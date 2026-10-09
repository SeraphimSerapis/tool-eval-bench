"""ScoredRun turns the flags into run_benchmark kwargs and the resume RunSettings."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.run_config import build_run_config
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.commands import resolve_scenarios
from tool_eval_bench.cli.dispatch import _resume_config_mismatches
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.scored_run import ScoredRun

# Supplied per call site by the runners, never by the builder.
RUNNER_OWNED = {
    "on_scenario_start",
    "on_scenario_result",
    "on_scenario_audit",
    "rate_limit_observer",
    "throughput_samples",
}
JUDGE_KWARGS = {
    "decision_judge_base_url",
    "decision_judge_model",
    "decision_judge_api_key",
    "decision_judge",
}
SCORING_FLAGS = [
    "--temperature",
    "0.3",
    "--timeout",
    "45",
    "--max-turns",
    "5",
    "--seed",
    "7",
    "--reference-date",
    "2026-03-20",
    "--parallel",
    "2",
    "--error-rate",
    "0.1",
    "--alpha",
    "0.5",
    "--weight-by-difficulty",
    "--system-prompt",
    "Be terse.",
    "--variant-seed",
    "3",
]
JUDGE_FLAGS = [
    "--decision-judge-base-url",
    "http://judge.test/v1",
    "--decision-judge-model",
    "judge-model",
]


def _request(flags: list[str], scenarios: list[Any] | None = None, **hidden: Any) -> ScoredRun:
    """Build a request the way a runner does; ``hidden`` sets dispatch's ``args._*`` attributes."""
    args = make_parser().parse_args(flags)
    for name, value in hidden.items():
        setattr(args, name, value)
    return ScoredRun.from_args(
        args,
        model="m",
        backend="vllm",
        base_url="http://bench.test/v1",
        api_key=None,
        wire_format="openai",
        scenarios=scenarios if scenarios is not None else resolve_scenarios(args),
        extra_params={"top_p": 0.9},
        scenario_packs=None,
        context_pressure_config={"ratio": 0.5},
    )


def _service_parameters() -> set[str]:
    params = inspect.signature(BenchmarkService.run_benchmark).parameters
    # scenario_ids is the API's alternative to scenarios; the CLI always resolves.
    return set(params) - {"self", "scenario_ids"}


def test_service_kwargs_cover_every_service_parameter_with_a_judge() -> None:
    request = _request([*SCORING_FLAGS, *JUDGE_FLAGS])

    assert set(request.service_kwargs()) == _service_parameters() - RUNNER_OWNED


def test_service_kwargs_omit_the_judge_without_a_judge_url() -> None:
    request = _request(SCORING_FLAGS)

    assert not request.audits_answers
    assert set(request.service_kwargs()) == _service_parameters() - RUNNER_OWNED - JUDGE_KWARGS


def test_judge_key_is_read_from_the_environment_only_with_a_judge_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOL_EVAL_DECISION_JUDGE_API_KEY", "judge-secret")

    assert _request(SCORING_FLAGS).judge is None
    judged = _request([*SCORING_FLAGS, *JUDGE_FLAGS]).service_kwargs()
    assert judged["decision_judge_api_key"] == "judge-secret"


def test_run_settings_match_what_the_service_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    full = resolve_scenarios(make_parser().parse_args([]))
    # A resume: the run executes a subset but records the original protocol.
    request = _request(
        [*SCORING_FLAGS, *JUDGE_FLAGS],
        full[:3],
        _resume_run_id="run-1",
        _resume_prior_results=[],
        _resume_scenarios=full,
    )
    assert request.scenarios == full[:3]
    stored: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def capture(settings: Any, *, scenarios: Any, metadata: Any, scenario_packs: Any) -> None:
        stored.update(settings=settings, scenarios=scenarios, scenario_packs=scenario_packs)
        raise _Stop

    monkeypatch.setattr(service_module, "build_run_config", capture)
    monkeypatch.setattr(service_module, "_collect_metadata_safe", _no_metadata)
    with pytest.raises(_Stop):
        asyncio.run(
            BenchmarkService(repo=None, reporter=None).run_benchmark(**request.service_kwargs())
        )

    settings = request.run_settings()
    assert stored["settings"] == settings
    assert settings.decision_judge is not None and settings.decision_judge["checks"]
    assert [s.id for s in stored["scenarios"]] == [s.id for s in request.config_scenarios()]
    assert [s.user_message for s in stored["scenarios"]] == [
        s.user_message for s in request.config_scenarios()
    ]


async def _no_metadata(*_args: Any) -> dict[str, Any]:
    return {}


def test_resume_check_compares_the_callers_scenarios_not_a_stale_args_attribute() -> None:
    full = resolve_scenarios(make_parser().parse_args([]))
    args = make_parser().parse_args([*SCORING_FLAGS, *JUDGE_FLAGS])
    stored = _request([*SCORING_FLAGS, *JUDGE_FLAGS], full)
    previous = build_run_config(
        stored.run_settings(),
        scenarios=stored.config_scenarios(),
        metadata={},
        scenario_packs=None,
    )
    # Left by an earlier resume plan; the explicit scenarios must still win.
    args._resume_scenarios = full[:3]

    mismatches = _resume_config_mismatches(
        previous,
        model="m",
        backend="vllm",
        base_url="http://bench.test/v1",
        scenarios=full,
        args=args,
        extra_params={"top_p": 0.9},
        scenario_packs=None,
        context_pressure={"ratio": 0.5},
    )

    assert mismatches == []
