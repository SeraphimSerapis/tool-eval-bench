"""Scoring integrity: callback errors, empty answers, trial statistics, and run settings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench import api
from tool_eval_bench.application import service as service_mod
from tool_eval_bench.application.service import BenchmarkService, validate_run_parameters
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.run_io import aggregate_trials
from tool_eval_bench.cli.scored_run import ScoredRun
from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult
from tool_eval_bench.domain.models import RUN_STATUS_INTERRUPTED
from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import run_all_scenarios, run_scenario, score_results
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter

BY_ID = {scenario.id: scenario for scenario in ALL_SCENARIOS}


class Replies(BackendAdapter):
    """Answers every turn with ``content`` and no tool calls."""

    def __init__(self, content: str = "done") -> None:
        self.content = content
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.calls += 1
        return ChatCompletionResult(content=self.content, finish_reason="stop")

    async def aclose(self) -> None:
        return None


def _scenario(sid: str, points: int = 2, category: Category = Category.A) -> ScenarioDefinition:
    status = {2: ScenarioStatus.PASS, 1: ScenarioStatus.PARTIAL, 0: ScenarioStatus.FAIL}[points]

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        return ScenarioEvaluation(status, points, f"{sid} graded")

    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=category,
        user_message=f"say hi {sid}",
        description="",
        handle_tool_call=lambda state, call: {"ok": True},
        evaluate=evaluate,
        tools_override=[],
    )


async def _run(adapter: BackendAdapter, scenario: ScenarioDefinition) -> ScenarioResult:
    return await run_scenario(
        adapter, model="m", base_url="http://x", api_key=None, scenario=scenario
    )


# -- progress callbacks ---------------------------------------------------------


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("hook", ["on_scenario_start", "on_scenario_result"])
async def test_a_failing_progress_callback_propagates_in_both_modes(
    concurrency: int, hook: str
) -> None:
    scenarios = [_scenario("S-1"), _scenario("S-2")]

    async def broken(scenario: ScenarioDefinition, *args: Any) -> None:
        if scenario.id == "S-1":
            raise BrokenPipeError("stderr closed")

    with pytest.raises(BrokenPipeError):
        await run_all_scenarios(
            Replies(),
            model="m",
            base_url="http://x",
            scenarios=scenarios,
            concurrency=concurrency,
            **{hook: broken},
        )


async def test_a_parallel_callback_error_does_not_rescore_the_scenario_as_a_crash() -> None:
    graded: list[ScenarioResult] = []

    async def broken(scenario: ScenarioDefinition, result: ScenarioResult, *args: Any) -> None:
        graded.append(result)
        if scenario.id == "S-1":
            raise BrokenPipeError("stderr closed")

    with pytest.raises(BrokenPipeError):
        await run_all_scenarios(
            Replies(),
            model="m",
            base_url="http://x",
            scenarios=[_scenario("S-1"), _scenario("S-2")],
            concurrency=2,
            on_scenario_result=broken,
        )

    # Every scenario finished with its evaluator's grade; none became model_crash.
    assert sorted((r.scenario_id, r.status, r.failure_kind) for r in graded) == [
        ("S-1", ScenarioStatus.PASS, None),
        ("S-2", ScenarioStatus.PASS, None),
    ]


# -- empty answers ----------------------------------------------------------------


@pytest.mark.parametrize("sid", ["TC-44", "TC-57"])
@pytest.mark.parametrize("content", ["", "   ", "[no content: MAX_TOKENS]"])
async def test_no_answer_and_no_tool_calls_fails(sid: str, content: str) -> None:
    result = await _run(Replies(content), BY_ID[sid])

    assert (result.status, result.points) == (ScenarioStatus.FAIL, 0)
    assert result.failure_kind == FailureKind.MISSING_STEP
    assert result.summary == "Returned no answer and made no tool calls."


async def test_a_visible_answer_keeps_its_grade() -> None:
    partial = await _run(Replies("I won't do that."), _scenario("S-1", points=1))
    empty = await _run(Replies(""), _scenario("S-1", points=1))

    assert (partial.status, partial.points) == (ScenarioStatus.PARTIAL, 1)
    assert (empty.status, empty.points) == (ScenarioStatus.FAIL, 0)


async def test_a_scenario_the_evaluator_already_failed_is_unchanged() -> None:
    result = await _run(Replies(""), _scenario("S-1", points=0))

    assert result.summary == "S-1 graded"


# -- trial statistics -------------------------------------------------------------


def _pass(sid: str) -> ScenarioResult:
    return ScenarioResult(sid, ScenarioStatus.PASS, 2, "ok")


def _timeout(sid: str) -> ScenarioResult:
    return ScenarioResult(sid, ScenarioStatus.FAIL, 0, "t/o", failure_kind=FailureKind.TIMEOUT)


def test_a_timeout_in_one_trial_is_not_a_miss() -> None:
    scenarios = [_scenario("S-1"), _scenario("S-2")]
    trials = [
        score_results([_pass("S-1"), _pass("S-2")], scenarios),
        score_results([_pass("S-1"), _timeout("S-2")], scenarios),
    ]

    stats = aggregate_trials(trials)

    assert (stats["pass_at_k"], stats["pass_hat_k"], stats["reliability_gap"]) == (100, 100, 0)
    assert stats["per_scenario"]["S-2"]["points"] == [2]
    assert stats["per_scenario"]["S-2"]["stddev"] == 0.0


def test_a_scenario_no_trial_could_grade_drops_out() -> None:
    scenarios = [_scenario("S-1"), _scenario("S-2")]
    trials = [score_results([_pass("S-1"), _timeout("S-2")], scenarios)] * 2

    stats = aggregate_trials(trials)

    assert list(stats["per_scenario"]) == ["S-1"]
    assert stats["pass_hat_k"] == 100


def test_scenarios_and_categories_come_from_every_trial() -> None:
    first = [_scenario("S-1")]
    both = [_scenario("S-1"), _scenario("S-2", points=0, category=Category.B)]
    trials = [
        score_results([_pass("S-1")], first),
        score_results([_pass("S-1"), ScenarioResult("S-2", ScenarioStatus.FAIL, 0, "x")], both),
    ]

    stats = aggregate_trials(trials)

    assert stats["per_scenario"]["S-2"]["points"] == [0]
    assert (stats["pass_at_k"], stats["pass_hat_k"]) == (50, 50)
    assert set(stats["per_category"]) == {Category.A.value, Category.B.value}


# -- run settings -----------------------------------------------------------------

INVALID_SETTINGS = [
    pytest.param({"max_turns": 0}, id="max_turns=0"),
    pytest.param({"error_rate": math.nan}, id="error_rate=nan"),
    pytest.param({"error_rate": 1.5}, id="error_rate=1.5"),
    pytest.param({"error_rate": -0.1}, id="error_rate=-0.1"),
    pytest.param({"alpha": 2.0}, id="alpha=2"),
    pytest.param({"alpha": math.nan}, id="alpha=nan"),
]
DEFAULTS = {"max_turns": 8, "error_rate": 0.0, "alpha": 0.7}


@pytest.mark.parametrize("settings", INVALID_SETTINGS)
def test_out_of_range_settings_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_run_parameters(**{**DEFAULTS, **settings})


@pytest.mark.parametrize(
    "settings",
    [
        {"max_turns": 1},
        {"error_rate": 0.0},
        {"error_rate": 1.0},
        {"alpha": 0.0},
        {"alpha": 1.0},
    ],
)
def test_boundary_settings_are_accepted(settings: dict[str, Any]) -> None:
    validate_run_parameters(**{**DEFAULTS, **settings})


@pytest.mark.parametrize("settings", INVALID_SETTINGS)
async def test_service_rejects_bad_settings_before_any_request(settings: dict[str, Any]) -> None:
    adapter = Replies()
    service = BenchmarkService(repo=None, reporter=None)
    service._adapter_for = lambda *a, **k: adapter  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        await service.run_benchmark(
            model="m", backend="vllm", base_url="http://x", scenarios=[_scenario("S-1")], **settings
        )
    assert adapter.calls == 0


@pytest.mark.parametrize("settings", INVALID_SETTINGS)
async def test_api_rejects_bad_settings_before_probing(
    settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_probe(**kwargs: Any) -> None:
        raise AssertionError("probed the server")

    monkeypatch.setattr(api, "build_run_context", no_probe)

    with pytest.raises(ValueError):
        await api.run_benchmark(
            model="m",
            base_url="http://127.0.0.1:9",
            scenarios=[_scenario("S-1")],
            persist=False,
            **settings,
        )


# -- resume with trials -----------------------------------------------------------


async def test_a_later_trial_of_a_resumed_run_is_a_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_metadata(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(service_mod, "_collect_metadata_safe", no_metadata)
    repo = RunRepository(db_path=str(tmp_path / "bench.sqlite"))
    service = BenchmarkService(repo=repo, reporter=MarkdownReporter(root=str(tmp_path / "runs")))
    monkeypatch.setattr(service, "_adapter_for", lambda *a, **k: Replies())
    protocol = [_scenario("S-1"), _scenario("S-2")]
    run_id = "20260101T000000Z-resume"
    repo.upsert_scenario_run(
        {"run_id": run_id, "status": "running", "config": {}, "scores": {}, "metadata": {}}
    )
    repo.mark_run_status(run_id, RUN_STATUS_INTERRUPTED)
    args = make_parser().parse_args(["--scenarios", "TC-01"])
    args._resume_run_id = run_id
    args._resume_prior_results = [_pass("S-1").to_dict()]
    args._resume_scenarios = protocol
    request = ScoredRun.from_args(
        args,
        model="m",
        backend="vllm",
        base_url="http://x",
        api_key=None,
        wire_format="openai",
        scenarios=[protocol[1]],
        extra_params=None,
        scenario_packs=None,
    )
    try:
        first = await service.run_benchmark(**request.service_kwargs())
        later = request.later_trial()
        second = await service.run_benchmark(**later.service_kwargs())
    finally:
        repo.close()

    assert first["run_id"] == run_id and first["status"] == "completed"
    assert second["run_id"] != run_id and second["status"] == "completed"
    assert [s.id for s in later.scenarios] == ["S-1", "S-2"]
    assert later.resume_prior_results is None and later.resume_scenarios is None
