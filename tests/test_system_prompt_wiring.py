"""Every hand-off that must carry a system prompt override.

A passing suite is not evidence that the override reaches the model: each of these
cuts was made in the source and left the rest of the suite green. They assert the
value at the boundary it crosses, so a dropped keyword fails here rather than in a
benchmark run whose scores quietly came from the built-in persona.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from tests.coverage_helpers import throughput_sample as _sample
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioStatus,
)

MARKER = "WIRING-MARKER"


def _scenario(sid: str = "TC-X") -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=Category.A,
        user_message="x",
        description="x",
        handle_tool_call=lambda state, call: {},
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = dict(
        trials=1,
        temperature=0.0,
        timeout=1.0,
        max_turns=2,
        reference_date=None,
        seed=1,
        parallel=1,
        error_rate=0.0,
        alpha=0.7,
        weight_by_difficulty=False,
        json_file=None,
        diff=None,
        output_dir=None,
        system_prompt=MARKER,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class _CapturingService:
    """Stands in for BenchmarkService and records what the run mode asked for."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "run_id": "r",
            "scores": {
                "final_score": 100,
                "rating": "Great",
                "scenario_results": [
                    {"scenario_id": "TC-X", "status": "pass", "points": 2, "summary": "ok"}
                ],
            },
        }


# ---------------------------------------------------------------------------
# CLI run modes
# ---------------------------------------------------------------------------


def test_json_run_mode_forwards_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli import dispatch

    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [_scenario()])
    monkeypatch.setattr(dispatch, "_emit_json_output", lambda *a, **k: None)
    service = _CapturingService()

    dispatch._run_json(service, "m", "vllm", "url", None, _args())

    assert service.calls[0]["system_prompt"] == MARKER


def test_plain_run_mode_forwards_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli import dispatch

    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [_scenario()])
    service = _CapturingService()

    dispatch._run_plain(service, Console(record=True), "m", "D", "vllm", "url", None, _args())

    assert service.calls[0]["system_prompt"] == MARKER


def test_live_display_run_mode_forwards_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli import dispatch

    class Display:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.results = {"TC-X": ScenarioResult("TC-X", ScenarioStatus.PASS, 2, "ok")}

        def start(self) -> None: ...

        def stop(self) -> None: ...

        async def on_scenario_start(self, *args: Any) -> None: ...

        async def on_scenario_result(self, *args: Any) -> None: ...

        def on_rate_limit(self, status: Any) -> None: ...

        def set_finished(self, *args: Any, **kwargs: Any) -> None: ...

    monkeypatch.setattr(dispatch, "BenchmarkDisplay", Display)
    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [_scenario()])
    service = _CapturingService()

    dispatch._run_with_live_display(
        service, Console(record=True), "m", "D", "vllm", "url", None, _args()
    )

    assert service.calls[0]["system_prompt"] == MARKER


def test_run_context_records_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The report's marker comes from here; without it a custom run looks default."""
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.utils import metadata

    captured: list[dict[str, Any]] = []

    async def collect(**kwargs: Any) -> None:
        captured.append(kwargs)
        return None

    monkeypatch.setattr(metadata, "collect_run_context", collect)
    monkeypatch.setattr(dispatch, "_scenario_selector_label", lambda args: "all")
    args = _args(no_think=False, context_pressure=None, label=None, no_probe_engine=True)

    dispatch._build_run_context(
        args,
        Console(record=True),
        model="m",
        backend="vllm",
        base_url="url",
        api_key=None,
        extra_params={},
    )

    assert captured[0]["system_prompt"] == MARKER


def test_main_drops_an_override_no_scenario_would_receive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """main() must apply the mode gate, not merely define it."""
    import sys

    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.storage import reports
    from tool_eval_bench.utils import metadata

    captured: list[dict[str, Any]] = []

    async def collect(**kwargs: Any) -> None:
        captured.append(kwargs)
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", collect)
    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: [_sample()])
    monkeypatch.setattr(
        reports.MarkdownReporter, "write_throughput_report", lambda *a, **k: tmp_path / "p.md"
    )
    monkeypatch.setattr(dispatch, "_persist_plugin_run", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--perf-only",
            "--no-warmup",
            "--system-prompt",
            "Be strict.",
            "--output-dir",
            str(tmp_path),
        ],
    )

    dispatch.main()

    assert captured[-1]["system_prompt"] is None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_forwards_the_override_and_persists_it_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service is the API's entry point, so it normalizes rather than trusting."""
    from tool_eval_bench.application import service as service_module
    from tool_eval_bench.application.service import BenchmarkService
    from tool_eval_bench.runner.orchestrator import score_results

    scenario = _scenario("TC-01")
    summary = score_results(
        [ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")],
        [scenario],
    )
    runner = AsyncMock(return_value=summary)
    monkeypatch.setattr(service_module, "run_all_scenarios", runner)

    service = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service, "_adapter_for", lambda *a, **k: object())

    run_data = await service.run_benchmark(
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[scenario],
        # Trailing newline: a caller that read a file keeps the same cohort as one
        # that passed the words inline.
        system_prompt="Be strict.\n",
    )

    assert runner.await_args is not None
    assert runner.await_args.kwargs["system_prompt"] == "Be strict."
    assert run_data["config"]["system_prompt"] == "Be strict."


@pytest.mark.asyncio
async def test_service_default_run_sends_and_persists_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.application import service as service_module
    from tool_eval_bench.application.service import BenchmarkService
    from tool_eval_bench.runner.orchestrator import score_results

    scenario = _scenario("TC-01")
    summary = score_results(
        [ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")],
        [scenario],
    )
    runner = AsyncMock(return_value=summary)
    monkeypatch.setattr(service_module, "run_all_scenarios", runner)

    service = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service, "_adapter_for", lambda *a, **k: object())

    run_data = await service.run_benchmark(
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[scenario],
    )

    assert runner.await_args is not None
    assert runner.await_args.kwargs["system_prompt"] is None
    assert "system_prompt" not in run_data["config"]


@pytest.mark.asyncio
async def test_service_rejects_a_blank_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI already refuses this; the API must not accept it either."""
    from tool_eval_bench.application.service import BenchmarkService

    service = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service, "_adapter_for", lambda *a, **k: object())

    with pytest.raises(ValueError, match="must not be empty"):
        await service.run_benchmark(
            model="m",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[_scenario("TC-01")],
            system_prompt="   ",
        )
