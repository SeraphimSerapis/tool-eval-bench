"""Behavioral tests for BenchmarkService locale filtering, cleanup, and DB resilience."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.review import ReviewerSettings
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioStatus,
)
from tool_eval_bench.storage.db import RunRepository


def _dummy_scenario(scenario_id: str = "CUSTOM-01") -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title="Dummy Scenario",
        category=Category.A,
        user_message="Say hello.",
        description="Respond politely.",
        handle_tool_call=lambda _state, _call: {"ok": True},
        evaluate=lambda _state: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )


class _TrackingAdapter:
    def __init__(self, *, response: str = "Hello!", fail_with: Exception | None = None) -> None:
        self._response = response
        self._fail_with = fail_with
        self.closed = False

    async def chat_completion(self, **_kwargs: Any) -> ChatCompletionResult:
        if self._fail_with is not None:
            raise self._fail_with
        return ChatCompletionResult(content=self._response, finish_reason="stop")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_run_benchmark_raises_when_locale_filters_all_selected_scenarios_to_empty() -> None:
    service = BenchmarkService(repo=None, reporter=None)
    unlocalized = _dummy_scenario("CUSTOM-UNLOCALIZED")

    with pytest.raises(ValueError, match="None of the selected scenarios support --locale de"):
        await service.run_benchmark(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[unlocalized],
            locale="de",
            probe_engine=False,
        )


@pytest.mark.asyncio
async def test_run_benchmark_resolves_scenario_ids_and_rejects_unknown_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrackingAdapter()
    service = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service, "_adapter_for", lambda *_a, **_k: adapter)

    with pytest.raises(ValueError, match="Unknown scenario IDs: NOPE-99"):
        await service.run_benchmark(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8000",
            scenario_ids=["TC-01", "NOPE-99"],
            probe_engine=False,
        )

    result = await service.run_benchmark(
        model="test-model",
        backend="vllm",
        base_url="http://localhost:8000",
        scenario_ids=["TC-01"],
        probe_engine=False,
    )
    assert [r["scenario_id"] for r in result["scores"]["scenario_results"]] == ["TC-01"]
    assert adapter.closed is True


@pytest.mark.asyncio
async def test_run_benchmark_closes_reviewer_adapter_on_success_failure_and_claim_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    scenario = _dummy_scenario("CUSTOM-REV")
    reviewer_settings = ReviewerSettings(model="critic", base_url="http://critic.local:8001")

    # 1. Normal completion closes both main and reviewer adapters in finally.
    main_ok = _TrackingAdapter(response="Done.")
    reviewer_ok = _TrackingAdapter(response='{"verdict": "approve", "feedback": ""}')
    service_ok = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service_ok, "_adapter_for", lambda *_a, **_k: main_ok)
    monkeypatch.setattr(
        "tool_eval_bench.application.service.build_adapter",
        lambda *_a, **_k: reviewer_ok,
    )

    result = await service_ok.run_benchmark(
        model="test-model",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[scenario],
        reviewer=reviewer_settings,
        probe_engine=False,
    )
    assert result["status"] == "completed"
    assert main_ok.closed is True
    assert reviewer_ok.closed is True

    # 2. Failure during run_all_scenarios still closes reviewer adapter in finally.
    main_err = _TrackingAdapter(response="Done.")
    reviewer_err = _TrackingAdapter(response='{"verdict": "approve", "feedback": ""}')
    service_err = BenchmarkService(repo=None, reporter=None)
    monkeypatch.setattr(service_err, "_adapter_for", lambda *_a, **_k: main_err)
    monkeypatch.setattr(
        "tool_eval_bench.application.service.build_adapter",
        lambda *_a, **_k: reviewer_err,
    )

    async def _explode(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("orchestrator crashed")

    monkeypatch.setattr("tool_eval_bench.application.service.run_all_scenarios", _explode)
    with pytest.raises(RuntimeError, match="orchestrator crashed"):
        await service_err.run_benchmark(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[scenario],
            reviewer=reviewer_settings,
            probe_engine=False,
        )
    assert main_err.closed is True
    assert reviewer_err.closed is True

    # 3. Failure in _claim_run (resuming an already-completed run) closes reviewer adapter too.
    db_path = str(tmp_path / "bench.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": "already-done-run",
                "status": "completed",
                "config": {"model": "test-model"},
                "scores": {},
                "metadata": {},
            }
        )
        main_claim = _TrackingAdapter(response="Done.")
        reviewer_claim = _TrackingAdapter(response='{"verdict": "approve", "feedback": ""}')
        service_claim = BenchmarkService(repo=repo, reporter=None)
        monkeypatch.setattr(service_claim, "_adapter_for", lambda *_a, **_k: main_claim)
        monkeypatch.setattr(
            "tool_eval_bench.application.service.build_adapter",
            lambda *_a, **_k: reviewer_claim,
        )
        with pytest.raises(ValueError, match="already completed"):
            await service_claim.run_benchmark(
                model="test-model",
                backend="vllm",
                base_url="http://localhost:8000",
                scenarios=[scenario],
                reviewer=reviewer_settings,
                resume_run_id="already-done-run",
                probe_engine=False,
            )
    assert main_claim.closed is True
    assert reviewer_claim.closed is True


@pytest.mark.asyncio
async def test_run_benchmark_survives_db_errors_on_checkpoint_clear_and_interrupted_mark(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scenario = _dummy_scenario("CUSTOM-DB")

    # 1. Checkpoint clear error after finalization must not fail a completed run.
    repo_clear_fail = MagicMock()
    repo_clear_fail.get.return_value = None
    repo_clear_fail.clear_checkpoints.side_effect = RuntimeError("sqlite busy on clear")

    async def _noop_checkpoint(*_a: Any, **_k: Any) -> None:
        return None

    repo_clear_fail.acheckpoint_scenario_result = _noop_checkpoint

    service_ok = BenchmarkService(repo=repo_clear_fail, reporter=None)
    monkeypatch.setattr(service_ok, "_adapter_for", lambda *_a, **_k: _TrackingAdapter())

    with caplog.at_level("WARNING", logger="tool_eval_bench.application.service"):
        completed = await service_ok.run_benchmark(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[scenario],
            probe_engine=False,
        )

    assert completed["status"] == "completed"
    assert any("Failed to clear checkpoints" in rec.message for rec in caplog.records)

    # 2. DB error while marking a failed run interrupted must not mask the run failure.
    caplog.clear()
    repo_mark_fail = MagicMock()
    repo_mark_fail.get.return_value = None
    repo_mark_fail.mark_run_status.side_effect = RuntimeError("disk full")
    repo_mark_fail.acheckpoint_scenario_result = _noop_checkpoint

    service_fail = BenchmarkService(repo=repo_mark_fail, reporter=None)
    monkeypatch.setattr(service_fail, "_adapter_for", lambda *_a, **_k: _TrackingAdapter())

    async def _fail_run(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("primary run failure")

    monkeypatch.setattr("tool_eval_bench.application.service.run_all_scenarios", _fail_run)

    with caplog.at_level("WARNING", logger="tool_eval_bench.application.service"):
        with pytest.raises(RuntimeError, match="primary run failure"):
            await service_fail.run_benchmark(
                model="test-model",
                backend="vllm",
                base_url="http://localhost:8000",
                scenarios=[scenario],
                probe_engine=False,
            )

    assert any("Could not mark run" in rec.message for rec in caplog.records)
