from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
)


def _evaluate_fail(state: ScenarioState) -> ScenarioEvaluation:
    return ScenarioEvaluation(status=ScenarioStatus.FAIL, points=0, summary="failed")


def _scenario(scenario_id: str, category: Category) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title=scenario_id,
        category=category,
        user_message="test",
        description="test",
        handle_tool_call=lambda state, call: {},
        evaluate=_evaluate_fail,
    )


def _run_context() -> RunContext:
    return RunContext(
        tool_version="2.0.0",
        git_sha="abc123",
        hostname="test-host",
        platform_info="test-platform",
        python_version="3.14",
        model="test-model",
        backend="vllm",
        base_url="http://localhost:8000",
    )


def test_scenario_seed_offset_uses_stable_sha256_digest() -> None:
    from tool_eval_bench.runner.orchestrator import _scenario_seed_offset

    expected = int.from_bytes(hashlib.sha256(b"TC-01").digest()[:4], "big")
    assert _scenario_seed_offset("TC-01") == expected


def test_plugin_metadata_serialization_uses_json_safe_dict() -> None:
    from tool_eval_bench.cli.bench import _metadata_for_storage

    metadata = _metadata_for_storage(_run_context())

    assert metadata["tool_version"] == "2.0.0"
    assert metadata["model"] == "test-model"


def test_plugin_persistence_errors_are_not_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli.bench import _persist_plugin_run
    from tool_eval_bench.storage import db as db_module

    class BrokenRepository:
        def __enter__(self) -> "BrokenRepository":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def upsert_scenario_run(self, run_data: dict) -> None:
            raise RuntimeError("sqlite unavailable")

    monkeypatch.setattr(db_module, "RunRepository", BrokenRepository)

    with pytest.raises(RuntimeError, match="sqlite unavailable"):
        _persist_plugin_run({"run_id": "plugin-run"})


def test_scenario_result_roundtrip_preserves_raw_trace() -> None:
    original = ScenarioResult(
        scenario_id="TC-01",
        status=ScenarioStatus.PASS,
        points=2,
        summary="passed",
        raw_log="assistant=starting\nassistant=done",
    )

    restored = ScenarioResult.from_dict(original.to_dict())

    assert restored.raw_log == original.raw_log


def test_scenario_result_roundtrip_preserves_hardmode_diagnostics() -> None:
    original = ScenarioResult(
        scenario_id="TC-80",
        status=ScenarioStatus.PARTIAL,
        points=1,
        summary="Recovered original state.",
        parallel_tool_turns=[1, 3],
        state_checkpoints=["unsafe mutation before availability check"],
    )

    restored = ScenarioResult.from_dict(original.to_dict())

    assert restored.parallel_tool_turns == [1, 3]
    assert restored.state_checkpoints == [
        "unsafe mutation before availability check",
    ]


@pytest.mark.asyncio
async def test_resume_rescores_and_reports_merged_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.application import service as service_module
    from tool_eval_bench.runner.orchestrator import score_results
    from tool_eval_bench.runner.service import BenchmarkService, _build_run_config

    prior = _scenario("PRIOR-A", Category.A)
    rerun = _scenario("RERUN-K", Category.K)
    rerun_result = ScenarioResult(
        scenario_id=rerun.id,
        status=ScenarioStatus.FAIL,
        points=0,
        summary="unsafe",
        raw_log="rerun trace",
    )
    rerun_summary = score_results([rerun_result], [rerun])
    monkeypatch.setattr(service_module, "ALL_SCENARIOS", [prior, rerun])
    monkeypatch.setattr(service_module, "run_all_scenarios", AsyncMock(return_value=rerun_summary))

    reporter = MagicMock()
    reporter.write_scenario_report.return_value = "report.md"
    repo = MagicMock()
    adapter = object()
    service = BenchmarkService(repo=repo, reporter=reporter)
    monkeypatch.setattr(service, "_adapter_for", lambda backend: adapter)

    result = await service.run_benchmark(
        model="test-model",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[rerun],
        resume_run_id="existing-run",
        resume_prior_results=[
            {
                "scenario_id": prior.id,
                "status": "pass",
                "points": 2,
                "summary": "passed",
                "raw_log": "prior trace",
            }
        ],
    )

    merged_summary = reporter.write_scenario_report.call_args.args[2]
    report_metadata = reporter.write_scenario_report.call_args.kwargs["scenario_metadata"]
    assert len(merged_summary.scenario_results) == 2
    assert merged_summary.rating.endswith("(safety-capped)")
    assert merged_summary.scenario_results[0].raw_log == "prior trace"
    assert list(report_metadata) == ["PRIOR-A", "RERUN-K"]
    assert report_metadata["PRIOR-A"].title == "PRIOR-A"
    assert report_metadata["RERUN-K"].category == Category.K
    assert result["config"]["scenario_ids"] == ["PRIOR-A", "RERUN-K"]
    assert repo.upsert_scenario_run.call_args.args[0]["report_path"] == "report.md"
    full_config = _build_run_config(
        model="test-model",
        backend="vllm",
        base_url="http://localhost:8000",
        scenarios=[prior, rerun],
        temperature=0.0,
        timeout_seconds=60.0,
        max_turns=8,
        seed=None,
        reference_date=None,
        concurrency=1,
        error_rate=0.0,
        alpha=0.7,
        extra_params=None,
        context_pressure_config=None,
        weight_by_difficulty=False,
        metadata={},
    )
    assert result["config"]["config_fingerprint"] == full_config["config_fingerprint"]


@pytest.mark.asyncio
async def test_report_failure_does_not_store_completed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed SQLite row must never exist without its Markdown artifact."""
    from tool_eval_bench.application import service as service_module
    from tool_eval_bench.application.service import BenchmarkService
    from tool_eval_bench.runner.orchestrator import score_results

    scenario = _scenario("REPORT-FAIL", Category.A)
    result = ScenarioResult(
        scenario_id=scenario.id,
        status=ScenarioStatus.PASS,
        points=2,
        summary="passed",
    )
    summary = score_results([result], [scenario])
    monkeypatch.setattr(service_module, "run_all_scenarios", AsyncMock(return_value=summary))

    repo = MagicMock()
    reporter = MagicMock()
    reporter.write_scenario_report.side_effect = OSError("report filesystem unavailable")
    service = BenchmarkService(repo=repo, reporter=reporter)
    monkeypatch.setattr(service, "_adapter_for", lambda backend: object())

    with pytest.raises(OSError, match="report filesystem unavailable"):
        await service.run_benchmark(
            model="test-model",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[scenario],
        )

    repo.upsert_scenario_run.assert_not_called()
