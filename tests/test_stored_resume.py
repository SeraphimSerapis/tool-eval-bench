"""Tests for resuming a run from its stored config alone (the web UI's resume path)."""

from __future__ import annotations

from typing import Any

import pytest

from tool_eval_bench.application.resume import (
    ResumeRejected,
    plan_stored_resume,
    result_requires_rerun,
)
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

_URL = "http://localhost:8000"


def _stored(**overrides: Any) -> dict[str, Any]:
    scenarios = [s for s in ALL_SCENARIOS if s.id in {"TC-01", "TC-02", "TC-03"}]
    values: dict[str, Any] = {
        "model": "m",
        "backend": "vllm",
        "base_url": _URL,
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": 7,
        "reference_date": "2026-01-15",
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": {"top_p": 0.9},
        "context_pressure_config": None,
        "weight_by_difficulty": False,
    }
    values.update(overrides)
    return {"config": build_run_config(RunSettings(**values), scenarios=scenarios, metadata={})}


def _result(sid: str, **overrides: Any) -> dict[str, Any]:
    return {
        "scenario_id": sid,
        "status": "fail",
        "points": 0,
        "summary": "s",
        "raw_log": "trace",
        **overrides,
    }


def test_plan_keeps_model_outcomes_and_reruns_infrastructure_failures() -> None:
    checkpoints = [
        _result("TC-01", status="pass", points=2),
        _result("TC-02", failure_kind="timeout"),
    ]
    plan = plan_stored_resume(_stored(), checkpoints, base_url=_URL)
    assert [r["scenario_id"] for r in plan.reusable] == ["TC-01"]
    assert [s.id for s in plan.remaining] == ["TC-02", "TC-03"]
    assert [s.id for s in plan.scenarios] == ["TC-01", "TC-02", "TC-03"]
    # Conditions the web UI can reproduce come back from the stored config.
    assert plan.settings.seed == 7
    assert plan.settings.reference_date == "2026-01-15"
    assert plan.settings.extra_params == {"top_p": 0.9}


def test_plan_rejects_a_different_endpoint() -> None:
    with pytest.raises(ResumeRejected, match="endpoint_id"):
        plan_stored_resume(_stored(), [], base_url="http://elsewhere:9000")


def test_plan_accepts_another_spelling_of_the_same_endpoint() -> None:
    plan = plan_stored_resume(_stored(), [], base_url=f"{_URL}/v1/")
    assert len(plan.remaining) == 3


def test_plan_rejects_runs_without_a_scenario_list() -> None:
    with pytest.raises(ResumeRejected, match="scenario list"):
        plan_stored_resume({"config": {"model": "m"}}, [], base_url=_URL)


def test_plan_rejects_held_out_pack_scenarios() -> None:
    run = _stored()
    run["config"]["scenario_ids"] = ["TC-01", "PACK-01"]
    with pytest.raises(ResumeRejected, match="--scenario-pack"):
        plan_stored_resume(run, [], base_url=_URL)


def test_result_requires_rerun_for_corrupt_checkpoints() -> None:
    assert result_requires_rerun({}) is True
    assert result_requires_rerun(_result("TC-01", raw_log=" ")) is True
    assert result_requires_rerun(_result("TC-01", status="weird")) is True
    assert result_requires_rerun(_result("TC-01")) is False
