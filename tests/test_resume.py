"""Gap tests for application/resume.py validation errors and prior-result merging."""

from __future__ import annotations

from typing import Any

import pytest

from tool_eval_bench.application.resume import (
    ResumeRejected,
    plan_stored_resume,
    prior_results_for_resume,
)
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

_URL = "http://localhost:8000"


def _stored_run(**overrides: Any) -> dict[str, Any]:
    scenarios = [s for s in ALL_SCENARIOS if s.id in {"TC-01", "TC-02"}]
    values: dict[str, Any] = {
        "model": "m",
        "backend": "vllm",
        "base_url": _URL,
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "context_pressure_config": None,
        "weight_by_difficulty": False,
    }
    values.update(overrides)
    return {
        "status": "interrupted",
        "run_type": "tool_eval",
        "config": build_run_config(RunSettings(**values), scenarios=scenarios, metadata={}),
    }


def test_plan_rejects_non_tool_eval_run_type() -> None:
    run = _stored_run()
    run["run_type"] = "perf"
    with pytest.raises(ResumeRejected, match="Only tool_eval runs"):
        plan_stored_resume(run, [], base_url=_URL)


def test_plan_rejects_non_dict_config() -> None:
    run = {"status": "interrupted", "run_type": "tool_eval", "config": ["bad"]}
    with pytest.raises(ResumeRejected, match="malformed"):
        plan_stored_resume(run, [], base_url=_URL)


def test_plan_rejects_invalid_wire_format() -> None:
    with pytest.raises(ResumeRejected, match="Unknown --format"):
        plan_stored_resume(_stored_run(), [], base_url=_URL, wire_format="bogus")


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("temperature", True),
        ("temperature", "not-a-float"),
        ("temperature", float("inf")),
        ("timeout_seconds", float("-inf")),
        ("error_rate", None),
        ("alpha", "bad"),
        ("max_turns", True),
        ("max_turns", "eight"),
        ("concurrency", None),
        ("concurrency", False),
    ],
)
def test_plan_rejects_invalid_numeric_config_fields(key: str, bad_value: Any) -> None:
    run = _stored_run()
    run["config"] = {**run["config"], key: bad_value}
    with pytest.raises(ResumeRejected, match=key):
        plan_stored_resume(run, [], base_url=_URL)


def test_plan_rejects_stored_reviewer_missing_model() -> None:
    run = _stored_run()
    run["config"] = {**run["config"], "reviewer": {"scope": "turns"}}
    with pytest.raises(ResumeRejected, match="no model"):
        plan_stored_resume(run, [], base_url=_URL)


def test_plan_rejects_unrebuildable_stored_reviewer() -> None:
    run = _stored_run()
    run["config"] = {
        **run["config"],
        "reviewer": {"model": "critic", "scope": "invalid_scope", "max_rounds": 1},
    }
    with pytest.raises(ResumeRejected, match="cannot be rebuilt"):
        plan_stored_resume(run, [], base_url=_URL)


def test_prior_results_for_resume_prefers_final_scores_over_checkpoints() -> None:
    prev_run = {
        "scores": {
            "scenario_results": [
                {"scenario_id": "TC-01", "status": "pass", "points": 2, "source": "final"},
                {"status": "fail"},  # missing scenario_id -> skipped
            ]
        }
    }
    checkpoints = [
        {"scenario_id": "TC-01", "status": "fail", "points": 0, "source": "checkpoint"},
        {"scenario_id": "TC-02", "status": "partial", "points": 1, "source": "checkpoint"},
    ]
    merged = {r["scenario_id"]: r for r in prior_results_for_resume(prev_run, checkpoints)}
    assert merged["TC-01"]["source"] == "final"
    assert merged["TC-02"]["source"] == "checkpoint"

    # Without checkpoints, only scored results with scenario_id are returned.
    assert [r["scenario_id"] for r in prior_results_for_resume(prev_run, [])] == ["TC-01"]
