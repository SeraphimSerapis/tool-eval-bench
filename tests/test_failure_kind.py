"""Tests for the failure_kind domain serialization contract."""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import (
    FailureKind,
    ScenarioResult,
    ScenarioStatus,
)

# ---------------------------------------------------------------------------
# Domain model round-trip
# ---------------------------------------------------------------------------


class TestScenarioResultFailureKind:
    def test_to_dict_includes_failure_kind(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="wrong tool",
            failure_kind=FailureKind.WRONG_TOOL,
        )
        d = result.to_dict()
        assert d["failure_kind"] == FailureKind.WRONG_TOOL

    def test_to_dict_omits_failure_kind_when_none(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            failure_kind=None,
        )
        d = result.to_dict()
        assert "failure_kind" not in d

    def test_from_dict_restores_failure_kind(self) -> None:
        data = {
            "scenario_id": "TC-01",
            "status": "fail",
            "points": 0,
            "summary": "timeout",
            "failure_kind": FailureKind.TIMEOUT,
        }
        result = ScenarioResult.from_dict(data)
        assert result.failure_kind == FailureKind.TIMEOUT

    def test_from_dict_defaults_failure_kind_to_none(self) -> None:
        data = {
            "scenario_id": "TC-01",
            "status": "pass",
            "points": 2,
            "summary": "ok",
        }
        result = ScenarioResult.from_dict(data)
        assert result.failure_kind is None
