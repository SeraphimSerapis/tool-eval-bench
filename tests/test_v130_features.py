"""Non-duplicate contract guards for arg_bytes serialization and Category O response_format."""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus


class TestArgBytesTracking:
    """Tests for per-tool-call argument size tracking."""

    def test_arg_bytes_in_scenario_result(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            tool_call_arg_bytes=150,
        )
        assert result.tool_call_arg_bytes == 150

    def test_arg_bytes_in_to_dict(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            tool_call_arg_bytes=250,
        )
        d = result.to_dict()
        assert d["tool_call_arg_bytes"] == 250

    def test_arg_bytes_zero_not_in_dict(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            tool_call_arg_bytes=0,
        )
        d = result.to_dict()
        assert "tool_call_arg_bytes" not in d

    def test_arg_bytes_default_zero(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
        )
        assert result.tool_call_arg_bytes == 0


class TestCategoryORegistration:
    """Contract guards for Category O response_format_override configuration."""

    def test_tc68_has_no_response_format(self) -> None:
        """TC-68 tests model restraint, so server-side response_format must stay unset."""
        from tool_eval_bench.evals.scenarios import STRUCTURED_SCENARIOS

        tc68 = next(s for s in STRUCTURED_SCENARIOS if s.id == "TC-68")
        assert tc68.response_format_override is None, (
            "TC-68 must NOT use response_format: it tests model restraint, not server enforcement"
        )

    def test_other_structured_scenarios_have_response_format(self) -> None:
        """TC-64, 65, 66, 67, 69 should all have response_format_override."""
        from tool_eval_bench.evals.scenarios import STRUCTURED_SCENARIOS

        for s in STRUCTURED_SCENARIOS:
            if s.id == "TC-68":
                continue
            assert s.response_format_override is not None, (
                f"{s.id} should have response_format_override"
            )
