"""Tests for Hard Mode scenarios (Category P).

Covers the 5 hardmode scenarios: adversarial near-duplicate tools,
ambiguous recipient, cascading error recovery, multi-constraint
composition, and stateful multi-turn corrections.
"""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.scenarios_hardmode import (
    HARDMODE_DISPLAY_DETAILS,
    HARDMODE_SCENARIOS,
)


class TestHardmodeRegistry:
    """Registry-level checks."""

    def test_scenario_count(self):
        assert len(HARDMODE_SCENARIOS) == 5

    def test_all_category_p(self):
        for s in HARDMODE_SCENARIOS:
            assert s.category == Category.P

    def test_ids_start_at_70(self):
        ids = [int(s.id.split("-")[1]) for s in HARDMODE_SCENARIOS]
        assert min(ids) == 70
        assert max(ids) == 74

    def test_unique_ids(self):
        ids = [s.id for s in HARDMODE_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_all_have_display_details(self):
        for s in HARDMODE_SCENARIOS:
            assert s.id in HARDMODE_DISPLAY_DETAILS

    def test_not_in_all_scenarios(self):
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        hardmode_ids = {s.id for s in HARDMODE_SCENARIOS}
        all_ids = {s.id for s in ALL_SCENARIOS}
        assert hardmode_ids.isdisjoint(all_ids)

    def test_in_all_scenarios_with_hardmode(self):
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
        hardmode_ids = {s.id for s in HARDMODE_SCENARIOS}
        combined_ids = {s.id for s in ALL_SCENARIOS_WITH_HARDMODE}
        assert hardmode_ids.issubset(combined_ids)


def _get_scenario(sid: str):
    return next(s for s in HARDMODE_SCENARIOS if s.id == sid)


def _make_call(name: str, args: dict, turn: int = 1) -> ToolCallRecord:
    return ToolCallRecord(id=f"call_{name}", name=name, raw_arguments=str(args), arguments=args, turn=turn)


class TestTC70AdversarialTools:
    """TC-70: Adversarial near-duplicate tools."""

    def test_pass_uses_global_directly(self):
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        call = _make_call("get_weather_global", {"location": "Tokyo"})
        sc.handle_tool_call(state, call)
        state.tool_calls.append(call)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_partial_recovers_from_error(self):
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        # First: wrong tool
        c1 = _make_call("get_weather", {"location": "Tokyo"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        # Then: correct tool
        c2 = _make_call("get_weather_global", {"location": "Tokyo"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_wrong_tool_no_recovery(self):
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        call = _make_call("get_weather", {"location": "Tokyo"})
        sc.handle_tool_call(state, call)
        state.tool_calls.append(call)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_has_tools_override(self):
        sc = _get_scenario("TC-70")
        assert sc.tools_override is not None
        names = [t["function"]["name"] for t in sc.tools_override]
        assert "get_weather" in names
        assert "get_weather_global" in names


class TestTC71AmbiguousRecipient:
    """TC-71: Ambiguous recipient — 3 Jordans."""

    def test_pass_asks_for_clarification(self):
        sc = _get_scenario("TC-71")
        state = ScenarioState()
        call = _make_call("get_contacts", {"query": "Jordan"})
        sc.handle_tool_call(state, call)
        state.tool_calls.append(call)
        state.assistant_messages.append(
            "I found three contacts named Jordan: Jordan Park (Engineering), "
            "Jordan Lee (Marketing), and Jordan Kim (Finance). Which Jordan "
            "would you like me to send the report to?"
        )
        state.final_answer = state.assistant_messages[-1]
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_sends_without_asking(self):
        sc = _get_scenario("TC-71")
        state = ScenarioState()
        c1 = _make_call("get_contacts", {"query": "Jordan"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        c2 = _make_call("send_email", {"to": "jordan.park@company.com", "body": "Report"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        state.final_answer = "I've sent the report to Jordan Park."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


class TestTC72CascadingRecovery:
    """TC-72: Cascading error recovery."""

    def test_pass_recovers_and_emails(self):
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        # Search
        c1 = _make_call("search_files", {"query": "Project Alpha"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        # Try corrupted file
        c2 = _make_call("read_file", {"file_id": "file_404"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        # Try alternative
        c3 = _make_call("read_file", {"file_id": "file_405"}, turn=3)
        sc.handle_tool_call(state, c3)
        state.tool_calls.append(c3)
        # Email
        c4 = _make_call("send_email", {"to": "dana@company.com", "body": "Budget is $1.2M"}, turn=4)
        sc.handle_tool_call(state, c4)
        state.tool_calls.append(c4)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_stops_at_error(self):
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        c1 = _make_call("read_file", {"file_id": "file_404"})
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        state.final_answer = "The file is corrupted."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


class TestTC74StatefulCorrections:
    """TC-74: Stateful multi-turn corrections."""

    def test_has_follow_up_messages(self):
        sc = _get_scenario("TC-74")
        assert len(sc.follow_up_messages) == 4

    def test_pass_all_corrections(self):
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        # Look up Sarah (from follow-up)
        c1 = _make_call("get_contacts", {"query": "sarah"}, turn=3)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        # Final event with all corrections applied
        c2 = _make_call("create_calendar_event", {
            "title": "Product Review",
            "date": "2026-03-25",
            "time": "14:00",
            "duration_minutes": 45,
            "attendees": ["mark.chen@company.com", "sarah.jones@company.com"],
        }, turn=5)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_no_event(self):
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        state.final_answer = "I'll schedule that for you."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL
