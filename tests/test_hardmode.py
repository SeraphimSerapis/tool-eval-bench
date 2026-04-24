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


# ===========================================================================
# Registry tests
# ===========================================================================

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

    def test_all_have_handlers_and_evaluators(self):
        for s in HARDMODE_SCENARIOS:
            assert callable(s.handle_tool_call)
            assert callable(s.evaluate)


# ===========================================================================
# Helpers
# ===========================================================================

def _get_scenario(sid: str):
    return next(s for s in HARDMODE_SCENARIOS if s.id == sid)


def _make_call(name: str, args: dict, turn: int = 1) -> ToolCallRecord:
    return ToolCallRecord(id=f"call_{name}_{turn}", name=name, raw_arguments=str(args), arguments=args, turn=turn)


# ===========================================================================
# TC-70: Adversarial Near-Duplicate Tools
# ===========================================================================

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

    def test_fail_no_tool_calls(self):
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        state.final_answer = "Tokyo is probably warm."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_has_tools_override(self):
        sc = _get_scenario("TC-70")
        assert sc.tools_override is not None
        names = [t["function"]["name"] for t in sc.tools_override]
        assert "get_weather" in names
        assert "get_weather_global" in names
        assert "web_search" in names

    def test_handler_returns_error_for_euro_tool_with_tokyo(self):
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        call = _make_call("get_weather", {"location": "Tokyo"})
        result = sc.handle_tool_call(state, call)
        assert "error" in result or "error" in str(result).lower()

    def test_handler_returns_data_for_global_tool(self):
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        call = _make_call("get_weather_global", {"location": "Tokyo"})
        result = sc.handle_tool_call(state, call)
        # Should contain weather data, not an error
        assert "temperature" in str(result)

    def test_handler_euro_tool_works_for_european_city(self):
        """get_weather should succeed for European cities."""
        sc = _get_scenario("TC-70")
        state = ScenarioState()
        call = _make_call("get_weather", {"location": "Berlin"})
        result = sc.handle_tool_call(state, call)
        assert "error" not in result  # no error key in the response dict


# ===========================================================================
# TC-71: Ambiguous Recipient
# ===========================================================================

class TestTC71AmbiguousRecipient:
    """TC-71: Ambiguous recipient — 3 Jordans."""

    def test_pass_asks_for_clarification_with_names(self):
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

    def test_partial_asks_clarification_without_naming_options(self):
        """Asks which Jordan but doesn't list them out."""
        sc = _get_scenario("TC-71")
        state = ScenarioState()
        call = _make_call("get_contacts", {"query": "Jordan"})
        sc.handle_tool_call(state, call)
        state.tool_calls.append(call)
        state.assistant_messages.append(
            "I found multiple contacts with that name. "
            "Could you clarify which one you'd like me to send the report to?"
        )
        state.final_answer = state.assistant_messages[-1]
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

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

    def test_fail_clarifies_but_also_sends_email(self):
        """Bug guard: model asks 'which Jordan?' but ALSO sends an email = FAIL."""
        sc = _get_scenario("TC-71")
        state = ScenarioState()
        c1 = _make_call("get_contacts", {"query": "Jordan"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        c2 = _make_call("send_email", {"to": "jordan.park@company.com", "body": "Report"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        state.assistant_messages.append(
            "I sent the report to Jordan Park. Which Jordan did you mean? "
            "I see Jordan Park, Jordan Lee, and Jordan Kim."
        )
        state.final_answer = state.assistant_messages[-1]
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_fail_no_contact_lookup(self):
        """Model tries to send email without looking up contacts first."""
        sc = _get_scenario("TC-71")
        state = ScenarioState()
        state.final_answer = "I don't have Jordan's email address."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_fail_sends_to_all_three(self):
        """Shotgunning to all Jordans is not acceptable."""
        sc = _get_scenario("TC-71")
        state = ScenarioState()
        c0 = _make_call("get_contacts", {"query": "Jordan"}, turn=1)
        sc.handle_tool_call(state, c0)
        state.tool_calls.append(c0)
        for i, addr in enumerate(["jordan.park@company.com", "jordan.lee@company.com", "jordan.kim@company.com"]):
            c = _make_call("send_email", {"to": addr, "body": "Report"}, turn=2 + i)
            sc.handle_tool_call(state, c)
            state.tool_calls.append(c)
        state.final_answer = "Sent to all three."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


# ===========================================================================
# TC-72: Cascading Error Recovery
# ===========================================================================

class TestTC72CascadingRecovery:
    """TC-72: Cascading error recovery."""

    def test_pass_recovers_and_emails(self):
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        c1 = _make_call("search_files", {"query": "Project Alpha"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        c2 = _make_call("read_file", {"file_id": "file_404"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        c3 = _make_call("read_file", {"file_id": "file_405"}, turn=3)
        sc.handle_tool_call(state, c3)
        state.tool_calls.append(c3)
        c4 = _make_call("send_email", {"to": "dana@company.com", "body": "Budget is $1.2M"}, turn=4)
        sc.handle_tool_call(state, c4)
        state.tool_calls.append(c4)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_pass_goes_straight_to_budget_file(self):
        """Smart model picks file_405 (budget) directly — also PASS."""
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        c1 = _make_call("search_files", {"query": "Project Alpha"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        c2 = _make_call("read_file", {"file_id": "file_405"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        c3 = _make_call("send_email", {"to": "dana@company.com", "body": "Total budget $1.2M"}, turn=3)
        sc.handle_tool_call(state, c3)
        state.tool_calls.append(c3)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_partial_reads_both_but_no_email(self):
        """Recovers from error and reads budget, but doesn't email."""
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        c1 = _make_call("read_file", {"file_id": "file_404"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        c2 = _make_call("read_file", {"file_id": "file_405"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        state.final_answer = "The budget is $1.2M."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_stops_at_error(self):
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        c1 = _make_call("read_file", {"file_id": "file_404"})
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        state.final_answer = "The file is corrupted."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_handler_file_404_always_errors(self):
        """file_404 is permanently corrupted, not just on first attempt."""
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        for _ in range(3):
            call = _make_call("read_file", {"file_id": "file_404"})
            result = sc.handle_tool_call(state, call)
            assert "error" in str(result).lower()

    def test_handler_file_405_returns_content(self):
        sc = _get_scenario("TC-72")
        state = ScenarioState()
        call = _make_call("read_file", {"file_id": "file_405"})
        result = sc.handle_tool_call(state, call)
        assert "1.2M" in str(result) or "750K" in str(result)


# ===========================================================================
# TC-73: Multi-Constraint Composition
# ===========================================================================

class TestTC73MultiConstraint:
    """TC-73: Multi-constraint composition."""

    def test_pass_full_chain(self):
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        # Search
        c1 = _make_call("web_search", {"query": "vegan restaurant Berlin near Alexanderplatz"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        state.assistant_messages.append(
            "I found Green Kitchen Berlin — it's vegan-friendly, 0.8km from Alexanderplatz, "
            "and open daily including Sundays."
        )
        # Contact lookup
        c2 = _make_call("get_contacts", {"query": "Lisa"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        # Email
        c3 = _make_call("send_email", {
            "to": "lisa.mueller@company.com",
            "body": "I recommend Green Kitchen Berlin for Sunday."
        }, turn=3)
        sc.handle_tool_call(state, c3)
        state.tool_calls.append(c3)
        state.final_answer = "I've emailed the Green Kitchen recommendation to Lisa."
        state.assistant_messages.append(state.final_answer)
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_partial_searches_and_filters_but_no_email(self):
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        c1 = _make_call("web_search", {"query": "restaurant Berlin Alexanderplatz vegan"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        state.assistant_messages.append(
            "I found Green Kitchen Berlin and Veganz Bistro as vegan options near Alexanderplatz."
        )
        state.final_answer = state.assistant_messages[-1]
        result = sc.evaluate(state)
        # searched + mentions_valid = 2 steps
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_no_search(self):
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        state.final_answer = "I'd recommend trying a restaurant in Berlin."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_fail_recommends_invalid_restaurant(self):
        """Recommends Mitte Brasserie which is closed Sundays and not vegan."""
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        c1 = _make_call("web_search", {"query": "restaurant Berlin Alexanderplatz"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        state.assistant_messages.append(
            "I recommend Mitte Brasserie, it's 1.5km from Alexanderplatz."
        )
        state.final_answer = state.assistant_messages[-1]
        result = sc.evaluate(state)
        # mentions_invalid = True and mentions_valid = False → constraint step fails
        assert result.status in (ScenarioStatus.PARTIAL, ScenarioStatus.FAIL)

    def test_handler_returns_three_restaurants(self):
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        call = _make_call("web_search", {"query": "restaurant Berlin vegan Alexanderplatz"})
        result = sc.handle_tool_call(state, call)
        result_str = str(result)
        assert "Green Kitchen" in result_str
        assert "Mitte Brasserie" in result_str
        assert "Veganz" in result_str

    def test_handler_generic_query_returns_generic_results(self):
        """Queries not about Berlin restaurants get generic results."""
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        call = _make_call("web_search", {"query": "best pizza NYC"})
        result = sc.handle_tool_call(state, call)
        result_str = str(result)
        assert "Green Kitchen" not in result_str

    def test_correctly_rejects_mitte_with_vegan_reasoning(self):
        """Model mentions Mitte Brasserie but correctly rejects it for not being vegan."""
        sc = _get_scenario("TC-73")
        state = ScenarioState()
        c1 = _make_call("web_search", {"query": "restaurant Berlin vegan Alexanderplatz"}, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        state.assistant_messages.append(
            "I found several options. I excluded Mitte Brasserie because it's not vegan. "
            "I recommend Green Kitchen Berlin — it's vegan-friendly and open daily."
        )
        c2 = _make_call("get_contacts", {"query": "Lisa"}, turn=2)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        c3 = _make_call("send_email", {
            "to": "lisa.mueller@company.com",
            "body": "I recommend Green Kitchen Berlin."
        }, turn=3)
        sc.handle_tool_call(state, c3)
        state.tool_calls.append(c3)
        state.final_answer = "Done — emailed Lisa the Green Kitchen recommendation."
        state.assistant_messages.append(state.final_answer)
        result = sc.evaluate(state)
        # Model correctly filtered — mentioning Mitte with rejection should NOT penalize
        assert result.status == ScenarioStatus.PASS


# ===========================================================================
# TC-74: Stateful Multi-Turn Corrections
# ===========================================================================

class TestTC74StatefulCorrections:
    """TC-74: Stateful multi-turn corrections."""

    def test_has_follow_up_messages(self):
        sc = _get_scenario("TC-74")
        assert len(sc.follow_up_messages) == 4

    def test_follow_ups_contain_all_corrections(self):
        """Each follow-up introduces a specific correction."""
        sc = _get_scenario("TC-74")
        fups = [m.lower() for m in sc.follow_up_messages]
        assert any("product review" in f for f in fups)   # title change
        assert any("wednesday" in f for f in fups)         # date change
        assert any("sarah" in f for f in fups)             # new attendee
        assert any("45" in f for f in fups)                # duration change
        assert any("2pm" in f for f in fups)               # time change

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

    def test_partial_some_corrections_lost(self):
        """Model gets title and date right but forgets time and duration."""
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        c1 = _make_call("get_contacts", {"query": "sarah"}, turn=3)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        c2 = _make_call("create_calendar_event", {
            "title": "Product Review",
            "date": "2026-03-25",
            "time": "10:00",  # Forgot the 2pm correction
            "duration_minutes": 30,  # Forgot the 45min correction
        }, turn=5)
        sc.handle_tool_call(state, c2)
        state.tool_calls.append(c2)
        result = sc.evaluate(state)
        # title_ok + date_ok + contacts_searched = 3/5 → PARTIAL
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_no_event(self):
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        state.final_answer = "I'll schedule that for you."
        result = sc.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_fail_only_original_event_no_corrections(self):
        """Model creates initial event and ignores all corrections."""
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        c1 = _make_call("create_calendar_event", {
            "title": "Team Sync",
            "date": "2026-03-24",
            "time": "10:00",
            "duration_minutes": 30,
        }, turn=1)
        sc.handle_tool_call(state, c1)
        state.tool_calls.append(c1)
        result = sc.evaluate(state)
        # 0/5 corrections → FAIL
        assert result.status == ScenarioStatus.FAIL

    def test_handler_contacts_sarah(self):
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        call = _make_call("get_contacts", {"query": "sarah"})
        result = sc.handle_tool_call(state, call)
        assert "Sarah Jones" in str(result)

    def test_handler_contacts_mark(self):
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        call = _make_call("get_contacts", {"query": "mark"})
        result = sc.handle_tool_call(state, call)
        assert "Mark Chen" in str(result)

    def test_handler_contacts_unknown_returns_empty(self):
        sc = _get_scenario("TC-74")
        state = ScenarioState()
        call = _make_call("get_contacts", {"query": "nobody"})
        result = sc.handle_tool_call(state, call)
        assert "results" in str(result) and "[]" in str(result)
