"""Evaluator contract tests — golden traces for TC-01 through TC-15.

Each evaluator is tested with canonical inputs that MUST always produce the
expected scoring tier.  These tests protect scoring semantics from accidental
regression when helpers or evaluator logic is changed.

Fixture coverage per scenario:
  - ``pass``    : canonical correct model behaviour
  - ``fail``    : canonical wrong behaviour
  - ``partial`` : middle-tier where applicable
  - ``paraphrased_refusal`` : alternate phrasing for restraint/refusal scenarios
  - ``wrong_order`` : correct tools, wrong dependency order
  - ``malformed_json_arg`` : common LLM mistake — wrapped value, wrong type
"""

from __future__ import annotations

import json

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    *,
    tool_calls: list[dict] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
    tool_results: list[dict] | None = None,
) -> ScenarioState:
    s = ScenarioState()
    s.final_answer = final_answer
    s.assistant_messages = assistant_messages or ([final_answer] if final_answer else [])
    for tc in tool_calls or []:
        s.tool_calls.append(
            ToolCallRecord(
                id=tc.get("id", f"c{len(s.tool_calls)}"),
                name=tc["name"],
                raw_arguments=json.dumps(tc.get("arguments", {})),
                arguments=tc.get("arguments", {}),
                turn=tc.get("turn", 1),
            )
        )
    for tr in tool_results or []:
        s.tool_results.append(
            ToolResultRecord(
                name=tr["name"],
                result=tr.get("result", {}),
                call_id=tr.get("call_id", "c0"),
            )
        )
    return s


def _get(scenario_id: str):
    return next(s for s in SCENARIOS if s.id == scenario_id)


# ---------------------------------------------------------------------------
# TC-01: Direct Specialist Match
# ---------------------------------------------------------------------------


class TestTC01Contract:
    sc = _get("TC-01")

    def test_pass_weather_only(self):
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            final_answer="8°C, overcast.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_web_search_instead(self):
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "Berlin weather"}}],
            final_answer="Looks cloudy.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_no_tools(self):
        s = _state(final_answer="It's probably cold in Berlin.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_wrong_location(self):
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Paris"}}],
            final_answer="Done.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_extra_web_search(self):
        """Called right tool but also unnecessary web_search → FAIL (not clean)."""
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Berlin"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "weather"}, "turn": 1},
            ],
            final_answer="Berlin is 8°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-02: Distractor Resistance
# ---------------------------------------------------------------------------


class TestTC02Contract:
    sc = _get("TC-02")

    def test_pass_stock_only(self):
        s = _state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
            final_answer="AAPL is $187.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_stock_plus_search(self):
        s = _state(
            tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "AAPL news"}, "turn": 2},
            ],
            final_answer="AAPL $187 and trending up.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_only_web_search(self):
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "AAPL price"}}],
            final_answer="Around $187.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_malformed_ticker_lowercase(self):
        """Common LLM mistake: lowercase ticker — evaluator normalises, should still pass."""
        s = _state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "aapl"}}],
            final_answer="$187.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-03: Implicit Tool Need
# ---------------------------------------------------------------------------


class TestTC03Contract:
    sc = _get("TC-03")

    def test_pass_lookup_then_email(self):
        s = _state(
            tool_calls=[
                {"name": "get_contacts", "arguments": {"query": "Sarah"}, "turn": 1},
                {
                    "name": "send_email",
                    "arguments": {
                        "to": "sarah.chen@company.com",
                        "subject": "Meeting",
                        "body": "The meeting was moved to 3pm.",
                    },
                    "turn": 2,
                },
            ],
            final_answer="Email sent to Sarah.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_wrong_email(self):
        s = _state(
            tool_calls=[
                {"name": "get_contacts", "arguments": {"query": "Sarah"}, "turn": 1},
                {
                    "name": "send_email",
                    "arguments": {
                        "to": "sarah@wrong.com",
                        "subject": "Meeting",
                        "body": "The meeting was moved to 3pm.",
                    },
                    "turn": 2,
                },
            ],
            final_answer="Sent.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_wrong_order_email_before_lookup(self):
        """Email sent before contact lookup — wrong dependency order → FAIL."""
        s = _state(
            tool_calls=[
                {
                    "name": "send_email",
                    "arguments": {
                        "to": "sarah.chen@company.com",
                        "subject": "Meeting",
                        "body": "The meeting was moved to 3pm.",
                    },
                    "turn": 1,
                },
                {"name": "get_contacts", "arguments": {"query": "Sarah"}, "turn": 2},
            ],
            final_answer="Sent.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_partial_asks_for_email(self):
        s = _state(
            final_answer="I can send the email. What is Sarah's email address?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# TC-04: Unit Handling
# ---------------------------------------------------------------------------


class TestTC04Contract:
    sc = _get("TC-04")

    def test_pass_fahrenheit_explicit(self):
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Tokyo", "units": "fahrenheit"}}
            ],
            final_answer="64°F in Tokyo.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_no_units_but_converted(self):
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
            final_answer="Tokyo is 64 Fahrenheit.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_celsius_returned(self):
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Tokyo", "units": "celsius"}}
            ],
            final_answer="18°C in Tokyo.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_malformed_units_string(self):
        """Model passes units as 'Fahrenheit' (mixed case) — normalised → PASS."""
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Tokyo", "units": "Fahrenheit"}}
            ],
            final_answer="64°F.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-05: Date and Time Parsing
# ---------------------------------------------------------------------------


class TestTC05Contract:
    sc = _get("TC-05")

    def test_pass_full_event(self):
        s = _state(
            tool_calls=[
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "Team Standup",
                        "date": "2026-03-23",
                        "time": "09:30",
                        "duration_minutes": 30,
                        "attendees": ["alex.stone@company.com", "jamie.liu@company.com"],
                    },
                }
            ],
            final_answer="Event created.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_correct_date_time_missing_attendees(self):
        s = _state(
            tool_calls=[
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "Standup",
                        "date": "2026-03-23",
                        "time": "09:30",
                        "duration_minutes": 30,
                    },
                }
            ],
            final_answer="Created.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_wrong_date(self):
        s = _state(
            tool_calls=[
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "date": "2026-03-20",
                        "time": "09:30",
                        "duration_minutes": 30,
                    },
                }
            ],
            final_answer="Done.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_pass_timezone_aware_time(self):
        """Timezone-aware ISO datetime should be accepted by flexible matcher."""
        s = _state(
            tool_calls=[
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "date": "2026-03-23",
                        "title": "Team Standup",
                        "time": "09:30:00+01:00",
                        "duration_minutes": 30,
                        "attendees": ["alex.stone@company.com", "jamie.liu@company.com"],
                    },
                }
            ],
            final_answer="Created.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-06: Multi-Value Extraction
# ---------------------------------------------------------------------------


class TestTC06Contract:
    sc = _get("TC-06")

    def test_prompt_requires_tool_use(self):
        """The prompt must require the translate_text tool so correct direct
        answers are not scored against a hidden requirement (forum #187)."""
        assert "translate_text" in self.sc.user_message

    def test_pass_two_separate_calls(self):
        s = _state(
            tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Where is the nearest hospital?",
                        "source_language": "English",
                        "target_language": "Spanish",
                    },
                    "turn": 1,
                },
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Where is the nearest hospital?",
                        "source_language": "English",
                        "target_language": "Japanese",
                    },
                    "turn": 1,
                },
            ],
            final_answer="Spanish: ¿Dónde está el hospital más cercano?\nJapanese: 最寄りの病院はどこですか？",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_only_one_language(self):
        s = _state(
            tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Where is the nearest hospital?",
                        "source_language": "English",
                        "target_language": "Spanish",
                    },
                }
            ],
            final_answer="Spanish: ¿Dónde está el hospital?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_bundled_targets(self):
        """Malformed: target_language with both languages in one string."""
        s = _state(
            tool_calls=[
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Where is the nearest hospital?",
                        "source_language": "English",
                        "target_language": "Spanish and Japanese",
                    },
                }
            ],
            final_answer="Both translations done.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-07: Search → Read → Act
# ---------------------------------------------------------------------------


class TestTC07Contract:
    sc = _get("TC-07")

    def test_pass_all_four_steps(self):
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 Budget Report"}, "turn": 1},
                {"name": "read_file", "arguments": {"file_id": "file_091"}, "turn": 2},
                {"name": "get_contacts", "arguments": {"query": "manager"}, "turn": 3},
                {
                    "name": "send_email",
                    "arguments": {
                        "to": "jordan.park@company.com",
                        "subject": "Budget",
                        "body": "Total: $4.4M",
                    },
                    "turn": 4,
                },
            ],
            final_answer="Email sent.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_three_steps(self):
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 Budget Report"}, "turn": 1},
                {"name": "read_file", "arguments": {"file_id": "file_091"}, "turn": 2},
                {"name": "get_contacts", "arguments": {"query": "manager"}, "turn": 3},
            ],
            final_answer="Found the report. Manager is Jordan Park.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_correct_chain_wrong_total(self):
        """All 4 steps done but wrong body total → 3 steps credit → PARTIAL."""
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 Budget Report"}, "turn": 1},
                {"name": "read_file", "arguments": {"file_id": "file_091"}, "turn": 2},
                {"name": "get_contacts", "arguments": {"query": "manager"}, "turn": 3},
                {
                    "name": "send_email",
                    "arguments": {
                        "to": "jordan.park@company.com",
                        "subject": "Budget",
                        "body": "Total: $5M",
                    },
                    "turn": 4,
                },
            ],
            final_answer="Sent.",
        )
        # body total wrong → send_email check fails → steps=3 → PARTIAL
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_wrong_order_email_before_read(self):
        """Sent email before reading file — wrong dependency order → FAIL."""
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "Q3 Budget Report"}, "turn": 1},
                {
                    "name": "send_email",
                    "arguments": {
                        "to": "jordan.park@company.com",
                        "subject": "Budget",
                        "body": "$4.4M",
                    },
                    "turn": 2,
                },
                {"name": "read_file", "arguments": {"file_id": "file_091"}, "turn": 3},
            ],
            final_answer="Done.",
        )
        # email before read → steps=2 → PARTIAL at best, not PASS
        result = self.sc.evaluate(s)
        assert result.status != ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-08: Conditional Branching
# ---------------------------------------------------------------------------


class TestTC08Contract:
    sc = _get("TC-08")

    def test_pass_weather_then_reminder(self):
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1},
                {
                    "name": "set_reminder",
                    "arguments": {
                        "message": "Bring an umbrella",
                        "datetime": "2026-03-21T08:00:00",
                    },
                    "turn": 2,
                },
            ],
            final_answer="Reminder set.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_weather_but_no_reminder_no_clarification(self):
        """Weather called but no reminder and no clarification → FAIL.

        The TC-08 evaluator only grants PARTIAL if the model asks for
        clarification after reading the weather.  A bare weather check
        without proceeding scores FAIL.
        """
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1}],
            final_answer="It will rain. Let me know if you want a reminder.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_partial_weather_then_asks_clarification(self):
        """PARTIAL: weather check done, model asks clarification instead of acting."""
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1}],
            # "Would you like" doesn't match — use a phrase that does
            final_answer="It's raining in Paris. Could you confirm if I should set the umbrella reminder?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_reminder_without_weather(self):
        s = _state(
            tool_calls=[
                {
                    "name": "set_reminder",
                    "arguments": {
                        "message": "Bring an umbrella",
                        "datetime": "2026-03-21T08:00:00",
                    },
                }
            ],
            final_answer="Reminder set.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_pass_tz_aware_reminder(self):
        """Timezone-aware reminder datetime → flexible match should accept."""
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1},
                {
                    "name": "set_reminder",
                    "arguments": {"message": "umbrella", "datetime": "2026-03-21T08:00:00+01:00"},
                    "turn": 2,
                },
            ],
            final_answer="Done.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-09: Parallel Independence
# ---------------------------------------------------------------------------


class TestTC09Contract:
    sc = _get("TC-09")

    def test_pass_both_tools(self):
        s = _state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "London"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}, "turn": 1},
            ],
            final_answer="London: 12°C. MSFT: $412.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_web_search_fallback(self):
        s = _state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "London weather MSFT price"}}
            ],
            final_answer="London cloudy, MSFT $412.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_only_weather(self):
        s = _state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "London"}}],
            final_answer="London: 12°C.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-10: Trivial Knowledge
# ---------------------------------------------------------------------------


class TestTC10Contract:
    sc = _get("TC-10")

    def test_pass_direct_answer(self):
        s = _state(final_answer="World War II ended in 1945.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_paraphrased_answer(self):
        s = _state(final_answer="The war ended in the year 1945 with Germany's surrender.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_used_web_search(self):
        s = _state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "when did WWII end"}}],
            final_answer="1945.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_wrong_year(self):
        s = _state(final_answer="World War II ended in 1944.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-11: Simple Math
# ---------------------------------------------------------------------------


class TestTC11Contract:
    sc = _get("TC-11")

    def test_pass_mental_math(self):
        s = _state(final_answer="15% of 200 is 30.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_paraphrased_answer(self):
        s = _state(final_answer="That's 30 dollars.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_calculator_correct(self):
        s = _state(
            tool_calls=[{"name": "calculator", "arguments": {"expression": "0.15 * 200"}}],
            final_answer="30.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_wrong_answer(self):
        s = _state(final_answer="15% of 200 is 25.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-12: Impossible Request
# ---------------------------------------------------------------------------


class TestTC12Contract:
    sc = _get("TC-12")

    def test_pass_canonical_refusal(self):
        s = _state(final_answer="I cannot delete your emails — no delete tool is available.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_paraphrased_refusal_with_delete(self):
        """Refusal that mentions 'delete' satisfies the evaluator."""
        s = _state(final_answer="I'm unable to delete your emails — that's not supported.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_paraphrased_refusal_available_tool(self):
        """Refusal that mentions 'available tool' satisfies the evaluator."""
        s = _state(final_answer="I can't do that — no available tool supports email deletion.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_refusal_without_delete_mention(self):
        """Refusal that doesn't mention 'delete' or 'available tool' → FAIL.

        The TC-12 evaluator requires one of those phrases alongside the refusal.
        This is a known evaluator strictness — documented in the contract test.
        """
        s = _state(final_answer="No tool exists for removing emails in the current toolset.")
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_used_send_email(self):
        s = _state(
            tool_calls=[
                {
                    "name": "send_email",
                    "arguments": {"to": "x", "subject": "del", "body": "deleted"},
                }
            ],
            final_answer="Done.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_refusal_mention(self):
        """Vague answer with no refusal language and no tool → FAIL (no delete mention)."""
        s = _state(final_answer="This request has been noted.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-13: Empty Results
# ---------------------------------------------------------------------------


class TestTC13Contract:
    sc = _get("TC-13")

    def test_pass_retry_different_query(self):
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "johnson proposal"}, "turn": 1},
                {"name": "search_files", "arguments": {"query": "johnson project"}, "turn": 2},
            ],
            final_answer="Found on retry.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_asks_clarification(self):
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "johnson proposal"}, "turn": 1}
            ],
            final_answer="No results found. Could you clarify the document name?",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_retry_same_query(self):
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "johnson proposal"}, "turn": 1},
                {"name": "search_files", "arguments": {"query": "johnson proposal"}, "turn": 2},
            ],
            final_answer="Still no results.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_gives_up(self):
        s = _state(
            tool_calls=[
                {"name": "search_files", "arguments": {"query": "johnson proposal"}, "turn": 1}
            ],
            final_answer="No documents found.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-14: Malformed Response (error recovery)
# ---------------------------------------------------------------------------


class TestTC14Contract:
    sc = _get("TC-14")

    def test_pass_acknowledged_and_searched(self):
        s = _state(
            tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "AAPL stock price"}, "turn": 2},
            ],
            tool_results=[
                {
                    "name": "get_stock_price",
                    "result": {"error": "Service temporarily unavailable."},
                    "call_id": "c0",
                },
            ],
            final_answer="The stock service was temporarily unavailable. Web search shows AAPL ~$187.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_searched_but_no_acknowledgement(self):
        s = _state(
            tool_calls=[
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1},
                {"name": "web_search", "arguments": {"query": "AAPL"}, "turn": 2},
            ],
            tool_results=[
                {
                    "name": "get_stock_price",
                    "result": {"error": "Rate limit exceeded."},
                    "call_id": "c0",
                },
            ],
            final_answer="AAPL is around $187.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_partial_acknowledged_no_search(self):
        s = _state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 1}],
            tool_results=[
                {
                    "name": "get_stock_price",
                    "result": {"error": "Service temporarily unavailable."},
                    "call_id": "c0",
                },
            ],
            final_answer="The stock service is temporarily unavailable. I cannot retrieve AAPL price right now.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_no_tool_calls(self):
        s = _state(final_answer="AAPL is $187.42.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_search_before_stock_error(self):
        """web_search called BEFORE stock error — no genuine recovery."""
        s = _state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "AAPL"}, "turn": 1},
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}, "turn": 2},
            ],
            tool_results=[
                {"name": "get_stock_price", "result": {"error": "Rate limit."}, "call_id": "c1"},
            ],
            final_answer="AAPL around $187.",
        )
        # search was before stock error → no genuine recovery credit
        result = self.sc.evaluate(s)
        assert result.status == ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-15: Conflicting Information
# ---------------------------------------------------------------------------


class TestTC15Contract:
    sc = _get("TC-15")

    def test_pass_search_then_calc(self):
        s = _state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "population of Iceland"}, "turn": 1},
                {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}, "turn": 2},
            ],
            final_answer="2% of 372,520 is 7,450.4.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_partial_mental_math_after_search(self):
        s = _state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "Iceland population"}, "turn": 1}
            ],
            final_answer="Population is ~372,520. 2% is 7450.4.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PARTIAL

    def test_fail_skipped_search(self):
        s = _state(
            tool_calls=[
                {"name": "calculator", "arguments": {"expression": "350000 * 0.02"}, "turn": 1}
            ],
            final_answer="7,000.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_fail_no_tools(self):
        s = _state(final_answer="About 7,000 people.")
        assert self.sc.evaluate(s).status == ScenarioStatus.FAIL

    def test_malformed_expression_comma_separator(self):
        """Common LLM: expression with comma as thousands separator → normalised."""
        s = _state(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "Iceland population"}, "turn": 1},
                {"name": "calculator", "arguments": {"expression": "372,520 * 0.02"}, "turn": 2},
            ],
            final_answer="7,450.4.",
        )
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS
