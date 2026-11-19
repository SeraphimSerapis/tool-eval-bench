"""End-to-end review regression tests for core, extended, and large_toolset scenarios."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import make_state, make_tool_call, simulate_results
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.locales import localized_variant
from tool_eval_bench.evals.variants import apply_variants


def _first_call_errors(scenario_id: str, tool_name: str):
    """Wrap a scenario so the first call to ``tool_name`` returns a transient error."""
    base = SCENARIOS[scenario_id]
    failed = False

    def handle(state: ScenarioState, call: ToolCallRecord):
        nonlocal failed
        if call.name == tool_name and not failed:
            failed = True
            return {"error": "Temporary upstream failure"}
        return base.handle_tool_call(state, call)

    return replace(base, handle_tool_call=handle)


# ---------------------------------------------------------------------------
# TC-01: Direct Specialist Match
# ---------------------------------------------------------------------------


def test_tc01_rejects_negative_temperature_and_accepts_zero_decimal() -> None:
    good = replay("TC-01", turn(("get_weather", {"location": "Berlin"})), turn(answer="8.0°C."))
    assert good.status is ScenarioStatus.PASS

    for wrong in ("Berlin is -8°C.", "Berlin is −8°C."):
        result = replay("TC-01", turn(("get_weather", {"location": "Berlin"})), turn(answer=wrong))
        assert result.status is ScenarioStatus.PARTIAL, wrong


def test_tc01_web_search_partial_requires_usable_berlin_query() -> None:
    good_partial = replay(
        "TC-01",
        turn(("web_search", {"query": "Berlin weather"})),
        turn(answer="Looks overcast."),
    )
    assert good_partial.status is ScenarioStatus.PARTIAL

    unrelated = replay(
        "TC-01",
        turn(("web_search", {"query": "cats"})),
        turn(answer="Berlin is 8°C."),
    )
    assert unrelated.status is ScenarioStatus.FAIL

    errored = replay(
        _first_call_errors("TC-01", "web_search"),
        turn(("web_search", {"query": "Berlin weather"})),
        turn(answer="Search failed."),
    )
    assert errored.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-02: Distractor Resistance
# ---------------------------------------------------------------------------


def test_tc02_rejects_negative_price_and_stray_side_effect_on_partial_path() -> None:
    good = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(answer="AAPL is $187.42."),
    )
    assert good.status is ScenarioStatus.PASS

    negative = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(answer="AAPL is -$187.42."),
    )
    assert negative.status is ScenarioStatus.PARTIAL

    stray_write = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL news"})),
        turn(
            (
                "send_email",
                {"to": "someone@example.com", "subject": "AAPL", "body": "Price $187.42"},
            )
        ),
        turn(answer="AAPL is $187.42."),
    )
    assert stray_write.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-03: Implicit Tool Need
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Hi Sarah, the meeting hasn\u2019t moved to 3pm.",
        "Hi Sarah, the meeting has **not** moved to 3pm.",
        "Hi Sarah, the meeting isn\u2019t **moved** to 3pm.",
    ],
)
def test_tc03_denial_with_curly_apostrophe_or_markdown_cannot_pass(body: str) -> None:
    result = replay(
        "TC-03",
        turn(("get_contacts", {"query": "Sarah"})),
        turn(
            (
                "send_email",
                {"to": "sarah.chen@company.com", "subject": "Meeting", "body": body},
            )
        ),
        turn(answer="Sent."),
    )
    assert result.status is ScenarioStatus.PARTIAL


def test_tc03_unchanged_room_with_curly_apostrophe_still_passes() -> None:
    result = replay(
        "TC-03",
        turn(("get_contacts", {"query": "Sarah"})),
        turn(
            (
                "send_email",
                {
                    "to": "sarah.chen@company.com",
                    "subject": "Meeting moved",
                    "body": "The meeting moved to 3pm. The room hasn\u2019t changed.",
                },
            )
        ),
        turn(answer="Sent."),
    )
    assert result.status is ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# TC-04: Unit Handling
# ---------------------------------------------------------------------------


def test_tc04_rejects_wrong_decimals_negative_celsius_and_stray_writes() -> None:
    good = replay(
        "TC-04",
        turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
        turn(answer="Tokyo is 64.0°F."),
    )
    assert good.status is ScenarioStatus.PASS

    localized_de = localized_variant(SCENARIOS["TC-04"], "de")
    de_ok = replay(
        localized_de,
        turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
        turn(answer="In Tokyo sind es 64,0 °F."),
    )
    assert de_ok.status is ScenarioStatus.PASS

    de_wrong_unit = replay(
        localized_de,
        turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
        turn(answer="In Tokyo sind es 64 Grad Celsius."),
    )
    assert de_wrong_unit.status is ScenarioStatus.PARTIAL

    for wrong in ("Tokyo is 64.9°F.", "Tokyo is -64°F.", "Tokyo is 64.0°C."):
        result = replay(
            "TC-04",
            turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
            turn(answer=wrong),
        )
        assert result.status is ScenarioStatus.PARTIAL, wrong

    stray_write = replay(
        "TC-04",
        turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
        turn(("send_email", {"to": "a@example.com", "subject": "Weather", "body": "64F"})),
        turn(answer="Tokyo is 64°F."),
    )
    assert stray_write.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-05: Date and Time Parsing
# ---------------------------------------------------------------------------


def test_tc05_accepts_hyphenated_standup_and_rejects_negated_attendee() -> None:
    good = replay(
        "TC-05",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Stand-up",
                    "date": "2026-03-23",
                    "time": "09:30",
                    "duration_minutes": 30,
                    "attendees": ["Alex", "Jamie"],
                },
            )
        ),
        turn(answer="Scheduled."),
    )
    assert good.status is ScenarioStatus.PASS

    negated = replay(
        "TC-05",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Standup",
                    "date": "2026-03-23",
                    "time": "09:30",
                    "duration_minutes": 30,
                    "attendees": ["not Alex", "Jamie"],
                },
            )
        ),
        turn(answer="Scheduled."),
    )
    assert negated.status is ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# TC-06: Multi-Value Extraction
# ---------------------------------------------------------------------------


def test_tc06_errored_translation_reports_unusable_result_not_extra_call() -> None:
    result = replay(
        _first_call_errors("TC-06", "translate_text"),
        turn(
            (
                "translate_text",
                {
                    "source_language": "English",
                    "target_language": "Spanish",
                    "text": "Where is the nearest hospital?",
                },
            ),
            (
                "translate_text",
                {
                    "source_language": "English",
                    "target_language": "Japanese",
                    "text": "Where is the nearest hospital?",
                },
            ),
        ),
        turn(answer="¿Dónde está el hospital más cercano? 最寄りの病院はどこですか？"),
    )
    assert result.status is ScenarioStatus.PARTIAL
    assert "unusable" in result.summary
    assert "invalid or extra" not in result.summary


# ---------------------------------------------------------------------------
# TC-07 & TC-38: Budget total precision and TC-38 grounding / retries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["TC-07", "TC-38"])
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("The Q3 budget total is $4.40M.", ScenarioStatus.PASS),
        ("The Q3 budget total is $4.40 million.", ScenarioStatus.PASS),
        ("The Q3 budget total is $4,400,000.00.", ScenarioStatus.PASS),
        ("The Q3 budget total is -$4.4M.", ScenarioStatus.PARTIAL),
        ("The Q3 budget total is $4M.", ScenarioStatus.PARTIAL),
        ("The Q3 budget total is $4.45M.", ScenarioStatus.PARTIAL),
    ],
)
def test_tc07_and_tc38_budget_total_exact_precision(
    scenario_id: str, body: str, expected: ScenarioStatus
) -> None:
    result = replay(
        scenario_id,
        turn(("search_files", {"query": "Q3 budget report"})),
        turn(("read_file", {"file_id": "file_091"})),
        turn(("get_contacts", {"query": "manager"})),
        turn(
            (
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Budget", "body": body},
            )
        ),
        turn(answer="Sent."),
    )
    assert result.status is expected, result.summary


def test_tc38_rejects_negated_queries_and_ungrounded_results_and_supports_fallback() -> None:
    negated = replay(
        "TC-38",
        turn(("search_files", {"query": "not budget"})),
        turn(("read_file", {"file_id": "file_091"})),
        turn(("get_contacts", {"query": "not manager"})),
        turn(
            (
                "send_email",
                {
                    "to": "jordan.park@company.com",
                    "subject": "Budget",
                    "body": "Total: $4.4M",
                },
            )
        ),
        turn(answer="Sent."),
    )
    assert negated.status is ScenarioStatus.FAIL

    ungrounded_state = make_state(
        tool_calls=[
            make_tool_call("search_files", {"query": "Q3 budget report"}, turn=1, call_id="c0"),
            make_tool_call("read_file", {"file_id": "file_091"}, turn=2, call_id="c1"),
            make_tool_call("get_contacts", {"query": "manager"}, turn=3, call_id="c2"),
            make_tool_call(
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Budget", "body": "Total: $4.4M"},
                turn=4,
                call_id="c3",
            ),
        ],
        tool_results=[
            ToolResultRecord("c0", "search_files", {"results": []}),
            ToolResultRecord("c1", "read_file", {"content": "unrelated notes"}),
            ToolResultRecord(
                "c2",
                "get_contacts",
                {"results": [{"name": "Jordan Park", "email": "jordan.park@company.com"}]},
            ),
            ToolResultRecord("c3", "send_email", {"status": "sent"}),
        ],
        final_answer="Sent.",
    )
    assert SCENARIOS["TC-38"].evaluate(ungrounded_state).status is ScenarioStatus.PARTIAL

    fallback = replay(
        _first_call_errors("TC-38", "get_contacts"),
        turn(("search_files", {"query": "Q3 budget report"})),
        turn(("get_contacts", {"query": "manager"})),
        turn(("get_org_chart", {"department": "Engineering"})),
        turn(("read_file", {"file_id": "file_091"})),
        turn(
            (
                "send_email",
                {
                    "to": "jordan.park@company.com",
                    "subject": "Budget",
                    "body": "Total: $4.4M",
                },
            )
        ),
        turn(answer="Sent."),
    )
    assert fallback.status is ScenarioStatus.PASS, fallback.summary


# ---------------------------------------------------------------------------
# TC-08: Conditional Branching (rainy side effects + dry variant phrasing)
# ---------------------------------------------------------------------------


def test_tc08_stray_write_fails_and_extra_read_reports_accurately() -> None:
    stray_write = replay(
        "TC-08",
        turn(("get_weather", {"location": "Paris"})),
        turn(
            (
                "set_reminder",
                {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
            )
        ),
        turn(("send_email", {"to": "a@example.com", "subject": "Rain", "body": "Rain"})),
        turn(answer="Done."),
    )
    assert stray_write.status is ScenarioStatus.FAIL

    extra_read = replay(
        "TC-08",
        turn(("get_weather", {"location": "Paris"})),
        turn(("web_search", {"query": "Paris weather"})),
        turn(
            (
                "set_reminder",
                {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
            )
        ),
        turn(answer="Done."),
    )
    assert extra_read.status is ScenarioStatus.PARTIAL
    assert "duplicate" not in extra_read.summary


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("It is **not** raining in Paris, so I did not set a reminder.", ScenarioStatus.PASS),
        ("It isn\u2019t currently raining in Paris.", ScenarioStatus.PASS),
        ("It is not currently raining in Paris, so no reminder needed.", ScenarioStatus.PASS),
        ("It is not raining yet, but rain is expected later.", ScenarioStatus.FAIL),
        ("It is raining in Paris.", ScenarioStatus.FAIL),
    ],
)
def test_tc08_dry_variant_markdown_and_adverb_phrasing(
    answer: str, expected: ScenarioStatus
) -> None:
    dry_scenario = apply_variants([SCENARIOS["TC-08"]], seed=1)[0]
    result = replay(
        dry_scenario,
        turn(("get_weather", {"location": "Paris"})),
        turn(answer=answer),
    )
    assert result.status is expected, f"{answer!r} -> {result.summary}"


# ---------------------------------------------------------------------------
# TC-09: Parallel Independence
# ---------------------------------------------------------------------------


def test_tc09_rejects_negative_values_and_stray_side_effects() -> None:
    for wrong in (
        "London is -12°C and MSFT is $412.78.",
        "London is 12°C and MSFT is -$412.78.",
    ):
        result = replay(
            "TC-09",
            turn(
                ("get_weather", {"location": "London"}),
                ("get_stock_price", {"ticker": "MSFT"}),
            ),
            turn(answer=wrong),
        )
        assert result.status is ScenarioStatus.PARTIAL, wrong

    stray_write = replay(
        "TC-09",
        turn(("web_search", {"query": "London weather MSFT"})),
        turn(("send_email", {"to": "a@example.com", "subject": "FYI", "body": "12C"})),
        turn(answer="London is 12°C and MSFT is $412.78."),
    )
    assert stray_write.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-10: Trivial Knowledge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("**1945**", ScenarioStatus.PASS),
        ("It was **1945**.", ScenarioStatus.PASS),
        ("WW2: 1945", ScenarioStatus.PASS),
        ("World War II ended in 1945.5.", ScenarioStatus.FAIL),
        ("World War II ended in -1945.", ScenarioStatus.FAIL),
        ("**1944**", ScenarioStatus.FAIL),
    ],
)
def test_tc10_markdown_terse_year_and_number_boundaries(
    answer: str, expected: ScenarioStatus
) -> None:
    result = replay("TC-10", turn(answer=answer))
    assert result.status is expected, f"{answer!r} -> {result.summary}"


# ---------------------------------------------------------------------------
# TC-11 & TC-39: Simple Math / Restraint Under Abundance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["TC-11", "TC-39"])
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("15% of 200 is 30.0.", ScenarioStatus.PASS),
        ("15% of 200 is 30.9.", ScenarioStatus.FAIL),
        ("15% of 200 is -30.", ScenarioStatus.FAIL),
    ],
)
def test_tc11_and_tc39_exact_numeric_boundaries(
    scenario_id: str, answer: str, expected: ScenarioStatus
) -> None:
    result = replay(scenario_id, turn(answer=answer))
    assert result.status is expected, f"{scenario_id} {answer!r} -> {result.summary}"


def test_tc11_stray_side_effect_alongside_calculator_fails() -> None:
    result = replay(
        "TC-11",
        turn(("calculator", {"expression": "0.15 * 200"})),
        turn(("run_code", {"language": "python", "code": "print(30)"})),
        turn(answer="15% of 200 is 30."),
    )
    assert result.status is ScenarioStatus.FAIL


def test_tc39_calculator_mock_evaluates_expression_instead_of_hardcoding_30() -> None:
    tc39 = SCENARIOS["TC-39"]
    wrong_expr = tc39.handle_tool_call(
        ScenarioState(), make_tool_call("calculator", {"expression": "15 * 200"})
    )
    assert wrong_expr["result"] == 3000.0

    invalid_expr = tc39.handle_tool_call(
        ScenarioState(), make_tool_call("calculator", {"expression": "not math"})
    )
    assert "error" in invalid_expr

    right_expr = tc39.handle_tool_call(
        ScenarioState(), make_tool_call("calculator", {"expression": "200 * 0.15"})
    )
    assert right_expr["result"] == 30.0


# ---------------------------------------------------------------------------
# TC-13: Empty Results
# ---------------------------------------------------------------------------


def test_tc13_clarification_requires_relevant_initial_search() -> None:
    good = replay(
        "TC-13",
        turn(("search_files", {"query": "proposal"})),
        turn(answer="No results came back. Could you clarify the exact file name?"),
    )
    assert good.status is ScenarioStatus.PASS

    unrelated = replay(
        "TC-13",
        turn(("search_files", {"query": "cats"})),
        turn(answer="No results came back. Could you clarify the exact file name?"),
    )
    assert unrelated.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-14: Malformed Response
# ---------------------------------------------------------------------------


def test_tc14_price_ticker_and_error_acknowledgement_boundaries() -> None:
    wrong_decimal = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL stock price"})),
        turn(answer="The stock tool failed. Web search says AAPL is $187.99."),
    )
    assert wrong_decimal.status is ScenarioStatus.PARTIAL

    negative_price = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL stock price"})),
        turn(answer="The stock tool failed. Web search says AAPL is -$187.42."),
    )
    assert negative_price.status is ScenarioStatus.PARTIAL

    wrong_ticker = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "MSFT"})),
        turn(("web_search", {"query": "Apple stock price"})),
        turn(answer="The stock tool failed. Web search says AAPL is $187.42."),
    )
    assert wrong_ticker.status is ScenarioStatus.FAIL

    bare_tool_name = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL stock price"})),
        turn(answer="I checked get_stock_price and web search: AAPL is $187.42."),
    )
    assert bare_tool_name.status is ScenarioStatus.PARTIAL

    negated_markdown = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL stock price"})),
        turn(answer="No **errors** occurred. AAPL is $187.42."),
    )
    assert negated_markdown.status is ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# TC-15: Conflicting Information
# ---------------------------------------------------------------------------


def test_tc15_underscore_expression_passes_and_negated_or_wrong_decimal_fails_pass() -> None:
    underscore_ok = replay(
        "TC-15",
        turn(("web_search", {"query": "population of Iceland"})),
        turn(("calculator", {"expression": "372_520 * 0.02"})),
        turn(answer="2% of 372,520 is 7,450.4."),
    )
    assert underscore_ok.status is ScenarioStatus.PASS

    for wrong in (
        "2% of Iceland's population is not 7,450.4.",
        "2% of Iceland's population is 7,450.99.",
        "2% of Iceland's population is -7,450.4.",
    ):
        result = replay(
            "TC-15",
            turn(("web_search", {"query": "population of Iceland"})),
            turn(("calculator", {"expression": "372520 * 0.02"})),
            turn(answer=wrong),
        )
        assert result.status is ScenarioStatus.PARTIAL, wrong

    negated_mental = replay(
        "TC-15",
        turn(("web_search", {"query": "population of Iceland"})),
        turn(answer="The answer is not 7450.4."),
    )
    assert negated_mental.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-16: German Language Tool Call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Das Wetter in München liegt bei 14,0 Grad.", ScenarioStatus.PASS),
        ("Das Wetter in München liegt bei 14.0 °C.", ScenarioStatus.PASS),
        ("Das Wetter in München liegt bei -14 Grad.", ScenarioStatus.PARTIAL),
        ("Das Wetter in München liegt bei 14,9 Grad.", ScenarioStatus.PARTIAL),
        ("Das Wetter in München liegt bei 14.5 Grad.", ScenarioStatus.PARTIAL),
        ("Das Wetter in München liegt bei 2.14 Grad.", ScenarioStatus.PARTIAL),
        ("Das Wetter in München liegt bei 14,0 °F.", ScenarioStatus.PARTIAL),
    ],
)
def test_tc16_temperature_decimal_and_sign_boundaries(
    answer: str, expected: ScenarioStatus
) -> None:
    result = replay(
        "TC-16",
        turn(("get_weather", {"location": "München"})),
        turn(answer=answer),
    )
    assert result.status is expected, f"{answer!r} -> {result.summary}"


# ---------------------------------------------------------------------------
# TC-17: Timezone-Aware Scheduling
# ---------------------------------------------------------------------------


def test_tc17_hyphenated_title_and_accurate_partial_summaries() -> None:
    good = replay(
        "TC-17",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Stand-up",
                    "date": "2026-03-24",
                    "time": "14:00",
                    "timezone": "Europe/Berlin",
                },
            )
        ),
        turn(answer="Erstellt."),
    )
    assert good.status is ScenarioStatus.PASS

    wrong_title = replay(
        "TC-17",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Wöchentlicher Sync",
                    "date": "2026-03-24",
                    "time": "14:00",
                    "timezone": "Europe/Berlin",
                },
            )
        ),
        turn(answer="Erstellt."),
    )
    assert wrong_title.status is ScenarioStatus.PARTIAL
    assert "title" in wrong_title.summary
    assert "date was wrong" not in wrong_title.summary

    wrong_tz = replay(
        "TC-17",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Standup",
                    "date": "2026-03-24",
                    "time": "14:00",
                    "timezone": "America/New_York",
                },
            )
        ),
        turn(answer="Erstellt."),
    )
    assert wrong_tz.status is ScenarioStatus.PARTIAL
    assert "defaulted to UTC" not in wrong_tz.summary
    assert "america/new_york" in wrong_tz.summary


# ---------------------------------------------------------------------------
# TC-19: Message Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        '{"message_1": "code_help", "message_2": "scheduling", "message_3": "billing", "message_4": "devops", "message_5": "research"}',
        '{"msg_1": "code_help", "msg_2": "scheduling", "msg_3": "billing", "msg_4": "devops", "msg_5": "research"}',
        (
            "**Classifications:**\n"
            "**Message 1:** code_help\n"
            "**Message 2:** scheduling\n"
            "**Message 3:** billing\n"
            "**Message 4:** devops\n"
            "**Message 5:** research"
        ),
    ],
)
def test_tc19_accepts_snake_case_json_and_bold_message_headings(answer: str) -> None:
    result = replay("TC-19", turn(answer=answer))
    assert result.status is ScenarioStatus.PASS, result.summary


def test_tc19_rejects_negated_snake_case_json() -> None:
    answer = (
        '{"message_1": "not code_help", "message_2": "not scheduling", '
        '"message_3": "not billing", "message_4": "not devops", "message_5": "research"}'
    )
    result = replay("TC-19", turn(answer=answer))
    assert result.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# TC-20: Data Extraction & Calculation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The average sales per region is $141,440.00.", ScenarioStatus.PASS),
        ("The average sales per region is not $141,440.", ScenarioStatus.PARTIAL),
        ("The average sales per region is $141,440.99.", ScenarioStatus.PARTIAL),
        ("The average sales per region is -$141,440.", ScenarioStatus.PARTIAL),
    ],
)
def test_tc20_average_negation_and_precision(answer: str, expected: ScenarioStatus) -> None:
    result = replay(
        "TC-20",
        turn(("search_files", {"query": "Q3 sales"})),
        turn(("read_file", {"file_id": "file_q3_sales"})),
        turn(answer=answer),
    )
    assert result.status is expected, f"{answer!r} -> {result.summary}"


# ---------------------------------------------------------------------------
# TC-37: Needle in a Haystack
# ---------------------------------------------------------------------------


def test_tc37_validates_location_units_result_and_temperature_precision() -> None:
    negated_loc = replay(
        "TC-37",
        turn(("get_weather", {"location": "not Berlin"})),
        turn(answer="Berlin is 8°C and overcast."),
    )
    assert negated_loc.status is ScenarioStatus.FAIL

    bad_units = replay(
        "TC-37",
        turn(("get_weather", {"location": "Berlin", "units": "metric"})),
        turn(answer="Berlin is 8°C and overcast."),
    )
    assert bad_units.status is ScenarioStatus.FAIL

    wrong_payload = make_state(
        tool_calls=[make_tool_call("get_weather", {"location": "Berlin"}, call_id="c0")],
        tool_results=[
            ToolResultRecord(
                "c0", "get_weather", {"location": "Berlin", "temperature": -4, "condition": "Snow"}
            )
        ],
        final_answer="Berlin is 8°C and overcast.",
    )
    assert SCENARIOS["TC-37"].evaluate(wrong_payload).status is ScenarioStatus.PARTIAL

    for wrong in ("Berlin is 8.9°C and overcast.", "Berlin is -8°C and overcast."):
        result = replay("TC-37", turn(("get_weather", {"location": "Berlin"})), turn(answer=wrong))
        assert result.status is ScenarioStatus.PARTIAL, wrong


# ---------------------------------------------------------------------------
# TC-40: Domain Confusion
# ---------------------------------------------------------------------------


def test_tc40_rejects_substring_and_negated_names_and_ungrounded_order_results() -> None:
    for bogus in ("kitchen", "not Sarah"):
        result = replay(
            "TC-40",
            turn(("get_order_status", {"order_id": bogus})),
            turn(answer="Sarah Chen's order has shipped."),
        )
        assert result.status is ScenarioStatus.PARTIAL, bogus
        assert "does not support" in result.summary

    cancelled_state = make_state(
        tool_calls=[make_tool_call("get_order_status", {"order_id": "Sarah Chen"}, call_id="c0")],
        tool_results=[
            ToolResultRecord(
                "c0",
                "get_order_status",
                {"order_id": "ORD-9999", "customer": "Bob", "status": "cancelled"},
            )
        ],
        final_answer="Sarah Chen's order has shipped.",
    )
    assert SCENARIOS["TC-40"].evaluate(cancelled_state).status is ScenarioStatus.PARTIAL

    none_orders_state = make_state(
        tool_calls=[
            make_tool_call(
                "get_customer_profile", {"customer_id": "Sarah Chen"}, turn=1, call_id="c0"
            ),
            make_tool_call("get_order_status", {"order_id": "ORD-2026-1847"}, turn=2, call_id="c1"),
        ],
        tool_results=[
            ToolResultRecord("c0", "get_customer_profile", {"name": "Sarah Chen", "orders": None}),
            ToolResultRecord(
                "c1",
                "get_order_status",
                {"order_id": "ORD-2026-1847", "customer": "Sarah Chen", "status": "shipped"},
            ),
        ],
        final_answer="Sarah Chen's order has shipped.",
    )
    assert SCENARIOS["TC-40"].evaluate(none_orders_state).status is ScenarioStatus.PARTIAL

    two_step_ok = simulate_results(
        make_state(
            tool_calls=[
                make_tool_call("get_customer_profile", {"customer_id": "Sarah Chen"}, turn=1),
                make_tool_call("get_order_status", {"order_id": "ORD-2026-1847"}, turn=2),
            ],
            final_answer="Sarah Chen's order ORD-2026-1847 has shipped.",
        ),
        SCENARIOS["TC-40"],
    )
    assert SCENARIOS["TC-40"].evaluate(two_step_ok).status is ScenarioStatus.PASS
