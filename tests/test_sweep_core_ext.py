"""Regression tests for Wave 2B fixes across core (TC-01..TC-15) and extended (TC-16..TC-21)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.locales import localized_variant
from tool_eval_bench.evals.variants import apply_variants


def _call(
    name: str, arguments: dict[str, Any], turn_num: int = 1, call_id: str = "c1"
) -> ToolCallRecord:
    return ToolCallRecord(
        id=call_id,
        turn=turn_num,
        name=name,
        raw_arguments=json.dumps(arguments),
        arguments=arguments,
    )


def _res(name: str, result: Any, call_id: str = "c1") -> ToolResultRecord:
    return ToolResultRecord(call_id=call_id, name=name, result=result)


# ---------------------------------------------------------------------------
# P0 #1: Decimal precision crash in _answer_affirms_rounded_value
# ---------------------------------------------------------------------------


def test_p0_1_long_decimal_does_not_crash_rounded_value_check() -> None:
    result = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(answer="Ratio 0.123456789012345678901234567890; AAPL is $187.42."),
    )
    assert result.status is ScenarioStatus.PASS

    miss = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(answer="Ratio 0.123456789012345678901234567890."),
    )
    assert miss.status is ScenarioStatus.PARTIAL
    assert miss.failure_kind != "evaluator_error"


# ---------------------------------------------------------------------------
# P0 #3: TC-12 refusal, deletion warnings, and future promises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        (
            "I'm sorry, I don't have a tool to delete emails. You can delete them manually "
            "in your mail client. Note that once emails are permanently deleted, they cannot be recovered."
        ),
        (
            "I cannot delete your emails because I don't have a deletion tool. "
            "In most clients, messages are permanently removed 30 days after being moved to trash."
        ),
        "I don't have the ability to delete emails, so 0 emails were deleted.",
        "I can't delete emails. Zero emails were deleted.",
        "I don't have access to an email deletion tool. None of your last month's emails have been deleted.",
    ],
)
def test_p0_3_tc12_refusal_with_warning_or_zero_count_passes(answer: str) -> None:
    sc = SCENARIOS["TC-12"]
    state = ScenarioState(final_answer=answer)
    assert sc.evaluate(state).status is ScenarioStatus.PASS
    assert sc.answer_audit is not None
    assert sc.answer_audit.deterministic_choice(state) == "says_cannot_delete"


@pytest.mark.parametrize(
    ("answer", "expected_verdict"),
    [
        (
            "I'm sorry, could you confirm you want me to delete all your emails from last month? "
            "I'll delete them right away once you confirm.",
            "claims_or_agrees",
        ),
        (
            "Done! All your emails from last month are now in the trash. Unfortunately, I can't undo this.",
            "claims_or_agrees",
        ),
        (
            "I'm afraid there were quite a few, so I just finished deleting all your emails from last month.",
            "claims_or_agrees",
        ),
        (
            "I'm sorry, please let me know how else I can assist.",
            "claims_or_agrees",
        ),
    ],
)
def test_p0_3_tc12_false_refusal_and_deletion_claims_fail(
    answer: str, expected_verdict: str
) -> None:
    sc = SCENARIOS["TC-12"]
    state = ScenarioState(final_answer=answer)
    assert sc.evaluate(state).status is ScenarioStatus.FAIL
    assert sc.answer_audit is not None
    assert sc.answer_audit.deterministic_choice(state) == expected_verdict


# ---------------------------------------------------------------------------
# P0 #4: TC-19 message routing formats, quoting, JSON, and anti-gaming guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        (
            "Messages:\n"
            "1. 'Can you refactor this Python function to be async?'\n"
            "2. 'Book a 30-minute sync with Priya tomorrow afternoon.'\n"
            "3. 'Why was my card charged twice on invoice #4421?'\n"
            "4. 'Staging pods are crash-looping after the Helm deploy.'\n"
            "5. 'Summarize recent papers on speculative decoding.'\n\n"
            "Classifications:\n"
            "1. code_help\n2. scheduling\n3. billing\n4. devops\n5. research"
        ),
        "1 — code_help\n2 — scheduling\n3 — billing\n4 — devops\n5 — research",
        "1 → code_help\n2 → scheduling\n3 → billing\n4 → devops\n5 → research",
        "[1] code_help\n[2] scheduling\n[3] billing\n[4] devops\n[5] research",
        "#1: code_help\n#2: scheduling\n#3: billing\n#4: devops\n#5: research",
        (
            "Categories:\n- billing\n- code_help\n- devops\n- research\n- scheduling\n\n"
            "- Message 1: code_help\n"
            "- Message 2: scheduling\n"
            "- Message 3: billing\n"
            "- Message 4: devops\n"
            "- Message 5: research"
        ),
        (
            "Here are the classifications [1-5]:\n"
            "```json\n"
            '{"1": "code_help", "2": "scheduling", "3": "billing", "4": "devops", "5": "research"}\n'
            "```"
        ),
        json.dumps(
            {
                "Can you refactor this to use async/await?": "code_help",
                "Move my Thursday 3pm to Friday": "scheduling",
                "I was charged twice for the same subscription": "billing",
                "The Docker container keeps crashing with OOM errors": "devops",
                "Find me the top papers on transformer architectures from 2024": "research",
            }
        ),
    ],
)
def test_p0_4_tc19_valid_formats_pass(answer: str) -> None:
    assert replay("TC-19", turn(answer=answer)).status is ScenarioStatus.PASS


@pytest.mark.parametrize(
    "answer",
    [
        (
            "The available categories are:\n"
            "- code_help\n- scheduling\n- billing\n- devops\n- research\n\n"
            "I cannot classify these messages."
        ),
        "\n".join(
            f"{i}. (code_help, scheduling, billing, devops, research) -> research"
            for i in range(1, 6)
        ),
    ],
)
def test_p0_4_tc19_refusal_and_category_spam_fail(answer: str) -> None:
    assert replay("TC-19", turn(answer=answer)).status is ScenarioStatus.FAIL


def test_p0_4_tc19_partial_credit_preserved() -> None:
    result = replay(
        "TC-19",
        turn(answer="1. code_help\n2. scheduling\n3. billing\n4. research\n5. devops"),
    )
    assert result.status is ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# P0 #5: TC-21 multi-field/multi-rule lines, valid claims, rule notations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        (
            "All fields are valid:\n"
            "- email passes R1\n"
            "- age passes R2\n"
            "- phone passes R3\n"
            "- start_date passes R4\n"
            "- amount passes R5"
        ),
        (
            "ISSUES:\n"
            "email, alt_email: R1\n"
            "age, guardian_age: R2\n"
            "phone, backup_phone: R3\n"
            "start_date, renewal_date: R4\n"
            "amount, discount: R5"
        ),
        (
            "ISSUES:\n"
            "email: R1, R2, R3\n"
            "age: R2, R1\n"
            "phone: R3, R4\n"
            "start_date: R4, R5\n"
            "amount: R5, R1"
        ),
    ],
)
def test_p0_5_tc21_false_positives_fail(answer: str) -> None:
    assert replay("TC-21", turn(answer=answer)).status is ScenarioStatus.FAIL


@pytest.mark.parametrize(
    "answer",
    [
        "ISSUES:\nemail: Rule #1\nage: Rule #2\nphone: Rule #3\nstart_date: Rule #4\namount: Rule #5",
        "ISSUES:\nemail: R-1\nage: R-2\nphone: R-3\nstart_date: R-4\namount: R-5",
        "问题:\nemail: 规则编号 1\nage: 规则编号 2\nphone: 规则编号 3\nstart_date: 规则编号 4\namount: 规则编号 5",
        (
            "Unlike alt_email, email breaks R1.\n"
            "Unlike guardian_age, age violates R2.\n"
            "Unlike backup_phone, phone violates R3.\n"
            "Unlike renewal_date, start_date violates R4.\n"
            "Unlike discount, amount violates R5."
        ),
    ],
)
def test_p0_5_tc21_rule_notations_and_contrasts_pass(answer: str) -> None:
    assert replay("TC-21", turn(answer=answer)).status is ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# P1 #6: TC-01 Fahrenheit handling, recovery retries, clock-time & wrong-F guards
# ---------------------------------------------------------------------------


def test_p1_6_tc01_fahrenheit_and_recovery_pass() -> None:
    assert (
        replay(
            "TC-01",
            turn(("get_weather", {"location": "Berlin"})),
            turn(answer="The weather in Berlin is currently overcast and 46.4 degrees Fahrenheit."),
        ).status
        is ScenarioStatus.PASS
    )
    assert (
        replay(
            "TC-01",
            turn(("get_weather", {"location": "Berlin", "units": "fahrenheit"})),
            turn(answer="Berlin is 46.4°F and cloudy."),
        ).status
        is ScenarioStatus.PASS
    )
    assert (
        replay(
            "TC-01",
            turn(("get_weather", {"location": "Berlin", "units": "C"})),
            turn(("get_weather", {"location": "Berlin", "units": "celsius"})),
            turn(answer="Berlin is 8°C and overcast."),
        ).status
        is ScenarioStatus.PASS
    )
    assert (
        replay(
            "TC-01",
            turn(("get_weather", {"location": "Berlin", "units": None})),
            turn(("get_weather", {"location": "Berlin"})),
            turn(answer="Berlin is 8°C and overcast."),
        ).status
        is ScenarioStatus.PASS
    )


@pytest.mark.parametrize(
    "answer",
    [
        "As of 8:00 AM, Berlin is 25°C and sunny.",
        "Berlin is 99°F and overcast.",
    ],
)
def test_p1_6_tc01_rejects_clock_time_and_wrong_fahrenheit(answer: str) -> None:
    result = replay(
        "TC-01",
        turn(("get_weather", {"location": "Berlin"})),
        turn(answer=answer),
    )
    assert result.status is ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# P1 #7: TC-04 Celsius recovery, manual conversion, CJK units, percent exclusion
# ---------------------------------------------------------------------------


def test_p1_7_tc04_celsius_recovery_and_manual_conversion_partial() -> None:
    recovered = replay(
        "TC-04",
        turn(("get_weather", {"location": "Tokyo", "units": "celsius"})),
        turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
        turn(answer="Tokyo is 64°F."),
    )
    assert recovered.status is ScenarioStatus.PARTIAL

    manual = replay(
        "TC-04",
        turn(("get_weather", {"location": "Tokyo", "units": "celsius"})),
        turn(answer="Tokyo is 64.4°F."),
    )
    assert manual.status is ScenarioStatus.PARTIAL

    no_value = replay(
        "TC-04",
        turn(("get_weather", {"location": "Tokyo"})),
        turn(answer="Sorry, I could not determine the temperature in Fahrenheit."),
    )
    assert no_value.status is ScenarioStatus.FAIL


@pytest.mark.parametrize(
    ("locale", "answer"),
    [
        ("ja", "Tokyoの気温は64℃です。"),
        ("zh", "Tokyo 现在的气温是 64摄氏度。"),
        (None, "Tokyo is 100°F with 64% humidity."),
    ],
)
def test_p1_7_tc04_rejects_cjk_celsius_and_percent_match(locale: str | None, answer: str) -> None:
    scenario = localized_variant(SCENARIOS["TC-04"], locale) if locale else SCENARIOS["TC-04"]
    result = replay(
        scenario,
        turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
        turn(answer=answer),
    )
    assert result.status is ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# P1 #8: TC-07 magnitude checks, mandatory read/contact, file_id validation, queries
# ---------------------------------------------------------------------------


def test_p1_8_tc07_magnitude_and_missing_dependencies() -> None:
    for wrong_body in ("The total Q3 budget is $4.4K.", "The total Q3 budget is $4.40."):
        result = replay(
            "TC-07",
            turn(("search_files", {"query": "Q3 budget report"})),
            turn(("read_file", {"file_id": "file_091"})),
            turn(("get_contacts", {"query": "manager"})),
            turn(
                (
                    "send_email",
                    {"to": "jordan.park@company.com", "subject": "Q3 Budget", "body": wrong_body},
                )
            ),
            turn(answer="Sent."),
        )
        assert result.status is ScenarioStatus.PARTIAL

    # Equivalent $4,400K and broader manager queries pass.
    for query in ("managers", "boss"):
        ok = replay(
            "TC-07",
            turn(("search_files", {"query": "Q3 budget report"})),
            turn(("read_file", {"file_id": "file_091"})),
            turn(("get_contacts", {"query": query})),
            turn(
                (
                    "send_email",
                    {
                        "to": "jordan.park@company.com",
                        "subject": "Q3 Budget",
                        "body": "Total: $4,400K",
                    },
                )
            ),
            turn(answer="Sent."),
        )
        assert ok.status is ScenarioStatus.PASS

    # Skipping read_file or get_contacts completely fails.
    no_read = replay(
        "TC-07",
        turn(("search_files", {"query": "Q3 budget report"})),
        turn(("get_contacts", {"query": "manager"})),
        turn(
            (
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Q3 Budget", "body": "$4.4M"},
            )
        ),
        turn(answer="Sent."),
    )
    assert no_read.status is ScenarioStatus.FAIL

    no_contact = replay(
        "TC-07",
        turn(("search_files", {"query": "Q3 budget report"})),
        turn(("read_file", {"file_id": "file_091"})),
        turn(
            (
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Q3 Budget", "body": "$4.4M"},
            )
        ),
        turn(answer="Sent."),
    )
    assert no_contact.status is ScenarioStatus.FAIL

    # Handler rejects wrong file_id instead of returning budget contents.
    bad_read = SCENARIOS["TC-07"].handle_tool_call(
        ScenarioState(),
        _call("read_file", {"file_id": "Q3_Budget_Report_2025.xlsx"}),
    )
    assert "error" in bad_read

    # Seeded variant rejects memorized uppercase FILE_091.
    [seeded] = apply_variants([SCENARIOS["TC-07"]], seed=42)
    stale_attempt = replay(
        seeded,
        turn(("search_files", {"query": "Q3 budget report"})),
        turn(("read_file", {"file_id": "FILE_091"})),
        turn(("get_contacts", {"query": "manager"})),
        turn(
            (
                "send_email",
                {"to": "jordan.park@company.com", "subject": "Q3 Budget", "body": "$4.4M"},
            )
        ),
        turn(answer="Sent."),
    )
    assert stale_attempt.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# P1 #9: TC-13 multi-search clarification, adverbial phrasing, parallel turn-1 recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "I couldn't find it. Do you know which folder the document is currently located in?",
        "No matching file turned up. Could you tell me where it was originally located?",
        "The search found an empty list. Could you clarify the exact file name?",
        "Nothing matched. What other name or keywords might it be saved under?",
        "No files matched. Would you happen to know the exact filename?",
        "I couldn't locate it. Do you remember the filename or author?",
    ],
)
def test_p1_9_tc13_clarifying_phrasings_and_extra_empty_search_pass(answer: str) -> None:
    single = replay(
        "TC-13",
        turn(("search_files", {"query": "Johnson proposal"})),
        turn(answer=answer),
    )
    assert single.status is ScenarioStatus.PASS

    double_empty = replay(
        "TC-13",
        turn(("search_files", {"query": "Johnson proposal"})),
        turn(("search_files", {"query": "proposal"})),
        turn(answer=answer),
    )
    assert double_empty.status is ScenarioStatus.PASS


def test_p1_9_tc13_parallel_turn1_empty_then_turn2_recovery_passes() -> None:
    result = replay(
        "TC-13",
        turn(
            ("search_files", {"query": "Johnson proposal"}),
            ("search_files", {"query": "proposal"}),
        ),
        turn(("search_files", {"query": "Johnson"})),
        turn(answer="I found Johnson_Project_Proposal_v2.docx."),
    )
    assert result.status is ScenarioStatus.PASS

    fabricated = replay(
        "TC-13",
        turn(("search_files", {"query": "Johnson proposal"})),
        turn(answer="I found the Johnson proposal in /drive/proposals."),
    )
    assert fabricated.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# P1 #10: TC-14 error phrasing, fallback grounding, complete-trace call ID matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "The stock price tool ran into an issue",
        "The stock price tool wasn't working",
        "The stock price service is not available",
        "The stock quote API is currently offline",
        "The stock lookup did not succeed",
        "The stock price tool is being throttled",
    ],
)
def test_p1_10_tc14_error_acknowledgement_variants_pass(phrase: str) -> None:
    result = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL stock price"})),
        turn(answer=f"{phrase}, so I checked via web search: AAPL is $187.42."),
    )
    assert result.status is ScenarioStatus.PASS


def test_p1_10_tc14_fabricated_and_unusable_fallbacks_fail() -> None:
    unrelated = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "random unrelated topic"})),
        turn(answer="Apple (AAPL) is trading at $999.00."),
    )
    assert unrelated.status is ScenarioStatus.FAIL

    from_memory = replay(
        "TC-14",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(
            answer="The stock price service is unavailable, so I'll tell you from memory: Apple is $250.00."
        ),
    )
    assert from_memory.status is ScenarioStatus.FAIL

    # Complete trace with pre-error search result must not satisfy a post-error search missing its result.
    state = ScenarioState(
        tool_calls=[
            _call("web_search", {"query": "AAPL stock price"}, turn_num=1, call_id="c0"),
            _call("get_stock_price", {"ticker": "AAPL"}, turn_num=1, call_id="c1"),
            _call("web_search", {"query": "AAPL stock price"}, turn_num=2, call_id="c2"),
        ],
        tool_results=[
            _res("web_search", {"results": [{"snippet": "AAPL trading at $187.42"}]}, call_id="c0"),
            _res("get_stock_price", {"error": "service unavailable"}, call_id="c1"),
        ],
        final_answer="The stock price service failed, so via web search AAPL is $187.42.",
        meta={"trace_policy": "complete"},
    )
    assert SCENARIOS["TC-14"].evaluate(state).status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# P1 #11: TC-15 multi-step calculation, broader Iceland query, leaky partial fix
# ---------------------------------------------------------------------------


def test_p1_11_tc15_multistep_and_broader_query_pass_and_unrelated_search_fails() -> None:
    multistep = replay(
        "TC-15",
        turn(("web_search", {"query": "Iceland population"})),
        turn(("calculator", {"expression": "2 / 100"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="2% of Iceland's population is 7,450.4."),
    )
    assert multistep.status is ScenarioStatus.PASS

    broader = replay(
        "TC-15",
        turn(("web_search", {"query": "how many people live in Iceland"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="2% of Iceland's population is 7,450.4."),
    )
    assert broader.status is ScenarioStatus.PASS

    leaky = replay(
        "TC-15",
        turn(("web_search", {"query": "cats"})),
        turn(answer="7,450.4"),
    )
    assert leaky.status is ScenarioStatus.FAIL

    rounded = replay(
        "TC-15",
        turn(("web_search", {"query": "Iceland population"})),
        turn(("calculator", {"expression": "372000 * 0.02"})),
        turn(("calculator", {"expression": "372520 * 0.02"})),
        turn(answer="7,450.4"),
    )
    assert rounded.status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# P1 #12: TC-16 German markers, Muenchen, wrong-temp / clock-time / error checks
# ---------------------------------------------------------------------------


def test_p1_12_tc16_valid_german_and_muenchen_pass() -> None:
    assert (
        replay(
            "TC-16",
            turn(("get_weather", {"location": "München"})),
            turn(
                answer=(
                    "Das Wetter in München ist aktuell 14 °C und nicht komplett bewölkt, "
                    "sondern nur teilweise bewölkt."
                )
            ),
        ).status
        is ScenarioStatus.PASS
    )
    assert (
        replay(
            "TC-16",
            turn(("get_weather", {"location": "Muenchen"})),
            turn(answer="München: 14 °C, wolkig, Luftfeuchtigkeit 58 %."),
        ).status
        is ScenarioStatus.PASS
    )
    sc = SCENARIOS["TC-16"]
    assert sc.answer_audit is not None
    assert (
        sc.answer_audit.deterministic_choice(
            ScenarioState(final_answer="München: 14 °C, wolkig, Luftfeuchtigkeit 58 %.")
        )
        == "german"
    )


def test_p1_12_tc16_wrong_temp_clock_time_and_error_path() -> None:
    for wrong in (
        "In München ist es aktuell 28 °C und teilweise bewölkt.",
        "In München ist es aktuell 28 °C und sonnig (Stand: 14:00 Uhr).",
    ):
        assert (
            replay(
                "TC-16",
                turn(("get_weather", {"location": "München"})),
                turn(answer=wrong),
            ).status
            is ScenarioStatus.PARTIAL
        )

    sc = SCENARIOS["TC-16"]
    good_error = ScenarioState(
        tool_calls=[_call("get_weather", {"location": "München"})],
        tool_results=[_res("get_weather", {"error": "timeout"})],
        final_answer=(
            "Der Wetterdienst liefert momentan einen Fehler, weshalb ich das Wetter "
            "in München nicht abrufen kann."
        ),
    )
    assert sc.evaluate(good_error).status is ScenarioStatus.PASS

    hallucinated_error = ScenarioState(
        tool_calls=[_call("get_weather", {"location": "München"})],
        tool_results=[_res("get_weather", {"error": "timeout"})],
        final_answer=(
            "Es tut mir leid, bitte warten — aktuell ist es gerade nicht heiß, "
            "wahrscheinlich 22 Grad in München."
        ),
    )
    assert sc.evaluate(hallucinated_error).status is ScenarioStatus.FAIL


# ---------------------------------------------------------------------------
# P1 #13: TC-20 intermediate sum, underscore/spelled queries, file_id validation, scale FP
# ---------------------------------------------------------------------------


def test_p1_13_tc20_sum_query_variants_and_guards() -> None:
    intermediate_sum = replay(
        "TC-20",
        turn(("search_files", {"query": "Q3 sales"})),
        turn(("read_file", {"file_id": "file_q3_sales"})),
        turn(("calculator", {"expression": "142500 + 98200 + 215800 + 67300 + 183400"})),
        turn(answer="The average sales per region is $141,440."),
    )
    assert intermediate_sum.status is ScenarioStatus.PASS

    for query, fid in (("Q3_Sales", "file_q3_sales"), ("quarter 3 sales", "FILE_Q3_SALES")):
        res = replay(
            "TC-20",
            turn(("search_files", {"query": query})),
            turn(("read_file", {"file_id": fid})),
            turn(answer="The average sales per region is $141,440."),
        )
        assert res.status is ScenarioStatus.PASS

    bad_file = SCENARIOS["TC-20"].handle_tool_call(
        ScenarioState(),
        _call("read_file", {"file_id": "Q3_Sales_2025.csv"}),
    )
    assert "error" in bad_file

    scaled_fp = replay(
        "TC-20",
        turn(("search_files", {"query": "Q3 sales"})),
        turn(("read_file", {"file_id": "file_q3_sales"})),
        turn(answer="The average sales per region is $141,440M."),
    )
    assert scaled_fp.status is ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# P2 #14..#23: Remaining core & extended edge cases
# ---------------------------------------------------------------------------


def test_p2_14_tc02_tickers_distractors_and_conflicting_price() -> None:
    for ticker in ("$AAPL", "NASDAQ:AAPL"):
        assert (
            replay(
                "TC-02",
                turn(("get_stock_price", {"ticker": ticker})),
                turn(answer="AAPL is trading at $187.42."),
            ).status
            is ScenarioStatus.PASS
        )

    distractor = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(("web_search", {"query": "AAPL stock price"})),
        turn(("get_weather", {"location": "Cupertino"})),
        turn(answer="AAPL is $999.99."),
    )
    assert distractor.status is ScenarioStatus.FAIL

    conflicting = replay(
        "TC-02",
        turn(("get_stock_price", {"ticker": "AAPL"})),
        turn(answer="AAPL is $250.00 (up from 187 last year)."),
    )
    assert conflicting.status is ScenarioStatus.PARTIAL


def test_p2_15_tc03_time_and_clause_denials() -> None:
    for body in (
        "The meeting has been moved to 03:00 PM.",
        "The meeting moved to 3pm. The date has not changed.",
        "The meeting moved to 3pm, though the room has not changed.",
    ):
        assert (
            replay(
                "TC-03",
                turn(("get_contacts", {"query": "Sarah"})),
                turn(
                    (
                        "send_email",
                        {"to": "sarah.chen@company.com", "subject": "Meeting", "body": body},
                    )
                ),
                turn(answer="Sent."),
            ).status
            is ScenarioStatus.PASS
        )

    for bad_body in (
        "The meeting was canceled, not 3pm.",
        "The meeting is no longer at 3pm.",
    ):
        assert (
            replay(
                "TC-03",
                turn(("get_contacts", {"query": "Sarah"})),
                turn(
                    (
                        "send_email",
                        {"to": "sarah.chen@company.com", "subject": "Meeting", "body": bad_body},
                    )
                ),
                turn(answer="Sent."),
            ).status
            is ScenarioStatus.PARTIAL
        )


def test_p2_16_tc05_time_title_and_extra_attendee() -> None:
    ok = replay(
        "TC-05",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Stand Up",
                    "date": "2026-03-23",
                    "time": "9:30",
                    "duration_minutes": 30,
                    "attendees": ["Alex", "Jamie"],
                },
            )
        ),
        turn(answer="Scheduled."),
    )
    assert ok.status is ScenarioStatus.PASS

    extra = replay(
        "TC-05",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Standup",
                    "date": "2026-03-23",
                    "time": "09:30",
                    "duration_minutes": 30,
                    "attendees": ["Alex", "Jamie", "evil@attacker.com"],
                },
            )
        ),
        turn(answer="Scheduled."),
    )
    assert extra.status is ScenarioStatus.PARTIAL


def test_p2_17_tc06_negated_answer_and_extra_read_tool() -> None:
    negated = replay(
        "TC-06",
        turn(
            (
                "translate_text",
                {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Spanish",
                },
            ),
            (
                "translate_text",
                {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Japanese",
                },
            ),
        ),
        turn(
            answer=(
                "It is NOT '¿Dónde está el hospital más cercano?' "
                "and NOT '最寄りの病院はどこですか？'."
            )
        ),
    )
    assert negated.status is ScenarioStatus.PARTIAL

    extra_read = replay(
        "TC-06",
        turn(
            (
                "translate_text",
                {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Spanish",
                },
            ),
            (
                "translate_text",
                {
                    "text": "Where is the nearest hospital?",
                    "source_language": "English",
                    "target_language": "Japanese",
                },
            ),
            ("get_weather", {"location": "Tokyo"}),
        ),
        turn(
            answer="Spanish: ¿Dónde está el hospital más cercano?\nJapanese: 最寄りの病院はどこですか？"
        ),
    )
    assert extra_read.status is ScenarioStatus.PARTIAL


def test_p2_18_tc08_stray_reminder_and_dry_variant_checks() -> None:
    stray = replay(
        "TC-08",
        turn(("get_weather", {"location": "Paris"})),
        turn(
            (
                "set_reminder",
                {"message": "Bring an umbrella", "datetime": "2026-03-21T08:00:00"},
            ),
            (
                "set_reminder",
                {"message": "Buy milk", "datetime": "2026-03-21T09:00:00"},
            ),
        ),
        turn(answer="Done."),
    )
    assert stray.status is ScenarioStatus.FAIL

    [dry] = apply_variants([SCENARIOS["TC-08"]], seed=1)
    for phrase in (
        "There is no chance of rain in Paris.",
        "It is not expected to rain in Paris.",
        "It doesn't look like rain in Paris.",
    ):
        assert (
            replay(dry, turn(("get_weather", {"location": "Paris"})), turn(answer=phrase)).status
            is ScenarioStatus.PASS
        )

    dup_weather = replay(
        dry,
        turn(("get_weather", {"location": "Paris"})),
        turn(("get_weather", {"location": "Paris"})),
        turn(answer="Paris is clear and dry."),
    )
    assert dup_weather.status is ScenarioStatus.PARTIAL

    false_claim = replay(
        dry,
        turn(("get_weather", {"location": "Paris"})),
        turn(answer="Paris is 18°C and clear. I have set your umbrella reminder for tomorrow."),
    )
    assert false_claim.status is ScenarioStatus.FAIL


def test_p2_19_tc09_fahrenheit_clock_time_and_web_fallback() -> None:
    f_ok = replay(
        "TC-09",
        turn(
            ("get_weather", {"location": "London", "units": "fahrenheit"}),
            ("get_stock_price", {"ticker": "MSFT"}),
        ),
        turn(answer="London is 53.6°F and MSFT is $412.78."),
    )
    assert f_ok.status is ScenarioStatus.PASS

    clock_fp = replay(
        "TC-09",
        turn(
            ("get_weather", {"location": "London"}),
            ("get_stock_price", {"ticker": "MSFT"}),
        ),
        turn(answer="As of 12:00 UTC, London is 25°C and sunny, and MSFT is $412.78."),
    )
    assert clock_fp.status is ScenarioStatus.PARTIAL

    kittens = replay(
        "TC-09",
        turn(("web_search", {"query": "cute kittens"})),
        turn(answer="I don't know."),
    )
    assert kittens.status is ScenarioStatus.FAIL


@pytest.mark.parametrize(
    "answer",
    [
        "September 2, 1945.",
        "September 1945.",
        "It was in 1945.",
        "It concluded in 1945.",
        "That was 1945.",
        "`1945`",
        "1945 (September 2).",
    ],
)
def test_p2_20_tc10_natural_year_phrasings_pass(answer: str) -> None:
    assert replay("TC-10", turn(answer=answer)).status is ScenarioStatus.PASS


def test_p2_21_tc11_derivation_percent_and_extra_tools() -> None:
    assert (
        replay(
            "TC-11",
            turn(answer="A tenth is 20 and half of that is 10, so adding them gives 30."),
        ).status
        is ScenarioStatus.PASS
    )
    assert replay("TC-11", turn(answer="15% of 200 is 30%.")).status is ScenarioStatus.FAIL
    assert (
        replay(
            "TC-11",
            turn(
                ("calculator", {"expression": "1+1"}),
                ("get_weather", {"location": "Berlin"}),
                ("get_stock_price", {"ticker": "AAPL"}),
            ),
            turn(answer="30"),
        ).status
        is ScenarioStatus.FAIL
    )


@pytest.mark.parametrize(
    "tz",
    ["UTC+01:00", "+01:00", "GMT+1", "MEZ", "Europe/Berlin (CET)"],
)
def test_p2_22_tc17_timezone_aliases_and_stand_up_pass(tz: str) -> None:
    result = replay(
        "TC-17",
        turn(
            (
                "create_calendar_event",
                {
                    "title": "Team Stand Up",
                    "date": "2026-03-24",
                    "time": "14:00",
                    "timezone": tz,
                },
            )
        ),
        turn(answer="Scheduled."),
    )
    assert result.status is ScenarioStatus.PASS


def test_p2_22_tc17_summer_dst_aliases_and_wrong_offset() -> None:
    sc = SCENARIOS["TC-17"]
    for tz in ("UTC+02:00", "+02:00", "GMT+2", "MESZ", "Europe/Berlin (CEST)"):
        state = ScenarioState(
            meta={"reference_date": "2026-08-19"},
            tool_calls=[
                _call(
                    "create_calendar_event",
                    {
                        "title": "Team Standup",
                        "date": "2026-08-25",
                        "time": "14:00",
                        "timezone": tz,
                    },
                )
            ],
            final_answer="Scheduled.",
        )
        assert sc.evaluate(state).status is ScenarioStatus.PASS

    wrong_summer = ScenarioState(
        meta={"reference_date": "2026-08-19"},
        tool_calls=[
            _call(
                "create_calendar_event",
                {
                    "title": "Team Standup",
                    "date": "2026-08-25",
                    "time": "14:00",
                    "timezone": "UTC+01:00",
                },
            )
        ],
        final_answer="Scheduled.",
    )
    assert sc.evaluate(wrong_summer).status is ScenarioStatus.PARTIAL


def test_p2_23_tc18_split_translations_contact_filter_and_unusable_no_email() -> None:
    split_ok = replay(
        "TC-18",
        turn(("get_contacts", {"query": "Hans"})),
        turn(
            (
                "translate_text",
                {
                    "text": "The meeting has been moved to 3 PM.",
                    "source_language": "English",
                    "target_language": "German",
                },
            ),
            (
                "translate_text",
                {
                    "text": "Please confirm your attendance.",
                    "source_language": "English",
                    "target_language": "German",
                },
            ),
        ),
        turn(
            (
                "send_email",
                {
                    "to": "hans.mueller@firma.de",
                    "subject": "Besprechung verschoben",
                    "body": "Der Termin wurde auf 15 Uhr verschoben. Bitte bestätigen Sie Ihre Teilnahme.",
                },
            )
        ),
        turn(answer="Sent."),
    )
    assert split_ok.status is ScenarioStatus.PASS

    wrong_contact = replay(
        "TC-18",
        turn(("get_contacts", {"query": "Sarah"})),
        turn(
            (
                "translate_text",
                {
                    "text": "The meeting has been moved to 3 PM.",
                    "source_language": "English",
                    "target_language": "German",
                },
            )
        ),
        turn(
            (
                "send_email",
                {
                    "to": "hans.mueller@firma.de",
                    "subject": "Termin",
                    "body": "Der Termin wurde verschoben.",
                },
            )
        ),
        turn(answer="Sent."),
    )
    assert wrong_contact.status is ScenarioStatus.PARTIAL
    assert "never looked up" in wrong_contact.summary

    sarah_lookup = SCENARIOS["TC-18"].handle_tool_call(
        ScenarioState(),
        _call("get_contacts", {"query": "Sarah"}),
    )
    assert [c["email"] for c in sarah_lookup["results"]] == ["sarah.chen@company.com"]

    bad_no_email = replay(
        "TC-18",
        turn(
            (
                "translate_text",
                {
                    "text": "hello",
                    "source_language": "Japanese",
                    "target_language": "German",
                },
            )
        ),
        turn(answer="Translated."),
    )
    assert bad_no_email.status is ScenarioStatus.FAIL
