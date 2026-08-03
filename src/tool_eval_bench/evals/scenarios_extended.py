"""Extended scenario packs — localization + ClawEval-inspired tests.

TC-16 to TC-21: adds 6 new scenarios in 2 new categories:
  F — Localization (language handling, timezone awareness)
  G — Structured Reasoning (ClawEval-inspired: routing, state tracking, constraint checking)
"""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)

# ---------------------------------------------------------------------------
# Helpers (shared via evals.helpers)
# ---------------------------------------------------------------------------
from tool_eval_bench.evals.helpers import (
    answer_contains_number as _answer_contains_number,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    has_explicit_tool_error as _has_explicit_tool_error,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    parse_math_expression as _parse_math_expression,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)

# ===================================================================
# TC-16: German Language Tool Call
# Category F — Localization
# ===================================================================


def _tc16_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "München",
                "temperature": 14,
                "units": "celsius",
                "condition": "Teilweise bewölkt",
                "humidity": 58,
            },
            "get_weather",
        )
    if call.name == "translate_text":
        target = _normalize(_as_str(call.arguments.get("target_language")))
        if target in ("german", "de", "deutsch"):
            return _noise(
                {
                    "translated": "Das aktuelle Wetter in München: 14°C, teilweise bewölkt, Luftfeuchtigkeit 58%."
                },
                "translate_text",
            )
        return _noise({"translated": _as_str(call.arguments.get("text"))}, "translate_text")
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "München: 14°C, teilweise bewölkt."}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc16_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asked in German — should use get_weather and respond in German."""
    used_weather = any(
        c.name == "get_weather"
        and (
            _includes_text(c.arguments.get("location"), "münchen")
            or _includes_text(c.arguments.get("location"), "munich")
        )
        for c in state.tool_calls
    )
    # Check that the final answer is in German.
    # Two paths: weather-specific vocabulary (when tool succeeds) or
    # generic German (when tool returns an error and model reports failure).
    answer_lower = state.final_answer.lower()
    weather_german = ["wetter", "grad", "bewölkt", "celsius", "münchen", "temperatur", "°c"]
    generic_german = [
        "tut mir leid",
        "bitte",
        "leider",
        "versuchen",
        "nicht",
        "dienst",
        "erneut",
        "verfügbar",
        "gerade",
        "aktuell",
    ]
    weather_score = sum(1 for w in weather_german if w in answer_lower)
    generic_score = sum(1 for w in generic_german if w in answer_lower)
    german_function_words = sum(
        1
        for word in (" und ", " ist ", " der ", " die ", " das ", " für ", " nicht ", " bitte ")
        if word in f" {answer_lower} "
    )
    answer_in_german = (weather_score >= 3 and german_function_words >= 1) or (
        generic_score >= 2 and german_function_words >= 1
    )
    # Check the model surfaced actual weather data (temp 14 or condition).
    has_data = _answer_contains_number(state.final_answer, "14") or "bewölkt" in answer_lower
    weather_call = next((c for c in state.tool_calls if c.name == "get_weather"), None)
    weather_failed = bool(weather_call and _has_explicit_tool_error(state, weather_call))
    answered_about_error = generic_score >= 2 and weather_failed and not has_data
    if used_weather and answer_in_german and (has_data or answered_about_error):
        return _pass("Used get_weather for München and responded in German.")
    if used_weather and answer_in_german and not has_data:
        return _partial(
            "Used get_weather and responded in German but did not surface the temperature."
        )
    if used_weather and not answer_in_german:
        return _partial("Found the weather correctly but responded in English instead of German.")
    if not used_weather and answer_in_german:
        return _partial("Responded in German but didn't use the weather tool.")
    return _fail(
        "Did not handle the German-language request as a tool call in the expected language."
    )


# ===================================================================
# TC-17: Timezone-Aware Scheduling (Europe/Berlin)
# Category F — Localization
# ===================================================================


def _tc17_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "create_calendar_event":
        return _noise(
            {
                "event_id": "evt_9917",
                "status": "created",
                "title": _as_str(call.arguments.get("title")),
                "date": _as_str(call.arguments.get("date")),
                "time": _as_str(call.arguments.get("time")),
                "timezone": _as_str(call.arguments.get("timezone")),
            },
            "create_calendar_event",
        )
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Hans Müller", "email": "hans.mueller@firma.de"}]}, "get_contacts"
        )
    return _generic_tool_fallback(call)


def _tc17_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Should create event at 14:00 Europe/Berlin, not UTC."""
    event = next((c for c in state.tool_calls if c.name == "create_calendar_event"), None)
    if not event:
        return _fail("Did not create the calendar event.")

    time_val = _as_str(event.arguments.get("time"))
    tz_val = _normalize(_as_str(event.arguments.get("timezone")))
    date_val = _as_str(event.arguments.get("date"))
    title_val = _normalize(_as_str(event.arguments.get("title")))

    correct_time = time_val == "14:00"
    # 2026-03-24 is before the EU DST transition, so CEST is not valid here.
    correct_tz = tz_val in ("europe/berlin", "cet", "utc+1")
    correct_date = date_val == "2026-03-24"  # next Tuesday from reference date (Friday 2026-03-20)
    has_title = "standup" in title_val or "meeting" in title_val or "besprechung" in title_val

    if correct_time and correct_tz and correct_date and has_title:
        return _pass("Scheduled for 14:00 Europe/Berlin on the correct date.")
    if correct_time and correct_date and not correct_tz:
        return _partial(
            "Got the time and date right, but defaulted to UTC instead of Europe/Berlin."
        )
    if correct_time and correct_tz:
        return _partial("Got the time and timezone right, but the date was wrong.")
    return _fail("Did not respect the Europe/Berlin timezone in the scheduling request.")


# ===================================================================
# TC-18: Multilingual Translation Consistency
# Category F — Localization
# ===================================================================


def _tc18_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "translate_text":
        target = _normalize(_as_str(call.arguments.get("target_language")))
        if target in ("german", "de", "deutsch"):
            return _noise(
                {
                    "translated": "Der Termin wurde auf 15 Uhr verschoben. Bitte bestätigen Sie Ihre Teilnahme."
                },
                "translate_text",
            )
        if target in ("english", "en"):
            return _noise(
                {
                    "translated": "The meeting has been moved to 3 PM. Please confirm your attendance."
                },
                "translate_text",
            )
        return _noise({"error": f"Unsupported language: {target}"}, "translate_text")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_1822"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {
                "results": [
                    {"name": "Hans Müller", "email": "hans.mueller@firma.de"},
                    {"name": "Sarah Chen", "email": "sarah.chen@company.com"},
                ]
            },
            "get_contacts",
        )
    return _generic_tool_fallback(call)


def _tc18_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Should translate the message to German and then email it.

    User says: 'Translate "The meeting has been moved to 3 PM. Please confirm
    your attendance." to German and email it to Hans.'
    """
    translate_calls = [c for c in state.tool_calls if c.name == "translate_text"]
    email_calls = [c for c in state.tool_calls if c.name == "send_email"]

    translations = [
        c
        for c in translate_calls
        if _normalize(_as_str(c.arguments.get("target_language"))) in ("german", "de", "deutsch")
    ]
    emails = [
        c
        for c in email_calls
        if _normalize(_as_str(c.arguments.get("to"))) == "hans.mueller@firma.de"
    ]
    translated_to_german = bool(translations)
    sent_email = bool(emails)
    email_has_german = any(
        _includes_text(c.arguments.get("body"), "verschoben")
        or _includes_text(c.arguments.get("body"), "termin")
        for c in emails
    )
    ordered = any(t.turn < e.turn for t in translations for e in emails)
    if translated_to_german and sent_email and email_has_german and ordered:
        return _pass("Translated to German and emailed the German version to Hans.")
    if translated_to_german and sent_email and not ordered:
        return _partial("Translated and emailed the message, but used the steps out of order.")
    if translated_to_german and sent_email and not email_has_german:
        return _partial(
            "Translated correctly but emailed the English version instead of the German one."
        )
    if translated_to_german and not sent_email:
        return _partial("Translated to German but didn't complete the email step.")
    return _fail("Did not translate and email correctly.")


# ===================================================================
# TC-19: Message Routing (ClawEval-inspired — Tier 1 Router role)
# Category G — Structured Reasoning
# ===================================================================


def _tc19_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """No tools needed — tests direct structured output."""
    return _generic_tool_fallback(call)


def _tc19_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks: classify 5 messages into categories. Model should not use tools.

    Expected: no tool calls, just a direct structured classification.
    """
    if len(state.tool_calls) > 0:
        return _fail("Used tools when direct classification was appropriate.")

    answer = state.final_answer.lower()

    # Require structured output (numbered list, bullet points, or clear per-message labeling)
    has_structure = bool(
        re.search(r"(?:^|\n)\s*(?:1[.)\]]|message\s*1)", answer, re.MULTILINE)
        or re.search(r"(?:^|\n)\s*[-•*]", answer, re.MULTILINE)
    )

    # Each label must be attached to the corresponding numbered message. A
    # keyword bag can pass without classifying anything at all.
    numbered_checks = [
        bool(re.search(r"(?:1\s*[.):\-].{0,80})(?:code_help|code|engineering)", answer)),
        bool(re.search(r"(?:2\s*[.):\-].{0,80})(?:scheduling|schedule|calendar)", answer)),
        bool(re.search(r"(?:3\s*[.):\-].{0,80})(?:billing|payment)", answer)),
        bool(re.search(r"(?:4\s*[.):\-].{0,80})(?:devops|deploy)", answer)),
        bool(re.search(r"(?:5\s*[.):\-].{0,80})research", answer)),
    ]
    bullet_lines = re.findall(r"(?:^|\n)\s*[-•*]\s*(.+)", answer)
    expected = [
        ("code_help", "code", "engineering"),
        ("scheduling", "schedule", "calendar"),
        ("billing", "payment"),
        ("devops", "deploy"),
        ("research",),
    ]
    bullet_correct = sum(
        any(label in line for label in labels)
        for line, labels in zip(bullet_lines[:5], expected, strict=False)
    )
    correct = max(sum(numbered_checks), bullet_correct)

    if correct >= 4 and has_structure:
        return _pass("Classified messages correctly in structured format without tool use.")
    if correct >= 4:
        return _partial(
            "Classifications correct but output lacked structured format (no list/labels)."
        )
    if correct >= 3:
        return _partial(f"Got {correct}/5 classifications right.")
    return _fail(f"Only {correct}/5 classifications correct.")


# ===================================================================
# TC-20: Numerical Data Extraction (ClawEval-inspired — data analysis)
# Category G — Structured Reasoning
# ===================================================================


def _tc20_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "read_file":
        return _noise(
            {
                "content": (
                    "Sales Report Q3 2025\n"
                    "Region A: $142,500 (↑12%)\n"
                    "Region B: $98,200 (↓3%)\n"
                    "Region C: $215,800 (↑8%)\n"
                    "Region D: $67,300 (↓15%)\n"
                    "Region E: $183,400 (↑22%)\n"
                    "Total: $707,200\n"
                    "Top performer: Region C\n"
                    "Largest decline: Region D"
                ),
            },
            "read_file",
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_q3_sales", "name": "Q3_Sales_2025.csv"}]}, "search_files"
        )
    if call.name == "calculator":
        result = _parse_math_expression(_as_str(call.arguments.get("expression", "")))
        payload = {"error": "Invalid expression."} if result is None else {"result": result}
        return _noise(payload, "calculator")
    return _generic_tool_fallback(call)


def _tc20_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks to find the file, read it, and calculate the average sales per region.

    Expected: search → read → calculator (or mental math), answer = $141,440
    """
    search = next((c for c in state.tool_calls if c.name == "search_files"), None)
    read = next((c for c in state.tool_calls if c.name == "read_file"), None)
    searched = bool(
        search
        and "q3" in _as_str(search.arguments.get("query")).lower()
        and "sales" in _as_str(search.arguments.get("query")).lower()
    )
    read_correct = bool(read and _as_str(read.arguments.get("file_id")) == "file_q3_sales")
    # Average = 707200 / 5 = 141440
    answer_has_avg = any(
        num in state.final_answer.replace(",", "")
        for num in ("141440", "141,440", "141440.0", "141440.00")
    )

    ordered = bool(searched and read_correct and search and read and search.turn < read.turn)
    if ordered and answer_has_avg:
        return _pass("Found, read, and calculated the correct average ($141,440).")
    if read_correct and answer_has_avg:
        return _partial("Got the right answer but skipped the file search step.")
    if searched and read_correct and not answer_has_avg:
        return _partial("Found and read the file but calculated incorrectly.")
    return _fail("Did not complete the search→read→calculate chain.")


# ===================================================================
# TC-21: Constraint Validation (ClawEval-inspired — input validator)
# Category G — Structured Reasoning
# ===================================================================


def _tc21_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """No tools needed — tests direct reasoning."""
    return _generic_tool_fallback(call)


def _tc21_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User provides a JSON payload with 5 deliberate errors, asks the model to find them all.

    The model should NOT use any tools — just analyze the data directly.
    Expected errors: invalid email, age > 150, phone not 10 digits, invalid date, negative amount.
    """
    if len(state.tool_calls) > 0:
        used_tools = ", ".join(c.name for c in state.tool_calls)
        return _fail(f"Used tools ({used_tools}) when direct analysis was appropriate.")

    answer = state.final_answer.lower()
    error_checks = [
        bool(
            re.search(
                r"email.{0,50}(?:invalid|malformed|bad)|(?:invalid|malformed|bad).{0,50}email",
                answer,
            )
        ),
        bool(
            re.search(
                r"age.{0,50}(?:too high|out of range|over 150)|(?:too high|out of range).{0,50}age",
                answer,
            )
        ),
        bool(
            re.search(
                r"phone.{0,50}(?:invalid|digits?|format)|(?:invalid|digits?|format).{0,50}phone",
                answer,
            )
        ),
        bool(
            re.search(
                r"date.{0,50}(?:invalid|impossible|month 13)|(?:invalid|impossible).{0,50}date",
                answer,
            )
        ),
        bool(
            re.search(
                r"amount.{0,50}(?:negative|below zero|must be positive)|(?:negative|below zero).{0,50}amount",
                answer,
            )
        ),
    ]
    found = sum(error_checks)
    if found >= 4:
        return _pass(f"Identified {found}/5 validation errors without using tools.")
    if found >= 3:
        return _partial(f"Found {found}/5 errors. Missed some validation issues.")
    return _fail(f"Only found {found}/5 validation errors.")


# ===================================================================
# Extended scenario registry
# ===================================================================

EXTENDED_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="TC-16",
        title="German Language Tool Call",
        category=Category.F,
        user_message="Wie ist das Wetter in München gerade?",
        description="Handle a German-language weather request, respond in German.",
        handle_tool_call=_tc16_handle,
        evaluate=_tc16_eval,
        difficulty=2,
    ),
    ScenarioDefinition(
        id="TC-17",
        title="Timezone-Aware Scheduling",
        category=Category.F,
        user_message="Erstelle einen Termin für nächsten Dienstag um 14 Uhr Berliner Zeit. Titel: Team Standup.",
        description="Schedule in Europe/Berlin timezone, not UTC.",
        handle_tool_call=_tc17_handle,
        evaluate=_tc17_eval,
        difficulty=3,
    ),
    ScenarioDefinition(
        id="TC-18",
        title="Translate & Forward",
        category=Category.F,
        user_message='Translate "The meeting has been moved to 3 PM. Please confirm your attendance." to German and email it to Hans.',
        description="Translate to German and email the translated version.",
        handle_tool_call=_tc18_handle,
        evaluate=_tc18_eval,
        difficulty=3,
    ),
    ScenarioDefinition(
        id="TC-19",
        title="Message Routing",
        category=Category.G,
        user_message=(
            "Classify each message into one category (code_help, scheduling, billing, devops, research):\n"
            "1. 'Can you refactor this to use async/await?'\n"
            "2. 'Move my Thursday 3pm to Friday'\n"
            "3. 'I was charged twice for the same subscription'\n"
            "4. 'The Docker container keeps crashing with OOM errors'\n"
            "5. 'Find me the top papers on transformer architectures from 2024'"
        ),
        description="Classify messages without using any tools.",
        handle_tool_call=_tc19_handle,
        evaluate=_tc19_eval,
        difficulty=2,
    ),
    ScenarioDefinition(
        id="TC-20",
        title="Data Extraction & Calculation",
        category=Category.G,
        user_message="Find the Q3 sales report file and tell me the average sales per region.",
        description="Search → read → calculate, result should be $141,440.",
        handle_tool_call=_tc20_handle,
        evaluate=_tc20_eval,
        difficulty=3,
    ),
    ScenarioDefinition(
        id="TC-21",
        title="Constraint Validation",
        category=Category.G,
        user_message=(
            "Check this API payload for errors. List all validation issues:\n"
            '{"email": "john@.com", "age": 200, "phone": "555-12", '
            '"date": "2020-13-45", "amount": -50}'
        ),
        description="Find all 5 validation errors without resorting to tools.",
        handle_tool_call=_tc21_handle,
        evaluate=_tc21_eval,
        difficulty=3,
    ),
]


EXTENDED_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-16": ScenarioDisplayDetail(
        "Pass if it calls get_weather for München and responds in German.",
        "Fail if it responds in English or misses the weather tool.",
    ),
    "TC-17": ScenarioDisplayDetail(
        "Pass if it creates the event at 14:00 with timezone Europe/Berlin for 2026-03-24.",
        "Fail if it uses UTC or gets the date wrong.",
    ),
    "TC-18": ScenarioDisplayDetail(
        "Pass if it translates to German first, then emails the German text to Hans.",
        "Fail if it skips translation or sends the English version.",
    ),
    "TC-19": ScenarioDisplayDetail(
        "Pass if it classifies 4+ messages correctly without using any tools.",
        "Fail if it uses tools or misclassifies most messages.",
    ),
    "TC-20": ScenarioDisplayDetail(
        "Pass if it searches, reads, and computes $141,440 average.",
        "Fail if it invents data or skips the file read.",
    ),
    "TC-21": ScenarioDisplayDetail(
        "Pass if it finds 4+ of the 5 validation errors without tools.",
        "Fail if it uses tools or misses most errors.",
    ),
}
