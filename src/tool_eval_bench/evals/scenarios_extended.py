"""Extended scenario packs — localization + ClawEval-inspired tests.

TC-16 to TC-21: adds 6 new scenarios in 2 new categories:
  F — Localization (language handling, timezone awareness)
  G — Structured Reasoning (ClawEval-inspired: routing, state tracking, constraint checking)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    answer_affirms_text as _answer_affirms_text,
)
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
    matching_tool_results as _matching_tool_results,
)

# ---------------------------------------------------------------------------
# Helpers (shared via evals.helpers)
# ---------------------------------------------------------------------------
from tool_eval_bench.evals.helpers import (
    next_weekday_after_reference as _next_weekday_after_reference,
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
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    utc_offset_aliases as _utc_offset_aliases,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _positive_argument_contains(value: Any, expected: str) -> bool:
    """Match a requested entity without accepting an explicit negation."""
    text = _as_str(value).strip().lower()
    target = expected.strip().lower()
    if not text or not re.search(rf"(?<!\w){re.escape(target)}(?!\w)", text):
        return False
    return not re.search(
        rf"\b(?:not|no|without|except|exclude|excluding)\s+(?:the\s+)?"
        rf"{re.escape(target)}\b",
        text,
    )


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _as_str(value).strip().replace(",", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def _status_is_success(payload: Any, *, statuses: set[str] | None = None) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    return _normalize(_as_str(payload.get("status"))) in (
        statuses
        or {
            "accepted",
            "created",
            "delivered",
            "ok",
            "queued",
            "scheduled",
            "sent",
            "set",
            "success",
        }
    )


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Callable[[Any], bool],
) -> bool:
    """Validate a result when present, retaining synthetic trace compatibility."""
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results:
        results = exact_results
    elif any(result.call_id in known_call_ids for result in state.tool_results):
        return False
    else:
        results = _matching_tool_results(state, call)
    if not results:
        return True
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


# ===================================================================
# TC-16: German Language Tool Call
# Category F — Localization
# ===================================================================

# German words with no English homograph. Deliberately excludes "die", "hat",
# "in", "bei" spellings that double as English words.
_GERMAN_MARKERS = re.compile(
    r"\b(?:bewölkt\w*|wetter|grad|beträgt|liegt|derzeit|aktuell|heute|morgen|himmel|"
    r"regen|sonnig|schnee|und|ist|sind|es|im|der|das|für|nicht|bitte|leider|wird|"
    r"gerade|verfügbar|erneut|versuchen|dienst)\b",
    re.IGNORECASE,
)


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


def _tc16_weather_result_is_munich(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    if "location" in payload and not (
        _positive_argument_contains(payload.get("location"), "münchen")
        or _positive_argument_contains(payload.get("location"), "munich")
    ):
        return False
    return _numeric_value(payload.get("temperature")) == 14 or _includes_text(
        payload.get("condition"), "bewölkt"
    )


def _tc16_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asked in German — should use get_weather and respond in German."""
    matching_weather_calls = [
        c
        for c in state.tool_calls
        if c.name == "get_weather"
        and (
            _positive_argument_contains(c.arguments.get("location"), "münchen")
            or _positive_argument_contains(c.arguments.get("location"), "munich")
        )
    ]
    used_weather = bool(matching_weather_calls)
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
    # "temperatur", "celsius", "münchen" and "°c" all appear verbatim in English
    # answers, so the vocabulary score alone cannot prove the reply is German.
    # Require one marker that has no English reading — but keep the marker set
    # wide enough that a terse, fully German answer ("München hat aktuell 14°C
    # bei bewölktem Himmel.") still qualifies.
    german_markers = len(_GERMAN_MARKERS.findall(answer_lower))
    answer_in_german = (weather_score >= 3 or generic_score >= 2) and german_markers >= 1
    # Check the model surfaced actual weather data (temp 14 or condition).
    has_data = _answer_contains_number(state.final_answer, "14") or "bewölkt" in answer_lower
    weather_success = any(
        _result_matches_if_present(state, call, _tc16_weather_result_is_munich)
        for call in matching_weather_calls
    )
    weather_failed = any(_has_explicit_tool_error(state, call) for call in matching_weather_calls)
    answered_about_error = generic_score >= 2 and weather_failed and not has_data
    if used_weather and answer_in_german and answered_about_error:
        if generic_score >= 4:
            return _pass("Used get_weather and clearly reported the service error in German.")
        return _partial(
            "Reported the weather tool error in German, but did not receive weather data."
        )
    if used_weather and answer_in_german and has_data and weather_success:
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


def _tc17_event_result_is_created(payload: Any) -> bool:
    return _status_is_success(payload, statuses={"accepted", "created", "ok", "success"})


def _tc17_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Should create event at 14:00 Europe/Berlin, not UTC."""
    event_calls = [c for c in state.tool_calls if c.name == "create_calendar_event"]
    event = event_calls[0] if event_calls else None
    if not event:
        return _fail("Did not create the calendar event.")
    if not _result_matches_if_present(state, event, _tc17_event_result_is_created):
        return _partial("The calendar mutation did not return a successful creation result.")

    time_val = _as_str(event.arguments.get("time"))
    tz_val = _normalize(_as_str(event.arguments.get("timezone")))
    date_val = _as_str(event.arguments.get("date"))
    title_val = _normalize(_as_str(event.arguments.get("title")))

    # "nächsten Dienstag" is relative to the run's reference date.
    expected_date = _next_weekday_after_reference(state, "tuesday")

    correct_time = time_val == "14:00"
    # Which offset spelling is correct depends on whether the target date falls
    # inside EU summer time, so derive the accepted aliases from the date rather
    # than assuming the March default.
    correct_tz = tz_val == "europe/berlin" or tz_val in _utc_offset_aliases(
        expected_date, "Europe/Berlin"
    )
    correct_date = date_val == expected_date
    has_title = any(
        _positive_argument_contains(title_val, title)
        for title in ("standup", "meeting", "besprechung")
    )

    if correct_time and correct_tz and correct_date and has_title:
        if len(event_calls) > 1:
            return _partial("Created the requested event but issued duplicate calendar mutations.")
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


def _tc18_translation_result_is_german(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    translated = _as_str(payload.get("translated"))
    return _includes_text(translated, "verschoben") and _includes_text(translated, "termin")


def _tc18_email_result_is_sent(payload: Any) -> bool:
    return _status_is_success(
        payload,
        statuses={"accepted", "delivered", "ok", "queued", "sent", "success"},
    )


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
    valid_translations = [
        c
        for c in translations
        if _normalize(_as_str(c.arguments.get("source_language"))) in ("english", "en")
        and _positive_argument_contains(c.arguments.get("text"), "meeting")
    ]
    emails = [
        c
        for c in email_calls
        if _normalize(_as_str(c.arguments.get("to"))) == "hans.mueller@firma.de"
    ]
    usable_translations = [
        call
        for call in valid_translations
        if _result_matches_if_present(state, call, _tc18_translation_result_is_german)
    ]
    usable_emails = [
        call
        for call in emails
        if _result_matches_if_present(state, call, _tc18_email_result_is_sent)
    ]
    translated_to_german = bool(translations)
    # Keep attempted, correctly addressed mutations visible to the partial
    # outcome.  Passing still requires a usable result and complete arguments.
    sent_email = bool(emails)
    email_has_german = any(
        _answer_affirms_text(_as_str(c.arguments.get("body")), "verschoben")
        or _answer_affirms_text(_as_str(c.arguments.get("body")), "termin")
        for c in usable_emails
    )
    ordered = any(t.turn < e.turn for t in usable_translations for e in usable_emails)
    if translated_to_german and sent_email and email_has_german and ordered:
        if len(translations) != 1 or len(emails) != 1 or len(email_calls) != 1:
            return _partial(
                "Translated and emailed the message but issued duplicate or incorrect email mutations."
            )
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


_TC19_MARKER = re.compile(
    r"(?:^|\n)[^\S\n]*\|?[^\S\n]*(?:message[^\S\n]*)?\*{0,2}([1-5])\*{0,2}[^\S\n]*[.):\-|]"
)


def _tc19_segments(answer: str) -> dict[int, str]:
    """Map each message number to the text the answer attaches to it."""
    markers = [(int(m.group(1)), m.start(), m.end()) for m in _TC19_MARKER.finditer(answer)]
    segments: dict[int, str] = {}
    for position, (index, _, end) in enumerate(markers):
        stop = markers[position + 1][1] if position + 1 < len(markers) else len(answer)
        segments.setdefault(index, answer[end:stop])
    return segments


def _tc19_label_is_asserted(segment: str, label: str) -> bool:
    """Match a category label as an assertion, not ``not <label>``."""
    return _answer_affirms_text(segment, label)


def _tc19_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks: classify 5 messages into categories. Model should not use tools.

    Expected: no tool calls, just a direct structured classification.
    """
    if len(state.tool_calls) > 0:
        return _fail("Used tools when direct classification was appropriate.")

    answer = state.final_answer.lower()

    # Require structured output (numbered list, bullets, table, or per-message labeling)
    has_structure = bool(
        re.search(r"(?:^|\n)\s*(?:1[.)\]]|message\s*1)", answer, re.MULTILINE)
        or re.search(r"(?:^|\n)\s*[-•*|]", answer, re.MULTILINE)
    )

    expected = [
        ("code_help", "code", "engineering"),
        ("scheduling", "schedule", "calendar"),
        ("billing", "payment"),
        ("devops", "deploy"),
        ("research",),
    ]
    # Each label must be attached to the corresponding message. A keyword bag
    # can pass without classifying anything at all — but the layout carrying
    # that association is the model's choice, so segment on the message marker
    # rather than assuming one list style. This covers "1.", "1)", "Message 1:",
    # "**1**" and markdown table rows alike.
    segments = _tc19_segments(answer)
    numbered_correct = sum(
        any(_tc19_label_is_asserted(segments[index], label) for label in labels)
        for index, labels in enumerate(expected, start=1)
        if index in segments
    )
    bullet_lines = re.findall(r"(?:^|\n)\s*[-•*]\s*(.+)", answer)
    bullet_correct = sum(
        any(_tc19_label_is_asserted(line, label) for label in labels)
        for line, labels in zip(bullet_lines[:5], expected, strict=False)
    )
    correct = max(numbered_correct, bullet_correct)

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


def _tc20_search_result_has_sales_file(payload: Any) -> bool:
    if (
        not isinstance(payload, dict)
        or "error" in payload
        or not isinstance(payload.get("results"), list)
    ):
        return False
    return any(
        isinstance(item, dict)
        and (
            _normalize(_as_str(item.get("file_id"))) == "file_q3_sales"
            or (
                _includes_text(item.get("name"), "q3") and _includes_text(item.get("name"), "sales")
            )
        )
        for item in payload["results"]
    )


def _tc20_read_result_has_sales_report(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    content = _as_str(payload.get("content"))
    return _includes_text(content, "707,200") or _includes_text(content, "707200")


def _tc20_calculator_result_is_average(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    value = _numeric_value(payload.get("result"))
    return value is not None and abs(value - 141440) < 1e-9


def _tc20_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks to find the file, read it, and calculate the average sales per region.

    Expected: search → read → calculator (or mental math), answer = $141,440
    """
    search = next((c for c in state.tool_calls if c.name == "search_files"), None)
    read = next((c for c in state.tool_calls if c.name == "read_file"), None)
    searched = bool(
        search
        and _positive_argument_contains(search.arguments.get("query"), "q3")
        and _positive_argument_contains(search.arguments.get("query"), "sales")
    )
    read_correct = bool(read and _as_str(read.arguments.get("file_id")) == "file_q3_sales")
    search_result_usable = bool(
        search and _result_matches_if_present(state, search, _tc20_search_result_has_sales_file)
    )
    read_result_usable = bool(
        read and _result_matches_if_present(state, read, _tc20_read_result_has_sales_report)
    )
    calculator_calls = [c for c in state.tool_calls if c.name == "calculator"]
    calculator_usable = True
    calculator_after_read = True
    if calculator_calls:
        calculator = calculator_calls[0]
        calculator_usable = _result_matches_if_present(
            state, calculator, _tc20_calculator_result_is_average
        )
        calculator_after_read = bool(read and read.turn < calculator.turn)
    # Average = 707200 / 5 = 141440
    answer_has_avg = _answer_contains_number(state.final_answer, "141440")

    ordered = bool(
        searched
        and read_correct
        and search
        and read
        and search.turn < read.turn
        and search_result_usable
        and read_result_usable
        and calculator_usable
        and calculator_after_read
    )
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


def _tc21_asserts_issue(answer: str, field: str, issue_pattern: str) -> bool:
    """Find an asserted validation issue, not a quoted or negated mention."""
    for clause in re.split(r"[.!?;\n]", answer):
        if not re.search(rf"\b{field}\b", clause, re.IGNORECASE):
            continue
        match = re.search(issue_pattern, clause, re.IGNORECASE)
        if not match:
            continue
        before = clause[: match.start()]
        negation = re.search(r"\b(?:not|never|no)\b(?:\s+\w+){0,4}\s*$", before, re.IGNORECASE)
        if negation and not re.search(
            r"(?:\b(?:valid|correct|acceptable)\b|\bwithin\s+range\b)",
            negation.group(),
            re.IGNORECASE,
        ):
            continue
        positive_quality_claim = False
        for quality in re.finditer(
            r"(?:\b(?:valid|correct|acceptable)\b|\bwithin\s+range\b)",
            clause,
            re.IGNORECASE,
        ):
            quality_before = clause[: quality.start()]
            if not re.search(
                r"(?:\b(?:not|never|no)\b(?:\s+\w+){0,2}|n't)\s*$",
                quality_before,
                re.IGNORECASE,
            ):
                positive_quality_claim = True
                break
        if positive_quality_claim:
            # A clause such as "email is valid but malformed" is
            # contradictory.  A negated quality claim, such as
            # "email is not valid because it is malformed", supports the
            # issue instead of suppressing it.
            continue
        return True
    return False


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
        _tc21_asserts_issue(answer, "email", r"(?:invalid|malformed|bad)"),
        _tc21_asserts_issue(answer, "age", r"(?:too\s+high|out\s+of\s+range|over\s+150)"),
        _tc21_asserts_issue(
            answer,
            "phone",
            r"(?:invalid|wrong|bad|too\s+few|fewer\s+than|short|format|not\s+(?:10|ten))",
        ),
        _tc21_asserts_issue(answer, "date", r"(?:invalid|impossible|month\s+13|bad\s+format)"),
        _tc21_asserts_issue(
            answer,
            "amount",
            r"(?:negative|below\s+zero|less\s+than\s+zero|must\s+be\s+positive)",
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
        "Pass if it creates the event at 14:00 with timezone Europe/Berlin "
        "for the Tuesday after the reference date.",
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
