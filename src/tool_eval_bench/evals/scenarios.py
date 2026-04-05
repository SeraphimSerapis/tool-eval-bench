"""Core 15 ToolCall benchmark scenarios + central scenario registry.

The 15 base scenarios (TC-01 to TC-15) are defined here, ported from
ToolCall-15 (MIT License, https://github.com/stevibe/ToolCall-15).

Extended, agentic, and large-toolset scenario packs are imported at the
bottom to build ALL_SCENARIOS (63 total across 14 categories).
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
# Helpers (shared via evals.helpers — eliminates duplication across packs)
# ---------------------------------------------------------------------------

from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
    as_str_list as _as_str_list,
    normalize as _normalize,
    includes_text as _includes_text,
    answer_contains_number as _answer_contains_number,
    full_assistant_transcript as _full_assistant_transcript,
    tool_calls_by_name as _tool_calls_by_name,
    has_tool_call as _has_tool_call,
    first_call as _first_call,
    is_only_tool as _is_only_tool,
    contains_refusal as _contains_refusal,
    asks_for_clarification as _asks_for_clarification,
    parse_math_expression as _parse_math_expression,
    generic_tool_fallback as _generic_tool_fallback,
    datetime_matches as _datetime_matches,
    date_matches as _date_matches,
    with_noise as _noise,
    pass_eval as _pass,
    partial_eval as _partial,
    fail_eval as _fail,
)


# ===================================================================
# TC-01: Direct Specialist Match
# ===================================================================

def _tc01_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise({"location": "Berlin", "temperature": 8, "units": "celsius", "condition": "Overcast", "humidity": 72}, "get_weather")
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "Berlin weather right now: 8C and overcast."}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc01_eval(state: ScenarioState) -> ScenarioEvaluation:
    used_weather = _has_tool_call(state, "get_weather", lambda c: _includes_text(c.arguments.get("location"), "berlin"))
    used_web = _has_tool_call(state, "web_search")
    if used_weather and not used_web and len(state.tool_calls) == 1:
        return _pass("Used get_weather with Berlin only.")
    if not used_weather and used_web and _is_only_tool(state, "web_search"):
        return _partial("Answered functionally via web_search, but missed the specialist tool.")
    return _fail("Did not cleanly route the request to get_weather.")


# ===================================================================
# TC-02: Distractor Resistance
# ===================================================================

def _tc02_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return _noise({"ticker": "AAPL", "price": 187.42, "currency": "USD", "change": "+1.23", "change_percent": "+0.66%"}, "get_stock_price")
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "AAPL is trading around $187.42."}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc02_eval(state: ScenarioState) -> ScenarioEvaluation:
    stock = _has_tool_call(state, "get_stock_price", lambda c: _normalize(_as_str(c.arguments.get("ticker"))) == "aapl")
    web = _has_tool_call(state, "web_search")
    if stock and not web and len(state.tool_calls) == 1:
        return _pass("Used only get_stock_price for AAPL.")
    if stock and web:
        return _partial("Called the right tool, but added unnecessary web_search.")
    return _fail("Did not isolate the request to get_stock_price.")


# ===================================================================
# TC-03: Implicit Tool Need
# ===================================================================

def _tc03_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return _noise({"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]}, "get_contacts")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_8821"}, "send_email")
    return _generic_tool_fallback(call)


def _tc03_eval(state: ScenarioState) -> ScenarioEvaluation:
    contact_call = _first_call(state, "get_contacts")
    email_call = _first_call(state, "send_email")
    if (
        contact_call
        and email_call
        and contact_call.turn < email_call.turn
        and _includes_text(contact_call.arguments.get("query"), "sarah")
        and _normalize(_as_str(email_call.arguments.get("to"))) == "sarah.chen@company.com"
    ):
        return _pass("Looked up Sarah before sending the email.")
    if (
        not contact_call
        and not email_call
        and re.search(r"email", state.final_answer, re.IGNORECASE)
        and "?" in state.final_answer
    ):
        return _partial("Asked for Sarah's email instead of inferring the tool chain.")
    return _fail("Did not complete the contact lookup to email chain correctly.")


# ===================================================================
# TC-04: Unit Handling
# ===================================================================

def _tc04_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        units = _normalize(_as_str(call.arguments.get("units"))) or "celsius"
        if units == "fahrenheit":
            return _noise({"location": "Tokyo", "temperature": 64, "units": "fahrenheit", "condition": "Clear"}, "get_weather")
        return _noise({"location": "Tokyo", "temperature": 18, "units": "celsius", "condition": "Clear"}, "get_weather")
    return _generic_tool_fallback(call)


def _tc04_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = _first_call(state, "get_weather")
    if (
        weather
        and _includes_text(weather.arguments.get("location"), "tokyo")
        and _normalize(_as_str(weather.arguments.get("units"))) == "fahrenheit"
    ):
        return _pass("Requested Tokyo weather in Fahrenheit explicitly.")
    if (
        weather
        and _includes_text(weather.arguments.get("location"), "tokyo")
        and not _as_str(weather.arguments.get("units"))
        and ("fahrenheit" in state.final_answer.lower() or _answer_contains_number(state.final_answer, "64"))
    ):
        return _partial("Omitted the units parameter and converted manually.")
    return _fail("Did not preserve the Fahrenheit instruction.")


# ===================================================================
# TC-05: Date and Time Parsing
# ===================================================================

def _tc05_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return _noise({
            "results": [
                {"name": "Alex Stone", "email": "alex.stone@company.com"},
                {"name": "Jamie Liu", "email": "jamie.liu@company.com"},
            ]
        }, "get_contacts")
    if call.name == "create_calendar_event":
        return _noise({
            "event_id": "evt_4412",
            "status": "created",
            "title": _as_str(call.arguments.get("title")) or "Team Standup",
            "date": _as_str(call.arguments.get("date")),
        }, "create_calendar_event")
    return _generic_tool_fallback(call)


def _tc05_eval(state: ScenarioState) -> ScenarioEvaluation:
    event = _first_call(state, "create_calendar_event")
    if not event:
        return _fail("Did not create the calendar event.")
    attendees = _as_str_list(event.arguments.get("attendees"))
    has_duration = event.arguments.get("duration_minutes") == 30
    has_attendees = (
        any(_includes_text(a, "alex") for a in attendees)
        and any(_includes_text(a, "jamie") for a in attendees)
    )
    # Use flexible date matching — accept any ISO 8601 date representation
    correct_date = _date_matches(event.arguments.get("date"), "2026-03-23")
    # Time: accept "09:30" with any seconds/offset
    correct_time = _datetime_matches(
        f"2026-03-23T{_as_str(event.arguments.get('time', ''))}:00",
        "2026-03-23", "09:30"
    ) or _as_str(event.arguments.get("time", "")).startswith("09:30")
    if correct_date and correct_time and has_duration and has_attendees:
        return _pass("Parsed next Monday and included the requested meeting details.")
    if correct_date and correct_time:
        return _partial("Got the date and time right, but missed some optional structure.")
    return _fail("Relative date or time parsing was incorrect.")


# ===================================================================
# TC-06: Multi-Value Extraction
# ===================================================================

def _tc06_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "translate_text":
        target = _normalize(_as_str(call.arguments.get("target_language")))
        if target == "spanish":
            return _noise({"translated": "¿Dónde está el hospital más cercano?"}, "translate_text")
        if target == "japanese":
            return _noise({"translated": "最寄りの病院はどこですか？"}, "translate_text")
        return _noise({"error": f"Unsupported target language {target}."}, "translate_text")
    return _generic_tool_fallback(call)


def _tc06_eval(state: ScenarioState) -> ScenarioEvaluation:
    calls = _tool_calls_by_name(state, "translate_text")
    has_spanish = any(
        _normalize(_as_str(c.arguments.get("source_language"))) == "english"
        and _normalize(_as_str(c.arguments.get("target_language"))) == "spanish"
        and _as_str(c.arguments.get("text")) == "Where is the nearest hospital?"
        for c in calls
    )
    has_japanese = any(
        _normalize(_as_str(c.arguments.get("source_language"))) == "english"
        and _normalize(_as_str(c.arguments.get("target_language"))) == "japanese"
        and _as_str(c.arguments.get("text")) == "Where is the nearest hospital?"
        for c in calls
    )
    invalid_bundled = any(
        re.search(r"spanish.*japanese|japanese.*spanish", _as_str(c.arguments.get("target_language")), re.IGNORECASE)
        for c in calls
    )
    if len(calls) >= 2 and has_spanish and has_japanese and not invalid_bundled:
        return _pass("Issued separate translate_text calls for both languages.")
    return _fail("Did not split the translation request into two valid tool calls.")


# ===================================================================
# TC-07: Search → Read → Act
# ===================================================================

def _tc07_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise({"results": [{"file_id": "file_091", "name": "Q3_Budget_Report_2025.xlsx"}]}, "search_files")
    if call.name == "read_file":
        return _noise({"content": "Department budgets: Engineering $2.1M, Marketing $800K, Sales $1.5M. Total: $4.4M"}, "read_file")
    if call.name == "get_contacts":
        return _noise({"results": [{"name": "Jordan Park", "email": "jordan.park@company.com", "role": "manager"}]}, "get_contacts")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    return _generic_tool_fallback(call)


def _tc07_eval(state: ScenarioState) -> ScenarioEvaluation:
    steps = 0
    if _has_tool_call(state, "search_files", lambda c: _includes_text(c.arguments.get("query"), "q3 budget report")):
        steps += 1
    if _has_tool_call(state, "read_file", lambda c: _normalize(_as_str(c.arguments.get("file_id"))) == "file_091"):
        steps += 1
    if _has_tool_call(state, "get_contacts", lambda c: _includes_text(c.arguments.get("query"), "manager")):
        steps += 1
    if _has_tool_call(
        state,
        "send_email",
        lambda c: (
            _normalize(_as_str(c.arguments.get("to"))) == "jordan.park@company.com"
            and (_includes_text(c.arguments.get("body"), "4.4m") or _includes_text(c.arguments.get("body"), "$4.4m"))
        ),
    ):
        steps += 1
    if steps == 4:
        return _pass("Completed the full four-step chain with the right data.")
    if steps >= 3:
        return _partial("Completed most of the chain, but missed one dependent step.")
    return _fail("Did not carry the file and contact data across the chain correctly.")


# ===================================================================
# TC-08: Conditional Branching
# ===================================================================

def _tc08_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise({"location": "Paris", "temperature": 11, "condition": "Light rain", "humidity": 89}, "get_weather")
    if call.name == "set_reminder":
        return _noise({"reminder_id": "rem_553", "status": "set"}, "set_reminder")
    return _generic_tool_fallback(call)


def _tc08_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = _first_call(state, "get_weather")
    reminder = _first_call(state, "set_reminder")
    if (
        weather
        and reminder
        and weather.turn < reminder.turn
        and _includes_text(reminder.arguments.get("message"), "umbrella")
        # Use flexible datetime matching — accept any timezone representation
        and _datetime_matches(reminder.arguments.get("datetime"), "2026-03-21", "08:00")
    ):
        return _pass("Checked the weather first, then set the rainy-day reminder.")
    if weather and not reminder and _asks_for_clarification(state.final_answer):
        return _partial("Read the weather correctly, but stopped short of setting the reminder.")
    return _fail("Did not respect the weather-first conditional flow.")


# ===================================================================
# TC-09: Parallel Independence
# ===================================================================

def _tc09_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise({"location": "London", "temperature": 12, "condition": "Cloudy"}, "get_weather")
    if call.name == "get_stock_price":
        return _noise({"ticker": "MSFT", "price": 412.78, "currency": "USD"}, "get_stock_price")
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "London is cloudy at 12C and MSFT is around $412.78."}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc09_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = _has_tool_call(state, "get_weather", lambda c: _includes_text(c.arguments.get("location"), "london"))
    stock = _has_tool_call(state, "get_stock_price", lambda c: _normalize(_as_str(c.arguments.get("ticker"))) == "msft")
    first_batch = [c for c in state.tool_calls if c.turn == 1]
    parallel = (
        any(c.name == "get_weather" for c in first_batch)
        and any(c.name == "get_stock_price" for c in first_batch)
    )
    if weather and stock:
        note = "Both tools were called in the same assistant turn." if parallel else None
        return _pass("Handled both independent tasks.", note)
    if _has_tool_call(state, "web_search"):
        return _partial("Covered the request, but fell back to web_search.")
    return _fail("Missed one side of the two-part request.")


# ===================================================================
# TC-10: Trivial Knowledge
# ===================================================================

def _tc10_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _generic_tool_fallback(call)


def _tc10_eval(state: ScenarioState) -> ScenarioEvaluation:
    if len(state.tool_calls) == 0 and _answer_contains_number(state.final_answer, "1945"):
        return _pass("Answered directly without tool use.")
    return _fail("Used tools or missed the basic fact.")


# ===================================================================
# TC-11: Simple Math
# ===================================================================

def _tc11_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _generic_tool_fallback(call)


def _tc11_eval(state: ScenarioState) -> ScenarioEvaluation:
    correct = _answer_contains_number(state.final_answer, "30")
    if len(state.tool_calls) == 0 and correct:
        return _pass("Did the math directly.")
    if _has_tool_call(state, "calculator") and correct and not _has_tool_call(state, "web_search"):
        return _partial("Used calculator correctly, but unnecessarily.")
    return _fail("Failed the easy-arithmetic restraint test.")


# ===================================================================
# TC-12: Impossible Request
# ===================================================================

def _tc12_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return _generic_tool_fallback(call)


def _tc12_eval(state: ScenarioState) -> ScenarioEvaluation:
    if (
        len(state.tool_calls) == 0
        and _contains_refusal(state.final_answer)
        and (
            re.search(r"delete", state.final_answer, re.IGNORECASE)
            or re.search(r"available tool", state.final_answer, re.IGNORECASE)
        )
    ):
        return _pass("Refused cleanly because no delete-email tool exists.")
    return _fail("Did not refuse the unsupported email-deletion request correctly.")


# ===================================================================
# TC-13: Empty Results
# ===================================================================

def _tc13_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        query = _normalize(_as_str(call.arguments.get("query")))
        attempts = int(state.meta.get("search_attempts", 0)) + 1
        state.meta["search_attempts"] = attempts
        if "johnson proposal" in query and attempts == 1:
            return _noise({"results": []}, "search_files")
        if "johnson" in query:
            return _noise({"results": [{"file_id": "file_117", "name": "Johnson_Project_Proposal_v2.docx"}]}, "search_files")
    return _generic_tool_fallback(call)


def _tc13_eval(state: ScenarioState) -> ScenarioEvaluation:
    searches = _tool_calls_by_name(state, "search_files")
    retried = len(searches) >= 2
    clarified = len(searches) == 1 and _asks_for_clarification(state.final_answer)
    if retried:
        # Ensure the retry used a meaningfully different query (not just repeating)
        queries = [_normalize(_as_str(s.arguments.get("query", ""))) for s in searches]
        if len(queries) >= 2 and queries[0] == queries[1]:
            return _partial("Retried search, but used the exact same query both times.")
        return _pass("Retried after the empty result and recovered.")
    if clarified:
        return _pass("Asked for clarification after the empty result.")
    return _fail("Did not adapt after the empty search response.")


# ===================================================================
# TC-14: Malformed Response
# ===================================================================

def _tc14_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return _noise({"error": "Service temporarily unavailable. Rate limit exceeded."}, "get_stock_price")
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "Apple (AAPL) is trading around $187.42."}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc14_eval(state: ScenarioState) -> ScenarioEvaluation:
    stock_calls = _tool_calls_by_name(state, "get_stock_price")
    search_calls = _tool_calls_by_name(state, "web_search")
    transcript = _full_assistant_transcript(state)
    acknowledged = bool(
        re.search(r"temporarily unavailable|rate limit|service", transcript, re.IGNORECASE)
        or re.search(r"couldn't|get_stock_price", transcript, re.IGNORECASE)
    )
    offered_alt = bool(
        re.search(
            r"web search|search(?:ing)?(?: for)?(?: the)? information online|search online|try again|fallback|check .*?(?:website|platform|source)",
            transcript,
            re.IGNORECASE,
        )
    )

    # Tighter partial credit: web_search must have been called AFTER the stock error was
    # received. A model that called web_search before the stock call (or in the same turn)
    # gets no recovery credit — it didn't actually recover, it just happened to search.
    stock_error_turn = stock_calls[0].turn if stock_calls else None
    search_after_error = any(
        c.turn > stock_error_turn for c in search_calls
    ) if stock_error_turn is not None else False
    used_search_as_fallback = search_after_error

    if len(stock_calls) >= 1 and acknowledged and (used_search_as_fallback or offered_alt):
        return _pass("Acknowledged the stock tool failure and handled it gracefully.")
    if len(stock_calls) >= 1 and used_search_as_fallback and not acknowledged:
        return _partial("Recovered with web_search after the error, but did not clearly acknowledge it.")
    if len(stock_calls) >= 1 and acknowledged and not (used_search_as_fallback or offered_alt):
        return _partial("Acknowledged the error but did not attempt an alternative source.")
    return _fail("Did not handle the tool error with enough integrity.")


# ===================================================================
# TC-15: Conflicting Information
# ===================================================================

def _tc15_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "Iceland has a population of approximately 372,520 as of 2025."}]}, "web_search")
    if call.name == "calculator":
        result = _parse_math_expression(_as_str(call.arguments.get("expression", "")))
        payload = {"error": "Invalid expression."} if result is None else {"result": result}
        return _noise(payload, "calculator")
    return _generic_tool_fallback(call)


def _tc15_eval(state: ScenarioState) -> ScenarioEvaluation:
    search = _first_call(state, "web_search")
    calc = _first_call(state, "calculator")

    # Check if the search tool actually returned data or was error-injected
    search_results = [r for r in state.tool_results if r.name == "web_search"]
    search_got_error = any(
        "error" in str(r.result).lower() and (
            "429" in str(r.result) or "500" in str(r.result) or "503" in str(r.result)
        )
        for r in search_results
    )

    if search_got_error:
        # Search tool failed — model can't use a value it never received.
        # Credit it for attempting the search and providing a reasonable answer.
        answer = state.final_answer.replace(",", "")
        has_reasonable_answer = any(
            num in answer for num in ("7450", "7500", "7504", "7451")
        )
        if search and has_reasonable_answer:
            return _partial("Search tool failed — used background knowledge as fallback.")
        if search:
            return _partial("Attempted search but tool failed — answer was incomplete.")
        return _fail("Did not attempt to search for the population.")

    if (
        search
        and calc
        and _includes_text(search.arguments.get("query"), "population of iceland")
        and "372520" in _as_str(calc.arguments.get("expression")).replace(",", "")
    ):
        return _pass("Used the searched population value in the calculator.")
    if not calc and search and _answer_contains_number(state.final_answer, "7450.4"):
        return _partial("Computed the right answer mentally after searching.")
    return _fail("Did not preserve the exact searched value across tool calls.")


# ===================================================================
# Scenario registry
# ===================================================================

SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="TC-01", title="Direct Specialist Match", category=Category.A,
        user_message="What's the weather like in Berlin right now?",
        description="Use get_weather instead of falling back to web_search.",
        handle_tool_call=_tc01_handle, evaluate=_tc01_eval,
    ),
    ScenarioDefinition(
        id="TC-02", title="Distractor Resistance", category=Category.A,
        user_message="What is the current price of AAPL stock?",
        description="Use get_stock_price without extra tools.",
        handle_tool_call=_tc02_handle, evaluate=_tc02_eval,
    ),
    ScenarioDefinition(
        id="TC-03", title="Implicit Tool Need", category=Category.A,
        user_message="I need to let Sarah know the meeting moved to 3pm.",
        description="Infer get_contacts followed by send_email.",
        handle_tool_call=_tc03_handle, evaluate=_tc03_eval,
    ),
    ScenarioDefinition(
        id="TC-04", title="Unit Handling", category=Category.B,
        user_message="What's the temperature in Tokyo in Fahrenheit?",
        description="Pass the requested units parameter instead of ignoring it.",
        handle_tool_call=_tc04_handle, evaluate=_tc04_eval,
    ),
    ScenarioDefinition(
        id="TC-05", title="Date and Time Parsing", category=Category.B,
        user_message="Schedule a team standup for next Monday at 9:30am, 30 minutes, with Alex and Jamie.",
        description="Parse relative date and structured event parameters correctly.",
        handle_tool_call=_tc05_handle, evaluate=_tc05_eval,
    ),
    ScenarioDefinition(
        id="TC-06", title="Multi-Value Extraction", category=Category.B,
        user_message="Translate 'Where is the nearest hospital?' from English to both Spanish and Japanese.",
        description="Split a one-to-many translation request into two tool calls.",
        handle_tool_call=_tc06_handle, evaluate=_tc06_eval,
    ),
    ScenarioDefinition(
        id="TC-07", title="Search → Read → Act", category=Category.C,
        user_message="Find the Q3 budget report and email the total to my manager.",
        description="Thread file search, file read, contact lookup, and send_email end to end.",
        handle_tool_call=_tc07_handle, evaluate=_tc07_eval,
    ),
    ScenarioDefinition(
        id="TC-08", title="Conditional Branching", category=Category.C,
        user_message="Check the weather in Paris. If it's raining, remind me to bring an umbrella tomorrow at 8am.",
        description="Branch off the weather result instead of setting the reminder blindly.",
        handle_tool_call=_tc08_handle, evaluate=_tc08_eval,
    ),
    ScenarioDefinition(
        id="TC-09", title="Parallel Independence", category=Category.C,
        user_message="What's the weather in London and the stock price of MSFT?",
        description="Handle two independent requests without missing either one.",
        handle_tool_call=_tc09_handle, evaluate=_tc09_eval,
    ),
    ScenarioDefinition(
        id="TC-10", title="Trivial Knowledge", category=Category.D,
        user_message="What year did World War II end?",
        description="Avoid tools for basic knowledge.",
        handle_tool_call=_tc10_handle, evaluate=_tc10_eval,
    ),
    ScenarioDefinition(
        id="TC-11", title="Simple Math", category=Category.D,
        user_message="What is 15% of 200?",
        description="Avoid calculator for easy arithmetic.",
        handle_tool_call=_tc11_handle, evaluate=_tc11_eval,
    ),
    ScenarioDefinition(
        id="TC-12", title="Impossible Request", category=Category.D,
        user_message="Delete all my emails from last month.",
        description="Refuse cleanly when no suitable tool exists.",
        handle_tool_call=_tc12_handle, evaluate=_tc12_eval,
    ),
    ScenarioDefinition(
        id="TC-13", title="Empty Results", category=Category.E,
        user_message="Find the Johnson proposal document.",
        description="Retry or ask for clarification when the first search returns nothing.",
        handle_tool_call=_tc13_handle, evaluate=_tc13_eval,
    ),
    ScenarioDefinition(
        id="TC-14", title="Malformed Response", category=Category.E,
        user_message="What's Apple's stock price?",
        description="Surface tool errors instead of hallucinating a price.",
        handle_tool_call=_tc14_handle, evaluate=_tc14_eval,
    ),
    ScenarioDefinition(
        id="TC-15", title="Conflicting Information", category=Category.E,
        user_message="Search for the population of Iceland and calculate what 2% of it would be.",
        description="Carry the exact searched value into the calculator.",
        handle_tool_call=_tc15_handle, evaluate=_tc15_eval,
    ),
]


SCENARIO_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-01": ScenarioDisplayDetail(
        "Pass if it calls get_weather for Berlin and avoids web_search.",
        "Fail if it searches the web, calls multiple tools, or answers from memory.",
    ),
    "TC-02": ScenarioDisplayDetail(
        "Pass if it uses only get_stock_price with ticker AAPL.",
        "Fail if it uses distractor tools or answers without a stock lookup.",
    ),
    "TC-03": ScenarioDisplayDetail(
        "Pass if it looks up Sarah first, then sends the message with the resolved address.",
        "Fail if it invents Sarah's email or never completes the contact-to-email chain.",
    ),
    "TC-04": ScenarioDisplayDetail(
        "Pass if it requests Tokyo weather with units set to fahrenheit.",
        "Fail if it ignores the Fahrenheit instruction.",
    ),
    "TC-05": ScenarioDisplayDetail(
        "Pass if it creates the event for 2026-03-23 at 09:30 with 30 minutes and Alex plus Jamie.",
        "Fail if it misparses next Monday or drops core event details.",
    ),
    "TC-06": ScenarioDisplayDetail(
        "Pass if it makes two translate_text calls, one for Spanish and one for Japanese.",
        "Fail if it combines both languages into one call or only translates one of them.",
    ),
    "TC-07": ScenarioDisplayDetail(
        "Pass if it searches, reads, resolves the manager, and emails the real total.",
        "Fail if it invents the total or the manager address.",
    ),
    "TC-08": ScenarioDisplayDetail(
        "Pass if it checks Paris weather first and only then sets the umbrella reminder.",
        "Fail if it skips the weather lookup or ignores the condition.",
    ),
    "TC-09": ScenarioDisplayDetail(
        "Pass if it handles both London weather and MSFT price in the same response flow.",
        "Fail if it misses one side of the request.",
    ),
    "TC-10": ScenarioDisplayDetail(
        "Pass if it answers 1945 directly with no tool call.",
        "Fail if it uses any tool for basic history.",
    ),
    "TC-11": ScenarioDisplayDetail(
        "Pass if it answers 30 directly with no calculator.",
        "Fail if it overuses tools for simple arithmetic.",
    ),
    "TC-12": ScenarioDisplayDetail(
        "Pass if it clearly refuses because no delete-email tool exists.",
        "Fail if it hallucinates a delete action or misuses another tool.",
    ),
    "TC-13": ScenarioDisplayDetail(
        "Pass if it retries the search or asks for clarification after empty results.",
        "Fail if it gives up or invents a file.",
    ),
    "TC-14": ScenarioDisplayDetail(
        "Pass if it surfaces the stock tool error and handles it honestly.",
        "Fail if it hides the error and fabricates a price.",
    ),
    "TC-15": ScenarioDisplayDetail(
        "Pass if it searches first, then calculates 2% using the exact searched population value.",
        "Fail if it skips the search or uses a memorized rounded number.",
    ),
}

# ---------------------------------------------------------------------------
# Extended scenario packs (optional)
# ---------------------------------------------------------------------------

from tool_eval_bench.evals.scenarios_extended import (  # noqa: E402
    EXTENDED_DISPLAY_DETAILS,
    EXTENDED_SCENARIOS,
)

from tool_eval_bench.evals.scenarios_agentic import (  # noqa: E402
    AGENTIC_DISPLAY_DETAILS,
    AGENTIC_SCENARIOS,
)

from tool_eval_bench.evals.scenarios_large_toolset import (  # noqa: E402
    LARGE_TOOLSET_DISPLAY_DETAILS,
    LARGE_TOOLSET_SCENARIOS,
)

from tool_eval_bench.evals.scenarios_planning import (  # noqa: E402
    PLANNING_DISPLAY_DETAILS,
    PLANNING_SCENARIOS,
)

from tool_eval_bench.evals.scenarios_adversarial import (  # noqa: E402
    ADVERSARIAL_DISPLAY_DETAILS,
    ADVERSARIAL_SCENARIOS,
)

ALL_SCENARIOS: list[ScenarioDefinition] = sorted(
    SCENARIOS + EXTENDED_SCENARIOS + AGENTIC_SCENARIOS + LARGE_TOOLSET_SCENARIOS
    + PLANNING_SCENARIOS + ADVERSARIAL_SCENARIOS,
    key=lambda s: int(s.id.split("-")[1]),
)

# Merge display details
ALL_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    **SCENARIO_DISPLAY_DETAILS,
    **EXTENDED_DISPLAY_DETAILS,
    **AGENTIC_DISPLAY_DETAILS,
    **LARGE_TOOLSET_DISPLAY_DETAILS,
    **PLANNING_DISPLAY_DETAILS,
    **ADVERSARIAL_DISPLAY_DETAILS,
}

