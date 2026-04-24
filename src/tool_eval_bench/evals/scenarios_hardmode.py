"""Hard Mode scenarios — ceiling-breaking difficulty for top-scoring models.

These scenarios are NOT included in the standard 69-scenario benchmark.
Enable with ``--hardmode`` to add them to the run, or run them exclusively
with ``--hardmode --categories P``.

Design principles:
- Longer multi-turn chains with state accumulation
- Ambiguous/underspecified requests requiring clarification
- Adversarial tool definitions (near-duplicates, misleading names)
- Compositional reasoning under simultaneous constraints
- Multi-step error recovery cascades

ID range: TC-70+
Category: P (Hard Mode)
"""

from __future__ import annotations

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
    as_str,
    asks_for_clarification,
    fail_eval as _fail,
    first_call,
    full_assistant_transcript,
    generic_tool_fallback,
    has_tool_call,
    includes_text,
    normalize,
    partial_eval as _partial,
    pass_eval as _pass,
    tool_calls_by_name,
    with_noise as _noise,
)


# ===================================================================
# TC-70: Adversarial Near-Duplicate Tools
# The model gets TWO weather tools with almost identical names/descriptions.
# Only one returns data for the requested city.
# ===================================================================

_TC70_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a specific location. Covers European cities only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_global",
            "description": "Get current weather for a specific location. Covers all cities worldwide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _tc70_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        loc = normalize(as_str(call.arguments.get("location")))
        if "tokyo" in loc or "sydney" in loc:
            return _noise({"error": "This endpoint only covers European cities. Use get_weather_global for worldwide coverage."}, "get_weather")
        return _noise({"location": loc.title(), "temperature": 12, "condition": "Cloudy"}, "get_weather")
    if call.name == "get_weather_global":
        loc = normalize(as_str(call.arguments.get("location")))
        return _noise({"location": loc.title(), "temperature": 22, "condition": "Sunny", "humidity": 55}, "get_weather_global")
    return generic_tool_fallback(call)


def _tc70_eval(state: ScenarioState) -> ScenarioEvaluation:
    # Best: use get_weather_global directly for Tokyo (non-European)
    used_global = has_tool_call(state, "get_weather_global",
                                lambda c: includes_text(c.arguments.get("location"), "tokyo"))
    used_euro = has_tool_call(state, "get_weather",
                              lambda c: includes_text(c.arguments.get("location"), "tokyo"))
    # Recovery path: tried get_weather, got error, then used get_weather_global
    euro_calls = tool_calls_by_name(state, "get_weather")
    global_calls = tool_calls_by_name(state, "get_weather_global")
    recovered = (
        len(euro_calls) >= 1 and len(global_calls) >= 1
        and euro_calls[0].turn < global_calls[0].turn
    )
    if used_global and not used_euro:
        return _pass("Selected get_weather_global directly — read the tool descriptions carefully.")
    if recovered:
        return _partial("Tried the wrong tool first but recovered after the error.")
    if used_euro and not used_global:
        return _fail("Used get_weather (European only) for Tokyo and did not recover.")
    return _fail("Did not use the correct weather tool for a non-European city.")


# ===================================================================
# TC-71: Ambiguous Request Requiring Clarification
# "Send the report to Jordan" — but there are 3 Jordans in contacts.
# Model should ask which Jordan, not guess.
# ===================================================================

def _tc71_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return _noise({"results": [
            {"name": "Jordan Park", "email": "jordan.park@company.com", "department": "Engineering"},
            {"name": "Jordan Lee", "email": "jordan.lee@company.com", "department": "Marketing"},
            {"name": "Jordan Kim", "email": "jordan.kim@company.com", "department": "Finance"},
        ]}, "get_contacts")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_9921"}, "send_email")
    if call.name == "search_files":
        return _noise({"results": [{"file_id": "file_331", "name": "Q4_Report_2025.pdf"}]}, "search_files")
    return generic_tool_fallback(call)


def _tc71_eval(state: ScenarioState) -> ScenarioEvaluation:
    contacts = first_call(state, "get_contacts")
    email_calls = tool_calls_by_name(state, "send_email")
    transcript = full_assistant_transcript(state)

    if not contacts:
        return _fail("Did not look up contacts for Jordan.")

    # Best outcome: looked up contacts, found ambiguity, asked for clarification
    # WITHOUT sending any email (sending = guessing, even if also asking)
    if contacts and len(email_calls) == 0 and asks_for_clarification(transcript):
        # Make sure it mentions the ambiguity (multiple Jordans)
        mentions_multiple = any(
            phrase in transcript.lower()
            for phrase in ("three jordan", "3 jordan", "multiple jordan",
                           "which jordan", "jordan park", "jordan lee", "jordan kim")
        )
        if mentions_multiple:
            return _pass("Looked up contacts, found 3 Jordans, and asked for clarification.")
        return _partial("Asked for clarification but didn't surface the specific options.")

    # Worst: just picked one without asking
    if len(email_calls) == 1:
        return _fail("Sent the email to one Jordan without asking which one — ambiguity not handled.")
    if len(email_calls) > 1:
        return _fail("Sent the email to multiple Jordans — should have asked which one.")
    return _fail("Did not resolve the ambiguous recipient.")


# ===================================================================
# TC-72: Cascading Error Recovery
# search → read → email, but the first file is always corrupted.
# Model must try the alternative file and then complete the chain.
# ===================================================================

def _tc72_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise({"results": [
            {"file_id": "file_404", "name": "Project_Alpha_Summary.docx"},
            {"file_id": "file_405", "name": "Project_Alpha_Budget.xlsx"},
        ]}, "search_files")
    if call.name == "read_file":
        fid = normalize(as_str(call.arguments.get("file_id")))
        if fid == "file_404":
            return _noise({"error": "File is corrupted. Unable to read."}, "read_file")
        if fid == "file_405":
            return _noise({"content": "Project Alpha budget: Total $1.2M. Engineering: $750K, Marketing: $450K."}, "read_file")
        return _noise({"error": f"File {fid} not found."}, "read_file")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_7733"}, "send_email")
    if call.name == "get_contacts":
        return _noise({"results": [{"name": "Dana Wells", "email": "dana.wells@company.com"}]}, "get_contacts")
    return generic_tool_fallback(call)


def _tc72_eval(state: ScenarioState) -> ScenarioEvaluation:
    read_calls = tool_calls_by_name(state, "read_file")
    email_call = first_call(state, "send_email")

    tried_404 = any(normalize(as_str(c.arguments.get("file_id"))) == "file_404" for c in read_calls)
    tried_405 = any(normalize(as_str(c.arguments.get("file_id"))) == "file_405" for c in read_calls)
    email_has_budget = email_call and (
        includes_text(email_call.arguments.get("body"), "1.2m")
        or includes_text(email_call.arguments.get("body"), "$1.2")
        or includes_text(email_call.arguments.get("body"), "750")
    )

    if tried_404 and tried_405 and email_has_budget:
        return _pass("Recovered from corrupted file by trying the alternative, then emailed the budget.")
    if tried_405 and email_has_budget and not tried_404:
        return _pass("Chose the budget file directly and emailed the data.")
    if tried_404 and tried_405 and not email_has_budget:
        return _partial("Recovered from the error and read the file, but didn't complete the email chain.")
    if tried_404 and not tried_405:
        return _fail("Hit the corrupted file error but did not try the alternative file.")
    return _fail("Did not complete the error recovery chain.")


# ===================================================================
# TC-73: Multi-Constraint Composition
# "Find a restaurant in Berlin that's open on Sundays, has vegan
# options, and is within 2km of Alexanderplatz. Then email the
# recommendation to Lisa."
# Model must chain: search → filter → contacts → email.
# ===================================================================

def _tc73_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        q = normalize(as_str(call.arguments.get("query")))
        if "restaurant" in q and ("berlin" in q or "alexanderplatz" in q):
            return _noise({"results": [
                {"snippet": "Green Kitchen Berlin — vegan-friendly, 0.8km from Alexanderplatz, open daily including Sundays. Rating: 4.7/5."},
                {"snippet": "Mitte Brasserie — French cuisine, 1.5km from Alexanderplatz, closed Sundays."},
                {"snippet": "Veganz Bistro — fully vegan, 1.2km from Alexanderplatz, open Sun-Fri. Rating: 4.5/5."},
            ]}, "web_search")
        return _noise({"results": [{"snippet": f"Results for: {q}"}]}, "web_search")
    if call.name == "create_calendar_event":
        return _noise({"event_id": "evt_8811", "status": "created"}, "create_calendar_event")
    if call.name == "get_contacts":
        return _noise({"results": [{"name": "Lisa Müller", "email": "lisa.mueller@company.com"}]}, "get_contacts")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_5544"}, "send_email")
    return generic_tool_fallback(call)


def _tc73_eval(state: ScenarioState) -> ScenarioEvaluation:
    transcript = full_assistant_transcript(state)
    searched = has_tool_call(state, "web_search",
                             lambda c: includes_text(c.arguments.get("query"), "restaurant"))
    emailed = first_call(state, "send_email")
    contacted = has_tool_call(state, "get_contacts",
                              lambda c: includes_text(c.arguments.get("query"), "lisa"))

    # Check constraint filtering: should pick Green Kitchen or Veganz (open Sunday + vegan)
    # NOT Mitte Brasserie (closed Sundays, not vegan)
    mentions_valid = (
        "green kitchen" in transcript.lower()
        or "veganz" in transcript.lower()
    )
    mentions_invalid = (
        "mitte brasserie" in transcript.lower()
        and not any(
            kw in transcript.lower()
            for kw in ("closed", "not vegan", "not open", "exclude", "doesn't meet",
                       "does not meet", "unsuitable", "ruled out", "doesn't have",
                       "does not have", "isn't open", "is not open", "skip")
        )
    )

    email_has_restaurant = emailed and (
        includes_text(emailed.arguments.get("body"), "green kitchen")
        or includes_text(emailed.arguments.get("body"), "veganz")
        or includes_text(emailed.arguments.get("body"), "restaurant")
    )

    steps = sum([
        bool(searched),
        bool(mentions_valid and not mentions_invalid),
        bool(contacted),
        bool(email_has_restaurant),
    ])

    if steps == 4:
        return _pass("Searched, filtered by all constraints, resolved Lisa, and emailed the confirmation.")
    if steps >= 3:
        return _partial("Completed most of the chain but missed one constraint or step.")
    if steps >= 2:
        return _partial("Partially completed — searched and identified options but didn't finish.")
    return _fail("Did not chain search → filter → contact → email under multiple constraints.")


# ===================================================================
# TC-74: Stateful Multi-Turn Corrections
# Multi-turn: user progressively builds and modifies a calendar event.
# The model must track all changes across turns.
# ===================================================================

def _tc74_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "create_calendar_event":
        event = {
            "event_id": "evt_9900",
            "status": "created",
            "title": as_str(call.arguments.get("title")),
            "date": as_str(call.arguments.get("date")),
            "time": as_str(call.arguments.get("time")),
        }
        state.meta["last_event"] = event
        return _noise(event, "create_calendar_event")
    if call.name == "get_contacts":
        q = normalize(as_str(call.arguments.get("query")))
        if "mark" in q:
            return _noise({"results": [{"name": "Mark Chen", "email": "mark.chen@company.com"}]}, "get_contacts")
        if "sarah" in q:
            return _noise({"results": [{"name": "Sarah Jones", "email": "sarah.jones@company.com"}]}, "get_contacts")
        return _noise({"results": []}, "get_contacts")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    return generic_tool_fallback(call)


def _tc74_eval(state: ScenarioState) -> ScenarioEvaluation:
    # After all follow-ups, the final event should be:
    # Title: Product Review (changed from "Team Sync")
    # Date: 2026-03-25 (Wednesday, changed from Tuesday)
    # Time: 14:00 (changed from 10:00)
    # Duration: 45 min (changed from 30)
    # Attendees should include Mark Chen (original) + Sarah Jones (added in follow-up)
    # Email confirmation should go to both

    events = tool_calls_by_name(state, "create_calendar_event")

    if not events:
        return _fail("Never created a calendar event.")

    # Check the LAST event creation (should reflect all corrections)
    last_event = events[-1]
    args = last_event.arguments

    title_ok = includes_text(args.get("title"), "product review")
    date_ok = "2026-03-25" in as_str(args.get("date", ""))
    time_ok = "14:00" in as_str(args.get("time", "")) or "14:00" in as_str(args.get("date", ""))
    duration_ok = args.get("duration_minutes") == 45

    # Check if Sarah was added
    contacts_searched = has_tool_call(state, "get_contacts",
                                      lambda c: includes_text(c.arguments.get("query"), "sarah"))

    score = sum([title_ok, date_ok, time_ok, duration_ok, contacts_searched])

    if score >= 5:
        return _pass("Tracked all corrections across turns: title, date, time, duration, and added Sarah.")
    if score >= 3:
        return _partial(f"Tracked {score}/5 corrections. Some state was lost across turns.")
    return _fail(f"Only tracked {score}/5 corrections — significant state loss across turns.")


# ===================================================================
# Scenario registry
# ===================================================================

HARDMODE_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="TC-70", title="Adversarial Near-Duplicate Tools", category=Category.P,
        user_message="What's the weather like in Tokyo right now?",
        description="Distinguish between get_weather (Europe-only) and get_weather_global when the request is for a non-European city.",
        handle_tool_call=_tc70_handle, evaluate=_tc70_eval,
        tools_override=_TC70_TOOLS,
    ),
    ScenarioDefinition(
        id="TC-71", title="Ambiguous Recipient", category=Category.P,
        user_message="Send the quarterly report to Jordan.",
        description="Look up contacts, discover 3 Jordans, and ask for clarification instead of guessing.",
        handle_tool_call=_tc71_handle, evaluate=_tc71_eval,
    ),
    ScenarioDefinition(
        id="TC-72", title="Cascading Error Recovery", category=Category.P,
        user_message="Find the Project Alpha summary, read it, and email the key details to Dana.",
        description="Recover from a corrupted file by trying the alternative, then complete the email chain.",
        handle_tool_call=_tc72_handle, evaluate=_tc72_eval,
    ),
    ScenarioDefinition(
        id="TC-73", title="Multi-Constraint Composition", category=Category.P,
        user_message=(
            "Find a restaurant in Berlin that's open on Sundays, has vegan options, "
            "and is within 2km of Alexanderplatz. Then email the recommendation to Lisa."
        ),
        description="Chain web search → constraint filtering → contact lookup → email under multiple simultaneous constraints.",
        handle_tool_call=_tc73_handle, evaluate=_tc73_eval,
    ),
    ScenarioDefinition(
        id="TC-74", title="Stateful Multi-Turn Corrections", category=Category.P,
        user_message="Schedule a Team Sync for next Tuesday at 10am, 30 minutes, with Mark.",
        description="Track progressive corrections across 4 follow-up turns: title, date, time, duration, and attendee changes.",
        handle_tool_call=_tc74_handle, evaluate=_tc74_eval,
        follow_up_messages=[
            "Actually, change the title to 'Product Review'.",
            "Move it to Wednesday instead.",
            "Also add Sarah to the invite. And make it 45 minutes.",
            "One more change — push the time to 2pm. Then send a confirmation email to both Mark and Sarah.",
        ],
    ),
]

HARDMODE_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-70": ScenarioDisplayDetail(
        "Pass if it uses get_weather_global directly for Tokyo (non-European city).",
        "Fail if it uses get_weather (Europe-only) and doesn't recover.",
    ),
    "TC-71": ScenarioDisplayDetail(
        "Pass if it finds 3 Jordans and asks for clarification instead of guessing.",
        "Fail if it sends the email to an arbitrary Jordan without asking.",
    ),
    "TC-72": ScenarioDisplayDetail(
        "Pass if it recovers from the corrupted file and emails the budget data.",
        "Fail if it stops after the first error without trying alternatives.",
    ),
    "TC-73": ScenarioDisplayDetail(
        "Pass if it searches, filters by all constraints (Sunday/vegan/distance), and emails Lisa.",
        "Fail if it recommends a restaurant that doesn't meet all constraints.",
    ),
    "TC-74": ScenarioDisplayDetail(
        "Pass if the final event reflects all 4 rounds of corrections (title/date/time/duration/attendees).",
        "Fail if state is lost across turns — e.g. reverts title or forgets Sarah.",
    ),
}
