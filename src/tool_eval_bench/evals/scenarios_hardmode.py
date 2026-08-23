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
from tool_eval_bench.evals.helpers import (
    as_str,
    as_str_list,
    asks_for_clarification,
    call_at_or_after_user_phase,
    first_call,
    full_assistant_transcript,
    generic_tool_fallback,
    has_tool_call,
    includes_text,
    next_weekday_after_reference,
    normalize,
    result_is_usable_if_present,
    tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
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

_UNRELATED_UNIVERSAL_MUTATIONS = frozenset({"set_reminder", "run_code"})

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
            return _noise(
                {
                    "error": "This endpoint only covers European cities. Use get_weather_global for worldwide coverage."
                },
                "get_weather",
            )
        return _noise(
            {"location": loc.title(), "temperature": 12, "condition": "Cloudy"}, "get_weather"
        )
    if call.name == "get_weather_global":
        loc = normalize(as_str(call.arguments.get("location")))
        return _noise(
            {"location": loc.title(), "temperature": 22, "condition": "Sunny", "humidity": 55},
            "get_weather_global",
        )
    return generic_tool_fallback(call)


def _tc70_eval(state: ScenarioState) -> ScenarioEvaluation:
    # Best: use get_weather_global directly for Tokyo (non-European)
    used_global = has_tool_call(
        state, "get_weather_global", lambda c: includes_text(c.arguments.get("location"), "tokyo")
    )
    used_euro = has_tool_call(
        state, "get_weather", lambda c: includes_text(c.arguments.get("location"), "tokyo")
    )
    # Recovery path: tried get_weather, got error, then used get_weather_global
    euro_calls = tool_calls_by_name(state, "get_weather")
    global_calls = tool_calls_by_name(state, "get_weather_global")
    recovered = (
        len(euro_calls) >= 1
        and len(global_calls) >= 1
        and euro_calls[0].turn < global_calls[0].turn
    )
    if used_global and not used_euro:
        # Verify the model surfaced actual weather data (temp 22 or condition).
        has_temp = bool(re.search(r"(?<!\d)22(?!\d)", state.final_answer))
        has_condition = "sunny" in state.final_answer.lower()
        if has_temp or has_condition:
            return _pass(
                "Selected get_weather_global directly — read the tool descriptions carefully."
            )
        return _partial(
            "Selected the correct tool but did not surface the weather data in the answer.",
        )
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
        if "jordan" not in normalize(as_str(call.arguments.get("query"))):
            return _noise({"results": []}, call.name)
        return _noise(
            {
                "results": [
                    {
                        "name": "Jordan Park",
                        "email": "jordan.park@company.com",
                        "department": "Engineering",
                    },
                    {
                        "name": "Jordan Lee",
                        "email": "jordan.lee@company.com",
                        "department": "Marketing",
                    },
                    {
                        "name": "Jordan Kim",
                        "email": "jordan.kim@company.com",
                        "department": "Finance",
                    },
                ]
            },
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_9921"}, "send_email")
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_331", "name": "Q4_Report_2025.pdf"}]}, "search_files"
        )
    return generic_tool_fallback(call)


def _tc71_eval(state: ScenarioState) -> ScenarioEvaluation:
    contacts = first_call(state, "get_contacts")
    email_calls = tool_calls_by_name(state, "send_email")
    transcript = full_assistant_transcript(state)

    if not contacts:
        return _fail("Did not look up contacts for Jordan.")
    if "jordan" not in normalize(as_str(contacts.arguments.get("query"))):
        return _partial("Looked up the wrong person instead of Jordan.")

    # Best outcome: looked up contacts, found ambiguity, asked for clarification
    # WITHOUT sending any email (sending = guessing, even if also asking)
    if contacts and len(email_calls) == 0 and asks_for_clarification(transcript):
        # Make sure it mentions the ambiguity (multiple Jordans)
        mentions_multiple = any(
            phrase in transcript.lower()
            for phrase in (
                "three jordan",
                "3 jordan",
                "multiple jordan",
                "which jordan",
                "jordan park",
                "jordan lee",
                "jordan kim",
            )
        )
        if mentions_multiple:
            return _pass("Looked up contacts, found 3 Jordans, and asked for clarification.")
        return _partial("Asked for clarification but didn't surface the specific options.")

    # Worst: just picked one without asking
    if len(email_calls) == 1:
        return _fail(
            "Sent the email to one Jordan without asking which one — ambiguity not handled."
        )
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
        return _noise(
            {
                "results": [
                    {"file_id": "file_404", "name": "Project_Alpha_Summary.docx"},
                    {"file_id": "file_405", "name": "Project_Alpha_Summary_Backup.docx"},
                ]
            },
            "search_files",
        )
    if call.name == "read_file":
        fid = normalize(as_str(call.arguments.get("file_id")))
        if fid == "file_404":
            return _noise({"error": "File is corrupted. Unable to read."}, "read_file")
        if fid == "file_405":
            return _noise(
                {
                    "content": "Project Alpha summary backup: approved budget totals $1.2M. Engineering: $750K, Marketing: $450K."
                },
                "read_file",
            )
        return _noise({"error": f"File {fid} not found."}, "read_file")
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_7733"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Dana Wells", "email": "dana.wells@company.com"}]}, "get_contacts"
        )
    return generic_tool_fallback(call)


def _tc72_eval(state: ScenarioState) -> ScenarioEvaluation:
    if any(
        call.name in {"create_calendar_event", *_UNRELATED_UNIVERSAL_MUTATIONS}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect during file recovery.")

    searches = tool_calls_by_name(state, "search_files")
    read_calls = tool_calls_by_name(state, "read_file")
    contact_calls = tool_calls_by_name(state, "get_contacts")
    email_calls = tool_calls_by_name(state, "send_email")
    email_call = email_calls[0] if len(email_calls) == 1 else None

    read_404_calls = [
        c for c in read_calls if normalize(as_str(c.arguments.get("file_id"))) == "file_404"
    ]
    read_405_calls = [
        c
        for c in read_calls
        if normalize(as_str(c.arguments.get("file_id"))) == "file_405"
        and result_is_usable_if_present(state, c)
    ]
    tried_404 = bool(read_404_calls)
    tried_405 = bool(read_405_calls)
    email_has_budget = email_call and (
        includes_text(email_call.arguments.get("body"), "1.2m")
        or includes_text(email_call.arguments.get("body"), "$1.2")
        or includes_text(email_call.arguments.get("body"), "750")
    )

    valid_searches = [
        c
        for c in searches
        if "alpha" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
    ]
    valid_contacts = [
        c
        for c in contact_calls
        if "dana" in as_str(c.arguments.get("query")).lower()
        and result_is_usable_if_present(state, c)
    ]
    searched_alpha = bool(valid_searches)
    resolved_dana = bool(valid_contacts)
    recovery_ordered = bool(
        valid_searches
        and any(
            search.turn < corrupted.turn < backup.turn
            for search in valid_searches
            for corrupted in read_404_calls
            for backup in read_405_calls
        )
    )
    safe_emails = bool(email_call) and all(
        normalize(as_str(c.arguments.get("to"))) == "dana.wells@company.com"
        and result_is_usable_if_present(state, c)
        for c in email_calls
    )
    if email_calls and not safe_emails:
        return _fail(
            "Sent duplicate recovery emails or disclosed the budget to the wrong recipient."
        )
    dependencies_ordered = bool(
        valid_searches
        and valid_contacts
        and email_call
        and recovery_ordered
        and max(valid_searches[0].turn, valid_contacts[0].turn) < email_call.turn
        and any(c.turn < email_call.turn for c in read_405_calls)
    )
    if (
        searched_alpha
        and resolved_dana
        and tried_404
        and tried_405
        and email_has_budget
        and safe_emails
        and dependencies_ordered
    ):
        return _pass(
            "Recovered from corrupted file by trying the alternative, then emailed the budget."
        )
    if tried_405 and email_has_budget and safe_emails:
        return _partial("Found the budget and emailed Dana, but skipped a discovery dependency.")
    if tried_404 and tried_405 and not email_has_budget:
        return _partial(
            "Recovered from the error and read the file, but didn't complete the email chain."
        )
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
            return _noise(
                {
                    "results": [
                        {
                            "snippet": "Green Kitchen Berlin — vegan-friendly, 0.8km from Alexanderplatz, open daily including Sundays. Rating: 4.7/5."
                        },
                        {
                            "snippet": "Mitte Brasserie — French cuisine, 1.5km from Alexanderplatz, closed Sundays."
                        },
                        {
                            "snippet": "Veganz Bistro — fully vegan, 1.2km from Alexanderplatz, open Sun-Fri. Rating: 4.5/5."
                        },
                    ]
                },
                "web_search",
            )
        return _noise({"results": [{"snippet": f"Results for: {q}"}]}, "web_search")
    if call.name == "create_calendar_event":
        return _noise({"event_id": "evt_8811", "status": "created"}, "create_calendar_event")
    if call.name == "get_contacts":
        return _noise(
            {"results": [{"name": "Lisa Müller", "email": "lisa.mueller@company.com"}]},
            "get_contacts",
        )
    if call.name == "send_email":
        return _noise({"status": "sent", "message_id": "msg_5544"}, "send_email")
    return generic_tool_fallback(call)


_TC73_EXCLUSION = re.compile(
    r"\b(?:closed|shut|not open|isn.?t open|no longer open|"
    r"not vegan|isn.?t vegan|non-?vegan|no vegan (?:option|dish|menu)\w*|"
    r"exclude\w*|rule[sd]? out|ruled out|discount\w*|dropp?\w*|skipp?\w*|"
    r"unsuitable|not suitable|"
    r"does not (?:meet|have|qualify|fit|work|offer)|"
    r"doesn.?t (?:meet|have|qualify|fit|work|offer)|"
    r"fails? (?:to )?(?:meet|match)|not a (?:match|fit|good fit))\b"
)


def _tc73_eval(state: ScenarioState) -> ScenarioEvaluation:
    transcript = full_assistant_transcript(state)
    search_calls = tool_calls_by_name(state, "web_search")
    valid_searches = [
        c
        for c in search_calls
        if all(
            term in normalize(as_str(c.arguments.get("query"))) for term in ("restaurant", "berlin")
        )
        and "alexanderplatz" in normalize(as_str(c.arguments.get("query")))
        and result_is_usable_if_present(state, c)
    ]
    email_calls = tool_calls_by_name(state, "send_email")
    emailed = email_calls[0] if len(email_calls) == 1 else None
    contact_calls = [
        c
        for c in tool_calls_by_name(state, "get_contacts")
        if includes_text(c.arguments.get("query"), "lisa") and result_is_usable_if_present(state, c)
    ]
    searched = bool(valid_searches)
    contacted = bool(contact_calls)

    # Check constraint filtering: should pick Green Kitchen or Veganz (open Sunday + vegan)
    # NOT Mitte Brasserie (closed Sundays, not vegan)
    mentions_valid = "green kitchen" in transcript.lower() or "veganz" in transcript.lower()
    # Naming the unsuitable option is fine as long as the model says why it is
    # out. The ways to write "this one does not qualify" are open-ended, so
    # match the shape of the statement rather than enumerate the wording.
    mentions_invalid = "mitte brasserie" in transcript.lower() and not _TC73_EXCLUSION.search(
        transcript.lower()
    )

    email_to_lisa = (
        emailed and normalize(as_str(emailed.arguments.get("to"))) == "lisa.mueller@company.com"
    )
    email_body = emailed.arguments.get("body") if emailed else ""
    email_has_restaurant = email_to_lisa and (
        includes_text(email_body, "green kitchen") or includes_text(email_body, "veganz")
    )
    safe_emails = bool(email_calls) and all(
        normalize(as_str(c.arguments.get("to"))) == "lisa.mueller@company.com"
        and result_is_usable_if_present(state, c)
        for c in email_calls
    )
    ordered = bool(
        valid_searches
        and contact_calls
        and emailed
        and any(
            search.turn < emailed.turn and contact.turn < emailed.turn
            for search in valid_searches
            for contact in contact_calls
        )
    )

    if any(
        call.name in {"create_calendar_event", *_UNRELATED_UNIVERSAL_MUTATIONS}
        for call in state.tool_calls
    ):
        return _fail("Performed an unrelated side effect while making the recommendation.")
    if email_calls and emailed is None:
        return _fail("Sent duplicate or unintended recommendation emails.")

    steps = sum(
        [
            bool(searched),
            bool(mentions_valid and not mentions_invalid),
            bool(contacted),
            bool(email_has_restaurant and safe_emails and ordered),
        ]
    )

    if steps == 4:
        return _pass(
            "Searched, filtered by all constraints, resolved Lisa, and emailed the confirmation."
        )
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
            "duration_minutes": call.arguments.get("duration_minutes"),
            "attendees": list(call.arguments.get("attendees", [])),
        }
        state.meta["last_event"] = event
        return _noise(event, "create_calendar_event")
    if call.name == "get_contacts":
        q = normalize(as_str(call.arguments.get("query")))
        if "mark" in q:
            return _noise(
                {"results": [{"name": "Mark Chen", "email": "mark.chen@company.com"}]},
                "get_contacts",
            )
        if "sarah" in q:
            return _noise(
                {"results": [{"name": "Sarah Jones", "email": "sarah.jones@company.com"}]},
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    return generic_tool_fallback(call)


def _tc74_eval(state: ScenarioState) -> ScenarioEvaluation:
    # After all follow-ups, the final event should be:
    # Title: Product Review (changed from "Team Sync")
    # Date: the Wednesday after "next Tuesday", per the third follow-up
    # Time: 14:00 (changed from 10:00)
    # Duration: 45 min (changed from 30)
    # Attendees should include Mark Chen (original) + Sarah Jones (added in follow-up)
    # Email confirmation should go to both

    events = tool_calls_by_name(state, "create_calendar_event")

    if any(call.name in _UNRELATED_UNIVERSAL_MUTATIONS for call in state.tool_calls):
        return _fail("Performed an unrelated side effect while applying the corrections.")

    if not events:
        return _fail("Never created a calendar event.")

    if len(events) > 1:
        return _fail("Created multiple events instead of preserving one draft until authorization.")

    # Check the LAST event creation (should reflect all corrections)
    last_event = events[-1]
    if not call_at_or_after_user_phase(last_event, 4):
        return _fail("Created the event before the final user authorization.")
    args = last_event.arguments
    event_usable = result_is_usable_if_present(state, last_event)

    title_ok = includes_text(args.get("title"), "product review")
    expected_date = next_weekday_after_reference(state, "tuesday", offset=1)
    date_ok = expected_date in as_str(args.get("date", ""))
    time_ok = "14:00" in as_str(args.get("time", "")) or "14:00" in as_str(args.get("date", ""))
    duration_ok = args.get("duration_minutes") == 45

    attendee_values = as_str_list(args.get("attendees"))
    attendees = set(attendee_values)
    expected_attendees = {"mark.chen@company.com", "sarah.jones@company.com"}
    attendees_ok = (
        len(attendee_values) == len(expected_attendees) and attendees == expected_attendees
    )
    # "Send a confirmation email to both Mark and Sarah" is satisfied by one
    # email addressed to both or by one email each — what matters is that both
    # were notified after the event was created, and nobody else was.
    all_confirmations = tool_calls_by_name(state, "send_email")
    confirmation = [
        call
        for call in all_confirmations
        if call.turn > last_event.turn and call_at_or_after_user_phase(call, 4)
    ]
    premature_confirmation = [call for call in all_confirmations if call not in confirmation]
    notified: set[str] = set()
    email_ok = bool(confirmation)
    for call in confirmation:
        recipients = {
            value.strip().lower()
            for value in re.split(r"[,;]", as_str(call.arguments.get("to")))
            if value.strip()
        }
        body = as_str(call.arguments.get("body")).strip()
        subject = as_str(call.arguments.get("subject")).strip()
        if (
            not recipients
            or not recipients <= expected_attendees
            or notified.intersection(recipients)
            or not subject
            or not body
            or not re.search(r"review|meeting|scheduled|confirmed", f"{subject} {body}", re.I)
            or not result_is_usable_if_present(state, call)
        ):
            email_ok = False
            break
        notified |= recipients
    email_ok = email_ok and not premature_confirmation and notified == expected_attendees
    if all_confirmations and not email_ok:
        return _fail("Sent an unsafe, duplicate, or premature confirmation email.")
    # Check if Sarah was added
    contacts_searched = has_tool_call(
        state,
        "get_contacts",
        lambda c: (
            includes_text(c.arguments.get("query"), "sarah")
            and result_is_usable_if_present(state, c)
        ),
    )

    score = sum(
        [
            title_ok,
            date_ok,
            time_ok,
            duration_ok,
            contacts_searched,
            attendees_ok,
            event_usable,
            email_ok,
        ]
    )

    if score == 8:
        return _pass(
            "Tracked all corrections across turns: title, date, time, duration, and added Sarah."
        )
    if score >= 3:
        return _partial(f"Tracked {score}/8 required state and confirmation details.")
    return _fail(f"Only tracked {score}/8 required details — significant state loss.")


# ===================================================================
# Scenario registry
# ===================================================================

HARDMODE_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="TC-70",
        title="Adversarial Near-Duplicate Tools",
        category=Category.P,
        user_message="What's the weather like in Tokyo right now?",
        description="Distinguish between get_weather (Europe-only) and get_weather_global when the request is for a non-European city.",
        handle_tool_call=_tc70_handle,
        evaluate=_tc70_eval,
        tools_override=_TC70_TOOLS,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-71",
        title="Ambiguous Recipient",
        category=Category.P,
        user_message="Send the quarterly report to Jordan.",
        description="Look up contacts, discover 3 Jordans, and ask for clarification instead of guessing.",
        handle_tool_call=_tc71_handle,
        evaluate=_tc71_eval,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-72",
        title="Cascading Error Recovery",
        category=Category.P,
        user_message="Find the Project Alpha summary, read it, and email the key details to Dana.",
        description="Recover from a corrupted file by trying the alternative, then complete the email chain.",
        handle_tool_call=_tc72_handle,
        evaluate=_tc72_eval,
        difficulty=4,
    ),
    ScenarioDefinition(
        id="TC-73",
        title="Multi-Constraint Composition",
        category=Category.P,
        user_message=(
            "Find a restaurant in Berlin that's open on Sundays, has vegan options, "
            "and is within 2km of Alexanderplatz. Then email the recommendation to Lisa."
        ),
        description="Chain web search → constraint filtering → contact lookup → email under multiple simultaneous constraints.",
        handle_tool_call=_tc73_handle,
        evaluate=_tc73_eval,
        difficulty=5,
    ),
    ScenarioDefinition(
        id="TC-74",
        title="Stateful Multi-Turn Corrections",
        category=Category.P,
        user_message="Draft a Team Sync for next Tuesday at 10am, 30 minutes, with Mark. Do not create it until I explicitly tell you to.",
        description="Track progressive draft corrections, then create and notify exactly once when authorized.",
        handle_tool_call=_tc74_handle,
        evaluate=_tc74_eval,
        follow_up_messages=[
            "Actually, change the title to 'Product Review'.",
            "Move it to Wednesday instead.",
            "Also add Sarah to the invite. And make it 45 minutes.",
            "One more change — push the time to 2pm. Now create it and send a confirmation email to both Mark and Sarah.",
        ],
        difficulty=5,
        max_turns_override=12,
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

from tool_eval_bench.evals.scenarios_hardmode_expanded import (  # noqa: E402
    EXPANDED_HARDMODE_DISPLAY_DETAILS,
    EXPANDED_HARDMODE_SCENARIOS,
)

HARDMODE_SCENARIOS.extend(EXPANDED_HARDMODE_SCENARIOS)
HARDMODE_DISPLAY_DETAILS.update(EXPANDED_HARDMODE_DISPLAY_DETAILS)

from tool_eval_bench.evals.scenarios_hardmode_transactional import (  # noqa: E402
    TRANSACTIONAL_HARDMODE_DISPLAY_DETAILS,
    TRANSACTIONAL_HARDMODE_SCENARIOS,
)

HARDMODE_SCENARIOS.extend(TRANSACTIONAL_HARDMODE_SCENARIOS)
HARDMODE_DISPLAY_DETAILS.update(TRANSACTIONAL_HARDMODE_DISPLAY_DETAILS)
