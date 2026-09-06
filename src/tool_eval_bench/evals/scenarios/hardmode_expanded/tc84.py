"""TC-84 — Long-Horizon Recovery With Constraint Retention."""

from __future__ import annotations

import copy
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
    next_weekday_after_reference,
    recipient_values,
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
from tool_eval_bench.evals.scenarios.hardmode_expanded._shared import (
    _EMAIL,
    _ROOMS,
    _STRING,
    _failed_result_if_present,
    _result_matches_if_present,
    _tool,
)

_TC84_MAX_FAILED_ATTEMPTS = 3


_TC84_TOOLS = [
    _tool("get_contacts", "Look up contacts.", {"query": _STRING}, ["query"]),
    _tool(
        "search_slots",
        "Search meeting slots.",
        {"date": _STRING, "period": _STRING, "duration_minutes": {"type": "integer"}},
        ["date", "period", "duration_minutes"],
    ),
    _tool(
        "search_rooms",
        "Search rooms.",
        {"office": _STRING, "minimum_capacity": {"type": "integer"}},
        ["office", "minimum_capacity"],
    ),
    _tool("search_files", "Search files.", {"query": _STRING}, ["query"]),
    _tool(
        "book_room",
        "Book a room.",
        {
            "room_id": _STRING,
            "date": _STRING,
            "time": _STRING,
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": _EMAIL},
        },
        ["room_id", "date", "time", "duration_minutes", "attendees"],
    ),
    _tool(
        "send_email",
        "Send email.",
        {
            "to": _EMAIL,
            "subject": _STRING,
            "body": _STRING,
            "attachments": {"type": "array", "items": _STRING},
        },
        ["to", "subject", "body"],
    ),
]


def _tc84_slot_date(state: ScenarioState) -> str:
    """The "next Wednesday" the prompt asks for, relative to the reference date."""
    return next_weekday_after_reference(state, "wednesday")


def _tc84_room_id(call: ToolCallRecord) -> str | None:
    """Return a hashable room id, treating malformed arguments as unknown."""
    room_id = call.arguments.get("room_id")
    return room_id if isinstance(room_id, str) else None


def _tc84_contact_call_is_valid(state: ScenarioState, call: ToolCallRecord) -> bool:
    query = as_str(call.arguments.get("query")).lower()
    if not any(name in query for name in ("elena", "ravi")):
        return False
    if not result_is_usable_if_present(state, call):
        return False
    return _result_matches_if_present(
        state,
        call,
        lambda payload: all(
            any(
                isinstance(item, dict)
                and name in as_str(item.get("name")).lower()
                and as_str(item.get("email")).lower() == f"{name}@company.com"
                for item in payload.get("results", [])
            )
            for name in ("elena", "ravi")
            if name in query
        ),
    )


def _tc84_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        query = as_str(call.arguments.get("query")).strip().lower()
        results = []
        if not query or "elena" in query:
            results.append({"name": "Elena", "email": "elena@company.com"})
        if not query or "ravi" in query:
            results.append({"name": "Ravi", "email": "ravi@company.com"})
        if not results:
            results = [
                {"name": "Elena", "email": "elena@company.com"},
                {"name": "Ravi", "email": "ravi@company.com"},
            ]
        return _noise({"results": results}, call.name)
    if call.name == "search_slots":
        # The offered slot has to be the day the prompt asked for, or the
        # simulator contradicts the reference date the model was given.
        return _noise(
            {
                "slots": [
                    {
                        "date": _tc84_slot_date(state),
                        "time": "14:00",
                        "duration_minutes": 45,
                    }
                ]
            },
            call.name,
        )
    if call.name == "search_rooms":
        # Once berlin_3a has lost the race it stops being advertised. Returning
        # it again would invite a retry the evaluator then has to punish.
        rooms = [
            copy.deepcopy(room)
            for room in _ROOMS
            if not (room["room_id"] == "berlin_3a" and state.meta.get("berlin_3a_failed"))
        ]
        return _noise({"rooms": rooms}, call.name)
    if call.name == "search_files":
        return _noise({"results": [{"file_id": "agenda_q2", "name": "Agenda_Q2.pdf"}]}, call.name)
    if call.name == "book_room":
        room_id = call.arguments.get("room_id")
        if room_id == "berlin_3a":
            state.meta["berlin_3a_failed"] = True
            return _noise({"error": "Room became unavailable during booking."}, call.name)
        if room_id == "berlin_5b":
            state.meta["successful_booking_turn"] = call.turn
            return _noise({"status": "booked", "booking_id": "booking_84"}, call.name)
        return _noise({"error": "Invalid room for Berlin workflow."}, call.name)
    if call.name == "send_email":
        return _noise({"status": "sent"}, call.name)
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc84_eval(state: ScenarioState) -> ScenarioEvaluation:
    bookings = tool_calls_by_name(state, "book_room")
    emails = tool_calls_by_name(state, "send_email")
    contacts = tool_calls_by_name(state, "get_contacts")
    valid_contacts = [call for call in contacts if _tc84_contact_call_is_valid(state, call)]
    contact_names = {
        name
        for call in valid_contacts
        for name in ("elena", "ravi")
        if name in as_str(call.arguments.get("query")).lower()
    }
    discovery_calls = {
        "get_contacts": valid_contacts if contact_names == {"elena", "ravi"} else [],
        "search_slots": [
            c
            for c in tool_calls_by_name(state, "search_slots")
            if c.arguments.get("date") == _tc84_slot_date(state)
            and c.arguments.get("period") == "afternoon"
            and c.arguments.get("duration_minutes") == 45
            and result_is_usable_if_present(state, c)
            and _result_matches_if_present(
                state,
                c,
                lambda payload: any(
                    isinstance(slot, dict)
                    and slot.get("date") == _tc84_slot_date(state)
                    and slot.get("time") == "14:00"
                    and slot.get("duration_minutes") == 45
                    for slot in payload.get("slots", [])
                ),
            )
        ],
        "search_rooms": [
            c
            for c in tool_calls_by_name(state, "search_rooms")
            if c.arguments.get("office") == "Berlin"
            and c.arguments.get("minimum_capacity") == 3
            and result_is_usable_if_present(state, c)
            and _result_matches_if_present(
                state,
                c,
                lambda payload: any(
                    isinstance(room, dict)
                    and room.get("office") == "Berlin"
                    and room.get("capacity", 0) >= 3
                    for room in payload.get("rooms", [])
                ),
            )
        ],
        "search_files": [
            c
            for c in tool_calls_by_name(state, "search_files")
            if "agenda" in as_str(c.arguments.get("query")).lower()
            and result_is_usable_if_present(state, c)
            and _result_matches_if_present(
                state,
                c,
                lambda payload: any(
                    isinstance(item, dict) and item.get("file_id") == "agenda_q2"
                    for item in payload.get("results", [])
                ),
            )
        ],
    }
    success = [
        call
        for call in bookings
        if _tc84_room_id(call) == "berlin_5b"
        and result_is_usable_if_present(state, call)
        and _result_matches_if_present(
            state,
            call,
            lambda payload: (
                payload.get("status") == "booked" and payload.get("booking_id") == "booking_84"
            ),
        )
    ]
    failures = [
        call
        for call in bookings
        if _tc84_room_id(call) == "berlin_3a" and _failed_result_if_present(state, call)
    ]
    # A failed booking mutates nothing, so a bounded retry is a reasonable race
    # response rather than a failure. What matters is that exactly one booking
    # succeeded, that it was the fallback room, and that the model stopped.
    if (
        any(_tc84_room_id(call) not in {"berlin_3a", "berlin_5b"} for call in bookings)
        or not state.meta.get("berlin_3a_failed")
        or len(success) != 1
        or not 1 <= len(failures) <= _TC84_MAX_FAILED_ATTEMPTS
    ):
        return _fail("Did not recover from the Berlin room booking race.")
    if not all(discovery_calls.values()):
        return _partial("Recovered the booking without completing the required discovery workflow.")
    booking = success[-1]
    first_failure, last_failure = failures[0], failures[-1]
    attendee_values = booking.arguments.get("attendees")
    attendee_set = set(attendee_values) if isinstance(attendee_values, list) else set()
    booking_ok = (
        isinstance(attendee_values, list)
        and len(attendee_values) == 2
        and booking.arguments.get("date") == _tc84_slot_date(state)
        and booking.arguments.get("time") == "14:00"
        and booking.arguments.get("duration_minutes") == 45
        and attendee_set == {"elena@company.com", "ravi@company.com"}
    )

    def _retained_constraints(call: ToolCallRecord) -> bool:
        attendees = call.arguments.get("attendees")
        return (
            call.arguments.get("date") == _tc84_slot_date(state)
            and call.arguments.get("time") == "14:00"
            and call.arguments.get("duration_minutes") == 45
            and isinstance(attendees, list)
            and len(attendees) == 2
            and set(attendees) == {"elena@company.com", "ravi@company.com"}
        )

    # Every attempt, not only the first: a retry that quietly drops an attendee
    # has lost the constraint just as surely as the successful booking would.
    failure_ok = all(_retained_constraints(call) for call in failures)
    expected_recipients = {"elena@company.com", "ravi@company.com"}
    accepted_agenda_refs = {"agenda_q2", "agenda_q2.pdf", "/documents/agenda_q2.pdf"}
    notified: set[str] = set()
    email_ok = bool(emails)
    unsafe_email = False
    for email in emails:
        recipients = set(recipient_values(email.arguments.get("to")))
        raw_attachments = email.arguments.get("attachments", [])
        attachments = (
            [as_str(value).strip().lower() for value in raw_attachments]
            if isinstance(raw_attachments, list)
            else []
        )
        unsafe_email = (
            email.turn <= booking.turn
            or not recipients
            or not recipients <= expected_recipients
            or bool(notified.intersection(recipients))
            or any(attachment not in accepted_agenda_refs for attachment in attachments)
            or not result_is_usable_if_present(state, email)
            or not _result_matches_if_present(
                state, email, lambda payload: payload.get("status") == "sent"
            )
        )
        if unsafe_email:
            break
        if (
            len(attachments) != 1
            or not as_str(email.arguments.get("subject")).strip()
            or not as_str(email.arguments.get("body")).strip()
        ):
            email_ok = False
            break
        notified |= recipients
    if unsafe_email:
        return _fail("Sent an unsafe, duplicate, premature, or over-attached notification.")
    email_ok = email_ok and notified == expected_recipients
    booking_ordered = (
        all(any(c.turn < first_failure.turn for c in calls) for calls in discovery_calls.values())
        and last_failure.turn < booking.turn
    )
    if failure_ok and booking_ok and email_ok and booking_ordered:
        return _pass("Recovered from the room race and completed the constrained Berlin workflow.")
    if booking_ok and not failure_ok:
        return _partial(
            "Recovered the valid booking but dropped a constraint on an earlier attempt."
        )
    if booking_ok:
        return _partial(
            "Recovered the valid booking but left the email or agenda workflow incomplete."
        )
    return _fail("Lost an original constraint while recovering from the booking race.")


SCENARIO = ScenarioDefinition(
    "TC-84",
    "Long-Horizon Recovery With Constraint Retention",
    Category.P,
    "Find a 45-minute slot next Wednesday afternoon for Elena and Ravi, use the Berlin office only, book the smallest room that fits three people, attach the agenda, and email both attendees.",
    "Recover from a booking race while retaining all workflow constraints.",
    _tc84_handle,
    _tc84_eval,
    tools_override=_TC84_TOOLS,
    difficulty=5,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it recovers from the booking race with constraints intact.",
    "Fail if recovery loses state.",
)
