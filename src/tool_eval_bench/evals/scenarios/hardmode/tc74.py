"""TC-74 — Stateful Multi-Turn Corrections."""

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
    call_at_or_after_user_phase,
    generic_tool_fallback,
    has_tool_call,
    includes_text,
    matching_tool_results,
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
from tool_eval_bench.evals.scenarios.hardmode._shared import (
    _UNRELATED_UNIVERSAL_MUTATIONS,
)


def _tc74_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "create_calendar_event":
        event = {
            "event_id": "evt_9900",
            "status": "created",
            "title": as_str(call.arguments.get("title")),
            "date": as_str(call.arguments.get("date")),
            "time": as_str(call.arguments.get("time")),
            "duration_minutes": call.arguments.get("duration_minutes"),
            "attendees": as_str_list(call.arguments.get("attendees")),
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
    event_usable = result_is_usable_if_present(state, last_event) and (
        not matching_tool_results(state, last_event)
        or any(
            isinstance(result.result, dict) and result.result.get("status") == "created"
            for result in matching_tool_results(state, last_event)
        )
    )

    title_ok = includes_text(args.get("title"), "product review")
    expected_date = next_weekday_after_reference(state, "tuesday", offset=1)
    date_ok = expected_date in as_str(args.get("date", ""))
    time_ok = "14:00" in as_str(args.get("time", ""))
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
        if call.turn >= last_event.turn and call_at_or_after_user_phase(call, 4)
    ]
    premature_confirmation = [call for call in all_confirmations if call not in confirmation]
    notified: set[str] = set()
    email_ok = bool(confirmation)
    for call in confirmation:
        # The confirmation may address both recipients in one `to` field, or
        # put one of them in `cc` — "to both" does not dictate which field each
        # recipient lands in, so collect from both fields.
        recipients = {
            value.strip().lower()
            for value in (
                re.split(r"[,;]", as_str(call.arguments.get("to")))
                + re.split(r"[,;]", as_str(call.arguments.get("cc")))
            )
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
            or (
                matching_tool_results(state, call)
                and not any(
                    isinstance(result.result, dict) and result.result.get("status") == "sent"
                    for result in matching_tool_results(state, call)
                )
            )
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
            and (
                not matching_tool_results(state, c)
                or any(
                    isinstance(result.result, dict)
                    and any(
                        normalize(as_str(item.get("email"))) == "sarah.jones@company.com"
                        for item in result.result.get("results", []) or []
                        if isinstance(item, dict)
                    )
                    for result in matching_tool_results(state, c)
                )
            )
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


SCENARIO = ScenarioDefinition(
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
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if the final event reflects all 4 rounds of corrections (title/date/time/duration/attendees).",
    "Fail if state is lost across turns — e.g. reverts title or forgets Sarah.",
)
