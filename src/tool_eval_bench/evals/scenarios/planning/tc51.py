"""TC-51 — Goal-Level Planning."""

from __future__ import annotations

from datetime import date, timedelta
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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    as_str_list as _as_str_list,
)
from tool_eval_bench.evals.helpers import (
    asks_for_clarification as _asks_clarification,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.planning._shared import (
    _UNRELATED_UNIVERSAL_MUTATIONS,
    _recipient_set,
    _result_has_status,
    _result_matches_if_present,
)

_TC51_ENGINEERING = frozenset({"alice@company.com", "bob@company.com", "carol@company.com"})


def _tc51_contacts_result_is_engineering(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    contacts = payload.get("results")
    if not isinstance(contacts, list):
        return False
    emails = {
        _as_str(contact.get("email")).strip().lower()
        for contact in contacts
        if isinstance(contact, dict)
    }
    return _TC51_ENGINEERING <= emails


def _tc51_event_result_is_created(payload: Any) -> bool:
    return _result_has_status(payload, "created", "evt_tc51")


def _tc51_email_result_is_sent(payload: Any) -> bool:
    return _result_has_status(payload, "sent")


def _tc51_friday(state: ScenarioState) -> str:
    """Return the date meant by ``this Friday`` for the benchmark date."""
    raw = _as_str(state.meta.get("reference_date")).strip() or "2026-03-20"
    try:
        reference = date.fromisoformat(raw)
    except ValueError:
        reference = date.fromisoformat("2026-03-20")
    days_ahead = (4 - reference.weekday()) % 7
    return (reference + timedelta(days=days_ahead)).isoformat()


def _tc51_date_is_friday(state: ScenarioState, value: Any) -> bool:
    # Older synthetic traces did not carry benchmark metadata.  Keep those
    # traces evaluable while enforcing the relative date for runtime traces.
    date_value = _as_str(value).strip()
    if "reference_date" not in state.meta:
        return bool(date_value)
    return date_value == _tc51_friday(state)


def _tc51_time_is_present(state: ScenarioState, value: Any) -> bool:
    return "reference_date" not in state.meta or bool(_as_str(value).strip())


def _tc51_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        query = _as_str(call.arguments.get("query", ""))
        if _includes_text(query, "engineering") or _includes_text(query, "team"):
            return _noise(
                {
                    "results": [
                        {"name": "Alice Chen", "email": "alice@company.com", "role": "engineer"},
                        {"name": "Bob Park", "email": "bob@company.com", "role": "engineer"},
                        {"name": "Carol Singh", "email": "carol@company.com", "role": "engineer"},
                    ]
                },
                "get_contacts",
            )
        return _noise({"results": []}, "get_contacts")
    if call.name == "create_calendar_event":
        return _noise({"status": "created", "event_id": "evt_tc51"}, "create_calendar_event")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "Popular team lunch restaurants near downtown."}]},
            "web_search",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc51_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Organize a team lunch for this Friday with the engineering team.'

    Model must autonomously decompose into: lookup contacts → create event → notify.
    No step-by-step guidance is provided.
    """
    contact_calls = [
        c
        for c in _tool_calls_by_name(state, "get_contacts")
        if (
            _includes_text(c.arguments.get("query", ""), "engineer")
            or _includes_text(c.arguments.get("query", ""), "team")
        )
        and _result_matches_if_present(state, c, _tc51_contacts_result_is_engineering)
    ]
    event_calls = _tool_calls_by_name(state, "create_calendar_event")
    email_calls = _tool_calls_by_name(state, "send_email")
    valid_events = [
        c
        for c in event_calls
        if _includes_text(c.arguments.get("title"), "lunch")
        and _tc51_date_is_friday(state, c.arguments.get("date"))
        and _tc51_time_is_present(state, c.arguments.get("time"))
        and len(_as_str_list(c.arguments.get("attendees"))) == len(_TC51_ENGINEERING)
        and set(_as_str_list(c.arguments.get("attendees"))) == _TC51_ENGINEERING
        and _result_matches_if_present(state, c, _tc51_event_result_is_created)
    ]
    valid_event = valid_events[0] if len(valid_events) == 1 else None

    if any(call.name in _UNRELATED_UNIVERSAL_MUTATIONS for call in state.tool_calls):
        return _fail("Performed an unrelated side effect while planning the lunch.")

    notified: set[str] = set()
    notifications_valid = bool(email_calls)
    for call in email_calls:
        recipients = _recipient_set(call.arguments.get("to"))
        if (
            call.turn <= (valid_event.turn if valid_event else -1)
            or not recipients
            or not recipients <= _TC51_ENGINEERING
            or notified.intersection(recipients)
            or not _as_str(call.arguments.get("subject")).strip()
            or not _as_str(call.arguments.get("body")).strip()
            or not _result_matches_if_present(state, call, _tc51_email_result_is_sent)
        ):
            notifications_valid = False
            break
        notified |= recipients
    notification_complete = notifications_valid and notified == _TC51_ENGINEERING
    got_contacts = bool(contact_calls)
    event_ordered = bool(
        valid_event and any(call.turn < valid_event.turn for call in contact_calls)
    )

    # Any extra event or recipient is an unsafe side effect, even if a later
    # call happens to satisfy the happy-path fields.
    if len(event_calls) > 1:
        return _fail("Created an invalid or duplicate event while planning the lunch.")
    if (
        email_calls
        and not notifications_valid
        and any(
            _as_str(call.arguments.get("subject")).strip()
            or _as_str(call.arguments.get("body")).strip()
            for call in email_calls
        )
    ):
        return _fail("Sent an invalid, duplicate, or unintended lunch notification.")

    if got_contacts and valid_event and event_ordered and notification_complete:
        return _pass("Autonomously decomposed goal: contacts → event → notify.")

    steps = sum([got_contacts, bool(valid_event and event_ordered), notification_complete])
    if steps == 2:
        missing = []
        if not got_contacts:
            missing.append("contact lookup")
        if not valid_event or not event_ordered:
            missing.append("event creation")
        if not notification_complete:
            missing.append("notification")
        return _partial(f"Completed 2/3 planning steps. Missing: {', '.join(missing)}.")
    relevant_calls = [
        c
        for c in state.tool_calls
        if c.name in {"get_contacts", "create_calendar_event", "send_email"}
    ]
    if (
        got_contacts
        or valid_event
        or any(c.name == "create_calendar_event" for c in relevant_calls)
        or len(relevant_calls) >= 2
    ):
        missing = []
        if not got_contacts:
            missing.append("contact lookup")
        if not valid_event or not event_ordered:
            missing.append("event creation")
        if not notification_complete:
            missing.append("notification")
        return _partial(f"Started planning but missing: {', '.join(missing)}.")
    # Asking for clarification is acceptable for an ambiguous goal
    if _asks_clarification(state.final_answer):
        return _partial(
            "Asked for clarification instead of planning — reasonable but not proactive."
        )
    return _fail("Did not decompose the goal into any tool actions.")


SCENARIO = ScenarioDefinition(
    id="TC-51",
    title="Goal-Level Planning",
    category=Category.M,
    user_message="Organize a team lunch for this Friday with the engineering team.",
    description="Autonomously decompose goal into contacts → event → notify workflow.",
    handle_tool_call=_tc51_handle,
    evaluate=_tc51_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it autonomously decomposes: contacts → calendar event → email notification.",
    "Fail if it doesn't break down the goal into tool actions.",
)
