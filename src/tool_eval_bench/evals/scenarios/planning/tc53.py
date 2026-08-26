"""TC-53 — Conditional Planning."""

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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
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
    _result_has_status,
    _result_matches_if_present,
)


def _tc53_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "London",
                "temperature": 11,
                "units": "celsius",
                "condition": "Rain",
                "humidity": 88,
                "forecast": "Rain expected through weekend.",
            },
            "get_weather",
        )
    if call.name == "create_calendar_event":
        return _noise({"status": "created", "event_id": "evt_tc53"}, "create_calendar_event")
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "get_contacts":
        return _noise(
            {
                "results": [
                    {"name": "Team Lead", "email": "lead@company.com"},
                    {"name": "Dev Team", "email": "dev-team@company.com"},
                ]
            },
            "get_contacts",
        )
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc53_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Check if it's going to rain in London this weekend, and if so,
    move our outdoor meeting to the office and notify the attendees.'

    Model must: check weather → see rain → act on it.
    The weather WILL show rain, so the conditional branch should be taken.
    Since there's no update_event tool, informing/notifying is the best action.
    """

    def weather_result_is_rain(payload: Any) -> bool:
        return "rain" in str(payload).lower()

    def event_result_is_created(payload: Any) -> bool:
        return _result_has_status(payload, "created", "evt_tc53")

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    weather_calls = [
        c
        for c in _tool_calls_by_name(state, "get_weather")
        if _includes_text(c.arguments.get("location", ""), "london")
        and _result_matches_if_present(state, c, weather_result_is_rain)
    ]
    checked_weather = bool(weather_calls)
    event_calls = _tool_calls_by_name(state, "create_calendar_event")

    if any(call.name in _UNRELATED_UNIVERSAL_MUTATIONS for call in state.tool_calls):
        return _fail("Performed an unrelated side effect while handling the weather plan.")

    answer = state.final_answer.lower()
    mentions_office = any(w in answer for w in ("office", "indoor", "inside", "moved", "relocated"))
    mentions_rain = "rain" in answer

    # Only the scenario's attendee addresses are authorized.  An address
    # appearing in an arbitrary tool result, such as a search snippet, is not
    # proof that it is an intended recipient.
    intended_recipients = {
        "lead@company.com",
        "dev-team@company.com",
        "team@company.com",
    }
    for contact_call in _tool_calls_by_name(state, "get_contacts"):
        for result in _matching_tool_results(state, contact_call):
            intended_recipients.update(
                address.lower()
                for address in re.findall(r"[\w.+-]+@[\w.-]+\.\w+", str(result.result))
            )
    notifications = [c for c in state.tool_calls if c.name == "send_email"]
    recipient_sets = [
        {
            value.strip().lower()
            for value in re.split(r"[,;]", _as_str(notification.arguments.get("to")))
            if value.strip()
        }
        for notification in notifications
    ]
    notified: set[str] = set()
    no_duplicate_recipients = True
    for recipients in recipient_sets:
        if notified.intersection(recipients):
            no_duplicate_recipients = False
            break
        notified |= recipients
    safe_notifications = (
        bool(recipient_sets)
        and all(
            values
            and values <= intended_recipients
            and any(weather.turn < notification.turn for weather in weather_calls)
            and _as_str(notification.arguments.get("subject")).strip()
            and _as_str(notification.arguments.get("body")).strip()
            and _result_matches_if_present(state, notification, email_result_is_sent)
            for notification, values in zip(notifications, recipient_sets, strict=True)
        )
        and no_duplicate_recipients
    )
    if len(event_calls) > 1:
        return _fail("Created duplicate calendar events while handling the conditional request.")
    valid_events = all(
        any(weather.turn <= event.turn for weather in weather_calls)
        and any(
            term in _as_str(event.arguments.get("title")).lower()
            for term in ("meeting", "sync", "office", "indoor")
        )
        and _result_matches_if_present(state, event, event_result_is_created)
        for event in event_calls
    )
    if event_calls and not valid_events:
        return _fail("Created an unrelated or premature calendar event.")
    if not checked_weather and _tool_calls_by_name(state, "get_weather"):
        return _fail("Claimed a rain-based action after the weather lookup failed.")
    if checked_weather and notifications and not safe_notifications:
        return _fail("Sent the weather relocation notice to an unverified recipient.")
    if (
        checked_weather
        and notifications
        and mentions_rain
        and mentions_office
        and safe_notifications
    ):
        return _pass("Checked weather → detected rain → moved to office → notified attendees.")
    if checked_weather and (event_calls or notifications) and mentions_rain:
        return _partial("Detected rain and partially acted on the conditional plan.")
    if checked_weather and mentions_rain:
        return _partial("Checked weather and noted rain but didn't take action (event/notify).")
    if checked_weather:
        return _partial("Checked weather but didn't follow through on the conditional plan.")
    return _fail("Did not check weather or execute the conditional workflow.")


SCENARIO = ScenarioDefinition(
    id="TC-53",
    title="Conditional Planning",
    category=Category.M,
    user_message="Check if it's going to rain in London this weekend, and if so, move our outdoor meeting to the office and notify the attendees.",
    description="Execute conditional workflow: weather check → branch on rain → act.",
    handle_tool_call=_tc53_handle,
    evaluate=_tc53_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it checks weather → detects rain → moves meeting to office → notifies.",
    "Fail if it ignores the conditional or doesn't act on the rain result.",
)
