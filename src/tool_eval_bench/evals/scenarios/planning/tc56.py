"""TC-56 — Notification Workflow."""

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
    answer_affirms_number as _answer_affirms_number,
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
    _call_index,
    _has_unexpected_tools,
    _is_tomorrow_morning,
    _result_has_status,
    _result_matches_if_present,
)


def _tc56_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        location = _as_str(call.arguments.get("location", "")).lower()
        if "nyc" in location or "new york" in location:
            return _noise(
                {
                    "location": "New York City",
                    "temperature": -3,
                    "units": "celsius",
                    "condition": "Snow",
                    "humidity": 75,
                },
                "get_weather",
            )
        return _noise(
            {"location": location, "temperature": 15, "condition": "Clear"},
            "get_weather",
        )
    if call.name == "send_email":
        return _noise({"status": "sent"}, "send_email")
    if call.name == "set_reminder":
        return _noise({"status": "set", "reminder_id": "rem_tc56"}, "set_reminder")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc56_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Check the weather in NYC. If it's below freezing, email me
    a warning and set a reminder to dress warmly tomorrow morning.'

    Must: get_weather(NYC) → see -3°C (below freezing) → send_email + set_reminder.
    Temperature IS below freezing, so the email should be sent.
    """

    def weather_result_is_freezing(payload: Any) -> bool:
        return "-3" in str(payload) or "snow" in str(payload).lower()

    def email_result_is_sent(payload: Any) -> bool:
        return _result_has_status(payload, "sent")

    def reminder_result_is_set(payload: Any) -> bool:
        return _result_has_status(payload, "set", "rem_tc56")

    weather_calls = [
        call
        for call in _tool_calls_by_name(state, "get_weather")
        if (
            _includes_text(call.arguments.get("location", ""), "nyc")
            or _includes_text(call.arguments.get("location", ""), "new york")
        )
        and _result_matches_if_present(state, call, weather_result_is_freezing)
    ]
    email_calls = [
        call
        for call in _tool_calls_by_name(state, "send_email")
        if (
            _as_str(call.arguments.get("to")).strip().lower() == "user@company.com"
            and _as_str(call.arguments.get("subject")).strip()
            and _as_str(call.arguments.get("body")).strip()
            and any(
                term
                in _as_str(call.arguments.get("subject")).lower()
                + " "
                + _as_str(call.arguments.get("body")).lower()
                for term in ("freez", "cold", "warm", "-3")
            )
        )
        and _result_matches_if_present(state, call, email_result_is_sent)
    ]
    reminder_calls = [
        call
        for call in _tool_calls_by_name(state, "set_reminder")
        if (
            "warm" in _as_str(call.arguments.get("message")).lower()
            and _is_tomorrow_morning(call.arguments.get("datetime"), state)
        )
        and _result_matches_if_present(state, call, reminder_result_is_set)
    ]
    checked_weather = bool(weather_calls)
    sent_email = bool(email_calls)
    set_reminder = bool(reminder_calls)

    weather_index = _call_index(state, weather_calls[0]) if weather_calls else -1
    email_after_weather = bool(email_calls and _call_index(state, email_calls[0]) > weather_index)
    reminder_after_weather = bool(
        reminder_calls and _call_index(state, reminder_calls[0]) > weather_index
    )
    email_calls_are_safe = len(email_calls) == len(_tool_calls_by_name(state, "send_email"))
    reminder_calls_are_safe = len(reminder_calls) == len(_tool_calls_by_name(state, "set_reminder"))

    answer = state.final_answer.lower()
    mentions_cold = any(
        _answer_affirms_number(state.final_answer, "-3") or w in answer
        for w in ("below freezing", "cold", "freez", "snow", "warning")
    )

    if (
        checked_weather
        and sent_email
        and set_reminder
        and mentions_cold
        and email_after_weather
        and reminder_after_weather
        and email_calls_are_safe
        and reminder_calls_are_safe
        and not _has_unexpected_tools(state, {"get_weather", "send_email", "set_reminder"})
    ):
        return _pass("Composed weather check → detected freezing → sent warning email.")
    if checked_weather and sent_email:
        return _partial("Checked weather and sent email but didn't note the freezing condition.")
    if checked_weather and set_reminder and mentions_cold:
        return _partial(
            "Set reminder instead of sending email — close but wrong notification channel."
        )
    if checked_weather and mentions_cold:
        return _partial("Detected freezing conditions but didn't send the warning email.")
    if checked_weather:
        return _partial("Checked weather but didn't act on the below-freezing result.")
    return _fail("Did not set up the weather alert workflow.")


SCENARIO = ScenarioDefinition(
    id="TC-56",
    title="Notification Workflow",
    category=Category.N,
    user_message="Check the weather in NYC. If it's below freezing, email me at user@company.com with a warning and set a reminder to dress warmly tomorrow morning.",
    description="Compose weather check → conditional → email notification.",
    handle_tool_call=_tc56_handle,
    evaluate=_tc56_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it checks NYC weather → detects freezing → sends warning email.",
    "Fail if it doesn't compose weather check with notification.",
)
