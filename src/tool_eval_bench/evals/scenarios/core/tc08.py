"""TC-08 — Conditional Branching."""

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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    asks_for_clarification as _asks_for_clarification,
)
from tool_eval_bench.evals.helpers import (
    datetime_matches as _datetime_matches,
)
from tool_eval_bench.evals.helpers import (
    days_after_reference as _days_after_reference,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
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
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc08_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {"location": "Paris", "temperature": 11, "condition": "Light rain", "humidity": 89},
            "get_weather",
        )
    if call.name == "set_reminder":
        return _noise({"reminder_id": "rem_553", "status": "set"}, "set_reminder")
    return _generic_tool_fallback(call)


def _tc08_weather_result_is_rainy(payload: Any) -> bool:
    """Return whether an explicit weather result supports the rainy branch."""
    if not isinstance(payload, dict) or not _includes_text(payload.get("condition"), "rain"):
        return False
    result_location = payload.get("location")
    return not result_location or _includes_text(result_location, "paris")


def _tc08_reminder_result_is_set(payload: Any) -> bool:
    """Return whether an explicit reminder result confirms creation."""
    if not isinstance(payload, dict) or "error" in payload:
        return False
    status = _normalize(_as_str(payload.get("status")))
    if not status:
        return bool(_as_str(payload.get("reminder_id")).strip())
    return status in {"accepted", "created", "ok", "scheduled", "set", "success"}


def _tc08_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather_calls = [
        call
        for call in _tool_calls_by_name(state, "get_weather")
        if _positive_argument_contains(call.arguments.get("location"), "paris")
    ]
    reminder_calls = [
        call
        for call in _tool_calls_by_name(state, "set_reminder")
        if (
            _positive_argument_contains(call.arguments.get("message"), "umbrella")
            # Use flexible datetime matching — accept any timezone representation
            and _datetime_matches(
                call.arguments.get("datetime"), _days_after_reference(state, 1), "08:00"
            )
        )
    ]
    rainy_weather_calls = [
        call
        for call in weather_calls
        if _result_matches_if_present(state, call, _tc08_weather_result_is_rainy)
    ]
    usable_reminder_calls = [
        call
        for call in reminder_calls
        if _result_matches_if_present(state, call, _tc08_reminder_result_is_set)
    ]
    ordered_pairs = [
        (weather, reminder)
        for weather in rainy_weather_calls
        for reminder in usable_reminder_calls
        if weather.turn < reminder.turn
    ]
    if ordered_pairs:
        if len(reminder_calls) != 1 or len(weather_calls) != 1 or len(state.tool_calls) != 2:
            return _partial(
                "Checked the weather but issued duplicate or additional reminder mutations."
            )
        return _pass("Checked the weather first, then set the rainy-day reminder.")

    if weather_calls and reminder_calls:
        # An explicit weather error means the condition is unknown, so retain
        # partial credit for the attempted chain without treating the reminder
        # as justified. A concrete non-rain result is a wrong branch instead.
        weather_result_is_error = any(
            not _result_is_usable_if_present(state, call) for call in weather_calls
        )
        if weather_result_is_error:
            return _partial(
                "The weather lookup returned an error, so the rainy branch could not be confirmed."
            )
        if not rainy_weather_calls:
            return _fail(
                "The weather result did not confirm rain, so the reminder was not justified."
            )

        reminder_result_is_error = any(
            not _result_is_usable_if_present(state, call) for call in reminder_calls
        )
        if reminder_result_is_error or not usable_reminder_calls:
            return _partial(
                "set_reminder did not return a usable result, so the reminder could not be confirmed."
            )
        return _fail("Did not respect the weather-first conditional flow.")

    if weather_calls and not reminder_calls and _asks_for_clarification(state.final_answer):
        return _partial("Read the weather correctly, but stopped short of setting the reminder.")
    return _fail("Did not respect the weather-first conditional flow.")


SCENARIO = ScenarioDefinition(
    id="TC-08",
    title="Conditional Branching",
    category=Category.C,
    user_message="Check the weather in Paris. If it's raining, remind me to bring an umbrella tomorrow at 8am.",
    description="Branch off the weather result instead of setting the reminder blindly.",
    handle_tool_call=_tc08_handle,
    evaluate=_tc08_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it checks Paris weather first and only then sets the umbrella reminder.",
    "Fail if it skips the weather lookup or ignores the condition.",
)
