"""TC-25 — Cross-Reference Prior Results."""

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
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
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
    positive_argument_contains as _positive_argument_contains,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc25_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "Berlin",
                "temperature": 5,
                "units": "celsius",
                "condition": "Clear",
                "humidity": 45,
            },
            "get_weather",
        )
    if call.name == "set_reminder":
        return _noise({"status": "created", "reminder_id": "rem_552"}, "set_reminder")
    return _generic_tool_fallback(call)


def _tc25_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Two-part request: 'What's the weather in Berlin? Also, if it's below 10°C, set a reminder to bring a coat.'

    The model should call get_weather ONCE, see 5°C < 10°C, then set the reminder.
    It should NOT call get_weather twice.
    """
    weather_calls = [c for c in state.tool_calls if c.name == "get_weather"]
    reminder_calls = [c for c in state.tool_calls if c.name == "set_reminder"]

    if len(weather_calls) == 0:
        return _fail("Did not check the weather.")

    if not any(
        _positive_argument_contains(call.arguments.get("location"), "berlin")
        for call in weather_calls
    ):
        return _fail("Checked the wrong location for the conditional weather request.")

    # Check if the weather tool actually returned data or was error-injected
    weather_results = _matching_tool_results(state, weather_calls[-1])
    weather_got_error = not _result_is_usable_if_present(state, weather_calls[-1])

    if weather_got_error:
        # The model never received the temperature — can't evaluate the conditional.
        # Credit retry attempts and graceful behavior.
        if len(weather_calls) > 1:
            return _partial("Weather tool failed — retried, good recovery attempt.")
        return _partial(
            "Weather tool returned an error — couldn't evaluate the temperature condition."
        )

    if len(weather_calls) > 1:
        return _partial("Called get_weather multiple times instead of reusing the first result.")

    recorded_temperatures: list[float] = []
    for result in weather_results:
        if isinstance(result.result, dict):
            temperature = result.result.get("temperature")
            if isinstance(temperature, (int, float)):
                recorded_temperatures.append(temperature)
    if recorded_temperatures and not any(temperature < 10 for temperature in recorded_temperatures):
        if reminder_calls:
            return _fail(
                "Set a coat reminder even though the recorded temperature was not below 10°C."
            )
        return _pass("Correctly skipped the reminder because the temperature was not below 10°C.")

    if len(reminder_calls) == 0:
        return _fail("Checked weather (5°C) but did not set the coat reminder.")

    if any(call.turn <= weather_calls[0].turn for call in reminder_calls):
        return _fail("Set the reminder before receiving the weather result.")

    coat_mentioned = any(
        "coat" in _as_str(c.arguments.get("message")).lower() for c in reminder_calls
    )
    if coat_mentioned:
        coat_call = next(
            call
            for call in reminder_calls
            if "coat" in _as_str(call.arguments.get("message")).lower()
        )
        if not _result_is_usable_if_present(state, coat_call):
            return _partial("The coat reminder call returned an explicit error.")
        if len(state.tool_calls) != 2:
            return _partial("Completed the conditional request with unnecessary additional calls.")
        return _pass("Checked weather once, recognized 5°C < 10°C, set coat reminder.")
    return _partial("Set a reminder but didn't mention bringing a coat.")


SCENARIO = ScenarioDefinition(
    id="TC-25",
    title="Cross-Reference Prior Results",
    category=Category.I,
    user_message="What's the weather in Berlin? Also, if it's below 10°C, set a reminder to bring a coat tomorrow morning.",
    description="Should call get_weather once, then conditionally set_reminder.",
    handle_tool_call=_tc25_handle,
    evaluate=_tc25_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_weather once, sees 5°C < 10, sets coat reminder.",
    "Fail if it calls get_weather twice or skips the reminder.",
)
