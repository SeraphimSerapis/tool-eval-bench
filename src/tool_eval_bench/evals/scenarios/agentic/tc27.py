"""TC-27 — Deduplication Awareness."""

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
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
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
    positive_argument_contains as _positive_argument_contains,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc27_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        units = _normalize(_as_str(call.arguments.get("units", "celsius")))
        if units == "fahrenheit":
            return _noise(
                {
                    "location": "London",
                    "temperature": 50,
                    "units": "fahrenheit",
                    "condition": "Rainy",
                    "humidity": 78,
                },
                "get_weather",
            )
        return _noise(
            {
                "location": "London",
                "temperature": 10,
                "units": "celsius",
                "condition": "Rainy",
                "humidity": 78,
            },
            "get_weather",
        )
    return _generic_tool_fallback(call)


def _tc27_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Get the weather in London in Celsius, and also the weather in London in Fahrenheit.'

    Should make exactly 2 get_weather calls (different units), not 1 or 3+.
    """
    weather_calls = [c for c in state.tool_calls if c.name == "get_weather"]

    if len(weather_calls) == 2:
        if any(
            not _positive_argument_contains(c.arguments.get("location"), "london")
            for c in weather_calls
        ):
            return _fail("Requested weather for the wrong location.")
        units_used = [
            _normalize(_as_str(c.arguments.get("units", "celsius"))) for c in weather_calls
        ]
        has_both = "celsius" in units_used and "fahrenheit" in units_used
        if has_both:
            for call in weather_calls:
                if not _result_is_usable_if_present(state, call):
                    return _partial("A weather request returned an explicit error.")

            recorded_results = [
                result.result
                for call in weather_calls
                for result in _matching_tool_results(state, call)
                if isinstance(result.result, dict)
            ]
            if recorded_results:
                recorded_by_unit = {
                    _normalize(_as_str(result.get("units"))): result.get("temperature")
                    for result in recorded_results
                }
                if (
                    recorded_by_unit.get("celsius") != 10
                    or recorded_by_unit.get("fahrenheit") != 50
                ):
                    return _partial("The answer was not grounded in the recorded weather results.")
            # Verify the model actually surfaced the temperature values
            has_celsius = _answer_affirms_number(state.final_answer, "10")
            has_fahrenheit = _answer_affirms_number(state.final_answer, "50")
            if has_celsius and has_fahrenheit:
                return _pass("Made exactly 2 calls with different units.")
            return _partial(
                "Called get_weather correctly with both units but did not surface "
                "the actual temperatures in the answer.",
                "Answer should include 10 (Celsius) and 50 (Fahrenheit).",
            )
        return _partial("Made 2 calls but didn't distinguish units correctly.")

    if len(weather_calls) == 1:
        return _partial("Only made 1 call — should have made 2 with different units.")

    if len(weather_calls) == 0:
        return _fail("Did not call get_weather at all.")

    return _partial(
        f"Made {len(weather_calls)} calls — expected exactly 2.", "Possible deduplication issue"
    )


SCENARIO = ScenarioDefinition(
    id="TC-27",
    title="Deduplication Awareness",
    category=Category.I,
    user_message="Get the weather in London in Celsius, and also the weather in London in Fahrenheit.",
    description="Should make exactly 2 calls (different units), not 1 or 3+.",
    handle_tool_call=_tc27_handle,
    evaluate=_tc27_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it makes exactly 2 get_weather calls (Celsius + Fahrenheit).",
    "Fail if it makes 1 call or 3+ calls.",
)
