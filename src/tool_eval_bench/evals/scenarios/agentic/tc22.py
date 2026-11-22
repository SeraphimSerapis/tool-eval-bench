"""TC-22 — Output Format Compliance."""

from __future__ import annotations

import json
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
from tool_eval_bench.evals.helpers import counted_calls as _counted_calls
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
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


def _tc22_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "Berlin",
                "temperature": 7,
                "units": "celsius",
                "condition": "Overcast",
                "humidity": 82,
            },
            "get_weather",
        )
    return _generic_tool_fallback(call)


_TC22_TEMP_STR = re.compile(r"7(?:\.0+)?(?:\s*°?\s*c(?:elsius)?)?", re.IGNORECASE)
_TC22_HUMIDITY_STR = re.compile(r"82(?:\.0+)?(?:\s*%)?")


def _tc22_values_match(parsed: dict[str, Any]) -> bool:
    """Accept raw numbers or unit-qualified strings matching 7°C, Overcast, 82%."""
    temp = parsed.get("temp")
    cond = parsed.get("condition")
    hum = parsed.get("humidity")
    temp_ok = (type(temp) in (int, float) and temp == 7) or (
        isinstance(temp, str) and bool(_TC22_TEMP_STR.fullmatch(temp.strip()))
    )
    cond_ok = isinstance(cond, str) and cond.strip().lower() == "overcast"
    hum_ok = (type(hum) in (int, float) and hum == 82) or (
        isinstance(hum, str) and bool(_TC22_HUMIDITY_STR.fullmatch(hum.strip()))
    )
    return temp_ok and cond_ok and hum_ok


@forbid_unrequested_side_effects()
def _tc22_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asked: 'Get the weather in Berlin. Respond with ONLY valid JSON, keys: temp, condition, humidity. No other text.'"""
    weather_calls = [c for c in state.tool_calls if c.name == "get_weather"]
    if not weather_calls:
        return _fail("Did not call get_weather.")

    berlin_calls = [
        call
        for call in weather_calls
        if _positive_argument_contains(call.arguments.get("location"), "berlin")
    ]
    if not berlin_calls:
        return _fail("Called get_weather for the wrong location.")

    weather_call = berlin_calls[-1]
    if not _result_is_usable_if_present(state, weather_call):
        return _partial("get_weather returned an error, so the JSON cannot claim weather data.")

    recorded_results = _matching_tool_results(state, weather_call)
    if recorded_results and not any(
        isinstance(result.result, dict)
        and result.result.get("temperature") == 7
        and result.result.get("condition") == "Overcast"
        and result.result.get("humidity") == 82
        for result in recorded_results
    ):
        return _partial("The JSON values were not grounded in the recorded weather result.")

    answer = state.final_answer.strip()
    # Try to parse the response as JSON
    try:
        parsed = json.loads(answer)
        if not isinstance(parsed, dict):
            return _partial("Returned JSON, but not the required object.")
        has_keys = all(k in parsed for k in ("temp", "condition", "humidity"))
        if has_keys:
            no_extra = set(parsed) == {"temp", "condition", "humidity"}
            correct_values = _tc22_values_match(parsed)
            if correct_values and no_extra:
                if len(_counted_calls(state.tool_calls)) != 1:
                    return _partial(
                        "Returned correct JSON, but made unnecessary additional tool calls."
                    )
                return _pass("Called get_weather and returned properly formatted JSON.")
            if correct_values and not no_extra:
                return _partial(
                    "Returned JSON with extra keys.",
                    f"Got keys: {list(parsed.keys())}",
                )
            return _partial(
                "Returned JSON with correct keys but wrong values.",
                f"Got temp={parsed.get('temp')!r}, condition={parsed.get('condition')!r}, "
                f"humidity={parsed.get('humidity')!r}.",
            )
        return _partial("Returned JSON but with wrong keys.", f"Got keys: {list(parsed.keys())}")
    except json.JSONDecodeError:
        pass

    # Maybe it's wrapped in markdown code fences — acceptable since chat models
    # commonly wrap structured output in code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict) and all(
                k in parsed for k in ("temp", "condition", "humidity")
            ):
                no_extra = set(parsed) == {"temp", "condition", "humidity"}
                correct_values = _tc22_values_match(parsed)
                if correct_values and no_extra:
                    return _partial(
                        "Returned correct JSON but violated the raw-JSON output contract."
                    )
                return _partial(
                    "Returned code-fenced JSON with correct keys but wrong values.",
                    f"Got temp={parsed.get('temp')!r}, condition={parsed.get('condition')!r}, "
                    f"humidity={parsed.get('humidity')!r}.",
                )
        except json.JSONDecodeError:
            pass

    return _fail("Did not return valid JSON as requested.", f"Got: {answer[:100]}")


SCENARIO = ScenarioDefinition(
    id="TC-22",
    title="Output Format Compliance",
    category=Category.H,
    user_message="Get the weather in Berlin. Respond with ONLY valid JSON, keys: temp, condition, humidity. No other text.",
    description="Must return raw JSON, not prose or markdown.",
    handle_tool_call=_tc22_handle,
    evaluate=_tc22_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it returns raw valid JSON with temp/condition/humidity keys.",
    "Fail if it adds prose, wraps in markdown, or uses wrong keys.",
)
