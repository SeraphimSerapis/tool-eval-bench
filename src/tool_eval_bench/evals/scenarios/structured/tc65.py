"""TC-65 — Tool → Structured Output."""

from __future__ import annotations

import json
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
    generic_tool_fallback,
    includes_text,
    normalize,
    positive_argument_contains,
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
from tool_eval_bench.evals.scenarios.structured._shared import (
    _extract_json_answer,
    _result_matches_if_present,
    _schema_text,
)

_TC65_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature_celsius": {"type": "number"},
                "condition": {"type": "string"},
                "recommendation": {"type": "string"},
            },
            "required": ["location", "temperature_celsius", "condition", "recommendation"],
            "additionalProperties": False,
        },
    },
}


def _tc65_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        if not positive_argument_contains(call.arguments.get("location"), "tokyo"):
            return _noise(
                {"error": "Weather report only available for Tokyo in this scenario."}, call.name
            )
        return _noise(
            {
                "location": "Tokyo",
                "temperature": 28,
                "units": "celsius",
                "condition": "Sunny",
                "humidity": 65,
            },
            "get_weather",
        )
    return generic_tool_fallback(call)


def _tc65_eval(state: ScenarioState) -> ScenarioEvaluation:
    def weather_result_is_tokyo(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and normalize(as_str(payload.get("location"))) == "tokyo"
            and payload.get("temperature") == 28
            and payload.get("condition") == "Sunny"
        )

    weather_calls = [
        call
        for call in state.tool_calls
        if call.name == "get_weather"
        and positive_argument_contains(call.arguments.get("location"), "tokyo")
        and _result_matches_if_present(state, call, weather_result_is_tokyo)
    ]
    if not weather_calls:
        if any(call.name == "get_weather" for call in state.tool_calls):
            return _partial("Called get_weather, but for the wrong requested location or result.")
        return _fail("Did not call get_weather before producing structured output.")
    if any(call.name != "get_weather" for call in state.tool_calls):
        return _partial("Called an unrelated tool during a structured weather request.")

    answer = _extract_json_answer(state.final_answer)

    # Extract JSON from fences if present
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called get_weather correctly but final output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Called get_weather but output is not a JSON object.")

    required = {"location", "temperature_celsius", "condition", "recommendation"}
    missing = required - set(data.keys())
    if missing:
        return _partial(f"Valid JSON but missing: {', '.join(sorted(missing))}.")

    # Verify the data comes from the tool result, not hallucinated
    if data.get("temperature_celsius") != 28:
        return _partial("Schema-compliant but temperature doesn't match tool result (28°C).")

    if not isinstance(data.get("location"), str) or not data["location"].strip():
        return _partial("Structured weather output has an invalid location field.")
    if not includes_text(data.get("location", ""), "tokyo"):
        return _partial("Schema-compliant but location doesn't match tool result.")
    if data.get("condition") != "Sunny":
        return _partial("Schema-compliant but condition doesn't match tool result.")

    if (
        type(data.get("temperature_celsius")) not in (int, float)
        or not isinstance(data.get("condition"), str)
        or not isinstance(data.get("recommendation"), str)
        or not data["recommendation"]
        or set(data) != required
    ):
        return _partial("Structured weather output contains schema-invalid fields.")
    return _pass("Called get_weather, then produced schema-compliant JSON with correct data.")


SCENARIO = ScenarioDefinition(
    id="TC-65",
    title="Tool → Structured Output",
    category=Category.O,
    user_message=(
        "Get the current weather in Tokyo and output it as JSON "
        "matching this schema. Include a recommendation for what to wear.\n\n"
        f"Schema:\n```json\n{_schema_text(_TC65_SCHEMA)}\n```"
    ),
    description="Call get_weather, then format the result as schema-compliant JSON.",
    handle_tool_call=_tc65_handle,
    evaluate=_tc65_eval,
    response_format_override=_TC65_SCHEMA,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_weather first, then outputs schema-compliant JSON with correct data.",
    "Fail if it skips the tool call or produces non-JSON output.",
)
