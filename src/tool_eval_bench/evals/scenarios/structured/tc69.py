"""TC-69 — Multi-Tool → Complex Schema."""

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
from tool_eval_bench.evals.helpers import (
    as_str,
    generic_tool_fallback,
    normalize,
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

_TC69_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "daily_briefing",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "weather": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "temperature": {"type": "number"},
                        "condition": {"type": "string"},
                    },
                    "required": ["location", "temperature", "condition"],
                    "additionalProperties": False,
                },
                "market": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "price": {"type": "number"},
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "flat"],
                        },
                    },
                    "required": ["ticker", "price", "direction"],
                    "additionalProperties": False,
                },
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["date", "weather", "market", "action_items"],
            "additionalProperties": False,
        },
    },
}


def _tc69_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {
                "location": "San Francisco",
                "temperature": 18,
                "units": "celsius",
                "condition": "Foggy",
                "humidity": 85,
            },
            "get_weather",
        )
    if call.name == "get_stock_price":
        return _noise(
            {
                "ticker": "AAPL",
                "price": 192.30,
                "currency": "USD",
                "change": "-2.15",
                "change_percent": "-1.11%",
            },
            "get_stock_price",
        )
    return generic_tool_fallback(call)


def _tc69_eval(state: ScenarioState) -> ScenarioEvaluation:
    def weather_result_is_san_francisco(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("location") == "San Francisco"
            and payload.get("temperature") == 18
            and payload.get("condition") == "Foggy"
        )

    def stock_result_is_aapl(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("ticker") == "AAPL"
            and payload.get("price") == 192.30
            and "-1.11" in str(payload)
        )

    weather_calls = [
        call
        for call in state.tool_calls
        if call.name == "get_weather"
        and normalize(as_str(call.arguments.get("location"))) == "san francisco"
        and _result_matches_if_present(state, call, weather_result_is_san_francisco)
    ]
    stock_calls = [
        call
        for call in state.tool_calls
        if call.name == "get_stock_price"
        and normalize(as_str(call.arguments.get("ticker"))) == "aapl"
        and _result_matches_if_present(state, call, stock_result_is_aapl)
    ]
    weather_call = weather_calls[-1] if weather_calls else None
    stock_call = stock_calls[-1] if stock_calls else None
    weather = bool(weather_calls)
    stock = bool(stock_calls)

    if not weather or not stock:
        missing = []
        if not weather:
            missing.append("get_weather")
        if not stock:
            missing.append("get_stock_price")
        if any(call.name == "get_weather" for call in state.tool_calls) and any(
            call.name == "get_stock_price" for call in state.tool_calls
        ):
            return _partial("Called a briefing tool, but its required arguments were invalid.")
        return _fail(f"Did not call required tools: {', '.join(missing)}.")
    if weather_call is None or stock_call is None:
        return _fail("Required briefing call records were unavailable.")
    if any(call.name not in {"get_weather", "get_stock_price"} for call in state.tool_calls):
        return _partial("Called an unrelated tool during the structured briefing.")

    answer = _extract_json_answer(state.final_answer)

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called both tools but output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Output is not a JSON object.")

    # Check top-level
    required_top = {"date", "weather", "market", "action_items"}
    missing_top = required_top - set(data.keys())
    if missing_top:
        return _partial(f"Missing top-level fields: {', '.join(sorted(missing_top))}.")

    # Validate weather sub-object
    w = data.get("weather", {})
    if not isinstance(w, dict) or not all(k in w for k in ("location", "temperature", "condition")):
        return _partial("Weather sub-object is missing required fields.")

    # Validate market sub-object
    m = data.get("market", {})
    if not isinstance(m, dict) or not all(k in m for k in ("ticker", "price", "direction")):
        return _partial("Market sub-object is missing required fields.")

    valid_directions = {"up", "down", "flat"}
    direction_val = m.get("direction")
    if not isinstance(direction_val, str) or direction_val not in valid_directions:
        return _partial(f"Market direction '{direction_val}' is not a valid enum value.")

    if (
        not isinstance(data.get("date"), str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["date"])
        or not isinstance(w.get("location"), str)
        or not isinstance(w.get("temperature"), (int, float))
        or isinstance(w.get("temperature"), bool)
        or not isinstance(w.get("condition"), str)
        or not isinstance(m.get("ticker"), str)
        or not isinstance(m.get("price"), (int, float))
        or isinstance(m.get("price"), bool)
        or set(w) != {"location", "temperature", "condition"}
        or set(m) != {"ticker", "price", "direction"}
        or set(data) != required_top
    ):
        return _partial("Nested briefing output contains schema-invalid fields.")

    # Validate action_items is array of strings
    actions = data.get("action_items", [])
    if (
        not isinstance(actions, list)
        or not actions
        or not all(isinstance(a, str) and a.strip() for a in actions)
    ):
        return _partial("action_items must be a non-empty array of actionable strings.")

    # Verify data integrity from tools
    if w.get("temperature") != 18:
        return _partial("Weather temperature doesn't match tool result (18°C).")
    if w.get("location") != "San Francisco" or w.get("condition") != "Foggy":
        return _partial("Weather fields don't match the San Francisco tool result.")

    if m.get("price") != 192.30:
        return _partial("Stock price doesn't match tool result (192.30).")
    if m.get("ticker") != "AAPL":
        return _partial("Market ticker doesn't match the AAPL tool result.")

    # Direction should be "down" since change is negative
    if m.get("direction") != "down":
        return _partial("Market direction should be 'down' (stock dropped -1.11%).")

    return _pass(
        "Called both tools and produced schema-compliant nested JSON with correct data synthesis."
    )


SCENARIO = ScenarioDefinition(
    id="TC-69",
    title="Multi-Tool → Complex Schema",
    category=Category.O,
    user_message=(
        "Create my daily briefing: check the weather in San Francisco "
        "and look up AAPL stock price. Output as JSON matching this schema "
        "with actionable items.\n\n"
        f"Schema:\n```json\n{_schema_text(_TC69_SCHEMA)}\n```"
    ),
    description="Call multiple tools and synthesize into complex nested schema.",
    handle_tool_call=_tc69_handle,
    evaluate=_tc69_eval,
    response_format_override=_TC69_SCHEMA,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls both tools and produces complex nested JSON with correct data synthesis.",
    "Fail if it misses a tool or produces invalid nested structure.",
)
