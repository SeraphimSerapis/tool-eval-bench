"""TC-67 — Enum Constraint + Analysis."""

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

_TC67_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "stock_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"type": "string"},
                "signal": {
                    "type": "string",
                    "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["ticker", "price", "currency", "signal", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _tc67_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return _noise(
            {
                "ticker": "NVDA",
                "price": 892.50,
                "currency": "USD",
                "change": "+15.30",
                "change_percent": "+1.74%",
                "volume": "42.3M",
            },
            "get_stock_price",
        )
    if call.name == "web_search":
        return _noise(
            {
                "results": [
                    {
                        "snippet": "NVIDIA (NVDA) reported record Q4 revenue of $22.1B, "
                        "up 265% year-over-year, driven by data center AI demand. "
                        "Analysts maintain buy ratings with average price target of $950.",
                    }
                ],
            },
            "web_search",
        )
    return generic_tool_fallback(call)


def _tc67_eval(state: ScenarioState) -> ScenarioEvaluation:
    def stock_result_is_nvda(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and normalize(as_str(payload.get("ticker"))) == "nvda"
            and payload.get("price") == 892.50
            and payload.get("currency") == "USD"
        )

    stock_calls = [
        call
        for call in state.tool_calls
        if call.name == "get_stock_price"
        and normalize(as_str(call.arguments.get("ticker"))) == "nvda"
        and _result_matches_if_present(state, call, stock_result_is_nvda)
    ]
    if not stock_calls:
        if any(call.name == "get_stock_price" for call in state.tool_calls):
            return _partial("Called get_stock_price, but looked up the wrong ticker or result.")
        return _fail("Did not call get_stock_price.")
    news = next(
        (
            call
            for call in state.tool_calls
            if call.name == "web_search"
            and any(
                word in as_str(call.arguments.get("query")).lower()
                for word in ("news", "nvidia", "nvda")
            )
            and _result_matches_if_present(
                state,
                call,
                lambda payload: (
                    "nvidia" in str(payload).lower() and "22.1b" in str(payload).lower()
                ),
            )
        ),
        None,
    )
    if news is None:
        return _partial("Produced a stock analysis without the required recent-news lookup.")
    if any(call.name not in {"get_stock_price", "web_search"} for call in state.tool_calls):
        return _partial("Called an unrelated tool during a structured stock analysis.")

    answer = _extract_json_answer(state.final_answer)

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called get_stock_price but output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Output is not a JSON object.")

    required = {"ticker", "price", "currency", "signal", "reasoning"}
    missing = required - set(data.keys())
    if missing:
        return _partial(f"Valid JSON but missing: {', '.join(sorted(missing))}.")

    valid_signals = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    signal_val = data.get("signal")
    if not isinstance(signal_val, str) or signal_val not in valid_signals:
        return _partial(f"Signal '{signal_val}' is not a valid enum value.")

    # Verify data comes from tool
    if data.get("price") != 892.50:
        return _partial("Price doesn't match tool result (892.50).")

    if normalize(as_str(data.get("ticker"))) != "nvda":
        return _partial("Ticker doesn't match tool result.")

    if not isinstance(data.get("reasoning"), str) or len(data["reasoning"]) < 20:
        return _partial("Reasoning field is too short — should explain the signal.")

    if data.get("currency") != "USD" or set(data) != required:
        return _partial("Analysis contains invalid currency or extra schema fields.")
    return _pass("Produced schema-compliant analysis with correct enum signal and tool data.")


SCENARIO = ScenarioDefinition(
    id="TC-67",
    title="Enum Constraint + Analysis",
    category=Category.O,
    user_message=(
        "Get the current stock price for NVDA and produce a stock analysis "
        "as JSON matching this schema. Research recent news "
        "to inform your signal.\n\n"
        f"Schema:\n```json\n{_schema_text(_TC67_SCHEMA)}\n```"
    ),
    description="Call tools and produce JSON with enum-constrained signal field.",
    handle_tool_call=_tc67_handle,
    evaluate=_tc67_eval,
    response_format_override=_TC67_SCHEMA,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it gets the stock price, researches news, and produces JSON with a valid enum signal.",
    "Fail if it skips tools or uses an invalid signal enum value.",
)
