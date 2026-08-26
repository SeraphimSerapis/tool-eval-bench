"""TC-09 — Parallel Independence."""

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
    first_call as _first_call,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    has_tool_call as _has_tool_call,
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
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc09_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise(
            {"location": "London", "temperature": 12, "condition": "Cloudy"}, "get_weather"
        )
    if call.name == "get_stock_price":
        return _noise({"ticker": "MSFT", "price": 412.78, "currency": "USD"}, "get_stock_price")
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "London is cloudy at 12C and MSFT is around $412.78."}]},
            "web_search",
        )
    return _generic_tool_fallback(call)


def _tc09_weather_result_is_london(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    return (
        _positive_argument_contains(payload.get("location"), "london")
        and _numeric_value(payload.get("temperature")) == 12
    )


def _tc09_stock_result_is_msft(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    price = _numeric_value(payload.get("price"))
    return (
        _normalize(_as_str(payload.get("ticker"))) == "msft"
        and price is not None
        and abs(price - 412.78) < 1e-9
    )


def _tc09_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = _has_tool_call(
        state,
        "get_weather",
        lambda c: _positive_argument_contains(c.arguments.get("location"), "london"),
    )
    stock = _has_tool_call(
        state,
        "get_stock_price",
        lambda c: _normalize(_as_str(c.arguments.get("ticker"))) == "msft",
    )
    first_batch = [c for c in state.tool_calls if c.turn == 1]
    parallel = any(c.name == "get_weather" for c in first_batch) and any(
        c.name == "get_stock_price" for c in first_batch
    )
    if weather and stock:
        weather_call = _first_call(state, "get_weather")
        stock_call = _first_call(state, "get_stock_price")
        if weather_call and not _result_matches_if_present(
            state, weather_call, _tc09_weather_result_is_london
        ):
            return _partial("The weather result was unusable, so its value cannot be confirmed.")
        if stock_call and not _result_matches_if_present(
            state, stock_call, _tc09_stock_result_is_msft
        ):
            return _partial("The stock result was unusable, so its value cannot be confirmed.")
        # Verify the model actually surfaced the tool result values.
        # Use digit-boundary match for temperature (12) to avoid false
        # positives from the stock price (412.78) which contains "12".
        # (?<!\d)12(?!\d) matches "12C", "12°C", "12 degrees" but not "412".
        has_temp = _answer_affirms_number(state.final_answer, "12")
        has_price = _answer_affirms_number(state.final_answer, "412")
        if has_temp and has_price:
            if len(state.tool_calls) != 2:
                return _partial("Handled both requested tools but added an unnecessary tool call.")
            note = "Both tools were called in the same assistant turn." if parallel else None
            return _pass("Handled both independent tasks.", note)
        return _partial(
            "Called the right tools but did not surface the actual results in the answer.",
            "Answer should include the temperature (12) and stock price (412.78).",
        )
    if _has_tool_call(state, "web_search"):
        return _partial("Covered the request, but fell back to web_search.")
    return _fail("Missed one side of the two-part request.")


SCENARIO = ScenarioDefinition(
    id="TC-09",
    title="Parallel Independence",
    category=Category.C,
    user_message="What's the weather in London and the stock price of MSFT?",
    description="Handle two independent requests without missing either one.",
    handle_tool_call=_tc09_handle,
    evaluate=_tc09_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it handles both London weather and MSFT price in the same response flow.",
    "Fail if it misses one side of the request.",
)
