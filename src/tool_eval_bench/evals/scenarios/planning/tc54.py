"""TC-54 — Cross-Tool Synthesis."""

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
    parse_math_expression as _parse_math_expression,
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
    _result_matches_if_present,
)


def _tc54_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        ticker = _as_str(call.arguments.get("ticker", "")).upper()
        if ticker == "MSFT":
            return _noise(
                {"ticker": "MSFT", "price": 425.80, "currency": "USD"},
                "get_stock_price",
            )
        return _noise({"error": f"Unknown ticker: {ticker}"}, "get_stock_price")
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if "usd" in query and ("jpy" in query or "yen" in query):
            return _noise(
                {"results": [{"snippet": "Current exchange rate: 1 USD = 149.50 JPY."}]},
                "web_search",
            )
        if "exchange" in query or "currency" in query or "yen" in query:
            return _noise(
                {"results": [{"snippet": "USD/JPY exchange rate: 149.50. Japanese Yen."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": f"Results for: {query}"}]},
            "web_search",
        )
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc54_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What's the local currency equivalent of MSFT's stock price
    in Tokyo right now?'

    Must combine: get_stock_price(MSFT) + web_search(USD/JPY rate) + calculator.
    No single tool solves this. Expected answer: ~63,627 JPY.
    """

    def stock_result_is_msft(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and _as_str(payload.get("ticker")).upper() == "MSFT"
            and payload.get("price") == 425.80
        )

    def exchange_result_is_usable(payload: Any) -> bool:
        return "149.50" in str(payload)

    def calculator_result_is_expected(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        value = payload.get("result")
        if value is None:
            return False
        try:
            return abs(float(value) - 63657.1) < 0.01
        except (TypeError, ValueError):
            return "63657" in str(value).replace(",", "")

    stock_calls = [
        call
        for call in _tool_calls_by_name(state, "get_stock_price")
        if _as_str(call.arguments.get("ticker", "")).upper() == "MSFT"
        and _result_matches_if_present(state, call, stock_result_is_msft)
    ]
    exchange_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if any(
            w in _as_str(call.arguments.get("query", "")).lower()
            for w in ("usd", "jpy", "yen", "exchange", "currency")
        )
        and _result_matches_if_present(state, call, exchange_result_is_usable)
    ]
    got_stock = bool(stock_calls)
    searched_exchange = bool(exchange_calls)

    answer = state.final_answer
    # Expected: 425.80 * 149.50 ≈ 63,657 JPY. Accept nearby rounded values
    # without allowing any arbitrary "63" substring to count as the result.
    has_reasonable = any(
        _answer_affirms_number(answer, str(value)) for value in range(63600, 63700)
    )

    calculator_calls = [
        call
        for call in _tool_calls_by_name(state, "calculator")
        if bool(
            (expression := _as_str(call.arguments.get("expression")).replace(",", ""))
            and "425.8" in expression
            and "149.5" in expression
            and "*" in expression
            and _parse_math_expression(expression) is not None
        )
        and _result_matches_if_present(state, call, calculator_result_is_expected)
    ]
    calculator = bool(calculator_calls)
    data_available_before_calculation = bool(
        stock_calls
        and exchange_calls
        and calculator_calls
        and max(_call_index(state, stock_calls[0]), _call_index(state, exchange_calls[0]))
        < _call_index(state, calculator_calls[0])
    )
    if got_stock and searched_exchange and calculator and not data_available_before_calculation:
        return _partial("Calculated before both source lookups completed.")
    if got_stock and searched_exchange and calculator and has_reasonable:
        if _has_unexpected_tools(state, {"get_stock_price", "web_search", "calculator"}):
            return _partial("Solved the conversion but also called an unrelated tool.")
        return _pass("Combined stock price + exchange rate + calculation — creative composition.")
    if got_stock and searched_exchange:
        if not _tool_calls_by_name(state, "calculator"):
            return _partial(
                "Got both data sources but did not call calculator to verify the exact conversion."
            )
        if not calculator:
            return _partial(
                "Called calculator but did not verify the required 425.8 * 149.5 USD/JPY conversion."
            )
        return _partial(
            "Called calculator and verified the conversion, but the final answer does not match the computed USD/JPY conversion."
        )
    if got_stock and not searched_exchange:
        return _partial("Got stock price but didn't look up the exchange rate.")
    if searched_exchange and not got_stock:
        return _partial("Searched exchange rate but didn't get the stock price.")
    return _fail("Did not combine tools to solve the cross-domain problem.")


SCENARIO = ScenarioDefinition(
    id="TC-54",
    title="Cross-Tool Synthesis",
    category=Category.N,
    user_message="What's the local currency equivalent of MSFT's stock price in Tokyo right now?",
    description="Combine stock price + exchange rate lookup + calculation.",
    handle_tool_call=_tc54_handle,
    evaluate=_tc54_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it combines stock price + exchange rate to calculate JPY equivalent.",
    "Fail if it doesn't creatively combine multiple tools.",
)
