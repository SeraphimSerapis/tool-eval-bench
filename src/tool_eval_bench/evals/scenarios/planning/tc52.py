"""TC-52 — Open-Ended Research."""

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
    _result_matches_if_present,
)


def _tc52_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        ticker = _as_str(call.arguments.get("ticker", "")).upper()
        if ticker == "AAPL":
            return _noise(
                {"ticker": "AAPL", "price": 178.50, "change": -2.3, "change_percent": -1.27},
                "get_stock_price",
            )
        return _noise({"error": f"Unknown ticker: {ticker}"}, "get_stock_price")
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", "")).lower()
        if (
            "market" in query
            or "s&p" in query
            or "index" in query
            or "nasdaq" in query
            or "benchmark" in query
        ):
            return _noise(
                {
                    "results": [
                        {
                            "snippet": "S&P 500 closed at 5,412.50, up 0.8% for the week. "
                            "NASDAQ composite at 17,234.12, up 1.2%."
                        },
                    ]
                },
                "web_search",
            )
        if "aapl" in query or "apple" in query:
            return _noise(
                {"results": [{"snippet": "Apple Inc (AAPL) reports Q1 revenue of $94.3B."}]},
                "web_search",
            )
        return _noise(
            {"results": [{"snippet": f"Search results for: {query}"}]},
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


def _tc52_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'I need to prepare a summary comparing our stock performance
    against the market. Our ticker is AAPL.'

    Model must research market data + get stock price + synthesize.
    Not told which tools to chain or in what order.
    """

    def stock_result_is_aapl(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and _as_str(payload.get("ticker")).upper() == "AAPL"
            and payload.get("price") == 178.50
        )

    def market_result_has_benchmark(payload: Any) -> bool:
        return "5,412.50" in str(payload) and "17,234.12" in str(payload)

    stock_calls = [
        call
        for call in _tool_calls_by_name(state, "get_stock_price")
        if _as_str(call.arguments.get("ticker", "")).upper() == "AAPL"
        and _result_matches_if_present(state, call, stock_result_is_aapl)
    ]
    market_calls = [
        call
        for call in _tool_calls_by_name(state, "web_search")
        if any(
            w in _as_str(call.arguments.get("query", "")).lower()
            for w in ("market", "s&p", "index", "nasdaq", "benchmark")
        )
        and _result_matches_if_present(state, call, market_result_has_benchmark)
    ]
    got_stock = bool(stock_calls)
    searched_market = bool(market_calls)
    answer = state.final_answer.lower()
    # Require actual data (AAPL price 178) plus comparison context.
    has_price = _answer_affirms_number(state.final_answer, "178")
    has_benchmark = any(
        _answer_affirms_number(state.final_answer, value)
        for value in ("5412", "5,412", "17234", "17,234")
    )
    mentions_comparison = (
        has_price
        and has_benchmark
        and any(
            w in answer
            for w in (
                "compare",
                "comparison",
                "versus",
                "vs",
                "against",
                "relative",
                "underperformed",
                "outperformed",
            )
        )
    )

    if got_stock and searched_market and mentions_comparison:
        return _pass("Autonomously researched market + stock and synthesized comparison.")
    if got_stock and searched_market:
        return _partial("Got both data sources but didn't synthesize a comparison.")
    if got_stock and not searched_market:
        return _partial("Got stock price but didn't research market benchmark for comparison.")
    if searched_market and not got_stock:
        return _partial("Researched market but didn't get specific AAPL stock price.")
    return _fail("Did not autonomously plan a research workflow.")


SCENARIO = ScenarioDefinition(
    id="TC-52",
    title="Open-Ended Research",
    category=Category.M,
    user_message="I need to prepare a summary comparing our stock performance against the market. Our ticker is AAPL.",
    description="Autonomously research market data + stock price and synthesize comparison.",
    handle_tool_call=_tc52_handle,
    evaluate=_tc52_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it gets AAPL stock price AND researches market benchmark, then synthesizes.",
    "Fail if it doesn't autonomously plan the research workflow.",
)
