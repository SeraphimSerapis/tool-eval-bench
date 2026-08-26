"""TC-78 — Independent Portfolio Valuation."""

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
    answer_affirms_number,
    as_str,
    result_is_usable_if_present,
    tool_calls_by_name,
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
from tool_eval_bench.evals.scenarios.hardmode_expanded._shared import (
    _PRICES,
    _STRING,
    _result_matches_if_present,
    _tool,
)

_TC78_TOOLS = [
    _tool(
        "get_stock_price",
        "Get the current stock price for a ticker.",
        {"ticker": _STRING},
        ["ticker"],
    ),
    _tool("calculator", "Perform arithmetic.", {"expression": _STRING}, ["expression"]),
]


def _tc78_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        ticker = as_str(call.arguments.get("ticker")).upper()
        price = _PRICES.get(ticker)
        if price is not None:
            return _noise({"ticker": ticker, "price_usd": price}, call.name)
    if call.name == "calculator":
        return _noise({"result": 935.0}, call.name)
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc78_eval(state: ScenarioState) -> ScenarioEvaluation:
    calls = tool_calls_by_name(state, "get_stock_price")
    tickers = {as_str(call.arguments.get("ticker")).strip().upper() for call in calls}
    expected_prices = _PRICES
    complete = all(
        any(
            as_str(call.arguments.get("ticker")).strip().upper() == ticker
            and result_is_usable_if_present(state, call)
            and _result_matches_if_present(
                state,
                call,
                lambda payload, expected_ticker=ticker, expected_price=price: (
                    payload.get("ticker") == expected_ticker
                    and payload.get("price_usd") == expected_price
                ),
            )
            for call in calls
        )
        for ticker, price in expected_prices.items()
    )
    total_ok = answer_affirms_number(state.final_answer, "935")
    wrong_lookup = any(ticker not in _PRICES for ticker in tickers)
    if complete and total_ok and not wrong_lookup:
        return _pass("Fetched all three independent prices and reported the $935 portfolio value.")
    if complete:
        return _partial("Fetched all required prices but did not report the clean correct total.")
    return _fail("Omitted a required ticker or invented the portfolio value.")


SCENARIO = ScenarioDefinition(
    "TC-78",
    "Independent Portfolio Valuation",
    Category.P,
    "Using current prices, calculate the value of 3 ACME shares, 2 BETA shares, and 5 CYGN shares.",
    "Fetch three independent prices and calculate the portfolio total.",
    _tc78_handle,
    _tc78_eval,
    tools_override=_TC78_TOOLS,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it fetches all prices and totals $935.", "Fail if a ticker is omitted."
)
