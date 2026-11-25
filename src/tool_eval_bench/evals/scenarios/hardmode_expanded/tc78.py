"""TC-78 — Independent Portfolio Valuation."""

from __future__ import annotations

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
    answer_affirms_number,
    answer_affirms_pattern,
    as_str,
    generic_tool_fallback,
    result_is_usable_if_present,
    strip_markdown_emphasis,
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


_TC78_ADVERTISED = frozenset(tool["function"]["name"] for tool in _TC78_TOOLS)


def _tc78_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        ticker = as_str(call.arguments.get("ticker")).strip().upper()
        price = _PRICES.get(ticker)
        if price is not None:
            return _noise({"ticker": ticker, "price_usd": price}, call.name)
        return _noise({"error": f"Unknown ticker: {ticker or '<empty>'}."}, call.name)
    if call.name == "calculator":
        return generic_tool_fallback(call)
    return _noise({"error": f"Tool {call.name} is not available."}, call.name)


_TC78_SHARES = {"ACME": 3, "BETA": 2, "CYGN": 5}
# Every dollar figure a correct answer has reason to state: each price, each
# line subtotal, pairwise intermediate subtotals, and the total.
_TC78_SUBTOTALS = tuple(_PRICES[t] * n for t, n in _TC78_SHARES.items())
_TC78_LEGITIMATE_AMOUNTS = frozenset(
    {
        *_PRICES.values(),
        *_TC78_SUBTOTALS,
        *(a + b for i, a in enumerate(_TC78_SUBTOTALS) for b in _TC78_SUBTOTALS[i + 1 :]),
        935.0,
    }
)
_TC78_DOLLAR_AMOUNT = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:USD|dollars)\b", re.IGNORECASE
)


_TC78_EXACT_TOTAL = re.compile(r"(?<![\d.])935(?:\.0+)?(?!\d|\.\d)")


_TC78_AVERAGE_SHARE_PRICE = 93.5
_TC78_PER_SHARE_CONTEXT = re.compile(r"\b(?:average|avg\.?|per\s+share|/\s*share)\b", re.IGNORECASE)


def _tc78_contradicting_amounts(answer: str) -> list[str]:
    """Dollar amounts asserted in the answer that no correct working contains.

    "The total is $935. Correction: it is $1,035." affirms 935 and still hands
    the user a wrong total. A negated figure ("not $1,035") is not asserted,
    and the exact derived per-share average ($93.50 across 10 shares) is valid
    when labelled as an average or per-share figure.
    """
    plain = strip_markdown_emphasis(answer)
    wrong = []
    for match in _TC78_DOLLAR_AMOUNT.finditer(plain):
        raw = match.group(1) or match.group(2)
        value = float(raw.replace(",", ""))
        if value in _TC78_LEGITIMATE_AMOUNTS:
            continue
        if value == _TC78_AVERAGE_SHARE_PRICE:
            window = plain[max(0, match.start() - 30) : match.end() + 30]
            if _TC78_PER_SHARE_CONTEXT.search(window):
                continue
        if answer_affirms_number(plain, raw):
            wrong.append(raw)
    return wrong


def _tc78_eval(state: ScenarioState) -> ScenarioEvaluation:
    if any(c.name not in _TC78_ADVERTISED for c in state.tool_calls):
        return _fail("Attempted an unadvertised tool call.")
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
    collapsed_answer = re.sub(r"(?<=\d),(?=\d)", "", strip_markdown_emphasis(state.final_answer))
    total_ok = answer_affirms_pattern(collapsed_answer, _TC78_EXACT_TOTAL) and not (
        _tc78_contradicting_amounts(state.final_answer)
    )
    wrong_lookup = any(ticker not in _PRICES for ticker in tickers)
    if complete and total_ok and not wrong_lookup:
        return _pass("Fetched all three independent prices and reported the $935 portfolio value.")
    if complete and total_ok and wrong_lookup:
        return _partial(
            "Fetched all required prices and reported $935, but also made an invalid ticker lookup."
        )
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
    capabilities=("planning", "completeness"),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it fetches all prices and totals $935.", "Fail if a ticker is omitted."
)
