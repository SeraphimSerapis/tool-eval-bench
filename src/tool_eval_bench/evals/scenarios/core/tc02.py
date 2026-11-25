"""TC-02 — Distractor Resistance."""

from __future__ import annotations

import re
from decimal import Decimal
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
    _is_negated,
    forbid_unrequested_side_effects,
)
from tool_eval_bench.evals.helpers import (
    answer_affirms_pattern as _answer_affirms_pattern,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    has_explicit_tool_error as _has_explicit_tool_error,
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
    _THOUSANDS_COMMA,
    _answer_affirms_rounded_value,
    _numeric_value,
    _only_error_retries_before_last,
    _result_matches_if_present,
)

_TC02_TICKER_RE = re.compile(r"(?:(?:nasdaq|nyse|us|xnas):\s*)?\$?aapl", re.IGNORECASE)
_TC02_PRICE_MENTION = re.compile(
    r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:usd|dollars?)\b",
    re.IGNORECASE,
)
_TC02_DOLLAR_AMOUNT = re.compile(r"(?<![+\-])\$\s*(\d+(?:\.\d+)?)(?!\d|\.\d)")
_TC02_ACCEPTED_DOLLAR_VALUES = frozenset(
    {Decimal("187"), Decimal("187.4"), Decimal("187.42"), Decimal("1.23"), Decimal("186.19")}
)


def _tc02_ticker_is_aapl(value: Any) -> bool:
    return bool(_TC02_TICKER_RE.fullmatch(_normalize(_as_str(value))))


def _tc02_has_conflicting_dollar_price(answer: str) -> bool:
    collapsed = _THOUSANDS_COMMA.sub("", answer)
    for match in _TC02_DOLLAR_AMOUNT.finditer(collapsed):
        prefix = collapsed[max(0, match.start() - 120) : match.start()]
        if _is_negated(prefix):
            continue
        if re.search(
            r"\b(?:from|previous|prior|yesterday|change|gained|up|down)\s+$", prefix, re.I
        ):
            continue
        amount = Decimal(match.group(1))
        if amount not in _TC02_ACCEPTED_DOLLAR_VALUES:
            return True
    return False


def _tc02_stock_result_is_aapl(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    if "ticker" in payload and _normalize(_as_str(payload.get("ticker"))) != "aapl":
        return False
    price = _numeric_value(payload.get("price"))
    return price is not None and abs(price - 187.42) < 1e-9


def _tc02_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return _noise(
            {
                "ticker": "AAPL",
                "price": 187.42,
                "currency": "USD",
                "change": "+1.23",
                "change_percent": "+0.66%",
            },
            "get_stock_price",
        )
    if call.name == "web_search":
        return _noise({"results": [{"snippet": "AAPL is trading around $187.42."}]}, "web_search")
    return _generic_tool_fallback(call)


@forbid_unrequested_side_effects()
def _tc02_eval(state: ScenarioState) -> ScenarioEvaluation:
    aapl_calls = [
        c
        for c in state.tool_calls
        if c.name == "get_stock_price" and _tc02_ticker_is_aapl(c.arguments.get("ticker"))
    ]
    stock = bool(aapl_calls)
    web = _has_tool_call(state, "web_search")
    only_aapl_lookups = bool(state.tool_calls) and len(aapl_calls) == len(state.tool_calls)
    has_price = _answer_affirms_rounded_value(state.final_answer, "187.42") and not (
        _tc02_has_conflicting_dollar_price(state.final_answer)
    )
    if only_aapl_lookups:
        # Same retry rule as TC-01: a retry after an error is recovery and the
        # last call is graded; a repeat after a usable price is redundant.
        if not _only_error_retries_before_last(state, state.tool_calls):
            return _partial(
                "Called get_stock_price for AAPL again after a usable result; "
                "the duplicate call was redundant."
            )
        stock_call = state.tool_calls[-1]
        if not _result_matches_if_present(state, stock_call, _tc02_stock_result_is_aapl):
            if _has_explicit_tool_error(state, stock_call) and _answer_affirms_pattern(
                state.final_answer, _TC02_PRICE_MENTION
            ):
                return _fail(
                    "get_stock_price returned an error; the answer cannot fabricate a stock price."
                )
            return _partial(
                "get_stock_price did not return a usable AAPL price, so no price can be confirmed."
            )
        # Verify the model surfaced the actual stock price. A rounded "$187"
        # is the same price; a different decimal tail ("$187.99") or a
        # conflicting dollar price ("$250.00 (up from 187)") is not.
        if has_price:
            return _pass("Used only get_stock_price for AAPL.")
        if re.search(r"\$\s*\d|\b\d+\.\d+\b", state.final_answer):
            return _partial(
                "Called get_stock_price correctly but stated a price that does not match "
                "the tool result ($187.42)."
            )
        return _partial(
            "Called get_stock_price correctly but did not surface the price in the answer.",
        )
    if (
        stock
        and web
        and all(c.name in {"get_stock_price", "web_search"} for c in state.tool_calls)
        and any(
            _result_matches_if_present(state, c, _tc02_stock_result_is_aapl) for c in aapl_calls
        )
        and has_price
    ):
        return _partial("Called the right tool, but added unnecessary web_search.")
    return _fail("Did not isolate the request to get_stock_price.")


SCENARIO = ScenarioDefinition(
    id="TC-02",
    title="Distractor Resistance",
    category=Category.A,
    user_message="What is the current price of AAPL stock?",
    description="Use get_stock_price without extra tools.",
    handle_tool_call=_tc02_handle,
    evaluate=_tc02_eval,
    difficulty=1,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it uses only get_stock_price with ticker AAPL and surfaces the price.",
    "Partial if it adds web_search alongside a valid AAPL lookup; "
    "fail if it uses unrelated distractor tools or answers without a stock lookup.",
)
