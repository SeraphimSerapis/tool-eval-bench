"""TC-02 — Distractor Resistance."""

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
    _result_matches_if_present,
)


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


def _tc02_eval(state: ScenarioState) -> ScenarioEvaluation:
    stock = _has_tool_call(
        state, "get_stock_price", lambda c: _normalize(_as_str(c.arguments.get("ticker"))) == "aapl"
    )
    web = _has_tool_call(state, "web_search")
    if stock and not web and len(state.tool_calls) == 1:
        stock_call = _first_call(state, "get_stock_price")
        if stock_call and not _result_matches_if_present(
            state, stock_call, _tc02_stock_result_is_aapl
        ):
            return _partial(
                "get_stock_price did not return a usable AAPL price, so no price can be confirmed."
            )
        # Verify the model surfaced the actual stock price.
        has_price = _answer_affirms_number(state.final_answer, "187")
        if has_price:
            return _pass("Used only get_stock_price for AAPL.")
        return _partial(
            "Called get_stock_price correctly but did not surface the price in the answer.",
        )
    if stock and web:
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
    "Pass if it uses only get_stock_price with ticker AAPL.",
    "Fail if it uses distractor tools or answers without a stock lookup.",
)
