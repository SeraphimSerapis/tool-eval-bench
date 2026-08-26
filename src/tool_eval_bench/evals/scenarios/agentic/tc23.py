"""TC-23 — Explicit Tool Prohibition."""

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
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
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


def _tc23_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return _noise({"ticker": "AAPL", "price": 178.50, "currency": "USD"}, "get_stock_price")
    return _generic_tool_fallback(call)


def _tc23_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User said: 'Without using any tools, explain what get_stock_price likely does based on its name.'"""
    if any(c.name == "get_stock_price" for c in state.tool_calls):
        return _fail("Called get_stock_price despite explicit instruction not to use tools.")
    if len(state.tool_calls) > 0:
        tools_used = ", ".join(c.name for c in state.tool_calls)
        return _fail(f"Used tools ({tools_used}) despite instruction to not use any.")

    answer = state.final_answer.lower()
    # Collapse all whitespace (including LF/CRLF, tabs, and runs of spaces)
    # so formatting such as headings, bullets, and line breaks cannot break
    # the semantic regex chains below. Meaning is unchanged: the chains still
    # require a retrieval/return/fetch action tied to stock/price/ticker and
    # to the function name, and negation of those facts still blocks a pass.
    answer = re.sub(r"\s+", " ", answer)
    # Should explain what the function does
    explains = bool(
        re.search(
            r"(?:get_stock_price|function).{0,80}(?:retriev|return|fetch|look up|obtain).{0,80}(?:stock|price|ticker)"
            r"|(?:retriev|return|fetch|look up|obtain).{0,80}(?:stock|price|ticker).{0,80}(?:function|get_stock_price)",
            answer,
        )
    ) and not re.search(
        r"(?:does not|doesn't|not|never)\s+"
        r"(?:(?:a|an|the)\s+)?"
        r"(?:function\s+(?:that|which)\s+)?"
        r"(?:retriev|return|fetch|look up|obtain)",
        answer,
    )
    if explains:
        return _pass("Explained the function without calling any tools.")
    return _partial("Did not use tools (good) but explanation was unclear.")


SCENARIO = ScenarioDefinition(
    id="TC-23",
    title="Explicit Tool Prohibition",
    category=Category.H,
    user_message="Without using any tools, explain what the get_stock_price function likely does based on its name and parameters.",
    description="Must NOT call any tools despite them being available.",
    handle_tool_call=_tc23_handle,
    evaluate=_tc23_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it explains get_stock_price without calling any tools.",
    "Fail if it calls get_stock_price or any other tool.",
)
