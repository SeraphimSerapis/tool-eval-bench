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
    answer_affirms_pattern as _answer_affirms_pattern,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
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
    strip_markdown_emphasis as _strip_markdown_emphasis,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)

# The description verb is the model's choice: "returns the current market
# price", "looks up the price", "gives you the price" and "queries a market
# data source" all describe the same function. An allowlist of four verbs
# graded vocabulary, and missed "looks up" because it was spelled "look up".
_TC23_VERB = (
    r"\b(?:retriev\w*|returns?|returning|fetch(?:es|ing)?|looks?\s*up|lookups?|"
    r"obtains?|obtaining|gets?\b|getting|provid\w*|gives?\b|giving|quer(?:y|ies|ying)|"
    r"reports?\b|pulls?\b|reads?\b|shows?\b|tells?\b|finds?\b|access(?:es|ing)?|"
    r"expos\w*|deliver\w*|surfac\w*|yields?\b)"
)


_TC23_ACTION = re.compile(rf"(?:{_TC23_VERB}).{{0,80}}(?:stock|price|ticker)")
_TC23_COORDINATED_NEGATION = re.compile(
    r"(?:does not|doesn't|not|never)\s+"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:function\s+(?:that|which)\s+)?"
    r"(?:\w+\s+){0,4}?(?:or|nor)\s+$"
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

    # Normalize apostrophe variants and Markdown emphasis before checking
    # whether the explanation asserts a stock-price lookup rather than negating it.
    answer = re.sub(r"\s+", " ", _normalize(_strip_markdown_emphasis(state.final_answer)))
    explains = bool(re.search(r"get_stock_price|\bfunction\b", answer)) and _answer_affirms_pattern(
        answer,
        _TC23_ACTION,
        skip=lambda prefix: _TC23_COORDINATED_NEGATION.search(prefix[-80:]) is not None,
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
