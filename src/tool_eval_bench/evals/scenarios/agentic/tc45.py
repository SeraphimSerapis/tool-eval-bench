"""TC-45 — tool_choice=required Compliance."""

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
    generic_tool_fallback_simple as _generic_tool_fallback,
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
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc45_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "calculator":
        expr = _as_str(call.arguments.get("expression", ""))
        result = _parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression"}, "calculator")
    if call.name == "web_search":
        query = _as_str(call.arguments.get("query", ""))
        return _noise({"results": [{"snippet": f"Result for: {query}"}]}, "web_search")
    return _generic_tool_fallback(call)


def _tc45_eval(state: ScenarioState) -> ScenarioEvaluation:
    """With tool_choice="required", the model MUST call at least one tool.

    The question "What is 7 * 8?" is trivial, so a model might skip tools.
    But with required, it should still use calculator or web_search.

    Pass:    Used calculator with '7*8' or '7 * 8'
    Partial: Used any tool (even wrong one) — at least honored required
    Fail:    No tool calls despite tool_choice="required"
    """
    if not state.tool_calls:
        return _fail("No tool calls despite tool_choice='required'.")

    if any(
        call.name in {"send_email", "create_calendar_event", "set_reminder", "run_code"}
        for call in state.tool_calls
    ):
        return _fail(
            "Performed an unauthorized side effect while satisfying tool_choice='required'."
        )

    unrelated_calls = [call for call in state.tool_calls if call.name != "calculator"]
    calc_calls = [c for c in state.tool_calls if c.name == "calculator"]
    if calc_calls:
        expr = _as_str(calc_calls[0].arguments.get("expression", ""))
        result = _parse_math_expression(expr)
        if (
            result is not None
            and abs(result - 56) < 0.01
            and _result_is_usable_if_present(state, calc_calls[0])
        ):
            # Verify the model surfaced the computed result.
            if unrelated_calls:
                return _partial("Used the calculator correctly but also called an unrelated tool.")
            if _answer_affirms_number(state.final_answer, "56"):
                return _pass(
                    "Used calculator with correct expression — honored tool_choice='required'."
                )
            return _partial(
                "Used calculator correctly but did not surface the result (56) in the answer.",
            )
        return _partial("Used calculator but expression didn't evaluate to 56.")
    return _partial(
        f"Used {state.tool_calls[0].name} instead of calculator — honored 'required' but wrong tool."
    )


SCENARIO = ScenarioDefinition(
    id="TC-45",
    title="tool_choice=required Compliance",
    category=Category.H,
    user_message="What is 7 times 8?",
    description="With tool_choice='required', the model MUST call at least one tool even for trivial math.",
    handle_tool_call=_tc45_handle,
    evaluate=_tc45_eval,
    tool_choice_override="required",
    tool_choice_after_first_call="auto",
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls calculator with 7*8 despite being a trivial question.",
    "Fail if it answers directly without calling any tool (tool_choice='required').",
)
