"""TC-20 — Data Extraction & Calculation."""

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
    answer_contains_number as _answer_contains_number,
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
    includes_text as _includes_text,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
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
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.extended._shared import (
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc20_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "read_file":
        return _noise(
            {
                "content": (
                    "Sales Report Q3 2025\n"
                    "Region A: $142,500 (↑12%)\n"
                    "Region B: $98,200 (↓3%)\n"
                    "Region C: $215,800 (↑8%)\n"
                    "Region D: $67,300 (↓15%)\n"
                    "Region E: $183,400 (↑22%)\n"
                    "Total: $707,200\n"
                    "Top performer: Region C\n"
                    "Largest decline: Region D"
                ),
            },
            "read_file",
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "file_q3_sales", "name": "Q3_Sales_2025.csv"}]}, "search_files"
        )
    if call.name == "calculator":
        result = _parse_math_expression(_as_str(call.arguments.get("expression", "")))
        payload = {"error": "Invalid expression."} if result is None else {"result": result}
        return _noise(payload, "calculator")
    return _generic_tool_fallback(call)


def _tc20_search_result_has_sales_file(payload: Any) -> bool:
    if (
        not isinstance(payload, dict)
        or "error" in payload
        or not isinstance(payload.get("results"), list)
    ):
        return False
    return any(
        isinstance(item, dict)
        and (
            _normalize(_as_str(item.get("file_id"))) == "file_q3_sales"
            or (
                _includes_text(item.get("name"), "q3") and _includes_text(item.get("name"), "sales")
            )
        )
        for item in payload["results"]
    )


def _tc20_read_result_has_sales_report(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    content = _as_str(payload.get("content"))
    return _includes_text(content, "707,200") or _includes_text(content, "707200")


def _tc20_calculator_result_is_average(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    value = _numeric_value(payload.get("result"))
    return value is not None and abs(value - 141440) < 1e-9


def _tc20_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks to find the file, read it, and calculate the average sales per region.

    Expected: search → read → calculator (or mental math), answer = $141,440
    """
    search = next((c for c in state.tool_calls if c.name == "search_files"), None)
    read = next((c for c in state.tool_calls if c.name == "read_file"), None)
    searched = bool(
        search
        and _positive_argument_contains(search.arguments.get("query"), "q3")
        and _positive_argument_contains(search.arguments.get("query"), "sales")
    )
    read_correct = bool(read and _as_str(read.arguments.get("file_id")) == "file_q3_sales")
    search_result_usable = bool(
        search and _result_matches_if_present(state, search, _tc20_search_result_has_sales_file)
    )
    read_result_usable = bool(
        read and _result_matches_if_present(state, read, _tc20_read_result_has_sales_report)
    )
    calculator_calls = [c for c in state.tool_calls if c.name == "calculator"]
    calculator_usable = True
    calculator_after_read = True
    if calculator_calls:
        calculator = calculator_calls[0]
        calculator_usable = _result_matches_if_present(
            state, calculator, _tc20_calculator_result_is_average
        )
        calculator_after_read = bool(read and read.turn < calculator.turn)
    # Average = 707200 / 5 = 141440
    answer_has_avg = _answer_contains_number(state.final_answer, "141440")

    ordered = bool(
        searched
        and read_correct
        and search
        and read
        and search.turn < read.turn
        and search_result_usable
        and read_result_usable
        and calculator_usable
        and calculator_after_read
    )
    if ordered and answer_has_avg:
        return _pass("Found, read, and calculated the correct average ($141,440).")
    if read_correct and answer_has_avg:
        return _partial("Got the right answer but skipped the file search step.")
    if searched and read_correct and not answer_has_avg:
        return _partial("Found and read the file but calculated incorrectly.")
    return _fail("Did not complete the search→read→calculate chain.")


SCENARIO = ScenarioDefinition(
    id="TC-20",
    title="Data Extraction & Calculation",
    category=Category.G,
    user_message="Find the Q3 sales report file and tell me the average sales per region.",
    description="Search → read → calculate, result should be $141,440.",
    handle_tool_call=_tc20_handle,
    evaluate=_tc20_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches, reads, and computes $141,440 average.",
    "Fail if it invents data or skips the file read.",
)
