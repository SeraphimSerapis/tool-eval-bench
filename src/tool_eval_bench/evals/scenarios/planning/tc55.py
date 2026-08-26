"""TC-55 — Data Pipeline."""

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


def _tc55_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return _noise(
            {
                "results": [
                    {"file_id": "q3_rev_na", "name": "Q3_Revenue_NA.xlsx"},
                    {"file_id": "q3_rev_emea", "name": "Q3_Revenue_EMEA.xlsx"},
                ]
            },
            "search_files",
        )
    if call.name == "read_file":
        fid = _as_str(call.arguments.get("file_id", ""))
        if fid == "q3_rev_na":
            return _noise(
                {
                    "content": "Q3 Revenue Report — North America\nTotal Revenue: $2,400,000\nSegments: Enterprise $1.4M, SMB $600K, Consumer $400K"
                },
                "read_file",
            )
        if fid == "q3_rev_emea":
            return _noise(
                {
                    "content": "Q3 Revenue Report — EMEA\nTotal Revenue: $1,800,000\nSegments: UK $900K, DACH $500K, Nordics $400K"
                },
                "read_file",
            )
        return _noise({"error": f"File not found: {fid}"}, "read_file")
    if call.name == "calculator":
        from tool_eval_bench.evals.helpers import parse_math_expression

        expr = _as_str(call.arguments.get("expression", ""))
        result = parse_math_expression(expr)
        if result is not None:
            return _noise({"result": result}, "calculator")
        return _noise({"error": "Invalid expression."}, "calculator")
    return _noise({"error": f"Tool {call.name} is not relevant."}, call.name)


def _tc55_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Find all Q3 revenue files and calculate the total revenue
    across all regions.'

    Must: search_files → read_file (×2) → calculator to sum.
    Total = $2,400,000 + $1,800,000 = $4,200,000.
    """

    def search_result_has_regions(payload: Any) -> bool:
        return all(identifier in str(payload) for identifier in ("q3_rev_na", "q3_rev_emea"))

    def read_result_has_amount(payload: Any, amount: str) -> bool:
        return amount in str(payload).replace(",", "")

    def calculator_result_is_total(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        value = payload.get("result")
        if value is None:
            return False
        try:
            return abs(float(value) - 4200000) < 0.01
        except (TypeError, ValueError):
            return "4200000" in str(value).replace(",", "")

    search_calls = [
        call
        for call in _tool_calls_by_name(state, "search_files")
        if "q3" in _as_str(call.arguments.get("query", "")).lower()
        and "revenue" in _as_str(call.arguments.get("query", "")).lower()
        and _result_matches_if_present(state, call, search_result_has_regions)
    ]
    read_na_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if _as_str(call.arguments.get("file_id", "")) == "q3_rev_na"
        and _result_matches_if_present(
            state, call, lambda payload: read_result_has_amount(payload, "2400000")
        )
    ]
    read_emea_calls = [
        call
        for call in _tool_calls_by_name(state, "read_file")
        if _as_str(call.arguments.get("file_id", "")) == "q3_rev_emea"
        and _result_matches_if_present(
            state, call, lambda payload: read_result_has_amount(payload, "1800000")
        )
    ]
    searched = bool(search_calls)
    read_na = bool(read_na_calls)
    read_emea = bool(read_emea_calls)
    answer = state.final_answer
    has_total = any(_answer_affirms_number(answer, value) for value in ("4200000", "4.2")) and any(
        marker in answer.lower() for marker in ("million", "4.2m", "$4.2", "4200000", "4,200,000")
    )

    calculator_calls = [
        call
        for call in _tool_calls_by_name(state, "calculator")
        if bool(
            (expression := _as_str(call.arguments.get("expression")).replace(",", ""))
            and "2400000" in expression
            and "1800000" in expression
            and "+" in expression
            and _parse_math_expression(expression) is not None
        )
        and _result_matches_if_present(state, call, calculator_result_is_total)
    ]
    calculator = bool(calculator_calls)
    dependencies_satisfied = bool(
        search_calls
        and read_na_calls
        and read_emea_calls
        and calculator_calls
        and _call_index(state, search_calls[0])
        < min(_call_index(state, read_na_calls[0]), _call_index(state, read_emea_calls[0]))
        and max(_call_index(state, read_na_calls[0]), _call_index(state, read_emea_calls[0]))
        < _call_index(state, calculator_calls[0])
    )
    if (
        searched
        and read_na
        and read_emea
        and calculator
        and has_total
        and not dependencies_satisfied
    ):
        return _partial("Calculated before both regional files had been read.")
    if searched and read_na and read_emea and calculator and has_total:
        if _has_unexpected_tools(state, {"search_files", "read_file", "calculator"}):
            return _partial("Aggregated the files but also called an unrelated tool.")
        return _pass("Built data pipeline: search → read ×2 → calculate total revenue.")
    if searched and read_na and read_emea and has_total:
        return _partial("Read both files and produced the total but didn't use the calculator.")
    if searched and (read_na or read_emea) and has_total:
        return _partial("Got the total but only read one of two files.")
    if searched and read_na and read_emea:
        return _partial("Read both files but didn't calculate the combined total.")
    if searched:
        return _partial("Found files but didn't read and aggregate them.")
    return _fail("Did not build a data pipeline to aggregate Q3 revenue files.")


SCENARIO = ScenarioDefinition(
    id="TC-55",
    title="Data Pipeline",
    category=Category.N,
    user_message="Find all Q3 revenue files and calculate the total revenue across all regions.",
    description="Build pipeline: search → read ×2 → calculate aggregate.",
    handle_tool_call=_tc55_handle,
    evaluate=_tc55_eval,
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches → reads both revenue files → calculates total ($4.2M).",
    "Fail if it doesn't build the multi-read data pipeline.",
)
