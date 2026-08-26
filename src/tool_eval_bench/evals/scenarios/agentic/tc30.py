"""TC-30 — Chained Conditional Execution."""

from __future__ import annotations

import ast
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
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
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


def _tc30_print_argument(statement: ast.stmt) -> ast.expr | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Name)
        or call.func.id != "print"
        or len(call.args) != 1
        or call.keywords
    ):
        return None
    return call.args[0]


def _tc30_is_integer(node: ast.expr, value: int) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == value


def _tc30_is_calculation(node: ast.expr, calculated_names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in calculated_names
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _tc30_is_integer(node.left, 2)
        and _tc30_is_integer(node.right, 2)
    )


def _tc30_prints_text(statements: list[ast.stmt], expected: str) -> bool:
    if len(statements) != 1:
        return False
    argument = _tc30_print_argument(statements[0])
    return (
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.strip().lower() == expected
    )


def _tc30_is_expected_workflow(code: str) -> bool:
    """Recognize the requested combined Python workflow without executing it."""
    try:
        module = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False

    calculated_names: set[str] = set()
    printed_calculation = False
    for index, statement in enumerate(module.body):
        if (
            isinstance(statement, ast.Assign)
            and not printed_calculation
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and _tc30_is_calculation(statement.value, calculated_names)
        ):
            calculated_names.add(statement.targets[0].id)
            continue

        printed = _tc30_print_argument(statement)
        if (
            printed is not None
            and not printed_calculation
            and _tc30_is_calculation(printed, calculated_names)
        ):
            printed_calculation = True
            continue

        if not isinstance(statement, ast.If) or not printed_calculation:
            return False
        comparison = statement.test
        if (
            not isinstance(comparison, ast.Compare)
            or len(comparison.ops) != 1
            or not isinstance(comparison.ops[0], ast.Eq)
            or len(comparison.comparators) != 1
        ):
            continue
        left = comparison.left
        right = comparison.comparators[0]
        compares_result_to_four = (
            _tc30_is_calculation(left, calculated_names) and _tc30_is_integer(right, 4)
        ) or (_tc30_is_integer(left, 4) and _tc30_is_calculation(right, calculated_names))
        if (
            compares_result_to_four
            and _tc30_prints_text(statement.body, "correct")
            and _tc30_prints_text(statement.orelse, "wrong")
            and index == len(module.body) - 1
        ):
            return True
        return False
    return False


def _tc30_is_print_calculation(code: str) -> bool:
    """Recognize exactly one print of the requested 2 + 2 calculation."""
    try:
        module = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    return len(module.body) == 1 and (
        (argument := _tc30_print_argument(module.body[0])) is not None
        and _tc30_is_calculation(argument, set())
    )


def _tc30_is_print_text(code: str, expected: str) -> bool:
    """Recognize exactly one print of the expected branch text."""
    try:
        module = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    return _tc30_prints_text(module.body, expected)


def _tc30_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "run_code":
        code = _as_str(call.arguments.get("code"))
        if _tc30_is_expected_workflow(code):
            return _noise({"stdout": "4\ncorrect", "stderr": "", "exit_code": 0}, "run_code")
        if _tc30_is_print_calculation(code):
            return _noise({"stdout": "4", "stderr": "", "exit_code": 0}, "run_code")
        if _tc30_is_print_text(code, "correct"):
            return _noise({"stdout": "correct", "stderr": "", "exit_code": 0}, "run_code")
        if _tc30_is_print_text(code, "wrong"):
            return _noise({"stdout": "wrong", "stderr": "", "exit_code": 0}, "run_code")
        return _noise({"stdout": "", "stderr": "Unknown code", "exit_code": 1}, "run_code")
    return _generic_tool_fallback(call)


def _tc30_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Run print(2+2) in Python. If the result is 4, run print("correct"), otherwise run print("wrong").'

    Should make 2 run_code calls. First gets "4", then runs "correct".
    Also accept: 1 run_code call that runs the "correct" branch — the model
    computed 2+2=4 mentally and skipped the unnecessary first call.
    """
    code_calls = [c for c in state.tool_calls if c.name == "run_code"]

    if not code_calls:
        return _fail("Did not use run_code at all.")

    if len(state.tool_calls) != len(code_calls):
        return _fail("Used an unrelated tool during the code workflow.")

    if len(code_calls) == 2:
        # The second call is only a dependent branch if the first call really
        # performed the requested calculation.  Keyword-only calls are not a
        # valid chain, even when the final branch text is correct.
        first_code = _as_str(code_calls[0].arguments.get("code"))
        if not _tc30_is_print_calculation(first_code):
            return _fail("Second branch call was not grounded in a 2+2 calculation.")
        if code_calls[1].turn <= code_calls[0].turn:
            return _fail("Ran the conditional branch before receiving the calculation result.")
        if not _result_is_usable_if_present(
            state, code_calls[0]
        ) or not _result_is_usable_if_present(state, code_calls[1]):
            return _partial("A run_code call returned an explicit error.")
        first_results = _matching_tool_results(state, code_calls[0])
        if first_results and not any(
            isinstance(result.result, dict) and _as_str(result.result.get("stdout")).strip() == "4"
            for result in first_results
        ):
            return _fail("The conditional branch was not grounded in a recorded result of 4.")
        second_code = _as_str(code_calls[1].arguments.get("code"))
        ran_correct = _tc30_is_print_text(second_code, "correct")
        ran_wrong = _tc30_is_print_text(second_code, "wrong")

        if ran_correct and not ran_wrong:
            return _pass("Correctly chained: ran 2+2, saw 4, ran 'correct'.")
        if ran_wrong:
            return _fail("Ran the 'wrong' branch despite the result being 4.")
        return _partial("Made 2 calls but the conditional logic was unclear.")

    if len(code_calls) > 2:
        return _partial(
            f"Made {len(code_calls)} run_code calls instead of one complete call or two dependent calls."
        )

    # A single call is valid only when it executes the complete conditional
    # workflow. Printing the branch alone skips the requested calculation.
    raw_first_code = _as_str(code_calls[0].arguments.get("code"))
    if _tc30_is_expected_workflow(raw_first_code):
        if not _result_is_usable_if_present(state, code_calls[0]):
            return _partial("The combined run_code call returned an explicit error.")
        return _pass("Ran the calculation and correct conditional branch in one call.")
    if _tc30_is_print_text(raw_first_code, "correct"):
        return _partial("Ran the correct branch but skipped the requested calculation.")
    if _tc30_is_print_calculation(raw_first_code):
        return _partial("Ran the calculation but didn't follow up with the conditional.")
    return _partial("Made 1 run_code call — conditional logic unclear.")


SCENARIO = ScenarioDefinition(
    id="TC-30",
    title="Chained Conditional Execution",
    category=Category.J,
    user_message='Run print(2+2) in Python. If the result is 4, then run print("correct"). Otherwise run print("wrong").',
    description="Must chain two run_code calls with conditional logic.",
    handle_tool_call=_tc30_handle,
    evaluate=_tc30_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it runs 2+2, sees 4, then runs print('correct').",
    "Fail if it skips the conditional or runs 'wrong'.",
)
