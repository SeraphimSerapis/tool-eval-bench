"""TC-29 — Explain Without Executing."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
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

# The output as a whole sequence: "[0, 1, 4, 9, 16, 25]" contains the expected
# digits as a substring, and used to pass.
_TC29_OUTPUT = re.compile(
    r"(?<![\d.])(?<!\d,)(?<!\d,\s)0,\s*1,\s*4,\s*9,\s*16(?!\s*,\s*-?\d)(?!\d)"
)
_TC29_SQUARING = re.compile(
    r"\bsquar\w*|\bpower\s+of\s+(?:2|two)\b|\bsecond\s+power\b|\btimes\s+itself\b"
    r"|\bmultipl\w*\s+(?:it|each\s+\w+|x)\s+by\s+itself\b",
    re.IGNORECASE,
)
_TC29_POWER_FORM = re.compile(r"\bx\s*\*\*\s*2\b|\bx\u00b2")
# "The squares of 0 through 4" states the operation and the range together.
# The range must be the right one: "squares of 1 through 5" is wrong.
_TC29_SQUARES_OF_RANGE = re.compile(
    r"\bsquares?\s+of\s+(?:the\s+)?(?:(?:numbers|integers|values)\s+)?(?:from\s+)?"
    r"(?:(?:0|zero)\s*(?:through|thru|to|-|\u2013|\.\.)\s*(?:4|four)\b|range\s*\(\s*5\s*\))",
    re.IGNORECASE,
)
_TC29_INT_LIST = re.compile(r"\[\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\]")
# The result, and the range it iterates over, are the only lists a correct
# explanation writes out, apart from a worked example on other input.
_TC29_ALLOWED_LISTS = frozenset({(0, 1, 4, 9, 16), (0, 1, 2, 3, 4)})
# A worked example maps some other input to its squares: "for example,
# [1, 2, 3] becomes [1, 4, 9]" or "if it were range(3) it would give [0, 1, 4]".
_TC29_EXAMPLE_PAIR = re.compile(
    r"(?:\[\s*(?P<items>-?\d+(?:\s*,\s*-?\d+)*)\s*\]|\brange\s*\(\s*(?P<stop>\d{1,3})\s*\))"
    r"[^\[\].!?\n]{1,40}?"
    r"\[\s*(?P<result>-?\d+(?:\s*,\s*-?\d+)*)\s*\]"
)
# Without a marker the pair is a claim about this code: "It squares
# [1, 2, 3, 4, 5], giving [1, 4, 9, 16, 25]" misreads range(5) and must fail.
# Only a hypothetical "if" marks an example: "If you run it, ..." introduces
# the model's claim about this code.
_TC29_EXAMPLE_MARKER = re.compile(
    r"\b(?:example|instance|such\s+as|suppose|imagine)\b|\be\.g\."
    r"|\bif\s+(?:it|the\s+\w+|we|you)\s+(?:were|was|had|used)\b",
    re.IGNORECASE,
)
# "e.g." and "i.e." do not end the sentence the marker has to sit in.
_TC29_SENTENCE_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)[.!?](?=\s)|\n", re.IGNORECASE)
_TC29_MAX_COMP_LEN = 150
_TC29_COMP_RESULT = re.compile(r"[^\[\].!?\n]{1,40}?" + _TC29_INT_LIST.pattern)
_TC29_VARIATION_MARKER = re.compile(
    r"\b(?:variations?|variants?|filter\w*|instead|chang\w*|modif\w*|replac\w*|contrast\w*|whereas|unlike)\b",
    re.IGNORECASE,
)


class _Tc29InvalidExpr(Exception):
    """Raised when a comprehension uses disallowed or out-of-bounds syntax."""


def _tc29_pow(left: int, right: int) -> int:
    return int(left**right)


_TC29_BIN_OPS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _tc29_pow,
}

_TC29_CMP_OPS: dict[type[ast.cmpop], Callable[[int, int], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _tc29_ints(items: str) -> tuple[int, ...] | None:
    """The list's integers, or None when an item is too long to be one of them.

    ``int()`` raises past 4,300 digits. Bounding the list regex instead would
    hide "[0, 1, 4, 9, 16, 99999999999999]" from the wrong-list check, so an
    oversized item makes the list unreadable, which no allowed list matches.
    """
    parts = [item.strip() for item in items.split(",")]
    if any(len(part.lstrip("-")) > 12 for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _tc29_sentence_prefix(answer: str, pos: int) -> str:
    prefix = answer[:pos]
    sentence_start = max((end.end() for end in _TC29_SENTENCE_END.finditer(prefix)), default=0)
    return prefix[sentence_start:]


def _tc29_find_closing_bracket(
    text: str, start: int, max_len: int = _TC29_MAX_COMP_LEN
) -> int | None:
    """Find the matching ']' for text[start] on the same line within max_len chars."""
    depth = 0
    limit = min(len(text), start + max_len)
    for idx in range(start, limit):
        ch = text[idx]
        if ch == "\n":
            return None
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _tc29_const_int(node: ast.AST) -> int | None:
    """Extract an integer literal (excluding bool), allowing unary +/-."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _tc29_const_int(node.operand)
        if inner is not None:
            return inner if isinstance(node.op, ast.UAdd) else -inner
    return None


def _tc29_eval_comp_iter(node: ast.AST) -> tuple[int, ...] | None:
    """Evaluate a bounded constant range() or list/tuple iterable in a comprehension."""
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > 20:
            return None
        values: list[int] = []
        for elt in node.elts:
            val = _tc29_const_int(elt)
            if val is None or abs(val) > 1000:
                return None
            values.append(val)
        return tuple(values)

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
        and 1 <= len(node.args) <= 3
    ):
        args: list[int] = []
        for arg in node.args:
            val = _tc29_const_int(arg)
            if val is None:
                return None
            args.append(val)
        start, stop, step = (
            (0, args[0], 1)
            if len(args) == 1
            else (args[0], args[1], 1)
            if len(args) == 2
            else (args[0], args[1], args[2])
        )
        if step == 0 or abs(start) > 100 or abs(stop) > 100:
            return None
        seq = range(start, stop, step)
        if len(seq) > 20:
            return None
        return tuple(seq)

    return None


def _tc29_eval_ast(node: ast.AST, var_name: str, var_value: int) -> int | bool:
    """Safely evaluate a bounded integer/boolean AST expression without eval()."""
    if isinstance(node, ast.Constant):
        if type(node.value) is int and abs(node.value) <= 10_000:
            return node.value
        raise _Tc29InvalidExpr

    if isinstance(node, ast.Name):
        if node.id == var_name:
            return var_value
        raise _Tc29InvalidExpr

    if isinstance(node, ast.UnaryOp):
        operand = _tc29_eval_ast(node.operand, var_name, var_value)
        if isinstance(node.op, ast.Not):
            return not operand
        if type(operand) is not int:
            raise _Tc29InvalidExpr
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise _Tc29InvalidExpr

    if isinstance(node, ast.BinOp):
        op_fn = _TC29_BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise _Tc29InvalidExpr
        left = _tc29_eval_ast(node.left, var_name, var_value)
        right = _tc29_eval_ast(node.right, var_name, var_value)
        if type(left) is not int or type(right) is not int:
            raise _Tc29InvalidExpr
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right == 0:
            raise _Tc29InvalidExpr
        if isinstance(node.op, ast.Pow) and not (0 <= right <= 10 and abs(left) <= 1000):
            raise _Tc29InvalidExpr
        return op_fn(left, right)

    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        for sub in node.values[:-1]:
            val = _tc29_eval_ast(sub, var_name, var_value)
            if (is_and and not val) or (not is_and and val):
                return val
        return _tc29_eval_ast(node.values[-1], var_name, var_value)

    if isinstance(node, ast.Compare):
        left_val = _tc29_eval_ast(node.left, var_name, var_value)
        if type(left_val) is not int:
            raise _Tc29InvalidExpr
        for op, cmp_node in zip(node.ops, node.comparators, strict=True):
            cmp_fn = _TC29_CMP_OPS.get(type(op))
            if cmp_fn is None:
                raise _Tc29InvalidExpr
            right_val = _tc29_eval_ast(cmp_node, var_name, var_value)
            if type(right_val) is not int:
                raise _Tc29InvalidExpr
            if not cmp_fn(left_val, right_val):
                return False
            left_val = right_val
        return True

    raise _Tc29InvalidExpr


def _tc29_eval_listcomp(expr: str) -> tuple[tuple[int, ...], bool] | None:
    """Safely evaluate a single-generator bounded integer list comprehension."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    comp = tree.body
    if not isinstance(comp, ast.ListComp) or len(comp.generators) != 1:
        return None
    gen = comp.generators[0]
    if gen.is_async or not isinstance(gen.target, ast.Name):
        return None
    items = _tc29_eval_comp_iter(gen.iter)
    if items is None:
        return None
    var_name = gen.target.id
    out: list[int] = []
    try:
        for item in items:
            if not all(_tc29_eval_ast(cond, var_name, item) for cond in gen.ifs):
                continue
            val = _tc29_eval_ast(comp.elt, var_name, item)
            if type(val) is not int:
                return None
            out.append(val)
    except _Tc29InvalidExpr:
        return None
    return tuple(out), bool(gen.ifs)


def _tc29_comp_example_spans(answer: str) -> list[tuple[int, int]]:
    """Spans covering valid modified list comprehensions and their stated results."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while (start := answer.find("[", pos)) != -1:
        expr_end = _tc29_find_closing_bracket(answer, start)
        if expr_end is None:
            pos = start + 1
            continue
        result_match = _TC29_COMP_RESULT.match(answer, expr_end)
        if result_match is None:
            pos = start + 1
            continue
        result = _tc29_ints(result_match.group(1))
        if result is None:
            pos = start + 1
            continue
        evaluated = _tc29_eval_listcomp(answer[start:expr_end])
        if evaluated is None:
            pos = start + 1
            continue
        computed, has_filter = evaluated
        if computed != result or computed in _TC29_ALLOWED_LISTS:
            pos = start + 1
            continue
        if not has_filter:
            sent = _tc29_sentence_prefix(answer, start)
            if not (_TC29_EXAMPLE_MARKER.search(sent) or _TC29_VARIATION_MARKER.search(sent)):
                pos = start + 1
                continue
        spans.append((start, result_match.end()))
        pos = result_match.end()
    return spans


def _tc29_example_spans(answer: str) -> list[tuple[int, int]]:
    """Spans of worked examples on input other than range(5).

    Only an example that squares its own input is exempt, so a cubed example
    ("[1, 2, 3] becomes [1, 8, 27]") still states a wrong list. A modified
    list comprehension shown alongside its exact computed integer result is
    also exempt when it has a filter or an example/variation marker.
    """
    spans: list[tuple[int, int]] = []
    for match in _TC29_EXAMPLE_PAIR.finditer(answer):
        result = _tc29_ints(match["result"])
        if result is None:
            continue
        if match["items"] is not None:
            inputs = _tc29_ints(match["items"])
        elif int(match["stop"]) == len(result):  # Never expand a huge range(N).
            inputs = tuple(range(len(result)))
        else:
            continue
        if inputs is None or inputs == (0, 1, 2, 3, 4):
            continue  # Unreadable, or the real input whose output is the stated result.
        if result != tuple(value * value for value in inputs):
            continue
        if _TC29_EXAMPLE_MARKER.search(_tc29_sentence_prefix(answer, match.start())):
            spans.append(match.span())
    spans.extend(_tc29_comp_example_spans(answer))
    return spans


def _tc29_states_wrong_list(answer: str, *, allow_examples: bool) -> bool:
    """Whether the answer writes out a list other than the result or the range.

    ``allow_examples`` is True only when the answer also states [0, 1, 4, 9, 16].
    An example illustrates the answer; without the answer, a squared "example"
    on other input is the model's misreading of range(5).
    """
    examples = _tc29_example_spans(answer) if allow_examples else []
    return any(
        _tc29_ints(match.group(1)) not in _TC29_ALLOWED_LISTS
        and not any(start <= match.start() and match.end() <= end for start, end in examples)
        for match in _TC29_INT_LIST.finditer(answer)
    )


def _tc29_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "run_code":
        return _noise({"stdout": "[0, 1, 4, 9, 16]", "stderr": "", "exit_code": 0}, "run_code")
    if call.name == "web_search":
        return _noise(
            {"results": [{"snippet": "List comprehension in Python creates a new list."}]},
            "web_search",
        )
    return _generic_tool_fallback(call)


def _tc29_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'What does this Python code do: [x**2 for x in range(5)]?'

    Should answer directly. Partial if it uses run_code. Fail if web_search.
    """
    used_run_code = any(c.name == "run_code" for c in state.tool_calls)
    used_web = any(c.name == "web_search" for c in state.tool_calls)
    answer = state.final_answer

    # "It does not return 0, 1, 4, 9, 16" states no result, and must not unlock
    # the worked-example exemption either.
    correct_output = _answer_affirms_pattern(answer, _TC29_OUTPUT)
    low_answer = _strip_markdown_emphasis(answer).lower()
    # Naming the construct is not an explanation of what it computes. The
    # answer has to say the values are squared, and not deny it ("it doesn't
    # square anything, it cubes"). Restating `x**2` counts as saying so only
    # next to the construct's name: "It applies x**2 to each item." just
    # echoes the question.
    names_comprehension = "list comprehension" in low_answer
    explains_comprehension = (
        _answer_affirms_pattern(answer, _TC29_SQUARING)
        and (
            names_comprehension
            or bool(
                re.search(
                    r"\b(?:each|every|all)\s+(?:the\s+)?(?:numbers?|integers?|values?|elements?|items?)\b",
                    low_answer,
                )
            )
        )
    ) or (
        (names_comprehension and _answer_affirms_pattern(answer, _TC29_POWER_FORM))
        or _answer_affirms_pattern(answer, _TC29_SQUARES_OF_RANGE)
    )
    wrong_list = _tc29_states_wrong_list(answer, allow_examples=correct_output)

    if used_web:
        return _fail("Used web_search for a basic Python question.")

    if any(c.name != "run_code" for c in state.tool_calls):
        return _fail("Used an unrelated tool for a code explanation that needed no tools.")

    if used_run_code:
        if correct_output and not wrong_list:
            return _partial("Got the right answer but unnecessarily executed the code.")
        return _fail("Executed the code but still gave wrong output.")

    if wrong_list:
        return _fail("Stated a result list other than [0, 1, 4, 9, 16].")
    if correct_output or explains_comprehension:
        return _pass("Correctly explained the code without using any tools.")
    return _fail("Did not explain the code correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-29",
    title="Explain Without Executing",
    category=Category.J,
    user_message="What does this Python code do: [x**2 for x in range(5)]?",
    description="Should explain directly without executing the code.",
    handle_tool_call=_tc29_handle,
    evaluate=_tc29_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it explains [0,1,4,9,16] directly without tools.",
    "Fail if it web-searches for a basic Python question or states a different result list.",
)
