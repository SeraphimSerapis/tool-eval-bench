"""TC-29 — Explain Without Executing."""

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
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
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


def _tc29_example_spans(answer: str) -> list[tuple[int, int]]:
    """Spans of worked examples on input other than range(5).

    Only an example that squares its own input is exempt, so a cubed example
    ("[1, 2, 3] becomes [1, 8, 27]") still states a wrong list.
    """
    spans = []
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
        prefix = answer[: match.start()]
        sentence_start = max((end.end() for end in _TC29_SENTENCE_END.finditer(prefix)), default=0)
        if _TC29_EXAMPLE_MARKER.search(prefix[sentence_start:]):
            spans.append(match.span())
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

    correct_output = bool(_TC29_OUTPUT.search(answer))
    low_answer = answer.lower()
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

    if used_run_code:
        if correct_output and not wrong_list:
            return _partial("Got the right answer but unnecessarily executed the code.")
        return _fail("Executed the code but still gave wrong output.")

    if state.tool_calls:
        return _fail("Used an unrelated tool for a code explanation that needed no tools.")

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
