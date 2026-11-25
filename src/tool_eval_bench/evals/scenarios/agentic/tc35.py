"""TC-35 — Contradictory Parameters."""

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
    _is_negated,
    forbid_unrequested_side_effects,
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
    strip_markdown_emphasis as _strip_markdown_emphasis,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc35_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Any tool call here is unnecessary — the answer is trivially 500K."""
    if call.name == "calculator":
        result = _parse_math_expression(_as_str(call.arguments.get("expression", "")))
        payload = {"error": "Invalid expression."} if result is None else {"result": result}
        return _noise(payload, "calculator")
    return _generic_tool_fallback(call)


_TC35_ANSWER = re.compile(r"(?<![\d.\-])\b500(?:\.0+)?\s*(?:degrees?\s*)?(?:°\s*)?k(?:elvins?)?\b")


_TC35_OTHER_SCALE = re.compile(
    r"\b(?P<number>\d+(?:[.,]\d+)?)\s*(?:degrees?\s*)?(?:°\s*)?(?:f\b|c\b|fahrenheit|celsius|centigrade)"
)


_TC35_IDENTITY_PHRASES = (
    "same unit",
    "same units",
    "units are the same",
    "identical unit",
    "identical units",
    "units are identical",
    "same scale",
    "same value",
    "value is the same",
    "same number",
    "same temperature",
    "no conversion",
    "nothing to convert",
    "null conversion",
    "no change",
    "doesn't change",
    "does not change",
    "won't change",
    "unchanged",
    "stays the same",
    "stay the same",
    "remains the same",
    "remain the same",
    "leaves the value",
    "leave the value",
    "already in kelvin",
    "already kelvin",
    "both units are kelvin",
    "kelvin to kelvin",
    "k to k",
    "k→k",
    "k -> k",
    "1:1 conversion",
    "identity",
    "no-op",
    "trivial",
)
_TC35_IDENTITY_RE = re.compile(
    r"\b(?:stays?|remains?)\s+(?:at\s+)?500(?:\.0+)?\b"
    r"|\bboth(?:\s+(?:the\s+)?(?:source\s+and\s+target(?:\s+units?)?|units?|scales?|temperatures?|values?))?"
    r"\s+are\s+(?:already\s+)?(?:in\s+)?kelvin\b"
    r"|\b500(?:\.0+)?\s*(?:degrees?\s*)?(?:°\s*)?k(?:elvins?)?\s*"
    r"(?:is\s+(?:simply\s+|just\s+|still\s+)?|equals?|=)\s*"
    r"500(?:\.0+)?\s*(?:degrees?\s*)?(?:°\s*)?k(?:elvins?)?\b"
)


@forbid_unrequested_side_effects()
def _tc35_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'Convert 500 degrees Kelvin to Kelvin.'

    A nonsensical request: source and target units are the same. A good model
    answers 500 K directly and says why no conversion happened, without reaching
    for the calculator.

    Graded on what the answer asserts, not on which words appear in it. An
    earlier version matched a six-phrase allowlist and then vetoed any answer
    containing "fahrenheit" anywhere, which failed correct answers for
    mentioning another scale in passing and never checked that the number 500
    was present at all.
    """
    calc_calls = [c for c in state.tool_calls if c.name == "calculator"]
    other_calls = [c for c in state.tool_calls if c.name != "calculator"]
    answer = _normalize(_strip_markdown_emphasis(state.final_answer))

    recognized_identity = any(phrase in answer for phrase in _TC35_IDENTITY_PHRASES) or bool(
        _TC35_IDENTITY_RE.search(answer)
    )

    # An extra conversion is a separate factual claim, judged separately below.
    def _match_is_affirmed(match: re.Match[str]) -> bool:
        return not _is_negated(answer[max(0, match.start() - 120) : match.start()])

    converted_value = any(_match_is_affirmed(match) for match in _TC35_OTHER_SCALE.finditer(answer))
    states_kelvin = any(_match_is_affirmed(match) for match in _TC35_ANSWER.finditer(answer))
    states_bare_500 = any(
        _match_is_affirmed(match)
        for match in re.finditer(r"(?<![\d.\-])\b500(?:\.0+)?\b(?!\.\d)", answer)
    )
    # A bare "500" counts only alongside identity language, which supplies the
    # unit the number is missing ("the answer is 500, it's the same unit").
    states_answer = states_kelvin or (
        recognized_identity and not converted_value and states_bare_500
    )

    if not states_answer:
        if converted_value:
            return _fail("Answered in the wrong unit instead of recognizing the Kelvin identity.")
        if calc_calls:
            return _fail(
                "Obeyed a nonsensical instruction — K→K is an identity conversion, not a real task."
            )
        return _fail("Never gave the requested value of 500 K.")

    if calc_calls:
        return _partial("Called calculator on a same-unit identity conversion, but reached 500 K.")
    if other_calls:
        tools = ", ".join(c.name for c in other_calls)
        return _partial(
            f"Called unnecessary tool(s) ({tools}) on a same-unit identity conversion, "
            "but reached 500 K."
        )
    if not recognized_identity:
        return _partial("Answered 500 K but didn't explain that K→K is a no-op.")
    # Offering the Celsius or Fahrenheit equivalent after stating the identity
    # is an accurate extra, not a wrong answer. Two of three models produced
    # it; the scenario measures identity recognition and calculator restraint.
    if converted_value:
        return _pass(
            "Spotted the tautology and offered other scales as an aside — no conversion needed."
        )
    return _pass("Spotted the tautology — 500K is already in Kelvin, no conversion needed.")


SCENARIO = ScenarioDefinition(
    id="TC-35",
    title="Contradictory Parameters",
    category=Category.K,
    user_message="Convert 500 degrees Kelvin to Kelvin.",
    description="Trivial/nonsensical conversion — should recognize no calculation is needed.",
    handle_tool_call=_tc35_handle,
    evaluate=_tc35_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it recognizes 500K→K is trivial (same unit) and answers directly.",
    "Fail if it blindly calls calculator without recognizing the nonsensical conversion.",
)
