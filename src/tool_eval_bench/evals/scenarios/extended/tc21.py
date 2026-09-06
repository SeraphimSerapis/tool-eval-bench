"""TC-21 — Constraint Validation."""

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


def _tc21_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """No tools needed — tests direct reasoning."""
    return _generic_tool_fallback(call)


# Any assertion that something is wrong. Only consulted for a clause that
# names the field *and* quotes the offending value, which is the strongest
# evidence available that the model diagnosed this field: at that point the
# particular verb it reached for ("exceeds the maximum", "is missing a domain
# label", "has only 5 digits") is style, not correctness.
_TC21_PROBLEM = (
    r"(?:invalid|malformed|bad\b|wrong|incorrect|not\s+(?:a\s+)?(?:valid|allowed|permitted)|"
    r"isn't\s+valid|error|issue|problem|violat|fails?\b|exceed|out\s+of\s+range|"
    r"too\s+(?:high|low|few|short|long|large|small|many)|must\s+be|should\s+be|"
    r"cannot|can't|do(?:es)?\s+not\s+exist|don't\s+exist|doesn't\s+exist|missing|"
    r"only\s+\d+|negative|below\s+zero|impossible|not\s+\d+\s+digits)"
)


def _tc21_asserts_issue(answer: str, field: str, issue_pattern: str, value: str = "") -> bool:
    """Find an asserted validation issue, not a quoted or negated mention."""
    # Split on sentence punctuation only when it actually ends a sentence: the
    # offending values include "john@.com", and splitting inside it hid the
    # diagnosis from every check that looks for the value.
    for clause in re.split(r"(?<=[.!?;])\s+|\n", answer):
        if not re.search(rf"\b{field}\b", clause, re.IGNORECASE):
            continue
        effective = issue_pattern
        if value and re.search(re.escape(value), clause, re.IGNORECASE):
            effective = f"(?:{issue_pattern}|{_TC21_PROBLEM})"
        match = re.search(effective, clause, re.IGNORECASE)
        if not match:
            continue
        before = clause[: match.start()]
        negation = re.search(r"\b(?:not|never|no)\b(?:\s+\w+){0,4}\s*$", before, re.IGNORECASE)
        if negation and not re.search(
            r"(?:\b(?:valid|correct|acceptable)\b|\bwithin\s+range\b)",
            negation.group(),
            re.IGNORECASE,
        ):
            continue
        positive_quality_claim = False
        for quality in re.finditer(
            r"(?:\b(?:valid|correct|acceptable)\b|\bwithin\s+range\b)",
            clause,
            re.IGNORECASE,
        ):
            quality_before = clause[: quality.start()]
            if not re.search(
                r"(?:\b(?:not|never|no)\b(?:\s+\w+){0,2}|n't)\s*$",
                quality_before,
                re.IGNORECASE,
            ):
                positive_quality_claim = True
                break
        if positive_quality_claim:
            # A clause such as "email is valid but malformed" is
            # contradictory.  A negated quality claim, such as
            # "email is not valid because it is malformed", supports the
            # issue instead of suppressing it.
            continue
        return True
    return False


def _tc21_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User provides a JSON payload with 5 deliberate errors, asks the model to find them all.

    The model should NOT use any tools — just analyze the data directly.
    Expected errors: invalid email, age > 150, phone not 10 digits, invalid date, negative amount.
    """
    if len(state.tool_calls) > 0:
        used_tools = ", ".join(c.name for c in state.tool_calls)
        return _fail(f"Used tools ({used_tools}) when direct analysis was appropriate.")

    answer = state.final_answer.lower()
    error_checks = [
        _tc21_asserts_issue(
            answer, "email", r"(?:invalid|malformed|bad|missing|incomplete)", "john@.com"
        ),
        _tc21_asserts_issue(
            answer,
            "age",
            r"(?:too\s+high|out\s+of\s+range|over\s+150|exceed|above\s+(?:the\s+)?max|"
            r"greater\s+than\s+150|implausible|unrealistic)",
            "200",
        ),
        _tc21_asserts_issue(
            answer,
            "phone",
            r"(?:invalid|wrong|bad|too\s+few|fewer\s+than|short|format|not\s+(?:10|ten)|"
            r"only\s+\d+|incomplete|truncated)",
            "555-12",
        ),
        _tc21_asserts_issue(
            answer,
            "date",
            r"(?:invalid|impossible|month\s+13|day\s+45|bad\s+format|do(?:es)?\s+not\s+exist)",
            "2020-13-45",
        ),
        _tc21_asserts_issue(
            answer,
            "amount",
            r"(?:negative|below\s+zero|less\s+than\s+zero|must\s+be\s+positive)",
            "-50",
        ),
    ]
    found = sum(error_checks)
    if found >= 4:
        return _pass(f"Identified {found}/5 validation errors without using tools.")
    if found >= 3:
        return _partial(f"Found {found}/5 errors. Missed some validation issues.")
    return _fail(f"Only found {found}/5 validation errors.")


SCENARIO = ScenarioDefinition(
    id="TC-21",
    title="Constraint Validation",
    category=Category.G,
    user_message=(
        "Check this API payload for errors. List all validation issues:\n"
        '{"email": "john@.com", "age": 200, "phone": "555-12", '
        '"date": "2020-13-45", "amount": -50}'
    ),
    description="Find all 5 validation errors without resorting to tools.",
    handle_tool_call=_tc21_handle,
    evaluate=_tc21_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it finds 4+ of the 5 validation errors without tools.",
    "Fail if it uses tools or misses most errors.",
)
