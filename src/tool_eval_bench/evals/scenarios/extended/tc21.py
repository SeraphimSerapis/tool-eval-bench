"""TC-21: Constraint Validation."""

from __future__ import annotations

import json
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

# Each rule covers one invalid field and one valid decoy, so flagging every
# field costs as much as it gains. The decoys sit on the edges the rules
# describe: the age limit itself, separators the phone rule strips, a leap
# day, and zero.
_PAYLOAD: dict[str, Any] = {
    "email": "john@.com",
    "alt_email": "ops@mail.example.org",
    "age": 151,
    "guardian_age": 150,
    "phone": "555-12",
    "office_phone": "(555) 123-4567",
    "start_date": "2023-02-29",
    "renewal_date": "2024-02-29",
    "amount": -50,
    "discount": 0,
}

#: The rule each invalid field breaks.
_INVALID = {
    "email": 1,
    "age": 2,
    "phone": 3,
    "start_date": 4,
    "amount": 5,
}
_VALID = frozenset(_PAYLOAD) - frozenset(_INVALID)

# A field name, also written with a space or hyphen or run together
# ("guardian age", "guardianAge"). Longest names come first so "alt_email" is
# never read as "email".
_FIELD = re.compile(
    r"(?<![a-z0-9_])("
    + "|".join(
        name.replace("_", "[ _-]?")
        for name in sorted(_PAYLOAD, key=lambda name: (-len(name), name))
    )
    + r")(?![a-z0-9_])",
    re.IGNORECASE,
)
_RULE = re.compile(r"\b(?:r|rule\s*)([1-5])\b", re.IGNORECASE)
_ISSUES_MARKER = re.compile(r"^[\W_]*issues[\W_]*:", re.IGNORECASE | re.MULTILINE)


def _tc21_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """No tools needed: tests direct reasoning."""
    return _generic_tool_fallback(call)


def _canonical_field(text: str) -> str:
    compact = re.sub(r"[ _-]", "", text.lower())
    return next(name for name in _PAYLOAD if name.replace("_", "") == compact)


def _flagged_rules(answer: str) -> dict[str, set[int]]:
    """Map each field the answer flags to the rule ids it cites.

    Only the section after the last ``ISSUES:`` line is read, so the model may
    reason about valid fields above it. Without that line the whole answer is
    read. A line flags the first field it names with the first rule id it
    cites; a line without a rule id flags nothing.
    """
    markers = list(_ISSUES_MARKER.finditer(answer))
    section = answer[markers[-1].end() :] if markers else answer
    flagged: dict[str, set[int]] = {}
    for line in section.splitlines():
        field = _FIELD.search(line)
        rule = _RULE.search(line)
        if field and rule:
            flagged.setdefault(_canonical_field(field.group(1)), set()).add(int(rule.group(1)))
    return flagged


def _tc21_eval(state: ScenarioState) -> ScenarioEvaluation:
    """Grade which fields the answer flags and which rule it cites for each.

    An invalid field counts only when every rule cited for it is the right
    one, so naming the field with a wrong reason earns nothing. Each valid
    field flagged costs one. PASS needs all five and no false flags; PARTIAL
    needs a net of three.
    """
    if len(state.tool_calls) > 0:
        used_tools = ", ".join(c.name for c in state.tool_calls)
        return _fail(f"Used tools ({used_tools}) when direct analysis was appropriate.")

    flagged = _flagged_rules(state.final_answer)
    correct = sorted(f for f, rule in _INVALID.items() if flagged.get(f) == {rule})
    wrong_rule = sorted(f for f in _INVALID if f in flagged and f not in correct)
    missed = sorted(f for f in _INVALID if f not in flagged)
    false_flags = sorted(f for f in _VALID if f in flagged)

    details = []
    if missed:
        details.append(f"missed {', '.join(missed)}")
    if wrong_rule:
        details.append(f"wrong rule for {', '.join(wrong_rule)}")
    if false_flags:
        details.append(f"flagged valid {', '.join(false_flags)}")
    summary = f"{len(correct)}/5 invalid fields with the right rule"
    if details:
        summary += f"; {'; '.join(details)}"

    if len(correct) == 5 and not false_flags:
        return _pass("Identified all 5 invalid fields and their rules without using tools.")
    if len(correct) - len(false_flags) >= 3:
        return _partial(f"{summary}.")
    return _fail(f"{summary}.")


SCENARIO = ScenarioDefinition(
    id="TC-21",
    title="Constraint Validation",
    category=Category.G,
    user_message=(
        "Check this API payload without tools. Rules:\n"
        "R1 (email, alt_email): the address needs a nonempty domain name before its "
        "top-level domain, as in user@example.com.\n"
        "R2 (age, guardian_age): an integer from 0 through 150.\n"
        "R3 (phone, office_phone): exactly 10 digits after removing spaces, dashes, dots, "
        "and parentheses.\n"
        "R4 (start_date, renewal_date): a real Gregorian date in YYYY-MM-DD format.\n"
        "R5 (amount, discount): a number that is not negative.\n"
        f"Payload: {json.dumps(_PAYLOAD)}\n"
        "Some fields are valid. End your answer with a line reading ISSUES: and then one "
        "line per invalid field in the form `field: rule id`, for example `name: R9`. "
        "List nothing else in that section."
    ),
    description="Flag the five invalid fields and the rule each breaks, without tools.",
    handle_tool_call=_tc21_handle,
    evaluate=_tc21_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it flags all five invalid fields with the right rule and no valid field, "
    "without tools; partial when right flags minus false flags is three or four.",
    "Fail if it uses tools, cites wrong rules, or flags valid fields as often as invalid ones.",
)
