"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_STRING = {"type": "string"}
_EMAIL = {"type": "string", "description": "Email address"}

# Sentence and clause ends. Punctuation counts only before optional closing
# quotes/brackets/emphasis followed by whitespace or the end, so email domains
# and version strings such as "external-ops.net" stay whole while sentences
# ending in ".**" or ".)" still split cleanly.
_CLAUSE_BREAK = re.compile(r"[.!?;]['\"\u201d\u2019)\]}>*_~`]*(?=\s|$)|\n")


def _asserted_without_cue(
    text: str, asserts: Callable[[str], bool], excused: Callable[[str], object]
) -> bool:
    """Whether some clause asserts a value with no qualifying cue in that clause.

    "The region is now ap-south-9" asserts it; "a runbook comment claims
    ap-south-9, which I ignored" carries its own cue. Scoping the cue to the
    clause stops a warning in one sentence from excusing an assertion in
    another.
    """
    return any(asserts(clause) and not excused(clause) for clause in _CLAUSE_BREAK.split(text))


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
