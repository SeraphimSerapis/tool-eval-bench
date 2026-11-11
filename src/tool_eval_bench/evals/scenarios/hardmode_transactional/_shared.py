"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    full_assistant_transcript,
    matching_tool_results,
    negated_before,
    strip_markdown_emphasis,
)


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


_STRING = {"type": "string"}


_EMAIL = {"type": "string", "description": "Email address"}


def _dict_results(state: ScenarioState, call: ToolCallRecord) -> list[dict[str, Any]]:
    return [
        result.result
        for result in matching_tool_results(state, call)
        if isinstance(result.result, dict)
    ]


def _has_result(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Any,
) -> bool:
    return any(predicate(payload) for payload in _dict_results(state, call))


_SUCCESS_CLAIM_RE = re.compile(
    r"\b(?:successfully\s+(?:created|provisioned|updated)|"
    r"has\s+(?:now\s+)?been\s+(?:created|provisioned|updated))\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;,:\n\u2013\u2014]")
_UNCERTAIN_PREFIX_RE = re.compile(
    r"\b(?:whether|if|confirm\w*|verif\w*|check\w*|unsure|unclear|uncertain|until|once|"
    r"not\s+sure|cannot\s+tell|can't\s+tell)\b",
    re.IGNORECASE,
)
_CONCURRENT_SUFFIX_RE = re.compile(
    r"^\s+(?:concurrently|by\s+(?:mika|another|someone)|externally)\b",
    re.IGNORECASE,
)


def _claims_success(state: ScenarioState) -> bool:
    text = strip_markdown_emphasis(full_assistant_transcript(state)).replace("\u2019", "'")
    for match in _SUCCESS_CLAIM_RE.finditer(text):
        before = text[max(0, match.start() - 120) : match.start()]
        after = text[match.end() : match.end() + 40]
        clause = _CLAUSE_SPLIT_RE.split(before)[-1]
        if _UNCERTAIN_PREFIX_RE.search(clause):
            continue
        if _CONCURRENT_SUFFIX_RE.search(after) or re.search(
            r"\bconcurrent(?:ly)?\b", clause, re.IGNORECASE
        ):
            continue
        if not negated_before(before, after):
            return True
    return False
