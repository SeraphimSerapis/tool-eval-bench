"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    full_assistant_transcript,
    matching_tool_results,
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


def _claims_success(state: ScenarioState) -> bool:
    transcript = full_assistant_transcript(state).lower()
    return any(
        phrase in transcript
        for phrase in (
            "successfully created",
            "successfully provisioned",
            "has been created",
            "has been provisioned",
            "successfully updated",
            "has been updated",
        )
    )
