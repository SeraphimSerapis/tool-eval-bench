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
    has_explicit_tool_error,
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


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Any,
) -> bool:
    """Validate a payload when a trace records one, preserving old fixtures.

    The original synthetic scenario tests record calls without result records.
    Runtime traces always have a result, so an observed payload must satisfy the
    scenario-specific contract instead of allowing the model to fabricate it.
    """
    results = matching_tool_results(state, call)
    if not results:
        return True
    for result in results:
        if not isinstance(result.result, dict):
            continue
        try:
            if predicate(result.result):
                return True
        except (AttributeError, TypeError, KeyError):
            continue
    return False


def _failed_result_if_present(state: ScenarioState, call: ToolCallRecord) -> bool:
    """Accept a simulated failure only when no result or an explicit error exists."""
    results = matching_tool_results(state, call)
    return not results or has_explicit_tool_error(state, call)


_PRICES = {"ACME": 100.0, "BETA": 80.0, "CYGN": 95.0}


_ORIGINAL_EVENT = {
    "event_id": "release_review",
    "date": "2026-03-24",
    "time": "11:00",
    "attendees": ["ana@company.com", "ben@company.com"],
}


_ROOMS = [
    {"room_id": "berlin_3a", "office": "Berlin", "capacity": 3, "available": True},
    {"room_id": "berlin_5b", "office": "Berlin", "capacity": 5, "available": True},
    {"room_id": "munich_3a", "office": "Munich", "capacity": 3, "available": True},
]
