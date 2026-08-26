"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results,
    result_is_usable_if_present,
)


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Any,
) -> bool:
    """Validate explicit tool results without breaking synthetic traces."""
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results:
        # A stable call ID is authoritative.  Only the nameless records used
        # by older synthetic tests may be ignored.  A known result under the
        # wrong call ID must not be reassigned to this call by name matching.
        if any(
            result.name not in {"", "unknown"} and result.name != call.name
            for result in exact_results
        ):
            return False
        results = [result for result in exact_results if result.name == call.name]
    elif any(
        result.call_id in known_call_ids and result.name == call.name
        for result in state.tool_results
    ):
        return False
    else:
        results = matching_tool_results(state, call)
    if not results:
        return True
    return result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


def _extract_json_answer(answer: str) -> str:
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", answer, re.DOTALL)
    return json_match.group(1).strip() if json_match else answer.strip()


def _schema_text(schema_dict: dict) -> str:
    """Extract the JSON schema body from a response_format dict and format it
    as compact text suitable for embedding in a user message.  This ensures
    the model sees the schema even if the backend ignores response_format."""
    js = schema_dict.get("json_schema", {})
    body = js.get("schema", {})
    return json.dumps(body, indent=2)
