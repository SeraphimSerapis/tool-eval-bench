"""TC-68 — Schema Violation Resistance."""

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
    as_str,
    generic_tool_fallback,
    normalize,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.scenarios.structured._shared import (
    _schema_text,
)

_TC68_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_status",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "blocked"],
                },
                "assignee": {"type": "string"},
            },
            "required": ["task_id", "status", "assignee"],
            "additionalProperties": False,
        },
    },
}


def _tc68_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return generic_tool_fallback(call)


def _tc68_eval(state: ScenarioState) -> ScenarioEvaluation:
    answer = state.final_answer.strip()

    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", answer, re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _fail("Output is not valid JSON.")

    if not isinstance(data, dict):
        return _fail("Output is not a JSON object.")

    required = {"task_id", "status", "assignee"}
    missing = required - set(data.keys())
    if missing:
        return _partial(f"Missing required fields: {', '.join(sorted(missing))}.")

    valid_statuses = {"pending", "in_progress", "completed", "blocked"}
    status_val = data.get("status")
    if not isinstance(status_val, str) or status_val not in valid_statuses:
        return _partial(f"Status '{status_val}' is not a valid enum value.")

    # Check for extra fields — the user asked for them, but schema forbids them
    extra = set(data.keys()) - required
    if extra:
        return _partial(
            f"Included extra fields ({', '.join(sorted(extra))}) that violate "
            "additionalProperties: false."
        )

    if not isinstance(data.get("task_id"), str) or not isinstance(data.get("assignee"), str):
        return _partial("task_id and assignee must be strings.")
    if (
        data.get("task_id") != "PROJ-127"
        or data.get("status") != "in_progress"
        or normalize(as_str(data.get("assignee"))) not in {"me", "myself", "current user"}
    ):
        return _partial("Structured task status does not match the requested task update.")

    # An unnecessary tool call degrades the result but does not invalidate a
    # schema-compliant answer; the task is JSON composition, not discovery.
    if len(state.tool_calls) > 0:
        return _partial("Produced schema-compliant JSON but called tools when none were needed.")
    return _pass(
        "Produced schema-compliant JSON without the forbidden extra fields, "
        "despite the user requesting them."
    )


SCENARIO = ScenarioDefinition(
    id="TC-68",
    title="Schema Violation Resistance",
    category=Category.O,
    user_message=(
        "Create a task status update for task PROJ-127: it's in progress, "
        "assigned to me. Also include the priority level, due date, "
        "and estimated hours remaining. Output as JSON matching this schema.\n\n"
        f"Schema:\n```json\n{_schema_text(_TC68_SCHEMA)}\n```"
    ),
    description=(
        "Resist user pressure to add extra fields that violate additionalProperties: false."
    ),
    handle_tool_call=_tc68_handle,
    evaluate=_tc68_eval,
    # NOTE: intentionally NO response_format_override here.
    # TC-68 tests whether the MODEL respects additionalProperties: false
    # on its own.  If we sent the schema via response_format, the SERVER
    # would enforce the constraint, making the test trivially passable.
    difficulty=4,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it produces JSON with ONLY the schema-allowed fields, despite user asking for more.",
    "Fail if it adds extra fields violating additionalProperties: false.",
)
