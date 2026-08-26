"""TC-64 — Simple Schema Compliance."""

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
    generic_tool_fallback,
    includes_text,
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

_TC64_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "movie_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
                "rating": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                },
                "genre": {
                    "type": "string",
                    "enum": ["action", "comedy", "drama", "horror", "sci-fi", "thriller"],
                },
                "summary": {"type": "string"},
            },
            "required": ["title", "year", "rating", "genre", "summary"],
            "additionalProperties": False,
        },
    },
}


def _tc64_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return generic_tool_fallback(call)


def _tc64_eval(state: ScenarioState) -> ScenarioEvaluation:
    # Model should produce valid JSON matching the schema — no tools needed
    if len(state.tool_calls) > 0:
        return _fail("Called tools when none were needed — this is a structured output task.")

    answer = state.final_answer.strip()

    # Try to extract JSON from code fences if present
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", answer, re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _fail("Output is not valid JSON.")

    if not isinstance(data, dict):
        return _fail("Output is not a JSON object.")

    required = {"title", "year", "rating", "genre", "summary"}
    missing = required - set(data.keys())
    if missing:
        return _partial(f"Valid JSON but missing required fields: {', '.join(sorted(missing))}.")

    # Validate types
    errors: list[str] = []
    if not isinstance(data.get("title"), str) or not data["title"]:
        errors.append("title must be a non-empty string")
    if type(data.get("year")) is not int:
        errors.append("year must be an integer")
    if type(data.get("rating")) not in (int, float):
        errors.append("rating must be a number")
    elif not (0 <= data["rating"] <= 10):
        errors.append("rating must be between 0 and 10")
    valid_genres = {"action", "comedy", "drama", "horror", "sci-fi", "thriller"}
    genre_val = data.get("genre")
    if not isinstance(genre_val, str) or genre_val not in valid_genres:
        errors.append(f"genre must be one of {sorted(valid_genres)}")
    if not isinstance(data.get("summary"), str) or not data["summary"]:
        errors.append("summary must be a non-empty string")

    # Check for extra fields (additionalProperties: false)
    extra = set(data.keys()) - required
    if extra:
        errors.append(f"extra fields not allowed: {', '.join(sorted(extra))}")

    if errors:
        return _partial(f"Valid JSON with schema violations: {'; '.join(errors)}.")

    # Check that it's about The Matrix (the requested movie)
    if not includes_text(data.get("title", ""), "matrix"):
        return _partial("Valid schema-compliant JSON but not about the requested movie.")

    return _pass("Produced valid, schema-compliant JSON for the requested movie review.")


SCENARIO = ScenarioDefinition(
    id="TC-64",
    title="Simple Schema Compliance",
    category=Category.O,
    user_message=(
        "Write a review of the movie 'The Matrix'. "
        "Output ONLY valid JSON matching this schema — no extra text.\n\n"
        f"Schema:\n```json\n{_schema_text(_TC64_SCHEMA)}\n```"
    ),
    description="Produce valid JSON matching a strict schema without tool calls.",
    handle_tool_call=_tc64_handle,
    evaluate=_tc64_eval,
    response_format_override=_TC64_SCHEMA,
    tools_override=[],
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it produces valid JSON matching the movie_review schema with correct types and enum values.",
    "Fail if it produces non-JSON output, uses tools, or violates the schema.",
)
