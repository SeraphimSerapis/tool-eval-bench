"""Category O — Structured Output scenarios (TC-64 to TC-69).

Tests whether models can produce valid JSON conforming to a given schema,
chain structured output with tool calls, handle nested schemas, and
refuse to produce output that violates schema constraints.

These scenarios use ``response_format`` with ``json_schema`` type to test
structured generation — a critical capability for agentic pipelines where
downstream systems expect machine-parseable output.
"""

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
    fail_eval as _fail,
    first_call,
    generic_tool_fallback,
    has_tool_call,
    includes_text,
    normalize,
    partial_eval as _partial,
    pass_eval as _pass,
    with_noise as _noise,
)


def _schema_text(schema_dict: dict) -> str:
    """Extract the JSON schema body from a response_format dict and format it
    as compact text suitable for embedding in a user message.  This ensures
    the model sees the schema even if the backend ignores response_format."""
    js = schema_dict.get("json_schema", {})
    body = js.get("schema", {})
    return json.dumps(body, indent=2)


# ===================================================================
# TC-64: Simple Schema Compliance
# ===================================================================
# The model must produce output conforming to a strict JSON schema
# without any tool calls — pure structured generation.

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
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.DOTALL)
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
    if not isinstance(data.get("year"), int):
        errors.append("year must be an integer")
    if not isinstance(data.get("rating"), (int, float)):
        errors.append("rating must be a number")
    elif not (0 <= data["rating"] <= 10):
        errors.append("rating must be between 0 and 10")
    valid_genres = {"action", "comedy", "drama", "horror", "sci-fi", "thriller"}
    if data.get("genre") not in valid_genres:
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


# ===================================================================
# TC-65: Tool Call → Structured Output
# ===================================================================
# The model must call a tool first, then format the result as structured JSON.

_TC65_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature_celsius": {"type": "number"},
                "condition": {"type": "string"},
                "recommendation": {"type": "string"},
            },
            "required": ["location", "temperature_celsius", "condition", "recommendation"],
            "additionalProperties": False,
        },
    },
}


def _tc65_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise({
            "location": "Tokyo",
            "temperature": 28,
            "units": "celsius",
            "condition": "Sunny",
            "humidity": 65,
        }, "get_weather")
    return generic_tool_fallback(call)


def _tc65_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = first_call(state, "get_weather")
    if not weather:
        return _fail("Did not call get_weather before producing structured output.")

    answer = state.final_answer.strip()

    # Extract JSON from fences if present
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called get_weather correctly but final output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Called get_weather but output is not a JSON object.")

    required = {"location", "temperature_celsius", "condition", "recommendation"}
    missing = required - set(data.keys())
    if missing:
        return _partial(f"Valid JSON but missing: {', '.join(sorted(missing))}.")

    # Verify the data comes from the tool result, not hallucinated
    if data.get("temperature_celsius") != 28:
        return _partial("Schema-compliant but temperature doesn't match tool result (28°C).")

    if not includes_text(data.get("location", ""), "tokyo"):
        return _partial("Schema-compliant but location doesn't match tool result.")

    return _pass("Called get_weather, then produced schema-compliant JSON with correct data.")


# ===================================================================
# TC-66: Nested Schema (Array of Objects)
# ===================================================================

_TC66_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "contact_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "total": {"type": "integer"},
                "contacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "department": {"type": "string"},
                        },
                        "required": ["name", "email", "department"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["query", "total", "contacts"],
            "additionalProperties": False,
        },
    },
}


def _tc66_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return _noise({
            "results": [
                {"name": "Alice Zhang", "email": "alice.zhang@company.com", "department": "Engineering"},
                {"name": "Bob Martinez", "email": "bob.martinez@company.com", "department": "Design"},
                {"name": "Carol Singh", "email": "carol.singh@company.com", "department": "Engineering"},
            ],
        }, "get_contacts")
    return generic_tool_fallback(call)


def _tc66_eval(state: ScenarioState) -> ScenarioEvaluation:
    contacts_call = first_call(state, "get_contacts")
    if not contacts_call:
        return _fail("Did not call get_contacts.")

    answer = state.final_answer.strip()

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called get_contacts correctly but output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Output is not a JSON object.")

    # Check top-level structure
    if not all(k in data for k in ("query", "total", "contacts")):
        return _partial("Missing required top-level fields.")

    contacts = data.get("contacts", [])
    if not isinstance(contacts, list):
        return _partial("'contacts' is not an array.")

    if len(contacts) < 3:
        return _partial(f"Expected 3 contacts, got {len(contacts)}.")

    # Validate each contact has required fields
    for i, c in enumerate(contacts):
        if not isinstance(c, dict):
            return _partial(f"Contact {i} is not an object.")
        if not all(k in c for k in ("name", "email", "department")):
            return _partial(f"Contact {i} is missing required fields.")

    # Verify total matches array length
    if data.get("total") != len(contacts):
        return _partial("'total' doesn't match contacts array length.")

    # Verify data integrity — contacts should come from tool result
    names = {c.get("name", "").lower() for c in contacts}
    expected_names = {"alice zhang", "bob martinez", "carol singh"}
    if not expected_names.issubset(names):
        return _partial("Contacts don't match tool result data.")

    return _pass("Produced schema-compliant nested JSON with correct contact data from tool.")


# ===================================================================
# TC-67: Schema with Enum Constraints + Tool Chaining
# ===================================================================

_TC67_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "stock_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"type": "string"},
                "signal": {
                    "type": "string",
                    "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["ticker", "price", "currency", "signal", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _tc67_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return _noise({
            "ticker": "NVDA",
            "price": 892.50,
            "currency": "USD",
            "change": "+15.30",
            "change_percent": "+1.74%",
            "volume": "42.3M",
        }, "get_stock_price")
    if call.name == "web_search":
        return _noise({
            "results": [{
                "snippet": "NVIDIA (NVDA) reported record Q4 revenue of $22.1B, "
                           "up 265% year-over-year, driven by data center AI demand. "
                           "Analysts maintain buy ratings with average price target of $950.",
            }],
        }, "web_search")
    return generic_tool_fallback(call)


def _tc67_eval(state: ScenarioState) -> ScenarioEvaluation:
    stock = first_call(state, "get_stock_price")
    if not stock:
        return _fail("Did not call get_stock_price.")

    answer = state.final_answer.strip()

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called get_stock_price but output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Output is not a JSON object.")

    required = {"ticker", "price", "currency", "signal", "reasoning"}
    missing = required - set(data.keys())
    if missing:
        return _partial(f"Valid JSON but missing: {', '.join(sorted(missing))}.")

    valid_signals = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    if data.get("signal") not in valid_signals:
        return _partial(f"Signal '{data.get('signal')}' is not a valid enum value.")

    # Verify data comes from tool
    if data.get("price") != 892.50:
        return _partial("Price doesn't match tool result (892.50).")

    if normalize(as_str(data.get("ticker"))) != "nvda":
        return _partial("Ticker doesn't match tool result.")

    if not isinstance(data.get("reasoning"), str) or len(data["reasoning"]) < 20:
        return _partial("Reasoning field is too short — should explain the signal.")

    return _pass("Produced schema-compliant analysis with correct enum signal and tool data.")


# ===================================================================
# TC-68: Schema Violation Resistance
# ===================================================================
# The user asks the model to include extra fields that aren't in the schema.
# A good model should refuse the extra fields and stay schema-compliant.

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
    if len(state.tool_calls) > 0:
        return _fail("Called tools when none were needed.")

    answer = state.final_answer.strip()

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.DOTALL)
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
    if data.get("status") not in valid_statuses:
        return _partial(f"Status '{data.get('status')}' is not a valid enum value.")

    # Check for extra fields — the user asked for them, but schema forbids them
    extra = set(data.keys()) - required
    if extra:
        return _partial(
            f"Included extra fields ({', '.join(sorted(extra))}) that violate "
            "additionalProperties: false."
        )

    return _pass(
        "Produced schema-compliant JSON without the forbidden extra fields, "
        "despite the user requesting them."
    )


# ===================================================================
# TC-69: Multi-Tool → Complex Structured Output
# ===================================================================
# The model must call multiple tools and synthesize results into a complex schema.

_TC69_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "daily_briefing",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "weather": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "temperature": {"type": "number"},
                        "condition": {"type": "string"},
                    },
                    "required": ["location", "temperature", "condition"],
                    "additionalProperties": False,
                },
                "market": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "price": {"type": "number"},
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "flat"],
                        },
                    },
                    "required": ["ticker", "price", "direction"],
                    "additionalProperties": False,
                },
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["date", "weather", "market", "action_items"],
            "additionalProperties": False,
        },
    },
}


def _tc69_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return _noise({
            "location": "San Francisco",
            "temperature": 18,
            "units": "celsius",
            "condition": "Foggy",
            "humidity": 85,
        }, "get_weather")
    if call.name == "get_stock_price":
        return _noise({
            "ticker": "AAPL",
            "price": 192.30,
            "currency": "USD",
            "change": "-2.15",
            "change_percent": "-1.11%",
        }, "get_stock_price")
    return generic_tool_fallback(call)


def _tc69_eval(state: ScenarioState) -> ScenarioEvaluation:
    weather = has_tool_call(state, "get_weather")
    stock = has_tool_call(state, "get_stock_price")

    if not weather or not stock:
        missing = []
        if not weather:
            missing.append("get_weather")
        if not stock:
            missing.append("get_stock_price")
        return _fail(f"Did not call required tools: {', '.join(missing)}.")

    answer = state.final_answer.strip()

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.DOTALL)
    if json_match:
        answer = json_match.group(1).strip()

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return _partial("Called both tools but output is not valid JSON.")

    if not isinstance(data, dict):
        return _partial("Output is not a JSON object.")

    # Check top-level
    required_top = {"date", "weather", "market", "action_items"}
    missing_top = required_top - set(data.keys())
    if missing_top:
        return _partial(f"Missing top-level fields: {', '.join(sorted(missing_top))}.")

    # Validate weather sub-object
    w = data.get("weather", {})
    if not isinstance(w, dict) or not all(k in w for k in ("location", "temperature", "condition")):
        return _partial("Weather sub-object is missing required fields.")

    # Validate market sub-object
    m = data.get("market", {})
    if not isinstance(m, dict) or not all(k in m for k in ("ticker", "price", "direction")):
        return _partial("Market sub-object is missing required fields.")

    valid_directions = {"up", "down", "flat"}
    if m.get("direction") not in valid_directions:
        return _partial(f"Market direction '{m.get('direction')}' is not a valid enum value.")

    # Validate action_items is array of strings
    actions = data.get("action_items", [])
    if not isinstance(actions, list) or not all(isinstance(a, str) for a in actions):
        return _partial("action_items must be an array of strings.")

    # Verify data integrity from tools
    if w.get("temperature") != 18:
        return _partial("Weather temperature doesn't match tool result (18°C).")

    if m.get("price") != 192.30:
        return _partial("Stock price doesn't match tool result (192.30).")

    # Direction should be "down" since change is negative
    if m.get("direction") != "down":
        return _partial("Market direction should be 'down' (stock dropped -1.11%).")

    return _pass(
        "Called both tools and produced schema-compliant nested JSON "
        "with correct data synthesis."
    )


# ===================================================================
# Scenario registry
# ===================================================================

STRUCTURED_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="TC-64", title="Simple Schema Compliance", category=Category.O,
        user_message=(
            "Write a review of the movie 'The Matrix'. "
            "Output ONLY valid JSON matching this schema — no extra text.\n\n"
            f"Schema:\n```json\n{_schema_text(_TC64_SCHEMA)}\n```"
        ),
        description="Produce valid JSON matching a strict schema without tool calls.",
        handle_tool_call=_tc64_handle, evaluate=_tc64_eval,
        response_format_override=_TC64_SCHEMA,
    ),
    ScenarioDefinition(
        id="TC-65", title="Tool → Structured Output", category=Category.O,
        user_message=(
            "Get the current weather in Tokyo and output it as JSON "
            "matching this schema. Include a recommendation for what to wear.\n\n"
            f"Schema:\n```json\n{_schema_text(_TC65_SCHEMA)}\n```"
        ),
        description="Call get_weather, then format the result as schema-compliant JSON.",
        handle_tool_call=_tc65_handle, evaluate=_tc65_eval,
        response_format_override=_TC65_SCHEMA,
    ),
    ScenarioDefinition(
        id="TC-66", title="Nested Schema (Array of Objects)", category=Category.O,
        user_message=(
            "Look up all engineering contacts and return the results "
            "as a JSON object matching this schema.\n\n"
            f"Schema:\n```json\n{_schema_text(_TC66_SCHEMA)}\n```"
        ),
        description="Call get_contacts and format as nested JSON with array of objects.",
        handle_tool_call=_tc66_handle, evaluate=_tc66_eval,
        response_format_override=_TC66_SCHEMA,
    ),
    ScenarioDefinition(
        id="TC-67", title="Enum Constraint + Analysis", category=Category.O,
        user_message=(
            "Get the current stock price for NVDA and produce a stock analysis "
            "as JSON matching this schema. Research recent news "
            "to inform your signal.\n\n"
            f"Schema:\n```json\n{_schema_text(_TC67_SCHEMA)}\n```"
        ),
        description="Call tools and produce JSON with enum-constrained signal field.",
        handle_tool_call=_tc67_handle, evaluate=_tc67_eval,
        response_format_override=_TC67_SCHEMA,
    ),
    ScenarioDefinition(
        id="TC-68", title="Schema Violation Resistance", category=Category.O,
        user_message=(
            "Create a task status update for task PROJ-127: it's in progress, "
            "assigned to me. Also include the priority level, due date, "
            "and estimated hours remaining. Output as JSON matching this schema.\n\n"
            f"Schema:\n```json\n{_schema_text(_TC68_SCHEMA)}\n```"
        ),
        description=(
            "Resist user pressure to add extra fields that violate "
            "additionalProperties: false."
        ),
        handle_tool_call=_tc68_handle, evaluate=_tc68_eval,
        # NOTE: intentionally NO response_format_override here.
        # TC-68 tests whether the MODEL respects additionalProperties: false
        # on its own.  If we sent the schema via response_format, the SERVER
        # would enforce the constraint, making the test trivially passable.
    ),
    ScenarioDefinition(
        id="TC-69", title="Multi-Tool → Complex Schema", category=Category.O,
        user_message=(
            "Create my daily briefing: check the weather in San Francisco "
            "and look up AAPL stock price. Output as JSON matching this schema "
            "with actionable items.\n\n"
            f"Schema:\n```json\n{_schema_text(_TC69_SCHEMA)}\n```"
        ),
        description="Call multiple tools and synthesize into complex nested schema.",
        handle_tool_call=_tc69_handle, evaluate=_tc69_eval,
        response_format_override=_TC69_SCHEMA,
    ),
]

STRUCTURED_DISPLAY_DETAILS: dict[str, ScenarioDisplayDetail] = {
    "TC-64": ScenarioDisplayDetail(
        "Pass if it produces valid JSON matching the movie_review schema with correct types and enum values.",
        "Fail if it produces non-JSON output, uses tools, or violates the schema.",
    ),
    "TC-65": ScenarioDisplayDetail(
        "Pass if it calls get_weather first, then outputs schema-compliant JSON with correct data.",
        "Fail if it skips the tool call or produces non-JSON output.",
    ),
    "TC-66": ScenarioDisplayDetail(
        "Pass if it calls get_contacts and produces nested JSON with array of contact objects.",
        "Fail if it doesn't call the tool or produces flat/incorrect structure.",
    ),
    "TC-67": ScenarioDisplayDetail(
        "Pass if it gets the stock price, researches news, and produces JSON with a valid enum signal.",
        "Fail if it skips tools or uses an invalid signal enum value.",
    ),
    "TC-68": ScenarioDisplayDetail(
        "Pass if it produces JSON with ONLY the schema-allowed fields, despite user asking for more.",
        "Fail if it adds extra fields violating additionalProperties: false.",
    ),
    "TC-69": ScenarioDisplayDetail(
        "Pass if it calls both tools and produces complex nested JSON with correct data synthesis.",
        "Fail if it misses a tool or produces invalid nested structure.",
    ),
}
