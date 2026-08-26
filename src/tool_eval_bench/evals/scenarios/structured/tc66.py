"""TC-66 — Nested Schema (Array of Objects)."""

from __future__ import annotations

import json
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
    result_is_usable_if_present,
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
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.structured._shared import (
    _extract_json_answer,
    _result_matches_if_present,
    _schema_text,
)

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
        return _noise(
            {
                "results": [
                    {
                        "name": "Alice Zhang",
                        "email": "alice.zhang@company.com",
                        "department": "Engineering",
                    },
                    {
                        "name": "Carol Singh",
                        "email": "carol.singh@company.com",
                        "department": "Engineering",
                    },
                ],
            },
            "get_contacts",
        )
    return generic_tool_fallback(call)


def _tc66_eval(state: ScenarioState) -> ScenarioEvaluation:
    def contacts_result_is_engineering(payload: Any) -> bool:
        expected = {
            ("Alice Zhang", "alice.zhang@company.com", "Engineering"),
            ("Carol Singh", "carol.singh@company.com", "Engineering"),
        }
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return False
        actual = {
            (
                item.get("name"),
                item.get("email"),
                item.get("department"),
            )
            for item in payload["results"]
            if isinstance(item, dict)
        }
        return actual == expected

    contacts_calls = [
        call
        for call in state.tool_calls
        if call.name == "get_contacts"
        and _result_matches_if_present(state, call, contacts_result_is_engineering)
    ]
    if not contacts_calls:
        return _fail("Did not call get_contacts.")
    contacts_call = contacts_calls[-1]
    if any(call.name != "get_contacts" for call in state.tool_calls):
        return _partial("Called an unrelated tool during a structured contact lookup.")
    # Listing every contact and filtering client-side is a valid strategy, so an
    # empty/broad query is fine — the contact data itself is verified below.
    # Only a query aimed at a different group is wrong.
    contacts_query = normalize(as_str(contacts_call.arguments.get("query")))
    if contacts_query not in ("", "all", "*", "contacts") and "engineering" not in contacts_query:
        return _partial("Called get_contacts, but looked up the wrong contact group.")
    if not result_is_usable_if_present(state, contacts_call):
        return _partial("get_contacts returned an error, so contacts cannot be confirmed.")

    answer = _extract_json_answer(state.final_answer)

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

    if len(contacts) != 2:
        return _partial(f"Expected exactly 2 engineering contacts, got {len(contacts)}.")

    # Validate each contact has required fields
    for i, c in enumerate(contacts):
        if not isinstance(c, dict):
            return _partial(f"Contact {i} is not an object.")
        if not all(k in c for k in ("name", "email", "department")):
            return _partial(f"Contact {i} is missing required fields.")
        if not all(isinstance(c.get(k), str) and c[k] for k in ("name", "email", "department")):
            return _partial(f"Contact {i} has schema-invalid field types.")
        if set(c) != {"name", "email", "department"}:
            return _partial(f"Contact {i} contains fields forbidden by the schema.")

    # Verify total matches array length
    if data.get("total") != len(contacts):
        return _partial("'total' doesn't match contacts array length.")

    # Verify data integrity — contacts should come from tool result
    expected_contacts = {
        ("Alice Zhang", "alice.zhang@company.com", "Engineering"),
        ("Carol Singh", "carol.singh@company.com", "Engineering"),
    }
    actual_contacts = {(c.get("name"), c.get("email"), c.get("department")) for c in contacts}
    if actual_contacts != expected_contacts:
        return _partial("Contacts don't match tool result data.")
    if data.get("query") != "engineering" or set(data) != {"query", "total", "contacts"}:
        return _partial("Top-level contact fields do not match the requested schema and query.")

    return _pass("Produced schema-compliant nested JSON with correct contact data from tool.")


SCENARIO = ScenarioDefinition(
    id="TC-66",
    title="Nested Schema (Array of Objects)",
    category=Category.O,
    user_message=(
        "Look up all engineering contacts and return the results "
        "as a JSON object matching this schema.\n\n"
        f"Schema:\n```json\n{_schema_text(_TC66_SCHEMA)}\n```"
    ),
    description="Call get_contacts and format as nested JSON with array of objects.",
    handle_tool_call=_tc66_handle,
    evaluate=_tc66_eval,
    response_format_override=_TC66_SCHEMA,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it calls get_contacts and produces nested JSON with array of contact objects.",
    "Fail if it doesn't call the tool or produces flat/incorrect structure.",
)
