"""No schema-violating answer may PASS a structured-output scenario.

An endpoint that rejects ``json_schema`` falls back to ``json_object``, which
promises valid JSON but not the schema's shape. The evaluators' hand-written
checks are then the only schema enforcement, so this walks each scenario's
schema, breaks a passing answer in every way the schema forbids, and requires
that none of the broken answers grade PASS.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import make_state

from tool_eval_bench.domain.scenarios import ScenarioDefinition, ScenarioStatus
from tool_eval_bench.evals.scenarios import STRUCTURED_SCENARIOS

# A passing trace per scenario: the tool calls it needs and a schema-valid answer.
_PASSING: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    "TC-64": (
        [],
        {
            "title": "The Matrix",
            "year": 1999,
            "rating": 8.7,
            "genre": "sci-fi",
            "summary": "A hacker learns the truth about reality.",
        },
    ),
    "TC-65": (
        [{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
        {
            "location": "Tokyo",
            "temperature_celsius": 28,
            "condition": "Sunny",
            "recommendation": "Wear light, breathable clothing.",
        },
    ),
    "TC-66": (
        [{"name": "get_contacts", "arguments": {"query": "engineering"}}],
        {
            "query": "engineering",
            "total": 2,
            "contacts": [
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
    ),
    "TC-67": (
        [
            {"name": "get_stock_price", "arguments": {"ticker": "NVDA"}},
            {"name": "web_search", "arguments": {"query": "NVDA recent news"}},
        ],
        {
            "ticker": "NVDA",
            "price": 892.50,
            "currency": "USD",
            "signal": "buy",
            "reasoning": "Revenue up 265% YoY on AI demand; analysts target $950.",
        },
    ),
    "TC-68": ([], {"task_id": "PROJ-127", "status": "in_progress", "assignee": "me"}),
    "TC-69": (
        [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}},
            {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
        ],
        {
            "date": "2026-03-20",
            "weather": {"location": "San Francisco", "temperature": 18, "condition": "Foggy"},
            "market": {"ticker": "AAPL", "price": 192.30, "direction": "down"},
            "action_items": ["Bring a jacket.", "Watch AAPL."],
        },
    ),
}

# Values of the wrong JSON type for each schema type. bool is listed for the
# numeric types because Python's bool is an int, a classic hand-check hole.
_WRONG_TYPE: dict[str, list[Any]] = {
    "string": [123, None, ["x"], {"x": 1}],
    "number": ["42", True, None, [1]],
    "integer": ["42", 1.5, True, None],
    "array": ["not-an-array", None, {"x": 1}],
    "object": ["not-an-object", None, [1]],
}


def _schema(scenario: ScenarioDefinition) -> dict[str, Any]:
    """The schema the scenario shows the model, even when it sends no response_format."""
    wrapper = scenario.response_format_override
    if wrapper is None:
        module = importlib.import_module(scenario.evaluate.__module__)
        wrapper = next(v for k, v in vars(module).items() if k.endswith("_SCHEMA"))
    return wrapper["json_schema"]["schema"]


def _violations(
    schema: dict[str, Any], value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], str, Any]]:
    """Yield ``(path, label, replacement)`` for every way to break ``value``."""
    kind = schema.get("type")
    if "enum" in schema:
        yield path, "value outside enum", "not_a_valid_value"
    for bad in _WRONG_TYPE.get(str(kind), []):
        yield path, f"{kind} replaced by {bad!r}", bad
    if "minimum" in schema:
        yield path, "below minimum", schema["minimum"] - 1
    if "maximum" in schema:
        yield path, "above maximum", schema["maximum"] + 1
    if kind == "object":
        for key in schema.get("required", []):
            yield path, f"missing {key}", {k: v for k, v in value.items() if k != key}
        if schema.get("additionalProperties") is False:
            yield path, "extra key", {**value, "unexpected": "x"}
        for key, sub in schema.get("properties", {}).items():
            yield from _violations(sub, value[key], (*path, key))
    if kind == "array" and value:
        yield from _violations(schema["items"], value[0], (*path, 0))


def _replace(data: Any, path: tuple[Any, ...], new: Any) -> Any:
    if not path:
        return new
    data = copy.deepcopy(data)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = new
    return data


def test_every_structured_scenario_has_a_passing_trace() -> None:
    assert {s.id for s in STRUCTURED_SCENARIOS} == set(_PASSING)


@pytest.mark.parametrize("scenario", STRUCTURED_SCENARIOS, ids=lambda s: s.id)
def test_schema_violations_never_pass(scenario: ScenarioDefinition) -> None:
    calls, answer = _PASSING[scenario.id]

    def grade(final: Any) -> ScenarioStatus:
        state = make_state(tool_calls=calls, final_answer=json.dumps(final))
        return scenario.evaluate(state).status

    assert grade(answer) == ScenarioStatus.PASS, "the passing trace no longer passes"

    violations = list(_violations(_schema(scenario), answer))
    assert violations
    leaks = [
        f"{'/'.join(map(str, path)) or '<root>'}: {label}"
        for path, label, bad in violations
        if grade(_replace(answer, path, bad)) == ScenarioStatus.PASS
    ]
    assert not leaks, f"{scenario.id} passed schema-invalid answers: {leaks}"
