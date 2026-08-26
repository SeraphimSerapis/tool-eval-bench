"""Declarative YAML scenario loader (pilot).

Provides a small, data-driven way to author simple tool-call scenarios without
writing Python evaluator functions.  This is intentionally limited compared to
the full Python scenario API; it is a pilot for the "YAML-first" direction
identified in the project assessment.

Supported YAML format::

    id: YAML-01
    title: Simple weather lookup
    category: A
    difficulty: 1
    description: Model calls get_weather for Berlin.
    user_message: What is the weather in Berlin?
    expected_tool_calls:
      - tool: get_weather
        arguments:
          location: Berlin
    tool_responses:
      get_weather:
        - match:
            location: Berlin
          response:
            temperature: 18
            condition: cloudy
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)


def _make_handler(
    tool_responses: dict[str, list[dict[str, Any]]],
) -> Any:
    """Build a handle_tool_call callable from declarative response rules."""

    def handle_tool_call(state: ScenarioState, record: ToolCallRecord) -> Any:
        rules = tool_responses.get(record.name, [])
        for rule in rules:
            match = rule.get("match") or {}
            if all(record.arguments.get(k) == v for k, v in match.items()):
                return rule.get("response", {"result": "ok"})
        # No rule matched — return a generic empty success so the conversation
        # can continue; the evaluator will flag the mismatch.
        return {"result": "ok"}

    return handle_tool_call


def _make_evaluator(
    expected_tool_calls: list[dict[str, Any]],
) -> Any:
    """Build an evaluator that checks expected tool calls and arguments."""

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        if not expected_tool_calls:
            if state.tool_calls:
                called = ", ".join(call.name for call in state.tool_calls)
                return ScenarioEvaluation(
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary=f"No tools expected, but called: {called}.",
                )
            return ScenarioEvaluation(
                status=ScenarioStatus.PASS,
                points=2,
                summary="No tools expected; none called.",
            )

        call_index = 0
        for expected in expected_tool_calls:
            tool = expected["tool"]
            args = expected.get("arguments", {})
            if call_index >= len(state.tool_calls):
                return ScenarioEvaluation(
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary=f"Missing expected tool call {tool}.",
                )
            actual = state.tool_calls[call_index]
            if actual.name != tool:
                return ScenarioEvaluation(
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary=f"Expected {tool}, got {actual.name}.",
                )
            for key, val in args.items():
                if actual.arguments.get(key) != val:
                    return ScenarioEvaluation(
                        status=ScenarioStatus.FAIL,
                        points=0,
                        summary=f"Argument {key} mismatch for {tool}.",
                    )
            call_index += 1

        if len(state.tool_calls) > call_index:
            return ScenarioEvaluation(
                status=ScenarioStatus.FAIL,
                points=0,
                summary="Extra tool calls made.",
            )

        return ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="All expected tool calls matched.",
        )

    return evaluate


def _required_string(data: dict[str, Any], field: str, path: Path) -> str:
    """Read a required non-empty string field with path-aware errors."""
    if field not in data:
        raise ValueError(f"Missing required field {field!r} in {path}")
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Required field {field!r} must be a non-empty string in {path}")
    return value


def _load_yaml_file(path: Path, raw_bytes: bytes | None = None) -> ScenarioDefinition:
    """Load a single YAML scenario file into a ScenarioDefinition.

    *raw_bytes* lets a caller that has already read the file (to hash it, say)
    hand the bytes over instead of forcing a second read.
    """
    raw = (raw_bytes if raw_bytes is not None else path.read_bytes()).decode("utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path}:\n{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Scenario YAML must be a mapping: {path}")

    scenario_id = _required_string(data, "id", path)
    title = _required_string(data, "title", path)
    category_value = _required_string(data, "category", path)
    user_message = _required_string(data, "user_message", path)
    try:
        category = Category(category_value)
    except ValueError as exc:
        raise ValueError(f"Invalid category in {path}: {exc}") from exc

    tool_responses = data.get("tool_responses", {})
    expected_tool_calls = data.get("expected_tool_calls", [])

    return ScenarioDefinition(
        id=scenario_id,
        title=title,
        category=category,
        user_message=user_message,
        description=data.get("description", ""),
        handle_tool_call=_make_handler(tool_responses),
        evaluate=_make_evaluator(expected_tool_calls),
        difficulty=data.get("difficulty"),
        held_out=bool(data.get("held_out", False)),
    )


def load_yaml_scenarios(
    directory: str | Path, *, held_out: bool = False
) -> list[ScenarioDefinition]:
    """Load all ``*.yaml`` scenario files from *directory* in sorted order.

    When ``held_out`` is set, every scenario in the directory is marked held-out
    regardless of its own flag, so a whole private pack can be protected without
    annotating each file.
    """
    return [
        scenario for scenario, _, _ in load_yaml_scenarios_with_bytes(directory, held_out=held_out)
    ]


def load_yaml_scenarios_with_bytes(
    directory: str | Path, *, held_out: bool = False
) -> list[tuple[ScenarioDefinition, Path, bytes]]:
    """Load scenarios with the file and raw bytes each was parsed from.

    Pack loading needs both the scenarios and a content hash over the same
    files.  Returning the source bytes lets it walk and read the directory once
    instead of twice.
    """
    root = Path(directory)
    loaded: list[tuple[ScenarioDefinition, Path, bytes]] = []
    for path in sorted(root.glob("*.yaml")):
        raw_bytes = path.read_bytes()
        scenario = _load_yaml_file(path, raw_bytes)
        if held_out and not scenario.held_out:
            scenario = replace(scenario, held_out=True)
        loaded.append((scenario, path, raw_bytes))
    return loaded
