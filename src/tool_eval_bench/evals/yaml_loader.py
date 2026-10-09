"""Declarative YAML scenario loader.

A data-driven way to author simple tool-call scenarios without writing Python.
Deliberately narrower than the Python API: it matches tool calls positionally
and cannot inspect tool results or run arbitrary logic.  Its job is held-out
packs, where a third party needs private scenarios without shipping executable
code.  See ``docs/scenario-packs.md``.

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
    answer_contains:
      - "18"
      - cloudy
    capabilities:           # optional; keys of CAPABILITY_LABELS
      - tool-selection

``answer_contains`` is what makes the middle tier reachable.  Every entry must
appear in the model's final answer, case-insensitively.  Getting the tool calls
right but never stating the result scores PARTIAL rather than PASS, which is
the distinction a three-tier benchmark exists to make.

The loader validates the whole file up front, because a mistake that only
surfaces at run time is scored as the model's failure.  Expected tools must be
among the universal tools every YAML scenario is offered.  ``difficulty`` is an
integer from 1 to 5.  A key the format does not define is rejected, so a typo
cannot quietly drop a check.  A plain scalar that YAML 1.1 reads differently
from JSON or YAML 1.2, such as ``2026-03-21``, ``14:30``, ``NO``, ``01234``, or
``1e5``, is rejected with its line and column; quote it to keep it a string.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from tool_eval_bench.domain.scenarios import (
    CAPABILITY_LABELS,
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

# YAML scenarios cannot declare tools; the orchestrator offers these to every one.
_OFFERED_TOOLS = frozenset(tool["function"]["name"] for tool in UNIVERSAL_TOOLS)


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


def _missing_from_answer(answer: str, required: list[str]) -> list[str]:
    """Return the required snippets absent from *answer*, case-insensitively."""
    haystack = (answer or "").lower()
    return [snippet for snippet in required if snippet.lower() not in haystack]


def _make_evaluator(
    expected_tool_calls: list[dict[str, Any]],
    answer_contains: list[str],
) -> Any:
    """Build an evaluator that checks tool calls, then the answer's content."""

    def _verdict(state: ScenarioState, summary: str) -> ScenarioEvaluation:
        """Score tool discipline that already passed against the answer text."""
        missing = _missing_from_answer(state.final_answer, answer_contains)
        if not missing:
            return ScenarioEvaluation(status=ScenarioStatus.PASS, points=2, summary=summary)
        return ScenarioEvaluation(
            status=ScenarioStatus.PARTIAL,
            points=1,
            summary=f"{summary} Answer never states: {', '.join(missing)}.",
        )

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        if not expected_tool_calls:
            if state.tool_calls:
                called = ", ".join(call.name for call in state.tool_calls)
                return ScenarioEvaluation(
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary=f"No tools expected, but called: {called}.",
                )
            return _verdict(state, "No tools expected; none called.")

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

        return _verdict(state, "All expected tool calls matched.")

    return evaluate


def _required_string(data: dict[str, Any], field: str, path: Path) -> str:
    """Read a required non-empty string field with path-aware errors."""
    if field not in data:
        raise ValueError(f"Missing required field {field!r} in {path}")
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Required field {field!r} must be a non-empty string in {path}")
    return value


def _string_list(data: dict[str, Any], field: str, path: Path) -> list[str]:
    """Read an optional list-of-strings field, rejecting a bare string.

    ``answer_contains: cloudy`` is the natural typo, and taken literally it
    would assert every character of the word separately.
    """
    value = data.get(field)
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, list):
        raise ValueError(f"Field {field!r} must be a list of strings in {path}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Field {field!r} must contain only non-empty strings in {path}")
    return list(value)


# YAML 1.1 resolves some plain scalars to types a pack author rarely means:
# ``2026-03-21`` becomes a date, ``14:30`` the sexagesimal int 870, ``NO`` the
# bool False, and ``01234`` the octal int 668.  Tool arguments arrive as JSON,
# so such a value can never match what a model sends, and a date cannot even be
# serialized back to it.  These forms are rejected rather than reinterpreted:
# a reinterpreting loader would grade an unchanged pack differently while its
# byte-based content hash stayed the same.
_YAML_BOOL = "tag:yaml.org,2002:bool"
_YAML_INT = "tag:yaml.org,2002:int"
_YAML_FLOAT = "tag:yaml.org,2002:float"
_YAML_STR = "tag:yaml.org,2002:str"
_YAML_TIMESTAMP = "tag:yaml.org,2002:timestamp"
_PLAIN_BOOLS = frozenset({"true", "True", "TRUE", "false", "False", "FALSE"})
_PLAIN_INT = re.compile(r"[-+]?(?:0|[1-9][0-9]*)|0x[0-9a-fA-F]+")
_PLAIN_FLOAT = re.compile(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")
# The reverse case: ``1e5`` or ``-.5`` is a number in JSON and YAML 1.2, but
# YAML 1.1 leaves it a string.
_YAML12_NUMBER = re.compile(r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|0o[0-7]+")
_QUOTE_IT = "quote it to keep it a string"


def _ambiguous_plain_scalar(tag: str, value: str) -> str | None:
    """How to fix a plain scalar with a YAML 1.1-only reading, or ``None`` if it has none."""
    if tag == _YAML_TIMESTAMP:
        return _QUOTE_IT
    if tag == _YAML_BOOL and value not in _PLAIN_BOOLS:
        return f"{_QUOTE_IT}, or write true or false"
    if tag == _YAML_INT and _PLAIN_INT.fullmatch(value) is None:
        return f"{_QUOTE_IT}, or write the number in plain decimal"
    if tag == _YAML_FLOAT and _PLAIN_FLOAT.fullmatch(value) is None:
        return f"{_QUOTE_IT}, or write the number in plain decimal"
    if tag == _YAML_STR and _YAML12_NUMBER.fullmatch(value) is not None:
        return f"{_QUOTE_IT}, or write a number as plain decimal such as 100000 or 0.5"
    return None


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses plain scalars YAML 1.1 reads ambiguously.

    Quoted and block scalars, and explicitly tagged ones, are untouched: the
    author has already said what they mean.
    """

    # Typed Any: the PyYAML stubs declare the anchor as a dict, but it is a str or None.
    def compose_scalar_node(self, anchor: Any) -> yaml.ScalarNode:
        event = self.peek_event()
        plain = event.tag in (None, "!") and event.implicit[0]
        node = super().compose_scalar_node(anchor)
        hint = _ambiguous_plain_scalar(node.tag, node.value) if plain else None
        if hint is not None:
            reading = node.tag.rsplit(":", 1)[-1]
            raise yaml.MarkedYAMLError(
                problem=(
                    f"ambiguous unquoted value {node.value!r} (YAML 1.1 reads it as {reading}); "
                    f"{hint}"
                ),
                problem_mark=node.start_mark,
            )
        return node


def _check_json_value(value: Any, where: str, path: Path) -> None:
    """Reject values that cannot travel as JSON, such as an explicitly tagged date."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Field {where!r} must be a finite number in {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_value(item, f"{where}[{index}]", path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Field {where!r} has a non-string key {key!r} in {path}")
            _check_json_value(item, f"{where}.{key}", path)
        return
    raise ValueError(
        f"Field {where!r} holds a {type(value).__name__}, which is not a JSON value, in {path}"
    )


def _mapping_field(value: Any, where: str, path: Path) -> dict[str, Any]:
    """Validate a field that must be a JSON object when present; ``None`` is not one."""
    if not isinstance(value, dict):
        raise ValueError(f"Field {where!r} must be a mapping in {path}")
    _check_json_value(value, where, path)
    return value


# Every key the format defines.  Anything else is most likely a typo, and a
# misspelled ``arguments`` would silently drop the check it was meant to make.
_SCENARIO_KEYS = frozenset(
    {
        "id",
        "title",
        "category",
        "difficulty",
        "description",
        "user_message",
        "expected_tool_calls",
        "tool_responses",
        "answer_contains",
        "capabilities",
        "held_out",
    }
)
_EXPECTED_CALL_KEYS = frozenset({"tool", "arguments"})
_RESPONSE_RULE_KEYS = frozenset({"match", "response"})


def _reject_unknown_keys(
    mapping: dict[Any, Any], allowed: frozenset[str], where: str, path: Path
) -> None:
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(
            f"Unknown key {names} in {where} in {path}; allowed: {', '.join(sorted(allowed))}"
        )


def _difficulty(data: dict[str, Any], path: Path) -> int | None:
    """Read ``difficulty``: absent, or an integer 1-5 as the domain model defines."""
    value = data.get("difficulty")
    if value is None:
        return None
    # bool is an int subclass, and ``difficulty: true`` is not a tier.
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError(f"Field 'difficulty' must be an integer from 1 to 5 in {path}")
    return value


def _expected_tool_calls(data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    """Validate ``expected_tool_calls`` as a list of ``{tool, arguments?}`` mappings.

    YAML scenarios always run against ``UNIVERSAL_TOOLS``, so an expected tool
    outside that set is one no model is ever offered.
    """
    value = data.get("expected_tool_calls")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field 'expected_tool_calls' must be a list in {path}")
    for index, entry in enumerate(value):
        where = f"expected_tool_calls[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"Field {where!r} must be a mapping in {path}")
        _reject_unknown_keys(entry, _EXPECTED_CALL_KEYS, where, path)
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError(f"Field {where + '.tool'!r} must be a non-empty string in {path}")
        if tool not in _OFFERED_TOOLS:
            raise ValueError(
                f"Field {where + '.tool'!r} names {tool!r}, which YAML scenarios never offer, "
                f"in {path}; choose from {', '.join(sorted(_OFFERED_TOOLS))}"
            )
        if "arguments" in entry:
            _mapping_field(entry["arguments"], f"{where}.arguments", path)
    return value


def _tool_responses(data: dict[str, Any], path: Path) -> dict[str, list[dict[str, Any]]]:
    """Validate ``tool_responses`` as a mapping of tool name to a list of rules."""
    value = data.get("tool_responses")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Field 'tool_responses' must be a mapping in {path}")
    for tool, rules in value.items():
        if not isinstance(tool, str):
            raise ValueError(f"Field 'tool_responses' has a non-string key {tool!r} in {path}")
        if not isinstance(rules, list):
            raise ValueError(f"Field {'tool_responses.' + tool!r} must be a list in {path}")
        for index, rule in enumerate(rules):
            where = f"tool_responses.{tool}[{index}]"
            if not isinstance(rule, dict):
                raise ValueError(f"Field {where!r} must be a mapping in {path}")
            _reject_unknown_keys(rule, _RESPONSE_RULE_KEYS, where, path)
            # An empty ``match:`` matches every call, as an absent one does.
            if rule.get("match") is not None:
                _mapping_field(rule["match"], f"{where}.match", path)
            if "response" in rule:
                response = rule["response"]
                if not isinstance(response, (dict, str)):
                    raise ValueError(
                        f"Field {where + '.response'!r} must be a mapping or a string in {path}"
                    )
                _check_json_value(response, f"{where}.response", path)
    return value


def _load_yaml_file(path: Path, raw_bytes: bytes | None = None) -> ScenarioDefinition:
    """Load a single YAML scenario file into a ScenarioDefinition.

    *raw_bytes* lets a caller that has already read the file (to hash it, say)
    hand the bytes over instead of forcing a second read.
    """
    raw = (raw_bytes if raw_bytes is not None else path.read_bytes()).decode("utf-8")
    try:
        data = yaml.load(raw, Loader=_StrictLoader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path}:\n{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Scenario YAML must be a mapping: {path}")
    _reject_unknown_keys(data, _SCENARIO_KEYS, "the scenario", path)

    scenario_id = _required_string(data, "id", path)
    title = _required_string(data, "title", path)
    category_value = _required_string(data, "category", path)
    user_message = _required_string(data, "user_message", path)
    try:
        category = Category(category_value)
    except ValueError as exc:
        raise ValueError(f"Invalid category in {path}: {exc}") from exc

    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"Field 'description' must be a string in {path}")
    difficulty = _difficulty(data, path)
    tool_responses = _tool_responses(data, path)
    expected_tool_calls = _expected_tool_calls(data, path)
    answer_contains = _string_list(data, "answer_contains", path)
    capabilities = _string_list(data, "capabilities", path)
    unknown = [tag for tag in capabilities if tag not in CAPABILITY_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown capabilities {unknown} in {path}; choose from {', '.join(CAPABILITY_LABELS)}"
        )

    return ScenarioDefinition(
        id=scenario_id,
        title=title,
        category=category,
        user_message=user_message,
        description=description or "",
        handle_tool_call=_make_handler(tool_responses),
        evaluate=_make_evaluator(expected_tool_calls, answer_contains),
        difficulty=difficulty,
        held_out=bool(data.get("held_out", False)),
        capabilities=tuple(capabilities),
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
