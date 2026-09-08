"""Versioned, opt-in fixtures that preserve scenario IDs and control comparability."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import replace
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import fail_eval

VARIANT_VERSION = 1


def substitute(value: Any, replacements: dict[str, str]) -> Any:
    """Replace complete fixture literals, including inside nested payloads."""
    if isinstance(value, str):
        if not replacements:
            return value
        pattern = (
            r"(?<![\w])(?:"
            + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
            + r")(?![\w])"
        )
        return re.sub(pattern, lambda match: replacements[match.group()], value)
    if isinstance(value, list):
        return [substitute(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, replacements) for key, item in value.items()}
    return value


def translated_state(state: ScenarioState, replacements: dict[str, str]) -> ScenarioState:
    result = copy.deepcopy(state)
    for call in result.tool_calls:
        call.arguments = substitute(call.arguments, replacements)
        call.raw_arguments = substitute(call.raw_arguments, replacements)
    for record in result.tool_results:
        record.result = substitute(record.result, replacements)
    result.assistant_messages = substitute(result.assistant_messages, replacements)
    result.final_answer = substitute(result.final_answer, replacements)
    result.meta = substitute(result.meta, replacements)
    return result


def identity_variant(scenario: ScenarioDefinition, seed: int) -> ScenarioDefinition:
    replacements = {
        literal: f"fixture_{hashlib.sha256(f'{scenario.id}:{seed}:{literal}'.encode()).hexdigest()[:12]}"
        + ("@example.com" if "@" in literal else "")
        for literal in scenario.variant_literals
    }
    inverse = {value: key for key, value in replacements.items()}

    def stale(value: Any) -> bool:
        return substitute(value, replacements) != value

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        if stale(call.arguments):
            return {"error": "This fixture does not contain the requested identifier."}
        original = translated_state(state, inverse)
        original_call = next(
            (c for c in original.tool_calls if c.id == call.id),
            replace(call, arguments=substitute(call.arguments, inverse)),
        )
        result = scenario.handle_tool_call(original, original_call)
        state.meta = substitute(original.meta, replacements)
        return substitute(result, replacements)

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        result = scenario.evaluate(translated_state(state, inverse))
        if any(stale(call.arguments) for call in state.tool_calls) or stale(state.final_answer):
            failure = fail_eval(
                "Used an identifier from a different fixture instead of the observed data."
            )
            return replace(
                failure, safety_violation=result.safety_violation, diagnostics=result.diagnostics
            )
        return replace(result, summary=substitute(result.summary, replacements))

    return replace(
        scenario,
        user_message=substitute(scenario.user_message, replacements),
        follow_up_messages=substitute(scenario.follow_up_messages, replacements),
        tools_override=substitute(scenario.tools_override, replacements),
        handle_tool_call=handle,
        evaluate=evaluate,
        variant_metadata={"version": VARIANT_VERSION, "seed": seed, "kind": "identifiers"},
    )


def apply_variants(
    scenarios: list[ScenarioDefinition], seed: int | None
) -> list[ScenarioDefinition]:
    if seed is None:
        return scenarios
    result = []
    for scenario in scenarios:
        if scenario.variant_metadata:
            if scenario.variant_metadata.get("seed") != seed:
                raise ValueError(
                    "Select variants from the original scenario definitions when changing the seed."
                )
            result.append(scenario)
        elif scenario.held_out:
            result.append(scenario)
        elif scenario.variant_factory:
            result.append(scenario.variant_factory(scenario, seed))
        elif scenario.variant_literals:
            result.append(identity_variant(scenario, seed))
        else:
            result.append(scenario)
    return result
