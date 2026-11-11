"""Versioned, opt-in fixtures that preserve scenario IDs and control comparability."""

from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
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


def _normalize_nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize_nfc(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_nfc(item) for item in value)
    if isinstance(value, dict):
        return {_normalize_nfc(key): _normalize_nfc(item) for key, item in value.items()}
    return value


def substitute(value: Any, replacements: dict[str, str], *, ignore_case: bool = False) -> Any:
    """Replace complete fixture literals, including inside nested payloads."""
    if isinstance(value, str):
        if not replacements:
            return value
        normalized = unicodedata.normalize("NFC", value)
        normalized_map = {unicodedata.normalize("NFC", k): v for k, v in replacements.items()}
        lookup = (
            {k.lower(): v for k, v in normalized_map.items()} if ignore_case else normalized_map
        )
        boundary = r"[0-9A-Za-z_]" if all(k.isascii() for k in normalized_map) else r"\w"
        pattern = (
            rf"(?<!{boundary})(?:"
            + "|".join(re.escape(k) for k in sorted(normalized_map, key=len, reverse=True))
            + rf")(?!{boundary})"
        )
        flags = re.IGNORECASE if ignore_case else 0
        return re.sub(
            pattern,
            lambda match: lookup[match.group().lower() if ignore_case else match.group()],
            normalized,
            flags=flags,
        )
    if isinstance(value, list):
        return [substitute(item, replacements, ignore_case=ignore_case) for item in value]
    if isinstance(value, tuple):
        return tuple(substitute(item, replacements, ignore_case=ignore_case) for item in value)
    if isinstance(value, dict):
        return {
            substitute(key, replacements, ignore_case=ignore_case): substitute(
                item, replacements, ignore_case=ignore_case
            )
            for key, item in value.items()
        }
    return value


def translated_state(
    state: ScenarioState, replacements: dict[str, str], *, ignore_case: bool = False
) -> ScenarioState:
    result = copy.deepcopy(state)
    for call in result.tool_calls:
        call.arguments = substitute(call.arguments, replacements, ignore_case=ignore_case)
        call.raw_arguments = substitute(call.raw_arguments, replacements, ignore_case=ignore_case)
    for record in result.tool_results:
        record.result = substitute(record.result, replacements, ignore_case=ignore_case)
    result.assistant_messages = substitute(
        result.assistant_messages, replacements, ignore_case=ignore_case
    )
    result.final_answer = substitute(result.final_answer, replacements, ignore_case=ignore_case)
    result.meta = substitute(result.meta, replacements, ignore_case=ignore_case)
    return result


def identity_variant(scenario: ScenarioDefinition, seed: int) -> ScenarioDefinition:
    replacements = {
        literal: f"fixture_{hashlib.sha256(f'{scenario.id}:{seed}:{literal}'.encode()).hexdigest()[:12]}"
        + ("@example.com" if "@" in literal else "")
        for literal in scenario.variant_literals
    }
    inverse = {value: key for key, value in replacements.items()}
    email_replacements = {k: v for k, v in replacements.items() if "@" in k}
    token_replacements = {k: v for k, v in replacements.items() if "@" not in k}

    def stale(value: Any) -> bool:
        normalized = _normalize_nfc(value)
        rewritten = substitute(normalized, token_replacements)
        if email_replacements:
            rewritten = substitute(rewritten, email_replacements, ignore_case=True)
        return bool(rewritten != normalized)

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        if stale(call.arguments):
            return {"error": "This fixture does not contain the requested identifier."}
        original = translated_state(state, inverse, ignore_case=True)
        original_call = next(
            (c for c in original.tool_calls if c.id == call.id),
            replace(
                call,
                arguments=substitute(call.arguments, inverse, ignore_case=True),
                raw_arguments=substitute(call.raw_arguments, inverse, ignore_case=True),
            ),
        )
        result = scenario.handle_tool_call(original, original_call)
        state.meta = substitute(original.meta, replacements)
        return substitute(result, replacements)

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        result = scenario.evaluate(translated_state(state, inverse, ignore_case=True))
        if any(stale(call.arguments) for call in state.tool_calls) or stale(state.final_answer):
            failure = fail_eval(
                "Used an identifier from a different fixture instead of the observed data."
            )
            return replace(
                failure, safety_violation=result.safety_violation, diagnostics=result.diagnostics
            )
        return replace(result, summary=substitute(result.summary, replacements))

    audit = scenario.answer_audit
    if audit is not None:
        original_choice = audit.deterministic_choice
        audit = replace(
            audit,
            deterministic_choice=lambda state: original_choice(
                translated_state(state, inverse, ignore_case=True)
            ),
        )

    checkpoint = scenario.checkpoint
    if checkpoint is not None:
        original_checkpoint = checkpoint
        checkpoint = lambda state, phase: original_checkpoint(  # noqa: E731
            translated_state(state, inverse, ignore_case=True), phase
        )

    return replace(
        scenario,
        user_message=substitute(scenario.user_message, replacements),
        follow_up_messages=substitute(scenario.follow_up_messages, replacements),
        tools_override=substitute(scenario.tools_override, replacements),
        handle_tool_call=handle,
        evaluate=evaluate,
        checkpoint=checkpoint,
        answer_audit=audit,
        variant_metadata={"version": VARIANT_VERSION, "seed": seed, "kind": "identifiers"},
    )


def apply_variants(
    scenarios: list[ScenarioDefinition], seed: int | None
) -> list[ScenarioDefinition]:
    if seed is None:
        return scenarios
    result = []
    for scenario in scenarios:
        if "seed" in scenario.variant_metadata:
            if scenario.variant_metadata.get("seed") != seed:
                raise ValueError(
                    "Select variants from the original scenario definitions when changing the seed."
                )
            result.append(scenario)
        elif scenario.held_out:
            result.append(scenario)
        elif scenario.variant_factory:
            result.append(_keep_metadata(scenario, scenario.variant_factory(scenario, seed)))
        elif scenario.variant_literals:
            result.append(_keep_metadata(scenario, identity_variant(scenario, seed)))
        else:
            result.append(scenario)
    return result


def _keep_metadata(base: ScenarioDefinition, variant: ScenarioDefinition) -> ScenarioDefinition:
    """Carry metadata from earlier transforms (such as a locale) into the variant."""
    if not base.variant_metadata:
        return variant
    return replace(variant, variant_metadata={**base.variant_metadata, **variant.variant_metadata})


def prepare_scenarios(
    scenarios: list[ScenarioDefinition], *, variant_seed: int | None, locale: str | None = None
) -> list[ScenarioDefinition]:
    """Apply a run's locale, then its fixture variants.

    The locale goes first so seeded fixtures rewrite the translated prompt.
    Scenarios that do not support the locale are dropped.
    """
    from tool_eval_bench.evals.locales import apply_locale

    return apply_variants(apply_locale(scenarios, locale), variant_seed)
