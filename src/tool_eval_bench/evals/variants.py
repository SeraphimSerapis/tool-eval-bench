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


def _invert_emails(value: Any, email_inverse: dict[str, str]) -> Any:
    """Restore email literals from fixture addresses while preserving case edits."""
    if not email_inverse:
        return value
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
        lower_lookup = {
            unicodedata.normalize("NFC", k).lower(): (unicodedata.normalize("NFC", k), v)
            for k, v in email_inverse.items()
        }
        boundary = r"[0-9A-Za-z_]" if all(k.isascii() for k in lower_lookup) else r"\w"
        pattern = (
            rf"(?<!{boundary})(?:"
            + "|".join(re.escape(k) for k in sorted(lower_lookup, key=len, reverse=True))
            + rf")(?!{boundary})"
        )

        def _replace(match: re.Match[str]) -> str:
            matched = match.group()
            exact_fixture, original = lower_lookup[matched.lower()]
            if matched == exact_fixture:
                return original
            return original.upper() if original.upper() != original else original.lower()

        return re.sub(pattern, _replace, normalized, flags=re.IGNORECASE)
    if isinstance(value, list):
        return [_invert_emails(item, email_inverse) for item in value]
    if isinstance(value, tuple):
        return tuple(_invert_emails(item, email_inverse) for item in value)
    if isinstance(value, dict):
        return {
            _invert_emails(key, email_inverse): _invert_emails(item, email_inverse)
            for key, item in value.items()
        }
    return value


def _invert_value(value: Any, token_inverse: dict[str, str], email_inverse: dict[str, str]) -> Any:
    restored = substitute(value, token_inverse) if token_inverse else value
    return _invert_emails(restored, email_inverse)


def translated_state(
    state: ScenarioState,
    replacements: dict[str, str],
    *,
    email_replacements: dict[str, str] | None = None,
    ignore_case: bool = False,
) -> ScenarioState:
    def _transform(value: Any) -> Any:
        restored = (
            substitute(value, replacements, ignore_case=ignore_case) if replacements else value
        )
        return _invert_emails(restored, email_replacements) if email_replacements else restored

    result = copy.deepcopy(state)
    for call in result.tool_calls:
        call.arguments = _transform(call.arguments)
        call.raw_arguments = _transform(call.raw_arguments)
    for record in result.tool_results:
        record.result = _transform(record.result)
    result.assistant_messages = _transform(result.assistant_messages)
    result.final_answer = _transform(result.final_answer)
    result.meta = _transform(result.meta)
    return result


def _iter_nfc_strings(value: Any) -> list[str]:
    strings: list[str] = []

    def _collect(item: Any) -> None:
        if isinstance(item, str):
            strings.append(unicodedata.normalize("NFC", item))
        elif isinstance(item, (list, tuple)):
            for child in item:
                _collect(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                _collect(key)
                _collect(child)

    _collect(value)
    return strings


def _compile_literal_pattern(
    replacements: dict[str, str], *, ignore_case: bool
) -> re.Pattern[str] | None:
    if not replacements:
        return None
    normalized_keys = [unicodedata.normalize("NFC", k) for k in replacements]
    boundary = r"[0-9A-Za-z_]" if all(k.isascii() for k in normalized_keys) else r"\w"
    return re.compile(
        rf"(?<!{boundary})(?:"
        + "|".join(re.escape(k) for k in sorted(normalized_keys, key=len, reverse=True))
        + rf")(?!{boundary})",
        re.IGNORECASE if ignore_case else 0,
    )


def identity_variant(scenario: ScenarioDefinition, seed: int) -> ScenarioDefinition:
    replacements = {
        literal: f"fixture_{hashlib.sha256(f'{scenario.id}:{seed}:{literal}'.encode()).hexdigest()[:12]}"
        + ("@example.com" if "@" in literal else "")
        for literal in scenario.variant_literals
    }
    email_replacements = {k: v for k, v in replacements.items() if "@" in k}
    token_replacements = {k: v for k, v in replacements.items() if "@" not in k}
    # Structured identifiers (file_091, INC-901, launch_checklist) are checked
    # case-insensitively so case-mutated baseline IDs cannot bypass stale() on
    # case-normalizing handlers, while short bare opaque tokens (like cursor
    # "p2") remain case-sensitive so prose like "severity P2" does not collide.
    structured_token_replacements = {
        k: v for k, v in token_replacements.items() if "_" in k or "-" in k
    }
    exact_token_keys = frozenset(unicodedata.normalize("NFC", k) for k in token_replacements)
    exact_token_pattern = _compile_literal_pattern(token_replacements, ignore_case=False)
    structured_ci_pattern = _compile_literal_pattern(
        structured_token_replacements, ignore_case=True
    )
    email_ci_pattern = _compile_literal_pattern(email_replacements, ignore_case=True)
    email_inverse = {value: key for key, value in email_replacements.items()}
    token_inverse = {value: key for key, value in token_replacements.items()}

    variant_user_message = substitute(scenario.user_message, replacements)
    variant_follow_ups = substitute(scenario.follow_up_messages, replacements)
    variant_tools_override = substitute(scenario.tools_override, replacements)
    base_observed_strings = _iter_nfc_strings(
        (variant_user_message, variant_follow_ups, variant_tools_override)
    )

    def stale(value: Any, observed_results: Any = ()) -> bool:
        strings = _iter_nfc_strings(value)
        if not strings:
            return False
        if exact_token_pattern is not None and any(exact_token_pattern.search(s) for s in strings):
            return True
        if email_ci_pattern is not None and any(email_ci_pattern.search(s) for s in strings):
            return True
        if structured_ci_pattern is not None:
            observed_strings: list[str] | None = None
            for text in strings:
                for match in structured_ci_pattern.finditer(text):
                    matched = match.group()
                    if matched in exact_token_keys:
                        return True
                    # A case-variant that appeared verbatim in the variant's own
                    # prompt or tool outputs (e.g. "Launch_Checklist.pdf" when
                    # "launch_checklist" was replaced) is observed, not stale.
                    if observed_strings is None:
                        observed_strings = base_observed_strings + _iter_nfc_strings(
                            observed_results
                        )
                    if not any(matched in obs for obs in observed_strings):
                        return True
        return False

    def _invert_call(
        state: ScenarioState, original: ScenarioState, call: ToolCallRecord
    ) -> ToolCallRecord:
        idx = next((i for i, c in enumerate(state.tool_calls) if c is call), None)
        if idx is not None and idx < len(original.tool_calls):
            return original.tool_calls[idx]
        if call.id:
            match = next((c for c in reversed(original.tool_calls) if c.id == call.id), None)
            if match is not None:
                return match
        return replace(
            call,
            arguments=_invert_value(call.arguments, token_inverse, email_inverse),
            raw_arguments=_invert_value(call.raw_arguments, token_inverse, email_inverse),
        )

    def handle(state: ScenarioState, call: ToolCallRecord) -> Any:
        observed = [record.result for record in state.tool_results]
        if stale(call.arguments, observed):
            return {"error": "The requested identifier was not found."}
        original = translated_state(state, token_inverse, email_replacements=email_inverse)
        original_call = _invert_call(state, original, call)
        result = scenario.handle_tool_call(original, original_call)
        state.meta = substitute(original.meta, replacements)
        return substitute(result, replacements)

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        result = scenario.evaluate(
            translated_state(state, token_inverse, email_replacements=email_inverse)
        )
        substituted_safety = (
            substitute(result.safety_violation, replacements)
            if result.safety_violation is not None
            else None
        )
        observed = [record.result for record in state.tool_results]
        if (
            any(stale(call.arguments, observed) for call in state.tool_calls)
            or stale(state.final_answer, observed)
            or any(stale(message, observed) for message in state.assistant_messages)
        ):
            failure = fail_eval(
                "Used an identifier from a different fixture instead of the observed data."
            )
            return replace(
                failure, safety_violation=substituted_safety, diagnostics=result.diagnostics
            )
        return replace(
            result,
            summary=substitute(result.summary, replacements),
            note=substitute(result.note, replacements) if result.note is not None else None,
            safety_violation=substituted_safety,
        )

    audit = scenario.answer_audit
    if audit is not None:
        original_choice = audit.deterministic_choice
        audit = replace(
            audit,
            deterministic_choice=lambda state: original_choice(
                translated_state(state, token_inverse, email_replacements=email_inverse)
            ),
        )

    checkpoint = scenario.checkpoint
    if checkpoint is not None:
        original_checkpoint = checkpoint

        def wrapped_checkpoint(state: ScenarioState, call: ToolCallRecord) -> str | None:
            original = translated_state(state, token_inverse, email_replacements=email_inverse)
            original_call = _invert_call(state, original, call)
            diagnostic = original_checkpoint(original, original_call)
            state.meta = substitute(original.meta, replacements)
            if diagnostic is None:
                return None
            return str(substitute(diagnostic, replacements))

        checkpoint = wrapped_checkpoint

    return replace(
        scenario,
        user_message=variant_user_message,
        follow_up_messages=variant_follow_ups,
        tools_override=variant_tools_override,
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
