"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    normalize as _normalize,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Callable[[Any], bool],
) -> bool:
    """Validate an explicit result while preserving old synthetic traces.

    Runtime traces always contain a result for every call.  A number of direct
    evaluator tests and imported traces intentionally contain calls only, so
    an absent result remains unknown rather than becoming a failure.  When a
    result is present, however, it must both be non-error and describe the
    value that the call is meant to provide.
    """
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results:
        results = exact_results
    elif any(result.call_id in known_call_ids for result in state.tool_results):
        # Runtime traces use stable IDs.  Once a trace proves that it is
        # ID-aware, a result from another same-named call must not be borrowed
        # to make this call look successful.
        return False
    else:
        results = _matching_tool_results(state, call)
    if not results:
        return True
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


def _positive_argument_contains(value: Any, expected: str) -> bool:
    """Match an entity in a tool argument without accepting a negation.

    Scenario arguments are model-authored input, not prose to summarize.  A
    substring check such as ``"not Berlin"`` or ``"not Sarah"`` must not be
    treated as the requested entity.  Keep the accepted surface broad enough
    for normal qualifiers such as ``Berlin, Germany`` while rejecting an
    explicit denial immediately before the entity.
    """
    text = _as_str(value).strip().lower()
    target = expected.strip().lower()
    if not text or not re.search(rf"(?<!\w){re.escape(target)}(?!\w)", text):
        return False
    return not re.search(
        rf"\b(?:not|no|without|except|exclude|excluding)\s+(?:the\s+)?"
        rf"{re.escape(target)}\b",
        text,
    )


def _numeric_value(value: Any) -> float | None:
    """Parse a scalar numeric tool result without accepting arbitrary text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _as_str(value).strip().replace(",", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def _status_is_success(payload: Any, *, statuses: set[str] | None = None) -> bool:
    """Return whether a structured mutation result explicitly succeeded."""
    if not isinstance(payload, dict) or "error" in payload:
        return False
    status = _normalize(_as_str(payload.get("status")))
    accepted = statuses or {
        "accepted",
        "created",
        "delivered",
        "ok",
        "queued",
        "scheduled",
        "sent",
        "set",
        "success",
    }
    return status in accepted


def _attendee_matches(value: str, name: str) -> bool:
    """Match a requested attendee by name or email local part."""
    text = value.strip().lower()
    target = name.strip().lower()
    if text == target or text.startswith(f"{target}@"):
        return True
    return bool(re.search(rf"(?<!\w){re.escape(target)}(?!\w)", text))


def _tc03_email_result_is_sent(payload: Any) -> bool:
    """Return whether an explicit send result confirms a successful delivery."""
    if not isinstance(payload, dict):
        return False
    return _normalize(_as_str(payload.get("status"))) in {
        "accepted",
        "delivered",
        "ok",
        "queued",
        "sent",
        "success",
    }


_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "english",
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
    "en-ca": "english",
    "en-au": "english",
    "spanish": "spanish",
    "es": "spanish",
    "es-es": "spanish",
    "es-419": "spanish",
    "espanol": "spanish",
    "español": "spanish",
    "castilian": "spanish",
    "spa": "spanish",
    "japanese": "japanese",
    "ja": "japanese",
    "ja-jp": "japanese",
    "日本語": "japanese",
    "jpn": "japanese",
}


def _language_alias(value: Any) -> str | None:
    """Resolve a language designator to the canonical language name used by
    the mock, or None when the string is not a language designator."""
    return _LANGUAGE_ALIASES.get(_normalize(_as_str(value)))


def _tc14_result_is_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        status = _normalize(_as_str(payload.get("status")))
        return "error" in payload or status in {
            "blocked",
            "cancelled",
            "canceled",
            "error",
            "failed",
        }
    return bool(re.search(r"\berror\b|\b(?:429|500|502|503|504)\b", _as_str(payload), re.I))


def _result_matches_error_if_present(state: ScenarioState, call: ToolCallRecord) -> bool:
    """Match an expected explicit error while keeping synthetic traces valid."""
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results:
        return any(_tc14_result_is_error(result.result) for result in exact_results)
    if any(result.call_id in known_call_ids for result in state.tool_results):
        return False
    results = _matching_tool_results(state, call)
    return not results or any(_tc14_result_is_error(result.result) for result in results)
