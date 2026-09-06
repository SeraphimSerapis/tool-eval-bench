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
    positive_argument_contains as _positive_argument_contains_shared,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)


def _positive_argument_contains(value: Any, expected: str) -> bool:
    """Match a requested entity without accepting an explicit negation."""
    return _positive_argument_contains_shared(value, expected)


def _numeric_value(value: Any) -> float | None:
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
    if not isinstance(payload, dict) or "error" in payload:
        return False
    return _normalize(_as_str(payload.get("status"))) in (
        statuses
        or {
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
    )


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Callable[[Any], bool],
) -> bool:
    """Validate a result when present, retaining synthetic trace compatibility."""
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results:
        results = exact_results
    elif any(result.call_id in known_call_ids for result in state.tool_results):
        return False
    else:
        results = _matching_tool_results(state, call)
    if not results:
        return True
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


_GERMAN_MARKERS = re.compile(
    r"\b(?:bewölkt\w*|wetter|grad|beträgt|liegt|derzeit|aktuell|heute|morgen|himmel|"
    r"regen|sonnig|schnee|und|ist|sind|es|im|der|das|für|nicht|bitte|leider|wird|"
    r"gerade|verfügbar|erneut|versuchen|dienst)\b",
    re.IGNORECASE,
)
