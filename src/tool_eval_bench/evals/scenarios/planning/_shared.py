"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
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
    recipient_values as _recipient_values,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)


def _result_matches_if_present(
    state: ScenarioState,
    call: ToolCallRecord,
    predicate: Any,
) -> bool:
    """Validate an explicit result while preserving synthetic trace support.

    Direct evaluator tests and imported historical traces may contain tool
    calls without result records.  Runtime traces always have results, so an
    explicit result must be both successful and consistent with the fixture
    the scenario promises.  If the trace has stable IDs, never borrow a
    same-named result from another call.
    """
    exact_results = [result for result in state.tool_results if result.call_id == call.id]
    known_call_ids = {candidate.id for candidate in state.tool_calls}
    if exact_results and any(result.name == call.name for result in exact_results):
        results = exact_results
    elif exact_results and all(result.name in {"", "unknown"} for result in exact_results):
        # Older synthetic tests attached a bare payload to call_0 without a
        # tool name.  It is evidence for the first call only when the caller
        # can identify that name, otherwise leave it unknown.
        results = []
    elif exact_results and all(
        any(candidate.name == result.name for candidate in state.tool_calls)
        for result in exact_results
    ):
        # Legacy fixtures sometimes default every result call_id to call_0.
        # If the named result belongs to another recorded call, do not let it
        # masquerade as this call's payload.
        results = []
    elif any(
        result.call_id in known_call_ids and result.name == call.name
        for result in state.tool_results
    ):
        return False
    else:
        results = _matching_tool_results(state, call)
    if not results:
        return True
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


def _result_has_status(payload: Any, status: str, identifier: str | None = None) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status", "")).strip().lower() != status:
        return False
    return identifier is None or identifier in str(payload)


def _has_unexpected_tools(state: ScenarioState, allowed: set[str]) -> bool:
    return any(call.name not in allowed for call in state.tool_calls)


def _call_index(state: ScenarioState, target: ToolCallRecord) -> int:
    """Find a call by identity, not dataclass equality, for dependency checks."""
    return next(index for index, call in enumerate(state.tool_calls) if call is target)


_UNRELATED_UNIVERSAL_MUTATIONS = frozenset({"set_reminder", "run_code"})


def _recipient_set(value: Any) -> set[str]:
    """Distinct addresses named by a recipient argument, string or array."""
    return set(_recipient_values(value))


def _is_tomorrow_morning(datetime_value: Any, state: ScenarioState) -> bool:
    """Return whether a set_reminder datetime is semantically tomorrow morning.

    Accepts either natural-language text ("tomorrow morning") or an ISO
    timestamp that resolves to the next calendar day in a morning window
    relative to the scenario reference date in ``state.meta``.

    Morning window (ISO path only): 05:00 inclusive through 12:00 exclusive
    (``5 <= hour < 12``). Timezone offsets are ignored — only calendar date and
    hour are compared (same ignore-offset idea as ``datetime_matches``, but
    this helper uses a next-day hour *window*, not an exact ``HH:MM`` match).
    The natural-language path keeps the historical substring check and does
    not enforce the hour window, so existing full-workflow tests stay green.
    """
    if not datetime_value:
        return False
    dt_str = _as_str(datetime_value).strip().lower()
    if not dt_str:
        return False

    # Natural-language form (backward compatible; hour window not applied).
    if re.search(r"\btomorrow\s+morning\b", dt_str) and not re.search(
        r"\b(?:not|never|day after|the day after)\s+tomorrow\b", dt_str
    ):
        return True

    ref = state.meta.get("reference_date")
    if ref is None:
        return False
    try:
        ref_dt = datetime.fromisoformat(str(ref).strip())
    except Exception:
        return False

    # Normalise ISO timestamps: strip a trailing Z and parse as naive
    # (Z means UTC, but for calendar-day semantics we compare dates only).
    parse_str = dt_str
    if parse_str.endswith("z"):
        parse_str = parse_str[:-1]
    try:
        target = datetime.fromisoformat(parse_str)
    except Exception:
        return False
    if ref_dt.tzinfo is None and target.tzinfo is not None:
        target = target.replace(tzinfo=None)
    if ref_dt.tzinfo is not None and target.tzinfo is None:
        target = target.replace(tzinfo=ref_dt.tzinfo)

    next_day = date(ref_dt.year, ref_dt.month, ref_dt.day) + timedelta(days=1)
    if target.date() != next_day:
        return False
    hour = target.time().hour
    return 5 <= hour < 12
