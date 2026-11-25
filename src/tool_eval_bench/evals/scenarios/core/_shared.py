"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    _is_negated,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    has_explicit_tool_error as _has_explicit_tool_error,
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
        return state.meta.get("trace_policy") != "complete"
    return _result_is_usable_if_present(state, call) and any(
        predicate(result.result) for result in results
    )


def _positive_argument_contains(value: Any, expected: str) -> bool:
    """Match a requested entity without accepting an explicit negation."""
    return _positive_argument_contains_shared(value, expected)


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
    """Match a requested attendee by name or email local part without negation."""
    text = value.strip().lower()
    target = name.strip().lower()
    if re.search(
        rf"\b(?:not|no|without|except|exclude|excluding)\s+(?:the\s+)?{re.escape(target)}\b", text
    ):
        return False
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


def _only_error_retries_before_last(state: ScenarioState, calls: list[ToolCallRecord]) -> bool:
    """Whether every call except the last came back with an explicit error.

    Repeating a correct call after a failure is recovery. Repeating it after a
    usable result is a redundant duplicate.
    """
    return all(_has_explicit_tool_error(state, call) for call in calls[:-1])


_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?:,\d{3})*(?!\d))")
_STATED_NUMBER = re.compile(r"(?<![\d.])(?:([-\u2212])[$€£¥]?)?(\d+)(?:\.(\d+))?(?!\d|\.\d)")
_CLOCK_BEFORE = re.compile(r"\d:$")
_CLOCK_AFTER = re.compile(r":\d{2}\b")
_PERCENT_AFTER = re.compile(r"\s*(?:%|(?:percent|prozent|por\s+ciento)\b)", re.IGNORECASE)
_SCALE_SUFFIX_AFTER = re.compile(
    r"\s*(?:[kmb]\b|mm\b|mil\b|thousand\b|million\b|billion\b)", re.IGNORECASE
)


def _parsed_stated_number(match: re.Match[str]) -> Decimal:
    sign = "-" if match.group(1) else ""
    fractional = f".{match.group(3)}" if match.group(3) is not None else ""
    return Decimal(f"{sign}{match.group(2)}{fractional}")


def _excluded_stated_match(
    collapsed: str,
    match: re.Match[str],
    *,
    exclude_clock_times: bool,
    exclude_percent: bool,
    exclude_scale_suffix: bool,
) -> bool:
    before = collapsed[: match.start()]
    after = collapsed[match.end() :]
    if exclude_clock_times and (_CLOCK_AFTER.match(after) or _CLOCK_BEFORE.search(before)):
        return True
    if exclude_percent and _PERCENT_AFTER.match(after):
        return True
    if exclude_scale_suffix and _SCALE_SUFFIX_AFTER.match(after):
        return True
    return False


def _answer_affirms_rounded_value(
    answer: str,
    value: str,
    *,
    exclude_clock_times: bool = True,
    exclude_percent: bool = True,
    exclude_scale_suffix: bool = False,
) -> bool:
    """Whether ``answer`` asserts ``value`` at the precision it chose.

    A stated number passes when it equals ``value`` rounded or truncated to
    the number of decimals the answer wrote, so 187, 187.4 and 187.42 all
    report a price of 187.42 while 187.99 and -187.42 do not. Negated
    mentions ("not 187.42") are rejected.
    """
    collapsed = _THOUSANDS_COMMA.sub("", answer)
    expected = Decimal(value)
    exp = expected.as_tuple().exponent
    expected_decimals = max(0, -exp) if isinstance(exp, int) else 0
    for match in _STATED_NUMBER.finditer(collapsed):
        if _excluded_stated_match(
            collapsed,
            match,
            exclude_clock_times=exclude_clock_times,
            exclude_percent=exclude_percent,
            exclude_scale_suffix=exclude_scale_suffix,
        ):
            continue
        stated = _parsed_stated_number(match)
        frac = match.group(3) or ""
        if len(frac) > expected_decimals:
            matches_expected = stated == expected
        else:
            quantum = Decimal(1).scaleb(-len(frac))
            try:
                accepted = {
                    expected.quantize(quantum, rounding=ROUND_HALF_UP),
                    expected.quantize(quantum, rounding=ROUND_DOWN),
                }
                matches_expected = stated in accepted
            except InvalidOperation:
                matches_expected = stated == expected
        if matches_expected and not _is_negated(
            collapsed[max(0, match.start() - 120) : match.start()]
        ):
            return True
    return False


def _answer_affirms_exact_value(
    answer: str,
    value: str,
    *,
    exclude_clock_times: bool = False,
    exclude_percent: bool = False,
    exclude_scale_suffix: bool = False,
) -> bool:
    """Whether ``answer`` asserts ``value`` with exact numeric equality.

    Accepts equivalent trailing zeros such as ``4.40`` for ``4.4`` or
    ``4,400,000.00`` for ``4400000``, while rejecting rounded-away digits
    (``4``), wrong decimals (``4.45``), wrong signs (``-4.4``), and negations.
    """
    collapsed = _THOUSANDS_COMMA.sub("", answer)
    expected = Decimal(value)
    for match in _STATED_NUMBER.finditer(collapsed):
        if _excluded_stated_match(
            collapsed,
            match,
            exclude_clock_times=exclude_clock_times,
            exclude_percent=exclude_percent,
            exclude_scale_suffix=exclude_scale_suffix,
        ):
            continue
        if _parsed_stated_number(match) == expected and not _is_negated(
            collapsed[max(0, match.start() - 120) : match.start()]
        ):
            return True
    return False
