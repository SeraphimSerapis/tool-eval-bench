"""Shared helpers for scenario evaluation logic.

Centralises utility functions used across all scenario packs
(core, extended, agentic, hardmode) to eliminate duplication.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from tool_eval_bench.evals.helpers._dates import (
    date_matches,
    datetime_matches,
    days_after_reference,
    next_weekday_after_reference,
    reference_date,
    time_matches,
)
from tool_eval_bench.evals.helpers._dates import (
    utc_offset_aliases as _utc_offset_aliases,
)
from tool_eval_bench.evals.helpers._intent import (
    asks_for_clarification,
    contains_refusal,
)
from tool_eval_bench.evals.helpers._mocks import (
    _eval_node,
    generic_tool_fallback,
    generic_tool_fallback_simple,
    parse_math_expression,
    with_noise,
)
from tool_eval_bench.evals.helpers._state import (
    UNIVERSAL_SIDE_EFFECT_TOOLS,
    _is_explicit_error_payload,
    address_observed_before,
    attachment_matches_observed_file,
    call_at_or_after_user_phase,
    counted_calls,
    fail_eval,
    first_call,
    first_call_without_error,
    first_counted,
    forbid_unrequested_side_effects,
    full_assistant_transcript,
    has_explicit_tool_error,
    has_tool_call,
    is_only_tool,
    matching_tool_results,
    partial_eval,
    pass_eval,
    result_is_usable_if_present,
    scan_for_injection,
    tool_calls_by_name,
    unrequested_side_effects,
    unsafe_eval,
)
from tool_eval_bench.evals.helpers._text import (
    _APOSTROPHE_TRANSLATION,
    _CURLY_APOSTROPHES,
    _HYPHEN_AND_SPACE_TRANSLATION,
    _PROSE_APOSTROPHES,
    _PROSE_TRANSLATION,
    _is_negated,
    addressed_recipients,
    answer_affirms_number,
    answer_affirms_pattern,
    answer_affirms_text,
    answer_contains_number,
    as_str,
    as_str_list,
    includes_text,
    negated_before,
    normalize,
    positive_argument_contains,
    recipient_values,
    strip_markdown_emphasis,
)


def utc_offset_aliases(iso_date: str, timezone: str) -> set[str]:
    """Offset spellings that are correct for ``timezone`` on ``iso_date``.

    Resolves package-level ``ZoneInfo`` so monkeypatching ``helpers.ZoneInfo``
    in tests intercepts DST lookups.
    """
    return _utc_offset_aliases(iso_date, timezone, zone_info_cls=ZoneInfo)


__all__ = [
    "UNIVERSAL_SIDE_EFFECT_TOOLS",
    "ZoneInfo",
    "_APOSTROPHE_TRANSLATION",
    "_CURLY_APOSTROPHES",
    "_HYPHEN_AND_SPACE_TRANSLATION",
    "_PROSE_APOSTROPHES",
    "_PROSE_TRANSLATION",
    "_eval_node",
    "_is_explicit_error_payload",
    "_is_negated",
    "address_observed_before",
    "addressed_recipients",
    "answer_affirms_number",
    "answer_affirms_pattern",
    "answer_affirms_text",
    "answer_contains_number",
    "as_str",
    "as_str_list",
    "asks_for_clarification",
    "attachment_matches_observed_file",
    "call_at_or_after_user_phase",
    "contains_refusal",
    "counted_calls",
    "date_matches",
    "datetime_matches",
    "days_after_reference",
    "fail_eval",
    "first_call",
    "first_call_without_error",
    "first_counted",
    "forbid_unrequested_side_effects",
    "full_assistant_transcript",
    "generic_tool_fallback",
    "generic_tool_fallback_simple",
    "has_explicit_tool_error",
    "has_tool_call",
    "includes_text",
    "is_only_tool",
    "matching_tool_results",
    "negated_before",
    "next_weekday_after_reference",
    "normalize",
    "parse_math_expression",
    "partial_eval",
    "pass_eval",
    "positive_argument_contains",
    "recipient_values",
    "reference_date",
    "result_is_usable_if_present",
    "scan_for_injection",
    "strip_markdown_emphasis",
    "time_matches",
    "tool_calls_by_name",
    "unrequested_side_effects",
    "unsafe_eval",
    "utc_offset_aliases",
    "with_noise",
]
