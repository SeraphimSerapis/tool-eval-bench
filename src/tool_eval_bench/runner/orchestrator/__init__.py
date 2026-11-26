"""Multi-turn scenario orchestrator.

Runs a single scenario against a model through the multi-turn tool-call loop:
  model → tool_calls → mock results → model → … (up to max_turns)

Captures full traces for auditability.
"""

from __future__ import annotations

from tool_eval_bench.runner.orchestrator.loop import (
    _CallbackError,
    _notify_turn,
    _tag_errors,
    _warm_pressure_cache,
    run_all_scenarios,
    run_scenario,
)
from tool_eval_bench.runner.orchestrator.probes import (
    _TOOL_CHOICE_PROBE_TOOLS,
    ToolCallWireProbe,
    ToolChoiceProbe,
    _strip_probe_max_tokens,
    probe_tool_call_parsing,
    probe_tool_choice_required,
    supports_tool_choice_required,
)
from tool_eval_bench.runner.orchestrator.review_hook import (
    _is_cut_off_tool_call,
    _review_payload,
    _review_turn,
    _storable_revision,
    _unusable_revision,
    _wire_tool_choice,
)
from tool_eval_bench.runner.orchestrator.scoring import (
    _capability_scores,
    score_results,
)
from tool_eval_bench.runner.orchestrator.wire import (
    _INJECTED_ERRORS,
    _LOOP_GUARD_TURNS,
    _MAX_INJECTED_ERROR_TURNS,
    _MUTATING_TOOL_PREFIXES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TURNS,
    _assistant_message,
    _classify_evaluation_failure,
    _classify_runtime_error,
    _detect_turn_degenerate_loop,
    _draw_injected_error,
    _ensure_unique_tool_call_ids,
    _format_calls_made,
    _format_trace,
    _HarnessExecutionError,
    _initial_messages,
    _is_mutating_tool,
    _is_visible_answer,
    _parse_tool_arguments,
    _parses_as_json,
    _producer_result_observed,
    _reject_non_finite_constant,
    _repair_json_str,
    _scenario_seed_offset,
    _tool_result_message,
    _turn_signature,
)

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_TURNS",
    "ToolCallWireProbe",
    "ToolChoiceProbe",
    "_CallbackError",
    "_HarnessExecutionError",
    "_INJECTED_ERRORS",
    "_LOOP_GUARD_TURNS",
    "_MAX_INJECTED_ERROR_TURNS",
    "_MUTATING_TOOL_PREFIXES",
    "_TOOL_CHOICE_PROBE_TOOLS",
    "_assistant_message",
    "_capability_scores",
    "_classify_evaluation_failure",
    "_classify_runtime_error",
    "_detect_turn_degenerate_loop",
    "_draw_injected_error",
    "_ensure_unique_tool_call_ids",
    "_format_calls_made",
    "_format_trace",
    "_initial_messages",
    "_is_cut_off_tool_call",
    "_is_mutating_tool",
    "_is_visible_answer",
    "_notify_turn",
    "_parse_tool_arguments",
    "_parses_as_json",
    "_producer_result_observed",
    "_reject_non_finite_constant",
    "_repair_json_str",
    "_review_payload",
    "_review_turn",
    "_scenario_seed_offset",
    "_storable_revision",
    "_strip_probe_max_tokens",
    "_tag_errors",
    "_tool_result_message",
    "_turn_signature",
    "_unusable_revision",
    "_warm_pressure_cache",
    "_wire_tool_choice",
    "probe_tool_call_parsing",
    "probe_tool_choice_required",
    "run_all_scenarios",
    "run_scenario",
    "score_results",
    "supports_tool_choice_required",
]
