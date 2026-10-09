"""Withhold held-out scenario content from machine-readable output.

The Markdown report already withholds titles, summaries, and traces for
scenarios loaded from a ``--scenario-pack``. ``--json``, ``--json-file``, and
the JSONL progress events on stderr apply the same rule unless the user passes
``--include-held-out``, because JSON output is what CI pipelines upload and
publishing a held-out trace burns the scenario. Full results stay in SQLite.

Result fields are kept by allowlist, so a field added to ``ScenarioResult``
later is withheld by default instead of leaking until someone notices.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from tool_eval_bench.cli.commands import resolve_pack_scenarios

HELD_OUT = "held out"

# Scores, timings, and counts: what the report publishes, plus numbers that say
# nothing about the prompt or the expected tool calls.
_PUBLISHED_FIELDS = frozenset(
    {
        "scenario_id",
        "status",
        "points",
        "failure_kind",
        "duration_seconds",
        "turn_count",
        "turn_budget_exceeded",
        "ttft_ms",
        "turn_latencies_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tool_call_arg_bytes",
    }
)


def held_out_ids(args: argparse.Namespace) -> frozenset[str]:
    """IDs whose content JSON output must withhold; empty with ``--include-held-out``."""
    if getattr(args, "include_held_out", False):
        return frozenset()
    return frozenset(scenario.id for scenario in resolve_pack_scenarios(args))


def redact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a held-out scenario result with only its published fields.

    The text fields every result carries keep their keys, with a placeholder
    value, so a consumer that reads ``summary`` or ``raw_log`` does not break.
    """
    redacted = {key: value for key, value in result.items() if key in _PUBLISHED_FIELDS}
    redacted.update(
        {
            "summary": HELD_OUT,
            "note": None,
            "tool_calls_made": [],
            "expected_behavior": HELD_OUT,
            "raw_log": HELD_OUT,
            "held_out": True,
        }
    )
    if result.get("safety_violation"):
        redacted["safety_violation"] = HELD_OUT
    return redacted


def redact_warnings(warnings: Iterable[str], ids: frozenset[str]) -> list[str]:
    """Replace the title and violation of held-out safety warnings.

    Warnings read ``"{id} ({title}): {violation}"``; the ID and the count stay.
    """
    redacted = []
    for warning in warnings:
        held_id = next((sid for sid in ids if warning.startswith(f"{sid} (")), None)
        redacted.append(f"{held_id}: {HELD_OUT}" if held_id is not None else warning)
    return redacted


def redact_run(run: dict[str, Any], ids: frozenset[str]) -> dict[str, Any]:
    """Return a copy of a run result with held-out scenario content withheld."""
    if not ids:
        return run
    redacted = dict(run)
    scores = run.get("scores")
    if isinstance(scores, dict):
        scores = dict(scores)
        if isinstance(scores.get("scenario_results"), list):
            scores["scenario_results"] = [
                redact_result(r) if isinstance(r, dict) and r.get("scenario_id") in ids else r
                for r in scores["scenario_results"]
            ]
        if scores.get("safety_warnings"):
            scores["safety_warnings"] = redact_warnings(scores["safety_warnings"], ids)
        redacted["scores"] = scores
    if run.get("safety_warnings"):
        redacted["safety_warnings"] = redact_warnings(run["safety_warnings"], ids)
    gate = run.get("safety_gate")
    if isinstance(gate, dict) and gate.get("warnings"):
        redacted["safety_gate"] = {**gate, "warnings": redact_warnings(gate["warnings"], ids)}
    return redacted
