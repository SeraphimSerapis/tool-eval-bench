"""Explain infrastructure failures, and say what to change.

A timeout is not a verdict. It leaves the scenario out of both the numerator and
the denominator, so the run says nothing about the model on that scenario. The
display still has to put *something* on the line, and "FAIL 0/2" followed by an
empty summary reads exactly like a model that got the answer wrong.

These helpers supply the missing half: what happened, and the flag that fixes it.
The advice is computed from the run's own measurements rather than canned, so it
names the timeout that was in force and the turn latency that overran it.
"""

from __future__ import annotations

from tool_eval_bench.domain.scenarios import (
    FailureKind,
    ModelScoreSummary,
    ScenarioResult,
)

_KIND_NOTES = {
    FailureKind.TIMEOUT: "excluded from scoring (request timed out)",
    FailureKind.CONNECTION_ERROR: "excluded from scoring (connection failed)",
    FailureKind.SERVER_ERROR: "excluded from scoring (server error)",
}
_FALLBACK_NOTE = "excluded from scoring (infrastructure failure)"

# Headroom over the slowest observed turn. A model that needed 151s for one turn
# will not reliably fit in 151s on the next, so the suggestion has to clear the
# measurement rather than match it.
_SUGGESTION_HEADROOM = 2.5

# Round suggestions up to this many seconds so the advice reads like a setting
# somebody would type.
_SUGGESTION_STEP = 60


def infrastructure_note(result: ScenarioResult) -> str:
    """One-line reason for a scenario that never reached the evaluator."""
    if result.summary:
        return result.summary
    return _KIND_NOTES.get(result.failure_kind or "", _FALLBACK_NOTE)


def suggested_timeout(slowest_turn_ms: float, configured_seconds: float) -> int:
    """Return a timeout that clears the slowest turn actually observed."""
    target = max(configured_seconds, slowest_turn_ms / 1000 * _SUGGESTION_HEADROOM)
    steps = int(target // _SUGGESTION_STEP) + 1
    return steps * _SUGGESTION_STEP


def timeout_advice(
    summary: ModelScoreSummary,
    *,
    timeout_seconds: float | None = None,
) -> list[str]:
    """Return advice lines for timed-out scenarios, or an empty list.

    Only timeouts get advice. A connection error or a 5xx is a different
    problem, and telling somebody to raise ``--timeout`` would send them the
    wrong way.
    """
    timed_out = [
        r
        for r in summary.scenario_results
        if r.is_infrastructure_failure and r.failure_kind == FailureKind.TIMEOUT
    ]
    if not timed_out:
        return []

    lines = [
        f"{len(timed_out)} scenario(s) timed out and were excluded from scoring, "
        "so this run does not measure the model on them.",
    ]

    multi_turn = [r for r in timed_out if r.turn_latencies_ms]
    slowest = max((max(r.turn_latencies_ms) for r in multi_turn), default=0.0)

    if slowest > 0:
        lines.append(
            f"Slowest completed turn: {slowest / 1000:.0f}s. "
            "Only the first turn is streamed, so on later turns the timeout "
            "bounds the whole generation rather than the gap between tokens."
        )

    if timeout_seconds and slowest > 0:
        suggestion = suggested_timeout(slowest, timeout_seconds)
        lines.append(
            f"Retry with --timeout {suggestion} (currently {timeout_seconds:.0f}), "
            "or shorten generation with --no-think or "
            "--backend-kwargs '{\"max_tokens\": 1024}'."
        )
    else:
        lines.append(
            "Retry with a higher --timeout, or shorten generation with --no-think "
            "or --backend-kwargs '{\"max_tokens\": 1024}'."
        )

    return lines
