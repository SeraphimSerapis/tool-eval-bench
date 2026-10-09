"""Pure speculative-decoding arithmetic shared by runners and report renderers.

vLLM's ``--per-request-spec-decode-metrics detailed`` returns, per request,
the number of tokens drafted and accepted at every speculative step. This
module turns those step arrays into per-position acceptance rates, and
derives draft-window utilization, without depending on any transport or
storage layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def window_utilization(acceptance_length: float, draft_window: float) -> float | None:
    """Fraction of drafted positions the verifier accepted: ``(τ − 1) ÷ window``.

    τ follows vLLM's convention and counts the verifier's bonus token, which
    was never drafted, so it is subtracted before dividing by the window.
    Because the draft window is drafted tokens per step, this equals α for
    counter backends; the report shows it next to the window to make the
    tuning question explicit. Returns None for an empty window.
    """
    if draft_window <= 0:
        return None
    return max(acceptance_length - 1.0, 0.0) / draft_window


def suggested_draft_window(acceptance_length: float, draft_window: float) -> int | None:
    """A smaller ``num_speculative_tokens`` worth trying, or None.

    The suggestion leaves 50% headroom over the accepted drafts per step
    (``τ − 1``), with a floor of 2. None when that would not shrink the
    current window, so a window of 2 is never told to "reduce" to 2.
    """
    optimal = max(int((acceptance_length - 1.0) * 1.5), 2)
    return optimal if optimal < round(draft_window) else None


def per_position_acceptance(
    steps: Iterable[tuple[Sequence[int], Sequence[int]]],
) -> list[float]:
    """Acceptance rate at each draft position, aggregated over step arrays.

    Each item in *steps* is ``(per_step_accepted, per_step_drafted)`` for one
    request. Position ``p`` counts as drafted in a step when that step drafted
    more than ``p`` tokens, and as accepted when more than ``p`` were accepted.
    Verification stops at the first rejection, so accepted counts are prefix
    lengths and this is exact rather than an estimate.

    Returns one rate per position up to the longest draft seen. Requests whose
    two arrays differ in length are skipped: they cannot be paired step by
    step, and silently truncating would bias later positions.
    """
    drafted_at: list[int] = []
    accepted_at: list[int] = []
    for accepted, drafted in steps:
        if len(accepted) != len(drafted):
            continue
        for a, d in zip(accepted, drafted, strict=True):
            if d > len(drafted_at):
                drafted_at.extend([0] * (d - len(drafted_at)))
                accepted_at.extend([0] * (d - len(accepted_at)))
            for pos in range(d):
                drafted_at[pos] += 1
            for pos in range(min(a, d)):
                accepted_at[pos] += 1
    return [acc / drf if drf else 0.0 for acc, drf in zip(accepted_at, drafted_at, strict=True)]
