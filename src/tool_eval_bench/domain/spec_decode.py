"""Pure speculative-decoding arithmetic shared by runners and report renderers.

vLLM's ``--per-request-spec-decode-metrics detailed`` returns, per request,
the number of tokens drafted and accepted at every speculative step. This
module turns those step arrays into per-position acceptance rates without
depending on any transport or storage layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


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
