"""How long a turn is allowed to take.

The orchestrator streams the first turn of a scenario and no others. That one
detail decides what a request timeout actually bounds, so the rule for turns
after the first lives here rather than being inlined at the call site.
"""

from __future__ import annotations

# Headroom applied to turn 1's measured latency when bounding a later,
# unstreamed turn. Later turns carry a longer prompt and, for reasoning models,
# often more to say, so matching turn 1 exactly would leave no margin.
UNSTREAMED_TURN_HEADROOM = 2.5


def unstreamed_turn_timeout(configured_seconds: float, first_turn_ms: float) -> float:
    """Return the timeout to allow for a turn that is not streamed.

    ``httpx``'s read timeout measures the gap between reads, not total elapsed
    time. On a streamed turn, tokens keep arriving and a long generation never
    trips it. On an unstreamed turn the whole response arrives at once, so the
    same number bounds the entire generation. A model comfortably inside the
    timeout on turn 1 can therefore blow it on turn 2 without having slowed
    down at all.

    Turn 1 is a free measurement of this model's speed on this exact prompt, and
    being streamed it cannot time out spuriously, so later turns get a multiple
    of what turn 1 actually took. A hung endpoint never completes turn 1, so it
    still fails at the configured timeout instead of being handed more rope.

    Returns ``configured_seconds`` unchanged when turn 1 produced no usable
    measurement.
    """
    if first_turn_ms <= 0:
        return configured_seconds
    return max(configured_seconds, first_turn_ms / 1000 * UNSTREAMED_TURN_HEADROOM)
