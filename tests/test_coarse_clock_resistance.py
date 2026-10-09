"""Nothing that must vary may take its variation from a coarse clock.

`time.time_ns()` and `time.monotonic()` advance in roughly 15.6ms steps on
Windows, against nanoseconds on Linux. Code that seeds a generator from the
clock, or measures a short interval with it, is therefore correct on the CI
that runs it and wrong on the platform nobody tested.
"""

from __future__ import annotations

import time

import pytest

from tool_eval_bench.runner.context_pressure import (
    ContextPressureConfig,
    build_pressure_messages,
)

FROZEN_NS = 1_756_000_000_000_000_000


@pytest.fixture
def frozen_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze `time.time_ns`, the way a 15.6ms tick freezes it in practice.

    `context_pressure` imports `time` inside the function under test, so the
    stdlib module is the only place to patch.
    """
    monkeypatch.setattr(time, "time_ns", lambda: FROZEN_NS)
    monkeypatch.setattr(time, "time", lambda: FROZEN_NS / 1e9)


def _user_content(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def test_unseeded_pressure_filler_varies_even_when_the_clock_does_not(frozen_wall_clock) -> None:
    """Identical filler between levels would hand the server a warm prefix cache.

    The noise injection exists to defeat prefix caching. Seeding it from
    `time.time_ns()` meant two builds inside one clock tick produced
    byte-identical text, which is the condition the noise is there to prevent.
    """
    config = ContextPressureConfig(ratio=0.5, fill_tokens=5000, detected_context=32768)

    first = _user_content(build_pressure_messages(config))
    second = _user_content(build_pressure_messages(config))

    differences = sum(1 for a, b in zip(first, second, strict=False) if a != b)
    assert differences > 0, "two builds produced identical filler under a frozen clock"


def test_seeded_pressure_filler_stays_reproducible() -> None:
    """The unseeded path may vary freely; the seeded path may not."""
    config = ContextPressureConfig(ratio=0.5, fill_tokens=5000, detected_context=32768)

    assert _user_content(build_pressure_messages(config, seed=7)) == _user_content(
        build_pressure_messages(config, seed=7)
    )
