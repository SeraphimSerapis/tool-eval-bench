"""Interval estimates for the toolscale report.

Deterministic on purpose: a bootstrap seeded from a constant gives the same
interval for the same data, so two reports of one run never disagree.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import fmean, stdev


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, in percent.

    Wilson rather than the normal approximation because cells near 0% or 100%
    are common here, and the normal interval collapses to a point there.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1))


def sample_stddev(values: Sequence[float]) -> float:
    """Sample standard deviation, 0 for fewer than two values."""
    return stdev(values) if len(values) >= 2 else 0.0


def bootstrap_mean_interval(
    values: Sequence[float], *, resamples: int = 2000, ci: float = 0.95
) -> tuple[float, float]:
    """Percentile-bootstrap interval for the mean of *values*.

    The toolscale contrasts are paired: each value is one target's difference
    between two cells, so resampling targets keeps each pair together.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(104729)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(resamples))
    alpha = 1 - ci
    low = means[int(alpha / 2 * resamples)]
    high = means[int((1 - alpha / 2) * resamples) - 1]
    return (low, high)


def slope_per_doubling(points: Sequence[tuple[int, float]]) -> float | None:
    """Least-squares slope of accuracy against log2(tool count).

    Read as "points of accuracy lost (negative) or gained per doubling of the
    tool list". ``None`` when fewer than two distinct counts were measured.
    """
    xs = [math.log2(count) for count, _ in points]
    ys = [acc for _, acc in points]
    if len(set(xs)) < 2:
        return None
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx
