"""Metrics over scored decisions: distance from a gold distribution and calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HIGH_CONFIDENCE = 0.9


@dataclass(frozen=True)
class ReliabilityBin:
    """Predictions whose confidence falls in ``[low, high)``, the last bin closed."""

    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    confidences: Sequence[float], corrects: Sequence[bool], *, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Group predictions by confidence and compare confidence to accuracy.

    Empty bins are omitted.  A well-calibrated model has ``accuracy`` close to
    ``mean_confidence`` in every bin.
    """
    if len(confidences) != len(corrects):
        raise ValueError("confidences and corrects must have the same length")
    members: list[list[int]] = [[] for _ in range(n_bins)]
    for i, conf in enumerate(confidences):
        members[min(int(conf * n_bins), n_bins - 1)].append(i)
    bins = []
    for b, idx in enumerate(members):
        if not idx:
            continue
        bins.append(
            ReliabilityBin(
                low=b / n_bins,
                high=(b + 1) / n_bins,
                count=len(idx),
                mean_confidence=sum(confidences[i] for i in idx) / len(idx),
                accuracy=sum(corrects[i] for i in idx) / len(idx),
            )
        )
    return bins


def expected_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Count-weighted mean gap between confidence and accuracy, in [0, 1]."""
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    return sum(b.count * abs(b.accuracy - b.mean_confidence) for b in bins) / total


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Stands in for a predicted probability of exactly 0 where the gold is not 0,
# which would make KL infinite.  The value matches the third-party scorer that
# reproduced the typed-decisions card's KL figures: prima-ratio's
# benchmarks/typed_decisions_bench.py, from discussion #5 on the dataset page.
KL_FLOOR = 1e-12


def kl_divergence(
    gold: Mapping[str, float], predicted: Mapping[str, float], *, floor: float = KL_FLOOR
) -> float:
    """KL(gold ‖ predicted) in nats, summed over the answers gold supports.

    An answer with gold probability 0 contributes nothing.  A predicted
    probability below ``floor``, including a missing answer, counts as ``floor``.
    """
    return sum(
        g * math.log(g / max(predicted.get(k, 0.0), floor)) for k, g in gold.items() if g > 0
    )


def brier_score(gold: Mapping[str, float], predicted: Mapping[str, float]) -> float:
    """Squared distance between two distributions over the same answers, in [0, 2]."""
    keys = gold.keys() | predicted.keys()
    return sum((predicted.get(k, 0.0) - gold.get(k, 0.0)) ** 2 for k in keys)


def argmax(distribution: Mapping[str, float]) -> str:
    """The most likely answer.  A tie goes to the answer listed first."""
    return max(distribution, key=lambda k: distribution[k])


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]; 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
