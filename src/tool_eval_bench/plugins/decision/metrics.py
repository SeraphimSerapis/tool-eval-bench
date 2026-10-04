"""Aggregate metrics over scored decision items: calibration, ordinal error, robustness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

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


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]; 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def confusion_matrix(
    pairs: Sequence[tuple[str, str]], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Count ``(gold, predicted)`` pairs.  Predictions outside ``labels`` are counted
    under their own name, so a hallucinated option is visible rather than dropped."""
    matrix: dict[str, dict[str, int]] = {label: {} for label in labels}
    for gold, predicted in pairs:
        row = matrix.setdefault(gold, {})
        row[predicted] = row.get(predicted, 0) + 1
    return matrix


@dataclass(frozen=True)
class Robustness:
    """How stable answers are when the options are reordered or renamed.

    ``invariance`` is the share of variant items whose answer, mapped back to
    the base item's option names, equals the base item's answer.  Accuracy and
    invariance differ: a model can be wrong the same way twice.
    """

    shuffled_accuracy: float | None
    shuffled_invariance: float | None
    opaque_accuracy: float | None
    opaque_invariance: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def robustness(
    base: Mapping[str, tuple[str, bool]],
    variants: Mapping[str, Sequence[tuple[str, str, bool]]],
) -> Robustness:
    """Compute variant accuracy and invariance against the base answers.

    ``base`` maps a base item id to ``(predicted, correct)``.  ``variants`` maps
    a variant kind to ``(base_id, predicted_in_base_names, correct)`` triples.
    A kind with no answered pairs yields ``None`` for both measures.
    """

    def measure(kind: str) -> tuple[float | None, float | None]:
        rows = [(b, p, c) for b, p, c in variants.get(kind, ()) if b in base]
        if not rows:
            return None, None
        accuracy = sum(c for _, _, c in rows) / len(rows) * 100
        invariance = sum(p == base[b][0] for b, p, _ in rows) / len(rows) * 100
        return accuracy, invariance

    shuffled = measure("shuffled")
    opaque = measure("opaque")
    return Robustness(shuffled[0], shuffled[1], opaque[0], opaque[1])
