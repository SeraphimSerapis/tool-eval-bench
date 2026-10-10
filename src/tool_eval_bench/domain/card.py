"""The data a shareable results card renders.

A card summarises one benchmark session on a single image: speed, long-context
behaviour, retrieval, and tool-calling quality. ``application.card_data``
assembles a :class:`CardData` from stored runs and ``storage.reports.card``
renders it. Everything here is already safe to publish: no session id, run id,
hostname, or server URL belongs on a card, so none has a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CardLayout = Literal["portrait", "landscape"]


@dataclass(frozen=True)
class PerfPoint:
    """One throughput cell, averaged over its measurement runs.

    ``tg_tps`` is per-request decode speed; ``tg_tps_total`` is the aggregate
    across all concurrent requests. ``ttft_ms`` is the cold time to first token.
    ``cached_ttft_ms`` is the time to first token when the depth prefix was
    already cached, and is ``None`` unless the sweep measured it.
    """

    depth: int
    concurrency: int
    pp_tps: float
    tg_tps: float
    tg_tps_total: float
    ttft_ms: float
    cached_ttft_ms: float | None = None
    pp_estimated: bool = False


@dataclass(frozen=True)
class NeedleCell:
    context_tokens: int
    depth_percent: float
    found: bool


@dataclass(frozen=True)
class NeedleSummary:
    """Needle-in-a-haystack retrieval grid.

    ``effective_context`` is the largest haystack retrieved at every depth, or
    ``None`` when no length passed at every depth. ``cells`` holds only cells
    that got an answer; a cell whose request errored, for example a context
    overflow, is left out and counted in ``errors`` so a server failure never
    reads as a retrieval miss.
    """

    cells: tuple[NeedleCell, ...]
    accuracy: float
    effective_context: int | None
    errors: int = 0


@dataclass(frozen=True)
class CategoryScore:
    """A tool-eval category, as a 0-100 percentage."""

    code: str
    name: str
    percent: float


@dataclass(frozen=True)
class ToolEvalSummary:
    """Tool-calling quality. ``score`` is 0-100.

    With ``trials`` above 1, ``score`` and category percentages are the mean
    across completed trials and the counts are per trial. ``suite_size`` is the
    scenario count of the public suite the run drew from, so a card can mark a
    run with fewer scenarios as a subset; ``None`` means unknown.
    """

    score: float
    rating: str
    passed: int
    partial: int
    failed: int
    total: int
    categories: tuple[CategoryScore, ...] = ()
    safety_capped: bool = False
    hardmode: bool = False
    trials: int = 1
    suite_size: int | None = None


@dataclass(frozen=True)
class AccuracyResult:
    """An external accuracy benchmark such as GSM8K. ``accuracy`` is 0-1."""

    name: str
    accuracy: float
    detail: str | None = None


@dataclass(frozen=True)
class CardData:
    """Everything one card shows. A ``None`` or empty section renders as not measured."""

    model: str
    tool_version: str
    created_at: str
    engine: str | None = None
    engine_version: str | None = None
    quantization: str | None = None
    hardware: str | None = None
    max_context: int | None = None
    perf_preset: str | None = None
    perf_pp: int | None = None
    perf_tg: int | None = None
    perf: tuple[PerfPoint, ...] = ()
    tool_eval: ToolEvalSummary | None = None
    needle: NeedleSummary | None = None
    accuracy: tuple[AccuracyResult, ...] = field(default_factory=tuple)
    seed: int | None = None
    # A short prefix of the config fingerprint, so a reader can tell whether two
    # cards measured the same configuration. It identifies a configuration, not
    # a person or a machine.
    fingerprint: str | None = None
