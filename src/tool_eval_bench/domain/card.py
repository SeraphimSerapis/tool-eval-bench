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
    ``tg_estimated`` marks a per-request decode speed derived by dividing the
    aggregate by the concurrency because the run did not store a measured one.
    """

    depth: int
    concurrency: int
    pp_tps: float
    tg_tps: float
    tg_tps_total: float
    ttft_ms: float
    cached_ttft_ms: float | None = None
    pp_estimated: bool = False
    tg_estimated: bool = False


@dataclass(frozen=True)
class NeedleCell:
    """One grid cell. An errored cell got no answer and has ``found=False``."""

    context_tokens: int
    depth_percent: float
    found: bool
    error: bool = False


@dataclass(frozen=True)
class NeedleSummary:
    """Needle-in-a-haystack retrieval grid.

    ``effective_context`` is the largest haystack retrieved at every depth, or
    ``None`` when no length passed at every depth. It is an estimate in
    tokens. ``cells`` holds every measured cell, including cells whose request
    errored, for example a context overflow. Those carry ``error=True`` and are
    counted in ``errors``, so a server failure never reads as a retrieval
    miss. ``accuracy`` is found cells over answered cells, so errors do not
    count against it; it is 0.0 when no cell was answered.
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
    across completed trials and the counts are per trial, rounded so they still
    sum to ``total``. ``total`` counts scored scenarios; ``excluded`` counts
    selected scenarios dropped from scoring after an infrastructure failure.

    ``suite_size`` is the scenario count of the suite the selection is compared
    against: the public suite, the public suite plus Hard Mode, or Hard Mode
    alone for ``hardmode_only``. A run is a subset when ``total + excluded`` is
    below it. ``None`` means unknown, for example a scenario pack or a run from
    an older registry, and then no subset mark is drawn. ``scenario_pack`` marks
    scenarios loaded from a ``--scenario-pack`` rather than the public suite.
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
    excluded: int = 0
    hardmode_only: bool = False
    scenario_pack: bool = False


@dataclass(frozen=True)
class AccuracyResult:
    """An external accuracy benchmark such as GSM8K. ``accuracy`` is 0-1.

    ``accuracy`` is correct answers over answered items, or NaN when every item
    errored. ``errors`` counts items whose request failed and ``items`` the
    items attempted, when known, so a card can say a score rests on only part
    of the set.
    """

    name: str
    accuracy: float
    detail: str | None = None
    errors: int = 0
    items: int | None = None


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
    # Grid cells that produced no usable sample, and whether the sweep stopped
    # before its plan finished, for example an interrupted or failed depth pass.
    # A partial sweep must not read as a complete one.
    perf_failed_cells: int = 0
    perf_incomplete: bool = False
