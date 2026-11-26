"""Card panels facade.

Each panel takes a :class:`Box` and returns SVG markup that stays inside it.
Implementations live in sibling modules (``_primitives``, ``_score_panels``,
``_perf_panels``, ``_needle_panel``) and are re-exported here for ``_layout``
and unit tests.
"""

from __future__ import annotations

from tool_eval_bench.storage.reports.card._needle_panel import (
    _needle_heatmap,
    needle_panel,
)
from tool_eval_bench.storage.reports.card._perf_panels import (
    _decode_callout,
    _span_extent,
    concurrency_panel,
    long_context_panel,
)
from tool_eval_bench.storage.reports.card._primitives import (
    HINT_H,
    LABEL,
    MID,
    GhostKind,
    bar_color,
    log_decades,
    log_step,
    missing_panel,
    nice_ticks,
    pill,
    spaced_labels,
)
from tool_eval_bench.storage.reports.card._score_panels import (
    Hero,
    accuracy_height,
    accuracy_min_height,
    accuracy_panel,
    chip_runs_for,
    hero_tile,
    hero_value_size,
    outcome_bar,
    score_block,
    smarts_height,
    smarts_panel,
)

__all__ = [
    "HINT_H",
    "LABEL",
    "MID",
    "GhostKind",
    "Hero",
    "_decode_callout",
    "_needle_heatmap",
    "_span_extent",
    "accuracy_height",
    "accuracy_min_height",
    "accuracy_panel",
    "bar_color",
    "chip_runs_for",
    "concurrency_panel",
    "hero_tile",
    "hero_value_size",
    "log_decades",
    "log_step",
    "long_context_panel",
    "missing_panel",
    "needle_panel",
    "nice_ticks",
    "outcome_bar",
    "pill",
    "score_block",
    "smarts_height",
    "smarts_panel",
    "spaced_labels",
]
