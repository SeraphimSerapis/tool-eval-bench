"""Pluggable benchmark abstraction.

Defines the ``BenchmarkPlugin`` protocol and ``BenchmarkResult`` container
that all benchmark modules (tool-eval, gsm8k, future MMLU, HumanEval, …)
implement.  Shared infrastructure (adapter, storage, reporting) stays in
the existing layers; each plugin owns its own orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS

# ---------------------------------------------------------------------------
# Universal result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Universal result container returned by every benchmark plugin.

    Fields
    ------
    plugin_name : str
        Short identifier, e.g. ``"tool-eval"``, ``"gsm8k"``.
    score : float
        Normalised 0–100 score (accuracy %, weighted score, etc.).
    score_label : str
        Human-friendly score string, e.g. ``"78 / 100"`` or ``"82.3 % accuracy"``.
    rating : str
        Star/letter rating derived from *score*.
    details : dict
        Plugin-specific breakdown (category scores, per-topic stats, …).
    item_results : list[dict]
        Per-item results — one entry per question / scenario.
    metadata : dict
        Run metadata (dataset version, few-shot config, …).
    duration_seconds : float
        Wall-clock time for the entire plugin run.
    total_tokens : int
        Aggregate prompt + completion tokens consumed.
    """

    plugin_name: str
    score: float
    score_label: str
    rating: str
    details: dict[str, Any] = field(default_factory=dict)
    item_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / SQLite storage."""
        return {
            "plugin_name": self.plugin_name,
            "score": round(self.score, 2),
            "score_label": self.score_label,
            "rating": self.rating,
            "details": self.details,
            "item_results": self.item_results,
            "metadata": self.metadata,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_tokens": self.total_tokens,
        }


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

OnPluginProgress = Callable[[int, int, dict[str, Any]], Awaitable[None]]
"""``(current, total, item_info)`` — called after each item completes."""


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------


class BenchmarkPlugin(ABC):
    """Interface that every pluggable benchmark module must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in CLI flags and storage, e.g. ``"gsm8k"``."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description for ``--help``."""
        ...

    @abstractmethod
    async def run(
        self,
        adapter: BackendAdapter,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        seed: int | None = None,
        extra_params: dict[str, Any] | None = None,
        on_progress: OnPluginProgress | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Execute the benchmark and return a result."""
        ...

    @abstractmethod
    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Return Markdown lines for inclusion in a run report."""
        ...
