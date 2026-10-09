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

from tool_eval_bench.domain.adapters import BackendAdapter, ChatCompletionResult
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
# Adapter result guard
# ---------------------------------------------------------------------------


class TransportRejectedError(RuntimeError):
    """The server rejected the request, so the result holds no model answer."""


def raise_for_transport_error(result: ChatCompletionResult) -> ChatCompletionResult:
    """Return *result*, or raise when the adapter degraded a rejection to a soft result.

    Adapters report a non-retryable HTTP rejection (401, 404, a 400 context
    overflow) as ``content="[server error N] ..."`` with
    ``transport_error_status`` set rather than raising.  Grading that text
    would score the error body as an answer, so plugins call this inside the
    ``try`` that already turns request exceptions into error rows.
    """
    if result.transport_error_status is not None:
        raise TransportRejectedError(
            f"server rejected the request with HTTP {result.transport_error_status}: "
            f"{result.content[:200]}"
        )
    return result


def final_answer_text(result: ChatCompletionResult) -> str | None:
    """Return the text to grade, or ``None`` when the token budget truncated the turn.

    ``content`` is graded when it holds any non-whitespace text.  When it is
    empty or whitespace only, the reasoning text stands in for models that put
    their final answer there, but only if generation finished normally.  An
    empty ``content`` with ``finish_reason == "length"`` means the budget ran
    out mid-thought; grading that unfinished reasoning can pick up a stray
    number or letter, so the caller records the item as truncated (wrong, and
    counted separately).
    """
    if result.content and result.content.strip():
        return result.content
    if result.finish_reason == "length":
        return None
    return result.reasoning or ""


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
