"""Pluggable benchmark abstraction.

Defines the ``BenchmarkPlugin`` protocol and ``BenchmarkResult`` container
that all benchmark modules (tool-eval, gsm8k, future MMLU, HumanEval, …)
implement.  Shared infrastructure (adapter, storage, reporting) stays in
the existing layers; each plugin owns its own orchestration.
"""

from __future__ import annotations

import re
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
    """The server rejected the request or sent an unreadable body, so there is no model answer."""


def raise_for_transport_error(result: ChatCompletionResult) -> ChatCompletionResult:
    """Return *result*, or raise when the adapter degraded a failure to a soft result.

    Adapters report a non-retryable HTTP rejection (401, 404, a 400 context
    overflow) as ``content="[server error N] ..."`` with
    ``transport_error_status`` set rather than raising, and a body that is not
    valid JSON as ``content="[malformed response]"`` with ``malformed`` set.
    Grading that text would score the placeholder as an answer, so plugins
    call this inside the ``try`` that already turns request exceptions into
    error rows.
    """
    if result.transport_error_status is not None:
        raise TransportRejectedError(
            f"server rejected the request with HTTP {result.transport_error_status}: "
            f"{result.content[:200]}"
        )
    if result.malformed:
        raise TransportRejectedError("server returned a response body that is not valid JSON")
    return result


_LEADING_THINK_RE = re.compile(
    r"^\s*<think(?:\s[^>]*)?>(.*?)</think>\s*",
    flags=re.DOTALL | re.IGNORECASE,
)
_CLOSED_THINK_RE = re.compile(
    r"<think(?:\s[^>]*)?>(.*?)</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_RE = re.compile(r"<think(?:\s[^>]*)?>", flags=re.IGNORECASE)


def strip_thinking(text: str, *, fallback_to_thought: bool = False) -> str | None:
    """Strip inline ``<think>...</think>`` blocks from *text*.

    Some OpenAI-compatible servers leave reasoning traces inline inside
    ``content`` wrapped in ``<think>...</think>`` tags instead of populating a
    separate ``reasoning_content`` field.  Closed blocks are removed so format
    checkers and answer extractors see only the visible answer.

    When *text* contains an unclosed ``<think>`` tag, any non-whitespace prefix
    before it is returned; otherwise an unclosed ``<think>`` returns ``None``
    when ``fallback_to_thought`` is ``False`` and ``""`` when ``True``, never
    the unfinished scratchpad.  When only closed ``<think>...</think>`` blocks
    are present with no visible text outside them, ``fallback_to_thought``
    controls whether the completed thought text stands in or ``None`` is
    returned.
    """
    if "<think" not in text.lower():
        return text

    thoughts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        thoughts.append(match.group(1))
        return ""

    cleaned = text
    while True:
        next_cleaned = _LEADING_THINK_RE.sub(_collect, cleaned, count=1)
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    cleaned = _CLOSED_THINK_RE.sub(_collect, cleaned)

    open_match = _OPEN_THINK_RE.search(cleaned)
    if open_match is not None:
        prefix = cleaned[: open_match.start()]
        if prefix.strip():
            return prefix.rstrip()
        return "" if fallback_to_thought else None

    if cleaned.strip():
        return cleaned
    if not fallback_to_thought:
        return None
    joined = "\n".join(t for t in thoughts if t.strip())
    return joined if joined else ""


def final_answer_text(result: ChatCompletionResult) -> str | None:
    """Return the text to grade, or ``None`` when the token budget truncated the turn.

    Inline ``<think>...</think>`` blocks are stripped from ``content`` without
    mutating *result*.  Visible non-whitespace text outside ``<think>`` always
    wins, even when ``finish_reason == "length"``.  When no visible text remains
    outside ``<think>``:

    - ``finish_reason == "length"`` returns ``None`` so callers record the item
      as truncated rather than grading unfinished reasoning.
    - An unclosed ``<think>`` block returns ``""`` when generation stopped
      without a separate ``reasoning`` fallback, never the half-written tail.
    - Otherwise completed inline ``<think>...</think>`` text or
      ``result.reasoning`` stands in for models that place their final answer in
      reasoning and stop normally.
    """
    finished = result.finish_reason != "length"
    if result.content and result.content.strip():
        stripped = strip_thinking(result.content, fallback_to_thought=False)
        if stripped is not None and stripped.strip():
            return stripped
        if not finished:
            return None
        from_thought = strip_thinking(result.content, fallback_to_thought=True) or ""
        if from_thought.strip():
            return from_thought

    if not finished:
        return None
    if result.reasoning and result.reasoning.strip():
        return strip_thinking(result.reasoning, fallback_to_thought=True) or ""
    return ""


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
