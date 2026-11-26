"""Toolscale plugin: orchestration and BenchmarkPlugin entry point.

Implements ``BenchmarkPlugin`` for the tool-count by similarity sweep from
issue #319. Each item is one streamed request with a generated tool list; the
first tool call's name is graded. See ``grid.py`` for how the lists are built,
``aggregate.py`` for cell and contrast summaries, ``report.py`` for Markdown
rendering, and ``docs/toolscale.md`` for the method.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, default_max_tokens
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
    final_answer_text,
    raise_for_transport_error,
)
from tool_eval_bench.domain.redaction import redact_urls
from tool_eval_bench.plugins.toolscale.aggregate import similarity_check, summarize
from tool_eval_bench.plugins.toolscale.catalog import Catalog
from tool_eval_bench.plugins.toolscale.grid import (
    CORRECT,
    ERROR,
    ToolscaleItem,
    build_messages,
    grade,
    openai_tools,
)
from tool_eval_bench.plugins.toolscale.report import render_report

__all__ = [
    "ToolscalePlugin",
    "render_report",
    "similarity_check",
    "summarize",
]

logger = logging.getLogger(__name__)


def _rating_for_accuracy(accuracy: float) -> str:
    if accuracy >= 95:
        return "★★★★★ Excellent"
    if accuracy >= 85:
        return "★★★★ Good"
    if accuracy >= 70:
        return "★★★ Adequate"
    if accuracy >= 50:
        return "★★ Weak"
    return "★ Poor"


def _log_id(item: ToolscaleItem, catalog: Catalog) -> str:
    """An item label for logs; grid coordinates only for a private catalog."""
    if catalog.is_custom:
        return (
            f"target-{item.target_rank}@N{item.count}/{item.similarity}/{item.cue}/"
            f"{item.density}/{item.descriptions}/t{item.trial}"
        )
    return item.item_id


class ToolscalePlugin(BenchmarkPlugin):
    """Tool-selection accuracy across tool count and description similarity."""

    @property
    def name(self) -> str:
        return "toolscale"

    @property
    def description(self) -> str:
        return "Tool-selection accuracy as tool count and description similarity vary"

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
        """Send every item, grade its first tool call, and aggregate the grid.

        Keyword arguments: ``items`` (from :func:`grid.build_items`),
        ``catalog`` (the :class:`Catalog` they came from), and
        ``concurrency``.
        """
        items: list[ToolscaleItem] = list(kwargs.get("items") or [])
        catalog: Catalog | None = kwargs.get("catalog")
        concurrency: int = kwargs.get("concurrency", 1)
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        total = len(items)
        if total == 0:
            return BenchmarkResult(
                plugin_name="toolscale",
                score=0.0,
                score_label="0/0",
                rating=_rating_for_accuracy(0),
                details={"total": 0, "correct": 0},
            )
        if catalog is None:
            raise ValueError("catalog is required with items")

        extra: dict[str, Any] = {}
        if seed is not None:
            extra["seed"] = seed
        if extra_params:
            extra.update(extra_params)
        max_tokens = default_max_tokens(extra)

        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = [{} for _ in range(total)]
        total_tokens = 0
        progress_counter = 0
        progress_lock = asyncio.Lock()
        t_start = time.monotonic()

        def row(item: ToolscaleItem, outcome: str, picked: str | None, **extra_fields: Any) -> dict:
            return {
                "item_id": item.item_id,
                "target": item.target.name,
                "target_rank": item.target_rank,
                "domain": item.target.domain,
                "count": item.count,
                "similarity": item.similarity,
                "cue": item.cue,
                "density": item.density,
                "descriptions": item.descriptions,
                "trial": item.trial,
                "position": round(item.position, 4),
                "query": item.query,
                "outcome": outcome,
                "picked": picked,
                "correct": outcome == CORRECT,
                "is_error": outcome == ERROR,
                **extra_fields,
            }

        async def eval_one(idx: int, item: ToolscaleItem) -> None:
            nonlocal total_tokens, progress_counter
            prompt_tokens: int | None = None
            arguments: str | None = None
            reply: str | None = None
            error: str | None = None
            try:
                async with sem:
                    response = await adapter.chat_completion(
                        model=model,
                        messages=build_messages(item),
                        tools=openai_tools(item),
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra or None,
                        # The scenario suite streams every turn; matching it keeps
                        # one tool-call parsing path and gap-based timeouts for
                        # models that think before calling.
                        stream=True,
                    )
                raise_for_transport_error(response)
                outcome, picked = grade(item, response)
                prompt_tokens = response.prompt_tokens
                total_tokens += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                if response.tool_calls:
                    arguments = response.tool_calls[0].arguments_str[:300]
                else:
                    # What the model said instead of calling: a clarifying
                    # question and a refusal read very differently.
                    reply = (final_answer_text(response) or "").strip()[:300] or None
            except Exception as exc:
                logger.debug("Error on toolscale item %s: %s", _log_id(item, catalog), exc)
                outcome, picked = ERROR, None
                # Kept for the report: a run of errors is only diagnosable if
                # it says whether the endpoint timed out, refused, or rejected.
                error = redact_urls(f"{type(exc).__name__}: {exc}")[:200]

            results[idx] = row(
                item,
                outcome,
                picked,
                prompt_tokens=prompt_tokens,
                arguments=arguments,
                reply=reply,
                error=error,
            )
            if on_progress:
                async with progress_lock:
                    progress_counter += 1
                    await on_progress(progress_counter, total, results[idx])

        gathered = await asyncio.gather(
            *(eval_one(i, item) for i, item in enumerate(items)), return_exceptions=True
        )
        for outcome in gathered:
            if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
                raise outcome
        for i, exc in enumerate(gathered):
            if isinstance(exc, BaseException):
                logger.error("Toolscale item %s crashed: %s", _log_id(items[i], catalog), exc)
                if not results[i]:
                    results[i] = row(
                        items[i],
                        ERROR,
                        None,
                        prompt_tokens=None,
                        arguments=None,
                        reply=None,
                        error=redact_urls(f"{type(exc).__name__}: {exc}")[:200],
                    )

        details = summarize(results, items, catalog)
        duration = time.monotonic() - t_start
        accuracy = details["accuracy"]
        return BenchmarkResult(
            plugin_name="toolscale",
            score=accuracy,
            score_label=(
                f"{details['accuracy_when_called']:.1f}% accurate when called · "
                f"{details['call_rate']:.1f}% call rate"
            ),
            rating=_rating_for_accuracy(accuracy),
            details=details,
            item_results=results,
            metadata={"dataset": f"toolscale catalog v{catalog.version} ({catalog.source})"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render the grid, effects, error mix, similarity check, and failures."""
        return render_report(result)
