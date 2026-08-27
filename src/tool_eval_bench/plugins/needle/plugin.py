"""Needle-in-a-haystack plugin — orchestrator and report rendering.

Implements ``BenchmarkPlugin`` for retrieval across a (context length x depth)
grid.  Unlike the dataset-backed plugins, every case is generated locally from
the shared filler pool, so the benchmark needs no download and no cache.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
)
from tool_eval_bench.plugins.needle.haystack import (
    NeedleCase,
    build_needle_messages,
    grade_response,
)

logger = logging.getLogger(__name__)

# A retrieved fact is short.  The budget covers a model that restates the
# question or thinks briefly before answering, without paying for an essay.
_MAX_ANSWER_TOKENS = 256


def _rating_for_accuracy(accuracy: float) -> str:
    if accuracy >= 98:
        return "★★★★★ Excellent"
    if accuracy >= 90:
        return "★★★★ Good"
    if accuracy >= 75:
        return "★★★ Adequate"
    if accuracy >= 50:
        return "★★ Weak"
    return "★ Poor"


class NeedlePlugin(BenchmarkPlugin):
    """Needle-in-a-haystack retrieval across a context-length/depth grid."""

    @property
    def name(self) -> str:
        return "needle"

    @property
    def description(self) -> str:
        return "Needle-in-a-haystack retrieval across context lengths and depths"

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
        """Run every case in the grid and score retrieval accuracy."""
        cases: list[NeedleCase] = list(kwargs.get("cases") or [])
        concurrency: int = kwargs.get("concurrency", 1)
        context_size: int = kwargs.get("context_size", 0)

        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        total = len(cases)
        if total == 0:
            return BenchmarkResult(
                plugin_name="needle",
                score=0.0,
                score_label="0/0",
                rating=_rating_for_accuracy(0),
                details={"retrieved": 0, "total": 0},
            )

        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = [{} for _ in range(total)]
        retrieved = 0
        error_count = 0
        total_tokens = 0
        progress_counter = 0
        progress_lock = asyncio.Lock()
        t_start = time.monotonic()

        extra: dict[str, Any] = {}
        if seed is not None:
            extra["seed"] = seed
        if extra_params:
            extra.update(extra_params)

        async def eval_one(idx: int, case: NeedleCase) -> None:
            nonlocal retrieved, total_tokens, error_count, progress_counter

            # Vary the haystack per cell so two cells never share a prefix the
            # server could serve from cache instead of actually reading.
            case_seed = None if seed is None else seed + idx
            # Building a 100K-token document is CPU work, not I/O; keep it off
            # the event loop so concurrent cells still overlap their requests.
            messages = await asyncio.to_thread(build_needle_messages, case, seed=case_seed)

            content = ""
            try:
                async with sem:
                    response = await adapter.chat_completion(
                        model=model,
                        messages=messages,
                        tools=None,
                        temperature=temperature,
                        max_tokens=_MAX_ANSWER_TOKENS,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra or None,
                    )
                content = response.content or response.reasoning or ""
                total_tokens += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                is_error = False
            except Exception as exc:
                logger.debug("Error on needle cell %s: %s", case.cell_id, exc)
                is_error = True
                error_count += 1

            found = False if is_error else grade_response(case, content)
            if found:
                retrieved += 1

            results[idx] = {
                "cell_id": case.cell_id,
                "context_tokens": case.context_tokens,
                "depth_percent": case.depth_percent,
                "question": case.question,
                "expected": case.answer,
                "found": found,
                "is_error": is_error,
                "model_response": content[:500],
            }

            if on_progress:
                async with progress_lock:
                    progress_counter += 1
                    await on_progress(progress_counter, total, results[idx])

        tasks = [eval_one(i, case) for i, case in enumerate(cases)]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, exc in enumerate(gather_results):
            if isinstance(exc, BaseException):
                logger.error("Needle cell %s crashed: %s", cases[i].cell_id, exc)
                if not results[i]:
                    case = cases[i]
                    results[i] = {
                        "cell_id": case.cell_id,
                        "context_tokens": case.context_tokens,
                        "depth_percent": case.depth_percent,
                        "question": case.question,
                        "expected": case.answer,
                        "found": False,
                        "is_error": True,
                        "model_response": "",
                    }
                    error_count += 1

        duration = time.monotonic() - t_start
        accuracy = retrieved / total * 100
        answered = total - error_count

        lengths = sorted({c.context_tokens for c in cases})
        depths = sorted({c.depth_percent for c in cases})

        by_length = {
            str(length): _accuracy_of(r for r in results if r["context_tokens"] == length)
            for length in lengths
        }
        by_depth = {
            f"{depth:.2f}": _accuracy_of(r for r in results if r["depth_percent"] == depth)
            for depth in depths
        }

        return BenchmarkResult(
            plugin_name="needle",
            score=round(accuracy, 2),
            score_label=f"{accuracy:.1f}% ({retrieved}/{total} needles retrieved)",
            rating=_rating_for_accuracy(accuracy),
            details={
                "retrieved": retrieved,
                "total": total,
                "answered": answered,
                "errors": error_count,
                "completion_rate": round(answered / total * 100, 2),
                "status": "incomplete" if error_count else "completed",
                "incomplete": error_count > 0,
                "accuracy": round(accuracy, 2),
                "context_size": context_size,
                "context_lengths": lengths,
                "depths": depths,
                "by_length": by_length,
                "by_depth": by_depth,
                "effective_context": _effective_context(lengths, by_length),
            },
            item_results=results,
            metadata={"dataset": "synthetic (generated locally)"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render the retrieval grid and the failing cells."""
        d = result.details
        lengths: list[int] = d.get("context_lengths", [])
        depths: list[float] = d.get("depths", [])
        effective = d.get("effective_context")

        lines = [
            "## Needle in a Haystack — Retrieval",
            "",
            f"**Retrieval Accuracy:** {d.get('accuracy', 0):.1f}% "
            f"({d.get('retrieved', 0)}/{d.get('total', 0)})",
            f"**Rating:** {result.rating}",
            f"**Context Window:** {d.get('context_size', 0):,} tokens",
            (
                f"**Effective Context:** {effective:,} tokens (largest fully retrieved length)"
                if effective
                else "**Effective Context:** none (no length retrieved every depth)"
            ),
            f"**Duration:** {result.duration_seconds:.1f}s",
            f"**Tokens:** {result.total_tokens:,}",
            "",
        ]

        if lengths and depths:
            found = {
                (r["context_tokens"], r["depth_percent"]): r["found"] for r in result.item_results
            }
            lines.extend(
                [
                    "### Retrieval Grid",
                    "",
                    "Rows are needle depth, columns are haystack size.",
                    "",
                    "| Depth | " + " | ".join(f"{length // 1024}K" for length in lengths) + " |",
                    "|---" * (len(lengths) + 1) + "|",
                ]
            )
            for depth in depths:
                cells = " | ".join(
                    "✅" if found.get((length, depth)) else "❌" for length in lengths
                )
                lines.append(f"| {depth:.0%} | {cells} |")
            lines.append("")

        by_length = d.get("by_length", {})
        if by_length:
            lines.extend(
                [
                    "### Accuracy by Haystack Size",
                    "",
                    "| Context | Accuracy |",
                    "|---|---:|",
                ]
            )
            for length in lengths:
                lines.append(f"| {length:,} | {by_length.get(str(length), 0):.1f}% |")
            lines.append("")

        failures = [r for r in result.item_results if not r.get("found")]
        if failures:
            lines.extend(
                [
                    f"### Missed Needles ({len(failures)} total)",
                    "",
                    "| Cell | Expected | Model response |",
                    "|---|---|---|",
                ]
            )
            for f in failures:
                response = (f.get("model_response") or "").replace("|", "\\|").replace("\n", " ")
                if len(response) > 120:
                    response = response[:117] + "…"
                if f.get("is_error"):
                    response = "*(request failed)*"
                lines.append(
                    f"| `{f['cell_id']}` | `{f['expected']}` | {response or '*(empty)*'} |"
                )
            lines.append("")

        return lines


def _accuracy_of(rows: Any) -> float:
    """Percentage of *rows* that retrieved their needle."""
    rows = list(rows)
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r["found"]) / len(rows) * 100, 1)


def _effective_context(lengths: list[int], by_length: dict[str, float]) -> int | None:
    """Largest haystack size that retrieved the needle at every depth.

    This is the number worth quoting: a model advertising 128K that misses a
    needle at 32K does not have 128K of usable context, whatever the config
    says.  ``None`` means even the smallest length tested had a miss.
    """
    passing = [length for length in lengths if by_length.get(str(length), 0.0) >= 100.0]
    return max(passing) if passing else None
