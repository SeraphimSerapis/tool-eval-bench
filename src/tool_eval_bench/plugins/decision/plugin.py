"""Decision-model plugin: orchestration and scoring.

Implements ``BenchmarkPlugin`` for models that answer by scoring options in one
forward pass (llama.cpp ``/v1/systemone``) instead of generating text.  The
items are built in, so there is no download.  A decision model has no sampling,
so ``temperature``, ``seed``, and ``extra_params`` are accepted for interface
compatibility and ignored.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.decision import DecisionBackend, DecisionUnsupportedError
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
)
from tool_eval_bench.plugins.decision.dataset import (
    CATEGORIES,
    CATEGORY_ROUTING,
    CATEGORY_URGENCY,
    QUESTION_NAME,
    VARIANT_BASE,
    DecisionItem,
    build_items,
)
from tool_eval_bench.plugins.decision.evaluator import score_item
from tool_eval_bench.plugins.decision.metrics import (
    HIGH_CONFIDENCE,
    confusion_matrix,
    expected_calibration_error,
    mean,
    percentile,
    reliability_bins,
    robustness,
)
from tool_eval_bench.plugins.decision.render import render_report

logger = logging.getLogger(__name__)


def _rating_for_accuracy(accuracy: float) -> str:
    if accuracy >= 90:
        return "★★★★★ Excellent"
    if accuracy >= 80:
        return "★★★★ Good"
    if accuracy >= 65:
        return "★★★ Adequate"
    if accuracy >= 45:
        return "★★ Weak"
    return "★ Poor"


class DecisionPlugin(BenchmarkPlugin):
    """Routing, moderation, urgency, yes/no, and abstention for decision models."""

    @property
    def name(self) -> str:
        return "decision"

    @property
    def description(self) -> str:
        return "Decision models: accuracy, calibration, and option-order robustness"

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
        """Run every item once and score accuracy, calibration, and robustness.

        Raises ``ValueError`` when ``adapter`` cannot make decision requests and
        ``DecisionUnsupportedError`` when the server does not serve them.
        """
        if not isinstance(adapter, DecisionBackend):
            raise ValueError(
                f"{type(adapter).__name__} cannot make decision requests; "
                "build the adapter with build_decision_adapter()"
            )
        items: list[DecisionItem] = list(kwargs.get("items") or build_items())
        concurrency: int = kwargs.get("concurrency", 1)
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        total = len(items)
        sem = asyncio.Semaphore(concurrency)
        rows: list[dict[str, Any]] = [{} for _ in range(total)]
        total_tokens = 0
        progress_counter = 0
        progress_lock = asyncio.Lock()
        t_start = time.monotonic()

        async def eval_one(idx: int, item: DecisionItem) -> None:
            nonlocal total_tokens, progress_counter
            try:
                async with sem:
                    result = await adapter.decide(
                        model=model,
                        state=item.state,
                        questions={QUESTION_NAME: item.question},
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                    )
                total_tokens += result.input_tokens
                score = score_item(item, result.answers[QUESTION_NAME])
                rows[idx] = _row(item, score, result.elapsed_ms, result.input_tokens)
            except DecisionUnsupportedError:
                # Every item would fail the same way; stop and say so once.
                raise
            except Exception as exc:
                logger.debug("Decision item %s failed: %s", item.id, exc)
                rows[idx] = _error_row(item, exc)

            if on_progress:
                async with progress_lock:
                    progress_counter += 1
                    await on_progress(progress_counter, total, rows[idx])

        outcomes = await asyncio.gather(
            *(eval_one(i, item) for i, item in enumerate(items)), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, DecisionUnsupportedError):
                raise outcome
            if isinstance(outcome, BaseException):
                raise outcome

        duration = time.monotonic() - t_start
        details = _summarize(rows)
        accuracy = details["accuracy"]
        return BenchmarkResult(
            plugin_name="decision",
            score=round(accuracy, 2),
            score_label=f"{accuracy:.1f}% ({details['correct']}/{details['total']})",
            rating=_rating_for_accuracy(accuracy),
            details=details,
            item_results=rows,
            metadata={"dataset": "built-in", "endpoint": "/v1/systemone"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        return render_report(result)


def _row(item: DecisionItem, score: Any, elapsed_ms: float, input_tokens: int) -> dict[str, Any]:
    return {
        "id": item.id,
        "category": item.category,
        "variant": item.variant,
        "base_id": item.base_id,
        "question_type": type(item.question).__name__,
        "state": item.state,
        "gold": score.gold,
        "predicted": score.predicted,
        # Variants that rename options are compared to the base item in its names.
        "predicted_base": item.to_base.get(score.predicted, score.predicted),
        "correct": score.correct,
        "is_error": False,
        "confidence": round(score.confidence, 6),
        "p_gold": round(score.p_gold, 6),
        "brier": round(score.brier, 6),
        "log_loss": round(score.log_loss, 6),
        "distribution": {k: round(v, 6) for k, v in score.distribution.items()},
        "level_error": None if score.level_error is None else round(score.level_error, 4),
        "elapsed_ms": round(elapsed_ms, 2),
        "input_tokens": input_tokens,
    }


def _error_row(item: DecisionItem, exc: Exception) -> dict[str, Any]:
    return {
        "id": item.id,
        "category": item.category,
        "variant": item.variant,
        "base_id": item.base_id,
        "question_type": type(item.question).__name__,
        "state": item.state,
        "gold": str(
            item.gold if not isinstance(item.gold, bool) else ("yes" if item.gold else "no")
        ),
        "predicted": None,
        "predicted_base": None,
        "correct": False,
        "is_error": True,
        "error": f"{type(exc).__name__}: {exc}"[:200],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item rows into the details block."""
    base = [r for r in rows if r["variant"] == VARIANT_BASE]
    answered = [r for r in base if not r["is_error"]]
    errors = sum(1 for r in rows if r["is_error"])
    correct = sum(1 for r in base if r["correct"])
    accuracy = correct / len(base) * 100 if base else 0.0

    categories: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in base if r["category"] == cat]
        if cat_rows:
            ok = sum(1 for r in cat_rows if r["correct"])
            categories[cat] = {
                "correct": ok,
                "total": len(cat_rows),
                "accuracy": round(ok / len(cat_rows) * 100, 1),
            }

    confidences = [r["confidence"] for r in answered]
    bins = reliability_bins(confidences, [r["correct"] for r in answered])
    calibration = {
        "ece": round(expected_calibration_error(bins), 4),
        "brier": round(mean([r["brier"] for r in answered]), 4),
        "log_loss": round(mean([r["log_loss"] for r in answered]), 4),
        "mean_confidence": round(mean(confidences), 4),
        "bins": [
            {
                "low": b.low,
                "high": b.high,
                "count": b.count,
                "mean_confidence": round(b.mean_confidence, 4),
                "accuracy": round(b.accuracy, 4),
            }
            for b in bins
        ],
        "high_confidence_errors": sum(
            1 for r in answered if not r["correct"] and r["confidence"] >= HIGH_CONFIDENCE
        ),
    }

    routing = [r for r in answered if r["category"] == CATEGORY_ROUTING]
    labels = list(dict.fromkeys(r["gold"] for r in routing))
    confusion = {
        "labels": labels,
        "matrix": confusion_matrix([(r["gold"], r["predicted"]) for r in routing], labels),
    }

    levels = [r for r in answered if r["category"] == CATEGORY_URGENCY]
    ordinal = {
        "count": len(levels),
        "mae": round(mean([r["level_error"] for r in levels]), 4),
        "within_one": round(
            sum(abs(int(r["predicted"]) - int(r["gold"])) <= 1 for r in levels) / len(levels) * 100,
            1,
        )
        if levels
        else 0.0,
    }

    base_answers = {r["id"]: (r["predicted"], r["correct"]) for r in answered}
    variant_rows: dict[str, list[tuple[str, str, bool]]] = {}
    for r in rows:
        if r["variant"] != VARIANT_BASE and not r["is_error"]:
            variant_rows.setdefault(r["variant"], []).append(
                (r["base_id"], r["predicted_base"], r["correct"])
            )

    elapsed = [r["elapsed_ms"] for r in rows if not r["is_error"]]
    return {
        "correct": correct,
        "total": len(base),
        "requests": len(rows),
        "answered": len(answered),
        "errors": errors,
        "completion_rate": round((len(rows) - errors) / len(rows) * 100, 2) if rows else 0.0,
        "status": "incomplete" if errors else "completed",
        "incomplete": errors > 0,
        "accuracy": round(accuracy, 2),
        "categories": categories,
        "calibration": calibration,
        "confusion": confusion,
        "ordinal": ordinal,
        "robustness": robustness(base_answers, variant_rows).to_dict(),
        "latency_ms": {
            "mean": round(mean(elapsed), 2),
            "p50": round(percentile(elapsed, 50), 2),
            "p95": round(percentile(elapsed, 95), 2),
        },
        "error_kinds": dict(Counter(r["error"].split(":", 1)[0] for r in rows if r["is_error"])),
    }
