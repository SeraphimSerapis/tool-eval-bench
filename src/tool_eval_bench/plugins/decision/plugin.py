"""Decision-model plugin: orchestration and scoring.

Implements ``BenchmarkPlugin`` for models that answer by scoring options in one
forward pass (llama.cpp ``/v1/systemone``) instead of generating text.  A decision
model has no sampling, so ``temperature``, ``seed``, and ``extra_params`` are
accepted for interface compatibility and ignored.

A run scores the vendored typed-decisions test split, one request per case with
all five questions, against soft gold distributions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.decision import (
    DecisionBackend,
    DecisionQuestion,
    DecisionResult,
    DecisionState,
    DecisionUnsupportedError,
)
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
)
from tool_eval_bench.plugins.decision.evaluator import score_against_gold
from tool_eval_bench.plugins.decision.metrics import (
    HIGH_CONFIDENCE,
    expected_calibration_error,
    mean,
    percentile,
    reliability_bins,
)
from tool_eval_bench.plugins.decision.render import render_report
from tool_eval_bench.plugins.decision.typed_decisions import (
    PRIOR_ACCURACY,
    QUESTION_TYPES,
    TEACHER_CEILING_ACCURACY,
    WORKFLOWS,
    DatasetInfo,
    TypedDecisionCase,
    dataset_info,
    load_cases,
)

logger = logging.getLogger(__name__)

NO_SUCCESSFUL_CASES = "Incomplete: no successful cases"


def _rating(accuracy: float, answered: int) -> str:
    if not answered:
        # Every request failed, so the 0% says nothing about the model.
        return NO_SUCCESSFUL_CASES
    # Gold is a teacher's output, so a fixed "excellent" threshold would be
    # meaningless here.  The bands are the card's two reference points.
    if accuracy >= TEACHER_CEILING_ACCURACY * 100:
        return "★★★ At the teacher self-agreement ceiling"
    if accuracy > PRIOR_ACCURACY * 100:
        return "★★ Above the prior baseline"
    return "★ No better than the prior baseline"


class DecisionPlugin(BenchmarkPlugin):
    """Typed-decisions cases scored against soft gold distributions."""

    @property
    def name(self) -> str:
        return "decision"

    @property
    def description(self) -> str:
        return "Decision models: accuracy, distance from gold, and calibration"

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
        """Run every case once and score it.

        ``cases`` narrows the run to a non-empty subset of the test split.
        ``concurrency`` sets how many requests are in flight.  Raises
        ``ValueError`` when ``adapter`` cannot make decision requests or
        ``cases`` is empty, ``DecisionUnsupportedError`` when the server does
        not serve them, and ``DatasetIntegrityError`` when the vendored data
        fails its check.
        """
        if not isinstance(adapter, DecisionBackend):
            raise ValueError(
                f"{type(adapter).__name__} cannot make decision requests; "
                "build the adapter with build_decision_adapter()"
            )
        concurrency: int = kwargs.get("concurrency", 1)
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        cases: list[TypedDecisionCase] | None = kwargs.get("cases")
        if cases is not None and not cases:
            raise ValueError("cases must not be empty")

        async def decide(
            state: DecisionState, questions: Mapping[str, DecisionQuestion]
        ) -> DecisionResult:
            return await adapter.decide(
                model=model,
                state=state,
                questions=questions,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
                base_url=base_url,
            )

        return await _run_cases(
            list(cases) if cases is not None else load_cases(),
            dataset_info(),
            decide,
            concurrency,
            on_progress,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        return render_report(result)


_Decide = Callable[[DecisionState, Mapping[str, DecisionQuestion]], Awaitable[DecisionResult]]


async def _gather_all(jobs: list[Awaitable[None]]) -> None:
    """Await every job and re-raise the first failure.

    Each job records its own per-case errors, so anything that escapes is
    either an unsupported endpoint or a bug, and both should stop the run.
    """
    for outcome in await asyncio.gather(*jobs, return_exceptions=True):
        if isinstance(outcome, BaseException):
            raise outcome


async def _run_cases(
    cases: list[TypedDecisionCase],
    info: DatasetInfo,
    decide: _Decide,
    concurrency: int,
    on_progress: OnPluginProgress | None,
) -> BenchmarkResult:
    # Progress counts decisions, not requests, so a tally of correct answers
    # adds up to the headline accuracy.
    total = sum(len(c.questions) for c in cases)
    sem = asyncio.Semaphore(concurrency)
    case_rows: list[list[dict[str, Any]]] = [[] for _ in cases]
    total_tokens = 0
    progress_counter = 0
    progress_lock = asyncio.Lock()
    t_start = time.monotonic()

    async def eval_case(idx: int, case: TypedDecisionCase) -> None:
        nonlocal total_tokens, progress_counter
        try:
            async with sem:
                # The whole case in one request, as the card's leaderboard was
                # scored.  Asking one question at a time changes the answers.
                result = await decide(case.state, case.questions)
            total_tokens += result.input_tokens
            case_rows[idx] = [_answered_row(case, name, result) for name in case.questions]
        except DecisionUnsupportedError:
            raise
        except Exception as exc:
            logger.debug("Decision case %s failed: %s", case.id, exc)
            case_rows[idx] = [_error_row(case, name, exc) for name in case.questions]

        if on_progress:
            async with progress_lock:
                for row in case_rows[idx]:
                    progress_counter += 1
                    await on_progress(progress_counter, total, row)

    await _gather_all([eval_case(i, case) for i, case in enumerate(cases)])

    duration = time.monotonic() - t_start
    rows = [row for group in case_rows for row in group]
    details = _summarize(rows, info)
    accuracy = details["accuracy"]
    return BenchmarkResult(
        plugin_name="decision",
        score=round(accuracy, 2),
        score_label=f"{accuracy:.1f}% ({details['correct']}/{details['total']} decisions)",
        rating=_rating(accuracy, details["answered"]),
        details=details,
        item_results=rows,
        metadata={
            "dataset": info.dataset,
            "dataset_revision": info.revision,
            "dataset_split": info.split,
            "dataset_license": info.license,
            "endpoint": "/v1/systemone",
        },
        duration_seconds=round(duration, 2),
        total_tokens=total_tokens,
    )


def _base_row(case: TypedDecisionCase, name: str) -> dict[str, Any]:
    return {
        "id": f"{case.id}/{name}",
        "case_id": case.id,
        "workflow": case.workflow,
        "question": name,
        "question_type": case.question_type(name),
        "state_preview": case.state_preview,
        "gold": case.gold[name].label,
        "gold_distribution": dict(case.gold[name].probabilities),
    }


def _answered_row(case: TypedDecisionCase, name: str, result: DecisionResult) -> dict[str, Any]:
    score = score_against_gold(case.questions[name], result.answers[name], case.gold[name])
    return {
        **_base_row(case, name),
        "predicted": score.predicted,
        "correct": score.correct,
        "is_error": False,
        "confidence": round(score.confidence, 6),
        "kl": round(score.kl, 6),
        "brier": round(score.brier, 6),
        "distribution": {k: round(v, 6) for k, v in score.distribution.items()},
        # Every decision in a case shares its request, and so its latency.
        "request_ms": round(result.elapsed_ms, 2),
    }


def _error_row(case: TypedDecisionCase, name: str, exc: Exception) -> dict[str, Any]:
    return {
        **_base_row(case, name),
        "predicted": None,
        "correct": False,
        "is_error": True,
        "error": f"{type(exc).__name__}: {exc}"[:200],
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in rows if not r["is_error"]]
    correct = sum(1 for r in rows if r["correct"])
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": round(correct / len(rows) * 100, 2) if rows else 0.0,
        # None, not 0, when nothing was answered: 0 would read as a perfect match.
        "kl": round(mean([r["kl"] for r in answered]), 4) if answered else None,
        "brier": round(mean([r["brier"] for r in answered]), 4) if answered else None,
    }


def _grouped(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str], order: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in order}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {name: _group(rs) for name, rs in groups.items() if rs}


def _summarize(rows: list[dict[str, Any]], info: DatasetInfo) -> dict[str, Any]:
    """Aggregate per-decision rows: overall, per question type, workflow, and question.

    Errors count as misses in accuracy.  KL, Brier, and calibration cover the
    answered decisions only, since an error has no distribution to compare.
    """
    answered = [r for r in rows if not r["is_error"]]
    errors = len(rows) - len(answered)
    overall = _group(rows)
    confidences = [r["confidence"] for r in answered]
    bins = reliability_bins(confidences, [r["correct"] for r in answered])
    request_ms = list({r["case_id"]: r["request_ms"] for r in answered}.values())
    cases = len({r["case_id"] for r in rows})
    return {
        "dataset": info.dataset,
        "dataset_revision": info.revision,
        "dataset_split": info.split,
        "dataset_license": info.license,
        "attribution": info.attribution,
        "correct": overall["correct"],
        "total": overall["total"],
        "cases": cases,
        "answered": len(answered),
        "errors": errors,
        "completion_rate": round(len(answered) / len(rows) * 100, 2) if rows else 0.0,
        "status": "incomplete" if errors else "completed",
        "incomplete": errors > 0,
        "accuracy": overall["accuracy"],
        "kl": overall["kl"],
        "brier": overall["brier"],
        "calibration": {
            "ece": round(expected_calibration_error(bins), 4),
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
        },
        "by_type": _grouped(rows, lambda r: r["question_type"], QUESTION_TYPES),
        "by_workflow": _grouped(rows, lambda r: r["workflow"], WORKFLOWS),
        "by_question": _grouped(rows, lambda r: f"{r['workflow']}/{r['question']}"),
        "latency_ms": {
            "mean": round(mean(request_ms), 2),
            "p50": round(percentile(request_ms, 50), 2),
            "p95": round(percentile(request_ms, 95), 2),
        },
        "error_kinds": dict(Counter(r["error"].split(":", 1)[0] for r in rows if r["is_error"])),
    }
