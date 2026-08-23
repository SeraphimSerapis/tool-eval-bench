"""MMLU benchmark plugin — orchestrator and report rendering.

Implements ``BenchmarkPlugin`` for the Massive Multitask Language
Understanding benchmark (14,042 test questions, 57 subjects).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
)
from tool_eval_bench.plugins.mmlu.dataset import (
    CATEGORIES,
    SUBJECT_CATEGORIES,
    MMLUItem,
    load_dataset,
)
from tool_eval_bench.plugins.mmlu.evaluator import evaluate_answer
from tool_eval_bench.plugins.mmlu.prompts import build_messages

logger = logging.getLogger(__name__)


def _rating_for_accuracy(accuracy: float) -> str:
    if accuracy >= 85:
        return "★★★★★ Excellent"
    if accuracy >= 70:
        return "★★★★ Good"
    if accuracy >= 55:
        return "★★★ Adequate"
    if accuracy >= 40:
        return "★★ Weak"
    return "★ Poor"


class MMLUPlugin(BenchmarkPlugin):
    """MMLU benchmark — 14K multiple-choice questions across 57 subjects."""

    @property
    def name(self) -> str:
        return "mmlu"

    @property
    def description(self) -> str:
        return "Massive Multitask Language Understanding (14K questions, 57 subjects)"

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
        """Run MMLU evaluation."""
        n_shots: int = kwargs.get("n_shots", 5)
        limit: int = kwargs.get("limit", 0)
        subjects_filter: list[str] | None = kwargs.get("subjects", None)
        concurrency: int = kwargs.get("concurrency", 1)
        preloaded: dict | None = kwargs.get("_preloaded_items")

        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        # Load dataset
        if preloaded is not None:
            all_items = preloaded.get("test", [])
            dev_items = preloaded.get("dev", [])
        else:
            on_download = kwargs.get("on_download_progress")
            all_items = load_dataset("test", on_progress=on_download)
            dev_items = load_dataset("dev") if n_shots > 0 else []

        logger.info("Loaded %d MMLU test questions", len(all_items))

        # Filter by subject/category if requested
        if subjects_filter:
            expanded: set[str] = set()
            for s in subjects_filter:
                # Allow category names like "STEM" to expand to all subjects
                if s in CATEGORIES:
                    expanded.update(subj for subj, cat in SUBJECT_CATEGORIES.items() if cat == s)
                else:
                    expanded.add(s)
            all_items = [it for it in all_items if it.subject in expanded]
            logger.info("Filtered to %d items for subjects: %s", len(all_items), expanded)

        # Apply limit
        if limit > 0:
            all_items = all_items[:limit]

        total = len(all_items)
        if total == 0:
            return BenchmarkResult(
                plugin_name="mmlu",
                score=0.0,
                score_label="0/0",
                rating=_rating_for_accuracy(0),
                details={"correct": 0, "total": 0},
            )

        # Group dev items by subject for few-shot
        dev_by_subject: dict[str, list[MMLUItem]] = defaultdict(list)
        for item in dev_items:
            dev_by_subject[item.subject].append(item)

        # Evaluate
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = [{}] * total
        correct_count = 0
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

        async def eval_one(idx: int, item: MMLUItem) -> None:
            nonlocal correct_count, total_tokens, error_count, progress_counter
            few_shots = dev_by_subject.get(item.subject, [])
            messages: list[ChatMessage] = build_messages(item, few_shots, n_shots=n_shots)

            try:
                async with sem:
                    response = await adapter.chat_completion(
                        model=model,
                        messages=messages,
                        tools=None,
                        temperature=temperature,
                        max_tokens=256,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra or None,
                    )

                content = response.content or response.reasoning or ""
                total_tokens += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                is_error = False
            except Exception as exc:
                logger.debug("Error on question %d: %s", item.index, exc)
                content = ""
                is_error = True
                error_count += 1

            if is_error:
                result_dict: dict[str, Any] = {
                    "index": item.index,
                    "subject": item.subject,
                    "category": item.category,
                    "question": item.question[:200],
                    "correct": False,
                    "is_error": True,
                    "extracted_answer": None,
                    "ground_truth": item.answer,
                    "extraction_method": "error",
                    "model_response": "",
                }
            else:
                result = evaluate_answer(content, item.answer)
                if result.correct:
                    correct_count += 1
                result_dict = {
                    "index": item.index,
                    "subject": item.subject,
                    "category": item.category,
                    "question": item.question[:200],
                    "correct": result.correct,
                    "extracted_answer": result.extracted_answer,
                    "ground_truth": result.ground_truth_letter,
                    "extraction_method": result.extraction_method,
                    "model_response": content[:1000],
                }

            results[idx] = result_dict

            if on_progress:
                async with progress_lock:
                    progress_counter += 1
                    await on_progress(progress_counter, total, results[idx])

        tasks = [eval_one(i, item) for i, item in enumerate(all_items)]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, exc in enumerate(gather_results):
            if isinstance(exc, BaseException):
                logger.error("MMLU question %d crashed: %s", i, exc)
                if not results[i]:
                    item = all_items[i]
                    results[i] = {
                        "index": item.index,
                        "subject": item.subject,
                        "category": item.category,
                        "question": item.question[:200],
                        "correct": False,
                        "is_error": True,
                        "extracted_answer": None,
                        "ground_truth": item.answer,
                        "extraction_method": "error",
                        "model_response": "",
                    }
                    error_count += 1

        duration = time.monotonic() - t_start
        answered = total - error_count
        accuracy = (correct_count / total * 100) if total > 0 else 0.0

        # Per-category breakdown
        cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        subj_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        method_counts: Counter[str] = Counter()

        for r in results:
            cat = r.get("category", "Other")
            subj = r.get("subject", "unknown")
            cat_stats[cat]["total"] += 1
            subj_stats[subj]["total"] += 1
            method_counts[r.get("extraction_method", "none")] += 1
            if r.get("correct"):
                cat_stats[cat]["correct"] += 1
                subj_stats[subj]["correct"] += 1

        return BenchmarkResult(
            plugin_name="mmlu",
            score=round(accuracy, 2),
            score_label=f"{accuracy:.1f}% ({correct_count}/{total})",
            rating=_rating_for_accuracy(accuracy),
            details={
                "correct": correct_count,
                "total": total,
                "answered": answered,
                "errors": error_count,
                "completion_rate": round(answered / total * 100, 2) if total else 0.0,
                "status": "incomplete" if error_count else "completed",
                "incomplete": error_count > 0,
                "accuracy": round(accuracy, 2),
                "n_shots": n_shots,
                "dataset_size": 14042,
                "categories": {
                    cat: {
                        "correct": s["correct"],
                        "total": s["total"],
                        "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] else 0,
                    }
                    for cat, s in sorted(cat_stats.items())
                },
                "extraction_methods": dict(method_counts),
            },
            item_results=results,
            metadata={"dataset": "cais/mmlu", "config": "all"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render Markdown report section for MMLU results."""
        d = result.details
        lines = [
            "## MMLU — Massive Multitask Language Understanding",
            "",
            f"**Accuracy:** {result.score:.1f}% ({d['correct']}/{d['total']})",
            f"**Rating:** {result.rating}",
            f"**Duration:** {result.duration_seconds:.1f}s",
            f"**Tokens:** {result.total_tokens:,}",
            f"**Prompting:** {d.get('n_shots', 5)}-shot",
            "",
        ]

        # Category breakdown
        cats = d.get("categories", {})
        if cats:
            lines.extend(
                [
                    "### Per-Category Accuracy",
                    "",
                    "| Category | Correct | Total | Accuracy |",
                    "|---|---|---|---|",
                ]
            )
            for cat in CATEGORIES:
                if cat in cats:
                    c = cats[cat]
                    lines.append(
                        f"| {cat} | {c['correct']} | {c['total']} | {c['accuracy']:.1f}% |"
                    )
            lines.append("")

        # Extraction methods
        methods = d.get("extraction_methods", {})
        if methods:
            lines.extend(["### Extraction Methods", ""])
            for method, count in sorted(methods.items(), key=lambda x: -x[1]):
                lines.append(f"- **{method}**: {count}")
            lines.append("")

        # Error analysis
        failures = [r for r in result.item_results if not r.get("correct")]
        errors = [r for r in failures if r.get("is_error")]
        no_extract = [
            r for r in failures if not r.get("is_error") and r.get("extraction_method") == "none"
        ]
        wrong_answer = [
            r
            for r in failures
            if not r.get("is_error")
            and r.get("extraction_method") != "none"
            and r.get("extracted_answer") is not None
        ]

        if failures:
            lines.extend(
                [
                    "### Error Analysis",
                    "",
                    f"- **Total failures**: {len(failures)} / {d['total']}",
                ]
            )
            if no_extract:
                lines.append(
                    f"- **No answer extracted**: {len(no_extract)} — model did not "
                    "produce a recognizable A/B/C/D answer"
                )
            if wrong_answer:
                lines.append(
                    f"- **Wrong answer**: {len(wrong_answer)} — model chose the wrong option"
                )
            if errors:
                lines.append(f"- **Server errors**: {len(errors)} — timeouts or API failures")
            lines.append("")

        # Full failures table (collapsible if > 30)
        if failures:
            use_details = len(failures) > 30
            lines.append(f"### Failed Questions ({len(failures)} total)")
            lines.append("")
            if use_details:
                lines.append("<details>")
                lines.append(f"<summary>Show all {len(failures)} failures</summary>")
                lines.append("")

            lines.extend(
                [
                    "| # | Subject | Question (excerpt) | Expected | Got "
                    "| Method | Response (excerpt) |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            for f in failures:
                question = (
                    (f.get("question", "") or "").replace("|", "\\|").replace("\n", " ").strip()
                )
                if len(question) > 120:
                    question = question[:117] + "…"
                resp = (
                    (f.get("model_response", "") or "")
                    .replace("|", "\\|")
                    .replace("\n", " ")
                    .strip()
                )
                if len(resp) > 150:
                    resp = resp[:147] + "…"
                lines.append(
                    f"| {f['index']} | {f.get('subject', '—')} | {question} "
                    f"| {f['ground_truth']} | {f.get('extracted_answer') or '—'} "
                    f"| {f.get('extraction_method', '—')} | {resp} |"
                )

            if use_details:
                lines.append("")
                lines.append("</details>")
            lines.append("")

        # Detailed failure samples (up to 5)
        non_error_failures = [f for f in failures if not f.get("is_error")]
        samples = non_error_failures[:5]
        if samples:
            lines.extend(
                [
                    "### Detailed Failure Samples",
                    "",
                ]
            )
            for f in samples:
                lines.append(f"#### Question #{f['index']} ({f.get('subject', '?')})")
                lines.append("")
                lines.append(
                    f"**Expected:** {f['ground_truth']} · "
                    f"**Got:** {f.get('extracted_answer') or '(none)'} · "
                    f"**Method:** {f.get('extraction_method', '?')}"
                )
                lines.append("")
                question = (f.get("question", "") or "").strip()
                lines.append("**Question:**")
                lines.append("")
                lines.append(f"> {question}")
                lines.append("")
                resp = (f.get("model_response", "") or "").strip()
                if resp:
                    lines.append("**Model response:**")
                    lines.append("")
                    lines.append("```")
                    lines.append(resp[:500])
                    lines.append("```")
                else:
                    lines.append("**Model response:** *(empty)*")
                lines.append("")

        return lines
