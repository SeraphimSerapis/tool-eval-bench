"""IFEval benchmark plugin — orchestrator and report rendering.

Implements ``BenchmarkPlugin`` for the Instruction Following Evaluation
benchmark (541 prompts, 25 constraint types).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
)
from tool_eval_bench.plugins.ifeval.dataset import IFEvalItem, load_dataset
from tool_eval_bench.plugins.ifeval.evaluator import evaluate_prompt

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


class IFEvalPlugin(BenchmarkPlugin):
    """IFEval benchmark — 541 prompts with 25 constraint types."""

    @property
    def name(self) -> str:
        return "ifeval"

    @property
    def description(self) -> str:
        return "Instruction Following Evaluation (541 prompts, 25 constraint types)"

    async def run(
        self,
        adapter: BackendAdapter,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        seed: int | None = None,
        extra_params: dict[str, Any] | None = None,
        on_progress: OnPluginProgress | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run IFEval evaluation."""
        limit: int = kwargs.get("limit", 0)
        concurrency: int = kwargs.get("concurrency", 1)
        preloaded = kwargs.get("_preloaded_items")

        # Load dataset
        if preloaded is not None:
            all_items = list(preloaded)
        else:
            on_download = kwargs.get("on_download_progress")
            all_items = load_dataset(on_progress=on_download)

        logger.info("Loaded %d IFEval prompts", len(all_items))

        if limit > 0:
            all_items = all_items[:limit]

        total = len(all_items)
        if total == 0:
            return BenchmarkResult(
                plugin_name="ifeval",
                score=0.0,
                score_label="0/0",
                rating=_rating_for_accuracy(0),
                details={"prompts_passed": 0, "total": 0},
            )

        # Evaluate
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = [{}] * total
        prompts_passed = 0
        error_count = 0
        instructions_passed = 0
        instructions_total = 0
        total_tokens = 0
        progress_counter = 0
        progress_lock = asyncio.Lock()
        t_start = time.monotonic()

        async def eval_one(idx: int, item: IFEvalItem) -> None:
            nonlocal prompts_passed, instructions_passed, instructions_total
            nonlocal total_tokens, error_count, progress_counter

            messages = [
                {"role": "user", "content": item.prompt},
            ]

            try:
                async with sem:
                    response = await adapter.chat_completion(
                        model=model,
                        messages=messages,
                        tools=None,
                        temperature=temperature,
                        max_tokens=4096,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra_params,
                    )

                content = response.content or response.reasoning or ""
                total_tokens += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                is_error = False
            except Exception as exc:
                logger.debug("Error on prompt %d: %s", item.key, exc)
                content = ""
                is_error = True
                error_count += 1

            if is_error:
                results[idx] = {
                    "key": item.key,
                    "prompt": item.prompt[:200],
                    "prompt_pass": False,
                    "is_error": True,
                    "instructions_passed": 0,
                    "instructions_total": len(item.instruction_id_list),
                    "instruction_details": [],
                    "model_response": "",
                }
                instructions_total += len(item.instruction_id_list)
            else:
                result = evaluate_prompt(content, item.instruction_id_list, item.kwargs)

                if result.prompt_pass:
                    prompts_passed += 1
                instructions_passed += result.instructions_passed
                instructions_total += result.instructions_total

                results[idx] = {
                    "key": item.key,
                    "prompt": item.prompt[:200],
                    "prompt_pass": result.prompt_pass,
                    "instructions_passed": result.instructions_passed,
                    "instructions_total": result.instructions_total,
                    "instruction_details": [
                        {
                            "id": ir.instruction_id,
                            "passed": ir.passed,
                            "error": ir.error,
                        }
                        for ir in result.instruction_results
                    ],
                    "model_response": content[:1000],
                }

            if on_progress:
                async with progress_lock:
                    progress_counter += 1
                    await on_progress(progress_counter, total, results[idx])

        tasks = [eval_one(i, item) for i, item in enumerate(all_items)]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, exc in enumerate(gather_results):
            if isinstance(exc, BaseException):
                logger.error("IFEval prompt %d crashed: %s", i, exc)

        duration = time.monotonic() - t_start
        answered = total - error_count
        prompt_accuracy = (prompts_passed / answered * 100) if answered > 0 else 0.0
        instruction_accuracy = (
            instructions_passed / instructions_total * 100 if instructions_total > 0 else 0
        )

        # Per-constraint-type breakdown
        constraint_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        for r in results:
            for detail in r.get("instruction_details", []):
                cid = detail["id"]
                constraint_stats[cid]["total"] += 1
                if detail["passed"]:
                    constraint_stats[cid]["passed"] += 1

        return BenchmarkResult(
            plugin_name="ifeval",
            score=round(prompt_accuracy, 2),
            score_label=(
                f"Prompt: {prompt_accuracy:.1f}% ({prompts_passed}/{total}) | "
                f"Instruction: {instruction_accuracy:.1f}% "
                f"({instructions_passed}/{instructions_total})"
            ),
            rating=_rating_for_accuracy(prompt_accuracy),
            details={
                "prompts_passed": prompts_passed,
                "total": total,
                "errors": error_count,
                "prompt_accuracy": round(prompt_accuracy, 2),
                "instruction_accuracy": round(instruction_accuracy, 2),
                "instructions_passed": instructions_passed,
                "instructions_total": instructions_total,
                "dataset_size": 541,
                "constraint_types": {
                    cid: {
                        "passed": s["passed"],
                        "total": s["total"],
                        "accuracy": round(s["passed"] / s["total"] * 100, 1) if s["total"] else 0,
                    }
                    for cid, s in sorted(constraint_stats.items())
                },
            },
            item_results=results,
            metadata={"dataset": "google/IFEval"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render Markdown report section for IFEval results."""
        d = result.details
        lines = [
            "## IFEval — Instruction Following Evaluation",
            "",
            f"**Prompt-level Accuracy:** {d.get('prompt_accuracy', 0):.1f}% "
            f"({d['prompts_passed']}/{d['total']})",
            f"**Instruction-level Accuracy:** {d.get('instruction_accuracy', 0):.1f}% "
            f"({d.get('instructions_passed', 0)}/{d.get('instructions_total', 0)})",
            f"**Rating:** {result.rating}",
            f"**Duration:** {result.duration_seconds:.1f}s",
            f"**Tokens:** {result.total_tokens:,}",
            "",
        ]

        # Per-constraint-type breakdown
        ct = d.get("constraint_types", {})
        if ct:
            lines.extend(
                [
                    "### Per-Constraint Accuracy",
                    "",
                    "| Constraint | Passed | Total | Accuracy |",
                    "|---|---|---|---|",
                ]
            )
            # Sort by accuracy ascending (worst first)
            sorted_ct = sorted(
                ct.items(),
                key=lambda x: x[1].get("accuracy", 0),
            )
            for cid, stats in sorted_ct:
                lines.append(
                    f"| `{cid}` | {stats['passed']} | {stats['total']} | {stats['accuracy']:.1f}% |"
                )
            lines.append("")

        # Error analysis
        failures = [r for r in result.item_results if not r.get("prompt_pass")]
        errors = [r for r in failures if r.get("is_error")]
        non_errors = [r for r in failures if not r.get("is_error")]

        if failures:
            lines.extend(
                [
                    "### Error Analysis",
                    "",
                    f"- **Total failed prompts**: {len(failures)} / {d['total']}",
                ]
            )
            if non_errors:
                lines.append(
                    f"- **Constraint violations**: {len(non_errors)} — model "
                    "responded but did not satisfy all instruction constraints"
                )
            if errors:
                lines.append(f"- **Server errors**: {len(errors)} — timeouts or API failures")
            lines.append("")

        # Full failures table (collapsible if > 30)
        if failures:
            use_details = len(failures) > 30
            lines.append(f"### Failed Prompts ({len(failures)} total)")
            lines.append("")
            if use_details:
                lines.append("<details>")
                lines.append(f"<summary>Show all {len(failures)} failures</summary>")
                lines.append("")

            lines.extend(
                [
                    "| Key | Prompt (excerpt) | Instructions | Passed | Failed Constraints |",
                    "|---|---|---:|---:|---|",
                ]
            )
            for f in failures:
                prompt = (f.get("prompt", "") or "").replace("|", "\\|").replace("\n", " ").strip()
                if len(prompt) > 120:
                    prompt = prompt[:117] + "…"
                total_instr = f.get("instructions_total", 0)
                passed_instr = f.get("instructions_passed", 0)
                failed_ids = [
                    d_item["id"]
                    for d_item in f.get("instruction_details", [])
                    if not d_item["passed"]
                ]
                failed_str = ", ".join(f"`{fid}`" for fid in failed_ids) if failed_ids else "—"
                lines.append(
                    f"| {f['key']} | {prompt} | {total_instr} | {passed_instr} | {failed_str} |"
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
                failed_ids = [
                    d_item for d_item in f.get("instruction_details", []) if not d_item["passed"]
                ]
                lines.append(f"#### Prompt #{f['key']}")
                lines.append("")
                lines.append(
                    f"**Passed:** {f.get('instructions_passed', 0)}/"
                    f"{f.get('instructions_total', 0)} instructions"
                )
                lines.append("")
                if failed_ids:
                    lines.append("**Failed constraints:**")
                    lines.append("")
                    for d_item in failed_ids:
                        err = d_item.get("error", "")
                        err_str = f" — {err}" if err else ""
                        lines.append(f"- `{d_item['id']}`{err_str}")
                    lines.append("")
                prompt = (f.get("prompt", "") or "").strip()
                lines.append("**Prompt:**")
                lines.append("")
                lines.append(f"> {prompt}")
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
