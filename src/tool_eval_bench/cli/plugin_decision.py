"""CLI runner for the Decision Models (/v1/systemone) benchmark."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from tool_eval_bench.cli.headless import report_run_failed
from tool_eval_bench.cli.plugin_lifecycle import (
    ProgressCallback,
    finalize_plugin_run,
    items_per_minute,
    run_plugin_session,
)
from tool_eval_bench.cli.plugin_progress import (
    PluginProgressDisplay,
    final_tally_line,
    status_icon,
    tally_line,
)
from tool_eval_bench.domain.models import RunContext


def print_decision_tables(console: Console, result: Any) -> None:
    """Print accuracy and distance from gold by question type and workflow, then reliability."""
    from tool_eval_bench.plugins.decision.render import bar, metric_text

    details = result.details

    for title, label, key in (
        ("Accuracy by question type", "Type", "by_type"),
        ("Accuracy by workflow", "Workflow", "by_workflow"),
    ):
        table = Table(title=title, title_style="bold", border_style="bright_cyan")
        table.add_column(label)
        table.add_column("Correct", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("KL", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("")
        for name, g in details[key].items():
            table.add_row(
                name,
                f"{g['correct']}/{g['total']}",
                f"{g['accuracy']:.1f}%",
                metric_text(g["kl"]),
                metric_text(g["brier"]),
                bar(g["accuracy"] / 100),
            )
        console.print(table)

    bins = details["calibration"]["bins"]
    if bins:
        rel = Table(
            title="Reliability (confidence vs accuracy)",
            title_style="bold",
            border_style="bright_cyan",
        )
        rel.add_column("Confidence", justify="right", style="dim")
        rel.add_column("n", justify="right")
        rel.add_column("Confidence")
        rel.add_column("Accuracy")
        for b in bins:
            rel.add_row(
                f"{b['low']:.1f}–{b['high']:.1f}",
                str(b["count"]),
                f"{bar(b['mean_confidence'])} {b['mean_confidence']:.2f}",
                f"{bar(b['accuracy'])} {b['accuracy']:.2f}",
            )
        console.print(rel)


def run_decision_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: RunContext | None = None,
) -> None:
    """Score a decision model on the typed-decisions test split and display results."""
    from tool_eval_bench.adapters.factory import build_decision_adapter
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.plugins.decision.plugin import DecisionPlugin
    from tool_eval_bench.plugins.decision.render import attribution_line, metric_text
    from tool_eval_bench.plugins.decision.typed_decisions import (
        DatasetIntegrityError,
        dataset_info,
        load_cases,
    )

    try:
        info = dataset_info()
    except DatasetIntegrityError as exc:
        report_run_failed(console, f"\n[bold red]Decision error:[/] {escape(str(exc))}")
        sys.exit(1)
    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]Typed Decisions {info.split} split · every question of a case in one "
            f"request · single forward pass{parallel_label}[/]\n"
            f"[dim]{escape(info.attribution)}[/]",
            title="[bold]⚖️  Decision Models — /v1/systemone[/]",
            border_style="bright_cyan",
        )
    )

    plugin = DecisionPlugin()
    adapter = build_decision_adapter(max_concurrency=parallel, **adapter_options(args))
    cases_holder: list[Any] = []

    def _prepare_cases() -> int:
        cases = load_cases()
        cases_holder.append(cases)
        return sum(len(c.questions) for c in cases)

    def _format_progress(
        display: PluginProgressDisplay, current: int, item_info: dict[str, Any]
    ) -> tuple[str, str]:
        return (
            tally_line(
                display.tally,
                rate=display.rate_per_minute(current),
                unit="decisions/min",
                accent="cyan",
            ),
            f"  {status_icon(item_info)} [bold]{item_info.get('id')}[/]",
        )

    def _format_finish(res: Any) -> tuple[int, str]:
        rows = res.item_results
        errs = sum(1 for r in rows if r["is_error"])
        right = sum(1 for r in rows if r["correct"])
        return (
            len(rows),
            final_tally_line(
                correct=right,
                wrong=len(rows) - right - errs,
                errors=errs,
                score=right / len(rows) * 100 if rows else 0.0,
                rate=items_per_minute(len(rows), res.duration_seconds),
                unit="decisions/min",
                accent="cyan",
            ),
        )

    async def _run_plugin(on_progress: ProgressCallback) -> Any:
        return await plugin.run(
            adapter,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=args.timeout,
            on_progress=on_progress,
            cases=cases_holder[0],
            concurrency=parallel,
        )

    result = run_plugin_session(
        console,
        benchmark_name="Decision",
        adapter=adapter,
        total=_prepare_cases,
        run_plugin=_run_plugin,
        format_progress=_format_progress,
        format_finish=_format_finish,
    )
    if result is None:
        return
    details = result.details
    cal = details["calibration"]

    console.print()
    print_decision_tables(console, result)
    console.print()
    console.print(
        f"  [bold]Decision Accuracy:[/] [bold cyan]{result.score:.1f}%[/] "
        f"({details['correct']}/{details['total']} decisions)"
    )
    console.print(
        f"  [bold]Distance from gold:[/] KL {metric_text(details['kl'])} · "
        f"Brier {metric_text(details['brier'])} · "
        f"ECE {cal['ece']:.3f} · {cal['high_confidence_errors']} confident disagreement(s)"
    )
    errs = details.get("errors", 0)
    if errs > 0:
        console.print(f"  [bold yellow]⚠ {errs} decision errors[/] (counted as misses)")
    console.print(f"  [bold]Rating:[/] {result.rating}")
    lat = details["latency_ms"]
    console.print(
        f"  [dim]p50 {lat['p50']:.1f} ms · p95 {lat['p95']:.1f} ms per request · "
        f"Duration: {result.duration_seconds:.1f}s · Input tokens: {result.total_tokens:,}[/]"
    )
    console.print(f"  [dim]{escape(attribution_line(details))}[/]")

    finalize_plugin_run(
        console,
        mode="decision",
        title="Decision Models",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "decision",
            # Runs on another item set or revision must not share a fingerprint.
            "dataset": details["dataset"],
            "dataset_revision": details["dataset_revision"],
            "dataset_split": details["dataset_split"],
            "cases": details["cases"],
        },
        report_metrics=[
            f"- **Accuracy**: **{result.score:.1f}%**",
            f"- **KL from gold**: {metric_text(details['kl'])}",
            f"- **Brier**: {metric_text(details['brier'])}",
            f"- **ECE**: {cal['ece']:.3f}",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
            f"- **Dataset**: {details['dataset']} ({details['dataset_split']} split, "
            f"revision `{details['dataset_revision']}`)",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        label=getattr(args, "label", None),
        run_context=run_context,
    )
