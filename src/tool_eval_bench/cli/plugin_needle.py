"""CLI runner for the Needle-in-a-Haystack retrieval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tool_eval_bench.cli.headless import HeadlessConsole, report_run_failed
from tool_eval_bench.cli.plugin_accuracy import print_truncated
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
    truncate,
)
from tool_eval_bench.domain.models import RunContext

# The prompt scaffolding, the question, and the answer budget all live inside
# the context window alongside the haystack.  Sizing haystacks against the raw
# window would push the largest cells past it and score a context overflow as a
# retrieval failure.
_NEEDLE_PROMPT_OVERHEAD = 2048

# The shallowest haystack worth probing.  Below this a miss says nothing about
# long-context retrieval.
_NEEDLE_MIN_TOKENS = 1024


def needle_context_lengths(context_size: int, steps: int) -> list[int]:
    """Evenly spaced haystack sizes from ``_NEEDLE_MIN_TOKENS`` to the window."""
    usable = context_size - _NEEDLE_PROMPT_OVERHEAD
    if usable <= _NEEDLE_MIN_TOKENS:
        return [max(1, usable)]
    if steps < 2:
        return [usable]
    span = usable - _NEEDLE_MIN_TOKENS
    return [int(_NEEDLE_MIN_TOKENS + span * i / (steps - 1)) for i in range(steps)]


def needle_depths(steps: int) -> list[float]:
    """Evenly spaced depths across the haystack, inclusive of both ends."""
    if steps < 2:
        return [0.5]
    return [round(i / (steps - 1), 4) for i in range(steps)]


def resolve_needle_context_size(
    console: Console,
    base_url: str,
    model: str,
    api_key: str | None,
    args: argparse.Namespace,
    run_context: RunContext | None = None,
) -> int | None:
    """Return the effective context window, or ``None`` when it cannot be found."""
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.context_pressure import (
        detect_context_size,
        detect_kv_capacity,
        reported_context_window,
    )

    if args.context_size:
        return int(args.context_size)

    context_size = asyncio.run(
        detect_context_size(
            base_url,
            model,
            api_key,
            client_factory=HTTPMeasurementClient,
            reported_context=reported_context_window(run_context),
        )
    )
    if context_size is None:
        report_run_failed(
            console,
            "\n[bold red]Error:[/] Could not auto-detect the context window. "
            "Use --context-size to specify it.",
        )
        return None

    # Same cap the pressure sweep applies: a server may have allocated far less
    # KV cache than the model architecture allows, and a haystack it cannot hold
    # measures the deployment rather than the model.
    kv_info = asyncio.run(
        detect_kv_capacity(
            base_url,
            api_key,
            metrics_url=getattr(args, "metrics_url", None),
            client_factory=HTTPMeasurementClient,
        )
    )
    if kv_info is not None and not kv_info.is_hybrid and kv_info.capacity < context_size:
        console.print(
            f"  [dim]⚠ KV cache capacity ({kv_info.capacity:,} tokens) < "
            f"max_model_len ({context_size:,}) — capping[/]"
        )
        context_size = kv_info.capacity
    return context_size


def print_needle_grid(console: Console, result: Any) -> None:
    """Print the retrieval grid as depth rows against haystack-size columns."""
    from tool_eval_bench.plugins.needle.plugin import estimated_size_label

    details = result.details
    lengths: list[int] = details.get("context_lengths", [])
    depths: list[float] = details.get("depths", [])
    if not lengths or not depths:
        return

    found = {(r["context_tokens"], r["depth_percent"]): r["found"] for r in result.item_results}

    table = Table(title="Retrieval grid", title_style="bold", border_style="bright_yellow")
    table.add_column("Depth", justify="right", style="dim")
    for length in lengths:
        table.add_column(estimated_size_label(length), justify="center")
    for depth in depths:
        cells = ["[green]●[/]" if found.get((length, depth)) else "[red]○[/]" for length in lengths]
        table.add_row(f"{depth:.0%}", *cells)
    console.print(table)


def run_needle_benchmark(
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
    """Run needle-in-a-haystack retrieval and display the grid."""
    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.plugins.needle.haystack import build_cases
    from tool_eval_bench.plugins.needle.plugin import ESTIMATE_NOTE, NeedlePlugin

    seed = getattr(args, "seed", None)

    context_size = resolve_needle_context_size(
        console, base_url, model, api_key, args, run_context=run_context
    )
    if context_size is None:
        if isinstance(console, HeadlessConsole):
            sys.exit(1)
        return

    lengths = needle_context_lengths(context_size, max(1, args.needle_lengths))
    depths = needle_depths(max(1, args.needle_depths))
    cases = build_cases(lengths, depths, seed=seed)

    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]{len(lengths)} haystack sizes × {len(depths)} depths = "
            f"{len(cases)} needles · up to ~{lengths[-1]:,} tokens (estimated)"
            f"{parallel_label}[/]",
            title="[bold]🪡 Needle in a Haystack — Retrieval[/]",
            border_style="bright_yellow",
        )
    )

    plugin = NeedlePlugin()
    adapter = build_adapter(base_url, **adapter_options(args))
    # A 100K-token prompt takes far longer to prefill than a scenario turn, so
    # the per-request timeout scales with the largest haystack in the grid.
    effective_timeout = max(args.timeout, 120.0 + lengths[-1] / 50_000 * 60.0)

    def _record_item(display: PluginProgressDisplay, item_info: dict[str, Any]) -> None:
        display.tally.record({**item_info, "correct": item_info.get("found")})

    def _format_progress(
        display: PluginProgressDisplay, current: int, item_info: dict[str, Any]
    ) -> tuple[str, str]:
        return (
            tally_line(
                display.tally,
                rate=display.rate_per_minute(current),
                unit="n/min",
                accent="yellow",
            ),
            f"  {status_icon(item_info, key='found')} "
            f"[bold]{item_info.get('cell_id')}[/]  "
            f"[dim italic]{truncate(item_info.get('model_response'))}[/]",
        )

    def _format_finish(res: Any) -> tuple[int, str]:
        d = res.details
        errs = d.get("errors", 0)
        return (
            d["total"],
            final_tally_line(
                correct=d["retrieved"],
                wrong=d["total"] - d["retrieved"] - errs,
                errors=errs,
                score=d["accuracy"],
                rate=items_per_minute(d["total"], res.duration_seconds),
                unit="n/min",
                accent="yellow",
            ),
        )

    async def _run_plugin(on_progress: ProgressCallback) -> Any:
        return await plugin.run(
            adapter,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout_seconds=effective_timeout,
            seed=seed,
            extra_params=extra_params,
            on_progress=on_progress,
            cases=cases,
            concurrency=parallel,
            context_size=context_size,
        )

    result = run_plugin_session(
        console,
        benchmark_name="Needle",
        adapter=adapter,
        total=len(cases),
        run_plugin=_run_plugin,
        format_progress=_format_progress,
        format_finish=_format_finish,
        record_item=_record_item,
    )
    if result is None:
        return
    details = result.details

    console.print()
    print_needle_grid(console, result)
    console.print()
    console.print(
        f"  [bold]Retrieval Accuracy:[/] [bold green]{details['accuracy']:.1f}%[/] "
        f"({details['retrieved']}/{details['total']})"
    )
    effective = details.get("effective_context")
    if effective:
        console.print(
            f"  [bold]Effective context:[/] [bold cyan]~{effective:,}[/] tokens "
            f"[dim](estimated; largest haystack retrieved at every depth)[/]"
        )
    else:
        console.print(
            "  [bold yellow]Effective context:[/] none — every haystack size missed a needle"
        )
    console.print(f"  [dim]{ESTIMATE_NOTE}[/]")
    errs = details.get("errors", 0)
    if errs > 0:
        console.print(f"  [bold yellow]⚠ {errs} errors[/] (counted as misses)")
    print_truncated(console, details)
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    finalize_plugin_run(
        console,
        mode="needle",
        title="Needle in a Haystack",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "needle",
            "context_size": context_size,
            "lengths": lengths,
            "depths": depths,
            "temperature": args.temperature,
            "seed": seed,
            "extra_params": extra_params or None,
        },
        report_metrics=[
            f"- **Retrieval Accuracy**: **{details['accuracy']:.1f}%**",
            "- **Effective Context**: "
            + (f"~{effective:,} tokens (estimated)" if effective else "none"),
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        label=getattr(args, "label", None),
        run_context=run_context,
    )
