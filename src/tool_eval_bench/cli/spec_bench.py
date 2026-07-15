"""Speculative-decoding / MTP benchmark runner for the CLI.

Extracted from the monolithic ``cli/bench.py``. Provides ``run_spec_bench``,
which wraps ``runner.speculative.run_spec_bench`` with a Rich progress UI,
a summary table, and a Markdown report.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rich.console import Console


def run_spec_bench(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    *,
    pp: int,
    tg: int,
    depths: list[int],
    spec_method: str = "auto",
    baseline_tg_tps: float | None = None,
    prompt_types: list[str] | None = None,
    metrics_url: str | None = None,
    output_dir: str | None = None,
    metadata_for_storage: Any = None,
    with_config_fingerprint: Any = None,
    persist_plugin_run: Any = None,
) -> list:
    """Run speculative decoding benchmark and display results.

    Returns a list of SpecDecodeSample objects.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.runner.speculative import SpecDecodeSample, run_spec_bench

    prompt_types = prompt_types or ["filler", "code", "structured"]

    console.print()
    baseline_str = f"  baseline={baseline_tg_tps:.1f} t/s" if baseline_tg_tps else ""
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]tg={tg}  depth={depths}  prompts={prompt_types}  method={spec_method}{baseline_str}[/]",
            title="[bold]🔮 Speculative Decoding Benchmark[/]",
            border_style="bright_magenta",
        )
    )
    console.print()

    completed: list[SpecDecodeSample] = []

    async def on_sample(sample: SpecDecodeSample, idx: int, total: int) -> None:
        completed.append(sample)
        label = f"{sample.prompt_type:>10} @ d{sample.depth}"
        if sample.error:
            console.print(f"  [red]✗[/] {label} — {sample.error}")
        else:
            parts = [
                f"  [green]✓[/] {label}",
                f"  [bold]{sample.effective_tg_tps:,.1f}[/] eff t/s",
                f"  [dim]{sample.tg_tps:,.1f} stream t/s[/]",
            ]

            if sample.acceptance_rate is not None:
                ar_pct = sample.acceptance_rate * 100
                ar_style = "green" if ar_pct >= 60 else "yellow" if ar_pct >= 40 else "red"
                parts.append(f"  [{ar_style}]α={ar_pct:.1f}%[/{ar_style}]")

            if sample.waste_ratio is not None:
                wr_pct = sample.waste_ratio * 100
                wr_style = "green" if wr_pct <= 20 else "yellow" if wr_pct <= 50 else "red"
                parts.append(f"  [{wr_style}]waste={wr_pct:.0f}%[/{wr_style}]")

            if sample.acceptance_length is not None:
                parts.append(f"  [dim]τ={sample.acceptance_length:.1f}[/]")

            if sample.draft_window is not None:
                parts.append(f"  [dim]win={sample.draft_window:.0f}[/]")

            if sample.speedup_ratio is not None:
                sp_style = (
                    "green"
                    if sample.speedup_ratio >= 1.2
                    else "yellow"
                    if sample.speedup_ratio >= 1.0
                    else "red"
                )
                parts.append(f"  [{sp_style}]{sample.speedup_ratio:.2f}x[/{sp_style}]")

            console.print("".join(parts))

    async def _run() -> None:
        await run_spec_bench(
            base_url,
            model,
            pp=pp,
            tg=tg,
            depths=depths,
            api_key=api_key,
            spec_method=spec_method,
            baseline_tg_tps=baseline_tg_tps,
            prompt_types=prompt_types,
            on_sample=on_sample,
            metrics_url=metrics_url,
        )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)

    # Summary table
    ok_samples = [s for s in completed if not s.error]
    has_speedup = any(s.speedup_ratio is not None for s in ok_samples)
    if ok_samples:
        console.print()
        table = Table(
            title="[bold]Speculative Decoding Results[/]",
            show_header=True,
            header_style="bold",
            border_style="bright_magenta",
        )
        has_draft = any(s.draft_tps is not None for s in ok_samples)

        table.add_column("Prompt", no_wrap=True, min_width=10)
        table.add_column("Depth", justify="right", no_wrap=True)
        table.add_column("Eff t/s", justify="right", min_width=7, no_wrap=True)
        table.add_column("α %", justify="right", min_width=6, no_wrap=True)
        table.add_column("Waste", justify="right", min_width=5, no_wrap=True)
        table.add_column("τ len", justify="right", min_width=5, no_wrap=True)
        if has_draft:
            table.add_column("Win", justify="right", no_wrap=True)
            table.add_column("Draft t/s", justify="right", min_width=9, no_wrap=True)
        if has_speedup:
            table.add_column("Speed", justify="right", no_wrap=True)
        table.add_column("TTFT ms", justify="right", min_width=7, no_wrap=True)
        table.add_column("Total ms", justify="right", min_width=8, no_wrap=True)

        def _depth_label(d: int) -> str:
            if d == 0:
                return "0"
            if d >= 1024 and d % 1024 == 0:
                return f"{d // 1024}K"
            return f"{d:,}"

        for s in ok_samples:
            ar_str = f"{s.acceptance_rate * 100:.1f}%" if s.acceptance_rate is not None else "—"
            wr_str = f"{s.waste_ratio * 100:.0f}%" if s.waste_ratio is not None else "—"
            al_str = f"{s.acceptance_length:.1f}" if s.acceptance_length is not None else "—"
            row: list[str] = [
                s.prompt_type,
                _depth_label(s.depth),
                f"{s.effective_tg_tps:,.1f}",
                ar_str,
                wr_str,
                al_str,
            ]
            if has_draft:
                row.append(f"{s.draft_window:.0f}" if s.draft_window is not None else "—")
                row.append(f"{s.draft_tps:,.1f}" if s.draft_tps is not None else "—")
            if has_speedup:
                row.append(f"{s.speedup_ratio:.2f}x" if s.speedup_ratio is not None else "—")
            row.extend(
                [
                    f"{s.ttft_ms:,.0f}",
                    f"{s.total_ms:,.0f}",
                ]
            )
            table.add_row(*row)

        console.print(table)

        # Show insights
        has_ar = any(s.acceptance_rate is not None for s in ok_samples)
        if has_ar:
            best = max(ok_samples, key=lambda s: s.acceptance_rate or 0)
            worst = min(
                ok_samples,
                key=lambda s: s.acceptance_rate if s.acceptance_rate is not None else float("inf"),
            )
            if best.acceptance_rate is not None and worst.acceptance_rate is not None:
                console.print(
                    f"\n  [dim]Highest acceptance:[/] [bold]{best.prompt_type}[/] "
                    f"({best.acceptance_rate * 100:.1f}%)  "
                    f"[dim]Lowest:[/] [bold]{worst.prompt_type}[/] "
                    f"({worst.acceptance_rate * 100:.1f}%)"
                )

            with_window = [
                s
                for s in ok_samples
                if s.draft_window is not None and s.acceptance_length is not None
            ]
            if with_window:
                draft_windows = [s.draft_window for s in with_window if s.draft_window is not None]
                acceptance_lengths = [
                    s.acceptance_length for s in with_window if s.acceptance_length is not None
                ]
                avg_window = sum(draft_windows) / len(draft_windows)
                avg_tau = sum(acceptance_lengths) / len(acceptance_lengths)
                utilization = (avg_tau / avg_window * 100) if avg_window > 0 else 0
                avg_waste = (
                    sum(s.waste_ratio for s in with_window if s.waste_ratio is not None)
                    / len(with_window)
                    * 100
                )
                util_style = (
                    "green" if utilization >= 50 else "yellow" if utilization >= 25 else "red"
                )
                console.print(
                    f"  [dim]Draft window:[/] [{util_style}]{avg_tau:.1f}/{avg_window:.0f} "
                    f"positions used ({utilization:.0f}% utilization)[/{util_style}]  "
                    f"[dim]Avg waste: {avg_waste:.0f}%[/]"
                )
                if utilization < 50:
                    optimal = max(int(avg_tau * 1.5), 2)
                    console.print(
                        f"  [yellow]💡 Consider reducing num_speculative_tokens to "
                        f"~{optimal} (currently ~{avg_window:.0f})[/]"
                    )
        else:
            console.print("\n  [dim]ℹ Acceptance rate: not available (optional).[/]")
            console.print(
                "  [dim]  Effective t/s (shown above) is the primary metric and "
                "already captures MTP/spec-decode speedup.[/]"
            )
            console.print(
                "  [dim]  For acceptance rate breakdown, ensure your server exposes "
                "/metrics with spec_decode counters[/]"
            )
            console.print(
                "  [dim]  (vLLM: enabled by default at http://<host>:<port>/metrics; "
                "llama.cpp: start with --metrics flag).[/]"
            )

    # Write report
    if ok_samples:
        from tool_eval_bench.utils.ids import build_run_id

        run_config = with_config_fingerprint(
            {
                "model": model,
                "base_url": base_url,
                "mode": "spec-bench",
                "method": spec_method,
            }
        )
        run_id = build_run_id(run_config)
        from tool_eval_bench.storage.reports import MarkdownReporter

        reporter = MarkdownReporter(root=output_dir)
        report_path = reporter.write_spec_decode_report(run_id, display_name, ok_samples)
        persist_plugin_run(
            {
                "run_id": run_id,
                "run_type": "spec-bench",
                "status": "completed",
                "config": run_config,
                "scores": {"samples": len(ok_samples)},
                "metadata": metadata_for_storage(None),
            }
        )
        console.print(f"\n  [dim]📄 Report saved to {report_path}[/]")

    try:
        from tool_eval_bench import __version__

        console.print(f"  [dim]tool-eval-bench v{__version__}[/]")
    except ImportError:
        pass
    console.print()
    return completed
