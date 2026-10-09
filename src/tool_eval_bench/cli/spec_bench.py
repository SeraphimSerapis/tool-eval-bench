"""Speculative-decoding / MTP benchmark runner for the CLI.

Extracted from the monolithic ``cli/bench.py``. Provides ``run_spec_bench``,
which wraps ``runner.speculative.run_spec_bench`` with a Rich progress UI,
a summary table, and a Markdown report.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.cli.headless import report_run_failed, report_run_saved
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.spec_decode import suggested_draft_window, window_utilization
from tool_eval_bench.runner.spec_detection import canonical_spec_method_hint
from tool_eval_bench.storage.reports.spec_decode import spec_decode_report


def load_spec_prompt_file(path: str | None) -> dict[str, str]:
    """Read a user workload for spec-bench: label to prompt text.

    Plain lines become ``custom#N``. A line that parses as a JSON object uses
    its ``prompt`` and optional ``label`` keys, which is the only way to carry
    a multi-line prompt. Blank lines and ``#`` comments are skipped.
    """
    if not path:
        return {}
    prompts: dict[str, str] = {}
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label = f"custom#{len(prompts) + 1}"
        text = line
        if line.startswith("{"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON line: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                raise ValueError(f"{path}:{number}: JSON line needs a string 'prompt'")
            text = record["prompt"]
            if isinstance(record.get("label"), str) and record["label"].strip():
                label = record["label"].strip()
        if label in prompts:
            raise ValueError(f"{path}:{number}: duplicate prompt label {label!r}")
        prompts[label] = text
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def _spec_bench_config(
    model: str,
    base_url: str,
    *,
    spec_method: str,
    runs: int,
    temperature: float,
    pp: int,
    tg: int,
    depths: list[int],
    prompt_types: list[str],
    baseline_tg_tps: float | None,
    custom_prompts: dict[str, str] | None,
) -> dict[str, Any]:
    """The stored spec-bench config; every key joins the comparison fingerprint.

    The workload (prompt lengths, depths, prompt selection, baseline) is part of
    it, so runs that measured different things never share a cohort. Custom
    prompt text is hashed rather than stored.
    """
    config: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "mode": "spec-bench",
        "method": canonical_spec_method_hint(spec_method) or "auto",
        "runs": runs,
        "temperature": temperature,
        "pp": pp,
        "tg": tg,
        # Sorted: the order cells run in does not change what was measured.
        "depths": sorted(depths),
        "prompt_types": sorted(prompt_types),
        "baseline_tg_tps": baseline_tg_tps,
    }
    selected = {
        label: text for label, text in (custom_prompts or {}).items() if label in prompt_types
    }
    if selected:
        payload = json.dumps(selected, sort_keys=True, ensure_ascii=False).encode("utf-8")
        config["custom_prompts_sha256"] = hashlib.sha256(payload).hexdigest()
    return config


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
    runs: int = 1,
    temperature: float = 0.0,
    custom_prompts: dict[str, str] | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    run_context: RunContext | None = None,
) -> list:
    """Run speculative decoding benchmark and display results.

    Returns one SpecDecodeSample per cell, failed cells included. The run is
    always stored; when any cell failed it is stored as failed and a
    ``run_failed`` message is reported, but this does not exit, because a
    tool-call run or plugin may still follow. The caller exits non-zero when
    spec-bench is the last mode. An interrupt or error mid-run stores the
    cells that finished as a failed run, then exits 1.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.domain.spec_decode import per_position_acceptance
    from tool_eval_bench.runner.speculative import SpecDecodeSample, run_spec_bench

    prompt_types = prompt_types or ["filler", "code", "structured"]
    method_label = canonical_spec_method_hint(spec_method) or spec_method

    console.print()
    baseline_str = f"  baseline={baseline_tg_tps:.1f} t/s" if baseline_tg_tps else ""
    runs_str = f"  runs={runs}" if runs > 1 else ""
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]tg={tg}  depth={depths}  prompts={prompt_types}  method={method_label}"
            f"{runs_str}  temperature={temperature:g}{baseline_str}[/]",
            title="[bold]🔮 Speculative Decoding Benchmark[/]",
            border_style="bright_magenta",
        )
    )
    console.print()

    completed: list[SpecDecodeSample] = []

    async def on_sample(sample: SpecDecodeSample, idx: int, total: int) -> None:
        completed.append(sample)
        label = f"{sample.prompt_type:>10} @ d{sample.depth}"
        if sample.runs > 1:
            label += f" ×{sample.runs}"
        if sample.failed_runs and not sample.error:
            label += f" [yellow]({sample.failed_runs} failed)[/]"
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
                if sample.acceptance_rate_range is not None:
                    lo, hi = sample.acceptance_rate_range
                    parts.append(f"[dim] ({lo * 100:.0f}–{hi * 100:.0f})[/]")

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
            client_factory=HTTPMeasurementClient,
            spec_method=spec_method,
            baseline_tg_tps=baseline_tg_tps,
            prompt_types=prompt_types,
            on_sample=on_sample,
            metrics_url=metrics_url,
            runs=runs,
            temperature=temperature,
            custom_prompts=custom_prompts,
        )

    def _persist(*, stopped_early: bool) -> int:
        """Store the run and write its report; return the failed-cell count.

        Like --perf-only, every run is stored, and a cell that failed on every
        run marks the whole run failed, as does a run that stopped early.
        """
        ok = [s for s in completed if not s.error]
        failed_count = len(completed) - len(ok)
        run = ModeRun(
            run_type="spec-bench",
            config=_spec_bench_config(
                model,
                base_url,
                spec_method=spec_method,
                runs=runs,
                temperature=temperature,
                pp=pp,
                tg=tg,
                depths=depths,
                prompt_types=prompt_types,
                baseline_tg_tps=baseline_tg_tps,
                custom_prompts=custom_prompts,
            ),
            scores={
                "samples": len(ok),
                # A count only: error text can quote the server URL.
                "failed": failed_count,
                # Runs that failed inside cells that still have results.
                "failed_runs": sum(s.failed_runs for s in ok),
                "results": [s.to_result() for s in ok],
            },
            status="failed" if failed_count or stopped_early else "completed",
        )
        finalized = finalize_mode_run(
            run,
            spec_decode_report(
                display_name,
                ok,
                label=label,
                temperature=temperature,
                failed=failed_count,
                stopped_early=stopped_early,
            ),
            run_context=run_context,
            output_dir=output_dir,
        )
        report_run_saved(console, run, finalized)
        console.print(f"\n  [dim]📄 Report saved to {finalized.report_path}[/]")
        return failed_count

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Keep the cells that finished; an interrupted run is a failed run.
        if completed:
            _persist(stopped_early=True)
        report_run_failed(console, "\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        if completed:
            _persist(stopped_early=True)
        report_run_failed(console, f"\n[bold red]Error: {exc}[/]")
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
            table.add_column("Steps/s", justify="right", min_width=7, no_wrap=True)
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
                row.append(
                    f"{s.verify_steps_per_s:,.1f}" if s.verify_steps_per_s is not None else "—"
                )
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
            sources = {s.acceptance_source for s in ok_samples if s.acceptance_source}
            if sources == {"response"}:
                console.print(
                    "\n  [dim]Acceptance source:[/] per-request response metrics "
                    "[green](exact, unaffected by concurrent traffic)[/]"
                )
            elif sources == {"timings"}:
                console.print(
                    "\n  [dim]Acceptance source:[/] per-request response timings "
                    "[green](exact, unaffected by concurrent traffic)[/]"
                )
            elif "prometheus" in sources:
                console.print(
                    "\n  [dim]Acceptance source:[/] Prometheus counter deltas "
                    "[yellow](server-wide; concurrent traffic skews per-request values)[/]"
                )
                console.print(
                    "  [dim]  vLLM can report exact per-request values: start the server "
                    "with --per-request-spec-decode-metrics summary|detailed[/]"
                )
            step_arrays = [
                (s.per_step_accepted, s.per_step_drafted)
                for s in ok_samples
                if s.per_step_accepted is not None and s.per_step_drafted is not None
            ]
            per_pos = per_position_acceptance(step_arrays) if step_arrays else []
            if per_pos:
                cells = []
                for pos, rate in enumerate(per_pos):
                    pct = rate * 100
                    style = "green" if pct >= 60 else "yellow" if pct >= 40 else "red"
                    cells.append(f"[dim]{pos}:[/][{style}]{pct:.0f}%[/{style}]")
                console.print(f"  [dim]Per-position acceptance:[/] {'  '.join(cells)}")
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
                utilization = (window_utilization(avg_tau, avg_window) or 0.0) * 100
                avg_waste = (
                    sum(s.waste_ratio for s in with_window if s.waste_ratio is not None)
                    / len(with_window)
                    * 100
                )
                util_style = (
                    "green" if utilization >= 50 else "yellow" if utilization >= 25 else "red"
                )
                console.print(
                    f"  [dim]Draft window:[/] [{util_style}]{avg_tau - 1:.1f}/{avg_window:.0f} "
                    f"drafted positions accepted ({utilization:.0f}% utilization)"
                    f"[/{util_style}]  [dim]Avg waste: {avg_waste:.0f}%[/]"
                )
                optimal = suggested_draft_window(avg_tau, avg_window)
                if utilization < 50 and optimal is not None:
                    console.print(
                        f"  [yellow]💡 Consider reducing num_speculative_tokens to "
                        f"~{optimal} (currently ~{avg_window:.0f})[/]"
                    )
            if not has_speedup:
                with_steps = [s for s in ok_samples if s.verify_steps_per_s]
                if with_steps:
                    ceilings = [
                        s.effective_tg_tps / s.verify_steps_per_s
                        for s in with_steps
                        if s.verify_steps_per_s
                    ]
                    console.print(
                        f"  [dim]Speedup ceiling:[/] [bold]{min(ceilings):.1f}–{max(ceilings):.1f}x[/] "
                        "[dim](eff t/s ÷ verify steps/s; the real figure needs "
                        "--baseline-tgs from a run without speculation)[/]"
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

    failed_count = _persist(stopped_early=False)
    if failed_count:
        # The caller decides whether to exit: a combined run still has modes to go.
        report_run_failed(
            console,
            f"[bold red]Speculative decoding benchmark failed in {failed_count} cell(s).[/]",
        )

    try:
        from tool_eval_bench import __version__

        console.print(f"  [dim]tool-eval-bench v{__version__}[/]")
    except ImportError:
        pass
    console.print()
    return completed
