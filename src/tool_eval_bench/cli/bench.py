"""CLI entry point for running tool-call benchmarks.

Defaults cascade:  .env file → TOOL_EVAL_* env vars → hardcoded fallbacks.

Usage:
    tool-eval-bench                           # uses .env / env vars
    tool-eval-bench --base-url URL            # override server
    tool-eval-bench --short                   # core 15 scenarios only

The --model flag is optional: if omitted, the CLI will query the server's
/v1/models endpoint and auto-select (1 model) or prompt the user (multiple).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from statistics import mean, stdev
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from tool_eval_bench.cli.display import BenchmarkDisplay
from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition, ScenarioResult, ScenarioStatus
from tool_eval_bench.runner.service import BenchmarkService


# Valid category letters for --categories
_VALID_CATEGORIES = {c.value for c in Category}


def _resolve_scenarios(args: argparse.Namespace) -> list[ScenarioDefinition]:
    """Resolve scenarios from --short, --scenarios, and --categories flags.

    Priority: --scenarios (individual IDs) > --categories > --short > all.
    """
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, SCENARIOS

    base = SCENARIOS if args.short else ALL_SCENARIOS

    if args.scenarios:
        requested = set(args.scenarios)
        return [s for s in base if s.id in requested]

    if args.categories:
        cats = {c.upper() for c in args.categories}
        return [s for s in base if s.category.value in cats]

    return list(base)


# ---------------------------------------------------------------------------
# Load .env (same logic as tui/settings.py, inlined to avoid import coupling)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env file into os.environ (does not overwrite existing vars)."""
    load_dotenv(override=False)


def _redact_url(url: str) -> str:
    """Mask the host in a URL for display.  e.g. http://192.168.10.5:8080 → http://***:8080"""
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    # Replace hostname with ***, keep port if present
    redacted_netloc = "***"
    if parsed.port:
        redacted_netloc = f"***:{parsed.port}"
    return urlunparse(parsed._replace(netloc=redacted_netloc))



# ---------------------------------------------------------------------------
# Model auto-detection
# ---------------------------------------------------------------------------

def _detect_model(
    base_url: str, api_key: str | None, console: Console,
    *, display_url: str | None = None,
) -> tuple[str, str]:
    """Query /v1/models and auto-select or let the user pick.

    Returns (api_id, display_name).
      - api_id:       what to send in API requests (e.g. "gemma4")
      - display_name: the real model path if available (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")
    """
    import httpx

    url = base_url.rstrip("/")
    models_endpoint = f"{url}/v1/models"
    # Handle base_url that already ends with /v1
    if url.endswith("/v1"):
        models_endpoint = f"{url}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build a display-safe endpoint URL for console output
    show_url = display_url or base_url
    show_endpoint = f"{show_url.rstrip('/')}/v1/models"
    if show_url.rstrip("/").endswith("/v1"):
        show_endpoint = f"{show_url.rstrip('/')}/models"
    console.print(f"[dim]  Querying {show_endpoint} …[/]", end=" ")

    used_fallback = False

    async def _fetch() -> tuple[httpx.Response, bool]:
        nonlocal used_fallback
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(models_endpoint, headers=headers)
            if resp.status_code == 404:
                fallback_url = f"{url}/models"
                resp = await client.get(fallback_url, headers=headers)
                used_fallback = True
            return resp, used_fallback

    try:
        resp, used_fallback = asyncio.run(_fetch())
        resp.raise_for_status()
    except httpx.ConnectError:
        console.print("[bold red]✗ cannot connect[/]")
        console.print(f"\n[red]Could not connect to {show_url}. Is the server running?[/]")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        console.print(f"[bold red]✗ HTTP {exc.response.status_code}[/]")
        console.print(f"\n[red]Server returned {exc.response.status_code}. Check the URL and API key.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]✗ {exc}[/]")
        sys.exit(1)

    if used_fallback:
        console.print("\n  [yellow]⚠ /v1/models returned 404, used /models fallback. "
                       "Check your server configuration.[/]")

    try:
        data = resp.json()
        model_list = data.get("data", [])
    except Exception:
        console.print("[bold red]✗ invalid response[/]")
        console.print("[red]Server returned invalid JSON from /v1/models.[/]")
        sys.exit(1)

    # Build (api_id, display_name) pairs
    # vLLM: "id" is the served alias, "root" is the actual model path
    # LiteLLM/others: may not have "root"
    models: list[tuple[str, str]] = []
    for m in model_list:
        api_id = m.get("id", "")
        if not api_id:
            continue
        root = m.get("root", "")
        # Use root as display name if it differs from the alias
        display = root if root and root != api_id else api_id
        models.append((api_id, display))

    if not models:
        console.print("[bold red]✗ no models found[/]")
        console.print("[red]The server returned an empty model list.[/]")
        sys.exit(1)

    if len(models) == 1:
        api_id, display = models[0]
        if display != api_id:
            console.print(f"[bold green]✓[/] [bold]{display}[/] [dim](alias: {api_id})[/]")
        else:
            console.print(f"[bold green]✓[/] [bold]{api_id}[/]")
        return api_id, display

    # Multiple models — let the user choose
    console.print(f"[bold cyan]found {len(models)} models[/]")
    console.print()
    console.print("[bold]Available models:[/]")
    for i, (api_id, display) in enumerate(models, 1):
        if display != api_id:
            console.print(f"  [bold cyan]{i}[/]) {display} [dim](alias: {api_id})[/]")
        else:
            console.print(f"  [bold cyan]{i}[/]) {api_id}")
    console.print()

    while True:
        try:
            choice = input(f"Select model [1-{len(models)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                api_id, display = models[idx]
                console.print(f"\n[dim]  Selected:[/] [bold]{display}[/]\n")
                return api_id, display
            console.print(f"[red]  Please enter a number between 1 and {len(models)}.[/]")
        except (ValueError, EOFError):
            console.print(f"[red]  Please enter a number between 1 and {len(models)}.[/]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Cancelled.[/]")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Plain-text fallback (for --json or --no-live)
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

STATUS_STYLE = {
    ScenarioStatus.PASS: f"{GREEN}✅ PASS{RESET}",
    ScenarioStatus.PARTIAL: f"{YELLOW}⚠️  PARTIAL{RESET}",
    ScenarioStatus.FAIL: f"{RED}❌ FAIL{RESET}",
}


async def _plain_on_start(scenario: ScenarioDefinition, idx: int, total: int) -> None:
    print(f"  {DIM}[{idx + 1}/{total}]{RESET} {scenario.id} {scenario.title}... ", end="", flush=True)


async def _plain_on_result(
    scenario: ScenarioDefinition, result: ScenarioResult, idx: int, total: int
) -> None:
    style = STATUS_STYLE.get(result.status, "?")
    print(f"{style}  ({result.points}/2) {DIM}{result.summary}{RESET}")


# ---------------------------------------------------------------------------
# Server warm-up
# ---------------------------------------------------------------------------

def _do_warmup(console: Console, base_url: str, model: str, api_key: str | None) -> None:
    """Send a trivial request to prime the server before benchmarking."""
    from tool_eval_bench.runner.throughput import warmup

    with console.status("[dim]  Warming up server…[/]", spinner="dots"):
        try:
            ms = asyncio.run(warmup(base_url, model, api_key, timeout=30.0))
            console.print(f"  [bold green]✓[/] Warm-up complete [dim]({ms:.0f} ms)[/]")
        except Exception as exc:
            console.print(f"  [bold yellow]⚠[/] Warm-up failed [dim]({exc})[/]")


# ---------------------------------------------------------------------------
# Throughput benchmark (--perf / --perf-only)
# ---------------------------------------------------------------------------

def _run_throughput(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    *,
    pp: int,
    tg: int,
    depths: list[int],
    concurrency_levels: list[int],
) -> list:
    """Run llama-bench style throughput sweep and display results.

    Returns a list of ThroughputSample objects for report persistence.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.runner.throughput import ThroughputSample, run_throughput_matrix

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]pp={pp}  tg={tg}  depth={depths}  concurrency={concurrency_levels}[/]",
            title="[bold]⚡ Throughput Benchmark[/]",
            border_style="bright_cyan",
        )
    )

    completed: list[ThroughputSample] = []

    async def on_sample(sample: ThroughputSample, idx: int, total: int) -> None:
        completed.append(sample)
        label = f"pp{sample.label_pp} @ d{sample.label_depth} c{sample.concurrency}"
        if sample.error:
            console.print(f"  [red]✗[/] {label} — {sample.error}")
        else:
            console.print(
                f"  [green]✓[/] {label}  "
                f"[bold]{sample.pp_tps:,.0f}[/] pp t/s  "
                f"[bold]{sample.tg_tps:,.1f}[/] tg t/s  "
                f"[dim]ttft={sample.ttft_ms:,.0f}ms  total={sample.total_ms:,.0f}ms[/]"
            )

    matrix_result_holder: list[object] = []

    async def run() -> None:
        result = await run_throughput_matrix(
            base_url, model,
            pp=pp, tg=tg,
            depths=depths,
            concurrency_levels=concurrency_levels,
            api_key=api_key,
            on_sample=on_sample,
        )
        matrix_result_holder.append(result)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)

    # Summary table
    ok_samples = [s for s in completed if not s.error]
    if ok_samples:
        console.print()
        table = Table(
            title="[bold]Throughput Results[/]",
            show_header=True,
            header_style="bold",
            border_style="bright_cyan",
            expand=True,
        )
        table.add_column("Test", min_width=20, no_wrap=True)
        table.add_column("pp t/s", justify="right", width=10)
        table.add_column("tg t/s", justify="right", width=10)
        table.add_column("TTFT (ms)", justify="right", width=10)
        table.add_column("Total (ms)", justify="right", width=10)
        table.add_column("Tokens", justify="right", width=12)

        for s in ok_samples:
            conc_label = f"  c{s.concurrency}" if s.concurrency > 1 else ""
            label = f"pp{s.label_pp} tg{s.tg_tokens} @ d{s.label_depth}{conc_label}"
            table.add_row(
                label,
                f"{s.pp_tps:,.0f}",
                f"{s.tg_tps:,.1f}",
                f"{s.ttft_ms:,.0f}",
                f"{s.total_ms:,.0f}",
                f"{s.pp_tokens}+{s.tg_tokens}",
            )

        console.print(table)

    # Post-run hints
    from rich.panel import Panel
    matrix_result = matrix_result_holder[0] if matrix_result_holder else None
    if matrix_result is not None and matrix_result.spec_decoding_detected:
        method_label = f" ({matrix_result.spec_decoding_method})" if matrix_result.spec_decoding_method else ""
        console.print(Panel(
            f"[bold yellow]⚡ Speculative decoding detected{method_label}[/]\n"
            "Standard [cyan]tg t/s[/] under-reports real throughput for spec-decode models.\n"
            "Re-run with [bold cyan]--spec-bench[/] for acceptance rate (α) and effective t/s.",
            border_style="yellow",
        ))
    if ok_samples and ok_samples[0].calibration_confidence == "heuristic":
        console.print(
            "[dim yellow]⚠ Token counts use 4 chars/token heuristic — pp t/s may be "
            "inaccurate for non-English or multilingual models.[/]"
        )

    console.print()
    return completed


# ---------------------------------------------------------------------------
# llama-benchy integration (--perf / --perf-only)
# ---------------------------------------------------------------------------

def _run_llama_benchy(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    *,
    pp: list[int],
    tg: list[int],
    depths: list[int],
    concurrency_levels: list[int],
    runs: int = 3,
    latency_mode: str = "generation",
    skip_coherence: bool = False,
    extra_args: list[str] | None = None,
) -> list:
    """Run llama-benchy externally and display results.

    Returns a list of ThroughputSample objects for report persistence.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.runner.llama_benchy import (
        LlamaBenchyResult,
        is_available,
        run_llama_benchy,
    )

    if not is_available():
        console.print(
            "[bold red]Error:[/] llama-benchy is not available.\n"
            "Install it with: [bold cyan]pip install llama-benchy[/]\n"
            "Or ensure [bold cyan]uvx[/] is on PATH for zero-install usage."
        )
        sys.exit(1)

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]pp={pp}  tg={tg}  depth={depths}  concurrency={concurrency_levels}  "
            f"runs={runs}  latency={latency_mode}[/]",
            title="[bold]⚡ llama-benchy Throughput Benchmark[/]",
            border_style="bright_cyan",
        )
    )
    console.print()

    # Calculate total runs for progress bar
    import re

    total_test_points = len(pp) * len(tg) * len(depths) * len(concurrency_levels)
    total_runs = total_test_points * runs

    benchy_result: LlamaBenchyResult | None = None

    async def run() -> None:
        nonlocal benchy_result

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("Initializing…", total=total_runs)
            current_test = ""
            completed_runs = 0

            def on_output(line: str) -> None:
                nonlocal current_test, completed_runs
                stripped = line.strip()
                if not stripped:
                    return

                # Parse test and run progress from llama-benchy output
                if stripped.startswith("Running test:"):
                    # e.g. "Running test: pp=2048, tg=128, depth=0, concurrency=1"
                    current_test = stripped.replace("Running test: ", "")
                    progress.update(task, description=current_test)
                elif re.match(r"\s*Run \d+/\d+", stripped):
                    # e.g. "  Run 1/3 (batch size 1)..."
                    completed_runs += 1
                    progress.update(task, completed=completed_runs)
                elif "Warming up" in stripped and "complete" not in stripped.lower():
                    progress.update(task, description="Warming up…")
                elif "Measuring latency" in stripped:
                    progress.update(task, description="Measuring latency…")
                elif "Average latency" in stripped:
                    # e.g. "Average latency (generation): 55.78 ms"
                    progress.update(task, description="Running benchmarks…")
                # Other informational lines are silently consumed

            benchy_result = await run_llama_benchy(
                base_url, model,
                api_key=api_key,
                tokenizer=display_name,
                pp=pp, tg=tg,
                depths=depths,
                concurrency_levels=concurrency_levels,
                runs=runs,
                latency_mode=latency_mode,
                skip_coherence=skip_coherence,
                extra_args=extra_args,
                on_output=on_output,
            )

            # Mark complete
            progress.update(task, completed=total_runs, description="[green]✓ Complete")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except RuntimeError as exc:
        console.print(f"\n[bold red]llama-benchy error:[/] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)

    if benchy_result is None:
        console.print("[bold red]No results from llama-benchy.[/]")
        return []

    # Display version info
    if benchy_result.version:
        console.print(f"\n  [dim]llama-benchy {benchy_result.version}[/]")
    if benchy_result.latency_ms > 0:
        console.print(f"  [dim]Estimated latency: {benchy_result.latency_ms:.1f} ms[/]")

    # Summary table
    ok_samples = [s for s in benchy_result.samples if not s.error]
    if ok_samples:
        console.print()

        # Pre-compute labels to size the Test column
        labels: list[str] = []
        for s in ok_samples:
            labels.append(f"pp{s.label_pp} tg{s.tg_tokens} @ d{s.label_depth}")
        test_col_width = max(len(lbl) for lbl in labels)

        table = Table(
            title="[bold]llama-benchy Results[/]",
            show_header=True,
            header_style="bold",
            border_style="bright_cyan",
            expand=True,
        )
        table.add_column("Test", min_width=test_col_width, no_wrap=True)
        table.add_column("c", justify="center", width=4)
        table.add_column("pp t/s", justify="right", width=9)
        table.add_column("tg t/s", justify="right", width=9)
        table.add_column("TTFT (ms)", justify="right", width=10)
        table.add_column("Total (ms)", justify="right", width=10)
        table.add_column("Tokens", justify="right", width=10)

        for lbl, s in zip(labels, ok_samples):
            table.add_row(
                lbl,
                f"c{s.concurrency}",
                f"{s.pp_tps:,.0f}",
                f"{s.tg_tps:,.1f}",
                f"{s.ttft_ms:,.0f}",
                f"{s.total_ms:,.0f}",
                f"{s.pp_tokens}+{s.tg_tokens}",
            )

        console.print(table)

    if ok_samples and ok_samples[0].calibration_confidence == "llama-benchy":
        console.print(
            "\n  [dim]ℹ Metrics sourced from llama-benchy — see "
            "[bold]https://github.com/eugr/llama-benchy[/] for methodology.[/]"
        )

    console.print()
    return ok_samples


# ---------------------------------------------------------------------------
# Speculative decoding / MTP benchmark
# ---------------------------------------------------------------------------

def _run_spec_bench(
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

    completed: list[SpecDecodeSample] = []

    async def on_sample(sample: SpecDecodeSample, idx: int, total: int) -> None:
        completed.append(sample)
        label = f"{sample.prompt_type:>10} @ d{sample.depth}"
        if sample.error:
            console.print(f"  [red]✗[/] {label} — {sample.error}")
        else:
            # Build status line
            parts = [
                f"  [green]✓[/] {label}",
                f"  [bold]{sample.effective_tg_tps:,.1f}[/] eff t/s",
                f"  [dim]{sample.tg_tps:,.1f} stream t/s[/]",
            ]

            if sample.acceptance_rate is not None:
                ar_pct = sample.acceptance_rate * 100
                ar_style = "green" if ar_pct >= 60 else "yellow" if ar_pct >= 40 else "red"
                parts.append(f"  [{ar_style}]α={ar_pct:.1f}%[/{ar_style}]")

            if sample.acceptance_length is not None:
                parts.append(f"  [dim]τ={sample.acceptance_length:.1f}[/]")

            if sample.speedup_ratio is not None:
                sp_style = "green" if sample.speedup_ratio >= 1.2 else "yellow" if sample.speedup_ratio >= 1.0 else "red"
                parts.append(f"  [{sp_style}]{sample.speedup_ratio:.2f}x[/{sp_style}]")

            console.print("".join(parts))

    async def run() -> None:
        await run_spec_bench(
            base_url, model,
            pp=pp, tg=tg,
            depths=depths,
            api_key=api_key,
            spec_method=spec_method,
            baseline_tg_tps=baseline_tg_tps,
            prompt_types=prompt_types,
            on_sample=on_sample,
            metrics_url=metrics_url,
        )

    try:
        asyncio.run(run())
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
        table.add_column("Prompt", no_wrap=True)
        table.add_column("Depth", justify="right", no_wrap=True)
        table.add_column("Eff t/s", justify="right", min_width=7, no_wrap=True)
        table.add_column("α %", justify="right", no_wrap=True)
        table.add_column("τ len", justify="right", no_wrap=True)
        if has_speedup:
            table.add_column("Speed", justify="right", no_wrap=True)
        table.add_column("TTFT", justify="right", no_wrap=True)
        table.add_column("Total ms", justify="right", no_wrap=True)

        def _depth_label(d: int) -> str:
            if d == 0:
                return "0"
            if d >= 1024 and d % 1024 == 0:
                return f"{d // 1024}K"
            return f"{d:,}"

        for s in ok_samples:
            ar_str = f"{s.acceptance_rate * 100:.1f}%" if s.acceptance_rate is not None else "—"
            al_str = f"{s.acceptance_length:.1f}" if s.acceptance_length is not None else "—"
            row: list[str] = [
                s.prompt_type,
                _depth_label(s.depth),
                f"{s.effective_tg_tps:,.1f}",
                ar_str,
                al_str,
            ]
            if has_speedup:
                row.append(f"{s.speedup_ratio:.2f}x" if s.speedup_ratio is not None else "—")
            row.extend([
                f"{s.ttft_ms:,.0f}",
                f"{s.total_ms:,.0f}",
            ])
            table.add_row(*row)

        console.print(table)

        # Show insights
        has_ar = any(s.acceptance_rate is not None for s in ok_samples)
        if has_ar:
            best = max(ok_samples, key=lambda s: s.acceptance_rate or 0)
            worst = min(ok_samples, key=lambda s: s.acceptance_rate if s.acceptance_rate is not None else float('inf'))
            if best.acceptance_rate is not None and worst.acceptance_rate is not None:
                console.print(
                    f"\n  [dim]Highest acceptance:[/] [bold]{best.prompt_type}[/] "
                    f"({best.acceptance_rate * 100:.1f}%)  "
                    f"[dim]Lowest:[/] [bold]{worst.prompt_type}[/] "
                    f"({worst.acceptance_rate * 100:.1f}%)"
                )
        else:
            console.print(
                "\n  [dim]ℹ Acceptance rate: not available (optional).[/]"
            )
            console.print(
                "  [dim]  Effective t/s (shown above) is the primary metric and "
                "already captures MTP/spec-decode speedup.[/]"
            )
            console.print(
                "  [dim]  For acceptance rate breakdown, ensure your server exposes "
                "/metrics with spec_decode counters[/]"
            )
            console.print(
                "  [dim]  (vLLM: enabled by default at http://\u003chost\u003e:\u003cport\u003e/metrics; "
                "llama.cpp: start with --metrics flag).[/]"
            )

    # Write report
    if ok_samples:
        from tool_eval_bench.storage.reports import MarkdownReporter
        from tool_eval_bench.utils.ids import build_run_id

        run_config = {"model": model, "base_url": base_url, "mode": "spec-bench", "method": spec_method}
        run_id = build_run_id(run_config)
        reporter = MarkdownReporter()
        report_path = reporter.write_spec_decode_report(run_id, display_name, ok_samples)
        console.print(f"\n  [dim]📄 Report saved to {report_path}[/]")

    console.print()
    return completed


# ---------------------------------------------------------------------------
# History and diff (extracted to cli/history.py)
# ---------------------------------------------------------------------------

from tool_eval_bench.cli.history import (  # noqa: E402
    compare_runs as _compare_runs,
    print_diff as _print_diff,
    print_history as _print_history,
)

from tool_eval_bench.cli.leaderboard import (  # noqa: E402
    export_runs as _export_runs,
    print_leaderboard as _print_leaderboard,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_int_list(value: str) -> list[int]:
    """Parse a space-or-comma separated list of ints."""
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run tool-eval-bench agentic tool-call benchmark"
    )
    parser.add_argument("--model", default=None, help="Model name/path (auto-detected if omitted)")
    parser.add_argument("--backend", default=None,
                        help="Backend label for reports: vllm, litellm, llamacpp "
                             "(all use the same OpenAI-compatible adapter; default: env/vllm)")
    parser.add_argument("--base-url", default=None, help="Server base URL (default: from .env)")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking/reasoning for models that support it "
                             "(sets enable_thinking=false via chat_template_kwargs)")
    parser.add_argument("--top-p", type=float, default=None, metavar="P",
                        help="Top-p (nucleus) sampling value (e.g. 0.9)")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Top-k sampling value (e.g. 40)")
    parser.add_argument("--min-p", type=float, default=None, metavar="P",
                        help="Min-p sampling threshold (e.g. 0.05)")
    parser.add_argument("--repeat-penalty", type=float, default=None, metavar="V",
                        help="Repetition penalty (e.g. 1.1)")
    parser.add_argument(
        "--backend-kwargs", type=str, default=None, metavar="JSON",
        help="JSON-encoded dict of extra parameters to pass to the backend "
             "(e.g. --backend-kwargs '{\"temperature\": 0.6, \"top_p\": 0.9}'). "
             "These are merged into the API request payload. Overrides any "
             "conflicting individual flags (--temperature, --top-p, etc.).",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per scenario")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Specific scenario IDs to run (e.g. TC-01 TC-07). Default: all.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        metavar="CAT",
        help="Run only scenarios from specific categories (e.g. --categories K A J). "
             "Letters A–O: A=Tool Selection, B=Parameter Precision, C=Multi-Step, "
             "D=Restraint, E=Error Recovery, F=Localization, G=Structured Reasoning, "
             "H=Instruction Following, I=Context & State, J=Code Patterns, "
             "K=Safety, L=Toolset Scale, M=Autonomous Planning, N=Creative Composition, "
             "O=Structured Output.",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of rich display")
    parser.add_argument("--no-live", action="store_true", help="Disable live updating display")
    parser.add_argument(
        "--redact-url", action="store_true",
        help="Mask the server URL in display output (useful for screenshots/recordings). "
             "The actual API connection is unaffected.",
    )
    parser.add_argument("--short", action="store_true", help="Run only the core 15 scenarios (skip extended + agentic)")
    parser.add_argument("--trials", type=int, default=1, help="Number of trial runs for statistical rigor (default: 1)")
    parser.add_argument(
        "--reference-date", default=None,
        help="Override benchmark reference date (YYYY-MM-DD). Default: 2026-03-20",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (passed to server)")
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run N scenarios concurrently (default: 1 = sequential). "
             "Values >1 speed up benchmarks but may increase server load.",
    )
    parser.add_argument(
        "--error-rate", type=float, default=0.0, metavar="RATE",
        help="Inject random tool errors at this rate (0.0–1.0) to test robustness. "
             "Simulates HTTP 429/500/503 errors. Best used with --trials 3+ for Pass@k/Pass^k analysis.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, metavar="WEIGHT",
        help="Quality/speed weight for deployability score (0.0–1.0, default: 0.7). "
             "Higher values weight quality more; lower values penalize slow models more.",
    )

    # Warm-up and throughput
    parser.add_argument("--no-warmup", action="store_true", help="Skip server warm-up request")
    parser.add_argument(
        "--perf", action="store_true",
        help="Run throughput benchmark (via llama-benchy) before tool-call scenarios. "
             "Requires llama-benchy to be installed (pip install llama-benchy) or uvx on PATH.",
    )
    parser.add_argument(
        "--perf-only", action="store_true",
        help="Run ONLY throughput benchmark via llama-benchy (skip tool-call scenarios)",
    )
    parser.add_argument(
        "--perf-legacy", action="store_true",
        help="Run built-in throughput benchmark before tool-call scenarios "
             "(simpler, no external dependencies)",
    )
    parser.add_argument(
        "--perf-legacy-only", action="store_true",
        help="Run ONLY built-in throughput benchmark (skip tool-call scenarios)",
    )
    parser.add_argument("--pp", type=int, default=2048, help="Prompt tokens for throughput benchmark (default: 2048)")
    parser.add_argument("--tg", type=int, default=128, help="Generation tokens for throughput benchmark (default: 128)")
    parser.add_argument("--depth", type=str, default="0,4096,8192", help="Context depths, comma separated (default: '0,4096,8192')")
    parser.add_argument("--concurrency", type=str, default="1,2,4", help="Concurrency levels (default: '1,2,4')")

    # llama-benchy tuning (used by --perf / --perf-only)
    parser.add_argument(
        "--benchy-runs", type=int, default=3,
        help="Number of measurement runs per test for llama-benchy (default: 3)",
    )
    parser.add_argument(
        "--benchy-latency-mode", default="generation",
        choices=["api", "generation", "none"],
        help="Latency measurement mode for llama-benchy (default: generation). "
             "'generation' is most accurate for local servers.",
    )
    parser.add_argument(
        "--benchy-args", type=str, default=None,
        help="Additional arguments to pass through to llama-benchy (quoted string, "
             "e.g. --benchy-args='--no-warmup --book-url URL')",
    )
    parser.add_argument(
        "--skip-coherence", action="store_true",
        help="Skip llama-benchy coherence check (not recommended — use only on "
             "air-gapped hosts that cannot reach gutenberg.org)",
    )

    # Speculative decoding / MTP benchmark
    parser.add_argument(
        "--spec-bench", action="store_true",
        help="Run speculative decoding / MTP benchmark (measures effective t/s, "
             "acceptance rate, and speedup vs baseline)",
    )
    parser.add_argument(
        "--spec-method", default="auto",
        choices=["auto", "mtp", "draft", "ngram", "eagle"],
        help="Speculative decoding method hint (default: auto-detect)",
    )
    parser.add_argument(
        "--baseline-tgs", type=float, default=None, metavar="TPS",
        help="Known baseline tg t/s (without spec decode) for speedup calculation. "
             "If omitted, speedup ratio won't be computed.",
    )
    parser.add_argument(
        "--spec-prompts", type=str, default="filler,code,structured",
        help="Prompt types for spec bench, comma separated (default: 'filler,code,structured'). "
             "Options: filler, code, structured",
    )
    parser.add_argument(
        "--metrics-url", type=str, default=None, metavar="URL",
        help="Direct URL to the Prometheus /metrics endpoint for spec-decode acceptance rate. "
             "Use when the API runs behind a proxy (e.g. LiteLLM) and /metrics lives on a "
             "different host/port (e.g. --metrics-url http://vllm-host:8080/metrics).",
    )

    # Context pressure
    parser.add_argument(
        "--context-pressure", type=float, default=None, metavar="RATIO",
        help="Fill context to RATIO (0.0–1.0) before each scenario to test tool-calling "
             "under context pressure (e.g. --context-pressure 0.75 fills 75%% of available context). "
             "Context window size is auto-detected from /v1/models or set via --context-size.",
    )
    parser.add_argument(
        "--context-size", type=int, default=None, metavar="TOKENS",
        help="Override auto-detected context window size (tokens). "
             "Required if auto-detection fails (e.g. --context-size 32768).",
    )
    parser.add_argument(
        "--context-pressure-sweep", type=str, default=None, metavar="START-END",
        help="Run scenarios at increasing context pressure from START to END "
             "(e.g. --context-pressure-sweep 0.5-1.0). Reports pass/fail at each "
             "level and identifies the breaking point. Use with --sweep-steps to "
             "control granularity.",
    )
    parser.add_argument(
        "--sweep-steps", type=int, default=5, metavar="N",
        help="Number of intervals for --context-pressure-sweep (default: 5, "
             "producing N+1 test levels). E.g. --sweep-steps 10 with range "
             "0.9-1.0 tests 0.90, 0.91, ..., 1.00.",
    )

    # Comparison and history
    parser.add_argument(
        "--diff", metavar="RUN_ID", default=None,
        help="Compare results against a previous run ID (use 'latest' for most recent)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None,
        help="Compare two stored runs by ID (e.g. --compare RUN_ID_OLD RUN_ID_NEW)",
    )
    parser.add_argument("--history", action="store_true", help="List recent benchmark runs and exit")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Show a ranked leaderboard of all benchmarked models and exit")
    parser.add_argument(
        "--export", metavar="FORMAT", default=None,
        choices=["csv", "json"],
        help="Export all stored results in CSV or JSON format and exit",
    )
    parser.add_argument(
        "--export-output", metavar="FILE", default=None,
        help="Output file for --export (default: stdout)",
    )

    # LLM-as-Judge
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="[WIP] Re-evaluate FAIL results using an LLM judge. "
             "Can upgrade FAIL → PARTIAL (never to PASS) to catch false negatives "
             "from string-matching evaluators. Module is implemented but not yet "
             "wired into the benchmark flow.",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None, metavar="MODEL",
        help="Model to use for LLM judging (default: same as benchmark model).",
    )

    # Experimental
    parser.add_argument(
        "--experimental-async", action="store_true",
        help="[WIP] Enable experimental async tool orchestration. "
             "Does not affect standard scenarios — adds async tool awareness.",
    )

    args = parser.parse_args()
    console = Console()

    # --history: show recent runs and exit
    if args.history:
        _print_history(console)
        return

    # --leaderboard: show ranked model comparison and exit
    if args.leaderboard:
        _print_leaderboard(console)
        return

    # --export: dump results in CSV/JSON and exit
    if args.export:
        _export_runs(console, fmt=args.export, output=args.export_output)
        return

    # --compare: diff two stored runs and exit
    if args.compare:
        _compare_runs(console, args.compare[0], args.compare[1])
        return

    # Cascade: CLI flag → env var → fallback
    model = args.model or os.getenv("TOOL_EVAL_MODEL") or None
    backend = args.backend or os.getenv("TOOL_EVAL_BACKEND", "vllm")
    base_url = args.base_url or os.getenv("TOOL_EVAL_BASE_URL", "")
    api_key = args.api_key or os.getenv("TOOL_EVAL_API_KEY")

    # Fallback: construct URL from TOOL_EVAL_HOST + TOOL_EVAL_PORT
    if not base_url:
        host = os.getenv("TOOL_EVAL_HOST", "")
        port = os.getenv("TOOL_EVAL_PORT", "")
        if host:
            base_url = f"http://{host}:{port}" if port else f"http://{host}"

    if not base_url:
        parser.error("--base-url is required (or set TOOL_EVAL_BASE_URL or TOOL_EVAL_HOST+TOOL_EVAL_PORT in .env)")

    # URL redaction for display (actual API calls use real base_url)
    display_url = _redact_url(base_url) if args.redact_url else base_url

    # Auto-detect model if not provided
    display_name: str | None = None
    if not model:
        console.print("\n[bold]🔧 Tool-Call Benchmark[/]")
        console.print(f"[dim]  Server: {display_url}[/]")
        model, display_name = _detect_model(base_url, api_key, console, display_url=display_url)
        console.print()

    # display_name is the human-readable model (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")
    # model is the API alias (e.g. "gemma4") — used in all API calls
    display_name = display_name or model

    # Build extra_params from sampling / thinking flags
    extra_params: dict[str, Any] = {}
    if args.no_think:
        extra_params["chat_template_kwargs"] = {"enable_thinking": False}
    if args.top_p is not None:
        extra_params["top_p"] = args.top_p
    if args.top_k is not None:
        extra_params["top_k"] = args.top_k
    if args.min_p is not None:
        extra_params["min_p"] = args.min_p
    if args.repeat_penalty is not None:
        extra_params["repetition_penalty"] = args.repeat_penalty

    # Merge --backend-kwargs (JSON blob) — wins over individual flags on conflict
    if args.backend_kwargs:
        try:
            bk = json.loads(args.backend_kwargs)
            if not isinstance(bk, dict):
                parser.error("--backend-kwargs must be a JSON object (dict), "
                             f"got {type(bk).__name__}")
            # Deep-merge: for dict-valued keys, merge nested dicts; else override
            for k, v in bk.items():
                if isinstance(v, dict) and isinstance(extra_params.get(k), dict):
                    extra_params[k].update(v)
                else:
                    extra_params[k] = v
        except json.JSONDecodeError as exc:
            parser.error(f"--backend-kwargs is not valid JSON: {exc}")

    # -- Validate --categories --
    if args.categories:
        invalid = {c.upper() for c in args.categories} - _VALID_CATEGORIES
        if invalid:
            parser.error(
                f"Unknown categories: {', '.join(sorted(invalid))}. "
                f"Valid: {', '.join(sorted(_VALID_CATEGORIES))}"
            )
        cats = [c.upper() for c in args.categories]
        from tool_eval_bench.domain.scenarios import CATEGORY_LABELS
        cat_names = ", ".join(
            f"{c} ({CATEGORY_LABELS[Category(c)]})" for c in cats
        )
        resolved_count = len(_resolve_scenarios(args))
        if not args.json:
            console.print(
                f"  [dim]📋 Categories: {cat_names} "
                f"({resolved_count} scenarios)[/]"
            )

    # -- Warm-up --
    if not args.no_warmup:
        _do_warmup(console, base_url, model, api_key)

    # -- Feature flags not yet wired into the run loop --
    if args.llm_judge:
        console.print(
            "\n  [bold yellow]⚠ --llm-judge:[/] The judge module is implemented "
            "(runner/judge.py) but not yet wired into the benchmark flow. "
            "Judge results will not be applied in this run.\n"
        )
    if args.experimental_async:
        console.print(
            "\n  [bold yellow]⚠ --experimental-async:[/] The async tool executor is "
            "implemented (runner/async_tools.py) but not yet integrated with "
            "the scenario orchestrator. This flag has no effect in this run.\n"
        )

    # -- Throughput benchmark (llama-benchy, the default) --
    throughput_samples: list = []
    if args.perf or args.perf_only:
        depths = _parse_int_list(args.depth)
        conc_levels = _parse_int_list(args.concurrency)

        # Parse extra args if provided
        benchy_extra: list[str] | None = None
        if args.benchy_args:
            import shlex
            benchy_extra = shlex.split(args.benchy_args)

        throughput_samples = _run_llama_benchy(
            console, model, display_name, base_url, api_key,
            pp=[args.pp], tg=[args.tg],
            depths=depths,
            concurrency_levels=conc_levels,
            runs=args.benchy_runs,
            latency_mode=args.benchy_latency_mode,
            skip_coherence=args.skip_coherence,
            extra_args=benchy_extra,
        )

        if args.perf_only:
            # Write standalone throughput report
            from tool_eval_bench.storage.reports import MarkdownReporter
            from tool_eval_bench.utils.ids import build_run_id

            run_config = {
                "model": model, "backend": backend,
                "base_url": base_url, "mode": "perf-only",
            }
            run_id = build_run_id(run_config)
            reporter = MarkdownReporter()
            report_path = reporter.write_throughput_report(
                run_id, display_name, throughput_samples,
            )
            console.print(f"\n  [dim]Report saved to {report_path}[/]\n")
            return

    # -- Legacy built-in throughput benchmark --
    if args.perf_legacy or args.perf_legacy_only:
        depths = _parse_int_list(args.depth)
        conc_levels = _parse_int_list(args.concurrency)
        legacy_samples = _run_throughput(
            console, model, display_name, base_url, api_key,
            pp=args.pp, tg=args.tg, depths=depths, concurrency_levels=conc_levels,
        )
        throughput_samples.extend(legacy_samples)

        if args.perf_legacy_only:
            from tool_eval_bench.storage.reports import MarkdownReporter
            from tool_eval_bench.utils.ids import build_run_id

            run_config = {"model": model, "backend": backend, "base_url": base_url, "mode": "perf-legacy-only"}
            run_id = build_run_id(run_config)
            reporter = MarkdownReporter()
            report_path = reporter.write_throughput_report(run_id, display_name, legacy_samples)
            console.print(f"\n  [dim]Report saved to {report_path}[/]\n")
            return

    # -- Speculative decoding / MTP benchmark --
    if args.spec_bench:
        spec_depths = _parse_int_list(args.depth)
        spec_prompts = [p.strip() for p in args.spec_prompts.split(",") if p.strip()]
        _run_spec_bench(
            console, model, display_name, base_url, api_key,
            pp=args.pp, tg=args.tg, depths=spec_depths,
            spec_method=args.spec_method,
            baseline_tg_tps=args.baseline_tgs,
            prompt_types=spec_prompts,
            metrics_url=args.metrics_url,
        )
        # If --spec-bench is the only mode requested (no --perf, no scenarios)
        if not args.perf and not args.perf_only:
            return

    # -- Context pressure sweep --
    if args.context_pressure_sweep is not None:
        _run_pressure_sweep(
            console, model, display_name, backend, base_url, api_key, args,
            display_url=display_url,
            extra_params=extra_params or None,
        )
        return

    # -- Context pressure --
    pressure_messages: list[dict] | None = None
    pressure_config_dict: dict | None = None
    if args.context_pressure is not None:
        from rich.progress import BarColumn, Progress, TextColumn

        from tool_eval_bench.runner.context_pressure import (
            build_pressure_messages,
            prepare_context_pressure,
        )

        ratio = max(0.0, min(1.0, args.context_pressure))
        try:
            pressure_cfg = asyncio.run(
                prepare_context_pressure(
                    base_url, model, api_key,
                    ratio=ratio,
                    context_size_override=args.context_size,
                )
            )

            if not args.json and pressure_cfg.fill_tokens > 0:
                with Progress(
                    TextColumn("  [bold cyan]⚡ Filling context[/]"),
                    BarColumn(bar_width=40),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("[dim]{task.completed:,}/{task.total:,} tokens[/]"),
                    console=console,
                ) as progress:
                    task = progress.add_task("fill", total=pressure_cfg.fill_tokens)
                    pressure_messages = build_pressure_messages(
                        pressure_cfg,
                        on_chunk=lambda tokens_so_far: progress.update(
                            task, completed=tokens_so_far,
                        ),
                    )
            else:
                pressure_messages = build_pressure_messages(pressure_cfg)

            pressure_config_dict = {
                "ratio": pressure_cfg.ratio,
                "fill_tokens": pressure_cfg.fill_tokens,
                "context_size": pressure_cfg.detected_context,
            }
            if not args.json:
                # Compute tool token estimate for selected scenarios
                from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

                selected_sc = _resolve_scenarios(args)

                max_toolset = UNIVERSAL_TOOLS
                for s in selected_sc:
                    if s.tools_override and len(s.tools_override) > len(max_toolset):
                        max_toolset = s.tools_override
                tool_tokens_est = len(json.dumps(max_toolset)) // 4
                num_tools = len(max_toolset)

                from tool_eval_bench.runner.context_pressure import (
                    _RESERVED_FOR_OUTPUT,
                )

                headroom = (
                    pressure_cfg.detected_context
                    - pressure_cfg.fill_tokens
                    - _RESERVED_FOR_OUTPUT
                )
                fill_k = pressure_cfg.fill_tokens / 1000
                tool_k = tool_tokens_est / 1000
                out_k = _RESERVED_FOR_OUTPUT / 1000
                head_k = headroom / 1000

                console.print(
                    f"  [dim]  {pressure_cfg.summary()} — "
                    f"{len(pressure_messages or [])} filler messages[/]"
                )
                console.print(
                    f"  [dim]  Budget: [bold]{fill_k:.0f}K[/] fill │ "
                    f"~{tool_k:.0f}K tools ({num_tools} loaded) │ "
                    f"{out_k:.0f}K output │ "
                    f"{head_k:.0f}K headroom[/]\n"
                )
        except ValueError as exc:
            console.print(f"\n[bold red]Error:[/] {exc}")
            sys.exit(1)

    # -- Tool-call scenarios --
    service = BenchmarkService()
    use_live = not args.json and not args.no_live
    trials = max(1, args.trials)

    if trials > 1 and not args.json:
        console.print(f"[dim]  Running {trials} trials for statistical measurement…[/]\n")

    if use_live:
        _run_with_live_display(
            service, console, model, display_name, backend, base_url, api_key, args,
            throughput_samples=throughput_samples,
            extra_params=extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config_dict,
            display_url=display_url,
        )
    elif args.json:
        _run_json(service, model, backend, base_url, api_key, args,
                  extra_params=extra_params or None,
                  context_pressure_messages=pressure_messages,
                  context_pressure_config=pressure_config_dict)
    else:
        _run_plain(service, console, model, display_name, backend, base_url, api_key, args,
                   throughput_samples=throughput_samples,
                   extra_params=extra_params or None,
                   context_pressure_messages=pressure_messages,
                   context_pressure_config=pressure_config_dict,
                   display_url=display_url)

# ---------------------------------------------------------------------------
# Multi-trial aggregation
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses percentile bootstrap — no scipy dependency. With N=3-5 trials
    we can't assume normality, so bootstrap is more appropriate than
    parametric CI.

    Returns (lower, upper) bounds for the given confidence level.
    """
    import random

    if len(values) <= 1:
        v = values[0] if values else 0.0
        return (v, v)

    # Deterministic bootstrap for reproducibility
    rng = random.Random(42)
    means = sorted(
        mean(rng.choices(values, k=len(values)))
        for _ in range(n_resamples)
    )

    alpha = 1 - ci
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return (round(means[lo_idx], 1), round(means[hi_idx], 1))


def _median(values: list[float]) -> float:
    """Median without importing statistics.median (already have mean, stdev)."""
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _aggregate_trials(
    summaries: list,  # list[ModelScoreSummary]
) -> dict:
    """Compute mean ± stddev, median, and 95% bootstrap CI across N trials.

    Returns a dict with aggregated statistics suitable for display and JSON.
    """
    n = len(summaries)
    if n <= 1:
        return {}

    final_scores = [s.final_score for s in summaries]
    total_points_list = [s.total_points for s in summaries]

    # Bootstrap CI for final score
    ci_lo, ci_hi = _bootstrap_ci([float(x) for x in final_scores])

    # Per-scenario aggregation
    scenario_ids = [r.scenario_id for r in summaries[0].scenario_results]
    scenario_stats: dict[str, dict] = {}
    pass_at_k_count = 0  # scenarios that passed at least once
    pass_hat_k_count = 0  # scenarios that passed every trial
    for sid in scenario_ids:
        points = []
        for s in summaries:
            r = next((r for r in s.scenario_results if r.scenario_id == sid), None)
            if r:
                points.append(r.points)
        passed_at_least_once = any(p == 2 for p in points)
        passed_every_time = all(p == 2 for p in points)
        if passed_at_least_once:
            pass_at_k_count += 1
        if passed_every_time:
            pass_hat_k_count += 1
        scenario_stats[sid] = {
            "mean": round(mean(points), 2),
            "stddev": round(stdev(points), 2) if len(points) > 1 else 0.0,
            "points": points,
            "pass_at_k": passed_at_least_once,
            "pass_hat_k": passed_every_time,
        }

    total_scenarios = len(scenario_ids)

    # Per-category aggregation
    cat_stats: dict[str, dict] = {}
    for cs in summaries[0].category_scores:
        cat_key = cs.category.value
        percents = []
        for s in summaries:
            cat_s = next((c for c in s.category_scores if c.category == cs.category), None)
            if cat_s:
                percents.append(cat_s.percent)
        cat_stats[cat_key] = {
            "label": cs.label,
            "mean_percent": round(mean(percents), 1),
            "stddev_percent": round(stdev(percents), 1) if len(percents) > 1 else 0.0,
        }

    # Pass@k / Pass^k rates (Claw-Eval methodology)
    pass_at_k_rate = round(100 * pass_at_k_count / total_scenarios, 1) if total_scenarios else 0.0
    pass_hat_k_rate = round(100 * pass_hat_k_count / total_scenarios, 1) if total_scenarios else 0.0

    return {
        "trials": n,
        "final_score_mean": round(mean(final_scores), 1),
        "final_score_stddev": round(stdev(final_scores), 1) if n > 1 else 0.0,
        "final_score_median": round(_median([float(x) for x in final_scores]), 1),
        "final_score_ci95": (ci_lo, ci_hi),
        "total_points_mean": round(mean(total_points_list), 1),
        "total_points_stddev": round(stdev(total_points_list), 1) if n > 1 else 0.0,
        "pass_at_k": pass_at_k_rate,
        "pass_hat_k": pass_hat_k_rate,
        "reliability_gap": round(pass_at_k_rate - pass_hat_k_rate, 1),
        "per_scenario": scenario_stats,
        "per_category": cat_stats,
    }


def _print_trials_summary(console: Console, agg: dict) -> None:
    """Print aggregated trial statistics."""
    if not agg:
        return

    from rich.panel import Panel

    n = agg["trials"]
    score_mean = agg["final_score_mean"]
    score_std = agg["final_score_stddev"]
    ci_lo, ci_hi = agg["final_score_ci95"]
    median = agg["final_score_median"]

    content = (
        f"  [bold]Trials:[/]  {n}\n"
        f"  [bold]Score:[/]   {score_mean:.1f} ± {score_std:.1f} / 100\n"
        f"  [bold]Median:[/]  {median:.1f}\n"
        f"  [bold]95% CI:[/]  [{ci_lo:.1f}, {ci_hi:.1f}]\n"
        f"  [bold]Points:[/]  {agg['total_points_mean']:.1f} ± {agg['total_points_stddev']:.1f}\n"
    )

    # Pass@k / Pass^k reliability metrics
    if "pass_at_k" in agg:
        pass_at = agg["pass_at_k"]
        pass_hat = agg["pass_hat_k"]
        gap = agg["reliability_gap"]
        content += (
            f"\n  [bold]Pass@{n}:[/]  {pass_at:.1f}%  [dim](capability ceiling)[/]\n"
            f"  [bold]Pass^{n}:[/]  {pass_hat:.1f}%  [dim](reliability floor)[/]\n"
        )
        if gap > 5:
            content += f"  [bold yellow]⚠ Gap:[/]    {gap:.1f}pp  [dim](high variance — consistency issue)[/]\n"
        elif gap > 0:
            content += f"  [bold]Gap:[/]     {gap:.1f}pp\n"

    # Show categories with variance
    cat_lines = []
    for cat_key, cs in agg["per_category"].items():
        if cs["stddev_percent"] > 0:
            cat_lines.append(f"    {cat_key} {cs['label']}: {cs['mean_percent']:.0f}% ± {cs['stddev_percent']:.1f}%")
    if cat_lines:
        content += "\n  [bold]Categories with variance:[/]\n" + "\n".join(cat_lines)

    # Show scenarios with variance
    unstable = [
        (sid, st) for sid, st in agg["per_scenario"].items()
        if st["stddev"] > 0
    ]
    if unstable:
        content += f"\n\n  [bold yellow]⚡ {len(unstable)} unstable scenario(s):[/]"
        for sid, st in unstable:
            pts_str = ",".join(str(p) for p in st["points"])
            content += f"\n    {sid}: {st['mean']:.1f} ± {st['stddev']:.1f}  [dim]({pts_str})[/]"

    console.print(Panel(content, title="[bold]📊 Trial Statistics[/]", border_style="bright_cyan", padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Context pressure sweep
# ---------------------------------------------------------------------------

def _parse_sweep_range(sweep_str: str) -> tuple[float, float]:
    """Parse 'START-END' into (start, end) floats, each clamped to [0, 1]."""
    parts = sweep_str.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. Expected format: START-END "
            f"(e.g. 0.5-1.0)"
        )
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(
            f"Invalid sweep range '{sweep_str}'. START and END must be numbers "
            f"(e.g. 0.5-1.0)"
        )
    start = max(0.0, min(1.0, start))
    end = max(0.0, min(1.0, end))
    if start >= end:
        raise ValueError(
            f"Sweep START ({start}) must be less than END ({end})"
        )
    return start, end


def _run_pressure_sweep(
    console: Console,
    model: str,
    display_name: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    display_url: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """Run scenarios at increasing context pressure and report breaking point."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
    from tool_eval_bench.runner.context_pressure import (
        ContextPressureConfig,
        build_pressure_messages,
        compute_fill_budget,
        detect_context_size,
    )
    from tool_eval_bench.runner.orchestrator import run_all_scenarios

    # Parse range
    try:
        start, end = _parse_sweep_range(args.context_pressure_sweep)
    except ValueError as exc:
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    steps = max(1, args.sweep_steps)
    levels = [start + i * (end - start) / steps for i in range(steps + 1)]
    # Ensure clean floating-point values
    levels = [round(lv, 4) for lv in levels]

    scenarios = _resolve_scenarios(args)
    if not scenarios:
        console.print("[bold red]Error:[/] No scenarios matched.")
        sys.exit(1)

    scenario_ids = [s.id for s in scenarios]

    console.print(f"\n[bold]⚡ Context Pressure Sweep[/] — {display_name}")
    console.print(
        f"[dim]  Backend: {backend}  |  Server: {display_url or base_url}[/]"
    )
    console.print(
        f"[dim]  Range: {start:.0%} → {end:.0%}  |  "
        f"{len(levels)} levels  |  "
        f"{len(scenarios)} scenario{'s' if len(scenarios) != 1 else ''}[/]\n"
    )

    # Detect context size once
    try:
        context_size: int | None = args.context_size
        if context_size is None:
            context_size = asyncio.run(
                detect_context_size(base_url, model, api_key)
            )
        if context_size is None:
            console.print(
                "[bold red]Error:[/] Could not auto-detect context size. "
                "Use --context-size to specify it."
            )
            sys.exit(1)
        console.print(f"  [dim]Context window: {context_size:,} tokens[/]\n")
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    # Status emojis
    _STATUS_EMOJI = {
        "pass": "✅",
        "partial": "⚠️ ",
        "fail": "❌",
    }

    # Run sweep
    adapter = OpenAICompatibleAdapter()
    level_results: list[dict[str, Any]] = []
    consecutive_all_fail = 0

    try:
        for level_idx, ratio in enumerate(levels):
            # Compute fill budget for this ratio
            fill_tokens = compute_fill_budget(context_size, ratio)
            cfg = ContextPressureConfig(
                ratio=ratio, fill_tokens=fill_tokens,
                detected_context=context_size,
            )

            # Build fresh pressure messages (unique per level)
            pressure_messages = build_pressure_messages(cfg)

            # Progress line
            pct_done = (level_idx + 1) / len(levels)
            bar_filled = int(pct_done * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            console.print(
                f"  [bold cyan]⚡[/] Sweep {level_idx + 1}/{len(levels)}: "
                f"[bold]{ratio:.0%}[/] pressure  {bar}  ",
                end="",
            )

            # Run scenarios at this pressure level
            try:
                summary = asyncio.run(
                    run_all_scenarios(
                        adapter,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        scenarios=scenarios,
                        temperature=0.0,
                        extra_params=extra_params,
                        context_pressure_messages=pressure_messages,
                    )
                )

                # Collect per-scenario results
                results_map: dict[str, str] = {}
                for sr in summary.scenario_results:
                    results_map[sr.scenario_id] = sr.status.value

                pass_count = sum(
                    1 for s in results_map.values() if s == "pass"
                )
                total = len(scenarios)
                score_pct = (pass_count / total * 100) if total else 0

                # Print inline status
                emoji_str = "  ".join(
                    _STATUS_EMOJI.get(results_map.get(sid, "fail"), "❌")
                    for sid in scenario_ids
                )
                console.print(f"{emoji_str}  [bold]{score_pct:.0f}%[/]")

                level_results.append({
                    "ratio": ratio,
                    "results": results_map,
                    "score_pct": score_pct,
                    "pass_count": pass_count,
                    "fill_tokens": fill_tokens,
                })

                # Early stop logic
                if pass_count == 0:
                    consecutive_all_fail += 1
                else:
                    consecutive_all_fail = 0

                if consecutive_all_fail >= 2:
                    console.print(
                        "  [dim]··· stopped (2 consecutive all-fail levels)[/]"
                    )
                    break

            except Exception as exc:
                console.print(f"[red]error: {exc}[/]")
                level_results.append({
                    "ratio": ratio,
                    "results": {sid: "fail" for sid in scenario_ids},
                    "score_pct": 0,
                    "pass_count": 0,
                    "fill_tokens": fill_tokens,
                    "error": str(exc),
                })
                consecutive_all_fail += 1
                if consecutive_all_fail >= 2:
                    console.print(
                        "  [dim]··· stopped (2 consecutive all-fail levels)[/]"
                    )
                    break

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
    finally:
        if hasattr(adapter, "aclose"):
            try:
                asyncio.run(adapter.aclose())
            except Exception:
                pass

    if not level_results:
        console.print("\n[bold red]No results collected.[/]")
        return

    # Summary panel
    console.print()

    # Build the visual summary
    lines: list[str] = []
    breaking_point: float | None = None
    first_degradation: float | None = None

    for lr in level_results:
        ratio = lr["ratio"]
        score = lr["score_pct"]
        emoji_str = "  ".join(
            _STATUS_EMOJI.get(lr["results"].get(sid, "fail"), "❌")
            for sid in scenario_ids
        )
        # Bar visualization (20 chars wide)
        bar_len = int(score / 100 * 20)
        if score >= 100:
            bar_color = "green"
        elif score >= 50:
            bar_color = "yellow"
        else:
            bar_color = "red"
        bar = f"[{bar_color}]{'█' * bar_len}[/]{'░' * (20 - bar_len)}"

        lines.append(
            f"  [bold]{ratio:5.0%}[/]  {emoji_str}  {score:4.0f}%  {bar}"
        )

        # Track breaking point (last level where all pass)
        all_pass = all(
            v == "pass" for v in lr["results"].values()
        )
        if all_pass:
            breaking_point = ratio
        if first_degradation is None and not all_pass:
            first_degradation = ratio

    lines.append("")
    if breaking_point is not None:
        lines.append(
            f"  [bold green]Breaking point:[/] {breaking_point:.0%} "
            f"(all scenarios pass)"
        )
    else:
        lines.append(
            "  [bold red]Breaking point:[/] none "
            "(no level had all scenarios pass)"
        )
    if first_degradation is not None:
        lines.append(
            f"  [bold yellow]Degradation:[/]    {first_degradation:.0%} "
            f"(first partial/fail)"
        )

    header = "  ".join(f"[dim]{sid}[/]" for sid in scenario_ids)
    lines.insert(0, f"  [dim]       {header}[/]")

    panel_content = "\n".join(lines)
    console.print(Panel(
        panel_content,
        title="[bold]⚡ Context Pressure Sweep Results[/]",
        border_style="bright_cyan",
        padding=(1, 1),
    ))
    console.print()

def _run_with_live_display(
    service: BenchmarkService,
    console: Console,
    model: str,
    display_name: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    throughput_samples: list | None = None,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[dict] | None = None,
    context_pressure_config: dict | None = None,
    display_url: str | None = None,
) -> None:
    """Run with Rich live display — the default visual mode."""
    from tool_eval_bench.runner.orchestrator import score_results

    scenarios = _resolve_scenarios(args)

    trials = max(1, args.trials)
    all_summaries = []

    # --- Trial 1: with live display ---
    display = BenchmarkDisplay(display_name, backend, display_url or base_url, scenarios)
    display.start()

    async def run_trial(*, show: bool = False) -> dict:
        callbacks: dict = {}
        if show:
            callbacks["on_scenario_start"] = display.on_scenario_start
            callbacks["on_scenario_result"] = display.on_scenario_result
        return await service.run_benchmark(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            scenarios=scenarios,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            max_turns=args.max_turns,
            reference_date=args.reference_date,
            seed=args.seed,
            throughput_samples=throughput_samples or [],
            concurrency=args.parallel,
            error_rate=args.error_rate,
            alpha=args.alpha,
            extra_params=extra_params,
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
            **callbacks,
        )

    async def run_all_trials() -> None:
        """Run all trials in a single event loop for connection reuse."""
        result = await run_trial(show=True)

        all_results = [
            display.results[s.id]
            for s in scenarios
            if s.id in display.results
        ]
        if all_results:
            summary = score_results(all_results, scenarios, alpha=args.alpha)
            all_summaries.append(summary)
            display.set_finished(summary, throughput_samples=throughput_samples)

            # --diff: compare against previous run
            if args.diff:
                _print_diff(console, all_results, args.diff)
        else:
            display.stop()

        # Print report path
        report_path = result.get("report_path")
        report_paths: list[str] = []
        if report_path:
            console.print(f"\n  [dim]📄 Full report: {report_path}[/]\n")
            report_paths.append(str(report_path))

        # --- Trials 2..N: silent runs (same event loop) ---
        if trials > 1:
            for t in range(2, trials + 1):
                console.print(f"  [dim]Running trial {t}/{trials}\u2026[/]", end=" ")
                trial_result = await run_trial(show=False)
                trial_scores = trial_result.get("scores", {})
                trial_score_results = trial_scores.get("scenario_results", [])

                # Collect report path
                trial_rp = trial_result.get("report_path")
                if trial_rp:
                    report_paths.append(str(trial_rp))

                # Reconstruct ScenarioResult objects from the persisted dict
                trial_sr = []
                for sr_dict in trial_score_results:
                    trial_sr.append(ScenarioResult(
                        scenario_id=sr_dict["scenario_id"],
                        status=ScenarioStatus(sr_dict["status"]),
                        points=sr_dict["points"],
                        summary=sr_dict.get("summary", ""),
                        duration_seconds=sr_dict.get("duration_seconds", 0.0),
                    ))
                if trial_sr:
                    trial_summary = score_results(trial_sr, scenarios, alpha=args.alpha)
                    all_summaries.append(trial_summary)
                    console.print(f"[bold]{trial_summary.final_score}[/]/100")

            agg = _aggregate_trials(all_summaries)
            _print_trials_summary(console, agg)

            # Write consolidated summary report
            if agg and len(all_summaries) > 1:
                from tool_eval_bench.storage.reports import MarkdownReporter
                reporter = MarkdownReporter()
                run_id_base = result.get("run_id", "summary")
                throughput = result.get("throughput_samples")
                summary_path = reporter.write_summary_report(
                    run_id=run_id_base,
                    model=display_name,
                    summaries=all_summaries,
                    agg=agg,
                    throughput_samples=throughput,
                    report_paths=report_paths,
                )
                console.print(f"  [dim]📊 Summary report: {summary_path}[/]\n")

    try:
        asyncio.run(run_all_trials())
    except KeyboardInterrupt:
        display.stop()
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        display.stop()
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)


def _run_json(
    service: BenchmarkService,
    model: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[dict] | None = None,
    context_pressure_config: dict | None = None,
) -> None:
    """Run and output raw JSON."""
    trials = max(1, args.trials)
    resolved = _resolve_scenarios(args)

    async def run() -> dict:
        return await service.run_benchmark(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            scenarios=resolved,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            max_turns=args.max_turns,
            reference_date=args.reference_date,
            seed=args.seed,
            concurrency=args.parallel,
            error_rate=args.error_rate,
            alpha=args.alpha,
            extra_params=extra_params,
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
        )

    try:
        results = []
        for _t in range(trials):
            results.append(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    if trials == 1:
        print(json.dumps(results[0], indent=2, default=str))
    else:
        # Aggregate trial data
        from tool_eval_bench.runner.orchestrator import score_results

        resolved_sc = _resolve_scenarios(args)
        summaries = []
        for r in results:
            sr_dicts = r.get("scores", {}).get("scenario_results", [])
            trial_sr = [
                ScenarioResult(
                    scenario_id=d["scenario_id"],
                    status=ScenarioStatus(d["status"]),
                    points=d["points"],
                    summary=d.get("summary", ""),
                )
                for d in sr_dicts
            ]
            if trial_sr:
                summaries.append(score_results(trial_sr, resolved_sc, alpha=args.alpha))

        agg = _aggregate_trials(summaries) if summaries else {}
        output = results[-1]  # last run as the primary result
        if agg:
            output["trial_statistics"] = agg
        print(json.dumps(output, indent=2, default=str))


def _run_plain(
    service: BenchmarkService,
    console: Console,
    model: str,
    display_name: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    throughput_samples: list | None = None,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[dict] | None = None,
    context_pressure_config: dict | None = None,
    display_url: str | None = None,
) -> None:
    """Run with simple line-by-line output."""
    console.print(f"\n[bold]Tool-Call Benchmark[/] — {display_name}")
    console.print(f"[dim]  Backend: {backend}  |  Server: {display_url or base_url}[/]\n")

    resolved = _resolve_scenarios(args)

    trials = max(1, args.trials)
    started = time.time()

    async def run(*, show: bool = False) -> dict:
        callbacks: dict = {}
        if show:
            callbacks["on_scenario_start"] = _plain_on_start
            callbacks["on_scenario_result"] = _plain_on_result
        return await service.run_benchmark(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            scenarios=resolved,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            max_turns=args.max_turns,
            reference_date=args.reference_date,
            seed=args.seed,
            throughput_samples=throughput_samples or [],
            concurrency=args.parallel,
            error_rate=args.error_rate,
            alpha=args.alpha,
            extra_params=extra_params,
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
            **callbacks,
        )

    try:
        all_results_dicts = []
        for t in range(1, trials + 1):
            if t > 1:
                console.print(f"\n[dim]  --- Trial {t}/{trials} ---[/]\n")
            all_results_dicts.append(asyncio.run(run(show=True)))
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Error: {exc}[/]")
        sys.exit(1)

    elapsed = time.time() - started
    scores = all_results_dicts[-1].get("scores", {})
    console.print(f"\n[bold]Score: {scores.get('final_score', 0)} / 100  — {scores.get('rating', '')}[/]")
    console.print(f"[dim]Completed in {elapsed:.1f}s[/]\n")

    # Show trial statistics if multiple trials
    if trials > 1:
        from tool_eval_bench.runner.orchestrator import score_results
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, SCENARIOS

        base = SCENARIOS if args.short else ALL_SCENARIOS
        summaries = []
        for r in all_results_dicts:
            sr_dicts = r.get("scores", {}).get("scenario_results", [])
            trial_sr = [
                ScenarioResult(
                    scenario_id=d["scenario_id"],
                    status=ScenarioStatus(d["status"]),
                    points=d["points"],
                    summary=d.get("summary", ""),
                )
                for d in sr_dicts
            ]
            if trial_sr:
                summaries.append(score_results(trial_sr, base, alpha=args.alpha))
        agg = _aggregate_trials(summaries) if summaries else {}
        _print_trials_summary(console, agg)

        if agg and len(summaries) > 1:
            from tool_eval_bench.storage.reports import MarkdownReporter
            reporter = MarkdownReporter()
            run_id_base = all_results_dicts[0].get("run_id", "summary") if all_results_dicts else "summary"
            rp_list = [str(r.get("report_path", "")) for r in all_results_dicts if r.get("report_path")]
            summary_path = reporter.write_summary_report(
                run_id=run_id_base,
                model=display_name,
                summaries=summaries,
                agg=agg,
                report_paths=rp_list,
            )
            console.print(f"  [dim]📊 Summary report: {summary_path}[/]\n")


if __name__ == "__main__":
    main()

