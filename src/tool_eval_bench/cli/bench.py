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
import logging
import os
import sys
import time
from statistics import mean, stdev
from typing import Any

from dotenv import load_dotenv  # noqa: F401  (re-exported via _load_dotenv)
from rich.console import Console

from tool_eval_bench.cli.commands import (
    resolve_all_scenarios_for_ids as _resolve_all_scenarios_for_ids,
)
from tool_eval_bench.cli.commands import (
    resolve_scenarios as _resolve_scenarios,
)
from tool_eval_bench.cli.display import BenchmarkDisplay
from tool_eval_bench.cli.helpers import (
    emit_headless_error as _headless_error,
)
from tool_eval_bench.cli.helpers import (
    load_dotenv_file as _load_dotenv,
)
from tool_eval_bench.cli.helpers import (
    metadata_for_storage as _metadata_for_storage,
)
from tool_eval_bench.cli.helpers import (
    parse_int_list as _parse_int_list,
)
from tool_eval_bench.cli.helpers import (
    parse_sweep_range as _parse_sweep_range,
)
from tool_eval_bench.cli.helpers import (
    persist_plugin_run as _persist_plugin_run,
)
from tool_eval_bench.cli.helpers import (
    redact_url as _redact_url,
)
from tool_eval_bench.cli.helpers import (
    with_config_fingerprint as _with_config_fingerprint,
)
from tool_eval_bench.cli.history import (
    compare_runs as _compare_runs,
)
from tool_eval_bench.cli.history import (
    print_diff as _print_diff,
)
from tool_eval_bench.cli.history import (
    print_history as _print_history,
)
from tool_eval_bench.cli.leaderboard import (
    export_runs as _export_runs,
)
from tool_eval_bench.cli.leaderboard import (
    print_leaderboard as _print_leaderboard,
)
from tool_eval_bench.cli.perf import (
    run_llama_benchy as _run_llama_benchy,
)
from tool_eval_bench.cli.perf import (
    run_throughput as _run_throughput,
)
from tool_eval_bench.cli.pressure import (
    run_pressure_sweep as _run_pressure_sweep,
)
from tool_eval_bench.cli.server import (
    DISCOVERY_PORTS as _DISCOVERY_PORTS,
)
from tool_eval_bench.cli.server import (
    discover_server as _discover_server,
)
from tool_eval_bench.cli.spec_bench import (
    run_spec_bench as _run_spec_bench,
)
from tool_eval_bench.domain.errors import (
    CONNECTION_FAILED,
    DETECTION_FAILED,
    HTTP_ERROR,
    INVALID_RESPONSE,
    MODEL_NOT_AVAILABLE,
    NO_MODELS,
    NO_SERVER,
)
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.runner.service import BenchmarkService
from tool_eval_bench.storage.reports import MarkdownReporter

logger = logging.getLogger(__name__)

# Valid category letters for --categories
_VALID_CATEGORIES = {c.value for c in Category}


# ---------------------------------------------------------------------------
# Model auto-detection
# ---------------------------------------------------------------------------


def _detect_model(
    base_url: str,
    api_key: str | None,
    console: Console,
    *,
    display_url: str | None = None,
    headless: bool = False,
) -> tuple[str, str]:
    """Query /v1/models and auto-select or let the user pick.

    Returns (api_id, display_name).
      - api_id:       what to send in API requests (e.g. "gemma4")
      - display_name: the real model path if available (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")

    When *headless* is True (e.g. ``--json`` mode), the interactive picker is
    skipped: the first available model is auto-selected and a JSONL event is
    emitted on stderr.  Connection errors produce structured JSON on stderr
    and use differentiated exit codes (2 = connection, 3 = no models).
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
    if not headless:
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
        if headless:
            _headless_error(
                CONNECTION_FAILED,
                f"Could not connect to {show_url}. Is the server running?",
                exit_code=2,
            )
        console.print("[bold red]✗ cannot connect[/]")
        console.print(f"\n[red]Could not connect to {show_url}. Is the server running?[/]")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        if headless:
            _headless_error(
                HTTP_ERROR,
                f"Server returned {exc.response.status_code}. Check the URL and API key.",
                exit_code=2,
            )
        console.print(f"[bold red]✗ HTTP {exc.response.status_code}[/]")
        console.print(
            f"\n[red]Server returned {exc.response.status_code}. Check the URL and API key.[/]"
        )
        sys.exit(1)
    except Exception as exc:
        if headless:
            _headless_error(DETECTION_FAILED, str(exc), exit_code=2)
        console.print(f"[bold red]✗ {exc}[/]")
        sys.exit(1)

    if used_fallback:
        if not headless:
            console.print(
                "\n  [yellow]⚠ /v1/models returned 404, used /models fallback. "
                "Check your server configuration.[/]"
            )

    try:
        data = resp.json()
        model_list = data.get("data", [])
    except Exception:
        status_code = resp.status_code
        content_type = resp.headers.get("Content-Type", "unknown")
        body_snippet = resp.text[:200]
        err_msg = (
            f"Server returned invalid JSON from /v1/models (HTTP {status_code}, "
            f"Content-Type: {content_type}). Body snippet: {body_snippet!r}"
        )
        if headless:
            _headless_error(INVALID_RESPONSE, err_msg, exit_code=2)
        console.print("[bold red]✗ invalid response[/]")
        console.print(f"[red]{err_msg}[/]")
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
        if headless:
            _headless_error(NO_MODELS, "The server returned an empty model list.", exit_code=3)
        console.print("[bold red]✗ no models found[/]")
        console.print("[red]The server returned an empty model list.[/]")
        sys.exit(1)

    if len(models) == 1:
        api_id, display = models[0]
        if not headless:
            if display != api_id:
                console.print(f"[bold green]✓[/] [bold]{display}[/] [dim](alias: {api_id})[/]")
            else:
                console.print(f"[bold green]✓[/] [bold]{api_id}[/]")
        return api_id, display

    # Multiple models — in headless mode, auto-select the first one
    if headless:
        api_id, display = models[0]
        msg = {
            "event": "model_auto_selected",
            "model": api_id,
            "display_name": display,
            "total_available": len(models),
            "available_models": [m[0] for m in models],
        }
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
        return api_id, display

    # Multiple models — interactive: let the user choose
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


def _probe_server(
    console: Console,
    base_url: str,
    api_key: str | None,
    *,
    headless: bool = False,
) -> None:
    """Check if a server is reachable and responsive, then exit.

    Useful for CI/CD pipelines and sparkrun recipes where the benchmark
    step runs right after server startup — this lets the orchestrator
    wait until the server is ready.

    Exits 0 if the server responds to /v1/models, exit 1 otherwise.
    """
    import httpx

    from tool_eval_bench.utils.urls import models_url

    endpoint = models_url(base_url)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _check() -> httpx.Response:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(endpoint, headers=headers)
            resp.raise_for_status()
            return resp

    try:
        resp = asyncio.run(_check())
        data = resp.json()
        model_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception as exc:
        if headless:
            msg = {
                "event": "probe_result",
                "status": "failed",
                "base_url": base_url,
                "error": str(exc) or type(exc).__name__,
            }
            sys.stderr.write(json.dumps(msg) + "\n")
            sys.stderr.flush()
        else:
            console.print(f"[bold red]✗[/] Server at {base_url} is not ready: {exc}")
        sys.exit(1)

    if headless:
        msg = {
            "event": "probe_result",
            "status": "ready",
            "base_url": base_url,
            "models": model_ids,
        }
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
    else:
        console.print(f"[bold green]✓[/] Server at {base_url} is ready")
        if model_ids:
            console.print(f"  Models: {', '.join(model_ids)}")
    sys.exit(0)


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
    print(
        f"  {DIM}[{idx + 1}/{total}]{RESET} {scenario.id} {scenario.title}... ", end="", flush=True
    )


async def _plain_on_result(
    scenario: ScenarioDefinition, result: ScenarioResult, idx: int, total: int
) -> None:
    style = STATUS_STYLE.get(result.status, "?")
    print(f"{style}  ({result.points}/2) {DIM}{result.summary}{RESET}")


# ---------------------------------------------------------------------------
# Pre-flight model availability check (issue #19)
# ---------------------------------------------------------------------------


def _preflight_model_check(
    console: Console,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    headless: bool = False,
) -> None:
    """Send a trivial chat completion to verify the model is actually usable.

    Some servers (vLLM, LiteLLM) list models in ``/v1/models`` even when they
    fail to load — returning HTTP 400 "Model not found" on the first real
    request.  Without this check, the benchmark silently produces misleading
    pass/partial/fail scores because 4xx responses are treated as "model
    returned no tool calls" by the adapter.

    This function sends a minimal 1-token completion request.  If the server
    returns 4xx/5xx, we abort with a clear error before any scenarios run.

    Exits with code 3 (model error) on failure.
    """
    import httpx

    from tool_eval_bench.utils.urls import chat_completions_url as _chat_url

    url = _chat_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _check() -> httpx.Response:
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(url, json=payload, headers=headers)

    try:
        resp = asyncio.run(_check())
        if resp.status_code >= 400:
            body = resp.text[:300].strip()
            if headless:
                _headless_error(
                    MODEL_NOT_AVAILABLE,
                    f"Model '{model}' is listed in /v1/models but returned "
                    f"HTTP {resp.status_code} on a test request: {body}",
                    exit_code=3,
                )
            console.print("[bold red]✗ Model not available[/]")
            console.print(
                f"[red]Model '{model}' is listed in /v1/models but returned "
                f"HTTP {resp.status_code} on a test request.[/]"
            )
            console.print(f"[dim]  {body}[/]")
            console.print(
                "\n[yellow]The server lists this model but cannot serve it. "
                "Check server logs for model loading errors.[/]"
            )
            sys.exit(3)
    except httpx.ConnectError:
        if headless:
            _headless_error(
                CONNECTION_FAILED,
                f"Could not connect to {base_url} for pre-flight check.",
                exit_code=2,
            )
        console.print(f"[bold red]✗ Cannot connect to {base_url}[/]")
        sys.exit(2)
    except Exception as exc:
        if headless:
            _headless_error(
                MODEL_NOT_AVAILABLE,
                f"Pre-flight check failed with unexpected error: {exc}",
                exit_code=3,
            )
        console.print(f"[bold red]✗ Pre-flight check failed:[/] {exc}")
        sys.exit(3)


# ---------------------------------------------------------------------------
# Server warm-up
# ---------------------------------------------------------------------------


def _do_warmup(console: Console, base_url: str, model: str, api_key: str | None) -> None:
    """Send a trivial request to prime the server before benchmarking.

    With speculative decoding (dflash, EAGLE, etc.), the first request triggers
    JIT compilation and CUDA graph capture which can take 30-60+ seconds.
    This is a one-time server-side cost — subsequent requests are fast.
    """
    from tool_eval_bench.runner.throughput import warmup

    with console.status(
        "[dim]  Warming up server… (first request may be slow with speculative decoding)[/]",
        spinner="dots",
    ):
        try:
            ms = asyncio.run(warmup(base_url, model, api_key, timeout=120.0))
            if ms > 10_000:  # >10s indicates JIT/CUDA graph compilation
                console.print(
                    f"  [bold green]✓[/] Warm-up complete [dim]({ms:.0f} ms — "
                    f"JIT/CUDA graph compilation on first request)[/]"
                )
            else:
                console.print(f"  [bold green]✓[/] Warm-up complete [dim]({ms:.0f} ms)[/]")
        except Exception as exc:
            # httpx timeout exceptions can have empty str(), so fall back
            # to the exception class name for a useful diagnostic.
            err_msg = str(exc) or type(exc).__name__
            console.print(f"  [bold yellow]⚠[/] Warm-up failed [dim]({err_msg})[/]")


# ---------------------------------------------------------------------------
# History and diff (extracted to cli/history.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GSM8K benchmark (--gsm8k / --gsm8k-only)
# ---------------------------------------------------------------------------


def _run_gsm8k_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run the GSM8K grade-school math benchmark and display results."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
    from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin

    n_shots = args.gsm8k_shots
    limit = args.gsm8k_limit
    shuffle = args.gsm8k_shuffle
    seed = getattr(args, "seed", None)
    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""
    limit_label = "all 1319" if limit == 0 else f"{limit}"

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]{n_shots}-shot CoT · {limit_label} questions"
            f"{' · shuffled' if shuffle else ''}{parallel_label}[/]",
            title="[bold]📐 GSM8K — Grade School Math[/]",
            border_style="bright_magenta",
        )
    )

    plugin = GSM8KPlugin()
    adapter = OpenAICompatibleAdapter()
    result_holder: list = []

    # -- Phase 1: Load dataset (with visible progress) --
    from tool_eval_bench.plugins.gsm8k.dataset import _find_cache_file, load_dataset

    cache_path = _find_cache_file()
    if cache_path.exists():
        console.print("  [dim]Loading GSM8K from cache…[/]", end=" ")
        dataset_items = load_dataset()
        console.print(f"[bold green]✓[/] [dim]{len(dataset_items)} questions[/]")
    else:
        # First use — download with visible progress
        try:
            import datasets as _ds  # noqa: F401

            method_hint = "via datasets lib"
        except ImportError:
            method_hint = "via REST API"
        console.print()
        with console.status(
            f"[bold]Downloading GSM8K dataset from HuggingFace…[/] [dim]({method_hint})[/]",
            spinner="dots",
        ) as status:

            def on_download(downloaded: int, total: int) -> None:
                pct = downloaded / total * 100 if total else 0
                status.update(
                    f"[bold]Downloading GSM8K dataset…[/] "
                    f"[dim]{downloaded:,}/{total:,} questions ({pct:.0f}%)[/]"
                )

            try:
                dataset_items = load_dataset(on_progress=on_download)
            except Exception as exc:
                console.print(
                    f"\n  [bold red]✗[/] Failed to download GSM8K dataset: {exc}\n"
                    "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                    "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
                )
                return

        console.print(
            f"  [bold green]✓[/] Downloaded [bold]{len(dataset_items)}[/] questions "
            f"[dim](cached to data/gsm8k/test.jsonl)[/]"
        )

    # -- Phase 2: Evaluate with model --
    async def run() -> None:
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        eval_total = limit if limit > 0 else len(dataset_items)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )

        stats_text = TextColumn("")
        stats_progress = Progress(stats_text, console=console)
        last_q_text = TextColumn("")
        last_q_progress = Progress(last_q_text, console=console)

        from rich.console import Group

        group = Group(progress, stats_progress, last_q_progress)

        correct_so_far = 0
        wrong_so_far = 0
        errors_so_far = 0
        t_start = time.monotonic()
        stats_progress.add_task("", total=None)
        last_q_progress.add_task("", total=None)

        with Live(group, console=console, refresh_per_second=4):
            task = progress.add_task("Evaluating…", total=eval_total)

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                nonlocal correct_so_far, wrong_so_far, errors_so_far
                if item_info.get("is_error"):
                    errors_so_far += 1
                elif item_info.get("correct"):
                    correct_so_far += 1
                else:
                    wrong_so_far += 1

                answered = correct_so_far + wrong_so_far
                pct = (correct_so_far / answered * 100) if answered > 0 else 0
                elapsed = time.monotonic() - t_start
                speed = current / elapsed * 60 if elapsed > 0 else 0  # questions/min

                # Build a compact status line
                status_parts = [
                    f"  [bold green]✓ {correct_so_far}[/]",
                    f"[bold red]✗ {wrong_so_far}[/]",
                ]
                if errors_so_far > 0:
                    status_parts.append(f"[bold yellow]⚠ {errors_so_far}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold magenta]{pct:.1f}%[/] accuracy",
                    "[dim]│[/]",
                    f"[dim]{speed:.1f} q/min[/]",
                ]
                stats_text.text_format = "  ".join(status_parts)

                progress.update(task, completed=current, total=total)

                # Show last completed question
                if item_info.get("is_error"):
                    icon = "[yellow]⚠[/]"
                elif item_info.get("correct", False):
                    icon = "[green]✓[/]"
                else:
                    icon = "[red]✗[/]"
                got = item_info.get("extracted_answer", "?")
                expected = item_info.get("ground_truth", "?")
                question = (item_info.get("question") or "").replace("\n", " ").strip()
                if len(question) > 90:
                    question = question[:87] + "…"
                last_q_text.text_format = (
                    f"  {icon} [bold]{got}[/]/{expected} [dim italic]{question}[/]"
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    n_shots=n_shots,
                    limit=limit,
                    shuffle=shuffle,
                    concurrency=args.parallel,
                    _preloaded_items=dataset_items,
                )
                result_holder.append(result)

                # Final state
                progress.update(
                    task, completed=result.details["total"], description="[green]✓ Complete"
                )
                final_speed = (
                    result.details["total"] / result.duration_seconds * 60
                    if result.duration_seconds > 0
                    else 0
                )
                errs = result.details.get("errors", 0)
                wrong = result.details["total"] - result.details["correct"] - errs
                parts = f"  [bold green]✓ {result.details['correct']}[/]  [bold red]✗ {wrong}[/]  "
                if errs > 0:
                    parts += f"[bold yellow]⚠ {errs} errors[/]  "
                parts += (
                    f"[dim]│[/]  "
                    f"[bold magenta]{result.score:.1f}%[/] accuracy  "
                    f"[dim]│[/]  "
                    f"[dim]{final_speed:.1f} q/min[/]"
                )
                stats_text.text_format = parts
                last_q_text.text_format = ""
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]GSM8K error:[/] {exc}")
        sys.exit(1)

    if not result_holder:
        console.print("[bold red]No GSM8K results.[/]")
        return

    result = result_holder[0]
    details = result.details

    # Display summary
    console.print()
    errs = details.get("errors", 0)
    answered = details["total"] - errs
    console.print(
        f"  [bold]GSM8K Accuracy:[/] [bold magenta]{result.score:.1f}%[/] "
        f"({details['correct']}/{answered})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (server timeouts/failures — excluded from accuracy)"
        )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    # Write report
    if output_dir or True:  # Always write reports
        from tool_eval_bench.storage.reports import MarkdownReporter
        from tool_eval_bench.utils.ids import build_run_id

        run_config = _with_config_fingerprint(
            {
                "model": model,
                "base_url": base_url,
                "mode": "gsm8k",
                "n_shots": n_shots,
                "limit": limit,
                "temperature": args.temperature,
                "seed": seed,
                "shuffle": shuffle,
            }
        )
        run_id = build_run_id(run_config)
        reporter = MarkdownReporter(root=output_dir)
        report_lines = plugin.render_report_section(result)
        _write_gsm8k_report(
            reporter, run_id, display_name, result, report_lines, run_context=run_context
        )

        # Persist to SQLite (project rule: every run → SQLite)
        _persist_plugin_run(
            {
                "run_id": run_id,
                "run_type": "gsm8k",
                "status": "completed",
                "config": run_config,
                "scores": {
                    "final_score": round(result.score),
                    "accuracy": result.score,
                    "rating": result.rating,
                    **result.details,
                },
                "metadata": _metadata_for_storage(run_context),
            }
        )

        console.print("\n  [dim]Report saved to runs/[/]\n")


def _write_gsm8k_report(
    reporter: Any,
    run_id: str,
    model: str,
    result: Any,
    report_lines: list[str],
    *,
    run_context: Any | None = None,
) -> None:
    """Write a standalone GSM8K Markdown report."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    folder = reporter.root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.md"

    md = [
        f"# GSM8K Benchmark — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: gsm8k",
        f"- **Accuracy**: **{result.score:.1f}%**",
        f"- **Rating**: {result.rating}",
        "",
    ]

    md.extend(report_lines)

    path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# MMLU benchmark (--mmlu / --mmlu-only)
# ---------------------------------------------------------------------------


def _run_mmlu_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run the MMLU benchmark and display results."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
    from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

    n_shots = args.mmlu_shots
    limit = args.mmlu_limit
    subjects_str = args.mmlu_subjects
    seed = getattr(args, "seed", None)
    limit_label = "all 14042" if limit == 0 else f"{limit}"
    subjects_list = [s.strip() for s in subjects_str.split(",")] if subjects_str else None
    subjects_label = f" · subjects: {subjects_str}" if subjects_str else ""

    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]{n_shots}-shot · {limit_label} questions{subjects_label}{parallel_label}[/]",
            title="[bold]🧠 MMLU — Massive Multitask Language Understanding[/]",
            border_style="bright_blue",
        )
    )

    plugin = MMLUPlugin()
    adapter = OpenAICompatibleAdapter()
    result_holder: list = []

    # -- Phase 1: Load dataset (with visible progress) --
    from tool_eval_bench.plugins.mmlu.dataset import _find_cache_file, load_dataset

    cache_path = _find_cache_file("test")
    if cache_path.exists():
        console.print("  [dim]Loading MMLU from cache…[/]", end=" ")
        test_items = load_dataset("test")
        console.print(f"[bold green]✓[/] [dim]{len(test_items)} questions[/]")
    else:
        from pathlib import Path as _Path

        partial_path = _Path("data") / "mmlu" / "test.partial.jsonl"
        resuming = partial_path.exists()
        # Check which download method will be used
        try:
            import datasets as _ds  # noqa: F401

            method_hint = "via datasets lib"
        except ImportError:
            method_hint = "via REST API"
        label = "Resuming MMLU download" if resuming else "Downloading MMLU dataset"
        console.print()
        with console.status(
            f"[bold]{label} from HuggingFace…[/] [dim]({method_hint})[/]",
            spinner="dots",
        ) as status:

            def on_download(downloaded: int, total: int) -> None:
                pct = downloaded / total * 100 if total else 0
                status.update(
                    f"[bold]{label}…[/] [dim]{downloaded:,}/{total:,} questions ({pct:.0f}%)[/]"
                )

            try:
                test_items = load_dataset("test", on_progress=on_download)
            except Exception as exc:
                console.print(
                    f"\n  [bold red]✗[/] Failed to download MMLU dataset: {exc}\n"
                    "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                    "  Progress is saved — re-run to resume from where it stopped.\n"
                    "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
                )
                return
        console.print(
            f"  [bold green]✓[/] Downloaded [bold]{len(test_items)}[/] questions "
            f"[dim](cached to data/mmlu/test.jsonl)[/]"
        )

    # Load dev split for few-shot
    dev_items = []
    if n_shots > 0:
        dev_cache = _find_cache_file("dev")
        if dev_cache.exists():
            dev_items = load_dataset("dev")
        else:
            with console.status("[dim]Downloading MMLU dev split…[/]", spinner="dots"):
                dev_items = load_dataset("dev")
            console.print(f"  [dim]Loaded {len(dev_items)} dev examples for few-shot[/]")

    preloaded = {"test": test_items, "dev": dev_items}

    # -- Phase 2: Evaluate with model --
    async def run() -> None:
        from rich.console import Group
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        eval_total = limit if limit > 0 else len(test_items)
        if subjects_list:
            # Adjust for filtering
            from tool_eval_bench.plugins.mmlu.dataset import CATEGORIES, SUBJECT_CATEGORIES

            expanded: set[str] = set()
            for s in subjects_list:
                if s in CATEGORIES:
                    expanded.update(subj for subj, cat in SUBJECT_CATEGORIES.items() if cat == s)
                else:
                    expanded.add(s)
            filtered = [it for it in test_items if it.subject in expanded]
            eval_total = min(eval_total, len(filtered)) if limit > 0 else len(filtered)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )

        stats_text = TextColumn("")
        stats_progress = Progress(stats_text, console=console)
        last_q_text = TextColumn("")
        last_q_progress = Progress(last_q_text, console=console)
        group = Group(progress, stats_progress, last_q_progress)

        correct_so_far = 0
        wrong_so_far = 0
        errors_so_far = 0
        t_start = time.monotonic()
        stats_progress.add_task("", total=None)
        last_q_progress.add_task("", total=None)

        with Live(group, console=console, refresh_per_second=4):
            task = progress.add_task("Evaluating…", total=eval_total)

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                nonlocal correct_so_far, wrong_so_far, errors_so_far
                if item_info.get("is_error"):
                    errors_so_far += 1
                elif item_info.get("correct"):
                    correct_so_far += 1
                else:
                    wrong_so_far += 1

                answered = correct_so_far + wrong_so_far
                pct = (correct_so_far / answered * 100) if answered > 0 else 0
                elapsed = time.monotonic() - t_start
                speed = current / elapsed * 60 if elapsed > 0 else 0

                status_parts = [
                    f"  [bold green]✓ {correct_so_far}[/]",
                    f"[bold red]✗ {wrong_so_far}[/]",
                ]
                if errors_so_far > 0:
                    status_parts.append(f"[bold yellow]⚠ {errors_so_far}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold blue]{pct:.1f}%[/] accuracy",
                    "[dim]│[/]",
                    f"[dim]{speed:.1f} q/min[/]",
                ]
                stats_text.text_format = "  ".join(status_parts)
                progress.update(task, completed=current, total=total)

                # Show last completed question details
                subj = item_info.get("subject", "?")
                if item_info.get("is_error"):
                    icon = "[yellow]⚠[/]"
                elif item_info.get("correct", False):
                    icon = "[green]✓[/]"
                else:
                    icon = "[red]✗[/]"
                got = item_info.get("extracted_answer", "?")
                expected = item_info.get("ground_truth", "?")
                question = (item_info.get("question") or "").replace("\n", " ").strip()
                if len(question) > 90:
                    question = question[:87] + "…"
                last_q_text.text_format = (
                    f"  {icon} [bold]{got}[/]/{expected} [dim]{subj}[/]  [dim italic]{question}[/]"
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    n_shots=n_shots,
                    limit=limit,
                    subjects=subjects_list,
                    concurrency=args.parallel,
                    _preloaded_items=preloaded,
                )
                result_holder.append(result)

                progress.update(
                    task, completed=result.details["total"], description="[green]✓ Complete"
                )
                final_speed = (
                    result.details["total"] / result.duration_seconds * 60
                    if result.duration_seconds > 0
                    else 0
                )
                errs = result.details.get("errors", 0)
                wrong = result.details["total"] - result.details["correct"] - errs
                parts = f"  [bold green]✓ {result.details['correct']}[/]  [bold red]✗ {wrong}[/]  "
                if errs > 0:
                    parts += f"[bold yellow]⚠ {errs} errors[/]  "
                parts += (
                    f"[dim]│[/]  "
                    f"[bold blue]{result.score:.1f}%[/] accuracy  "
                    f"[dim]│[/]  "
                    f"[dim]{final_speed:.1f} q/min[/]"
                )
                stats_text.text_format = parts
                last_q_text.text_format = ""
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]MMLU error:[/] {exc}")
        sys.exit(1)

    if not result_holder:
        console.print("[bold red]No MMLU results.[/]")
        return

    result = result_holder[0]
    details = result.details

    console.print()
    errs = details.get("errors", 0)
    answered = details["total"] - errs
    console.print(
        f"  [bold]MMLU Accuracy:[/] [bold blue]{result.score:.1f}%[/] "
        f"({details['correct']}/{answered})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (server timeouts/failures — excluded from accuracy)"
        )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    # Show category breakdown
    cats = details.get("categories", {})
    if cats:
        parts = [f"{cat}: {c['accuracy']:.1f}%" for cat, c in sorted(cats.items())]
        console.print(f"  [dim]{' · '.join(parts)}[/]")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    # Write report
    from datetime import datetime, timezone

    from tool_eval_bench.storage.reports import MarkdownReporter
    from tool_eval_bench.utils.ids import build_run_id

    run_config = _with_config_fingerprint(
        {
            "model": model,
            "base_url": base_url,
            "mode": "mmlu",
            "n_shots": n_shots,
            "limit": limit,
            "temperature": args.temperature,
            "seed": seed,
            "subjects": subjects_str,
        }
    )
    run_id = build_run_id(run_config)
    reporter = MarkdownReporter(root=output_dir)
    report_lines = plugin.render_report_section(result)

    now = datetime.now(timezone.utc)
    folder = reporter.root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.md"
    md = [
        f"# MMLU Benchmark — {display_name}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: mmlu",
        f"- **Accuracy**: **{result.score:.1f}%**",
        f"- **Rating**: {result.rating}",
        "",
    ]
    md.extend(report_lines)
    path.write_text("\n".join(md), encoding="utf-8")

    # Persist to SQLite (project rule: every run → SQLite)
    _persist_plugin_run(
        {
            "run_id": run_id,
            "run_type": "mmlu",
            "status": "completed",
            "config": run_config,
            "scores": {
                "final_score": round(result.score),
                "accuracy": result.score,
                "rating": result.rating,
                **result.details,
            },
            "metadata": _metadata_for_storage(run_context),
        }
    )

    console.print("\n  [dim]Report saved to runs/[/]\n")


# ---------------------------------------------------------------------------
# IFEval benchmark (--ifeval / --ifeval-only)
# ---------------------------------------------------------------------------


def _run_ifeval_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run the IFEval instruction-following benchmark and display results."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
    from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

    limit = args.ifeval_limit
    seed = getattr(args, "seed", None)
    limit_label = "all 541" if limit == 0 else f"{limit}"

    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n[dim]{limit_label} prompts · 25 constraint types{parallel_label}[/]",
            title="[bold]📋 IFEval — Instruction Following Evaluation[/]",
            border_style="bright_green",
        )
    )

    plugin = IFEvalPlugin()
    adapter = OpenAICompatibleAdapter()
    result_holder: list = []

    # -- Phase 1: Load dataset --
    from tool_eval_bench.plugins.ifeval.dataset import _find_cache_file, load_dataset

    cache_path = _find_cache_file()
    if cache_path.exists():
        console.print("  [dim]Loading IFEval from cache…[/]", end=" ")
        dataset_items = load_dataset()
        console.print(f"[bold green]✓[/] [dim]{len(dataset_items)} prompts[/]")
    else:
        from pathlib import Path as _Path

        partial_path = _Path("data") / "ifeval" / "prompts.partial.jsonl"
        resuming = partial_path.exists()
        try:
            import datasets as _ds  # noqa: F401

            method_hint = "via datasets lib"
        except ImportError:
            method_hint = "via REST API"
        label = "Resuming IFEval download" if resuming else "Downloading IFEval dataset"
        console.print()
        with console.status(
            f"[bold]{label} from HuggingFace…[/] [dim]({method_hint})[/]",
            spinner="dots",
        ) as status:

            def on_download(downloaded: int, total: int) -> None:
                pct = downloaded / total * 100 if total else 0
                status.update(
                    f"[bold]{label}…[/] [dim]{downloaded:,}/{total:,} prompts ({pct:.0f}%)[/]"
                )

            try:
                dataset_items = load_dataset(on_progress=on_download)
            except Exception as exc:
                console.print(
                    f"\n  [bold red]✗[/] Failed to download IFEval dataset: {exc}\n"
                    "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                    "  Progress is saved — re-run to resume from where it stopped.\n"
                    "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
                )
                return
        console.print(
            f"  [bold green]✓[/] Downloaded [bold]{len(dataset_items)}[/] prompts "
            f"[dim](cached to data/ifeval/prompts.jsonl)[/]"
        )

    # -- Phase 2: Evaluate with model --
    async def run() -> None:
        from rich.console import Group
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        eval_total = limit if limit > 0 else len(dataset_items)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )

        stats_text = TextColumn("")
        stats_progress = Progress(stats_text, console=console)
        last_q_text = TextColumn("")
        last_q_progress = Progress(last_q_text, console=console)
        group = Group(progress, stats_progress, last_q_progress)

        prompts_passed = 0
        prompts_failed = 0
        errors_so_far = 0
        instructions_passed = 0
        instructions_total = 0
        t_start = time.monotonic()
        stats_progress.add_task("", total=None)
        last_q_progress.add_task("", total=None)

        with Live(group, console=console, refresh_per_second=4):
            task = progress.add_task("Evaluating…", total=eval_total)

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                nonlocal prompts_passed, prompts_failed, errors_so_far
                nonlocal instructions_passed, instructions_total
                if item_info.get("is_error"):
                    errors_so_far += 1
                elif item_info.get("prompt_pass"):
                    prompts_passed += 1
                else:
                    prompts_failed += 1
                instructions_passed += item_info.get("instructions_passed", 0)
                instructions_total += item_info.get("instructions_total", 0)

                answered = prompts_passed + prompts_failed
                prompt_pct = (prompts_passed / answered * 100) if answered > 0 else 0
                inst_pct = (
                    (instructions_passed / instructions_total * 100)
                    if instructions_total > 0
                    else 0
                )
                elapsed = time.monotonic() - t_start
                speed = current / elapsed * 60 if elapsed > 0 else 0

                status_parts = [
                    f"  [bold green]✓ {prompts_passed}[/]",
                    f"[bold red]✗ {prompts_failed}[/]",
                ]
                if errors_so_far > 0:
                    status_parts.append(f"[bold yellow]⚠ {errors_so_far}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold green]{prompt_pct:.1f}%[/] prompt",
                    f"[bold cyan]{inst_pct:.1f}%[/] instr",
                    "[dim]│[/]",
                    f"[dim]{speed:.1f} p/min[/]",
                ]
                stats_text.text_format = "  ".join(status_parts)
                progress.update(task, completed=current, total=total)

                # Show last completed prompt
                if item_info.get("is_error"):
                    icon = "[yellow]⚠[/]"
                elif item_info.get("prompt_pass", False):
                    icon = "[green]✓[/]"
                else:
                    icon = "[red]✗[/]"
                ip = item_info.get("instructions_passed", 0)
                it = item_info.get("instructions_total", 0)
                prompt = (item_info.get("prompt") or "").replace("\n", " ").strip()
                if len(prompt) > 90:
                    prompt = prompt[:87] + "…"
                last_q_text.text_format = (
                    f"  {icon} [bold]{ip}[/]/{it} constraints  [dim italic]{prompt}[/]"
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    limit=limit,
                    concurrency=args.parallel,
                    _preloaded_items=dataset_items,
                )
                result_holder.append(result)

                progress.update(
                    task, completed=result.details["total"], description="[green]✓ Complete"
                )
                d = result.details
                final_speed = (
                    d["total"] / result.duration_seconds * 60 if result.duration_seconds > 0 else 0
                )
                errs = d.get("errors", 0)
                wrong = d["total"] - d["prompts_passed"] - errs
                parts = f"  [bold green]✓ {d['prompts_passed']}[/]  [bold red]✗ {wrong}[/]  "
                if errs > 0:
                    parts += f"[bold yellow]⚠ {errs} errors[/]  "
                parts += (
                    f"[dim]│[/]  "
                    f"[bold green]{d['prompt_accuracy']:.1f}%[/] prompt  "
                    f"[bold cyan]{d.get('instruction_accuracy', 0):.1f}%[/] instr  "
                    f"[dim]│[/]  "
                    f"[dim]{final_speed:.1f} p/min[/]"
                )
                stats_text.text_format = parts
                last_q_text.text_format = ""
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]IFEval error:[/] {exc}")
        sys.exit(1)

    if not result_holder:
        console.print("[bold red]No IFEval results.[/]")
        return

    result = result_holder[0]
    details = result.details

    console.print()
    errs = details.get("errors", 0)
    answered = details["total"] - errs
    console.print(
        f"  [bold]IFEval Prompt Accuracy:[/] [bold green]{details.get('prompt_accuracy', 0):.1f}%[/] "
        f"({details['prompts_passed']}/{answered})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (server timeouts/failures — excluded from accuracy)"
        )
    console.print(
        f"  [bold]IFEval Instruction Accuracy:[/] [bold cyan]"
        f"{details.get('instruction_accuracy', 0):.1f}%[/] "
        f"({details.get('instructions_passed', 0)}/{details.get('instructions_total', 0)})"
    )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    # Write report
    from datetime import datetime, timezone

    from tool_eval_bench.storage.reports import MarkdownReporter
    from tool_eval_bench.utils.ids import build_run_id

    run_config = _with_config_fingerprint(
        {
            "model": model,
            "base_url": base_url,
            "mode": "ifeval",
            "limit": limit,
            "temperature": args.temperature,
            "seed": seed,
        }
    )
    run_id = build_run_id(run_config)
    reporter = MarkdownReporter(root=output_dir)
    report_lines = plugin.render_report_section(result)

    now = datetime.now(timezone.utc)
    folder = reporter.root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.md"
    md = [
        f"# IFEval Benchmark — {display_name}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: ifeval",
        f"- **Prompt Accuracy**: **{details.get('prompt_accuracy', 0):.1f}%**",
        f"- **Instruction Accuracy**: **{details.get('instruction_accuracy', 0):.1f}%**",
        f"- **Rating**: {result.rating}",
        "",
    ]
    md.extend(report_lines)
    path.write_text("\n".join(md), encoding="utf-8")

    # Persist to SQLite (project rule: every run → SQLite)
    _persist_plugin_run(
        {
            "run_id": run_id,
            "run_type": "ifeval",
            "status": "completed",
            "config": run_config,
            "scores": {
                "final_score": round(result.score),
                "accuracy": result.score,
                "rating": result.rating,
                **result.details,
            },
            "metadata": _metadata_for_storage(run_context),
        }
    )

    console.print("\n  [dim]Report saved to runs/[/]\n")


def _make_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Extracted from ``main()`` so that tests and external tools can introspect
    the full argument list without calling ``parse_args()`` (which would consume
    sys.argv).
    """
    parser = argparse.ArgumentParser(
        description="Run tool-eval-bench agentic tool-call benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Connection --------------------------------------------------------
    conn = parser.add_argument_group("connection")
    conn.add_argument("--model", default=None, help="Model name/path (auto-detected if omitted)")
    conn.add_argument(
        "--backend",
        default=None,
        help="Backend label for reports: vllm, litellm, llamacpp "
        "(all use the same OpenAI-compatible adapter; default: env/vllm)",
    )
    conn.add_argument(
        "--base-url",
        default=None,
        help="Server base URL (default: auto-discover on localhost, or from .env)",
    )
    conn.add_argument("--api-key", default=None, help="API key")
    conn.add_argument(
        "--probe",
        action="store_true",
        help="Check if a server is reachable and exit (exit 0 = ready, exit 1 = not found)",
    )

    # -- Sampling ----------------------------------------------------------
    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature (default: 0.0)"
    )
    sampling.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking/reasoning (sets enable_thinking=false)",
    )
    sampling.add_argument(
        "--top-p", type=float, default=None, metavar="P", help="Top-p (nucleus) sampling (e.g. 0.9)"
    )
    sampling.add_argument(
        "--top-k", type=int, default=None, metavar="K", help="Top-k sampling (e.g. 40)"
    )
    sampling.add_argument(
        "--min-p",
        type=float,
        default=None,
        metavar="P",
        help="Min-p sampling threshold (e.g. 0.05)",
    )
    sampling.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        metavar="V",
        help="Repetition penalty (e.g. 1.1)",
    )
    sampling.add_argument("--seed", type=int, default=None, help="Random seed (passed to server)")
    sampling.add_argument(
        "--backend-kwargs",
        type=str,
        default=None,
        metavar="JSON",
        help="JSON dict merged into API payload; overrides individual flags "
        '(e.g. \'{"temperature": 0.6, "top_p": 0.9}\')',
    )

    # -- Scenario selection ------------------------------------------------
    select = parser.add_argument_group("scenario selection")
    select.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Specific scenario IDs to run (e.g. TC-01 TC-07). Default: all.",
    )
    select.add_argument(
        "--categories",
        nargs="*",
        default=None,
        metavar="CAT",
        help="Run only specific categories (e.g. --categories K A J). "
        "Letters A–O map to the 15 benchmark categories.",
    )
    select.add_argument(
        "--short",
        action="store_true",
        help="Run only the core 15 scenarios (skip extended + agentic)",
    )
    select.add_argument(
        "--hardmode",
        action="store_true",
        help="Include Hard Mode scenarios (Category P) — ceiling-breaking difficulty "
        "for models that score 100%% on the standard benchmark",
    )
    select.add_argument(
        "--hardmode-only",
        action="store_true",
        help="Run ONLY Hard Mode scenarios (Category P) — shortcut for --hardmode --categories P",
    )

    # -- Run control -------------------------------------------------------
    run_ctrl = parser.add_argument_group("run control")
    run_ctrl.add_argument(
        "--timeout", type=float, default=60.0, help="Request timeout in seconds (default: 60)"
    )
    run_ctrl.add_argument(
        "--max-turns", type=int, default=8, help="Max turns per scenario (default: 8)"
    )
    run_ctrl.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trial runs for statistical rigor (default: 1)",
    )
    run_ctrl.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run N scenarios concurrently (default: 1 = sequential)",
    )
    run_ctrl.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        metavar="RATE",
        help="Inject random tool errors at this rate (0.0–1.0) for robustness testing",
    )
    run_ctrl.add_argument("--no-warmup", action="store_true", help="Skip server warm-up request")
    run_ctrl.add_argument(
        "--reference-date", default=None, help="Override benchmark reference date (YYYY-MM-DD)"
    )
    run_ctrl.add_argument(
        "--skip-tool-eval",
        action="store_true",
        help="Skip tool-call scenarios (use with --perf / --spec-bench)",
    )

    # -- Output ------------------------------------------------------------
    output = parser.add_argument_group("output")
    output.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of rich display"
    )
    output.add_argument(
        "--json-file",
        default=None,
        metavar="PATH",
        help="Write JSON results to PATH instead of stdout "
        "(implies --json; keeps stdout clean for logging)",
    )
    output.add_argument("--no-live", action="store_true", help="Disable live updating display")
    output.add_argument(
        "--redact-url",
        action="store_true",
        help="Mask the server URL in display output (for screenshots/recordings)",
    )
    output.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        metavar="W",
        help="Quality/speed weight for deployability score (0–1, default: 0.7)",
    )
    output.add_argument(
        "--no-probe-engine",
        action="store_true",
        help="Skip inference engine probing (no /version, /health HTTP calls)",
    )
    output.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for report files (default: ./runs/)",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which scenarios would run, then exit (no server needed)",
    )

    # -- Throughput (llama-benchy) -----------------------------------------
    perf_grp = parser.add_argument_group("throughput benchmark (llama-benchy)")
    perf_grp.add_argument(
        "--perf", action="store_true", help="Run throughput benchmark before tool-call scenarios"
    )
    perf_grp.add_argument(
        "--perf-only",
        action="store_true",
        help="Run ONLY throughput benchmark (skip tool-call scenarios)",
    )
    perf_grp.add_argument(
        "--perf-legacy",
        action="store_true",
        help="Use built-in throughput benchmark (no external deps)",
    )
    perf_grp.add_argument(
        "--perf-legacy-only", action="store_true", help="Run ONLY built-in throughput benchmark"
    )
    perf_grp.add_argument("--pp", type=int, default=2048, help="Prompt tokens (default: 2048)")
    perf_grp.add_argument("--tg", type=int, default=128, help="Generation tokens (default: 128)")
    perf_grp.add_argument(
        "--depth",
        type=str,
        default="0,4096,8192",
        help="Context depths, comma separated (default: '0,4096,8192')",
    )
    perf_grp.add_argument(
        "--concurrency", type=str, default="1,2,4", help="Concurrency levels (default: '1,2,4')"
    )
    perf_grp.add_argument(
        "--benchy-runs", type=int, default=3, help="Measurement runs per test point (default: 3)"
    )
    perf_grp.add_argument(
        "--benchy-latency-mode",
        default="generation",
        choices=["api", "generation", "none"],
        help="Latency measurement mode (default: generation)",
    )
    perf_grp.add_argument(
        "--benchy-args",
        type=str,
        default=None,
        help="Pass-through args for llama-benchy (quoted string)",
    )
    perf_grp.add_argument(
        "--skip-coherence",
        action="store_true",
        help="Deprecated: llama-benchy coherence check is now always skipped (retained for compatibility)",
    )

    # -- GSM8K benchmark ----------------------------------------------------
    gsm8k_grp = parser.add_argument_group("GSM8K benchmark")
    gsm8k_grp.add_argument(
        "--gsm8k",
        action="store_true",
        help="Run GSM8K (Grade School Math) benchmark after tool-call scenarios",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-only",
        action="store_true",
        help="Run ONLY the GSM8K benchmark (skip tool-call scenarios)",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-shots",
        type=int,
        default=8,
        metavar="N",
        help="Number of few-shot CoT examples (0–8, default: 8)",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-limit",
        type=int,
        default=200,
        metavar="N",
        help="Max questions to evaluate (default: 200, 0 = all 1319)",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-shuffle",
        action="store_true",
        help="Shuffle question order (uses --seed for reproducibility)",
    )

    # -- MMLU benchmark -----------------------------------------------------
    mmlu_grp = parser.add_argument_group("MMLU benchmark")
    mmlu_grp.add_argument(
        "--mmlu",
        action="store_true",
        help="Run MMLU (Massive Multitask Language Understanding) benchmark",
    )
    mmlu_grp.add_argument(
        "--mmlu-only",
        action="store_true",
        help="Run ONLY the MMLU benchmark (skip tool-call scenarios)",
    )
    mmlu_grp.add_argument(
        "--mmlu-shots",
        type=int,
        default=5,
        metavar="N",
        help="Number of few-shot examples per subject (0–5, default: 5)",
    )
    mmlu_grp.add_argument(
        "--mmlu-limit",
        type=int,
        default=500,
        metavar="N",
        help="Max questions to evaluate (default: 500, 0 = all 14042)",
    )
    mmlu_grp.add_argument(
        "--mmlu-subjects",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated subjects or categories (e.g. 'STEM,abstract_algebra')",
    )

    # -- IFEval benchmark ---------------------------------------------------
    ifeval_grp = parser.add_argument_group("IFEval benchmark")
    ifeval_grp.add_argument(
        "--ifeval", action="store_true", help="Run IFEval (Instruction Following) benchmark"
    )
    ifeval_grp.add_argument(
        "--ifeval-only",
        action="store_true",
        help="Run ONLY the IFEval benchmark (skip tool-call scenarios)",
    )
    ifeval_grp.add_argument(
        "--ifeval-limit",
        type=int,
        default=0,
        metavar="N",
        help="Max prompts to evaluate (default: 0 = all 541)",
    )

    # -- Speculative decoding benchmark ------------------------------------
    spec_grp = parser.add_argument_group("speculative decoding benchmark")
    spec_grp.add_argument(
        "--spec-bench",
        action="store_true",
        help="Run spec-decode / MTP benchmark (effective t/s, acceptance rate)",
    )
    spec_grp.add_argument(
        "--spec-live",
        action="store_true",
        help="Live-monitor speculative decoding stats (polls /metrics, runs until Ctrl+C)",
    )
    spec_grp.add_argument(
        "--spec-live-interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Poll interval for --spec-live in seconds (default: 1.0)",
    )
    spec_grp.add_argument(
        "--spec-method",
        default="auto",
        choices=["auto", "mtp", "draft", "dflash", "ngram", "eagle"],
        help="Spec-decode method hint (default: auto-detect)",
    )
    spec_grp.add_argument(
        "--baseline-tgs",
        type=float,
        default=None,
        metavar="TPS",
        help="Baseline tg t/s for speedup ratio calculation",
    )
    spec_grp.add_argument(
        "--spec-prompts",
        type=str,
        default="filler,code,structured",
        help="Prompt types, comma separated (default: 'filler,code,structured')",
    )
    spec_grp.add_argument(
        "--metrics-url",
        type=str,
        default=None,
        metavar="URL",
        help="Prometheus /metrics URL for acceptance rate (when API is behind a proxy)",
    )

    # -- Context pressure --------------------------------------------------
    pressure = parser.add_argument_group("context pressure")
    pressure.add_argument(
        "--context-pressure",
        type=float,
        default=None,
        metavar="RATIO",
        help="Fill context to RATIO (0.0–1.0) before each scenario",
    )
    pressure.add_argument(
        "--context-size",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Override auto-detected context window size (tokens)",
    )
    pressure.add_argument(
        "--context-pressure-sweep",
        type=str,
        default=None,
        metavar="START-END",
        help="Sweep pressure from START to END (e.g. 0.5-1.0)",
    )
    pressure.add_argument(
        "--sweep-steps",
        type=int,
        default=5,
        metavar="N",
        help="Number of pressure levels to test (default: 5)",
    )

    # -- History & comparison ----------------------------------------------
    hist_grp = parser.add_argument_group("history & comparison")
    hist_grp.add_argument(
        "--diff",
        metavar="RUN_ID",
        default=None,
        help="Compare against a previous run (use 'latest' for most recent)",
    )
    hist_grp.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        default=None,
        help="Compare two stored runs by ID",
    )
    hist_grp.add_argument(
        "--history", action="store_true", help="List recent benchmark runs and exit"
    )
    hist_grp.add_argument(
        "--leaderboard", action="store_true", help="Show ranked model leaderboard and exit"
    )
    hist_grp.add_argument(
        "--export",
        metavar="FORMAT",
        default=None,
        choices=["csv", "json"],
        help="Export all results in CSV or JSON format and exit",
    )
    hist_grp.add_argument(
        "--export-output",
        metavar="FILE",
        default=None,
        help="Output file for --export (default: stdout)",
    )
    hist_grp.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume a previous run — skip scenarios that already passed",
    )

    # -- Scoring options ---------------------------------------------------
    scoring = parser.add_argument_group("scoring")
    scoring.add_argument(
        "--weight-by-difficulty",
        action="store_true",
        help="Weight scenario scores by difficulty tier (1×trivial … 5×very hard)",
    )

    # -- Hidden / WIP (not shown in --help) --------------------------------
    parser.add_argument("--llm-judge", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--judge-model", type=str, default=None, metavar="MODEL", help=argparse.SUPPRESS
    )
    parser.add_argument("--experimental-async", action="store_true", help=argparse.SUPPRESS)

    return parser


# Set of argument dest names that are intentionally suppressed (not in ARGS_SCHEMA).
# Used by the drift-detection test in tests/test_api.py.
_HIDDEN_ARGS: frozenset[str] = frozenset({"llm_judge", "judge_model", "experimental_async", "help"})


def main() -> None:
    _load_dotenv()
    parser = _make_parser()
    args = parser.parse_args()

    # --json-file implies --json
    if args.json_file:
        args.json = True

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

    # --dry-run: show what would run and exit (no server needed)
    if args.dry_run:
        scenarios = _resolve_scenarios(args)
        from tool_eval_bench.domain.scenarios import CATEGORY_LABELS

        if args.json:
            import json as _json

            cat_counts: dict[str, int] = {}
            for s in scenarios:
                cat_counts[s.category.value] = cat_counts.get(s.category.value, 0) + 1
            out = {
                "event": "dry_run",
                "total_scenarios": len(scenarios),
                "estimated_minutes": round(len(scenarios) * 0.3, 1),
                "categories": {
                    cat: {
                        "label": CATEGORY_LABELS.get(
                            next(s.category for s in scenarios if s.category.value == cat),
                            cat,
                        ),
                        "count": count,
                    }
                    for cat, count in sorted(cat_counts.items())
                },
                "scenarios": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "category": s.category.value,
                        "difficulty": s.difficulty,
                    }
                    for s in scenarios
                ],
            }
            sys.stdout.write(_json.dumps(out, indent=2) + "\n")
        else:
            console.print(f"\n[bold]Dry run:[/] {len(scenarios)} scenarios would execute\n")
            console.print(
                f"  Estimated time: ~{len(scenarios) * 0.3:.0f} minutes (at ~18s/scenario)\n"
            )
            cat_counts = {}
            for s in scenarios:
                label = CATEGORY_LABELS.get(s.category, s.category.value)
                cat_counts[label] = cat_counts.get(label, 0) + 1
            for label, count in sorted(cat_counts.items()):
                console.print(f"  {label}: {count} scenarios")
            console.print()
            _diff_stars = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★", 5: "★★★★★"}
            for s in scenarios:
                d = _diff_stars.get(s.difficulty, "?") if s.difficulty else "?"
                console.print(f"  [dim]{s.id}[/]  {d:>5s}  {s.title}")
            console.print()
        sys.exit(0)

    # Cascade: CLI flag → env var → auto-discovery
    model = args.model or os.getenv("TOOL_EVAL_MODEL") or None
    backend = args.backend or os.getenv("TOOL_EVAL_BACKEND", "")
    base_url = args.base_url or os.getenv("TOOL_EVAL_BASE_URL", "")
    api_key = args.api_key or os.getenv("TOOL_EVAL_API_KEY")

    # Fallback: construct URL from TOOL_EVAL_HOST + TOOL_EVAL_PORT
    if not base_url:
        host = os.getenv("TOOL_EVAL_HOST", "")
        port = os.getenv("TOOL_EVAL_PORT", "")
        if host:
            base_url = f"http://{host}:{port}" if port else f"http://{host}"

    # Auto-discovery: probe localhost on common inference server ports
    if not base_url:
        if not args.json:
            console.print("\n[dim]  No --base-url provided, scanning localhost…[/]")
        discovered = _discover_server(headless=args.json, console=console)
        if discovered:
            base_url, discovered_backend = discovered
            if not backend:
                backend = discovered_backend
        else:
            if args.json:
                _headless_error(
                    NO_SERVER,
                    "No inference server found on localhost. "
                    "Tried ports: " + ", ".join(str(p) for p, _, _ in _DISCOVERY_PORTS),
                    exit_code=2,
                )
            parser.error(
                "No inference server found on localhost. "
                "Use --base-url or set TOOL_EVAL_BASE_URL in .env"
            )

    # Default backend if still unset
    if not backend:
        backend = "vllm"

    # --probe: check if server is reachable and exit
    if args.probe:
        _probe_server(console, base_url, api_key, headless=args.json)
        return

    # URL redaction for display (actual API calls use real base_url)
    display_url = _redact_url(base_url) if args.redact_url else base_url

    # Auto-detect model if not provided
    display_name: str | None = None
    if not model:
        if not args.json:
            console.print("\n[bold]🔧 Tool-Call Benchmark[/]")
            console.print(f"[dim]  Server: {display_url}[/]")
        model, display_name = _detect_model(
            base_url,
            api_key,
            console,
            display_url=display_url,
            headless=args.json,
        )
        if not args.json:
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
                parser.error(
                    f"--backend-kwargs must be a JSON object (dict), got {type(bk).__name__}"
                )
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

        cat_names = ", ".join(f"{c} ({CATEGORY_LABELS[Category(c)]})" for c in cats)
        resolved_count = len(_resolve_scenarios(args))
        if not args.json:
            console.print(f"  [dim]📋 Categories: {cat_names} ({resolved_count} scenarios)[/]")

    # -- spec-live: standalone live monitor (exits after session) --
    if args.spec_live:
        # Map CLI choice names to internal method identifiers
        _method_map = {"draft": "draft_model"}
        raw_method = args.spec_method
        spec_method_hint = _method_map.get(raw_method, raw_method) if raw_method != "auto" else None

        from tool_eval_bench.cli.spec_live_display import run_spec_live

        try:
            asyncio.run(
                run_spec_live(
                    base_url,
                    api_key=api_key,
                    metrics_url=args.metrics_url,
                    model_name=display_name,
                    poll_interval=args.spec_live_interval,
                    spec_method=spec_method_hint,
                )
            )
        except KeyboardInterrupt:
            pass
        return

    # -- Pre-flight: verify the model actually works (issue #19) --
    # Some servers list models in /v1/models but fail on real requests.
    # Without this check, the benchmark produces misleading scores.
    _preflight_model_check(console, base_url, model, api_key, headless=args.json)

    # -- Warm-up --
    if not args.no_warmup and not args.json:
        _do_warmup(console, base_url, model, api_key)

    # -- Feature flags not yet wired into the run loop --
    if args.llm_judge and not args.json:
        console.print(
            "\n  [bold yellow]⚠ --llm-judge:[/] The judge module is implemented "
            "(runner/judge.py) but not yet wired into the benchmark flow. "
            "Judge results will not be applied in this run.\n"
        )
    if args.experimental_async and not args.json:
        console.print(
            "\n  [bold yellow]⚠ --experimental-async:[/] The async tool executor is "
            "implemented (runner/async_tools.py) but not yet integrated with "
            "the scenario orchestrator. This flag has no effect in this run.\n"
        )

    # -- Build RunContext (issue #6: full execution context metadata) --
    # Built early so perf-only and spec-bench paths also get engine detection.
    run_context = None
    try:
        from tool_eval_bench.utils.metadata import collect_run_context

        # Determine scenario selector description
        resolved_sc = _resolve_scenarios(args)
        if args.scenarios:
            scenario_sel = ", ".join(args.scenarios)
        elif args.categories:
            scenario_sel = (
                f"categories {', '.join(c.upper() for c in args.categories)} ({len(resolved_sc)})"
            )
        elif args.short:
            scenario_sel = f"short ({len(resolved_sc)})"
        else:
            scenario_sel = f"all ({len(resolved_sc)})"

        trials = max(1, args.trials)
        run_context = asyncio.run(
            collect_run_context(
                model=model,
                backend=backend,
                base_url=base_url,
                api_key=api_key,
                temperature=args.temperature,
                max_turns=args.max_turns,
                timeout_seconds=args.timeout,
                seed=args.seed,
                scenario_selector=scenario_sel,
                trials=trials,
                parallel=args.parallel,
                error_rate=args.error_rate,
                thinking_enabled=not args.no_think,
                extra_params=extra_params or None,
                context_pressure=args.context_pressure,
                probe_engine=not args.no_probe_engine,
            )
        )
        if not args.json and run_context.engine_name:
            engine_str = run_context.engine_name
            if run_context.engine_version:
                engine_str += f" {run_context.engine_version}"
            console.print(f"  [dim]🔍 Engine: {engine_str}[/]")
    except Exception as exc:
        logger.warning("Failed to build RunContext: %s", exc)

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
            console,
            model,
            display_name,
            base_url,
            api_key,
            pp=[args.pp],
            tg=[args.tg],
            depths=depths,
            concurrency_levels=conc_levels,
            runs=args.benchy_runs,
            latency_mode=args.benchy_latency_mode,
            skip_coherence=True,
            extra_args=benchy_extra,
            # When we've already done our own warmup, tell llama-benchy to
            # skip its redundant warmup phase (saves 2 extra requests).
            skip_warmup=not args.no_warmup,
        )

        if args.perf_only:
            # Write standalone throughput report
            from tool_eval_bench.utils.ids import build_run_id

            run_config = _with_config_fingerprint(
                {
                    "model": model,
                    "backend": backend,
                    "base_url": base_url,
                    "mode": "perf-only",
                }
            )
            run_id = build_run_id(run_config)
            reporter = MarkdownReporter(root=args.output_dir)
            report_path = reporter.write_throughput_report(
                run_id,
                display_name,
                throughput_samples,
                run_context=run_context,
            )
            _persist_plugin_run(
                {
                    "run_id": run_id,
                    "run_type": "perf",
                    "status": "completed",
                    "config": run_config,
                    "scores": {"samples": len(throughput_samples)},
                    "metadata": _metadata_for_storage(run_context),
                }
            )
            console.print(f"\n  [dim]Report saved to {report_path}[/]\n")
            return

    # -- Legacy built-in throughput benchmark --
    if args.perf_legacy or args.perf_legacy_only:
        depths = _parse_int_list(args.depth)
        conc_levels = _parse_int_list(args.concurrency)
        legacy_samples = _run_throughput(
            console,
            model,
            display_name,
            base_url,
            api_key,
            pp=args.pp,
            tg=args.tg,
            depths=depths,
            concurrency_levels=conc_levels,
        )
        throughput_samples.extend(legacy_samples)

        if args.perf_legacy_only:
            from tool_eval_bench.utils.ids import build_run_id

            run_config = _with_config_fingerprint(
                {
                    "model": model,
                    "backend": backend,
                    "base_url": base_url,
                    "mode": "perf-legacy-only",
                }
            )
            run_id = build_run_id(run_config)
            reporter = MarkdownReporter(root=args.output_dir)
            report_path = reporter.write_throughput_report(
                run_id,
                display_name,
                legacy_samples,
                run_context=run_context,
            )
            _persist_plugin_run(
                {
                    "run_id": run_id,
                    "run_type": "perf-legacy",
                    "status": "completed",
                    "config": run_config,
                    "scores": {"samples": len(legacy_samples)},
                    "metadata": _metadata_for_storage(run_context),
                }
            )
            console.print(f"\n  [dim]Report saved to {report_path}[/]\n")
            return

    # -- Speculative decoding / MTP benchmark --
    if args.spec_bench:
        spec_depths = _parse_int_list(args.depth)
        spec_prompts = [p.strip() for p in args.spec_prompts.split(",") if p.strip()]
        _run_spec_bench(
            console,
            model,
            display_name,
            base_url,
            api_key,
            pp=args.pp,
            tg=args.tg,
            depths=spec_depths,
            spec_method=args.spec_method,
            baseline_tg_tps=args.baseline_tgs,
            prompt_types=spec_prompts,
            metrics_url=args.metrics_url,
            output_dir=args.output_dir,
            metadata_for_storage=_metadata_for_storage,
            with_config_fingerprint=_with_config_fingerprint,
            persist_plugin_run=_persist_plugin_run,
        )
        # If --spec-bench is the only mode, or user explicitly skipped tool-eval
        if args.skip_tool_eval or (
            not args.perf
            and not args.perf_only
            and not args.gsm8k
            and not args.gsm8k_only
            and not args.mmlu
            and not args.mmlu_only
            and not args.ifeval
            and not args.ifeval_only
        ):
            return

    # -- Context pressure sweep --
    if args.context_pressure_sweep is not None:
        _run_pressure_sweep(
            console,
            model,
            display_name,
            backend,
            base_url,
            api_key,
            args,
            display_url=display_url,
            extra_params=extra_params or None,
            parse_sweep_range=_parse_sweep_range,
            resolve_scenarios=_resolve_scenarios,
            with_config_fingerprint=_with_config_fingerprint,
            persist_plugin_run=_persist_plugin_run,
            metadata_for_storage=_metadata_for_storage,
        )
        return

    # -- Context pressure --
    pressure_messages: list[dict] | None = None
    pressure_config_dict: dict | None = None
    if args.context_pressure is not None:
        from rich.progress import BarColumn, Progress, TextColumn

        from tool_eval_bench.runner.context_pressure import (
            build_pressure_messages,
            calibrate_pressure_messages,
            prepare_context_pressure,
        )

        ratio = max(0.0, min(1.0, args.context_pressure))
        try:
            pressure_cfg = asyncio.run(
                prepare_context_pressure(
                    base_url,
                    model,
                    api_key,
                    ratio=ratio,
                    context_size_override=args.context_size,
                    metrics_url=args.metrics_url,
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
                            task,
                            completed=tokens_so_far,
                        ),
                        seed=args.seed,
                    )
            else:
                pressure_messages = build_pressure_messages(
                    pressure_cfg,
                    seed=args.seed,
                )

            # Calibrate using server-side tokenizer for exact token counts
            pressure_messages, actual_fill_tokens = asyncio.run(
                calibrate_pressure_messages(
                    pressure_messages,
                    pressure_cfg.fill_tokens,
                    base_url,
                    model,
                    api_key,
                    seed=args.seed,
                )
            )

            pressure_config_dict = {
                "ratio": pressure_cfg.ratio,
                "fill_tokens": actual_fill_tokens,
                "fill_tokens_target": pressure_cfg.fill_tokens,
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

                budget = pressure_cfg.budget_breakdown(tool_tokens=tool_tokens_est)
                fill_k = pressure_cfg.fill_tokens / 1024
                tool_k = tool_tokens_est / 1024
                out_k = _RESERVED_FOR_OUTPUT / 1024
                head_k = budget["remaining_headroom_tokens"] / 1024

                console.print(
                    f"  [dim]  {pressure_cfg.summary()} — "
                    f"{len(pressure_messages or [])} filler messages[/]"
                )
                console.print(
                    f"  [dim]  Budget: [bold]{fill_k:.0f}K[/] fill │ "
                    f"~{tool_k:.0f}K tools ({num_tools} loaded) │ "
                    f"{out_k:.0f}K output │ "
                    f"{head_k:.0f}K scenario headroom[/]\n"
                )
            # Auto-scale timeout for context pressure: large fills need
            # significant prefill time.  Without this, a 182K fill at the
            # default 60s timeout will fail while the same level passes in
            # a --context-pressure-sweep (which has its own auto-scaling).
            fill_tokens_for_timeout = actual_fill_tokens or pressure_cfg.fill_tokens
            if fill_tokens_for_timeout > 0:
                fill_scaling = max(0, fill_tokens_for_timeout / 50_000) * 60.0
                scaled_timeout = max(args.timeout, 120.0 + fill_scaling)
                if scaled_timeout > args.timeout:
                    logger.info(
                        "Auto-scaling timeout from %.0fs to %.0fs for %d fill tokens",
                        args.timeout,
                        scaled_timeout,
                        fill_tokens_for_timeout,
                    )
                    args.timeout = scaled_timeout

        except ValueError as exc:
            console.print(f"\n[bold red]Error:[/] {exc}")
            sys.exit(1)

    # -- GSM8K benchmark --
    if args.gsm8k or args.gsm8k_only:
        _run_gsm8k_benchmark(
            console,
            model,
            display_name,
            base_url,
            api_key,
            args,
            extra_params=extra_params or None,
            output_dir=args.output_dir,
            run_context=run_context,
        )
        if args.gsm8k_only:
            return

    # -- MMLU benchmark --
    if args.mmlu or args.mmlu_only:
        _run_mmlu_benchmark(
            console,
            model,
            display_name,
            base_url,
            api_key,
            args,
            extra_params=extra_params or None,
            output_dir=args.output_dir,
            run_context=run_context,
        )
        if args.mmlu_only:
            return

    # -- IFEval benchmark --
    if args.ifeval or args.ifeval_only:
        _run_ifeval_benchmark(
            console,
            model,
            display_name,
            base_url,
            api_key,
            args,
            extra_params=extra_params or None,
            output_dir=args.output_dir,
            run_context=run_context,
        )
        if args.ifeval_only:
            return

    # -- Skip tool-call scenarios if requested --
    if args.skip_tool_eval:
        any_benchmark = (
            args.perf
            or args.perf_only
            or args.spec_bench
            or args.spec_live
            or args.gsm8k
            or args.gsm8k_only
            or args.mmlu
            or args.mmlu_only
            or args.ifeval
            or args.ifeval_only
        )
        if not any_benchmark:
            console.print(
                "\n  [yellow]⚠ --skip-tool-eval has no effect without "
                "--perf, --perf-only, --spec-bench, --gsm8k, --mmlu, or --ifeval.[/]\n"
            )
        return

    # -- Tool-call scenarios --
    service = BenchmarkService(
        reporter=MarkdownReporter(root=args.output_dir),
    )
    use_live = not args.json and not args.no_live
    trials = max(1, args.trials)

    # -- Resume: skip scenarios that already passed in a prior run --
    # When resuming, we reuse the original run_id and merge results after.
    resume_prior_results: list[dict] | None = None
    if args.resume:
        from tool_eval_bench.storage.db import RunRepository

        resume_repo = RunRepository()
        prev_run = resume_repo.get(args.resume)
        resume_repo.close()
        if prev_run is None:
            console.print(
                f"\n  [bold red]✗[/] Run '{args.resume}' not found in history.\n"
                "  [dim]Use --history to list available runs.[/]\n"
            )
            sys.exit(1)

        # --- B1: Validate configuration compatibility ---
        prev_config = prev_run.get("config") or {}
        prev_model = prev_config.get("model", "")
        prev_backend = prev_config.get("backend", "")
        mismatches: list[str] = []
        if prev_model and prev_model != model:
            mismatches.append(f"model ({prev_model} → {model})")
        if prev_backend and prev_backend != backend:
            mismatches.append(f"backend ({prev_backend} → {backend})")
        if mismatches:
            console.print(
                f"\n  [bold red]✗ Resume aborted: configuration mismatch[/]\n"
                f"  [dim]Prior run differs in: {', '.join(mismatches)}[/]\n"
                f"  [dim]Start a fresh run instead of resuming.[/]\n"
            )
            sys.exit(1)

        prev_results = (prev_run.get("scores") or {}).get("scenario_results", [])
        if not prev_results:
            if not args.json:
                console.print(
                    f"  [dim]ℹ No scenario results in run {args.resume} — running all.[/]"
                )
        else:
            passed_ids = {r["scenario_id"] for r in prev_results if r.get("status") == "pass"}

            # --- B5: Reject legacy passes without raw_log traces ---
            traceless = {
                r["scenario_id"]
                for r in prev_results
                if r.get("status") == "pass" and not r.get("raw_log")
            }
            if traceless and not args.json:
                console.print(
                    f"  [bold yellow]⚠[/] {len(traceless)} prior passes lack traces"
                    " — will be rerun for full-trace compliance"
                )
            # Remove traceless results from passed so they get rerun
            passed_ids -= traceless

            if not passed_ids:
                if not args.json:
                    console.print(
                        f"  [dim]ℹ No usable passed scenarios in run {args.resume}"
                        " — running all.[/]"
                    )
            else:
                # Override --scenarios to exclude already-passed IDs
                resolved = _resolve_scenarios(args)
                remaining = [s for s in resolved if s.id not in passed_ids]
                if not args.json:
                    console.print(
                        f"  [bold cyan]↻ Resume:[/] {len(passed_ids)} scenarios already passed "
                        f"in [dim]{args.resume}[/], "
                        f"running {len(remaining)} remaining"
                    )
                if not remaining:
                    console.print(
                        "\n  [bold green]✓[/] All scenarios already passed — nothing to re-run.\n"
                    )
                    return
                # Inject the filtered list as --scenarios so it flows through
                args.scenarios = [s.id for s in remaining]
                # Store prior results for post-run merge (only those with traces)
                resume_prior_results = [
                    r for r in prev_results if r.get("status") == "pass" and r.get("raw_log")
                ]
        # Store resume_run_id on args so the run_benchmark helpers can pass it
        args._resume_run_id = args.resume
    else:
        args._resume_run_id = None
    # Store prior results on args for service merge
    args._resume_prior_results = resume_prior_results

    if trials > 1 and not args.json:
        console.print(f"[dim]  Running {trials} trials for statistical measurement…[/]\n")

    if use_live:
        _run_with_live_display(
            service,
            console,
            model,
            display_name,
            backend,
            base_url,
            api_key,
            args,
            throughput_samples=throughput_samples,
            extra_params=extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config_dict,
            display_url=display_url,
            run_context=run_context,
        )
    elif args.json:
        _run_json(
            service,
            model,
            backend,
            base_url,
            api_key,
            args,
            extra_params=extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config_dict,
            run_context=run_context,
        )
    else:
        _run_plain(
            service,
            console,
            model,
            display_name,
            backend,
            base_url,
            api_key,
            args,
            throughput_samples=throughput_samples,
            extra_params=extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config_dict,
            display_url=display_url,
            run_context=run_context,
        )


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
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_resamples))

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
            cat_lines.append(
                f"    {cat_key} {cs['label']}: {cs['mean_percent']:.0f}% ± {cs['stddev_percent']:.1f}%"
            )
    if cat_lines:
        content += "\n  [bold]Categories with variance:[/]\n" + "\n".join(cat_lines)

    # Show scenarios with variance
    unstable = [(sid, st) for sid, st in agg["per_scenario"].items() if st["stddev"] > 0]
    if unstable:
        content += f"\n\n  [bold yellow]⚡ {len(unstable)} unstable scenario(s):[/]"
        for sid, st in unstable:
            pts_str = ",".join(str(p) for p in st["points"])
            content += f"\n    {sid}: {st['mean']:.1f} ± {st['stddev']:.1f}  [dim]({pts_str})[/]"

    console.print(
        Panel(
            content,
            title="[bold]📊 Trial Statistics[/]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )
    console.print()


# ---------------------------------------------------------------------------


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
    run_context: Any | None = None,
) -> None:
    """Run with Rich live display — the default visual mode."""
    from tool_eval_bench.runner.orchestrator import score_results

    scenarios = _resolve_scenarios(args)

    trials = max(1, args.trials)
    all_summaries = []

    # --- Trial 1: with live display ---
    display = BenchmarkDisplay(
        display_name, backend, display_url or base_url, scenarios, run_context=run_context
    )
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
            run_context=run_context,
            weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
            resume_run_id=getattr(args, "_resume_run_id", None),
            resume_prior_results=getattr(args, "_resume_prior_results", None),
            **callbacks,
        )

    async def run_all_trials() -> None:
        """Run all trials in a single event loop for connection reuse."""
        result = await run_trial(show=True)

        # When resuming, the service has already merged prior results into
        # result["scores"].  Use that merged summary for display instead of
        # re-scoring only the rerun subset from display.results (which would
        # show an inflated score — e.g. 100% from 5/5 reruns when the full
        # set was 50% on 35/69).
        has_resume = bool(getattr(args, "_resume_prior_results", None))
        merged_scores = result.get("scores", {}) if has_resume else None

        if merged_scores and has_resume:
            # Reconstruct full summary from the merged service result
            from tool_eval_bench.domain.scenarios import (
                ScenarioResult as _SR,
            )

            merged_sr = [
                _SR.from_dict(sr_dict) for sr_dict in merged_scores.get("scenario_results", [])
            ]
            merged_scenario_defs = _resolve_all_scenarios_for_ids(
                [sr.scenario_id for sr in merged_sr]
            )
            summary = score_results(
                merged_sr,
                merged_scenario_defs,
                alpha=args.alpha,
                weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
            )
            all_summaries.append(summary)
            display.set_finished(summary, throughput_samples=throughput_samples)
        else:
            all_results = [display.results[s.id] for s in scenarios if s.id in display.results]
            if all_results:
                summary = score_results(
                    all_results,
                    scenarios,
                    alpha=args.alpha,
                    weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
                )
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
                # Reconstruct ScenarioResult objects from the persisted dict
                trial_sr = [ScenarioResult.from_dict(sr_dict) for sr_dict in trial_score_results]
                if trial_sr:
                    trial_summary = score_results(
                        trial_sr,
                        scenarios,
                        alpha=args.alpha,
                        weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
                    )
                    all_summaries.append(trial_summary)
                    console.print(f"[bold]{trial_summary.final_score}[/]/100")

            agg = _aggregate_trials(all_summaries)
            _print_trials_summary(console, agg)

            # Write consolidated summary report
            if agg and len(all_summaries) > 1:
                reporter = MarkdownReporter(root=args.output_dir)
                run_id_base = result.get("run_id", "summary")
                throughput = result.get("throughput_samples")
                summary_path = reporter.write_summary_report(
                    run_id=run_id_base,
                    model=display_name,
                    summaries=all_summaries,
                    agg=agg,
                    throughput_samples=throughput,
                    report_paths=report_paths,
                    run_context=run_context,
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


# ---------------------------------------------------------------------------
# JSONL progress callbacks for headless mode (sparkrun integration)
# ---------------------------------------------------------------------------


async def _stderr_progress_start(
    scenario: ScenarioDefinition,
    idx: int,
    total: int,
) -> None:
    """Emit a JSONL progress event to stderr when a scenario starts."""
    msg = {
        "event": "scenario_start",
        "scenario_id": scenario.id,
        "title": scenario.title,
        "category": scenario.category.value,
        "index": idx,
        "total": total,
    }
    sys.stderr.write(json.dumps(msg) + "\n")
    sys.stderr.flush()


async def _stderr_progress_result(
    scenario: ScenarioDefinition,
    result: ScenarioResult,
    idx: int,
    total: int,
) -> None:
    """Emit a JSONL progress event to stderr when a scenario completes."""
    msg = {
        "event": "scenario_result",
        "scenario_id": scenario.id,
        "status": result.status.value,
        "points": result.points,
        "index": idx,
        "total": total,
        "duration_seconds": round(result.duration_seconds, 2),
    }
    sys.stderr.write(json.dumps(msg) + "\n")
    sys.stderr.flush()


def _emit_json_output(data: dict[str, Any], *, json_file: str | None = None) -> None:
    """Write versioned JSON output to stdout or a file.

    When *json_file* is set, the JSON is written to that path (keeps stdout
    clean for sparkrun / subprocess consumers).  Otherwise it goes to stdout.
    """
    from tool_eval_bench.api import format_result

    envelope = format_result(data)
    text = json.dumps(envelope, indent=2, default=str)

    if json_file:
        from pathlib import Path

        out = Path(json_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        # Also emit a final JSONL event so callers know where to find it
        msg = {
            "event": "benchmark_complete",
            "json_file": str(out),
            "final_score": envelope.get("final_score"),
        }
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
    else:
        print(text)


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
    run_context: Any | None = None,
) -> None:
    """Run and output raw JSON (with optional JSONL progress on stderr)."""
    trials = max(1, args.trials)
    resolved = _resolve_scenarios(args)
    json_file = getattr(args, "json_file", None)

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
            run_context=run_context,
            weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
            resume_run_id=getattr(args, "_resume_run_id", None),
            resume_prior_results=getattr(args, "_resume_prior_results", None),
            on_scenario_start=_stderr_progress_start,
            on_scenario_result=_stderr_progress_result,
        )

    try:
        results = []
        for _t in range(trials):
            results.append(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        error_data = {"error": str(exc)}
        _emit_json_output(error_data, json_file=json_file)
        sys.exit(1)

    if trials == 1:
        _emit_json_output(results[0], json_file=json_file)
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
        _emit_json_output(output, json_file=json_file)


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
    run_context: Any | None = None,
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
            run_context=run_context,
            weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
            resume_run_id=getattr(args, "_resume_run_id", None),
            resume_prior_results=getattr(args, "_resume_prior_results", None),
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
    console.print(
        f"\n[bold]Score: {scores.get('final_score', 0)} / 100  — {scores.get('rating', '')}[/]"
    )
    if scores.get("weighted_score") is not None:
        console.print(
            f"[bold]Weighted Score: {scores['weighted_score']} / 100[/]  [dim](difficulty-weighted)[/]"
        )
    console.print(f"[dim]Completed in {elapsed:.1f}s[/]\n")

    # Show trial statistics if multiple trials
    if trials > 1:
        from tool_eval_bench.runner.orchestrator import score_results

        resolved_sc = _resolve_scenarios(args)
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
                summaries.append(score_results(trial_sr, resolved_sc, alpha=args.alpha))
        agg = _aggregate_trials(summaries) if summaries else {}
        _print_trials_summary(console, agg)

        if agg and len(summaries) > 1:
            reporter = MarkdownReporter(root=args.output_dir)
            run_id_base = (
                all_results_dicts[0].get("run_id", "summary") if all_results_dicts else "summary"
            )
            rp_list = [
                str(r.get("report_path", "")) for r in all_results_dicts if r.get("report_path")
            ]
            summary_path = reporter.write_summary_report(
                run_id=run_id_base,
                model=display_name,
                summaries=summaries,
                agg=agg,
                report_paths=rp_list,
                run_context=run_context,
            )
            console.print(f"  [dim]📊 Summary report: {summary_path}[/]\n")


if __name__ == "__main__":
    main()
