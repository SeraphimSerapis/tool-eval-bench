"""CLI dispatch and compatibility implementation for running benchmarks.

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
from typing import Any

from dotenv import load_dotenv  # noqa: F401  (re-exported via _load_dotenv)
from rich.console import Console

from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import model_probe as _model_probe
from tool_eval_bench.cli.compare_report import (
    run_compare_report_command as _run_compare_report_command,
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
    persist_plugin_run as _persist_plugin_run,
)
from tool_eval_bench.cli.helpers import prior_results_for_resume
from tool_eval_bench.cli.helpers import safety_gate_failed as _safety_gate_failed
from tool_eval_bench.cli.history import compare_runs as _compare_runs
from tool_eval_bench.cli.history import (
    print_diff as _print_diff,
)
from tool_eval_bench.cli.history import print_history as _print_history
from tool_eval_bench.cli.leaderboard import export_runs as _export_runs
from tool_eval_bench.cli.leaderboard import print_leaderboard as _print_leaderboard
from tool_eval_bench.cli.legacy_parser import _make_parser  # noqa: F401
from tool_eval_bench.cli.local_commands import handle_local_command as _handle_local_command
from tool_eval_bench.cli.perf import (
    run_llama_benchy as _run_llama_benchy,
)
from tool_eval_bench.cli.perf import (
    run_throughput as _run_throughput,
)
from tool_eval_bench.cli.plugin_runners import (
    _run_gsm8k_benchmark,
    _run_ifeval_benchmark,
    _run_mmlu_benchmark,
)
from tool_eval_bench.cli.pressure import (
    run_pressure_sweep as _run_pressure_sweep,
)
from tool_eval_bench.cli.probe import preflight_model_check as _preflight_model_check
from tool_eval_bench.cli.probe import warmup_server as _do_warmup
from tool_eval_bench.cli.resolve import (
    parse_int_list as _parse_int_list,
)
from tool_eval_bench.cli.resolve import (
    parse_sweep_range as _parse_sweep_range,
)
from tool_eval_bench.cli.resolve import (
    redact_url as _redact_url,
)
from tool_eval_bench.cli.resolve import (
    resolve_all_scenarios_for_ids as _resolve_all_scenarios_for_ids,
)
from tool_eval_bench.cli.resolve import (
    resolve_packs as _resolve_packs,
)
from tool_eval_bench.cli.resolve import (
    resolve_scenarios as _resolve_scenarios,
)
from tool_eval_bench.cli.resolve import (
    with_config_fingerprint as _with_config_fingerprint,
)
from tool_eval_bench.cli.run_io import aggregate_trials as _aggregate_trials
from tool_eval_bench.cli.run_io import bootstrap_ci as _bootstrap_ci  # noqa: F401
from tool_eval_bench.cli.run_io import emit_json_output as _emit_json_output
from tool_eval_bench.cli.run_io import median as _median  # noqa: F401
from tool_eval_bench.cli.run_io import stderr_progress_result as _stderr_progress_result
from tool_eval_bench.cli.run_io import stderr_progress_start as _stderr_progress_start
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
    NO_SERVER,
)
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.storage.reports import MarkdownReporter

logger = logging.getLogger(__name__)

# Valid category letters for --categories
_VALID_CATEGORIES = {c.value for c in Category}


def _pack_attestations(args: Any) -> list[dict[str, Any]] | None:
    """Content-hash records for any held-out packs in this run, or None.

    Recorded in the run config so a published score can be tied to a specific
    held-out set without publishing the set itself.
    """
    packs = _resolve_packs(args)
    return [pack.to_dict() for pack in packs] or None


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
    """Compatibility wrapper preserving the historical asyncio patch seam."""
    _model_probe.asyncio = asyncio
    return _model_probe._detect_model(
        base_url,
        api_key,
        console,
        display_url=display_url,
        headless=headless,
    )


def _probe_server(
    console: Console,
    base_url: str,
    api_key: str | None,
    *,
    headless: bool = False,
) -> None:
    """Compatibility wrapper preserving the historical asyncio patch seam."""
    _model_probe.asyncio = asyncio
    _model_probe._probe_server(console, base_url, api_key, headless=headless)


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


# ---------------------------------------------------------------------------
# History and diff (extracted to cli/history.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GSM8K benchmark (--gsm8k / --gsm8k-only)
# ---------------------------------------------------------------------------


# Set of argument dest names that are intentionally suppressed (not in ARGS_SCHEMA).
# Used by the drift-detection test in tests/test_api.py.
_HIDDEN_ARGS: frozenset[str] = frozenset({"command", "help"})


def main() -> None:
    _load_dotenv()
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.parser import parse_cli_args

    parser, args = parse_cli_args(make_parser)

    console = Console()

    if getattr(args, "command", None) == "compare-report":
        _run_compare_report_command(args, console)
        return

    # --json-file implies --json
    if args.json_file:
        args.json = True

    if _handle_local_command(
        args,
        console,
        resolve_scenarios=_resolve_scenarios,
        print_history=_print_history,
        print_leaderboard=_print_leaderboard,
        export_runs=_export_runs,
        compare_runs=_compare_runs,
    ):
        return

    # Cascade: CLI flag → env var → auto-discovery
    model = args.model or os.getenv("TOOL_EVAL_MODEL") or None
    backend = args.backend or os.getenv("TOOL_EVAL_BACKEND", "")
    backend_explicit = bool(backend)
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

    # Authoritative backend detection: a port-based guess (from localhost
    # auto-discovery, above) or a hardcoded default both get overridden here
    # by actually asking the server what it is — via its /metrics namespace,
    # llama.cpp's /props /health fingerprint, or vLLM's /version endpoint.
    # This is what tells a remote llama.cpp box apart from vLLM/SGLang instead
    # of silently defaulting to "vllm". Skipped when the user pinned --backend
    # / TOOL_EVAL_BACKEND explicitly, or opted out of engine probing.
    if not backend_explicit and base_url and not args.probe and not args.no_probe_engine:
        from tool_eval_bench.utils.metadata import probe_backend_hint as _probe_backend_hint

        hint = asyncio.run(_probe_backend_hint(base_url, api_key))
        if hint:
            backend, backend_label = hint
            if args.json:
                sys.stderr.write(
                    json.dumps({"event": "backend_detected", "backend": backend}) + "\n"
                )
                sys.stderr.flush()
            else:
                console.print(f"[dim]  Detected backend: {backend_label}[/]")

    # Default backend if still unset (detection above was inconclusive, or skipped)
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

    # -- Validate --scenario-pack / --pack-only --
    # Load packs up front: a missing, empty, or colliding pack must abort before
    # the run starts, not surface as a traceback partway through resolution.
    try:
        packs = _resolve_packs(args)
        _resolve_scenarios(args)
    except ValueError as exc:
        parser.error(str(exc))
    if packs and not args.json:
        total = sum(len(p.scenarios) for p in packs)
        names = ", ".join(f"{p.name} ({p.content_hash})" for p in packs)
        console.print(f"  [dim]🔒 Held-out packs: {names} — {total} scenario(s)[/]")

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
    if not args.no_preflight:
        _preflight_model_check(
            console,
            base_url,
            model,
            api_key,
            headless=args.json,
            timeout_seconds=args.timeout,
            temperature=args.temperature,
            extra_params=extra_params or None,
        )

    # -- Warm-up --
    if not args.no_warmup and not args.json:
        _do_warmup(console, base_url, model, api_key)

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
            tokenizer=getattr(args, "tokenizer", None),
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
    pressure_messages: list[ChatMessage] | None = None
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

    # -- External benchmark plugins --
    from tool_eval_bench.cli.plugin_runners import run_selected_plugins

    if run_selected_plugins(
        console,
        model,
        display_name,
        base_url,
        api_key,
        args,
        runners={
            "gsm8k": _run_gsm8k_benchmark,
            "mmlu": _run_mmlu_benchmark,
            "ifeval": _run_ifeval_benchmark,
        },
        extra_params=extra_params or None,
        output_dir=args.output_dir,
        run_context=run_context,
    ):
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
        prev_checkpoints = resume_repo.get_checkpoints(args.resume) if prev_run else []
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

        prev_results = prior_results_for_resume(prev_run, prev_checkpoints)
        if prev_checkpoints and not args.json:
            console.print(
                f"  [dim]ℹ Recovered {len(prev_checkpoints)} checkpointed scenario(s) from "
                f"interrupted run {args.resume}.[/]"
            )
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
    context_pressure_messages: list[ChatMessage] | None = None,
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
            scenario_packs=_pack_attestations(args),
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
        if _safety_gate_failed(args, result):
            raise SystemExit(2)

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


def _run_json(
    service: BenchmarkService,
    model: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[ChatMessage] | None = None,
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
            scenario_packs=_pack_attestations(args),
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
        if _safety_gate_failed(args, results[0]):
            raise SystemExit(2)
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
        if _safety_gate_failed(args, output):
            raise SystemExit(2)


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
    context_pressure_messages: list[ChatMessage] | None = None,
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
            scenario_packs=_pack_attestations(args),
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
    if _safety_gate_failed(args, all_results_dicts[-1]):
        raise SystemExit(2)

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
