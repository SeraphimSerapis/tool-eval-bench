"""Endpoint discovery, target resolution, preflight gating, and context-pressure setup."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.adapters.wire_format import resolve_wire_format as _resolve_wire_format
from tool_eval_bench.application.reviewer import mask_credentials_in_adapter_logs, reviewer_origin
from tool_eval_bench.application.run_context import build_run_context, identify_backend
from tool_eval_bench.cli.arg_validation import (
    _build_extra_params,
    _scenario_selector_label,
    _validate_scenario_selection,
)
from tool_eval_bench.cli.headless import report_run_failed
from tool_eval_bench.cli.helpers import emit_headless_error as _headless_error
from tool_eval_bench.cli.modes import any_plugin_selected as _any_plugin_selected
from tool_eval_bench.cli.probe import preflight_model_check as _preflight_model_check
from tool_eval_bench.cli.probe import warmup_server as _do_warmup
from tool_eval_bench.cli.provider_env import resolve_provider as _resolve_provider
from tool_eval_bench.cli.resolve import redact_url as _redact_url
from tool_eval_bench.cli.resolve import resolve_scenarios as _resolve_scenarios
from tool_eval_bench.cli.reviewer import (
    build_reviewer_settings,
    connection_api_key,
    reviewer_requested,
    same_server,
)
from tool_eval_bench.cli.scored_run import DECISION_JUDGE_API_KEY_ENV, DECISION_JUDGE_MODEL_ENV
from tool_eval_bench.cli.server import DISCOVERY_PORTS as _DISCOVERY_PORTS
from tool_eval_bench.cli.server import discover_server as _discover_server
from tool_eval_bench.domain.errors import NO_SERVER
from tool_eval_bench.domain.models import ChatMessage, RunContext
from tool_eval_bench.domain.review import ReviewerSettings, without_auth_headers
from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.runner.perf_presets import PerfPlan, failed_cell_count
from tool_eval_bench.utils.headers import attach_session_id as _attach_session_id
from tool_eval_bench.utils.headers import merge_headers as _merge_headers
from tool_eval_bench.utils.headers import parse_header_env as _parse_header_env
from tool_eval_bench.utils.headers import parse_header_pairs as _parse_header_pairs

if TYPE_CHECKING:
    from tool_eval_bench.runner.throughput import ThroughputSample

logger = logging.getLogger("tool_eval_bench.cli.dispatch")


@dataclass(frozen=True)
class _PerfSweep:
    """What :func:`tool_eval_bench.cli.perf_mode._run_perf_plan` measured.

    ``plan`` holds only the passes that ran, the one that failed included.
    ``incomplete`` means a planned pass was skipped, failed, or interrupted;
    ``interrupted`` means Ctrl+C stopped a later pass.
    """

    samples: list[ThroughputSample]
    plan: PerfPlan
    incomplete: bool = False
    interrupted: bool = False

    @property
    def failed_cells(self) -> int:
        return failed_cell_count(self.samples)


@dataclass(frozen=True)
class _Endpoint:
    """The connection ``main`` resolved before a model is chosen."""

    base_url: str
    api_key: str | None
    backend: str
    wire_format: str
    model: str | None
    # Pre-flight requests are single-shot conversations of their own.
    probe_headers: Mapping[str, str]


@dataclass
class _Target:
    """Everything the mode handlers need about a resolved endpoint.

    ``main`` resolves the endpoint once (URL cascade, wire format, backend
    probe, model detection) and each mode reads it from here, rather than
    threading a dozen locals through every branch.
    """

    args: argparse.Namespace
    parser: argparse.ArgumentParser
    console: Console
    model: str
    display_name: str
    backend: str
    base_url: str
    display_url: str
    api_key: str | None
    wire_format: str
    extra_params: dict[str, Any]
    run_context: RunContext | None
    #: What the throughput sweep ran, set by ``_run_throughput_mode``. Its
    #: plan leaves out a skipped depth pass, so the stored config never claims
    #: a grid that was not attempted.
    perf_sweep: _PerfSweep | None = None


@dataclass(frozen=True)
class _PressureFill:
    """Context-pressure filler for the scored run, and the config it records."""

    messages: list[ChatMessage]
    config: dict[str, Any]


def _resolve_endpoint(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    console: Console,
    *,
    discover_server_fn: Callable[..., tuple[str, str] | None] | None = None,
) -> _Endpoint:
    """Resolve the server, credentials, headers, wire format, and backend label."""
    # Cascade: CLI flag → provider-scoped env → generic env → auto-discovery.
    # A selected provider owns its api_key and model: the generic
    # TOOL_EVAL_API_KEY is for whatever the generic base URL points at, and
    # sending it to a vendor API would just muddy a failed-auth diagnosis.
    discover = discover_server_fn if discover_server_fn is not None else _discover_server
    try:
        provider = _resolve_provider(args.provider or os.getenv("TOOL_EVAL_PROVIDER"), os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        if provider is not None:
            model = args.model or provider.model
            backend = (
                args.backend or os.getenv("TOOL_EVAL_BACKEND", "") or provider.backend_label or ""
            )
            base_url = args.base_url or provider.base_url
            api_key = args.api_key or provider.api_key
            env_headers = provider.headers
            session_header = args.session_header or provider.session_header
        else:
            model = args.model or os.getenv("TOOL_EVAL_MODEL") or None
            backend = args.backend or os.getenv("TOOL_EVAL_BACKEND", "")
            base_url = args.base_url or os.getenv("TOOL_EVAL_BASE_URL", "")
            api_key = args.api_key or os.getenv("TOOL_EVAL_API_KEY")
            env_headers = _parse_header_env(os.getenv("TOOL_EVAL_HEADERS"))
            session_header = args.session_header or os.getenv("TOOL_EVAL_SESSION_HEADER") or None
        # Headers merge rather than replace: a --header adds to the provider's.
        request_headers = _merge_headers(env_headers, _parse_header_pairs(args.header))
    except ValueError as exc:
        parser.error(str(exc))
    backend_explicit = bool(backend)
    # Left on the namespace so the plugin and pressure runners, which take
    # ``args`` rather than the resolved connection, build their adapters the
    # same way (see cli.helpers.adapter_options).
    args._request_headers = request_headers
    args._session_header = session_header
    # Pre-flight requests are single-shot conversations of their own.
    probe_headers = _attach_session_id(request_headers, session_header)

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
        discovered = discover(headless=args.json, console=console, redact=args.redact_url)
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

    # Which request format the endpoint speaks (native Gemini vs OpenAI-compatible).
    # Detected from the URL unless --format pins it.
    try:
        wire_format = _resolve_wire_format(args.format, base_url)
    except ValueError as exc:
        parser.error(str(exc))
    # An explicit label (flag, env, or provider) is kept, including "unknown".
    # A label from local discovery is only the fallback for an inconclusive probe.
    detection = asyncio.run(
        identify_backend(
            backend if backend_explicit else None,
            base_url=base_url,
            api_key=api_key,
            wire_format=wire_format,
            probe=bool(base_url) and not args.probe and not args.no_probe_engine,
            fallback=backend or "unknown",
        )
    )
    backend = detection.backend
    if detection.server_name:
        if args.json:
            sys.stderr.write(json.dumps({"event": "backend_detected", "backend": backend}) + "\n")
            sys.stderr.flush()
        else:
            console.print(f"[dim]  Detected backend: {escape(detection.server_name)}[/]")

    return _Endpoint(
        base_url=base_url,
        api_key=api_key,
        backend=backend,
        wire_format=wire_format,
        model=model,
        probe_headers=probe_headers,
    )


def _detect_decision_judge_model(
    args: argparse.Namespace,
    console: Console,
    *,
    detect_model_fn: Callable[..., tuple[str, str]],
) -> None:
    """Ask the judge server for its model when no flag or env var named one.

    A single served model is taken. Several open the interactive picker, and
    stop a ``--json`` run rather than guess, since the judge model is recorded.
    """
    judge_url = getattr(args, "decision_judge_base_url", None)
    if judge_url is None or getattr(args, "decision_judge_model", None) is not None:
        return
    # The judge host is masked wherever it is printed or stored.
    display_url = _redact_url(judge_url)
    if not args.json:
        console.print(f"[dim]  Decision judge: {escape(display_url)}[/]")
    model, _ = detect_model_fn(
        judge_url,
        os.environ.get(DECISION_JUDGE_API_KEY_ENV),
        console,
        display_url=display_url,
        headless=args.json,
        multiple_models_hint=(
            f"pass --decision-judge-model or set {DECISION_JUDGE_MODEL_ENV} to choose one"
        ),
    )
    args.decision_judge_model = model


def _resolve_reviewer(
    args: argparse.Namespace,
    console: Console,
    endpoint: _Endpoint,
    extra_params: dict[str, Any],
    *,
    detect_model_fn: Callable[..., tuple[str, str]],
) -> None:
    """Leave the reviewer on ``args._reviewer``, asking its server for the model if needed.

    A single served model is taken. Several open the interactive picker, and
    stop a ``--json`` run rather than guess, since the reviewer is recorded.
    """
    args._reviewer = None
    if not reviewer_requested(args):
        return
    connection: dict[str, Any] = args._reviewer_connection
    reviewer_url: str | None = connection["base_url"]
    model: str | None = connection["model"]
    # The origin only: some gateways authenticate by a path segment.
    display_url = reviewer_origin(reviewer_url) if reviewer_url else "(benchmarked server)"
    if model is None and reviewer_url is not None:
        # Discovery authenticates exactly as the reviewer's requests will.
        shared = same_server(reviewer_url, endpoint.base_url)
        discovery_key = connection_api_key(connection, endpoint.base_url, endpoint.api_key)
        headers = connection["headers"]
        if headers is None and shared:
            inherited = getattr(args, "_request_headers", None)
            # Discovery authenticates as the chat requests will: a reviewer
            # with its own key drops the run's Authorization header.
            headers = (
                inherited if discovery_key == endpoint.api_key else without_auth_headers(inherited)
            )
        session_header = connection["session_header"] or (
            getattr(args, "_session_header", None) if shared else None
        )
        if session_header:
            headers = _attach_session_id(headers or {}, session_header)
        # HTTPX logs the discovery URL, query string included: mask what the
        # reviewer's URL, key, and headers carry while it runs.
        unmask = mask_credentials_in_adapter_logs(
            ReviewerSettings(
                model="(discovery)",
                base_url=reviewer_url,
                api_key=discovery_key,
                extra_headers=headers,
            ),
            api_key=endpoint.api_key,
            headers=getattr(args, "_request_headers", None),
        )
        try:
            model, _ = detect_model_fn(
                reviewer_url,
                discovery_key,
                console,
                display_url=display_url,
                headless=args.json,
                wire_format=_resolve_wire_format(
                    getattr(args, "reviewer_format", None), reviewer_url
                ),
                headers=headers,
                multiple_models_hint=(
                    "pass --reviewer-model or set TOOL_EVAL_REVIEWER_MODEL to choose one"
                ),
                show_body=False,
            )
        finally:
            unmask()
    if model is None:  # prepare_reviewer_args rejects a reviewer with neither.
        return
    args._reviewer = build_reviewer_settings(
        args,
        model=model,
        main_base_url=endpoint.base_url,
        main_api_key=endpoint.api_key,
        main_headers=getattr(args, "_request_headers", None),
        main_session_header=getattr(args, "_session_header", None),
        main_extra_params=extra_params,
    )
    if not args.json:
        console.print(f"[dim]  Reviewer: {escape(model)} @ {escape(display_url)}[/]")
        if _any_plugin_selected(args):
            console.print(
                "[dim]  The reviewer applies to the tool-call scenarios; plugin "
                "benchmarks run without it.[/]"
            )


def _resolve_target(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    console: Console,
    endpoint: _Endpoint,
    *,
    detect_model_fn: Callable[..., tuple[str, str]],
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
) -> _Target:
    """Pick the model and request parameters; the run context is attached later."""
    base_url, api_key, wire_format = endpoint.base_url, endpoint.api_key, endpoint.wire_format

    # URL redaction for display (actual API calls use real base_url)
    display_url = _redact_url(base_url) if args.redact_url else base_url

    # Auto-detect model if not provided
    model = endpoint.model
    display_name: str | None = None
    if not model:
        if not args.json:
            console.print("\n[bold]🔧 Tool-Call Benchmark[/]")
            console.print(f"[dim]  Server: {escape(display_url)}[/]")
        model, display_name = detect_model_fn(
            base_url,
            api_key,
            console,
            display_url=display_url,
            headless=args.json,
            wire_format=wire_format,
            headers=endpoint.probe_headers,
        )
        if not args.json:
            console.print()

    _detect_decision_judge_model(args, console, detect_model_fn=detect_model_fn)

    # display_name is the human-readable model (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")
    # model is the API alias (e.g. "gemma4") — used in all API calls
    display_name = display_name or model

    extra_params = _build_extra_params(args, parser)
    _resolve_reviewer(args, console, endpoint, extra_params, detect_model_fn=detect_model_fn)

    _validate_scenario_selection(args, parser, console, resolve_scenarios_fn=resolve_scenarios_fn)

    return _Target(
        args=args,
        parser=parser,
        console=console,
        model=model,
        display_name=display_name,
        backend=endpoint.backend,
        base_url=base_url,
        display_url=display_url,
        api_key=api_key,
        wire_format=wire_format,
        extra_params=extra_params,
        run_context=None,
    )


def _check_endpoint_ready(
    args: argparse.Namespace,
    console: Console,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    wire_format: str,
    extra_params: dict[str, Any],
    headers: Mapping[str, str] | None = None,
    display_url: str | None = None,
    preflight_fn: Callable[..., None] | None = None,
    warmup_fn: Callable[..., None] | None = None,
) -> None:
    """Verify the model answers a real request, then warm the server.

    Some servers list a model in ``/v1/models`` and still fail the first real
    request.  Without this gate the benchmark scores the failure as the model's.
    """
    preflight = preflight_fn if preflight_fn is not None else _preflight_model_check
    warmup = warmup_fn if warmup_fn is not None else _do_warmup
    if not args.no_preflight:
        preflight(
            console,
            base_url,
            model,
            api_key,
            headless=args.json,
            wire_format=wire_format,
            timeout_seconds=args.timeout,
            temperature=args.temperature,
            extra_params=extra_params or None,
            headers=headers,
            display_url=display_url,
        )

    if not args.no_warmup and not args.json:
        warmup(
            console,
            base_url,
            model,
            api_key,
            wire_format=wire_format,
            temperature=args.temperature,
            extra_params=extra_params or None,
            headers=headers,
            display_url=display_url,
        )


def _build_run_context(
    args: argparse.Namespace,
    console: Console,
    *,
    model: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    extra_params: dict[str, Any],
    build_run_context_fn: Callable[..., Any] | None = None,
    scenario_selector_label_fn: Callable[[argparse.Namespace], str] | None = None,
) -> RunContext | None:
    """Collect execution-context metadata for the report.

    Built before the mode branches so the throughput and spec-decode paths get
    engine detection too.  Failure is not fatal: the run proceeds with a report
    that lacks the context block.
    """
    context_builder = (
        build_run_context_fn if build_run_context_fn is not None else build_run_context
    )
    selector_label = (
        scenario_selector_label_fn
        if scenario_selector_label_fn is not None
        else _scenario_selector_label
    )
    run_context = asyncio.run(
        context_builder(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout,
            seed=args.seed,
            scenario_selector=selector_label(args),
            trials=max(1, args.trials),
            parallel=args.parallel,
            error_rate=args.error_rate,
            extra_params=extra_params,
            context_pressure=args.context_pressure,
            system_prompt=getattr(args, "system_prompt", None),
            label=args.label,
            probe_engine=not args.no_probe_engine,
        )
    )
    if run_context is not None and not args.json and run_context.engine_name:
        engine_str = run_context.engine_name
        if run_context.engine_version:
            engine_str += f" {run_context.engine_version}"
        console.print(f"  [dim]🔍 Engine: {escape(engine_str)}[/]")
    session_id = getattr(args, "_session_id", None)
    if run_context is not None and session_id:
        # Every mode persists ``run_context.to_dict()``, so this one field is
        # what tags each stored row of the invocation with its session.
        run_context = replace(run_context, session_id=session_id)
    return run_context


def _ready_target(
    target: _Target,
    probe_headers: Mapping[str, str],
    *,
    check_endpoint_ready_fn: Callable[..., None] | None = None,
    build_run_context_fn: Callable[..., RunContext | None] | None = None,
) -> _Target:
    """Gate on a model that answers, then attach the run context every mode reports."""
    args, console = target.args, target.console
    check_ready = (
        check_endpoint_ready_fn if check_endpoint_ready_fn is not None else _check_endpoint_ready
    )
    build_ctx = build_run_context_fn if build_run_context_fn is not None else _build_run_context
    # A decision model has no chat endpoint to preflight or warm; the plugin
    # reports an unreachable or unsupported endpoint itself.
    if not args.decision_bench_only:
        check_ready(
            args,
            console,
            base_url=target.base_url,
            model=target.model,
            api_key=target.api_key,
            wire_format=target.wire_format,
            extra_params=target.extra_params,
            headers=probe_headers,
            display_url=target.display_url,
        )

    run_context = build_ctx(
        args,
        console,
        model=target.model,
        backend=target.backend,
        base_url=target.base_url,
        api_key=target.api_key,
        extra_params=target.extra_params,
    )
    return replace(target, run_context=run_context)


def _prepare_context_pressure(
    target: _Target,
    *,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
) -> _PressureFill | None:
    """Build the ``--context-pressure`` or ``--context-fill-tokens`` filler.

    Scales ``args.timeout`` for a large fill; that must happen before the
    resume check and the run read it.
    """
    args, console = target.args, target.console
    base_url, model, api_key = target.base_url, target.model, target.api_key
    fill_tokens_requested = getattr(args, "context_fill_tokens", None)
    if args.context_pressure is None and fill_tokens_requested is None:
        return None

    from rich.progress import BarColumn, Progress, TextColumn

    from tool_eval_bench.runner.context_pressure import (
        _RESERVED_FOR_OUTPUT,
        _RESERVED_FOR_SCENARIO,
        build_pressure_messages,
        calibrate_pressure_messages,
        prepare_context_pressure,
        reported_context_window,
    )

    resolver = resolve_scenarios_fn if resolve_scenarios_fn is not None else _resolve_scenarios
    ratio = max(0.0, min(1.0, args.context_pressure or 0.0))
    try:
        pressure_cfg = asyncio.run(
            prepare_context_pressure(
                base_url,
                model,
                api_key,
                ratio=ratio,
                context_size_override=args.context_size,
                metrics_url=args.metrics_url,
                client_factory=HTTPMeasurementClient,
                reported_context=reported_context_window(target.run_context),
                fill_tokens=fill_tokens_requested,
            )
        )
        if ratio > 0 and pressure_cfg.fill_tokens == 0:
            # A light ratio may legitimately give a small fill; none at all
            # means the window cannot hold the reserve or the ratio rounds to
            # nothing, and the run would be stored as pressured while the
            # model saw no filler.
            raise ValueError(
                f"--context-pressure {ratio:g} gives no filler in a "
                f"{pressure_cfg.detected_context:,}-token context window: "
                f"{_RESERVED_FOR_OUTPUT + _RESERVED_FOR_SCENARIO:,} tokens are reserved for "
                "output and the scenario, so either the window is too small or the ratio "
                "too low. Check --context-size; on llama.cpp the window is per slot (-c "
                "divided by --parallel)."
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
        estimated: list[bool] = []
        pressure_messages, actual_fill_tokens = asyncio.run(
            calibrate_pressure_messages(
                pressure_messages,
                pressure_cfg.fill_tokens,
                base_url,
                model,
                api_key,
                client_factory=HTTPMeasurementClient,
                seed=args.seed,
                on_estimated=lambda: estimated.append(True),
            )
        )

        pressure_config_dict: dict[str, Any] = {
            "ratio": pressure_cfg.ratio,
            "fill_tokens": actual_fill_tokens,
            "fill_tokens_target": pressure_cfg.fill_tokens,
            "context_size": pressure_cfg.detected_context,
        }
        if fill_tokens_requested is not None:
            pressure_config_dict["fill_tokens_requested"] = fill_tokens_requested
        if estimated:
            # No usable /tokenize: fill_tokens is the 4 chars/token estimate,
            # which runs well above real counts on common tokenizers.
            pressure_config_dict["fill_tokens_estimated"] = True
        if not args.json:
            # Compute tool token estimate for selected scenarios
            from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

            selected_sc = resolver(args)

            max_toolset = UNIVERSAL_TOOLS
            for s in selected_sc:
                if s.tools_override and len(s.tools_override) > len(max_toolset):
                    max_toolset = s.tools_override
            tool_tokens_est = len(json.dumps(max_toolset)) // 4
            num_tools = len(max_toolset)

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
                f"{head_k:.0f}K scenario headroom[/]"
            )
            console.print(
                "  [dim]  An unscored warm-up request prefills the filler once so every "
                "scenario starts from the server's prefix cache[/]\n"
            )
        # Auto-scale timeout for context pressure: large fills need
        # significant prefill time.  Without this, a 182K fill at the
        # default 120s timeout will fail while the same level passes in
        # a --context-pressure-sweep (which has its own auto-scaling).
        # Scaled from the target, not the calibrated count: the timeout is
        # persisted, fingerprinted, and checked on resume, and unseeded
        # filler calibrates to a slightly different count every run.
        if pressure_cfg.fill_tokens > 0:
            fill_scaling = pressure_cfg.fill_tokens / 50_000 * 60.0
            scaled_timeout = max(args.timeout, 120.0 + fill_scaling)
            if scaled_timeout > args.timeout:
                logger.info(
                    "Auto-scaling timeout from %.0fs to %.0fs for %d fill tokens",
                    args.timeout,
                    scaled_timeout,
                    pressure_cfg.fill_tokens,
                )
                args.timeout = scaled_timeout

    except ValueError as exc:
        report_run_failed(console, f"\n[bold red]Error:[/] {escape(str(exc))}")
        sys.exit(1)
    return _PressureFill(messages=pressure_messages, config=pressure_config_dict)
