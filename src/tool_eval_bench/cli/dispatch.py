"""CLI dispatch router and compatibility surface for running benchmarks.

Defaults cascade:  .env file → TOOL_EVAL_* env vars → hardcoded fallbacks.
With --provider NAME (or TOOL_EVAL_PROVIDER), the TOOL_EVAL_<NAME>_* triple
replaces the generic base URL, API key, and model, so one .env can hold several
endpoints for A/B runs.

Implementation of the focused phases lives in:
  - ``cli.arg_validation``: flag normalization, range checks, scenario validation
  - ``cli.session_setup``: endpoint discovery, target/reviewer setup, preflight, context pressure
  - ``cli.perf_mode``: llama-benchy sweep planning, execution, persistence
  - ``cli.resume_mode``: checkpoint recovery and config mismatch checks
  - ``cli.scored_runners``: live Rich, headless JSON, and plain scenario runners
  - ``cli.probe``: ``--probe`` reachability, tool-call wire, and chat-template checks

All extracted modules are imported eagerly at top level so test patchers that
walk ``sys.modules`` reach every alias, and compatibility patch seams on
``cli.dispatch`` (and ``cli.bench``) remain live through call-time forwarding.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv  # noqa: F401  (re-exported via _load_dotenv)
from rich.console import Console
from rich.markup import escape

from tool_eval_bench.application.decision_audit import (
    with_selected_checks,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.application.resume import (
    result_requires_rerun as _resume_result_requires_rerun,  # noqa: F401
)
from tool_eval_bench.application.run_config import RunSettings  # noqa: F401  (cli.bench name)
from tool_eval_bench.application.run_context import build_run_context
from tool_eval_bench.application.run_queries import invocation_session
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import (
    arg_validation as _arg_validation,
)
from tool_eval_bench.cli import (
    model_probe as _model_probe,
)
from tool_eval_bench.cli import (
    perf_mode as _perf_mode,
)
from tool_eval_bench.cli import (
    probe as _probe_mod,
)
from tool_eval_bench.cli import (
    resume_mode as _resume_mode,
)
from tool_eval_bench.cli import (
    scored_runners as _scored_runners,
)
from tool_eval_bench.cli import (
    session_setup as _session_setup,
)
from tool_eval_bench.cli.arg_validation import (
    _build_extra_params,
    _drop_unused_system_prompt,  # noqa: F401
    _prepare_args,
    _resolve_system_prompt,  # noqa: F401
)
from tool_eval_bench.cli.compare_report import (
    run_compare_report_command as _run_compare_report_command,
)
from tool_eval_bench.cli.display import BenchmarkDisplay
from tool_eval_bench.cli.headless import (
    HeadlessConsole,
    emit_run_failed,
    headless_usage_errors,
    json_logging,
)
from tool_eval_bench.cli.helpers import (
    emit_headless_error as _headless_error,  # noqa: F401
)
from tool_eval_bench.cli.helpers import (
    load_dotenv_file as _load_dotenv,
)
from tool_eval_bench.cli.helpers import (
    metadata_for_storage as _metadata_for_storage,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.helpers import (
    persist_plugin_run as _persist_plugin_run,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.helpers import (
    safety_gate_failed as _safety_gate_failed,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.history import compare_runs as _compare_runs
from tool_eval_bench.cli.history import print_diff as _print_diff
from tool_eval_bench.cli.history import print_history as _print_history
from tool_eval_bench.cli.leaderboard import export_runs as _export_runs
from tool_eval_bench.cli.leaderboard import print_leaderboard as _print_leaderboard
from tool_eval_bench.cli.legacy_parser import _make_parser  # noqa: F401
from tool_eval_bench.cli.local_commands import handle_local_command as _handle_local_command
from tool_eval_bench.cli.modes import any_plugin_selected as _any_plugin_selected
from tool_eval_bench.cli.modes import sends_system_prompt as _sends_system_prompt
from tool_eval_bench.cli.perf import _resolve_benchy_tokenizer
from tool_eval_bench.cli.perf import run_llama_benchy as _run_llama_benchy
from tool_eval_bench.cli.perf_mode import (
    _exit_if_throughput_failed,
    _PerfSweep,  # noqa: F401
    _throughput_config,  # noqa: F401
)
from tool_eval_bench.cli.plugin_runners import (
    _run_decision_benchmark,
    _run_gsm8k_benchmark,
    _run_ifeval_benchmark,
    _run_mmlu_benchmark,
    _run_needle_benchmark,
    _run_toolscale_benchmark,
)
from tool_eval_bench.cli.pressure import run_pressure_sweep as _run_pressure_sweep
from tool_eval_bench.cli.probe import (
    check_chat_template_fidelity as _check_chat_template_fidelity,
)
from tool_eval_bench.cli.probe import (
    check_tool_call_wire_parsing as _check_tool_call_wire_parsing,
)
from tool_eval_bench.cli.probe import preflight_model_check as _preflight_model_check
from tool_eval_bench.cli.probe import warmup_server as _do_warmup
from tool_eval_bench.cli.resolve import (
    parse_int_list as _parse_int_list,
)
from tool_eval_bench.cli.resolve import (
    parse_sweep_range as _parse_sweep_range,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.resolve import redact_url as _redact_url  # noqa: F401
from tool_eval_bench.cli.resolve import (
    resolve_all_scenarios_for_ids as _resolve_all_scenarios_for_ids,  # noqa: F401
)
from tool_eval_bench.cli.resolve import resolve_scenarios as _resolve_scenarios
from tool_eval_bench.cli.resolve import (
    with_config_fingerprint as _with_config_fingerprint,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.resume_mode import (
    _load_resumable,
    _resume_config_mismatches,  # noqa: F401
    _resumed_session_id,
)
from tool_eval_bench.cli.run_io import aggregate_trials as _aggregate_trials
from tool_eval_bench.cli.run_io import bootstrap_ci as _bootstrap_ci  # noqa: F401
from tool_eval_bench.cli.run_io import emit_json_output as _emit_json_output
from tool_eval_bench.cli.run_io import median as _median  # noqa: F401
from tool_eval_bench.cli.run_io import (
    stderr_progress_result as _stderr_progress_result,  # noqa: F401
)
from tool_eval_bench.cli.run_io import (
    stderr_progress_start as _stderr_progress_start,  # noqa: F401
)
from tool_eval_bench.cli.scored_run import JudgeConnection
from tool_eval_bench.cli.scored_runners import (
    _pack_attestations,
    _plain_on_audit,  # noqa: F401
    _plain_verdict,  # noqa: F401
    _print_trials_summary,  # noqa: F401
    _report_model_label,  # noqa: F401
)
from tool_eval_bench.cli.server import DISCOVERY_PORTS as _DISCOVERY_PORTS  # noqa: F401
from tool_eval_bench.cli.server import discover_server as _discover_server
from tool_eval_bench.cli.session_setup import _Endpoint, _PressureFill, _Target
from tool_eval_bench.cli.spec_bench import load_spec_prompt_file
from tool_eval_bench.cli.spec_bench import run_spec_bench as _run_spec_bench
from tool_eval_bench.domain.models import ChatMessage, RunContext
from tool_eval_bench.runner.perf_presets import plan_perf
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.storage.reports.throughput import throughput_report
from tool_eval_bench.utils.urls import redact_urls as _redact_urls

if TYPE_CHECKING:
    from tool_eval_bench.runner.throughput import ThroughputSample

logger = logging.getLogger(__name__)

# Set of argument dest names that are intentionally suppressed (not in ARGS_SCHEMA).
# Used by the drift-detection test in tests/test_api.py.
_HIDDEN_ARGS: frozenset[str] = frozenset({"command", "help"})


# ---------------------------------------------------------------------------
# Compatibility seam wrappers (preserving patch targets on cli.dispatch / cli.bench)
# ---------------------------------------------------------------------------


def _decision_judge_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The service's judge kwargs; kept as a ``cli.bench`` name, the runners use ``ScoredRun``."""
    judge = JudgeConnection.from_args(args)
    return judge.service_kwargs() if judge is not None else {}


def _detect_model(
    base_url: str,
    api_key: str | None,
    console: Console,
    *,
    display_url: str | None = None,
    headless: bool = False,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
    multiple_models_hint: str | None = None,
    show_body: bool = True,
) -> tuple[str, str]:
    """Compatibility wrapper preserving the historical asyncio patch seam."""
    prior_asyncio = _model_probe.asyncio
    _model_probe.asyncio = asyncio
    try:
        return _model_probe._detect_model(
            base_url,
            api_key,
            console,
            display_url=display_url,
            headless=headless,
            wire_format=wire_format,
            headers=headers,
            multiple_models_hint=multiple_models_hint,
            show_body=show_body,
        )
    finally:
        _model_probe.asyncio = prior_asyncio


def _probe_server(
    console: Console,
    base_url: str,
    api_key: str | None,
    *,
    display_url: str | None = None,
    headless: bool = False,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
) -> None:
    """Compatibility wrapper preserving the historical asyncio patch seam."""
    prior_asyncio = _model_probe.asyncio
    _model_probe.asyncio = asyncio
    try:
        _model_probe._probe_server(
            console,
            base_url,
            api_key,
            display_url=display_url,
            headless=headless,
            wire_format=wire_format,
            headers=headers,
        )
    finally:
        _model_probe.asyncio = prior_asyncio


def _validate_explicit_scenarios(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _arg_validation._validate_explicit_scenarios(
        args, parser, resolve_scenarios_fn=_resolve_scenarios
    )


def _scenario_selector_label(args: argparse.Namespace) -> str:
    return _arg_validation._scenario_selector_label(args, resolve_scenarios_fn=_resolve_scenarios)


def _resolve_endpoint(
    args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console
) -> _Endpoint:
    return _session_setup._resolve_endpoint(
        args, parser, console, discover_server_fn=_discover_server
    )


def _resolve_target(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    console: Console,
    endpoint: _Endpoint,
) -> _Target:
    return _session_setup._resolve_target(
        args,
        parser,
        console,
        endpoint,
        detect_model_fn=_detect_model,
        resolve_scenarios_fn=_resolve_scenarios,
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
) -> None:
    _session_setup._check_endpoint_ready(
        args,
        console,
        base_url=base_url,
        model=model,
        api_key=api_key,
        wire_format=wire_format,
        extra_params=extra_params,
        headers=headers,
        display_url=display_url,
        preflight_fn=_preflight_model_check,
        warmup_fn=_do_warmup,
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
) -> RunContext | None:
    return _session_setup._build_run_context(
        args,
        console,
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        extra_params=extra_params,
        build_run_context_fn=build_run_context,
        scenario_selector_label_fn=_scenario_selector_label,
    )


def _ready_target(target: _Target, probe_headers: Mapping[str, str]) -> _Target:
    return _session_setup._ready_target(
        target,
        probe_headers,
        check_endpoint_ready_fn=_check_endpoint_ready,
        build_run_context_fn=_build_run_context,
    )


def _prepare_context_pressure(target: _Target) -> _PressureFill | None:
    return _session_setup._prepare_context_pressure(target, resolve_scenarios_fn=_resolve_scenarios)


def _scored_run_follows(args: argparse.Namespace) -> bool:
    """Whether a scored tool-call run follows the throughput sweep in ``_run_cli``.

    A context-pressure sweep runs scenarios but stores its own runs, so it does
    not carry ``--perf`` samples. ``tests/test_cli_dispatch_golden.py`` pins
    this against the routing for every mode that ends the CLI early.
    """
    return _sends_system_prompt(args) and args.context_pressure_sweep is None


def _run_throughput_mode(target: _Target) -> tuple[list, bool]:
    return _perf_mode._run_throughput_mode(
        target,
        resolve_tokenizer_fn=_resolve_benchy_tokenizer,
        run_benchy_fn=_run_llama_benchy,
        scored_run_follows_fn=_scored_run_follows,
        throughput_report_fn=throughput_report,
    )


def _save_throughput(target: _Target, throughput_samples: list[ThroughputSample]) -> int:
    return _perf_mode._save_throughput(
        target, throughput_samples, throughput_report_fn=throughput_report
    )


def _plan_resume(target: _Target, pressure: _PressureFill | None) -> None:
    _resume_mode._plan_resume(
        target,
        pressure,
        resolve_scenarios_fn=_resolve_scenarios,
        pack_attestations_fn=_pack_attestations,
    )


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
    wire_format: str = "openai",
) -> None:
    _scored_runners._run_with_live_display(
        service,
        console,
        model,
        display_name,
        backend,
        base_url,
        api_key,
        args,
        throughput_samples=throughput_samples,
        extra_params=extra_params,
        context_pressure_messages=context_pressure_messages,
        context_pressure_config=context_pressure_config,
        display_url=display_url,
        run_context=run_context,
        wire_format=wire_format,
        display_cls=BenchmarkDisplay,
        reporter_cls=MarkdownReporter,
        resolve_scenarios_fn=_resolve_scenarios,
        pack_attestations_fn=_pack_attestations,
        print_diff_fn=_print_diff,
        aggregate_trials_fn=_aggregate_trials,
    )


def _run_json(
    service: BenchmarkService,
    model: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    throughput_samples: list | None = None,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[ChatMessage] | None = None,
    context_pressure_config: dict | None = None,
    run_context: Any | None = None,
    wire_format: str = "openai",
) -> None:
    _scored_runners._run_json(
        service,
        model,
        backend,
        base_url,
        api_key,
        args,
        throughput_samples=throughput_samples,
        extra_params=extra_params,
        context_pressure_messages=context_pressure_messages,
        context_pressure_config=context_pressure_config,
        run_context=run_context,
        wire_format=wire_format,
        resolve_scenarios_fn=_resolve_scenarios,
        pack_attestations_fn=_pack_attestations,
        emit_json_output_fn=_emit_json_output,
        aggregate_trials_fn=_aggregate_trials,
    )


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
    wire_format: str = "openai",
) -> None:
    _scored_runners._run_plain(
        service,
        console,
        model,
        display_name,
        backend,
        base_url,
        api_key,
        args,
        throughput_samples=throughput_samples,
        extra_params=extra_params,
        context_pressure_messages=context_pressure_messages,
        context_pressure_config=context_pressure_config,
        display_url=display_url,
        run_context=run_context,
        wire_format=wire_format,
        reporter_cls=MarkdownReporter,
        resolve_scenarios_fn=_resolve_scenarios,
        pack_attestations_fn=_pack_attestations,
        print_diff_fn=_print_diff,
        aggregate_trials_fn=_aggregate_trials,
    )


def _run_scored_mode(
    target: _Target,
    throughput_samples: list[ThroughputSample],
    pressure: _PressureFill | None,
) -> None:
    _scored_runners._run_scored_mode(
        target,
        throughput_samples,
        pressure,
        service_cls=BenchmarkService,
        reporter_cls=MarkdownReporter,
        plan_resume_fn=_plan_resume,
        run_live_fn=_run_with_live_display,
        run_json_fn=_run_json,
        run_plain_fn=_run_plain,
    )


# ---------------------------------------------------------------------------
# Thin mode wrappers & top-level router
# ---------------------------------------------------------------------------


def _run_spec_bench_mode(target: _Target) -> bool:
    """Run the speculative-decoding benchmark. Returns True when the CLI is done."""
    args, console = target.args, target.console
    model, base_url, api_key = target.model, target.base_url, target.api_key
    display_name = target.display_name
    if args.spec_bench:
        # Spec-bench reads --pp, --tg, and --depth but ignores --perf-preset:
        # unset values keep the short grid it has always used.
        workload = plan_perf(
            "short",
            pp=args.pp,
            tg=args.tg,
            depths=None if args.depth is None else _parse_int_list(args.depth),
        )
        spec_depths = workload.depths
        spec_prompts = [p.strip() for p in args.spec_prompts.split(",") if p.strip()]
        custom_prompts = load_spec_prompt_file(args.spec_prompt_file)
        # A file with no explicit selection runs every prompt in it.
        if custom_prompts and not any(label in custom_prompts for label in spec_prompts):
            spec_prompts = spec_prompts + list(custom_prompts)
        spec_samples = _run_spec_bench(
            console,
            model,
            display_name,
            base_url,
            api_key,
            pp=workload.pp,
            tg=workload.tg,
            depths=spec_depths,
            spec_method=args.spec_method,
            baseline_tg_tps=args.baseline_tgs,
            prompt_types=spec_prompts,
            metrics_url=args.metrics_url,
            runs=args.spec_runs,
            temperature=args.temperature,
            custom_prompts=custom_prompts,
            output_dir=args.output_dir,
            label=args.label,
            run_context=target.run_context,
        )
        # Done unless accuracy plugins or the scored run still follow;
        # --skip-tool-eval drops only the scored run.
        if not _any_plugin_selected(args) and (
            args.skip_tool_eval or (not args.perf and not args.perf_only)
        ):
            # The failed row and its run_failed message are already out; as with
            # --perf-only, only the last mode turns a failed cell into exit 1.
            if any(sample.error for sample in spec_samples):
                sys.exit(1)
            return True
    return False


def _run_pressure_sweep_mode(target: _Target) -> bool:
    """Run the context-pressure sweep. Returns True when the CLI is done."""
    args, console = target.args, target.console
    if args.context_pressure_sweep is not None:
        _run_pressure_sweep(
            console,
            target.model,
            target.display_name,
            target.backend,
            target.base_url,
            target.api_key,
            args,
            display_url=target.display_url,
            extra_params=target.extra_params or None,
            label=args.label,
            run_context=target.run_context,
        )
        return True
    return False


def _run_spec_live_mode(target: _Target) -> None:
    """``--spec-live``: standalone live monitor (exits after session)."""
    args = target.args
    from tool_eval_bench.cli.spec_live_display import run_spec_live
    from tool_eval_bench.runner.spec_detection import canonical_spec_method_hint

    spec_method_hint = canonical_spec_method_hint(args.spec_method)

    try:
        asyncio.run(
            run_spec_live(
                target.base_url,
                api_key=target.api_key,
                metrics_url=args.metrics_url,
                model_name=target.display_name,
                poll_interval=args.spec_live_interval,
                spec_method=spec_method_hint,
                redact_endpoint=args.redact_url,
            )
        )
    except KeyboardInterrupt:
        pass


def _run_decision_live_mode(target: _Target) -> None:
    """``--decision-live``: standalone canary monitor (exits after session)."""
    args = target.args
    from tool_eval_bench.cli.decision_live_display import run_decision_live
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.domain.decision import DecisionUnsupportedError
    from tool_eval_bench.plugins.decision.typed_decisions import DatasetIntegrityError

    try:
        asyncio.run(
            run_decision_live(
                target.base_url,
                model=target.model,
                api_key=target.api_key,
                metrics_url=args.metrics_url,
                display_url=target.display_url,
                interval=args.decision_live_interval,
                timeout_seconds=args.timeout,
                adapter_options=adapter_options(args),
            )
        )
    except KeyboardInterrupt:
        pass
    except (DecisionUnsupportedError, DatasetIntegrityError) as exc:
        detail = _redact_urls(str(exc)) if args.redact_url else str(exc)
        target.console.print(f"\n[bold red]Decision monitor error:[/] {escape(detail)}")
        sys.exit(1)


def _run_plugins_mode(target: _Target) -> bool:
    """Run the selected accuracy plugins. Returns True when the CLI is done."""
    args = target.args
    from tool_eval_bench.cli.plugin_runners import run_selected_plugins

    return run_selected_plugins(
        target.console,
        target.model,
        target.display_name,
        target.base_url,
        target.api_key,
        args,
        runners={
            "gsm8k": _run_gsm8k_benchmark,
            "mmlu": _run_mmlu_benchmark,
            "ifeval": _run_ifeval_benchmark,
            "needle": _run_needle_benchmark,
            "toolscale": _run_toolscale_benchmark,
            "decision": _run_decision_benchmark,
        },
        extra_params=target.extra_params or None,
        output_dir=args.output_dir,
        run_context=target.run_context,
    )


def _skip_tool_eval_mode(target: _Target) -> bool:
    """``--skip-tool-eval``: stop before the scenarios. Returns True when the CLI is done."""
    args = target.args
    if not args.skip_tool_eval:
        return False
    any_benchmark = (
        args.perf
        or args.perf_only
        or args.spec_bench
        or args.spec_live
        or _any_plugin_selected(args)
    )
    if not any_benchmark:
        message = (
            "--skip-tool-eval has no effect without "
            "--perf, --perf-only, --spec-bench, --gsm8k, --mmlu, --ifeval, "
            "--needle, --toolscale, or --decision-bench."
        )
        if args.json:
            logger.warning(message)
        else:
            target.console.print(f"\n  [yellow]⚠ {message}[/]\n")
    return True


def main() -> None:
    _load_dotenv()
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.parser import parse_cli_args

    parser, args = parse_cli_args(make_parser)

    console = Console()

    if getattr(args, "command", None) == "compare-report":
        _run_compare_report_command(args, console)
        return

    # --json-file implies --json; _prepare_args records that on the namespace.
    headless = bool(args.json or args.json_file)
    if headless:
        headless_usage_errors(parser)
    with json_logging(headless):
        try:
            _run_cli(args, parser, console)
        except KeyboardInterrupt:
            # The runners handle an interrupt during the scenarios; this covers
            # the setup before them, so --json stderr stays JSON lines.
            if headless:
                emit_run_failed("interrupted")
            else:
                console.print("\n[bold red]Interrupted.[/]")
            sys.exit(1)


def _run_cli(args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console) -> None:
    _prepare_args(args, parser, console)

    if getattr(args, "scenarios_from_diff", None):
        from tool_eval_bench.cli.retest_diffs import apply_scenarios_from_diff

        apply_scenarios_from_diff(args, parser, console)

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

    if args.json:
        # stdout carries only the result envelope; see cli.headless.
        console = HeadlessConsole()
    _validate_explicit_scenarios(args, parser)
    args._session_id = uuid.uuid4().hex
    if args.resume and not args.probe:
        # _plan_resume reads the run again later; checking now keeps a bad
        # target from costing the preflight, warm-up, and any perf sweep.
        prev_run, _ = _load_resumable(args, console)
        args._session_id = _resumed_session_id(prev_run) or args._session_id
    # The RunContext carries the id into each row's metadata. The scope covers
    # a run whose context could not be built, which stores no metadata.
    with invocation_session(args._session_id):
        _run_session(args, parser, console)


def _run_session(
    args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console
) -> None:
    """Run the benchmark modes of one invocation; ``_run_cli`` sets up its session."""
    endpoint = _resolve_endpoint(args, parser, console)
    if args.probe:
        _probe_mod.run_probe_mode(
            args,
            console,
            endpoint,
            parser=parser,
            probe_server_fn=_probe_server,
            build_extra_params_fn=_build_extra_params,
            asyncio_mod=asyncio,
            check_tool_call_fn=_check_tool_call_wire_parsing,
            check_template_fn=_check_chat_template_fidelity,
        )
        return

    target = _resolve_target(args, parser, console, endpoint)
    if args.spec_live:
        _run_spec_live_mode(target)
        return
    if args.decision_live:
        _run_decision_live_mode(target)
        return

    target = _ready_target(target, endpoint.probe_headers)
    throughput_samples, finished = _run_throughput_mode(target)
    if finished:
        return
    # --perf samples ride in the scored run's report. When no scored run
    # follows, save them on their own now, before a mode that may end the
    # process with sys.exit. A failed cell still exits 1, but only after that
    # mode has had its turn.
    perf_failed_count = 0
    if throughput_samples and not _scored_run_follows(args):
        perf_failed_count = _save_throughput(target, throughput_samples)
        throughput_samples = []
    if _run_spec_bench_mode(target) or _run_pressure_sweep_mode(target):
        _exit_if_throughput_failed(target, perf_failed_count)
        return

    pressure = _prepare_context_pressure(target)
    if _scored_run_follows(args):
        # Combined plugin flags (--gsm8k, --mmlu, --ifeval, --needle,
        # --toolscale, --decision-bench) run after the tool-call scenarios, as
        # the CLI help states. Defer a --fail-on-safety exit (2) until any
        # requested plugins finish so they are not skipped.
        safety_exit: SystemExit | None = None
        try:
            _run_scored_mode(target, throughput_samples, pressure)
        except SystemExit as exc:
            if exc.code != 2 or not _any_plugin_selected(args):
                raise
            safety_exit = exc
        _run_plugins_mode(target)
        if safety_exit is not None:
            raise safety_exit
        return

    if _run_plugins_mode(target) or _skip_tool_eval_mode(target):
        _exit_if_throughput_failed(target, perf_failed_count)
        return
    _run_scored_mode(target, throughput_samples, pressure)


if __name__ == "__main__":
    main()
