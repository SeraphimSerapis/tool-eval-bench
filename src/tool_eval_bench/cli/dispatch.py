"""CLI dispatch and compatibility implementation for running benchmarks.

Defaults cascade:  .env file → TOOL_EVAL_* env vars → hardcoded fallbacks.
With --provider NAME (or TOOL_EVAL_PROVIDER), the TOOL_EVAL_<NAME>_* triple
replaces the generic base URL, API key, and model, so one .env can hold several
endpoints for A/B runs.

Usage:
    tool-eval-bench                           # uses .env / env vars
    tool-eval-bench --base-url URL            # override server
    tool-eval-bench --provider gemini         # TOOL_EVAL_GEMINI_* from .env
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
import sqlite3
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv  # noqa: F401  (re-exported via _load_dotenv)
from rich.console import Console

from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.adapters.wire_format import resolve_wire_format as _resolve_wire_format
from tool_eval_bench.application.decision_audit import (
    decision_judge_config,
    with_selected_checks,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.application.run_config import (
    RunSettings,  # noqa: F401  (cli.bench name)
    build_run_config,
    resume_mismatches,
)
from tool_eval_bench.application.run_context import build_run_context, identify_backend
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import model_probe as _model_probe
from tool_eval_bench.cli.command_registry import PLUGIN_FLAG_STEMS
from tool_eval_bench.cli.compare_report import (
    run_compare_report_command as _run_compare_report_command,
)
from tool_eval_bench.cli.display import BenchmarkDisplay, decision_audit_line
from tool_eval_bench.cli.headless import (
    HeadlessConsole,
    emit_run_failed,
    headless_usage_errors,
    json_logging,
    report_run_failed,
    report_run_saved,
)
from tool_eval_bench.cli.helpers import (
    emit_headless_error as _headless_error,
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
from tool_eval_bench.cli.helpers import prior_results_for_resume
from tool_eval_bench.cli.helpers import (
    safety_gate_failed as _safety_gate_failed,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.helpers import trial_safety_warnings as _trial_safety_warnings
from tool_eval_bench.cli.helpers import trials_safety_gate_failed as _trials_safety_gate_failed
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
from tool_eval_bench.cli.plugin_runners import (
    _run_decision_benchmark,
    _run_gsm8k_benchmark,
    _run_ifeval_benchmark,
    _run_mmlu_benchmark,
    _run_needle_benchmark,
)
from tool_eval_bench.cli.pressure import (
    run_pressure_sweep as _run_pressure_sweep,
)
from tool_eval_bench.cli.probe import preflight_model_check as _preflight_model_check
from tool_eval_bench.cli.probe import warmup_server as _do_warmup
from tool_eval_bench.cli.provider_env import resolve_provider as _resolve_provider
from tool_eval_bench.cli.resolve import (
    parse_int_list as _parse_int_list,
)
from tool_eval_bench.cli.resolve import (
    parse_sweep_range as _parse_sweep_range,  # noqa: F401  (cli.bench name)
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
    with_config_fingerprint as _with_config_fingerprint,  # noqa: F401  (cli.bench name)
)
from tool_eval_bench.cli.run_io import aggregate_trials as _aggregate_trials
from tool_eval_bench.cli.run_io import bootstrap_ci as _bootstrap_ci  # noqa: F401
from tool_eval_bench.cli.run_io import emit_json_output as _emit_json_output
from tool_eval_bench.cli.run_io import median as _median  # noqa: F401
from tool_eval_bench.cli.run_io import stderr_progress_audit as _stderr_progress_audit
from tool_eval_bench.cli.run_io import stderr_progress_result as _stderr_progress_result
from tool_eval_bench.cli.run_io import stderr_progress_start as _stderr_progress_start
from tool_eval_bench.cli.scored_run import JudgeConnection, ScoredRun
from tool_eval_bench.cli.server import (
    DISCOVERY_PORTS as _DISCOVERY_PORTS,
)
from tool_eval_bench.cli.server import (
    discover_server as _discover_server,
)
from tool_eval_bench.cli.spec_bench import (
    load_spec_prompt_file,
)
from tool_eval_bench.cli.spec_bench import (
    run_spec_bench as _run_spec_bench,
)
from tool_eval_bench.domain.errors import (
    NO_SERVER,
)
from tool_eval_bench.domain.models import ChatMessage, RunContext
from tool_eval_bench.domain.scenarios import (
    AuditPhase,
    Category,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
    scenario_report_metadata,
)
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.storage.reports.throughput import throughput_report
from tool_eval_bench.utils.headers import attach_session_id as _attach_session_id
from tool_eval_bench.utils.headers import merge_headers as _merge_headers
from tool_eval_bench.utils.headers import parse_header_env as _parse_header_env
from tool_eval_bench.utils.headers import parse_header_pairs as _parse_header_pairs
from tool_eval_bench.utils.system_prompt import MAX_SYSTEM_PROMPT_BYTES, normalize_system_prompt
from tool_eval_bench.utils.urls import redact_urls as _redact_urls

if TYPE_CHECKING:
    from tool_eval_bench.runner.throughput import ThroughputSample

logger = logging.getLogger(__name__)

# Valid category letters for --categories
_VALID_CATEGORIES = {c.value for c in Category}


def _resume_result_requires_rerun(result: dict[str, Any]) -> bool:
    """Return whether a checkpoint is missing, corrupt, or infrastructure-only.

    A model outcome is evidence, including ordinary failures and partials.  It
    must not be silently retried into a better score under the same run ID.
    """
    if not result.get("scenario_id"):
        return True
    if result.get("status") not in {status.value for status in ScenarioStatus}:
        return True
    if not isinstance(result.get("points"), int) or not isinstance(result.get("summary"), str):
        return True
    if not isinstance(result.get("raw_log"), str) or not result["raw_log"].strip():
        return True
    return result.get("status") == ScenarioStatus.FAIL.value and result.get("failure_kind") in {
        "timeout",
        "connection_error",
        "server_error",
    }


def _resume_config_mismatches(
    previous: dict[str, Any],
    *,
    model: str,
    backend: str,
    base_url: str,
    scenarios: list[ScenarioDefinition],
    args: argparse.Namespace,
    extra_params: dict[str, Any] | None,
    scenario_packs: list[dict[str, Any]] | None,
    context_pressure: dict[str, Any] | None,
) -> list[str]:
    """Compare every user-controlled scoring condition persisted in a run.

    The current config is built the way the service builds the one it stores,
    so both sides share one schema and the comparison cannot drift from it.
    """
    request = ScoredRun.from_args(
        args,
        model=model,
        backend=backend,
        base_url=base_url,
        # Neither reaches the stored config.
        api_key=None,
        wire_format="openai",
        scenarios=scenarios,
        extra_params=extra_params,
        scenario_packs=scenario_packs,
        context_pressure_config=context_pressure,
        # The caller's scenarios are the protocol being compared, whatever an
        # earlier resume left on args.
        resume_scenarios=scenarios,
    )
    # Metadata only feeds the fingerprint, which resume does not compare.
    current = build_run_config(
        request.run_settings(),
        scenarios=request.config_scenarios(),
        metadata={},
        scenario_packs=scenario_packs,
    )
    return resume_mismatches(
        previous,
        current,
        base_url=base_url,
        judge_base_url=request.judge.base_url if request.judge else None,
    )


def _execution_scenarios(args: argparse.Namespace) -> list[ScenarioDefinition]:
    """Return the explicit resume subset, including an intentionally empty one."""
    subset = getattr(args, "_resume_remaining_scenarios", None)
    return subset if subset is not None else _resolve_scenarios(args)


def _score_stored_trial(
    result: dict[str, Any], args: argparse.Namespace, scenarios: list[ScenarioDefinition]
) -> ModelScoreSummary | None:
    """Score one finished trial the way the service scored the row it stored.

    Every runner's trial statistics go through here so they agree. The stored
    results keep ``failure_kind``, so an infrastructure failure stays out of the
    score as it does in the service. Each result is scored against its own
    definition, which for a resumed run is the whole original protocol rather
    than the rerun subset. ``scenarios`` are the definitions the trial executed;
    they and the resume protocol take precedence over the registry, as in the
    service. Returns None for a trial with no results.
    """
    from tool_eval_bench.runner.orchestrator import score_results

    results = [
        ScenarioResult.from_dict(data)
        for data in result.get("scores", {}).get("scenario_results", [])
    ]
    if not results:
        return None
    known = {
        scenario.id: scenario
        for scenario in _resolve_all_scenarios_for_ids([r.scenario_id for r in results])
    }
    known.update({s.id: s for s in getattr(args, "_resume_scenarios", None) or []})
    known.update({s.id: s for s in scenarios})
    return score_results(
        results,
        [known[r.scenario_id] for r in results],
        alpha=args.alpha,
        weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
    )


def _resolve_diff_target(args: argparse.Namespace) -> None:
    """Fix ``--diff``'s comparison run before the scored run starts.

    The run being benchmarked is stored when it finishes, so resolving
    ``latest`` afterwards would compare the run with itself. ``latest`` means
    the newest completed tool-call run. Leaves ``args._diff_target`` (None when
    there is nothing to compare against); ``--json`` has no console diff.
    """
    args._diff_target = None
    diff = getattr(args, "diff", None)
    if not diff:
        return
    if args.json:
        logger.warning("--diff prints a console table and is ignored with --json")
        return
    if diff.lower() != "latest":
        args._diff_target = diff
        return
    from tool_eval_bench.application.run_queries import resolve_run

    try:
        resolved = resolve_run(diff, run_type="tool_eval", status="completed")
    except (sqlite3.Error, OSError) as exc:
        # The diff is a convenience printed after the run; an unreadable
        # history database must not stop the benchmark from starting.
        logger.warning("--diff skipped: could not read previous runs (%s)", exc)
        args.diff = None
        return
    args._diff_target = resolved[0] if resolved is not None else None


def _show_diff(console: Console, results: list[ScenarioResult], args: argparse.Namespace) -> None:
    """Print the ``--diff`` table against the run fixed by ``_resolve_diff_target``."""
    if not getattr(args, "diff", None):
        return
    target = getattr(args, "_diff_target", None)
    if target is None:
        console.print("\n  [yellow]No previous runs found for comparison.[/]\n")
        return
    _print_diff(console, results, target)


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
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Compatibility wrapper preserving the historical asyncio patch seam."""
    _model_probe.asyncio = asyncio
    return _model_probe._detect_model(
        base_url,
        api_key,
        console,
        display_url=display_url,
        headless=headless,
        wire_format=wire_format,
        headers=headers,
    )


def _probe_server(
    console: Console,
    base_url: str,
    api_key: str | None,
    *,
    headless: bool = False,
    wire_format: str = "openai",
    headers: Mapping[str, str] | None = None,
) -> None:
    """Compatibility wrapper preserving the historical asyncio patch seam."""
    _model_probe.asyncio = asyncio
    _model_probe._probe_server(
        console, base_url, api_key, headless=headless, wire_format=wire_format, headers=headers
    )


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


async def _plain_on_audit(
    scenario: ScenarioDefinition, result: ScenarioResult, phase: AuditPhase
) -> None:
    line = decision_audit_line(scenario.id, result, phase)
    if line is not None:
        print(line.plain, flush=True)


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


@dataclass
class _Target:
    """Everything the mode handlers below need about a resolved endpoint.

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


def _benchy_extra_args(args: argparse.Namespace) -> list[str] | None:
    if not args.benchy_args:
        return None
    import shlex

    return shlex.split(args.benchy_args)


def _stored_tokenizer(tokenizer: str | None) -> str | None:
    """Reduce a local tokenizer path to its basename; keep a Hugging Face repo id.

    A path can name the user's home directory or a private mount, and the
    stored config is exported and shared. A repo id such as ``Qwen/Qwen3-8B``
    identifies the tokenizer and carries no local detail.
    """
    if tokenizer is None:
        return None
    if tokenizer.startswith(("/", "./", "../", "~")) or os.path.exists(tokenizer):
        return Path(tokenizer).name
    return tokenizer


def _throughput_config(target: _Target) -> dict[str, Any]:
    """The stored config of a throughput run that no scored run carries.

    The workload fields keep different sweeps out of one fingerprint cohort.
    ``--benchy-args`` is stored with credentials stripped, because it can
    carry ``--api-key``.
    """
    from tool_eval_bench.runner.llama_benchy import redact_arguments

    args = target.args
    return {
        "model": target.model,
        "backend": target.backend,
        "base_url": target.base_url,
        "mode": "perf-only",
        "pp": args.pp,
        "tg": args.tg,
        "depths": _parse_int_list(args.depth),
        "concurrency": _parse_int_list(args.concurrency),
        "runs": args.benchy_runs,
        "latency_mode": args.benchy_latency_mode,
        # The tokenizer builds the prompts and counts their tokens.
        "tokenizer": _stored_tokenizer(getattr(args, "tokenizer", None)),
        "benchy_args": redact_arguments(_benchy_extra_args(args) or []),
    }


def _run_throughput_mode(target: _Target) -> tuple[list, bool]:
    """Run the llama-benchy throughput sweep.

    Returns the samples and whether the CLI is finished; ``--perf-only`` writes
    its own report and stops, while ``--perf`` hands its samples to whatever
    runs next. ``_run_cli`` persists them itself when no scored run follows.
    """
    args, console = target.args, target.console
    if not (args.perf or args.perf_only):
        return [], False

    throughput_samples = _run_llama_benchy(
        console,
        target.model,
        target.display_name,
        target.base_url,
        target.api_key,
        pp=[args.pp],
        tg=[args.tg],
        depths=_parse_int_list(args.depth),
        concurrency_levels=_parse_int_list(args.concurrency),
        runs=args.benchy_runs,
        latency_mode=args.benchy_latency_mode,
        skip_coherence=True,
        extra_args=_benchy_extra_args(args),
        # llama-benchy always warms up, as llama-bench does: its global
        # warm-up requests plus a discarded first run of every cell. Its
        # --no-warmup drops both, so each cell's first measured run would be
        # cold. tool-eval-bench's --no-warmup covers only its own warm-up.
        skip_warmup=False,
        tokenizer=getattr(args, "tokenizer", None),
        backend=target.backend,
    )

    if not args.perf_only:
        # Failed cells too: the stored row counts them, and every report and
        # display of the samples drops them itself.
        return throughput_samples, False

    _exit_if_throughput_failed(target, _save_throughput(target, throughput_samples))
    return throughput_samples, True


def _save_throughput(target: _Target, throughput_samples: list[ThroughputSample]) -> int:
    """Persist a throughput sweep as its own ``perf`` run and return its failed-cell count.

    Used by ``--perf-only`` and by ``--perf`` when no scored run follows to
    carry the samples (spec-bench, a sweep, a plugin-only run, or
    ``--skip-tool-eval``). Both store the same config, so the same measurement
    lands in the same fingerprint cohort. The caller decides when a failed
    cell ends the CLI; see :func:`_exit_if_throughput_failed`.
    """
    args, console = target.args, target.console
    failed_count = sum(bool(sample.error) for sample in throughput_samples)
    successful_count = len(throughput_samples) - failed_count
    scores: dict[str, Any] = {"samples": len(throughput_samples)}
    if failed_count:
        scores.update({"successful": successful_count, "failed": failed_count})
    # Failed cells stay a count: their error text can quote the server URL.
    scores["results"] = [sample.to_result() for sample in throughput_samples if not sample.error]
    run_context = target.run_context
    run = ModeRun(
        run_type="perf",
        config=_throughput_config(target),
        scores=scores,
        status="failed" if failed_count else "completed",
    )
    finalized = finalize_mode_run(
        run,
        throughput_report(
            target.display_name,
            throughput_samples,
            label=args.label,
            backend=target.backend,
            # The same redacted form the run context stores, whatever --redact-url says.
            server=_redact_url(target.base_url),
            served_model=target.model,
            model_root=run_context.server_model_root if run_context is not None else None,
        ),
        run_context=run_context,
        output_dir=args.output_dir,
    )
    report_path = finalized.report_path
    report_run_saved(console, run, finalized)
    console.print(f"\n  [dim]Report saved to {report_path}[/]\n")
    return failed_count


def _exit_if_throughput_failed(target: _Target, failed_count: int) -> None:
    """Report a failed throughput sweep and exit 1; a no-op when every cell succeeded."""
    if failed_count:
        report_run_failed(
            target.console,
            f"[bold red]Throughput benchmark failed in {failed_count} cell(s).[/]",
        )
        sys.exit(1)


def _reject_unknown_categories(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    invalid = {c.upper() for c in args.categories or []} - _VALID_CATEGORIES
    if invalid:
        parser.error(
            f"Unknown categories: {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(_VALID_CATEGORIES))}"
        )


def _validate_scenario_selection(
    args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console
) -> None:
    """Reject a bad pack or category before any request is made.

    Packs load up front on purpose: a missing, empty, or colliding pack must
    abort before the run starts rather than surface as a traceback partway
    through scenario resolution.
    """
    try:
        packs = _resolve_packs(args)
        _resolve_scenarios(args)
    except ValueError as exc:
        parser.error(str(exc))
    if packs and not args.json:
        total = sum(len(p.scenarios) for p in packs)
        names = ", ".join(f"{p.name} ({p.content_hash})" for p in packs)
        console.print(f"  [dim]🔒 Held-out packs: {names} — {total} scenario(s)[/]")

    if args.categories:
        _reject_unknown_categories(args, parser)
        cats = [c.upper() for c in args.categories]
        from tool_eval_bench.domain.scenarios import CATEGORY_LABELS

        cat_names = ", ".join(f"{c} ({CATEGORY_LABELS[Category(c)]})" for c in cats)
        resolved_count = len(_resolve_scenarios(args))
        if not args.json:
            console.print(f"  [dim]📋 Categories: {cat_names} ({resolved_count} scenarios)[/]")


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
) -> None:
    """Verify the model answers a real request, then warm the server.

    Some servers list a model in ``/v1/models`` and still fail the first real
    request.  Without this gate the benchmark scores the failure as the model's.
    """
    if not args.no_preflight:
        _preflight_model_check(
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
        )

    if not args.no_warmup and not args.json:
        _do_warmup(
            console,
            base_url,
            model,
            api_key,
            wire_format=wire_format,
            temperature=args.temperature,
            extra_params=extra_params or None,
            headers=headers,
        )


def _decision_judge_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The service's judge kwargs; kept as a ``cli.bench`` name, the runners use ``ScoredRun``."""
    judge = JudgeConnection.from_args(args)
    return judge.service_kwargs() if judge is not None else {}


def _resolve_system_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Materialize ``--system-prompt-file`` into ``args.system_prompt``.

    Downstream consumers (run modes, resume-compat check, RunContext) read only
    ``args.system_prompt``, so the file variant is resolved once, right after
    argument parsing, and both variants leave here in the canonical form that is
    persisted and sent (see ``normalize_system_prompt``).
    """
    file_prompt = getattr(args, "system_prompt_file", None)
    inline_prompt = getattr(args, "system_prompt", None)
    if file_prompt is not None and inline_prompt is not None:
        parser.error("--system-prompt and --system-prompt-file are mutually exclusive")
    if file_prompt is not None:
        source = f"--system-prompt-file '{file_prompt}'"
        try:
            with open(file_prompt, "rb") as handle:
                raw = handle.read(MAX_SYSTEM_PROMPT_BYTES + 1)
        except OSError as exc:
            parser.error(f"Cannot read {source}: {exc}")
        # Judged on the file, not the stripped text: a bounded read must never
        # silently drop the tail of a larger file.
        if len(raw) > MAX_SYSTEM_PROMPT_BYTES:
            parser.error(f"{source} is larger than {MAX_SYSTEM_PROMPT_BYTES} bytes")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            parser.error(f"{source} is not valid UTF-8")
    elif inline_prompt is not None:
        source, text = "--system-prompt", inline_prompt
    else:
        return
    try:
        args.system_prompt = normalize_system_prompt(text)
    except ValueError as exc:
        parser.error(f"{source}: {exc}")


def _any_plugin_selected(args: argparse.Namespace) -> bool:
    """Whether any accuracy plugin runs, through ``--<plugin>`` or ``--<plugin>-only``."""
    return any(
        getattr(args, stem) or getattr(args, f"{stem}_only") for stem in PLUGIN_FLAG_STEMS.values()
    )


def _sends_system_prompt(args: argparse.Namespace) -> bool:
    """Whether this invocation runs tool-call scenarios, the override's only consumer.

    Also gates the checks that only matter when scenarios run, such as rejecting
    an empty scenario selection.

    Mirrors the routing in ``main()``: ``--perf-only``, ``--spec-live``, a lone
    ``--spec-bench``, any ``--<plugin>-only`` run, and ``--skip-tool-eval`` all
    stop before the scenarios; a context-pressure sweep is a scenario run.

    Probe, ``--dry-run``, and the storage commands are not considered here: they
    record nothing, so an unused flag is inert, and they already accept the rest
    of the run-control group in silence.
    """
    if args.spec_live or args.decision_live or args.perf_only:
        return False
    other_benchmarks = args.perf or _any_plugin_selected(args)
    if args.spec_bench and (args.skip_tool_eval or not other_benchmarks):
        return False
    if args.context_pressure_sweep is not None:
        return True
    if args.skip_tool_eval:
        return False
    return not any(getattr(args, f"{stem}_only") for stem in PLUGIN_FLAG_STEMS.values())


def _drop_unused_system_prompt(args: argparse.Namespace, console: Console) -> None:
    """Ignore, with a warning, an override that no scenario would receive.

    Left on the namespace it would be recorded in the run context and marked in
    a report as if the model had been run under it.
    """
    if args.system_prompt is None or _sends_system_prompt(args):
        return
    message = (
        "--system-prompt applies only to tool-call scenarios and is ignored: "
        "this invocation runs none."
    )
    if args.json:
        logger.warning(message)
    else:
        console.print(f"\n  [yellow]⚠ {message}[/]\n")
    args.system_prompt = None


def _scenario_selector_label(args: argparse.Namespace) -> str:
    """Describe the scenario selection for the run report."""
    resolved = _resolve_scenarios(args)
    if args.scenarios:
        return ", ".join(args.scenarios)
    if args.categories:
        return f"categories {', '.join(c.upper() for c in args.categories)} ({len(resolved)})"
    if args.short:
        return f"short ({len(resolved)})"
    return f"all ({len(resolved)})"


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
    """Collect execution-context metadata for the report.

    Built before the mode branches so the throughput and spec-decode paths get
    engine detection too.  Failure is not fatal: the run proceeds with a report
    that lacks the context block.
    """
    run_context = asyncio.run(
        build_run_context(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout,
            seed=args.seed,
            scenario_selector=_scenario_selector_label(args),
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
        console.print(f"  [dim]🔍 Engine: {engine_str}[/]")
    return run_context


def _run_spec_bench_mode(target: _Target) -> bool:
    """Run the speculative-decoding benchmark. Returns True when the CLI is done."""
    args, console = target.args, target.console
    model, base_url, api_key = target.model, target.base_url, target.api_key
    display_name = target.display_name
    if args.spec_bench:
        spec_depths = _parse_int_list(args.depth)
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
            pp=args.pp,
            tg=args.tg,
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
        # If --spec-bench is the only mode, or user explicitly skipped tool-eval
        if args.skip_tool_eval or (
            not args.perf and not args.perf_only and not _any_plugin_selected(args)
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
    model, base_url, api_key = target.model, target.base_url, target.api_key
    display_name, display_url = target.display_name, target.display_url
    backend, extra_params = target.backend, target.extra_params
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
            label=args.label,
            run_context=target.run_context,
        )
        return True
    return False


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


@dataclass(frozen=True)
class _PressureFill:
    """Context-pressure filler for the scored run, and the config it records."""

    messages: list[ChatMessage]
    config: dict[str, Any]


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
        _run_cli(args, parser, console)


def _run_cli(args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console) -> None:
    _prepare_args(args, parser, console)

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
    endpoint = _resolve_endpoint(args, parser, console)
    if args.probe:
        _run_probe_mode(args, console, endpoint)
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
    if _run_plugins_mode(target) or _skip_tool_eval_mode(target):
        _exit_if_throughput_failed(target, perf_failed_count)
        return
    _run_scored_mode(target, throughput_samples, pressure)


def _scored_run_follows(args: argparse.Namespace) -> bool:
    """Whether a scored tool-call run follows the throughput sweep in ``_run_cli``.

    A context-pressure sweep runs scenarios but stores its own runs, so it does
    not carry ``--perf`` samples. ``tests/test_cli_dispatch_golden.py`` pins
    this against the routing for every mode that ends the CLI early.
    """
    return _sends_system_prompt(args) and args.context_pressure_sweep is None


def _prepare_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console
) -> None:
    """Normalize the parsed flags that every later step reads."""
    # --json-file implies --json
    if args.json_file:
        args.json = True
    if args.json and (args.spec_live or args.decision_live):
        flag = "--spec-live" if args.spec_live else "--decision-live"
        parser.error(f"{flag} is an interactive monitor and cannot be combined with --json")

    # Resolve the system-prompt override from its file, if given, so every
    # downstream consumer (run, resume-compat check, RunContext) sees plain
    # text on args.system_prompt.
    _resolve_system_prompt(args, parser)
    _drop_unused_system_prompt(args, console)
    _validate_run_ranges(args, parser)
    try:
        judge_config = decision_judge_config(
            getattr(args, "decision_judge_base_url", None),
            getattr(args, "decision_judge_model", None),
            judge_set=getattr(args, "decision_judge", None),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if judge_config is not None and (
        getattr(args, "command", None) not in {None, "run", "resume"}
        or not _sends_system_prompt(args)
        or args.context_pressure_sweep is not None
    ):
        parser.error("Decision judge audits require a run or resume of tool-call scenarios")


def _validate_run_ranges(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject out-of-range run settings before any request.

    Checked here rather than with an argparse ``type=`` so ``--json`` reports
    them as ``invalid_arguments``. ``0 <= x <= 1`` also rejects NaN.
    """
    max_turns = getattr(args, "max_turns", None)
    if max_turns is not None and max_turns < 1:
        parser.error(f"--max-turns must be at least 1, got {max_turns}")
    for flag, attr in (
        ("--error-rate", "error_rate"),
        ("--alpha", "alpha"),
        ("--context-pressure", "context_pressure"),
    ):
        value = getattr(args, attr, None)
        if value is not None and not (0.0 <= value <= 1.0):
            parser.error(f"{flag} must be between 0 and 1, got {value}")


def _validate_explicit_scenarios(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Validate explicit scenario IDs before server discovery or benchmark
    # requests. This keeps a typo from turning into an empty run or a server
    # failure. Probe/history/dry-run keep their own command semantics.
    if args.probe:
        return
    if args.scenario_pack and args.context_pressure_sweep is not None:
        # The sweep report renders every trace and its config records no
        # pack attestation, so running a held-out pack there would publish it.
        parser.error(
            "--scenario-pack cannot be combined with --context-pressure-sweep: "
            "the sweep report publishes full traces"
        )
    try:
        resolved = _resolve_scenarios(args)
    except ValueError as exc:
        parser.error(str(exc))
    # A filter that matches nothing would otherwise persist a completed
    # 0-scenario run scored 0.  Modes that never send scenarios (--perf-only,
    # plugin-only, --skip-tool-eval) ignore the selection, and a resume runs
    # whatever its checkpoint left, which may legitimately be nothing.
    if not resolved and not getattr(args, "resume", None) and _sends_system_prompt(args):
        _reject_unknown_categories(args, parser)
        message = "No scenarios matched the selected filters"
        requested = {c.upper() for c in args.categories or []}
        if "P" in requested and not (args.hardmode or args.hardmode_only):
            message += "; Category P is Hard Mode, so add --hardmode"
        parser.error(message)


def _resolve_endpoint(
    args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console
) -> _Endpoint:
    """Resolve the server, credentials, headers, wire format, and backend label."""
    # Cascade: CLI flag → provider-scoped env → generic env → auto-discovery.
    # A selected provider owns its api_key and model: the generic
    # TOOL_EVAL_API_KEY is for whatever the generic base URL points at, and
    # sending it to a vendor API would just muddy a failed-auth diagnosis.
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
            console.print(f"[dim]  Detected backend: {detection.server_name}[/]")

    return _Endpoint(
        base_url=base_url,
        api_key=api_key,
        backend=backend,
        wire_format=wire_format,
        model=model,
        probe_headers=probe_headers,
    )


def _run_probe_mode(args: argparse.Namespace, console: Console, endpoint: _Endpoint) -> None:
    """``--probe``: check whether the server is reachable."""
    _probe_server(
        console,
        endpoint.base_url,
        endpoint.api_key,
        headless=args.json,
        wire_format=endpoint.wire_format,
        headers=endpoint.probe_headers,
    )


def _resolve_target(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    console: Console,
    endpoint: _Endpoint,
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
            console.print(f"[dim]  Server: {display_url}[/]")
        model, display_name = _detect_model(
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

    # display_name is the human-readable model (e.g. "Intel/gemma-4-31B-it-int4-AutoRound")
    # model is the API alias (e.g. "gemma4") — used in all API calls
    display_name = display_name or model

    extra_params = _build_extra_params(args, parser)

    _validate_scenario_selection(args, parser, console)

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


def _build_extra_params(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    """Build extra_params from the sampling and thinking flags."""
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
    return extra_params


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
        target.console.print(f"\n[bold red]Decision monitor error:[/] {exc}")
        sys.exit(1)


def _ready_target(target: _Target, probe_headers: Mapping[str, str]) -> _Target:
    """Gate on a model that answers, then attach the run context every mode reports."""
    args, console = target.args, target.console
    # A decision model has no chat endpoint to preflight or warm; the plugin
    # reports an unreachable or unsupported endpoint itself.
    if not args.decision_bench_only:
        _check_endpoint_ready(
            args,
            console,
            base_url=target.base_url,
            model=target.model,
            api_key=target.api_key,
            wire_format=target.wire_format,
            extra_params=target.extra_params,
            headers=probe_headers,
        )

    run_context = _build_run_context(
        args,
        console,
        model=target.model,
        backend=target.backend,
        base_url=target.base_url,
        api_key=target.api_key,
        extra_params=target.extra_params,
    )
    return replace(target, run_context=run_context)


def _prepare_context_pressure(target: _Target) -> _PressureFill | None:
    """Build the ``--context-pressure`` filler.

    Scales ``args.timeout`` for a large fill; that must happen before the
    resume check and the run read it.
    """
    args, console = target.args, target.console
    base_url, model, api_key = target.base_url, target.model, target.api_key
    if args.context_pressure is None:
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
                client_factory=HTTPMeasurementClient,
                reported_context=reported_context_window(target.run_context),
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
        if estimated:
            # No usable /tokenize: fill_tokens is the 4 chars/token estimate,
            # which runs well above real counts on common tokenizers.
            pressure_config_dict["fill_tokens_estimated"] = True
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
        report_run_failed(console, f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)
    return _PressureFill(messages=pressure_messages, config=pressure_config_dict)


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
            "--needle, or --decision-bench."
        )
        if args.json:
            logger.warning(message)
        else:
            target.console.print(f"\n  [yellow]⚠ {message}[/]\n")
    return True


def _plan_resume(target: _Target, pressure: _PressureFill | None) -> None:
    """Preserve completed model outcomes under the original run ID.

    Leaves the plan on ``args._resume_*`` (and narrows ``args.scenarios`` to
    the rerun subset), where ``ScoredRun.from_args`` reads it; exits 1 when the
    run cannot be resumed.
    """
    args, console = target.args, target.console
    resume_prior_results: list[dict] | None = None
    resume_scenarios: list[ScenarioDefinition] | None = None
    if args.resume:
        from tool_eval_bench.application.run_queries import resume_state

        resumable = resume_state(args.resume)
        prev_run = resumable[0] if resumable else None
        prev_checkpoints = resumable[1] if resumable else []
        if prev_run is None:
            report_run_failed(
                console,
                f"\n  [bold red]✗[/] Run '{args.resume}' not found in history.\n"
                "  [dim]Use --history to list available runs.[/]\n",
            )
            sys.exit(1)

        if prev_run.get("status") == "completed":
            report_run_failed(
                console,
                "\n  [bold red]✗ Resume aborted: run is already completed[/]\n"
                "  [dim]Completed scenario outcomes are immutable. Start a fresh run to retry.[/]\n",
            )
            sys.exit(1)

        # Resolve before changing args.scenarios.  These definitions are needed
        # to rescore checkpointed Hard Mode and held-out-pack results.
        resume_scenarios = _resolve_scenarios(args)

        # Validate every persisted benchmark condition, not merely model/backend.
        prev_config = prev_run.get("config") or {}
        mismatches = _resume_config_mismatches(
            prev_config,
            model=target.model,
            backend=target.backend,
            base_url=target.base_url,
            scenarios=resume_scenarios,
            args=args,
            extra_params=target.extra_params or None,
            scenario_packs=_pack_attestations(args),
            context_pressure=pressure.config if pressure is not None else None,
        )
        if mismatches:
            report_run_failed(
                console,
                f"\n  [bold red]✗ Resume aborted: configuration mismatch[/]\n"
                f"  [dim]Prior run differs in: {', '.join(mismatches)}[/]\n"
                f"  [dim]Start a fresh run instead of resuming.[/]\n",
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
            prior_by_id = {
                result["scenario_id"]: result
                for result in prev_results
                if result.get("scenario_id") and not _resume_result_requires_rerun(result)
            }
            rerun_ids = {
                scenario.id
                for scenario in resume_scenarios
                if _resume_result_requires_rerun(
                    next(
                        (
                            result
                            for result in prev_results
                            if result.get("scenario_id") == scenario.id
                        ),
                        {},
                    )
                )
            }

            if not prior_by_id:
                if not args.json:
                    console.print(
                        f"  [dim]ℹ No reusable scenario outcomes in run {args.resume}"
                        " — running all.[/]"
                    )
            else:
                remaining = [s for s in resume_scenarios if s.id in rerun_ids]
                if not args.json:
                    console.print(
                        f"  [bold cyan]↻ Resume:[/] preserving {len(prior_by_id)} completed outcomes "
                        f"in [dim]{args.resume}[/], "
                        f"running {len(remaining)} remaining"
                    )
                # An empty subset is intentional: service finalizes the fully
                # checkpointed interrupted run without changing any outcome.
                args.scenarios = [s.id for s in remaining]
                args._resume_remaining_scenarios = remaining
                resume_prior_results = list(prior_by_id.values())
        # Store resume_run_id on args so the run_benchmark helpers can pass it
        args._resume_run_id = args.resume
    else:
        args._resume_run_id = None
    # Store prior results on args for service merge
    args._resume_prior_results = resume_prior_results
    args._resume_scenarios = resume_scenarios
    if not hasattr(args, "_resume_remaining_scenarios"):
        args._resume_remaining_scenarios = None


def _run_scored_mode(
    target: _Target,
    throughput_samples: list[ThroughputSample],
    pressure: _PressureFill | None,
) -> None:
    """Run the tool-call scenarios through the live, JSON, or plain runner."""
    args, console = target.args, target.console
    service = BenchmarkService(
        reporter=MarkdownReporter(root=args.output_dir),
    )
    use_live = not args.json and not args.no_live
    trials = max(1, args.trials)

    _plan_resume(target, pressure)
    _resolve_diff_target(args)

    if trials > 1 and not args.json:
        console.print(f"[dim]  Running {trials} trials for statistical measurement…[/]\n")

    pressure_messages = pressure.messages if pressure is not None else None
    pressure_config = pressure.config if pressure is not None else None
    if use_live:
        _run_with_live_display(
            service,
            console,
            target.model,
            target.display_name,
            target.backend,
            target.base_url,
            target.api_key,
            args,
            throughput_samples=throughput_samples,
            extra_params=target.extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config,
            display_url=target.display_url,
            run_context=target.run_context,
            wire_format=target.wire_format,
        )
    elif args.json:
        _run_json(
            service,
            target.model,
            target.backend,
            target.base_url,
            target.api_key,
            args,
            throughput_samples=throughput_samples,
            extra_params=target.extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config,
            run_context=target.run_context,
            wire_format=target.wire_format,
        )
    else:
        _run_plain(
            service,
            console,
            target.model,
            target.display_name,
            target.backend,
            target.base_url,
            target.api_key,
            args,
            throughput_samples=throughput_samples,
            extra_params=target.extra_params or None,
            context_pressure_messages=pressure_messages,
            context_pressure_config=pressure_config,
            display_url=target.display_url,
            run_context=target.run_context,
            wire_format=target.wire_format,
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
    wire_format: str = "openai",
) -> None:
    """Run with Rich live display — the default visual mode."""
    from tool_eval_bench.runner.orchestrator import score_results

    scenarios = _execution_scenarios(args)

    trials = max(1, args.trials)
    all_summaries = []

    # --- Trial 1: with live display ---
    display = BenchmarkDisplay(
        display_name, backend, display_url or base_url, scenarios, run_context=run_context
    )
    display.start()

    async def run_trial(request: ScoredRun, *, show: bool = False) -> dict:
        callbacks: dict = {}
        if request.audits_answers:
            callbacks["on_scenario_audit"] = display.on_scenario_audit
        if show:
            callbacks["on_scenario_start"] = display.on_scenario_start
            callbacks["on_scenario_result"] = display.on_scenario_result
            callbacks["rate_limit_observer"] = display.on_rate_limit
        return await service.run_benchmark(
            **request.service_kwargs(),
            throughput_samples=throughput_samples or [],
            **callbacks,
        )

    async def run_all_trials() -> None:
        """Run all trials in a single event loop for connection reuse."""
        request = ScoredRun.from_args(
            args,
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            wire_format=wire_format,
            scenarios=scenarios,
            extra_params=extra_params,
            scenario_packs=_pack_attestations(args),
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
            run_context=run_context,
        )
        result = await run_trial(request, show=True)
        trial_results = [result]

        # When resuming, the service has already merged prior results into
        # result["scores"].  Use that merged summary for display instead of
        # re-scoring only the rerun subset from display.results (which would
        # show an inflated score — e.g. 100% from 5/5 reruns when the full
        # set was 50% on 35/69).
        has_resume = bool(getattr(args, "_resume_prior_results", None))
        merged = _score_stored_trial(result, args, scenarios) if has_resume else None

        if merged is not None:
            all_summaries.append(merged)
            display.set_finished(merged, throughput_samples=throughput_samples)
            _show_diff(console, merged.scenario_results, args)
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
                _show_diff(console, all_results, args)
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
            later = request.later_trial()
            for t in range(2, trials + 1):
                console.print(f"  [dim]Running trial {t}/{trials}\u2026[/]", end=" ")
                trial_result = await run_trial(later, show=False)
                trial_results.append(trial_result)

                # Collect report path
                trial_rp = trial_result.get("report_path")
                if trial_rp:
                    report_paths.append(str(trial_rp))

                trial_summary = _score_stored_trial(trial_result, args, scenarios)
                if trial_summary is not None:
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
                    scenario_metadata=scenario_report_metadata(scenarios),
                    scenario_packs=_pack_attestations(args),
                )
                console.print(f"  [dim]📊 Summary report: {summary_path}[/]\n")
        if _trials_safety_gate_failed(args, trial_results):
            raise SystemExit(2)

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
    throughput_samples: list | None = None,
    extra_params: dict[str, Any] | None = None,
    context_pressure_messages: list[ChatMessage] | None = None,
    context_pressure_config: dict | None = None,
    run_context: Any | None = None,
    wire_format: str = "openai",
) -> None:
    """Run and output raw JSON (with optional JSONL progress on stderr)."""
    trials = max(1, args.trials)
    resolved = _execution_scenarios(args)
    json_file = getattr(args, "json_file", None)

    async def run(request: ScoredRun) -> dict:
        return await service.run_benchmark(
            **request.service_kwargs(),
            throughput_samples=throughput_samples or [],
            on_scenario_start=_stderr_progress_start,
            on_scenario_result=_stderr_progress_result,
            on_scenario_audit=_stderr_progress_audit,
        )

    try:
        request = ScoredRun.from_args(
            args,
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            wire_format=wire_format,
            scenarios=resolved,
            extra_params=extra_params,
            scenario_packs=_pack_attestations(args),
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
            run_context=run_context,
        )
        results = [asyncio.run(run(request))]
        later = request.later_trial()
        for _t in range(1, trials):
            results.append(asyncio.run(run(later)))
    except KeyboardInterrupt:
        emit_run_failed("interrupted")
        sys.exit(1)
    except Exception as exc:
        # Shareable output like a report, so a quoted request URL is redacted.
        error_data = {"error": _redact_urls(str(exc))}
        _emit_json_output(error_data, json_file=json_file)
        sys.exit(1)

    if trials == 1:
        _emit_json_output(results[0], json_file=json_file)
    else:
        # Aggregate trial data
        scored = (_score_stored_trial(r, args, resolved) for r in results)
        summaries = [summary for summary in scored if summary is not None]

        agg = _aggregate_trials(summaries) if summaries else {}
        output = results[-1]  # last run as the primary result
        if agg:
            output["trial_statistics"] = agg
        # The envelope's top-level safety_warnings and safety_gate are what
        # consumers check, so they cover every trial; scores.safety_warnings
        # stays the last trial's.
        union = _trial_safety_warnings(results)
        output["safety_warnings"] = union
        output["safety_gate"] = {"passed": not union, "warnings": union}
        _emit_json_output(output, json_file=json_file)
    if _trials_safety_gate_failed(args, results):
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
    wire_format: str = "openai",
) -> None:
    """Run with simple line-by-line output."""
    console.print(f"\n[bold]Tool-Call Benchmark[/] — {display_name}")
    console.print(f"[dim]  Backend: {backend}  |  Server: {display_url or base_url}[/]\n")

    resolved = _execution_scenarios(args)

    trials = max(1, args.trials)
    started = time.time()

    async def run(request: ScoredRun, *, show: bool = False) -> dict:
        callbacks: dict = {}
        if request.audits_answers:
            callbacks["on_scenario_audit"] = _plain_on_audit
        if show:
            callbacks["on_scenario_start"] = _plain_on_start
            callbacks["on_scenario_result"] = _plain_on_result
        return await service.run_benchmark(
            **request.service_kwargs(),
            throughput_samples=throughput_samples or [],
            **callbacks,
        )

    try:
        request = ScoredRun.from_args(
            args,
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            wire_format=wire_format,
            scenarios=resolved,
            extra_params=extra_params,
            scenario_packs=_pack_attestations(args),
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
            run_context=run_context,
        )
        all_results_dicts = [asyncio.run(run(request, show=True))]
        later = request.later_trial()
        for t in range(2, trials + 1):
            console.print(f"\n[dim]  --- Trial {t}/{trials} ---[/]\n")
            all_results_dicts.append(asyncio.run(run(later, show=True)))
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
    if getattr(args, "diff", None):
        # The diff follows the score above, so it compares the same (last) trial.
        _show_diff(
            console,
            [ScenarioResult.from_dict(r) for r in scores.get("scenario_results") or []],
            args,
        )

    # Show trial statistics if multiple trials
    if trials > 1:
        scored = (_score_stored_trial(r, args, resolved) for r in all_results_dicts)
        summaries = [summary for summary in scored if summary is not None]
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
                scenario_metadata=scenario_report_metadata(resolved),
                scenario_packs=_pack_attestations(args),
            )
            console.print(f"  [dim]📊 Summary report: {summary_path}[/]\n")
    if _trials_safety_gate_failed(args, all_results_dicts):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
