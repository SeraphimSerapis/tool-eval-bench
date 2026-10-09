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
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv  # noqa: F401  (re-exported via _load_dotenv)
from rich.console import Console

from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
from tool_eval_bench.adapters.wire_format import resolve_wire_format as _resolve_wire_format
from tool_eval_bench.application.decision_audit import decision_judge_config, with_selected_checks
from tool_eval_bench.application.run_context import build_run_context, identify_backend
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import model_probe as _model_probe
from tool_eval_bench.cli.command_registry import PLUGIN_FLAG_STEMS
from tool_eval_bench.cli.compare_report import (
    run_compare_report_command as _run_compare_report_command,
)
from tool_eval_bench.cli.display import BenchmarkDisplay, decision_audit_line
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
from tool_eval_bench.cli.run_io import stderr_progress_audit as _stderr_progress_audit
from tool_eval_bench.cli.run_io import stderr_progress_result as _stderr_progress_result
from tool_eval_bench.cli.run_io import stderr_progress_start as _stderr_progress_start
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
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.utils.headers import attach_session_id as _attach_session_id
from tool_eval_bench.utils.headers import parse_header_env as _parse_header_env
from tool_eval_bench.utils.headers import parse_header_pairs as _parse_header_pairs
from tool_eval_bench.utils.system_prompt import MAX_SYSTEM_PROMPT_BYTES, normalize_system_prompt
from tool_eval_bench.utils.urls import endpoint_identity

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
    """Compare every user-controlled scoring condition persisted in a run."""
    from tool_eval_bench.evals.variants import apply_variants

    scenarios = apply_variants(scenarios, getattr(args, "variant_seed", None))
    judge_config = decision_judge_config(
        getattr(args, "decision_judge_base_url", None),
        getattr(args, "decision_judge_model", None),
        judge_set=getattr(args, "decision_judge", None),
    )
    current = {
        "scenario_variants": {s.id: s.variant_metadata for s in scenarios if s.variant_metadata},
        "model": model,
        "backend": backend,
        "base_url": _redact_url(base_url),
        "endpoint_id": endpoint_identity(base_url),
        "temperature": args.temperature,
        "timeout_seconds": args.timeout,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "reference_date": args.reference_date,
        "system_prompt": getattr(args, "system_prompt", None),
        "decision_judge": (
            with_selected_checks(judge_config, scenarios) if judge_config is not None else None
        ),
        "scenario_ids": [scenario.id for scenario in scenarios],
        "concurrency": args.parallel,
        "error_rate": args.error_rate,
        "alpha": args.alpha,
        "extra_params": extra_params,
        "weight_by_difficulty": getattr(args, "weight_by_difficulty", False),
        "scenario_packs": scenario_packs,
    }
    # Older persisted runs predate some fields. Validate every condition they
    # did record, while modern runs receive the full strict comparison.
    mismatches = [
        key
        for key, value in current.items()
        if (key in previous and previous[key] != value)
        or (key == "scenario_variants" and previous.get(key, {}) != value)
        # A run on the built-in prompt persists no system_prompt key at all, so
        # absence means "built-in": resuming it with an override would otherwise
        # merge two personas into one result.
        or (key in {"system_prompt", "decision_judge"} and previous.get(key) != value)
    ]
    if "decision_judge" in mismatches:
        mismatches = [key for key in mismatches if key != "decision_judge"]
        mismatches.append(
            _judge_mismatch(previous.get("decision_judge"), current["decision_judge"])
        )
    pressure = _pressure_mismatch(previous.get("context_pressure"), context_pressure)
    if pressure is not None:
        mismatches.append(pressure)
    return mismatches


def _pressure_mismatch(previous: Any, current: dict[str, Any] | None) -> str | None:
    """Name a context-pressure difference that would merge two fill levels into one run.

    A run without pressure persists no ``context_pressure`` key; the key has been
    written since the option existed, so absence always means "no pressure".

    The ratio and the fill target decide what the model sees. The detected
    ``context_size`` is not compared: a restarted server can report a slightly
    different KV capacity, and the fill target is quantised to whole filler
    chunks, so drift that leaves the fill unchanged must not block a resume.
    The calibrated ``fill_tokens`` is not compared either, because an unseeded
    run draws fresh filler and lands within the calibration tolerance, never
    on the same count twice.
    """
    if not previous and not current:
        return None
    if not previous or not current:
        was = previous["ratio"] if previous else "off"
        now = current["ratio"] if current else "off"
        return f"context_pressure (was {was}, now {now})"
    if previous.get("ratio") != current["ratio"]:
        return f"context_pressure ratio (was {previous.get('ratio')}, now {current['ratio']})"
    # Runs from before tokenizer calibration recorded no target.
    old_target = previous.get("fill_tokens_target")
    if old_target is not None and old_target != current["fill_tokens_target"]:
        return (
            f"context_pressure fill (was {old_target:,} tokens, "
            f"now {current['fill_tokens_target']:,}; the context size changed)"
        )
    return None


def _judge_mismatch(previous: Any, current: Any) -> str:
    """Name the judge difference a user must undo to resume, when it is a set or check."""
    if isinstance(previous, dict) and "set" not in previous:
        # Audited before judge sets existed, which meant TC-89 only. No set
        # reproduces that selection, so say why rather than name a bare key.
        return "decision_judge (a TC-89-only audit from an earlier version)"
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return "decision_judge"
    if previous["set"] != current["set"]:
        return f"decision_judge set (was {previous['set']}, now {current['set']})"
    old_checks = set(previous.get("checks") or [])
    new_checks = set(current.get("checks") or [])
    if old_checks == new_checks:
        # A URL or model change; the generic key already says enough.
        return "decision_judge"
    changes = []
    if removed := sorted(old_checks - new_checks):
        changes.append(f"was {', '.join(removed)}")
    if added := sorted(new_checks - old_checks):
        changes.append(f"now {', '.join(added)}")
    return f"decision_judge checks ({'; '.join(changes)})"


def _execution_scenarios(args: argparse.Namespace) -> list[ScenarioDefinition]:
    """Return the explicit resume subset, including an intentionally empty one."""
    subset = getattr(args, "_resume_remaining_scenarios", None)
    return subset if subset is not None else _resolve_scenarios(args)


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
    run_context: Any | None


def _run_throughput_mode(target: _Target) -> tuple[list, bool]:
    """Run the llama-benchy throughput sweep.

    Returns the samples and whether the CLI is finished; ``--perf-only`` writes
    its own report and stops, while ``--perf`` hands its samples to the
    tool-call run that follows.
    """
    args, console = target.args, target.console
    if not (args.perf or args.perf_only):
        return [], False

    benchy_extra: list[str] | None = None
    if args.benchy_args:
        import shlex

        benchy_extra = shlex.split(args.benchy_args)

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
        extra_args=benchy_extra,
        # We already warmed the server, so skip llama-benchy's own warm-up and
        # save two requests.
        skip_warmup=not args.no_warmup,
        tokenizer=getattr(args, "tokenizer", None),
        backend=target.backend,
    )

    if not args.perf_only:
        return [sample for sample in throughput_samples if not sample.error], False

    from tool_eval_bench.utils.ids import build_run_id

    run_config = _with_config_fingerprint(
        {
            "model": target.model,
            "backend": target.backend,
            "base_url": target.base_url,
            "mode": "perf-only",
        }
    )
    run_id = build_run_id(run_config)
    reporter = MarkdownReporter(root=args.output_dir)
    report_path = reporter.write_throughput_report(
        run_id,
        target.display_name,
        throughput_samples,
        run_context=target.run_context,
    )
    failed_count = sum(bool(sample.error) for sample in throughput_samples)
    successful_count = len(throughput_samples) - failed_count
    scores = {"samples": len(throughput_samples)}
    if failed_count:
        scores.update({"successful": successful_count, "failed": failed_count})
    _persist_plugin_run(
        {
            "run_id": run_id,
            "run_type": "perf",
            "status": "failed" if failed_count else "completed",
            "config": run_config,
            "scores": scores,
            "metadata": _metadata_for_storage(target.run_context),
            "report_path": str(report_path),
        }
    )
    console.print(f"\n  [dim]Report saved to {report_path}[/]\n")
    if failed_count:
        console.print(f"[bold red]Throughput benchmark failed in {failed_count} cell(s).[/]")
        sys.exit(1)
    return throughput_samples, True


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
    base_url = getattr(args, "decision_judge_base_url", None)
    if base_url is None:
        return {}
    return {
        "decision_judge_base_url": base_url,
        "decision_judge_model": getattr(args, "decision_judge_model", None),
        "decision_judge_api_key": os.environ.get("TOOL_EVAL_DECISION_JUDGE_API_KEY"),
        "decision_judge": getattr(args, "decision_judge", None),
    }


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


def _sends_system_prompt(args: argparse.Namespace) -> bool:
    """Whether this invocation runs tool-call scenarios, the override's only consumer.

    Mirrors the routing in ``main()``: ``--perf-only``, ``--spec-live``, a lone
    ``--spec-bench``, any ``--<plugin>-only`` run, and ``--skip-tool-eval`` all
    stop before the scenarios; a context-pressure sweep is a scenario run.

    Probe, ``--dry-run``, and the storage commands are not considered here: they
    record nothing, so an unused flag is inert, and they already accept the rest
    of the run-control group in silence.
    """
    if args.spec_live or args.decision_live or args.perf_only:
        return False
    other_benchmarks = args.perf or any(
        getattr(args, stem) or getattr(args, f"{stem}_only") for stem in PLUGIN_FLAG_STEMS.values()
    )
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
            runs=args.spec_runs,
            temperature=args.temperature,
            custom_prompts=custom_prompts,
            output_dir=args.output_dir,
            metadata_for_storage=_metadata_for_storage,
            with_config_fingerprint=_with_config_fingerprint,
            persist_plugin_run=_persist_plugin_run,
            label=args.label,
            run_context=target.run_context,
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
            and not args.needle
            and not args.needle_only
            and not args.decision_bench
            and not args.decision_bench_only
        ):
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
            parse_sweep_range=_parse_sweep_range,
            resolve_scenarios=_resolve_scenarios,
            with_config_fingerprint=_with_config_fingerprint,
            persist_plugin_run=_persist_plugin_run,
            metadata_for_storage=_metadata_for_storage,
            label=args.label,
            run_context=target.run_context,
        )
        return True
    return False


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

    # Resolve the system-prompt override from its file, if given, so every
    # downstream consumer (run, resume-compat check, RunContext) sees plain
    # text on args.system_prompt.
    _resolve_system_prompt(args, parser)
    _drop_unused_system_prompt(args, console)
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

    # Validate explicit scenario IDs before server discovery or benchmark
    # requests. This keeps a typo from turning into an empty run or a server
    # failure. Probe/history/dry-run keep their own command semantics.
    if not args.probe:
        try:
            _resolve_scenarios(args)
        except ValueError as exc:
            parser.error(str(exc))

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
        request_headers = {**env_headers, **_parse_header_pairs(args.header)}
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

    # --probe: check if server is reachable and exit
    if args.probe:
        _probe_server(
            console,
            base_url,
            api_key,
            headless=args.json,
            wire_format=wire_format,
            headers=probe_headers,
        )
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
            wire_format=wire_format,
            headers=probe_headers,
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

    _validate_scenario_selection(args, parser, console)

    # -- spec-live: standalone live monitor (exits after session) --
    if args.spec_live:
        # Map CLI choice names to internal method identifiers
        _method_map = {
            "draft": "draft_model",
            "standalone": "draft_model",
            "nextn": "mtp",
        }
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

    # -- decision-live: standalone canary monitor (exits after session) --
    if args.decision_live:
        from tool_eval_bench.cli.decision_live_display import run_decision_live
        from tool_eval_bench.cli.helpers import adapter_options
        from tool_eval_bench.domain.decision import DecisionUnsupportedError
        from tool_eval_bench.plugins.decision.typed_decisions import DatasetIntegrityError

        try:
            asyncio.run(
                run_decision_live(
                    base_url,
                    model=model,
                    api_key=api_key,
                    metrics_url=args.metrics_url,
                    display_url=display_url,
                    interval=args.decision_live_interval,
                    timeout_seconds=args.timeout,
                    adapter_options=adapter_options(args),
                )
            )
        except KeyboardInterrupt:
            pass
        except (DecisionUnsupportedError, DatasetIntegrityError) as exc:
            console.print(f"\n[bold red]Decision monitor error:[/] {exc}")
            sys.exit(1)
        return

    # A decision model has no chat endpoint to preflight or warm; the plugin
    # reports an unreachable or unsupported endpoint itself.
    if not args.decision_bench_only:
        _check_endpoint_ready(
            args,
            console,
            base_url=base_url,
            model=model,
            api_key=api_key,
            wire_format=wire_format,
            extra_params=extra_params,
            headers=probe_headers,
        )

    run_context = _build_run_context(
        args,
        console,
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        extra_params=extra_params,
    )

    target = _Target(
        args=args,
        parser=parser,
        console=console,
        model=model,
        display_name=display_name,
        backend=backend,
        base_url=base_url,
        display_url=display_url,
        api_key=api_key,
        wire_format=wire_format,
        extra_params=extra_params,
        run_context=run_context,
    )

    throughput_samples, finished = _run_throughput_mode(target)
    if finished:
        return

    if _run_spec_bench_mode(target):
        return

    if _run_pressure_sweep_mode(target):
        return

    # -- Context pressure --
    pressure_messages: list[ChatMessage] | None = None
    pressure_config_dict: dict | None = None
    if args.context_pressure is not None:
        from rich.progress import BarColumn, Progress, TextColumn

        from tool_eval_bench.runner.context_pressure import (
            build_pressure_messages,
            calibrate_pressure_messages,
            llamacpp_reported_context,
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
                    client_factory=HTTPMeasurementClient,
                    reported_context=llamacpp_reported_context(run_context),
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
                    client_factory=HTTPMeasurementClient,
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
            "needle": _run_needle_benchmark,
            "decision": _run_decision_benchmark,
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
            or args.needle
            or args.needle_only
            or args.decision_bench
            or args.decision_bench_only
        )
        if not any_benchmark:
            console.print(
                "\n  [yellow]⚠ --skip-tool-eval has no effect without "
                "--perf, --perf-only, --spec-bench, --gsm8k, --mmlu, --ifeval, "
                "--needle, or --decision-bench.[/]\n"
            )
        return

    # -- Tool-call scenarios --
    service = BenchmarkService(
        reporter=MarkdownReporter(root=args.output_dir),
    )
    use_live = not args.json and not args.no_live
    trials = max(1, args.trials)

    # -- Resume: preserve completed model outcomes under the original run ID --
    resume_prior_results: list[dict] | None = None
    resume_scenarios: list[ScenarioDefinition] | None = None
    if args.resume:
        from tool_eval_bench.application.run_queries import resume_state

        resumable = resume_state(args.resume)
        prev_run = resumable[0] if resumable else None
        prev_checkpoints = resumable[1] if resumable else []
        if prev_run is None:
            console.print(
                f"\n  [bold red]✗[/] Run '{args.resume}' not found in history.\n"
                "  [dim]Use --history to list available runs.[/]\n"
            )
            sys.exit(1)

        if prev_run.get("status") == "completed":
            console.print(
                "\n  [bold red]✗ Resume aborted: run is already completed[/]\n"
                "  [dim]Completed scenario outcomes are immutable. Start a fresh run to retry.[/]\n"
            )
            sys.exit(1)

        # Resolve before changing args.scenarios.  These definitions are needed
        # to rescore checkpointed Hard Mode and held-out-pack results.
        resume_scenarios = _resolve_scenarios(args)

        # Validate every persisted benchmark condition, not merely model/backend.
        prev_config = prev_run.get("config") or {}
        mismatches = _resume_config_mismatches(
            prev_config,
            model=model,
            backend=backend,
            base_url=base_url,
            scenarios=resume_scenarios,
            args=args,
            extra_params=extra_params or None,
            scenario_packs=_pack_attestations(args),
            context_pressure=pressure_config_dict,
        )
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
            wire_format=wire_format,
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
            wire_format=wire_format,
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
            wire_format=wire_format,
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

    async def run_trial(*, show: bool = False) -> dict:
        callbacks: dict = _decision_judge_kwargs(args)
        if callbacks:
            callbacks["on_scenario_audit"] = display.on_scenario_audit
        if show:
            callbacks["on_scenario_start"] = display.on_scenario_start
            callbacks["on_scenario_result"] = display.on_scenario_result
            callbacks["rate_limit_observer"] = display.on_rate_limit
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
            system_prompt=getattr(args, "system_prompt", None),
            variant_seed=getattr(args, "variant_seed", None),
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
            resume_scenarios=getattr(args, "_resume_scenarios", None),
            scenario_packs=_pack_attestations(args),
            wire_format=wire_format,
            extra_headers=getattr(args, "_request_headers", None),
            session_header=getattr(args, "_session_header", None),
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
            resume_defs = getattr(args, "_resume_scenarios", None) or []
            known_defs = {scenario.id: scenario for scenario in resume_defs}
            known_defs.update(
                {
                    scenario.id: scenario
                    for scenario in _resolve_all_scenarios_for_ids(
                        [sr.scenario_id for sr in merged_sr]
                    )
                }
            )
            merged_scenario_defs = [known_defs[sr.scenario_id] for sr in merged_sr]
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
    wire_format: str = "openai",
) -> None:
    """Run and output raw JSON (with optional JSONL progress on stderr)."""
    trials = max(1, args.trials)
    resolved = _execution_scenarios(args)
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
            system_prompt=getattr(args, "system_prompt", None),
            variant_seed=getattr(args, "variant_seed", None),
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
            resume_scenarios=getattr(args, "_resume_scenarios", None),
            scenario_packs=_pack_attestations(args),
            wire_format=wire_format,
            extra_headers=getattr(args, "_request_headers", None),
            session_header=getattr(args, "_session_header", None),
            **_decision_judge_kwargs(args),
            on_scenario_start=_stderr_progress_start,
            on_scenario_result=_stderr_progress_result,
            on_scenario_audit=_stderr_progress_audit,
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

        resolved_sc = getattr(args, "_resume_scenarios", None) or _execution_scenarios(args)
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
    wire_format: str = "openai",
) -> None:
    """Run with simple line-by-line output."""
    console.print(f"\n[bold]Tool-Call Benchmark[/] — {display_name}")
    console.print(f"[dim]  Backend: {backend}  |  Server: {display_url or base_url}[/]\n")

    resolved = _execution_scenarios(args)

    trials = max(1, args.trials)
    started = time.time()

    async def run(*, show: bool = False) -> dict:
        callbacks: dict = _decision_judge_kwargs(args)
        if callbacks:
            callbacks["on_scenario_audit"] = _plain_on_audit
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
            system_prompt=getattr(args, "system_prompt", None),
            variant_seed=getattr(args, "variant_seed", None),
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
            resume_scenarios=getattr(args, "_resume_scenarios", None),
            scenario_packs=_pack_attestations(args),
            wire_format=wire_format,
            extra_headers=getattr(args, "_request_headers", None),
            session_header=getattr(args, "_session_header", None),
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
