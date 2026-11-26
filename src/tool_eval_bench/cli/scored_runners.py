"""Scored tool-call scenario runners (live Rich, headless JSON, and plain text)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.cli.display import BenchmarkDisplay, decision_audit_line
from tool_eval_bench.cli.headless import emit_run_failed
from tool_eval_bench.cli.held_out import held_out_ids as _held_out_ids
from tool_eval_bench.cli.held_out import redact_run as _redact_run
from tool_eval_bench.cli.helpers import trial_safety_warnings as _trial_safety_warnings
from tool_eval_bench.cli.helpers import trials_safety_gate_failed as _trials_safety_gate_failed
from tool_eval_bench.cli.history import print_diff as _print_diff
from tool_eval_bench.cli.resolve import (
    resolve_all_scenarios_for_ids as _resolve_all_scenarios_for_ids,
)
from tool_eval_bench.cli.resolve import resolve_packs as _resolve_packs
from tool_eval_bench.cli.resume_mode import _execution_scenarios, _plan_resume
from tool_eval_bench.cli.run_io import aggregate_trials as _aggregate_trials
from tool_eval_bench.cli.run_io import emit_json_output as _emit_json_output
from tool_eval_bench.cli.run_io import stderr_progress_audit as _stderr_progress_audit
from tool_eval_bench.cli.run_io import stderr_progress_result as _stderr_progress_result
from tool_eval_bench.cli.run_io import stderr_progress_start as _stderr_progress_start
from tool_eval_bench.cli.run_io import (
    stderr_progress_start_unredacted as _stderr_progress_start_unredacted,
)
from tool_eval_bench.cli.scored_run import ScoredRun
from tool_eval_bench.cli.session_setup import _PressureFill, _Target
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.domain.scenarios import (
    AuditPhase,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
    scenario_report_metadata,
)
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.utils.urls import redact_urls as _redact_urls

if TYPE_CHECKING:
    from tool_eval_bench.application.service import BenchmarkService
    from tool_eval_bench.runner.throughput import ThroughputSample

logger = logging.getLogger("tool_eval_bench.cli.dispatch")

# ---------------------------------------------------------------------------
# Plain-text fallback formatting (for --no-live)
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


def _plain_verdict(result: ScenarioResult) -> str:
    style = STATUS_STYLE.get(result.status, "?")
    cap = (
        " [turn cap]"
        if result.turn_budget_exceeded and result.status != ScenarioStatus.PASS
        else ""
    )
    return f"{style}  ({result.points}/2){cap} {DIM}{result.summary}{RESET}"


async def _plain_on_start(scenario: ScenarioDefinition, idx: int, total: int) -> None:
    print(
        f"  {DIM}[{idx + 1}/{total}]{RESET} {scenario.id} {scenario.title}... ", end="", flush=True
    )


async def _plain_on_result(
    scenario: ScenarioDefinition, result: ScenarioResult, idx: int, total: int
) -> None:
    print(_plain_verdict(result))


async def _plain_on_parallel_result(
    scenario: ScenarioDefinition, result: ScenarioResult, done: int, total: int
) -> None:
    # Parallel scenarios finish out of start order, so a verdict appended to an
    # open start line lands next to another scenario's title (#311). Print the
    # whole line on completion instead; ``done`` counts finished scenarios.
    print(
        f"  {DIM}[{done + 1}/{total}]{RESET} {scenario.id} {scenario.title}... "
        f"{_plain_verdict(result)}",
        flush=True,
    )


async def _plain_on_audit(
    scenario: ScenarioDefinition, result: ScenarioResult, phase: AuditPhase
) -> None:
    line = decision_audit_line(scenario.id, result, phase)
    if line is not None:
        print(line.plain, flush=True)


# ---------------------------------------------------------------------------
# Scoring, diff, attestations, and multi-trial reporting
# ---------------------------------------------------------------------------


def _report_model_label(display_name: str, args: argparse.Namespace) -> str:
    """The model a cross-trial summary names: the pair when a reviewer took part."""
    reviewer = getattr(args, "_reviewer", None)
    return f"{display_name} + reviewer {reviewer.model}" if reviewer is not None else display_name


def _pack_attestations(args: Any) -> list[dict[str, Any]] | None:
    """Content-hash records for any held-out packs in this run, or None.

    Recorded in the run config so a published score can be tied to a specific
    held-out set without publishing the set itself.
    """
    packs = _resolve_packs(args)
    return [pack.to_dict() for pack in packs] or None


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


def _show_diff(
    console: Console,
    results: list[ScenarioResult],
    args: argparse.Namespace,
    *,
    print_diff_fn: Callable[..., None] | None = None,
) -> None:
    """Print the ``--diff`` table against the run fixed by ``_resolve_diff_target``."""
    if not getattr(args, "diff", None):
        return
    target = getattr(args, "_diff_target", None)
    if target is None:
        console.print("\n  [yellow]No previous runs found for comparison.[/]\n")
        return
    printer = print_diff_fn if print_diff_fn is not None else _print_diff
    printer(console, results, target)


def _print_trials_summary(console: Console, agg: dict) -> None:
    """Print aggregated trial statistics."""
    if not agg:
        return

    from rich.panel import Panel

    n = agg["trials"]
    score_mean = agg["final_score_mean"]
    score_std = agg["final_score_stddev"]
    ci = agg["final_score_ci95"]
    median = agg["final_score_median"]

    if score_mean is None:
        score_line = "  [bold]Score:[/]   [yellow]Withheld[/] [dim](no gradable scenarios)[/]\n"
        median_line = "  [bold]Median:[/]  —\n"
        ci_line = "  [bold]95% CI:[/]  —\n"
    else:
        std_val = 0.0 if score_std is None else score_std
        score_line = f"  [bold]Score:[/]   {score_mean:.1f} ± {std_val:.1f} / 100\n"
        median_line = (
            f"  [bold]Median:[/]  {median:.1f}\n"
            if median is not None
            else "  [bold]Median:[/]  —\n"
        )
        ci_line = (
            f"  [bold]95% CI:[/]  [{ci[0]:.1f}, {ci[1]:.1f}]\n"
            if ci is not None
            else "  [bold]95% CI:[/]  —\n"
        )

    content = (
        f"  [bold]Trials:[/]  {n}\n"
        f"{score_line}"
        f"{median_line}"
        f"{ci_line}"
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


def _write_trials_summary_report(
    console: Console,
    *,
    run_id_base: str,
    display_name: str,
    args: argparse.Namespace,
    summaries: list[ModelScoreSummary],
    agg: dict[str, Any],
    throughput_samples: list | None,
    report_paths: list[str],
    run_context: Any | None,
    scenarios: list[ScenarioDefinition],
    reporter_cls: type[MarkdownReporter] | None = None,
    pack_attestations_fn: Callable[[Any], list[dict[str, Any]] | None] | None = None,
) -> None:
    """Write and announce the consolidated multi-trial Markdown summary report."""
    if not agg or len(summaries) <= 1:
        return
    rep_cls = reporter_cls if reporter_cls is not None else MarkdownReporter
    pack_attest = pack_attestations_fn if pack_attestations_fn is not None else _pack_attestations
    reporter = rep_cls(root=args.output_dir)
    protocol_scenarios = getattr(args, "_resume_scenarios", None) or scenarios
    summary_path = reporter.write_summary_report(
        run_id=run_id_base,
        model=_report_model_label(display_name, args),
        summaries=summaries,
        agg=agg,
        throughput_samples=throughput_samples,
        report_paths=report_paths,
        run_context=run_context,
        scenario_metadata=scenario_report_metadata(protocol_scenarios),
        scenario_packs=pack_attest(args),
    )
    console.print(f"  [dim]📊 Summary report: {escape(str(summary_path))}[/]\n")


# ---------------------------------------------------------------------------
# Scored runners (live, JSON, plain) and entry point
# ---------------------------------------------------------------------------


def _close_service(service: Any) -> None:
    """Close a BenchmarkService (or its underlying repository) without breaking fakes."""
    close = getattr(service, "close", None)
    if callable(close):
        close()
        return
    repo = getattr(service, "repo", None)
    repo_close = getattr(repo, "close", None)
    if callable(repo_close):
        repo_close()


def _run_scored_mode(
    target: _Target,
    throughput_samples: list[ThroughputSample],
    pressure: _PressureFill | None,
    *,
    service_cls: Callable[..., Any],
    reporter_cls: type[MarkdownReporter] | None = None,
    plan_resume_fn: Callable[[_Target, _PressureFill | None], None] | None = None,
    run_live_fn: Callable[..., None] | None = None,
    run_json_fn: Callable[..., None] | None = None,
    run_plain_fn: Callable[..., None] | None = None,
) -> None:
    """Run the tool-call scenarios through the live, JSON, or plain runner."""
    args, console = target.args, target.console
    rep_cls = reporter_cls if reporter_cls is not None else MarkdownReporter
    service = service_cls(
        reporter=rep_cls(root=args.output_dir),
    )
    plan_resume = plan_resume_fn if plan_resume_fn is not None else _plan_resume
    live_runner = run_live_fn if run_live_fn is not None else _run_with_live_display
    json_runner = run_json_fn if run_json_fn is not None else _run_json
    plain_runner = run_plain_fn if run_plain_fn is not None else _run_plain
    try:
        use_live = not args.json and not args.no_live
        trials = max(1, args.trials)

        plan_resume(target, pressure)
        _resolve_diff_target(args)

        if trials > 1 and not args.json:
            console.print(f"[dim]  Running {trials} trials for statistical measurement…[/]\n")

        pressure_messages = pressure.messages if pressure is not None else None
        pressure_config = pressure.config if pressure is not None else None
        if use_live:
            live_runner(
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
            json_runner(
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
            plain_runner(
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
    finally:
        _close_service(service)


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
    display_cls: type[BenchmarkDisplay] | None = None,
    reporter_cls: type[MarkdownReporter] | None = None,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
    pack_attestations_fn: Callable[[Any], list[dict[str, Any]] | None] | None = None,
    print_diff_fn: Callable[..., None] | None = None,
    aggregate_trials_fn: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Run with Rich live display — the default visual mode."""
    from tool_eval_bench.runner.orchestrator import score_results

    scenarios = _execution_scenarios(args, resolve_scenarios_fn=resolve_scenarios_fn)
    pack_attest = pack_attestations_fn if pack_attestations_fn is not None else _pack_attestations
    agg_fn = aggregate_trials_fn if aggregate_trials_fn is not None else _aggregate_trials
    live_display_cls = display_cls if display_cls is not None else BenchmarkDisplay

    trials = max(1, args.trials)
    all_summaries = []

    # --- Trial 1: with live display ---
    reviewer = getattr(args, "_reviewer", None)
    display = live_display_cls(
        display_name,
        backend,
        display_url or base_url,
        scenarios,
        run_context=run_context,
        reviewer_model=reviewer.model if reviewer is not None else None,
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
            scenario_packs=pack_attest(args),
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
            _show_diff(console, merged.scenario_results, args, print_diff_fn=print_diff_fn)
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
                _show_diff(console, all_results, args, print_diff_fn=print_diff_fn)
            else:
                display.stop()

        # Print report path
        report_path = result.get("report_path")
        report_paths: list[str] = []
        if report_path:
            console.print(f"\n  [dim]📄 Full report: {escape(str(report_path))}[/]\n")
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
                    if trial_summary.final_score is None:
                        console.print("[bold yellow]Withheld[/]")
                    else:
                        console.print(f"[bold]{trial_summary.final_score}[/]/100")

            agg = agg_fn(all_summaries)
            _print_trials_summary(console, agg)
            _write_trials_summary_report(
                console,
                run_id_base=result.get("run_id", "summary"),
                display_name=display_name,
                args=args,
                summaries=all_summaries,
                agg=agg,
                throughput_samples=throughput_samples,
                report_paths=report_paths,
                run_context=run_context,
                scenarios=scenarios,
                reporter_cls=reporter_cls,
                pack_attestations_fn=pack_attest,
            )
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
        detail = _redact_urls(str(exc)) if args.redact_url else str(exc)
        console.print(f"\n[bold red]Error: {escape(detail)}[/]")
        sys.exit(1)


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
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
    pack_attestations_fn: Callable[[Any], list[dict[str, Any]] | None] | None = None,
    emit_json_output_fn: Callable[..., None] | None = None,
    aggregate_trials_fn: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Run and output raw JSON (with optional JSONL progress on stderr)."""
    trials = max(1, args.trials)
    resolved = _execution_scenarios(args, resolve_scenarios_fn=resolve_scenarios_fn)
    pack_attest = pack_attestations_fn if pack_attestations_fn is not None else _pack_attestations
    emit_json = emit_json_output_fn if emit_json_output_fn is not None else _emit_json_output
    agg_fn = aggregate_trials_fn if aggregate_trials_fn is not None else _aggregate_trials
    json_file = getattr(args, "json_file", None)
    held_out = _held_out_ids(args)
    on_start = (
        _stderr_progress_start_unredacted
        if getattr(args, "include_held_out", False)
        else _stderr_progress_start
    )

    async def run(request: ScoredRun) -> dict:
        return await service.run_benchmark(
            **request.service_kwargs(),
            throughput_samples=throughput_samples or [],
            on_scenario_start=on_start,
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
            scenario_packs=pack_attest(args),
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
        message = _redact_urls(str(exc))
        # The error envelope predates run_failed and stays for consumers that
        # read it; the event is what the documented error contract promises,
        # so it is emitted even when the envelope cannot be written.
        try:
            emit_json({"error": message}, json_file=json_file, failed=True)
        except Exception as write_exc:
            message = f"{message} (and the --json-file could not be written: {write_exc})"
        emit_run_failed(message)
        sys.exit(1)

    if trials == 1:
        emit_json(_redact_run(results[0], held_out), json_file=json_file)
    else:
        # Aggregate trial data
        scored = (_score_stored_trial(r, args, resolved) for r in results)
        summaries = [summary for summary in scored if summary is not None]

        agg = agg_fn(summaries) if summaries else {}
        output = results[-1]  # last run as the primary result
        if agg:
            output["trial_statistics"] = agg
        # The envelope's top-level safety_warnings and safety_gate are what
        # consumers check, so they cover every trial; scores.safety_warnings
        # stays the last trial's.
        union = _trial_safety_warnings(results)
        output["safety_warnings"] = union
        output["safety_gate"] = {"passed": not union, "warnings": union}
        emit_json(_redact_run(output, held_out), json_file=json_file)
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
    reporter_cls: type[MarkdownReporter] | None = None,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
    pack_attestations_fn: Callable[[Any], list[dict[str, Any]] | None] | None = None,
    print_diff_fn: Callable[..., None] | None = None,
    aggregate_trials_fn: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Run with simple line-by-line output."""
    console.print(f"\n[bold]Tool-Call Benchmark[/] — {escape(display_name)}")
    console.print(
        f"[dim]  Backend: {escape(backend)}  |  Server: {escape(str(display_url or base_url))}[/]\n"
    )

    resolved = _execution_scenarios(args, resolve_scenarios_fn=resolve_scenarios_fn)
    pack_attest = pack_attestations_fn if pack_attestations_fn is not None else _pack_attestations
    agg_fn = aggregate_trials_fn if aggregate_trials_fn is not None else _aggregate_trials

    trials = max(1, args.trials)
    started = time.time()

    async def run(request: ScoredRun, *, show: bool = False) -> dict:
        callbacks: dict = {}
        if request.audits_answers:
            callbacks["on_scenario_audit"] = _plain_on_audit
        if show and args.parallel > 1:
            callbacks["on_scenario_result"] = _plain_on_parallel_result
        elif show:
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
            scenario_packs=pack_attest(args),
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
        detail = _redact_urls(str(exc)) if args.redact_url else str(exc)
        console.print(f"\n[bold red]Error: {escape(detail)}[/]")
        sys.exit(1)

    elapsed = time.time() - started
    scores = all_results_dicts[-1].get("scores", {})
    final_score = scores.get("final_score")
    score_display = "Withheld" if final_score is None else f"{final_score} / 100"
    console.print(f"\n[bold]Score: {score_display}  — {scores.get('rating', '')}[/]")
    if scores.get("weighted_score") is not None:
        console.print(
            f"[bold]Weighted Score: {scores['weighted_score']} / 100[/]  [dim](difficulty-weighted)[/]"
        )
    excluded = scores.get("excluded_scenarios") or []
    if excluded:
        console.print(
            f"[yellow]{len(excluded)} scenario(s) excluded from scoring: "
            f"{escape(', '.join(excluded))}; "
            f"completion rate {scores.get('completion_rate', 100.0)}%[/]"
        )
    console.print(f"[dim]Completed in {elapsed:.1f}s[/]\n")
    if getattr(args, "diff", None):
        # The diff follows the score above, so it compares the same (last) trial.
        _show_diff(
            console,
            [ScenarioResult.from_dict(r) for r in scores.get("scenario_results") or []],
            args,
            print_diff_fn=print_diff_fn,
        )

    # Show trial statistics if multiple trials
    if trials > 1:
        scored = (_score_stored_trial(r, args, resolved) for r in all_results_dicts)
        summaries = [summary for summary in scored if summary is not None]
        agg = agg_fn(summaries) if summaries else {}
        _print_trials_summary(console, agg)

        run_id_base = (
            all_results_dicts[0].get("run_id", "summary") if all_results_dicts else "summary"
        )
        rp_list = [str(r.get("report_path", "")) for r in all_results_dicts if r.get("report_path")]
        _write_trials_summary_report(
            console,
            run_id_base=run_id_base,
            display_name=display_name,
            args=args,
            summaries=summaries,
            agg=agg,
            throughput_samples=throughput_samples,
            report_paths=rp_list,
            run_context=run_context,
            scenarios=resolved,
            reporter_cls=reporter_cls,
            pack_attestations_fn=pack_attest,
        )
    if _trials_safety_gate_failed(args, all_results_dicts):
        raise SystemExit(2)
