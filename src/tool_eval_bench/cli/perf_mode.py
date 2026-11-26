"""Throughput benchmark planning, execution, and persistence for ``--perf`` / ``--perf-only``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.text import Text

from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.cli.headless import HeadlessConsole, report_run_failed, report_run_saved
from tool_eval_bench.cli.modes import any_plugin_selected as _any_plugin_selected
from tool_eval_bench.cli.modes import sends_system_prompt as _sends_system_prompt
from tool_eval_bench.cli.perf import PerfPassFailed, _resolve_benchy_tokenizer
from tool_eval_bench.cli.perf import run_llama_benchy as _run_llama_benchy
from tool_eval_bench.cli.resolve import parse_int_list as _parse_int_list
from tool_eval_bench.cli.resolve import redact_url as _redact_url
from tool_eval_bench.cli.session_setup import _Target
from tool_eval_bench.runner.perf_presets import (
    DEFAULT_PERF_PRESET,
    PerfPass,
    PerfPlan,
    depth0_speeds,
    estimate_pass_seconds,
    failed_cell_count,
    format_duration,
    plan_perf,
)
from tool_eval_bench.storage.reports.throughput import throughput_report

if TYPE_CHECKING:
    from tool_eval_bench.runner.throughput import ThroughputSample

logger = logging.getLogger("tool_eval_bench.cli.dispatch")


def _default_scored_run_follows(args: argparse.Namespace) -> bool:
    return _sends_system_prompt(args) and args.context_pressure_sweep is None


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


def _perf_plan(target: _Target) -> PerfPlan:
    """Resolve ``--perf-preset`` and any explicit workload flags into passes.

    The served window comes from the run context; ``long`` drops the depths
    that do not fit it. The plan is a pure function of the target, so the run
    and the stored config always agree.
    """
    args = target.args
    run_context = getattr(target, "run_context", None)
    window = run_context.max_model_len if run_context is not None else None
    return plan_perf(
        getattr(args, "perf_preset", DEFAULT_PERF_PRESET),
        pp=args.pp,
        tg=args.tg,
        runs=args.benchy_runs,
        depths=None if args.depth is None else _parse_int_list(args.depth),
        concurrency=None if args.concurrency is None else _parse_int_list(args.concurrency),
        context_window=window if type(window) is int and window > 0 else None,
    )


def _stored_passes(plan: PerfPlan) -> list[dict[str, Any]] | None:
    """The ``passes`` a run config records, or None for a single-pass sweep."""
    if len(plan.passes) < 2:
        return None
    return [perf_pass.to_config() for perf_pass in plan.passes]


def _throughput_config(target: _Target) -> dict[str, Any]:
    """The stored config of a throughput run that no scored run carries.

    The workload fields keep different sweeps out of one fingerprint cohort.
    ``--benchy-args`` is stored with credentials stripped, because it can
    carry ``--api-key``.

    ``perf_preset`` is stored but not fingerprinted: the concrete grid already
    is, and a short-preset run must keep the fingerprint it had before presets
    existed. It reads ``custom`` when explicit flags changed the preset's grid.
    ``passes`` appears only when the sweep ran more than one llama-benchy
    invocation, because ``depths`` and ``concurrency`` alone would describe a
    full cross product that never ran.
    """
    from tool_eval_bench.runner.llama_benchy import redact_arguments

    args = target.args
    plan = target.perf_sweep.plan if target.perf_sweep is not None else _perf_plan(target)
    config: dict[str, Any] = {
        "model": target.model,
        "backend": target.backend,
        "base_url": target.base_url,
        "mode": "perf-only",
        "pp": plan.pp,
        "tg": plan.tg,
        "depths": plan.depths,
        "concurrency": plan.concurrency,
        "runs": plan.runs,
        "latency_mode": args.benchy_latency_mode,
        # The tokenizer builds the prompts and counts their tokens.
        "tokenizer": _stored_tokenizer(getattr(args, "tokenizer", None)),
        "benchy_args": redact_arguments(_benchy_extra_args(args) or []),
        "perf_preset": plan.stored_preset,
    }
    if len(plan.passes) > 1:
        config["passes"] = _stored_passes(plan)
    return config


def _depth_list(depths: tuple[int, ...]) -> str:
    """``depth 65,536`` or ``depths 65,536 and 131,072``."""
    labels = [f"{depth:,}" for depth in depths]
    if len(labels) == 1:
        return f"depth {labels[0]}"
    return f"depths {', '.join(labels[:-1])} and {labels[-1]}"


def _announce_perf_plan(console: Console, plan: PerfPlan) -> None:
    """Warn about cells the served window cannot hold, or that it could not check."""
    if not plan.request_fits:
        console.print(
            f"  [yellow]⚠ pp {plan.pp:,} plus tg {plan.tg:,} does not fit the "
            f"{plan.context_window:,}-token context window, so even depth-0 cells will fail.[/]"
        )
    if plan.preset != "long":
        return
    if plan.context_window is None:
        console.print(
            "  [yellow]⚠ The served context window is unknown, so the depth pass keeps "
            f"every depth up to {max(plan.depths):,}. A depth that does not fit will fail.[/]"
        )
    elif plan.dropped_depths:
        console.print(
            f"  [dim]Skipping {_depth_list(plan.dropped_depths)} in the depth pass: depth plus "
            f"pp and tg does not fit the {plan.context_window:,}-token context window.[/]"
        )


def _print_pass_estimate(
    console: Console, plan: PerfPlan, perf_pass: PerfPass, samples: list[ThroughputSample]
) -> None:
    """Print a rough runtime estimate for *perf_pass* from the depth-0 speeds so far."""
    speeds = depth0_speeds(samples)
    if speeds is None:
        console.print(
            "  [dim]No single-stream depth-0 result, so no runtime estimate for the depth pass.[/]"
        )
        return
    pp_tps, tg_tps = speeds
    seconds = estimate_pass_seconds(
        perf_pass, pp=plan.pp, tg=plan.tg, runs=plan.runs, pp_tps=pp_tps, tg_tps=tg_tps
    )
    console.print(
        f"  [bold]⏱ Depth pass estimate: at least {format_duration(seconds)}.[/] "
        f"[dim]{len(perf_pass.depths)} depths x {plan.runs + 1} runs each, the discarded "
        f"warm-up included, at the depth-0 speeds of {pp_tps:,.0f} pp t/s and "
        f"{tg_tps:,.1f} tg t/s. Both slow down as depth grows, so expect longer.[/]"
    )


def _unfinished_pass_samples(plan: PerfPlan, perf_pass: PerfPass) -> list[ThroughputSample]:
    """One failed sample per cell of a pass that stopped without results.

    The error names the cell and is fixed text rather than llama-benchy's,
    because a failed cell's error is shown in reports and the exception text
    can quote the server URL.
    """
    from tool_eval_bench.runner.throughput import ThroughputSample

    return [
        ThroughputSample(
            pp_tokens=plan.pp,
            tg_tokens=plan.tg,
            depth=depth,
            concurrency=level,
            requested_pp=plan.pp,
            requested_depth=depth,
            calibration_confidence="llama-benchy",
            error=(
                f"pp{plan.pp} tg{plan.tg} @ d{depth} c{level}: not measured, "
                "llama-benchy stopped before this cell finished"
            ),
        )
        for depth in perf_pass.depths
        for level in perf_pass.concurrency
    ]


@dataclass(frozen=True)
class _PerfSweep:
    """What :func:`_run_perf_plan` measured.

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


def _perf_notice(console: Console, markup: str) -> None:
    """Print a throughput warning, or log it under ``--json`` so stderr carries it.

    A ``HeadlessConsole`` drops every print, and these warnings change what
    the stored run means. ``json_logging`` turns the log record into a
    ``log`` event with URLs redacted.
    """
    if isinstance(console, HeadlessConsole):
        logger.warning("%s", " ".join(Text.from_markup(markup).plain.split()))
    else:
        console.print(markup)


def _run_perf_plan(
    target: _Target,
    plan: PerfPlan,
    *,
    resolve_tokenizer_fn: Callable[..., str | None] | None = None,
    run_benchy_fn: Callable[..., list[ThroughputSample]] | None = None,
    scored_run_follows_fn: Callable[[argparse.Namespace], bool] | None = None,
) -> _PerfSweep:
    """Run every pass of *plan* and return their samples as one list.

    The depth pass is skipped when the first pass produced nothing usable: a
    server that failed every depth-0 cell will not survive 128K of context.
    A first pass that stops without results ends the CLI, as a single-pass
    sweep always has. A later pass that stops keeps what came before: its
    cells are counted as failed and the sweep is marked incomplete.
    """
    args, console = target.args, target.console
    resolve_tok = (
        resolve_tokenizer_fn if resolve_tokenizer_fn is not None else _resolve_benchy_tokenizer
    )
    run_benchy = run_benchy_fn if run_benchy_fn is not None else _run_llama_benchy
    scored_follows = (
        scored_run_follows_fn if scored_run_follows_fn is not None else _default_scored_run_follows
    )
    _announce_perf_plan(console, plan)
    tokenizer = getattr(args, "tokenizer", None)
    if len(plan.passes) > 1:
        # Resolved once, so the tokenizer line is printed once, not per pass.
        tokenizer = resolve_tok(
            console, target.model, target.display_name, target.base_url, tokenizer
        )
    samples: list[ThroughputSample] = []
    for index, perf_pass in enumerate(plan.passes):
        if index > 0:
            if not any(not sample.error for sample in samples):
                _perf_notice(
                    console,
                    "  [yellow]Skipping the depth pass: the first pass has no successful cell.[/]",
                )
                return _PerfSweep(samples, plan.first_passes(index), incomplete=True)
            _print_pass_estimate(console, plan, perf_pass, samples)
        try:
            samples.extend(
                run_benchy(
                    console,
                    target.model,
                    target.display_name,
                    target.base_url,
                    target.api_key,
                    pp=[plan.pp],
                    tg=[plan.tg],
                    depths=list(perf_pass.depths),
                    concurrency_levels=list(perf_pass.concurrency),
                    runs=plan.runs,
                    latency_mode=args.benchy_latency_mode,
                    skip_coherence=True,
                    extra_args=_benchy_extra_args(args),
                    # llama-benchy always warms up, as llama-bench does: its global
                    # warm-up requests plus a discarded first run of every cell. Its
                    # --no-warmup drops both, so each cell's first measured run would
                    # be cold. tool-eval-bench's --no-warmup covers only its own.
                    skip_warmup=False,
                    tokenizer=tokenizer,
                    backend=target.backend,
                    enable_prefix_caching=perf_pass.prefix_caching,
                    title=f"pass {index + 1}/{len(plan.passes)}" if len(plan.passes) > 1 else None,
                )
            )
        except PerfPassFailed as failure:
            has_downstream = not args.perf_only and (
                scored_follows(args)
                or args.spec_bench
                or args.context_pressure_sweep is not None
                or _any_plugin_selected(args)
            )
            if index == 0 and (failure.interrupted or not has_downstream):
                report_run_failed(console, failure.markup)
                sys.exit(1)
            unfinished = _unfinished_pass_samples(plan, perf_pass)
            # Ctrl+C is reported once the earlier passes are saved.
            reason = "" if failure.interrupted else f"{failure.markup.strip()}\n"
            if index == 0:
                _perf_notice(
                    console,
                    f"{reason}  [yellow]⚠ Throughput pass stopped. Counting its "
                    f"{len(unfinished)} cell(s) as failed and continuing.[/]",
                )
            else:
                _perf_notice(
                    console,
                    f"{reason}  [yellow]⚠ Pass {index + 1} stopped. Keeping the earlier passes "
                    f"and counting its {len(unfinished)} cell(s) as failed.[/]",
                )
            samples.extend(unfinished)
            return _PerfSweep(
                samples,
                plan.first_passes(index + 1),
                incomplete=True,
                interrupted=failure.interrupted,
            )
    return _PerfSweep(samples, plan)


def _save_throughput(
    target: _Target,
    throughput_samples: list[ThroughputSample],
    *,
    throughput_report_fn: Callable[..., Any] | None = None,
) -> int:
    """Persist a throughput sweep as its own ``perf`` run and return its failed-cell count.

    Used by ``--perf-only`` and by ``--perf`` when no scored run follows to
    carry the samples (spec-bench, a sweep, a plugin-only run, or
    ``--skip-tool-eval``). Both store the same config, so the same measurement
    lands in the same fingerprint cohort. The caller decides when a failed
    cell ends the CLI; see :func:`_exit_if_throughput_failed`.

    ``samples`` and ``successful`` count result rows, of which a prefix-cached
    cell has two; ``failed`` counts grid cells, as every message does.
    ``incomplete`` is present, and True, when a planned pass did not finish.
    """
    args, console = target.args, target.console
    report_builder = throughput_report_fn if throughput_report_fn is not None else throughput_report
    sweep = target.perf_sweep
    failed_count = failed_cell_count(throughput_samples)
    incomplete = sweep is not None and sweep.incomplete
    successful_count = sum(not sample.error for sample in throughput_samples)
    scores: dict[str, Any] = {"samples": len(throughput_samples)}
    if failed_count:
        scores.update({"successful": successful_count, "failed": failed_count})
    if incomplete:
        scores["incomplete"] = True
    # Failed cells stay a count: their error text can quote the server URL.
    scores["results"] = [sample.to_result() for sample in throughput_samples if not sample.error]
    run_context = target.run_context
    run = ModeRun(
        run_type="perf",
        config=_throughput_config(target),
        scores=scores,
        status="failed" if failed_count or incomplete else "completed",
        unfingerprinted=("perf_preset",),
    )
    finalized = finalize_mode_run(
        run,
        report_builder(
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


def _run_throughput_mode(
    target: _Target,
    *,
    resolve_tokenizer_fn: Callable[..., str | None] | None = None,
    run_benchy_fn: Callable[..., list[ThroughputSample]] | None = None,
    scored_run_follows_fn: Callable[[argparse.Namespace], bool] | None = None,
    throughput_report_fn: Callable[..., Any] | None = None,
) -> tuple[list, bool]:
    """Run the llama-benchy throughput sweep.

    Returns the samples and whether the CLI is finished; ``--perf-only`` writes
    its own report and stops, while ``--perf`` hands its samples to whatever
    runs next. ``_run_cli`` persists them itself when no scored run follows.
    A multi-pass preset still yields one sample list and one stored run.
    Ctrl+C during a later pass saves the earlier passes on their own and
    exits 1, whatever mode was meant to follow.

    The sweep's stored preset, passes, and completeness are left on ``args``
    for ``ScoredRun.from_args``, so a scored run records the plan that ran.
    """
    args = target.args
    if not (args.perf or args.perf_only):
        return [], False

    scored_follows = (
        scored_run_follows_fn if scored_run_follows_fn is not None else _default_scored_run_follows
    )
    sweep = _run_perf_plan(
        target,
        _perf_plan(target),
        resolve_tokenizer_fn=resolve_tokenizer_fn,
        run_benchy_fn=run_benchy_fn,
        scored_run_follows_fn=scored_follows,
    )
    target.perf_sweep = sweep
    args._perf_preset = sweep.plan.stored_preset
    args._perf_passes = _stored_passes(sweep.plan)
    args._perf_incomplete = sweep.incomplete
    throughput_samples = sweep.samples

    if sweep.interrupted:
        _save_throughput(target, throughput_samples, throughput_report_fn=throughput_report_fn)
        skipped = (
            " The tool-call scenarios were not run."
            if not args.perf_only and scored_follows(args)
            else ""
        )
        report_run_failed(
            target.console,
            f"[bold red]Interrupted.[/] The finished throughput passes are saved.{skipped}",
        )
        sys.exit(1)

    if not args.perf_only:
        if sweep.failed_cells and scored_follows(args):
            # The scored run keeps the status its scenarios earn; the sweep's
            # failure is recorded beside it, so say so here.
            state = ", the sweep is incomplete" if sweep.incomplete else ""
            _perf_notice(
                target.console,
                f"  [yellow]⚠ Throughput: {sweep.failed_cells} cell(s) failed{state}. The scored "
                "run records this under its throughput results.[/]",
            )
        # Failed cells too: the stored row counts them, and every report and
        # display of the samples drops them itself.
        return throughput_samples, False

    _exit_if_throughput_failed(
        target,
        _save_throughput(target, throughput_samples, throughput_report_fn=throughput_report_fn),
    )
    return throughput_samples, True
