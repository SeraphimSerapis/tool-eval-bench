"""Resume validation and checkpoint recovery planning for ``--resume``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from typing import Any

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.adapters.wire_format import resolve_wire_format as _resolve_wire_format
from tool_eval_bench.application.resume import prior_results_for_resume
from tool_eval_bench.application.resume import (
    result_requires_rerun as _resume_result_requires_rerun,
)
from tool_eval_bench.application.run_config import build_run_config, resume_mismatches
from tool_eval_bench.cli.headless import report_run_failed
from tool_eval_bench.cli.resolve import resolve_packs as _resolve_packs
from tool_eval_bench.cli.resolve import resolve_scenarios as _resolve_scenarios
from tool_eval_bench.cli.scored_run import ScoredRun
from tool_eval_bench.cli.session_setup import _PressureFill, _Target
from tool_eval_bench.domain.scenarios import ScenarioDefinition


def _default_pack_attestations(args: Any) -> list[dict[str, Any]] | None:
    packs = _resolve_packs(args)
    return [pack.to_dict() for pack in packs] or None


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
    wire_format: str | None = None,
) -> list[str]:
    """Compare every user-controlled scoring condition persisted in a run.

    The current config is built the way the service builds the one it stores,
    so both sides share one schema and the comparison cannot drift from it.
    ``wire_format`` is the run's ``--format`` choice; None detects it from
    ``base_url``. It decides how the endpoint identity canonicalises the URL.
    """
    resolved_format = _resolve_wire_format(wire_format, base_url)
    request = ScoredRun.from_args(
        args,
        model=model,
        backend=backend,
        base_url=base_url,
        # Never reaches the stored config.
        api_key=None,
        wire_format=resolved_format,
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
        wire_format=resolved_format,
    )


def _execution_scenarios(
    args: argparse.Namespace,
    *,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
) -> list[ScenarioDefinition]:
    """Return the explicit resume subset, including an intentionally empty one."""
    subset = getattr(args, "_resume_remaining_scenarios", None)
    if subset is not None:
        return subset
    resolver = resolve_scenarios_fn if resolve_scenarios_fn is not None else _resolve_scenarios
    return resolver(args)


def _load_resumable(
    args: argparse.Namespace, console: Console
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the ``--resume`` run and its checkpoints; exit 1 when it cannot resume."""
    from tool_eval_bench.application.run_queries import resume_state

    resumable = resume_state(args.resume)
    if resumable is None:
        report_run_failed(
            console,
            f"\n  [bold red]✗[/] Run '{args.resume}' not found in history.\n"
            "  [dim]Use --history to list available runs.[/]\n",
        )
        sys.exit(1)
    prev_run, prev_checkpoints = resumable
    if prev_run.get("status") == "completed":
        report_run_failed(
            console,
            "\n  [bold red]✗ Resume aborted: run is already completed[/]\n"
            "  [dim]Completed scenario outcomes are immutable. Start a fresh run to retry.[/]\n",
        )
        sys.exit(1)
    return prev_run, prev_checkpoints


def _resumed_session_id(prev_run: Mapping[str, Any]) -> str | None:
    """The session a ``--resume`` invocation joins, or None to start a new one.

    A resume finishes the interrupted invocation's scored run under its
    original run ID, so it keeps that invocation's session id too. The plugin,
    standalone perf, and other rows written before the interruption then share
    a session with the completed scored run, and a card of that session shows
    them together. Throughput from ``--perf`` alongside scenarios is the
    exception: those samples ride in memory until the scored run is saved, so
    an interruption loses them. Any rows the resume invocation writes join the
    same session. A run stored before session ids existed has none, so the
    resume starts a fresh session.

    The column wins over the metadata copy: a write without a session, such as
    a Python API resume, replaces the metadata but keeps the column.
    """
    session_id = prev_run.get("session_id") or (prev_run.get("metadata") or {}).get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def _plan_resume(
    target: _Target,
    pressure: _PressureFill | None,
    *,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
    pack_attestations_fn: Callable[[Any], list[dict[str, Any]] | None] | None = None,
) -> None:
    """Preserve completed model outcomes under the original run ID.

    Leaves the plan on ``args._resume_*`` (and narrows ``args.scenarios`` to
    the rerun subset), where ``ScoredRun.from_args`` reads it; exits 1 when the
    run cannot be resumed.
    """
    args, console = target.args, target.console
    resolver = resolve_scenarios_fn if resolve_scenarios_fn is not None else _resolve_scenarios
    pack_attest = (
        pack_attestations_fn if pack_attestations_fn is not None else _default_pack_attestations
    )
    resume_prior_results: list[dict] | None = None
    resume_scenarios: list[ScenarioDefinition] | None = None
    if args.resume:
        prev_run, prev_checkpoints = _load_resumable(args, console)

        # Resolve before changing args.scenarios.  These definitions are needed
        # to rescore checkpointed Hard Mode and held-out-pack results.
        resume_scenarios = resolver(args)

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
            scenario_packs=pack_attest(args),
            context_pressure=pressure.config if pressure is not None else None,
            wire_format=target.wire_format,
        )
        if mismatches:
            report_run_failed(
                console,
                f"\n  [bold red]✗ Resume aborted: configuration mismatch[/]\n"
                f"  [dim]Prior run differs in: {escape(', '.join(mismatches))}[/]\n"
                f"  [dim]Start a fresh run instead of resuming.[/]\n",
            )
            sys.exit(1)

        escaped_resume = escape(str(args.resume))
        prev_results = prior_results_for_resume(prev_run, prev_checkpoints)
        if prev_checkpoints and not args.json:
            console.print(
                f"  [dim]ℹ Recovered {len(prev_checkpoints)} checkpointed scenario(s) from "
                f"interrupted run {escaped_resume}.[/]"
            )
        if not prev_results:
            if not args.json:
                console.print(
                    f"  [dim]ℹ No scenario results in run {escaped_resume} — running all.[/]"
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
                        f"  [dim]ℹ No reusable scenario outcomes in run {escaped_resume}"
                        " — running all.[/]"
                    )
            else:
                remaining = [s for s in resume_scenarios if s.id in rerun_ids]
                if not args.json:
                    console.print(
                        f"  [bold cyan]↻ Resume:[/] preserving {len(prior_by_id)} completed outcomes "
                        f"in [dim]{escaped_resume}[/], "
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
