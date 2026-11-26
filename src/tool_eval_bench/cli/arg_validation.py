"""Argument normalization and pre-flight flag validation for CLI runs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.application.decision_audit import (
    check_decision_judge_url,
    decision_judge_config,
)
from tool_eval_bench.cli.helpers import read_prompt_option
from tool_eval_bench.cli.modes import mode_conflict as _mode_conflict
from tool_eval_bench.cli.modes import sends_system_prompt as _sends_system_prompt
from tool_eval_bench.cli.resolve import resolve_packs as _resolve_packs
from tool_eval_bench.cli.resolve import resolve_scenarios as _resolve_scenarios
from tool_eval_bench.cli.reviewer import prepare_reviewer_args, reviewer_requested
from tool_eval_bench.cli.scored_run import (
    DECISION_JUDGE_BASE_URL_ENV,
    DECISION_JUDGE_MODEL_ENV,
)
from tool_eval_bench.domain.review import CONVERSATION_KEYS
from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition

logger = logging.getLogger("tool_eval_bench.cli.dispatch")

# Valid category letters for --categories
_VALID_CATEGORIES = {c.value for c in Category}


def _runs_a_storage_command(args: argparse.Namespace) -> bool:
    """Whether the invocation only reads or manages stored runs (history, export, ...).

    Those commands run no scenario, so a reviewer flag there would be ignored
    without a word; ``handle_local_command`` routes the same flags.
    """
    return bool(
        args.history
        or args.leaderboard
        or args.export
        or args.compare
        or getattr(args, "serve", False)
        or getattr(args, "card", None) is not None
        or getattr(args, "card_run", None)
        or getattr(args, "delete", None) is not None
    )


def _judge_requested(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in ("decision_judge", "decision_judge_base_url", "decision_judge_model")
    )


def _apply_decision_judge_env(args: argparse.Namespace) -> None:
    """Fill unset judge connection flags from ``TOOL_EVAL_DECISION_JUDGE_*``.

    Only once a judge flag asks for a judge: like ``TOOL_EVAL_BASE_URL``, the
    variables configure a connection and never switch audits on. A judge URL
    in ``.env`` would otherwise reject every command that is not a scored run.
    """
    if not _judge_requested(args):
        return
    for attr, env in (
        ("decision_judge_base_url", DECISION_JUDGE_BASE_URL_ENV),
        ("decision_judge_model", DECISION_JUDGE_MODEL_ENV),
    ):
        if getattr(args, attr, None) is None:
            setattr(args, attr, os.getenv(env) or None)


def _resolve_system_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Materialize ``--system-prompt-file`` into ``args.system_prompt``.

    Downstream consumers (run modes, resume-compat check, RunContext) read only
    ``args.system_prompt``, so the file variant is resolved once, right after
    argument parsing, and both variants leave here in the canonical form that is
    persisted and sent (see ``normalize_system_prompt``).
    """
    args.system_prompt = read_prompt_option(
        getattr(args, "system_prompt", None),
        getattr(args, "system_prompt_file", None),
        inline_flag="--system-prompt",
        file_flag="--system-prompt-file",
        parser=parser,
    )


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


def _validate_run_ranges(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject out-of-range run settings before any request.

    Checked here rather than with an argparse ``type=`` so ``--json`` reports
    them as ``invalid_arguments``. ``0 <= x <= 1`` also rejects NaN.
    """
    for flag, attr, minimum in (
        ("--max-turns", "max_turns", 1),
        ("--trials", "trials", 1),
        ("--spec-runs", "spec_runs", 1),
        ("--benchy-runs", "benchy_runs", 1),
        ("--pp", "pp", 1),
        ("--tg", "tg", 1),
        ("--context-size", "context_size", 1),
        ("--sweep-steps", "sweep_steps", 2),
        ("--needle-depths", "needle_depths", 1),
        ("--needle-lengths", "needle_lengths", 1),
        ("--gsm8k-limit", "gsm8k_limit", 0),
        ("--mmlu-limit", "mmlu_limit", 0),
        ("--ifeval-limit", "ifeval_limit", 0),
        ("--toolscale-limit", "toolscale_limit", 0),
    ):
        int_val = getattr(args, attr, None)
        if int_val is not None and int_val < minimum:
            parser.error(f"{flag} must be at least {minimum}, got {int_val}")

    gsm8k_shots = getattr(args, "gsm8k_shots", None)
    if gsm8k_shots is not None and not (0 <= gsm8k_shots <= 8):
        parser.error(f"--gsm8k-shots must be between 0 and 8, got {gsm8k_shots}")
    mmlu_shots = getattr(args, "mmlu_shots", None)
    if mmlu_shots is not None and not (0 <= mmlu_shots <= 5):
        parser.error(f"--mmlu-shots must be between 0 and 5, got {mmlu_shots}")

    for flag, attr in (
        ("--error-rate", "error_rate"),
        ("--alpha", "alpha"),
        ("--context-pressure", "context_pressure"),
        ("--min-p", "min_p"),
    ):
        value = getattr(args, attr, None)
        if value is not None and not (0.0 <= value <= 1.0):
            parser.error(f"{flag} must be between 0 and 1, got {value}")

    top_p = getattr(args, "top_p", None)
    if top_p is not None and not (0.0 < top_p <= 1.0):
        parser.error(f"--top-p must be in (0, 1], got {top_p}")

    temperature = getattr(args, "temperature", None)
    if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
        parser.error(f"--temperature must be a finite number >= 0, got {temperature}")

    for flag, attr in (
        ("--timeout", "timeout"),
        ("--repeat-penalty", "repeat_penalty"),
        ("--baseline-tgs", "baseline_tgs"),
        ("--spec-live-interval", "spec_live_interval"),
        ("--decision-live-interval", "decision_live_interval"),
    ):
        pos_val = getattr(args, attr, None)
        if pos_val is not None and (not math.isfinite(pos_val) or pos_val <= 0):
            parser.error(f"{flag} must be a finite number greater than 0, got {pos_val}")

    fill_tokens = getattr(args, "context_fill_tokens", None)
    if fill_tokens is not None:
        if fill_tokens < 1:
            parser.error(f"--context-fill-tokens must be at least 1, got {fill_tokens}")
        if getattr(args, "context_pressure", None) is not None:
            parser.error("--context-fill-tokens cannot be combined with --context-pressure")
        if getattr(args, "context_pressure_sweep", None) is not None:
            parser.error("--context-fill-tokens cannot be combined with --context-pressure-sweep")


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
    conflict = _mode_conflict(args)
    if conflict is not None:
        parser.error(conflict)

    # Resolve the system-prompt override from its file, if given, so every
    # downstream consumer (run, resume-compat check, RunContext) sees plain
    # text on args.system_prompt.
    _resolve_system_prompt(args, parser)
    _drop_unused_system_prompt(args, console)
    _validate_run_ranges(args, parser)
    _apply_decision_judge_env(args)
    judge_url = getattr(args, "decision_judge_base_url", None)
    judge_model = getattr(args, "decision_judge_model", None)
    try:
        if _judge_requested(args) and judge_url is None:
            raise ValueError(
                "A decision judge requires --decision-judge-base-url or "
                f"{DECISION_JUDGE_BASE_URL_ENV}"
            )
        if judge_url is not None and judge_model is None:
            # _resolve_target asks this URL for the model, so check it first.
            check_decision_judge_url(judge_url)
        else:
            decision_judge_config(
                judge_url, judge_model, judge_set=getattr(args, "decision_judge", None)
            )
    except ValueError as exc:
        parser.error(str(exc))
    if judge_url is not None and (
        getattr(args, "command", None) not in {None, "run", "resume"}
        or not _sends_system_prompt(args)
        or args.context_pressure_sweep is not None
    ):
        parser.error("Decision judge audits require a run or resume of tool-call scenarios")
    prepare_reviewer_args(args, parser)
    if reviewer_requested(args):
        if (
            getattr(args, "command", None) not in {None, "run", "resume"}
            or not _sends_system_prompt(args)
            or args.context_pressure_sweep is not None
            or getattr(args, "probe", False)
            or _runs_a_storage_command(args)
        ):
            parser.error("A reviewer requires a run or resume of tool-call scenarios")
        if getattr(args, "scenario_pack", None):
            parser.error(
                "A reviewer cannot be combined with --scenario-pack: it would receive "
                "the held-out scenarios"
            )
        # Checked here, before any request: the flags alone decide it.
        replaced = sorted(CONVERSATION_KEYS & _build_extra_params(args, parser).keys())
        if replaced:
            parser.error(
                f"--backend-kwargs {', '.join(replaced)} replaces the conversation, which "
                "would drop the reviewer's revision requests"
            )
    if args.resume and (not _sends_system_prompt(args) or args.context_pressure_sweep is not None):
        parser.error("--resume requires a run of tool-call scenarios")
    if getattr(args, "check_tools", False) and not args.probe:
        parser.error("--check-tools requires probe or --probe")
    if getattr(args, "check_template", False) and not args.probe:
        parser.error("--check-template requires probe or --probe")
    if getattr(args, "retest_diffs", False) and not getattr(args, "compare", None):
        parser.error("--retest-diffs requires compare <left> <right> or --compare RUN_A RUN_B")
    for retest_flag, retest_attr in (
        ("--base-url-a", "base_url_a"),
        ("--base-url-b", "base_url_b"),
        ("--api-key-a", "api_key_a"),
        ("--api-key-b", "api_key_b"),
    ):
        if getattr(args, retest_attr, None) is not None and not getattr(
            args, "retest_diffs", False
        ):
            parser.error(f"{retest_flag} requires --retest-diffs")
    # The service applies the same rule, but only after preflight, warm-up, and
    # any perf sweep. The original string is kept: it is stored and fingerprinted.
    if args.reference_date:
        try:
            if len(args.reference_date) != 10:
                raise ValueError("not YYYY-MM-DD")
            datetime.strptime(args.reference_date, "%Y-%m-%d")
        except ValueError:
            parser.error(
                f"Invalid --reference-date '{args.reference_date}'. "
                "Expected format: YYYY-MM-DD (e.g. 2026-03-20)"
            )


def _reject_unknown_categories(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    invalid = {c.upper() for c in args.categories or []} - _VALID_CATEGORIES
    if invalid:
        parser.error(
            f"Unknown categories: {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(_VALID_CATEGORIES))}"
        )


def _validate_explicit_scenarios(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
) -> None:
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
    resolver = resolve_scenarios_fn if resolve_scenarios_fn is not None else _resolve_scenarios
    try:
        resolved = resolver(args)
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


def _validate_scenario_selection(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    console: Console,
    *,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
) -> None:
    """Reject a bad pack or category before any request is made.

    Packs load up front on purpose: a missing, empty, or colliding pack must
    abort before the run starts rather than surface as a traceback partway
    through scenario resolution.
    """
    resolver = resolve_scenarios_fn if resolve_scenarios_fn is not None else _resolve_scenarios
    try:
        packs = _resolve_packs(args)
        resolver(args)
    except ValueError as exc:
        parser.error(str(exc))
    if packs and not args.json:
        total = sum(len(p.scenarios) for p in packs)
        names = ", ".join(f"{escape(p.name)} ({escape(p.content_hash)})" for p in packs)
        console.print(f"  [dim]🔒 Held-out packs: {names} — {total} scenario(s)[/]")

    if args.categories:
        _reject_unknown_categories(args, parser)
        cats = [c.upper() for c in args.categories]
        from tool_eval_bench.domain.scenarios import CATEGORY_LABELS

        cat_names = ", ".join(f"{c} ({CATEGORY_LABELS[Category(c)]})" for c in cats)
        resolved_count = len(resolver(args))
        if not args.json:
            console.print(f"  [dim]📋 Categories: {cat_names} ({resolved_count} scenarios)[/]")


def _scenario_selector_label(
    args: argparse.Namespace,
    *,
    resolve_scenarios_fn: Callable[[Any], list[ScenarioDefinition]] | None = None,
) -> str:
    """Describe the scenario selection for the run report."""
    resolver = resolve_scenarios_fn if resolve_scenarios_fn is not None else _resolve_scenarios
    resolved = resolver(args)
    if args.scenarios:
        return ", ".join(args.scenarios)
    if args.categories:
        return f"categories {', '.join(c.upper() for c in args.categories)} ({len(resolved)})"
    if args.short:
        return f"short ({len(resolved)})"
    return f"all ({len(resolved)})"
