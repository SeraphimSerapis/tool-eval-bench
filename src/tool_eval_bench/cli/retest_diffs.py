"""Targeted multi-trial re-testing of discordant scenarios between two stored runs.

Supports two workflows without importing ``storage.db``:
1. ``tool-eval-bench compare <left> <right> --retest-diffs`` runs N trials on A
   and N trials on B for only the discordant scenarios, then prints a paired
   multi-trial summary table and verdict.
2. ``tool-eval-bench run --scenarios-from-diff <run_a> <run_b>`` populates
   ``--scenarios`` with the discordant scenario IDs so users with one model
   loaded in VRAM at a time can run targeted trials separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from tool_eval_bench.application.comparison import (
    compute_paired_comparison,
    exact_two_sided_sign_p_value,
    extract_result_maps,
    is_infrastructure_failure_dict,
)
from tool_eval_bench.application.run_queries import invocation_session, resolve_run
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.commands import resolve_packs
from tool_eval_bench.cli.headless import exit_invalid_arguments
from tool_eval_bench.domain.engines import ACCEPTED_BACKEND_LABELS, BACKEND_LABELS
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.utils.urls import redact_url, redact_urls, validate_http_url

DEFAULT_RETEST_TRIALS = 5
_DEFAULT_LOCAL_BASE_URL = "http://localhost:8000"


def _format_signed_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}"


def resolve_discordant_scenarios_for_diff(
    run_id_a: str,
    run_id_b: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Resolve two stored runs and compute their paired comparison payload.

    Raises :class:`ValueError` when either run is missing or lacks scenario
    results.
    """
    resolved_a = resolve_run(run_id_a)
    if resolved_a is None:
        if run_id_a.lower() == "latest":
            raise ValueError("No runs found in database for 'latest'")
        raise ValueError(f"Run '{run_id_a}' not found in database")

    resolved_b = resolve_run(run_id_b)
    if resolved_b is None:
        if run_id_b.lower() == "latest":
            raise ValueError("No runs found in database for 'latest'")
        raise ValueError(f"Run '{run_id_b}' not found in database")

    id_a, run_a = resolved_a
    id_b, run_b = resolved_b
    results_a = (run_a.get("scores") or {}).get("scenario_results")
    results_b = (run_b.get("scores") or {}).get("scenario_results")
    if not isinstance(results_a, list) or not results_a:
        raise ValueError(f"Run '{id_a}' has no scenario results to compare")
    if not isinstance(results_b, list) or not results_b:
        raise ValueError(f"Run '{id_b}' has no scenario results to compare")

    map_a, map_b, all_ids = extract_result_maps(results_a, results_b)
    paired = compute_paired_comparison(map_a, map_b, all_ids)
    return id_a, run_a, id_b, run_b, paired


def _resolve_discordant_definitions(
    discordant_ids: Sequence[str],
    args: argparse.Namespace,
) -> tuple[list[ScenarioDefinition], list[dict[str, Any]] | None]:
    """Resolve ScenarioDefinitions for discordant IDs from public + pack registries."""
    packs = resolve_packs(args)
    by_id: dict[str, ScenarioDefinition] = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}
    for pack in packs:
        for scenario in pack.scenarios:
            by_id[scenario.id] = scenario

    missing = [sid for sid in discordant_ids if sid not in by_id]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Discordant scenarios not found in the public registry: {joined}. "
            "If they came from a held-out pack, pass --scenario-pack <DIR>."
        )

    attestations = [dict(pack.to_dict()) for pack in packs] or None
    return [by_id[sid] for sid in discordant_ids], attestations


def apply_scenarios_from_diff(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    console: Console,
) -> None:
    """Populate ``args.scenarios`` from ``--scenarios-from-diff <run_a> <run_b>``.

    Exits with status 2 on invalid combinations, missing runs, or missing
    held-out scenario packs. Exits cleanly with status 0 when the two runs
    have zero discordant scenarios.
    """
    pair = getattr(args, "scenarios_from_diff", None)
    if not pair:
        return
    if getattr(args, "scenarios", None):
        parser.error("--scenarios-from-diff cannot be combined with --scenarios")
    if getattr(args, "resume", None):
        parser.error("--scenarios-from-diff cannot be combined with --resume")
    if getattr(args, "compare", None):
        parser.error("--scenarios-from-diff cannot be combined with --compare")

    try:
        id_a, _, id_b, _, paired = resolve_discordant_scenarios_for_diff(pair[0], pair[1])
    except ValueError as exc:
        parser.error(str(exc))

    discordant_ids: list[str] = list(paired["discordant_scenario_ids"])
    if not discordant_ids:
        if getattr(args, "json", False):
            payload = {
                "event": "no_discordant_scenarios",
                "run_a": id_a,
                "run_b": id_b,
                "discordant_scenario_ids": [],
                "verdict": paired["verdict"],
            }
            text = json.dumps(payload, indent=2) + "\n"
            json_file = getattr(args, "json_file", None)
            if json_file:
                from pathlib import Path

                out_path = Path(json_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
        else:
            console.print(
                f"\n  [green]Runs '{escape(id_a)}' and '{escape(id_b)}' have "
                "0 discordant scenarios; nothing to re-run.[/]\n"
            )
        raise SystemExit(0)

    try:
        _resolve_discordant_definitions(discordant_ids, args)
    except ValueError as exc:
        parser.error(str(exc))

    args.scenarios = discordant_ids


def _extract_variant_setting(config: Mapping[str, Any], key: str) -> Any:
    """Read ``locale`` or ``variant_seed`` from config or ``scenario_variants``."""
    direct_key = "variant_seed" if key == "seed" else key
    if config.get(direct_key) is not None:
        return config[direct_key]
    variants = config.get("scenario_variants")
    if isinstance(variants, Mapping):
        for meta in variants.values():
            if isinstance(meta, Mapping) and meta.get(key) is not None:
                return meta[key]
    return None


def _build_side_kwargs(
    run_id: str,
    run_record: Mapping[str, Any],
    *,
    base_url: str,
    api_key: str | None,
    backend_override: str | None,
    scenarios: list[ScenarioDefinition],
    scenario_packs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Reconstruct ``BenchmarkService.run_benchmark`` kwargs from a stored run."""
    raw_cfg = run_record.get("config")
    cfg: Mapping[str, Any] = raw_cfg if isinstance(raw_cfg, Mapping) else {}

    model_raw = cfg.get("model") or run_record.get("model")
    model = str(model_raw).strip() if model_raw is not None else ""
    if not model:
        raise ValueError(f"Stored run '{run_id}' has no model name in its config")

    backend = str(backend_override or cfg.get("backend") or "unknown").lower()
    if backend not in ACCEPTED_BACKEND_LABELS:
        raise ValueError(f"Unsupported backend '{backend}'. Supported: {', '.join(BACKEND_LABELS)}")

    raw_temp = cfg.get("temperature")
    temperature = float(raw_temp) if isinstance(raw_temp, (int, float)) else 0.0

    raw_turns = cfg.get("max_turns")
    max_turns = (
        int(raw_turns)
        if isinstance(raw_turns, int) and not isinstance(raw_turns, bool) and raw_turns >= 1
        else 8
    )

    raw_timeout = cfg.get("timeout_seconds")
    timeout_seconds = (
        float(raw_timeout)
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0
        else DEFAULT_REQUEST_TIMEOUT_SECONDS
    )

    raw_seed = cfg.get("seed")
    seed = int(raw_seed) if isinstance(raw_seed, int) and not isinstance(raw_seed, bool) else None

    raw_variant_seed = _extract_variant_setting(cfg, "seed")
    variant_seed = (
        int(raw_variant_seed)
        if isinstance(raw_variant_seed, int) and not isinstance(raw_variant_seed, bool)
        else None
    )

    raw_locale = _extract_variant_setting(cfg, "locale")
    locale = str(raw_locale) if isinstance(raw_locale, str) and raw_locale else None

    raw_ref_date = cfg.get("reference_date")
    reference_date = str(raw_ref_date) if isinstance(raw_ref_date, str) and raw_ref_date else None

    raw_alpha = cfg.get("alpha")
    alpha = float(raw_alpha) if isinstance(raw_alpha, (int, float)) else 0.7

    raw_extra = cfg.get("extra_params")
    extra_params = dict(raw_extra) if isinstance(raw_extra, Mapping) else None

    weight_by_difficulty = bool(cfg.get("weight_by_difficulty", False))
    raw_prompt = cfg.get("system_prompt")
    system_prompt = raw_prompt if isinstance(raw_prompt, str) else None

    return {
        "model": model,
        "backend": backend,
        "base_url": base_url,
        "api_key": api_key,
        "scenarios": list(scenarios),
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "max_turns": max_turns,
        "seed": seed,
        "variant_seed": variant_seed,
        "locale": locale,
        "reference_date": reference_date,
        "alpha": alpha,
        "extra_params": extra_params,
        "weight_by_difficulty": weight_by_difficulty,
        "system_prompt": system_prompt,
        "scenario_packs": scenario_packs,
    }


def _summarize_side_trials(
    trial_runs: Sequence[Mapping[str, Any]],
    scenario_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-scenario mean points, Pass@k, and Pass^k across trials."""
    stats: dict[str, dict[str, Any]] = {}
    for sid in scenario_ids:
        points: list[int] = []
        passes: list[bool] = []
        for run in trial_runs:
            results = (run.get("scores") or {}).get("scenario_results") or []
            matched = next(
                (r for r in results if isinstance(r, Mapping) and str(r.get("scenario_id")) == sid),
                None,
            )
            if matched is None or is_infrastructure_failure_dict(matched):
                continue
            points.append(int(matched.get("points") or 0))
            passes.append(matched.get("status") == "pass")

        if points:
            stats[sid] = {
                "graded_trials": len(points),
                "mean_points": sum(points) / len(points),
                "points": points,
                "pass_at_k": any(passes),
                "pass_hat_k": all(passes),
            }
        else:
            stats[sid] = {
                "graded_trials": 0,
                "mean_points": None,
                "points": [],
                "pass_at_k": False,
                "pass_hat_k": False,
            }
    return stats


def _trial_level_pct_deltas(
    runs_a: Sequence[Mapping[str, Any]],
    runs_b: Sequence[Mapping[str, Any]],
    scenario_ids: Sequence[str],
) -> list[float]:
    """Compute paired percentage-point deltas per trial index across discordant IDs."""
    deltas: list[float] = []
    for run_a, run_b in zip(runs_a, runs_b, strict=False):
        res_a = (run_a.get("scores") or {}).get("scenario_results") or []
        res_b = (run_b.get("scores") or {}).get("scenario_results") or []
        map_a = {
            str(r["scenario_id"]): r for r in res_a if isinstance(r, Mapping) and "scenario_id" in r
        }
        map_b = {
            str(r["scenario_id"]): r for r in res_b if isinstance(r, Mapping) and "scenario_id" in r
        }
        paired = compute_paired_comparison(map_a, map_b, scenario_ids, bootstrap_samples=0)
        if paired["shared_count"] > 0:
            deltas.append(float(paired["mean_pct_delta"]))
    return deltas


def _fail_retest(console: Console, args: argparse.Namespace, message: str) -> NoReturn:
    if getattr(args, "json", False):
        exit_invalid_arguments(message)
    console.print(f"\n[bold red]Error:[/] {escape(message)}\n")
    raise SystemExit(2)


def run_retest_diffs_command(args: argparse.Namespace, console: Console) -> None:
    """Execute ``compare <left> <right> --retest-diffs`` and print summary table."""
    if getattr(args, "json", False):
        exit_invalid_arguments("--retest-diffs does not support --json")

    compare_pair = getattr(args, "compare", None)
    if not compare_pair or len(compare_pair) != 2:
        _fail_retest(
            console, args, "--retest-diffs requires two run IDs via compare <left> <right>"
        )

    trials = getattr(args, "trials", DEFAULT_RETEST_TRIALS)
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        _fail_retest(console, args, f"--trials must be an integer >= 1, got {trials!r}")

    try:
        id_a, run_a, id_b, run_b, initial_paired = resolve_discordant_scenarios_for_diff(
            compare_pair[0], compare_pair[1]
        )
    except ValueError as exc:
        _fail_retest(console, args, str(exc))

    discordant_ids: list[str] = list(initial_paired["discordant_scenario_ids"])
    if not discordant_ids:
        console.print(
            f"\n  [green]Runs '{escape(id_a)}' and '{escape(id_b)}' have no discordant "
            "scenarios to re-test.[/]\n"
        )
        return

    try:
        discordant_scenarios, scenario_packs = _resolve_discordant_definitions(discordant_ids, args)
    except ValueError as exc:
        _fail_retest(console, args, str(exc))

    env_url = os.getenv("TOOL_EVAL_BASE_URL") or _DEFAULT_LOCAL_BASE_URL
    url_a = getattr(args, "base_url_a", None) or getattr(args, "base_url", None) or env_url
    url_b = getattr(args, "base_url_b", None) or getattr(args, "base_url", None) or env_url
    try:
        validate_http_url(url_a, what="Baseline A base URL")
        validate_http_url(url_b, what="Candidate B base URL")
    except ValueError as exc:
        _fail_retest(console, args, str(exc))

    env_key = os.getenv("TOOL_EVAL_API_KEY") or None
    key_a = getattr(args, "api_key_a", None) or getattr(args, "api_key", None) or env_key
    key_b = getattr(args, "api_key_b", None) or getattr(args, "api_key", None) or env_key
    backend_override = getattr(args, "backend", None)

    try:
        kwargs_a = _build_side_kwargs(
            id_a,
            run_a,
            base_url=url_a,
            api_key=key_a,
            backend_override=backend_override,
            scenarios=discordant_scenarios,
            scenario_packs=scenario_packs,
        )
        kwargs_b = _build_side_kwargs(
            id_b,
            run_b,
            base_url=url_b,
            api_key=key_b,
            backend_override=backend_override,
            scenarios=discordant_scenarios,
            scenario_packs=scenario_packs,
        )
    except ValueError as exc:
        _fail_retest(console, args, str(exc))

    redact = bool(getattr(args, "redact_url", False))
    disp_url_a = redact_url(url_a) if redact else url_a
    disp_url_b = redact_url(url_b) if redact else url_b

    trial_word = "trial" if trials == 1 else "trials"
    console.print(
        f"\n  [bold]Targeted Discordant Re-Test:[/] {len(discordant_scenarios)} scenarios, "
        f"{trials} {trial_word} per side"
    )
    console.print(
        f"  [dim]A: {escape(kwargs_a['model'])} @ {escape(disp_url_a)}[/]\n"
        f"  [dim]B: {escape(kwargs_b['model'])} @ {escape(disp_url_b)}[/]\n"
    )

    service = BenchmarkService(reporter=MarkdownReporter(root=getattr(args, "output_dir", None)))

    async def _run_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        runs_a: list[dict[str, Any]] = []
        runs_b: list[dict[str, Any]] = []
        for idx in range(1, trials + 1):
            console.print(f"  [dim]Running baseline A trial {idx}/{trials}...[/]")
            call_a = {**kwargs_a, "scenarios": list(discordant_scenarios)}
            runs_a.append(await service.run_benchmark(**call_a))
        for idx in range(1, trials + 1):
            console.print(f"  [dim]Running candidate B trial {idx}/{trials}...[/]")
            call_b = {**kwargs_b, "scenarios": list(discordant_scenarios)}
            runs_b.append(await service.run_benchmark(**call_b))
        return runs_a, runs_b

    try:
        with invocation_session(uuid.uuid4().hex):
            trial_runs_a, trial_runs_b = asyncio.run(_run_all())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        detail = redact_urls(str(exc)) if redact else str(exc)
        console.print(f"\n[bold red]Error:[/] {escape(detail)}\n")
        sys.exit(1)

    _render_retest_summary(
        console,
        id_a=id_a,
        run_a=run_a,
        id_b=id_b,
        run_b=run_b,
        discordant_ids=discordant_ids,
        trial_runs_a=trial_runs_a,
        trial_runs_b=trial_runs_b,
        trials=trials,
    )


def _render_retest_summary(
    console: Console,
    *,
    id_a: str,
    run_a: Mapping[str, Any],
    id_b: str,
    run_b: Mapping[str, Any],
    discordant_ids: Sequence[str],
    trial_runs_a: Sequence[Mapping[str, Any]],
    trial_runs_b: Sequence[Mapping[str, Any]],
    trials: int,
) -> None:
    """Render the Paired Discordant Re-Test Summary Table and multi-trial verdict."""
    import random

    init_res_a = (run_a.get("scores") or {}).get("scenario_results") or []
    init_res_b = (run_b.get("scores") or {}).get("scenario_results") or []
    init_map_a, init_map_b, _ = extract_result_maps(init_res_a, init_res_b)

    stats_a = _summarize_side_trials(trial_runs_a, discordant_ids)
    stats_b = _summarize_side_trials(trial_runs_b, discordant_ids)

    table = Table(
        title=f"[bold]Paired Discordant Re-Test Summary (k={trials})[/]",
        show_header=True,
        header_style="bold",
        border_style="bright_cyan",
        expand=True,
    )
    table.add_column("ID", width=10, no_wrap=True)
    table.add_column("Initial A -> B", justify="center", min_width=14)
    table.add_column("A Mean (Pass@k / Pass^k)", justify="center", min_width=22)
    table.add_column("B Mean (Pass@k / Pass^k)", justify="center", min_width=22)
    table.add_column("Multi-Trial Δ", justify="right", min_width=13)

    status_symbols = {"pass": "✅", "partial": "⚠️", "fail": "❌"}
    scenario_mean_deltas: list[float] = []
    b_wins = 0
    a_wins = 0

    for sid in discordant_ids:
        ra = init_map_a.get(sid) or {}
        rb = init_map_b.get(sid) or {}
        pts_a = int(ra.get("points") or 0)
        pts_b = int(rb.get("points") or 0)
        sym_a = status_symbols.get(str(ra.get("status") or "fail"), "?")
        sym_b = status_symbols.get(str(rb.get("status") or "fail"), "?")
        initial_cell = f"{sym_a} {pts_a}/2 -> {sym_b} {pts_b}/2"

        sa = stats_a[sid]
        sb = stats_b[sid]
        mean_a = sa["mean_points"]
        mean_b = sb["mean_points"]

        cell_a = (
            f"{mean_a:.2f}/2 ({'yes' if sa['pass_at_k'] else 'no'} / "
            f"{'yes' if sa['pass_hat_k'] else 'no'})"
            if mean_a is not None
            else "[yellow]infra error[/]"
        )
        cell_b = (
            f"{mean_b:.2f}/2 ({'yes' if sb['pass_at_k'] else 'no'} / "
            f"{'yes' if sb['pass_hat_k'] else 'no'})"
            if mean_b is not None
            else "[yellow]infra error[/]"
        )

        if mean_a is not None and mean_b is not None:
            diff_pts = mean_b - mean_a
            scenario_mean_deltas.append(diff_pts)
            if diff_pts > 1e-9:
                b_wins += 1
                delta_cell = f"[bold green]+{diff_pts:.2f}[/]"
            elif diff_pts < -1e-9:
                a_wins += 1
                delta_cell = f"[bold red]{diff_pts:.2f}[/]"
            else:
                delta_cell = "[dim]0.00[/]"
        else:
            delta_cell = "[yellow]excluded[/]"

        table.add_row(escape(sid), initial_cell, cell_a, cell_b, delta_cell)

    console.print()
    console.print(table)

    if not scenario_mean_deltas:
        console.print(
            Panel(
                "  [yellow]Multi-trial verdict:[/] No gradable discordant scenarios "
                "completed without infrastructure errors.",
                border_style="yellow",
                padding=(0, 2),
            )
        )
        console.print()
        return

    gradable_n = len(scenario_mean_deltas)
    mean_pct_delta = (sum(scenario_mean_deltas) / (2.0 * gradable_n)) * 100.0
    trial_pct_deltas = _trial_level_pct_deltas(trial_runs_a, trial_runs_b, discordant_ids)

    if len(trial_pct_deltas) >= 2 and len(set(trial_pct_deltas)) > 1:
        rng = random.Random(0)
        k_n = len(trial_pct_deltas)
        resamples = sorted(sum(rng.choices(trial_pct_deltas, k=k_n)) / k_n for _ in range(2000))
        ci_low = resamples[int(0.025 * 2000)]
        ci_high = resamples[int(0.975 * 2000) - 1]
    else:
        ci_low = mean_pct_delta
        ci_high = mean_pct_delta

    # Combine scenario-level sign test with unanimous multi-trial consistency.
    trial_b_wins = sum(1 for d in trial_pct_deltas if d > 1e-9)
    trial_a_wins = sum(1 for d in trial_pct_deltas if d < -1e-9)
    scenario_sign_p = exact_two_sided_sign_p_value(a_wins, b_wins)
    trial_sign_p = exact_two_sided_sign_p_value(trial_a_wins, trial_b_wins)
    effective_p = min(scenario_sign_p, trial_sign_p)

    unanimous_b = len(trial_pct_deltas) >= DEFAULT_RETEST_TRIALS and all(
        d > 1e-9 for d in trial_pct_deltas
    )
    unanimous_a = len(trial_pct_deltas) >= DEFAULT_RETEST_TRIALS and all(
        d < -1e-9 for d in trial_pct_deltas
    )

    if abs(mean_pct_delta) < 1e-9:
        verdict_line = (
            "[dim]Multi-trial verdict:[/] Converged; multi-trial means match and the "
            "initial gap was single-run variance."
        )
    elif (effective_p < 0.05 or unanimous_b) and mean_pct_delta > 0:
        verdict_line = (
            "[bold green]Multi-trial verdict:[/] Confirmed improvement; candidate B "
            "consistently outperforms baseline A on the discordant scenarios."
        )
    elif (effective_p < 0.05 or unanimous_a) and mean_pct_delta < 0:
        verdict_line = (
            "[bold red]Multi-trial verdict:[/] Confirmed regression; candidate B "
            "consistently trails baseline A on the discordant scenarios."
        )
    else:
        verdict_line = (
            "[bold yellow]Multi-trial verdict:[/] Inconclusive across trials; "
            "variance still overlaps zero."
        )

    summary_text = (
        f"  [bold]Discordant subset delta:[/] {_format_signed_pct(mean_pct_delta)} pts, "
        f"95% CI [{_format_signed_pct(ci_low)}, {_format_signed_pct(ci_high)}] pts, "
        f"{b_wins} B higher, {a_wins} A higher\n"
        f"  {verdict_line}"
    )
    console.print(Panel(summary_text, border_style="bright_cyan", padding=(0, 2)))
    console.print()
