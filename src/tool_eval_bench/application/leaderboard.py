"""Leaderboard ranking and paired-comparison statistics shared by the CLI and web UI.

These are pure functions over stored run records. Both delivery layers render
the same rows, so the grouping, eligibility, and cohort rules live here once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tool_eval_bench.application.run_config import CONFIG_FINGERPRINT_KEY, cohort_config
from tool_eval_bench.utils.ids import build_config_fingerprint


def shorten_model_name(name: str) -> str:
    """Shorten a model identifier for leaderboard display.

    Handles three common formats:
      1. HF-style  "Qwen/Qwen3.6-35B-A3B-FP8"       → kept as-is
      2. Filesystem "/home/…/models--Org--Model/…"    → "Org/Model"
      3. Bare path  "/data/models/my-model"           → "my-model"

    Only the last two path components (owner/model) are kept so the
    leaderboard column stays readable without hard-truncation (#15).
    """
    # Strip leading/trailing whitespace
    name = name.strip()

    # Detect HuggingFace cache paths: "models--Org--ModelName"
    # e.g. /home/user/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B/snapshots/abc
    if "models--" in name:
        for segment in name.replace("\\", "/").split("/"):
            if segment.startswith("models--"):
                parts = segment.split("--", 2)  # ["models", "Org", "Model"]
                if len(parts) == 3:
                    return f"{parts[1]}/{parts[2]}"
                if len(parts) == 2:
                    return parts[1]
                break  # unexpected format, fall through

    # Detect filesystem paths (Unix or Windows)
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (
        len(normalized) > 2 and normalized[1] == ":" and normalized[2] == "/"
    ):
        # Absolute path (Unix, Windows drive, or UNC) — keep last two components
        parts = [p for p in normalized.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        if parts:
            return parts[-1]

    # Already a short name (HF-style "owner/model" or bare alias)
    return name


def cohort_fingerprint(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Fingerprint benchmark conditions shared across different models.

    The schema decides which config keys to drop, and ``cohort_config`` applies
    each field's ``fingerprint_view``.  It is an exclusion rather than an
    allow-list so a key that only an older version stored still splits
    cohorts.  The per-model ``config_fingerprint`` is excluded because it folds
    in the model, so the code identity it carries is added back from the run's
    stored metadata: evaluators are code, and runs from different versions or
    commits are not comparable.  Never use the running ``__version__`` here; it
    would describe this process, not the run.
    """
    return build_config_fingerprint(
        {
            "config": cohort_config(dict(config)),
            "tool_version": metadata.get("tool_version"),
            "git_sha": metadata.get("git_sha"),
        }
    )


def cohort_label(config: dict[str, Any], cohort_fingerprint: str) -> str:
    """Build a compact label that explains why rows are ranked separately."""
    label = [
        f"{config.get('backend', '?')}/{config.get('scenario_count', '?')}",
    ]
    seed = config.get("seed")
    if seed is not None:
        label.append(f"seed={seed}")
    if config.get("weight_by_difficulty"):
        label.append("difficulty-weighted")
    if config.get("context_pressure"):
        label.append("context-pressure")
    if config.get("scenario_packs"):
        label.append("scenario-pack")

    # Keep a short stable identifier so two cohorts with the same readable
    # settings remain visibly distinct when another comparison field differs.
    return f"{', '.join(label)} [{cohort_fingerprint[:8]}]"


def is_rank_eligible(run: dict[str, Any], config: dict[str, Any], scores: dict[str, Any]) -> bool:
    """Reject explicit partial, interrupted, and incomplete persisted runs."""
    # Records written before run status existed remain exportable.  New records
    # must explicitly be completed and contain every configured scenario.
    status = run.get("status")
    if status is None:
        return True
    if status != "completed":
        return False
    completion_rate = scores.get("completion_rate")
    if isinstance(completion_rate, (int, float)) and completion_rate < 100:
        return False
    expected = config.get("scenario_count")
    results = scores.get("scenario_results")
    if not isinstance(expected, int) or expected <= 0 or not isinstance(results, list):
        return False
    return (
        len({result.get("scenario_id") for result in results if result.get("scenario_id")})
        == expected
    )


# ---------------------------------------------------------------------------
# Data extraction from stored runs
# ---------------------------------------------------------------------------


def extract_leaderboard_rows(
    runs: list[dict],
) -> list[dict[str, Any]]:
    """Extract normalized rows from stored run data.

    Groups by model and deterministic configuration fingerprint so that only
    comparable executions are ranked together. Takes the best run per group.
    """
    # Group runs by comparable configuration
    by_config: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        model = run.get("model", "unknown")
        config = run.get("config") or {}
        scores = run.get("scores") or {}
        run_type = run.get("run_type", "tool_eval")

        # Skip non-tool-eval runs (plugins have their own display), incomplete
        # records, and rows that have not reached a completed reportable state.
        if run_type != "tool_eval" or not is_rank_eligible(run, config, scores):
            continue

        # Runs stored before config_fingerprint existed cannot be rebuilt
        # through the current schema, so hash everything they recorded.
        fingerprint = config.get(CONFIG_FINGERPRINT_KEY) or build_config_fingerprint(
            {
                "config": config,
                "metadata": run.get("metadata") or {},
            }
        )
        key = (model, fingerprint)
        by_config.setdefault(key, []).append(run)

    rows: list[dict[str, Any]] = []
    for (model, fingerprint), group_runs in by_config.items():
        # A retry must not silently become a best-of score.  Persisted modern
        # runs select the latest declared execution.  The score-max fallback is
        # retained only for status-less legacy imports that cannot distinguish
        # trials from historical duplicates.
        if all(run.get("status") is None for run in group_runs):
            best = max(group_runs, key=lambda r: (r.get("scores") or {}).get("final_score", 0))
        else:
            best = max(group_runs, key=lambda r: (r.get("created_at", ""), r.get("run_id", "")))
        scores = best.get("scores") or {}
        config = best.get("config") or {}
        scenario_count = config.get("scenario_count", len(scores.get("scenario_results", [])))
        backend = config.get("backend", "?")

        # Extract per-category percentages
        cat_scores: dict[str, float] = {}
        for cs in scores.get("category_scores", []):
            cat_scores[cs["category"]] = cs.get("percent", 0)

        # Count scenarios by status
        results = scores.get("scenario_results", [])
        passes = sum(1 for r in results if r.get("status") == "pass")
        partials = sum(1 for r in results if r.get("status") == "partial")
        fails = sum(1 for r in results if r.get("status") == "fail")

        # Extract metadata (issue #6)
        metadata = best.get("metadata") or {}
        row_cohort = cohort_fingerprint(config, metadata)

        rows.append(
            {
                "model": model,
                "run_id": best.get("run_id", "?"),
                "date": best.get("created_at", "?")[:19],
                "final_score": scores.get("final_score", 0),
                "rating": scores.get("rating", "?"),
                "total_points": scores.get("total_points", 0),
                "max_points": scores.get("max_points", 0),
                "passes": passes,
                "partials": partials,
                "fails": fails,
                "completion_count": len(
                    {r.get("scenario_id") for r in results if r.get("scenario_id")}
                ),
                "cat_scores": cat_scores,
                "scenario_count": scenario_count,
                "backend": backend,
                "config_fingerprint": fingerprint,
                "cohort_fingerprint": row_cohort,
                "cohort_label": cohort_label(config, row_cohort),
                "total_tokens": scores.get("total_tokens", 0),
                "token_efficiency": scores.get("token_efficiency"),
                "median_turn_ms": scores.get("median_turn_ms"),
                "deployability": scores.get("deployability"),
                "safety_warnings": len(scores.get("safety_warnings", [])),
                "num_runs": len(group_runs),
                # Issue #6 metadata fields
                "tool_version": metadata.get("tool_version"),
                "engine_name": metadata.get("engine_name"),
                "engine_version": metadata.get("engine_version"),
                "quantization": metadata.get("quantization"),
                "max_model_len": metadata.get("max_model_len"),
                "temperature": metadata.get("temperature"),
                "server_model_root": metadata.get("server_model_root"),
            }
        )

    # Keep cohorts together, then rank scores within each comparable cohort.
    rows.sort(key=lambda r: (r["cohort_label"], -r["final_score"], r["model"]))
    previous_cohort: str | None = None
    cohort_rank = 0
    for row in rows:
        if row["cohort_fingerprint"] != previous_cohort:
            previous_cohort = row["cohort_fingerprint"]
            cohort_rank = 0
        cohort_rank += 1
        row["cohort_rank"] = cohort_rank
    return rows


# ---------------------------------------------------------------------------
# Leaderboard display


def compute_mcnemar_stats(
    map_a: Mapping[str, Mapping[str, Any]],
    map_b: Mapping[str, Mapping[str, Any]],
    all_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute McNemar's test for statistical significance between two runs.

    Uses a 2×2 contingency table of pass/not-pass outcomes for scenarios
    present in both runs.  The chi-squared statistic (with continuity
    correction) is computed without scipy — only stdlib math is needed.

    Interpretation:
        p < 0.05  → statistically significant difference
        p >= 0.05 → no significant difference (could be noise)

    Values are returned unrounded; callers format them for display.
    """
    # Build contingency table
    # b = A pass, B fail (B regressed)
    # c = A fail, B pass (B improved)
    b = 0  # A pass & B not-pass
    c = 0  # A not-pass & B pass

    for sc_id in all_ids:
        ra = map_a.get(sc_id)
        rb = map_b.get(sc_id)
        if not ra or not rb:
            continue  # skip scenarios not in both runs

        a_pass = ra.get("status") == "pass"
        b_pass = rb.get("status") == "pass"

        if a_pass and not b_pass:
            b += 1
        elif not a_pass and b_pass:
            c += 1

    n_discordant = b + c
    if n_discordant == 0:
        return {
            "discordant_pairs": 0,
            "a_only_pass": 0,
            "b_only_pass": 0,
            "chi2": 0.0,
            "p_value": 1.0,
            "significant": False,
            "direction": "neither",
        }

    # The exact two-sided binomial test is reliable for the small paired
    # samples common in targeted scenario runs. For larger runs, use the
    # continuity-corrected chi-squared approximation. Clamp the correction at
    # zero so balanced discordance produces chi²=0 and p=1.
    corrected_difference = max(0, abs(b - c) - 1)
    chi2 = corrected_difference**2 / n_discordant
    if n_discordant <= 25:
        smaller_side = min(b, c)
        p_value = min(
            1.0,
            2
            * sum(math.comb(n_discordant, idx) for idx in range(smaller_side + 1))
            / 2**n_discordant,
        )
    else:
        # Survival function for chi-squared with df=1 (no scipy needed).
        p_value = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0

    return {
        "discordant_pairs": n_discordant,
        "a_only_pass": b,
        "b_only_pass": c,
        "chi2": chi2,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "direction": "B" if c > b else "A" if b > c else "neither",
    }
