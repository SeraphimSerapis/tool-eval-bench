"""Leaderboard ranking and paired-comparison statistics shared by the CLI and web UI.

These are pure functions over stored run records. Both delivery layers render
the same rows, so the grouping, eligibility, and cohort rules live here once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tool_eval_bench.application.run_config import CONFIG_FINGERPRINT_KEY, cohort_config
from tool_eval_bench.domain.scenarios import INFRASTRUCTURE_FAILURE_KINDS
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


_UNPROBED_ENGINE_BACKENDS: frozenset[str] = frozenset(
    {
        "vllm",
        "sglang",
        "llamacpp",
        "llama.cpp",
        "llama_cpp",
        "litellm",
        "openai",
        "local",
    }
)


def cohort_fingerprint(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Fingerprint benchmark conditions shared across different models.

    The schema decides which config keys to drop, and ``cohort_config`` applies
    each field's ``fingerprint_view``.  It is an exclusion rather than an
    allow-list so a key that only an older version stored still splits
    cohorts.  When the run recorded explicit rubric provenance
    (``eval_rubric_version``), cohorts key on those rubric versions instead of
    unrelated ``tool_version``/``git_sha`` commits.  Legacy runs that predate
    rubric versioning keep the historical ``tool_version`` + ``git_sha``
    payload so existing stored rows retain identical cohort hashes.  Never use
    the running ``__version__`` here; it would describe this process, not the
    run.
    """
    eval_rubric_version = metadata.get("eval_rubric_version")
    if isinstance(eval_rubric_version, str) and eval_rubric_version:
        return build_config_fingerprint(
            {
                "config": cohort_config(dict(config)),
                "eval_rubric_version": eval_rubric_version,
                "locale_version": metadata.get("locale_version"),
                "noise_version": metadata.get("noise_version"),
            }
        )
    return build_config_fingerprint(
        {
            "config": cohort_config(dict(config)),
            "tool_version": metadata.get("tool_version"),
            "git_sha": metadata.get("git_sha"),
        }
    )


def _has_response_format_fallback(scenario_results: Any) -> bool:
    """Return whether any stored scenario result recorded a response_format fallback."""
    if not isinstance(scenario_results, Sequence) or isinstance(scenario_results, (str, bytes)):
        return False
    for result in scenario_results:
        if not isinstance(result, Mapping):
            continue
        diagnostics = result.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        if diagnostics.get("response_format_fallback"):
            return True
        rf_note = diagnostics.get("response_format")
        if isinstance(rf_note, str):
            lowered = rf_note.lower()
            if "fell back" in lowered or "fallback" in lowered or "downgrad" in lowered:
                return True
    return False


def comparability_warnings(
    run: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    scores: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return deterministic comparability warnings for a single benchmark run."""
    raw_cfg = run.get("config")
    cfg: Mapping[str, Any] = (
        config if config is not None else (raw_cfg if isinstance(raw_cfg, Mapping) else {})
    )
    raw_scores = run.get("scores")
    sc: Mapping[str, Any] = (
        scores if scores is not None else (raw_scores if isinstance(raw_scores, Mapping) else {})
    )
    raw_meta = run.get("metadata")
    meta: Mapping[str, Any] = (
        metadata if metadata is not None else (raw_meta if isinstance(raw_meta, Mapping) else {})
    )

    warnings: list[str] = []

    score_withheld = (
        ("final_score" in sc and sc["final_score"] is None)
        or (scores is None and "final_score" in run and run["final_score"] is None)
        or sc.get("rating") == "Withheld"
    )
    if score_withheld:
        warnings.append("Score withheld (no gradable scenarios)")

    completion_rate = sc.get("completion_rate")
    if (
        not isinstance(completion_rate, bool)
        and isinstance(completion_rate, (int, float))
        and math.isfinite(completion_rate)
        and completion_rate < 100
    ):
        warnings.append(
            f"Incomplete run ({completion_rate:.1f}% completion; infrastructure failures excluded)"
        )

    git_sha = meta.get("git_sha")
    if isinstance(git_sha, str) and git_sha.endswith("-dirty"):
        warnings.append(f"Run from dirty git working tree ({git_sha})")

    limit = cfg.get("limit")
    if not isinstance(limit, bool) and isinstance(limit, int) and limit > 0:
        warnings.append(f"Evaluated subset only (limit={limit})")

    raw_backend = cfg.get("backend") or meta.get("backend")
    backend = raw_backend.strip().lower() if isinstance(raw_backend, str) else ""
    engine_name = meta.get("engine_name")
    has_engine_name = isinstance(engine_name, str) and bool(engine_name.strip())
    if bool(cfg.get("no_probe_engine")) or (
        not has_engine_name and backend in _UNPROBED_ENGINE_BACKENDS
    ):
        warnings.append("Serving engine was not probed")

    if _has_response_format_fallback(sc.get("scenario_results")):
        warnings.append("Structured output fell back from json_schema")

    return warnings


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
    reviewer = config.get("reviewer")
    if isinstance(reviewer, dict):
        label.append(f"reviewer={shorten_model_name(str(reviewer.get('model', '?')))}")

    # Keep a short stable identifier so two cohorts with the same readable
    # settings remain visibly distinct when another comparison field differs.
    return f"{', '.join(label)} [{cohort_fingerprint[:8]}]"


def _finite_number(value: object, default: float = 0.0) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value if math.isfinite(value) else default


def is_rank_eligible(run: dict[str, Any], config: dict[str, Any], scores: dict[str, Any]) -> bool:
    """Reject explicit partial, interrupted, and incomplete persisted runs."""
    # Records written before run status existed remain exportable.  New records
    # must explicitly be completed and contain every configured scenario.
    status = run.get("status")
    if status is None:
        return True
    if status != "completed":
        return False
    if scores.get("final_score") is None:
        return False
    completion_rate = scores.get("completion_rate")
    if isinstance(completion_rate, (int, float)) and (
        not math.isfinite(completion_rate) or completion_rate < 100
    ):
        return False
    expected = config.get("scenario_count")
    results = scores.get("scenario_results")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or not isinstance(results, list)
    ):
        return False
    return (
        len(
            {
                result.get("scenario_id")
                for result in results
                if isinstance(result, dict) and result.get("scenario_id")
            }
        )
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
            best = max(
                group_runs,
                key=lambda r: _finite_number((r.get("scores") or {}).get("final_score"), 0),
            )
        else:
            best = max(
                group_runs,
                key=lambda r: (str(r.get("created_at") or ""), str(r.get("run_id") or "")),
            )
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
        final_score = _finite_number(scores.get("final_score", 0), 0)

        reviewer = config.get("reviewer")
        rows.append(
            {
                "model": model,
                # A paired run scores the model together with its reviewer.
                "reviewer_model": reviewer.get("model") if isinstance(reviewer, dict) else None,
                "run_id": best.get("run_id", "?"),
                "date": str(best.get("created_at") or "?")[:19],
                "final_score": final_score,
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
                "comparability_warnings": comparability_warnings(best, config, scores, metadata),
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
    rows.sort(
        key=lambda r: (
            r["cohort_label"],
            r["cohort_fingerprint"],
            -r["final_score"],
            r["model"],
        )
    )
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

    Uses a 2x2 contingency table of pass/not-pass outcomes for scenarios
    present in both runs, excluding pairs where either run recorded an
    infrastructure failure. Computed with stdlib math only.

    Interpretation:
        p < 0.05  -> statistically significant difference
        p >= 0.05 -> no significant difference

    Values are returned unrounded; callers format them for display.
    """
    b = 0  # A pass & B not-pass
    c = 0  # A not-pass & B pass

    for sc_id in dict.fromkeys(all_ids):
        ra = map_a.get(sc_id)
        rb = map_b.get(sc_id)
        if not ra or not rb:
            continue  # skip scenarios not in both runs
        if (
            ra.get("failure_kind") in INFRASTRUCTURE_FAILURE_KINDS
            or rb.get("failure_kind") in INFRASTRUCTURE_FAILURE_KINDS
        ):
            continue

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
