"""Redaction and enrichment applied to every run record the web UI returns.

Stored runs keep held-out pack content and redacted endpoint URLs; the UI must
not undo either. Everything the API or the live stream emits goes through here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tool_eval_bench.cli.held_out import HELD_OUT, redact_result, redact_run
from tool_eval_bench.domain.redaction import redact_url
from tool_eval_bench.domain.scenarios import CATEGORY_LABELS
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

_PUBLIC_SCENARIO_MAP = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}

# Key fragments that mark a credential. "token" alone is not one of them:
# max_tokens, fill_tokens, and token_efficiency are benchmark settings, so only
# an exact "token" or a "*_token" suffix (hf_token, access_token) counts.
_SENSITIVE_SUBSTRINGS = ("api_key", "apikey", "secret", "password", "authorization", "credential")
_URL_KEYS = frozenset({"base_url", "decision_judge_base_url", "metrics_url", "server"})


def is_sensitive_key(key: str) -> bool:
    """Whether a config or metadata key names a credential."""
    lowered = key.lower()
    if lowered == "token" or lowered.endswith("_token"):
        return True
    return any(part in lowered for part in _SENSITIVE_SUBSTRINGS)


def sanitize_mapping(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of *data* with credential keys removed and URLs redacted."""
    if not isinstance(data, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        key_str = str(key)
        if is_sensitive_key(key_str):
            continue
        if key_str in _URL_KEYS and isinstance(value, str) and value:
            cleaned[key_str] = redact_url(value)
        elif isinstance(value, Mapping):
            cleaned[key_str] = sanitize_mapping(value)
        elif isinstance(value, list):
            cleaned[key_str] = [
                sanitize_mapping(item) if isinstance(item, Mapping) else item for item in value
            ]
        else:
            cleaned[key_str] = value
    return cleaned


def stored_held_out_ids(run: Mapping[str, Any]) -> frozenset[str]:
    """Identify held-out scenario IDs in a stored run record.

    Anything outside the public registry counts as held out, so a pack whose
    attestation is missing still stays hidden.
    """
    held_out: set[str] = set()
    config = run.get("config")
    if isinstance(config, Mapping):
        packs = config.get("scenario_packs")
        if isinstance(packs, list):
            for pack in packs:
                if isinstance(pack, Mapping):
                    ids = pack.get("scenario_ids")
                    if isinstance(ids, list):
                        held_out.update(str(sid) for sid in ids if sid)
    scores = run.get("scores")
    if isinstance(scores, Mapping):
        results = scores.get("scenario_results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                sid = result.get("scenario_id")
                if not isinstance(sid, str) or not sid:
                    continue
                if result.get("held_out") is True or sid not in _PUBLIC_SCENARIO_MAP:
                    held_out.add(sid)
    return frozenset(held_out)


def enrich_scenario_result(result: dict[str, Any], held_out: frozenset[str]) -> dict[str, Any]:
    """Attach scenario title, category, difficulty, and safety metadata for the UI."""
    sid = str(result.get("scenario_id", ""))
    enriched = dict(result)
    if sid in held_out:
        enriched["title"] = HELD_OUT
        enriched["category"] = "held_out"
        enriched["category_label"] = "Held-Out Pack"
        enriched["difficulty"] = None
        enriched["safety_critical"] = False
        enriched["held_out"] = True
        return enriched

    scenario = _PUBLIC_SCENARIO_MAP.get(sid)
    if scenario is not None:
        enriched["title"] = scenario.title
        enriched["category"] = scenario.category.value
        enriched["category_label"] = CATEGORY_LABELS.get(scenario.category, scenario.category.value)
        enriched["difficulty"] = scenario.difficulty
        enriched["safety_critical"] = bool(
            scenario.safety_critical_on_fail or result.get("safety_violation")
        )
        enriched["held_out"] = False
    else:
        enriched.setdefault("title", sid)
        enriched.setdefault("category", "?")
        enriched.setdefault("category_label", "Unknown")
        enriched.setdefault("difficulty", None)
        enriched.setdefault("safety_critical", False)
        enriched.setdefault("held_out", False)
    return enriched


def sanitize_result(
    result: Mapping[str, Any],
    held_out: frozenset[str],
    *,
    include_traces: bool = False,
) -> dict[str, Any]:
    """Redact and enrich one scenario result for the live view or a checkpoint list.

    Traces are dropped by default: live views show verdicts, and keeping every
    raw log in the in-memory event history would grow without bound.
    """
    raw = dict(result)
    sid = str(raw.get("scenario_id", ""))
    if sid in held_out or raw.get("held_out") is True:
        enriched = enrich_scenario_result(redact_result(raw), held_out | {sid})
    else:
        enriched = enrich_scenario_result(raw, held_out)
    if not include_traces:
        enriched.pop("raw_log", None)
    return enriched


def sanitize_run_for_web(run: Mapping[str, Any], *, include_traces: bool = True) -> dict[str, Any]:
    """Redact held-out scenario content, strip secrets, and enrich scenario metadata."""
    held_out = stored_held_out_ids(run)
    redacted = redact_run(dict(run), held_out)
    config = sanitize_mapping(redacted.get("config"))
    metadata = sanitize_mapping(redacted.get("metadata"))

    scores_raw = redacted.get("scores")
    scores: dict[str, Any] = dict(scores_raw) if isinstance(scores_raw, Mapping) else {}
    results_raw = scores.get("scenario_results")
    passes = partials = fails = 0
    if isinstance(results_raw, list):
        enriched_results: list[dict[str, Any]] = []
        for item in results_raw:
            if not isinstance(item, Mapping):
                continue
            status = item.get("status")
            if status == "pass":
                passes += 1
            elif status == "partial":
                partials += 1
            elif status == "fail":
                fails += 1
            row = enrich_scenario_result(dict(item), held_out)
            if not include_traces:
                row.pop("raw_log", None)
            enriched_results.append(row)
        scores["scenario_results"] = enriched_results

    safety_warnings = scores.get("safety_warnings")
    warning_list = safety_warnings if isinstance(safety_warnings, list) else []
    label = metadata.get("label") or config.get("label")
    scenario_count = config.get(
        "scenario_count",
        len(results_raw) if isinstance(results_raw, list) else 0,
    )

    out: dict[str, Any] = {
        "run_id": redacted.get("run_id"),
        "created_at": redacted.get("created_at"),
        "status": redacted.get("status", "completed"),
        "model": redacted.get("model") or config.get("model", "unknown"),
        "run_type": redacted.get("run_type", "tool_eval"),
        "report_path": redacted.get("report_path"),
        "label": label,
        "backend": metadata.get("backend") or config.get("backend", "unknown"),
        "final_score": scores.get("final_score"),
        "rating": scores.get("rating"),
        "deployability": scores.get("deployability"),
        "responsiveness": scores.get("responsiveness"),
        "total_points": scores.get("total_points"),
        "max_points": scores.get("max_points"),
        "passes": passes,
        "partials": partials,
        "fails": fails,
        "scenario_count": scenario_count,
        "safety_warnings_count": len(warning_list),
        "held_out_count": len(held_out),
        "config": config,
        "metadata": metadata,
        "scores": scores,
    }
    if "safety_gate" in redacted and isinstance(redacted["safety_gate"], Mapping):
        out["safety_gate"] = dict(redacted["safety_gate"])
    return out
