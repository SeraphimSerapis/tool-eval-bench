"""What the web UI shows for a stored run of each type.

A tool-call run has a score. The other modes store measurements: throughput
cells, speculative-decoding samples, pressure levels, retrieval grids, and
accuracy counts. :func:`headline` picks the one figure a run list can show for
a run, and :func:`detail` lays its measurements out as stats and a table, so
the browser renders every mode the same way without deriving any figure
itself. Throughput, needle, and accuracy figures come from the same readers
the results card uses, so the two never disagree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from tool_eval_bench.application.card_data import accuracy_result, needle_summary, perf_points

TOOL_EVAL = "tool_eval"

RUN_TYPE_LABELS: dict[str, str] = {
    TOOL_EVAL: "Tool calls",
    "perf": "Throughput",
    "spec-bench": "Spec decode",
    "context-pressure": "Context pressure",
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "ifeval": "IFEval",
    "needle": "Needle",
    "toolscale": "Toolscale",
    "decision": "Decision",
}

_ACCURACY_NAMES = {"gsm8k": "GSM8K", "mmlu": "MMLU", "ifeval": "IFEval"}
# Generic stats skip these: they repeat the headline or hold the raw rows.
_GENERIC_SKIP = frozenset({"final_score", "rating"})
_GENERIC_LIMIT = 12


def run_type(run: Mapping[str, Any]) -> str:
    """The stored run type; rows from before the column existed are tool-call runs."""
    return str(run.get("run_type") or TOOL_EVAL)


def type_label(kind: str) -> str:
    """A display name for a run type, falling back to the stored name."""
    return RUN_TYPE_LABELS.get(kind, kind.replace("_", " ").replace("-", " ").capitalize())


def headline(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """The figure a run list shows in place of a score, or ``None``.

    Returns ``{"label", "value", "unit", "detail"}``. A tool-call run returns
    ``None``: it shows its score. So does a run with nothing measured.
    """
    kind = run_type(run)
    scores = _scores(run)
    if kind == TOOL_EVAL:
        return None
    if kind == "perf":
        points = perf_points(run)
        if not points:
            return None
        single = [point for point in points if point.concurrency == 1] or list(points)
        point = min(single, key=lambda p: p.depth)
        where = f"depth {point.depth:,}" if point.depth else None
        return _figure("decode", point.tg_tps, "tok/s", detail=where)
    if kind == "spec-bench":
        results = _rows(scores.get("results"))
        effective = _mean_of(results, "effective_tg_tps") or _mean_of(results, "tg_tps")
        if effective is None:
            return None
        notes = []
        acceptance = _mean_of(results, "acceptance_rate")
        if acceptance is not None:
            notes.append(f"{acceptance * 100:.0f}% accepted")
        speedup = _mean_of(results, "speedup_ratio")
        if speedup is not None:
            notes.append(f"{speedup:.2f}× speedup")
        return _figure("effective decode", effective, "tok/s", detail=", ".join(notes) or None)
    if kind == "context-pressure":
        breaking = _number(scores.get("breaking_point"))
        if breaking is not None:
            return _figure("holds to", breaking * 100, "% fill")
        degraded = _number(scores.get("first_degradation"))
        if degraded is not None:
            return _figure("degrades at", degraded * 100, "% fill")
        return None
    if kind == "needle":
        summary = needle_summary(run)
        context = summary.effective_context
        return _figure(
            "retrieval",
            summary.accuracy * 100,
            "%",
            detail=f"effective to {context:,} tokens" if context else None,
        )
    if kind in _ACCURACY_NAMES:
        result = accuracy_result(_ACCURACY_NAMES[kind], run)
        if result is None or math.isnan(result.accuracy):
            return None
        return _figure("accuracy", result.accuracy * 100, "%", detail=result.detail)
    accuracy = _number(scores.get("accuracy"))
    return _figure("accuracy", accuracy, "%") if accuracy is not None else None


def detail(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """A run's measurements as ``{"stats": [...], "table": {...} | None}``.

    Each stat is ``{"label", "value", "unit"}``. A table has ``columns``
    (``{"key", "label", "unit"}``) and ``rows`` keyed by column. A tool-call
    run returns ``None``: the UI has its own view for scenario results.
    """
    kind = run_type(run)
    if kind == TOOL_EVAL:
        return None
    scores = _scores(run)
    stored = run.get("config")
    config: Mapping[str, Any] = stored if isinstance(stored, Mapping) else {}
    if kind == "perf":
        return _perf_detail(run, scores, config)
    if kind == "spec-bench":
        return _spec_detail(scores, config)
    if kind == "context-pressure":
        return _pressure_detail(scores)
    if kind == "needle":
        return _needle_detail(run)
    if kind in _ACCURACY_NAMES:
        result = accuracy_result(_ACCURACY_NAMES[kind], run)
        if result is not None:
            accuracy = None if math.isnan(result.accuracy) else result.accuracy * 100
            stats = [
                _stat("Accuracy", accuracy, "%"),
                _stat("Items", result.items, ""),
                _stat("Errors", result.errors, ""),
            ]
            if result.detail:
                stats.append(_stat("Sample", result.detail, ""))
            return {"stats": stats, "table": None}
    return {"stats": _generic_stats(scores), "table": None}


def session_entry(run: Mapping[str, Any]) -> dict[str, Any]:
    """The fields a "same session" list needs for one run, without scores or config."""
    kind = run_type(run)
    scores = _scores(run)
    return {
        "run_id": run.get("run_id"),
        "run_type": kind,
        "type_label": type_label(kind),
        "status": run.get("status") or "completed",
        "created_at": run.get("created_at"),
        "final_score": scores.get("final_score") if kind == TOOL_EVAL else None,
        "headline": headline(run),
    }


# ---------------------------------------------------------------------------
# Per-mode details
# ---------------------------------------------------------------------------


def _perf_detail(
    run: Mapping[str, Any], scores: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    points = perf_points(run)
    stats = [
        _stat("Preset", config.get("perf_preset"), ""),
        _stat("Prompt tokens", _number(config.get("pp")), ""),
        _stat("Generated tokens", _number(config.get("tg")), ""),
        _stat("Cells", len(points), ""),
        _stat("Failed cells", _number(scores.get("failed")) or 0, ""),
    ]
    rows = [
        {
            "depth": point.depth,
            "concurrency": point.concurrency,
            "pp_tps": _round(point.pp_tps),
            "tg_tps": _round(point.tg_tps),
            "tg_tps_total": _round(point.tg_tps_total),
            "ttft_ms": _round(point.ttft_ms, 0),
            "cached_ttft_ms": _round(point.cached_ttft_ms, 0),
        }
        for point in points
    ]
    columns = [
        _column("depth", "Depth", ""),
        _column("concurrency", "Concurrency", ""),
        _column("pp_tps", "Prefill", "tok/s"),
        _column("tg_tps", "Decode per request", "tok/s"),
        _column("tg_tps_total", "Decode total", "tok/s"),
        _column("ttft_ms", "TTFT", "ms"),
    ]
    if any(row["cached_ttft_ms"] is not None for row in rows):
        columns.append(_column("cached_ttft_ms", "Cached TTFT", "ms"))
    return {"stats": _present(stats), "table": _table(columns, rows)}


def _spec_detail(scores: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    results = _rows(scores.get("results"))
    acceptance = _mean_of(results, "acceptance_rate")
    stats = [
        _stat("Method", config.get("method"), ""),
        _stat("Cells", len(results), ""),
        _stat("Failed cells", _number(scores.get("failed")) or 0, ""),
        _stat("Mean acceptance", acceptance * 100 if acceptance is not None else None, "%"),
        _stat("Mean speedup", _mean_of(results, "speedup_ratio"), "×", digits=2),
    ]
    rows = [
        {
            "prompt_type": row.get("prompt_type"),
            "depth": _number(row.get("depth")),
            "tg_tps": _round(_number(row.get("tg_tps"))),
            "effective_tg_tps": _round(_number(row.get("effective_tg_tps"))),
            "acceptance_rate": _round(_scaled(row.get("acceptance_rate"), 100)),
            "acceptance_length": _round(_number(row.get("acceptance_length")), 2),
            "speedup_ratio": _round(_number(row.get("speedup_ratio")), 2),
            "ttft_ms": _round(_number(row.get("ttft_ms")), 0),
        }
        for row in results
    ]
    columns = [
        _column("prompt_type", "Prompt", ""),
        _column("depth", "Depth", ""),
        _column("tg_tps", "Decode", "tok/s"),
        _column("effective_tg_tps", "Effective", "tok/s"),
        _column("acceptance_rate", "Acceptance", "%"),
        _column("acceptance_length", "Accepted length", ""),
        _column("speedup_ratio", "Speedup", "×"),
        _column("ttft_ms", "TTFT", "ms"),
    ]
    # Drop columns no row measured, such as speedup without a baseline.
    columns = [c for c in columns if any(row[c["key"]] is not None for row in rows)]
    return {"stats": _present(stats), "table": _table(columns, rows)}


def _pressure_detail(scores: Mapping[str, Any]) -> dict[str, Any]:
    levels = _rows(scores.get("level_results"))
    stats = [
        _stat("Levels", _number(scores.get("levels")), ""),
        _stat("Planned levels", _number(scores.get("planned_levels")), ""),
        _stat("Holds to", _scaled(scores.get("breaking_point"), 100), "% fill"),
        _stat("Degrades at", _scaled(scores.get("first_degradation"), 100), "% fill"),
        _stat("Stopped", scores.get("stop_reason"), ""),
    ]
    rows = [
        {
            "ratio": _round(_scaled(level.get("ratio"), 100)),
            "fill_tokens": _number(level.get("fill_tokens")),
            "score_pct": _round(_number(level.get("score_pct"))),
            "pass_count": _number(level.get("pass_count")),
            # The error text can quote the server URL, so the table only flags it.
            "outcome": "error" if level.get("error") else "",
        }
        for level in levels
    ]
    columns = [
        _column("ratio", "Fill", "%"),
        _column("fill_tokens", "Fill tokens", ""),
        _column("score_pct", "Score", "%"),
        _column("pass_count", "Passed", ""),
        _column("outcome", "Outcome", ""),
    ]
    return {"stats": _present(stats), "table": _table(columns, rows)}


def _needle_detail(run: Mapping[str, Any]) -> dict[str, Any]:
    summary = needle_summary(run)
    stats = [
        _stat("Retrieval", summary.accuracy * 100, "%"),
        _stat("Effective context", summary.effective_context, "tokens"),
        _stat("Errored cells", summary.errors, ""),
    ]
    rows = [
        {
            "context_tokens": cell.context_tokens,
            "depth_percent": _round(cell.depth_percent),
            "outcome": "error" if cell.error else "found" if cell.found else "missed",
        }
        for cell in summary.cells
    ]
    columns = [
        _column("context_tokens", "Context", "tokens"),
        _column("depth_percent", "Depth", "%"),
        _column("outcome", "Outcome", ""),
    ]
    return {"stats": _present(stats), "table": _table(columns, rows)}


def _generic_stats(scores: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Top-level numeric scores, for a mode without its own layout."""
    stats = [
        _stat(key.replace("_", " ").capitalize(), _round(float(value), 2), "")
        for key, value in scores.items()
        if key not in _GENERIC_SKIP and _number(value) is not None
    ]
    return stats[:_GENERIC_LIMIT]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scores(run: Mapping[str, Any]) -> Mapping[str, Any]:
    scores = run.get("scores")
    return scores if isinstance(scores, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    """A finite number, or ``None`` for anything else, booleans included."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _scaled(value: Any, factor: float) -> float | None:
    number = _number(value)
    return number * factor if number is not None else None


def _mean_of(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [number for row in rows if (number := _number(row.get(key))) is not None]
    return fmean(values) if values else None


def _round(value: float | None, digits: int = 1) -> float | None:
    if value is None:
        return None
    return round(value) if digits == 0 else round(value, digits)


def _figure(
    label: str, value: float | None, unit: str, *, detail: str | None = None
) -> dict[str, Any]:
    return {"label": label, "value": _round(value), "unit": unit, "detail": detail}


def _stat(label: str, value: Any, unit: str, *, digits: int = 1) -> dict[str, Any]:
    if isinstance(value, float):
        value = _round(value, digits)
    return {"label": label, "value": value, "unit": unit}


def _present(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [stat for stat in stats if stat["value"] is not None and stat["value"] != ""]


def _column(key: str, label: str, unit: str) -> dict[str, str]:
    return {"key": key, "label": label, "unit": unit}


def _table(columns: list[dict[str, str]], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return {"columns": columns, "rows": rows} if rows else None
