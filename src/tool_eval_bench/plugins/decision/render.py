"""Markdown rendering for the decision-model report.

The report is the visualization: text bars for probability distributions,
a reliability table, and a confusion matrix.  Bars use block characters so they
survive any Markdown viewer and a terminal.
"""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.plugin import BenchmarkResult
from tool_eval_bench.plugins.decision.metrics import HIGH_CONFIDENCE

_BAR_WIDTH = 10
# How many confident mistakes to list before pointing at the full trace.
_MAX_MISTAKES = 10


def bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    """Render ``fraction`` in [0, 1] as a fixed-width block bar."""
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _cell(text: str, limit: int = 90) -> str:
    flat = text.replace("|", "\\|").replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def render_report(result: BenchmarkResult) -> list[str]:
    d = result.details
    rows = result.item_results
    lines = [
        "## Decision Models — Single-Pass Scoring",
        "",
        f"**Accuracy:** {result.score:.1f}% ({d['correct']}/{d['total']} base items)",
        f"**Rating:** {result.rating}",
        f"**Requests:** {d['requests']} ({d['errors']} errors)",
        f"**Duration:** {result.duration_seconds:.1f}s · **Input tokens:** {result.total_tokens:,}",
        "",
        *_categories(d),
        *_calibration(d),
        *_confusion(d),
        *_ordinal(d),
        *_robustness(d),
        *_latency(d),
        *_mistakes(rows),
        *_trace(rows),
    ]
    return lines


def _categories(d: dict[str, Any]) -> list[str]:
    cats = d.get("categories", {})
    if not cats:
        return []
    lines = [
        "### Accuracy by Category",
        "",
        "| Category | Correct | Total | Accuracy | |",
        "|---|---|---|---|---|",
    ]
    for name, c in cats.items():
        lines.append(
            f"| {name} | {c['correct']} | {c['total']} | {c['accuracy']:.1f}% "
            f"| `{bar(c['accuracy'] / 100)}` |"
        )
    return [*lines, ""]


def _calibration(d: dict[str, Any]) -> list[str]:
    cal = d.get("calibration", {})
    bins = cal.get("bins", [])
    if not bins:
        return []
    lines = [
        "### Calibration",
        "",
        f"- **ECE:** {cal['ece']:.3f} (0 is perfectly calibrated)",
        f"- **Brier score:** {cal['brier']:.3f} (0 best, 2 worst)",
        f"- **Log loss:** {cal['log_loss']:.3f}",
        f"- **Mean confidence:** {cal['mean_confidence']:.1%} against "
        f"{d['accuracy'] / 100:.1%} accuracy on answered base items",
        f"- **Confident mistakes** (wrong at ≥{HIGH_CONFIDENCE:.0%} confidence): "
        f"{cal['high_confidence_errors']}",
        "",
        "Each row groups predictions by stated confidence. A calibrated model has "
        "accuracy close to confidence in every row.",
        "",
        "| Confidence | n | Mean confidence | Accuracy | Gap | Confidence · Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for b in bins:
        gap = b["accuracy"] - b["mean_confidence"]
        lines.append(
            f"| {b['low']:.1f}–{b['high']:.1f} | {b['count']} | {b['mean_confidence']:.2f} "
            f"| {b['accuracy']:.2f} | {gap:+.2f} "
            f"| `{bar(b['mean_confidence'])}` `{bar(b['accuracy'])}` |"
        )
    return [*lines, ""]


def _confusion(d: dict[str, Any]) -> list[str]:
    conf = d.get("confusion", {})
    labels: list[str] = conf.get("labels", [])
    if not labels:
        return []
    matrix: dict[str, dict[str, int]] = conf["matrix"]
    # A prediction outside the option list is a hallucinated label; keep its column.
    extras = sorted({p for row in matrix.values() for p in row} - set(labels))
    columns = [*labels, *extras]
    lines = [
        "### Routing Confusion Matrix",
        "",
        "Rows are the gold team, columns are the predicted team.",
        "",
        "| gold \\ predicted | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for gold in labels:
        row = matrix.get(gold, {})
        cells = [
            f"**{row.get(c, 0)}**" if c == gold else str(row.get(c, 0) or "·") for c in columns
        ]
        lines.append(f"| {gold} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _ordinal(d: dict[str, Any]) -> list[str]:
    o = d.get("ordinal", {})
    if not o.get("count"):
        return []
    return [
        "### Urgency Scale",
        "",
        f"- **Mean absolute error:** {o['mae']:.2f} levels (expected level against gold)",
        f"- **Within one level:** {o['within_one']:.1f}%",
        "",
    ]


def _robustness(d: dict[str, Any]) -> list[str]:
    r = d.get("robustness", {})
    if all(v is None for v in r.values()):
        return []

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f}%"

    return [
        "### Robustness",
        "",
        "Routing items asked again with the options changed. Invariance is the share of "
        "answers that match the original answer, right or wrong.",
        "",
        "| Variant | Accuracy | Invariance |",
        "|---|---|---|",
        f"| Options in reverse order | {pct(r.get('shuffled_accuracy'))} "
        f"| {pct(r.get('shuffled_invariance'))} |",
        f"| Options renamed to opaque labels | {pct(r.get('opaque_accuracy'))} "
        f"| {pct(r.get('opaque_invariance'))} |",
        "",
    ]


def _latency(d: dict[str, Any]) -> list[str]:
    lat = d.get("latency_ms", {})
    if not lat:
        return []
    return [
        "### Latency",
        "",
        f"- **p50:** {lat['p50']:.1f} ms · **p95:** {lat['p95']:.1f} ms · "
        f"**mean:** {lat['mean']:.1f} ms per request",
        "",
    ]


def _distribution_lines(row: dict[str, Any]) -> list[str]:
    dist: dict[str, float] = row["distribution"]
    width = max(len(k) for k in dist)
    lines = []
    # Score levels read best in scale order; options read best most likely first.
    if row["question_type"] == "ScoreQuestion":
        ordered = sorted(dist.items(), key=lambda kv: int(kv[0]))
    else:
        ordered = sorted(dist.items(), key=lambda kv: -kv[1])
    for name, p in ordered:
        marks = []
        if name == row["gold"]:
            marks.append("gold")
        if name == row["predicted"]:
            marks.append("pred")
        suffix = f"  ◀ {'+'.join(marks)}" if marks else ""
        lines.append(f"    {name:<{width}}  {bar(p)} {p:.3f}{suffix}")
    return lines


def _mistakes(rows: list[dict[str, Any]]) -> list[str]:
    wrong = [r for r in rows if r["variant"] == "base" and not r["is_error"] and not r["correct"]]
    if not wrong:
        return []
    wrong.sort(key=lambda r: -r["confidence"])
    lines = [
        f"### Mistakes ({len(wrong)} base items)",
        "",
        "Most confident first. A confident mistake is worse than an uncertain one.",
        "",
        "| Item | Message | Gold | Predicted | Confidence |",
        "|---|---|---|---|---|",
    ]
    for r in wrong[:_MAX_MISTAKES]:
        lines.append(
            f"| {r['id']} | {_cell(r['state'])} | {r['gold']} | {r['predicted']} "
            f"| {r['confidence']:.2f} |"
        )
    if len(wrong) > _MAX_MISTAKES:
        lines.append("")
        lines.append(f"{len(wrong) - _MAX_MISTAKES} more in the full trace below.")
    return [*lines, ""]


def _trace(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Full Trace",
        "",
        "<details>",
        f"<summary>Probability distribution for all {len(rows)} requests</summary>",
        "",
        "```",
    ]
    for r in rows:
        if r["is_error"]:
            lines.append(f"{r['id']}  ERROR  {r.get('error', '')}")
            lines.append(f"    {_cell(r['state'], 100)}")
            continue
        mark = "✓" if r["correct"] else "✗"
        lines.append(f"{r['id']}  {mark}  {_cell(r['state'], 100)}")
        lines.extend(_distribution_lines(r))
    lines.extend(["```", "", "</details>", ""])
    return lines
