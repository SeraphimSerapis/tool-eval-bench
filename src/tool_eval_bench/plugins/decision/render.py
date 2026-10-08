"""Markdown rendering for the decision-model report.

The report is the visualization: accuracy bars per question type, workflow, and
question, a reliability table, and the predicted and gold distribution of every
decision.  Bars use block characters so they survive any Markdown viewer and a
terminal.
"""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.plugin import BenchmarkResult
from tool_eval_bench.plugins.decision.metrics import HIGH_CONFIDENCE
from tool_eval_bench.plugins.decision.typed_decisions import (
    PRIOR_ACCURACY,
    TEACHER_CEILING_ACCURACY,
)

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


def attribution_line(d: dict[str, Any]) -> str:
    """Credit the third-party dataset a typed-decisions result was scored on."""
    return (
        f"Data: {d['attribution']}. Third-party data, unmodified apart from format "
        "conversion; see NOTICE next to the data file."
    )


def metric_text(value: float | None) -> str:
    """Three decimals, or n/a when no decision was answered."""
    return "n/a" if value is None else f"{value:.3f}"


def render_report(result: BenchmarkResult) -> list[str]:
    d = result.details
    cal = d["calibration"]
    return [
        "## Decision Models — Typed Decisions",
        "",
        f"> {attribution_line(d)}",
        "",
        f"**Accuracy:** {result.score:.1f}% ({d['correct']}/{d['total']} decisions)",
        f"**KL from gold:** {metric_text(d['kl'])} · **Brier:** {metric_text(d['brier'])} · "
        f"**ECE:** {cal['ece']:.3f}",
        f"**Rating:** {result.rating}",
        f"**Requests:** {d['cases']} cases, all questions of a case in one request "
        f"({d['errors']} decision errors)",
        f"**Duration:** {result.duration_seconds:.1f}s · **Input tokens:** {result.total_tokens:,}",
        "",
        "Gold is the mean of three samples from a teacher model of roughly 4B-class "
        "capability, so these scores measure agreement with that teacher, not "
        f"correctness. The dataset card puts a prior that ignores the input at "
        f"{PRIOR_ACCURACY:.1%} accuracy and the teacher's agreement with itself at "
        f"{TEACHER_CEILING_ACCURACY:.1%}. The rating uses those two points: ★ at or "
        "below the prior, ★★ above it, ★★★ at or above the teacher's self-agreement. "
        "Ceilings differ a lot between questions, so read the per-question table too.",
        "",
        "Accuracy compares the most likely answer with the gold label. KL is "
        "KL(gold ‖ prediction) in nats and Brier the summed squared difference from "
        "the gold distribution, both averaged per decision. ECE is top-label over ten "
        "bins; the card's ECE definition is unpublished, so this ECE is not "
        "comparable with the card's.",
        "",
        *_table("By Question Type", "Type", d["by_type"]),
        *_table("By Workflow", "Workflow", d["by_workflow"]),
        *_table("By Question", "Question", d["by_question"]),
        *_calibration(cal),
        *_latency(d),
        *_mistakes(result.item_results),
        *_trace(result.item_results),
    ]


def _table(title: str, label: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"| {label} | Correct | Total | Accuracy | KL | Brier | |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, g in groups.items():
        lines.append(
            f"| {name} | {g['correct']} | {g['total']} | {g['accuracy']:.1f}% "
            f"| {metric_text(g['kl'])} | {metric_text(g['brier'])} | `{bar(g['accuracy'] / 100)}` |"
        )
    return [*lines, ""]


def _calibration(cal: dict[str, Any]) -> list[str]:
    bins = cal.get("bins", [])
    if not bins:
        return []
    lines = [
        "### Calibration",
        "",
        f"- **ECE:** {cal['ece']:.3f} (top-label, ten bins, against the gold label)",
        f"- **Mean confidence:** {cal['mean_confidence']:.1%}",
        f"- **Confident disagreements** (≥{HIGH_CONFIDENCE:.0%}): {cal['high_confidence_errors']}",
        "",
        "| Confidence | n | Mean confidence | Accuracy | Gap |",
        "|---|---|---|---|---|",
    ]
    for b in bins:
        lines.append(
            f"| {b['low']:.1f}–{b['high']:.1f} | {b['count']} | {b['mean_confidence']:.2f} "
            f"| {b['accuracy']:.2f} | {b['accuracy'] - b['mean_confidence']:+.2f} |"
        )
    return [*lines, ""]


def _latency(d: dict[str, Any]) -> list[str]:
    if not d["answered"]:
        # With no successful request there is no latency to report, only zeros.
        return []
    lat = d["latency_ms"]
    return [
        "### Latency",
        "",
        f"- **p50:** {lat['p50']:.1f} ms · **p95:** {lat['p95']:.1f} ms · "
        f"**mean:** {lat['mean']:.1f} ms per request",
        "",
    ]


def _mistakes(rows: list[dict[str, Any]]) -> list[str]:
    wrong = [r for r in rows if not r["is_error"] and not r["correct"]]
    if not wrong:
        return []
    wrong.sort(key=lambda r: -r["confidence"])
    lines = [
        f"### Disagreements with Gold ({len(wrong)} decisions)",
        "",
        "Most confident first.",
        "",
        "| Decision | Gold | Gold p | Predicted | Confidence |",
        "|---|---|---|---|---|",
    ]
    for r in wrong[:_MAX_MISTAKES]:
        lines.append(
            f"| {r['id']} | {r['gold']} | {r['gold_distribution'].get(r['gold'], 0.0):.2f} "
            f"| {r['predicted']} | {r['confidence']:.2f} |"
        )
    if len(wrong) > _MAX_MISTAKES:
        lines.append("")
        lines.append(f"{len(wrong) - _MAX_MISTAKES} more in the full trace below.")
    return [*lines, ""]


def _pairs(dist: dict[str, float], order: list[str] | None = None) -> str:
    keys = [k for k in order or [] if k in dist]
    keys += [k for k in dist if k not in keys]
    return " ".join(f"{k}={dist[k]:.3f}" for k in keys)


def _trace(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Full Trace",
        "",
        "<details>",
        f"<summary>Predicted and gold distribution for all {len(rows)} decisions</summary>",
        "",
        "```",
    ]
    case_id = None
    for r in rows:
        if r["case_id"] != case_id:
            case_id = r["case_id"]
            lines.append(f"{case_id}  {_cell(r['state_preview'], 100)}")
        if r["is_error"]:
            lines.append(f"  {r['question']} ({r['question_type']})  ERROR  {r.get('error', '')}")
            continue
        mark = "✓" if r["correct"] else "✗"
        lines.append(
            f"  {r['question']} ({r['question_type']})  {mark}  pred {r['predicted']}  "
            f"gold {r['gold']}  KL {r['kl']:.3f}  Brier {r['brier']:.3f}"
        )
        lines.append(f"      pred  {_pairs(r['distribution'])}")
        lines.append(f"      gold  {_pairs(r['gold_distribution'], list(r['distribution']))}")
    lines.extend(["```", "", "</details>", ""])
    return lines
