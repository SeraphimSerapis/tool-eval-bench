"""Markdown report rendering for toolscale benchmark results."""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.domain.plugin import BenchmarkResult
from tool_eval_bench.plugins.toolscale.aggregate import _label, _slices
from tool_eval_bench.plugins.toolscale.grid import AXES

# Failures listed in the Markdown report. Every item is still counted; past
# this the list stops being readable and the cell tables carry the signal.
_MAX_LISTED_FAILURES = 200

_AXIS_NOUNS = {"cue": "cue", "density": "fillers", "descriptions": "descriptions"}

# Title and reading of each condition-axis effect table.
_AXIS_EFFECT_TEXT = {
    "cue": (
        "Cue Effect",
        "Accuracy with the literal query minus accuracy with the implied query, on the "
        "identical tool list, paired by target. Positive means the model relied on the "
        "query naming the distinguishing word.",
    ),
    "density": (
        "Density Effect",
        "Accuracy with sparse fillers minus accuracy with crowded fillers, with the same "
        "target, neighbours, and slot layout, paired by target. Positive means a catalog "
        "dense with near-duplicate clusters costs accuracy.",
    ),
    "descriptions": (
        "Description Effect",
        "Accuracy with clean descriptions minus accuracy with degraded ones, on the identical "
        "tool list and query, paired by target. Terse descriptions share one line per family, "
        "so near-duplicates differ only by name; bare tools have no descriptions at all. "
        "Positive means the model relied on the descriptions.",
    ),
}


def _report_present(d: dict[str, Any]) -> dict[str, list[str]]:
    """The values of each condition axis a stored run swept."""
    return {
        "cue": d.get("cues", ["literal"]),
        "density": d.get("densities", ["sparse"]),
        "descriptions": d.get("descriptions", ["clean"]),
    }


def _cell(
    details: dict[str, Any], count: int, level: str, slice_: dict[str, str]
) -> dict[str, Any] | None:
    baseline = {axis: values[0] for axis, values in AXES}
    for cell in details.get("cells", []):
        if (
            cell["count"] == count
            and cell["similarity"] == level
            and all(cell.get(a, baseline[a]) == v for a, v in slice_.items())
        ):
            return cell
    return None


def _entry_label(
    entry: dict[str, Any],
    present: dict[str, list[str]],
    skip: tuple[str, ...] = (),
    fallback: str | None = None,
) -> str:
    """A stored cell or effect's values on the varied axes, for a table's Slice column."""
    baseline = {axis: values[0] for axis, values in AXES}
    slice_ = {axis: entry.get(axis, baseline[axis]) for axis, _ in AXES}
    return _label(slice_, present, skip=skip, fallback=fallback)


def _slice_heading(slice_: dict[str, str], present: dict[str, list[str]]) -> str:
    """``implied cue, crowded fillers``: the cue always, other axes when varied."""
    parts = [f"{slice_['cue']} cue"]
    for axis in ("density", "descriptions"):
        if len(present[axis]) > 1:
            parts.append(f"{slice_[axis]} {_AXIS_NOUNS[axis]}")
    return ", ".join(parts)


def _table_text(value: Any, limit: int) -> str:
    """Model or catalog text made safe for one Markdown table cell.

    GFM splits a row on any unescaped pipe, even inside a code span, and
    CommonMark ends a line at a lone carriage return, so both are neutralised.
    """
    text = re.sub(r"[\r\n]+", " ", str(value))
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    # Escape after truncating, so a cut never separates a backslash from its pipe.
    return text.replace("|", "\\|")


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def _grid_table(
    d: dict[str, Any],
    counts: list[int],
    levels: list[str],
    slice_: dict[str, str],
    render: Any,
) -> list[str]:
    lines = [
        "| Similarity | " + " | ".join(f"N={n}" for n in counts) + " |",
        "|---" + "|---:" * len(counts) + "|",
    ]
    for level in levels:
        row = []
        for count in counts:
            cell = _cell(d, count, level, slice_)
            row.append("" if cell is None else render(cell))
        lines.append(f"| {level} | " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _axis_effect_lines(
    title: str, reading: str, effects: list[dict[str, Any]], axis: str, present: dict
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        reading,
        "",
        "| Slice | Versus | N | Similarity | Delta (pts) | 95% CI | Targets |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for e in effects:
        lines.append(
            f"| {_entry_label(e, present, skip=(axis,), fallback='all')} | {e['versus']} | "
            f"{e['count']} | "
            f"{e['similarity']} | {_signed(e['delta'])} | "
            f"{_signed(e['ci_low'])} to {_signed(e['ci_high'])} | {e['targets']} |"
        )
    lines.append("")
    return lines


def render_report(result: BenchmarkResult) -> list[str]:
    """Markdown for the run report."""
    d = result.details
    counts: list[int] = d.get("counts", [])
    levels: list[str] = d.get("similarities", [])
    present = _report_present(d)
    slices = _slices(present)
    catalog = d.get("catalog", {})
    trials = d.get("trials", 1)

    lines = [
        "## Toolscale: Tool Count x Description Similarity",
        "",
        f"**Accuracy when a tool was called:** {d.get('accuracy_when_called', 0):.1f}% "
        f"({d.get('grid_correct', 0)}/{d.get('grid_called', 0)})",
        f"**Call rate:** {d.get('call_rate', 0):.1f}% "
        f"({d.get('grid_called', 0)}/{d.get('grid_items', 0)})",
        f"**Combined accuracy (stored score):** {d.get('accuracy', 0):.1f}%",
        f"**Rating:** {result.rating}",
        f"**Catalog:** v{catalog.get('version', '?')} ({catalog.get('source', '?')}, "
        f"{catalog.get('tools', 0)} tools, sha256 `{str(catalog.get('sha256', ''))[:12]}`)",
        f"**Cues:** {', '.join(present['cue'])} · **Fillers:** {', '.join(present['density'])}"
        f" · **Descriptions:** {', '.join(present['descriptions'])} · **Trials:** {trials}",
        f"**Duration:** {result.duration_seconds:.1f}s",
        f"**Tokens:** {result.total_tokens:,}",
        "",
        "Each item offers the target tool, 3 neighbours from one similarity tier, and unrelated "
        "fillers up to N tools. Only the first tool call's name is graded. The two headline "
        "numbers are reported apart on purpose: accuracy when called measures tool confusion, "
        "the call rate measures whether the model acted at all (a clarifying question counts "
        "as no call). Request errors count as misses; intervals are 95% Wilson.",
        "",
    ]

    for axis, title in (("cue", "Cue"), ("density", "Fillers"), ("descriptions", "Descriptions")):
        breakdown = d.get(f"by_{axis}", {})
        if len(present[axis]) < 2 or not breakdown:
            continue
        lines.extend(
            [
                f"### Headline by {title}",
                "",
                f"| {title} | Accuracy when called | Call rate | Combined accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for value in present[axis]:
            h = breakdown.get(value, {})
            lines.append(
                f"| {value} | {h.get('accuracy_when_called', 0):.1f}% "
                f"({h.get('grid_correct', 0)}/{h.get('grid_called', 0)}) | "
                f"{h.get('call_rate', 0):.1f}% | {h.get('accuracy', 0):.1f}% |"
            )
        lines.append("")

    if counts and levels:

        def combined(c: dict[str, Any]) -> str:
            text = f"{c['accuracy']:.1f}% ({c['ci_low']:.0f}-{c['ci_high']:.0f})"
            if trials > 1:
                text += f" ±{c['trial_stddev']:.1f}"
            return text

        for slice_ in slices:
            heading = _slice_heading(slice_, present)
            lines.extend(
                [f"### Accuracy When a Tool Was Called ({heading})", ""]
                + _grid_table(
                    d,
                    counts,
                    levels,
                    slice_,
                    lambda c: f"{c['accuracy_when_called']:.1f}% ({c['correct']}/{c['called']})",
                )
            )
            lines.extend(
                [f"### Call Rate ({heading})", ""]
                + _grid_table(
                    d,
                    counts,
                    levels,
                    slice_,
                    lambda c: f"{c['call_rate']:.1f}% ({c['called']}/{c['items']})",
                )
            )
            lines.extend(
                [f"### Combined Accuracy ({heading})", ""]
                + _grid_table(d, counts, levels, slice_, combined)
            )
        if trials > 1:
            lines.extend(["*± is the standard deviation of cell accuracy across trials.*", ""])

    controls = d.get("control") or {}
    for label, control in controls.items():
        lines.extend(
            [
                f"**Control (N=1, target only, {label}):** {control['accuracy']:.1f}% "
                f"({control['correct']}/{control['items']}); call rate "
                f"{control['call_rate']:.1f}%, truncated {control['outcomes'].get('truncated', 0)}",
                "",
            ]
        )

    for axis, key in (
        ("cue", "cue_effects"),
        ("density", "density_effects"),
        ("descriptions", "description_effects"),
    ):
        effects = d.get(key, [])
        if effects:
            title, reading = _AXIS_EFFECT_TEXT[axis]
            lines.extend(_axis_effect_lines(title, reading, effects, axis, present))

    sim_effects = d.get("similarity_effects", [])
    if sim_effects:
        lines.extend(
            [
                "### Similarity Effect at Fixed N",
                "",
                "Accuracy with distinct neighbours minus accuracy with more similar neighbours, "
                "paired by target. Positive means similarity costs accuracy.",
                "",
                "| Slice | N | Versus | Delta (pts) | 95% CI | Targets |",
                "|---|---:|---|---:|---|---:|",
            ]
        )
        for e in sim_effects:
            label = _entry_label(e, present)
            lines.append(
                f"| {label} | {e['count']} | {e['versus']} | {_signed(e['delta'])} | "
                f"{_signed(e['ci_low'])} to {_signed(e['ci_high'])} | {e['targets']} |"
            )
        lines.append("")

    count_effects = d.get("count_effects", [])
    if count_effects:
        lines.extend(
            [
                "### Count Effect at Fixed Similarity",
                "",
                "Accuracy at the smallest N minus accuracy at the largest, paired by target. "
                "A positive delta means more tools cost accuracy. The slope is a least-squares "
                "fit of cell accuracy against log2(N), in points per doubling: negative means "
                "accuracy falls as the list grows.",
                "",
                "| Slice | Similarity | From N | To N | Delta (pts) | 95% CI | Slope per doubling |",
                "|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for e in count_effects:
            slope = e.get("slope_per_doubling")
            label = _entry_label(e, present)
            lines.append(
                f"| {label} | {e['similarity']} | {e['from_count']} | {e['to_count']} | "
                f"{_signed(e['delta'])} | {_signed(e['ci_low'])} to {_signed(e['ci_high'])} | "
                f"{'' if slope is None else _signed(slope)} |"
            )
        lines.append("")

    cells = d.get("cells", [])
    if cells:
        lines.extend(
            [
                "### Error Mix",
                "",
                "A wrong neighbour in the near tier is similarity confusion. A wrong filler is a "
                "count or attention failure. In the distinct tier both are unrelated tools, so "
                "their rates are the baseline.",
                "",
                "| Slice | N | Similarity | Correct | Wrong neighbour | Wrong filler | "
                "Hallucinated | No call | Truncated | Error | Prompt tokens |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cells:
            o = c["outcomes"]
            tokens = c.get("mean_prompt_tokens")
            label = _entry_label(c, present)
            lines.append(
                f"| {label} | {c['count']} | {c['similarity']} | "
                f"{o['correct']} | {o['wrong_neighbour']} | {o['wrong_filler']} | "
                f"{o['hallucinated']} | {o['no_call']} | {o['truncated']} | {o['error']} | "
                f"{'' if tokens is None else f'{tokens:,}'} |"
            )
        lines.append("")

    queries = d.get("query_check") or {}
    if queries:
        lines.extend(
            [
                "### Query Check",
                "",
                "Word overlap (TF-IDF cosine) between each target's query and its own definition, "
                "and the margin over its closest family sibling. A literal query should lean "
                "toward the target; an implied query should not, so the model has to infer.",
                "",
                "| Query | Query to target | Margin over closest sibling | Sibling overlaps more |",
                "|---|---:|---:|---:|",
            ]
        )
        for cue, q in queries.items():
            lines.append(
                f"| {cue} | {q['query_to_target']:.3f} | {q['margin_over_sibling']:+.3f} | "
                f"{q['sibling_wins']:.1f}% |"
            )
        lines.append("")

    check = d.get("similarity_check") or {}
    if check:
        lines.extend(
            [
                "### Similarity Check",
                "",
                f"Measured with {check.get('metric', 'tf-idf cosine')} on the clean definitions, "
                "whatever description level the run used. The tiers are defined by the "
                "catalog's structure; this confirms the ordering.",
                "",
                "| Tier | Target to neighbours |",
                "|---|---:|",
            ]
        )
        for level, score in check.get("target_to_neighbours", {}).items():
            lines.append(f"| {level} | {score:.3f} |")
        if check.get("target_to_fillers") is not None:
            lines.append(f"| fillers | {check['target_to_fillers']:.3f} |")
        lines.append("")
        crowding = check.get("filler_crowding") or {}
        if crowding:
            lines.extend(
                [
                    "Filler crowding, each filler's similarity to its closest other filler "
                    "on the largest list:",
                    "",
                    "| Fillers | Closest-filler similarity |",
                    "|---|---:|",
                    *(f"| {density} | {value:.3f} |" for density, value in crowding.items()),
                    "",
                ]
            )
        nearest = check.get("nearest_by_count", [])
        if nearest:
            lines.extend(
                [
                    "Target's nearest-neighbour similarity within the offered list:",
                    "",
                    "| Similarity | " + " | ".join(f"N={n}" for n in counts) + " |",
                    "|---" + "|---:" * len(counts) + "|",
                ]
            )
            lookup = {(n["similarity"], n["count"]): n["nearest"] for n in nearest}
            for level in levels:
                values = [lookup.get((level, count)) for count in counts]
                lines.append(
                    f"| {level} | "
                    + " | ".join("" if v is None else f"{v:.3f}" for v in values)
                    + " |"
                )
            lines.append("")
        ref = check.get("category_l_reference")
        if ref:
            lines.extend(
                [
                    f"**Category L reference:** {ref['scenario']} offers {ref['count']} tools; "
                    f"its target `{ref['target']}` has nearest neighbour `{ref['nearest']}` at "
                    f"{ref['nearest_similarity']:.3f} (mean to all others "
                    f"{ref['mean_similarity']:.3f}) on the same scale.",
                    "",
                ]
            )

    positions = d.get("by_position", [])
    if positions:
        lines.extend(
            [
                "### Accuracy by Target Position",
                "",
                "| Target position | Items | Accuracy |",
                "|---|---:|---:|",
            ]
        )
        for p in positions:
            lines.append(f"| {p['bucket']} | {p['items']} | {p['accuracy']:.1f}% |")
        lines.append("")

    failures = [r for r in result.item_results if not r.get("correct")]
    if failures:
        lines.append(f"### Failed Items ({len(failures)} total)")
        lines.append("")
        if catalog.get("source") == "custom":
            lines.extend(
                [
                    "Tool names and queries are withheld because this run used a private "
                    "catalog. Counts by outcome are in the error mix above.",
                    "",
                ]
            )
            return lines
        # Wrong picks first: they are the confusion signal, and a long run of
        # clarifying questions would otherwise push them past the listing cap.
        failures.sort(key=lambda f: (f.get("picked") is None, f["item_id"]))
        lines.extend(["| Item | Query | Picked | Outcome |", "|---|---|---|---|"])
        for f in failures[:_MAX_LISTED_FAILURES]:
            query = _table_text(f.get("query", ""), 100)
            if f.get("picked"):
                picked = f"`{_table_text(f['picked'], 80)}`"
            elif f.get("error"):
                picked = f"*(request failed)* {_table_text(f['error'], 80)}"
            elif f.get("reply"):
                picked = f"*(no call)* {_table_text(f['reply'], 80)}"
            else:
                picked = "*(none)*"
            lines.append(f"| `{f['item_id']}` | {query} | {picked} | {f['outcome']} |")
        if len(failures) > _MAX_LISTED_FAILURES:
            lines.append("")
            lines.append(
                f"*{len(failures) - _MAX_LISTED_FAILURES} more failures not listed. Every item "
                "is counted in the tables above, and the stored run keeps each item's outcome.*"
            )
        lines.append("")

    return lines
