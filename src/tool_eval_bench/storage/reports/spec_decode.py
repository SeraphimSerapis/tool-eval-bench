"""Speculative decoding benchmark report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tool_eval_bench.storage.reports._common import (
    markdown_label,
    report_filename,
)


def write_spec_decode_report(
    root: Path,
    run_id: str,
    model: str,
    spec_samples: list[Any],
    label: str | None = None,
) -> Path:
    """Write a Markdown report for speculative decoding benchmark results."""
    now = datetime.now(timezone.utc)
    folder = root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / report_filename(run_id, label)

    label_line = [f"- **Label**: {markdown_label(label)}"] if label else []
    md = [
        f"# Speculative Decoding Benchmark — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: spec-bench",
        *label_line,
    ]

    # Detect method
    methods = {s.spec_method for s in spec_samples if hasattr(s, "spec_method")}
    method_str = ", ".join(sorted(methods)) if methods else "unknown"
    md.append(f"- **Spec Method**: {method_str}")

    # Check if acceptance rate is available
    has_ar = any(getattr(s, "acceptance_rate", None) is not None for s in spec_samples)
    if not has_ar:
        md.extend(
            [
                "",
                "> [!NOTE]",
                "> Acceptance rate metrics were not available from the server.",
                "> Effective t/s (wall-clock based) still captures the real benefit",
                "> of MTP / speculative decoding.",
            ]
        )

    md.append("")

    # Results table
    md.extend(["## Results", ""])

    has_draft = any(getattr(s, "draft_tps", None) is not None for s in spec_samples)

    if has_ar and has_draft:
        md.append(
            "| Prompt | Depth | Eff t/s | Stream t/s | α (accept) | Waste "
            "| τ (length) | Window | Draft t/s | Speedup | TTFT (ms) | Total (ms) | Tokens |"
        )
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    elif has_ar:
        md.append(
            "| Prompt | Depth | Eff t/s | Stream t/s | α (accept) | Waste "
            "| τ (length) | Speedup | TTFT (ms) | Total (ms) | Tokens |"
        )
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        md.append(
            "| Prompt | Depth | Eff t/s | Stream t/s | Speedup | TTFT (ms) | Total (ms) | Tokens |"
        )
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for s in spec_samples:
        eff = getattr(s, "effective_tg_tps", 0)
        tg = getattr(s, "tg_tps", 0)
        ar = getattr(s, "acceptance_rate", None)
        wr = getattr(s, "waste_ratio", None)
        al = getattr(s, "acceptance_length", None)
        dw = getattr(s, "draft_window", None)
        dt = getattr(s, "draft_tps", None)
        sp = getattr(s, "speedup_ratio", None)
        ttft = getattr(s, "ttft_ms", 0)
        total = getattr(s, "total_ms", 0)
        tg_tok = getattr(s, "tg_tokens", 0)
        prompt_type = getattr(s, "prompt_type", "?")
        depth = getattr(s, "depth", 0)

        sp_str = f"{sp:.2f}x" if sp is not None else "—"

        if has_ar and has_draft:
            ar_str = f"{ar * 100:.1f}%" if ar is not None else "—"
            wr_str = f"{wr * 100:.0f}%" if wr is not None else "—"
            al_str = f"{al:.1f}" if al is not None else "—"
            dw_str = f"{dw:.0f}" if dw is not None else "—"
            dt_str = f"{dt:,.1f}" if dt is not None else "—"
            md.append(
                f"| {prompt_type} | {depth} | {eff:,.1f} | {tg:,.1f} "
                f"| {ar_str} | {wr_str} | {al_str} | {dw_str} | {dt_str} | {sp_str} "
                f"| {ttft:,.0f} | {total:,.0f} | {tg_tok} |"
            )
        elif has_ar:
            ar_str = f"{ar * 100:.1f}%" if ar is not None else "—"
            wr_str = f"{wr * 100:.0f}%" if wr is not None else "—"
            al_str = f"{al:.1f}" if al is not None else "—"
            md.append(
                f"| {prompt_type} | {depth} | {eff:,.1f} | {tg:,.1f} "
                f"| {ar_str} | {wr_str} | {al_str} | {sp_str} "
                f"| {ttft:,.0f} | {total:,.0f} | {tg_tok} |"
            )
        else:
            md.append(
                f"| {prompt_type} | {depth} | {eff:,.1f} | {tg:,.1f} "
                f"| {sp_str} "
                f"| {ttft:,.0f} | {total:,.0f} | {tg_tok} |"
            )

    md.append("")

    # Acceptance rate bar chart (if available)
    if has_ar:
        md.extend(["## Acceptance Rate by Prompt Type", ""])
        md.append("```")
        for s in spec_samples:
            ar = getattr(s, "acceptance_rate", None)
            pt = getattr(s, "prompt_type", "?")
            depth = getattr(s, "depth", 0)
            if ar is not None:
                bar_len = int(ar * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                md.append(f"  {pt:>10} d{depth:<5} {bar} {ar * 100:.1f}%")
            else:
                md.append(f"  {pt:>10} d{depth:<5} {'·' * 40} n/a")
        md.append("```")
        md.append("")

    # Per-prompt-type summary
    prompt_groups: dict[str, list] = {}
    for s in spec_samples:
        pt = getattr(s, "prompt_type", "?")
        prompt_groups.setdefault(pt, []).append(s)

    if len(prompt_groups) > 1:
        md.extend(["## Per-Prompt-Type Summary", ""])
        if has_draft:
            md.append(
                "| Prompt Type | Avg Eff t/s | Avg Stream t/s | Avg α | Avg Waste | Avg Draft t/s |"
            )
            md.append("|---|---:|---:|---:|---:|---:|")
        else:
            md.append("| Prompt Type | Avg Eff t/s | Avg Stream t/s | Avg α | Avg Waste |")
            md.append("|---|---:|---:|---:|---:|")

        for pt, group in sorted(prompt_groups.items()):
            avg_eff = sum(s.effective_tg_tps for s in group) / len(group)
            avg_tg = sum(s.tg_tps for s in group) / len(group)
            ars = [s.acceptance_rate for s in group if s.acceptance_rate is not None]
            avg_ar = f"{sum(ars) / len(ars) * 100:.1f}%" if ars else "—"
            wrs = [s.waste_ratio for s in group if getattr(s, "waste_ratio", None) is not None]
            avg_wr = f"{sum(wrs) / len(wrs) * 100:.0f}%" if wrs else "—"
            if has_draft:
                dts = [s.draft_tps for s in group if getattr(s, "draft_tps", None) is not None]
                avg_dt = f"{sum(dts) / len(dts):,.1f}" if dts else "—"
                md.append(
                    f"| {pt} | {avg_eff:,.1f} | {avg_tg:,.1f} | {avg_ar} | {avg_wr} | {avg_dt} |"
                )
            else:
                md.append(f"| {pt} | {avg_eff:,.1f} | {avg_tg:,.1f} | {avg_ar} | {avg_wr} |")

        md.append("")

    # Draft efficiency section (when window data is available)
    with_window = [
        s
        for s in spec_samples
        if getattr(s, "draft_window", None) is not None
        and getattr(s, "acceptance_length", None) is not None
    ]
    if with_window:
        avg_window = sum(s.draft_window for s in with_window) / len(with_window)
        avg_tau = sum(s.acceptance_length for s in with_window) / len(with_window)
        utilization = (avg_tau / avg_window * 100) if avg_window > 0 else 0
        avg_waste = (
            sum(s.waste_ratio for s in with_window if getattr(s, "waste_ratio", None) is not None)
            / len(with_window)
            * 100
        )

        md.extend(
            [
                "## Draft Efficiency",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Avg Draft Window | {avg_window:.0f} tokens/step |",
                f"| Avg Acceptance Length (τ) | {avg_tau:.1f} tokens/step |",
                f"| Window Utilization | {utilization:.0f}% |",
                f"| Avg Waste | {avg_waste:.0f}% |",
            ]
        )
        if utilization < 50:
            optimal = max(int(avg_tau * 1.5), 2)
            md.extend(
                [
                    "",
                    "> [!WARNING]",
                    f"> Window utilization is low ({utilization:.0f}%). "
                    f"Only {avg_tau:.1f} of {avg_window:.0f} drafted positions are accepted on average.",
                    f"> Consider reducing `num_speculative_tokens` to ~{optimal} for better GPU efficiency.",
                ]
            )
        md.append("")

    # Interpretation guide
    md.extend(
        [
            "## Interpretation Guide",
            "",
            "- **Eff t/s** (Effective t/s): Output tokens ÷ wall-clock generation time. "
            "This is what users experience. Higher is better.",
            "- **Stream t/s**: Token generation rate measured from SSE stream timing. "
            "For standard decoding, this matches Eff t/s. For spec decode, Eff t/s "
            "is typically higher.",
            "- **α (accept)**: Acceptance rate — % of draft tokens accepted by the verifier. "
            "Higher means the draft model/MTP heads predict well for this workload.",
            "- **Waste**: Fraction of drafted tokens rejected (1 − α). Lower is better. "
            "High waste means the draft model is poorly aligned with the target.",
            "- **τ (length)**: Average acceptance length — tokens accepted per speculative step. "
            "Higher means more tokens generated per verification pass.",
            "- **Window**: Average tokens drafted per speculative step (the configured draft window). "
            "Compare with τ to see window utilization.",
            "- **Draft t/s**: Rate at which draft tokens are generated, regardless of acceptance. "
            "Compare with Eff t/s to see draft overhead.",
            "- **Speedup**: Effective t/s ÷ baseline t/s. Values > 1.0x indicate spec decode "
            "is providing a benefit.",
            "",
            "> [!TIP]",
            "> Acceptance rates vary significantly by prompt type. Code and structured tasks",
            "> typically show higher acceptance rates than creative/open-ended generation",
            "> because future tokens are more predictable.",
            "",
        ]
    )

    path.write_text("\n".join(md), encoding="utf-8")
    return path
