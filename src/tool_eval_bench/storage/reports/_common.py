"""Shared helpers for the Markdown report writers.

Path resolution, label sanitising, and the table and trace fragments that more
than one report type renders.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from pathlib import Path
from typing import Any

from tool_eval_bench.domain.models import RunContext

_HELD_OUT_LABEL = "held out"


def _default_reports_root() -> str:
    """Resolve default reports root relative to the current working directory.

    Reports are written under ``./runs/`` in whichever directory the user
    invokes the CLI from — not relative to the installed package location
    (which would land inside ``.venv/``).
    """
    return str(Path.cwd() / "runs")


def slugify_label(label: str | None, *, max_length: int = 80) -> str:
    """Return a filesystem-safe slug of a run label for report filenames.

    Lowercases, collapses whitespace/punctuation to single dashes, and keeps
    ``.`` ``-`` ``_`` so version-like strings stay readable ("3.75" →
    "3.75"). Empty labels yield ``""`` so callers can fall back to plain
    run-ID filenames. Labels without an ASCII representation use a stable hash
    marker so every non-empty label still produces an identifiable file.
    """
    if not label:
        return ""
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalized.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    if not slug:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
        slug = f"label-{digest}"
    return slug[:max_length]


def safe_label_text(label: str) -> str:
    """Make control characters visible without changing persisted metadata."""
    visible: list[str] = []
    for char in label:
        if char == "\n":
            visible.append(r"\n")
        elif char == "\r":
            visible.append(r"\r")
        elif char == "\t":
            visible.append(r"\t")
        elif unicodedata.category(char).startswith("C"):
            visible.append(f"\\u{ord(char):04x}")
        else:
            visible.append(char)
    return "".join(visible)


def markdown_label(label: str, *, table_cell: bool = False) -> str:
    """Render a label as inert inline HTML for Markdown reports."""
    visible = html.escape(safe_label_text(label), quote=False)
    if table_cell:
        visible = visible.replace("|", "&#124;")
    return f"<code>{visible}</code>"


def report_filename(run_id: str, label: str | None, suffix: str = "") -> str:
    """Report file name: ``{run_id}--{slug}{suffix}.md``, or plain without a label.

    ``suffix`` is used for derived reports (e.g. ``_summary``), so the slug
    stays adjacent to the run ID: ``{run_id}--{slug}_summary.md``.
    """
    slug = slugify_label(label)
    if not slug:
        return f"{run_id}{suffix}.md"
    return f"{run_id}--{slug}{suffix}.md"


def _trace_block(raw_log: str) -> list[str]:
    """Return a Markdown fence long enough to preserve a complete raw trace."""
    longest = max((len(match) for match in re.findall(r"`+", raw_log)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", raw_log or "(empty trace)", fence]


def _markdown_table_cell(value: object) -> str:
    """Render untrusted report text as one inert Markdown table cell."""
    text = safe_label_text(str(value))
    return html.escape(text, quote=False).replace("|", "&#124;").replace("\\n", "<br>")


def _markdown_heading(value: object) -> str:
    """Render an untrusted heading label without allowing extra Markdown lines."""
    return html.escape(safe_label_text(str(value)), quote=False).replace("\\n", " ")


def _render_held_out_note(
    held_out_ids: list[str], scenario_packs: list[dict[str, Any]] | None
) -> list[str]:
    """Explain the redaction and attest to which pack produced the numbers.

    The content hash lets a reader confirm two reports were scored against the
    same held-out set — the check they would otherwise perform by reading the
    scenarios, which is exactly what must not be published.
    """
    md = [
        "",
        f"> **{len(held_out_ids)} held-out scenario(s)** — titles, summaries, and traces are "
        "withheld so publishing this report does not publish the scenarios. Statuses and "
        "points are scored identically to public scenarios.",
    ]
    for pack in scenario_packs or []:
        name = pack.get("name", "?")
        count = pack.get("scenario_count", "?")
        digest = pack.get("content_hash", "?")
        md.append(f"> - pack `{name}`: {count} scenario(s), content hash `{digest}`")
    return md


def _render_run_context(ctx: RunContext) -> list[str]:
    """Render RunContext as Markdown tables for embedding in reports."""
    md: list[str] = []

    # -- Run Context table (Tier 2: CLI parameters) --
    md.extend(
        [
            "## Run Context",
            "",
            "| Parameter | Value |",
            "|---|---|",
        ]
    )
    if ctx.label:
        md.append(f"| **Label** | {markdown_label(ctx.label, table_cell=True)} |")
    md.extend(
        [
            f"| Backend | {ctx.backend} |",
            f"| Server | `{ctx.base_url}` |",
            f"| Model (API) | `{ctx.model}` |",
        ]
    )
    if ctx.server_model_root and ctx.server_model_root != ctx.model:
        md.append(f"| Model (Root) | `{ctx.server_model_root}` |")
    md.extend(
        [
            f"| Temperature | {ctx.temperature} |",
            f"| Seed | {ctx.seed if ctx.seed is not None else '—'} |",
            f"| Max Turns | {ctx.max_turns} |",
            f"| Timeout | {ctx.timeout_seconds}s |",
            f"| Scenarios | {ctx.scenario_selector} |",
            f"| Parallel | {ctx.parallel} {'(sequential)' if ctx.parallel <= 1 else ''} |",
            f"| Error Rate | {ctx.error_rate} |",
            f"| Thinking | {'enabled' if ctx.thinking_enabled else 'disabled'} |",
        ]
    )
    if ctx.context_pressure is not None:
        md.append(f"| Context Pressure | {ctx.context_pressure:.0%} |")
    if ctx.extra_params:
        import json as _json

        md.append(f"| Extra Params | `{_json.dumps(ctx.extra_params)}` |")
    md.append("")

    # -- Inference Engine table (Tier 3: best-effort) --
    has_engine_info = any(
        [
            ctx.engine_name,
            ctx.engine_version,
            ctx.max_model_len,
            ctx.quantization,
            ctx.gpu_count,
            ctx.spec_decoding,
        ]
    )
    if has_engine_info:
        md.extend(
            [
                "## Inference Engine",
                "",
                "| Property | Value |",
                "|---|---|",
            ]
        )
        if ctx.engine_name:
            version_str = f" {ctx.engine_version}" if ctx.engine_version else ""
            md.append(f"| Engine | {ctx.engine_name}{version_str} |")
        if ctx.max_model_len:
            md.append(f"| Max Model Length | {ctx.max_model_len:,} |")
        if ctx.quantization:
            md.append(f"| Quantization | {ctx.quantization} |")
        if ctx.gpu_count:
            md.append(f"| GPU Count | {ctx.gpu_count} |")
        if ctx.spec_decoding:
            md.append(f"| Spec Decoding | {ctx.spec_decoding} |")
        md.extend(
            [
                f"| Host | `{ctx.hostname}` |",
                f"| Platform | `{ctx.platform_info}` |",
                f"| Python | {ctx.python_version} |",
                "",
            ]
        )
    else:
        # Minimal environment info even without engine probes
        md.extend(
            [
                "## Environment",
                "",
                "| Property | Value |",
                "|---|---|",
                f"| Host | `{ctx.hostname}` |",
                f"| Platform | `{ctx.platform_info}` |",
                f"| Python | {ctx.python_version} |",
                "",
            ]
        )

    return md
