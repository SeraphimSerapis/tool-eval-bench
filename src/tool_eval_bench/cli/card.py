"""The ``card`` command: render stored runs as a shareable results card."""

from __future__ import annotations

import argparse
import importlib
from datetime import date, datetime
from pathlib import Path
from types import ModuleType

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.application.card_data import CardDataUnavailable, load_card_data
from tool_eval_bench.domain.card import CardData
from tool_eval_bench.storage.reports import _default_reports_root, slugify_label

PNG_INSTALL_HINT = "pip install 'tool-eval-bench[card]'"
_RENDERER_MODULE = "tool_eval_bench.storage.reports.card"


def _renderer() -> ModuleType:
    """Import the renderer on use, so the command module loads without it."""
    return importlib.import_module(_RENDERER_MODULE)


def card_date(card: CardData) -> str:
    """``YYYYMMDD`` of the newest run on the card, or today for a row without a timestamp.

    The run date rather than today's keeps a regenerated card on the same file.
    """
    try:
        return datetime.fromisoformat(card.created_at).strftime("%Y%m%d")
    except ValueError:
        return date.today().strftime("%Y%m%d")


def output_paths(
    card: CardData, *, output: str | None, output_dir: str | None, png: bool
) -> tuple[Path, Path | None]:
    """Return the SVG path and, when PNG is wanted, the PNG path.

    An ``-o`` path ending in ``.png`` asks for PNG by itself; the SVG goes
    beside it. Without ``-o`` the card lands in ``<output-dir>/cards/``.
    """
    if output:
        target = Path(output)
        if target.suffix.lower() == ".png":
            return target.with_suffix(".svg"), target
        return target, target.with_suffix(".png") if png else None
    slug = slugify_label(card.model) or "model"
    svg = Path(output_dir or _default_reports_root()) / "cards" / f"{slug}-{card_date(card)}.svg"
    return svg, svg.with_suffix(".png") if png else None


def fill_tip(card: CardData) -> str | None:
    """Name the flags that would fill the sections this card leaves empty."""
    flags: list[str] = []
    if not card.perf:
        flags.append("--perf --perf-preset long")
    if card.needle is None:
        flags.append("--needle")
    if not card.accuracy:
        flags.append("--gsm8k --mmlu --ifeval")
    tip = f"add {' '.join(flags)} to the next run" if flags else ""
    if card.tool_eval is None:
        scenarios = "run the tool-call scenarios (without --skip-tool-eval or a --*-only flag)"
        tip = f"{tip} and {scenarios}" if tip else scenarios
    return f"To fill the empty sections, {tip}." if tip else None


def run_card_command(args: argparse.Namespace, console: Console) -> None:
    """Render a card from stored runs and write it to disk."""
    scale = args.card_scale
    if not scale > 0:
        console.print(f"[bold red]Error:[/] --scale must be greater than 0, got {scale}")
        raise SystemExit(2)
    try:
        card = load_card_data(
            session=args.card or "latest",
            run_ids=tuple(args.card_run or ()),
            model=args.model,
            hardware=args.card_hardware,
        )
    except CardDataUnavailable as exc:
        console.print(f"[bold red]Error:[/] {escape(str(exc))}")
        raise SystemExit(1) from None

    svg_path, png_path = output_paths(
        card, output=args.card_output, output_dir=args.output_dir, png=args.card_png
    )
    renderer = _renderer()
    svg = renderer.render_card_svg(card, layout=args.card_layout)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    console.print(f"[bold green]✓[/] Card saved to {escape(str(svg_path))}")

    png_missing = False
    if png_path is not None:
        try:
            png = renderer.svg_to_png(svg, scale=scale)
        except renderer.PngExportUnavailable:
            png_missing = True
            console.print(
                "[bold red]PNG not written: export needs the optional renderer.[/] "
                + escape(PNG_INSTALL_HINT)
            )
        else:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.write_bytes(png)
            console.print(f"[bold green]✓[/] PNG saved to {escape(str(png_path))}")

    tip = fill_tip(card)
    if tip:
        console.print(f"[dim]Tip: {escape(tip)}[/]")
    if png_missing:
        # The SVG is written, but an explicit PNG request failed; scripts must see it.
        raise SystemExit(1)
