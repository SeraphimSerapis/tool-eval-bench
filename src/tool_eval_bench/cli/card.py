"""The ``card`` command: render stored runs as a shareable results card."""

from __future__ import annotations

import argparse
import importlib
import re
from datetime import datetime
from pathlib import Path
from types import ModuleType

from rich.console import Console
from rich.markup import escape

from tool_eval_bench.application.card_data import CardDataUnavailable, load_card_data
from tool_eval_bench.domain.card import CardData
from tool_eval_bench.storage.reports import _default_reports_root, slugify_label

_RENDERER_MODULE = "tool_eval_bench.storage.reports.card"
MIN_SCALE = 0.25
MAX_SCALE = 8.0
# File types a user might plausibly ask for that a card cannot be written as.
# Any other "suffix" is part of the name, as in ``Qwen3.5-27B``.
_OTHER_FILE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".tif", ".tiff", ".heic", ".ico"}
    | {".pdf", ".eps", ".ps", ".html", ".htm", ".md", ".txt", ".json"}
)
# The fingerprint prefix in the default filename: enough to keep two
# configurations of one model on one day apart. It names a configuration, not
# a session, so the filename is as safe to share as the card.
_FILENAME_FINGERPRINT_CHARS = 8
_ACCURACY_FLAGS = (("GSM8K", "--gsm8k"), ("MMLU", "--mmlu"), ("IFEval", "--ifeval"))


def _renderer() -> ModuleType:
    """Import the renderer on use, so the command module loads without it."""
    return importlib.import_module(_RENDERER_MODULE)


def card_date(card: CardData) -> str:
    """``YYYYMMDD`` of the newest run on the card, or ``undated`` without a readable timestamp.

    The run date rather than today's keeps a regenerated card on the same file.
    """
    try:
        return datetime.fromisoformat(card.created_at.replace("Z", "+00:00")).strftime("%Y%m%d")
    except ValueError:
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", card.created_at)
        return "".join(match.groups()) if match else "undated"


def default_filename(card: CardData) -> str:
    """``<model>-<YYYYMMDD>[-<fingerprint>].svg``, with the model name slugified."""
    parts = [slugify_label(card.model) or "model", card_date(card)]
    if card.fingerprint:
        parts.append(card.fingerprint[:_FILENAME_FINGERPRINT_CHARS])
    return "-".join(parts) + ".svg"


class CardOutputError(ValueError):
    """``-o`` names a file type the card cannot be written as."""


def output_paths(
    card: CardData, *, output: str | None, output_dir: str | None, png: bool
) -> tuple[Path, Path | None]:
    """Return the SVG path and, when PNG is wanted, the PNG path.

    An ``-o`` path ending in ``.png`` asks for PNG by itself; the SVG goes
    beside it. An ``-o`` directory, or a path ending in a separator, gets the
    default filename inside it. A suffix naming another image or document type
    raises :class:`CardOutputError`. Any other name gets ``.svg`` appended, so a
    dotted model name such as ``Qwen3.5-27B`` keeps every character. Without
    ``-o`` the card lands in ``<output-dir>/cards/``.
    """
    if output:
        target = Path(output)
        if target.is_dir() or output.endswith(("/", "\\")):
            target = target / default_filename(card)
        suffix = target.suffix.lower()
        if suffix in _OTHER_FILE_SUFFIXES:
            raise CardOutputError(
                f"-o {output}: a card is written as .svg or .png, not {target.suffix}"
            )
        if suffix == ".png":
            return target.with_suffix(".svg"), target
        if suffix != ".svg":
            target = target.with_name(target.name + ".svg")
        return target, target.with_suffix(".png") if png else None
    svg = Path(output_dir or _default_reports_root()) / "cards" / default_filename(card)
    return svg, svg.with_suffix(".png") if png else None


def fill_tip(card: CardData) -> str | None:
    """Name the flags that would fill the sections this card leaves empty."""
    flags: list[str] = []
    if not card.perf:
        flags.append("--perf --perf-preset long")
    if card.needle is None:
        flags.append("--needle")
    measured = {result.name for result in card.accuracy}
    flags.extend(flag for name, flag in _ACCURACY_FLAGS if name not in measured)
    tip = f"add {' '.join(flags)} to the next run" if flags else ""
    if card.tool_eval is None:
        scenarios = "include the tool-call scenarios"
        tip = f"{tip} and {scenarios}" if tip else scenarios
    return f"To fill the empty sections, {tip}." if tip else None


def _error(console: Console, message: str, code: int) -> SystemExit:
    console.print(f"[bold red]Error:[/] {escape(message)}")
    return SystemExit(code)


def _warn(console: Console, message: str) -> None:
    console.print(f"[yellow]Warning:[/] {escape(message)}")


def _session_argument(args: argparse.Namespace) -> str | None:
    """The SESSION the user named, or ``None`` for the default latest session."""
    session = args.card
    if session is None or session.strip().lower() == "latest":
        return None
    return str(session)


def run_card_command(args: argparse.Namespace, console: Console) -> None:
    """Render a card from stored runs and write it to disk."""
    scale = args.card_scale
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise _error(
            console, f"--scale must be between {MIN_SCALE:g} and {MAX_SCALE:g}, got {scale:g}", 2
        )
    session = _session_argument(args)
    run_ids = tuple(args.card_run or ())
    if session is not None and run_ids:
        raise _error(
            console,
            f"SESSION {session} and --run both pick the runs for a card; pass one of them",
            2,
        )
    if args.model and (session is not None or run_ids):
        chosen = "--run" if run_ids else "SESSION"
        _warn(console, f"--model only narrows the latest session; it is ignored with {chosen}")

    try:
        card = load_card_data(
            session=session or "latest",
            run_ids=run_ids,
            model=args.model,
            hardware=args.card_hardware,
            warn=lambda message: _warn(console, message),
        )
    except CardDataUnavailable as exc:
        raise _error(console, str(exc), 1) from None

    try:
        svg_path, png_path = output_paths(
            card, output=args.card_output, output_dir=args.output_dir, png=args.card_png
        )
    except CardOutputError as exc:
        raise _error(console, str(exc), 2) from None
    renderer = _renderer()
    svg = renderer.render_card_svg(card, layout=args.card_layout)
    try:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(svg, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise _error(console, f"Could not write {svg_path}: {exc.strerror or exc}", 1) from None
    console.print(f"[bold green]✓[/] Card saved to {escape(str(svg_path))}")

    png_failed = False
    if png_path is not None:
        try:
            png = renderer.svg_to_png(svg, scale=scale)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.write_bytes(png)
        except renderer.PngExportUnavailable:
            png_failed = True
            console.print("[bold red]PNG not written.[/] " + escape(renderer.INSTALL_HINT))
        except OSError as exc:
            png_failed = True
            _warn(console, f"PNG not written to {png_path}: {exc.strerror or exc}")
        except (ValueError, RuntimeError) as exc:
            png_failed = True
            _warn(console, f"PNG not written: the renderer failed: {exc}")
        else:
            console.print(f"[bold green]✓[/] PNG saved to {escape(str(png_path))}")
    elif (stale := svg_path.with_suffix(".png")).exists():
        _warn(
            console,
            f"{stale} is left from an earlier render and may not match this SVG; "
            "pass --png to refresh it",
        )

    tip = fill_tip(card)
    if tip:
        console.print(f"[dim]Tip: {escape(tip)}[/]")
    if png_failed:
        # The SVG is written, but an explicit PNG request failed; scripts must see it.
        raise SystemExit(1)
