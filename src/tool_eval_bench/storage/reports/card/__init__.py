"""Shareable benchmark card: one session summarised as an SVG, optionally a PNG.

The SVG is plain, self-contained markup generated with the standard library.
PNG export needs the optional ``resvg-py`` package (the ``card`` extra) and
renders with the bundled fonts only, so a PNG looks the same on every machine.
"""

from __future__ import annotations

import importlib
from contextlib import ExitStack
from importlib import resources

from tool_eval_bench.domain.card import CardData, CardLayout
from tool_eval_bench.storage.reports.card._fonts import font_file_names
from tool_eval_bench.storage.reports.card._layout import render

__all__ = ["PngExportUnavailable", "render_card_svg", "svg_to_png"]

INSTALL_HINT = "pip install 'tool-eval-bench[card]'"


class PngExportUnavailable(RuntimeError):
    """Raised when PNG export is requested without the optional renderer installed."""


def render_card_svg(data: CardData, *, layout: CardLayout = "portrait") -> str:
    """Render ``data`` as an SVG document. The same input always gives the same bytes.

    ``portrait`` is 1080x1350 (4:5) and ``landscape`` is 1600x900 (16:9).
    """
    return render(data, layout)


def svg_to_png(svg: str, *, scale: float = 2.0) -> bytes:
    """Rasterise a card SVG at ``scale`` times its size using the bundled fonts."""
    if not scale > 0:
        raise ValueError("scale must be positive")
    try:
        resvg_py = importlib.import_module("resvg_py")
    except ImportError as exc:
        raise PngExportUnavailable(
            f"PNG export needs the optional resvg-py renderer. Install it with: {INSTALL_HINT}"
        ) from exc
    fonts_dir = resources.files(__name__).joinpath("fonts")
    with ExitStack() as stack:
        paths = [
            str(stack.enter_context(resources.as_file(fonts_dir.joinpath(name))))
            for name in font_file_names()
        ]
        png = resvg_py.svg_to_bytes(
            svg_string=svg,
            zoom=scale,
            font_files=paths,
            skip_system_fonts=True,
            font_family="Inter",
            sans_serif_family="Inter",
            monospace_family="JetBrains Mono NL",
        )
    return bytes(png)
