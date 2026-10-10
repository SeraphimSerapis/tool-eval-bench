"""Tests for the shareable card renderer in ``storage.reports.card``."""

from __future__ import annotations

import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from importlib import resources

import pytest

from tests.card_fixtures import ALL, extreme, full_long, real_sparse, tool_eval_only
from tool_eval_bench.domain.card import (
    AccuracyResult,
    CardData,
    CardLayout,
    CategoryScore,
    NeedleCell,
    NeedleSummary,
    PerfPoint,
    ToolEvalSummary,
)
from tool_eval_bench.storage.reports.card import (
    PngExportUnavailable,
    render_card_svg,
    svg_to_png,
)
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    cached_ttft_label,
    engine_label,
    is_subset,
    plottable,
    score_details,
    short_version,
    ttft_note,
)
from tool_eval_bench.storage.reports.card._fonts import font_file_names, text_width
from tool_eval_bench.storage.reports.card._layout import (
    CHIP_GAP,
    CHIP_PAD,
    REPO,
    Slot,
    _model_lines,
    allocate,
    chip_labels,
    footer_receipt,
)
from tool_eval_bench.storage.reports.card._panels import (
    _span_extent,
    log_decades,
    log_step,
    nice_ticks,
    spaced_labels,
)
from tool_eval_bench.storage.reports.card._svg import ELLIPSIS, first_fit

SVG_NS = "{http://www.w3.org/2000/svg}"
LAYOUTS: tuple[CardLayout, ...] = ("portrait", "landscape")
SIZES = {"portrait": (1080, 1350), "landscape": (1600, 900)}
MINUS = "\u2212"


def _parse(svg: str) -> ET.Element:
    # The input is markup the renderer under test just produced, not external data.
    return ET.fromstring(svg)  # noqa: S314


def _texts(svg: str) -> list[str]:
    return ["".join(el.itertext()) for el in _parse(svg).iter(f"{SVG_NS}text")]


def _all_text(svg: str) -> str:
    return " | ".join(_texts(svg))


# --------------------------------------------------------------------------- whole card


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", sorted(ALL))
def test_every_fixture_renders_valid_svg_at_layout_size(name: str, layout: CardLayout) -> None:
    svg = render_card_svg(ALL[name](), layout=layout)
    root = _parse(svg)
    width, height = SIZES[layout]
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("width") == str(width)
    assert root.get("height") == str(height)
    assert root.get("viewBox") == f"0 0 {width} {height}"
    assert not re.search(r'="[^"]*\b(nan|inf)\b', svg, re.IGNORECASE)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_render_is_deterministic(layout: CardLayout) -> None:
    first = render_card_svg(full_long(), layout=layout)
    assert render_card_svg(full_long(), layout=layout) == first


def test_unknown_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown card layout"):
        render_card_svg(full_long(), layout="square")  # type: ignore[arg-type]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_untrusted_text_is_escaped_and_never_becomes_markup(layout: CardLayout) -> None:
    hostile = "<script>alert('x')</script>&\"' \x00\x1b\x7f<foreignObject>"
    data = replace(
        full_long(),
        model=hostile,
        engine=hostile,
        quantization=hostile,
        hardware='<a href="https://evil.example">x</a>',
        fingerprint=hostile,
        tool_version=hostile,
        accuracy=(AccuracyResult(hostile, 0.5),),
    )
    svg = render_card_svg(data, layout=layout)
    root = _parse(svg)
    assert "<script" not in svg
    assert "evil.example" not in svg or "&lt;a href" in svg
    for el in root.iter():
        assert el.tag.removeprefix(SVG_NS) not in {"script", "foreignObject", "image", "a"}
        assert not any("href" in key for key in el.attrib)
    assert "<script>" in root.findtext(f"{SVG_NS}title", "")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_only_link_is_the_repository(layout: CardLayout) -> None:
    for build in ALL.values():
        svg = render_card_svg(build(), layout=layout)
        assert svg.count("://") == 1  # the SVG namespace
        assert re.findall(r"github\.com/[\w./-]+", svg) == [REPO]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_footer_is_receipt_only(layout: CardLayout) -> None:
    svg = render_card_svg(full_long(), layout=layout)
    footer = next(t for t in _texts(svg) if t.startswith("tool-eval-bench v"))
    assert footer == "tool-eval-bench v1.14.0 \u00b7 seed 42 \u00b7 cfg a1b2c3d4e5f6"
    assert "preset" not in footer.lower()


# Hints and labels each fixture must show in both layouts. Compared case-insensitively
# because some panels start a sentence with the same phrase.
EXPECTED_TEXT: dict[str, tuple[str, ...]] = {
    "a_full_long": (
        "LONG PRESET",
        "TTFT cold: full context prefill \u00b7 cached: +2K tokens on cached context",
        "cold, full context prefill",
        "402 ms",
        "97 scenarios",
        "tok/s peak",
    ),
    "b_tool_eval_only": (
        "69 scenarios \u00b7 mean of 3 trials",
        "Speed not measured",
        "--perf --perf-preset long",
        "--needle",
        "--gsm8k",
    ),
    "c_perf_short": (
        "SHORT PRESET",
        "--perf --perf-preset long",
        "The long preset sweeps to 128K.",
        "--needle",
        "run without --perf-only",
    ),
    "d_extreme": (
        "Per-depth grid not stored for this run \u00b7 1 errored",
        "estimated",
        f"{MINUS}99.9%",
        "tool-eval-bench v2.0.0.dev41+g1234abcd",
    ),
    "e_needle_misses": (
        "35/43 found \u00b7 2 errored",
        "not run",
        "--perf --concurrency 1,2,4,8",
        "run without --perf-only",
    ),
    "f_real_sparse": (
        "3 scenarios \u00b7 subset",
        "vLLM 0.21.1rc1",
        "--perf --perf-preset long",
        "--needle",
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", sorted(EXPECTED_TEXT))
def test_fixture_shows_expected_labels_and_hints(name: str, layout: CardLayout) -> None:
    text = _all_text(render_card_svg(ALL[name](), layout=layout)).casefold()
    for expected in EXPECTED_TEXT[name]:
        assert expected.casefold() in text, expected


@pytest.mark.parametrize("layout", LAYOUTS)
def test_short_preset_shows_depth_strip_instead_of_chart(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(ALL["c_perf_short"](), layout=layout))
    assert "Sweep stopped at 8K" in text
    assert "Decode tok/s" not in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_errored_needle_cells_render_hatched(layout: CardLayout) -> None:
    svg = render_card_svg(ALL["e_needle_misses"](), layout=layout)
    assert 'fill="url(#hatch)"' in svg
    assert 'fill="url(#hatch)"' not in render_card_svg(full_long(), layout=layout)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_safety_cap_is_called_out(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(extreme(), layout=layout))
    assert "\u26a0 Safety" in text


def test_engine_version_is_shortened_on_the_card() -> None:
    text = _all_text(render_card_svg(real_sparse()))
    assert "vLLM 0.21.1rc1" in text
    assert "g97e4022c6" not in text


def test_long_model_name_is_shortened_not_overflowing() -> None:
    data = extreme()
    for layout in LAYOUTS:
        text = _all_text(render_card_svg(data, layout=layout))
        assert data.model not in text
        assert ELLIPSIS in text


def _degenerate() -> CardData:
    nan, inf = math.nan, math.inf
    return CardData(
        model="",
        tool_version="",
        created_at="",
        max_context=0,
        perf_pp=0,
        perf=(
            PerfPoint(0, 1, nan, 0.0, 0.0, inf, nan),
            PerfPoint(16384, 1, 0.0, 1.0, 0.0, 0.0, -5.0),
            PerfPoint(0, 4, 1.0, nan, inf, 1.0),
        ),
        tool_eval=ToolEvalSummary(
            score=nan,
            rating="",
            passed=0,
            partial=0,
            failed=0,
            total=0,
            categories=(CategoryScore("A", "x", nan), CategoryScore("B", "y", -40)),
            trials=0,
        ),
        needle=NeedleSummary(
            cells=(NeedleCell(0, nan, True), NeedleCell(-5, 0.5, False)),
            accuracy=nan,
            effective_context=None,
            errors=5,
        ),
        accuracy=(AccuracyResult("", nan), AccuracyResult("X", -0.5)),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_degenerate_values_render_without_invalid_coordinates(layout: CardLayout) -> None:
    svg = render_card_svg(_degenerate(), layout=layout)
    _parse(svg)
    assert not re.search(r"\b(nan|inf)\b", svg, re.IGNORECASE)
    assert "unknown model" in _all_text(svg)


def test_plottable_drops_unplottable_points_and_cells() -> None:
    cleaned = plottable(_degenerate())
    assert [(p.depth, p.concurrency) for p in cleaned.perf] == [(16384, 1)]
    assert cleaned.perf[0].cached_ttft_ms is None
    assert cleaned.needle is not None and cleaned.needle.cells == ()
    assert plottable(full_long()) == full_long()


# --------------------------------------------------------------------------- PNG export


def _png_size(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    return width, height


@pytest.mark.parametrize("layout", LAYOUTS)
def test_png_export_uses_scale(layout: CardLayout) -> None:
    pytest.importorskip("resvg_py")
    png = svg_to_png(render_card_svg(full_long(), layout=layout), scale=2.0)
    width, height = SIZES[layout]
    assert _png_size(png) == (width * 2, height * 2)


def test_png_export_without_renderer_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "resvg_py", None)
    with pytest.raises(PngExportUnavailable, match=re.escape("tool-eval-bench[card]")):
        svg_to_png(render_card_svg(full_long()))


@pytest.mark.parametrize("scale", [0, -1, math.nan])
def test_png_export_rejects_bad_scale_before_importing(
    scale: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "resvg_py", None)
    with pytest.raises(ValueError, match="scale"):
        svg_to_png("<svg/>", scale=scale)


def test_bundled_fonts_and_licenses_are_package_data() -> None:
    fonts = resources.files("tool_eval_bench.storage.reports.card").joinpath("fonts")
    for name in font_file_names():
        assert fonts.joinpath(name).read_bytes()[:4] in (b"\x00\x01\x00\x00", b"OTTO")
    licenses = [p.name for p in fonts.iterdir() if p.name.endswith(".txt")]
    assert sum("OFL" in name for name in licenses) == 2
    assert "SOURCES.txt" in licenses


# --------------------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0"),
        (512, "512"),
        (4096, "4K"),
        (131072, "128K"),
        (100_000, "100K"),
        (123_456, "123.5K"),
        (1024 * 1024, "1M"),
        (1536 * 1024, "1.5M"),
        (2_000_000, "2M"),
    ],
)
def test_tokens(n: int, expected: str) -> None:
    assert fmt.tokens(n) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1234.4, "1,234"), (142.3, "142"), (12.34, "12.3"), (0.854, "0.85"), (-1, "n/a")],
)
def test_rate(value: float, expected: str) -> None:
    assert fmt.rate(value) == expected


def test_rate_non_finite() -> None:
    assert fmt.rate(math.nan) == fmt.rate(math.inf) == "n/a"


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (5, ("5.0", "ms")),
        (850, ("850", "ms")),
        (999.6, ("1.0", "s")),
        (2400, ("2.4", "s")),
        (412_000, ("412", "s")),
        (1_234_567, ("1,235", "s")),
        (math.inf, ("n/a", "")),
        (-3, ("n/a", "")),
    ],
)
def test_duration(ms: float, expected: tuple[str, str]) -> None:
    assert fmt.duration(ms) == expected


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.712, "71%"),
        (1.0, "100%"),
        (0.0, "0%"),
        (0.9998, "99.9%"),
        (0.003, "0.3%"),
        (0.000001, "0.1%"),
        (7.0, "700%"),
        (-0.5, f"{MINUS}50%"),
        (math.nan, "n/a"),
    ],
)
def test_percent_never_rounds_to_a_fake_extreme(fraction: float, expected: str) -> None:
    assert fmt.percent(fraction) == expected


@pytest.mark.parametrize(
    ("change", "expected"),
    [(0.12, "+12%"), (-0.32, f"{MINUS}32%"), (-0.9998, f"{MINUS}99.9%"), (math.inf, "n/a")],
)
def test_signed_percent(change: float, expected: str) -> None:
    assert fmt.signed_percent(change) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0"), (2.5, "2.5"), (50, "50"), (1200, "1.2K"), (9000, "9K"), (12_000, "12K")],
)
def test_axis_rate(value: float, expected: str) -> None:
    assert fmt.axis_rate(value) == expected


# --------------------------------------------------------------------------- data helpers


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.21.1rc1.dev292+g97e4022c6.d20260526", "0.21.1rc1"),
        ("2.0.0.dev41", "2.0.0"),
        ("0.11.0", "0.11.0"),
        ("b6750", "b6750"),
    ],
)
def test_short_version(version: str, expected: str) -> None:
    assert short_version(version) == expected


def test_engine_label() -> None:
    assert engine_label("vLLM", "0.12.0rc1.dev999+gdeadbeef") == "vLLM 0.12.0rc1"
    assert engine_label("vLLM", None) == "vLLM"
    assert engine_label(None, None) is None


@pytest.mark.parametrize(
    ("total", "suite_size", "subset"),
    [(3, 69, True), (69, 69, False), (69, 97, True), (97, 97, False), (3, None, False)],
)
def test_is_subset(total: int, suite_size: int | None, subset: bool) -> None:
    assert is_subset(total, suite_size) is subset


def test_score_details_keeps_count_and_mentions_trials() -> None:
    te = tool_eval_only().tool_eval
    assert te is not None
    details = score_details(te, passed=True)
    assert details[0] == "69 scenarios \u00b7 56 of 69 passed \u00b7 mean of 3 trials"
    assert details[-1] == "69 scenarios \u00b7 3 trials"
    assert all(d.startswith("69 scenarios") for d in details)


def test_score_details_marks_subset() -> None:
    te = real_sparse().tool_eval
    assert te is not None
    assert score_details(te) == ["3 scenarios \u00b7 subset"]
    assert score_details(replace(te, total=1, passed=1)) == ["1 scenario \u00b7 subset"]


def test_ttft_note_describes_cached_only_when_measured() -> None:
    assert ttft_note(full_long())[0] == (
        "TTFT cold: full context prefill \u00b7 cached: +2K tokens on cached context"
    )
    no_cache = replace(
        full_long(), perf=tuple(replace(p, cached_ttft_ms=None) for p in full_long().perf)
    )
    assert all("cached" not in note for note in ttft_note(no_cache))


def test_cached_ttft_label() -> None:
    assert cached_ttft_label(full_long(), "402 ms")[0] == "cached, +2K new tokens: 402 ms"
    no_pp = replace(full_long(), perf_pp=None)
    assert cached_ttft_label(no_pp, "402 ms") == ("cached context: 402 ms", "cached: 402 ms")


def _chips_width(labels: list[str]) -> float:
    widths = [text_width(lbl, 14, weight=500) + 2 * CHIP_PAD for lbl in labels]
    return sum(widths) + CHIP_GAP * max(len(labels) - 1, 0)


def test_chip_labels_fit_and_shorten_context_first() -> None:
    data = full_long()
    roomy = chip_labels(data, 2000)
    assert roomy == ["vLLM 0.11.0", "FP8", "RTX PRO 6000 Blackwell 96GB", "256K context"]
    tight_width = _chips_width(roomy) - 20
    tight = chip_labels(data, tight_width)
    assert tight[-1] == "256K ctx"
    assert _chips_width(tight) <= tight_width


def test_chip_labels_share_the_cut_between_long_labels() -> None:
    data = extreme()
    labels = chip_labels(data, 760)
    assert len(labels) == 4
    assert _chips_width(labels) <= 760
    assert labels[1].endswith(ELLIPSIS) and labels[2].endswith(ELLIPSIS)
    assert chip_labels(data, 50) == []


def test_footer_receipt_drops_local_version_segment_before_truncating() -> None:
    data = extreme()
    full = "tool-eval-bench v2.0.0.dev41+g1234abcd \u00b7 seed 2147483647 \u00b7 cfg ffffffffffff"
    public = "tool-eval-bench v2.0.0.dev41 \u00b7 seed 2147483647 \u00b7 cfg ffffffffffff"
    assert footer_receipt(data, 2000) == full
    room = text_width(public, 12, font="mono")
    assert footer_receipt(data, room) == public
    squeezed = footer_receipt(data, room - 60)
    assert squeezed.endswith(ELLIPSIS) or "cfg" in squeezed
    assert text_width(squeezed, 12, font="mono") <= room - 60


def test_footer_receipt_omits_missing_seed_and_fingerprint() -> None:
    data = replace(full_long(), seed=None, fingerprint=None)
    assert footer_receipt(data, 2000) == "tool-eval-bench v1.14.0"


# --------------------------------------------------------------------------- layout and scales


def _model_width(line: str, size: int) -> float:
    return text_width(line, size, font="display", weight=700, tracking=-1.0)


def test_model_lines_keep_short_names_whole() -> None:
    assert _model_lines("gemma-3-27b-it", 900, 64, 30, 2) == (64, ["gemma-3-27b-it"])


def test_model_lines_break_long_names_within_width() -> None:
    name = extreme().model
    size, lines = _model_lines(name, 900, 64, 30, 2)
    assert len(lines) == 2
    assert all(_model_width(line, size) <= 900 for line in lines)
    assert lines[0].startswith("org-with-a-long-name/")


def test_model_lines_single_line_uses_middle_ellipsis() -> None:
    name = extreme().model
    size, lines = _model_lines(name, 600, 48, 30, 1)
    assert size == 30 and len(lines) == 1
    assert ELLIPSIS in lines[0] and lines[0].startswith("org")
    assert _model_width(lines[0], size) <= 600


def test_text_width_scales_with_size_and_glyphs() -> None:
    assert text_width("", 14) == 0
    assert text_width("WWW", 14) > text_width("iii", 14)
    assert text_width("abc", 28) == pytest.approx(2 * text_width("abc", 14))
    assert text_width("abc", 14, tracking=2) == pytest.approx(text_width("abc", 14) + 6)


def test_nice_ticks() -> None:
    assert nice_ticks(1501) == [0, 500, 1000, 1500, 2000]
    assert nice_ticks(0) == nice_ticks(math.nan) == [0.0, 1.0]
    ticks = nice_ticks(9000)
    assert ticks[0] == 0 and ticks[-1] >= 9000
    assert len({round(b - a, 6) for a, b in zip(ticks, ticks[1:], strict=False)}) == 1


def test_log_decades() -> None:
    assert log_decades([40, 400_000]) == (1, 6)
    assert log_decades([100]) == (2, 3)
    assert log_decades([0, -1, math.nan]) == (0, 1)


def test_log_step_spaces_labels() -> None:
    assert log_step(5, 200) == 1
    assert log_step(5, 40) == 3
    assert log_step(1, 5) == 1


def test_spaced_labels_keep_ends_and_drop_collisions() -> None:
    assert spaced_labels([], []) == []
    assert spaced_labels([5], [10]) == [True]
    assert spaced_labels([0, 10, 20, 100], [30] * 4) == [True, False, False, True]
    assert spaced_labels([0, 50, 100], [10] * 3) == [True, True, True]


def _slot(min_h: float, max_h: float, sink: bool = False) -> Slot:
    return Slot(lambda box: "", min_h, max_h, sink)


def test_allocate_grows_evenly_then_feeds_sinks() -> None:
    assert allocate(300, [_slot(50, 100), _slot(50, 100, sink=True)]) == [100, 200]
    assert allocate(150, [_slot(50, 200), _slot(50, 60)]) == [90, 60]
    assert allocate(300, [_slot(50, 100)]) == [300]


def test_allocate_shrinks_proportionally_when_short() -> None:
    assert allocate(50, [_slot(50, 60), _slot(50, 60)]) == [25, 25]


def test_first_fit_prefers_longest_wording_that_fits() -> None:
    room = text_width("short", 14) + 1
    assert first_fit(["much longer wording", "short"], room, 14) == "short"
    assert first_fit(["fits easily"], 1000, 14) == "fits easily"
    assert first_fit(["nothing here fits"], 40, 14).endswith(ELLIPSIS)


def test_span_extent() -> None:
    line = [(0.0, 0.0), (100.0, 100.0)]
    assert _span_extent([line], 20, 30) == (20, 30)
    assert _span_extent([line], 200, 300) is None
