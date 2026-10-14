"""Tests for the shareable card renderer in ``storage.reports.card``."""

from __future__ import annotations

import inspect
import math
import re
import struct
import sys
import types
import xml.etree.ElementTree as ET
from dataclasses import replace
from importlib import resources

import pytest

from tests.card_fixtures import (
    ALL,
    accuracy_only,
    all_errored_needle,
    compact_smarts,
    excluded_scenarios,
    extreme,
    full_long,
    hardmode_only,
    interrupted_long,
    needle_eleven_depths,
    needle_misses,
    needle_single_cell,
    no_single_stream,
    perf_short,
    real_sparse,
    scenario_pack,
    sparse_perf_accuracy,
    subset_3_of_69,
    tool_eval_only,
    window_capped_deep,
    window_capped_long,
)
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
    INSTALL_HINT,
    PngExportUnavailable,
    render_card_svg,
    svg_to_png,
)
from tool_eval_bench.storage.reports.card import _format as fmt
from tool_eval_bench.storage.reports.card._data import (
    accuracy_note,
    cached_ttft_label,
    concurrency_series,
    engine_label,
    is_subset,
    needs_caution,
    plottable,
    public_version,
    scope_label,
    score_details,
    ttft_note,
    window_capped,
)
from tool_eval_bench.storage.reports.card._fonts import (
    FAMILIES,
    font_file_names,
    text_width,
    unsupported_characters,
)
from tool_eval_bench.storage.reports.card._layout import (
    CHIP_GAP,
    CHIP_PAD,
    NEEDLE_HINT,
    REPO,
    SPEED_HINT,
    TOOL_HINT,
    Slot,
    _model_lines,
    allocate,
    chip_labels,
    footer_receipt,
    hint_rows,
    scope_ids,
    stamp_runs,
)
from tool_eval_bench.storage.reports.card._panels import (
    LABEL,
    _decode_callout,
    _needle_heatmap,
    _span_extent,
    bar_color,
    chip_runs_for,
    log_decades,
    log_step,
    nice_ticks,
    outcome_bar,
    spaced_labels,
)
from tool_eval_bench.storage.reports.card._svg import ELLIPSIS, Run, first_fit, runs_text, text
from tool_eval_bench.storage.reports.card._theme import FAIL, GOOD, PARTIAL, PASS, Box

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


_FONT_OF = {family: key for key, family in FAMILIES.items()}
_DESCENDERS = set("gjpqy(),;_|/[]{}Q")


def _style(el: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
    keys = ("font-size", "font-family", "font-weight", "letter-spacing")
    return {**inherited, **{k: el.get(k, "") for k in keys if el.get(k) is not None}}


def _text_box(el: ET.Element) -> tuple[str, tuple[float, float, float, float]]:
    """Approximate ink box of a <text>, summing each tspan with its own font and dx."""
    base = _style(el, {"font-size": "16", "font-family": "", "font-weight": "400"})
    runs = [(el.text or "", base, 0.0)]
    runs += [(c.text or "", _style(c, base), float(c.get("dx") or 0)) for c in el]
    width, size, descent = 0.0, 0.0, 0.0
    for chunk, style, dx in runs:
        px = float(style["font-size"])
        width += dx + text_width(
            chunk,
            px,
            font=_FONT_OF.get(style["font-family"], "sans"),  # type: ignore[arg-type]
            weight=int(style["font-weight"]),
            tracking=float(style.get("letter-spacing") or 0),
        )
        if chunk:
            size = max(size, px)
            descent = max(descent, (0.21 if set(chunk) & _DESCENDERS else 0.02) * px)
    x, y = float(el.get("x", 0)), float(el.get("y", 0))
    anchor = el.get("text-anchor", "start")
    x0 = x - width if anchor == "end" else x - width / 2 if anchor == "middle" else x
    content = "".join(r[0] for r in runs)
    return content, (x0, y - 0.73 * size, x0 + width, y + descent)


def _overlaps(svg: str) -> list[tuple[str, str]]:
    boxes = [_text_box(el) for el in _parse(svg).iter(f"{SVG_NS}text")]
    boxes = [(c, b) for c, b in boxes if c.strip()]
    found = []
    for i, (ca, a) in enumerate(boxes):
        for cb, b in boxes[i + 1 :]:
            if a[0] < b[2] - 1 and b[0] < a[2] - 1 and a[1] < b[3] - 1 and b[1] < a[3] - 1:
                found.append((ca, cb))
    return found


def _panels(svg: str, *, dashed: bool) -> list[tuple[float, float, float, float]]:
    """Panel frames as (x, y, w, h): solid ones use the panel gradient fully opaque."""
    out = []
    for el in _parse(svg).iter(f"{SVG_NS}rect"):
        if not re.fullmatch(r"url\(#c[0-9a-f]{8}-panel\)", el.get("fill", "")):
            continue
        if (el.get("opacity") is not None) is dashed:
            out.append(tuple(float(el.get(k, 0)) for k in ("x", "y", "width", "height")))
    return out  # type: ignore[return-value]


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
        "cached: +2K tokens on cached context",
        "cold, full context prefill",
        "402 ms",
        "97 scenarios",
        "tok/s max aggregate",
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
        "The long preset charts decode and TTFT to 128K.",
        "--needle",
        "drop --perf-only",
    ),
    "d_extreme": (
        "Per-depth grid not stored for this run \u00b7 1 errored",
        "estimated",
        f"{MINUS}99.9%",
        "tool-eval-bench v2.0.0.dev41",
    ),
    "e_needle_misses": (
        "35/43 found \u00b7 2 errored",
        "--perf --concurrency 1,2,4,8",
        "--perf-only",
    ),
    "f_real_sparse": (
        "subset: 3 of 69 scenarios",
        "vLLM 0.21.1rc1.dev292",
        "--perf --perf-preset long",
        "--needle",
        "24 of 28 errored",
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", sorted(EXPECTED_TEXT))
def test_fixture_shows_expected_labels_and_hints(name: str, layout: CardLayout) -> None:
    text = _all_text(render_card_svg(ALL[name](), layout=layout)).casefold()
    for expected in EXPECTED_TEXT[name]:
        assert expected.casefold() in text, expected


@pytest.mark.parametrize("layout", LAYOUTS)
def test_perf_card_says_tool_calling_is_missing_once(layout: CardLayout) -> None:
    for data in (perf_short(), replace(perf_short(), accuracy=accuracy_only().accuracy)):
        text = _all_text(render_card_svg(data, layout=layout))
        assert "drop --perf-only" in text
        assert "Tool calling not scored" not in text
        assert text.count("TOOL CALLING") == 1


@pytest.mark.parametrize("layout", LAYOUTS)
def test_plugin_only_card_names_the_only_flag(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(accuracy_only(), layout=layout))
    assert "drop --*-only" in text
    assert "--perf-only" not in text


def test_narrow_hint_strip_shortens_its_message_instead_of_truncating() -> None:
    texts = _texts(render_card_svg(accuracy_only()))
    assert any(t == "Not scored" for t in texts)
    assert not any(t.endswith("\u2026") and "not" in t.casefold() for t in texts)


def test_roomy_accuracy_panel_draws_one_ring_per_benchmark() -> None:
    svg = render_card_svg(accuracy_only())
    texts = _texts(svg)
    for result in accuracy_only().accuracy:
        assert result.name in texts
        assert fmt.percent(result.accuracy) in texts
    # Each ring is a track plus an arc; the rows layout draws no circles.
    rows = render_card_svg(accuracy_only(), layout="landscape")
    assert svg.count("<circle") >= rows.count("<circle") + 2 * len(accuracy_only().accuracy)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_concurrency_legend_names_the_per_request_rate_at_peak(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(full_long(), layout=layout))
    peak = max(concurrency_series(plottable(full_long())), key=lambda p: p.tg_tps_total)
    assert f"per request {fmt.rate(peak.tg_tps)}" in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_short_preset_suggests_long_preset_instead_of_claiming_it_stopped(
    layout: CardLayout,
) -> None:
    text = _all_text(render_card_svg(perf_short(), layout=layout))
    assert "Short preset: depths 0 to 8K" in text
    assert "--perf --perf-preset long" in text
    assert "stopped" not in text.lower()
    assert "Decode tok/s" not in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_errored_needle_cells_render_hatched(layout: CardLayout) -> None:
    hatch = re.compile(r'fill="url\(#c[0-9a-f]{8}-hatch\)"')
    assert hatch.search(render_card_svg(needle_misses(), layout=layout))
    assert not hatch.search(render_card_svg(full_long(), layout=layout))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_safety_cap_is_explained_once(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(extreme(), layout=layout))
    assert text.count("Safety cap") == 1
    assert "capped" in text


def test_outcome_bar_has_no_safety_pill() -> None:
    te = extreme().tool_eval
    assert te is not None and te.safety_capped
    bar = outcome_bar(Box(0, 0, 400, 36), te)
    assert "Safety" not in bar and "capped" not in bar


def test_engine_version_keeps_dev_and_drops_local_segment() -> None:
    text = _all_text(render_card_svg(real_sparse()))
    assert "vLLM 0.21.1rc1.dev292" in text
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
        (999_999, "1M"),
        (999_499, "999.5K"),
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
        (0, ("0", "ms")),
        (0.01, ("<0.1", "ms")),
        (412_000, ("6.9", "min")),
        (7_200_000, ("2.0", "h")),
        (1_234_567, ("20.6", "min")),
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
    [
        (0, "0"),
        (2.5, "2.5"),
        (50, "50"),
        (1200, "1.2K"),
        (9000, "9K"),
        (12_000, "12K"),
        (999_600, "1M"),
        (2_500_000, "2.5M"),
    ],
)
def test_axis_rate(value: float, expected: str) -> None:
    assert fmt.axis_rate(value) == expected


# --------------------------------------------------------------------------- data helpers


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.21.1rc1.dev292+g97e4022c6.d20260526", "0.21.1rc1.dev292"),
        ("0.1.dev19023+g1234abc", "0.1.dev19023"),
        ("0.25.2.dev0", "0.25.2.dev0"),
        ("2.0.0.dev41", "2.0.0.dev41"),
        ("0.11.0", "0.11.0"),
        ("b6750", "b6750"),
    ],
)
def test_public_version_drops_only_the_local_segment(version: str, expected: str) -> None:
    assert public_version(version) == expected


def test_engine_label() -> None:
    assert engine_label("vLLM", "0.12.0rc1.dev999+gdeadbeef") == "vLLM 0.12.0rc1.dev999"
    assert engine_label("vLLM", None) == "vLLM"
    assert engine_label(None, None) is None


@pytest.mark.parametrize(
    ("total", "excluded", "suite_size", "subset"),
    [
        (3, 0, 69, True),
        (69, 0, 69, False),
        (66, 3, 69, False),
        (69, 0, 97, True),
        (28, 0, 28, False),
        (3, 0, None, False),
    ],
)
def test_is_subset(total: int, excluded: int, suite_size: int | None, subset: bool) -> None:
    te = ToolEvalSummary(80.0, "", total, 0, 0, total, suite_size=suite_size, excluded=excluded)
    assert is_subset(te) is subset


def test_score_details_keeps_count_and_mentions_trials() -> None:
    te = tool_eval_only().tool_eval
    assert te is not None
    details = score_details(te, passed=True)
    assert details[0] == "69 scenarios \u00b7 56 of 69 passed \u00b7 mean of 3 trials"
    assert details[-1] == "69 scenarios"
    assert all(d.startswith("69 scenarios") for d in details)


def test_scope_marks_subset_exclusions_and_suite() -> None:
    sparse = real_sparse().tool_eval
    assert sparse is not None
    assert scope_label(sparse) == "subset: 3 of 69 scenarios"
    excluded = excluded_scenarios().tool_eval
    assert excluded is not None
    assert scope_label(excluded) == "66 scenarios \u00b7 3 excluded"
    assert needs_caution(excluded) and not is_subset(excluded)
    hard = hardmode_only().tool_eval
    assert hard is not None
    assert scope_label(hard) == "28 scenarios \u00b7 Hard Mode only"
    assert "28 \u00b7 Hard Mode only" in score_details(hard)
    pack = scenario_pack().tool_eval
    assert pack is not None
    assert scope_label(pack) == "40 scenarios \u00b7 scenario pack"
    assert not needs_caution(pack)


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


def test_footer_receipt_never_shows_the_local_version_segment() -> None:
    data = extreme()
    public = "tool-eval-bench v2.0.0.dev41 \u00b7 seed 2147483647 \u00b7 cfg ffffffffffff"
    assert footer_receipt(data, 2000) == public
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


def _slot(
    min_h: float,
    max_h: float,
    sink: bool = False,
    *,
    hint: bool = False,
    rigid: bool = False,
    floor: float | None = None,
) -> Slot:
    return Slot(lambda box: "", min_h, max_h, sink, hint=hint, rigid=rigid, floor=floor)


def test_allocate_grows_evenly_then_feeds_sinks() -> None:
    assert allocate(300, [_slot(50, 100), _slot(50, 100, sink=True)]) == [100, 200]
    assert allocate(150, [_slot(50, 200), _slot(50, 60)]) == [90, 60]
    assert allocate(300, [_slot(50, 100)]) == [300]


def test_allocate_shrinks_proportionally_when_short() -> None:
    assert allocate(50, [_slot(50, 60), _slot(50, 60)]) == [25, 25]


def test_allocate_squeezes_slack_above_floors_before_scaling() -> None:
    assert allocate(180, [_slot(100, 200, floor=60), _slot(100, 200)]) == [80, 100]


def test_allocate_feeds_data_panels_before_hint_strips() -> None:
    hint = _slot(88, 88, hint=True, rigid=True)
    assert allocate(400, [_slot(100, 150), hint]) == [312, 88]
    assert allocate(400, [_slot(100, 150, rigid=True), hint]) == [150, 250]
    assert allocate(400, [_slot(100, 150), _slot(100, 150, sink=True), hint]) == [150, 162, 88]


def test_first_fit_prefers_longest_wording_that_fits() -> None:
    room = text_width("short", 14) + 1
    assert first_fit(["much longer wording", "short"], room, 14) == "short"
    assert first_fit(["fits easily"], 1000, 14) == "fits easily"
    assert first_fit(["nothing here fits"], 40, 14).endswith(ELLIPSIS)


def test_span_extent() -> None:
    line = [(0.0, 0.0), (100.0, 100.0)]
    assert _span_extent([line], 20, 30) == (20, 30)
    assert _span_extent([line], 200, 300) is None


# --------------------------------------------------------------------------- fix-round regressions


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", sorted(ALL))
def test_no_text_overlaps(name: str, layout: CardLayout) -> None:
    # Covers the compact accuracy rows (name over detail), the safety pill over
    # the outcome counts, and narrow empty states drawing flags over titles.
    assert _overlaps(render_card_svg(ALL[name](), layout=layout)) == []


def test_accuracy_row_keeps_error_count_when_detail_does_not_fit() -> None:
    mmlu = real_sparse().accuracy[1]
    assert accuracy_note(mmlu) == "28 of 14,042 q \u00b7 24 of 28 errored"
    assert accuracy_note(mmlu, detail=False) == "24 of 28 errored"
    assert "24 of 28 errored" in _all_text(render_card_svg(compact_smarts()))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_needle_legend_percent_counts_answered_cells_only(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(needle_misses(), layout=layout))
    assert "35/43 found" in text
    match = re.search(r"35/43 found \u00b7 (\d+)%", text)
    assert match is None or match.group(1) == "81"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_errored_needle_is_neutral(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(all_errored_needle(), layout=layout))
    assert "all 15 cells errored" in text
    assert "effective context" not in text
    assert "0% found" not in text
    for context in ("8K", "16K", "32K"):
        assert context in text


def _needle_cells(svg: str) -> list[ET.Element]:
    root = _parse(f'<svg xmlns="http://www.w3.org/2000/svg">{svg}</svg>')
    # Grid cells carry these opacities; legend swatches and the effective-context bar do not.
    return [
        el
        for el in root.iter(f"{SVG_NS}rect")
        if el.get("fill") in (PASS, FAIL) and el.get("opacity") in ("0.72", "0.95")
    ]


def test_needle_single_cell_is_capped_not_a_slab() -> None:
    needle = needle_single_cell().needle
    assert needle is not None
    cells = _needle_cells(_needle_heatmap(Box(0, 0, 900, 300), needle))
    assert len(cells) == 1
    assert float(cells[0].get("width", 0)) <= 160


def test_needle_eleven_depths_draws_every_row() -> None:
    needle = needle_eleven_depths().needle
    assert needle is not None
    cells = _needle_cells(_needle_heatmap(Box(0, 0, 420, 160), needle))
    assert len(cells) == 22
    assert all(float(c.get("height", 0)) > 0 for c in cells)


def test_unsupported_characters_ignores_bundled_glyphs() -> None:
    assert unsupported_characters("Qwen3 \u901a\u4e49 \u2605 \u00b7 \u00e9") == "\u901a\u4e49"
    assert unsupported_characters("plain ascii") == ""


@pytest.mark.parametrize(
    ("model", "fallback"), [("gemma-3", False), ("\u901a\u4e49\u5343\u95ee", True)]
)
def test_png_loads_system_fonts_only_for_missing_glyphs(
    model: str, fallback: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def svg_to_bytes(**kwargs: object) -> bytes:
        calls.append(kwargs)
        return b"png"

    monkeypatch.setitem(sys.modules, "resvg_py", types.SimpleNamespace(svg_to_bytes=svg_to_bytes))
    assert svg_to_png(render_card_svg(replace(full_long(), model=model))) == b"png"
    assert calls[0]["skip_system_fonts"] is not fallback


@pytest.mark.parametrize(
    ("value", "expected"),
    [(99.6, "99.6"), (99.96, "99.9"), (100.0, "100"), (0.4, "0.4"), (0.0, "0"), (86.4, "86")],
)
def test_score_never_rounds_to_a_fake_extreme(value: float, expected: str) -> None:
    assert fmt.score(value) == expected


def test_hero_and_categories_show_near_perfect_scores_honestly() -> None:
    te = replace(full_long().tool_eval, score=99.6)  # type: ignore[type-var]
    te = replace(te, categories=(CategoryScore("A", "Tool Selection", 99.7),))
    text = _all_text(render_card_svg(replace(full_long(), tool_eval=te)))
    assert "99.6/100" in text
    assert "100/100" not in text
    assert "99.7" in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_heroes_use_lowest_concurrency_without_c1(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(no_single_stream(), layout=layout))
    assert "Not measured" not in text.split("TOOL CALLING")[0]
    assert "per request at c=2, estimated" in text
    assert "per request at c=2 by context depth" in text
    assert "per request (est.)" in text
    assert "c=2 to 8" in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_window_capped_long_preset_says_so_without_suggesting_long(layout: CardLayout) -> None:
    assert window_capped(plottable(window_capped_long()))
    text = _all_text(render_card_svg(window_capped_long(), layout=layout))
    assert "capped at 16K by the 32K window" in text
    assert "--perf-preset long" not in text
    deep = _all_text(render_card_svg(window_capped_deep(), layout=layout))
    assert "capped at 64K by the 96K window" in deep


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"perf_failed_cells": 1}, id="a-cell-failed"),
        pytest.param({"max_context": 256 * 1024}, id="128k-fits-the-window"),
    ],
)
def test_short_long_sweep_is_not_blamed_on_the_window(changes: dict[str, int]) -> None:
    data = replace(window_capped_long(), **changes)
    assert not window_capped(plottable(data))
    assert "capped at" not in _all_text(render_card_svg(data))


def test_window_cap_headroom_matches_the_planner() -> None:
    from tool_eval_bench.runner import perf_presets
    from tool_eval_bench.storage.reports.card import _data

    assert _data.CHAT_TEMPLATE_HEADROOM == perf_presets.CHAT_TEMPLATE_HEADROOM
    assert _data.LONG_PRESET_DEPTH == max(perf_presets.PRESET_DEFAULTS["long"].depths)


def test_window_that_barely_misses_128k_is_still_a_cap() -> None:
    data = replace(window_capped_long(), max_context=128 * 1024)
    assert window_capped(plottable(data))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_errored_accuracy_is_not_a_zero_score(layout: CardLayout) -> None:
    def zeros(text: str) -> int:
        return len(re.findall(r"(?<![\d.])0%", text))

    for base in (full_long(), accuracy_only()):
        unmeasured, zero = (
            _all_text(
                render_card_svg(
                    replace(base, accuracy=(AccuracyResult("GSM8K", value, errors=5, items=5),)),
                    layout=layout,
                )
            )
            for value in (math.nan, 0.0)
        )
        assert "n/a" in unmeasured and "errored" in unmeasured
        # Axis ticks elsewhere can read 0%; the accuracy value must not add one.
        assert zeros(unmeasured) == zeros(zero) - 1


def test_short_window_capped_long_preset_uses_depth_strip_without_long_flag() -> None:
    data = replace(window_capped_long(), max_context=8192, perf=window_capped_long().perf[:1])
    text = _all_text(render_card_svg(data))
    assert "Long preset capped at 0 by the 8K window" in text
    assert "--perf-preset long" not in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_interrupted_sweep_is_marked_and_never_suggests_long(layout: CardLayout) -> None:
    text = _all_text(render_card_svg(interrupted_long(), layout=layout))
    assert "incomplete" in text
    assert "3 cells failed" in text
    assert "depth pass incomplete" in text
    assert "--perf-preset long" not in text


def test_stamp_keeps_warnings_when_detail_does_not_fit() -> None:
    data = plottable(interrupted_long())
    roomy = "".join(r.content for r in stamp_runs(data, 2000))
    assert roomy.startswith("pp 2,048") and "incomplete \u00b7 3 cells failed" in roomy
    tight = stamp_runs(data, 300)
    assert [r.content for r in tight] == ["incomplete \u00b7 3 cells failed"]


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        (100.0, (f"{MINUS}70%", "at 64K")),
        (0.001, ("30.0", "tok/s at 64K")),
        (0.5, ("30.0", "tok/s at 64K")),
    ],
)
def test_decode_callout_guards_tiny_bases(base: float, expected: tuple[str, str]) -> None:
    series = [PerfPoint(0, 1, 1.0, base, base, 1.0), PerfPoint(65536, 1, 1.0, 30.0, 30.0, 1.0)]
    assert _decode_callout(series) == expected


@pytest.mark.parametrize("layout", LAYOUTS)
def test_subset_withholds_stars_and_marks_scope(layout: CardLayout) -> None:
    for data in (subset_3_of_69(), real_sparse()):
        text = _all_text(render_card_svg(data, layout=layout))
        assert "subset: 3 of 69 scenarios" in text
        assert "Excellent" not in text
        assert "\u2605" not in text


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    ("build", "marker"),
    [
        (hardmode_only, "Hard Mode only"),
        (scenario_pack, "scenario pack"),
        (excluded_scenarios, "3 excluded"),
    ],
)
def test_suite_markers_are_rendered(build: object, marker: str, layout: CardLayout) -> None:
    text = _all_text(render_card_svg(build(), layout=layout))  # type: ignore[operator]
    assert marker in text
    assert "subset" not in text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sparse_card_gives_space_to_data_not_placeholders(layout: CardLayout) -> None:
    svg = render_card_svg(sparse_perf_accuracy(), layout=layout)
    solid = sum(w * h for _, _, w, h in _panels(svg, dashed=False))
    dashed = sum(w * h for _, _, w, h in _panels(svg, dashed=True))
    assert dashed < 0.25 * solid
    assert "ACCURACY" in _all_text(svg)


def test_hint_rows_group_three_to_a_row_and_one_when_alone() -> None:
    hints = [SPEED_HINT, NEEDLE_HINT, TOOL_HINT]
    assert len(hint_rows(hints, 968)) == 1
    assert len(hint_rows(hints, 468)) == 3
    assert len(hint_rows(hints, 968, alone=True)) == 3
    assert hint_rows([], 968) == []


@pytest.mark.parametrize(
    ("ms", "expected"),
    [(10, "10 ms"), (1000, "1 s"), (60_000, "1 min"), (600_000, "10 min"), (3.6e6, "1 h")],
)
def test_axis_duration_switches_units(ms: float, expected: str) -> None:
    assert fmt.axis_duration(ms) == expected


def _chip_text(chips: list[list[Run]]) -> list[str]:
    return ["".join(r.content for r in runs) for runs in chips]


def test_accuracy_chips_never_drop_results_silently() -> None:
    results = compact_smarts().accuracy
    roomy = _chip_text(chip_runs_for(results, 2000, 13))
    assert roomy[0].startswith("GSM8K75%") and "24 of 28 errored" in roomy[0]
    tight = _chip_text(chip_runs_for(results, 260, 13))
    shown = sum(1 for c in tight if not c.startswith("+"))
    more = [c for c in tight if c.startswith("+")]
    assert more == [f"+{len(results) - shown} more"] or shown == len(results)


def test_accuracy_values_are_coloured_by_band() -> None:
    assert bar_color(92) == PASS
    assert bar_color(70) == GOOD
    assert bar_color(45) == PARTIAL
    assert bar_color(10) == FAIL
    chip = chip_runs_for((AccuracyResult("MMLU", 0.25),), 2000, 13)[0]
    assert chip[1].fill == FAIL


@pytest.mark.parametrize("layout", LAYOUTS)
def test_value_and_unit_share_one_text_element(layout: CardLayout) -> None:
    root = _parse(render_card_svg(full_long(), layout=layout))
    heroes = [
        el
        for el in root.iter(f"{SVG_NS}text")
        if (el.text or "") == "151" and [c.text for c in el] == ["tok/s"]
    ]
    assert heroes, "decode value and unit should be one <text> with a <tspan>"


def test_runs_text_resets_inherited_tracking() -> None:
    svg = runs_text(0, 0, [Run("A", 13, "#fff", tracking=1.6), Run("b", 13, "#fff", gap=4)])
    assert 'letter-spacing="1.6"' in svg and 'letter-spacing="0"' in svg and 'dx="4"' in svg


def test_ids_are_prefixed_per_card_and_every_reference_resolves() -> None:
    first = render_card_svg(full_long())
    second = render_card_svg(tool_eval_only())
    prefixes = {re.findall(r'id="(c[0-9a-f]{8})-', svg)[0] for svg in (first, second)}
    assert len(prefixes) == 2
    for svg in (first, second):
        ids = set(re.findall(r'id="([\w-]+)"', svg))
        assert len(set(re.findall(r'id="(c[0-9a-f]{8})-', svg))) == 1
        refs = set(re.findall(r"url\(#([\w-]+)\)", svg))
        assert refs and refs <= ids
    assert render_card_svg(full_long()) == first


def test_scope_ids_leaves_text_alone() -> None:
    svg = (
        '<svg><defs><g id="panel"/></defs><rect fill="url(#panel)"/><text>url(#panel)</text></svg>'
    )
    scoped = scope_ids(svg)
    assert "<text>url(#panel)</text>" in scoped
    assert re.search(r'fill="url\(#c[0-9a-f]{8}-panel\)"', scoped)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_long_context_panel_gets_a_usable_height(layout: CardLayout) -> None:
    svg = render_card_svg(full_long(), layout=layout)
    root = _parse(svg)
    title = next(el for el in root.iter(f"{SVG_NS}text") if (el.text or "") == "LONG CONTEXT")
    y = float(title.get("y", 0))
    panel = next(p for p in _panels(svg, dashed=False) if p[1] < y < p[1] + p[3])
    assert panel[3] >= 280


@pytest.mark.parametrize("layout", LAYOUTS)
def test_labels_stay_legible_at_phone_width(layout: CardLayout) -> None:
    assert LABEL >= 13
    sizes = [
        float(el.get("font-size", 16))
        for el in _parse(render_card_svg(full_long(), layout=layout)).iter()
        if el.get("font-size")
    ]
    assert min(sizes) >= 12


def test_ring_shows_pass_count_not_a_percentage() -> None:
    text = _texts(render_card_svg(tool_eval_only()))
    assert "56/69" in text
    assert not any(re.fullmatch(r"\d+(\.\d)?%", t) for t in text)


def test_text_helper_has_no_raw_escape_hatch() -> None:
    assert "raw" not in inspect.signature(text).parameters


def test_install_hint_is_the_one_shared_message(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "git+https://github.com/SeraphimSerapis/tool-eval-bench.git" in INSTALL_HINT
    assert "tool-eval-bench[card]" in INSTALL_HINT
    monkeypatch.setitem(sys.modules, "resvg_py", None)
    with pytest.raises(PngExportUnavailable) as caught:
        svg_to_png(render_card_svg(full_long()))
    assert str(caught.value) == INSTALL_HINT
