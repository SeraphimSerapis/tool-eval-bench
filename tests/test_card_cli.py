"""The ``card`` command: argument translation, output paths, and the PNG fallback."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tool_eval_bench.application.card_data import CardDataUnavailable
from tool_eval_bench.cli import card as card_cli
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.local_commands import handle_local_command
from tool_eval_bench.cli.parser import parse_cli_args, translate_argv
from tool_eval_bench.domain.card import (
    AccuracyResult,
    CardData,
    NeedleSummary,
    PerfPoint,
    ToolEvalSummary,
)

FULL_CARD = CardData(
    model="Qwen3-32B-AWQ",
    tool_version="3.1.0",
    created_at="2026-10-10T12:13:00+00:00",
    perf=(PerfPoint(0, 1, 1000.0, 50.0, 50.0, 200.0),),
    tool_eval=ToolEvalSummary(82.0, "★★★★ Good", 4, 1, 1, 6),
    needle=NeedleSummary(cells=(), accuracy=0.75, effective_context=1024),
    accuracy=(
        AccuracyResult("GSM8K", 0.9),
        AccuracyResult("MMLU", 0.7),
        AccuracyResult("IFEval", 0.8),
    ),
)
EMPTY_CARD = CardData(model="M", tool_version="3.1.0", created_at="not a date")


class _PngExportUnavailable(RuntimeError):
    pass


class _Renderer:
    """Stands in for ``storage.reports.card`` so these tests pin the CLI, not the drawing."""

    PngExportUnavailable = _PngExportUnavailable
    INSTALL_HINT = "pip install 'tool-eval-bench[card]'"

    def __init__(self, *, png: bool = True, png_error: Exception | None = None) -> None:
        self.png = png
        self.png_error = png_error
        self.layouts: list[str] = []
        self.scales: list[float] = []

    def render_card_svg(self, data: CardData, *, layout: str = "portrait") -> str:
        self.layouts.append(layout)
        return f"<svg>{data.model}</svg>"

    def svg_to_png(self, svg: str, *, scale: float = 2.0) -> bytes:
        self.scales.append(scale)
        if self.png_error is not None:
            raise self.png_error
        if not self.png:
            raise _PngExportUnavailable("resvg-py is not installed")
        return b"\x89PNG" + svg.encode()


def _args(argv: list[str]) -> argparse.Namespace:
    return parse_cli_args(make_parser, argv)[1]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    card: CardData = FULL_CARD,
    renderer: _Renderer | None = None,
) -> tuple[str, _Renderer, dict[str, Any]]:
    renderer = renderer or _Renderer()
    loaded: dict[str, Any] = {}

    def load(**kwargs: Any) -> CardData:
        loaded.update(kwargs)
        return card

    monkeypatch.setattr(card_cli, "load_card_data", load)
    monkeypatch.setattr(card_cli, "_renderer", lambda: renderer)
    console = Console(record=True, width=300)
    card_cli.run_card_command(_args(argv), console)
    return console.export_text(), renderer, loaded


class TestTranslation:
    def test_defaults(self) -> None:
        assert translate_argv(["card"]) == [
            "--card=latest",
            "--card-layout",
            "portrait",
            "--card-scale",
            "2.0",
        ]

    def test_every_option_reaches_the_flat_namespace(self) -> None:
        args = _args(
            [
                "card",
                "sess-1",
                "--run",
                "a",
                "--run",
                "b",
                "--hardware",
                "RTX 5090 32GB",
                "--layout",
                "landscape",
                "-o",
                "out.svg",
                "--png",
                "--scale",
                "3",
                "--model",
                "m",
                "--output-dir",
                "d",
            ]
        )
        assert args.card == "sess-1"
        assert args.card_run == ["a", "b"]
        assert args.card_hardware == "RTX 5090 32GB"
        assert args.card_layout == "landscape"
        assert args.card_output == "out.svg"
        assert args.card_png is True
        assert args.card_scale == 3.0
        assert args.model == "m"
        assert args.output_dir == "d"

    def test_flat_flag_defaults_to_latest(self) -> None:
        assert _args(["--card"]).card == "latest"
        assert _args([]).card is None

    def test_unknown_option_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            translate_argv(["card", "--base-url", "http://x"])
        assert exc_info.value.code == 2

    def test_focused_help_is_card_specific(self) -> None:
        from tool_eval_bench.cli.parser import _command_help

        text = _command_help("card").format_help()
        assert "--hardware" in text
        assert "--png" in text
        assert "auto-detected" not in text
        assert "--base-url" not in text


class TestPaths:
    def test_default_path_uses_model_slug_and_run_date(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        text, renderer, loaded = _run(
            monkeypatch, ["card", "--output-dir", str(tmp_path), "--hardware", "H100"]
        )
        svg = tmp_path / "cards" / "qwen3-32b-awq-20261010.svg"
        assert svg.read_text(encoding="utf-8") == "<svg>Qwen3-32B-AWQ</svg>"
        assert not svg.with_suffix(".png").exists()
        assert str(svg) in text
        assert renderer.layouts == ["portrait"]
        assert {key: value for key, value in loaded.items() if key != "warn"} == {
            "session": "latest",
            "run_ids": (),
            "model": None,
            "hardware": "H100",
        }

    def test_default_name_carries_the_fingerprint_prefix_not_the_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        card = dataclasses.replace(FULL_CARD, fingerprint="0123abcd99")
        _run(monkeypatch, ["card", "--output-dir", str(tmp_path)], card=card)
        assert (tmp_path / "cards" / "qwen3-32b-awq-20261010-0123abcd.svg").exists()

    def test_default_root_is_runs_under_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _run(monkeypatch, ["card"], card=EMPTY_CARD)
        # An unparseable timestamp is not replaced by today's date.
        assert (tmp_path / "runs" / "cards" / "m-undated.svg").exists()

    @pytest.mark.parametrize(
        ("created_at", "expected"),
        [
            ("2026-10-10T12:13:00Z", "20261010"),
            ("2026-06-02 08:25:16 garbage", "20260602"),
        ],
    )
    def test_card_date_reads_the_run_timestamp(self, created_at: str, expected: str) -> None:
        card = dataclasses.replace(EMPTY_CARD, created_at=created_at)
        assert card_cli.card_date(card) == expected

    def test_output_directory_gets_the_default_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _run(monkeypatch, ["card", "-o", str(tmp_path)])
        assert (tmp_path / "qwen3-32b-awq-20261010.svg").exists()

    def test_output_without_a_suffix_becomes_svg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _run(monkeypatch, ["card", "-o", str(tmp_path / "mycard")])
        assert (tmp_path / "mycard.svg").exists()

    @pytest.mark.parametrize("name", ["Qwen3.5-27B", "qwen3.5", "card.v2"])
    def test_dotted_name_keeps_every_character(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
    ) -> None:
        _run(monkeypatch, ["card", "-o", str(tmp_path / name)])
        assert [p.name for p in tmp_path.iterdir()] == [f"{name}.svg"]

    def test_dotted_png_name_puts_the_svg_beside_it(self, tmp_path: Path) -> None:
        svg, png = card_cli.output_paths(
            EMPTY_CARD, output=str(tmp_path / "Qwen3.5-27B.png"), output_dir=None, png=False
        )
        assert (svg.name, png) == ("Qwen3.5-27B.svg", tmp_path / "Qwen3.5-27B.png")

    @pytest.mark.parametrize("name", ["card.PDF", "card.jpeg", "card.html"])
    def test_other_file_types_are_rejected(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(card_cli.CardOutputError):
            card_cli.output_paths(
                EMPTY_CARD, output=str(tmp_path / name), output_dir=None, png=False
            )

    def test_unknown_suffix_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(monkeypatch, ["card", "-o", str(tmp_path / "card.jpg")])
        assert exc_info.value.code == 2
        assert list(tmp_path.iterdir()) == []

    def test_stale_png_is_flagged(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "card.png").write_bytes(b"old")
        text, _, _ = _run(monkeypatch, ["card", "-o", str(tmp_path / "card.svg")])
        assert "card.png is left from an earlier render" in text
        assert (tmp_path / "card.png").read_bytes() == b"old"

    def test_png_beside_the_svg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        out = tmp_path / "share" / "card.svg"
        text, renderer, _ = _run(
            monkeypatch, ["card", "-o", str(out), "--png", "--scale", "3", "--layout", "landscape"]
        )
        assert out.exists()
        assert out.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
        assert renderer.scales == [3.0]
        assert renderer.layouts == ["landscape"]
        assert "PNG saved to" in text

    def test_png_output_path_implies_png(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "card.PNG"
        _run(monkeypatch, ["card", "-o", str(out)])
        assert out.read_bytes().startswith(b"\x89PNG")
        assert out.with_suffix(".svg").exists()

    def test_run_ids_and_model_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, _, loaded = _run(
            monkeypatch,
            ["card", "--run", "r1", "--run", "r2", "--model", "qwen", "-o", str(tmp_path / "c")],
        )
        assert loaded["run_ids"] == ("r1", "r2")
        assert loaded["model"] == "qwen"

    def test_model_is_flagged_when_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        text, _, _ = _run(
            monkeypatch, ["card", "--run", "r1", "--model", "qwen", "-o", str(tmp_path / "c")]
        )
        assert "--model only narrows the latest session; it is ignored with --run" in text

    def test_loader_warnings_are_printed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def load(*, warn: Any, **_: Any) -> CardData:
            warn("The runs come from different engine versions [a, b]")
            return FULL_CARD

        monkeypatch.setattr(card_cli, "load_card_data", load)
        monkeypatch.setattr(card_cli, "_renderer", _Renderer)
        console = Console(record=True, width=300)
        card_cli.run_card_command(_args(["card", "-o", str(tmp_path / "c")]), console)
        assert "Warning: The runs come from different engine versions [a, b]" in (
            console.export_text()
        )

    def test_session_and_run_together_are_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def load(**_: Any) -> CardData:
            raise AssertionError("a conflicting selection must fail before any read")

        monkeypatch.setattr(card_cli, "load_card_data", load)
        console = Console(record=True, width=300)
        with pytest.raises(SystemExit) as exc_info:
            card_cli.run_card_command(_args(["card", "abcdef", "--run", "r1"]), console)
        assert exc_info.value.code == 2
        assert "pass one of them" in console.export_text()


class TestFallbacksAndErrors:
    @pytest.mark.parametrize("png_flag", [["--png"], []])
    def test_missing_resvg_writes_the_svg_then_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, png_flag: list[str]
    ) -> None:
        out = tmp_path / ("card.svg" if png_flag else "card.png")
        monkeypatch.setattr(card_cli, "load_card_data", lambda **_: EMPTY_CARD)
        monkeypatch.setattr(card_cli, "_renderer", lambda: _Renderer(png=False))
        console = Console(record=True, width=300)
        with pytest.raises(SystemExit) as exc_info:
            card_cli.run_card_command(_args(["card", "-o", str(out), *png_flag]), console)
        assert exc_info.value.code == 1
        assert out.with_suffix(".svg").exists()
        assert not out.with_suffix(".png").exists()
        text = console.export_text()
        # The brackets survive Rich markup, so the command is copy-pasteable.
        assert "pip install 'tool-eval-bench[card]'" in text
        # The tip still prints before the exit.
        assert "Tip:" in text

    def test_no_runs_exits_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def load(**_: Any) -> CardData:
            raise CardDataUnavailable("No benchmark session found [here]")

        monkeypatch.setattr(card_cli, "load_card_data", load)
        console = Console(record=True, width=300)
        with pytest.raises(SystemExit) as exc_info:
            card_cli.run_card_command(_args(["card", "--output-dir", str(tmp_path)]), console)
        assert exc_info.value.code == 1
        assert "No benchmark session found [here]" in console.export_text()
        assert not (tmp_path / "cards").exists()

    @pytest.mark.parametrize(
        ("png_error", "message"),
        [
            (ValueError("scale too large"), "the renderer failed: scale too large"),
            (RuntimeError("resvg crashed"), "the renderer failed: resvg crashed"),
        ],
    )
    def test_renderer_failure_keeps_the_svg_and_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, png_error: Exception, message: str
    ) -> None:
        out = tmp_path / "card.svg"
        with pytest.raises(SystemExit) as exc_info:
            _run(
                monkeypatch,
                ["card", "-o", str(out), "--png"],
                renderer=_Renderer(png_error=png_error),
            )
        assert exc_info.value.code == 1
        assert out.exists()
        assert not out.with_suffix(".png").exists()

    def test_unwritable_output_is_a_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(card_cli, "load_card_data", lambda **_: FULL_CARD)
        monkeypatch.setattr(card_cli, "_renderer", _Renderer)
        console = Console(record=True, width=300)
        with pytest.raises(SystemExit) as exc_info:
            card_cli.run_card_command(_args(["card", "-o", str(blocker / "card.svg")]), console)
        assert exc_info.value.code == 1
        assert "Could not write" in console.export_text()

    def test_unwritable_png_keeps_the_svg_and_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "card.png").mkdir()
        with pytest.raises(SystemExit) as exc_info:
            _run(monkeypatch, ["card", "-o", str(tmp_path / "card.svg"), "--png"])
        assert exc_info.value.code == 1
        assert (tmp_path / "card.svg").exists()

    @pytest.mark.parametrize("scale", ["0.25", "8"])
    def test_scale_bounds_are_inclusive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scale: str
    ) -> None:
        _, renderer, _ = _run(
            monkeypatch, ["card", "-o", str(tmp_path / "c.svg"), "--png", "--scale", scale]
        )
        assert renderer.scales == [float(scale)]

    @pytest.mark.parametrize("scale", ["0", "-1", "nan", "0.2", "8.5", "inf"])
    def test_bad_scale_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, scale: str) -> None:
        def load(**_: Any) -> CardData:
            raise AssertionError("a bad scale must fail before any read")

        monkeypatch.setattr(card_cli, "load_card_data", load)
        with pytest.raises(SystemExit) as exc_info:
            card_cli.run_card_command(_args(["card", "--scale", scale]), Console(record=True))
        assert exc_info.value.code == 2


class TestTip:
    def test_full_card_has_no_tip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        text, _, _ = _run(monkeypatch, ["card", "-o", str(tmp_path / "c.svg")])
        assert "Tip" not in text

    def test_empty_card_names_every_missing_section(self) -> None:
        tip = card_cli.fill_tip(EMPTY_CARD)
        assert tip == (
            "To fill the empty sections, add --perf --perf-preset long --needle "
            "--gsm8k --mmlu --ifeval to the next run and include the tool-call scenarios."
        )

    def test_each_missing_accuracy_benchmark_is_named(self) -> None:
        # GSM8K is on the card, so only the failed or skipped ones are suggested.
        card = dataclasses.replace(FULL_CARD, accuracy=(AccuracyResult("GSM8K", 0.9),))
        assert card_cli.fill_tip(card) == (
            "To fill the empty sections, add --mmlu --ifeval to the next run."
        )

    def test_only_tool_eval_missing(self) -> None:
        card = CardData(
            model="M",
            tool_version="1",
            created_at="",
            perf=FULL_CARD.perf,
            needle=FULL_CARD.needle,
            accuracy=FULL_CARD.accuracy,
        )
        assert card_cli.fill_tip(card) == (
            "To fill the empty sections, include the tool-call scenarios."
        )

    def test_tip_is_printed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        text, _, _ = _run(monkeypatch, ["card", "-o", str(tmp_path / "c.svg")], card=EMPTY_CARD)
        assert "Tip: To fill the empty sections, add --perf --perf-preset long" in text


class TestLocalCommandRouting:
    def _handlers(self) -> dict[str, Any]:
        def unexpected(*_: Any, **__: Any) -> None:
            raise AssertionError("only the card handler should run")

        return {
            "resolve_scenarios": unexpected,
            "print_history": unexpected,
            "print_leaderboard": unexpected,
            "export_runs": unexpected,
            "compare_runs": unexpected,
        }

    @pytest.mark.parametrize("argv", [["card"], ["--card"], ["--card-run", "r1"]])
    def test_card_is_a_local_command(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        calls: list[argparse.Namespace] = []
        monkeypatch.setattr(card_cli, "run_card_command", lambda args, _console: calls.append(args))
        assert handle_local_command(_args(argv), Console(record=True), **self._handlers())
        assert len(calls) == 1

    def test_json_is_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            handle_local_command(_args(["--card", "--json"]), Console(), **self._handlers())
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "--card does not support --json" in captured.out + captured.err


def _store_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import open_repository

    monkeypatch.chdir(tmp_path)
    with open_repository() as repo:
        for run_type, scores in (
            ("needle", {"accuracy": 50.0, "effective_context": None, "cells": []}),
            ("gsm8k", {"accuracy": 80.0, "total": 20, "n_shots": 8}),
        ):
            repo.upsert_scenario_run(
                {
                    "run_id": f"run-{run_type}",
                    "run_type": run_type,
                    "config": {"model": "org/Tiny-1B", "base_url": "http://***:1"},
                    "scores": scores,
                    "metadata": {"session_id": "sess-e2e", "hostname": "box-private"},
                }
            )


def test_main_renders_a_stored_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End to end through ``main()`` and the real renderer: stored rows in, a card out."""
    from tool_eval_bench.cli.dispatch import main

    _store_session(tmp_path, monkeypatch)
    # Simulate a missing [card] extra: the SVG is still written, then exit 1.
    monkeypatch.setitem(sys.modules, "resvg_py", None)
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", "card", "--png", "--hardware", "M3 Max"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    (svg,) = (tmp_path / "runs" / "cards").glob("tiny-1b-*.svg")
    text = svg.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<svg")
    assert "Tiny-1B" in text
    assert "M3 Max" in text
    assert "box-private" not in text
    assert "sess-e2e" not in text
    assert not svg.with_suffix(".png").exists()


def test_real_png_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("resvg_py")
    from tool_eval_bench.cli.dispatch import main

    _store_session(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["tool-eval-bench", "card", "-o", str(tmp_path / "c.png"), "--scale", "1"]
    )
    main()
    assert (tmp_path / "c.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "c.svg").exists()


def test_card_in_an_empty_directory_creates_no_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.cli.dispatch import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", "card"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "runs").exists()
