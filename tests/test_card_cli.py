"""The ``card`` command: argument translation, output paths, and the PNG fallback."""

from __future__ import annotations

import argparse
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
    accuracy=(AccuracyResult("GSM8K", 0.9),),
)
EMPTY_CARD = CardData(model="M", tool_version="3.1.0", created_at="not a date")


class _PngExportUnavailable(RuntimeError):
    pass


class _Renderer:
    """Stands in for ``storage.reports.card`` so these tests pin the CLI, not the drawing."""

    PngExportUnavailable = _PngExportUnavailable

    def __init__(self, *, png: bool = True) -> None:
        self.png = png
        self.layouts: list[str] = []
        self.scales: list[float] = []

    def render_card_svg(self, data: CardData, *, layout: str = "portrait") -> str:
        self.layouts.append(layout)
        return f"<svg>{data.model}</svg>"

    def svg_to_png(self, svg: str, *, scale: float = 2.0) -> bytes:
        self.scales.append(scale)
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
        assert loaded == {"session": "latest", "run_ids": (), "model": None, "hardware": "H100"}

    def test_default_root_is_runs_under_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _run(monkeypatch, ["card"], card=EMPTY_CARD)
        # An unparseable timestamp falls back to today's date.
        assert len(list((tmp_path / "runs" / "cards").glob("m-*.svg"))) == 1

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

    @pytest.mark.parametrize("scale", ["0", "-1", "nan"])
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
            "--gsm8k --mmlu --ifeval to the next run and run the tool-call scenarios "
            "(without --skip-tool-eval or a --*-only flag)."
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
        tip = card_cli.fill_tip(card)
        assert tip is not None
        assert tip.startswith("To fill the empty sections, run the tool-call scenarios")

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
