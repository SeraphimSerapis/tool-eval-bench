"""Unit tests for CLI helpers, scenario resolution, and plugin runner edge cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from tool_eval_bench.domain.scenarios import Category

# ---------------------------------------------------------------------------
# _resolve_scenarios
# ---------------------------------------------------------------------------


def _resolve_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "short": False,
        "scenarios": None,
        "categories": None,
        "hardmode": False,
        "hardmode_only": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveScenarios:
    def test_default_returns_all_scenarios(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        result = _resolve_scenarios(_resolve_args())
        assert len(result) == len(ALL_SCENARIOS)
        assert all(s.category != Category.P for s in result)

    def test_short_returns_core_scenarios(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import SCENARIOS

        result = _resolve_scenarios(_resolve_args(short=True))
        assert len(result) == len(SCENARIOS)

    def test_hardmode_adds_hardmode_to_all(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, HARDMODE_SCENARIOS

        result = _resolve_scenarios(_resolve_args(hardmode=True))
        assert len(result) == len(ALL_SCENARIOS) + len(HARDMODE_SCENARIOS)
        assert any(s.category == Category.P for s in result)

    def test_hardmode_adds_hardmode_to_short(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import HARDMODE_SCENARIOS, SCENARIOS

        result = _resolve_scenarios(_resolve_args(short=True, hardmode=True))
        assert len(result) == len(SCENARIOS) + len(HARDMODE_SCENARIOS)

    def test_categories_filter(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(categories=["A", "K"]))
        assert all(s.category in {Category.A, Category.K} for s in result)
        assert len(result) > 0

    def test_categories_lowercase(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(categories=["a"]))
        assert all(s.category == Category.A for s in result)

    def test_scenarios_override_all(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(scenarios=["TC-01", "TC-02"], categories=["K"]))
        assert [s.id for s in result] == ["TC-01", "TC-02"]

    def test_explicit_hardmode_ids_opt_in_without_hardmode_flag(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(scenarios=["TC-85", "TC-88"]))

        assert [s.id for s in result] == ["TC-85", "TC-88"]
        assert all(s.category == Category.P for s in result)

    def test_explicit_ids_override_short_pool(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(short=True, scenarios=["TC-70"]))

        assert [s.id for s in result] == ["TC-70"]

    def test_unknown_explicit_id_fails_at_resolution(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        with pytest.raises(ValueError, match=r"Unknown scenarios: NOPE-99"):
            _resolve_scenarios(_resolve_args(scenarios=["TC-01", "NOPE-99"]))

    def test_hardmode_only_stays_restrictive_for_explicit_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        with pytest.raises(ValueError, match=r"Unknown scenarios: TC-01"):
            _resolve_scenarios(_resolve_args(hardmode_only=True, scenarios=["TC-01"]))

    def test_hardmode_only_with_categories_filters_within_hardmode(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(hardmode_only=True, categories=["P"]))
        assert all(s.category == Category.P for s in result)
        assert len(result) > 0

    def test_hardmode_only_with_non_matching_category_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios

        result = _resolve_scenarios(_resolve_args(hardmode_only=True, categories=["A"]))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _resolve_all_scenarios_for_ids
# ---------------------------------------------------------------------------


class TestResolveAllScenariosForIds:
    def test_resolves_known_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids(["TC-01", "TC-02"])
        assert [s.id for s in result] == ["TC-01", "TC-02"]

    def test_ignores_unknown_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids(["TC-01", "UNKNOWN-99"])
        assert [s.id for s in result] == ["TC-01"]

    def test_resolves_hardmode_ids(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids(["TC-70", "TC-84"])
        assert [s.id for s in result] == ["TC-70", "TC-84"]
        assert all(s.category == Category.P for s in result)

    def test_empty_list_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_all_scenarios_for_ids

        result = _resolve_all_scenarios_for_ids([])
        assert result == []


class TestScenarioSelectorDispatch:
    @pytest.mark.parametrize(
        "argv",
        [
            ["--scenarios", "TC-85"],
            ["run", "--scenarios", "TC-85"],
        ],
    )
    def test_legacy_and_subcommand_parsers_preserve_hardmode_id(self, argv: list[str]) -> None:
        from tool_eval_bench.cli.bench import _make_parser
        from tool_eval_bench.cli.parser import parse_cli_args

        _, args = parse_cli_args(_make_parser, argv)

        assert args.scenarios == ["TC-85"]

    def test_run_rejects_unknown_id_before_model_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from tool_eval_bench.cli import dispatch

        monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
        monkeypatch.setattr(
            dispatch,
            "_detect_model",
            lambda *args, **kwargs: pytest.fail("model discovery should not run"),
        )
        monkeypatch.setattr(
            dispatch,
            "_discover_server",
            lambda *args, **kwargs: pytest.fail("server discovery should not run"),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "tool-eval-bench",
                "--scenarios",
                "NOPE-99",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            dispatch.main()

        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _parse_int_list / _parse_sweep_range / _with_config_fingerprint
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_comma_separated(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("1,2,3") == [1, 2, 3]

    def test_space_separated(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("10 20 30") == [10, 20, 30]

    def test_mixed_separators(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("1, 2 3") == [1, 2, 3]

    def test_empty_returns_empty(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("") == []

    def test_extra_whitespace_ignored(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        assert _parse_int_list("  1 ,  2  ") == [1, 2]

    def test_invalid_raises(self) -> None:
        from tool_eval_bench.cli.bench import _parse_int_list

        with pytest.raises(ValueError):
            _parse_int_list("1,abc,3")


class TestParseSweepRange:
    def test_valid_range(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        assert _parse_sweep_range("0.5-1.0") == (0.5, 1.0)

    def test_clamps_above_one(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        assert _parse_sweep_range("0.5-1.5") == (0.5, 1.0)

    def test_rejects_single_value(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError):
            _parse_sweep_range("0.5")

    def test_rejects_non_numeric(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError):
            _parse_sweep_range("abc-def")

    def test_rejects_start_gte_end(self) -> None:
        from tool_eval_bench.cli.bench import _parse_sweep_range

        with pytest.raises(ValueError):
            _parse_sweep_range("0.8-0.8")

        with pytest.raises(ValueError):
            _parse_sweep_range("0.9-0.5")


class TestWithConfigFingerprint:
    def test_adds_fingerprint(self) -> None:
        from tool_eval_bench.cli.bench import _with_config_fingerprint

        config = {"model": "test", "backend": "vllm"}
        result = _with_config_fingerprint(config)
        assert "config_fingerprint" in result
        assert result["model"] == "test"
        assert result["backend"] == "vllm"

    def test_fingerprint_is_deterministic(self) -> None:
        from tool_eval_bench.cli.bench import _with_config_fingerprint

        config = {"model": "test", "backend": "vllm"}
        fp1 = _with_config_fingerprint(config)["config_fingerprint"]
        fp2 = _with_config_fingerprint(config)["config_fingerprint"]
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# _emit_json_output
# ---------------------------------------------------------------------------


class TestEmitJsonOutput:
    def test_writes_to_file(self, tmp_path: Path) -> None:
        from tool_eval_bench.cli.bench import _emit_json_output

        out_file = tmp_path / "result.json"
        data = {"final_score": 87, "model": "test"}
        _emit_json_output(data, json_file=str(out_file))

        assert out_file.exists()
        parsed = json.loads(out_file.read_text(encoding="utf-8"))
        assert parsed["final_score"] == 87
        assert parsed["model"] == "test"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from tool_eval_bench.cli.bench import _emit_json_output

        out_file = tmp_path / "nested" / "dir" / "result.json"
        _emit_json_output({"final_score": 90}, json_file=str(out_file))
        assert out_file.exists()

    def test_prints_to_stdout(self, capsys) -> None:
        from tool_eval_bench.cli.bench import _emit_json_output

        _emit_json_output({"final_score": 75})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["final_score"] == 75


# ---------------------------------------------------------------------------
# Trial aggregation helpers
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_single_value_and_identical_values(self) -> None:
        from tool_eval_bench.cli.bench import _bootstrap_ci

        assert _bootstrap_ci([85.0]) == (85.0, 85.0)
        assert _bootstrap_ci([90.0, 90.0, 90.0, 90.0]) == (90.0, 90.0)

    def test_ci_contains_mean_is_deterministic_and_narrows(self) -> None:
        from statistics import mean

        from tool_eval_bench.cli.bench import _bootstrap_ci

        values = [80.0, 85.0, 90.0, 82.0, 88.0]
        lo, hi = _bootstrap_ci(values)
        assert lo <= mean(values) <= hi
        assert _bootstrap_ci([70.0, 80.0, 90.0]) == _bootstrap_ci([70.0, 80.0, 90.0])

        narrow = _bootstrap_ci([89.0, 90.0, 91.0, 90.0, 90.0])
        wide = _bootstrap_ci([60.0, 80.0, 100.0, 70.0, 90.0])
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


class TestMedian:
    def test_median_odd_even_and_single(self) -> None:
        from tool_eval_bench.cli.bench import _median

        assert _median([1.0, 3.0, 5.0]) == 3.0
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert _median([42.0]) == 42.0


class TestPassAtKMetrics:
    """Verify Pass@k / Pass^k computation in trial aggregation."""

    def test_pass_at_k_computation(self) -> None:
        from tool_eval_bench.cli.bench import _aggregate_trials
        from tool_eval_bench.domain.scenarios import (
            Category,
            CategoryScore,
            ModelScoreSummary,
            ScenarioResult,
            ScenarioStatus,
        )

        def make_summary(results: list[tuple[str, int]]) -> ModelScoreSummary:
            return ModelScoreSummary(
                final_score=50.0,
                total_points=sum(p for _, p in results),
                max_points=6,
                rating="★★ Below Average",
                scenario_results=[
                    ScenarioResult(
                        scenario_id=sid,
                        status=ScenarioStatus.PASS if pts == 2 else ScenarioStatus.FAIL,
                        points=pts,
                        summary="",
                    )
                    for sid, pts in results
                ],
                category_scores=[
                    CategoryScore(
                        category=Category.A,
                        label="A",
                        earned=sum(p for _, p in results),
                        max_points=6,
                        percent=50.0,
                    ),
                ],
            )

        trial_1 = make_summary([("SC-A", 2), ("SC-B", 2), ("SC-C", 0)])
        trial_2 = make_summary([("SC-A", 2), ("SC-B", 0), ("SC-C", 0)])

        agg = _aggregate_trials([trial_1, trial_2])

        assert agg["per_scenario"]["SC-A"]["pass_at_k"] is True
        assert agg["per_scenario"]["SC-A"]["pass_hat_k"] is True
        assert agg["per_scenario"]["SC-B"]["pass_at_k"] is True
        assert agg["per_scenario"]["SC-B"]["pass_hat_k"] is False
        assert agg["per_scenario"]["SC-C"]["pass_at_k"] is False
        assert agg["per_scenario"]["SC-C"]["pass_hat_k"] is False
        assert agg["pass_at_k"] == round(100 * 2 / 3, 1)
        assert agg["pass_hat_k"] == round(100 * 1 / 3, 1)
        assert agg["reliability_gap"] == round(agg["pass_at_k"] - agg["pass_hat_k"], 1)


# ---------------------------------------------------------------------------
# Plugin runner edge cases (accuracy, needle, toolscale)
# ---------------------------------------------------------------------------


class TestPluginRunnerEdgeCases:
    """Gap coverage for decomposed plugin runners under ``cli/plugin_*.py``."""

    def test_mmlu_test_and_dev_load_failures_exit_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tool_eval_bench.cli import plugin_runners

        args = argparse.Namespace(
            mmlu_shots=5,
            mmlu_limit=10,
            mmlu_subjects=None,
            seed=None,
            parallel=1,
            temperature=0.0,
            timeout=10.0,
            format=None,
        )

        # Test split load fails -> immediate exit 1
        monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", lambda *a, **k: None)
        with pytest.raises(SystemExit) as exc:
            plugin_runners._run_mmlu_benchmark(
                Console(record=True), "m", "m", "http://h/v1", None, args
            )
        assert exc.value.code == 1

        # Test split loads, dev split fails -> exit 1
        calls = iter([[{"subject": "math"}], None])
        monkeypatch.setattr(
            plugin_runners, "load_dataset_with_progress", lambda *a, **k: next(calls)
        )
        with pytest.raises(SystemExit) as exc2:
            plugin_runners._run_mmlu_benchmark(
                Console(record=True), "m", "m", "http://h/v1", None, args
            )
        assert exc2.value.code == 1

    def test_ifeval_dataset_load_failure_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_eval_bench.cli import plugin_runners

        args = argparse.Namespace(
            ifeval_limit=5,
            seed=None,
            parallel=1,
            temperature=0.0,
            timeout=10.0,
            format=None,
        )
        monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", lambda *a, **k: None)
        with pytest.raises(SystemExit) as exc:
            plugin_runners._run_ifeval_benchmark(
                Console(record=True), "m", "m", "http://h/v1", None, args
            )
        assert exc.value.code == 1

    def test_needle_caps_context_window_to_kv_cache_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tool_eval_bench.cli import plugin_runners
        from tool_eval_bench.runner import context_pressure

        async def fake_context(*_a: Any, **_kw: Any) -> int:
            return 131_072

        async def fake_kv(*_a: Any, **_kw: Any) -> Any:
            return SimpleNamespace(capacity=32_768, is_hybrid=False)

        monkeypatch.setattr(context_pressure, "detect_context_size", fake_context)
        monkeypatch.setattr(context_pressure, "detect_kv_capacity", fake_kv)

        console = Console(record=True, width=160)
        args = argparse.Namespace(context_size=None, metrics_url=None)
        resolved = plugin_runners._resolve_needle_context_size(
            console, "http://h/v1", "m", None, args
        )
        assert resolved == 32_768
        assert (
            "KV cache capacity (32,768 tokens) < max_model_len (131,072)" in console.export_text()
        )

    def test_toolscale_warns_on_custom_catalog_lint_and_closes_adapter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_eval_bench.adapters import factory
        from tool_eval_bench.cli import plugin_runners
        from tool_eval_bench.plugins.toolscale import catalog as ts_catalog
        from tool_eval_bench.plugins.toolscale import grid as ts_grid
        from tool_eval_bench.plugins.toolscale.plugin import ToolscalePlugin

        closed: list[bool] = []

        class _FakeAdapter:
            async def aclose(self) -> None:
                closed.append(True)

        fake_catalog = SimpleNamespace(
            is_custom=True, version="custom", source="file", sha256="deadbeef"
        )
        fake_item = SimpleNamespace(
            count=4,
            similarity="near",
            cue="literal",
            density="sparse",
            descriptions="terse",
            target_rank=0,
        )
        fake_item_bare = SimpleNamespace(
            count=4,
            similarity="near",
            cue="literal",
            density="sparse",
            descriptions="bare",
            target_rank=0,
        )

        async def fake_run(self: Any, adapter: Any, *, on_progress: Any, **kwargs: Any) -> Any:
            await on_progress(
                1,
                2,
                {
                    "count": 4,
                    "similarity": "near",
                    "cue": "literal",
                    "density": "sparse",
                    "descriptions": "terse",
                    "correct": True,
                    "outcome": "correct",
                },
            )
            return SimpleNamespace(
                score=100.0,
                rating="★★★★★ Excellent",
                duration_seconds=1.0,
                total_tokens=42,
                details={
                    "total": 2,
                    "grid_items": 2,
                    "grid_called": 2,
                    "grid_correct": 2,
                    "grid_errors": 0,
                    "accuracy": 100.0,
                    "accuracy_when_called": 100.0,
                    "call_rate": 100.0,
                    "errors": 0,
                    "counts": [4],
                    "similarities": ["near"],
                    "cues": ["literal"],
                    "densities": ["sparse"],
                    "descriptions": ["terse", "bare"],
                    "cells": [
                        {
                            "count": 4,
                            "similarity": "near",
                            "cue": "literal",
                            "density": "sparse",
                            "descriptions": "terse",
                            "accuracy_when_called": 100.0,
                            "call_rate": 100.0,
                        }
                    ],
                    "by_descriptions": {
                        "terse": {"accuracy_when_called": 100.0, "call_rate": 100.0},
                        "bare": {"accuracy_when_called": 50.0, "call_rate": 100.0},
                    },
                    "control": {"literal": {"accuracy": 100.0, "correct": 1, "items": 1}},
                },
            )

        monkeypatch.setattr(ts_catalog, "load_catalog_file", lambda _p: fake_catalog)
        monkeypatch.setattr(ts_catalog, "lint_catalog", lambda _c: ["duplicate name"])
        monkeypatch.setattr(ts_grid, "build_items", lambda *_a, **_kw: [fake_item, fake_item_bare])
        monkeypatch.setattr(factory, "build_adapter", lambda *_a, **_kw: _FakeAdapter())
        monkeypatch.setattr(ToolscalePlugin, "run", fake_run)
        monkeypatch.setattr(ToolscalePlugin, "render_report_section", lambda self, res: ["report"])
        monkeypatch.setattr("tool_eval_bench.application.run_queries.persist_run", lambda _r: None)

        args = argparse.Namespace(
            seed=0,
            trials=1,
            toolscale_catalog=str(tmp_path / "custom.yaml"),
            toolscale_limit=1,
            toolscale_counts="4",
            toolscale_similarity="near",
            toolscale_cues="literal",
            toolscale_fillers="sparse",
            toolscale_descriptions="terse,bare",
            parallel=1,
            temperature=0.0,
            timeout=10.0,
            format=None,
            label=None,
        )
        console = Console(record=True, width=160)
        plugin_runners._run_toolscale_benchmark(
            console, "m", "m", "http://h/v1", None, args, output_dir=str(tmp_path)
        )

        out = console.export_text()
        assert "1 catalog lint finding(s)" in out
        assert "Description effects compare against clean descriptions" in out
        assert closed == [True]
