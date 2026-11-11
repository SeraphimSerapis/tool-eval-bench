"""Tests for paired 3-tier statistical comparison and discordant re-testing."""

from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.application.comparison import (
    compute_paired_comparison,
    exact_two_sided_sign_p_value,
    extract_result_maps,
    is_infrastructure_failure_dict,
)
from tool_eval_bench.application.leaderboard import compute_mcnemar_stats
from tool_eval_bench.cli.dispatch import main
from tool_eval_bench.cli.history import compare_runs
from tool_eval_bench.cli.retest_diffs import run_retest_diffs_command
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app

SECRET_PACK_TITLE = "Audit confidential payroll ledger"
SECRET_PACK_PROMPT = "Locate duplicate wire transfer WIRE-9981 and flag it"
SECRET_PACK_SUMMARY = "Confidential summary for WIRE-9981"
SECRET_PACK_TRACE = "CONFIDENTIAL TRACE WIRE-9981"


def _write_pack(root: Path, *ids: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for scenario_id in ids:
        content = (
            f"id: {scenario_id}\n"
            f"title: {SECRET_PACK_TITLE}\n"
            "category: A\n"
            "difficulty: 3\n"
            f"user_message: {SECRET_PACK_PROMPT}\n"
            "expected_tool_calls: []\n"
            "tool_responses: {}\n"
        )
        (root / f"{scenario_id.lower()}.yaml").write_text(content, encoding="utf-8")
    return root


def _scenario_row(
    scenario_id: str,
    *,
    points: int,
    status: str | None = None,
    failure_kind: str | None = None,
    summary: str = "ok",
    raw_log: str = "trace",
    duration_seconds: float = 1.0,
) -> dict[str, Any]:
    resolved_status = status or ("pass" if points == 2 else ("partial" if points == 1 else "fail"))
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "status": resolved_status,
        "points": points,
        "summary": summary,
        "raw_log": raw_log,
        "duration_seconds": duration_seconds,
    }
    if failure_kind is not None:
        row["failure_kind"] = failure_kind
    return row


def _stored_run(
    run_id: str,
    results: list[dict[str, Any]],
    *,
    model: str = "qwen3-32b",
    backend: str = "vllm",
    fingerprint: str = "fp-shared",
    config_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_pts = sum(int(r.get("points") or 0) for r in results)
    max_pts = max(2 * len(results), 1)
    final_score = round((total_pts / max_pts) * 100)
    cfg: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "temperature": 0.0,
        "max_turns": 8,
        "scenario_count": len(results),
        "config_fingerprint": fingerprint,
    }
    if config_extra:
        cfg.update(config_extra)
    return {
        "run_id": run_id,
        "status": "completed",
        "run_type": "tool_eval",
        "model": model,
        "config": cfg,
        "metadata": {
            "model": model,
            "backend": backend,
            "tool_version": "2.8.0",
            "git_sha": "abc1234",
        },
        "scores": {
            "final_score": final_score,
            "rating": "Good",
            "total_points": total_pts,
            "max_points": max_pts,
            "completion_rate": 100,
            "total_tokens": 1000,
            "category_scores": [{"category": "A", "label": "Selection", "percent": final_score}],
            "scenario_results": results,
        },
    }


def _seed_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runs: list[dict[str, Any]],
) -> str:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        for run in runs:
            repo.upsert_scenario_run(run)
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    return db_path


class TestMcNemarAndPairedComparison:
    def test_mcnemar_excludes_infrastructure_failures_and_keeps_model_failures(self) -> None:
        map_a = {
            "TC-01": _scenario_row("TC-01", points=2),
            "TC-02": _scenario_row("TC-02", points=0, failure_kind="timeout"),
            "TC-03": _scenario_row("TC-03", points=2),
            "TC-04": _scenario_row("TC-04", points=0, failure_kind="wrong_tool"),
        }
        map_b = {
            "TC-01": _scenario_row("TC-01", points=0, failure_kind="server_error"),
            "TC-02": _scenario_row("TC-02", points=2),
            "TC-03": _scenario_row("TC-03", points=2),
            "TC-04": _scenario_row("TC-04", points=2),
        }
        stats = compute_mcnemar_stats(
            map_a,
            map_b,
            ["TC-01", "TC-02", "TC-03", "TC-04", "TC-04"],
        )
        assert set(stats.keys()) == {
            "discordant_pairs",
            "a_only_pass",
            "b_only_pass",
            "chi2",
            "p_value",
            "significant",
            "direction",
        }
        # TC-01 and TC-02 are excluded due to server_error / timeout; only TC-04 is discordant.
        assert stats["discordant_pairs"] == 1
        assert stats["a_only_pass"] == 0
        assert stats["b_only_pass"] == 1

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("timeout", True),
            ("connection_error", True),
            ("server_error", True),
            ("wrong_tool", False),
            ("empty_turn", False),
            (None, False),
        ],
    )
    def test_is_infrastructure_failure_dict(self, kind: str | None, expected: bool) -> None:
        row = _scenario_row("TC-01", points=0, failure_kind=kind)
        assert is_infrastructure_failure_dict(row) is expected

    def test_partial_credit_transitions_detected_when_binary_mcnemar_sees_zero(self) -> None:
        # Six scenarios improve from 0 (fail) to 1 (partial). Binary McNemar sees
        # fail -> fail on all 6 (0 discordant pairs), whereas 3-tier paired stats
        # detect all 6 improvements with exact two-sided p = 2 / 64 = 0.03125 < 0.05.
        ids = [f"TC-0{i}" for i in range(1, 7)]
        map_a = {sid: _scenario_row(sid, points=0, status="fail") for sid in ids}
        map_b = {sid: _scenario_row(sid, points=1, status="partial") for sid in ids}

        paired = compute_paired_comparison(map_a, map_b, ids, seed=0)
        assert paired["shared_count"] == 6
        assert paired["infra_excluded_ids"] == []
        assert paired["discordant_scenario_ids"] == ids
        assert paired["improved_ids"] == ids
        assert paired["regressed_ids"] == []
        assert paired["mean_pct_delta"] == pytest.approx(50.0)
        assert paired["ci95_low_pct"] == pytest.approx(50.0)
        assert paired["ci95_high_pct"] == pytest.approx(50.0)
        assert paired["sign_p_value"] == pytest.approx(0.03125)
        assert paired["mcnemar"]["discordant_pairs"] == 0
        assert paired["verdict"] == "significant_improvement"

    def test_paired_bootstrap_ci_is_deterministic_and_excludes_infra(self) -> None:
        results_a = [
            _scenario_row("TC-01", points=0),
            _scenario_row("TC-02", points=2),
            _scenario_row("TC-03", points=1),
            _scenario_row("TC-04", points=2),
            _scenario_row("TC-05", points=0, failure_kind="connection_error"),
            _scenario_row("TC-06", points=2),
        ]
        results_b = [
            _scenario_row("TC-01", points=2),
            _scenario_row("TC-02", points=1),
            _scenario_row("TC-03", points=2),
            _scenario_row("TC-04", points=2),
            _scenario_row("TC-05", points=2),
            _scenario_row("TC-07", points=2),
        ]
        map_a, map_b, all_ids = extract_result_maps(results_a, results_b)
        first = compute_paired_comparison(map_a, map_b, all_ids, seed=7)
        second = compute_paired_comparison(map_a, map_b, all_ids, seed=7)

        assert first == second
        assert first["shared_count"] == 4
        assert first["infra_excluded_ids"] == ["TC-05"]
        assert first["discordant_scenario_ids"] == ["TC-01", "TC-02", "TC-03"]
        assert first["improved_ids"] == ["TC-01", "TC-03"]
        assert first["regressed_ids"] == ["TC-02"]
        # Deltas on TC-01..TC-04 are +2, -1, +1, 0 -> sum = +2 over 2*4=8 max points -> +25.0%
        assert first["mean_pct_delta"] == pytest.approx(25.0)
        assert first["ci95_low_pct"] < first["mean_pct_delta"] < first["ci95_high_pct"]
        assert first["verdict"] == "inconclusive_noise"

    def test_all_verdicts_and_status_only_discordance(self) -> None:
        # 1. no_shared_scenarios (only infra shared)
        infra_only = compute_paired_comparison(
            {"TC-01": _scenario_row("TC-01", points=0, failure_kind="timeout")},
            {"TC-01": _scenario_row("TC-01", points=2)},
            ["TC-01"],
        )
        assert infra_only["verdict"] == "no_shared_scenarios"
        assert infra_only["mean_pct_delta"] == 0.0
        assert infra_only["sign_p_value"] == 1.0

        # 2. identical
        identical = compute_paired_comparison(
            {"TC-01": _scenario_row("TC-01", points=2)},
            {"TC-01": _scenario_row("TC-01", points=2)},
            ["TC-01"],
        )
        assert identical["verdict"] == "identical"
        assert identical["discordant_scenario_ids"] == []

        # 3. significant_regression (6 regressions, 0 improvements -> p = 0.03125)
        ids = [f"TC-0{i}" for i in range(1, 7)]
        reg = compute_paired_comparison(
            {sid: _scenario_row(sid, points=2) for sid in ids},
            {sid: _scenario_row(sid, points=1) for sid in ids},
            ids,
        )
        assert reg["verdict"] == "significant_regression"
        assert reg["mean_pct_delta"] == pytest.approx(-50.0)

        # 4. status mismatch with equal points still registers as discordant
        status_diff = compute_paired_comparison(
            {"TC-01": _scenario_row("TC-01", points=0, status="fail")},
            {"TC-01": _scenario_row("TC-01", points=0, status="partial")},
            ["TC-01"],
        )
        assert status_diff["discordant_scenario_ids"] == ["TC-01"]
        assert status_diff["verdict"] == "inconclusive_noise"

        # Exact sign test symmetry and clamping at 1.0 when counts match
        assert exact_two_sided_sign_p_value(2, 2) == 1.0
        assert exact_two_sided_sign_p_value(1, 7) == exact_two_sided_sign_p_value(7, 1)


class TestCompareRunsCli:
    def test_compare_runs_shows_paired_ci_infra_note_and_retest_commands(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_a = _stored_run(
            "run_alpha",
            [
                _scenario_row("TC-01", points=0),
                _scenario_row("TC-02", points=2),
                _scenario_row("TC-03", points=0, failure_kind="timeout"),
            ],
        )
        run_b = _stored_run(
            "run_beta",
            [
                _scenario_row("TC-01", points=2),
                _scenario_row("TC-02", points=2),
                _scenario_row("TC-03", points=2),
            ],
        )
        _seed_db(tmp_path, monkeypatch, [run_a, run_b])

        console = Console(file=StringIO(), width=140, no_color=True)
        compare_runs(console, "run_alpha", "run_beta")
        output = console.file.getvalue()

        assert "Paired quality delta: +50.00 pts, 95% CI [" in output
        assert "3-tier exact sign test: p=1.0000" in output
        assert "Excluded infrastructure failures: 1 pair skipped: TC-03" in output
        assert "Discordant scenario IDs: 1 total: TC-01" in output
        assert "Inconclusive difference" in output
        assert "tool-eval-bench compare run_alpha run_beta --retest-diffs --trials 5" in output
        assert "tool-eval-bench run --scenarios-from-diff run_alpha run_beta --trials 5" in output
        assert "McNemar's test:" in output

    def test_compare_runs_does_not_leak_held_out_titles_or_summaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pack_meta = [
            {
                "name": "priv",
                "scenario_count": 1,
                "scenario_ids": ["HO-01"],
                "content_hash": "deadbeef12345678",
            }
        ]
        run_a = _stored_run(
            "run_ho_a",
            [
                _scenario_row(
                    "HO-01",
                    points=0,
                    summary=SECRET_PACK_SUMMARY,
                    raw_log=SECRET_PACK_TRACE,
                )
            ],
            config_extra={"scenario_packs": pack_meta},
        )
        run_b = _stored_run(
            "run_ho_b",
            [
                _scenario_row(
                    "HO-01",
                    points=2,
                    summary=SECRET_PACK_SUMMARY,
                    raw_log=SECRET_PACK_TRACE,
                )
            ],
            config_extra={"scenario_packs": pack_meta},
        )
        _seed_db(tmp_path, monkeypatch, [run_a, run_b])

        console = Console(file=StringIO(), width=140, no_color=True)
        compare_runs(console, "run_ho_a", "run_ho_b")
        output = console.file.getvalue()

        assert "HO-01" in output
        assert SECRET_PACK_TITLE not in output
        assert SECRET_PACK_SUMMARY not in output
        assert SECRET_PACK_TRACE not in output

    @pytest.mark.parametrize(
        ("pts_a", "pts_b", "infra", "expected_snippet"),
        [
            ([2, 2], [2, 2], False, "Identical outcomes across all shared scenarios."),
            ([0] * 6, [2] * 6, False, "Statistically significant improvement"),
            ([2] * 6, [0] * 6, False, "Statistically significant regression"),
            ([0, 0], [2, 2], True, "No shared non-infrastructure scenarios to compare."),
        ],
    )
    def test_compare_runs_verdict_variants(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        pts_a: list[int],
        pts_b: list[int],
        infra: bool,
        expected_snippet: str,
    ) -> None:
        fk = "server_error" if infra else None
        res_a = [
            _scenario_row(f"TC-0{i + 1}", points=p, failure_kind=fk) for i, p in enumerate(pts_a)
        ]
        res_b = [_scenario_row(f"TC-0{i + 1}", points=p) for i, p in enumerate(pts_b)]
        _seed_db(
            tmp_path,
            monkeypatch,
            [_stored_run("ra", res_a), _stored_run("rb", res_b)],
        )
        console = Console(file=StringIO(), width=140, no_color=True)
        compare_runs(console, "ra", "rb")
        output = console.file.getvalue()
        assert expected_snippet in output


class TestCompareRetestDiffsCli:
    def test_retest_diffs_runs_default_five_trials_per_side_with_hardmode_and_endpoints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_a = _stored_run(
            "base_run",
            [
                _scenario_row("TC-01", points=2),
                _scenario_row("TC-04", points=0),
                _scenario_row("TC-70", points=1),
            ],
            model="model-alpha",
            backend="vllm",
            config_extra={
                "temperature": 0.2,
                "max_turns": 6,
                "weight_by_difficulty": True,
                "system_prompt": "Be concise.",
                "scenario_variants": {
                    "TC-01": {"locale": "de-DE", "seed": 99},
                },
            },
        )
        run_b = _stored_run(
            "cand_run",
            [
                _scenario_row("TC-01", points=2),
                _scenario_row("TC-04", points=2),
                _scenario_row("TC-70", points=2),
            ],
            model="model-beta",
            backend="sglang",
            config_extra={
                "temperature": 0.4,
                "max_turns": 10,
                "locale": "fr-FR",
                "variant_seed": 12,
            },
        )
        _seed_db(tmp_path, monkeypatch, [run_a, run_b])

        recorded_calls: list[dict[str, Any]] = []

        async def _fake_run_benchmark(self: Any, **kwargs: Any) -> dict[str, Any]:
            recorded_calls.append(kwargs)
            is_b = kwargs["model"] == "model-beta"
            return _stored_run(
                f"trial_{len(recorded_calls)}",
                [
                    _scenario_row("TC-04", points=2 if is_b else 0),
                    _scenario_row("TC-70", points=2 if is_b else 1),
                ],
                model=kwargs["model"],
            )

        monkeypatch.setattr(
            "tool_eval_bench.cli.retest_diffs.BenchmarkService.run_benchmark",
            _fake_run_benchmark,
        )

        console_buf = StringIO()
        with patch(
            "tool_eval_bench.cli.dispatch.Console",
            return_value=Console(file=console_buf, width=140, no_color=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "tool-eval-bench",
                    "compare",
                    "base_run",
                    "cand_run",
                    "--retest-diffs",
                    "--base-url-a",
                    "http://host-a.local:8000",
                    "--base-url-b",
                    "http://host-b.local:8001",
                    "--api-key-a",
                    "key-a",
                    "--api-key-b",
                    "key-b",
                ],
            )
            main()

        # Default trials=5 per side -> 10 total calls (5 for A, 5 for B)
        assert len(recorded_calls) == 10
        calls_a = recorded_calls[:5]
        calls_b = recorded_calls[5:]

        for call in calls_a:
            assert [s.id for s in call["scenarios"]] == ["TC-04", "TC-70"]
            assert call["model"] == "model-alpha"
            assert call["backend"] == "vllm"
            assert call["base_url"] == "http://host-a.local:8000"
            assert call["api_key"] == "key-a"
            assert call["temperature"] == pytest.approx(0.2)
            assert call["max_turns"] == 6
            assert call["locale"] == "de-DE"
            assert call["variant_seed"] == 99
            assert call["weight_by_difficulty"] is True
            assert call["system_prompt"] == "Be concise."

        for call in calls_b:
            assert [s.id for s in call["scenarios"]] == ["TC-04", "TC-70"]
            assert call["model"] == "model-beta"
            assert call["backend"] == "sglang"
            assert call["base_url"] == "http://host-b.local:8001"
            assert call["api_key"] == "key-b"
            assert call["temperature"] == pytest.approx(0.4)
            assert call["max_turns"] == 10
            assert call["locale"] == "fr-FR"
            assert call["variant_seed"] == 12
            assert call["weight_by_difficulty"] is False
            assert call["system_prompt"] is None

        output = console_buf.getvalue()
        assert "Paired Discordant Re-Test Summary (k=5)" in output
        assert "TC-04" in output
        assert "TC-70" in output
        assert "0.00/2 (no / no)" in output
        assert "2.00/2 (yes / yes)" in output
        assert "Confirmed improvement" in output

    def test_legacy_flat_compare_retest_diffs_defaults_to_five_trials_and_handles_zero_diffs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_a = _stored_run("same_a", [_scenario_row("TC-01", points=2)])
        run_b = _stored_run("same_b", [_scenario_row("TC-01", points=2)])
        _seed_db(tmp_path, monkeypatch, [run_a, run_b])

        called = False

        async def _unexpected(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(
            "tool_eval_bench.cli.retest_diffs.BenchmarkService.run_benchmark",
            _unexpected,
        )

        console_buf = StringIO()
        with patch(
            "tool_eval_bench.cli.dispatch.Console",
            return_value=Console(file=console_buf, width=140, no_color=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["tool-eval-bench", "--compare", "same_a", "same_b", "--retest-diffs"],
            )
            main()

        assert called is False
        assert "have no discordant scenarios to re-test" in console_buf.getvalue()

    def test_retest_diffs_held_out_pack_requires_flag_and_never_leaks_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pack_dir = _write_pack(tmp_path / "held_out_pack", "HO-77")
        run_a = _stored_run(
            "ho_a",
            [
                _scenario_row(
                    "HO-77",
                    points=0,
                    summary=SECRET_PACK_SUMMARY,
                    raw_log=SECRET_PACK_TRACE,
                )
            ],
        )
        run_b = _stored_run(
            "ho_b",
            [
                _scenario_row(
                    "HO-77",
                    points=2,
                    summary=SECRET_PACK_SUMMARY,
                    raw_log=SECRET_PACK_TRACE,
                )
            ],
        )
        _seed_db(tmp_path, monkeypatch, [run_a, run_b])

        # Without --scenario-pack: exits 2 with clear guidance
        buf_err = StringIO()
        with patch(
            "tool_eval_bench.cli.dispatch.Console",
            return_value=Console(file=buf_err, width=140, no_color=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "tool-eval-bench",
                    "compare",
                    "ho_a",
                    "ho_b",
                    "--retest-diffs",
                    "--trials",
                    "2",
                ],
            )
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2
        assert "--scenario-pack" in buf_err.getvalue()

        # With --scenario-pack: resolves HO-77 cleanly and never prints secret title/summary
        async def _fake_pack_run(self: Any, **kwargs: Any) -> dict[str, Any]:
            assert [s.id for s in kwargs["scenarios"]] == ["HO-77"]
            assert kwargs["scenario_packs"] is not None
            return _stored_run(
                "ho_trial",
                [
                    _scenario_row(
                        "HO-77",
                        points=1,
                        summary=SECRET_PACK_SUMMARY,
                        raw_log=SECRET_PACK_TRACE,
                    )
                ],
            )

        monkeypatch.setattr(
            "tool_eval_bench.cli.retest_diffs.BenchmarkService.run_benchmark",
            _fake_pack_run,
        )
        buf_ok = StringIO()
        with patch(
            "tool_eval_bench.cli.dispatch.Console",
            return_value=Console(file=buf_ok, width=140, no_color=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "tool-eval-bench",
                    "compare",
                    "ho_a",
                    "ho_b",
                    "--retest-diffs",
                    "--trials",
                    "2",
                    "--scenario-pack",
                    str(pack_dir),
                ],
            )
            main()

        out = buf_ok.getvalue()
        assert "HO-77" in out
        assert "Converged" in out
        assert SECRET_PACK_TITLE not in out
        assert SECRET_PACK_SUMMARY not in out
        assert SECRET_PACK_TRACE not in out

    @pytest.mark.parametrize(
        "argv",
        [
            ["tool-eval-bench", "compare", "a", "b", "--report", "--retest-diffs"],
            ["tool-eval-bench", "compare", "a", "b", "--base-url-a", "http://localhost:8000"],
            ["tool-eval-bench", "--compare", "a", "b", "--report", "--retest-diffs"],
            ["tool-eval-bench", "--base-url-b", "http://localhost:8000", "--dry-run"],
            ["tool-eval-bench", "compare", "a", "b", "--retest-diffs", "--trials", "0"],
            ["tool-eval-bench", "compare", "a", "b", "--retest-diffs", "--base-url", "ftp://bad"],
            ["tool-eval-bench", "compare", "missing_a", "b", "--retest-diffs"],
        ],
    )
    def test_retest_diffs_rejects_invalid_arguments_with_exit_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
    ) -> None:
        _seed_db(
            tmp_path,
            monkeypatch,
            [
                _stored_run("a", [_scenario_row("TC-01", points=0)]),
                _stored_run("b", [_scenario_row("TC-01", points=2)]),
            ],
        )
        buf = StringIO()
        with patch(
            "tool_eval_bench.cli.dispatch.Console",
            return_value=Console(file=buf, width=140, no_color=True),
        ):
            monkeypatch.setattr("sys.argv", argv)
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_retest_diffs_rejects_boolean_trials_and_handles_all_infra_or_regression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_db(
            tmp_path,
            monkeypatch,
            [
                _stored_run("ra", [_scenario_row("TC-01", points=2)]),
                _stored_run("rb", [_scenario_row("TC-01", points=0)]),
            ],
        )
        console = Console(file=StringIO(), width=140, no_color=True)

        # Direct call with bool trials must be rejected (bool is a subclass of int in Python).
        bad_args = argparse.Namespace(compare=["ra", "rb"], trials=True, json=False)
        with pytest.raises(SystemExit) as exc_bool:
            run_retest_diffs_command(bad_args, console)
        assert exc_bool.value.code == 2

        # Confirmed regression across 5 unanimous trials
        async def _reg_run(self: Any, **kwargs: Any) -> dict[str, Any]:
            pts = 2 if kwargs["base_url"].endswith("8001") else 0
            return _stored_run("t", [_scenario_row("TC-01", points=pts)])

        monkeypatch.setattr(
            "tool_eval_bench.cli.retest_diffs.BenchmarkService.run_benchmark",
            _reg_run,
        )
        reg_console = Console(file=StringIO(), width=140, no_color=True)
        ok_args = argparse.Namespace(
            compare=["ra", "rb"],
            trials=5,
            base_url=None,
            base_url_a="http://localhost:8001",
            base_url_b="http://localhost:8002",
            api_key=None,
            api_key_a=None,
            api_key_b=None,
            backend=None,
            scenario_pack=None,
            redact_url=True,
            output_dir=str(tmp_path / "runs"),
            json=False,
        )
        run_retest_diffs_command(ok_args, reg_console)
        reg_out = reg_console.file.getvalue()
        assert "Confirmed regression" in reg_out

        # All-infrastructure failures across trials
        async def _infra_run(self: Any, **kwargs: Any) -> dict[str, Any]:
            return _stored_run(
                "t_infra",
                [_scenario_row("TC-01", points=0, failure_kind="timeout")],
            )

        monkeypatch.setattr(
            "tool_eval_bench.cli.retest_diffs.BenchmarkService.run_benchmark",
            _infra_run,
        )
        infra_console = Console(file=StringIO(), width=140, no_color=True)
        run_retest_diffs_command(ok_args, infra_console)
        assert "No gradable discordant scenarios" in infra_console.file.getvalue()

        # Inconclusive multi-trial variance (trials vary so bootstrap CI runs and overlaps zero)
        seq = iter([0, 2, 0, 2, 1, 2, 0, 2, 0, 2])

        async def _noisy_run(self: Any, **kwargs: Any) -> dict[str, Any]:
            return _stored_run("t_noisy", [_scenario_row("TC-01", points=next(seq))])

        monkeypatch.setattr(
            "tool_eval_bench.cli.retest_diffs.BenchmarkService.run_benchmark",
            _noisy_run,
        )
        noisy_console = Console(file=StringIO(), width=140, no_color=True)
        run_retest_diffs_command(ok_args, noisy_console)
        noisy_out = noisy_console.file.getvalue()
        assert "Inconclusive across trials" in noisy_out

    def test_retest_diffs_validates_stored_config_and_empty_or_latest_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Empty DB with 'latest'
        _seed_db(tmp_path, monkeypatch, [])
        console = Console(file=StringIO(), width=140, no_color=True)
        with pytest.raises(SystemExit) as exc_latest:
            run_retest_diffs_command(
                argparse.Namespace(compare=["latest", "b"], trials=1, json=False),
                console,
            )
        assert exc_latest.value.code == 2

        # Stored run with empty scenario_results or unsupported backend
        empty_run = _stored_run("empty_r", [])
        bad_backend_run = _stored_run(
            "bad_b",
            [_scenario_row("TC-01", points=2)],
            backend="not-a-backend",
        )
        good_run = _stored_run("good_r", [_scenario_row("TC-01", points=0)])
        _seed_db(tmp_path, monkeypatch, [empty_run, bad_backend_run, good_run])

        with pytest.raises(SystemExit) as exc_empty:
            run_retest_diffs_command(
                argparse.Namespace(compare=["empty_r", "good_r"], trials=1, json=False),
                console,
            )
        assert exc_empty.value.code == 2

        with pytest.raises(SystemExit) as exc_backend:
            run_retest_diffs_command(
                argparse.Namespace(
                    compare=["good_r", "bad_b"],
                    trials=1,
                    base_url="http://localhost:8000",
                    base_url_a=None,
                    base_url_b=None,
                    api_key=None,
                    api_key_a=None,
                    api_key_b=None,
                    backend=None,
                    scenario_pack=None,
                    json=False,
                ),
                console,
            )
        assert exc_backend.value.code == 2


class TestRunScenariosFromDiffCli:
    def test_run_scenarios_from_diff_populates_discordant_ids_in_dry_run_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = _stored_run(
            "diff_a",
            [
                _scenario_row("TC-01", points=2),
                _scenario_row("TC-05", points=0),
                _scenario_row("TC-70", points=1),
                _scenario_row("TC-09", points=0, failure_kind="timeout"),
            ],
        )
        run_b = _stored_run(
            "diff_b",
            [
                _scenario_row("TC-01", points=2),
                _scenario_row("TC-05", points=1),
                _scenario_row("TC-70", points=2),
                _scenario_row("TC-09", points=2),
            ],
        )
        _seed_db(tmp_path, monkeypatch, [run_a, run_b])

        monkeypatch.setattr(
            "sys.argv",
            [
                "tool-eval-bench",
                "run",
                "--scenarios-from-diff",
                "diff_a",
                "diff_b",
                "--dry-run",
                "--json",
            ],
        )
        with pytest.raises(SystemExit) as exc_dry:
            main()
        assert exc_dry.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_scenarios"] == 2
        assert [s["id"] for s in payload["scenarios"]] == ["TC-05", "TC-70"]

    def test_run_scenarios_from_diff_exits_zero_when_no_discordant_scenarios(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_db(
            tmp_path,
            monkeypatch,
            [
                _stored_run("eq_a", [_scenario_row("TC-01", points=2)]),
                _stored_run("eq_b", [_scenario_row("TC-01", points=2)]),
            ],
        )

        # JSON output on 0 discordant scenarios
        monkeypatch.setattr(
            "sys.argv",
            [
                "tool-eval-bench",
                "run",
                "--scenarios-from-diff",
                "eq_a",
                "eq_b",
                "--json",
            ],
        )
        with pytest.raises(SystemExit) as exc_json:
            main()
        assert exc_json.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["event"] == "no_discordant_scenarios"
        assert data["discordant_scenario_ids"] == []
        assert data["verdict"] == "identical"

        # --json-file output on 0 discordant scenarios
        json_out = tmp_path / "zero_diff.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "tool-eval-bench",
                "run",
                "--scenarios-from-diff",
                "eq_a",
                "eq_b",
                "--json-file",
                str(json_out),
            ],
        )
        with pytest.raises(SystemExit) as exc_jf:
            main()
        assert exc_jf.value.code == 0
        assert (
            json.loads(json_out.read_text(encoding="utf-8"))["event"] == "no_discordant_scenarios"
        )

        # Human console output on 0 discordant scenarios
        buf = StringIO()
        with patch(
            "tool_eval_bench.cli.dispatch.Console",
            return_value=Console(file=buf, width=140, no_color=True),
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["tool-eval-bench", "run", "--scenarios-from-diff", "eq_a", "eq_b"],
            )
            with pytest.raises(SystemExit) as exc_human:
                main()
        assert exc_human.value.code == 0
        assert "0 discordant scenarios" in buf.getvalue()

    @pytest.mark.parametrize(
        "argv",
        [
            [
                "tool-eval-bench",
                "run",
                "--scenarios-from-diff",
                "missing_1",
                "missing_2",
                "--dry-run",
            ],
            [
                "tool-eval-bench",
                "run",
                "--scenarios-from-diff",
                "d1",
                "d2",
                "--scenarios",
                "TC-01",
                "--dry-run",
            ],
            [
                "tool-eval-bench",
                "--scenarios-from-diff",
                "d1",
                "d2",
                "--resume",
                "d1",
            ],
        ],
    )
    def test_run_scenarios_from_diff_invalid_inputs_exit_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
    ) -> None:
        _seed_db(
            tmp_path,
            monkeypatch,
            [
                _stored_run("d1", [_scenario_row("TC-01", points=0)]),
                _stored_run("d2", [_scenario_row("TC-01", points=2)]),
            ],
        )
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


class TestWebComparePairedStatsAndDomSafety:
    @pytest.mark.asyncio
    async def test_api_compare_includes_paired_stats_excludes_infra_and_redacts_packs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pack_meta = [
            {
                "name": "secret-pack",
                "scenario_count": 1,
                "scenario_ids": ["PACK-01"],
                "content_hash": "abcdef1234567890",
            }
        ]
        run_left = _stored_run(
            "web_left",
            [
                _scenario_row("TC-01", points=0),
                _scenario_row("TC-02", points=0, failure_kind="timeout"),
                _scenario_row(
                    "PACK-01",
                    points=1,
                    summary=SECRET_PACK_SUMMARY,
                    raw_log=SECRET_PACK_TRACE,
                ),
            ],
            config_extra={"scenario_packs": pack_meta},
        )
        run_right = _stored_run(
            "web_right",
            [
                _scenario_row("TC-01", points=1),
                _scenario_row("TC-02", points=2),
                _scenario_row(
                    "PACK-01",
                    points=2,
                    summary=SECRET_PACK_SUMMARY,
                    raw_log=SECRET_PACK_TRACE,
                ),
            ],
            config_extra={"scenario_packs": pack_meta},
        )
        _seed_db(tmp_path, monkeypatch, [run_left, run_right])

        app = create_app(token=TEST_TOKEN)
        async with web_client(app) as client:
            resp = await client.get(
                "/api/compare",
                params={"left": "web_left", "right": "web_right"},
            )
            assert resp.status_code == 200
            body_text = resp.text
            assert SECRET_PACK_SUMMARY not in body_text
            assert SECRET_PACK_TRACE not in body_text

            data = resp.json()
            deltas_by_id = {d["scenario_id"]: d for d in data["deltas"]}
            assert deltas_by_id["PACK-01"]["title"] == "held out"
            assert deltas_by_id["PACK-01"]["a"]["raw_log"] == "held out"
            assert deltas_by_id["PACK-01"]["b"]["raw_log"] == "held out"
            assert "mcnemar" in data
            paired = data["paired_stats"]
            assert paired["shared_count"] == 2
            assert paired["infra_excluded_ids"] == ["TC-02"]
            assert paired["discordant_scenario_ids"] == ["TC-01", "PACK-01"]
            assert paired["improved_ids"] == ["TC-01", "PACK-01"]
            assert paired["regressed_ids"] == []
            assert paired["mean_pct_delta"] == pytest.approx(50.0)
            assert paired["verdict"] == "inconclusive_noise"

    def test_web_app_js_uses_safe_dom_construction_for_compare_stats(self) -> None:
        js_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "tool_eval_bench"
            / "web"
            / "static"
            / "app.js"
        )
        source = js_path.read_text(encoding="utf-8")
        assert "innerHTML" not in source
        assert "outerHTML" not in source
        assert "insertAdjacentHTML" not in source
        assert "paired_stats" in source
        assert "ci95_low_pct" in source
        assert "discordant_scenario_ids" in source
        assert "--scenarios " in source
