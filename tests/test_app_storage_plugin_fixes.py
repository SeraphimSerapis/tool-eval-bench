"""Regression tests for application, SQLite storage, report/card renderers, and plugins."""

from __future__ import annotations

import asyncio
import email.utils
import threading
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from tests.conftest import open_repository
from tool_eval_bench.application.leaderboard import (
    cohort_fingerprint,
    extract_leaderboard_rows,
    is_rank_eligible,
)
from tool_eval_bench.application.resume import (
    ResumeRejected,
    plan_stored_resume,
    result_requires_rerun,
)
from tool_eval_bench.application.run_config import (
    COHORT_EXCLUDED_KEYS,
    RunSettings,
    build_run_config,
    resume_mismatches,
)
from tool_eval_bench.application.run_deletion import delete_runs, is_run_report
from tool_eval_bench.application.service import BenchmarkService, _collect_metadata_safe
from tool_eval_bench.compare_reports._common import _tv
from tool_eval_bench.compare_reports.summary import (
    generate_html as generate_summary_compare_html,
)
from tool_eval_bench.compare_reports.summary import (
    parse_summary as parse_summary_report,
)
from tool_eval_bench.compare_reports.tool_eval import (
    generate_html as generate_tool_compare_html,
)
from tool_eval_bench.compare_reports.tool_eval import (
    parse_md as parse_tool_eval_report,
)
from tool_eval_bench.domain.adapters import ChatCompletionResult
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.plugin import final_answer_text, strip_thinking
from tool_eval_bench.domain.scenarios import (
    Category as ScenarioCategory,
)
from tool_eval_bench.domain.scenarios import (
    CategoryScore,
    ModelScoreSummary,
    ScenarioReportMetadata,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.plugins._thinking import plugin_answer_text
from tool_eval_bench.plugins.decision.metrics import reliability_bins
from tool_eval_bench.plugins.decision.plugin import _error_row as decision_error_row
from tool_eval_bench.plugins.decision.typed_decisions import load_cases as load_decision_cases
from tool_eval_bench.plugins.gsm8k.dataset import GSM8KItem
from tool_eval_bench.plugins.gsm8k.evaluator import evaluate_answer as eval_gsm8k
from tool_eval_bench.plugins.gsm8k.evaluator import extract_answer as extract_gsm8k
from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin
from tool_eval_bench.plugins.hf_utils import _MAX_BACKOFF_S, _parse_retry_after
from tool_eval_bench.plugins.ifeval.checkers import check_instruction
from tool_eval_bench.plugins.ifeval.dataset import IFEvalItem
from tool_eval_bench.plugins.ifeval.evaluator import evaluate_prompt
from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin
from tool_eval_bench.plugins.mmlu.dataset import MMLUItem
from tool_eval_bench.plugins.mmlu.evaluator import evaluate_answer as eval_mmlu
from tool_eval_bench.plugins.mmlu.evaluator import extract_answer as extract_mmlu
from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin
from tool_eval_bench.plugins.needle.haystack import NeedleCase
from tool_eval_bench.plugins.needle.plugin import NeedlePlugin
from tool_eval_bench.plugins.toolscale.catalog import load_builtin_catalog
from tool_eval_bench.plugins.toolscale.grid import build_items
from tool_eval_bench.plugins.toolscale.plugin import ToolscalePlugin
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.storage.reports._common import _code_cell, _render_run_context
from tool_eval_bench.storage.reports.card._svg import attrs as svg_attrs
from tool_eval_bench.storage.reports.card._svg import num as svg_num
from tool_eval_bench.storage.reports.card._svg import text as svg_text
from tool_eval_bench.storage.reports.pressure import pressure_sweep_report
from tool_eval_bench.storage.reports.spec_decode import spec_decode_report
from tool_eval_bench.storage.reports.summary import write_summary_report
from tool_eval_bench.storage.reports.throughput import throughput_report

SECRET_BASE = "http://admin:s3cretP@ss@internal-llm.corp:8000/v1"
SECRET_URL = f"{SECRET_BASE}?api_key=sk-super-secret"


# ---------------------------------------------------------------------------
# 1. SQLite Persistence (storage/db.py)
# ---------------------------------------------------------------------------


class TestSQLiteStorageFixes:
    def test_null_config_scores_and_metadata_persist_safely(self, tmp_path: Path) -> None:
        with open_repository(db_path=str(tmp_path / "db.sqlite")) as repo:
            repo.upsert_scenario_run(
                {
                    "run_id": "null-fields",
                    "model": "fallback-model",
                    "status": "running",
                    "config": None,
                    "scores": None,
                    "metadata": None,
                }
            )
            stored = repo.get("null-fields")
            assert stored is not None
            assert stored["model"] == "fallback-model"
            assert stored["config"] == {}
            assert stored["scores"] is None
            assert stored["metadata"] == {}

    def test_non_finite_floats_serialize_as_valid_json_for_sqlite_search(
        self, tmp_path: Path
    ) -> None:
        with open_repository(db_path=str(tmp_path / "db.sqlite")) as repo:
            repo.upsert_scenario_run(
                {
                    "run_id": "nan-run",
                    "config": {"model": "qwen", "backend": "vllm", "weird": float("nan")},
                    "scores": {"final_score": float("inf"), "nested": [1.0, float("-inf")]},
                    "metadata": {"label": "canary-nightly", "ratio": float("nan")},
                }
            )
            repo.checkpoint_scenario_result(
                "nan-run",
                {"scenario_id": "TC-01", "status": "pass", "latency": float("nan")},
            )
            # SQLite json_valid() must accept metadata_json and config_json so search works.
            found = repo.list(search="canary-nightly")
            assert [r["run_id"] for r in found] == ["nan-run"]
            assert found[0]["scores"]["final_score"] is None
            assert found[0]["scores"]["nested"] == [1.0, None]
            cp = repo.get_checkpoints("nan-run")
            assert cp[0]["latency"] is None

    @pytest.mark.asyncio
    async def test_concurrent_async_checkpoints_and_main_thread_writes_do_not_race(
        self, tmp_path: Path
    ) -> None:
        with open_repository(db_path=str(tmp_path / "race.sqlite")) as repo:
            repo.upsert_scenario_run(
                {"run_id": "r1", "status": "running", "config": {"model": "m"}, "scores": {}}
            )
            errors: list[BaseException] = []

            def hammer_sync() -> None:
                try:
                    for i in range(25):
                        repo.mark_run_status("r1", "running" if i % 2 == 0 else "interrupted")
                        _ = repo.get("r1")
                        _ = repo.list(limit=5)
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            worker = threading.Thread(target=hammer_sync)
            worker.start()
            await asyncio.gather(
                *(
                    repo.acheckpoint_scenario_result(
                        "r1", {"scenario_id": f"TC-{i:02d}", "status": "pass", "points": 2}
                    )
                    for i in range(1, 26)
                )
            )
            worker.join()
            assert errors == []
            assert len(repo.get_checkpoints("r1")) == 25


# ---------------------------------------------------------------------------
# 2. Report Deletion Safety (application/run_deletion.py)
# ---------------------------------------------------------------------------


class TestRunDeletionSafety:
    RUN_ID = "2026-10-26T13-30-22.083000Z_bbbb2222"

    def test_symlink_report_file_is_never_unlinked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        victim = tmp_path / "precious.md"
        victim.write_text("do not delete", encoding="utf-8")

        report_dir = tmp_path / "runs" / "2026" / "10"
        report_dir.mkdir(parents=True)
        link = report_dir / f"{self.RUN_ID}.md"
        link.symlink_to(victim)

        assert not is_run_report(link, self.RUN_ID)

        with RunRepository() as repo:
            repo.upsert_scenario_run(
                {
                    "run_id": self.RUN_ID,
                    "status": "interrupted",
                    "config": {"model": "m"},
                    "scores": {},
                    "report_path": str(link),
                }
            )

        res = delete_runs([self.RUN_ID])
        assert res.deleted == (self.RUN_ID,)
        assert res.reports_removed == ()
        assert len(res.reports_kept) == 1
        assert res.reports_kept[0].reason == "not a report this tool wrote"
        assert victim.read_text(encoding="utf-8") == "do not delete"
        assert link.is_symlink()

    def test_symlink_month_dir_traversal_and_invalid_months_are_rejected(
        self, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / f"{self.RUN_ID}.md"
        target.write_text("x", encoding="utf-8")

        year_dir = tmp_path / "runs" / "2026"
        year_dir.mkdir(parents=True)
        month_link = year_dir / "10"
        month_link.symlink_to(outside)
        assert not is_run_report(month_link / f"{self.RUN_ID}.md", self.RUN_ID)

        bad_month = tmp_path / "runs" / "2026" / "13" / f"{self.RUN_ID}.md"
        bad_month.parent.mkdir(parents=True)
        bad_month.write_text("x", encoding="utf-8")
        assert not is_run_report(bad_month, self.RUN_ID)

        zero_month = tmp_path / "runs" / "2026" / "00" / f"{self.RUN_ID}.md"
        zero_month.parent.mkdir(parents=True)
        zero_month.write_text("x", encoding="utf-8")
        assert not is_run_report(zero_month, self.RUN_ID)

        valid = tmp_path / "runs" / "2026" / "12" / f"{self.RUN_ID}.md"
        valid.parent.mkdir(parents=True)
        valid.write_text("x", encoding="utf-8")
        assert is_run_report(valid, self.RUN_ID)
        assert not is_run_report(
            tmp_path / "runs" / "2026" / ".." / "2026" / "12" / f"{self.RUN_ID}.md",
            self.RUN_ID,
        )
        assert not is_run_report(valid, "../evil")
        assert not is_run_report(valid, "2026/evil")


# ---------------------------------------------------------------------------
# 3. Lifecycle, Resume & Leaderboard (application/service.py, resume.py, leaderboard.py)
# ---------------------------------------------------------------------------


class TestLifecycleResumeAndLeaderboard:
    @pytest.mark.asyncio
    async def test_empty_resume_without_prior_results_is_rejected(self, tmp_path: Path) -> None:
        with open_repository(db_path=str(tmp_path / "db.sqlite")) as repo:
            svc = BenchmarkService(repo=repo, reporter=MarkdownReporter(root=str(tmp_path)))
            with pytest.raises(ValueError, match="No scenarios to run"):
                await svc.run_benchmark(
                    model="m",
                    backend="vllm",
                    base_url="http://localhost:8000",
                    scenarios=[],
                    resume_run_id="r-empty",
                    resume_prior_results=[],
                )

    @pytest.mark.asyncio
    async def test_resume_deduplicates_prior_results_and_supports_hardmode_fallback(
        self, tmp_path: Path
    ) -> None:
        tc70 = next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == "TC-70")
        prior_old = ScenarioResult(
            scenario_id=tc70.id,
            status=ScenarioStatus.PARTIAL,
            points=1,
            summary="older checkpoint",
            expected_behavior="exp",
            raw_log="trace 1",
        ).to_dict()
        prior_latest = ScenarioResult(
            scenario_id=tc70.id,
            status=ScenarioStatus.PASS,
            points=2,
            summary="newer checkpoint",
            expected_behavior="exp",
            raw_log="trace 2",
        ).to_dict()

        with open_repository(db_path=str(tmp_path / "db.sqlite")) as repo:
            svc = BenchmarkService(repo=repo, reporter=MarkdownReporter(root=str(tmp_path)))
            run_data = await svc.run_benchmark(
                model="m",
                backend="vllm",
                base_url="http://localhost:8000",
                scenarios=[],
                resume_run_id="r-hm",
                resume_prior_results=[prior_old, prior_latest],
            )
        results = run_data["scores"]["scenario_results"]
        assert len(results) == 1
        assert results[0]["scenario_id"] == "TC-70"
        assert results[0]["points"] == 2
        assert results[0]["summary"] == "newer checkpoint"

    @pytest.mark.asyncio
    async def test_collect_metadata_safe_redacts_credentialed_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tool_eval_bench.utils import metadata as meta_mod

        async def boom(_cfg: Any) -> dict[str, Any]:
            req = httpx.Request("GET", f"{SECRET_BASE}/models?api_key=sk-super-secret")
            httpx.Response(503, request=req).raise_for_status()
            raise AssertionError("unreachable")

        monkeypatch.setattr(meta_mod, "collect_run_metadata", boom)
        meta = await _collect_metadata_safe("m", "vllm", SECRET_URL, None)
        assert "s3cretP@ss" not in meta["error"]
        assert "sk-super-secret" not in meta["error"]
        assert "internal-llm.corp" not in meta["error"]
        assert "http://***:8000/v1/models" in meta["error"]

    def test_stored_resume_guards_completed_bool_points_and_corrupt_numbers(self) -> None:
        assert (
            result_requires_rerun(
                {
                    "scenario_id": "TC-01",
                    "status": "pass",
                    "points": True,
                    "summary": "ok",
                    "raw_log": "trace",
                }
            )
            is True
        )

        scenarios = [s for s in ALL_SCENARIOS if s.id == "TC-01"]
        cfg = build_run_config(
            RunSettings(
                model="m",
                backend="vllm",
                base_url="http://localhost:8000",
                temperature=0.0,
                timeout_seconds=30.0,
                max_turns=6,
                seed=None,
                reference_date=None,
                concurrency=1,
                error_rate=0.0,
                alpha=0.7,
                extra_params=None,
                context_pressure_config=None,
                weight_by_difficulty=False,
            ),
            scenarios=scenarios,
            metadata={},
        )
        with pytest.raises(ResumeRejected, match="already completed"):
            plan_stored_resume(
                {"status": "completed", "config": cfg},
                [],
                base_url="http://localhost:8000",
            )
        with pytest.raises(ResumeRejected, match="temperature"):
            plan_stored_resume(
                {"status": "interrupted", "config": {**cfg, "temperature": float("nan")}},
                [],
                base_url="http://localhost:8000",
            )
        with pytest.raises(ResumeRejected, match="max_turns"):
            plan_stored_resume(
                {"status": "interrupted", "config": {**cfg, "max_turns": None}},
                [],
                base_url="http://localhost:8000",
            )

    def test_leaderboard_handles_nulls_and_rejects_bool_scenario_count(self) -> None:
        assert not is_rank_eligible(
            {"status": "completed"},
            {"scenario_count": True},
            {"completion_rate": 100, "scenario_results": [{"scenario_id": "TC-01"}]},
        )
        assert not is_rank_eligible(
            {"status": "completed"},
            {"scenario_count": 1},
            {"completion_rate": float("nan"), "scenario_results": [{"scenario_id": "TC-01"}]},
        )

        rows = extract_leaderboard_rows(
            [
                {
                    "run_id": "r1",
                    "model": "m1",
                    "status": "completed",
                    "created_at": None,
                    "config": {"backend": "vllm", "scenario_count": 1, "config_fingerprint": "fp1"},
                    "scores": {
                        "final_score": float("nan"),
                        "completion_rate": 100,
                        "scenario_results": [{"scenario_id": "TC-01", "status": "pass"}],
                    },
                    "metadata": {},
                }
            ]
        )
        assert len(rows) == 1
        assert rows[0]["final_score"] == 0
        assert rows[0]["date"] == "?"

    def test_cohort_ignores_operational_keys_while_fingerprint_and_resume_keep_them(
        self,
    ) -> None:
        assert {"concurrency", "timeout_seconds", "alpha"} <= COHORT_EXCLUDED_KEYS

        scenarios = [s for s in ALL_SCENARIOS if s.id == "TC-01"]
        metadata = {"tool_version": "0.1.0", "git_sha": "abc1234"}
        base_settings = RunSettings(
            model="model-a",
            backend="vllm",
            base_url="http://localhost:8000",
            temperature=0.0,
            timeout_seconds=30.0,
            max_turns=6,
            seed=None,
            reference_date=None,
            concurrency=1,
            error_rate=0.0,
            alpha=0.7,
            extra_params=None,
            context_pressure_config=None,
            weight_by_difficulty=False,
        )
        cfg_base = build_run_config(base_settings, scenarios=scenarios, metadata=metadata)
        cfg_parallel = build_run_config(
            replace(base_settings, concurrency=8), scenarios=scenarios, metadata=metadata
        )
        cfg_timeout = build_run_config(
            replace(base_settings, timeout_seconds=600.0), scenarios=scenarios, metadata=metadata
        )
        cfg_alpha = build_run_config(
            replace(base_settings, alpha=0.2), scenarios=scenarios, metadata=metadata
        )
        cfg_model_b_operational = build_run_config(
            replace(
                base_settings,
                model="model-b",
                concurrency=8,
                timeout_seconds=600.0,
                alpha=0.2,
            ),
            scenarios=scenarios,
            metadata=metadata,
        )

        # Positive: operational keys do not split cross-model leaderboard cohorts.
        base_cohort = cohort_fingerprint(cfg_base, metadata)
        assert cohort_fingerprint(cfg_parallel, metadata) == base_cohort
        assert cohort_fingerprint(cfg_timeout, metadata) == base_cohort
        assert cohort_fingerprint(cfg_alpha, metadata) == base_cohort
        assert cohort_fingerprint(cfg_model_b_operational, metadata) == base_cohort

        # Negative: per-model config_fingerprint still distinguishes operational keys.
        assert cfg_parallel["config_fingerprint"] != cfg_base["config_fingerprint"]
        assert cfg_timeout["config_fingerprint"] != cfg_base["config_fingerprint"]
        assert cfg_alpha["config_fingerprint"] != cfg_base["config_fingerprint"]

        # Negative: evaluation-affecting settings still split cohorts.
        for scoring_override in (
            {"temperature": 0.7},
            {"max_turns": 4},
            {"seed": 42},
            {"error_rate": 0.1},
            {"weight_by_difficulty": True},
            {"system_prompt": "Be concise."},
        ):
            cfg_scoring = build_run_config(
                replace(base_settings, **scoring_override),
                scenarios=scenarios,
                metadata=metadata,
            )
            assert cohort_fingerprint(cfg_scoring, metadata) != base_cohort

        # Negative: --resume mismatch detection still enforces operational keys.
        assert resume_mismatches(cfg_base, cfg_parallel, base_url=base_settings.base_url) == [
            "concurrency"
        ]
        assert resume_mismatches(cfg_base, cfg_timeout, base_url=base_settings.base_url) == [
            "timeout_seconds"
        ]
        assert resume_mismatches(cfg_base, cfg_alpha, base_url=base_settings.base_url) == ["alpha"]

        # Stored resume reconstructs and validates operational keys strictly.
        cfg_interrupted = build_run_config(
            replace(base_settings, concurrency=4, timeout_seconds=180.0, alpha=0.4),
            scenarios=scenarios,
            metadata=metadata,
        )
        plan = plan_stored_resume(
            {"status": "interrupted", "config": cfg_interrupted},
            [],
            base_url=base_settings.base_url,
        )
        assert plan.settings.concurrency == 4
        assert plan.settings.timeout_seconds == 180.0
        assert plan.settings.alpha == 0.4
        for bad_key, bad_val in (
            ("concurrency", True),
            ("timeout_seconds", float("nan")),
            ("alpha", None),
        ):
            with pytest.raises(ResumeRejected, match=bad_key):
                plan_stored_resume(
                    {"status": "interrupted", "config": {**cfg_base, bad_key: bad_val}},
                    [],
                    base_url=base_settings.base_url,
                )

        # End-to-end leaderboard: model-a and model-b with different operational
        # flags rank together in one cohort.
        rows = extract_leaderboard_rows(
            [
                {
                    "run_id": "run-a",
                    "model": "model-a",
                    "status": "completed",
                    "created_at": "2026-01-01T00:00:00Z",
                    "config": cfg_base,
                    "scores": {
                        "final_score": 82.0,
                        "completion_rate": 100,
                        "scenario_results": [{"scenario_id": "TC-01", "status": "pass"}],
                    },
                    "metadata": metadata,
                },
                {
                    "run_id": "run-b",
                    "model": "model-b",
                    "status": "completed",
                    "created_at": "2026-01-01T00:05:00Z",
                    "config": cfg_model_b_operational,
                    "scores": {
                        "final_score": 91.0,
                        "completion_rate": 100,
                        "scenario_results": [{"scenario_id": "TC-01", "status": "pass"}],
                    },
                    "metadata": metadata,
                },
            ]
        )
        assert [r["model"] for r in rows] == ["model-b", "model-a"]
        assert [r["cohort_rank"] for r in rows] == [1, 2]
        assert rows[0]["cohort_fingerprint"] == rows[1]["cohort_fingerprint"] == base_cohort


# ---------------------------------------------------------------------------
# 4. Reports, Held-Out Redaction, Escaping & Comparison Renderers
# ---------------------------------------------------------------------------


class TestReportAndComparisonFixes:
    def test_held_out_safety_warning_without_title_parens_is_still_redacted(
        self, tmp_path: Path
    ) -> None:
        summary = ModelScoreSummary(
            total_points=0,
            max_points=2,
            final_score=0,
            rating="Weak",
            safety_warnings=[
                "HOLD-01: leaked secret answer 42",
                "HOLD-02 (Secret Title): leaked second answer",
            ],
            category_scores=[
                CategoryScore(ScenarioCategory.A, "Tool Selection", 0, 2, 0.0, fail_count=1),
            ],
            scenario_results=[
                ScenarioResult(
                    scenario_id="HOLD-01",
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary="secret summary",
                    expected_behavior="secret expected",
                    raw_log="secret trace",
                )
            ],
        )
        meta = {
            "HOLD-01": ScenarioReportMetadata(
                "Secret Title 1", ScenarioCategory.A, difficulty=3, held_out=True
            ),
            "HOLD-02": ScenarioReportMetadata(
                "Secret Title 2", ScenarioCategory.A, difficulty=3, held_out=True
            ),
        }
        path = MarkdownReporter(root=str(tmp_path)).write_scenario_report(
            "r-held",
            "m",
            summary,
            scenario_metadata=meta,
        )
        text = path.read_text(encoding="utf-8")
        assert "leaked secret answer" not in text
        assert "Secret Title" not in text
        assert "secret summary" not in text
        assert "secret trace" not in text
        assert "> - HOLD-01: _held out_" in text
        assert "> - HOLD-02: _held out_" in text

    def test_pressure_sweep_withholds_held_out_scenarios_and_redacts_level_error(self) -> None:
        report = pressure_sweep_report(
            model="m",
            backend="vllm",
            server=SECRET_URL,
            context_size=8192,
            level_results=[
                {
                    "ratio": 0.5,
                    "fill_tokens": 4096,
                    "score_pct": 0.0,
                    "error": f"503 from {SECRET_URL}/chat/completions",
                    "scenario_results": [
                        {
                            "scenario_id": "HOLD-99",
                            "status": "fail",
                            "points": 0,
                            "summary": "top-secret-pack-summary",
                            "expected_behavior": "top-secret-expected",
                            "tool_calls_made": ["secret_tool"],
                            "raw_log": "top-secret-raw-trace",
                        }
                    ],
                }
            ],
            breaking_point=0.5,
            first_degradation=0.5,
            label=None,
            scenario_metadata={
                "HOLD-99": ScenarioReportMetadata(
                    "Pack Title", ScenarioCategory.A, difficulty=4, held_out=True
                )
            },
        )
        full = "\n".join([*report.header, *report.body])
        assert "s3cretP@ss" not in full
        assert "sk-super-secret" not in full
        assert "top-secret-pack-summary" not in full
        assert "top-secret-expected" not in full
        assert "secret_tool" not in full
        assert "top-secret-raw-trace" not in full
        assert "HOLD-99" in full

    def test_throughput_report_redacts_sample_error_urls(self) -> None:
        bad_sample = SimpleNamespace(
            error=f"HTTP 502 at {SECRET_BASE}/chat/completions?api_key=sk-super-secret"
        )
        rep = throughput_report(
            "m",
            [bad_sample],
            label=None,
            backend="vllm",
            server=SECRET_URL,
            served_model="m",
            model_root=None,
        )
        text = "\n".join([*rep.header, *rep.body])
        assert "s3cretP@ss" not in text
        assert "sk-super-secret" not in text
        assert "internal-llm.corp" not in text
        assert "http://***:8000/v1/chat/completions" in text

    def test_code_cell_and_run_context_escape_pipes_and_backticks_roundtrip(self) -> None:
        tricky = "org/model`v1|beta"
        rendered = f"| Model (API) | {_code_cell(tricky)} |"
        assert _tv("Model (API)", rendered, strip_bt=True) == tricky

        ctx = RunContext(
            tool_version="1.0",
            git_sha=None,
            hostname="h",
            platform_info="linux",
            python_version="3.13",
            model="m",
            backend="vllm",
            base_url="http://***:8000",
            extra_params={"stop": ["<|im_end|>", "```"]},
        )
        ctx_md = "\n".join(_render_run_context(ctx))
        extra_line = next(line for line in ctx_md.splitlines() if "Extra Params" in line)
        # Unescaped pipes split GFM cells; every pipe inside the value must be escaped.
        unescaped_pipes = len(extra_line.replace("\\|", "")) - len(
            extra_line.replace("\\|", "").replace("|", "")
        )
        assert unescaped_pipes == 3  # leading, middle, trailing table pipes only

    def test_summary_report_handles_empty_summaries_and_spec_decode_clamps_bars(
        self, tmp_path: Path
    ) -> None:
        out = write_summary_report(tmp_path, "empty-sum", "m", [], {"trials": 0})
        assert "Cross-Trial Summary" in out.read_text(encoding="utf-8")

        sample = SimpleNamespace(
            spec_method="mtp",
            runs=1,
            prompt_type="code",
            depth=0,
            effective_tg_tps=100.0,
            tg_tps=80.0,
            acceptance_rate=1.5,
            acceptance_rate_range=None,
            waste_ratio=None,
            acceptance_length=2.5,
            draft_window=4.0,
            draft_tps=None,
            speedup_ratio=1.25,
            ttft_ms=20.0,
            total_ms=200.0,
            tg_tokens=64,
        )
        spec_md = "\n".join(spec_decode_report("m", [sample], label=None, temperature=0.0).body)
        assert "| Avg Waste | — |" in spec_md
        assert "█" * 40 in spec_md

    def test_svg_primitives_strip_invalid_xml_chars_and_nan_coords(self) -> None:
        raw_attr = "hi\x00\x1fthere"
        fragment = (
            "<svg xmlns='http://www.w3.org/2000/svg'>"
            + svg_text(float("nan"), float("inf"), "bad\x00\x07name", size=12, fill="#000\x01")
            + f"<g {svg_attrs(data_label=raw_attr)}/>"
            + "</svg>"
        )
        assert svg_num(float("nan")) == "0"
        assert svg_num(float("-inf")) == "0"
        ET.fromstring(fragment)  # noqa: S314 - parsing locally rendered SVG fragment

    def test_compare_reports_parse_difficulty_tiers_pack_ids_and_escape_km(
        self, tmp_path: Path
    ) -> None:
        summary_a = ModelScoreSummary(
            total_points=2,
            max_points=2,
            final_score=100,
            rating="Excellent",
            category_scores=[
                CategoryScore(ScenarioCategory.A, "Tool Selection", 2, 2, 100.0, pass_count=1)
            ],
            scenario_results=[
                ScenarioResult(
                    scenario_id="HOLD-01",
                    status=ScenarioStatus.PASS,
                    points=2,
                    summary="secret",
                    expected_behavior="exp",
                    raw_log="trace",
                )
            ],
        )
        summary_b = ModelScoreSummary(
            total_points=0,
            max_points=2,
            final_score=0,
            rating="Poor",
            category_scores=[
                CategoryScore(ScenarioCategory.A, "Tool Selection", 0, 2, 0.0, fail_count=1)
            ],
            scenario_results=[
                ScenarioResult(
                    scenario_id="HOLD-01",
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary="secret",
                    expected_behavior="exp",
                    raw_log="trace",
                )
            ],
        )
        meta = {
            "HOLD-01": ScenarioReportMetadata(
                "Secret Title", ScenarioCategory.A, difficulty=4, held_out=True
            )
        }
        reporter = MarkdownReporter(root=str(tmp_path))
        pa = reporter.write_scenario_report("ra", "ModelA", summary_a, scenario_metadata=meta)
        pb = reporter.write_scenario_report("rb", "ModelB", summary_b, scenario_metadata=meta)

        da = parse_tool_eval_report(str(pa))
        db_ = parse_tool_eval_report(str(pb))
        assert [d["tier"] for d in da["difficulties"]] == ["Hard"]
        assert da["difficulties"][0]["rate"] == 100
        assert [s["id"] for s in da["scenarios"]] == ["HOLD-01"]
        assert da["scenarios"][0]["title"] == "_held out_"

        # Inject untrusted text into fallback median_turn_time to verify km() escaping.
        da["median_turn_time"] = "<script>alert(1)</script>"
        out_html = tmp_path / "cmp.html"
        generate_tool_compare_html(da, db_, str(out_html))
        html_text = out_html.read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in html_text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text
        assert "Secret Title" not in html_text

        sa = reporter.write_summary_report(
            "sa",
            "ModelA",
            [summary_b],
            {
                "trials": 1,
                "final_score_mean": 0.0,
                "final_score_stddev": 0.0,
                "total_points_mean": 0.0,
                "total_points_stddev": 0.0,
                "per_scenario": {"HOLD-01": {"pass_at_k": False, "pass_hat_k": False}},
            },
            scenario_metadata=meta,
        )
        parsed_sum = parse_summary_report(str(sa))
        assert [s["id"] for s in parsed_sum["scenarios"]] == ["HOLD-01"]
        assert parsed_sum["never_passes"] == [{"id": "HOLD-01", "issue": "_held out_"}]
        parsed_sum["median_turn_s"] = "<img src=x onerror=alert(1)>"
        sum_html = tmp_path / "sum_cmp.html"
        generate_summary_compare_html(parsed_sum, parsed_sum, str(sum_html))
        assert "<img src=x onerror=alert(1)>" not in sum_html.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. External Benchmark Plugins (plugins/)
# ---------------------------------------------------------------------------


class _StaticAdapter:
    def __init__(self, result: ChatCompletionResult | Exception) -> None:
        self._result = result

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class TestPluginFixes:
    def test_thinking_strip_and_unclosed_truncation_detection(self) -> None:
        assert plugin_answer_text is final_answer_text

        # Closed <think> blocks (including attributes, case, and non-mutation of raw content).
        assert strip_thinking("<think>draft 99, comma</think>\nFINAL ANSWER") == "FINAL ANSWER"
        assert strip_thinking('<THINK mode="fast">step 1</THINK>Answer: B') == "Answer: B"
        assert strip_thinking("<think>done</think>   ", fallback_to_thought=False) is None
        raw_closed = ChatCompletionResult(
            content="<think>thought</think>\n42", finish_reason="stop"
        )
        assert final_answer_text(raw_closed) == "42"
        assert raw_closed.content == "<think>thought</think>\n42"
        assert (
            final_answer_text(
                ChatCompletionResult(content="<think>done</think>\n42", finish_reason="length")
            )
            == "42"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(content="<think>#### 12</think>", finish_reason="stop")
            )
            == "#### 12"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(content="<think>#### 12</think>", finish_reason="length")
            )
            is None
        )

        # Multiple <think> blocks (consecutive leading, inline mid-answer, thought-only).
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>first</think>\n<think>second</think>\n42",
                    finish_reason="stop",
                )
            )
            == "42"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>plan</think>Result: <think>recheck</think>42",
                    finish_reason="stop",
                )
            )
            == "Result: 42"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>part 1</think>  <think>#### 7</think>\n",
                    finish_reason="stop",
                )
            )
            == "part 1\n#### 7"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>part 1</think>  <think>#### 7</think>\n",
                    finish_reason="length",
                )
            )
            is None
        )

        # Unclosed <think> blocks (length truncation vs stop vs visible prefix before <think>).
        assert strip_thinking("<think>unfinished reasoning", fallback_to_thought=False) is None
        assert strip_thinking("<think>unfinished reasoning", fallback_to_thought=True) == ""
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>cut off mid-thought 42", finish_reason="length"
                )
            )
            is None
        )
        assert (
            final_answer_text(
                ChatCompletionResult(content="<think>cut off mid-thought 42", finish_reason="stop")
            )
            == ""
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>done</think>\n<think>unfinished 99", finish_reason="length"
                )
            )
            is None
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>done</think>\n<think>unfinished 99", finish_reason="stop"
                )
            )
            == ""
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="Answer: C\n<think>post-answer scratchpad", finish_reason="length"
                )
            )
            == "Answer: C"
        )

        # reasoning_content (result.reasoning) fallback interaction.
        assert (
            final_answer_text(
                ChatCompletionResult(content="", reasoning="#### 4", finish_reason="stop")
            )
            == "#### 4"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(content="", reasoning="#### 4", finish_reason="length")
            )
            is None
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>   </think>", reasoning="#### 4", finish_reason="stop"
                )
            )
            == "#### 4"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>unfinished", reasoning="#### 4", finish_reason="stop"
                )
            )
            == "#### 4"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="<think>draft</think>42", reasoning="999", finish_reason="stop"
                )
            )
            == "42"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(
                    content="", reasoning="<think>inner</think>#### 8", finish_reason="stop"
                )
            )
            == "#### 8"
        )
        assert (
            final_answer_text(
                ChatCompletionResult(content="", reasoning=None, finish_reason="stop")
            )
            == ""
        )

        # Literal prose without <think> tags.
        for literal in (
            "The answer is 42.",
            "<thinking>keep me</thinking> 42",
            "if a < think and b > 0: return 42",
        ):
            assert strip_thinking(literal) == literal
            assert (
                final_answer_text(ChatCompletionResult(content=literal, finish_reason="stop"))
                == literal
            )

    @pytest.mark.parametrize(
        ("text", "expected_val", "expected_method"),
        [
            (r"Therefore we obtain \boxed{42}.", 42.0, "boxed"),
            (r"First \boxed{10}, wait actually \boxed{-1,250.5}", -1250.5, "boxed"),
            (r"Result: \boxed{\text{\$18}}", 18.0, "boxed"),
            ("#### **42**", 42.0, "marker"),
            ("#### 25%", 25.0, "marker"),
            ("The answer is $-18.", -18.0, "answer_pattern"),
            ("The answer is -$18.", -18.0, "answer_pattern"),
            ("Net balance: €1,250.50", 1250.5, "last_number"),
            (
                "<think>#### 999\nthe answer is 888</think>\nThe answer is **42**.",
                42.0,
                "answer_pattern",
            ),
        ],
    )
    def test_gsm8k_answer_extraction_edge_cases(
        self, text: str, expected_val: float, expected_method: str
    ) -> None:
        val, method = extract_gsm8k(text)
        assert val == pytest.approx(expected_val)
        assert method == expected_method
        assert eval_gsm8k(text, expected_val).correct is True

    @pytest.mark.parametrize(
        ("response", "expected_letter", "expected_method"),
        [
            ("**B**", "B", "exact"),
            ("**(D)**", "D", "exact"),
            ("`A`", "A", "exact"),
            (r"Hence \boxed{C}.", "C", "answer_pattern"),
            (r"Hence \boxed{\text{(b)}}.", "B", "answer_pattern"),
            ("The answer is **D**.", "D", "answer_pattern"),
            ("**Answer:** **A**", "A", "answer_pattern"),
            ("<think>Maybe A or C? Answer: A</think>\nB", "B", "exact"),
        ],
    )
    def test_mmlu_answer_extraction_and_ground_truth_bounds(
        self, response: str, expected_letter: str, expected_method: str
    ) -> None:
        assert extract_mmlu(response) == (expected_letter, expected_method)
        for bad_gt in (-1, 4, True):
            with pytest.raises(ValueError, match="ground_truth"):
                eval_mmlu("A", bad_gt)  # type: ignore[arg-type]

    def test_ifeval_checkers_and_inline_thinking_handling(self) -> None:
        # Non-word-char boundaries in forbidden_words
        assert not check_instruction(
            "keywords:forbidden_words", "We use C++ today.", {"forbidden_words": ["C++"]}
        )
        assert check_instruction(
            "keywords:forbidden_words", "Variable XC++ is synthetic.", {"forbidden_words": ["C++"]}
        )
        assert not check_instruction(
            "keywords:forbidden_words", "It costs $100 flat.", {"forbidden_words": ["$100"]}
        )
        assert check_instruction(
            "keywords:forbidden_words", "It costs $1000 flat.", {"forbidden_words": ["$100"]}
        )
        assert check_instruction(
            "keywords:forbidden_words", "Normal text.", {"forbidden_words": ["", "   "]}
        )

        # Single-line fenced JSON
        assert check_instruction(
            "detectable_format:json_format", '```json {"ok": true, "n": 1} ```', {}
        )

        # Malformed nth_paragraph type does not crash
        assert not check_instruction(
            "length_constraints:nth_paragraph_first_word",
            "Hello world",
            {"nth_paragraph": True, "first_word": "hello"},
        )
        assert not check_instruction(
            "length_constraints:nth_paragraph_first_word",
            "Hello world",
            {"nth_paragraph": "1", "first_word": "hello"},
        )

        # Inline <think> with commas and lowercase does not fail no_comma + english_capital
        res = evaluate_prompt(
            "<think>Wait, let me think carefully, step 1.</think>\nHELLO WORLD",
            ["punctuation:no_comma", "change_case:english_capital"],
            [{}, {}],
        )
        assert res.prompt_pass is True

    @pytest.mark.asyncio
    async def test_plugins_reject_negative_limit_and_handle_unclosed_think_truncation(
        self,
    ) -> None:
        adapter = _StaticAdapter(
            ChatCompletionResult(
                content="<think>I think the answer is 12, A, K7QM-2XPD-9WLR",
                finish_reason="length",
            )
        )
        for plugin in (GSM8KPlugin(), MMLUPlugin(), IFEvalPlugin()):
            with pytest.raises(ValueError, match="limit"):
                await plugin.run(adapter, model="m", base_url="http://x", limit=-1)

        g_res = await GSM8KPlugin().run(
            adapter,
            model="m",
            base_url="http://x",
            n_shots=0,
            _preloaded_items=[GSM8KItem(0, "Q?", "#### 12", 12.0)],
        )
        assert g_res.details["correct"] == 0
        assert g_res.details["truncated"] == 1

        m_res = await MMLUPlugin().run(
            adapter,
            model="m",
            base_url="http://x",
            n_shots=0,
            _preloaded_items={
                "test": [MMLUItem(0, "Q?", "anatomy", ["a", "b", "c", "d"], 0)],
                "dev": [],
            },
        )
        assert m_res.details["correct"] == 0
        assert m_res.details["truncated"] == 1

        i_res = await IFEvalPlugin().run(
            adapter,
            model="m",
            base_url="http://x",
            _preloaded_items=[IFEvalItem(1, "No commas.", ["punctuation:no_comma"], [{}])],
        )
        assert i_res.details["prompts_passed"] == 0
        assert i_res.details["truncated"] == 1

        n_case = NeedleCase(
            context_tokens=1024,
            depth_percent=0.5,
            needle="Passphrase is K7QM-2XPD-9WLR.",
            question="Passphrase?",
            answer="K7QM-2XPD-9WLR",
        )
        n_res = await NeedlePlugin().run(adapter, model="m", base_url="http://x", cases=[n_case])
        assert n_res.details["retrieved"] == 0

    @pytest.mark.asyncio
    async def test_decision_metrics_toolscale_redaction_and_hf_retry_after(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            reliability_bins([0.5], [True], n_bins=0)
        bins = reliability_bins([-0.2, float("nan"), 1.2], [False, False, True], n_bins=10)
        assert bins[0].low == 0.0 and bins[0].count == 2
        assert bins[-1].low == pytest.approx(0.9) and bins[-1].count == 1

        # URL redaction in toolscale and decision error rows
        leaky_exc = RuntimeError(f"503 from {SECRET_BASE}/chat/completions?api_key=sk-super-secret")
        case = load_decision_cases()[0]
        q_name = next(iter(case.questions))
        d_err = decision_error_row(case, q_name, leaky_exc)
        assert "s3cretP@ss" not in d_err["error"]
        assert "sk-super-secret" not in d_err["error"]
        assert "http://***:8000/v1/chat/completions" in d_err["error"]

        catalog = load_builtin_catalog()
        items = build_items(
            catalog, counts=[8], similarities=["distinct"], limit=1, trials=1, seed=1
        )[:1]
        ts_res = await ToolscalePlugin().run(
            _StaticAdapter(leaky_exc),
            model="m",
            base_url=SECRET_URL,
            items=items,
            catalog=catalog,
        )
        ts_report = "\n".join(ToolscalePlugin().render_report_section(ts_res))
        assert "s3cretP@ss" not in ts_report
        assert "sk-super-secret" not in ts_report

        ts_reply_msg = ChatCompletionResult(
            content="<think>missing optional filter</think>Which region should I query?",
            finish_reason="stop",
        )
        ts_abstain = await ToolscalePlugin().run(
            _StaticAdapter(ts_reply_msg),
            model="m",
            base_url="http://localhost:8000/v1",
            items=items,
            catalog=catalog,
        )
        assert ts_abstain.item_results[0]["reply"] == "Which region should I query?"
        assert (
            ts_reply_msg.content
            == "<think>missing optional filter</think>Which region should I query?"
        )

        # Retry-After parsing (numeric, HTTP-date, malformed, clamped)
        assert _parse_retry_after("4.5", 2.0) == pytest.approx(4.5)
        assert _parse_retry_after("9999", 2.0) == _MAX_BACKOFF_S
        assert _parse_retry_after("not-a-date", 2.0) == 2.0
        future = email.utils.format_datetime(
            datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True
        )
        parsed = _parse_retry_after(future, 2.0)
        assert 5.0 <= parsed <= 15.0
