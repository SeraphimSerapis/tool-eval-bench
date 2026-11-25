"""Tests for rubric versioning, CI AST fingerprint gate, cohorts, and comparability warnings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.application.comparison import (
    compare_comparability_warnings,
    compute_paired_comparison,
    extract_result_maps,
)
from tool_eval_bench.application.leaderboard import (
    cohort_fingerprint,
    comparability_warnings,
    extract_leaderboard_rows,
    is_rank_eligible,
)
from tool_eval_bench.application.run_config import cohort_config
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.rubric import (
    EVAL_RUBRIC_VERSION,
    LOCALE_VERSION,
    NOISE_VERSION,
    RUBRIC_AST_FINGERPRINT,
    compute_rubric_ast_fingerprint,
)
from tool_eval_bench.utils.ids import build_config_fingerprint
from tool_eval_bench.utils.metadata import collect_run_context

# ---------------------------------------------------------------------------
# CI AST fingerprint gate
# ---------------------------------------------------------------------------


def test_rubric_ast_fingerprint_matches_source() -> None:
    """CI gate: detect scoring/evaluator/orchestrator logic edits without version review."""
    actual = compute_rubric_ast_fingerprint()
    assert len(actual) == 16
    assert actual == RUBRIC_AST_FINGERPRINT, (
        f"Rubric AST fingerprint changed: expected {RUBRIC_AST_FINGERPRINT!r}, got {actual!r}.\n"
        "A Python file under src/tool_eval_bench/evals/ or "
        "src/tool_eval_bench/runner/orchestrator.py changed in a way that altered its AST.\n"
        "- If this change alters scoring verdicts, localized prompts, or noise injection, bump "
        "EVAL_RUBRIC_VERSION, LOCALE_VERSION, or NOISE_VERSION in "
        "src/tool_eval_bench/domain/rubric.py and set RUBRIC_AST_FINGERPRINT = "
        f"{actual!r}.\n"
        "- If this is a strictly verdict-preserving refactor, update RUBRIC_AST_FINGERPRINT = "
        f"{actual!r} only and explain why verdicts are unchanged in the PR description."
    )


# ---------------------------------------------------------------------------
# AST fingerprint normalization tests
# ---------------------------------------------------------------------------


def _write_synthetic_package(root: Path, eval_src: str, orch_src: str) -> Path:
    eval_file = root / "evals" / "scenarios" / "sample.py"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(eval_src, encoding="utf-8")

    orch_file = root / "runner" / "orchestrator.py"
    orch_file.parent.mkdir(parents=True, exist_ok=True)
    orch_file.write_text(orch_src, encoding="utf-8")
    return root


class TestRubricAstFingerprint:
    def test_docstring_comment_and_whitespace_edits_do_not_change_fingerprint(
        self, tmp_path: Path
    ) -> None:
        pkg_a = _write_synthetic_package(
            tmp_path / "a",
            eval_src=(
                '"""Original module docstring."""\n'
                "\n"
                "class Scorer:\n"
                '    """Original class docstring."""\n'
                "\n"
                "    def grade(self, x: int) -> int:\n"
                '        """Original method docstring."""\n'
                "        # original comment\n"
                "        return x + 1\n"
                "\n"
                "async def run_check(y: str) -> str:\n"
                '    """Original async docstring."""\n'
                "    return y\n"
            ),
            orch_src='"""Orch doc."""\ndef loop() -> None:\n    return None\n',
        )
        pkg_b = _write_synthetic_package(
            tmp_path / "b",
            eval_src=(
                '"""Completely rewritten module docstring.\n\nWith extra lines."""\n'
                "# extra top-level comment\n"
                "class Scorer:\n"
                "    # docstring removed completely from class\n"
                "    def grade(self,   x: int) -> int:\n"
                '        """Updated method docstring."""\n'
                "        return (x + 1)  # trailing comment\n"
                "\n\n\n"
                "async def run_check(y: str) -> str:\n"
                "    return y\n"
            ),
            orch_src="# no docstring\ndef loop() -> None:\n    return None\n",
        )

        fp_a = compute_rubric_ast_fingerprint(pkg_a)
        fp_b = compute_rubric_ast_fingerprint(pkg_b)
        assert fp_a == fp_b
        assert len(fp_a) == 16

    def test_logic_edit_in_evals_or_orchestrator_changes_fingerprint(self, tmp_path: Path) -> None:
        base_eval = "def grade(x: int) -> int:\n    return x + 1\n"
        base_orch = "def loop() -> int:\n    return 0\n"

        base_root = _write_synthetic_package(tmp_path / "base", base_eval, base_orch)
        eval_changed = _write_synthetic_package(
            tmp_path / "eval_changed",
            "def grade(x: int) -> int:\n    return x + 2\n",
            base_orch,
        )
        orch_changed = _write_synthetic_package(
            tmp_path / "orch_changed",
            base_eval,
            "def loop() -> int:\n    return 1\n",
        )
        none_vs_zero = _write_synthetic_package(
            tmp_path / "none_vs_zero",
            base_eval,
            "def loop() -> int | None:\n    return None\n",
        )

        fp_base = compute_rubric_ast_fingerprint(base_root)
        assert compute_rubric_ast_fingerprint(eval_changed) != fp_base
        assert compute_rubric_ast_fingerprint(orch_changed) != fp_base
        assert compute_rubric_ast_fingerprint(none_vs_zero) != fp_base

    def test_non_docstring_string_literal_changes_fingerprint(self, tmp_path: Path) -> None:
        pkg_a = _write_synthetic_package(
            tmp_path / "a",
            'def prompt() -> str:\n    """doc"""\n    return "alpha"\n',
            "x = 1\n",
        )
        pkg_b = _write_synthetic_package(
            tmp_path / "b",
            'def prompt() -> str:\n    """doc"""\n    return "beta"\n',
            "x = 1\n",
        )
        assert compute_rubric_ast_fingerprint(pkg_a) != compute_rubric_ast_fingerprint(pkg_b)

    def test_missing_required_paths_raise_file_not_found(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="evals"):
            compute_rubric_ast_fingerprint(empty)

        evals_only = tmp_path / "evals_only"
        (evals_only / "evals").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="orchestrator"):
            compute_rubric_ast_fingerprint(evals_only)


# ---------------------------------------------------------------------------
# RunContext & metadata collection
# ---------------------------------------------------------------------------


class TestRubricRunContext:
    def test_run_context_rubric_fields_default_none_and_preserve_positional_args(self) -> None:
        ctx = RunContext(
            "2.9.0",
            "abc1234",
            "host",
            "Linux",
            "3.13",
            "model-a",
            "vllm",
            "http://***:8000",
            0.0,
            8,
            120.0,
            None,
            "all",
            1,
            1,
            0.0,
            True,
            None,
            None,
            None,
            None,
            None,
            "vLLM",
            "0.8.5",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "sess-1",
        )
        assert ctx.session_id == "sess-1"
        assert ctx.eval_rubric_version is None
        assert ctx.locale_version is None
        assert ctx.noise_version is None
        serialized = ctx.to_dict()
        assert "eval_rubric_version" not in serialized
        assert "locale_version" not in serialized
        assert "noise_version" not in serialized

    @pytest.mark.asyncio
    async def test_collect_run_context_populates_rubric_versions(self) -> None:
        ctx = await collect_run_context(
            model="qwen3-32b",
            backend="vllm",
            base_url="http://localhost:8000",
            probe_engine=False,
        )
        assert ctx.eval_rubric_version == EVAL_RUBRIC_VERSION
        assert ctx.locale_version == LOCALE_VERSION
        assert ctx.noise_version == NOISE_VERSION
        serialized = ctx.to_dict()
        assert serialized["eval_rubric_version"] == EVAL_RUBRIC_VERSION
        assert serialized["locale_version"] == LOCALE_VERSION
        assert serialized["noise_version"] == NOISE_VERSION


# ---------------------------------------------------------------------------
# Rubric-versioned cohort_fingerprint
# ---------------------------------------------------------------------------


class TestRubricCohortFingerprint:
    @staticmethod
    def _base_config() -> dict[str, Any]:
        return {
            "model": "qwen3-32b",
            "backend": "vllm",
            "temperature": 0.0,
            "max_turns": 8,
            "scenario_count": 2,
            "scenario_ids": ["TC-01", "TC-02"],
            "seed": 42,
            "error_rate": 0.0,
            "weight_by_difficulty": False,
        }

    def test_stable_across_unrelated_commits_and_tool_versions(self) -> None:
        cfg = self._base_config()
        meta_a = {
            "tool_version": "2.9.0",
            "git_sha": "1111111",
            "eval_rubric_version": "1",
            "locale_version": "1",
            "noise_version": "1",
        }
        meta_b = {
            "tool_version": "2.10.0",
            "git_sha": "2222222",
            "eval_rubric_version": "1",
            "locale_version": "1",
            "noise_version": "1",
        }
        assert cohort_fingerprint(cfg, meta_a) == cohort_fingerprint(cfg, meta_b)

    @pytest.mark.parametrize(
        "bumped_field",
        ["eval_rubric_version", "locale_version", "noise_version"],
    )
    def test_splits_when_any_rubric_version_bumps(self, bumped_field: str) -> None:
        cfg = self._base_config()
        base_meta = {
            "tool_version": "2.9.0",
            "git_sha": "1111111",
            "eval_rubric_version": "1",
            "locale_version": "1",
            "noise_version": "1",
        }
        bumped_meta = {**base_meta, bumped_field: "2"}
        assert cohort_fingerprint(cfg, base_meta) != cohort_fingerprint(cfg, bumped_meta)

    def test_legacy_fallback_without_eval_rubric_version(self) -> None:
        cfg = self._base_config()
        legacy_a = {"tool_version": "2.8.0", "git_sha": "1111111"}
        legacy_b = {"tool_version": "2.8.0", "git_sha": "2222222"}
        modern = {
            **legacy_a,
            "eval_rubric_version": "1",
            "locale_version": "1",
            "noise_version": "1",
        }

        expected_legacy = build_config_fingerprint(
            {
                "config": cohort_config(dict(cfg)),
                "tool_version": "2.8.0",
                "git_sha": "1111111",
            }
        )
        assert cohort_fingerprint(cfg, legacy_a) == expected_legacy
        assert cohort_fingerprint(cfg, legacy_a) != cohort_fingerprint(cfg, legacy_b)
        assert cohort_fingerprint(cfg, legacy_a) != cohort_fingerprint(cfg, modern)


# ---------------------------------------------------------------------------
# Withheld eligibility & comparability_warnings
# ---------------------------------------------------------------------------


def _clean_run(**overrides: Any) -> dict[str, Any]:
    run: dict[str, Any] = {
        "run_id": "20261125T100000Z_abc",
        "model": "qwen3-32b",
        "status": "completed",
        "run_type": "tool_eval",
        "created_at": "2026-11-25T10:00:00Z",
        "config": {
            "model": "qwen3-32b",
            "backend": "vllm",
            "scenario_count": 2,
            "scenario_ids": ["TC-01", "TC-02"],
            "config_fingerprint": "fp-clean",
        },
        "metadata": {
            "tool_version": "2.9.0",
            "git_sha": "abc1234",
            "engine_name": "vLLM",
            "eval_rubric_version": "1",
            "locale_version": "1",
            "noise_version": "1",
        },
        "scores": {
            "final_score": 85,
            "rating": "★★★★ Good",
            "total_points": 3,
            "max_points": 4,
            "completion_rate": 100,
            "scenario_results": [
                {"scenario_id": "TC-01", "status": "pass", "points": 2},
                {"scenario_id": "TC-02", "status": "partial", "points": 1},
            ],
        },
    }
    run.update(overrides)
    return run


class TestRankEligibilityAndWarnings:
    def test_is_rank_eligible_rejects_withheld_final_score(self) -> None:
        withheld = _clean_run()
        withheld["scores"]["final_score"] = None
        withheld["scores"]["rating"] = "Withheld"

        assert not is_rank_eligible(withheld, withheld["config"], withheld["scores"])
        assert extract_leaderboard_rows([withheld]) == []

    def test_is_rank_eligible_accepts_zero_score_and_clean_run(self) -> None:
        zero_run = _clean_run()
        zero_run["scores"]["final_score"] = 0
        assert is_rank_eligible(zero_run, zero_run["config"], zero_run["scores"])

    def test_clean_run_has_no_comparability_warnings(self) -> None:
        run = _clean_run()
        assert comparability_warnings(run) == []
        rows = extract_leaderboard_rows([run])
        assert len(rows) == 1
        assert rows[0]["comparability_warnings"] == []

    def test_warns_on_withheld_score(self) -> None:
        run = _clean_run()
        run["scores"]["final_score"] = None
        assert comparability_warnings(run) == ["Score withheld (no gradable scenarios)"]

        # Flat web-style summary dict with final_score=None
        assert comparability_warnings({"final_score": None}) == [
            "Score withheld (no gradable scenarios)"
        ]
        # Zero score is NOT withheld
        run["scores"]["final_score"] = 0
        assert comparability_warnings(run) == []

    def test_warns_on_incomplete_completion_rate(self) -> None:
        run = _clean_run()
        run["scores"]["completion_rate"] = 85.5
        assert comparability_warnings(run) == [
            "Incomplete run (85.5% completion; infrastructure failures excluded)"
        ]

        # Non-finite or bool values do not crash or falsely warn
        run["scores"]["completion_rate"] = float("nan")
        assert comparability_warnings(run) == []
        run["scores"]["completion_rate"] = True
        assert comparability_warnings(run) == []

    def test_warns_on_dirty_git_tree(self) -> None:
        run = _clean_run()
        run["metadata"]["git_sha"] = "abc1234-dirty"
        assert comparability_warnings(run) == ["Run from dirty git working tree (abc1234-dirty)"]

    def test_warns_on_positive_subset_limit(self) -> None:
        run = _clean_run()
        run["config"]["limit"] = 10
        assert comparability_warnings(run) == ["Evaluated subset only (limit=10)"]

        for non_limit in (0, -5, None, False, True):
            run["config"]["limit"] = non_limit
            assert comparability_warnings(run) == []

    @pytest.mark.parametrize(
        "backend",
        ["vllm", "sglang", "llamacpp", "llama.cpp", "litellm", "openai", "local"],
    )
    def test_warns_when_serving_engine_unprobed_for_self_hosted_backends(
        self, backend: str
    ) -> None:
        run = _clean_run()
        run["config"]["backend"] = backend
        run["metadata"].pop("engine_name")
        assert comparability_warnings(run) == ["Serving engine was not probed"]

    def test_no_probe_engine_flag_warns_even_with_hosted_backend(self) -> None:
        run = _clean_run()
        run["config"]["backend"] = "gemini"
        run["config"]["no_probe_engine"] = True
        assert comparability_warnings(run) == ["Serving engine was not probed"]

        # Hosted backend without engine_name and without no_probe_engine does not warn
        run["config"].pop("no_probe_engine")
        run["metadata"].pop("engine_name")
        assert comparability_warnings(run) == []

    def test_warns_on_response_format_fallback_and_ignores_normal_diagnostics(self) -> None:
        run = _clean_run()
        # Normal orchestrator diagnostics must NOT trigger the warning
        run["scores"]["scenario_results"] = [
            {
                "scenario_id": "TC-64",
                "status": "pass",
                "points": 2,
                "diagnostics": {
                    "response_format": (
                        "requested; backend enforcement is not independently measured"
                    )
                },
            },
            {
                "scenario_id": "TC-65",
                "status": "pass",
                "points": 2,
                "diagnostics": {
                    "response_format": (
                        "withheld on tool-enabled turns; schema enforced by the prompt"
                    )
                },
            },
        ]
        assert comparability_warnings(run) == []

        # Orchestrator fallback message triggers once even across multiple scenarios
        fallback_diag = (
            "endpoint rejected it; fell back json_schema -> json_object, "
            "schema enforced only by the prompt"
        )
        run["scores"]["scenario_results"][0]["diagnostics"]["response_format"] = fallback_diag
        run["scores"]["scenario_results"][1]["diagnostics"]["response_format"] = fallback_diag
        assert comparability_warnings(run) == ["Structured output fell back from json_schema"]

        # Explicit boolean diagnostic key also triggers
        run["scores"]["scenario_results"] = [
            {
                "scenario_id": "TC-64",
                "status": "pass",
                "points": 2,
                "diagnostics": {"response_format_fallback": True},
            }
        ]
        assert comparability_warnings(run) == ["Structured output fell back from json_schema"]

    def test_extract_leaderboard_rows_includes_comparability_warnings(self) -> None:
        run = _clean_run()
        run["metadata"]["git_sha"] = "deadbee-dirty"
        run["config"]["limit"] = 2
        rows = extract_leaderboard_rows([run])
        assert len(rows) == 1
        assert rows[0]["comparability_warnings"] == [
            "Run from dirty git working tree (deadbee-dirty)",
            "Evaluated subset only (limit=2)",
        ]


# ---------------------------------------------------------------------------
# Comparison comparability warnings
# ---------------------------------------------------------------------------


class TestComparisonComparabilityWarnings:
    def test_matching_rubric_versions_across_commits_produce_no_warnings(self) -> None:
        run_a = _clean_run(run_id="run_a")
        run_b = _clean_run(run_id="run_b")
        run_b["metadata"]["git_sha"] = "9999999"
        run_b["metadata"]["tool_version"] = "2.10.0"

        assert compare_comparability_warnings(run_a, run_b) == []

        map_a, map_b, all_ids = extract_result_maps(
            run_a["scores"]["scenario_results"],
            run_b["scores"]["scenario_results"],
        )
        paired = compute_paired_comparison(map_a, map_b, all_ids, run_a=run_a, run_b=run_b)
        assert paired["comparability_warnings"] == []

    def test_surfaces_per_run_warnings_and_rubric_mismatches(self) -> None:
        run_a = _clean_run(run_id="run_a")
        run_a["metadata"]["git_sha"] = "aaa1111-dirty"

        run_b = _clean_run(run_id="run_b")
        run_b["metadata"]["eval_rubric_version"] = "2"
        run_b["metadata"]["locale_version"] = "2"
        run_b["metadata"]["noise_version"] = "2"
        run_b["config"]["limit"] = 15

        warnings = compare_comparability_warnings(run_a, run_b)
        assert warnings == [
            "Run A: Run from dirty git working tree (aaa1111-dirty)",
            "Run B: Evaluated subset only (limit=15)",
            "Eval rubric version mismatch (1 vs 2)",
            "Locale version mismatch (1 vs 2)",
            "Noise version mismatch (1 vs 2)",
        ]

        map_a, map_b, all_ids = extract_result_maps(
            run_a["scores"]["scenario_results"],
            run_b["scores"]["scenario_results"],
        )
        paired = compute_paired_comparison(map_a, map_b, all_ids, run_a=run_a, run_b=run_b)
        assert paired["comparability_warnings"] == warnings

    def test_modern_vs_legacy_and_legacy_vs_legacy_version_mismatches(self) -> None:
        modern = _clean_run(run_id="modern")
        legacy_a = _clean_run(run_id="legacy_a")
        legacy_b = _clean_run(run_id="legacy_b")
        for key in ("eval_rubric_version", "locale_version", "noise_version"):
            legacy_a["metadata"].pop(key)
            legacy_b["metadata"].pop(key)
        legacy_b["metadata"]["tool_version"] = "2.7.0"
        legacy_b["metadata"]["git_sha"] = "0000000"

        assert compare_comparability_warnings(modern, legacy_a) == [
            "Eval rubric version mismatch (1 vs unversioned)",
            "Locale version mismatch (1 vs unversioned)",
            "Noise version mismatch (1 vs unversioned)",
        ]
        assert compare_comparability_warnings(legacy_a, legacy_b) == [
            "Tool version mismatch (2.9.0 vs 2.7.0)",
            "Git SHA mismatch (abc1234 vs 0000000)",
        ]

    def test_paired_comparison_without_run_objects_checks_map_diagnostics(self) -> None:
        map_a = {
            "TC-64": {
                "scenario_id": "TC-64",
                "status": "pass",
                "points": 2,
                "diagnostics": {
                    "response_format": (
                        "endpoint rejected it; fell back json_schema -> json_object, "
                        "schema enforced only by the prompt"
                    )
                },
            }
        }
        map_b = {"TC-64": {"scenario_id": "TC-64", "status": "pass", "points": 2}}

        paired = compute_paired_comparison(map_a, map_b, ["TC-64"])
        assert paired["comparability_warnings"] == [
            "Run A: Structured output fell back from json_schema"
        ]

        # Single-sided run_a without run_b does not emit cross-run mismatch warnings
        single = compute_paired_comparison(
            map_a,
            map_b,
            ["TC-64"],
            run_a={"config": {"backend": "gemini"}, "metadata": {"eval_rubric_version": "1"}},
        )
        assert single["comparability_warnings"] == [
            "Run A: Structured output fell back from json_schema"
        ]
