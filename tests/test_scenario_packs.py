"""Held-out packs must stay secret while remaining verifiable.

A published benchmark leaks into training data, so official numbers need a
private scenario set. That creates two obligations that pull in opposite
directions: readers must be able to confirm two scores were measured against
the same set (attestation via content hash), and the report must not disclose
the set itself (redaction of titles, summaries, and traces).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tool_eval_bench.application.service import _build_run_config
from tool_eval_bench.cli.commands import resolve_pack_scenarios, resolve_packs, resolve_scenarios
from tool_eval_bench.domain.scenarios import (
    Category,
    ModelScoreSummary,
    ScenarioReportMetadata,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.evals.packs import load_scenario_pack, load_scenario_packs, pack_content_hash
from tool_eval_bench.storage.reports import MarkdownReporter

SECRET_TITLE = "Reconcile the quarterly ledger"
SECRET_PROMPT = "Find the duplicate invoice in ACME-4417 and refund it"


def _pack_yaml(scenario_id: str, *, title: str = SECRET_TITLE) -> str:
    return (
        f"id: {scenario_id}\n"
        f"title: {title}\n"
        "category: A\n"
        "difficulty: 3\n"
        f"user_message: {SECRET_PROMPT}\n"
        "expected_tool_calls: []\n"
        "tool_responses: {}\n"
    )


def _write_pack(root: Path, *ids: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for scenario_id in ids:
        (root / f"{scenario_id.lower()}.yaml").write_text(_pack_yaml(scenario_id))
    return root


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "scenarios": None,
        "categories": None,
        "short": False,
        "hardmode": False,
        "hardmode_only": False,
        "scenario_pack": None,
        "pack_only": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestPackLoading:
    def test_pack_scenarios_are_marked_held_out(self, tmp_path: Path) -> None:
        pack = load_scenario_pack(_write_pack(tmp_path / "private", "HO-1", "HO-2"))

        assert [s.id for s in pack.scenarios] == ["HO-1", "HO-2"]
        assert all(s.held_out for s in pack.scenarios)

    def test_missing_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_scenario_pack(tmp_path / "nope")

    def test_empty_pack_is_rejected(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no \\*.yaml"):
            load_scenario_pack(empty)

    def test_duplicate_ids_across_packs_are_rejected(self, tmp_path: Path) -> None:
        _write_pack(tmp_path / "one", "HO-1")
        _write_pack(tmp_path / "two", "HO-1")

        with pytest.raises(ValueError, match="appears in both"):
            load_scenario_packs([str(tmp_path / "one"), str(tmp_path / "two")])

    def test_pack_ids_colliding_with_public_suite_are_rejected(self, tmp_path: Path) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        _write_pack(tmp_path / "clash", ALL_SCENARIOS[0].id)

        with pytest.raises(ValueError, match="collide with the public suite"):
            resolve_scenarios(_args(scenario_pack=[str(tmp_path / "clash")]))


class TestPackContentHash:
    def test_hash_is_stable_for_identical_content(self, tmp_path: Path) -> None:
        first = _write_pack(tmp_path / "a", "HO-1")
        second = _write_pack(tmp_path / "b", "HO-1")

        assert pack_content_hash(first) == pack_content_hash(second)

    def test_editing_a_scenario_changes_the_hash(self, tmp_path: Path) -> None:
        pack = _write_pack(tmp_path / "a", "HO-1")
        before = pack_content_hash(pack)

        (pack / "ho-1.yaml").write_text(_pack_yaml("HO-1", title="Different task"))

        assert pack_content_hash(pack) != before

    def test_renaming_a_file_changes_the_hash(self, tmp_path: Path) -> None:
        pack = _write_pack(tmp_path / "a", "HO-1")
        before = pack_content_hash(pack)

        (pack / "ho-1.yaml").rename(pack / "renamed.yaml")

        assert pack_content_hash(pack) != before

    def test_hash_is_folded_into_config_fingerprint(self, tmp_path: Path) -> None:
        pack = load_scenario_pack(_write_pack(tmp_path / "private", "HO-1"))
        common: dict[str, object] = {
            "model": "m",
            "backend": "vllm",
            "base_url": "http://localhost:8000/v1",
            "scenarios": list(pack.scenarios),
            "temperature": 0.0,
            "timeout_seconds": 120.0,
            "max_turns": 8,
            "seed": None,
            "reference_date": None,
            "concurrency": 1,
            "error_rate": 0.0,
            "alpha": 0.7,
            "extra_params": None,
            "context_pressure_config": None,
            "weight_by_difficulty": False,
            "metadata": {},
        }
        attested = _build_run_config(**common, scenario_packs=[pack.to_dict()])  # type: ignore[arg-type]
        edited = dict(pack.to_dict())
        edited["content_hash"] = "0" * 16
        rehashed = _build_run_config(**common, scenario_packs=[edited])  # type: ignore[arg-type]

        assert attested["config_fingerprint"] != rehashed["config_fingerprint"]
        assert attested["scenario_packs"][0]["content_hash"] == pack.content_hash

    def test_attestation_records_no_scenario_content(self, tmp_path: Path) -> None:
        pack = load_scenario_pack(_write_pack(tmp_path / "private", "HO-1"))

        record = repr(pack.to_dict())

        assert SECRET_TITLE not in record
        assert SECRET_PROMPT not in record


class TestPackSelection:
    def test_packs_extend_the_public_suite_by_default(self, tmp_path: Path) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS

        resolved = resolve_scenarios(
            _args(scenario_pack=[str(_write_pack(tmp_path / "p", "HO-1"))])
        )

        assert len(resolved) == len(ALL_SCENARIOS) + 1
        assert "HO-1" in {s.id for s in resolved}

    def test_pack_only_skips_the_public_suite(self, tmp_path: Path) -> None:
        resolved = resolve_scenarios(
            _args(scenario_pack=[str(_write_pack(tmp_path / "p", "HO-1", "HO-2"))], pack_only=True)
        )

        assert [s.id for s in resolved] == ["HO-1", "HO-2"]

    def test_pack_only_without_a_pack_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires at least one"):
            resolve_scenarios(_args(pack_only=True))

    def test_packs_are_hashed_once_per_run(self, tmp_path: Path) -> None:
        args = _args(scenario_pack=[str(_write_pack(tmp_path / "p", "HO-1"))])
        first = resolve_packs(args)

        # A pack edited mid-run must not make the attestation disagree with the
        # scenarios that were actually executed.
        (tmp_path / "p" / "ho-1.yaml").write_text(_pack_yaml("HO-1", title="Swapped"))

        assert resolve_packs(args) is first
        assert [s.title for s in resolve_pack_scenarios(args)] == [SECRET_TITLE]

    def test_no_pack_flag_loads_nothing(self) -> None:
        assert resolve_packs(_args()) == []

    def test_dry_run_reports_a_bad_pack_without_a_traceback(self, capsys) -> None:
        from rich.console import Console

        from tool_eval_bench.cli.local_commands import _render_dry_run

        args = _args(pack_only=True, json=False)
        with pytest.raises(SystemExit) as exit_info:
            _render_dry_run(args, Console(), resolve_scenarios)

        assert exit_info.value.code == 2
        assert "requires at least one" in capsys.readouterr().out


class TestReportRedaction:
    def _summary(self) -> ModelScoreSummary:
        return ModelScoreSummary(
            scenario_results=[
                ScenarioResult(
                    scenario_id="PUB-1",
                    status=ScenarioStatus.PASS,
                    points=2,
                    summary="Called get_weather correctly",
                    raw_log="USER: what is the weather",
                ),
                ScenarioResult(
                    scenario_id="HO-1",
                    status=ScenarioStatus.FAIL,
                    points=0,
                    summary=f"Never called refund_invoice for {SECRET_PROMPT}",
                    raw_log=f"USER: {SECRET_PROMPT}",
                    note=SECRET_PROMPT,
                ),
            ],
            category_scores=[],
            total_points=2,
            max_points=4,
            final_score=50,
            rating="C",
        )

    def _write(self, tmp_path: Path, packs: list[dict[str, object]] | None) -> str:
        reporter = MarkdownReporter(root=str(tmp_path / "runs"))
        path = reporter.write_scenario_report(
            "run-1",
            "m",
            self._summary(),
            scenario_metadata={
                "PUB-1": ScenarioReportMetadata(
                    title="Public weather lookup", category=Category.A, difficulty=1
                ),
                "HO-1": ScenarioReportMetadata(
                    title=SECRET_TITLE, category=Category.A, difficulty=3, held_out=True
                ),
            },
            scenario_packs=packs,
        )
        return path.read_text(encoding="utf-8")

    def test_held_out_prompt_title_and_trace_are_withheld(self, tmp_path: Path) -> None:
        report = self._write(tmp_path, None)

        assert SECRET_TITLE not in report
        assert SECRET_PROMPT not in report

    def test_held_out_scores_remain_visible(self, tmp_path: Path) -> None:
        report = self._write(tmp_path, None)

        assert "HO-1" in report
        assert "held out" in report
        assert "**50** / 100" in report

    def test_public_scenarios_keep_full_traces(self, tmp_path: Path) -> None:
        report = self._write(tmp_path, None)

        assert "Public weather lookup" in report
        assert "USER: what is the weather" in report

    def test_report_attests_to_the_pack_hash(self, tmp_path: Path) -> None:
        report = self._write(
            tmp_path,
            [{"name": "private", "scenario_count": 1, "content_hash": "deadbeefdeadbeef"}],
        )

        assert "deadbeefdeadbeef" in report
        assert "pack `private`" in report
