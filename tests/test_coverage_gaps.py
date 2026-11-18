"""Non-duplicate storage and speculative report regression tests."""

from __future__ import annotations

from tests.conftest import open_repository
from tool_eval_bench.runner.speculative import SpecDecodeSample
from tool_eval_bench.storage.reports import MarkdownReporter


class TestRunRepositoryExtended:
    def test_get_nonexistent(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        assert repo.get("no_such_id") is None
        repo.close()

    def test_get_latest_empty(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        assert repo.get_latest() is None
        repo.close()

    def test_get_latest_with_data(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run(
            {"run_id": "r1", "config": {"model": "m"}, "scores": {}, "metadata": {}}
        )
        latest = repo.get_latest()
        assert latest is not None and latest["run_id"] == "r1"
        repo.close()

    def test_get_latest_filtered_by_model(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run(
            {"run_id": "r1", "config": {"model": "a"}, "scores": {}, "metadata": {}}
        )
        repo.upsert_scenario_run(
            {"run_id": "r2", "config": {"model": "b"}, "scores": {}, "metadata": {}}
        )
        assert repo.get_latest(model="a")["run_id"] == "r1"
        assert repo.get_latest(model="c") is None
        repo.close()

    def test_get_scenario_results(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run(
            {
                "run_id": "r1",
                "config": {"model": "m"},
                "scores": {"scenario_results": [{"id": "TC-01", "status": "pass"}]},
                "metadata": {},
            }
        )
        results = repo.get_scenario_results("r1")
        assert results is not None and len(results) == 1
        repo.close()

    def test_get_scenario_results_missing(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        assert repo.get_scenario_results("missing") is None
        repo.close()

    def test_get_scenario_results_no_scores(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run({"run_id": "r1", "config": {"model": "m"}, "metadata": {}})
        assert repo.get_scenario_results("r1") is None
        repo.close()

    def test_upsert_updates_existing(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run(
            {"run_id": "r1", "config": {"model": "m"}, "scores": {"v": 1}, "metadata": {}}
        )
        repo.upsert_scenario_run(
            {"run_id": "r1", "config": {"model": "m"}, "scores": {"v": 2}, "metadata": {}}
        )
        assert repo.get("r1")["scores"]["v"] == 2
        repo.close()

    def test_list_filter_by_model(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        repo.upsert_scenario_run(
            {"run_id": "r1", "config": {"model": "a"}, "scores": {}, "metadata": {}}
        )
        repo.upsert_scenario_run(
            {"run_id": "r2", "config": {"model": "b"}, "scores": {}, "metadata": {}}
        )
        assert len(repo.list(model="a")) == 1
        repo.close()

    def test_del_safety_net(self, tmp_path):
        repo = open_repository(db_path=str(tmp_path / "test.sqlite"))
        del repo  # should not raise


class TestSpecDecodeReport:
    def test_basic_spec_report(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [
            SpecDecodeSample(
                tg_tokens=128,
                total_ms=2000,
                ttft_ms=200,
                tg_tps=50.0,
                spec_method="mtp",
                prompt_type="code",
            ),
            SpecDecodeSample(
                tg_tokens=128,
                total_ms=3000,
                ttft_ms=300,
                tg_tps=40.0,
                spec_method="mtp",
                prompt_type="structured",
            ),
        ]
        path = reporter.write_spec_decode_report("spec_001", "test-model", samples)
        content = path.read_text(encoding="utf-8")
        assert "Speculative Decoding Benchmark" in content
        assert "mtp" in content
        assert "Interpretation Guide" in content

    def test_spec_report_with_acceptance_rate(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [
            SpecDecodeSample(
                tg_tokens=128,
                total_ms=2000,
                ttft_ms=200,
                tg_tps=50.0,
                spec_method="mtp",
                prompt_type="code",
                acceptance_rate=0.85,
                acceptance_length=3.2,
            ),
        ]
        path = reporter.write_spec_decode_report("spec_002", "test-model", samples)
        content = path.read_text(encoding="utf-8")
        assert "α (accept)" in content
        assert "85.0%" in content
        assert "Acceptance Rate by Prompt Type" in content

    def test_spec_report_no_acceptance_rate(self, tmp_path):
        reporter = MarkdownReporter(root=str(tmp_path))
        samples = [SpecDecodeSample(tg_tokens=64, total_ms=1000, tg_tps=30.0)]
        path = reporter.write_spec_decode_report("spec_003", "model", samples)
        content = path.read_text(encoding="utf-8")
        assert "[!NOTE]" in content
        assert "not available" in content
