"""A crash must not discard a partially completed run.

Scenario suites take minutes per scenario, so losing 60 finished scenarios to a
Ctrl-C at scenario 61 is the difference between a benchmark people run and one
they don't. Each finished scenario is checkpointed to SQLite; an interrupted run
is marked as such and can be picked up with ``--resume``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import open_repository
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli.helpers import prior_results_for_resume
from tool_eval_bench.domain.models import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_INTERRUPTED,
    RUN_STATUS_RUNNING,
)
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.storage.db import RunRepository


@pytest.fixture
def repo(tmp_path: Path):
    r = open_repository(db_path=str(tmp_path / "bench.sqlite"))
    yield r
    r.close()


def _scenario(sid: str) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title=sid,
        category=Category.A,
        user_message="",
        description="",
        handle_tool_call=lambda state, call: None,
        evaluate=lambda state: None,  # type: ignore[arg-type,return-value]
    )


def test_resume_reruns_a_checkpoint_without_required_trace_evidence() -> None:
    from tool_eval_bench.cli.dispatch import _resume_result_requires_rerun

    checkpoint = {
        "scenario_id": "TC-01",
        "status": "partial",
        "points": 1,
        "summary": "model outcome",
        "raw_log": "",
    }

    assert _resume_result_requires_rerun(checkpoint) is True


class TestCheckpointStorage:
    def test_roundtrip_preserves_order_and_payload(self, repo: RunRepository) -> None:
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-01", "status": "pass"})
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-02", "status": "fail"})

        stored = repo.get_checkpoints("R1")

        assert [r["scenario_id"] for r in stored] == ["TC-01", "TC-02"]
        assert stored[1]["status"] == "fail"

    def test_checkpoints_are_scoped_per_run(self, repo: RunRepository) -> None:
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-01"})
        repo.checkpoint_scenario_result("R2", {"scenario_id": "TC-02"})

        assert [r["scenario_id"] for r in repo.get_checkpoints("R1")] == ["TC-01"]
        assert [r["scenario_id"] for r in repo.get_checkpoints("R2")] == ["TC-02"]

    def test_rerunning_a_scenario_overwrites_rather_than_duplicates(
        self, repo: RunRepository
    ) -> None:
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-01", "status": "fail"})
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-01", "status": "pass"})

        stored = repo.get_checkpoints("R1")
        assert len(stored) == 1
        assert stored[0]["status"] == "pass"

    def test_result_without_scenario_id_is_ignored(self, repo: RunRepository) -> None:
        repo.checkpoint_scenario_result("R1", {"status": "pass"})
        assert repo.get_checkpoints("R1") == []

    def test_clear_removes_only_the_target_run(self, repo: RunRepository) -> None:
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-01"})
        repo.checkpoint_scenario_result("R2", {"scenario_id": "TC-02"})

        repo.clear_checkpoints("R1")

        assert repo.get_checkpoints("R1") == []
        assert len(repo.get_checkpoints("R2")) == 1

    def test_corrupt_checkpoint_is_skipped_not_fatal(self, repo: RunRepository) -> None:
        repo.checkpoint_scenario_result("R1", {"scenario_id": "TC-01"})
        with repo._conn as conn:
            conn.execute(
                "INSERT INTO run_checkpoints(run_id, scenario_id, created_at, result_json) "
                "VALUES('R1','TC-02','2026-01-01','{not json')"
            )

        stored = repo.get_checkpoints("R1")
        assert [r["scenario_id"] for r in stored] == ["TC-01"]

    def test_mark_run_status_does_not_touch_scores(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(
            {
                "run_id": "R1",
                "status": RUN_STATUS_RUNNING,
                "config": {"model": "m"},
                "scores": {"final_score": 42},
            }
        )
        repo.mark_run_status("R1", RUN_STATUS_INTERRUPTED)

        stored = repo.get("R1")
        assert stored is not None
        assert stored["status"] == RUN_STATUS_INTERRUPTED
        assert stored["scores"] == {"final_score": 42}

    def test_busy_timeout_is_configured(self, repo: RunRepository) -> None:
        """Concurrent runs share one DB file; a locked write should wait, not fail."""
        assert repo._conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0


class TestResumeResultSelection:
    def test_completed_run_uses_its_final_scores(self) -> None:
        prev = {"scores": {"scenario_results": [{"scenario_id": "TC-01", "status": "pass"}]}}
        assert prior_results_for_resume(prev, []) == [{"scenario_id": "TC-01", "status": "pass"}]

    def test_interrupted_run_falls_back_to_checkpoints(self) -> None:
        """Without this the entire interrupted run would have to be redone."""
        prev: dict = {"scores": {}}
        checkpoints = [
            {"scenario_id": "TC-01", "status": "pass"},
            {"scenario_id": "TC-02", "status": "fail"},
        ]

        recovered = prior_results_for_resume(prev, checkpoints)

        assert {r["scenario_id"] for r in recovered} == {"TC-01", "TC-02"}

    def test_final_scores_win_over_checkpoints_for_the_same_scenario(self) -> None:
        prev = {"scores": {"scenario_results": [{"scenario_id": "TC-01", "status": "pass"}]}}
        checkpoints = [{"scenario_id": "TC-01", "status": "fail"}]

        recovered = prior_results_for_resume(prev, checkpoints)

        assert recovered == [{"scenario_id": "TC-01", "status": "pass"}]

    def test_entries_without_scenario_id_are_dropped(self) -> None:
        prev = {"scores": {"scenario_results": [{"status": "pass"}]}}
        assert prior_results_for_resume(prev, []) == []

    def test_missing_scores_key_is_tolerated(self) -> None:
        assert prior_results_for_resume({}, []) == []
        assert prior_results_for_resume({"scores": None}, []) == []


class TestServiceCheckpointing:
    @pytest.fixture
    def scenario(self) -> ScenarioDefinition:
        return _scenario("TC-01")

    @pytest.mark.asyncio
    async def test_completed_run_claims_then_completes_and_clears(
        self, monkeypatch: pytest.MonkeyPatch, repo: RunRepository, scenario
    ) -> None:
        from tool_eval_bench.application import service as service_module
        from tool_eval_bench.runner.orchestrator import score_results

        result = ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok"
        )
        summary = score_results([result], [scenario])

        async def fake_run(adapter, **kwargs):
            await kwargs["on_scenario_result"](scenario, result, 1, 1)
            return summary

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        service = BenchmarkService(repo=repo, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        run_data = await service.run_benchmark(
            model="m", backend="vllm", base_url="http://localhost:8000", scenarios=[scenario]
        )

        stored = repo.get(run_data["run_id"])
        assert stored is not None
        assert stored["status"] == RUN_STATUS_COMPLETED
        # Final scores supersede the checkpoints, so they are cleaned up.
        assert repo.get_checkpoints(run_data["run_id"]) == []

    @pytest.mark.asyncio
    async def test_completed_run_id_is_not_reclaimed_or_mutated(
        self, monkeypatch: pytest.MonkeyPatch, repo: RunRepository, scenario
    ) -> None:
        """A retry must use a new run ID, never overwrite completed evidence."""
        from tool_eval_bench.application import service as service_module

        repo.upsert_scenario_run(
            {
                "run_id": "completed-run",
                "status": RUN_STATUS_COMPLETED,
                "config": {"model": "m"},
                "scores": {"final_score": 0},
            }
        )
        run_all = MagicMock()
        monkeypatch.setattr(service_module, "run_all_scenarios", run_all)
        service = BenchmarkService(repo=repo, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        with pytest.raises(ValueError, match="immutable"):
            await service.run_benchmark(
                model="m",
                backend="vllm",
                base_url="http://localhost:8000",
                scenarios=[scenario],
                resume_run_id="completed-run",
            )

        assert repo.get("completed-run")["scores"] == {"final_score": 0}
        run_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupted_run_keeps_partial_results_and_is_resumable(
        self, monkeypatch: pytest.MonkeyPatch, repo: RunRepository
    ) -> None:
        from tool_eval_bench.application import service as service_module

        done = _scenario("TC-01")
        never_ran = _scenario("TC-02")
        finished = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            raw_log="trace",
        )

        async def fake_run(adapter, **kwargs):
            await kwargs["on_scenario_result"](done, finished, 1, 2)
            raise KeyboardInterrupt

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        service = BenchmarkService(repo=repo, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        with pytest.raises(KeyboardInterrupt):
            await service.run_benchmark(
                model="m",
                backend="vllm",
                base_url="http://localhost:8000",
                scenarios=[done, never_ran],
            )

        interrupted = [r for r in repo.list(limit=10) if r["status"] == RUN_STATUS_INTERRUPTED]
        assert len(interrupted) == 1
        run_id = interrupted[0]["run_id"]

        checkpoints = repo.get_checkpoints(run_id)
        assert [c["scenario_id"] for c in checkpoints] == ["TC-01"]
        assert checkpoints[0]["raw_log"] == "trace"

        # The resume path can now rebuild the finished work from the checkpoints.
        recovered = prior_results_for_resume(interrupted[0], checkpoints)
        assert [r["scenario_id"] for r in recovered] == ["TC-01"]

    @pytest.mark.asyncio
    async def test_resumed_interruption_keeps_the_full_protocol_config(
        self, monkeypatch: pytest.MonkeyPatch, repo: RunRepository
    ) -> None:
        """A second interruption must not reduce the stored scenario identity."""
        from tool_eval_bench.application import service as service_module

        prior = _scenario("TC-01")
        rerun = _scenario("TC-02")
        rerun_result = ScenarioResult(
            scenario_id=rerun.id,
            status=ScenarioStatus.FAIL,
            points=0,
            summary="timeout",
            failure_kind="timeout",
        )

        async def fake_run(adapter, **kwargs):
            await kwargs["on_scenario_result"](rerun, rerun_result, 1, 1)
            raise KeyboardInterrupt

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        service = BenchmarkService(repo=repo, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        with pytest.raises(KeyboardInterrupt):
            await service.run_benchmark(
                model="m",
                backend="vllm",
                base_url="http://localhost:8000",
                scenarios=[rerun],
                resume_run_id="interrupted-run",
                resume_prior_results=[
                    {
                        "scenario_id": prior.id,
                        "status": "pass",
                        "points": 2,
                        "summary": "done",
                    }
                ],
                resume_scenarios=[prior, rerun],
            )

        stored = repo.get("interrupted-run")
        assert stored is not None
        assert stored["status"] == RUN_STATUS_INTERRUPTED
        assert stored["config"]["scenario_ids"] == [prior.id, rerun.id]

    @pytest.mark.asyncio
    async def test_caller_callback_still_runs_after_checkpointing(
        self, monkeypatch: pytest.MonkeyPatch, repo: RunRepository, scenario
    ) -> None:
        from tool_eval_bench.application import service as service_module
        from tool_eval_bench.runner.orchestrator import score_results

        result = ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok"
        )
        summary = score_results([result], [scenario])
        seen: list[str] = []

        async def user_callback(sc, res, index, total) -> None:
            seen.append(f"{res.scenario_id}:{index}/{total}")

        async def fake_run(adapter, **kwargs):
            await kwargs["on_scenario_result"](scenario, result, 1, 1)
            return summary

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        service = BenchmarkService(repo=repo, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        await service.run_benchmark(
            model="m",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[scenario],
            on_scenario_result=user_callback,
        )

        assert seen == ["TC-01:1/1"]

    @pytest.mark.asyncio
    async def test_checkpoint_failure_never_aborts_the_run(
        self, monkeypatch: pytest.MonkeyPatch, scenario
    ) -> None:
        """Bookkeeping is best-effort — a full disk must not kill a live run."""
        from tool_eval_bench.application import service as service_module
        from tool_eval_bench.runner.orchestrator import score_results

        result = ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok"
        )
        summary = score_results([result], [scenario])

        broken = MagicMock()
        broken.checkpoint_scenario_result.side_effect = OSError("disk full")
        broken.upsert_scenario_run.side_effect = [OSError("disk full"), None]

        async def fake_run(adapter, **kwargs):
            await kwargs["on_scenario_result"](scenario, result, 1, 1)
            return summary

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        service = BenchmarkService(repo=broken, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        run_data = await service.run_benchmark(
            model="m", backend="vllm", base_url="http://localhost:8000", scenarios=[scenario]
        )

        assert run_data["scores"]["final_score"] == 100

    @pytest.mark.asyncio
    async def test_persistence_disabled_skips_checkpointing_entirely(
        self, monkeypatch: pytest.MonkeyPatch, scenario
    ) -> None:
        from tool_eval_bench.application import service as service_module
        from tool_eval_bench.runner.orchestrator import score_results

        result = ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="ok"
        )
        summary = score_results([result], [scenario])
        seen: list[str] = []

        async def user_callback(sc, res, index, total) -> None:
            seen.append(res.scenario_id)

        async def fake_run(adapter, **kwargs):
            await kwargs["on_scenario_result"](scenario, result, 1, 1)
            return summary

        monkeypatch.setattr(service_module, "run_all_scenarios", fake_run)
        service = BenchmarkService(repo=None, reporter=None)
        monkeypatch.setattr(service, "_adapter_for", lambda *_args, **_kwargs: object())

        await service.run_benchmark(
            model="m",
            backend="vllm",
            base_url="http://localhost:8000",
            scenarios=[scenario],
            on_scenario_result=user_callback,
        )

        assert seen == ["TC-01"]
