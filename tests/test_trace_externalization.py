"""Traces live in their own table, not inside the run's scores blob.

Traces dominate a run's stored bytes — a 69-scenario run's raw logs run to
megabytes — and ``history``, ``leaderboard``, and ``export`` all list many runs
at once while reading nothing but scores. Keeping traces in ``scores_json``
meant every listing deserialized every trace. They are still required for
resume and for full-trace reports, so a single-run ``get()`` must return them
unchanged, including for rows written before the split.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import open_repository
from tool_eval_bench.storage.db import RunRepository

TRACE_A = "USER: what is the weather\nASSISTANT: get_weather(Berlin)\nTOOL: 21C"
TRACE_B = "USER: book a flight\nASSISTANT: search_flights(BER, JFK)"


def _run(run_id: str = "run-1", *, traces: bool = True) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "completed",
        "config": {"model": "m"},
        "scores": {
            "final_score": 75,
            "scenario_results": [
                {
                    "scenario_id": "TC-01",
                    "status": "pass",
                    "points": 2,
                    "raw_log": TRACE_A if traces else "",
                },
                {
                    "scenario_id": "TC-02",
                    "status": "fail",
                    "points": 0,
                    "raw_log": TRACE_B if traces else "",
                },
            ],
        },
        "metadata": {},
    }


@pytest.fixture
def repo(tmp_path: Path):
    repository = open_repository(db_path=str(tmp_path / "b.sqlite"))
    yield repository
    repository.close()


def _stored_blob(repo: RunRepository, run_id: str) -> str:
    with sqlite3.connect(repo.db_path) as conn:
        row = conn.execute(
            "SELECT scores_json FROM scenario_runs WHERE run_id=?", (run_id,)
        ).fetchone()
    return str(row[0])


class TestRoundTrip:
    def test_get_returns_traces_unchanged(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        stored = repo.get("run-1")

        assert stored is not None
        assert [r["raw_log"] for r in stored["scores"]["scenario_results"]] == [TRACE_A, TRACE_B]

    def test_scores_blob_no_longer_carries_traces(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        blob = _stored_blob(repo, "run-1")

        assert TRACE_A not in blob
        assert "raw_log" not in blob
        assert json.loads(blob)["final_score"] == 75

    def test_scores_other_than_traces_survive(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        results = repo.get("run-1")["scores"]["scenario_results"]  # type: ignore[index]

        assert [(r["scenario_id"], r["status"], r["points"]) for r in results] == [
            ("TC-01", "pass", 2),
            ("TC-02", "fail", 0),
        ]

    def test_caller_dict_is_not_mutated(self, repo: RunRepository) -> None:
        run_data = _run()

        repo.upsert_scenario_run(run_data)

        assert run_data["scores"]["scenario_results"][0]["raw_log"] == TRACE_A

    def test_rerun_overwrites_a_scenario_trace(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())
        rerun = _run()
        rerun["scores"]["scenario_results"][0]["raw_log"] = "USER: retried"

        repo.upsert_scenario_run(rerun)

        assert repo.get_traces("run-1")["TC-01"] == "USER: retried"

    def test_get_scenario_results_includes_traces(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        results = repo.get_scenario_results("run-1")

        assert results is not None
        assert results[0]["raw_log"] == TRACE_A

    def test_get_latest_includes_traces(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        latest = repo.get_latest()

        assert latest is not None
        assert latest["scores"]["scenario_results"][0]["raw_log"] == TRACE_A


class TestListingSkipsTraces:
    def test_list_omits_traces(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        (listed,) = repo.list()

        assert listed["scores"]["final_score"] == 75
        assert listed["scores"]["scenario_results"][0].get("raw_log") is None

    def test_get_can_opt_out_of_traces(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run())

        stored = repo.get("run-1", include_traces=False)

        assert stored is not None
        assert stored["scores"]["scenario_results"][0].get("raw_log") is None


class TestBackwardCompatibility:
    def test_traces_stored_inline_by_older_versions_are_still_returned(
        self, repo: RunRepository
    ) -> None:
        # Rows written before the split keep raw_log inside scores_json and have
        # no scenario_traces entries at all.
        with sqlite3.connect(repo.db_path) as conn:
            conn.execute(
                "INSERT INTO scenario_runs(run_id, created_at, status, model, config_json, "
                "scores_json, metadata_json, run_type) VALUES(?,?,?,?,?,?,?,?)",
                (
                    "legacy",
                    "2026-01-01T00:00:00+00:00",
                    "completed",
                    "m",
                    "{}",
                    json.dumps(_run()["scores"]),
                    "{}",
                    "tool_eval",
                ),
            )

        stored = repo.get("legacy")

        assert stored is not None
        assert stored["scores"]["scenario_results"][0]["raw_log"] == TRACE_A

    def test_runs_without_scenario_results_round_trip(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(
            {"run_id": "perf", "status": "completed", "config": {}, "scores": {"throughput": 12}}
        )

        stored = repo.get("perf")

        assert stored is not None
        assert stored["scores"] == {"throughput": 12}

    def test_empty_traces_are_not_externalized(self, repo: RunRepository) -> None:
        repo.upsert_scenario_run(_run(traces=False))

        assert repo.get_traces("run-1") == {}
