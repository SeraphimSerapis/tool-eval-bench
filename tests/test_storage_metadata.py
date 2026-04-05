"""Test for storage module — scenario run persistence."""

from pathlib import Path

from tool_eval_bench.storage.db import RunRepository


def test_scenario_run_roundtrip(tmp_path: Path) -> None:
    repo = RunRepository(db_path=str(tmp_path / "bench.sqlite"))

    run_data = {
        "run_id": "r1",
        "status": "completed",
        "config": {"model": "test-model", "backend": "vllm", "base_url": "http://localhost:8080"},
        "scores": {"final_score": 75, "total_points": 45, "max_points": 60},
        "metadata": {"git_sha": "abc123", "host": "local"},
    }
    repo.upsert_scenario_run(run_data)

    out = repo.get("r1")
    assert out is not None
    assert out["model"] == "test-model"
    assert out["scores"]["final_score"] == 75
    assert out["metadata"]["git_sha"] == "abc123"


def test_list_runs(tmp_path: Path) -> None:
    repo = RunRepository(db_path=str(tmp_path / "bench.sqlite"))

    for i in range(3):
        repo.upsert_scenario_run({
            "run_id": f"r{i}",
            "status": "completed",
            "config": {"model": "test-model", "backend": "vllm"},
            "scores": {},
            "metadata": {},
        })

    runs = repo.list(limit=10)
    assert len(runs) == 3

    runs_filtered = repo.list(limit=1)
    assert len(runs_filtered) == 1
