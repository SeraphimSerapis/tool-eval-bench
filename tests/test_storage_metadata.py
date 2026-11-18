"""Test for storage module — scenario run persistence."""

from pathlib import Path

from tests.conftest import open_repository


def test_scenario_run_roundtrip(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "bench.sqlite"))

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
    repo = open_repository(db_path=str(tmp_path / "bench.sqlite"))

    for i in range(3):
        repo.upsert_scenario_run(
            {
                "run_id": f"r{i}",
                "status": "completed",
                "config": {"model": "test-model", "backend": "vllm"},
                "scores": {},
                "metadata": {},
            }
        )

    runs = repo.list(limit=10)
    assert len(runs) == 3

    runs_filtered = repo.list(limit=1)
    assert len(runs_filtered) == 1


def test_upsert_replaces_config_for_resumed_run(tmp_path: Path) -> None:
    repo = open_repository(db_path=str(tmp_path / "bench.sqlite"))
    repo.upsert_scenario_run(
        {
            "run_id": "resumed",
            "config": {"model": "test-model", "scenario_ids": ["TC-01"]},
            "scores": {"final_score": 50},
        }
    )
    repo.upsert_scenario_run(
        {
            "run_id": "resumed",
            "config": {"model": "test-model", "scenario_ids": ["TC-01", "TC-02"]},
            "scores": {"final_score": 100},
        }
    )

    out = repo.get("resumed")
    assert out is not None
    assert out["config"]["scenario_ids"] == ["TC-01", "TC-02"]


def test_null_config_scores_and_metadata_persist_safely(tmp_path: Path) -> None:
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


def test_non_finite_floats_serialize_as_valid_json_for_sqlite_search(tmp_path: Path) -> None:
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
