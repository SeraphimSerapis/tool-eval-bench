from __future__ import annotations

import sqlite3
from pathlib import Path

from tool_eval_bench.storage.db import RunRepository


def test_old_database_is_migrated_to_current_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE scenario_runs (
              run_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              status TEXT NOT NULL,
              model TEXT NOT NULL,
              config_json TEXT NOT NULL,
              scores_json TEXT,
              metadata_json TEXT
            )
            """
        )
        conn.execute("PRAGMA user_version = 0")

    repo = RunRepository(db_path=str(db_path))
    try:
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(scenario_runs)")}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert {"run_type", "report_path"} <= columns
        assert version == 2

        repo.upsert_scenario_run(
            {
                "run_id": "migrated",
                "status": "completed",
                "config": {"model": "test"},
                "scores": {},
                "report_path": "runs/migrated.md",
            }
        )
        stored = repo.get("migrated")
        assert stored is not None
        assert stored["report_path"] == "runs/migrated.md"
    finally:
        repo.close()
