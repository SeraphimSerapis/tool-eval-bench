"""SQLite persistence for benchmark runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_db_path() -> str:
    """Resolve default DB path relative to the project root.

    This ensures the same database is used regardless of the working
    directory when invoking the CLI.
    """
    # Walk up from storage/db.py → storage → tool_eval_bench → src → project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return str(project_root / "data" / "benchmarks.sqlite")


class RunRepository:
    """Handles SQLite persistence for scenario-based benchmark runs.

    Keeps a single persistent connection for the repository's lifetime,
    avoiding per-operation connection overhead.  Call ``close()`` explicitly
    when done, or rely on ``__del__`` for cleanup.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = Path(db_path or _default_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        # WAL mode: crash-safe and allows concurrent reads during active runs
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn:
            self._conn.close()

    def __del__(self) -> None:  # safety net
        try:
            self.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        with self._conn as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_runs (
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

    def upsert_scenario_run(self, run_data: dict[str, Any]) -> None:
        """Persist a scenario-based benchmark run."""
        with self._conn as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO scenario_runs(run_id, created_at, status, model, config_json, scores_json, metadata_json)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                  status=excluded.status,
                  scores_json=excluded.scores_json,
                  metadata_json=excluded.metadata_json
                """,
                (
                    run_data["run_id"],
                    now,
                    run_data.get("status", "completed"),
                    run_data.get("config", {}).get("model", "unknown"),
                    json.dumps(run_data.get("config", {})),
                    json.dumps(run_data.get("scores", {})),
                    json.dumps(run_data.get("metadata", {})),
                ),
            )

    def get(self, run_id: str) -> dict | None:
        """Retrieve a single run by ID."""
        with self._conn as conn:
            row = conn.execute(
                "SELECT run_id, created_at, status, model, config_json, scores_json, metadata_json "
                "FROM scenario_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "created_at": row[1],
            "status": row[2],
            "model": row[3],
            "config": json.loads(row[4]),
            "scores": json.loads(row[5]) if row[5] else None,
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    def list(self, limit: int = 20, model: str | None = None) -> list[dict]:
        """List recent runs, optionally filtered by model."""
        query = (
            "SELECT run_id, created_at, status, model, config_json, scores_json, metadata_json "
            "FROM scenario_runs"
        )
        params: list[str | int] = []
        if model:
            query += " WHERE model = ?"
            params.append(model)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._conn as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "run_id": r[0],
                "created_at": r[1],
                "status": r[2],
                "model": r[3],
                "config": json.loads(r[4]),
                "scores": json.loads(r[5]) if r[5] else None,
                "metadata": json.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    def get_latest(self, model: str | None = None) -> dict | None:
        """Get the most recent run, optionally for a specific model."""
        runs = self.list(limit=1, model=model)
        return runs[0] if runs else None

    def get_scenario_results(self, run_id: str) -> list[dict] | None:
        """Extract per-scenario results from a stored run."""
        run = self.get(run_id)
        if not run or not run.get("scores"):
            return None
        return run["scores"].get("scenario_results", [])

