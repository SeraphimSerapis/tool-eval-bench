"""SQLite persistence for benchmark runs."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

_SCHEMA_VERSION = 4

# Wait rather than fail when another process holds the write lock (concurrent
# runs against different endpoints share one database file).
_BUSY_TIMEOUT_MS = 10_000


def _default_db_path() -> str:
    """Resolve default DB path relative to the current working directory.

    The database is stored under ``./data/`` in whichever directory the user
    invokes the CLI from — not relative to the installed package location
    (which would land inside ``.venv/``).
    """
    return str(Path.cwd() / "data" / "benchmarks.sqlite")


def _split_traces(scores: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return *scores* with per-scenario traces removed, plus the traces.

    The caller's dict is left untouched — it is also handed to the report writer
    and returned to the CLI, both of which still expect full traces.
    """
    results = scores.get("scenario_results")
    if not isinstance(results, list):
        return scores, {}
    traces: dict[str, str] = {}
    stripped: list[Any] = []
    for result in results:
        if not isinstance(result, dict):
            stripped.append(result)
            continue
        scenario_id = result.get("scenario_id")
        raw_log = result.get("raw_log")
        if scenario_id and raw_log:
            traces[str(scenario_id)] = str(raw_log)
            result = {k: v for k, v in result.items() if k != "raw_log"}
        stripped.append(result)
    return {**scores, "scenario_results": stripped}, traces


def _merge_traces(scores: dict[str, Any] | None, traces: dict[str, str]) -> dict[str, Any] | None:
    """Put externalized traces back on their scenario results."""
    if not scores or not traces:
        return scores
    results = scores.get("scenario_results")
    if not isinstance(results, list):
        return scores
    merged = [
        {**r, "raw_log": traces.get(str(r.get("scenario_id")), r.get("raw_log", ""))}
        if isinstance(r, dict)
        else r
        for r in results
    ]
    return {**scores, "scenario_results": merged}


class RunRepository:
    """Handles SQLite persistence for scenario-based benchmark runs.

    Keeps a single persistent connection for the repository's lifetime,
    avoiding per-operation connection overhead.  Call ``close()`` explicitly
    when done, or rely on ``__del__`` for cleanup.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = Path(db_path or _default_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets ``acheckpoint_scenario_result`` run a
        # write on a worker thread instead of stalling the event loop.  That is
        # only safe while access stays serialised, which ``_write_lock`` and the
        # single-worker executor below guarantee.
        self._conn: sqlite3.Connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._write_lock = threading.Lock()
        self._writer: ThreadPoolExecutor | None = None
        # WAL mode: crash-safe and allows concurrent reads during active runs
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        self._init_db()

    def close(self) -> None:
        """Close the underlying SQLite connection and any writer thread."""
        if self._writer is not None:
            self._writer.shutdown(wait=True)
            self._writer = None
        if self._conn:
            self._conn.close()

    def __enter__(self) -> "RunRepository":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # safety net
        try:
            self.close()
        except (sqlite3.Error, AttributeError):
            logging.getLogger(__name__).debug("Error closing DB connection in __del__")

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
                  metadata_json TEXT,
                  run_type TEXT NOT NULL DEFAULT 'tool_eval',
                  report_path TEXT
                )
                """
            )
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(scenario_runs)").fetchall()
            }
            if version < 1:
                if "run_type" not in columns:
                    conn.execute(
                        "ALTER TABLE scenario_runs ADD COLUMN run_type TEXT NOT NULL DEFAULT 'tool_eval'"
                    )
                version = 1
            if version < 2:
                if "report_path" not in columns:
                    conn.execute("ALTER TABLE scenario_runs ADD COLUMN report_path TEXT")
                version = 2
            if version < 3:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_checkpoints (
                      run_id TEXT NOT NULL,
                      scenario_id TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      result_json TEXT NOT NULL,
                      PRIMARY KEY (run_id, scenario_id)
                    )
                    """
                )
                version = 3
            if version < 4:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scenario_traces (
                      run_id TEXT NOT NULL,
                      scenario_id TEXT NOT NULL,
                      raw_log TEXT NOT NULL,
                      PRIMARY KEY (run_id, scenario_id)
                    )
                    """
                )
                version = 4
            if version != int(conn.execute("PRAGMA user_version").fetchone()[0]):
                conn.execute(f"PRAGMA user_version = {version}")

    def upsert_scenario_run(self, run_data: dict[str, Any]) -> None:
        """Persist a scenario-based benchmark run.

        Uses INSERT OR REPLACE so that resumed runs can update their
        original row.  New runs should generally produce unique IDs
        (microsecond timestamp + random nonce) so collisions don't occur.

        Traces are stored in ``scenario_traces`` rather than inside
        ``scores_json``: they dominate the blob (a 69-scenario run's traces run
        to megabytes) and every ``list()`` for ``history`` or ``leaderboard``
        would otherwise deserialize all of them just to read a score.
        """
        scores, traces = _split_traces(run_data.get("scores", {}))
        with self._conn as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO scenario_runs(
                    run_id, created_at, status, model, config_json,
                    scores_json, metadata_json, run_type, report_path
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                  created_at=excluded.created_at,
                  status=excluded.status,
                  model=excluded.model,
                  config_json=excluded.config_json,
                  scores_json=excluded.scores_json,
                  metadata_json=excluded.metadata_json,
                  run_type=excluded.run_type,
                  report_path=excluded.report_path
                """,
                (
                    run_data["run_id"],
                    now,
                    run_data.get("status", "completed"),
                    run_data.get("config", {}).get("model", "unknown"),
                    json.dumps(run_data.get("config", {})),
                    json.dumps(scores),
                    json.dumps(run_data.get("metadata", {})),
                    run_data.get("run_type", "tool_eval"),
                    run_data.get("report_path"),
                ),
            )
            conn.executemany(
                """
                INSERT INTO scenario_traces(run_id, scenario_id, raw_log)
                VALUES(?,?,?)
                ON CONFLICT(run_id, scenario_id) DO UPDATE SET raw_log=excluded.raw_log
                """,
                [
                    (run_data["run_id"], scenario_id, raw_log)
                    for scenario_id, raw_log in traces.items()
                ],
            )

    def mark_run_status(self, run_id: str, status: str) -> None:
        """Update only the status of an existing run row.

        Used to flip a checkpointed run to ``interrupted`` without rewriting
        scores that were never computed.
        """
        with self._conn as conn:
            conn.execute("UPDATE scenario_runs SET status=? WHERE run_id=?", (status, run_id))

    def checkpoint_scenario_result(self, run_id: str, result: dict[str, Any]) -> None:
        """Durably record one finished scenario mid-run.

        Idempotent per (run_id, scenario_id) so a re-run of the same scenario
        within a trial overwrites rather than duplicates.
        """
        scenario_id = result.get("scenario_id")
        if not scenario_id:
            return
        # Held across the write because the async path runs this on a worker
        # thread while the main thread may also be using the connection.
        with self._write_lock, self._conn as conn:
            conn.execute(
                """
                INSERT INTO run_checkpoints(run_id, scenario_id, created_at, result_json)
                VALUES(?,?,?,?)
                ON CONFLICT(run_id, scenario_id) DO UPDATE SET
                  created_at=excluded.created_at,
                  result_json=excluded.result_json
                """,
                (
                    run_id,
                    scenario_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(result),
                ),
            )

    async def acheckpoint_scenario_result(self, run_id: str, result: dict[str, Any]) -> None:
        """Checkpoint a result without blocking the event loop.

        ``checkpoint_scenario_result`` is a synchronous INSERT that fsyncs.  At
        ``--parallel 1`` that is invisible, but with several scenarios in flight
        it stalls every pending HTTP request for the duration of the commit, and
        a contended write can hold the loop for the full busy timeout.  The work
        runs on one dedicated thread, so writes stay ordered.
        """
        loop = asyncio.get_running_loop()
        if self._writer is None:
            self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-repo-write")
        await loop.run_in_executor(self._writer, self.checkpoint_scenario_result, run_id, result)

    def get_checkpoints(self, run_id: str) -> List[dict[str, Any]]:
        """Return checkpointed scenario results for a run, oldest first."""
        with self._conn as conn:
            rows = conn.execute(
                "SELECT result_json FROM run_checkpoints WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for (payload,) in rows:
            try:
                results.append(json.loads(payload))
            except json.JSONDecodeError:
                logging.getLogger(__name__).warning(
                    "Discarding corrupt checkpoint in run %s", run_id
                )
        return results

    def clear_checkpoints(self, run_id: str) -> None:
        """Drop checkpoints once the run's final scores are persisted."""
        with self._conn as conn:
            conn.execute("DELETE FROM run_checkpoints WHERE run_id=?", (run_id,))

    def get_traces(self, run_id: str) -> dict[str, str]:
        """Return externalized traces for a run, keyed by scenario ID."""
        with self._conn as conn:
            rows = conn.execute(
                "SELECT scenario_id, raw_log FROM scenario_traces WHERE run_id=?",
                (run_id,),
            ).fetchall()
        return {scenario_id: raw_log for scenario_id, raw_log in rows}

    def get(self, run_id: str, *, include_traces: bool = True) -> dict | None:
        """Retrieve a single run by ID, with traces rehydrated by default.

        Resume and comparison need the traces; pass ``include_traces=False`` to
        skip the second query when only scores are wanted.
        """
        with self._conn as conn:
            row = conn.execute(
                "SELECT run_id, created_at, status, model, config_json, "
                "scores_json, metadata_json, run_type, report_path "
                "FROM scenario_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        scores = json.loads(row[5]) if row[5] else None
        if include_traces:
            scores = _merge_traces(scores, self.get_traces(run_id))
        return {
            "run_id": row[0],
            "created_at": row[1],
            "status": row[2],
            "model": row[3],
            "config": json.loads(row[4]),
            "scores": scores,
            "metadata": json.loads(row[6]) if row[6] else {},
            "run_type": row[7] if len(row) > 7 else "tool_eval",
            "report_path": row[8] if len(row) > 8 else None,
        }

    def list(self, limit: int = 20, model: str | None = None) -> List[dict[str, Any]]:
        """List recent runs, optionally filtered by model.

        Scenario results come back without ``raw_log``; listing is used by
        ``history``, ``leaderboard``, and ``export``, none of which read traces.
        Use ``get()`` when a full run is needed.
        """
        query = (
            "SELECT run_id, created_at, status, model, config_json, "
            "scores_json, metadata_json, run_type, report_path "
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
                "run_type": r[7] if len(r) > 7 else "tool_eval",
                "report_path": r[8] if len(r) > 8 else None,
            }
            for r in rows
        ]

    def get_latest(self, model: str | None = None) -> dict | None:
        """Get the most recent run, optionally for a specific model."""
        runs = self.list(limit=1, model=model)
        if not runs:
            return None
        return self.get(runs[0]["run_id"])

    def get_scenario_results(
        self, run_id: str, *, include_traces: bool = True
    ) -> List[dict[str, Any]] | None:
        """Extract per-scenario results from a stored run.

        Traces are rehydrated by default.  A 69-scenario run's traces come to
        megabytes, so callers that only read scores (the run diff, for one)
        should pass ``include_traces=False`` and skip both the second query and
        the merge that rebuilds every result dict.
        """
        run = self.get(run_id, include_traces=include_traces)
        if not run or not run.get("scores"):
            return None
        return run["scores"].get("scenario_results", [])
