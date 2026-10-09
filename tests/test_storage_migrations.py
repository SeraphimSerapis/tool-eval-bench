from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import open_repository
from tool_eval_bench.storage import db
from tool_eval_bench.storage.db import _SCHEMA_VERSION


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

    repo = open_repository(db_path=str(db_path))
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scenario_runs)")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"run_type", "report_path"} <= columns
    assert {"run_checkpoints", "scenario_traces"} <= tables
    assert version == _SCHEMA_VERSION

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


def _make_pre_v1_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE scenario_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "status TEXT NOT NULL, model TEXT NOT NULL, config_json TEXT NOT NULL, "
            "scores_json TEXT, metadata_json TEXT)"
        )
    # The context manager commits but does not close; Windows needs the close.
    conn.close()


def _open_after_barrier(path: str, barrier: Any, outcomes: Any) -> None:
    from tool_eval_bench.storage.db import RunRepository

    barrier.wait()
    try:
        with RunRepository(path) as repo:
            repo.list()
        outcomes.put("ok")
    except Exception as exc:  # noqa: BLE001 - the parent asserts on the message
        outcomes.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="needs fork so the children share no interpreter state with pytest",
)
@pytest.mark.parametrize("legacy", [False, True], ids=["new-file", "pre-v1"])
def test_concurrent_first_open_neither_locks_nor_duplicates_columns(
    tmp_path: Path, legacy: bool
) -> None:
    # Separate processes, because SQLite file locks do not contend between
    # connections the same way inside one process. Before the fix, about a
    # quarter of openers failed with "database is locked" on the WAL switch or
    # "duplicate column name" on the migration.
    ctx = multiprocessing.get_context("fork")
    failures: list[str] = []
    for trial in range(6):
        db_path = tmp_path / f"trial{trial}.sqlite"
        if legacy:
            _make_pre_v1_database(db_path)
        barrier = ctx.Barrier(4)
        outcomes = ctx.Queue()
        workers = [
            ctx.Process(target=_open_after_barrier, args=(str(db_path), barrier, outcomes))
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        results = [outcomes.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=30)
        failures.extend(r for r in results if r != "ok")
    assert failures == []


class _LockedThenOk:
    """Stands in for the connection: the WAL pragma fails ``fail_times`` times."""

    def __init__(self, fail_times: int, message: str = "database is locked") -> None:
        self.fail_times = fail_times
        self.message = message
        self.calls = 0

    def execute(self, sql: str) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise sqlite3.OperationalError(self.message)


def test_wal_switch_retries_while_another_process_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "_WAL_RETRY_DELAY_S", 0)
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    real = repo._conn
    fake = _LockedThenOk(fail_times=3)
    repo._conn = fake
    try:
        repo._enable_wal()
    finally:
        repo._conn = real
    assert fake.calls == 4


def test_wal_switch_gives_up_after_the_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "_WAL_RETRY_DELAY_S", 0)
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    real = repo._conn
    fake = _LockedThenOk(fail_times=db._WAL_RETRY_ATTEMPTS)
    repo._conn = fake
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            repo._enable_wal()
    finally:
        repo._conn = real
    assert fake.calls == db._WAL_RETRY_ATTEMPTS


def test_wal_switch_does_not_retry_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "_WAL_RETRY_DELAY_S", 0)
    repo = open_repository(db_path=str(tmp_path / "b.sqlite"))
    real = repo._conn
    fake = _LockedThenOk(fail_times=1, message="disk I/O error")
    repo._conn = fake
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            repo._enable_wal()
    finally:
        repo._conn = real
    assert fake.calls == 1


def test_add_column_tolerates_only_an_existing_column() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (a TEXT)")
        db._add_column(conn, "ALTER TABLE t ADD COLUMN b TEXT")
        db._add_column(conn, "ALTER TABLE t ADD COLUMN b TEXT")
        assert [row[1] for row in conn.execute("PRAGMA table_info(t)")] == ["a", "b"]
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            db._add_column(conn, "ALTER TABLE missing ADD COLUMN c TEXT")
    finally:
        conn.close()
