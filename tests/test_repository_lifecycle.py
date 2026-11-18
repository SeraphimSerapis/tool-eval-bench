"""The CLI history paths must not leak their SQLite connection.

`RunRepository` holds a connection for its lifetime.  Three `cli/history.py`
entry points used to construct one and rely on `__del__`, which is
non-deterministic and, under WAL, can strand `-wal` and `-shm` files.  These
tests pin the close to the code path rather than to garbage collection.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from tool_eval_bench.cli import history


def _console() -> Console:
    """A console that renders and discards, without leaking a file handle."""
    return Console(file=StringIO(), width=100)


def test_print_history_closes_the_repository_when_there_are_no_runs() -> None:
    repo = MagicMock()
    repo.list.return_value = []

    with patch("tool_eval_bench.storage.db.RunRepository", return_value=repo):
        history.print_history(_console())

    repo.close.assert_called_once()


def test_print_diff_closes_the_repository_when_the_run_is_missing() -> None:
    repo = MagicMock()
    repo.get_latest.return_value = None

    with patch("tool_eval_bench.storage.db.RunRepository", return_value=repo):
        history.print_diff(_console(), [], "latest")

    repo.close.assert_called_once()


def test_compare_runs_closes_the_repository_when_it_exits_on_a_missing_run() -> None:
    """`compare_runs` calls `sys.exit(1)` here, which used to skip the close."""
    repo = MagicMock()
    repo.get.return_value = None
    repo.get_latest.return_value = None

    with patch("tool_eval_bench.storage.db.RunRepository", return_value=repo):
        with pytest.raises(SystemExit):
            history.compare_runs(_console(), "missing-a", "missing-b")

    repo.close.assert_called_once()


def test_init_closes_connection_when_wal_setup_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from tool_eval_bench.storage import db as db_mod

    closed = {"count": 0}
    real_connect = sqlite3.connect

    class SpyConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, *args, **kwargs):
            return self._inner.execute(*args, **kwargs)

        def close(self) -> None:
            closed["count"] += 1
            self._inner.close()

    created: list[SpyConnection] = []

    def spy_connect(path, **kwargs):
        spy = SpyConnection(real_connect(path, **kwargs))
        created.append(spy)
        return spy

    monkeypatch.setattr(db_mod.sqlite3, "connect", spy_connect)
    monkeypatch.setattr(
        db_mod.RunRepository,
        "_enable_wal",
        lambda self: (_ for _ in ()).throw(sqlite3.OperationalError("wal locked")),
    )

    with pytest.raises(sqlite3.OperationalError, match="wal locked"):
        db_mod.RunRepository(str(tmp_path / "fail.sqlite"))

    assert created and closed["count"] >= 1
    with pytest.raises(sqlite3.ProgrammingError):
        created[0]._inner.execute("SELECT 1")


def test_close_still_closes_connection_when_writer_shutdown_fails(tmp_path) -> None:
    import sqlite3

    from tool_eval_bench.storage.db import RunRepository

    repo = RunRepository(str(tmp_path / "writer-fail.sqlite"))
    failing_writer = MagicMock()
    failing_writer.shutdown.side_effect = RuntimeError("shutdown failed")
    repo._writer = failing_writer

    with pytest.raises(RuntimeError, match="shutdown failed"):
        repo.close()

    assert repo._writer is None
    with pytest.raises(sqlite3.ProgrammingError):
        repo._conn.execute("SELECT 1")


def test_persist_run_closes_repository_on_exit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    from tool_eval_bench.application import run_queries
    from tool_eval_bench.storage.db import RunRepository

    opened: list[RunRepository] = []

    def make_repo() -> RunRepository:
        repo = RunRepository(str(tmp_path / "persist.sqlite"))
        opened.append(repo)
        return repo

    monkeypatch.setattr(run_queries, "_repository", make_repo)

    run_queries.persist_run(
        {"run_id": "r-close", "status": "completed", "config": {"model": "m"}, "scores": {}}
    )

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0]._conn.execute("SELECT 1")


def test_context_manager_closes_connection_on_exit(tmp_path) -> None:
    import sqlite3

    from tool_eval_bench.storage.db import RunRepository

    db_path = tmp_path / "ctx.sqlite"
    with RunRepository(str(db_path)) as repo:
        assert db_path.exists()
        conn = repo._conn

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
