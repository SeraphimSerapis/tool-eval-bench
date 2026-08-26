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
