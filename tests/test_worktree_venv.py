"""Regression coverage for shared virtual environments in Git worktrees."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tool_eval_bench.utils.metadata import _git_env_without_repository

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "link_worktree_venv.py"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(  # noqa: S603 — fixed Git executable
        ["git", *args], cwd=cwd, env=_git_env_without_repository(), text=True
    ).strip()


def test_link_worktree_venv_ignores_inherited_git_dir(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    primary.mkdir()
    _git("init", "-q", cwd=primary)
    _git("config", "user.email", "test@example.com", cwd=primary)
    _git("config", "user.name", "Test", cwd=primary)
    (primary / "tracked.txt").write_text("tracked\n")
    _git("add", "tracked.txt", cwd=primary)
    _git("commit", "-qm", "initial", cwd=primary)

    python = primary / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
    _git("worktree", "add", "-qb", "topic", str(linked), cwd=primary)

    polluted_env = os.environ.copy()
    polluted_env["GIT_DIR"] = str(primary / ".git")
    polluted_env["GIT_WORK_TREE"] = str(primary)
    subprocess.run(  # noqa: S603 — repository-owned script under the project interpreter
        [sys.executable, str(_SCRIPT)], cwd=linked, env=polluted_env, check=True
    )

    target = linked / ".venv"
    assert target.is_symlink()
    assert target.resolve() == (primary / ".venv").resolve()
    assert _git("config", "--bool", "core.bare", cwd=primary) == "false"
