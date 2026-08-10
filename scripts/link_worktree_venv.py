"""Link the primary checkout's virtual environment into a Git worktree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_GIT_REPOSITORY_ENV_VARS = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)


def _clean_git_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _GIT_REPOSITORY_ENV_VARS:
        env.pop(name, None)
    return env


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(  # noqa: S603 — fixed Git executable
        ["git", "-C", str(cwd), *args],
        stderr=subprocess.DEVNULL,
        env=_clean_git_env(),
        text=True,
    ).strip()


def _worktree_roots(current_root: Path) -> list[Path]:
    output = _git("worktree", "list", "--porcelain", cwd=current_root)
    return [
        Path(line.removeprefix("worktree "))
        for line in output.splitlines()
        if line.startswith("worktree ")
    ]


def _has_project_venv(path: Path) -> bool:
    return any(
        candidate.is_file()
        for candidate in (path / "bin" / "python", path / "Scripts" / "python.exe")
    )


def link_worktree_venv(current_root: Path) -> bool:
    """Create the worktree link, returning whether a link was created."""
    target = current_root / ".venv"
    if target.exists() or target.is_symlink():
        return False

    sources = [
        root / ".venv"
        for root in _worktree_roots(current_root)
        if root != current_root and not (root / ".venv").is_symlink()
    ]
    source = next((candidate for candidate in sources if _has_project_venv(candidate)), None)
    if source is None:
        print("No primary worktree .venv found; create one before running project checks.")
        return False

    try:
        target.symlink_to(source.resolve(), target_is_directory=True)
    except OSError as exc:
        print(f"Could not link {target} to {source}: {exc}", file=sys.stderr)
        return False
    print(f"Linked {target} -> {source.resolve()}")
    return True


def main() -> int:
    try:
        current_root = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
    except (OSError, subprocess.CalledProcessError):
        return 0
    link_worktree_venv(current_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
