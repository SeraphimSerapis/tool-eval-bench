**Pytest temp dirs** — passing local runs now drop `/tmp/pytest-of-*`
session trees (`tmp_path_retention_policy = failed`, keep one failed
session). Nested worktrees created by `test_worktree_venv.py` no longer
accumulate on disk after a green suite.
