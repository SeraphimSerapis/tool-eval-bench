CI runs five checks on a pull request instead of thirteen. Ruff and mypy now run
once rather than four and three times: both are version-independent, and mypy is
pinned to `python_version = "3.11"` whatever interpreter it runs on. The macOS
runner is gone, having recorded no finding of its own. The Docker and wheel smoke
tests share a `packaging` job, and the `llama-benchy` tests fold into the main
`test` job.

Python 3.11 and Windows moved to a `test-extended` job that runs after merge
rather than on every pull request. Windows stays in CI because it has caught real
product bugs, but those came from running the suite there at all rather than from
gating each change on it.

The dependency audit moved to its own workflow, on a weekly schedule and on pull
requests that touch `uv.lock` or `pyproject.toml`. The locked set does not change
between those, but the vulnerability database does.
