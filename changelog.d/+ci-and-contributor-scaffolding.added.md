Added a concurrency group so pushes to a pull request stop queueing redundant matrix runs, a
`pre-commit` job so the hooks cannot rot, a `pip-audit` job, a Dependabot config for the unpinned
dependency floors, and coverage upload as an artifact. Pre-commit gained the usual safety hooks,
including `detect-private-key` and `check-added-large-files`. A bare `pytest` now excludes the live
tests and a local `--cov` run fails on the same 80% floor CI enforces. Also added issue templates
and a `CODEOWNERS`.
