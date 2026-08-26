Code scanning now reports findings the project can act on. The quality queries
were producing 102 open alerts, 101 of them without a security severity, which
buried the one that had one: an exponential-backtracking regex in the
Prometheus label parser. Auditing all 102 found two real defects, both fixed
here, and showed the rest to be rules this codebase's conventions make
structurally wrong: `...` in a `Protocol` body, private constants shared
between sibling modules, deliberate re-exports, `ruff format`'s string
wrapping, iterating an `Enum`, and a final `return` that mypy requires and
CodeQL calls unreachable. Those rules are now excluded, each with the reason
recorded next to it in `.github/codeql/codeql-config.yml`.

The two real defects: the leaderboard's grouping loop assigned
`scenario_count` and `backend` and never read them, and the orchestrator
guarded its parallel-path warning with `if concurrency > 1` on a path the
sequential branch has already returned from.
