**Reviewer** — `--reviewer-base-url`, `--reviewer-model`, and the other `--reviewer*` flags pair the
benchmarked model with a second model that reviews each turn before the harness acts on it. On
feedback the benchmarked model revises the turn, and the harness runs and grades the revision
instead of the rejected turn. `--reviewer-scope` picks every turn, tool-call turns, or final replies, and
`--reviewer-max-rounds` caps reviews per turn. Reviewer sampling inherits from the main flags unless
a `--reviewer-*` flag overrides it; the API key and headers travel only to the same server. A
reviewer error or an unusable revision lets the current draft run and is counted in the report;
a reviewer that fails its startup review stops the run before anything is stored. The pair is the system
under test: a reviewed run stores its reviewer, gets its own fingerprint and leaderboard cohort, and
refuses held-out packs. The web launcher has a Reviewer section and resumes paired runs, and the
Python API takes `reviewer=ReviewerSettings(...)`. See `docs/reviewer.md`.
