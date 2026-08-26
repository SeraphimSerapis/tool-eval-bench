The IFEval and MMLU plugins no longer assign a `content` fallback in their
per-item error branches. Nothing read it: an item that raises sets
`is_error`, and `content` is only read on the path that requires `is_error` to
be false. Removing it means one less branch to trace to establish that. The
migration test also drops a `try`/`finally` that closed a repository the
`open_repository` helper already closes through an autouse fixture.
