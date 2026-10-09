**Context pressure detects Strata's context window.** `--context-pressure`,
`--context-pressure-sweep`, and `--needle` no longer ask for `--context-size` against Strata. Its
model listing declares no window that the detection reads, so they now use the window the run
metadata already records from Strata's `/props` `n_ctx`, or its `/health` `max_context` when
`/props` has none. Both carry the per-request limit Strata enforces. An explicit `--context-size`
still wins, and other backends detect their window exactly as before.
