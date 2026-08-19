**`--label` run annotations** — an arbitrary string (`--label "tonyd2wild
tool hardening 646c55f"`) is now recorded on every report an execution
generates: a `Label` row in the tool-eval Run Context table, a `- **Label**:`
header line in GSM8K / MMLU / IFEval / throughput / spec-decode /
context-pressure-sweep reports, and the metadata persisted to SQLite (visible
via `history` and `export`). A filesystem-safe slug of the label is also
appended to report filenames (`<run_id>--<slug>.md`,
`<run_id>--<slug>_summary.md`), so all artifacts of one execution share a
grep-able marker while the timestamped run ID remains the leading identity.
Report rendering makes control characters visible and prevents Markdown or
terminal-markup injection; labels without an ASCII slug receive a stable hash
marker. The label is an annotation only: it never changes the config
fingerprint or run ID, so identical runs with different labels stay
comparable.
