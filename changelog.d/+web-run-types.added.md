**The web UI shows throughput, spec-decode, needle, and accuracy runs as what they are.** These runs
used to appear as "completed · unknown" with an empty score ring, and opening one showed
"0 / 0 / 0" and no results. Each now carries a type label and its main figure in the run list, such
as effective decode speed for spec decode or retrieval for needle, and its page shows its
measurements as a table. A **Same session** panel links every run to the other runs its command
stored, so the spec-decode run of a `--spec-bench` invocation is one click from its scored run.
