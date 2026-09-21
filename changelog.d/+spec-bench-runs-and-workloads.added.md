`--spec-bench` repeats each depth × prompt cell `--spec-runs` times (default 3) and pools the
counters, showing the per-run α range next to the pooled value; a single request is only a few
dozen speculative steps. `--spec-prompt-file` adds your own workload as plain lines or JSON
lines with `prompt` and an optional `label`. `--temperature` now reaches the benchmark requests
(greedy remains the default and the report records the value, since acceptance falls as
sampling temperature rises). Rows report verify **Steps/s**, a lower bound on the no-spec decode
rate, and the summary derives a speedup ceiling from it when no `--baseline-tgs` was given.
