**`--system-prompt` / `--system-prompt-file`** — a run can now replace the built-in
"helpful assistant" system prompt with its own text for every scenario, from the CLI or
`run_benchmark(system_prompt=...)`. The benchmark reference-date line is still appended
after the override and remains authoritative, because relative-time scenarios depend on it.
An override is persisted in the run config and folded into `config_fingerprint`, so runs
with different prompts land in different comparison cohorts, and a `--resume` that changes
the prompt — including resuming a run recorded before this option existed — is flagged as a
config mismatch. A run without the flag persists and fingerprints exactly as before, so no
historical run is re-cohorted. The prompt is capped at 32 KiB, must be valid UTF-8, and is
ignored with a warning on invocations that run no scenarios (`--perf-only`, a plugin, a
lone `--spec-bench`, `--skip-tool-eval`). The run report marks that a custom prompt was
used, without reproducing it.
