**`--system-prompt` / `--system-prompt-file`** — a run can now replace the built-in
"helpful assistant" system prompt with its own text for every scenario. The benchmark
reference-date line is always kept (relative-time scenarios depend on it). The override is
part of the run's `config_fingerprint` and persisted config, so runs with different prompts
land in different comparison cohorts, and a `--resume` that changes the prompt is flagged as
a config mismatch. The run report marks a custom prompt in the Run Context table.
