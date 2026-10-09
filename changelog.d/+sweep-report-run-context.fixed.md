**Context-pressure sweep and spec-bench reports show the inference engine.** The Markdown reports
for `--context-pressure-sweep` and `--spec-bench` under `runs/YYYY/MM/` now carry the
tool-eval-bench version and the same Inference Engine table that scenario and throughput reports
show: engine name and version, `max_model_len`, quantization, GPU and slot counts, spec decoding,
and the host. The database already stored this context. These reports leave out the CLI parameter
table, because both modes run with their own temperature, timeout, and concurrency, so that table
would misstate the run. Reports already written are unchanged.
