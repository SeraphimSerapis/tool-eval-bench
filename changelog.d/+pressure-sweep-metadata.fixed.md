**Context-pressure sweeps and spec-bench runs record the deployment.** `--context-pressure-sweep`
and `--spec-bench` saved their runs with empty metadata, so the database held no engine name or
version, `max_model_len`, quantization, slot count, or thinking setting for them, and `history`
showed no engine. Both now store the same run context that throughput and plugin runs store. The
comparison fingerprint is unchanged: like throughput and plugin runs, their cohort comes from the
config alone, so runs saved before this fix still compare with new ones. Runs already stored keep
their empty metadata.
