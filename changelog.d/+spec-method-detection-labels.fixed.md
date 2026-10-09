**Spec-decode method no longer read from model names.** `--spec-bench` and the `--perf` probe
named the method `eagle`, `ngram`, or `mtp` whenever that word appeared anywhere in `/metrics`, so a
vLLM server serving `acme/eagle-7b` or `deepseek-ai/DeepSeek-V3-MTP` reported that method whatever
it actually drafted with. None of vLLM, SGLang, or llama.cpp puts the method in its metrics, so
these servers now report `unknown` unless you pass `--spec-method`. Detection and `--spec-live` now
share one rule: only a `spec_method` or `speculative_method` label, or a `method` label on a
speculative series, names the method. `--spec-live` also stops reading the method from HELP text
and from inside other label values. Strata still reports `mtp`.
