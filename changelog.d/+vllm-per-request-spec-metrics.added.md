`--spec-bench` reads vLLM's per-request `metrics.speculative_decoding` response field when the
server runs with `--per-request-spec-decode-metrics`, and prefers it over Prometheus counter
deltas: it is exact and scoped to the request, so concurrent traffic no longer skews acceptance
rate and acceptance length. `detailed` mode also yields a per-position acceptance curve in the
summary and the Markdown report. Both now name the acceptance source. Servers without the flag
keep the Prometheus and llama.cpp timings paths unchanged, and the cross-talk warning now fires
only when a run actually used Prometheus deltas.
