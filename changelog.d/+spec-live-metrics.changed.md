**Spec-live backend metrics:** the live monitor now supports current vLLM and llama.cpp
speculative counters, current SGLang gauges, and per-position acceptance data. It
aggregates counters across engine series, avoids summing replicated gauges, includes the
verifier bonus token in acceptance length, and leaves method or drafter labels unknown
unless the server reports them explicitly. The request benchmark applies the same
acceptance-length convention and aggregates vLLM counter series across engines.
