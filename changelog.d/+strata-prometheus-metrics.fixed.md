**Strata's Prometheus metrics no longer read as vLLM.** Strata's optional Prometheus `/metrics`
format, served for `Accept: text/plain` or `?format=prometheus`, exports vLLM's `vllm:` metric names
next to its own `strata:` namespace. Detection now checks `strata:` before `vllm:`, so such a
response labels the run Strata.
