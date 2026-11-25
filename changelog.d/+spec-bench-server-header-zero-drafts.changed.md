**Spec-bench zero-draft responses without `/metrics`**: a llama.cpp or Splash response whose
`timings` object omits `draft_n` now counts as an exact zero when the response's own `Server` header
names the engine, also when `/metrics` is unavailable (llama-server without `--metrics`, or a proxy
that hides it). Before, that zero was inferred only when `/metrics` identified llama.cpp. Both
engines write `draft_n` only for a request that drafted.
