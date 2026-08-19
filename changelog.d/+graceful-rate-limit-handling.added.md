**Graceful rate-limit handling** — hosted endpoints with per-minute quotas
(Gemini, OpenAI, and similar) no longer turn a benchmark run into a string of
infrastructure failures. HTTP 429 now draws on its own retry budget (6 by
default, separate from the 2 generic transient retries), honors `Retry-After`
up to a full 60s quota window, and backs off exponentially with half jitter.
A rate limit observed by one request pauses every in-flight request, and the
adapter then paces subsequent requests apart — widening on each 429, decaying
back to unthrottled after sustained success — so retries do not walk straight
back into the same limit. Pacing stays completely off until a 429 is actually
seen, so local vLLM / llama.cpp runs are unaffected. Throttling is reported in
the live progress footer and as a one-line note under the results
(`⏳ Rate limited  12 retries, 38s waiting on the endpoint's quota`) instead of
interleaving retry log lines with scenario results.
