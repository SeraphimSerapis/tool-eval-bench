**Daily quota exhaustion no longer wastes the retry budget** — Google's
Gemini API reports a per-day quota limit as a plain HTTP 429, the same as a
per-minute one, and even attaches a `RetryInfo` delay that looks like normal
backoff advice. Retrying inside any request-level budget cannot help until
the quota resets, so a scenario against an exhausted daily quota used to
burn all 6 rate-limit retries (minutes) before the surrounding per-scenario
timeout fired first, reporting an uninformative `timeout` rather than the
real cause. A daily-quota 429 is now detected from the response body and
fails immediately with a `Daily quota exhausted for <model> (limit: N/day)`
log line instead. Also: the Retry-After hint used to require the standard
header; Gemini instead sends it as `RetryInfo.retryDelay` in the JSON error
body, which backoff now reads too.
