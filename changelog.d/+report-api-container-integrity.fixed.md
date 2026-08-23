**Reports, API, and containers:** Markdown reports contain hostile trace
fences and escaped table text, persisted endpoint URLs omit hosts and query
credentials, `run_benchmark()` forwards difficulty weighting, and Docker
images retain source version provenance without shipping Git metadata. The
runtime and builder also use one pinned base-image digest.
