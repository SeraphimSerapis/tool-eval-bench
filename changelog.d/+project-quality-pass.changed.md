**Project quality pass.** Hardened runtime reliability, query performance,
module structure, test hygiene, and documentation across the repository:

- **Bug fixes and runtime hardening.** Propagated cancellation (`BaseException`)
  through concurrent plugin runs (`gsm8k`, `mmlu`, `ifeval`, `needle`,
  `toolscale`), migrated GSM8K's REST fallback to `hf_utils.download_rows_paginated`
  for retries and partial-file resume, classified `httpx.NetworkError`,
  `ProtocolError`, and `ProxyError` as `connection_error`, unified
  `LOCALE_VERSION = "2"` under `domain/rubric.py`, fixed `RateLimitCoordinator`
  slot accounting on rate-limit updates, scaled `httpx` connection pools from
  configured concurrency, restricted Anthropic string-to-list wrapping in
  `extra_params` to `stop_sequences`, ensured SQLite handles close cleanly on
  scored-run exit, filtered completed `tool_eval` runs in SQL for leaderboard
  and export (returning `[]` on empty JSON export), and surfaced paired
  `comparability_warnings` in CLI `compare` and the Web UI Compare view.
- **Storage and throughput performance.** Upgraded SQLite storage to schema
  version 6 with composite and partial indexes on `(run_type, status, created_at DESC)`,
  `(created_at DESC)`, and `(report_path)`, added bounded session-prefix seeks,
  lightweight `get_latest()`, and batch `get_runs()`, parsed Prometheus text in
  a single linear pass in `runner/spec_live.py`, offloaded synchronous git,
  SQLite, and report I/O off the event loop, and cached symmetric TF-IDF cosine
  pairs in `plugins/toolscale/`.
- **Module decomposition.** Split large modules behind compatibility facades:
  `runner/orchestrator/` (`loop`, `probes`, `review_hook`, `scoring`, `wire`),
  `evals/helpers/` (`_text`, `_intent`, `_state`, `_dates`, `_mocks`),
  `storage/reports/card/` panel submodules (`_primitives`, `_score_panels`,
  `_perf_panels`, `_needle_panel`), `plugins/toolscale/` (`aggregate`, `report`),
  native ES modules under `web/static/`, and focused CLI modules for dispatch,
  plugin runners, and the `spec-live` dashboard, with per-module branch coverage
  floors enforced in `scripts/check_module_coverage.py`.
- **Tests and documentation.** Streamlined redundant tests, sped up slow test
  harnesses, tightened `README.md` and `AGENTS.md`, and aligned `docs/`,
  `CONTRIBUTING.md`, and `RELEASING.md` with current scoring diagnostics,
  rubric-versioned cohorts, SQLite schema v6, and CLI and API options.
