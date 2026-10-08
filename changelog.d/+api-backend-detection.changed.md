**Python API backend detection**: `run_benchmark()` now identifies the server and records its
engine metadata the same way the CLI does, so its `metadata` matches `tool-eval-bench --json` for the
same server instead of recording `unknown` and a host-only dictionary. Hosted Gemini and Anthropic
endpoints are labelled by their wire format without probing, and an explicit `backend=` is kept. The
backend label and engine facts are part of the comparison fingerprint, so API runs that used to
record `unknown` start a new leaderboard cohort. Pass `probe_engine=False` (new, the equivalent of
`--no-probe-engine`) to send no detection requests, or pass `backend=` to pin the label.
The old API metadata keys are replaced: `host` by `hostname`, `platform` by `platform_info`,
`config.model`, `config.backend` and `config.base_url` by top-level `model`, `backend` and
`base_url`, and the `backend_probe` fields by top-level `server_model_id`, `server_model_root` and
`max_model_len`. `pid` is no longer recorded.
