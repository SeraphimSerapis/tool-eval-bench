**One endpoint, one cohort, however its base URL is written.** `http://host:8000`,
`http://host:8000/`, `http://host:8000/v1` and `http://host:8000/v1/` send identical requests but
were four comparison cohorts, and `--resume` refused to continue a run under another spelling.
The endpoint identity now hashes the root the requests are built from, and the redacted
`base_url` no longer enters `config_fingerprint`. A native Gemini base keeps its API version,
because a bare Gemini host means `v1beta`. As with the mode-run fingerprint change in
[#264](https://github.com/SeraphimSerapis/tool-eval-bench/pull/264), new runs do not group with
runs stored by earlier versions. A run started before upgrading still resumes from the URL it was
started with.
