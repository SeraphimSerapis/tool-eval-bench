Reports under `runs/`, stored rows, `--json` error output, and `probe_result` events no longer contain the server URL's credentials or host. These paths wrote them unredacted:

- The context-pressure sweep report's `Server` line held the raw `--base-url`, userinfo and query string included, unless `--redact-url` was passed.
- httpx's status-error message quotes the full request URL. It reached scored-run traces and stored scores, sweep scenario traces, and sweep level errors in both the report and the stored row.
- llama-benchy cell errors in throughput reports and stored scores carried llama-benchy's own request errors as-is.
- The headless `--json` error envelope, the stderr `error` events, and both `probe_result` events quoted the raw URL or the status-error message.

Every URL written to these outputs is now redacted the same way the stored config already was. `--redact-url` only affects the console.
