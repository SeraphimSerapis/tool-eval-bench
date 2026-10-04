**`decision-live`** — a live terminal monitor for decision models, in the style of `--spec-live`
(`tool-eval-bench decision-live`, or `--decision-live`). It sends one canary item at a time to
`/v1/systemone`, scores it against its gold label, and redraws the probability bars for the latest
answer, rolling accuracy and calibration error with trend lines, a confidence histogram, per-category
accuracy, and latency. A server panel adds requests per second, input tokens per second, and slot and
queue counts from llama.cpp `/metrics`, which include other clients' traffic. A wrong answer at 90%
confidence or more raises a banner. Ctrl+R resets the session. `--decision-live-interval` sets the
pause between probes. See `docs/decision-models.md`.
