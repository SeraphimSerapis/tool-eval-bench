**`decision-live`** — a live terminal monitor for decision models, in the style of `--spec-live`
(`tool-eval-bench decision-live`, or `--decision-live`). Each probe sends one Typed Decisions test
case, all five questions in one request, to `/v1/systemone`, cycling the 400 cases in a fixed
shuffled order. The screen shows each answer of the latest case against gold, rolling accuracy and
calibration error with trend lines, Brier against the gold distribution, a confidence histogram,
accuracy per question type, latency, and input tokens. A server panel adds requests per second,
input tokens per second, and slot and queue counts from llama.cpp `/metrics`, which include other
clients' traffic. A wrong answer at 90% confidence or more raises a banner. Ctrl+R resets the
session. `--decision-live-interval` sets the pause between probes. See `docs/decision-models.md`.
