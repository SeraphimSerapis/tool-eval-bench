# Benchmark scoring decisions

## Visible correctness, safety outcomes, and fixture identity

TC-88 scores observable constraints independently of exposed reasoning. Transport
observations live in diagnostics because reasoning visibility is endpoint-dependent.
Revisit this split only if a backend-independent observable planning contract replaces it.

Safety warnings come from explicit unsafe outcomes, including recovered intermediate
mutations. Category membership alone cannot distinguish an unsafe action from harmless
incompletion. The existing rating threshold remains, gated on an observed violation.
Historical results need fresh execution to establish the new safety outcome field.

Fixture variants retain scenario IDs but record seed, version, and kind in comparison
fingerprints. This avoids multiplying the public registry for alternate environments.
A variant does not replace a private held-out pack or establish empirical difficulty.
Revisit the representation if variants need independently selectable public identities.

## Decision-model scoring

The decision-model benchmark scores headline accuracy against the gold label on the pinned
Typed Decisions test split and reports KL divergence, Brier score, and ECE beside it, never
folded in. A blended number would hide whether accuracy or calibration moved, and the two
predict different deployment risks.

Each case is one request carrying all five questions about its state. Sending one question
per request was rejected: the dataset card says answers change when questions are split, so
scores would stop being comparable with its published figures. The hand-written items were
dropped rather than kept beside the dataset, because they gave no external reference point.

Revisit when the pinned dataset revision changes or the card publishes its ECE definition.

## Web UI access control

`tool-eval-bench serve` requires a per-process session token, printed as a sign-in link and
exchanged for an `HttpOnly`, `SameSite=Strict` cookie. Requests that change state must also
come from the UI's origin and send `application/json`. The server holds `TOOL_EVAL_API_KEY`
and can start runs that spend it, so a page in another tab must not be able to drive it.

Origin and `Host` checks alone were rejected: they stop cross-site forgery but not a DNS
rebinding page, which runs on its own origin with a matching `Host`. Scoping the environment
key to `TOOL_EVAL_BASE_URL` was rejected too, because the CLI sends it to any `--base-url` and
the two entry points would disagree. The token closes both paths for every bind address.

The UI stops only runs it launched and resumes only `interrupted` runs whose stored config it
can rebuild, refusing the rest with a pointer to `tool-eval-bench resume`. Revisit if the UI
gains multi-user access, which would need real accounts instead of one shared token.
