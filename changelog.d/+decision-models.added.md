**Decision-model benchmark** — `--decision`, `--decision-only`, and `plugin decision` score models
served on llama.cpp's `/v1/systemone`, which answer by scoring options in one forward pass instead
of generating text. A run sends 111 requests (63 base items across routing, moderation, urgency,
yes/no, and abstention, plus reversed-order and opaque-label routing variants) and reports accuracy,
calibration (ECE, Brier score, log loss, confident mistakes), a reliability table, a routing
confusion matrix, urgency error, option-order robustness, and latency, with a probability trace for
every request. `--decision-only` skips the chat preflight and warmup, since a decision model may have
no chat endpoint. A server without `/v1/systemone` aborts the run with a message rather than scoring
every item as a miss. See `docs/decision-models.md`.
