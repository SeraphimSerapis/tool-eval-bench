**Decision-model benchmark** — `--decision-bench`, `--decision-bench-only`, and `plugin decision`
score models served on llama.cpp's `/v1/systemone`, which answer by scoring options in one forward
pass instead of generating text. A run sends the 400 cases of the `test` split of
[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (LocalLLaMA on
Hugging Face, Apache-2.0), one request per case with its state and all five questions, as on the
dataset card's leaderboard. Each of the 2,000 decisions is scored against a soft gold distribution:
accuracy against the gold label, KL from gold, Brier, top-label ECE, and confident disagreements,
overall and by question type, workflow, and question, with a reliability table, latency, and the
predicted and gold distribution of every decision. The rating uses the card's prior (47.0%) and
teacher self-agreement (73.5%) as its thresholds, and a run in which every request failed is
rated incomplete instead. The data is vendored at a pinned revision with
its license and a notice of the format conversion, checked against a manifest on every run, and
credited in every report and terminal summary; the dataset id and revision enter stored results and
the comparison fingerprint. `--decision-bench-only` skips the chat preflight and warmup, since a
decision model may have no chat endpoint. A server without `/v1/systemone` aborts the run with a
message rather than scoring every case as a miss. See `docs/decision-models.md`.
