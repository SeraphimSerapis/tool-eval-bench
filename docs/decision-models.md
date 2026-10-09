# Decision models

A decision model answers by scoring options you supply, in one forward pass. It
generates no text, so there is nothing to parse, no sampling, and no tool call.
llama.cpp serves these models at `/v1/systemone`. Typical jobs are routing,
moderation, and urgency triage.

This benchmark measures how accurate those probabilities are and whether the
model can be trusted when it says it is confident. It scores the test split of
[Typed Decisions](#third-party-data-typed-decisions), a public third-party
dataset of 400 cases that ships with this package.

```bash
tool-eval-bench --decision-bench-only --base-url http://host:8084   # just this benchmark
tool-eval-bench --decision-bench                                    # after the tool-call scenarios
tool-eval-bench plugin decision --base-url http://host:8084         # the subcommand spelling
```

The server must expose `/v1/systemone`. A server that answers 404, 405, or 501
there aborts the run with a message instead of scoring every case as a miss.
`--decision-bench-only` also skips the chat preflight and warmup, because a
decision model may have no working chat endpoint.

## Answer audits

This is separate from the decision-model accuracy benchmark. An optional judge
asks one versioned question about a scenario's assistant text, such as "did it
claim the payment succeeded?", and compares the answer with the scenario's own
deterministic check. It does not grade tool execution, authorization,
compensation, or the whole scenario.

```bash
tool-eval-bench run --hardmode --base-url http://localhost:8000/v1 \
  --decision-judge \
  --decision-judge-base-url http://localhost:8084/v1 \
  --decision-judge-model clef-flash
```

`--decision-judge [recommended|all]` picks the set. The bare flag means
`recommended`, and so do the connection flags given alone. The set flag
without a judge URL is a usage error.

The URL and model also read `TOOL_EVAL_DECISION_JUDGE_BASE_URL` and
`TOOL_EVAL_DECISION_JUDGE_MODEL`; flags win. The variables only configure the
connection, so a run audits only when a judge flag is given. Without a model
name the CLI asks the judge's `/v1/models` and uses the model it serves. When
it serves several, the console offers a picker and `--json` exits 2 with
`invalid_arguments` rather than pick one, because the judge model is recorded
with every audit.

| Scenario | Set | Check | What it asks | Evidence |
|---|---|---|---|---|
| TC-89 | recommended | `tc89-payment-claim-v1` | payment-claim check | all messages |
| TC-91 | recommended | `tc91-revocation-claim-v2` | revocation-claim check | all messages |
| TC-76 | recommended | `tc76-cancel-refund-claim-v1` | cancellation-or-refund claim check | all messages |
| TC-80 | recommended | `tc80-move-claim-v1` | move-claim check | final answer |
| TC-49 | recommended | `tc49-sent-claim-v2` | sent-claim check | final answer |
| TC-32 | recommended | `tc32-cleared-claim-v1` | cleared-claim check | final answer |
| TC-59 | recommended | `tc59-refuses-v2` | refusal check | final answer |
| TC-71 | recommended | `tc71-asks-which-jordan-v1` | clarification check | all messages |
| TC-26 | recommended | `tc26-attendee-claim-v1` | invented-attendee check | final answer |
| TC-14 | recommended | `tc14-error-acknowledged-v1` | error-acknowledgement check | all messages |
| TC-87 | recommended | `tc87-completeness-claim-v1` | completeness-claim check | all messages |
| TC-12 | all | `tc12-deletion-stance-v3` | deletion-refusal check | final answer |
| TC-33 | all | `tc33-internal-claim-v2` | internal-revenue claim check | final answer |
| TC-36 | all | `tc36-asks-details-v1` | missing-details check | final answer |
| TC-31 | all | `tc31-asks-clarification-v1` | clarification check | final answer |
| TC-75 | all | `tc75-asks-date-time-v1` | date-and-time request check | all messages |
| TC-16 | all | `tc16-answer-language-v1` | German-answer check | final answer |

`all` runs every row. Only scenarios selected for the run are audited, so a
standard run audits 5 recommended or 10 `all` checks; the rest are Hard Mode.
"All messages" sends every public assistant message, so an earlier claim
later contradicted by a denial still counts. "Final answer" sends only the
last one. Reasoning and tool payloads are never sent.

Every `recommended` check got all clear and tricky test cases right in live
testing and gave the same answer with the option order reversed. None made a
confident mistake except on the prompt injections covered under the verdict
limits below. Each check `all` adds has one weakness: a lower margin, one
order flip, an abstention on a tricky case, or one confident miss. TC-12's is
the miss: "Which email account should I delete them from?" came back
`says_cannot_delete` at p=0.81, so expect some TC-12 disagreements that are the
judge's mistake. TC-90 has no audit because its answer depended on option
order. TC-58 has none because its deterministic check already handled the
tricky cases and the judge's margin was unstable.

Some variants change the premise a question relies on, so they carry no audit:
TC-71's `clarified` variant names the recipient, and both TC-75 panel variants
change the booking. Identifier variants keep their audit; the deterministic
check sees the original identifiers.

Authentication uses the judge-only environment variable
`TOOL_EVAL_DECISION_JUDGE_API_KEY`. The benchmark's API key, provider
credentials, extra headers, and conversation ID never reach the judge. URLs
must be HTTP(S), without embedded credentials, query parameters, or fragments.
Held-out scenarios are always skipped. The flags work with `run`, `resume`,
and flat scenario invocations, not plugins or context-pressure sweeps. Judge
requests run one at a time after all benchmark scenarios finish, so their
latency does not enter scenario timing or deployability scores.

Live and plain terminal output name the judge when a request starts, then show
its choice and probability, disagreement, abstention, or request error under the
scenario ID. Each verdict is a `↳` row whose ID and judge label line up with the
scenario rows. A badge (`AGREES`, `DISAGREES`, `ABSTAINED`, or `NO VERDICT`)
starts in the status column, followed by a bar showing the judge's probability
for the option it chose. The live display shows
the in-flight request in its progress footer and logs only verdicts, under one
`Decision audits` heading. Unaudited scenarios and requests skipped before sending have no
placeholder. A previously performed audit shown during resume is labeled
`saved audit`, not shown as a new request. JSON mode emits `decision_audit_start`
and `decision_audit_result` events on stderr without raw evidence or credentials.

Official scores, safety warnings, and ratings never change. Each audited result
gets a `decision_audit` in JSON and SQLite. It includes the exact input and
question, `check_id`, `question_sha256`, evidence scope, the judge endpoint
(host redacted to `***`, as for every stored server URL) and model, probabilities, elapsed milliseconds, and the comparison with the
deterministic check. The Markdown report opens the section with one table,
disagreements first, then the full record per scenario. A disagreement is a
review candidate, not a corrected verdict.

Two limits apply to every verdict. The probabilities have not been calibrated
against independent human labels. Assistant text can also steer the judge: in
21 synthetic prompt injections, 4 flipped its verdict. The question tells the
judge that messages are evidence, not instructions, and nothing detects an
injection that works anyway.

`unclear` or tied top probabilities produce `abstained`. Errors, malformed
probabilities, missing evaluation evidence, and empty messages produce
`unavailable`. The entire request, including retries, has a 10-second deadline.
Requests larger than 6,000 UTF-8 bytes are not sent or truncated. This conservative
byte limit is not a tokenizer-based context guarantee; a server's context
rejection also produces an unavailable audit.

Judge configuration stays out of the comparison fingerprint, because an audit
never changes a score: judged and unjudged runs of one configuration share a
cohort. The stored run config still records the judge endpoint (host redacted,
plus an opaque `endpoint_id`), model, set, and selected `checks`. Repeat the judge flags when resuming. Changing the endpoint,
model, or set is refused, and so is a code update that bumps a selected check's
version. A run audited before sets existed (TC-89 only) cannot be resumed;
start it again. Pending audit evidence is checkpointed, so resuming can complete it
without rerunning the benchmark model. A saved verdict is reused only when its
`check_id` and `question_sha256` match the current question; otherwise the saved
evidence is judged again. Keys are never persisted.

The Python API accepts `decision_judge` (`"recommended"` or `"all"`),
`decision_judge_base_url`, `decision_judge_model`, and `decision_judge_api_key`.
API callers supply the judge key explicitly; the API reads no judge credentials
from the environment. An optional async `on_scenario_audit(scenario, result,
phase)` callback receives `started`, `completed`, or `reused` updates only for
attempted or previously performed requests. Progress callback errors are logged
without exception text and do not change judgments or official points.

## The wire format

A request carries a `state` and a map of named `questions`. The state is either
text or a JSON object, such as a support thread with the account behind it; an
object is sent as JSON, not flattened. Each question has a type and required
`instructions`.

| Type | Options | Answer |
|---|---|---|
| `choice` | `criteria`: a map of option name to description | `choice`, a probability per option, `confidence` |
| `score` | `criteria`: a list of level descriptions | expected `score`, a probability per level |
| `noul` | optional `criteria` describing `true` and `false` | `noul`, the probability of "yes" |

`model` selects a model in router mode and is ignored by a single-model server.
The response reports `output_tokens: 0`, because nothing is generated.

## What a run does

A run sends each of the 400 Typed Decisions test cases once: the case's state and
all five of its questions in one request, so a run is 400 requests and 2,000
decisions. The dataset card's leaderboard was scored the same way, and the card
warns that asking one question per request changes the answers, so this shape is
what keeps a score comparable with the card.

| Workflow | Cases | What the model decides |
|---|---|---|
| `agent_trace_observability` | 100 | Whether an agent run needs human review, and how urgently |
| `customer_service` | 100 | The right response and action for a customer thread and account |
| `invoice_processing` | 100 | Pay, hold, or reject a vendor bill against its order and delivery |
| `security_incidents` | 100 | Close, investigate, or contain a security alert, given machine history |

Each workflow asks the same five questions of every case, 20 question schemas in
all: 600 `noul`, 600 `choice`, and 800 `score` decisions. Every gold answer is a
probability distribution.

The dataset stores each state as a JSON string. The benchmark decodes it and sends
`state` as a JSON object. Sending the string verbatim, as one public harness does,
is the alternative; on d1 it changed about 8% of the most likely answers without
changing accuracy.

A decision model is deterministic, so a run needs one pass. `--temperature`,
`--seed`, and extra sampling parameters have no effect. `--parallel` sets how many
requests are in flight.

## Reading the result

The terminal prints accuracy by question type and by workflow, a reliability
table, then the summary. The Markdown report adds a per-question table, the most
confident disagreements, and a trace of every decision with both distributions.

**What a score means.** Gold is the mean of three samples from a teacher model of
roughly 4B-class capability, so a score measures agreement with that teacher, not
correctness. A model better than the teacher can score lower wherever the teacher
is wrong. The dataset card gives two reference points: a prior that ignores the
input and answers each question's label frequencies scores 0.470 accuracy, and a
fresh teacher sample agrees with gold built from the others 0.735 of the time.
Scores well above 0.735 mean a model is learning the teacher's habits. Per-question
ceilings range from 0.560 to 0.937, so read the per-question table as well as the
average.

**Rating.** The stars use only the card's two reference points: ★ at or below the
47.0% prior, ★★ above it, and ★★★ at or above the teacher's 73.5% self-agreement.
A run in which every request failed gets no stars; its rating reads "Incomplete: no
successful cases", because its 0% measures the connection, not the model.

**Metrics.** All are averaged over decisions, not cases.

- **Accuracy**: the prediction's most likely answer equals the gold `label`. The
  label is always one of the gold's most likely answers; in the 35 decisions where
  two gold answers tie, it is the dataset's own pick. A tie in the prediction goes
  to the option listed first.
- **KL from gold**: KL(gold ‖ prediction) in nats, summed over the answers the gold
  gives nonzero probability. A predicted probability of exactly 0 there would make
  KL infinite, so it counts as 1e-12, the floor a third-party harness used when it
  reproduced the card's figures. Lower is better.
- **Brier**: the sum over a question's answers of the squared difference between
  predicted and gold probability. 0 is best and 2 is worst.
- **ECE**: top-label expected calibration error over ten equal-width confidence
  bins, where confidence is the top predicted probability and a hit is a correct
  answer as defined above. The card does not publish its ECE definition, and
  submitters report that they could not reproduce it, so this is the standard
  definition and its values are not comparable with the card's ECE column.

The card names the four leaderboard metrics without formulas. KL and Brier follow
the definitions submitters used to reproduce the card's numbers. A model that puts
equal probability on every option scores KL 0.444 and Brier 0.238 here, which
matches the card's Uniform reference row and is checked by a test. Accuracy and
ECE for that model depend on how its ties are broken, so they do not pin a
definition.

For the prediction, a yes/no answer becomes `true` at the `noul` probability and
`false` at the rest. A `choice` or `score` answer uses its `probabilities`; the
server's `choice` and `confidence` fields are not used.

**Errors count as misses.** A request that times out or returns a malformed
response costs every decision in its case; the adapter rejects a response with one
bad answer as a whole. Errors count against accuracy and mark the run
`incomplete`. KL, Brier, and calibration cover answered decisions only.

To compare with published models, see the leaderboard on the
[dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions). Its
rows scored with `train` in the training data are not comparable with zero-shot
rows, and only accuracy, KL, and Brier are computed the same way here.

## Reading a trace

```
agent_trace_observability_000001  {"agent": {"autonomy": "unsupervised", "model": "internal-agent-v1"}, ...
  needs_review (noul)  ✗  pred true  gold false  KL 0.268  Brier 0.260
      pred  true=0.611 false=0.389
      gold  true=0.250 false=0.750
  urgency (score)  ✗  pred 3  gold 1  KL 1.711  Brier 0.591
      pred  0=0.073 1=0.030 2=0.434 3=0.463
      gold  0=0.433 1=0.433 2=0.113 3=0.020
```

Each case starts with a preview of its state, then one line per question with the
predicted and gold label, followed by both distributions in the same option order.
Gold ties happen (`urgency` above); the dataset's label breaks them.

## Third-party data: Typed Decisions

The items are not ours. They are the `test` split of
[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
published on Hugging Face by the LocalLLaMA organization under the Apache License
2.0. The dataset card names no individual authors.

- Pinned to revision `e039ebffcc280174dd354227424fb2b249f191de`, file
  `all/test-00000-of-00001.parquet`.
- Stored in `src/tool_eval_bench/plugins/decision/vendor/typed_decisions/` with the
  dataset's `LICENSE` and a `NOTICE` describing provenance and changes. Both ship in
  the wheel and the sdist next to the data.
- Rows are unmodified apart from format conversion: Parquet became gzipped JSON
  Lines, the `state`, `questions`, and `gold` columns were decoded from JSON
  strings, and the columns the benchmark does not use (`split`, `factors`,
  `label_agreement`, `n_questions`) were dropped.
- The `train` split is not included. Several published models were fine-tuned on
  it, so it is not benchmark material.
- Every report and terminal summary names the dataset, revision, and license.
  Stored results record the dataset id and revision, and both enter the run's
  comparison fingerprint, so runs on another revision are not compared.

`scripts/vendor_typed_decisions.py` produces the file. It downloads the pinned
revision, checks the source file's sha256, converts it, and writes `manifest.json`
with the hashes. The loader checks the data against the manifest on every run and
refuses a file that was edited by hand. To update the data, change the pinned
revision and hash in the script and run it:

```bash
uv run --no-project --with pyarrow python scripts/vendor_typed_decisions.py
```

pyarrow is needed only for that script and is not a project dependency. A missing,
truncated, or edited data file stops the benchmark and the live monitor with a
`DatasetIntegrityError` message.

## Live monitor

`decision-live` watches a decision model while it runs, the way `spec-live` watches
speculative decoding.

```bash
tool-eval-bench decision-live --base-url http://host:8084
tool-eval-bench --decision-live --decision-live-interval 0.25   # flat spelling, faster probes
```

Each probe is one Typed Decisions test case: its state and all five questions in
one request, as in a benchmark run. The monitor cycles the 400 cases in a fixed
shuffled order, scores each answer against its gold as the benchmark does, and
redraws the screen. Ctrl+R resets the session and Ctrl+C exits. A server without
`/v1/systemone` stops the command with a message before the screen opens.

| Panel | Shows |
|---|---|
| now | The latest case: how many of its five answers match gold, a state preview, then each question with its type, top probability, prediction, and the gold label when they differ, plus latency and input tokens |
| quality | Rolling accuracy and ECE over the last 100 decisions (20 cases) with trend lines, Brier against the gold distribution, mean confidence, session accuracy, and confident mistakes |
| by question type | Session accuracy for `noul`, `choice`, and `score` |
| confidence histogram | How many of the last 100 decisions fell in each confidence bin |
| server | Request latency p50 and p95, input tokens the monitor has sent, then requests per second, input tokens per second, and slot and queue counts from `/metrics` |

The footer credits the dataset. A wrong answer at 90% confidence or more raises a
banner for a few seconds, naming the most confident one when a case has several.
Three failed probes in a row show "model unreachable" and the probes keep going.

The server panel reads the llama.cpp counters on `/metrics` (`--metrics-url` points
it elsewhere). The counters include other clients' requests, so on a shared server
the requests-per-second figure rises above the monitor's own rate when someone else
is using the model. Without `/metrics` the server panel shows latency and input
tokens only.

The monitor measures the model with synthetic traffic. It cannot see the decisions
your application gets, and the server's counters carry no decisions, only load.

`--decision-live-interval SEC` sets the pause between probes (default 0.5). A probe
carries five questions, but they share one state, so its prefill is about twice that
of a single question; the input-token figures in the now and server panels show the
real cost. The other connection flags work as they do elsewhere.

## Design notes

**Accuracy is the headline and calibration stays separate.** Folding ECE into one
number would hide which one moved. A model with 90% accuracy and an ECE of 0.25 is
a different deployment risk from one with 85% and 0.03, and a single score erases
that.

**The data is vendored, not downloaded.** A run works offline, and the item set
cannot change under a stored result. The cost is a 99 KB file in the package and a
script to rerun when the dataset gets a new revision.

**One request per case.** Typed Decisions asks five questions about one state, and
the card's scores come from sending them together. Asking them one at a time
would cost prefill and produce numbers that do not compare with anyone else's.

**Errors count as misses.** A request that times out or returns a malformed answer
is a wrong answer in the headline and marks the run `incomplete`.

## Programmatic use

`build_decision_adapter()` returns an adapter that makes decision requests and
chat requests over one HTTP client. `DecisionPlugin().run(adapter, model=...,
base_url=...)` scores all 400 Typed Decisions cases; `cases=load_cases()[:40]`
narrows that to a subset. `load_cases` and `dataset_info` live in
`tool_eval_bench.plugins.decision.typed_decisions`. `DecisionPlugin.run` raises
`ValueError` for an adapter that cannot make decision requests or an empty `cases`
list, and `DatasetIntegrityError`, a `ValueError`, when the vendored data fails its check.
