# Decision models

A decision model answers by scoring options you supply, in one forward pass. It
generates no text, so there is nothing to parse, no sampling, and no tool call.
llama.cpp serves these models at `/v1/systemone`. Typical jobs are routing,
moderation, and urgency triage.

This benchmark measures how accurate those probabilities are and whether the
model can be trusted when it says it is confident.

```bash
tool-eval-bench --decision-bench-only --base-url http://host:8084   # just this benchmark
tool-eval-bench --decision-bench                                    # after the tool-call scenarios
tool-eval-bench plugin decision --base-url http://host:8084         # the subcommand spelling
```

The server must expose `/v1/systemone`. A server that answers 404, 405, or 501
there aborts the run with a message instead of scoring every item as a miss.
`--decision-bench-only` also skips the chat preflight and warmup, because a
decision model may have no working chat endpoint.

## Answer audits

This is separate from the decision-model accuracy benchmark. An optional judge
checks TC-89's assistant messages for a claim that payment succeeded. It does
not grade tool execution, authorization, compensation, or the whole scenario.

```bash
tool-eval-bench run --scenarios TC-89 --base-url http://localhost:8000/v1 \
  --decision-judge-base-url http://localhost:8084/v1 \
  --decision-judge-model clef-flash
```

Both judge flags are required. Authentication uses the judge-only environment
variable `TOOL_EVAL_DECISION_JUDGE_API_KEY`. The benchmark's API key, provider
credentials, extra headers, and conversation ID never reach the judge. URLs
must be HTTP(S), without embedded credentials, query parameters, or fragments.
Other scenarios do not contact the judge; held-out scenarios are always skipped.
The flags work with `run`, `resume`, and flat scenario invocations, not plugins
or context-pressure sweeps.

The versioned question offers `payment_claim`, `no_payment_claim`, and `unclear`.
It examines all public assistant messages, including earlier assertions later
contradicted by a denial. Reasoning and tool payloads are not sent. Judge requests
start after all benchmark scenarios finish, so their latency does not enter
scenario timing or deployability scores.

Live and plain terminal output name the judge when a request starts, then show
its choice and probability, disagreement, abstention, or request error under the
scenario ID. Unaudited scenarios and requests skipped before sending have no
placeholder. A previously performed audit shown during resume is labeled
`saved audit`, not shown as a new request. JSON mode emits `decision_audit_start`
and `decision_audit_result` events on stderr without raw evidence or credentials.

Official scores, safety warnings, and ratings never change. Each eligible result
gets a `decision_audit` in JSON and SQLite, also rendered in the Markdown report.
It includes the exact input and question, check version, judge endpoint and model,
probabilities, elapsed milliseconds, and comparison with the deterministic claim
check. A disagreement is a review candidate, not a corrected verdict. These
probabilities have not been calibrated against independent human labels.

`unclear` or tied top probabilities produce `abstained`. Errors, malformed
probabilities, missing evaluation evidence, and empty messages produce
`unavailable`. The entire request, including retries, has a 10-second deadline.
Requests larger than 6,000 UTF-8 bytes are not sent or truncated. This conservative
byte limit is not a tokenizer-based context guarantee; a server's context
rejection also produces an unavailable audit.

Judge configuration joins the comparison fingerprint. Repeat the connection
flags when resuming; changing or dropping them is refused. Pending audit evidence
is checkpointed, so resuming can complete it without rerunning the benchmark model.
Completed audits are preserved. Keys are never persisted.

The Python API accepts `decision_judge_base_url`, `decision_judge_model`, and
`decision_judge_api_key`. API callers supply the judge key explicitly; the API
reads no judge credentials from the environment. An optional async
`on_scenario_audit(scenario, result, phase)` callback receives `started`,
`completed`, or `reused` updates only for attempted or previously performed
requests. Progress callback errors are logged without exception text and do not
change judgments or official points.

## The wire format

A request carries a `state` (the text to judge) and a map of named `questions`.
Each question has a type and required `instructions`.

| Type | Options | Answer |
|---|---|---|
| `choice` | `criteria`: a map of option name to description | `choice`, a probability per option, `confidence` |
| `score` | `criteria`: a list of level descriptions | expected `score`, a probability per level |
| `noul` | none | `noul`, the probability of "yes" |

`model` selects a model in router mode and is ignored by a single-model server.
The response reports `output_tokens: 0`, because nothing is generated.

## What a run does

Every item is one request with one question. There are 63 base items, plus 48
routing variants that test robustness, so a run is 111 requests.

| Category | Question type | Items | What it tests |
|---|---|---|---|
| routing | `choice`, 4 teams | 24 | 16 plain items, plus 8 harder ones with negation, typos, and no keyword to match |
| moderation | `choice`, 3 classes | 10 | Normal, spam, and abusive messages |
| urgency | `score`, 4 levels | 9 | Whether the expected level lands near the gold level |
| yes-no | `noul` | 12 | Anger, refund requests, and personal contact details |
| abstain | `choice` with a `none` option | 8 | Whether the model declines when no option fits |

The two routing variants ask each routing item again with different options.
`shuffled` lists the options in reverse order. `opaque` renames them to
meaningless labels and keeps only the descriptions, so a model that memorised the
word "billing" fails and one that reads the criteria passes.

The headline score is accuracy on the 63 base items. Variants never count toward
it. They feed the robustness section.

A decision model is deterministic, so a run needs one pass. `--temperature`,
`--seed`, and extra sampling parameters have no effect. `--parallel` sets how many
requests are in flight.

## Reading the result

The terminal prints three tables, then the summary. The Markdown report holds the
same views plus a trace of every request.

**Accuracy by category.** Shows where a model is weak. A small model that routes
perfectly but misreads urgency has a different profile from one that is uniformly
mediocre.

**Calibration.** The probabilities are the product, so they should mean what they
say. The report gives:

- **ECE**, the count-weighted gap between confidence and accuracy over ten
  confidence bins. 0 is perfect.
- **Brier score**, the squared distance from the one-hot gold answer. 0 is best and
  2 is worst.
- **Log loss**, which punishes a low probability on the gold answer.
- **Confident mistakes**, the number of wrong answers given at 90% confidence or
  more. These are the failures that cause damage in a router, because nothing
  flags them for review.

The reliability table groups predictions by confidence. A calibrated model has
accuracy close to confidence in every row. Accuracy above confidence means the
model under-claims. Accuracy below it means the model over-claims.

Confidence here is the probability of the predicted answer, taken from
`probabilities`. The server's own `confidence` field is a different figure and is
kept in the raw response.

**Routing confusion matrix.** Gold team on the rows, predicted team on the columns.
A prediction outside the option list gets its own column.

**Urgency scale.** Mean absolute error of the expected level against the gold
level, and the share of items within one level. Exact-match accuracy alone treats
"today" for "right now" the same as "can wait" for "right now".

**Robustness.** For each variant, accuracy and invariance. Invariance is the share
of answers that match the model's answer to the original item, right or wrong.
Low shuffled invariance means position bias.

## Reading a trace

```
urg-01  ✗  It would be nice to have a dark mode at some point.
    0  ░░░░░░░░░░ 0.031  ◀ gold
    1  █░░░░░░░░░ 0.107
    2  █░░░░░░░░░ 0.133
    3  ███████░░░ 0.729  ◀ pred
```

Choice options are listed most likely first. Score levels are listed in scale
order. `◀ gold` marks the correct answer and `◀ pred` the model's pick.

## Live monitor

`decision-live` watches a decision model while it runs, the way `spec-live` watches
speculative decoding.

```bash
tool-eval-bench decision-live --base-url http://host:8084
tool-eval-bench --decision-live --decision-live-interval 0.25   # flat spelling, faster probes
```

It sends one built-in item at a time, in a fixed mixed order, scores the answer
against its gold label, and redraws the screen. Ctrl+R resets the session and Ctrl+C
exits. A server without `/v1/systemone` stops the command with a message before the
screen opens.

| Panel | Shows |
|---|---|
| now | The latest message, its probability bars, `◀ gold` and `◀ predicted` marks, latency, and input tokens |
| quality | Rolling accuracy and ECE over the last 100 probes with trend lines, mean confidence, session accuracy, and confident mistakes |
| by category | Session accuracy per category |
| confidence histogram | How many of the last 100 answers fell in each confidence bin |
| server | Latency p50 and p95, then requests per second, input tokens per second, and slot and queue counts from `/metrics` |

A wrong answer at 90% confidence or more raises a banner for a few seconds. Three
failed probes in a row show "model unreachable" and the probes keep going.

The server panel reads the llama.cpp counters on `/metrics` (`--metrics-url` points
it elsewhere). The counters include other clients' requests, so on a shared server
the requests-per-second figure rises above the monitor's own rate when someone else
is using the model. Without `/metrics` the server panel shows latency only.

The monitor measures the model with synthetic traffic. It cannot see the decisions
your application gets, and the server's counters carry no decisions, only load.

`--decision-live-interval SEC` sets the pause between probes (default 0.5). The
other connection flags work as they do elsewhere.

## Design notes

**Accuracy is the headline and calibration stays separate.** Folding ECE into one
number would hide which one moved. A model with 90% accuracy and an ECE of 0.25 is
a different deployment risk from one with 85% and 0.03, and a single score erases
that.

**Variants are scored apart from the base items.** They reuse the base messages,
so counting them would weight routing three times as heavily as the other
categories.

**Items are built in and single-question.** There is no download. The wire format
allows several questions per request, which is where a server saves prefill.
This benchmark asks one at a time so each probability maps to one gold label.

**Errors count as misses.** A request that times out or returns a malformed answer
is a wrong answer in the headline and marks the run `incomplete`.

## Programmatic use

`build_decision_adapter()` returns an adapter that makes decision requests and
chat requests over one HTTP client. `DecisionPlugin.run` raises `ValueError` for an
adapter that cannot make decision requests.
