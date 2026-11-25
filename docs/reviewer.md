# Reviewer: pairing the model with a second model

A reviewer is a second model that reads each turn the benchmarked model
proposes before the harness acts on it. The reviewer either approves the turn
or returns feedback. On feedback, the benchmarked model revises the turn, and
the harness runs the revision instead. A tool call that a revision replaces
therefore never reaches the mock tools, and a replaced reply never reaches the
evaluator. When the revision fails or gives the harness nothing to act on, the
current draft runs: the original turn, or the last revision that succeeded (see
[failures](#failures-never-fail-a-scenario)).

The pair is the system under test. A reviewed run stores the reviewer in its
config, gets its own comparison fingerprint and leaderboard cohort, and never
ranks against a solo run of the same model.

## Quick start

```bash
# Reviewer on another server; its model comes from that server's /v1/models
tool-eval-bench run --base-url http://localhost:8003 \
  --reviewer-base-url http://localhost:8001

# Another model on the same server, for example a larger sibling
tool-eval-bench run --base-url http://localhost:8000/v1 --model small \
  --reviewer-model large

# Review only tool calls, with a cooler reviewer and two rounds per turn
tool-eval-bench run --base-url http://localhost:8003 \
  --reviewer-base-url http://localhost:8001 \
  --reviewer-scope tools --reviewer-temperature 0 --reviewer-max-rounds 2
```

Any `--reviewer*` flag enables the reviewer. `--reviewer` alone enables the
reviewer that `TOOL_EVAL_REVIEWER_BASE_URL`, `TOOL_EVAL_REVIEWER_MODEL`, and
`TOOL_EVAL_REVIEWER_API_KEY` configure. Like `TOOL_EVAL_BASE_URL`, those
variables never switch a reviewer on by themselves, so a reviewer in `.env`
cannot silently turn every run into a paired run.
`--reviewer-provider NAME` reads the reviewer's URL, key, model, and headers
from `TOOL_EVAL_<NAME>_*`, the same variables `--provider` reads.

## How a turn is reviewed

1. The benchmarked model proposes a turn: tool calls, a reply, or both.
2. When the turn falls under `--reviewer-scope`, the reviewer receives the
   assistant's system prompt, the conversation so far, the tool definitions
   (with a note when the step disabled or required tool calls), and the
   proposed turn. It answers with one JSON object,
   `{"verdict": "approve" | "revise", "feedback": "..."}`.
3. On `approve`, the harness runs the turn as usual.
4. On `revise`, the harness sends the benchmarked model the same request
   plus one extra user message. That message shows the rejected turn and the
   feedback, and tells the model that the turn did not run. The model's new
   answer replaces the rejected turn.
5. While rounds remain (`--reviewer-max-rounds`, default 1) and the revision
   still falls under the scope, the reviewer reviews the revision too. The
   revision that follows the last allowed review runs without another review.

The review message never enters the conversation history. The next turn sees
the accepted turn only, so the conversation keeps the shape the scenario
evaluators expect.

`--reviewer-scope` selects the turns:

| Scope | The reviewer checks |
|---|---|
| `turns` (default) | every turn with a tool call or a visible reply |
| `tools` | turns with at least one tool call |
| `final` | replies without a tool call, the turns that end a phase |

The reviewer never checks a turn with neither a tool call nor visible text,
such as a turn the token ceiling cut off: the harness already tags those turns.

The reviewer never receives context-pressure filler. The harness drops the
filler messages from the reviewer's copy of the conversation and says how many
it dropped, so the reviewer reads the task instead of tens of thousands of
tokens of unrelated history.

## Settings and inheritance

A setting the reviewer flags leave unset inherits from the main run.

| Reviewer flag | Inherits from |
|---|---|
| `--reviewer-temperature` | `--temperature` |
| `--reviewer-top-p`, `--reviewer-top-k`, `--reviewer-min-p`, `--reviewer-repeat-penalty` | the matching main flag |
| `--reviewer-no-think` / `--reviewer-think` | `--no-think` |
| `--reviewer-seed` | `--seed` |
| `--reviewer-timeout` | `--timeout`, including its context-pressure scaling |
| `--reviewer-backend-kwargs JSON` | merged over `--backend-kwargs` |
| `--reviewer-format` | detected from the reviewer URL; the main `--format` only when the reviewer has no URL of its own |
| `--reviewer-api-key`, `--reviewer-header`, `--reviewer-session-header` | the main key, headers, and session header, **only on the same server** |
| `--reviewer-system-prompt(-file)` | never inherited; the built-in reviewer prompt applies |

The reviewer never inherits keys of `--backend-kwargs` that shape the request
rather than the sampling (`model`, `messages`, `tools`, `tool_choice`,
`parallel_tool_calls`, `response_format`, `stream`, `stream_options`, `system`,
`contents`, `systemInstruction`, `toolConfig`, `tool_config`, `functions`,
`function_call`, `cachedContent`, `cached_content`), nor an output
schema nested in `output_config.format` or in `generationConfig`
(`responseSchema`, `responseJsonSchema`, `responseMimeType`): an inherited
`model` would let the report name one reviewer while the endpoint runs another,
and an inherited schema would force the model's answer format on the verdict.
Credential-like keys (`api_key`, `token`, `authorization`, `password`,
`secret`, and similar) are not inherited either, at any depth, since they
belong to the benchmarked endpoint. `--reviewer-backend-kwargs` and
`ReviewerSettings.extra_params` reject them, because extra params are stored
with the run; the reviewer's key and headers have their own flags.
`--reviewer-backend-kwargs` merges over the inherited `--backend-kwargs` the
way `--backend-kwargs` merges over the individual flags: nested objects merge
key by key. It may not set `model`, `messages`, `system`, `contents`, or
`systemInstruction` (use `--reviewer-model` and `--reviewer-system-prompt`), so
the stored reviewer is always the one the requests reach. A `null` value removes an inherited key, which matters when the
reviewer speaks another wire format, for example
`--reviewer-backend-kwargs '{"top_k": null, "repetition_penalty": null}'`.
`--reviewer-think` removes an inherited `enable_thinking: false`;
`--reviewer-no-think` sets it. An explicit `--reviewer-temperature` or
`--reviewer-seed` also removes an inherited `temperature` or `seed` key, which
would otherwise override it on the wire.

The reviewer uses the main API key, headers, and session header only when it
talks to the benchmarked server: when it has no URL of its own, or when its URL
has the same scheme, host, and port. A credential issued for one server never
travels to another. A reviewer elsewhere authenticates with
`--reviewer-api-key`, `TOOL_EVAL_REVIEWER_API_KEY`, or its provider's key. The
same rule binds those defaults to their own URL: when `--reviewer-base-url`
points at another origin than `TOOL_EVAL_REVIEWER_BASE_URL` or the provider's
URL, only an explicit `--reviewer-api-key` and `--reviewer-header` go there.
`TOOL_EVAL_REVIEWER_API_KEY` without `TOOL_EVAL_REVIEWER_BASE_URL` belongs to the
benchmarked server, so a `--reviewer-base-url` on another origin needs
`--reviewer-api-key`.
Model discovery on the reviewer's `/v1/models` uses the same key and headers
as its chat requests. A reviewer with a key of its own on the benchmarked
server keeps the run's other headers but not its `Authorization`, which the
adapter would otherwise apply over the reviewer's key.

The main `--system-prompt` describes the assistant, not the reviewer, so the
reviewer never inherits it; the reviewer still sees that prompt as part of the
conversation it reviews. `--reviewer-system-prompt` replaces the built-in
reviewer prompt. A replacement must still ask for the JSON verdict above. The
parser also accepts a reply that starts with `APPROVE` or `REVISE`.

## Failures never fail a scenario

Before the first scenario, the service sends the reviewer one trivial review
with the real prompt, sampling, and token ceiling. When that review fails (a
wrong URL, model, or key, an HTTP error, or an endpoint that returns no text at
all), the run stops before it stores anything, because a reviewer that fails from the start
would turn the paired run into a solo run without notice. A reply without a
verdict gets one more attempt; a second reply without a verdict refuses the
reviewer, because its turns would all run unreviewed.

During the run, the harness lets a turn run unreviewed when the reviewer
cannot be reached, answers with an HTTP error, or replies without a verdict.
The harness keeps the draft when the revision request fails, or when the
revision has nothing to act on: no answer and no tool call, tool arguments cut
off at the token ceiling, or tool-call markup the server did not parse. A discarded revision still counts toward the
scenario's tokens. Every such case counts as an error in the per-scenario
review statistics and appears in the trace as `revision_error_*`, so the
report shows how many turns went unreviewed (`review_*=error`) or kept their
draft (`revision_error_*`). The stored error details name an
HTTP status, an exception type, or the length of a reply without a verdict,
never the text itself, since an error body can echo request headers. A usage
count the server sends in another type than a non-negative integer counts as
zero rather than failing the scenario.

While a reviewer is configured, the harness masks known credential values as
`***`, best effort (it matches the values, not transformed echoes):

- in adapter and HTTPX log lines: the reviewer's API key, credential-like header
  values, and URL credentials, and the run's API key and credential-like header
  values;
- in the request the reviewer receives: the run's API key, credential-like
  header values, and URL credentials;
- in reviewer feedback, before it reaches the trace or the benchmarked model:
  the reviewer's own credentials.

## What is stored

- **Run config.** The `reviewer` block holds the model, the redacted origin
  (scheme, masked host, and port; never the path, which some gateways use for
  authentication), the endpoint identity, the wire format, the scope, the rounds, the resolved
  temperature, timeout, seed, and extra params, and a custom prompt when one
  was given. It never
  holds a key or a header. Everything except the redacted URL and the timeout
  enters the comparison fingerprint and the leaderboard cohort.
- **Resume.** `--resume` rejects a run whose reviewer differs from the current
  flags, or a paired run resumed without a reviewer (`reviewer (was X, now off)`).
- **Per scenario.** `review` counts reviews, approvals, requested revisions,
  completed revisions, reviewer errors, failed revisions, reviewer tokens, and
  reviewer latency. The scenario's own token counts include the revisions,
  because the benchmarked model produced them, and each reviewed turn's
  latency covers the whole round: draft, reviews, and revisions.
- **Trace.** `review_<turn>.<round>=approve|revise: ...|error: ...` records
  each verdict; `rejected_draft_*` and `rejected_tool_call_*` record a draft
  that a revision replaced and that therefore never ran; `revision_error_*`
  records a revision the harness could not use, after which the draft ran.
- **Report.** The header names the reviewer, and a `## Reviewer` section lists
  the settings, the totals, and one row per scenario.
- **Leaderboard.** A paired row reads `model + reviewer <reviewer>`, and its
  cohort label carries `reviewer=<reviewer>`. CSV and JSON exports, from the
  CLI and the web UI, carry a `reviewer_model` field. The `--trials` cross-trial
  summary names the pair the same way, and the web UI's run list, run page,
  and comparison view show `+ reviewer <reviewer>` next to the model.
- **Terminal.** The CLI names the reviewer by its redacted origin, never its
  path, and leaves the body of an invalid `/v1/models` reply out of the error.
  `history`, `compare`, and the HTML `compare --report` of two Markdown reports
  (run reports and cross-trial summaries) name the pair as
  `model + reviewer <reviewer>`.
- **Results card.** `card` labels a paired tool-calling score with
  `+ reviewer <reviewer>`; throughput and accuracy sections stay the model's
  own.

## Limits

- **Held-out packs.** A reviewer cannot be combined with `--scenario-pack`,
  because it would receive every private prompt, tool result, and answer.
  The service and the runner also refuse held-out scenarios, so the Python API
  cannot send one either.
- **Resume.** The CLI compares a resumed run's stored reviewer with the
  current reviewer flags. The web launcher rebuilds the reviewer from the
  stored block; since that block keeps only a redacted origin and no key,
  enter the reviewer's Base URL (blank when it ran on the benchmarked server)
  and key in the launcher's Reviewer section before pressing Resume. The
  stored endpoint identity must match.
  A resume whose scenarios all finished only merges stored results and sends
  the reviewer nothing, so the reviewer need not be reachable. On the CLI, pass
  `--reviewer-model` as well: with a reviewer URL alone the CLI first asks the
  reviewer's `/v1/models` for its model.
  `BenchmarkService.run_benchmark(resume_run_id=...)` trusts its caller for
  every setting, the reviewer among them, as it did before the reviewer
  existed.
- **Conversation overrides.** `--backend-kwargs` or `extra_params` with
  `messages` or `contents` replace the whole conversation, which would drop
  every revision request, so a reviewer refuses them.
- **Not yet supported.** `compare --retest-diffs` refuses reviewed runs,
  rather than quietly retesting the model alone.
- **Scope of the feature.** The reviewer covers the tool-call scenarios of
  `run` and `resume` (and flat invocations that run them). Accuracy plugins,
  throughput, spec-bench, and context-pressure sweeps run without it, and the
  CLI rejects reviewer flags for invocations that run no tool-call scenarios.
- **Cost.** Every reviewed turn adds at least one reviewer request, and every
  revision adds a review and a request to the benchmarked model.
- **Steering.** The reviewer reads tool results and assistant text, so text
  inside them can steer it, as it can steer the benchmarked model.

## Web UI

`tool-eval-bench serve` has a collapsible **Reviewer (optional)** section next
to the endpoint fields: Base URL (blank for the benchmarked server), a
**Probe** button that lists the reviewer's models, model, API key, scope,
reviews per turn, and temperature. A blank model starts a solo run. The
reviewer probe never sends `TOOL_EVAL_API_KEY`, and the run's key reaches the
reviewer only on the same origin. The form clears a reviewer API key when the
reviewer URL changes to another origin, so a key typed for one server never
reaches the next. The live view names the pair.

A web resume restores the reviewer's key but not custom headers, which the
stored block never holds. A reviewer that needs a header (`--reviewer-header`)
fails its startup review on a web resume rather than running unreviewed;
resume such a run from the CLI with the same flags.

## Python API

```python
import asyncio
from tool_eval_bench.api import ReviewerSettings, run_benchmark

result = asyncio.run(run_benchmark(
    model="small",
    base_url="http://localhost:8003",
    reviewer=ReviewerSettings(
        model="large",
        base_url="http://localhost:8001",  # None: the benchmarked server
        temperature=0.0,                    # None: inherits temperature
        scope="turns",                      # turns, tools, or final
        max_rounds=1,
    ),
))
```

A `None` field inherits from the call. Sampling fields (`temperature`,
`timeout_seconds`, `seed`, `extra_params`) always inherit. Connection fields
(`api_key`, `wire_format`, `extra_headers`, `session_header`) inherit only when
`base_url` is `None`; unlike the CLI, the API does not compare origins, so a
reviewer with its own `base_url` needs its own key.
