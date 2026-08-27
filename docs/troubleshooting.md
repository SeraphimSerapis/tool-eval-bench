# Troubleshooting

Failure modes that come up often, and what each one actually means. For the full flag list and
JSON output shape, see [cli-reference.md](cli-reference.md).

## Start by isolating the endpoint

`probe` checks readiness without spending a benchmark run:

```bash
tool-eval-bench probe
tool-eval-bench probe --base-url http://localhost:8000
```

Its exit code tells you where the problem is:

| Exit | Meaning | Usual cause |
|---|---|---|
| 0 | Ready | |
| 1 | Runtime error, or server not ready | The process is up but not serving yet |
| 2 | Connection or HTTP error | Wrong host or port, or nothing listening |
| 3 | No models found | The server started with zero models loaded |

## The server is running but nothing is discovered

Auto-discovery scans common localhost ports for a model-list response. It finds vLLM, llama.cpp,
SGLang, LiteLLM, Ollama, and TGI on their defaults, and nothing else. Pass `--base-url` explicitly
for a non-standard port, a remote host, or anything behind a proxy.

llama.cpp is the common exception even locally: it may expose the model list at `/models` rather
than `/v1/models`, so discovery can miss a server that is otherwise healthy.

## Exit 3, or the wrong model gets benchmarked

The model is auto-detected from the server's model list. When the server has several models loaded,
name the one you mean:

```bash
tool-eval-bench run --model Qwen/Qwen3-8B --base-url http://localhost:8000
```

## The run stops at the pre-flight gate

Some endpoints have provider-specific startup behavior that the strict gate reads as not ready.
When you know the endpoint is good, skip it:

```bash
tool-eval-bench run --no-preflight
```

## Timeouts on thinking models

A timed-out scenario shows as `⏱  TIMEOUT` with `–/2` points, not as a failure. It is excluded from
both the numerator and the denominator, so the run says nothing about the model on that scenario.
Check `completion_rate` before comparing the score to anything.

The run prints what to change, computed from the slowest turn it actually measured. Take that
suggestion first.

Reasoning models routinely exceed the 120-second default. Raise it:

```bash
tool-eval-bench run --timeout 600
```

Or shorten what the model generates, which fixes the cause rather than the symptom:

```bash
tool-eval-bench run --no-think                                    # reasoning off entirely
tool-eval-bench run --backend-kwargs '{"reasoning_effort": "low"}'  # if your server honours it
tool-eval-bench run --backend-kwargs '{"max_tokens": 1024}'         # hard cap on generation
```

`--no-think` changes both the latency and the tool-calling behavior you are measuring, so it is a
diagnostic rather than a setting to keep. Scenarios that score provider-exposed reasoning, such as
TC-88, cannot reach a pass without it.

### Why a later turn times out when the first one did not

Only the first turn of a scenario is streamed. On a streamed turn the read timeout measures the gap
between tokens, so a long generation never trips it. Every later turn arrives as one response, which
makes the same number bound the whole generation instead.

A model that comfortably finished turn 1 in 150 seconds under a 120-second timeout will therefore
die on turn 2 without having slowed down at all. To stop that, later turns are given a multiple of
what turn 1 actually took, so a model that has demonstrated it is slow gets room while a hung
endpoint still fails at the configured timeout. Raising `--timeout` further is still the fix when
even that is not enough.

Eleven of the 88 scenarios have follow-up turns, so on a slow deployment this shows up as a handful
of exclusions rather than a whole-run failure.

## Rate limits against a hosted endpoint

Hosted APIs return 429s under sustained load. The runner already shares backoff across scenarios
and paces requests adaptively rather than retrying blindly, so the usual fix is patience rather
than configuration.

What matters is how the result is reported. A scenario lost to a persistent 429 is dropped from
both the numerator and the denominator instead of being scored zero, because it measures the
serving environment rather than the model. Check `completion_rate` before comparing that run to
anything else.

## A score moved and you do not know why

Two runs are only comparable when their `config_fingerprint` matches. The fingerprint covers the
code identity and the configuration, so a changed flag, a different scenario selection, or a new
version puts the run in a different cohort. The leaderboard groups on it for exactly this reason,
and `history` will show you both runs' fingerprints.

Also check `completion_rate` on both. A run graded on 60 of 69 scenarios is not comparable to one
graded on all 69, however similar the headline scores look.

## Backend-specific behavior

Some capabilities are not uniform across serving stacks, and the benchmark degrades rather than
failing when one is missing:

| Behavior | vLLM | SGLang | LiteLLM | llama.cpp |
|---|---|---|---|---|
| `/v1/models` discovery | Yes | Yes | Yes | May be at `/models` |
| `parallel_tool_calls` | Yes | Yes | Yes | Not supported |
| Streaming `usage` stats | Yes | Varies | Varies | No |
| `tool_choice: "required"` | Yes | Yes | Yes | Version-dependent |
| Large toolsets (52 tools) | Yes | Yes | Yes | May exceed the context window |

Category L runs 52 tools at once, so on llama.cpp it can exceed the context window on smaller
builds. That shows up as failures concentrated in one category rather than a uniformly lower score.

## Parsing the JSON output in CI

`--json` writes JSONL to stderr, one object per line, not a single document. Parse it line by line,
or use `--json-file PATH` to get the final result as one file.

To fail a CI job on safety-critical failures specifically, `--fail-on-safety` exits 2.

## Still stuck

Backend-specific issues are worth reporting, since the adapter layer is where most of them belong:
[open an issue](https://github.com/SeraphimSerapis/tool-eval-bench/issues).
