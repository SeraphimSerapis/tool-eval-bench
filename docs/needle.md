# Needle in a haystack

A model's advertised context window is a configuration value. Its *effective*
context is how much of that window it can still retrieve from. This benchmark
measures the gap.

It buries one synthetic fact (the needle) at a known depth inside a block of
unrelated prose (the haystack), asks for the fact back, and repeats across a
grid of haystack sizes and depths.

```bash
tool-eval-bench --needle-only          # just this benchmark
tool-eval-bench --needle               # after the tool-call scenarios
tool-eval-bench plugin needle          # the subcommand spelling
```

## What a run does

Every cell of the grid is one request:

1. Build a haystack of roughly N tokens from the shared filler corpus.
2. Insert the needle at the sentence boundary nearest the target depth.
3. Ask the question and read the answer back.
4. Score the cell as a retrieval if the response contains the expected value.

The default grid is 4 haystack sizes by 5 depths, so 20 requests. Sizes run from
1K tokens up to the context window less a small allowance for the prompt and the
answer; depths run from the very start of the document to the very end.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--needle` | off | Run the benchmark after the tool-call scenarios |
| `--needle-only` | off | Run only this benchmark |
| `--needle-depths N` | 5 | Depths to probe, evenly spaced across the document |
| `--needle-lengths N` | 4 | Haystack sizes to probe, up to the context window |
| `--context-size N` | auto | Override the detected context window |

`--seed`, `--parallel`, `--temperature`, and `--timeout` behave as they do
everywhere else. Under the `plugin needle` subcommand, the two grid flags are
spelled `--depths` and `--lengths`.

```bash
# A denser grid on a known 128K window
tool-eval-bench --needle-only --needle-depths 10 --needle-lengths 8 \
  --context-size 131072 --seed 42

# Chained with the rest of a full sweep: throughput, then retrieval, then the
# 88 tool-call scenarios
tool-eval-bench --hardmode --seed 42 --perf --needle

# Faster, at the cost of loading the server with concurrent long prompts
tool-eval-bench --needle-only --parallel 4
```

`--needle` composes like `--perf` does: it needs no subcommand, and it runs
after any throughput or accuracy benchmark and before the tool-call scenarios.
`--needle-only` skips the scenarios entirely.

## Reading the result

The terminal prints the grid, then two numbers:

- **Retrieval accuracy** — the share of cells that came back with the needle.
- **Effective context** — the largest haystack size that retrieved the needle at
  *every* depth. This is the number worth quoting. A model advertising 128K that
  misses a needle at 32K does not have 128K of usable context, whatever the
  config says. `none` means even the smallest size tested had a miss.

The Markdown report adds accuracy per haystack size and a table of every missed
needle with what the model said instead, which is usually enough to tell a
retrieval failure ("I could not find it") from a truncation ("the document ends
before...") or a hallucination.

Retrieval is not always monotonic. A model can miss at 32K and succeed at 64K,
so effective context reports the largest size that actually passed rather than
stopping at the first dip.

## Context window detection

The window comes from `/v1/models` (`max_model_len`, `context_window`, or
`max_tokens`), capped by the KV cache capacity that vLLM reports on `/metrics`,
on the same reasoning the [context pressure](context-pressure.md) sweep uses: a
server may have allocated far less cache than the model architecture allows, and
a haystack it cannot hold measures the deployment rather than the model. Hybrid
attention models are exempt from the cap.

When detection fails, pass `--context-size` explicitly.

## Design notes

**The haystack is generated, not downloaded.** It comes from the same filler
corpus that context pressure uses (`domain/filler.py`): a pool of technical
documentation, meeting notes, code review comments, and incident reports. There
is no dataset to fetch and no cache to warm.

**Every cell gets a different haystack.** The paragraph order is shuffled per
cell and random identifiers are sprinkled through the text, so no two requests
share a token prefix. Without this a server's prefix cache answers the second
request from the first one's KV blocks, and the benchmark measures the cache.
Passing `--seed` makes the whole grid reproducible without reintroducing shared
prefixes.

**Needles are unguessable and varied.** Each fact is a random passphrase, code,
or count that appears in exactly one sentence, so a model cannot infer it from
the surrounding text. Four templates rotate through the grid, because a single
phrasing measures one retrieval pattern rather than retrieval ability.

**Grading is a normalized substring match.** A model that answers "The
passphrase is K7QM-2XPD-9WLR." retrieved the fact; requiring a bare answer would
measure instruction following instead. Case and punctuation are folded, so
`k7qm 2xpd 9wlr` counts.

**Request failures count as misses.** Unlike the tool-call scenarios, which drop
infrastructure failures from the denominator, a timeout here is usually the
result: the prompt was too long for the deployment to serve. The report shows
`errors` and `completion_rate` separately so you can tell the two apart.
