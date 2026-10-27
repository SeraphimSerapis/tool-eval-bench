# Toolscale: tool count × description similarity

Tool-selection accuracy usually drops as the tool list grows. Most published
results cannot say why, because adding tools usually adds near-duplicates too,
so count and similarity move together. Toolscale varies the two independently
and reports each effect on its own.

```bash
tool-eval-bench --toolscale-only                 # just this benchmark
tool-eval-bench --toolscale                      # after the tool-call scenarios
tool-eval-bench plugin toolscale --trials 3      # the subcommand spelling
```

Category L (TC-37 to TC-40) is unchanged. It still measures one point, a fixed
52-tool set, and its scores stay comparable with earlier runs.

## What a run does

Every item is one streamed request with a generated tool list:

```
target  +  3 neighbours from one similarity tier  +  (N - 4) unrelated fillers
```

- **Similarity (s)** picks where the three neighbours come from. N stays fixed.
- **Count (N)** grows by adding fillers only. The target's neighbourhood stays
  fixed.

The user message is one of the target's two queries (the **cue**, below). The
model sees the same system prompt the tool-call scenarios use. Only the first
tool call's **name** is graded; arguments are recorded but not scored.

### Cues: literal and implied

Every tool has two queries with exactly one defensible answer each:

| Cue | The query... | Example for `list_draft_invoices` |
|---|---|---|
| `literal` | names the qualifier that tells the target from its siblings | "Show me the draft invoices on billing account AC-4471 that haven't gone out yet." |
| `implied` | describes the situation, so the qualifier has to be inferred | "Which invoices on billing account AC-4471 have I prepared but not sent to the customer yet?" |

Both cues see the identical tool list, so literal minus implied is paired too.
The literal cue is the clean control. The implied cue removes the keyword
shortcut, so a model has to understand the descriptions rather than match a
word. Lint rules keep each implied query free of its own qualifier and every
sibling's.

### Similarity tiers

The built-in catalog has 17 domains, 4 families per domain, and 4 variants per
family: 272 tools. That structure defines the tiers.

| Tier | Neighbours come from | Example for `list_draft_invoices` |
|---|---|---|
| `near` | the other 3 variants of the target's family. Same job, one qualifier differs | `list_overdue_invoices`, `list_paid_invoices`, `list_voided_invoices` |
| `related` | one variant from each of the 3 other families in the target's domain | `issue_card_refund`, `get_sales_tax_rate`, `schedule_sms_payment_reminder` |
| `distinct` | other domains | weather, shipping, music |

Fillers always come from other domains, so they never add a near-duplicate of
the target.

### Filler density: sparse and crowded

| Density | Fillers are... | At N=64 |
|---|---|---|
| `sparse` (default) | one tool from each of N-4 unrelated families, so no two fillers are alike | 60 tools from 60 families |
| `crowded` | whole families, so the list is dense with near-duplicate clusters | 60 tools from 15 families, 4 near-duplicates each |

This is the enterprise-catalog condition: plenty of tools that look alike, just
not like the target. Both densities walk the same family order and keep the
target's neighbours and the slot layout, so sparse minus crowded is paired.
The control and N=4 have no fillers, so crowded skips them. The report
confirms the axis by measuring each filler's similarity to its closest other
filler.

### The control cell

N=1 offers the target alone. It separates "cannot call a tool at all" from
"picked the wrong tool". N=4 with near neighbours is the pure disambiguation
ceiling: four near-duplicates and nothing else.

### Pairing

The grid is built so every contrast compares like with like:

- every cell uses the same targets;
- for one target and trial, the fillers are identical across tiers, and the
  neighbours sit in the same list positions in every tier;
- fillers are nested as N grows: the fillers at N=8 are a subset of those at
  N=16;
- the target sits at the same relative position in the list at every N, so
  position effects do not leak into the count axis.

Each trial redraws the fillers, the neighbour variants, and the target's
position, so `--trials` captures set-composition noise as well as sampling
noise.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--toolscale` | off | Run the sweep after the tool-call scenarios |
| `--toolscale-only` | off | Run only this benchmark |
| `--toolscale-counts LIST` | `1,4,8,16,32,64` | Tool counts. `1` is the control; other values must be at least 4. Above the sparse maximum (65 built-in), only crowded cells run, up to 248 |
| `--toolscale-similarity LIST` | `near,related,distinct` | Tiers to sweep |
| `--toolscale-cues LIST` | `literal,implied` | Query cues to sweep |
| `--toolscale-fillers LIST` | `sparse` | Filler densities to sweep: `sparse`, `crowded` |
| `--toolscale-limit N` | 0 (one per family, 68) | Targets per cell, spread across domains first |
| `--toolscale-catalog PATH` | built-in | A private catalog with the same schema |
| `--trials N` | 1 | Repetitions inside one run; 3 or more gives a spread per cell |

`--seed`, `--parallel`, and `--temperature` behave as they do everywhere else.
`--timeout` does too, except that it is raised when a long tool list needs
more time to prefill, the way the needle benchmark stretches it for long
haystacks. Under the `plugin toolscale` subcommand the grid flags are
spelled `--counts`, `--similarity`, `--cues`, `--fillers`, `--limit`, and
`--catalog`.

Unlike the tool-call scenarios, where `--trials N` stores N separate runs, a
toolscale run keeps all its trials in one stored run.

### Cost

The default grid is 68 targets × (1 control + 5 counts × 3 tiers) × 2 cues =
2,176 requests per trial. `--toolscale-cues literal` halves that. Adding
`--toolscale-fillers sparse,crowded` adds 68 × 4 counts × 3 tiers per cue.
Prompts run from a few hundred tokens at N=1 to roughly 8K at N=64, 16K at
N=128, and 33K at N=248. The report shows the measured mean prompt size per
cell. Each list is shuffled, so prefix caching helps little. A quick smoke run:

```bash
tool-eval-bench plugin toolscale --limit 17 --counts 1,4,16,64 --cues implied --parallel 4
```

The token budget per request follows the scenario suite's default, which
leaves thinking models room to reason before they call. A response that hits
the budget with no call is recorded as `truncated`.

## Reading the result

The headline is two numbers, reported apart on purpose:

- **Accuracy when called.** Of the items where the model called a tool, the
  share where it called the right one. This is the tool-confusion measure.
- **Call rate.** The share of answered items where the model called any tool.
  A model that asks a clarifying question, often about an optional parameter,
  or declines a high-impact action, is abstaining, not confusing tools. A
  request error has no model answer, so it is left out of the call rate; a
  turn that ran out of token budget without calling counts as no call.

A single blended accuracy hid this in calibration. A mid-sized model scored 83%
combined, and every miss was a clarifying question; it never picked a wrong
tool. The combined accuracy is still stored as the run's score so runs sort,
but it is not the headline. It is correct over all grid items, errors
included as misses; the N=1 control is reported on its own, except in a
control-only run, where it is the score. Read the two together:
accuracy when called alone can be gamed by declining hard items.

The terminal prints both per cell, one row per tier and cue, plus the cue
effect and the controls. The Markdown report adds:

- **Headline by cue.** Both numbers for each cue.
- **Grids per cue.** Accuracy when called, call rate, and combined accuracy for
  every cell, with a 95% Wilson interval and the standard deviation across
  trials when there is more than one.
- **Cue effect.** Literal minus implied accuracy at each N and tier, paired by
  target. Positive means the model relied on the query naming the
  distinguishing word.
- **Density effect.** Sparse minus crowded accuracy at each N, tier, and cue,
  paired by target, when the run swept both densities. Positive means a
  catalog dense with near-duplicate clusters costs accuracy.
- **Similarity effect at fixed N.** Distinct accuracy minus near (or related)
  accuracy, paired by target, with a bootstrap interval. Positive means
  similarity costs accuracy.
- **Count effect at fixed similarity.** Accuracy at the smallest N minus the
  largest, paired by target, plus the least-squares slope per doubling of N.
  Positive delta means more tools cost accuracy.
- **Error mix.** Each wrong answer is classified by the role of the tool picked.
  A wrong *neighbour* in the near tier is similarity confusion. A wrong *filler*
  is a count or attention failure. In the distinct tier both are unrelated
  tools, so their rates are the baseline. `hallucinated` means a name that was
  not offered; `no_call` means the model answered in text.
- **Query check.** Word overlap between each target's query and its own
  definition, and the margin over its closest sibling, per cue. Literal queries
  lean toward the target; implied ones should not. This confirms the implied
  cue really removes the keyword shortcut.
- **Similarity check.** The measured similarity of each tier (below).
- **Accuracy by target position.** First, middle, and last third of the list.
- **Failed items.** Query, the tool picked (or the start of the model's reply
  when it did not call), and the outcome (first 200).

Request errors count as misses, as in every accuracy plugin, and appear in the
error mix as `error`, so a reader can tell them from wrong picks.

### The similarity check

The tiers are defined by the catalog's structure. The report confirms them with
a measured, dependency-free metric: TF-IDF cosine over the words a model reads
in a tool definition (name, description, and parameter names), with inverse
document frequency fitted on the catalog. It reports:

- mean similarity between a target and its three neighbours, per tier. It must
  rise from distinct to related to near;
- mean similarity between a target and its fillers;
- the target's nearest-neighbour similarity within the offered list at each N.
  At fixed tier this should stay roughly flat as fillers are added, which is
  what "fixed similarity" means here;
- a **Category L reference**: TC-37's target, `get_weather`, measured against
  the rest of the 52-tool set on the same scale. That places Category L's single
  point on the toolscale chart.

The test suite asserts the tier ordering for every tool in the built-in
catalog, so a catalog edit that blurs the tiers fails CI.

Lexical overlap undercounts paraphrase. That is acceptable because the metric
checks the structural tiers rather than defining them.

## Private catalogs

A public catalog ends up in training data. `--toolscale-catalog PATH` runs a
private catalog with the same JSON schema as
`src/tool_eval_bench/plugins/toolscale/catalog/v1.json`:

```json
{
  "version": "lab-2026-10",
  "domains": [
    {
      "id": "billing",
      "families": [
        {
          "id": "invoice_listing",
          "variants": [
            {
              "name": "list_draft_invoices",
              "qualifier": "draft",
              "description": "List draft invoices on one billing account that have been prepared but not yet sent ...",
              "parameters": {"account_id": {"type": "string", "description": "Billing account identifier"}},
              "required": ["account_id"],
              "query": "Show me the draft invoices on billing account AC-4471 that haven't gone out yet.",
              "query_implied": "Which invoices on billing account AC-4471 have I prepared but not sent yet?"
            }
          ]
        }
      ]
    }
  ]
}
```

A catalog needs at least two domains, four families per domain, and four
variants per family. `query_implied` is optional per tool, but the implied cue
needs it on every tool; run a catalog without it with `--toolscale-cues literal`.

The largest N is set by the domain with the most families, since every
target's fillers must come from outside its own domain. For sparse fillers it
is the number of families outside that domain, plus one. For crowded fillers
it is the number of tools in those families after setting aside the three
largest, plus four. With a private catalog:

- the report keeps every table but withholds tool names and queries;
- stored results replace target names with `target-<n>`;
- the live progress line shows grid coordinates, not item IDs;
- the catalog's SHA-256 goes into the run's config fingerprint, so runs on
  different catalogs never compare.

`lint_catalog()` in `plugins/toolscale/catalog.py` checks the authoring rules
that keep items unambiguous: every variant name carries its qualifier, each
literal query names its own qualifier and no sibling's, each implied query
names neither, and no query contains a tool name. A private catalog that breaks them still runs, with a warning.

## Calibration

Two local models ran the full default grid (68 targets, both cues, seed 42) in
October 2026. A strong model also ran every one of the 272 tools as a target,
with each cue, at N=4 and N=64.

| Model | Accuracy when called (literal / implied) | Call rate (literal / implied) | Wrong picks |
|---|---:|---:|---:|
| DeepSeek V4.1 Flash | 100.0% / 99.7% | 97.8% / 98.7% | 3 of 2,004 |
| Gemma 4 | 100.0% / 99.9% | 83.2% / 86.4% | 1 of 1,730 |

What it showed:

- **Tool confusion was nearly absent.** No similarity tier, cue, or count up to
  64 produced a measurable drop in accuracy when called. The paired effects are
  computed on combined accuracy, and the two whose intervals excluded zero
  (DeepSeek, distinct tier at N=4) came from extra clarifying questions, not
  wrong picks. Two of the four wrong picks traced to one ambiguous implied
  query, since rewritten.
- **Abstention separated the models.** Gemma asked a clarifying question,
  usually about an optional parameter, on about one item in six. The call rate
  carried the whole difference in combined accuracy.
- **Neighbours changed willingness to act.** Gemma called more often when the
  three neighbours were near-duplicates of the target than when they were
  unrelated: on the literal cue, 91% against 84% at N=4.
- **Crowding did not matter.** On the implied cue at N=8 to 64, crowded lists
  were far denser than sparse ones (closest-filler similarity 0.605 against
  0.174), yet accuracy when called stayed at 99.7% or better for both models
  in both conditions, and no density effect excluded zero. Gemma's only wrong
  picks in that run were near-tier siblings at N=32 and N=64 ("since
  midnight" read as this month, walk-in collection read as drive-through), in
  sparse and crowded lists alike. Four misses in about 1,400 calls are too few
  to call a count-by-similarity interaction, but that cell is where to look
  with weaker models or more trials.
- **Up to 248 tools, still no confusion.** With crowded lists at N=64, 128,
  and 248 on the implied cue, DeepSeek was correct on all 537 calls and Gemma
  on 483 of 486 (99.4%). Gemma's misses were the same shipping item twice and
  one near sibling at N=248. Gemma's call rate rose with the list, from 87% at
  N=64 to 94% at N=248 in the near tier: more tools made it more willing to
  act, not less. A first Gemma attempt lost two thirds of its requests to a
  server outage; the rerun had no errors, and the report now prints each
  failed request's reason.
- **Calibration fixes.** One literal query that fit three tools ("premium
  cabin"), one implied query that a faster service also satisfied, and one
  production-deploy variant that the strong model refused even when it was
  the only tool offered (replaced with a low-risk sibling) were rewritten.

The published drops this benchmark set out to separate come from conditions it
holds clean on purpose: hundreds of tools, inconsistent or terse descriptions,
and selection deep in a conversation. Crowded catalogs and lists of up to 248
tools are now covered and, for these two models, made no difference.
Description quality and conversation depth are the next to add.

## Limits

- Argument correctness is not graded.
- With the built-in catalog, sparse lists stop at 65 tools and crowded lists
  at 248. Above 65 only crowded cells exist: a finite catalog runs out of
  unrelated families before it runs out of tools, as a real 200-tool catalog
  would. Clean sparse lists past 65 would need a larger catalog.
- Some providers cap tools per request. OpenAI rejects more than 128; the run
  warns above that, and a rejected request is recorded as an error, not a wrong
  pick. Local OpenAI-compatible servers have no tool cap, only their context
  window.
- With clean, well-written descriptions and at most 64 tools, current models
  rarely confuse tools on either cue (see Calibration). Expect the call rate to
  separate models before accuracy when called does.
- Implied queries need a short, common-sense inference that the target's
  description supports. Lint can check that they do not give the answer away,
  not that the inference is fair; that rests on review and calibration.
- Crowded fillers stay outside the target's domain. Crowding the target's own
  neighbourhood beyond its three neighbours would need larger families.
