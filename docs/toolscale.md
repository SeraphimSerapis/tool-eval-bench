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

The user message is the target's query, which names the target unambiguously.
The model sees the same system prompt the tool-call scenarios use. Only the
first tool call's **name** is graded; arguments are recorded but not scored.

### Similarity tiers

The built-in catalog has 17 domains, 4 families per domain, and 4 variants per
family: 272 tools. That structure defines the tiers.

| Tier | Neighbours come from | Example for `list_draft_invoices` |
|---|---|---|
| `near` | the other 3 variants of the target's family. Same job, one qualifier differs | `list_overdue_invoices`, `list_paid_invoices`, `list_voided_invoices` |
| `related` | one variant from each of the 3 other families in the target's domain | `issue_card_refund`, `get_sales_tax_rate`, `schedule_sms_payment_reminder` |
| `distinct` | other domains | weather, shipping, music |

Fillers always come from other domains, at most one tool per family, so the
fillers never add a near-duplicate of the target or of each other.

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
| `--toolscale-counts LIST` | `1,4,8,16,32,64` | Tool counts. `1` is the control; other values must be at least 4 |
| `--toolscale-similarity LIST` | `near,related,distinct` | Tiers to sweep |
| `--toolscale-limit N` | 0 (one per family, 68) | Targets per cell, spread across domains first |
| `--toolscale-catalog PATH` | built-in | A private catalog with the same schema |
| `--trials N` | 1 | Repetitions inside one run; 3 or more gives a spread per cell |

`--seed`, `--parallel`, `--temperature`, and `--timeout` behave as they do
everywhere else. Under the `plugin toolscale` subcommand the grid flags are
spelled `--counts`, `--similarity`, `--limit`, and `--catalog`.

Unlike the tool-call scenarios, where `--trials N` stores N separate runs, a
toolscale run keeps all its trials in one stored run.

### Cost

The default grid is 68 targets × (1 control + 5 counts × 3 tiers) = 1,088
requests per trial. Prompts run from a few hundred tokens at N=1 to roughly
6K at N=64. The report shows the measured mean prompt size per cell. Each list
is shuffled, so prefix caching helps little. A quick smoke run:

```bash
tool-eval-bench plugin toolscale --limit 17 --counts 1,4,16,64 --parallel 4
```

The token budget per request follows the scenario suite's default, which
leaves thinking models room to reason before they call. A response that hits
the budget with no call is recorded as `truncated`.

## Reading the result

The terminal prints the accuracy grid, the control, and the count effect per
tier. The Markdown report adds:

- **Accuracy grid.** Accuracy per cell with a 95% Wilson interval, and the
  standard deviation across trials when there is more than one.
- **Accuracy when a tool was called.** The same cells, counting only items
  where the model called a tool. Safety-tuned models sometimes decline a
  high-impact action, such as sharing an editable link, and ask for confirmation
  instead. That is abstention, not confusion, and the gap between the two grids
  shows how much of the headline drop it explains.
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
              "query": "Show me the draft invoices on billing account AC-4471 that haven't gone out yet."
            }
          ]
        }
      ]
    }
  ]
}
```

A catalog needs at least two domains, four families per domain, and four
variants per family. The largest N it supports is the number of families
outside the smallest domain, plus one. With a private catalog:

- the report keeps every table but withholds tool names and queries;
- stored results replace target names with `target-<n>`;
- the live progress line shows grid coordinates, not item IDs;
- the catalog's SHA-256 goes into the run's config fingerprint, so runs on
  different catalogs never compare.

`lint_catalog()` in `plugins/toolscale/catalog.py` checks the authoring rules
that keep items unambiguous: every variant name carries its qualifier, each
query names its own qualifier and no sibling's, and no query contains a tool
name. A private catalog that breaks them still runs, with a warning.

## Limits

- Fillers are unrelated to each other. A realistic enterprise catalog, where
  the fillers are also near-duplicates of one another, is a separate axis this
  version does not sweep.
- Argument correctness is not graded.
- N stops at 64 with the built-in catalog. Larger lists need a larger catalog,
  and several providers cap the number of tools per request.
- Queries name their qualifier literally, so every item has exactly one
  defensible answer. That trades some difficulty for an axis you can audit.
  A strong model can come close to saturating the grid. The signal then is in
  the slopes, the error mix, and weaker or smaller models, not the headline.
- The catalog was calibrated against one strong model. Every target ran at
  N=4 and N=64 with near and distinct neighbours. The one query that model
  found ambiguous was rewritten. A production-deploy variant that the model
  declined even when it was the only tool offered was replaced with a
  low-risk sibling, because an item a model will not act on measures safety
  tuning, not selection. Some abstention on other side-effecting tools remains;
  the "when called" grid separates it.
