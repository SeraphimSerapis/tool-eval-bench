# Held-out scenario packs

Private scenarios that stay private. A pack's scores appear in reports while its titles,
summaries, and traces are withheld, so publishing a number does not burn the pack.



Every scenario in this repository is public — prompt, mock tool responses, and evaluator. That is what makes the benchmark auditable, and it is also its expiry date: a published benchmark eventually lands in training data, and a memorized answer looks exactly like a capable one.

A **pack** is a directory of YAML scenarios — the declarative format loaded by `evals/yaml_loader.py`, with `evals/yaml_scenarios/` in this repo as three worked examples — kept outside the repo. Point a run at one to score against scenarios the model cannot have seen:

```bash
# Public suite + a private pack (69 + N scenarios)
tool-eval-bench run --scenario-pack ~/private/tool-eval-holdout

# Only the private pack
tool-eval-bench run --scenario-pack ~/private/tool-eval-holdout --pack-only

# Multiple packs
tool-eval-bench run --scenario-pack ~/packs/a --scenario-pack ~/packs/b
```

Pack scenarios are scored identically to public ones — they contribute to `final_score`, category percentages, and difficulty weighting. Two things differ:

- **The report withholds them.** Titles, summaries, and traces for pack scenarios are replaced with `held out` in the Markdown artifact, and therefore in any HTML comparison generated from it. Only the scenario ID, difficulty, status, points, and failure kind are published. This is a deliberate exception to the full-trace rule: publishing a held-out trace burns the scenario. Full traces are still stored in SQLite for your own inspection. Redaction is report-scoped — the live display and `--dry-run` still show pack titles locally so you can see what is running.
- **The report attests to which pack produced the number.** Each pack is hashed by filename and file bytes, and the hash is recorded in the run config and folded into `config_fingerprint`. Readers can confirm two published scores were measured against the same held-out set — and that it was not edited in between — without seeing its contents. Editing or renaming a scenario changes the hash, so `compare` will flag the runs as non-comparable.

Scenario IDs must not collide with the public suite or with another pack; a collision is an error rather than a silent override. The check covers every public scenario, Hard Mode included, whatever the selection flags, so it also applies under `--pack-only`.

`--scenario-pack` cannot be combined with `--context-pressure-sweep`. The sweep report publishes the full trace of every scenario and has no held-out handling, so the combination is refused before any request is made. A single `--context-pressure` run is a scored run and withholds pack traces as described above.

A pack is validated in full when it loads, before any request is made. A malformed scenario, an unknown or repeated key such as a misspelled `arguments`, a tool the scenario is never offered, a `match` key the tool has no parameter for, a `response` that is empty or a list rather than a mapping or string, a `difficulty` outside 1 to 5, or an unquoted value YAML 1.1 reads ambiguously (such as `2026-03-21`, `14:30`, or `NO`) stops the run with the file and line. A pack that cannot be graded would otherwise lower every model's attested score. [The YAML alternative](adding-a-scenario.md#the-yaml-alternative) lists the rules.
