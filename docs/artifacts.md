# Run IDs, artifacts, and labels

Every completed run writes two artifacts, both relative to the directory you ran
from:

| Artifact | Path | Contents |
|---|---|---|
| Markdown report | `runs/YYYY/MM/<run_id>.md` | Per-scenario verdicts with the full conversation trace |
| SQLite record | `data/benchmarks.sqlite` | The same data, queryable, plus traces for held-out packs |

Scenarios loaded from a held-out [scenario pack](scenario-packs.md) keep their
status and points in the Markdown report but withhold titles, summaries, and
traces, so publishing a score does not publish the pack. Full traces stay in
SQLite for local inspection.

## Run ID

Each execution gets a unique ID: `YYYY-MM-DDTHH-MM-SS.ffffffZ_<short_hash>`.

The persisted URL masks its authority and drops query parameters. An opaque
endpoint identity keeps retries against different deployments separate without
recording the host or any credentials.

## Config fingerprint

Stored tool-evaluation configs also carry a deterministic `config_fingerprint`,
so leaderboard entries only group runs that are actually comparable. The
fingerprint covers the code identity (version and git SHA) as well as the CLI
flags, because the scenarios and evaluators *are* code — two runs from different
commits are not comparable even when every flag matches.

The leaderboard ranks only completed runs at 100% completion. Runs from
different cohorts stay visible but receive no misleading global rank.

The version is derived from git by setuptools-scm, so a build installed straight
from a commit reports which commit it came from rather than claiming to be the
last tagged release. `git_sha` resolves against the installed package's own
checkout, is `None` for wheel installs, and gains a `-dirty` suffix when the
working tree has uncommitted changes.

## Labeling runs (`--label`)

`--label "..."` attaches an arbitrary string to an execution. Every report that
execution generates carries it: a `Label` row in the tool-eval Run Context table,
a `- **Label**:` header line in the plugin, throughput, spec-decode, and
pressure-sweep reports, and the persisted metadata shown in `history` and
included in `export`.

Report filenames gain a safe slug of the label, so all files from one execution
end with the same marker:

```
runs/2026/08/<run_id>--nightly-qwen3-2026-08.md
runs/2026/08/<run_id>--nightly-qwen3-2026-08_summary.md
```

The full label is persisted unchanged. Reports render it as inert inline code;
line breaks and control characters show as visible escapes, so a label cannot
alter the Markdown structure. Only the filename uses a slug: lowercased,
punctuation collapsed to dashes, `.-_` kept, capped at 80 characters. A label
with no ASCII representation gets a deterministic `label-<hash>` marker.

The label is an annotation only. It does not affect the run ID or the
`config_fingerprint`, so identical runs with different labels stay comparable.

## Resuming

Every scenario result is checkpointed to SQLite the moment it finishes, so a
Ctrl-C or a dropped connection midway through the suite costs you only the
scenario in flight. Interrupted runs appear in `tool-eval-bench history` marked
`interrupted — resumable`.

`tool-eval-bench resume RUN_ID` replays the finished work from the checkpoints
and runs only the missing, corrupt, or infrastructure-failed scenarios. Pass,
partial, and ordinary fail outcomes are immutable evidence under that run ID.
Start a new run when you want another scored attempt.
