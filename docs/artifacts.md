# Run IDs, artifacts, and labels

Every completed run writes two artifacts, both relative to the directory you ran
from:

| Artifact | Path | Contents |
|---|---|---|
| Markdown report | `runs/YYYY/MM/<run_id>.md` | Per-scenario verdicts with the full conversation trace |
| SQLite record | `data/benchmarks.sqlite` | The same data, queryable, plus traces for held-out packs |

Scenarios loaded from a held-out [scenario pack](scenario-packs.md) keep their
status and points in the Markdown report but withhold titles, summaries, and
traces, so publishing a score does not publish the pack. The cross-trial
`_summary.md` written by `--trials` withholds their summaries the same way.
`--json`, `--json-file`, and the stderr progress events withhold the same
content unless you pass `--include-held-out`. Full traces stay in SQLite for local inspection.

Context-pressure sweeps, spec-bench, `--perf-only` throughput runs, and plugin
runs share one report header: run ID, date, mode, the `tool-eval-bench` version,
the label, the facts that mode measured with, and the probed inference engine and
host. They leave out the tool-eval Run Context table, because these modes do not
use the scenario parameters it lists. When no RunContext could be built, the
version line and the engine table are omitted and the rest of the report is
unchanged. When only the engine probe fails, the engine table gives way to a
short Environment table of host, platform, and Python version.

Both artifacts are meant to be shareable, so every server URL in them is
redacted the same way: the host becomes `***` with the port kept, and userinfo,
query, and fragment are dropped. That covers report headers, stored configs,
and HTTP error messages quoted in traces and error fields. `--json` error
output, meaning the error envelope and the headless `error` events, follows the
same rule, and so do both `probe_result` events. The `server_discovered` event
is the exception: it reports the local URL it found unredacted, because a
consumer needs it to connect. `--redact-url` only changes what the console
shows.

## Run ID

Each execution gets a unique ID: `YYYY-MM-DDTHH-MM-SS.ffffffZ_<short_hash>`.

The persisted URL masks its authority and drops query parameters. An opaque
endpoint identity keeps retries against different deployments separate without
recording the host or any credentials. Spellings of one base URL that build the
same requests share an identity: `http://host:8000`, `http://host:8000/` and
`http://host:8000/v1` are one endpoint. A native Gemini base keeps its API
version, because a bare Gemini host means `v1beta`.

## Config fingerprint

Stored configs also carry a deterministic `config_fingerprint`, so leaderboard
entries and comparisons only group runs that are actually comparable. The
fingerprint covers the code identity (version and git SHA) and the discovered
deployment facts (engine, context window, quantization, GPU count, slot count,
speculative decoding) as well as the CLI flags, because the scenarios,
evaluators and measurement loops *are* code: two runs from different commits are
not comparable even when every flag matches. Scored runs, context-pressure
sweeps, spec-bench, throughput-only runs and plugins all use this one rule.

The leaderboard ranks only completed runs at 100% completion. It ranks models
against each other within a cohort: the same settings and the same
`tool-eval-bench` version and commit, read from each run's stored metadata.
Deployment facts are deliberately not part of the cohort, since models on one
server often differ in quantization or context window. Runs from different
cohorts stay visible but receive no misleading global rank. `export` restarts
the CSV `rank` column per cohort. CSV adds `cohort` and `cohort_fingerprint`
columns; JSON carries the same values as `cohort_label` and `cohort_fingerprint`.

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
included in `export`. Sweep, spec-bench, throughput, and plugin runs take the
label from the command line, so they keep it even when no RunContext could be
built. Scored runs read it from the RunContext, so a scored run whose context
could not be built has no label.

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
scenario in flight. Interrupted scored runs appear in `tool-eval-bench history`
marked `interrupted — resumable`. Throughput, context-pressure, spec-bench, and
plugin (GSM8K, MMLU, IFEval) runs cannot be resumed, so `history` shows their
bare status.

`tool-eval-bench resume RUN_ID` replays the finished work from the checkpoints
and runs only the missing, corrupt, or infrastructure-failed scenarios. Pass,
partial, and ordinary fail outcomes are immutable evidence under that run ID.
Start a new run when you want another scored attempt.
