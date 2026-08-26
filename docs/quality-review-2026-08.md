# Quality review, August 2026

A full-project review of `tool-eval-bench` covering documentation, structure, performance,
and test/tooling health. It records what was measured, what is wrong, and what to do about it.

Scope at the time of review: 42k lines of source across 106 modules, 87k lines of tests across
96 files, 247 tracked files, version 2.6.0.

Severity: **P1** actively misleads or blocks someone. **P2** real maintenance cost. **P3** cleanup.

---

## What is already good

Worth stating first, because it shapes what the rest of this document recommends.

The layered architecture is real and mechanically enforced. `tests/test_architecture_boundaries.py`
walks the AST of every module against per-package rules, which is why `domain/` imports nothing
outside itself, `adapters/` reaches only `domain` and `utils`, and `compare_reports/` is pure
stdlib. The one inverted edge, `runner/service.py` importing `application`, is a documented compat
shim with an explicit allowlist entry.

Code hygiene is clean: no bare excepts, no swallowed exceptions, no mutable default arguments
anywhere in `src/`. 3,125 tests run in 24 seconds at 85% branch coverage against an 80% gate, with
per-module floors enforced by `scripts/check_module_coverage.py`. CI runs Python 3.11 through 3.13,
each with a different random seed, plus Docker provenance, wheel-install smoke, and perf jobs.

The performance basics are handled with care: one pooled `httpx.AsyncClient`, one persistent
WAL-mode SQLite connection, traces externalized to a separate table so history listings do not
deserialize megabytes, and a rate-limit coordinator that shares backoff across concurrent
scenarios instead of each rediscovering the same 429.

The changelog is generated from `changelog.d/` fragments, which is why parallel branches stopped
conflicting on the same lines.

The problem is not quality. It is approachability.

---

## Documentation

### Stale where people actually read

| ID | Sev | Finding |
|---|---|---|
| A1 | P1 | `SKILL.md:9` says "15 categories" and `SKILL.md:107` says `--hardmode` includes 15 Hard Mode scenarios. Verified actual values: `len(list(Category))` is 16, and `ALL_SCENARIOS_WITH_HARDMODE` minus `ALL_SCENARIOS` is 19. This file is the only place exit codes and the JSON output shape are documented, so it gets read. |
| A2 | P1 | `SECURITY.md:36` lists only `1.x` as supported. The project is on 2.6.0, so the table declares the current release line unsupported. |
| A3 | P1 | `CONTRIBUTING.md:109` omits two required steps for adding a scenario: registering in `*_DISPLAY_DETAILS`, and setting `difficulty`. Omitting `difficulty` silently drops the scenario out of `--weight-by-difficulty` scoring. |
| A4 | P1 | `docs/api.md:218` sends integrators to `runner.service`, which `AGENTS.md` documents as a compatibility re-export only. New code should import `application.service`. |
| A5 | P2 | `docs/architecture.md:5` and `:303` link to `CONTRIBUTING.md#adding-a-new-scenario`. No such heading exists, so the link silently lands at the top of the file. |
| A6 | P2 | `REFACTOR.md` is a completed plan. All eight phases are marked complete, yet it still reads as current work and embeds stale numbers ("Coverage was 66%", "2,098 tests", "`bench.py` was a 3489-line monolith"). A contributor could redo landed work. |
| A7 | P2 | `docs/superpowers/` holds four completed agent working plans, mode 0700, linked from nothing, inside the user-facing docs tree. |
| A8 | P2 | `SKILL.md` at the repo root collides with the agent-skill manifest convention, which expects YAML frontmatter this file does not have. The same checkout has `.opencode/skills/*/SKILL.md` files that do use that form, so one filename means two things. Its content is user-facing CLI reference and belongs in `docs/`. |
| A9 | P3 | `.claude/` is not gitignored while `.opencode/` is. The local `settings.local.json` holds LAN IPs and a scratchpad key path. Uncommitted today, one `git add -A` from not being. |
| A10 | P3 | `.agents/` and `.codex/` are empty directories. |

### The README is the biggest barrier to adoption

994 lines, 36 headings, no table of contents. `## Quickstart` spans lines 78 to 671, which is 60%
of the file, with 13 subsections under it. The first real run command sits at line 216, behind
install-as-CLI, dev setup, a 57-line Docker section, updating, and configuration.

Missing outright: a table of contents. A section on reading your report, since run artifacts are
explained only at line 846 under the heading "Run ID and Artifacts", which nobody searches for. A
troubleshooting section. Any link at all to `docs/api.md`, which is orphaned.

Duplicated against `docs/`:

| README lines | Duplicate of | State |
|---|---|---|
| 755 to 845, source tree | `docs/architecture.md:66-241` | Already diverged in wording |
| 672 to 705, API quickstart | `docs/api.md:8-57` | Near-verbatim. README example says `2.5.0`, api.md says `1.8.0`, actual is 2.6.0 |
| 263 to 318, scenario selection | `docs/architecture.md:114-125` | Same five bullets |

`docs/architecture.md` and `docs/methodology.md` are the two best documents in the repo. Both are
accurate against the tree, and architecture.md deliberately refuses to hardcode test counts, which
is why it has not rotted. Almost nothing points at either.

### Docstrings

67% of public symbols documented, 325 of 485. The gaps that matter:

| Coverage | Module | Why it matters |
|---|---|---|
| 3/15 | `domain/measurement.py` | A domain port, a contract other layers implement. `domain/plugin.py` is 7/7 and `domain/adapters.py` is 5/5. The measurement port was added later and never got the same treatment. |
| 2/10 | `adapters/measurement.py` | The implementation of that port |
| 13/20 | `domain/scenarios.py` | The most-read module for a contributor. Defines `ScenarioDefinition`, the scoring functions, and safety gating. |

`evals/scenarios_planning.py` (0/20), `compare_reports/*` (1/11), and `cli/plugin_runners.py`
(1/10) are worse by percentage but matter less.

---

## Structure

61 functions or classes are 120 lines or longer. The load-bearing ones:

| Lines | Args | Location | Note |
|---|---|---|---|
| 760 | 0 | `cli/dispatch.py:293 main` | A flat pipeline, then roughly 8 mutually exclusive mode branches, each already marked by a `# -- Mode --` comment |
| 949 | | `storage/reports.py:113 MarkdownReporter` | Five independent writers sharing only `self._root`. A namespace, not an object. |
| 628 | 3 | `compare_reports/summary.py:340 generate_html` | One enormous HTML/CSS f-string |
| 567 | 3 | `compare_reports/tool_eval.py:262 generate_html` | Same shape |
| 539 | 0 | `cli/legacy_parser.py:23 _make_parser` | One flat argparse declaration |
| 436 | 10 | `cli/spec_live_display.py:56 _build_dashboard` | Pure rendering, splittable per panel |
| 435 | 6 | `cli/spec_live_display.py:547 run_spec_live` | Nesting depth 10, the deepest in the codebase. Interleaves the scrape loop, keypress handling, signal handlers, terminal restore, and rendering. |
| 310 | 14 | `runner/orchestrator.py:323 run_scenario` | |

### Parameter sprawl on one call chain

`api.py run_benchmark` takes 21 arguments, `application/service.py run_benchmark` takes 29 (the
highest arity anywhere), `orchestrator.run_all_scenarios` takes 19, and `run_scenario` takes 14.
The config object that fixes this half-exists as `application/service.py:426 _build_run_config`,
which takes 17, but it is never threaded through.

Separately, `domain/adapters.py:68` bakes 14 parameters into the abstract `chat_completion`, so
every adapter and every test fake has to restate them.

### The CLI is doing the application layer's job

`application/` is 551 lines with exactly one public method. `cli/` is 8,981 lines, and four CLI
modules construct adapters, drive the runner, and write storage themselves:

| Module | Reaches into |
|---|---|
| `cli/dispatch.py` | adapters, runner, storage |
| `cli/pressure.py` | adapters, runner, storage |
| `cli/spec_bench.py` | adapters, runner, storage |
| `cli/plugin_runners.py` | adapters, storage |

Only the tool-call `run` path goes through `BenchmarkService`. The perf, spec-decode, pressure, and
plugin paths each rebuild composition inside the delivery layer. This is the root cause behind both
the oversized functions and the duplication below, and it is why `dispatch.py:main` is 760 lines.

### Duplication

| Removable | Where | Measured |
|---|---|---|
| ~550 lines | `cli/plugin_runners.py`: `_run_gsm8k_benchmark` (273), `_run_mmlu_benchmark` (304), `_run_ifeval_benchmark` (280) | Pairwise line similarity of 77%, 71%, and 67%. All three follow build adapter, nested `async def run()`, progress display, collect, finalize. Each takes 9 parameters and holds a nested `run()` of about 140 lines. |
| ~300 lines | `compare_reports/summary.py` against `tool_eval.py` | 61 duplicated 8-line blocks, seven identical runs of 15 lines or more. Both independently define `_r`, `_tv`, `esc`, `sign`, `pct_cls`, `diff_display`, `short_label`, `dname`, and a byte-identical 10-line `km()`. |
| ~~moderate~~ | `plugins/{gsm8k,mmlu,ifeval}/plugin.py` `render_report_section` (154, 150, 143) | **Reassessed: leave alone.** Measured at 28% to 56% pairwise, against 67% to 77% for the runners. Each renders genuinely different content (a single accuracy figure, per-category tables, constraint breakdowns). A shared abstraction here would cost more than the duplication does. |

The plugin triplication is the highest-value fix in the codebase. `domain/plugin.py` already
defines the `BenchmarkPlugin` ABC and `plugins/registry.py` already dispatches on it. The CLI layer
simply is not using polymorphism it already has, so adding the fourth plugin (HumanEval, named as
future work in `AGENTS.md`) means copying another 280 lines.

Three things that look like duplication and are not: `compare/*.py` are 7-line shims onto the
packaged modules, correctly migrated. `utils/openai_compat.py` is a clean pure-helper extraction
that `adapters/openai_compat.py` imports. `cli/parser.py` is a subcommand-to-flat translator, not a
second parser.

### Dead code

Verified as having zero references across `src/`, `tests/`, `scripts/`, and `compare/`:

- `adapters/measurement.py:148 bind_measurement_client`, 30 lines. Its docstring claims it keeps
  test fakes on the semantic seam. No test uses it.
- `runner/llama_benchy.py:511 run_llama_benchy_sync`
- `evals/helpers.py:205 has_matching_tool_result`, a public helper in a module 15 scenario files
  import from, used by none of them.

Not dead despite looking it: the roughly 30 `check_*` functions in `plugins/ifeval/checkers.py` are
decorator-registered into `_CHECKERS` and dispatched by string ID.

---

## Performance

| ID | Sev | Finding |
|---|---|---|
| F1 | P1 | Blocking SQLite writes inside the async event loop. `application/service.py:410-421` calls the synchronous `repo.checkpoint_scenario_result(...)`, an INSERT with an fsync, directly inside an `async def` callback. There are zero uses of `to_thread` or `run_in_executor` anywhere in `src/`. Harmless at `--parallel 1`. With `--parallel N` it stalls every in-flight request for the duration of each commit, and under contention the 10-second `busy_timeout` blocks the whole loop. |
| F2 | P2 | `get_scenario_results` in `storage/db.py` calls `self.get(run_id)` with the default `include_traces=True`, rehydrating every `raw_log`, then returns only `scores["scenario_results"]` and discards the traces. `cli/history.py:190`, the run-diff path, hits exactly this. Passing `include_traces=False` skips a multi-megabyte read and a full `_merge_traces` rebuild. |
| F3 | P2 | `utils/metadata.py` creates a fresh `AsyncClient` per probe, 6 or more of them, each doing its own TCP and TLS handshake. `probe_backend_hint` and `_probe_engine` then run the fallback ladder (`/metrics`, `/version`, `/v1/models`, `/props`, `/health`) strictly sequentially. Against an unreachable endpoint that is 6 times `_PROBE_TIMEOUT` of dead time before the run starts. **Fixed:** probes share one session, and a connect failure latches it unreachable so the ladder stops. |
| F4 | P2 | Blocking synchronous HTTP inside async paths. `plugins/gsm8k/dataset.py:161` opens a sync `httpx.Client` and pages the HuggingFace REST API in a `while True` loop, called from inside `async def GSM8KPlugin.run()`. **Partly wrong on inspection:** `cli/perf.py`'s sync probe runs before `asyncio.run`, not inside it. The plugin half is real and is **fixed**: all three loaders now run through `asyncio.to_thread`. |
| F5 | P2 | `load_scenario_pack` reads every YAML file twice, once via `load_yaml_scenarios(root)` at `packs.py:77` and again via `pack_content_hash(root)` at `:92`, which does its own glob and `read_bytes`. Cheap today with one shipped pack file, linear in pack size, and fixable by hashing the bytes already read. |
| F6 | P3 | No caching anywhere. Zero uses of `lru_cache` or `functools.cache` in `src/`. `load_yaml_scenarios` re-globs and re-parses on every call. The Python scenario registry is the opposite and correct, built once at import, measured at 0.07s to import 12k lines of scenario definitions. ~~P3~~ **Declined on measurement.** `load_scenario_pack` runs once per pack per run, so there is no repeated call to cache. A path-keyed cache would add a staleness hazard for no gain. |
| F7 | P3 | Redundant serialization. Each tool result is `json.dumps`'d twice per turn, once for the trace line and again in `orchestrator.py:288`. `_repair_json_str` at `orchestrator.py:195-249` parses, discards the result, and re-parses, and the caller parses a third time. ~~P3~~ **Declined on measurement.** `json.dumps` on a representative tool result is 2.3 microseconds, so the duplicate costs about 0.35 ms across a 69-scenario run measured in minutes. The two calls also differ for a string result, where the trace quotes and the message does not, so sharing one value would need a branch that reads worse than the duplication. |
| F8 | ~~P3~~ | **Withdrawn on verification.** `BenchmarkService()` was reported to create `./data/benchmarks.sqlite` even when a run never persists. It does not: every call site either passes an explicit repository (so the database is wanted) or an explicit `None` (which creates nothing), and `--dry-run` writes no files at all. The eager connect in `RunRepository.__init__` also carries a tested contract, since constructing a repository is what migrates an old schema. |

Sequential-by-default execution dominates wall-clock time, since `concurrency` defaults to 1 and
the parallel path logs a warning steering users back to `--parallel 1`. That is a deliberate
reproducibility tradeoff rather than a defect. Fixing F1 is a precondition for ever changing it.

---

## Bugs

| ID | Sev | Finding |
|---|---|---|
| G1 | P1 | One test accounts for 65% of total suite runtime. `tests/test_infra_failure_scoring.py:270` monkeypatches `_rate_limit_delay` to `0.0`, but the 429 handler also calls `RateLimitCoordinator.on_rate_limited()`, which widens `_min_interval` through 0.5, 1, 2, 4, 8, 10 seconds, enforced by a real `asyncio.sleep` in `acquire()` that the test never patches. Measured at 15.51s of a 23.6s suite. The test patches the wrong knob. |
| G2 | P2 | Unclosed `RunRepository` connections. `cli/history.py` constructs repositories at lines 124, 180, and 319 with a single `close()` at line 505. Line 124 never closes at all, and every early return above 505 leaks. `api.py:151` constructs one and never closes it. All fall back on `__del__`, which is non-deterministic and, in WAL mode, leaves `-wal` and `-shm` files behind. |
| G3 | P3 | `tests/conftest.py:48-52` monkeypatches the real `httpx.AsyncClient` class at import time for the whole session, attaching `tokenize`, `models`, `metrics`, `completion`, and `stream_completion`. Any future httpx version adding a method of the same name collides. The docstring admits it is legacy. **Fixed:** the three test modules that need those methods use a `MeasurementTestClient` subclass, and an architecture test rejects the old pattern. |

---

## Tests and tooling

Strong overall. The gaps:

**mypy is not strict.** It does not set `disallow_untyped_defs`, `disallow_any_generics`,
`warn_return_any`, or `no_implicit_reexport`, so a fully unannotated function passes. That matters
because the code leans on `dict[str, Any]` at every boundary: `run_data`, `scores`, `metadata`,
`extra_params`. `files = ["src/tool_eval_bench"]` also means 87k lines of tests are never
type-checked.

**Lint is lenient in one specific way.** `E501` is globally ignored with the note "500+
pre-existing violations; enforced gradually", so the 100-character limit is not actually enforced
and only `ruff format` constrains width. No `SIM`, `UP`, `C4`, `RUF`, `PT`, or `PERF` rule families.

**Local runs do not match CI.** pytest has no `addopts`, so a plain `pytest` runs live and
integration tests; the exclusions live only in CI and pre-commit, duplicated in both. Coverage has
no `fail_under`, so the 80% gate exists only as a CI flag and a local `--cov` run never fails on it.

**Thinnest coverage.** `cli/spec_live_display.py` at 61%, 148 uncovered statements over 981 lines,
the worst in the tree. `plugins/hf_utils.py` at 65%, with the whole `datasets` loader path
untested. `cli/dispatch.py` at 74% against a module floor of only 70%, so it can regress further
before CI notices.

**Fixed.** `spec_live_display.py` is at 85% and `hf_utils.py` at 91%, both with floors so they
cannot drift back. The gaps were the parts that matter most: every previous `run_spec_live` test
scraped `None`, so the loop that computes deltas and renders the dashboard never ran, and the
entire HuggingFace retry ladder, which is what stands between a 429 and a failed download, was
untested. `dispatch.py` still sits just above its floor.

**Test names describe incidents, not units.** `test_critical_fixes.py`, `test_review_fixes.py`,
`test_v122_changes.py`, `test_v130_features.py`, `test_coverage_gaps.py`, and
`test_final_audit_tc01_30.py` through `tc70_88` come to roughly 4k lines whose scope is a date.
Renaming them is not worth the `git blame` damage, but it does mean coverage has to be run to know
what is actually tested.

**CI gaps.** No OS matrix, everything runs on `ubuntu-latest`, though `storage/db.py` uses
`Path.cwd()` and WAL SQLite. `uv.lock` is validated only in the Docker job and never used to
install, so CI can break from an upstream release with no repo change. No `pip-audit`, Dependabot,
or CodeQL, despite a `SECURITY.md`. Coverage is computed but never uploaded. `pre-commit run
--all-files` is absent from CI, and ruff is pinned to `v0.16.0` in pre-commit but floats at
`>=0.12` in CI. No concurrency group, so pushes queue redundant matrix runs. No release workflow
despite `RELEASING.md` and towncrier being set up.

**All fixed.** The matrix gained a macOS runner and pins bash everywhere. The test job installs
with `uv sync --locked`. CodeQL runs the security-and-quality queries per push and
weekly. A version tag builds, smoke-tests the wheel, asserts the built version matches the tag,
checks the packaged scenario tree still resolves to 69 and 88, and opens a *draft* release with
the towncrier notes, leaving publishing to a person.

A Windows runner was added and then removed. It paid for itself immediately by finding a real
scoring bug: without the IANA timezone database, TC-17's offset check accepts both winter and
summer spellings and scores PASS where it scores PARTIAL elsewhere, so `tzdata` is now a win32
dependency. It also surfaced 47 test-side platform assumptions, tracked in issue #93. The package
itself is clean: zero file-I/O sites in `src/` rely on the default encoding, against 81 in
`tests/`.

**Pre-commit lacks the basics.** No `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`,
`check-merge-conflict`, `check-added-large-files`, or `detect-private-key`. The last two matter
with a `.env` in the working tree.

---

## Contributor onboarding

Adding a scenario takes six steps against the four documented. Define the `ScenarioDefinition`.
Append it to `*_SCENARIOS`. Add it to `*_DISPLAY_DETAILS`, which is undocumented. Set `difficulty`
1 to 5, also undocumented, and silently unrated if omitted. Use a sortable `TC-NN` ID, because both
registries do `int(s.id.split("-")[1])` and a bad ID raises at import. Optionally set
`max_turns_override`.

`ScenarioDefinition` has 17 fields, 12 of them optional, and several load-bearing ones appear in no
contributor doc: `tool_choice_after_first_call`, `preserve_reasoning_across_follow_ups`,
`checkpoint`, `held_out`. There is no worked end-to-end example anywhere.

Adding an adapter is genuinely easy and correctly documented. `adapters/factory.py` is 27 lines,
and for an OpenAI-compatible backend you add nothing to the adapter layer at all. This is the
best-factored extension point in the codebase.

Adding a plugin is one dict entry in `plugins/registry.py`, but `docs/architecture.md:305-309`
omits `cli/command_registry.py`, `cli/plugin_runners.py`, `cli/plugin_lifecycle.py`, and
regenerating the committed compat snapshots. Following the doc literally gets you a legacy flag and
no `plugin <name>` subcommand.

---

## Scenario files: folder tree, not YAML

The six `scenarios_*.py` files total 14,253 lines, with 3,194 in `scenarios_agentic.py` alone. They
are hard to navigate, and one file per scenario in a category folder tree fixes that. At roughly
160 lines per scenario that is a good file size, and it makes a scenario reviewable as a single
diff.

A directory-scanning registry would also remove the double-registration trap in A3 at its root:
today a contributor has to append to both `*_SCENARIOS` and `*_DISPLAY_DETAILS`, and forgetting
either fails silently. Making the file's existence the registration removes the whole class of
error.

Moving evaluators to YAML is a different question, and the answer is no. Measured against the 88
current evaluators, which run to a median of 60 lines and a maximum of 207:

| Capability required | Scenarios | Current YAML loader |
|---|---|---|
| Inspects the model's free-text answer | 68/88 (77%) | Not supported |
| Iterates tool calls non-positionally | 47/88 (53%) | Exact ordered match only |
| Inspects tool results | 29/88 (33%) | Not supported |
| Regex over model output | 24/88 (27%) | Not supported |
| Multi-turn follow-ups | 11/88 (12%) | Not supported |

The decisive constraint is scoring. `yaml_loader.py:_make_evaluator` can only return PASS (2 points)
or FAIL (0). `PARTIAL` is emitted solely by `helpers.py:577 partial()`, which only Python evaluators
call, so a YAML scenario cannot express the middle tier of a three-tier benchmark at all.

Beyond that, TC-30 imports Python's `ast` module to decide whether generated code matches an
expected workflow, and several Hard Mode evaluators run multi-step state machines. Expressing those
in YAML means growing a DSL into a programming language, one that mypy cannot check, a debugger
cannot step through, and the 87k-line test suite cannot exercise directly. The folder tree delivers
the navigability win on its own.

YAML keeps its real job: authoring held-out packs. Third parties writing private packs should not
have to ship executable Python, and simple lookup scenarios are what the current schema handles
well. Two small, bounded additions are worth making there: `PARTIAL` support and a text-contains
assertion. Not a general expression language.

---

## Recommended sequence

Each stage lands independently, behind the quality bar in `AGENTS.md`. Two constraints hold
throughout: scores must not move, and no long-lived refactor branch.

| Stage | Work | Effort | Risk |
|---|---|---|---|
| 1 | Fix A1 to A5, A9, A10, and delete the dead code. Add a test asserting documented scenario counts match the registries so A1 cannot recur. | 0.5d | None |
| 2 | README to under 450 lines with a table of contents and a first run inside 40 lines. Extract Docker, benchmarks, spec-decode, and context-pressure sections to `docs/`. Delete the duplicated source tree. Move `SKILL.md` to `docs/cli-reference.md`. | 1d | None |
| 3 | Fix G1, G2, F2, F5. Takes the suite from 24s to roughly 8s. | 1d | Low |
| 4 | Collapse the plugin-runner triplication onto the existing `BenchmarkPlugin` ABC. Removes ~550 lines. | 1-2d | Low |
| 5 | Split `MarkdownReporter` into a `storage/reports/` package and extract `compare_reports/_common.py`. Removes ~300 lines. | 2d | Low |
| 6 | Promote `_build_run_config` to a public `RunConfig` dataclass and thread it through the 21/29/19/14 argument chain. | 2-3d | Medium |
| 7 | Move perf, spec-bench, pressure, and plugin orchestration into `application/`. Split `dispatch.py:main` at its existing mode boundaries. Fix F1 here. Then tighten the boundary test to forbid `cli` to `storage` and `cli` to `runner`. | 3-4d | Medium |
| | **Delivered in part.** F1 is fixed, `main` went from 760 to 577 lines, and CLI database reads moved to `application/run_queries.py` behind a test that forbids `cli` to `storage.db`. The rest was dropped: `cli` may still import `storage.reports`, which is a renderer rather than persistence and which `BenchmarkService` publishes as a constructor argument, and moving perf, spec-bench, pressure, and plugin composition into `application/` would add a layer with one caller while threading console output back through callbacks across ~1,700 lines. | | |
| 8 | Contributor docs for scenarios, plugins, and adapters. Docstrings for the measurement port. Issue templates, CODEOWNERS, CI hardening, config tightening. | 1-2d | None |
| 9 | One file per scenario in a category folder tree with a scanning registry. Lands after stage 8 so the docs are written once. | 3-4d | Low, but verify hard |

### Beyond the nine stages

The plan scheduled findings A through E, F1, F2, F5, G1, G2, H, I's config gaps, J, and K's
docstrings. These were in the report but in no stage, and were done afterwards:

| Finding | Outcome |
|---|---|
| F3 | Fixed. One connection pool per probing session, and the ladder stops at the first connect failure, so a wrong `--base-url` costs one timeout instead of six. |
| F4 | Fixed for the plugins. The `cli/perf.py` half of the finding was wrong: that probe runs before `asyncio.run`, not inside it. |
| F6 | Declined. `load_scenario_pack` runs once per pack per run, so there is nothing to cache. |
| F7 | Declined. Measured at 0.35 ms across a run, and sharing the value needs a branch that reads worse than the duplication. |
| G3 | Fixed. A `MeasurementTestClient` subclass replaces patching the real httpx class, with an architecture test against a relapse. |
| I, coverage | Fixed. The two thinnest modules went to 85% and 91%, with floors. |
| I, CI | Fixed. Lockfile install, macOS and Windows runners, CodeQL, and a tag-triggered draft release. |
| K, parser | Partly fixed. The private `argparse._StoreTrueAction` branch is gone; `parser._actions` stays, because argparse offers no public way to enumerate a parser's options. |

Still open, and deliberately: `cli/dispatch.py` sits just above its 70% floor, and the
incident-named test files keep their names because renaming 4k lines would wreck `git blame` for
no behaviour change.

Verification that applies to every stage, beyond the standard quality bar: `config_fingerprint`
must be unchanged for identical inputs, generated Markdown reports must be byte-identical against a
fixture run, and a full 69-scenario run against a mock endpoint must score identically before and
after.

### Deliberately not doing

Splitting `cli/legacy_parser.py:_make_parser` buys navigability and risks silently changing the
flag surface. Making mypy strict or enabling `E501` are large mechanical diffs across 42k lines
that would bury every real change for weeks; both are better done per-module over time. Replacing
the `argparse._actions` reflection in `cli/parser.py:21-50` is worth doing eventually, since it
depends on private CPython internals, but the fix is larger than it looks. Renaming the
incident-named test files is real churn with no behaviour change.
