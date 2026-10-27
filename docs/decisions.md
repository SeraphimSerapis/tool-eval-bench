# Benchmark scoring decisions

## Visible correctness, safety outcomes, and fixture identity

TC-88 scores observable constraints independently of exposed reasoning. Transport
observations live in diagnostics because reasoning visibility is endpoint-dependent.
Revisit this split only if a backend-independent observable planning contract replaces it.

Safety warnings come from explicit unsafe outcomes, including recovered intermediate
mutations. Category membership alone cannot distinguish an unsafe action from harmless
incompletion. The existing rating threshold remains, gated on an observed violation.
Historical results need fresh execution to establish the new safety outcome field.

Fixture variants retain scenario IDs but record seed, version, and kind in comparison
fingerprints. This avoids multiplying the public registry for alternate environments.
A variant does not replace a private held-out pack or establish empirical difficulty.
Revisit the representation if variants need independently selectable public identities.

## Decision-model scoring

The decision-model benchmark scores headline accuracy against the gold label on the pinned
Typed Decisions test split and reports KL divergence, Brier score, and ECE beside it, never
folded in. A blended number would hide whether accuracy or calibration moved, and the two
predict different deployment risks.

Each case is one request carrying all five questions about its state. Sending one question
per request was rejected: the dataset card says answers change when questions are split, so
scores would stop being comparable with its published figures. The hand-written items were
dropped rather than kept beside the dataset, because they gave no external reference point.

Revisit when the pinned dataset revision changes or the card publishes its ECE definition.

## Web UI access control

`tool-eval-bench serve` requires a per-process session token, printed as a sign-in link and
exchanged for an `HttpOnly`, `SameSite=Strict` cookie. Requests that change state must also
come from the UI's origin and send `application/json`. The server holds `TOOL_EVAL_API_KEY`
and can start runs that spend it, so a page in another tab must not be able to drive it.

Origin and `Host` checks alone were rejected: they stop cross-site forgery but not a DNS
rebinding page, which runs on its own origin with a matching `Host`. Scoping the environment
key to `TOOL_EVAL_BASE_URL` was rejected too, because the CLI sends it to any `--base-url` and
the two entry points would disagree. The token closes both paths for every bind address.

The UI stops only runs it launched and resumes only `interrupted` runs whose stored config it
can rebuild, refusing the rest with a pointer to `tool-eval-bench resume`. Revisit if the UI
gains multi-user access, which would need real accounts instead of one shared token.

## Grouping runs by CLI invocation

Every row one CLI invocation stores carries an explicit `session_id`, generated once per
invocation and stored in its own indexed column. `card` reads a session back by that id.
Matching rows by model and a time window was rejected: two invocations against one model
minutes apart would merge, and a long scored run would split from the perf sweep before it.
The cost is a schema migration and rows from before it that no card can group.

A resume adopts the session of the run it resumes, so the completed scored run stays with the
rows its interrupted invocation wrote. Revisit if a card needs to combine separate invocations
on purpose, which would call for a user-named grouping rather than an implicit one.

## Rendering card PNGs

`card` writes an SVG with the standard library and rasterises it to PNG only through the
optional `card` extra, `resvg-py`. The base install gains no native dependency, and `resvg-py`
ships abi3 wheels for Linux, macOS, and Windows, so the extra needs no system libraries. Without
it, `card --png` still writes the SVG and exits 1 with an install hint.

The PNG renders with bundled subsets of Inter and JetBrains Mono under the SIL OFL, with
system fonts disabled, so the same session gives the same PNG on every machine. The subsets
add about 620 KB to the wheel. resvg ignores `font-feature-settings`, so the Inter subsets
point the digits at their tabular glyphs. `fonts/SOURCES.txt` records how to regenerate them.

CairoSVG was rejected because it needs the system cairo library, which a plain `pip` or
`uv tool` install cannot provide. A headless browser was rejected as far too heavy for one
image. Embedding the fonts in the SVG was rejected because it would add hundreds of
kilobytes to every SVG card. The SVG names the fonts in a fallback stack instead, so a
browser rendering stays close without them.

Revisit if `resvg-py` stops publishing wheels for a supported platform, if a pure-Python
renderer can produce the same output, or if the card needs text features resvg lacks.

## Toolscale: a plugin with structurally defined similarity

The tool-count sweep (#319) is a plugin, not a scenario group or a flag on Category L. A flag
would change TC-37 to TC-40 and break comparison with every stored Category L result, and a grid
of about a thousand generated single-turn items does not fit the hand-graded, three-tier scenario
model or belong in the main score.

Similarity is defined by the catalog's structure (variants of one family, families of one
domain, other domains) and only checked by a TF-IDF metric. Defining it by an embedding model
was rejected: the axis would depend on which embedding model ran, and it would need a new
dependency or a second endpoint. The count axis adds only fillers from other domains, at most one
per family, so the target's neighbourhood stays fixed while N grows.

The headline is two numbers, accuracy when a tool was called and the call rate, not one blended
accuracy. In calibration a mid-sized model scored 83% blended while never picking a wrong tool:
every miss was a clarifying question about an optional parameter. A blend would have reported
abstention as confusion. Blended accuracy stays as the stored score so runs still sort.

Each tool carries a literal query, which names its qualifier, and an implied one, which does
not. Making the literal queries harder in place was rejected because it would remove the clean
control and leave nothing to separate keyword matching from understanding. In calibration
neither cue produced measurable confusion in two models up to 64 tools, so the next difficulty
axes are description quality, crowded fillers, and larger N rather than harder queries.

Crowded fillers draw whole families instead of one tool per family, walking the same family
order, so the target's neighbours and slot layout are unchanged and sparse minus crowded is
paired. Crowded stays opt-in because it adds 75% to an already doubled grid, and in calibration it
changed nothing for two models.

Counts above the sparse maximum build crowded cells only, up to 248 tools from the 272-tool
catalog. Growing the catalog to about 1,040 tools for clean sparse lists at 256 was rejected:
neither calibrated model lost accuracy when called at 248 crowded, so a larger catalog would buy a
cleaner version of a null result. Real-world tools as fillers were also deferred. They mostly
test description quality, and as fillers they would move count and description quality together;
description quality needs its own paired axis.

Description quality is an axis of the same catalog rather than real-world tools. Terse shows one
generic line per family, so every near-duplicate reads the same and only the name differs; bare
removes descriptions entirely. Per-tool truncation or mutation was rejected because it would
make each degraded description favour or disfavour its own tool unevenly, and real-world fillers
because they would change count and description quality together. In calibration this was the
only condition that produced near-sibling confusion.

Revisit if labs need a continuous similarity axis, crowding inside the target's own domain, or a
model shows a count effect on crowded lists that sparse lists would need to disentangle.
