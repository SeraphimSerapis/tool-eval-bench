// Stored runs rail, multi-select deletion, and Run Inspector view.

import {
  absTime,
  byId,
  fmtFigure,
  fmtNumber,
  fmtSeconds,
  fmtTokens,
  h,
  icon,
  median,
  pctOf,
  prefersReducedMotion,
  relTime,
  setChildren,
  splitModel,
  tierFor,
} from "./dom.js";
import { fetchJson, postJson, state } from "./state.js";
import {
  banner,
  bar,
  chip,
  emptyState,
  hasFigure,
  isToolEval,
  legend,
  metric,
  modelBlock,
  ppfBar,
  preserveScroll,
  ring,
  segmented,
  stat,
  tile,
  typeChip,
} from "./components.js";

let nav = null;

export function initHistory(navApi) {
  nav = navApi;
}

export async function loadRunsList() {
  const search = byId("history-search")?.value || "";
  const status = byId("history-status-filter")?.value || "";
  const runType = byId("history-type-filter")?.value || "";
  const params = new URLSearchParams({ limit: "100" });
  if (search) params.set("q", search);
  if (status) params.set("status", status);
  if (runType) params.set("run_type", runType);

  const container = byId("runs-list-container");
  const seq = ++state.runsRequestSeq;
  try {
    const data = await fetchJson(`/api/runs?${params.toString()}`);
    if (seq !== state.runsRequestSeq) return;
    if (data.version) {
      const pill = byId("app-version");
      if (pill) {
        pill.textContent = `v${String(data.version).split("+")[0]}`;
        pill.title = `tool-eval-bench ${data.version}`;
        pill.hidden = false;
      }
    }
    state.runs = data.runs || [];
    // Only runs still on screen may stay selected, so a delete never
    // reaches a run the current filters hide.
    const shown = new Set(state.runs.map((r) => r.run_id));
    for (const id of state.deleteSelection) if (!shown.has(id)) state.deleteSelection.delete(id);
    renderRunsList();
    updateSelectBar();
    if (!state.selectedRunId && state.runs.length > 0) {
      selectRun(state.runs[0].run_id);
    } else if (!state.runs.length && !state.selectedRunId) {
      renderInspector();
    }
  } catch (err) {
    if (seq !== state.runsRequestSeq) return;
    setChildren(container, emptyState("alert", "Could not load runs", err.message));
  }
}

let searchDebounce = null;
export function scheduleRunsListLoad() {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(loadRunsList, 250);
}

function runCard(run) {
  const selected = run.run_id === state.selectedRunId;
  const picked = state.selecting && state.deleteSelection.has(run.run_id);
  const score = run.final_score;
  return h(
    "button",
    {
      type: "button",
      class: state.selecting ? "run-card run-item is-selecting" : "run-card run-item",
      dataset: { runId: run.run_id },
      "aria-current": selected && !state.selecting ? "true" : null,
      "aria-pressed": state.selecting ? String(picked) : null,
      onclick: () => (state.selecting ? toggleDeleteSelection(run.run_id) : selectRun(run.run_id)),
    },
    state.selecting ? h("span", { class: "run-check", "aria-hidden": "true" }, picked ? icon("check") : null) : null,
    modelBlock(run.model, "run-model"),
    run.reviewer_model ? h("span", { class: "run-meta muted", text: `+ reviewer ${run.reviewer_model}` }) : null,
    isToolEval(run)
      ? ring(score, {
          label: score === null || score === undefined ? null : Math.round(score),
          title:
            score === null || score === undefined
              ? run.rating === "Withheld"
                ? "Score withheld (no gradable scenarios)"
                : `No score, ${run.status}`
              : `Score ${score} of 100`,
        })
      : metric(run.headline),
    h(
      "div",
      { class: "run-meta" },
      chip(run.status),
      isToolEval(run) ? null : typeChip(run),
      isToolEval(run) || (run.backend && run.backend !== "unknown") ? h("span", { text: run.backend || "unknown" }) : null,
      h("span", { title: absTime(run.created_at), text: relTime(run.created_at) }),
      run.label ? h("span", { class: "chip chip-accent" }, icon("tag"), run.label) : null,
      run.safety_warnings_count > 0
        ? h("span", { class: "chip chip-fail", title: "Safety warnings" }, icon("shield"), run.safety_warnings_count)
        : null
    )
  );
}

export function renderRunsList() {
  const container = byId("runs-list-container");
  if (!container) return;
  const count = byId("runs-count");
  if (count) count.textContent = state.runs.length ? String(state.runs.length) : "";
  if (!state.runs.length) {
    setChildren(
      container,
      emptyState("inbox", "No matching runs", "Run a benchmark from the CLI, or start one from Launch & Live.")
    );
    return;
  }
  setChildren(container, ...state.runs.map(runCard));
}

// ------------------------------------------------------- Deleting runs

export function setSelecting(on) {
  state.selecting = on;
  state.deleteSelection.clear();
  const toggle = byId("runs-select-toggle");
  if (toggle) {
    toggle.setAttribute("aria-pressed", String(on));
    toggle.textContent = on ? "Done" : "Select";
  }
  const selectBar = byId("runs-select-bar");
  if (selectBar) selectBar.hidden = !on;
  renderRunsList();
  updateSelectBar();
}

function toggleDeleteSelection(runId) {
  if (state.deleteSelection.has(runId)) state.deleteSelection.delete(runId);
  else state.deleteSelection.add(runId);
  renderRunsList();
  updateSelectBar();
  // Re-rendering replaced the button; keep keyboard focus on the same run.
  byId("runs-list-container")?.querySelector(`[data-run-id="${CSS.escape(runId)}"]`)?.focus();
}

function updateSelectBar() {
  const count = state.deleteSelection.size;
  const label = byId("runs-select-count");
  if (label) label.textContent = `${count} selected`;
  const del = byId("runs-delete-selected");
  if (del) del.disabled = count === 0;
  const all = byId("runs-select-all");
  if (all) {
    all.textContent = count > 0 && count === state.runs.length ? "Clear" : "Select all";
    all.disabled = state.runs.length === 0;
  }
}

export function toggleSelectAll() {
  if (state.runs.length && state.deleteSelection.size === state.runs.length) state.deleteSelection.clear();
  else for (const run of state.runs) state.deleteSelection.add(run.run_id);
  renderRunsList();
  updateSelectBar();
}

function setRunsStatus(message, kind = "") {
  const el = byId("runs-status");
  if (!el) return;
  el.textContent = message;
  el.className = kind ? `rail-status is-${kind}` : "rail-status";
}

// Resolves true only when the user picked Delete; Esc, a click outside,
// and Cancel all resolve false.
function confirmDelete(runs) {
  const dialog = byId("delete-dialog");
  if (!dialog) return Promise.resolve(false);
  const count = runs.length;
  const shown = runs.slice(0, 5);
  byId("delete-dialog-title").textContent = count === 1 ? "Delete this run?" : `Delete ${count} runs?`;
  setChildren(
    byId("delete-dialog-body"),
    h("p", {
      text:
        count === 1
          ? "The run and its Markdown report are deleted. This cannot be undone."
          : "The runs and their Markdown reports are deleted. This cannot be undone.",
    }),
    h(
      "ul",
      { class: "dialog-list" },
      shown.map((r) => h("li", {}, h("span", { text: splitModel(r.model).name }), h("span", { class: "mono muted", text: r.run_id })))
    ),
    count > shown.length ? h("p", { class: "muted", text: `and ${count - shown.length} more` }) : null,
    runs.some((r) => r.status === "running")
      ? h("p", { class: "dialog-note", text: "A run still executing in another process writes its row back when it finishes." })
      : null
  );
  dialog.returnValue = "";
  return new Promise((resolve) => {
    dialog.addEventListener(
      "close",
      () => {
        const confirmed = dialog.returnValue === "delete";
        dialog.returnValue = "";
        resolve(confirmed);
      },
      { once: true }
    );
    dialog.showModal();
  });
}

export async function deleteRuns(runs) {
  if (!runs.length || !(await confirmDelete(runs))) return;
  setRunsStatus("");
  let res;
  try {
    res = await postJson("/api/runs/delete", { run_ids: runs.map((r) => r.run_id) });
  } catch (err) {
    setRunsStatus(`Could not delete: ${err.message}`, "error");
    return;
  }
  const gone = new Set([...res.deleted, ...res.missing]);
  if (gone.has(state.selectedRunId)) {
    state.selectedRunId = null;
    state.selectedRun = null;
    // Clear the deleted run now; loadRunsList then opens the newest one.
    renderInspector();
  }
  if (gone.has(state.compareLeftId)) state.compareLeftId = "";
  if (gone.has(state.compareRightId)) state.compareRightId = "";
  if (!state.compareLeftId || !state.compareRightId) state.lastComparison = null;
  if (gone.has(state.liveRunId)) {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    state.liveRunId = "";
    state.liveActiveScenario = null;
    state.liveTurns = [];
    state.liveResults = [];
    state.liveNote = null;
    nav?.onRunsDeleted();
  }
  if (state.selecting) setSelecting(false);
  const runCount = res.deleted.length;
  const reportCount = res.reports_removed;
  let message = `Deleted ${runCount} ${runCount === 1 ? "run" : "runs"} and ${reportCount} ${reportCount === 1 ? "report" : "reports"}.`;
  if (res.reports_kept.length) {
    const kept = res.reports_kept.length;
    message += ` Kept ${kept} ${kept === 1 ? "report" : "reports"}: ${res.reports_kept[0].reason}.`;
  }
  setRunsStatus(message, res.reports_kept.length ? "warn" : "");
  await loadRunsList();
}

export async function selectRun(runId) {
  state.selectedRunId = runId;
  state.expandedScenarios.clear();
  state.scenarioFilter = "all";
  renderRunsList();
  const pane = byId("run-inspector");
  if (!pane) return;
  const seq = ++state.selectRunSeq;
  try {
    const run = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
    if (seq !== state.selectRunSeq) return;
    state.selectedRun = run;
    pane.scrollTop = 0;
    renderInspector({ animate: true });
  } catch (err) {
    if (seq !== state.selectRunSeq) return;
    setChildren(pane, emptyState("alert", `Could not load run ${runId}`, err.message));
  }
}

const SCENARIO_FILTERS = [
  ["all", "All", () => true],
  ["fail", "Fail", (r) => r.status === "fail"],
  ["partial", "Partial", (r) => r.status === "partial"],
  ["pass", "Pass", (r) => r.status === "pass"],
  ["safety", "Safety", (r) => Boolean(r.safety_critical || r.safety_violation)],
  ["audited", "Audited", (r) => Boolean(r.decision_audit)],
];

function toggleScenario(sid) {
  if (state.expandedScenarios.has(sid)) state.expandedScenarios.delete(sid);
  else state.expandedScenarios.add(sid);
  const pane = byId("run-inspector");
  if (pane) preserveScroll(pane, () => renderInspector());
}

function revealScenario(sid) {
  state.scenarioFilter = "all";
  state.expandedScenarios.add(sid);
  const pane = byId("run-inspector");
  if (!pane) return;
  preserveScroll(pane, () => renderInspector());
  const row = pane.querySelector(`[data-scenario-id="${CSS.escape(sid)}"]`);
  row?.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  row?.querySelector(".disclosure")?.focus({ preventScroll: true });
}

function scenarioDetail(r, colspan) {
  const calls = r.tool_calls_made || [];
  const audit = r.decision_audit;
  const meta = [
    ["Expected", r.expected_behavior || "—"],
    ["Tool calls", calls.length ? calls.join(", ") : "None"],
    ["Turns", r.turn_count ?? 1],
    ["Tokens", (r.prompt_tokens || 0) + (r.completion_tokens || 0)],
    ["Time to first token", r.ttft_ms ? `${Math.round(r.ttft_ms)} ms` : "—"],
    r.safety_violation ? ["Safety violation", r.safety_violation] : null,
    audit ? ["Decision audit", `${audit.choice || audit.status}${audit.disagreement ? " (disagrees)" : ""}`] : null,
  ].filter(Boolean);
  return h(
    "tr",
    { class: "detail-row" },
    h(
      "td",
      { colspan },
      h(
        "div",
        { class: "detail" },
        h("dl", { class: "detail-meta" }, meta.map(([k, v]) => h("div", {}, h("dt", { text: k }), h("dd", { text: v })))),
        h("pre", { class: "trace", tabindex: "0", "aria-label": `Trace for ${r.scenario_id}`, text: r.raw_log || "No raw trace available." })
      )
    )
  );
}

function scenarioRows(results) {
  const rows = [];
  for (const r of results) {
    const sid = r.scenario_id;
    const open = state.expandedScenarios.has(sid);
    rows.push(
      h(
        "tr",
        {
          class: `scenario-summary-row row-click ${open ? "row-open" : ""}`,
          dataset: { scenarioId: sid },
          onclick: (ev) => {
            if (!ev.target.closest("button")) toggleScenario(sid);
          },
        },
        h(
          "td",
          {},
          h(
            "button",
            {
              type: "button",
              class: "disclosure",
              "aria-expanded": String(open),
              onclick: () => toggleScenario(sid),
            },
            icon("chevron"),
            h("span", { class: "mono muted", text: sid })
          )
        ),
        h(
          "td",
          {},
          h("span", { class: "cell-title", text: r.title || sid }),
          r.held_out ? [" ", h("span", { class: "chip chip-plain", text: "held out" })] : null,
          r.decision_audit
            ? [
                " ",
                h("span", {
                  class: `chip ${r.decision_audit.disagreement ? "chip-partial" : "chip-info"}`,
                  text: `audit: ${r.decision_audit.choice || r.decision_audit.status}`,
                }),
              ]
            : null
        ),
        h("td", {}, h("span", { class: "cat-letter", title: r.category_label || r.category, text: r.category || "?" })),
        h(
          "td",
          { class: "nowrap" },
          chip(r.status),
          r.safety_violation ? [" ", h("span", { class: "chip chip-fail", title: r.safety_violation }, icon("shield"))] : null
        ),
        h("td", { class: "num", text: `${r.points ?? 0}/2` }),
        h(
          "td",
          { class: "num nowrap" },
          fmtSeconds(r.duration_seconds),
          r.ttft_ms ? h("span", { class: "muted", text: ` · ${Math.round(r.ttft_ms)}ms` }) : null
        ),
        h("td", { class: "cell-summary", text: r.summary || "" })
      )
    );
    if (open) rows.push(scenarioDetail(r, 7));
  }
  return rows;
}

function mapStats(results) {
  const latency = median(results.map((r) => r.duration_seconds));
  const ttft = median(results.map((r) => r.ttft_ms));
  const turns = results.reduce((sum, r) => sum + (r.turn_count || 1), 0) / Math.max(results.length, 1);
  const item = (label, value) => h("div", {}, h("dt", { text: label }), h("dd", { class: "mono", text: value }));
  return h(
    "dl",
    { class: "detail-meta map-stats" },
    item("Median scenario time", latency === null ? "—" : fmtSeconds(latency)),
    item("Median time to first token", ttft === null ? "—" : `${Math.round(ttft)} ms`),
    item("Average turns", turns.toFixed(1))
  );
}

// The other runs the same CLI invocation stored: a scored run's spec-decode,
// throughput, needle, or accuracy rows, or the scored run of one of those.
function sessionCard(entries) {
  if (!entries || !entries.length) return null;
  const figure = (entry) => {
    if (entry.final_score !== null && entry.final_score !== undefined) return `Score ${Math.round(entry.final_score)}`;
    if (isToolEval(entry) && entry.rating === "Withheld") return "Score withheld";
    if (hasFigure(entry.headline)) return `${entry.headline.label} ${fmtFigure(entry.headline.value, entry.headline.unit)}`;
    return "Nothing measured";
  };
  return h(
    "section",
    { class: "card card-pad session-card", "aria-labelledby": "session-card-title" },
    h("div", { class: "eyebrow", id: "session-card-title", text: `Same session · ${entries.length}` }),
    h("p", { class: "muted session-hint", text: "Stored by the same command." }),
    h(
      "ul",
      { class: "session-list" },
      entries.map((entry) =>
        h(
          "li",
          {},
          h(
            "button",
            { type: "button", class: "session-item", dataset: { runId: entry.run_id }, onclick: () => selectRun(entry.run_id) },
            typeChip(entry),
            h("span", { class: "session-figure", text: figure(entry) }),
            hasFigure(entry.headline) && entry.headline.detail ? h("span", { class: "muted", text: entry.headline.detail }) : null,
            h("span", { class: "session-end" }, chip(entry.status), h("span", { class: "muted", title: absTime(entry.created_at), text: relTime(entry.created_at) }))
          )
        )
      )
    )
  );
}

// Stats and a measurement table for a run that is not a tool-call run.
function modeBody(detail) {
  const stats = detail && detail.stats ? detail.stats : [];
  const table = detail && detail.table ? detail.table : null;
  if (!stats.length && !table) {
    return [h("div", { class: "section-spacer" }, banner("info", "info", "This run stored no measurements to show."))];
  }
  const out = [];
  if (stats.length) out.push(h("div", { class: "stats" }, stats.map((s) => stat(s.label, fmtFigure(s.value, s.unit)))));
  if (table) {
    const numeric = table.columns.map((c) => table.rows.some((row) => typeof row[c.key] === "number"));
    out.push(
      h("h2", { class: "section-title" }, "Measurements", h("span", { class: "count", text: table.rows.length })),
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table" },
          h(
            "thead",
            {},
            h("tr", {}, table.columns.map((c, i) => h("th", { scope: "col", class: numeric[i] ? "num" : null, text: c.unit ? `${c.label} (${c.unit})` : c.label })))
          ),
          h(
            "tbody",
            {},
            table.rows.map((row) =>
              h(
                "tr",
                {},
                table.columns.map((c, i) => {
                  const value = row[c.key];
                  const text = value === null || value === undefined || value === "" ? "—" : typeof value === "number" ? fmtNumber(value) : String(value);
                  return h("td", { class: numeric[i] ? "num" : null, text });
                })
              )
            )
          )
        )
      )
    );
  }
  return out;
}

export function renderInspector(opts = {}) {
  const pane = byId("run-inspector");
  if (!pane) return;
  const run = state.selectedRun;
  if (!run) {
    setChildren(
      pane,
      emptyState("layers", "Pick a run to inspect", "Scores, category breakdowns, and full multi-turn traces appear here.")
    );
    return;
  }
  const scores = run.scores || {};
  const meta = run.metadata || {};
  const warnings = scores.safety_warnings || [];
  const categories = scores.category_scores || [];
  const results = scores.scenario_results || [];
  const total = results.length;
  const { org, name } = splitModel(run.model);

  const toolEval = isToolEval(run);
  const actions = [
    // Compare pairs scenario results, which only tool-call runs have.
    toolEval
      ? h(
          "button",
          {
            type: "button",
            class: "btn",
            id: "inspect-compare-btn",
            onclick: () => {
              state.compareLeftId = run.run_id;
              nav?.switchView("compare");
            },
          },
          icon("swap"),
          "Compare"
        )
      : null,
    run.status === "running"
      ? h("button", { type: "button", class: "btn btn-primary", id: "inspect-watch-live-btn", onclick: () => nav?.openLiveRun(run.run_id) }, icon("eye"), "Watch live")
      : null,
    run.status === "interrupted"
      ? h("button", { type: "button", class: "btn btn-primary", id: "inspect-resume-btn", onclick: () => nav?.resumeRun(run.run_id) }, icon("play"), "Resume")
      : null,
    h("button", { type: "button", class: "btn btn-danger", id: "inspect-delete-btn", onclick: () => deleteRuns([run]) }, icon("trash"), "Delete"),
  ];

  const hero = h(
    "div",
    { class: "card hero" },
    toolEval
      ? ring(run.final_score, {
          cls: "ring-lg",
          animate: opts.animate,
          sub:
            run.final_score !== null && run.final_score !== undefined
              ? "of 100"
              : run.rating === "Withheld"
                ? "withheld"
                : null,
        })
      : metric(run.headline, { cls: "metric-lg", withLabel: true }),
    h(
      "div",
      { class: "hero-title" },
      org ? h("span", { class: "hero-org", text: org }) : null,
      h("h1", { class: "hero-name", text: name, title: run.model }),
      h(
        "div",
        { class: "hero-meta" },
        chip(run.status),
        toolEval ? null : typeChip(run),
        run.rating ? h("span", { class: "meta-item", text: run.rating }) : null,
        run.reviewer_model
          ? h("span", { class: "meta-item", title: "The score belongs to the model and its reviewer together", text: `+ reviewer ${run.reviewer_model}` })
          : null,
        h("span", { class: "meta-item mono", title: "Run ID", text: run.run_id }),
        toolEval || (run.backend && run.backend !== "unknown")
          ? h("span", { class: "meta-item" }, icon("server"), run.backend || "unknown")
          : null,
        !toolEval && run.headline && run.headline.detail ? h("span", { class: "meta-item", text: run.headline.detail }) : null,
        meta.engine_name ? h("span", { class: "meta-item" }, icon("cpu"), `${meta.engine_name} ${meta.engine_version || ""}`.trim()) : null,
        meta.quantization ? h("span", { class: "meta-item mono", text: meta.quantization }) : null,
        h("span", { class: "meta-item", title: absTime(run.created_at) }, icon("calendar"), relTime(run.created_at)),
        run.label ? h("span", { class: "chip chip-accent" }, icon("tag"), run.label) : null,
        run.held_out_count ? h("span", { class: "chip chip-plain", text: `${run.held_out_count} held out` }) : null
      )
    ),
    h("div", { class: "hero-actions" }, actions)
  );

  const session = sessionCard(run.session);
  if (!toolEval) {
    setChildren(pane, hero, session, ...modeBody(run.detail));
    return;
  }

  const p = run.passes || 0;
  const q = run.partials || 0;
  const f = run.fails || 0;
  const stats = h(
    "div",
    { class: "stats" },
    stat(
      "Pass / partial / fail",
      h("span", {}, h("span", { class: "delta-pos", text: p }), " / ", h("span", { text: q }), " / ", h("span", { class: "delta-neg", text: f })),
      {
        extra: ppfBar(p, q, f),
      }
    ),
    stat("Points", run.total_points ?? 0, {
      small: `/ ${run.max_points ?? 0}`,
      extra: bar(pctOf(run.total_points || 0, run.max_points || 0), tierFor(pctOf(run.total_points || 0, run.max_points || 0))),
    }),
    stat("Deployability", run.deployability ?? "—", { icon: "zap", small: run.deployability !== null && run.deployability !== undefined ? "/ 100" : null }),
    stat("Responsiveness", run.responsiveness ?? "—", { icon: "clock", small: run.responsiveness !== null && run.responsiveness !== undefined ? "/ 100" : null }),
    scores.total_tokens ? stat("Tokens", fmtTokens(scores.total_tokens), { icon: "message" }) : null,
    scores.degenerate_loop_scenarios > 0
      ? stat("Degenerate loops", scores.degenerate_loop_scenarios, { cls: "stat-fail", icon: "alert" })
      : null,
    scores.answer_in_thinking_turns > 0
      ? stat("Answer in thinking", scores.answer_in_thinking_turns, {
          icon: "message",
          small: scores.answer_in_thinking_turns === 1 ? "turn" : "turns",
        })
      : null,
    warnings.length ? stat("Safety warnings", warnings.length, { cls: "stat-fail", icon: "shield" }) : null
  );

  const safety = warnings.length
    ? h(
        "div",
        { class: "section-spacer" },
        banner("danger", "shield", h("strong", { text: `Safety gate warnings (${warnings.length})` }), h("ul", {}, warnings.map((w) => h("li", { text: String(w) }))))
      )
    : null;

  const catCard = categories.length
    ? h(
        "div",
        { class: "card card-pad" },
        h("div", { class: "eyebrow", text: "Category breakdown" }),
        h(
          "ul",
          { class: "cat-list", "aria-label": "Category scores" },
          categories.map((cs) => {
            const pct = Number(cs.percent || 0);
            return h(
              "li",
              { class: `cat-row ${tierFor(pct)}` },
              h("span", { class: "cat-letter", text: cs.category }),
              h("span", { class: "cat-name" }, h("span", { text: cs.label || cs.category }), bar(pct, tierFor(pct))),
              h("span", { class: "cat-pct", text: `${Math.round(pct)}%` }),
              h("span", { class: "cat-pts", text: `${cs.earned}/${cs.max_points}` })
            );
          })
        )
      )
    : null;

  const mapCard = total
    ? h(
        "div",
        { class: "card card-pad" },
        h("div", { class: "eyebrow", text: `Scenario map · ${total}` }),
        h(
          "div",
          { class: "tile-grid", role: "group", "aria-label": "Scenario outcomes; select one to open its trace" },
          results.map((r) =>
            tile(r.held_out ? "held" : r.status, `${r.scenario_id} · ${r.title || ""} · ${r.status}`, {
              pressed: state.expandedScenarios.has(r.scenario_id),
              onclick: () => revealScenario(r.scenario_id),
            })
          )
        ),
        legend(),
        mapStats(results)
      )
    : null;

  const filterDef = SCENARIO_FILTERS.find(([id]) => id === state.scenarioFilter) || SCENARIO_FILTERS[0];
  const filtered = results.filter(filterDef[2]);
  const filterOptions = SCENARIO_FILTERS.filter(([id, , test]) => id !== "audited" || results.some(test)).map(([id, label, test]) => [
    id,
    label,
    results.filter(test).length,
  ]);

  const traces = total
    ? [
        h("h2", { class: "section-title" }, "Scenario traces", h("span", { class: "count", text: `${filtered.length} of ${total}` })),
        h(
          "div",
          { class: "toolbar" },
          segmented(
            filterOptions,
            state.scenarioFilter,
            (id) => {
              state.scenarioFilter = id;
              preserveScroll(pane, () => renderInspector());
            },
            "Filter scenarios"
          ),
          h("span", { class: "muted", text: "Select a row to open its multi-turn trace." })
        ),
        h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            { class: "table" },
            h(
              "thead",
              {},
              h("tr", {}, ["Scenario", "Title", "Cat", "Verdict", "Pts", "Latency", "Summary"].map((t, i) => h("th", { class: i === 4 || i === 5 ? "num" : null, scope: "col", text: t })))
            ),
            h("tbody", {}, filtered.length ? scenarioRows(filtered) : h("tr", {}, h("td", { colspan: 7, class: "muted", text: "No scenarios match this filter." })))
          )
        ),
      ]
    : [h("div", { class: "section-spacer" }, banner("info", "info", "This run has no per-scenario results to show."))];

  setChildren(
    pane,
    hero,
    session,
    safety,
    stats,
    catCard || mapCard ? h("div", { class: "two-col" }, catCard, mapCard) : null,
    ...traces
  );
}
