// tool-eval-bench web UI.
//
// Every element is built with document.createElement and textContent; no
// HTML strings are parsed, so stored run data (model output, tool results,
// labels) can never become markup. Styles live in styles.css; the only
// dynamic styling is a numeric custom property set through the CSSOM.
(() => {
  "use strict";

  // ------------------------------------------------------------------ DOM

  const SVG_NS = "http://www.w3.org/2000/svg";

  // Stroke icons as [tag, attributes] lists, drawn on a 24x24 grid.
  const ICONS = {
    check: [["polyline", { points: "20 6 9 17 4 12" }]],
    x: [["path", { d: "M18 6 6 18M6 6l12 12" }]],
    minus: [["path", { d: "M5 12h14" }]],
    pause: [["path", { d: "M9 5v14M15 5v14" }]],
    play: [["polygon", { points: "6 4 20 12 6 20 6 4" }]],
    stop: [["rect", { x: "6", y: "6", width: "12", height: "12", rx: "2" }]],
    activity: [["polyline", { points: "22 12 18 12 15 21 9 3 6 12 2 12" }]],
    chevron: [["polyline", { points: "9 6 15 12 9 18" }]],
    search: [["circle", { cx: "11", cy: "11", r: "7" }], ["path", { d: "M21 21l-4.3-4.3" }]],
    shield: [["path", { d: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z" }]],
    alert: [
      ["path", { d: "M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" }],
      ["path", { d: "M12 9v4M12 17h.01" }],
    ],
    info: [["circle", { cx: "12", cy: "12", r: "9" }], ["path", { d: "M12 11v5M12 8h.01" }]],
    server: [
      ["rect", { x: "3", y: "4", width: "18", height: "7", rx: "2" }],
      ["rect", { x: "3", y: "13", width: "18", height: "7", rx: "2" }],
      ["path", { d: "M7 7.5h.01M7 16.5h.01" }],
    ],
    cpu: [
      ["rect", { x: "6", y: "6", width: "12", height: "12", rx: "2" }],
      ["path", { d: "M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" }],
    ],
    calendar: [
      ["rect", { x: "3", y: "5", width: "18", height: "16", rx: "2" }],
      ["path", { d: "M3 10h18M8 3v4M16 3v4" }],
    ],
    tag: [["path", { d: "M20 12l-8 8-9-9V3h8l9 9z" }], ["circle", { cx: "7.5", cy: "7.5", r: "1.5" }]],
    download: [["path", { d: "M12 3v12M7 10l5 5 5-5M5 21h14" }]],
    radio: [
      ["circle", { cx: "12", cy: "12", r: "2" }],
      ["path", { d: "M16.2 7.8a6 6 0 0 1 0 8.4M7.8 16.2a6 6 0 0 1 0-8.4M19.1 4.9a10 10 0 0 1 0 14.2M4.9 19.1a10 10 0 0 1 0-14.2" }],
    ],
    wrench: [
      ["path", { d: "M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.5-.5-.5-2.5 2.5-2.5z" }],
    ],
    message: [["path", { d: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" }]],
    clock: [["circle", { cx: "12", cy: "12", r: "9" }], ["polyline", { points: "12 7 12 12 15 14" }]],
    zap: [["polygon", { points: "13 2 3 14 12 14 11 22 21 10 12 10 13 2" }]],
    swap: [["path", { d: "M7 4v16M3 8l4-4 4 4M17 20V4M21 16l-4 4-4-4" }]],
    inbox: [
      ["path", { d: "M22 12h-6l-2 3h-4l-2-3H2" }],
      ["path", { d: "M5.5 5.1 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.5-6.9A2 2 0 0 0 16.8 4H7.2a2 2 0 0 0-1.7 1.1z" }],
    ],
    trophy: [
      ["path", { d: "M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4zM17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3" }],
    ],
    sliders: [["path", { d: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" }]],
    layers: [["path", { d: "M12 2 2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" }]],
    arrowRight: [["path", { d: "M5 12h14M12 5l7 7-7 7" }]],
    eye: [
      ["path", { d: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z" }],
      ["circle", { cx: "12", cy: "12", r: "3" }],
    ],
  };

  function icon(name) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    for (const [tag, attrs] of ICONS[name] || []) {
      const node = document.createElementNS(SVG_NS, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      svg.appendChild(node);
    }
    return svg;
  }

  function append(el, children) {
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  // h("div", { class, text, dataset, style, onclick, ...attributes }, ...children)
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    let value;
    for (const [key, val] of Object.entries(props || {})) {
      if (val === null || val === undefined || val === false) continue;
      if (key === "class") el.className = val;
      else if (key === "text") el.textContent = String(val);
      else if (key === "value") value = val;
      else if (key === "dataset") Object.assign(el.dataset, val);
      else if (key === "style") {
        for (const [prop, v] of Object.entries(val)) el.style.setProperty(prop, String(v));
      } else if (key.startsWith("on") && typeof val === "function") {
        el.addEventListener(key.slice(2), val);
      } else el.setAttribute(key, val === true ? "" : String(val));
    }
    append(el, children);
    // Set after the children exist, so a <select> can select its option.
    if (value !== undefined) el.value = String(value);
    return el;
  }

  // replaceChildren() would stringify a null child into the text "null".
  function setChildren(el, ...children) {
    if (!el) return null;
    el.replaceChildren();
    return append(el, children);
  }

  const byId = (id) => document.getElementById(id);
  const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- State

  const state = {
    activeView: "history",
    runs: [],
    selectedRunId: null,
    selectedRun: null,
    scenarioFilter: "all",
    expandedScenarios: new Set(),
    leaderboardCohort: "",
    leaderboardQuery: "",
    compareLeftId: "",
    compareRightId: "",
    compareFilter: "all",
    expandedCompareScenarios: new Set(),
    lastComparison: null,
    launcherSchema: null,
    probedModels: [],
    liveRunId: null,
    liveRunStatus: null,
    liveModel: null,
    liveTotal: 0,
    liveActiveScenario: null,
    liveTurns: [],
    liveResults: [],
    liveManaged: true,
    liveNote: null,
    eventSource: null,
    liveViewReady: null,
    runsRequestSeq: 0,
    selectRunSeq: 0,
    compareRuns: [],
  };

  // ------------------------------------------------------------ Formatting

  function splitModel(model) {
    const parts = String(model || "unknown").split("/").filter(Boolean);
    const name = parts.pop() || "unknown";
    return { org: parts.length ? parts[parts.length - 1] : "", name };
  }

  function tierFor(pct) {
    if (pct === null || pct === undefined || Number.isNaN(Number(pct))) return "";
    if (pct >= 75) return "tier-pass";
    if (pct >= 45) return "tier-partial";
    return "tier-fail";
  }

  function heatClass(pct) {
    if (pct >= 90) return "heat-90";
    if (pct >= 75) return "heat-75";
    if (pct >= 60) return "heat-60";
    return "heat-low";
  }

  function parseDate(iso) {
    if (!iso) return null;
    const text = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`;
    const date = new Date(text);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  const relFormatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

  function relTime(iso) {
    const date = parseDate(iso);
    if (!date) return iso ? String(iso).slice(0, 16).replace("T", " ") : "—";
    const seconds = (date.getTime() - Date.now()) / 1000;
    const units = [
      ["year", 31536000],
      ["month", 2592000],
      ["week", 604800],
      ["day", 86400],
      ["hour", 3600],
      ["minute", 60],
    ];
    for (const [unit, size] of units) {
      if (Math.abs(seconds) >= size) return relFormatter.format(Math.round(seconds / size), unit);
    }
    return "just now";
  }

  function absTime(iso) {
    const date = parseDate(iso);
    return date ? date.toLocaleString() : String(iso || "");
  }

  function fmtSeconds(value) {
    return value === undefined || value === null ? "—" : `${Number(value).toFixed(1)}s`;
  }

  function fmtTokens(value) {
    if (!value) return "—";
    return value >= 1000 ? `${(value / 1000).toFixed(value >= 100000 ? 0 : 1)}k` : String(value);
  }

  function pctOf(part, whole) {
    return whole ? Math.round((part / whole) * 100) : 0;
  }

  // ------------------------------------------------------------------ API

  async function fetchJson(url, options) {
    const resp = await fetch(url, options);
    let payload = null;
    try {
      payload = await resp.json();
    } catch (_err) {
      // A proxy error page or a bare 500 is not JSON; report the status instead.
    }
    if (!resp.ok) {
      throw new Error((payload && payload.error) || `Request failed (${resp.status})`);
    }
    if (payload === null) throw new Error("Server returned a non-JSON response");
    return payload;
  }

  // The server refuses state-changing requests that are not application/json,
  // which is what stops a cross-site page from forging them.
  function postJson(url, body) {
    return fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  async function ensureSchema() {
    if (!state.launcherSchema) state.launcherSchema = await fetchJson("/api/schema");
    return state.launcherSchema;
  }

  // ----------------------------------------------------------- Components

  const STATUS_ICONS = {
    pass: "check",
    completed: "check",
    partial: "minus",
    interrupted: "pause",
    fail: "x",
    failed: "x",
    running: "activity",
  };

  // Status always pairs a color with an icon and a word, never color alone.
  function chip(status, label) {
    const key = String(status || "").toLowerCase();
    const known = STATUS_ICONS[key];
    return h(
      "span",
      { class: `chip ${known ? `chip-${key}` : "chip-plain"}` },
      known ? icon(known) : null,
      label ?? (status || "unknown")
    );
  }

  function ring(value, opts = {}) {
    const hasValue = value !== null && value !== undefined && !Number.isNaN(Number(value));
    const pct = hasValue ? Math.max(0, Math.min(100, Number(value))) : 0;
    const tier = opts.tier ?? (hasValue ? tierFor(pct) : "");
    const el = h(
      "div",
      {
        class: `ring ${tier} ${hasValue ? "" : "ring-empty"} ${opts.cls || ""}`,
        role: "img",
        "aria-label": opts.title || (hasValue ? `Score ${Math.round(pct)} of 100` : "No score"),
        style: { "--value": opts.animate ? 0 : pct },
      },
      h("span", { class: "ring-label", text: opts.label ?? (hasValue ? Math.round(pct) : "—") }),
      opts.sub ? h("span", { class: "ring-sub", text: opts.sub }) : null
    );
    if (opts.animate) {
      // Two frames: the first paints 0, the second lets the transition run.
      requestAnimationFrame(() => requestAnimationFrame(() => el.style.setProperty("--value", pct)));
    }
    return el;
  }

  function bar(pct, cls = "") {
    return h("div", { class: `bar ${cls}`, role: "presentation" }, h("span", { style: { "--value": pct } }));
  }

  function ppfBar(p, q, f) {
    return h(
      "div",
      { class: "ppf", role: "presentation" },
      h("span", { class: "p", style: { "--value": p } }),
      h("span", { class: "q", style: { "--value": q } }),
      h("span", { class: "f", style: { "--value": f } })
    );
  }

  function emptyState(iconName, title, text, ...actions) {
    return h(
      "div",
      { class: "empty" },
      h("div", { class: "empty-icon" }, icon(iconName)),
      h("h2", { text: title }),
      text ? h("p", { text }) : null,
      actions.length ? h("div", { class: "quick-links" }, actions) : null
    );
  }

  function stat(label, value, opts = {}) {
    return h(
      "div",
      { class: `card stat ${opts.cls || ""}` },
      h("div", { class: "stat-label" }, opts.icon ? icon(opts.icon) : null, label),
      h("div", { class: "stat-value" }, value, opts.small ? h("small", { text: ` ${opts.small}` }) : null),
      opts.extra || null
    );
  }

  function banner(kind, iconName, ...content) {
    return h("div", { class: `banner banner-${kind}`, role: kind === "danger" ? "alert" : "status" }, icon(iconName), h("div", {}, content));
  }

  function segmented(options, current, onPick, label) {
    return h(
      "div",
      { class: "segmented", role: "group", "aria-label": label },
      options.map(([id, text, count]) =>
        h(
          "button",
          { type: "button", "aria-pressed": String(current === id), onclick: () => onPick(id) },
          text,
          count !== undefined ? h("span", { class: "count", text: count }) : null
        )
      )
    );
  }

  function tile(status, title, opts = {}) {
    return h("button", {
      type: "button",
      class: `tile tile-${status}`,
      title,
      "aria-label": title,
      "aria-pressed": opts.pressed ? "true" : null,
      onclick: opts.onclick,
      tabindex: opts.onclick ? null : "-1",
    });
  }

  function legend() {
    return h(
      "div",
      { class: "legend" },
      [
        ["pass", "Pass"],
        ["partial", "Partial"],
        ["fail", "Fail"],
        ["held", "Held out"],
      ].map(([cls, text]) => h("span", {}, h("i", { class: `tile-${cls}` }), text))
    );
  }

  function modelBlock(model, cls) {
    const { org, name } = splitModel(model);
    return h(
      "div",
      { class: cls, title: model },
      org ? h("span", { class: cls === "run-model" ? "run-org" : "org", text: org }) : null,
      h("span", { class: cls === "run-model" ? "run-name" : "name", text: name })
    );
  }

  function preserveScroll(container, render) {
    const top = container.scrollTop;
    render();
    container.scrollTop = top;
  }

  // ----------------------------------------------------------------- Tabs

  const VIEWS = ["history", "live", "leaderboard", "compare"];

  function switchView(viewName) {
    state.activeView = viewName;
    for (const tab of document.querySelectorAll(".tab")) {
      const active = tab.dataset.view === viewName;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const view of VIEWS) {
      const section = byId(`view-${view}`);
      if (section) section.hidden = view !== viewName;
    }
    if (viewName === "leaderboard") loadLeaderboardView();
    else if (viewName === "compare") loadCompareView();
    else if (viewName === "live") loadLiveView();
  }

  function onTabKeydown(ev) {
    const tabs = [...document.querySelectorAll(".tab")];
    const index = tabs.indexOf(ev.currentTarget);
    let next = null;
    if (ev.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
    else if (ev.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
    else if (ev.key === "Home") next = tabs[0];
    else if (ev.key === "End") next = tabs[tabs.length - 1];
    if (!next) return;
    ev.preventDefault();
    next.focus();
    switchView(next.dataset.view);
  }

  // -------------------------------------------------------------- History

  async function loadRunsList() {
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
      renderRunsList();
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
  function scheduleRunsListLoad() {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(loadRunsList, 250);
  }

  function runCard(run) {
    const selected = run.run_id === state.selectedRunId;
    const score = run.final_score;
    return h(
      "button",
      {
        type: "button",
        class: "run-card run-item",
        dataset: { runId: run.run_id },
        "aria-current": selected ? "true" : null,
        onclick: () => selectRun(run.run_id),
      },
      modelBlock(run.model, "run-model"),
      ring(score, {
        label: score === null || score === undefined ? null : Math.round(score),
        title: score === null || score === undefined ? `No score, ${run.status}` : `Score ${score} of 100`,
      }),
      h(
        "div",
        { class: "run-meta" },
        chip(run.status),
        h("span", { text: run.backend || "unknown" }),
        h("span", { title: absTime(run.created_at), text: relTime(run.created_at) }),
        run.label ? h("span", { class: "chip chip-accent" }, icon("tag"), run.label) : null,
        run.safety_warnings_count > 0
          ? h("span", { class: "chip chip-fail", title: "Safety warnings" }, icon("shield"), run.safety_warnings_count)
          : null
      )
    );
  }

  function renderRunsList() {
    const container = byId("runs-list-container");
    if (!container) return;
    const count = byId("runs-count");
    if (count) count.textContent = state.runs.length ? String(state.runs.length) : "";
    if (!state.runs.length) {
      setChildren(container,
        emptyState("inbox", "No matching runs", "Run a benchmark from the CLI, or start one from Launch & Live.")
      );
      return;
    }
    setChildren(container, ...state.runs.map(runCard));
  }

  async function selectRun(runId) {
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
              ? [" ", h("span", { class: `chip ${r.decision_audit.disagreement ? "chip-partial" : "chip-info"}`, text: `audit: ${r.decision_audit.choice || r.decision_audit.status}` })]
              : null
          ),
          h("td", {}, h("span", { class: "cat-letter", title: r.category_label || r.category, text: r.category || "?" })),
          h("td", { class: "nowrap" }, chip(r.status), r.safety_violation ? [" ", h("span", { class: "chip chip-fail", title: r.safety_violation }, icon("shield"))] : null),
          h("td", { class: "num", text: `${r.points ?? 0}/2` }),
          h("td", { class: "num nowrap" }, fmtSeconds(r.duration_seconds), r.ttft_ms ? h("span", { class: "muted", text: ` · ${Math.round(r.ttft_ms)}ms` }) : null),
          h("td", { class: "cell-summary", text: r.summary || "" })
        )
      );
      if (open) rows.push(scenarioDetail(r, 7));
    }
    return rows;
  }

  function median(values) {
    const sorted = values.filter((v) => typeof v === "number" && !Number.isNaN(v)).sort((a, b) => a - b);
    if (!sorted.length) return null;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
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

  function renderInspector(opts = {}) {
    const pane = byId("run-inspector");
    if (!pane) return;
    const run = state.selectedRun;
    if (!run) {
      setChildren(pane,
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

    const actions = [
      h(
        "button",
        {
          type: "button",
          class: "btn",
          id: "inspect-compare-btn",
          onclick: () => {
            state.compareLeftId = run.run_id;
            switchView("compare");
          },
        },
        icon("swap"),
        "Compare"
      ),
      run.status === "running"
        ? h("button", { type: "button", class: "btn btn-primary", id: "inspect-watch-live-btn", onclick: () => openLiveRun(run.run_id) }, icon("eye"), "Watch live")
        : null,
      run.status === "interrupted"
        ? h("button", { type: "button", class: "btn btn-primary", id: "inspect-resume-btn", onclick: () => resumeInterruptedRun(run.run_id) }, icon("play"), "Resume")
        : null,
    ];

    const hero = h(
      "div",
      { class: "card hero" },
      ring(run.final_score, { cls: "ring-lg", animate: opts.animate, sub: run.final_score !== null && run.final_score !== undefined ? "of 100" : null }),
      h(
        "div",
        { class: "hero-title" },
        org ? h("span", { class: "hero-org", text: org }) : null,
        h("h1", { class: "hero-name", text: name, title: run.model }),
        h(
          "div",
          { class: "hero-meta" },
          chip(run.status),
          run.rating ? h("span", { class: "meta-item", text: run.rating }) : null,
          h("span", { class: "meta-item mono", title: "Run ID", text: run.run_id }),
          h("span", { class: "meta-item" }, icon("server"), run.backend || "unknown"),
          meta.engine_name ? h("span", { class: "meta-item" }, icon("cpu"), `${meta.engine_name} ${meta.engine_version || ""}`.trim()) : null,
          meta.quantization ? h("span", { class: "meta-item mono", text: meta.quantization }) : null,
          h("span", { class: "meta-item", title: absTime(run.created_at) }, icon("calendar"), relTime(run.created_at)),
          run.label ? h("span", { class: "chip chip-accent" }, icon("tag"), run.label) : null,
          run.held_out_count ? h("span", { class: "chip chip-plain", text: `${run.held_out_count} held out` }) : null
        )
      ),
      h("div", { class: "hero-actions" }, actions)
    );

    const p = run.passes || 0;
    const q = run.partials || 0;
    const f = run.fails || 0;
    const stats = h(
      "div",
      { class: "stats" },
      stat("Pass / partial / fail", h("span", {}, h("span", { class: "delta-pos", text: p }), " / ", h("span", { text: q }), " / ", h("span", { class: "delta-neg", text: f })), {
        extra: ppfBar(p, q, f),
      }),
      stat("Points", run.total_points ?? 0, { small: `/ ${run.max_points ?? 0}`, extra: bar(pctOf(run.total_points || 0, run.max_points || 0), tierFor(pctOf(run.total_points || 0, run.max_points || 0))) }),
      stat("Deployability", run.deployability ?? "—", { icon: "zap", small: run.deployability !== null && run.deployability !== undefined ? "/ 100" : null }),
      stat("Responsiveness", run.responsiveness ?? "—", { icon: "clock", small: run.responsiveness !== null && run.responsiveness !== undefined ? "/ 100" : null }),
      scores.total_tokens ? stat("Tokens", fmtTokens(scores.total_tokens), { icon: "message" }) : null,
      warnings.length ? stat("Safety warnings", warnings.length, { cls: "stat-fail", icon: "shield" }) : null
    );

    const safety = warnings.length
      ? h("div", { class: "section-spacer" }, banner("danger", "shield", h("strong", { text: `Safety gate warnings (${warnings.length})` }), h("ul", {}, warnings.map((w) => h("li", { text: String(w) })))))
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
            segmented(filterOptions, state.scenarioFilter, (id) => {
              state.scenarioFilter = id;
              preserveScroll(pane, () => renderInspector());
            }, "Filter scenarios"),
            h("span", { class: "muted", text: "Select a row to open its multi-turn trace." })
          ),
          h(
            "div",
            { class: "table-wrap" },
            h(
              "table",
              { class: "table" },
              h("thead", {}, h("tr", {}, ["Scenario", "Title", "Cat", "Verdict", "Pts", "Latency", "Summary"].map((t, i) => h("th", { class: i === 4 || i === 5 ? "num" : null, scope: "col", text: t })))),
              h("tbody", {}, filtered.length ? scenarioRows(filtered) : h("tr", {}, h("td", { colspan: 7, class: "muted", text: "No scenarios match this filter." })))
            )
          ),
        ]
      : [h("div", { class: "section-spacer" }, banner("info", "info", "This run has no per-scenario results to show."))];

    setChildren(pane,
      hero,
      safety,
      stats,
      catCard || mapCard ? h("div", { class: "two-col" }, catCard, mapCard) : null,
      ...traces
    );
  }

  // ---------------------------------------------------------- Leaderboard

  let leaderboardDebounce = null;

  async function loadLeaderboardView() {
    const workspace = byId("leaderboard-workspace");
    if (!workspace) return;
    const params = new URLSearchParams({ limit: "200" });
    if (state.leaderboardCohort) params.set("cohort", state.leaderboardCohort);
    if (state.leaderboardQuery) params.set("q", state.leaderboardQuery);
    try {
      const [data, schema] = await Promise.all([
        fetchJson(`/api/leaderboard?${params.toString()}`),
        ensureSchema().catch(() => null),
      ]);
      renderLeaderboardView(data, schema);
    } catch (err) {
      setChildren(workspace, emptyState("alert", "Could not load the leaderboard", err.message));
    }
  }

  function leaderboardRow(row, categories) {
    const total = (row.passes || 0) + (row.partials || 0) + (row.fails || 0);
    const open = () => {
      switchView("history");
      selectRun(row.run_id);
    };
    return h(
      "tr",
      {
        class: "lb-row row-click",
        dataset: { runId: row.run_id },
        tabindex: "0",
        onclick: open,
        onkeydown: (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            open();
          }
        },
      },
      h("td", { class: "sticky-col sticky-rank" }, h("span", { class: `rank ${row.cohort_rank <= 3 ? `rank-${row.cohort_rank}` : ""}`, text: row.cohort_rank })),
      h("td", { class: "sticky-col sticky-model" }, modelBlock(row.model, "model-cell")),
      h(
        "td",
        {},
        h("div", { class: `score-cell ${tierFor(row.final_score)}` }, h("span", { text: row.final_score }), bar(row.final_score, tierFor(row.final_score))),
        row.rating ? h("div", { class: "rating", text: row.rating }) : null
      ),
      h(
        "td",
        {},
        h("div", { class: "mono", text: `${row.passes}/${row.partials}/${row.fails}` }),
        total ? ppfBar(row.passes || 0, row.partials || 0, row.fails || 0) : null
      ),
      categories.map((cat) => {
        const pct = row.cat_scores?.[cat];
        return h(
          "td",
          { class: "heat-col" },
          pct === undefined || pct === null
            ? h("span", { class: "heat heat-none", text: "—" })
            : h("span", { class: `heat ${heatClass(pct)}`, text: Math.round(pct) })
        );
      }),
      h("td", { class: "num", text: fmtTokens(row.total_tokens) }),
      h("td", { class: "num", text: row.num_runs })
    );
  }

  function renderLeaderboardView(data, schema) {
    const workspace = byId("leaderboard-workspace");
    if (!workspace) return;
    const rows = data.rows || [];
    const cohorts = data.cohorts || [];
    const categories = data.categories || [];
    const labels = Object.fromEntries((schema?.categories || []).map((c) => [c.id, c.label]));

    const head = h(
      "div",
      { class: "page-head" },
      h(
        "div",
        {},
        h("h1", { class: "page-title", text: "Leaderboard" }),
        h("p", { class: "page-sub", text: "Runs rank only against others with the same benchmark conditions. Each cohort below is one comparable set." })
      ),
      h(
        "div",
        { class: "topbar-actions" },
        h("a", { class: "btn", href: "/api/export?format=csv", download: "tool-eval-leaderboard.csv" }, icon("download"), "CSV"),
        h("a", { class: "btn", href: "/api/export?format=json", download: "tool-eval-leaderboard.json" }, icon("download"), "JSON")
      )
    );

    const toolbar = h(
      "div",
      { class: "toolbar" },
      h("label", { class: "visually-hidden", for: "lb-cohort-select", text: "Cohort" }),
      h(
        "select",
        {
          id: "lb-cohort-select",
          class: "select",
          value: state.leaderboardCohort,
          onchange: (ev) => {
            state.leaderboardCohort = ev.target.value;
            loadLeaderboardView();
          },
        },
        h("option", { value: "", text: `All cohorts (${cohorts.length})` }),
        cohorts.map((c) => h("option", { value: c.cohort_fingerprint, text: `${c.cohort_label} · ${c.count}` }))
      ),
      h(
        "div",
        { class: "input-with-icon grow" },
        icon("search"),
        h("label", { class: "visually-hidden", for: "lb-search-input", text: "Filter models" }),
        h("input", {
          type: "search",
          id: "lb-search-input",
          class: "input",
          placeholder: "Filter models",
          value: state.leaderboardQuery,
          oninput: (ev) => {
            state.leaderboardQuery = ev.target.value;
            clearTimeout(leaderboardDebounce);
            leaderboardDebounce = setTimeout(async () => {
              await loadLeaderboardView();
              const input = byId("lb-search-input");
              if (input) {
                input.focus();
                input.setSelectionRange(input.value.length, input.value.length);
              }
            }, 250);
          },
        })
      )
    );

    if (!rows.length) {
      setChildren(workspace,
        head,
        toolbar,
        emptyState("trophy", "No ranked runs yet", "Completed tool-call runs that finished every scenario appear here.")
      );
      return;
    }

    const groups = new Map();
    for (const row of rows) {
      if (!groups.has(row.cohort_fingerprint)) groups.set(row.cohort_fingerprint, []);
      groups.get(row.cohort_fingerprint).push(row);
    }

    const tables = [...groups.values()].map((group) =>
      h(
        "section",
        { class: "cohort", "aria-label": `Cohort ${group[0].cohort_label}` },
        h(
          "div",
          { class: "cohort-head" },
          icon("layers"),
          h("h2", { text: group[0].cohort_label }),
          h("span", { class: "chip chip-plain", text: `${group.length} model${group.length === 1 ? "" : "s"}` })
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
              h(
                "tr",
                {},
                h("th", { scope: "col", class: "sticky-col sticky-rank", text: "#" }),
                h("th", { scope: "col", class: "sticky-col sticky-model", text: "Model" }),
                h("th", { scope: "col", class: "col-score", text: "Score" }),
                h("th", { scope: "col", class: "col-ppf", text: "P/Pt/F" }),
                categories.map((cat) => h("th", { scope: "col", class: "heat-col", title: labels[cat] || cat }, h("abbr", { title: labels[cat] || cat, text: cat }))),
                h("th", { scope: "col", class: "num", text: "Tokens" }),
                h("th", { scope: "col", class: "num", text: "Runs" })
              )
            ),
            h("tbody", {}, group.map((row) => leaderboardRow(row, categories)))
          )
        )
      )
    );

    const catLegend = categories.length
      ? h("div", { class: "cat-legend" }, categories.map((cat) => h("span", {}, h("b", { text: cat }), ` ${labels[cat] || ""}`)))
      : null;

    setChildren(workspace, head, toolbar, ...tables, catLegend);
  }

  // -------------------------------------------------------------- Compare

  async function loadCompareView() {
    const workspace = byId("compare-workspace");
    if (!workspace) return;
    // Independent of the History filters, which would otherwise decide what
    // the pickers can offer.
    try {
      const data = await fetchJson("/api/runs?limit=200&run_type=tool_eval");
      state.compareRuns = data.runs || [];
    } catch (_err) {
      // Keep whatever list we had; the empty state below covers a first failure.
    }
    const toolRuns = state.compareRuns.filter((r) => r.scenario_count > 0);
    const completed = toolRuns.filter((r) => r.status === "completed");
    if (!state.compareLeftId && completed.length >= 2) state.compareLeftId = completed[1].run_id;
    else if (!state.compareLeftId && completed.length === 1) state.compareLeftId = completed[0].run_id;
    if (!state.compareRightId && completed.length >= 1) state.compareRightId = completed[0].run_id;

    if (state.compareLeftId && state.compareRightId) {
      try {
        state.lastComparison = await fetchJson(
          `/api/compare?left=${encodeURIComponent(state.compareLeftId)}&right=${encodeURIComponent(state.compareRightId)}`
        );
      } catch (err) {
        state.lastComparison = { error: err.message };
      }
    }
    renderCompareView(toolRuns);
  }

  function runPicker(id, selectedId, toolRuns, onPick) {
    return h(
      "select",
      { id, class: "select", value: selectedId, onchange: (ev) => onPick(ev.target.value) },
      toolRuns.map((r) => {
        const { name } = splitModel(r.model);
        const score = r.final_score ?? "—";
        return h("option", { value: r.run_id, text: `${name} · ${score} · ${relTime(r.created_at)} · ${r.run_id.slice(0, 15)}` });
      })
    );
  }

  function vsCard(side, run, selectId, selectedId, toolRuns, onPick) {
    return h(
      "div",
      { class: "card vs-card" },
      ring(run?.final_score),
      h(
        "div",
        { class: "hero-title" },
        h("span", { class: "eyebrow", text: side }),
        h("label", { class: "visually-hidden", for: selectId, text: side }),
        run ? h("strong", { text: splitModel(run.model).name, title: run.model }) : null,
        runPicker(selectId, selectedId, toolRuns, onPick)
      )
    );
  }

  function compareRows(deltas, cmp) {
    const rows = [];
    for (const d of deltas) {
      const sid = d.scenario_id;
      const open = state.expandedCompareScenarios.has(sid);
      const toggle = () => {
        if (open) state.expandedCompareScenarios.delete(sid);
        else state.expandedCompareScenarios.add(sid);
        const ws = byId("compare-workspace");
        const scroller = document.scrollingElement;
        const top = scroller.scrollTop;
        renderCompareView(state.compareRuns.filter((r) => r.scenario_count > 0));
        scroller.scrollTop = top;
        ws?.querySelector(`[data-scenario-id="${CSS.escape(sid)}"] .disclosure`)?.focus({ preventScroll: true });
      };
      const pDelta = d.points_delta;
      rows.push(
        h(
          "tr",
          {
            class: `cmp-scenario-row row-click ${open ? "row-open" : ""}`,
            dataset: { scenarioId: sid },
            onclick: (ev) => {
              if (!ev.target.closest("button")) toggle();
            },
          },
          h("td", {}, h("button", { type: "button", class: "disclosure", "aria-expanded": String(open), onclick: toggle }, icon("chevron"), h("span", { class: "mono muted", text: sid }))),
          h("td", { class: "cell-title", text: d.title }),
          h("td", { class: "nowrap" }, d.a ? [chip(d.a.status), h("span", { class: "muted mono", text: ` ${d.a.points}/2` })] : h("span", { class: "muted", text: "—" })),
          h("td", { class: "nowrap" }, d.b ? [chip(d.b.status), h("span", { class: "muted mono", text: ` ${d.b.points}/2` })] : h("span", { class: "muted", text: "—" })),
          h(
            "td",
            {},
            pDelta === null || pDelta === undefined
              ? h("span", { class: "chip chip-plain", text: d.outcome })
              : h("span", { class: `delta-chip ${pDelta > 0 ? "up" : pDelta < 0 ? "down" : ""}`, text: pDelta > 0 ? `+${pDelta}` : pDelta === 0 ? "=" : String(pDelta) })
          ),
          h("td", { class: "num", text: d.duration_delta === null || d.duration_delta === undefined ? "—" : `${d.duration_delta > 0 ? "+" : ""}${d.duration_delta}s` }),
          h("td", { class: "cell-summary", text: d.b?.summary || d.a?.summary || "" })
        )
      );
      if (open) {
        rows.push(
          h(
            "tr",
            { class: "detail-row" },
            h(
              "td",
              { colspan: 7 },
              h(
                "div",
                { class: "detail trace-pair" },
                h("div", {}, h("div", { class: "eyebrow", text: `A · ${splitModel(cmp.run_a.model).name}` }), h("pre", { class: "trace", tabindex: "0", text: d.a?.raw_log || "Not present in run A" })),
                h("div", {}, h("div", { class: "eyebrow", text: `B · ${splitModel(cmp.run_b.model).name}` }), h("pre", { class: "trace", tabindex: "0", text: d.b?.raw_log || "Not present in run B" }))
              )
            )
          )
        );
      }
    }
    return rows;
  }

  function renderCompareView(toolRuns) {
    const workspace = byId("compare-workspace");
    if (!workspace) return;
    const head = h(
      "div",
      { class: "page-head" },
      h(
        "div",
        {},
        h("h1", { class: "page-title", text: "Compare runs" }),
        h("p", { class: "page-sub", text: "Scenario-by-scenario verdicts, latency shifts, and McNemar's paired test for any two tool-call runs." })
      )
    );
    if (!toolRuns.length) {
      setChildren(workspace, head, emptyState("swap", "Nothing to compare yet", "Complete at least two tool-call runs to compare them."));
      return;
    }

    const cmp = state.lastComparison;
    const runA = cmp?.run_a;
    const runB = cmp?.run_b;
    const versus = h(
      "div",
      { class: "versus" },
      vsCard("Baseline A", runA, "cmp-left-select", state.compareLeftId, toolRuns, (id) => {
        state.compareLeftId = id;
        loadCompareView();
      }),
      h(
        "div",
        { class: "vs-badge" },
        h("span", { text: "VS" }),
        h(
          "button",
          {
            type: "button",
            id: "cmp-swap-btn",
            class: "btn btn-ghost",
            title: "Swap A and B",
            onclick: () => {
              [state.compareLeftId, state.compareRightId] = [state.compareRightId, state.compareLeftId];
              loadCompareView();
            },
          },
          icon("swap"),
          h("span", { class: "visually-hidden", text: "Swap A and B" })
        )
      ),
      vsCard("Candidate B", runB, "cmp-right-select", state.compareRightId, toolRuns, (id) => {
        state.compareRightId = id;
        loadCompareView();
      })
    );

    const body = [];
    if (cmp?.error) {
      body.push(h("div", { class: "section-spacer" }, banner("danger", "alert", cmp.error)));
    } else if (cmp?.summary) {
      const s = cmp.summary;
      const mc = cmp.mcnemar || {};
      if (cmp.config_mismatch) {
        body.push(
          h(
            "div",
            { class: "section-spacer" },
            banner("warn", "alert", h("strong", { text: "Configuration mismatch: " }), (cmp.config_diffs || []).join(", "), ". McNemar's result may reflect the setup rather than the models.")
          )
        );
      }
      const scoreDelta = s.score_delta;
      const verdict =
        mc.discordant_pairs === 0
          ? ["info", "info", "Identical pass/fail outcomes", "No discordant pairs, so there is nothing to test."]
          : mc.significant
            ? ["pass", "check", `Significant: ${mc.direction === "B" ? "candidate B" : "baseline A"} is better`, `p = ${mc.p_value} · χ² = ${mc.chi2} · ${mc.discordant_pairs} discordant pairs (A only ${mc.a_only_pass}, B only ${mc.b_only_pass})`]
            : ["plain", "minus", "Not significant", `p = ${mc.p_value} · χ² = ${mc.chi2} · ${mc.discordant_pairs} discordant pairs (A only ${mc.a_only_pass}, B only ${mc.b_only_pass})`];
      body.push(
        h(
          "div",
          { class: "stats" },
          stat("Score A → B", `${s.score_a ?? "?"} → ${s.score_b ?? "?"}`, {
            extra: scoreDelta === null || scoreDelta === undefined ? null : h("div", { class: scoreDelta >= 0 ? "delta-pos" : "delta-neg", text: `${scoreDelta >= 0 ? "+" : ""}${scoreDelta} points of score` }),
          }),
          stat("Points", `${s.total_points_a} → ${s.total_points_b}`, { small: `(${s.total_delta >= 0 ? "+" : ""}${s.total_delta})` }),
          stat("Improved", s.improved, { cls: "stat-pass", icon: "check" }),
          stat("Regressed", s.regressed, { cls: "stat-fail", icon: "x" }),
          stat("Unchanged", s.unchanged, { icon: "minus" })
        ),
        h(
          "div",
          { class: "card card-pad section-spacer" },
          h("div", { class: "eyebrow", text: "McNemar's paired test" }),
          h("div", { class: "verdict" }, h("span", { class: `chip chip-${verdict[0]}` }, icon(verdict[1]), verdict[2]), h("span", { class: "muted mono", text: verdict[3] }))
        )
      );

      const all = cmp.deltas || [];
      const filters = [
        ["all", "All", () => true],
        ["changed", "Changed", (d) => d.outcome !== "unchanged"],
        ["regressed", "Regressed", (d) => d.outcome === "regressed"],
        ["improved", "Improved", (d) => d.outcome === "improved"],
      ];
      const active = filters.find(([id]) => id === state.compareFilter) || filters[0];
      const deltas = all.filter(active[2]);
      body.push(
        h("h2", { class: "section-title" }, "Scenario deltas", h("span", { class: "count", text: `${deltas.length} of ${all.length}` })),
        h(
          "div",
          { class: "toolbar" },
          segmented(
            filters.map(([id, label, test]) => [id, label, all.filter(test).length]),
            state.compareFilter,
            (id) => {
              state.compareFilter = id;
              renderCompareView(toolRuns);
            },
            "Filter scenario deltas"
          ),
          h("span", { class: "muted", text: "Select a row to compare both traces side by side." })
        ),
        h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            { class: "table" },
            h("thead", {}, h("tr", {}, ["Scenario", "Title", "Baseline A", "Candidate B", "Δ pts", "Δ time", "Summary"].map((t, i) => h("th", { scope: "col", class: i === 5 ? "num" : null, text: t })))),
            h("tbody", {}, deltas.length ? compareRows(deltas, cmp) : h("tr", {}, h("td", { colspan: 7, class: "muted", text: "No scenarios match this filter." })))
          )
        )
      );
    }

    setChildren(workspace, head, versus, ...body);
  }

  // ----------------------------------------------------------------- Live

  // Builds the launcher form once. Later calls only refresh the observer pane,
  // so switching tabs or resuming never resets what the user typed.
  function loadLiveView() {
    if (!state.liveViewReady) {
      state.liveViewReady = (async () => {
        const workspace = byId("live-workspace");
        if (!workspace) return false;
        try {
          await ensureSchema();
        } catch (err) {
          setChildren(workspace, emptyState("alert", "Could not load the launcher", err.message));
          state.liveViewReady = null; // allow a retry on the next visit
          return false;
        }
        renderLiveWorkspace();
        return true;
      })();
      return state.liveViewReady;
    }
    return state.liveViewReady.then((ready) => {
      if (ready) renderLiveObserverPane();
      return ready;
    });
  }

  function field(label, control, hint) {
    return h("div", { class: "field" }, h("label", { class: "field-label", for: control.id, text: label }), control, hint ? h("span", { class: "field-hint", text: hint }) : null);
  }

  function setProbeStatus(kind, text) {
    const el = byId("probe-feedback");
    if (!el) return;
    el.className = `probe-status ${kind ? `is-${kind}` : ""}`;
    el.textContent = text;
  }

  async function probeServer() {
    const urlInput = byId("launch-base-url");
    setProbeStatus("busy", "Probing server…");
    try {
      const res = await postJson("/api/probe", {
        base_url: urlInput?.value.trim() || undefined,
        wire_format: byId("launch-wire-format")?.value || "auto",
        api_key: byId("launch-api-key")?.value.trim() || undefined,
      });
      if (!res.reachable) {
        setProbeStatus("error", res.error || "Unreachable");
        return;
      }
      state.probedModels = res.models || [];
      if (urlInput && !urlInput.value.trim() && res.resolved_base_url) urlInput.value = res.resolved_base_url;
      const backendSel = byId("launch-backend");
      if (backendSel && res.backend && res.backend !== "unknown") backendSel.value = res.backend;
      const modelInput = byId("launch-model");
      if (modelInput && !modelInput.value.trim() && state.probedModels.length) modelInput.value = state.probedModels[0].id;
      byId("probed-models-datalist")?.replaceChildren(
        ...state.probedModels.map((m) => h("option", { value: m.id, text: m.display_name || m.id }))
      );
      setProbeStatus("ok", `Connected to ${res.server_name || res.backend || "server"} · ${state.probedModels.length} model${state.probedModels.length === 1 ? "" : "s"}`);
    } catch (err) {
      setProbeStatus("error", err.message);
    }
  }

  function renderLiveWorkspace() {
    const workspace = byId("live-workspace");
    const schema = state.launcherSchema;
    if (!workspace || !schema) return;
    const defaults = schema.defaults || {};

    const input = (id, props = {}) => h("input", { id, class: "input", ...props });
    const select = (id, options, value) =>
      h("select", { id, class: "select", value }, options.map(([v, t]) => h("option", { value: v, text: t })));

    const endpoint = h(
      "div",
      { class: "card launcher-section" },
      h("div", { class: "eyebrow" }, icon("server"), "Endpoint"),
      h(
        "div",
        { class: "field" },
        h("label", { class: "field-label", for: "launch-base-url", text: "Base URL" }),
        h(
          "div",
          { class: "input-row" },
          input("launch-base-url", { class: "input mono", placeholder: "http://localhost:8000", value: defaults.base_url || "", autocomplete: "off" }),
          h("button", { type: "button", id: "probe-server-btn", class: "btn", onclick: probeServer }, icon("radio"), "Probe")
        ),
        h("div", { id: "probe-feedback", class: "probe-status", role: "status", text: "Leave empty to auto-discover a local server." })
      ),
      field(
        "Model",
        input("launch-model", { class: "input mono", list: "probed-models-datalist", placeholder: "e.g. Qwen/Qwen3-8B", value: defaults.model || "", required: true, autocomplete: "off" })
      ),
      h("datalist", { id: "probed-models-datalist" }, state.probedModels.map((m) => h("option", { value: m.id, text: m.display_name || m.id }))),
      h(
        "div",
        { class: "grid-2" },
        field("Backend", select("launch-backend", (schema.backends || []).map((b) => [b, b]), defaults.backend || "unknown")),
        field("Wire format", select("launch-wire-format", (schema.wire_formats || ["auto"]).map((w) => [w, w]), "auto"))
      ),
      field("API key", input("launch-api-key", { type: "password", class: "input mono", placeholder: "Uses TOOL_EVAL_API_KEY if blank", autocomplete: "off" }), "Sent to the endpoint only; never stored.")
    );

    const presetSelect = h(
      "select",
      {
        id: "launch-preset",
        class: "select",
        onchange: (ev) => {
          const box = byId("custom-selection-fields");
          if (box) box.hidden = ev.target.value !== "custom";
        },
      },
      (schema.presets || []).map((p) => h("option", { value: p.id, text: p.label }))
    );

    const suite = h(
      "div",
      { class: "card launcher-section" },
      h("div", { class: "eyebrow" }, icon("layers"), "Scenarios"),
      field("Suite", presetSelect),
      h(
        "div",
        { id: "custom-selection-fields", class: "field", hidden: true },
        h("span", { class: "field-label", text: "Categories" }),
        h(
          "div",
          { class: "checks", role: "group", "aria-label": "Categories" },
          (schema.categories || []).map((cat) =>
            h(
              "label",
              { class: "check-chip", title: cat.label },
              h("input", { type: "checkbox", class: "launch-cat-cb", value: cat.id }),
              h("span", {}, h("b", { class: "mono", text: cat.id }), h("span", { class: "count", text: cat.count }))
            )
          )
        ),
        field("Scenario IDs", input("launch-scenarios", { class: "input mono", placeholder: "TC-01, TC-12, TC-46" }), "Optional; overrides the categories.")
      )
    );

    const advanced = h(
      "details",
      { class: "card launcher-section advanced" },
      h("summary", {}, icon("chevron"), icon("sliders"), "Advanced settings"),
      h(
        "div",
        { class: "grid-2" },
        field("Temperature", input("launch-temperature", { type: "number", class: "input mono", step: "0.05", min: "0", max: "2", value: defaults.temperature ?? 0 })),
        field("Concurrency", input("launch-parallel", { type: "number", class: "input mono", step: "1", min: "1", max: "32", value: defaults.concurrency ?? 1 })),
        field("Max turns", input("launch-max-turns", { type: "number", class: "input mono", step: "1", min: "1", max: "25", value: defaults.max_turns ?? 8 })),
        field("Timeout (s)", input("launch-timeout", { type: "number", class: "input mono", step: "5", min: "5", max: "1800", value: defaults.timeout_seconds ?? 120 }))
      ),
      h("div", { class: "section-spacer" }, field("Run label", input("launch-label", { placeholder: "e.g. vllm-0.11-fp8-baseline" })))
    );

    const form = h(
      "form",
      { id: "launch-run-form", onsubmit: submitLaunch, novalidate: true },
      endpoint,
      suite,
      advanced,
      h("div", { id: "launch-error-msg", class: "banner banner-danger", role: "alert", hidden: true }),
      h("button", { type: "submit", id: "launch-submit-btn", class: "btn btn-primary btn-block" }, icon("play"), "Start benchmark run")
    );

    setChildren(workspace,
      h("aside", { class: "launcher", "aria-label": "Launch a benchmark run" }, h("h1", { class: "rail-title", text: "Launch a run" }), h("div", { class: "section-spacer" }, form)),
      h("section", { id: "live-observer-pane", class: "observer", "aria-label": "Live run progress", "aria-live": "polite" })
    );
    renderLiveObserverPane();
  }

  function showLaunchError(message) {
    const box = byId("launch-error-msg");
    if (!box) return;
    box.replaceChildren(icon("alert"), h("div", { text: message }));
    box.hidden = false;
  }

  async function submitLaunch(ev) {
    ev.preventDefault();
    const box = byId("launch-error-msg");
    if (box) box.hidden = true;
    const preset = byId("launch-preset")?.value || "short";
    const categoriesChecked = [...document.querySelectorAll(".launch-cat-cb:checked")].map((cb) => cb.value);
    const explicitScenarios = byId("launch-scenarios")?.value || "";
    const payload = {
      base_url: byId("launch-base-url")?.value || "",
      model: byId("launch-model")?.value || "",
      backend: byId("launch-backend")?.value || "unknown",
      wire_format: byId("launch-wire-format")?.value || "auto",
      api_key: byId("launch-api-key")?.value || undefined,
      preset,
      categories: preset === "custom" && categoriesChecked.length ? categoriesChecked : undefined,
      scenarios: preset === "custom" && explicitScenarios.trim() ? explicitScenarios : undefined,
      temperature: Number(byId("launch-temperature")?.value || 0),
      concurrency: Number(byId("launch-parallel")?.value || 1),
      max_turns: Number(byId("launch-max-turns")?.value || 8),
      timeout_seconds: Number(byId("launch-timeout")?.value || 120),
      label: byId("launch-label")?.value || undefined,
    };
    const button = byId("launch-submit-btn");
    if (button) button.disabled = true;
    try {
      const started = await postJson("/api/runs", payload);
      connectLiveStream(started.run_id, { model: started.model, total: started.total_scenarios });
      loadRunsList();
    } catch (err) {
      showLaunchError(err.message);
    } finally {
      if (button) button.disabled = false;
    }
  }

  function turnRow(t) {
    const calls = (t.tool_calls || []).map((c) => `${c.name}(${JSON.stringify(c.arguments || {})})`).join(", ");
    const text = calls || t.content || "[completed]";
    return h(
      "li",
      { class: "turn" },
      icon(calls ? "wrench" : "message"),
      h("span", { class: "mono muted", text: `T${t.turn}` }),
      h("span", { class: "turn-text", text: text.length > 400 ? `${text.slice(0, 400)}…` : text }),
      h("span", { class: "mono muted", text: t.latency_ms !== undefined && t.latency_ms !== null ? `${Math.round(t.latency_ms)}ms` : "" })
    );
  }

  function renderLiveObserverPane() {
    const pane = byId("live-observer-pane");
    if (!pane) return;

    if (!state.liveRunId) {
      const watchable = state.runs.filter((r) => r.status === "running" || r.status === "interrupted");
      setChildren(pane,
        emptyState(
          "activity",
          "Live run observer",
          "Start a run on the left, or follow one that is already going, including runs started from the CLI.",
          ...watchable.map((r) =>
            h(
              "button",
              { type: "button", class: "btn watch-existing-btn", dataset: { runId: r.run_id }, onclick: () => connectLiveStream(r.run_id) },
              r.status === "running" ? icon("eye") : icon("pause"),
              `${splitModel(r.model).name} · ${r.status}`
            )
          )
        )
      );
      return;
    }

    const results = state.liveResults;
    const done = results.length;
    const total = state.liveTotal || done || 1;
    const pct = Math.min(100, Math.round((done / Math.max(total, 1)) * 100));
    const passes = results.filter((r) => r.status === "pass").length;
    const partials = results.filter((r) => r.status === "partial").length;
    const fails = results.filter((r) => r.status === "fail").length;
    const running = state.liveRunStatus === "running";
    const { org, name } = splitModel(state.liveModel || state.liveRunId);

    const actions = [
      running && state.liveManaged
        ? h("button", { type: "button", id: "live-stop-btn", class: "btn btn-danger", onclick: stopLiveRun }, icon("stop"), "Stop")
        : null,
      state.liveRunStatus === "interrupted"
        ? h("button", { type: "button", id: "live-resume-btn", class: "btn btn-primary", onclick: () => resumeInterruptedRun(state.liveRunId) }, icon("play"), "Resume")
        : null,
      state.liveRunStatus === "completed"
        ? h(
            "button",
            {
              type: "button",
              id: "live-inspect-btn",
              class: "btn btn-primary",
              onclick: () => {
                const rid = state.liveRunId;
                switchView("history");
                selectRun(rid);
              },
            },
            icon("arrowRight"),
            "Open in inspector"
          )
        : null,
    ];

    const hero = h(
      "div",
      { class: "card live-hero" },
      ring(pct, { tier: "tier-info", label: `${pct}%`, title: `${done} of ${total} scenarios done` }),
      h(
        "div",
        { class: "hero-title" },
        org ? h("span", { class: "hero-org", text: org }) : null,
        h("h2", { class: "hero-name", text: name }),
        h(
          "div",
          { class: "hero-meta" },
          running ? h("span", { class: "live-dot", "aria-hidden": "true" }) : null,
          chip(state.liveRunStatus),
          h("span", { class: "meta-item mono", text: state.liveRunId }),
          h("span", { class: "meta-item progress-text", dataset: { progress: "" }, text: `${done}/${total} (${pct}%)` }),
          state.liveManaged ? null : h("span", { class: "chip chip-plain", text: "started by another process" })
        ),
        h("progress", { class: "visually-hidden", value: done, max: total, "aria-label": "Scenarios completed" }),
        bar(pct, `bar-live ${running ? "is-running" : ""}`)
      ),
      h("div", { class: "hero-actions" }, actions)
    );

    const active = state.liveActiveScenario;
    const now = active
      ? h(
          "div",
          { class: "card now-running" },
          h(
            "div",
            { class: "now-head" },
            h("span", { class: "eyebrow" }, "Now running"),
            h("span", { class: "mono muted", text: active.scenario_id }),
            h("strong", { text: active.title }),
            h("span", { class: "chip chip-info", text: `Category ${active.category}` }),
            h("span", { class: "muted", text: `scenario ${done + 1} of ${total}` })
          ),
          state.liveTurns.length
            ? h("ol", { class: "turns" }, state.liveTurns.map(turnRow))
            : h("p", { class: "muted", text: "Waiting for the first turn…" })
        )
      : null;

    const pending = Math.max(0, total - done - (active ? 1 : 0));
    const map = h(
      "div",
      { class: "card card-pad section-spacer" },
      h("div", { class: "eyebrow", text: "Scenario map" }),
      h(
        "div",
        { class: "tile-grid", role: "group", "aria-label": "Scenario outcomes so far" },
        results.map((r) => tile(r.held_out ? "held" : r.status, `${r.scenario_id} · ${r.title || ""} · ${r.status}`)),
        active ? tile("active", `${active.scenario_id} · running`) : null,
        Array.from({ length: pending }, () => tile("pending", "Not started"))
      ),
      legend()
    );

    const recent = results.slice().reverse();
    const table = h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: "table" },
        h("thead", {}, h("tr", {}, ["Scenario", "Title", "Verdict", "Pts", "Duration", "Summary"].map((t, i) => h("th", { scope: "col", class: i === 3 || i === 4 ? "num" : null, text: t })))),
        h(
          "tbody",
          {},
          recent.length
            ? recent.map((r) =>
                h(
                  "tr",
                  {},
                  h("td", { class: "mono muted", text: r.scenario_id }),
                  h("td", {}, h("span", { class: "cell-title", text: r.title || r.scenario_id }), r.preserved ? [" ", h("span", { class: "chip chip-plain", text: "kept" })] : null),
                  h("td", {}, chip(r.status)),
                  h("td", { class: "num", text: `${r.points ?? 0}/2` }),
                  h("td", { class: "num", text: fmtSeconds(r.duration_seconds) }),
                  h("td", { class: "cell-summary", text: r.summary || "" })
                )
              )
            : h("tr", {}, h("td", { colspan: 6, class: "muted", text: "Waiting for the first scenario to finish…" }))
        )
      )
    );

    setChildren(pane,
      hero,
      state.liveNote ? h("div", { class: "section-spacer" }, banner("warn", "info", state.liveNote)) : null,
      h(
        "div",
        { class: "stats" },
        stat("Passed", passes, { cls: "stat-pass", icon: "check" }),
        stat("Partial", partials, { cls: "stat-partial", icon: "minus" }),
        stat("Failed", fails, { cls: "stat-fail", icon: "x" }),
        stat("Remaining", Math.max(0, total - done), { icon: "clock" })
      ),
      now,
      map,
      h("h2", { class: "section-title" }, "Finished scenarios", h("span", { class: "count", text: "newest first" })),
      table
    );
  }

  async function stopLiveRun() {
    if (!state.liveRunId) return;
    try {
      await postJson(`/api/runs/${encodeURIComponent(state.liveRunId)}/stop`);
    } catch (err) {
      state.liveNote = `Could not stop the run: ${err.message}`;
      renderLiveObserverPane();
      return;
    }
    state.liveRunStatus = "interrupted";
    state.liveActiveScenario = null;
    renderLiveObserverPane();
    loadRunsList();
  }

  function setLiveBadge(on) {
    const badge = byId("live-active-badge");
    if (badge) badge.hidden = !on;
  }

  function connectLiveStream(runId, initialMeta) {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    state.liveRunId = runId;
    state.liveRunStatus = "running";
    state.liveModel = initialMeta?.model || runId;
    state.liveTotal = initialMeta?.total || 0;
    state.liveActiveScenario = null;
    state.liveTurns = [];
    state.liveResults = [];
    state.liveManaged = true;
    state.liveNote = null;
    setLiveBadge(true);
    renderLiveObserverPane();

    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    state.eventSource = es;

    const finish = (status, note) => {
      if (status) state.liveRunStatus = status;
      state.liveActiveScenario = null;
      state.liveNote = note || null;
      setLiveBadge(false);
      es.close();
      if (state.eventSource === es) state.eventSource = null;
      renderLiveObserverPane();
      loadRunsList();
    };

    es.addEventListener("run_started", (ev) => {
      const data = JSON.parse(ev.data);
      state.liveModel = data.model || state.liveModel;
      state.liveTotal = data.total || state.liveTotal;
      state.liveRunStatus = data.status || "running";
      state.liveManaged = data.managed !== false;
      state.liveNote = null;
      renderLiveObserverPane();
    });

    es.addEventListener("scenario_start", (ev) => {
      const data = JSON.parse(ev.data);
      state.liveActiveScenario = data;
      state.liveTurns = [];
      state.liveTotal = data.total || state.liveTotal;
      renderLiveObserverPane();
    });

    es.addEventListener("scenario_turn", (ev) => {
      state.liveTurns.push(JSON.parse(ev.data));
      renderLiveObserverPane();
    });

    es.addEventListener("scenario_result", (ev) => {
      const data = JSON.parse(ev.data);
      // Preserved results replayed for a resume do not end the active scenario.
      if (!data.preserved) {
        state.liveActiveScenario = null;
        state.liveTurns = [];
      }
      if (data.result && !state.liveResults.some((r) => r.scenario_id === data.result.scenario_id)) {
        state.liveResults.push(data.preserved ? { ...data.result, preserved: true } : data.result);
      }
      renderLiveObserverPane();
    });

    es.addEventListener("run_complete", () => finish("completed"));

    es.addEventListener("run_interrupted", (ev) => {
      const data = JSON.parse(ev.data);
      const reason = data.reason && data.reason !== "interrupted" ? data.reason : null;
      finish("interrupted", reason ? `Run stopped: ${reason}` : null);
    });

    // Named stream_error rather than "error", which EventSource reserves for
    // connection failures.
    es.addEventListener("stream_error", (ev) => {
      finish(null, JSON.parse(ev.data).error || "The live stream reported an error.");
    });

    es.addEventListener("stream_idle", (ev) => {
      finish(null, JSON.parse(ev.data).message || "The live stream stopped after a long idle period.");
    });

    // EventSource reconnects on its own after a dropped connection, and the
    // server replays the run's history, so only a permanent failure ends it.
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        finish(null, "Live stream closed. Reopen the run to keep watching.");
      } else {
        state.liveNote = "Connection lost, reconnecting…";
        renderLiveObserverPane();
      }
    };
    es.onopen = () => {
      if (state.liveNote === "Connection lost, reconnecting…") {
        state.liveNote = null;
        renderLiveObserverPane();
      }
    };
  }

  async function resumeInterruptedRun(runId) {
    switchView("live");
    // loadLiveView renders the form only once, so values typed earlier are
    // still there to read.
    await loadLiveView();
    const baseUrl = byId("launch-base-url")?.value.trim() || "";
    const apiKey = byId("launch-api-key")?.value.trim() || "";
    const wireFormat = byId("launch-wire-format")?.value || "auto";
    try {
      const res = await postJson(`/api/runs/${encodeURIComponent(runId)}/resume`, {
        base_url: baseUrl || undefined,
        api_key: apiKey || undefined,
        wire_format: wireFormat,
      });
      connectLiveStream(res.run_id, { model: res.model, total: res.total_scenarios });
    } catch (err) {
      showLaunchError(err.message);
    }
  }

  function openLiveRun(runId) {
    switchView("live");
    loadLiveView().then(() => connectLiveStream(runId));
  }

  // ----------------------------------------------------------------- Boot

  window.webWorkbench = { state, switchView, loadRunsList, selectRun, fetchJson };

  document.addEventListener("DOMContentLoaded", () => {
    for (const tab of document.querySelectorAll(".tab")) {
      tab.addEventListener("click", () => switchView(tab.dataset.view));
      tab.addEventListener("keydown", onTabKeydown);
    }
    byId("refresh-all-btn")?.addEventListener("click", () => {
      loadRunsList();
      if (state.activeView === "leaderboard") loadLeaderboardView();
      else if (state.activeView === "compare") loadCompareView();
    });
    byId("history-search")?.addEventListener("input", scheduleRunsListLoad);
    byId("history-status-filter")?.addEventListener("change", loadRunsList);
    byId("history-type-filter")?.addEventListener("change", loadRunsList);
    renderInspector();
    loadRunsList();
  });
})();
