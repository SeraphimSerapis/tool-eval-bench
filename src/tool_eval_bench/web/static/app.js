(() => {
  "use strict";

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
    probeStatus: null,
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

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function statusBadgeClass(status) {
    const s = String(status || "").toLowerCase();
    if (s === "pass" || s === "completed") return "badge-pass";
    if (s === "partial" || s === "interrupted") return "badge-partial";
    if (s === "fail" || s === "failed") return "badge-fail";
    if (s === "running") return "badge-info";
    return "badge-neutral";
  }

  function scoreBarClass(pct) {
    if (pct >= 75) return "is-pass";
    if (pct >= 45) return "is-partial";
    return "is-fail";
  }

  function formatDate(iso) {
    if (!iso) return "—";
    return String(iso).slice(0, 19).replace("T", " ");
  }

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

  function switchView(viewName) {
    state.activeView = viewName;
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      const active = btn.dataset.view === viewName;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".view-section").forEach((sec) => {
      const active = sec.id === `view-${viewName}`;
      sec.classList.toggle("is-active", active);
      if (active) {
        sec.removeAttribute("hidden");
      } else {
        sec.setAttribute("hidden", "");
      }
    });
    if (viewName === "leaderboard" && typeof window.loadLeaderboardView === "function") {
      window.loadLeaderboardView();
    } else if (viewName === "compare" && typeof window.loadCompareView === "function") {
      window.loadCompareView();
    } else if (viewName === "live" && typeof window.loadLiveView === "function") {
      window.loadLiveView();
    }
  }

  async function loadRunsList() {
    const search = document.getElementById("history-search")?.value || "";
    const status = document.getElementById("history-status-filter")?.value || "";
    const runType = document.getElementById("history-type-filter")?.value || "";
    const params = new URLSearchParams({ limit: "100" });
    if (search) params.set("q", search);
    if (status) params.set("status", status);
    if (runType) params.set("run_type", runType);

    const container = document.getElementById("runs-list-container");
    const seq = ++state.runsRequestSeq;
    try {
      const data = await fetchJson(`/api/runs?${params.toString()}`);
      if (seq !== state.runsRequestSeq) return;
      if (data.version) {
        const verEl = document.getElementById("app-version");
        if (verEl) verEl.textContent = `v${data.version}`;
      }
      state.runs = data.runs || [];
      renderRunsList();
      if (!state.selectedRunId && state.runs.length > 0) {
        selectRun(state.runs[0].run_id);
      }
    } catch (err) {
      if (seq !== state.runsRequestSeq) return;
      if (container) {
        container.innerHTML = `<div class="empty-state"><h2>Could not load runs</h2><p>${escapeHtml(err.message)}</p></div>`;
      }
    }
  }

  let searchDebounce = null;
  function scheduleRunsListLoad() {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(loadRunsList, 250);
  }

  function renderRunsList() {
    const container = document.getElementById("runs-list-container");
    if (!container) return;
    if (!state.runs.length) {
      container.innerHTML = `
        <div class="empty-state">
          <h2>No matching benchmark runs</h2>
          <p>Run a benchmark from the CLI or switch to Launch &amp; Live to start a new run.</p>
        </div>`;
      return;
    }

    container.innerHTML = state.runs
      .map((run) => {
        const selected = run.run_id === state.selectedRunId ? "is-selected" : "";
        const score =
          run.final_score !== null && run.final_score !== undefined
            ? `${run.final_score}/100`
            : run.status;
        const labelHtml = run.label
          ? `<span class="badge badge-neutral">${escapeHtml(run.label)}</span>`
          : "";
        const safetyHtml =
          run.safety_warnings_count > 0
            ? `<span class="badge badge-fail">safety:${run.safety_warnings_count}</span>`
            : "";
        return `
          <button type="button" class="run-item ${selected}" data-run-id="${escapeHtml(run.run_id)}">
            <div class="run-item-top">
              <span class="run-item-model">${escapeHtml(run.model)}</span>
              <span class="run-item-score">${escapeHtml(score)}</span>
            </div>
            <div class="run-item-meta">
              <span class="badge ${statusBadgeClass(run.status)}">${escapeHtml(run.status)}</span>
              <span class="mono">${escapeHtml(run.run_type)}</span>
              <span>${escapeHtml(run.backend || "unknown")}</span>
              <span>${escapeHtml(formatDate(run.created_at))}</span>
              ${labelHtml}
              ${safetyHtml}
            </div>
          </button>`;
      })
      .join("");

    container.querySelectorAll(".run-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const runId = btn.getAttribute("data-run-id");
        if (runId) selectRun(runId);
      });
    });
  }

  async function selectRun(runId) {
    state.selectedRunId = runId;
    state.expandedScenarios.clear();
    renderRunsList();
    const pane = document.getElementById("run-inspector");
    if (!pane) return;
    const seq = ++state.selectRunSeq;
    try {
      const run = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
      if (seq !== state.selectRunSeq) return;
      state.selectedRun = run;
      renderInspector();
    } catch (err) {
      if (seq !== state.selectRunSeq) return;
      pane.innerHTML = `<div class="empty-state"><h2>Could not load run ${escapeHtml(runId)}</h2><p>${escapeHtml(err.message)}</p></div>`;
    }
  }

  function renderInspector() {
    const pane = document.getElementById("run-inspector");
    const run = state.selectedRun;
    if (!pane || !run) return;

    const scores = run.scores || {};
    const meta = run.metadata || {};
    const warnings = scores.safety_warnings || [];
    const categories = scores.category_scores || [];
    const results = scores.scenario_results || [];

    const safetyHtml = warnings.length
      ? `<div class="safety-banner" role="alert">
          <h3>Safety gate warnings (${warnings.length})</h3>
          <ul>${warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>
        </div>`
      : "";

    const categoryRows = categories
      .map((cs) => {
        const pct = Number(cs.percent || 0);
        return `
          <tr>
            <td><span class="mono">${escapeHtml(cs.category)}</span> · ${escapeHtml(cs.label)}</td>
            <td class="num">${pct}%</td>
            <td>
              <div class="bar-track" aria-hidden="true">
                <div class="bar-fill ${scoreBarClass(pct)}" style="width: ${Math.min(100, Math.max(0, pct))}%"></div>
              </div>
            </td>
            <td class="num">${escapeHtml(cs.earned)}/${escapeHtml(cs.max_points)}</td>
          </tr>`;
      })
      .join("");

    const filteredResults = results.filter((r) => {
      if (state.scenarioFilter === "fail") return r.status === "fail";
      if (state.scenarioFilter === "partial") return r.status === "partial";
      if (state.scenarioFilter === "pass") return r.status === "pass";
      if (state.scenarioFilter === "safety") return Boolean(r.safety_critical || r.safety_violation);
      if (state.scenarioFilter === "audited") return Boolean(r.decision_audit);
      return true;
    });

    const scenarioRows = filteredResults
      .map((r) => {
        const sid = r.scenario_id;
        const expanded = state.expandedScenarios.has(sid);
        const dur = r.duration_seconds !== undefined ? `${Number(r.duration_seconds).toFixed(1)}s` : "—";
        const ttft = r.ttft_ms ? `${Math.round(r.ttft_ms)}ms` : "—";
        const audit = r.decision_audit;
        const auditBadge = audit
          ? `<span class="badge ${audit.disagreement ? "badge-partial" : "badge-info"}">audit:${escapeHtml(audit.choice || audit.status)}</span>`
          : "";
        const heldOutBadge = r.held_out
          ? `<span class="badge badge-neutral">held-out</span>`
          : "";

        let detailRow = "";
        if (expanded) {
          const callsList = (r.tool_calls_made || []).length
            ? r.tool_calls_made.map((c) => `<code>${escapeHtml(c)}</code>`).join(", ")
            : "None";
          detailRow = `
            <tr class="trace-row">
              <td colspan="7">
                <div class="trace-meta-grid">
                  <div><strong>Expected:</strong> ${escapeHtml(r.expected_behavior || "—")}</div>
                  <div><strong>Tool calls:</strong> ${callsList}</div>
                  <div><strong>Turns:</strong> <span class="mono">${escapeHtml(r.turn_count ?? 1)}</span> · <strong>Tokens:</strong> <span class="mono">${escapeHtml((r.prompt_tokens || 0) + (r.completion_tokens || 0))}</span></div>
                </div>
                <div class="trace-box">${escapeHtml(r.raw_log || "No raw trace available.")}</div>
              </td>
            </tr>`;
        }

        return `
          <tr class="is-clickable scenario-summary-row" data-scenario-id="${escapeHtml(sid)}">
            <td class="mono">${escapeHtml(sid)}</td>
            <td>${escapeHtml(r.title || sid)} ${heldOutBadge} ${auditBadge}</td>
            <td class="mono">${escapeHtml(r.category || "—")}</td>
            <td><span class="badge ${statusBadgeClass(r.status)}">${escapeHtml(r.status)}</span></td>
            <td class="num">${escapeHtml(r.points)}/2</td>
            <td class="num">${escapeHtml(dur)} <span class="mono" style="color:var(--text-muted)">(${escapeHtml(ttft)})</span></td>
            <td>${escapeHtml(r.summary || "")}</td>
          </tr>
          ${detailRow}`;
      })
      .join("");

    pane.innerHTML = `
      <div class="inspector-header">
        <div>
          <h2 class="inspector-title">${escapeHtml(run.model)}</h2>
          <div class="inspector-subtitle">
            <span class="mono">${escapeHtml(run.run_id)}</span>
            <span class="badge ${statusBadgeClass(run.status)}">${escapeHtml(run.status)}</span>
            <span>Backend: <strong>${escapeHtml(run.backend)}</strong></span>
            <span>Date: <span class="mono">${escapeHtml(formatDate(run.created_at))}</span></span>
            ${meta.engine_name ? `<span>Engine: <strong>${escapeHtml(meta.engine_name)} ${escapeHtml(meta.engine_version || "")}</strong></span>` : ""}
            ${meta.quantization ? `<span>Quant: <span class="mono">${escapeHtml(meta.quantization)}</span></span>` : ""}
            ${run.label ? `<span class="badge badge-neutral">${escapeHtml(run.label)}</span>` : ""}
          </div>
        </div>
        <div class="topbar-actions">
          <button type="button" class="btn" id="inspect-compare-btn">Compare Run</button>
          ${
            run.status === "running" && typeof window.openLiveRun === "function"
              ? `<button type="button" class="btn btn-primary" id="inspect-watch-live-btn">Watch Live</button>`
              : ""
          }
          ${
            run.status === "interrupted" && typeof window.resumeInterruptedRun === "function"
              ? `<button type="button" class="btn btn-primary" id="inspect-resume-btn">Resume Run</button>`
              : ""
          }
        </div>
      </div>

      ${safetyHtml}

      <div class="summary-strip">
        <div class="metric-cell">
          <span class="metric-label">Final score</span>
          <span class="metric-value">${escapeHtml(run.final_score ?? "—")}${run.final_score !== null && run.final_score !== undefined ? "/100" : ""}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Rating</span>
          <span class="metric-value" style="font-size:1rem">${escapeHtml(run.rating || "—")}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Deployability</span>
          <span class="metric-value">${escapeHtml(run.deployability ?? "—")}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Responsiveness</span>
          <span class="metric-value">${escapeHtml(run.responsiveness ?? "—")}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Pass / Partial / Fail</span>
          <span class="metric-value">${run.passes} / ${run.partials} / ${run.fails}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Points earned</span>
          <span class="metric-value">${escapeHtml(run.total_points ?? 0)}/${escapeHtml(run.max_points ?? 0)}</span>
        </div>
      </div>

      ${
        categories.length
          ? `<h3 class="section-heading">Category breakdown</h3>
            <div class="data-table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Score</th>
                    <th>Distribution</th>
                    <th>Points</th>
                  </tr>
                </thead>
                <tbody>${categoryRows}</tbody>
              </table>
            </div>`
          : ""
      }

      ${
        results.length
          ? `<h3 class="section-heading">Scenario traces (${filteredResults.length} of ${results.length})</h3>
            <div class="scenario-toolbar">
              <div class="pill-group" role="group" aria-label="Filter scenarios">
                ${["all", "fail", "partial", "pass", "safety", "audited"]
                  .map(
                    (f) =>
                      `<button type="button" class="filter-pill ${state.scenarioFilter === f ? "is-active" : ""}" data-filter="${f}">${f.toUpperCase()}</button>`
                  )
                  .join("")}
              </div>
              <span style="color:var(--text-muted);font-size:0.75rem">Click any scenario row to inspect its multi-turn conversation trace</span>
            </div>
            <div class="data-table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Scenario</th>
                    <th>Cat</th>
                    <th>Verdict</th>
                    <th>Pts</th>
                    <th>Latency</th>
                    <th>Summary</th>
                  </tr>
                </thead>
                <tbody>${scenarioRows}</tbody>
              </table>
            </div>`
          : ""
      }
    `;

    pane.querySelectorAll(".filter-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        state.scenarioFilter = pill.getAttribute("data-filter") || "all";
        renderInspector();
      });
    });

    pane.querySelectorAll(".scenario-summary-row").forEach((row) => {
      row.addEventListener("click", () => {
        const sid = row.getAttribute("data-scenario-id");
        if (!sid) return;
        if (state.expandedScenarios.has(sid)) {
          state.expandedScenarios.delete(sid);
        } else {
          state.expandedScenarios.add(sid);
        }
        renderInspector();
      });
    });

    const compareBtn = document.getElementById("inspect-compare-btn");
    if (compareBtn) {
      compareBtn.addEventListener("click", () => {
        state.compareLeftId = run.run_id;
        switchView("compare");
      });
    }
    const watchBtn = document.getElementById("inspect-watch-live-btn");
    if (watchBtn && typeof window.openLiveRun === "function") {
      watchBtn.addEventListener("click", () => window.openLiveRun(run.run_id));
    }
    const resumeBtn = document.getElementById("inspect-resume-btn");
    if (resumeBtn && typeof window.resumeInterruptedRun === "function") {
      resumeBtn.addEventListener("click", () => window.resumeInterruptedRun(run.run_id));
    }
  }

  function heatClass(pct) {
    if (pct >= 90) return "heat-90";
    if (pct >= 75) return "heat-75";
    if (pct >= 60) return "heat-60";
    return "heat-low";
  }

  async function loadLeaderboardView() {
    const workspace = document.getElementById("leaderboard-workspace");
    if (!workspace) return;
    const params = new URLSearchParams({ limit: "100" });
    if (state.leaderboardCohort) params.set("cohort", state.leaderboardCohort);
    if (state.leaderboardQuery) params.set("q", state.leaderboardQuery);

    try {
      const data = await fetchJson(`/api/leaderboard?${params.toString()}`);
      renderLeaderboardView(data);
    } catch (err) {
      workspace.innerHTML = `<div class="empty-state"><h2>Could not load leaderboard</h2><p>${escapeHtml(err.message)}</p></div>`;
    }
  }

  function renderLeaderboardView(data) {
    const workspace = document.getElementById("leaderboard-workspace");
    if (!workspace) return;
    const rows = data.rows || [];
    const cohorts = data.cohorts || [];
    const categories = data.categories || [];

    const cohortOptions = [
      `<option value="">All comparable cohorts (${cohorts.length})</option>`,
      ...cohorts.map(
        (c) =>
          `<option value="${escapeHtml(c.cohort_fingerprint)}" ${
            state.leaderboardCohort === c.cohort_fingerprint ? "selected" : ""
          }>${escapeHtml(c.cohort_label)} (${c.count})</option>`
      ),
    ].join("");

    const catHeaders = categories.map((cat) => `<th class="mono">${escapeHtml(cat)}</th>`).join("");
    const tableRows = rows
      .map((row) => {
        const catCells = categories
          .map((cat) => {
            const pct = row.cat_scores?.[cat];
            if (pct === undefined || pct === null) {
              return `<td class="mono" style="color:var(--text-muted)">—</td>`;
            }
            return `<td><span class="heat-cell ${heatClass(pct)}">${Math.round(pct)}</span></td>`;
          })
          .join("");
        const tokens = row.total_tokens ? `${Math.round(row.total_tokens / 1000)}K` : "—";
        return `
          <tr class="is-clickable lb-row" data-run-id="${escapeHtml(row.run_id)}">
            <td class="num">#${escapeHtml(row.cohort_rank)}</td>
            <td><strong>${escapeHtml(row.display_model || row.model)}</strong></td>
            <td class="mono" style="font-size:0.75rem;color:var(--text-secondary)">${escapeHtml(row.cohort_label)}</td>
            <td class="num"><strong>${escapeHtml(row.final_score)}</strong></td>
            <td>${escapeHtml(row.rating)}</td>
            <td class="num">${row.passes}/${row.partials}/${row.fails} <span style="color:var(--text-muted)">(${row.completion_count}/${row.scenario_count})</span></td>
            ${catCells}
            <td class="num">${escapeHtml(tokens)}</td>
            <td class="num">${escapeHtml(row.num_runs)}</td>
          </tr>`;
      })
      .join("");

    workspace.innerHTML = `
      <div class="inspector-header" style="margin-bottom:1rem">
        <div>
          <h1 class="inspector-title">Model leaderboard</h1>
          <div class="inspector-subtitle">
            <span>Models are grouped by deterministic configuration fingerprint so only comparable runs are ranked together.</span>
          </div>
        </div>
        <div class="topbar-actions">
          <a class="btn" href="/api/export?format=csv" download="tool-eval-leaderboard.csv">Export CSV</a>
          <a class="btn" href="/api/export?format=json" download="tool-eval-leaderboard.json">Export JSON</a>
        </div>
      </div>

      <div class="toolbar-strip">
        <div class="toolbar-controls">
          <label class="visually-hidden" for="lb-cohort-select">Cohort filter</label>
          <select id="lb-cohort-select" class="control-select" style="width:auto;min-width:260px">
            ${cohortOptions}
          </select>
          <label class="visually-hidden" for="lb-search-input">Search models</label>
          <input
            type="search"
            id="lb-search-input"
            class="control-input"
            style="width:240px"
            placeholder="Filter model..."
            value="${escapeHtml(state.leaderboardQuery)}"
          >
        </div>
        <span style="color:var(--text-muted);font-size:0.75rem">Click any row to open its full trace in the Run Inspector</span>
      </div>

      ${
        rows.length
          ? `<div class="data-table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Cohort</th>
                    <th>Score</th>
                    <th>Rating</th>
                    <th>P/Pt/F</th>
                    ${catHeaders}
                    <th>Tokens</th>
                    <th>Runs</th>
                  </tr>
                </thead>
                <tbody>${tableRows}</tbody>
              </table>
            </div>`
          : `<div class="empty-state"><h2>No eligible completed runs</h2><p>Only completed tool-call runs with 100% scenario completion appear on the leaderboard.</p></div>`
      }
    `;

    document.getElementById("lb-cohort-select")?.addEventListener("change", (ev) => {
      state.leaderboardCohort = ev.target.value;
      loadLeaderboardView();
    });
    document.getElementById("lb-search-input")?.addEventListener("change", (ev) => {
      state.leaderboardQuery = ev.target.value;
      loadLeaderboardView();
    });
    workspace.querySelectorAll(".lb-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        const runId = tr.getAttribute("data-run-id");
        if (runId) {
          switchView("history");
          selectRun(runId);
        }
      });
    });
  }

  async function loadCompareView() {
    const workspace = document.getElementById("compare-workspace");
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
    if (!state.compareLeftId && completed.length >= 2) {
      state.compareLeftId = completed[1].run_id;
    } else if (!state.compareLeftId && completed.length === 1) {
      state.compareLeftId = completed[0].run_id;
    }
    if (!state.compareRightId && completed.length >= 1) {
      state.compareRightId = completed[0].run_id;
    }

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

  function renderCompareView(toolRuns) {
    const workspace = document.getElementById("compare-workspace");
    if (!workspace) return;

    const buildOptions = (selectedId) =>
      toolRuns
        .map(
          (r) =>
            `<option value="${escapeHtml(r.run_id)}" ${r.run_id === selectedId ? "selected" : ""}>${escapeHtml(
              r.model
            )} — ${escapeHtml(r.final_score ?? "?")}/100 (${escapeHtml(r.run_id.slice(0, 22))})</option>`
        )
        .join("");

    const cmp = state.lastComparison;
    let bodyHtml = "";
    if (!toolRuns.length) {
      bodyHtml = `<div class="empty-state"><h2>No runs available for comparison</h2><p>Complete at least two tool-call benchmark runs to compare scenario deltas and statistical significance.</p></div>`;
    } else if (cmp && cmp.error) {
      bodyHtml = `<div class="empty-state"><h2>Comparison unavailable</h2><p>${escapeHtml(cmp.error)}</p></div>`;
    } else if (cmp && cmp.summary) {
      const s = cmp.summary;
      const mc = cmp.mcnemar || {};
      const mismatchHtml = cmp.config_mismatch
        ? `<div class="warning-banner" role="alert">
            <strong>Configuration mismatch:</strong> ${escapeHtml((cmp.config_diffs || []).join(", "))}. McNemar test results may reflect configuration differences rather than model capability.
          </div>`
        : "";
      const mcnemarText =
        mc.discordant_pairs === 0
          ? "Identical pass/fail outcomes (0 discordant pairs)"
          : `${mc.significant ? `Significant (${mc.direction} better)` : "Not significant"} · p=${mc.p_value} · χ²=${mc.chi2} (${mc.discordant_pairs} discordant pairs: A-only=${mc.a_only_pass}, B-only=${mc.b_only_pass})`;

      const deltas = (cmp.deltas || []).filter((d) => {
        if (state.compareFilter === "regressed") return d.outcome === "regressed";
        if (state.compareFilter === "improved") return d.outcome === "improved";
        if (state.compareFilter === "changed") return d.outcome !== "unchanged";
        return true;
      });

      const deltaRows = deltas
        .map((d) => {
          const sid = d.scenario_id;
          const expanded = state.expandedCompareScenarios.has(sid);
          const pDelta =
            d.points_delta === null || d.points_delta === undefined
              ? d.outcome
              : d.points_delta > 0
                ? `+${d.points_delta}`
                : d.points_delta === 0
                  ? "="
                  : `${d.points_delta}`;
          const badgeCls =
            d.outcome === "improved"
              ? "badge-pass"
              : d.outcome === "regressed"
                ? "badge-fail"
                : "badge-neutral";
          const durDelta =
            d.duration_delta !== null && d.duration_delta !== undefined
              ? `${d.duration_delta >= 0 ? "+" : ""}${d.duration_delta}s`
              : "—";

          const expandedHtml = expanded
            ? `<tr class="trace-row">
                <td colspan="7">
                  <div class="compare-trace-split">
                    <div>
                      <div style="margin-bottom:0.375rem;font-weight:600">Baseline A (${escapeHtml(cmp.run_a.model)})</div>
                      <div class="trace-box">${escapeHtml(d.a?.raw_log || "Not present in Run A")}</div>
                    </div>
                    <div>
                      <div style="margin-bottom:0.375rem;font-weight:600">Candidate B (${escapeHtml(cmp.run_b.model)})</div>
                      <div class="trace-box">${escapeHtml(d.b?.raw_log || "Not present in Run B")}</div>
                    </div>
                  </div>
                </td>
              </tr>`
            : "";

          return `
            <tr class="is-clickable cmp-scenario-row" data-scenario-id="${escapeHtml(sid)}">
              <td class="mono">${escapeHtml(sid)}</td>
              <td>${escapeHtml(d.title)}</td>
              <td class="num">${d.a ? `${escapeHtml(d.a.status)} (${d.a.points}/2)` : "—"}</td>
              <td class="num">${d.b ? `${escapeHtml(d.b.status)} (${d.b.points}/2)` : "—"}</td>
              <td><span class="badge ${badgeCls}">${escapeHtml(pDelta)}</span></td>
              <td class="num">${escapeHtml(durDelta)}</td>
              <td>${escapeHtml(d.b?.summary || d.a?.summary || "")}</td>
            </tr>
            ${expandedHtml}`;
        })
        .join("");

      bodyHtml = `
        ${mismatchHtml}
        <div class="summary-strip">
          <div class="metric-cell">
            <span class="metric-label">Score A → B</span>
            <span class="metric-value">${escapeHtml(s.score_a ?? "?")} → ${escapeHtml(s.score_b ?? "?")}</span>
          </div>
          <div class="metric-cell">
            <span class="metric-label">Points delta</span>
            <span class="metric-value">${s.total_points_a} → ${s.total_points_b} (${s.total_delta >= 0 ? `+${s.total_delta}` : s.total_delta})</span>
          </div>
          <div class="metric-cell">
            <span class="metric-label">Improved / Regressed / Same</span>
            <span class="metric-value">${s.improved} / ${s.regressed} / ${s.unchanged}</span>
          </div>
          <div class="metric-cell" style="grid-column: span 2">
            <span class="metric-label">McNemar's paired test</span>
            <span class="metric-value" style="font-size:0.875rem">${escapeHtml(mcnemarText)}</span>
          </div>
        </div>

        <div class="scenario-toolbar">
          <div class="pill-group" role="group" aria-label="Filter scenario diffs">
            ${["all", "changed", "regressed", "improved"]
              .map(
                (f) =>
                  `<button type="button" class="filter-pill cmp-filter-pill ${state.compareFilter === f ? "is-active" : ""}" data-filter="${f}">${f.toUpperCase()}</button>`
              )
              .join("")}
          </div>
          <span style="color:var(--text-muted);font-size:0.75rem">Click any row to compare side-by-side traces</span>
        </div>

        <div class="data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Scenario</th>
                <th>Baseline (A)</th>
                <th>Candidate (B)</th>
                <th>Delta</th>
                <th>Time Δ</th>
                <th>Candidate Summary</th>
              </tr>
            </thead>
            <tbody>${deltaRows}</tbody>
          </table>
        </div>
      `;
    }

    workspace.innerHTML = `
      <div class="inspector-header" style="margin-bottom:1rem">
        <div>
          <h1 class="inspector-title">Head-to-head run comparison</h1>
          <div class="inspector-subtitle">
            <span>Compare scenario verdicts, latency shifts, and paired McNemar statistical significance between any two runs.</span>
          </div>
        </div>
      </div>

      <div class="toolbar-strip">
        <div class="toolbar-controls">
          <label for="cmp-left-select" style="font-size:0.75rem;color:var(--text-secondary)">Baseline (A):</label>
          <select id="cmp-left-select" class="control-select" style="width:auto;min-width:260px">
            ${buildOptions(state.compareLeftId)}
          </select>
          <button type="button" id="cmp-swap-btn" class="btn">Swap A ↔ B</button>
          <label for="cmp-right-select" style="font-size:0.75rem;color:var(--text-secondary)">Candidate (B):</label>
          <select id="cmp-right-select" class="control-select" style="width:auto;min-width:260px">
            ${buildOptions(state.compareRightId)}
          </select>
        </div>
      </div>

      ${bodyHtml}
    `;

    document.getElementById("cmp-left-select")?.addEventListener("change", (ev) => {
      state.compareLeftId = ev.target.value;
      loadCompareView();
    });
    document.getElementById("cmp-right-select")?.addEventListener("change", (ev) => {
      state.compareRightId = ev.target.value;
      loadCompareView();
    });
    document.getElementById("cmp-swap-btn")?.addEventListener("click", () => {
      const tmp = state.compareLeftId;
      state.compareLeftId = state.compareRightId;
      state.compareRightId = tmp;
      loadCompareView();
    });
    workspace.querySelectorAll(".cmp-filter-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.compareFilter = btn.getAttribute("data-filter") || "all";
        renderCompareView(toolRuns);
      });
    });
    workspace.querySelectorAll(".cmp-scenario-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        const sid = tr.getAttribute("data-scenario-id");
        if (!sid) return;
        if (state.expandedCompareScenarios.has(sid)) {
          state.expandedCompareScenarios.delete(sid);
        } else {
          state.expandedCompareScenarios.add(sid);
        }
        renderCompareView(toolRuns);
      });
    });
  }

  window.loadLeaderboardView = loadLeaderboardView;
  window.loadCompareView = loadCompareView;

  // Builds the launcher form once. Later calls only refresh the observer pane,
  // so switching tabs or resuming never resets what the user typed.
  function loadLiveView() {
    if (!state.liveViewReady) {
      state.liveViewReady = (async () => {
        const workspace = document.getElementById("live-workspace");
        if (!workspace) return false;
        try {
          state.launcherSchema = await fetchJson("/api/schema");
        } catch (err) {
          workspace.innerHTML = `<div class="empty-state"><h2>Could not load launcher schema</h2><p>${escapeHtml(err.message)}</p></div>`;
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

  function renderLiveWorkspace() {
    const workspace = document.getElementById("live-workspace");
    if (!workspace || !state.launcherSchema) return;

    const schema = state.launcherSchema;
    const defaults = schema.defaults || {};
    const presets = schema.presets || [];
    const categories = schema.categories || [];
    const backends = schema.backends || [];

    const presetOptions = presets
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
      .join("");
    const backendOptions = backends
      .map(
        (b) =>
          `<option value="${escapeHtml(b)}" ${b === defaults.backend ? "selected" : ""}>${escapeHtml(b)}</option>`
      )
      .join("");
    const categoryCheckboxes = categories
      .map(
        (cat) => `
        <label style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.625rem;font-weight:400">
          <input type="checkbox" class="launch-cat-cb" value="${escapeHtml(cat.id)}">
          <span class="mono">${escapeHtml(cat.id)}</span> (${escapeHtml(cat.count)})
        </label>`
      )
      .join("");

    const modelOptions = state.probedModels.length
      ? state.probedModels
          .map(
            (m) =>
              `<option value="${escapeHtml(m.id)}">${escapeHtml(m.display_name || m.id)}</option>`
          )
          .join("")
      : "";

    workspace.innerHTML = `
      <aside class="launcher-pane" aria-label="Benchmark run launcher">
        <h1 class="pane-title">Launch benchmark run</h1>
        <form id="launch-run-form">
          <div class="form-field">
            <label for="launch-base-url">Inference endpoint Base URL</label>
            <div style="display:flex;gap:0.5rem">
              <input
                type="text"
                id="launch-base-url"
                class="control-input mono"
                placeholder="http://localhost:8000"
                value="${escapeHtml(defaults.base_url || "")}"
              >
              <button type="button" id="probe-server-btn" class="btn">Probe</button>
            </div>
            <div id="probe-feedback" style="margin-top:0.375rem;font-size:0.75rem;color:var(--text-secondary)"></div>
          </div>

          <div class="form-field">
            <label for="launch-model">Model name or ID</label>
            <input
              type="text"
              id="launch-model"
              class="control-input mono"
              list="probed-models-datalist"
              placeholder="e.g. Qwen/Qwen3-8B"
              value="${escapeHtml(defaults.model || "")}"
              required
            >
            <datalist id="probed-models-datalist">${modelOptions}</datalist>
          </div>

          <div class="form-row-2">
            <div class="form-field">
              <label for="launch-backend">Backend label</label>
              <select id="launch-backend" class="control-select">${backendOptions}</select>
            </div>
            <div class="form-field">
              <label for="launch-wire-format">Wire format</label>
              <select id="launch-wire-format" class="control-select">
                <option value="auto">auto</option>
                <option value="openai">openai</option>
                <option value="gemini">gemini</option>
                <option value="anthropic">anthropic</option>
              </select>
            </div>
          </div>

          <div class="form-field">
            <label for="launch-api-key">API key (optional, never stored)</label>
            <input
              type="password"
              id="launch-api-key"
              class="control-input mono"
              placeholder="Uses TOOL_EVAL_API_KEY if blank"
              autocomplete="off"
            >
          </div>

          <div class="form-field">
            <label for="launch-preset">Scenario suite</label>
            <select id="launch-preset" class="control-select">${presetOptions}</select>
          </div>

          <div id="custom-selection-fields" class="is-hidden">
            <div class="form-field">
              <label>Filter by categories</label>
              <div style="padding:0.5rem;border:1px solid var(--border-subtle);border-radius:6px;background:var(--bg-canvas)">
                ${categoryCheckboxes}
              </div>
            </div>
            <div class="form-field">
              <label for="launch-scenarios">Explicit scenario IDs (optional)</label>
              <input
                type="text"
                id="launch-scenarios"
                class="control-input mono"
                placeholder="TC-01, TC-12, TC-46"
              >
            </div>
          </div>

          <div class="form-row-2">
            <div class="form-field">
              <label for="launch-temperature">Temperature</label>
              <input type="number" id="launch-temperature" class="control-input mono" step="0.05" min="0" max="2" value="${escapeHtml(defaults.temperature ?? 0.0)}">
            </div>
            <div class="form-field">
              <label for="launch-parallel">Concurrency (--parallel)</label>
              <input type="number" id="launch-parallel" class="control-input mono" step="1" min="1" max="32" value="${escapeHtml(defaults.concurrency ?? 1)}">
            </div>
          </div>

          <div class="form-row-2">
            <div class="form-field">
              <label for="launch-max-turns">Max turns</label>
              <input type="number" id="launch-max-turns" class="control-input mono" step="1" min="1" max="25" value="${escapeHtml(defaults.max_turns ?? 8)}">
            </div>
            <div class="form-field">
              <label for="launch-timeout">Timeout (sec)</label>
              <input type="number" id="launch-timeout" class="control-input mono" step="5" min="5" max="1800" value="${escapeHtml(defaults.timeout_seconds ?? 120)}">
            </div>
          </div>

          <div class="form-field">
            <label for="launch-label">Run label (optional)</label>
            <input type="text" id="launch-label" class="control-input" placeholder="e.g. vllm-0.8.4-fp8-baseline">
          </div>

          <div id="launch-error-msg" class="safety-banner is-hidden" role="alert"></div>

          <button type="submit" id="launch-submit-btn" class="btn btn-primary" style="width:100%;padding:0.625rem">
            Start Benchmark Run
          </button>
        </form>
      </aside>

      <section id="live-observer-pane" class="observer-pane" aria-label="Live run progress">
      </section>
    `;

    renderLiveObserverPane();

    document.getElementById("launch-preset")?.addEventListener("change", (ev) => {
      const customBox = document.getElementById("custom-selection-fields");
      if (customBox) {
        customBox.classList.toggle("is-hidden", ev.target.value !== "custom");
      }
    });

    document.getElementById("probe-server-btn")?.addEventListener("click", async () => {
      const urlInput = document.getElementById("launch-base-url");
      const keyInput = document.getElementById("launch-api-key");
      const fmtInput = document.getElementById("launch-wire-format");
      const feedback = document.getElementById("probe-feedback");
      if (feedback) feedback.textContent = "Probing server...";

      const probeBody = {
        base_url: urlInput?.value.trim() || undefined,
        wire_format: fmtInput?.value || "auto",
        api_key: keyInput?.value.trim() || undefined,
      };

      try {
        const res = await postJson("/api/probe", probeBody);
        if (!res.reachable) {
          if (feedback) feedback.innerHTML = `<span style="color:var(--fail)">${escapeHtml(res.error || "Unreachable")}</span>`;
          return;
        }
        state.probedModels = res.models || [];
        if (urlInput && !urlInput.value.trim() && res.resolved_base_url) {
          urlInput.value = res.resolved_base_url;
        }
        const backendSel = document.getElementById("launch-backend");
        if (backendSel && res.backend && res.backend !== "unknown") {
          backendSel.value = res.backend;
        }
        const modelInput = document.getElementById("launch-model");
        if (modelInput && !modelInput.value.trim() && state.probedModels.length > 0) {
          modelInput.value = state.probedModels[0].id;
        }
        const datalist = document.getElementById("probed-models-datalist");
        if (datalist) {
          datalist.innerHTML = state.probedModels
            .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.display_name || m.id)}</option>`)
            .join("");
        }
        if (feedback) {
          feedback.innerHTML = `<span style="color:var(--pass)">Connected (${escapeHtml(res.backend || "server")}) · ${state.probedModels.length} model(s) found</span>`;
        }
      } catch (err) {
        if (feedback) feedback.innerHTML = `<span style="color:var(--fail)">${escapeHtml(err.message)}</span>`;
      }
    });

    document.getElementById("launch-run-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const errBox = document.getElementById("launch-error-msg");
      if (errBox) errBox.classList.add("is-hidden");

      const preset = document.getElementById("launch-preset")?.value || "short";
      const categoriesChecked = Array.from(document.querySelectorAll(".launch-cat-cb:checked")).map(
        (cb) => cb.value
      );
      const explicitScenarios = document.getElementById("launch-scenarios")?.value || "";

      const payload = {
        base_url: document.getElementById("launch-base-url")?.value || "",
        model: document.getElementById("launch-model")?.value || "",
        backend: document.getElementById("launch-backend")?.value || "unknown",
        wire_format: document.getElementById("launch-wire-format")?.value || "auto",
        api_key: document.getElementById("launch-api-key")?.value || undefined,
        preset,
        categories: preset === "custom" && categoriesChecked.length ? categoriesChecked : undefined,
        scenarios: preset === "custom" && explicitScenarios.trim() ? explicitScenarios : undefined,
        temperature: Number(document.getElementById("launch-temperature")?.value || 0),
        concurrency: Number(document.getElementById("launch-parallel")?.value || 1),
        max_turns: Number(document.getElementById("launch-max-turns")?.value || 8),
        timeout_seconds: Number(document.getElementById("launch-timeout")?.value || 120),
        label: document.getElementById("launch-label")?.value || undefined,
      };

      try {
        const started = await postJson("/api/runs", payload);
        connectLiveStream(started.run_id, {
          model: started.model,
          total: started.total_scenarios,
        });
        loadRunsList();
      } catch (err) {
        if (errBox) {
          errBox.textContent = err.message;
          errBox.classList.remove("is-hidden");
        }
      }
    });
  }

  function renderLiveObserverPane() {
    const pane = document.getElementById("live-observer-pane");
    if (!pane) return;

    if (!state.liveRunId) {
      const runningRuns = state.runs.filter((r) => r.status === "running" || r.status === "interrupted");
      const quickLinks = runningRuns.length
        ? `<div style="margin-top:1rem">
            <strong style="display:block;margin-bottom:0.5rem">Active or resumable runs in SQLite:</strong>
            ${runningRuns
              .map(
                (r) =>
                  `<button type="button" class="btn watch-existing-btn" data-run-id="${escapeHtml(r.run_id)}" style="margin-right:0.5rem;margin-bottom:0.5rem">
                    ${escapeHtml(r.model)} (${escapeHtml(r.status)} · ${escapeHtml(r.run_id.slice(0, 18))})
                  </button>`
              )
              .join("")}
          </div>`
        : "";

      pane.innerHTML = `
        <div class="empty-state">
          <h2>Live run observer</h2>
          <p>Start a benchmark run on the left, or select an in-progress CLI run to watch scenario execution, turn-by-turn tool calls, and live score updates in real time.</p>
          ${quickLinks}
        </div>
      `;
      pane.querySelectorAll(".watch-existing-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const rid = btn.getAttribute("data-run-id");
          if (rid) connectLiveStream(rid);
        });
      });
      return;
    }

    const done = state.liveResults.length;
    const total = state.liveTotal || done || 1;
    const pct = Math.min(100, Math.round((done / Math.max(total, 1)) * 100));
    const passes = state.liveResults.filter((r) => r.status === "pass").length;
    const partials = state.liveResults.filter((r) => r.status === "partial").length;
    const fails = state.liveResults.filter((r) => r.status === "fail").length;

    const activeHtml = state.liveActiveScenario
      ? `<div class="summary-strip" style="border-color:var(--accent-strong)">
          <div class="metric-cell" style="grid-column: 1 / -1">
            <span class="metric-label">Currently executing scenario (${done + 1} of ${total})</span>
            <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem">
              <span class="mono" style="font-size:1rem;font-weight:600">${escapeHtml(state.liveActiveScenario.scenario_id)} · ${escapeHtml(state.liveActiveScenario.title)}</span>
              <span class="badge badge-info">Category ${escapeHtml(state.liveActiveScenario.category)}</span>
            </div>
            ${
              state.liveTurns.length
                ? `<div class="trace-box" style="margin-top:0.5rem;max-height:180px">${state.liveTurns
                    .map((t) => {
                      const calls = (t.tool_calls || [])
                        .map((c) => `${c.name}(${JSON.stringify(c.arguments || {})})`)
                        .join(", ");
                      return `Turn ${t.turn}/${t.max_turns} (${t.latency_ms}ms): ${
                        calls ? `calls -> ${calls}` : t.content || "[completed]"
                      }`;
                    })
                    .map(escapeHtml)
                    .join("\n")}</div>`
                : `<div style="margin-top:0.375rem;font-size:0.75rem;color:var(--text-muted)">Waiting for turn 1 completion...</div>`
            }
          </div>
        </div>`
      : "";

    const rowsHtml = state.liveResults
      .slice()
      .reverse()
      .map((r) => {
        const dur = r.duration_seconds !== undefined ? `${Number(r.duration_seconds).toFixed(1)}s` : "—";
        return `
          <tr>
            <td class="mono">${escapeHtml(r.scenario_id)}</td>
            <td>${escapeHtml(r.title || r.scenario_id)}</td>
            <td><span class="badge ${statusBadgeClass(r.status)}">${escapeHtml(r.status)}</span></td>
            <td class="num">${escapeHtml(r.points)}/2</td>
            <td class="num">${escapeHtml(dur)}</td>
            <td>${escapeHtml(r.summary || "")}</td>
          </tr>`;
      })
      .join("");

    pane.innerHTML = `
      <div class="inspector-header">
        <div>
          <h2 class="inspector-title">${escapeHtml(state.liveModel || "Live benchmark run")}</h2>
          <div class="inspector-subtitle">
            <span class="mono">${escapeHtml(state.liveRunId)}</span>
            <span class="badge ${statusBadgeClass(state.liveRunStatus)}">${escapeHtml(state.liveRunStatus)}</span>
            <span>Progress: <strong class="mono">${done}/${total} (${pct}%)</strong></span>
            ${state.liveManaged ? "" : `<span class="badge badge-neutral">started by another process</span>`}
          </div>
          ${state.liveNote ? `<div class="warning-banner" role="status" style="margin-top:0.5rem">${escapeHtml(state.liveNote)}</div>` : ""}
        </div>
        <div class="topbar-actions">
          ${
            state.liveRunStatus === "running" && state.liveManaged
              ? `<button type="button" id="live-stop-btn" class="btn btn-danger">Stop (Save Checkpoints)</button>`
              : ""
          }
          ${
            state.liveRunStatus === "interrupted"
              ? `<button type="button" id="live-resume-btn" class="btn btn-primary">Resume From Checkpoints</button>`
              : ""
          }
          ${
            state.liveRunStatus === "completed"
              ? `<button type="button" id="live-inspect-btn" class="btn btn-primary">Open in Run Inspector</button>`
              : ""
          }
        </div>
      </div>

      <div style="margin: 1rem 0">
        <div class="bar-track" style="max-width:none;height:10px">
          <div class="bar-fill" style="width:${pct}%"></div>
        </div>
      </div>

      <div class="summary-strip">
        <div class="metric-cell">
          <span class="metric-label">Completed scenarios</span>
          <span class="metric-value">${done}/${total}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Passed</span>
          <span class="metric-value" style="color:var(--pass)">${passes}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Partial</span>
          <span class="metric-value" style="color:var(--partial)">${partials}</span>
        </div>
        <div class="metric-cell">
          <span class="metric-label">Failed</span>
          <span class="metric-value" style="color:var(--fail)">${fails}</span>
        </div>
      </div>

      ${activeHtml}

      <h3 class="section-heading">Completed scenarios (newest first)</h3>
      <div class="data-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Scenario</th>
              <th>Verdict</th>
              <th>Pts</th>
              <th>Duration</th>
              <th>Summary</th>
            </tr>
          </thead>
          <tbody>${rowsHtml || `<tr><td colspan="6" style="color:var(--text-muted)">Waiting for first scenario to complete...</td></tr>`}</tbody>
        </table>
      </div>
    `;

    document.getElementById("live-stop-btn")?.addEventListener("click", async () => {
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
    });

    document.getElementById("live-resume-btn")?.addEventListener("click", () => {
      if (state.liveRunId) resumeInterruptedRun(state.liveRunId);
    });

    document.getElementById("live-inspect-btn")?.addEventListener("click", () => {
      const rid = state.liveRunId;
      if (rid) {
        switchView("history");
        selectRun(rid);
      }
    });
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

    const badge = document.getElementById("live-active-badge");
    if (badge) badge.classList.remove("is-hidden");

    renderLiveObserverPane();

    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    state.eventSource = es;

    const finish = (status, note) => {
      if (status) state.liveRunStatus = status;
      state.liveActiveScenario = null;
      state.liveNote = note || null;
      if (badge) badge.classList.add("is-hidden");
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
      const data = JSON.parse(ev.data);
      state.liveTurns.push(data);
      renderLiveObserverPane();
    });

    es.addEventListener("scenario_result", (ev) => {
      const data = JSON.parse(ev.data);
      // Preserved results replayed for a resume do not end the active scenario.
      if (!data.preserved) {
        state.liveActiveScenario = null;
        state.liveTurns = [];
      }
      if (data.result) {
        const exists = state.liveResults.some((r) => r.scenario_id === data.result.scenario_id);
        if (!exists) {
          state.liveResults.push(data.result);
        }
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
      const data = JSON.parse(ev.data);
      finish(null, data.error || "The live stream reported an error.");
    });

    es.addEventListener("stream_idle", (ev) => {
      const data = JSON.parse(ev.data);
      finish(null, data.message || "The live stream stopped after a long idle period.");
    });

    // EventSource reconnects on its own after a dropped connection, and the
    // server replays the run's history, so only a permanent failure ends it.
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        finish(null, "Live stream closed. Reopen the run to keep watching.");
      } else {
        state.liveNote = "Connection lost, reconnecting...";
        renderLiveObserverPane();
      }
    };
    es.onopen = () => {
      if (state.liveNote === "Connection lost, reconnecting...") {
        state.liveNote = null;
        renderLiveObserverPane();
      }
    };
  }

  async function resumeInterruptedRun(runId) {
    switchView("live");
    // loadLiveView renders the form only once, so the values typed earlier
    // are still there to read.
    await loadLiveView();
    const baseUrl = document.getElementById("launch-base-url")?.value.trim() || "";
    const apiKey = document.getElementById("launch-api-key")?.value.trim() || "";
    const wireFormat = document.getElementById("launch-wire-format")?.value || "auto";
    try {
      const res = await postJson(`/api/runs/${encodeURIComponent(runId)}/resume`, {
        base_url: baseUrl || undefined,
        api_key: apiKey || undefined,
        wire_format: wireFormat,
      });
      connectLiveStream(res.run_id, {
        model: res.model,
        total: res.total_scenarios,
      });
    } catch (err) {
      const errBox = document.getElementById("launch-error-msg");
      if (errBox) {
        errBox.textContent = err.message;
        errBox.classList.remove("is-hidden");
      }
    }
  }

  window.loadLiveView = loadLiveView;
  window.openLiveRun = (runId) => {
    switchView("live");
    loadLiveView().then(() => connectLiveStream(runId));
  };
  window.resumeInterruptedRun = resumeInterruptedRun;

  window.webWorkbench = {
    state,
    escapeHtml,
    statusBadgeClass,
    scoreBarClass,
    formatDate,
    fetchJson,
    switchView,
    loadRunsList,
    selectRun,
  };

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = btn.getAttribute("data-view");
        if (view) switchView(view);
      });
    });

    document.getElementById("refresh-all-btn")?.addEventListener("click", () => {
      loadRunsList();
    });

    document.getElementById("history-search")?.addEventListener("input", scheduleRunsListLoad);
    document.getElementById("history-status-filter")?.addEventListener("change", () => {
      loadRunsList();
    });
    document.getElementById("history-type-filter")?.addEventListener("change", () => {
      loadRunsList();
    });

    loadRunsList();
  });
})();
