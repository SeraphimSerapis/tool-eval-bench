// Head-to-head run comparison view with paired 3-tier statistics and McNemar's test.

import { byId, h, icon, relTime, setChildren, splitModel } from "./dom.js";
import { fetchJson, state } from "./state.js";
import { banner, chip, emptyState, ring, segmented, stat } from "./components.js";

export async function loadCompareView() {
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
      const score = r.final_score ?? (r.rating === "Withheld" ? "Withheld" : "—");
      const pair = r.reviewer_model ? ` + reviewer ${r.reviewer_model}` : "";
      return h("option", { value: r.run_id, text: `${name}${pair} · ${score} · ${relTime(r.created_at)} · ${r.run_id.slice(0, 15)}` });
    })
  );
}

function vsCard(side, run, selectId, selectedId, toolRuns, onPick) {
  return h(
    "div",
    { class: "card vs-card" },
    ring(run?.final_score, {
      sub:
        run && (run.final_score === null || run.final_score === undefined) && run.rating === "Withheld"
          ? "withheld"
          : null,
    }),
    h(
      "div",
      { class: "hero-title" },
      h("span", { class: "eyebrow", text: side }),
      h("label", { class: "visually-hidden", for: selectId, text: side }),
      run ? h("strong", { text: splitModel(run.model).name, title: run.model }) : null,
      run && run.reviewer_model
        ? h("span", { class: "eyebrow", text: `+ reviewer ${run.reviewer_model}` })
        : null,
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

export function renderCompareView(toolRuns) {
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
    const ps = cmp.paired_stats || {};
    const infraExcluded = Array.isArray(ps.infra_excluded_ids) ? ps.infra_excluded_ids : [];
    const discordantIds = Array.isArray(ps.discordant_scenario_ids) ? ps.discordant_scenario_ids : [];
    const comparabilityWarnings = Array.isArray(ps.comparability_warnings) ? ps.comparability_warnings : [];
    const fmtSignedPct = (val) => {
      const num = Number(val ?? 0);
      return `${num > 0 ? "+" : ""}${num.toFixed(2)}`;
    };
    if (cmp.config_mismatch) {
      body.push(
        h(
          "div",
          { class: "section-spacer" },
          banner(
            "warn",
            "alert",
            h("strong", { text: "Configuration mismatch: " }),
            (cmp.config_diffs || []).join(", "),
            ". McNemar's result may reflect the setup rather than the models."
          )
        )
      );
    }
    if (comparabilityWarnings.length > 0) {
      body.push(
        h(
          "div",
          { class: "section-spacer" },
          banner(
            "warn",
            "alert",
            h("strong", {
              text: `Comparability ${comparabilityWarnings.length === 1 ? "warning" : "warnings"} (${comparabilityWarnings.length})`,
            }),
            h(
              "ul",
              {},
              comparabilityWarnings.map((warning) => h("li", { text: String(warning) }))
            )
          )
        )
      );
    }
    if (infraExcluded.length > 0) {
      body.push(
        h(
          "div",
          { class: "section-spacer" },
          banner(
            "warn",
            "alert",
            h("strong", { text: "Excluded infrastructure failures: " }),
            `${infraExcluded.length} shared ${infraExcluded.length === 1 ? "scenario" : "scenarios"} skipped from paired statistics: ${infraExcluded.join(", ")}.`
          )
        )
      );
    }
    const scoreDelta = s.score_delta;
    const pairedVerdictMap = {
      no_shared_scenarios: ["info", "info", "No shared non-infra scenarios"],
      identical: ["info", "info", "Identical outcomes"],
      significant_improvement: ["pass", "check", "Significant improvement in B"],
      significant_regression: ["fail", "x", "Significant regression in B"],
      inconclusive_noise: ["partial", "alert", "Inconclusive across shared scenarios"],
    };
    const pv = pairedVerdictMap[ps.verdict] || ["plain", "minus", "Paired comparison"];
    const ciDetail =
      ps.shared_count > 0
        ? `${fmtSignedPct(ps.mean_pct_delta)} pts · 95% CI [${fmtSignedPct(ps.ci95_low_pct)}, ${fmtSignedPct(ps.ci95_high_pct)}] pts · 3-tier sign p = ${ps.sign_p_value ?? 1} · ${ps.shared_count} shared`
        : "No shared gradable scenarios to compute paired confidence interval.";
    const selectorSnippet = discordantIds.length ? `--scenarios ${discordantIds.join(" ")}` : "";
    const verdict =
      mc.discordant_pairs === 0
        ? ["info", "info", "Identical pass/fail outcomes", "No discordant pairs, so there is nothing to test."]
        : mc.significant
          ? [
              "pass",
              "check",
              `Significant: ${mc.direction === "B" ? "candidate B" : "baseline A"} is better`,
              `p = ${mc.p_value} · χ² = ${mc.chi2} · ${mc.discordant_pairs} discordant pairs (A only ${mc.a_only_pass}, B only ${mc.b_only_pass})`,
            ]
          : [
              "plain",
              "minus",
              "Not significant",
              `p = ${mc.p_value} · χ² = ${mc.chi2} · ${mc.discordant_pairs} discordant pairs (A only ${mc.a_only_pass}, B only ${mc.b_only_pass})`,
            ];
    const scoreLabel = (val, runObj) =>
      val !== null && val !== undefined ? String(val) : runObj?.rating === "Withheld" ? "Withheld" : "—";
    body.push(
      h(
        "div",
        { class: "stats" },
        stat("Score A → B", `${scoreLabel(s.score_a, cmp.run_a)} → ${scoreLabel(s.score_b, cmp.run_b)}`, {
          extra:
            scoreDelta === null || scoreDelta === undefined
              ? null
              : h("div", { class: scoreDelta >= 0 ? "delta-pos" : "delta-neg", text: `${scoreDelta >= 0 ? "+" : ""}${scoreDelta} points of score` }),
        }),
        stat("Points", `${s.total_points_a} → ${s.total_points_b}`, { small: `(${s.total_delta >= 0 ? "+" : ""}${s.total_delta})` }),
        stat("Improved", s.improved, { cls: "stat-pass", icon: "check" }),
        stat("Regressed", s.regressed, { cls: "stat-fail", icon: "x" }),
        stat("Unchanged", s.unchanged, { icon: "minus" })
      ),
      h(
        "div",
        { class: "card card-pad section-spacer" },
        h("div", { class: "eyebrow", text: "Paired 3-tier signal vs noise" }),
        h(
          "div",
          { class: "verdict" },
          h("span", { class: `chip chip-${pv[0]}` }, icon(pv[1]), pv[2]),
          h("span", { class: "muted mono", text: ciDetail })
        ),
        selectorSnippet
          ? h(
              "div",
              { class: "toolbar section-spacer" },
              h("span", { class: "muted", text: `Re-test ${discordantIds.length} discordant ${discordantIds.length === 1 ? "scenario" : "scenarios"}:` }),
              h("code", { id: "cmp-discordant-selector", class: "mono", text: selectorSnippet }),
              h("button", {
                type: "button",
                class: "btn btn-ghost",
                onclick: (ev) => {
                  const btn = ev.currentTarget;
                  navigator.clipboard
                    ?.writeText(selectorSnippet)
                    .then(() => {
                      if (!(btn instanceof HTMLElement)) return;
                      btn.textContent = "Copied";
                      setTimeout(() => {
                        btn.textContent = "Copy --scenarios";
                      }, 1500);
                    })
                    .catch(() => {});
                },
                text: "Copy --scenarios",
              })
            )
          : null
      ),
      h(
        "div",
        { class: "card card-pad section-spacer" },
        h("div", { class: "eyebrow", text: "McNemar's paired test" }),
        h(
          "div",
          { class: "verdict" },
          h("span", { class: `chip chip-${verdict[0]}` }, icon(verdict[1]), verdict[2]),
          h("span", { class: "muted mono", text: verdict[3] })
        )
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
          h(
            "thead",
            {},
            h("tr", {}, ["Scenario", "Title", "Baseline A", "Candidate B", "Δ pts", "Δ time", "Summary"].map((t, i) => h("th", { scope: "col", class: i === 5 ? "num" : null, text: t })))
          ),
          h("tbody", {}, deltas.length ? compareRows(deltas, cmp) : h("tr", {}, h("td", { colspan: 7, class: "muted", text: "No scenarios match this filter." })))
        )
      )
    );
  }

  setChildren(workspace, head, versus, ...body);
}
