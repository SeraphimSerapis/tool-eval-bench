// Cohort leaderboard view.

import { byId, fmtTokens, h, heatClass, icon, setChildren, tierFor } from "./dom.js";
import { ensureSchema, fetchJson, state } from "./state.js";
import { bar, emptyState, modelBlock, ppfBar } from "./components.js";

let nav = null;
let leaderboardDebounce = null;

export function initLeaderboard(navApi) {
  nav = navApi;
}

export async function loadLeaderboardView() {
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
  const open = () => nav?.openRun(row.run_id);
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
    h(
      "td",
      { class: "sticky-col sticky-model" },
      modelBlock(row.model, "model-cell"),
      // A paired score belongs to the model and its reviewer together.
      row.reviewer_model ? h("div", { class: "run-meta muted", text: `+ reviewer ${row.reviewer_model}` }) : null
    ),
    h(
      "td",
      {},
      row.final_score !== null && row.final_score !== undefined
        ? h("div", { class: `score-cell ${tierFor(row.final_score)}` }, h("span", { text: row.final_score }), bar(row.final_score, tierFor(row.final_score)))
        : h("div", { class: "score-cell" }, h("span", { class: "muted", text: "Withheld" })),
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

export function renderLeaderboardView(data, schema) {
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
    setChildren(
      workspace,
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
