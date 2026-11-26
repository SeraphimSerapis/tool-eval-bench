// tool-eval-bench web UI entry point: tab routing, view wiring, and boot.

import { byId } from "./dom.js";
import { fetchJson, state } from "./state.js";
import {
  deleteRuns,
  initHistory,
  loadRunsList,
  renderInspector,
  scheduleRunsListLoad,
  selectRun,
  setSelecting,
  toggleSelectAll,
} from "./history.js";
import { initLeaderboard, loadLeaderboardView } from "./leaderboard.js";
import { loadCompareView } from "./compare.js";
import {
  initLive,
  loadLiveView,
  openLiveRun,
  renderLiveObserverPane,
  resumeInterruptedRun,
  setLiveBadge,
} from "./live.js";

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

// Navigation mediator breaking cross-view import cycles.
const nav = {
  switchView,
  openRun(runId) {
    switchView("history");
    selectRun(runId);
  },
  openLiveRun,
  resumeRun: resumeInterruptedRun,
  refreshRuns: loadRunsList,
  onRunsDeleted() {
    setLiveBadge(false);
    renderLiveObserverPane();
  },
};

initHistory(nav);
initLeaderboard(nav);
initLive(nav);

window.webWorkbench = { state, switchView, loadRunsList, selectRun, fetchJson };

function boot() {
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
  byId("runs-select-toggle")?.addEventListener("click", () => setSelecting(!state.selecting));
  byId("runs-select-all")?.addEventListener("click", toggleSelectAll);
  byId("runs-delete-selected")?.addEventListener("click", () =>
    deleteRuns(state.runs.filter((r) => state.deleteSelection.has(r.run_id)))
  );
  byId("history-type-filter")?.addEventListener("change", loadRunsList);
  renderInspector();
  loadRunsList();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
