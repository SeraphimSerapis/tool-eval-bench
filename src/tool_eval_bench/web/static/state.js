// Shared application state and JSON API client helpers.

// Sentinel option in the model picker for "type an id the probe did not list".
export const MODEL_CUSTOM = "__custom__";

export const state = {
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
  probeAttempted: false,
  liveRunId: null,
  liveRunStatus: null,
  liveModel: null,
  liveReviewer: null,
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
  // Selection mode in the run list, for deleting several runs at once.
  selecting: false,
  deleteSelection: new Set(),
};

export async function fetchJson(url, options) {
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
export function postJson(url, body) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

export async function ensureSchema() {
  if (!state.launcherSchema) state.launcherSchema = await fetchJson("/api/schema");
  return state.launcherSchema;
}
