// Run launcher form, server/reviewer probing, and SSE live observer pane.

import { byId, fmtSeconds, h, icon, setChildren, splitModel } from "./dom.js";
import { ensureSchema, MODEL_CUSTOM, postJson, state } from "./state.js";
import { banner, bar, chip, emptyState, legend, ring, stat, tile } from "./components.js";

let nav = null;

export function initLive(navApi) {
  nav = navApi;
}

// Builds the launcher form once. Later calls only refresh the observer pane,
// so switching tabs or resuming never resets what the user typed.
export function loadLiveView() {
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

// The probe fills a <select> rather than a <datalist>: browsers filter
// datalist options by the input's current value, so auto-filling the first
// model hid every other entry and made the list look unclickable. A free
// text field stays available behind the "Custom model ID…" entry for
// servers that advertise nothing or report names only the user knows.
function syncModelPicker(options = {}) {
  const selectEl = byId("launch-model-select");
  const customInput = byId("launch-model");
  if (!selectEl || !customInput) return;
  const customSelected = selectEl.value === MODEL_CUSTOM || selectEl.value === "";
  customInput.hidden = !customSelected;
  if (options.focus && customSelected) customInput.focus();
}

function readModelField() {
  const selectEl = byId("launch-model-select");
  if (!selectEl || selectEl.value === MODEL_CUSTOM) return (byId("launch-model")?.value || "").trim();
  return selectEl.value;
}

function fillModelPicker(models, preferred = "") {
  const selectEl = byId("launch-model-select");
  const customInput = byId("launch-model");
  if (!selectEl || !customInput) return;
  const list = (models || []).filter((m) => m && m.id);
  const typed = customInput.value.trim();
  const wasCustom = selectEl.value === MODEL_CUSTOM;
  const wanted = (preferred || (wasCustom ? typed : selectEl.value) || "").trim();

  const options = list.length
    ? list.map((m) => h("option", { value: m.id, text: m.display_name || m.id }))
    : [
        h("option", {
          value: "",
          text: state.probeAttempted ? "Endpoint reported no models" : "Probe the endpoint to list models",
          disabled: true,
        }),
      ];
  options.push(h("option", { value: MODEL_CUSTOM, text: "Custom model ID…" }));
  selectEl.replaceChildren(...options);

  if (list.some((m) => m.id === wanted)) {
    selectEl.value = wanted;
  } else if (wasCustom && typed) {
    selectEl.value = MODEL_CUSTOM;
  } else if (list.length) {
    selectEl.value = list[0].id;
  } else {
    selectEl.value = MODEL_CUSTOM;
    if (wanted) customInput.value = wanted;
  }
  syncModelPicker();
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
    state.probeAttempted = true;
    if (urlInput && !urlInput.value.trim() && res.resolved_base_url) urlInput.value = res.resolved_base_url;
    const backendSel = byId("launch-backend");
    if (backendSel && res.backend && res.backend !== "unknown") backendSel.value = res.backend;
    fillModelPicker(state.probedModels);
    setProbeStatus("ok", `Connected to ${res.server_name || res.backend || "server"} · ${state.probedModels.length} model${state.probedModels.length === 1 ? "" : "s"}`);
  } catch (err) {
    setProbeStatus("error", err.message);
  }
}

function urlOrigin(url) {
  const trimmed = (url || "").trim();
  if (!trimmed) return "";
  try {
    return new URL(trimmed).origin;
  } catch {
    return trimmed;
  }
}

// The server a reviewer request goes to: its own URL, or the benchmarked one
// when the reviewer URL is blank.
function effectiveReviewerOrigin() {
  const own = byId("launch-reviewer-base-url")?.value.trim() || "";
  return urlOrigin(own || byId("launch-base-url")?.value || "");
}

// A reviewer key belongs to the server it was typed for. When the effective
// reviewer origin changes, the key is cleared, so it never travels there.
function bindReviewerKeyToOrigin() {
  const key = byId("launch-reviewer-api-key");
  if (!key) return;
  key.dataset.origin = effectiveReviewerOrigin();
  delete key.dataset.cleared;
}

// Returns whether it cleared the key.
function forgetReviewerKeyOnNewOrigin() {
  const key = byId("launch-reviewer-api-key");
  if (!key || !key.value) return false;
  const now = effectiveReviewerOrigin();
  const bound = key.dataset.origin;
  if (bound === undefined || bound === "") {
    // Typed before either URL was known: it belongs to the first one entered.
    key.dataset.origin = now;
    return false;
  }
  if (bound !== now) {
    key.value = "";
    delete key.dataset.origin;
    // Remembered until the next probe reports it: the URL field's change
    // event usually clears the key before the probe button's click runs.
    key.dataset.cleared = "1";
    const feedback = byId("reviewer-probe-feedback");
    if (feedback) {
      feedback.className = "probe-status is-error";
      feedback.textContent = "The reviewer API key was cleared: it was entered for another server.";
    }
    return true;
  }
  return false;
}

async function probeReviewer() {
  const keyInput = byId("launch-reviewer-api-key");
  const cleared = forgetReviewerKeyOnNewOrigin() || keyInput?.dataset.cleared === "1";
  if (keyInput) delete keyInput.dataset.cleared;
  const feedback = byId("reviewer-probe-feedback");
  const setStatus = (kind, text) => {
    if (!feedback) return;
    feedback.className = `probe-status ${kind ? `is-${kind}` : ""}`;
    feedback.textContent = text;
  };
  // A blank reviewer URL means the benchmarked server, as for a launch, and
  // then the benchmark's wire format applies too.
  const ownUrl = byId("launch-reviewer-base-url")?.value.trim() || "";
  const url = ownUrl || byId("launch-base-url")?.value.trim() || "";
  if (!url) {
    setStatus("error", "Enter the reviewer's Base URL, or the benchmark Base URL to review on that server.");
    return;
  }
  setStatus("busy", "Probing reviewer…");
  try {
    // The same rule as a launch: without a key of its own, a reviewer on the
    // benchmarked server's origin uses the run's key; elsewhere it gets none.
    const ownKey = byId("launch-reviewer-api-key")?.value.trim() || "";
    const mainUrl = byId("launch-base-url")?.value || "";
    const sameOrigin = urlOrigin(mainUrl) !== "" && urlOrigin(url) === urlOrigin(mainUrl);
    const runKey = byId("launch-api-key")?.value.trim() || "";
    const sentKey = ownKey || (sameOrigin ? runKey : "");
    const res = await postJson("/api/probe", {
      purpose: "reviewer",
      base_url: url,
      api_key: sentKey || undefined,
      wire_format: ownUrl ? undefined : byId("launch-wire-format")?.value || "auto",
    });
    if (!res.reachable) {
      // The probe never falls back to TOOL_EVAL_API_KEY, which belongs to
      // the benchmarked server; a launch on that server does use it.
      // On the benchmark's origin the run's key field would be used; elsewhere
      // only the Reviewer API key field is.
      const hint = sentKey
        ? ""
        : sameOrigin
          ? " · if the server takes its key from TOOL_EVAL_API_KEY, enter it in the API key field to probe"
          : " · if the reviewer needs a key, enter it in the Reviewer API key field";
      setStatus("error", `${res.error || "Unreachable"}${hint}`);
      return;
    }
    const models = (res.models || []).filter((m) => m && m.id);
    setChildren(byId("launch-reviewer-models"), ...models.map((m) => h("option", { value: m.id, text: m.display_name || m.id })));
    const modelInput = byId("launch-reviewer-model");
    const current = modelInput ? modelInput.value.trim() : "";
    let replaced = "";
    if (modelInput && models.length && !models.some((m) => m.id === current)) {
      if (current) {
        // As in the benchmark picker: a model this server does not list
        // belongs to the previous one, so the first listed model replaces it.
        replaced = current;
        modelInput.value = models[0].id;
      } else if (models.length === 1) {
        // A blank model is a solo run: fill it only when the choice is clear.
        modelInput.value = models[0].id;
      }
    }
    const note = cleared ? " · the API key entered for another server was cleared" : "";
    const swap = replaced ? ` · ${replaced} is not listed here, now ${modelInput.value}` : "";
    setStatus("ok", `Reviewer reachable · ${models.length} model${models.length === 1 ? "" : "s"}${swap}${note}`);
  } catch (err) {
    setStatus("error", err.message);
  }
}

// Reviewer fields go out only with a reviewer model: a blank model is a solo run.
function readReviewerFields() {
  forgetReviewerKeyOnNewOrigin();
  const model = byId("launch-reviewer-model")?.value.trim() || "";
  if (!model) return { ok: true, fields: {} };
  const rounds = readNumericField("launch-reviewer-max-rounds", 1, "Reviews per turn");
  if (!rounds.ok) return rounds;
  if (!Number.isInteger(rounds.value) || rounds.value < 1) return { ok: false, error: "Reviews per turn must be a whole number of at least 1." };
  const tempEl = byId("launch-reviewer-temperature");
  const tempRaw = tempEl?.value.trim() || "";
  const temperature = tempRaw === "" ? { ok: true, value: undefined } : readNumericField("launch-reviewer-temperature", undefined, "Reviewer temperature");
  if (!temperature.ok) return temperature;
  return {
    ok: true,
    fields: {
      reviewer_model: model,
      reviewer_base_url: byId("launch-reviewer-base-url")?.value.trim() || undefined,
      reviewer_api_key: byId("launch-reviewer-api-key")?.value.trim() || undefined,
      reviewer_scope: byId("launch-reviewer-scope")?.value || "turns",
      reviewer_max_rounds: rounds.value,
      reviewer_temperature: temperature.value,
    },
  };
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
    h(
      "div",
      { class: "field" },
      h("label", { class: "field-label", for: "launch-model-select", text: "Model" }),
      h(
        "div",
        { class: "model-picker" },
        h("select", { id: "launch-model-select", class: "select", onchange: () => syncModelPicker({ focus: true }) }),
        input("launch-model", { class: "input mono", placeholder: "e.g. Qwen/Qwen3-8B", autocomplete: "off", hidden: true })
      ),
      h("span", { class: "field-hint", text: "Probe the endpoint to fill this list, or pick Custom model ID." })
    ),
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
      field("Seed", input("launch-seed", { type: "number", class: "input mono", step: "1", placeholder: "Unseeded" }), "Optional; blank leaves the run unseeded."),
      field("Concurrency", input("launch-parallel", { type: "number", class: "input mono", step: "1", min: "1", max: "32", value: defaults.concurrency ?? 1 })),
      field("Max turns", input("launch-max-turns", { type: "number", class: "input mono", step: "1", min: "1", max: "25", value: defaults.max_turns ?? 8 })),
      field("Timeout (s)", input("launch-timeout", { type: "number", class: "input mono", step: "5", min: "5", max: "1800", value: defaults.timeout_seconds ?? 120 }))
    ),
    h("div", { class: "section-spacer" }, field("Run label", input("launch-label", { placeholder: "e.g. vllm-0.11-fp8-baseline" })))
  );

  const reviewerSection = h(
    "details",
    { class: "card launcher-section advanced", id: "launch-reviewer-section" },
    h("summary", {}, icon("chevron"), icon("eye"), "Reviewer (optional)"),
    h("span", {
      class: "field-hint",
      text: "A second model reviews each turn before it runs, and the benchmarked model revises a rejected turn. The score belongs to the pair. Leave the model empty for a solo run.",
    }),
    h(
      "div",
      { class: "field" },
      h("label", { class: "field-label", for: "launch-reviewer-base-url", text: "Reviewer base URL" }),
      h(
        "div",
        { class: "input-row" },
        input("launch-reviewer-base-url", { class: "input mono", placeholder: "Blank: the benchmarked server", autocomplete: "off", onchange: forgetReviewerKeyOnNewOrigin }),
        h("button", { type: "button", id: "probe-reviewer-btn", class: "btn", onclick: probeReviewer }, icon("radio"), "Probe")
      ),
      h("div", { id: "reviewer-probe-feedback", class: "probe-status", role: "status" })
    ),
    field("Reviewer model", input("launch-reviewer-model", { class: "input mono", placeholder: "Probe to list models, or type an ID", autocomplete: "off", list: "launch-reviewer-models" })),
    h("datalist", { id: "launch-reviewer-models" }),
    field(
      "Reviewer API key",
      input("launch-reviewer-api-key", { type: "password", class: "input mono", placeholder: "Blank: the run's key, on the same server only", autocomplete: "off", oninput: bindReviewerKeyToOrigin }),
      "Sent to the reviewer only; never stored."
    ),
    h(
      "div",
      { class: "grid-2" },
      field("Scope", select("launch-reviewer-scope", [["turns", "Every turn"], ["tools", "Tool-call turns"], ["final", "Final replies"]], "turns")),
      field("Reviews per turn", input("launch-reviewer-max-rounds", { type: "number", class: "input mono", step: "1", min: "1", max: "5", value: 1 })),
      field("Reviewer temperature", input("launch-reviewer-temperature", { type: "number", class: "input mono", step: "0.05", min: "0", max: "2", placeholder: "Same as the run" }))
    )
  );

  const form = h(
    "form",
    { id: "launch-run-form", onsubmit: submitLaunch, novalidate: true },
    endpoint,
    reviewerSection,
    suite,
    advanced,
    h("div", { id: "launch-error-msg", class: "banner banner-danger", role: "alert", hidden: true }),
    h("button", { type: "submit", id: "launch-submit-btn", class: "btn btn-primary btn-block" }, icon("play"), "Start benchmark run")
  );

  setChildren(
    workspace,
    h("aside", { class: "launcher", "aria-label": "Launch a benchmark run" }, h("h1", { class: "rail-title", text: "Launch a run" }), h("div", { class: "section-spacer" }, form)),
    h("section", { id: "live-observer-pane", class: "observer", "aria-label": "Live run progress", "aria-live": "polite" })
  );
  fillModelPicker(state.probedModels, defaults.model || "");
  renderLiveObserverPane();
}

function showLaunchError(message) {
  const box = byId("launch-error-msg");
  if (!box) return;
  box.replaceChildren(icon("alert"), h("div", { text: message }));
  box.hidden = false;
}

function readNumericField(id, fallback, label) {
  const el = byId(id);
  if (!el) return { ok: true, value: fallback };
  if (el.validity?.badInput) return { ok: false, error: `${label} must be a valid number.` };
  const raw = el.value.trim();
  if (raw === "") return { ok: true, value: fallback };
  const num = Number(raw);
  if (!Number.isFinite(num)) return { ok: false, error: `${label} must be a finite number.` };
  return { ok: true, value: num };
}

// The seed is optional, so a blank field must stay unseeded instead of turning
// into 0, while an explicit 0 is a seed the server accepts. Fractions are
// refused here because /api/runs takes a whole number.
function readSeedField() {
  const el = byId("launch-seed");
  if (!el) return { ok: true, value: undefined };
  if (el.validity?.badInput) return { ok: false, error: "Seed must be a whole number." };
  const raw = el.value.trim();
  if (raw === "") return { ok: true, value: undefined };
  const num = Number(raw);
  if (!Number.isFinite(num) || !Number.isInteger(num)) {
    return { ok: false, error: "Seed must be a whole number." };
  }
  return { ok: true, value: num };
}

async function submitLaunch(ev) {
  ev.preventDefault();
  const box = byId("launch-error-msg");
  if (box) box.hidden = true;
  const temperature = readNumericField("launch-temperature", 0, "Temperature");
  const concurrency = readNumericField("launch-parallel", 1, "Concurrency");
  const maxTurns = readNumericField("launch-max-turns", 8, "Max turns");
  const timeoutSeconds = readNumericField("launch-timeout", 120, "Timeout (s)");
  const seed = readSeedField();
  const reviewer = readReviewerFields();
  for (const parsed of [temperature, concurrency, maxTurns, timeoutSeconds, seed, reviewer]) {
    if (!parsed.ok) {
      showLaunchError(parsed.error);
      return;
    }
  }
  const preset = byId("launch-preset")?.value || "short";
  const categoriesChecked = [...document.querySelectorAll(".launch-cat-cb:checked")].map((cb) => cb.value);
  const explicitScenarios = byId("launch-scenarios")?.value || "";
  const payload = {
    base_url: byId("launch-base-url")?.value || "",
    model: readModelField(),
    backend: byId("launch-backend")?.value || "unknown",
    wire_format: byId("launch-wire-format")?.value || "auto",
    api_key: byId("launch-api-key")?.value || undefined,
    preset,
    categories: preset === "custom" && categoriesChecked.length ? categoriesChecked : undefined,
    scenarios: preset === "custom" && explicitScenarios.trim() ? explicitScenarios : undefined,
    temperature: temperature.value,
    // undefined is dropped by JSON.stringify, so a blank field sends no seed.
    seed: seed.value,
    concurrency: concurrency.value,
    max_turns: maxTurns.value,
    timeout_seconds: timeoutSeconds.value,
    label: byId("launch-label")?.value || undefined,
    ...reviewer.fields,
  };
  if (!payload.model) {
    showLaunchError("Pick a probed model or enter a custom model ID.");
    return;
  }
  const button = byId("launch-submit-btn");
  if (button) button.disabled = true;
  try {
    const started = await postJson("/api/runs", payload);
    connectLiveStream(started.run_id, { model: started.model, reviewer: started.reviewer_model, total: started.total_scenarios });
    nav?.refreshRuns();
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

export function renderLiveObserverPane() {
  const pane = byId("live-observer-pane");
  if (!pane) return;

  if (!state.liveRunId) {
    const watchable = state.runs.filter((r) => r.status === "running" || r.status === "interrupted");
    setChildren(
      pane,
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
            onclick: () => nav?.openRun(state.liveRunId),
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
        // The score belongs to the pair; the heading stays the benchmarked model.
        state.liveReviewer
          ? h("span", { class: "meta-item", title: "Reviewed by a second model", text: `+ reviewer ${state.liveReviewer}` })
          : null,
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
      h(
        "thead",
        {},
        h("tr", {}, ["Scenario", "Title", "Verdict", "Pts", "Duration", "Summary"].map((t, i) => h("th", { scope: "col", class: i === 3 || i === 4 ? "num" : null, text: t })))
      ),
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

  setChildren(
    pane,
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
  nav?.refreshRuns();
}

export function setLiveBadge(on) {
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
  state.liveReviewer = initialMeta?.reviewer || null;
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
    nav?.refreshRuns();
  };

  es.addEventListener("run_started", (ev) => {
    const data = JSON.parse(ev.data);
    state.liveModel = data.model || state.liveModel;
    state.liveReviewer = data.reviewer_model || state.liveReviewer;
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

export async function resumeInterruptedRun(runId) {
  nav?.switchView("live");
  // loadLiveView renders the form only once, so values typed earlier are
  // still there to read.
  await loadLiveView();
  const baseUrl = byId("launch-base-url")?.value.trim() || "";
  const apiKey = byId("launch-api-key")?.value.trim() || "";
  const wireFormat = byId("launch-wire-format")?.value || "auto";
  forgetReviewerKeyOnNewOrigin();
  try {
    const res = await postJson(`/api/runs/${encodeURIComponent(runId)}/resume`, {
      base_url: baseUrl || undefined,
      api_key: apiKey || undefined,
      wire_format: wireFormat,
      // Used only when the run was reviewed: the stored reviewer URL is redacted.
      reviewer_base_url: byId("launch-reviewer-base-url")?.value.trim() || undefined,
      reviewer_api_key: byId("launch-reviewer-api-key")?.value.trim() || undefined,
    });
    connectLiveStream(res.run_id, { model: res.model, reviewer: res.reviewer_model, total: res.total_scenarios });
  } catch (err) {
    showLaunchError(err.message);
    if (state.liveRunId === runId) {
      state.liveNote = `Could not resume the run: ${err.message}`;
      renderLiveObserverPane();
    }
  }
}

export function openLiveRun(runId) {
  nav?.switchView("live");
  loadLiveView().then(() => connectLiveStream(runId));
}
