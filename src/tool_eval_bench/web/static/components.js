// Shared UI primitive components built with safe DOM nodes.

import { fmtFigure, fmtNumber, h, icon, splitModel, tierFor } from "./dom.js";

export const STATUS_ICONS = {
  pass: "check",
  completed: "check",
  partial: "minus",
  interrupted: "pause",
  fail: "x",
  failed: "x",
  running: "activity",
};

// Status always pairs a color with an icon and a word, never color alone.
export function chip(status, label) {
  const key = String(status || "").toLowerCase();
  const known = STATUS_ICONS[key];
  return h(
    "span",
    { class: `chip ${known ? `chip-${key}` : "chip-plain"}` },
    known ? icon(known) : null,
    label ?? (status || "unknown")
  );
}

export function ring(value, opts = {}) {
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

// Tool-call runs have a 0-100 score. Every other mode stores measurements,
// and the server picks the one figure (its headline) the UI shows instead.
export function isToolEval(run) {
  return (run.run_type || "tool_eval") === "tool_eval";
}

export function hasFigure(headline) {
  return Boolean(headline) && headline.value !== null && headline.value !== undefined;
}

export function metric(headline, opts = {}) {
  const has = hasFigure(headline);
  const value = has ? Number(headline.value) : null;
  // The list slot is narrow: large values drop their decimals.
  const shown = has ? (Math.abs(value) >= 100 ? Math.round(value).toLocaleString() : fmtNumber(value)) : "—";
  return h(
    "div",
    {
      class: `metric ${has ? "" : "metric-empty"} ${opts.cls || ""}`,
      role: "img",
      "aria-label": has ? `${headline.label} ${fmtFigure(headline.value, headline.unit)}` : "Nothing measured",
      title: has && headline.detail ? headline.detail : null,
    },
    h("span", { class: "metric-value", text: shown }),
    has && headline.unit ? h("span", { class: "metric-unit", text: headline.unit }) : null,
    has && opts.withLabel ? h("span", { class: "metric-label", text: headline.label }) : null
  );
}

export function typeChip(run) {
  return h("span", { class: "chip chip-type", text: run.type_label || run.run_type });
}

export function bar(pct, cls = "") {
  return h("div", { class: `bar ${cls}`, role: "presentation" }, h("span", { style: { "--value": pct } }));
}

export function ppfBar(p, q, f) {
  return h(
    "div",
    { class: "ppf", role: "presentation" },
    h("span", { class: "p", style: { "--value": p } }),
    h("span", { class: "q", style: { "--value": q } }),
    h("span", { class: "f", style: { "--value": f } })
  );
}

export function emptyState(iconName, title, text, ...actions) {
  return h(
    "div",
    { class: "empty" },
    h("div", { class: "empty-icon" }, icon(iconName)),
    h("h2", { text: title }),
    text ? h("p", { text }) : null,
    actions.length ? h("div", { class: "quick-links" }, actions) : null
  );
}

export function stat(label, value, opts = {}) {
  return h(
    "div",
    { class: `card stat ${opts.cls || ""}` },
    h("div", { class: "stat-label" }, opts.icon ? icon(opts.icon) : null, label),
    h("div", { class: "stat-value" }, value, opts.small ? h("small", { text: ` ${opts.small}` }) : null),
    opts.extra || null
  );
}

export function banner(kind, iconName, ...content) {
  return h(
    "div",
    { class: `banner banner-${kind}`, role: kind === "danger" ? "alert" : "status" },
    icon(iconName),
    h("div", {}, content)
  );
}

export function segmented(options, current, onPick, label) {
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

export function tile(status, title, opts = {}) {
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

export function legend() {
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

export function modelBlock(model, cls) {
  const { org, name } = splitModel(model);
  return h(
    "div",
    { class: cls, title: model },
    org ? h("span", { class: cls === "run-model" ? "run-org" : "org", text: org }) : null,
    h("span", { class: cls === "run-model" ? "run-name" : "name", text: name })
  );
}

export function preserveScroll(container, render) {
  const top = container.scrollTop;
  render();
  container.scrollTop = top;
}
