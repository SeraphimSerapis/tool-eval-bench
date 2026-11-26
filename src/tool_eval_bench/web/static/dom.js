// Shared SVG icon definitions, safe DOM builders, and pure formatting helpers.
//
// Every element is built with document.createElement and textContent; no
// HTML strings are parsed, so stored run data (model output, tool results,
// labels) can never become markup. Styles live in styles.css; the only
// dynamic styling is a numeric custom property set through the CSSOM.

const SVG_NS = "http://www.w3.org/2000/svg";

// Stroke icons as [tag, attributes] lists, drawn on a 24x24 grid.
export const ICONS = {
  check: [["polyline", { points: "20 6 9 17 4 12" }]],
  x: [["path", { d: "M18 6 6 18M6 6l12 12" }]],
  minus: [["path", { d: "M5 12h14" }]],
  pause: [["path", { d: "M9 5v14M15 5v14" }]],
  play: [["polygon", { points: "6 4 20 12 6 20 6 4" }]],
  stop: [["rect", { x: "6", y: "6", width: "12", height: "12", rx: "2" }]],
  activity: [["polyline", { points: "22 12 18 12 15 21 9 3 6 12 2 12" }]],
  chevron: [["polyline", { points: "9 6 15 12 9 18" }]],
  search: [
    ["circle", { cx: "11", cy: "11", r: "7" }],
    ["path", { d: "M21 21l-4.3-4.3" }],
  ],
  shield: [["path", { d: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z" }]],
  alert: [
    ["path", { d: "M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" }],
    ["path", { d: "M12 9v4M12 17h.01" }],
  ],
  info: [
    ["circle", { cx: "12", cy: "12", r: "9" }],
    ["path", { d: "M12 11v5M12 8h.01" }],
  ],
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
  tag: [
    ["path", { d: "M20 12l-8 8-9-9V3h8l9 9z" }],
    ["circle", { cx: "7.5", cy: "7.5", r: "1.5" }],
  ],
  download: [["path", { d: "M12 3v12M7 10l5 5 5-5M5 21h14" }]],
  radio: [
    ["circle", { cx: "12", cy: "12", r: "2" }],
    ["path", { d: "M16.2 7.8a6 6 0 0 1 0 8.4M7.8 16.2a6 6 0 0 1 0-8.4M19.1 4.9a10 10 0 0 1 0 14.2M4.9 19.1a10 10 0 0 1 0-14.2" }],
  ],
  wrench: [
    ["path", { d: "M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.5-.5-.5-2.5 2.5-2.5z" }],
  ],
  message: [["path", { d: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" }]],
  clock: [
    ["circle", { cx: "12", cy: "12", r: "9" }],
    ["polyline", { points: "12 7 12 12 15 14" }],
  ],
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
  trash: [["path", { d: "M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6" }]],
};

export function icon(name) {
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

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

// h("div", { class, text, dataset, style, onclick, ...attributes }, ...children)
export function h(tag, props, ...children) {
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
export function setChildren(el, ...children) {
  if (!el) return null;
  el.replaceChildren();
  return append(el, children);
}

export const byId = (id) => document.getElementById(id);
export const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------ Formatting

export function splitModel(model) {
  const parts = String(model || "unknown").split("/").filter(Boolean);
  const name = parts.pop() || "unknown";
  return { org: parts.length ? parts[parts.length - 1] : "", name };
}

export function tierFor(pct) {
  if (pct === null || pct === undefined || Number.isNaN(Number(pct))) return "";
  if (pct >= 75) return "tier-pass";
  if (pct >= 45) return "tier-partial";
  return "tier-fail";
}

export function heatClass(pct) {
  if (pct >= 90) return "heat-90";
  if (pct >= 75) return "heat-75";
  if (pct >= 60) return "heat-60";
  return "heat-low";
}

export function parseDate(iso) {
  if (!iso) return null;
  const text = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`;
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? null : date;
}

const relFormatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

export function relTime(iso) {
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

export function absTime(iso) {
  const date = parseDate(iso);
  return date ? date.toLocaleString() : String(iso || "");
}

export function fmtSeconds(value) {
  return value === undefined || value === null ? "—" : `${Number(value).toFixed(1)}s`;
}

export function fmtTokens(value) {
  if (!value) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 100000 ? 0 : 1)}k` : String(value);
}

export function pctOf(part, whole) {
  return whole ? Math.round((part / whole) * 100) : 0;
}

export function fmtNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function fmtFigure(value, unit) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value !== "number") return String(value);
  const text = fmtNumber(value);
  if (!unit) return text;
  return unit === "%" || unit === "×" ? `${text}${unit}` : `${text} ${unit}`;
}

export function median(values) {
  const sorted = values.filter((v) => typeof v === "number" && !Number.isNaN(v)).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
