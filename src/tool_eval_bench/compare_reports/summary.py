#!/usr/bin/env python3
"""Generate a head-to-head comparison HTML from two cross-trial summary markdown files.

Usage:
    python compare_summary.py <summary_a.md> <summary_b.md> <output.html>
"""

import re
import sys
from pathlib import Path
from typing import Any

# ─── Parser ─────────────────────────────────────────────────────────────────


def parse_summary(fp: str) -> dict:
    txt = Path(fp).read_text(encoding="utf-8")
    d: dict = {}

    d["model_name"] = _r(r"^# Cross-Trial Summary — (.+)$", txt, fl=re.M) or "Unknown"
    d["run_id"] = _r(r"\*\*Run ID\*\*:\s*`(.+?)`", txt)
    d["date"] = _r(r"\*\*Date\*\*:\s*`(.+?)`", txt)
    d["date_short"] = d["date"][:10] if d["date"] else ""
    d["version"] = _r(r"\*\*tool-eval-bench\*\*:\s*`(.+?)`", txt)
    d["trials"] = int(_r(r"\*\*Trials\*\*:\s*(\d+)", txt) or 0)

    d["backend"] = _tv("Backend", txt)
    d["model_api"] = _tv("Model (API)", txt, strip_bt=True)
    d["model_root"] = _tv("Model (Root)", txt, strip_bt=True)
    d["temperature"] = _tv("Temperature", txt)
    d["thinking"] = _tv("Thinking", txt)

    # Headline scores - mean
    hm = re.search(r"\*\*Final Score\*\*.*?\|\s*\*\*([\d.]+)\s*±\s*([\d.]+)\*\*", txt)
    d["mean_score"] = float(hm.group(1)) if hm else 0.0
    d["std_score"] = float(hm.group(2)) if hm else 0.0

    pm = re.search(r"\*\*Total Points\*\*.*?\|\s*\*\*([\d.]+)\s*±\s*([\d.]+)\*\*", txt)
    d["mean_points"] = float(pm.group(1)) if pm else 0.0
    d["std_points"] = float(pm.group(2)) if pm else 0.0

    # Rating — the Rating row is a pipe-delimited table line; take the last (summary) column
    rating_m = re.search(r"^\|\s*\*\*Rating\*\*\s*\|(.+)", txt, re.M)
    if rating_m:
        cells = [c.strip() for c in rating_m.group(1).split("|") if c.strip()]
        d["rating"] = cells[-1] if cells else ""
    else:
        d["rating"] = ""

    # Safety warnings (sum across trials or max)
    sws = re.findall(r"\*\*Safety Warnings\*\*.*?\|\s*(\d+)", txt)
    d["safety_warnings"] = [int(x) for x in sws] if sws else []

    # Reliability metrics
    d["pass_at_8"] = _r(r"\*\*Pass@8\*\*.*?\|\s*([\d.]+)%", txt)
    d["pass_8"] = _r(r"\*\*Pass\^8\*\*.*?\|\s*([\d.]+)%", txt)
    d["reliability_gap"] = _r(r"\*\*Reliability Gap\*\*.*?\|\s*([\d.]+)pp", txt)
    d["ci_95"] = _r(r"\*\*95% CI\*\*.*?\|\s*\[([^\]]+)\]", txt)

    # Category variance
    d["categories"] = _parse_cat_var(txt)

    # Per-scenario results
    d["scenarios"] = _parse_summary_scenarios(txt)

    # Never passes, flaky, consistently partial
    d["never_passes"] = _parse_never_passes(txt)
    d["flaky"] = _parse_flaky(txt)
    d["consistent_partials"] = _parse_consistent_partials(txt)

    # Deployability
    d["quality"] = int(_r(r"\*\*Quality\*\*.*?\|\s*(\d+)\s*/\s*100", txt) or 0)
    d["responsiveness"] = int(_r(r"\*\*Responsiveness\*\*.*?\|\s*(\d+)\s*/\s*100", txt) or 0)
    d["deployability"] = int(_r(r"\*\*Deployability\*\*.*?\|\s*\*\*(\d+)\*\*", txt) or 0)
    d["median_turn"] = _r(r"\*\*Median Turn\*\*.*?\|\s*([\d.]+)s", txt)

    return d


def _r(pat, txt, group=1, fl=0):
    m = re.search(pat, txt, fl)
    return m.group(group).strip() if m else ""


def _tv(field, txt, strip_bt=False):
    m = re.search(rf"\*\*{re.escape(field)}\*\*\s*\|\s*(.+)", txt)
    if not m:
        return ""
    v = m.group(1).strip()
    return v.strip("`") if strip_bt else v


def _parse_cat_var(txt):
    rows: list[dict[str, Any]] = []
    block = re.search(r"## Category Variance\n\n(.+?)(?=\n## |\Z)", txt, re.S)
    if not block:
        return rows
    lines = block.group(1).strip().split("\n")
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 3 or not cols[1] or cols[1].startswith("-"):
            continue
        if cols[1] == "Category":
            continue
        # Parse: | Name | T1 | T2 | ... | Mean/Variance |
        # The variance is the last column
        variance_str = cols[-1] if len(cols) > 2 else ""
        # Get mean from trial columns (columns 2 to second-to-last)
        trial_vals = []
        for c in cols[2:-1]:
            try:
                trial_vals.append(int(c.rstrip("%")))
            except (ValueError, IndexError):
                pass
        mean_pct = round(sum(trial_vals) / len(trial_vals)) if trial_vals else 0
        rows.append(
            {
                "name": cols[1],
                "mean": mean_pct,
                "variance": variance_str,
            }
        )
    return rows


def _parse_summary_scenarios(txt):
    rows: list[dict[str, Any]] = []
    block = re.search(r"## Per-Scenario Results\n\n(.+?)(?=\n## |\Z)", txt, re.S)
    if not block:
        return rows
    lines = block.group(1).strip().split("\n")
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 3 or not cols[1].startswith("TC-"):
            continue
        if all(c in ("---", "") for c in cols):
            continue
        # Trial results are cols 2..(n-2), last two are Pass@k and Pass^k
        trial_results = []
        for c in cols[2:-2]:
            trial_results.append(c)
        passk = cols[-2].strip() if len(cols) >= 3 else ""
        passk8 = cols[-1].strip() if len(cols) >= 3 else ""

        # Calculate pass/partial/fail counts
        passes = sum(1 for r in trial_results if r == "✅")
        partials = sum(1 for r in trial_results if r == "⚠️")
        fails = sum(1 for r in trial_results if r == "❌")

        rows.append(
            {
                "id": cols[1],
                "trials": trial_results,
                "passes": passes,
                "partials": partials,
                "fails": fails,
                "passk": "✓" in passk,
                "passk8": "✓" in passk8,
            }
        )
    return rows


def _parse_never_passes(txt):
    items: list[dict[str, Any]] = []
    block = re.search(r"### ❌ Never Passes.*?\n\n(.+?)(?=\n### |\n## |\Z)", txt, re.S)
    if not block:
        return items
    for m in re.finditer(
        r"\|\s*\*\*(TC-\d+)\*\*\s*\|\s*(.+?)\s*\|",
        block.group(1),
        re.S,
    ):
        issue = " ".join(m.group(2).split())
        items.append({"id": m.group(1), "issue": issue})
    return items


def _parse_flaky(txt):
    items = []
    block = re.search(r"### 🔀 Flaky.*?\n\n(.+?)(?=\n### |\n## |\Z)", txt, re.S)
    if block:
        for line in block.group(1).split("\n"):
            m = re.match(r"\|\s*\*\*(TC-\d+)\*\*\s*\|\s*(.+?)\s*\|", line)
            if m:
                items.append({"id": m.group(1), "results": m.group(2).strip()})
    return items


def _parse_consistent_partials(txt):
    items = []
    block = re.search(r"### ⚠️ Consistently Partial.*?\n\n(.+?)(?=\n### |\n## |\Z)", txt, re.S)
    if block:
        for line in block.group(1).split("\n"):
            m = re.match(r"\|\s*(TC-\d+)\s*\|\s*(.+?)\s*\|", line)
            if m:
                items.append({"id": m.group(1), "issue": m.group(2).strip()})
    return items


# ─── Helpers ────────────────────────────────────────────────────────────────


def short_label(name: str, api: str) -> tuple:
    combined = f"{name} {api}".lower()
    if "nvfp4" in combined or ("nvidia" in combined and "fp4" in combined):
        return "NVFP4", "NVIDIA FP4 optimized (vLLM)"
    if "gguf" in combined or "q8" in combined:
        return "GGUF Q8", "Q8_K_XL GGUF quantization"
    if "deepseek" in combined:
        short = name.replace("deepseek-ai/", "").replace("deepseek-v4-flash-dspark", "DeepSeek V4")
        return short, name
    short = api.split("/")[-1] if api else name
    return short, name


def dname(d: dict) -> str:
    return d["model_api"] or d["model_name"]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def sign(v: int) -> str:
    return f"{'+' if v >= 0 else ''}{v}"


def pct_cls(w, r):
    if w > r:
        return "font-semibold text-emerald-700"
    if w < r:
        return "text-rose-600"
    return ""


def diff_display(wp, rp):
    dv = round(wp - rp)
    if dv > 0:
        return f"+{dv}", "diff-positive"
    if dv < 0:
        return f"{dv}", "diff-negative"
    return "\u2014", "text-slate-500"


def turn_time_display(wmt, rmt, winner_raw, runner_raw):
    if wmt is not None and rmt is not None:
        delta = wmt - rmt
        cls = "diff-positive" if delta <= 0 else "diff-negative"
        return f"{wmt:.1f}s", f"{rmt:.1f}s", f"{delta:+.1f}s", cls
    return winner_raw or "\u2014", runner_raw or "\u2014", "\u2014", "text-slate-500"


def _pct_or_dash(value: str | None) -> str:
    if not value:
        return "\u2014"
    return f"{float(value):.1f}"


def _pp_or_dash(value: float | None) -> str:
    if value is None:
        return "\u2014"
    return f"{value:.1f}pp"


def _is_infrastructure_failure(d: dict) -> bool:
    """Detect runs where every scenario failed due to server/connection errors."""
    if d["mean_score"] != 0.0:
        return False
    issues = " ".join(s["issue"].lower() for s in d.get("never_passes", []))
    markers = (
        "server error",
        "connection attempts failed",
        "internal server error",
        "connection error",
        "timed out",
        "timeout",
    )
    return any(m in issues for m in markers)


def _infrastructure_summary(d: dict) -> str:
    issues = [s["issue"] for s in d.get("never_passes", [])]
    if not issues:
        return "All scenarios scored 0 — this run may be invalid."
    counts: dict[str, int] = {}
    for issue in issues:
        key = issue.split(" for url ")[0]
        counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
    parts = [f"{text} ({count} scenarios)" for text, count in top]
    return "; ".join(parts)


# ─── HTML Generation ────────────────────────────────────────────────────────

HTML_HEAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tool-Eval Bench Summary Comparison \u2022 {title}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Space+Grotesk:wght@500;600&amp;display=swap');
    :root{{--primary:#0f172a}}
    body{{font-family:'Inter',system-ui,sans-serif}}
    .font-display{{font-family:'Space Grotesk','Inter',system-ui,sans-serif;font-weight:600}}
    .score-card{{transition:transform .2s cubic-bezier(.4,0,.2,1)}}
    .metric-bar{{height:6px;background:linear-gradient(90deg,#64748b,#0ea47a);border-radius:9999px}}
    .comparison-table th{{font-weight:700;color:#1e2937}}
    .section-header{{font-size:1.05rem;letter-spacing:-.015em}}
    .verdict-win{{background:linear-gradient(135deg,#0ea47a,#15803d);color:#fff}}
    .model-label{{font-size:.75rem;font-weight:700;letter-spacing:.5px}}
    .stat-value{{font-feature-settings:"tnum"}}
    .light-card{{background:#fff;border:1px solid #94a3b8;box-shadow:0 2px 4px rgb(15 23 42 / .08)}}
    .diff-positive{{color:#15803d;font-weight:600}}
    .diff-negative{{color:#b91c1c;font-weight:600}}
    table{{border-collapse:separate;border-spacing:0}}
    .scenario-row:hover{{background-color:#f8fafc}}
    .winner-badge{{background:#15803d;color:#fff;font-size:.65rem;padding:1px 9px;border-radius:9999px;font-weight:700;letter-spacing:.5px}}
  </style>
</head>"""

HTML_FOOTER = """</body>\n</html>"""


def generate_html(da: dict, db: dict, out: str) -> None:
    w, r = (da, db) if da["mean_score"] >= db["mean_score"] else (db, da)
    wl_raw, wd = short_label(w["model_name"], w["model_api"])
    rl_raw, rd = short_label(r["model_name"], r["model_api"])
    wl = esc(wl_raw)
    rl = esc(rl_raw)
    wdn, rdn = dname(w), dname(r)

    dates = sorted(set(d["date_short"] for d in (da, db) if d["date_short"]))
    date_str = dates[0] if len(dates) == 1 else f"{dates[0]} \u2014 {dates[-1]}"
    vs = sorted(set(d["version"] for d in (da, db) if d["version"]))
    ver = " / ".join(vs) if len(vs) > 1 else (vs[0] if vs else "")

    diff_score = round(w["mean_score"] - r["mean_score"], 1)
    diff_pts = round(w["mean_points"] - r["mean_points"], 1)

    # Additional metrics
    w_quality = w.get("quality", 0)
    r_quality = r.get("quality", 0)
    w_resp = w.get("responsiveness", 0)
    r_resp = r.get("responsiveness", 0)
    w_deploy = w.get("deployability", 0)
    r_deploy = r.get("deployability", 0)
    qd = w_quality - r_quality
    dd = w_deploy - r_deploy
    rd_ = w_resp - r_resp

    try:
        wmt = float(w["median_turn"]) if w.get("median_turn") else None
        rmt = float(r["median_turn"]) if r.get("median_turn") else None
    except (ValueError, TypeError):
        wmt = rmt = None

    mt_g, mt_r, mt_d, mt_cls = turn_time_display(
        wmt, rmt, w.get("median_turn"), r.get("median_turn")
    )

    # Safety
    w_safe_max = max(w["safety_warnings"]) if w["safety_warnings"] else 0
    r_safe_max = max(r["safety_warnings"]) if r["safety_warnings"] else 0

    # Reliability
    def _to_float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    w_pass8 = _to_float(w.get("pass_8"))
    r_pass8 = _to_float(r.get("pass_8"))
    w_gap = _to_float(w.get("reliability_gap"))
    r_gap = _to_float(r.get("reliability_gap"))

    w_infra_fail = _is_infrastructure_failure(w)
    r_infra_fail = _is_infrastructure_failure(r)

    # Category rows
    r_cats = {c["name"]: c for c in r["categories"]}
    cat_rows = []
    for wc in w["categories"]:
        rp = r_cats.get(wc["name"], {})
        wp = wc["mean"]
        rpv = rp.get("mean", 0)
        dv, dc = diff_display(wp, rpv)
        cat_rows.append(
            {
                "name": wc["name"],
                "w": f"{int(wp)}%",
                "r": f"{int(rpv)}%",
                "w_v": wc.get("variance", ""),
                "r_v": rp.get("variance", ""),
                "w_b": wp > rpv,
                "r_b": wp < rpv,
                "diff": dv,
                "dc": dc,
            }
        )

    temp = da.get("temperature") or "?"
    think = da.get("thinking") or "?"

    lines: list[str] = []

    # ─── HEAD ───
    lines.append(HTML_HEAD.format(title=esc(wdn)))
    lines.append('<body class="bg-slate-50 text-slate-900">')
    lines.append('  <div class="max-w-[1080px] mx-auto px-6 py-8">')

    # ─── HEADER ───
    lines.append(f"""    <div class="flex items-center justify-between mb-8">
      <div>
        <div class="flex items-center gap-x-3">
          <div class="w-10 h-10 bg-slate-900 rounded-2xl flex items-center justify-center">
            <i class="fa-solid fa-chart-line text-white text-2xl"></i>
          </div>
          <div>
            <h1 class="font-display text-3xl font-semibold tracking-tighter">Tool-Eval Bench</h1>
            <p class="text-slate-600 text-sm -mt-0.5">Cross-trial summary comparison</p>
          </div>
        </div>
      </div>
      <div class="text-right text-sm">
        <div class="text-slate-600">Date of runs</div>
        <div class="font-semibold text-slate-800">{esc(date_str)}</div>
        {f'<div class="text-xs text-slate-400 mt-0.5">tool-eval-bench {esc(ver)}</div>' if ver else ""}
      </div>
    </div>""")

    # ─── MODEL CARDS ───
    w_rating_parts = w["rating"].split("(")
    w_rating_display = w_rating_parts[0].strip() if w_rating_parts else w["rating"]
    r_rating_parts = r["rating"].split("(")
    r_rating_display = r_rating_parts[0].strip() if r_rating_parts else r["rating"]

    lines.append(f"""    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
      <!-- Runner-up -->
      <div class="light-card rounded-3xl p-5 border border-slate-300">
        <div class="flex items-start justify-between">
          <div>
            <div class="model-label text-rose-700 tracking-widest">RUNNER-UP</div>
            <div class="font-semibold text-xl tracking-tight mt-1">{esc(rdn)}</div>
            <div class="text-xs text-slate-600 mt-0.5">{esc(rd)}</div>
          </div>
          <div class="text-right">
            <div class="text-4xl font-bold tabular-nums stat-value text-slate-800">{r["mean_score"]:.1f}</div>
            <div class="text-xs uppercase font-bold text-slate-500 -mt-1">\u00b1{r["std_score"]:.1f} mean</div>
          </div>
        </div>
        <div class="mt-4 text-xs flex items-center gap-x-2">
          <div class="px-2.5 py-0.5 bg-slate-200 rounded-full text-slate-700 font-semibold">{r["trials"]} trials</div>
          <div class="px-2.5 py-0.5 bg-rose-200 text-rose-800 rounded-full text-[10px] font-bold">{esc(r_rating_display)}</div>
        </div>
      </div>
      <!-- Winner -->
      <div class="light-card rounded-3xl p-5 border border-emerald-300 shadow-sm ring-1 ring-emerald-200">
        <div class="flex items-start justify-between">
          <div>
            <div class="flex items-center gap-x-2">
              <span class="model-label text-emerald-700 tracking-widest">WINNER</span>
              <span class="winner-badge"><i class="fa-solid fa-trophy mr-1"></i> BEST</span>
            </div>
            <div class="font-semibold text-xl tracking-tight mt-1 text-emerald-900">{esc(wdn)}</div>
            <div class="text-xs text-emerald-800 mt-0.5">{esc(wd)}</div>
          </div>
          <div class="text-right">
            <div class="text-4xl font-bold tabular-nums stat-value text-emerald-800">{w["mean_score"]:.1f}</div>
            <div class="text-xs uppercase font-bold text-emerald-700 -mt-1">\u00b1{w["std_score"]:.1f} mean</div>
          </div>
        </div>
        <div class="mt-4 text-xs flex items-center gap-x-2">
          <div class="px-2.5 py-0.5 bg-emerald-200 text-emerald-800 rounded-full font-semibold">{w["trials"]} trials</div>
          <div class="px-2.5 py-0.5 bg-emerald-300 text-emerald-900 rounded-full text-[10px] font-bold">{esc(w_rating_display)}</div>
        </div>
      </div>
    </div>""")

    # ─── VERDICT BANNER ───
    vp = [f"{sign(int(diff_score))} points mean"]
    if w_gap is not None and r_gap is not None:
        if w_gap < r_gap:
            vp.append("More reliable")
        elif r_gap < w_gap:
            vp.append("Lower variance")
    if mt_g and mt_r and mt_g != "\u2014" and mt_r != "\u2014":
        if float(mt_g.rstrip("s")) < float(mt_r.rstrip("s")):
            vp.append("Faster")
    if r_safe_max > w_safe_max:
        vp.append("Safer")
    vp_text = " \u2022 ".join(esc(v) for v in vp)

    if w_infra_fail or r_infra_fail:
        warn_models = []
        if w_infra_fail:
            warn_models.append(
                f"<strong>{esc(dname(w))}</strong>: {esc(_infrastructure_summary(w))}"
            )
        if r_infra_fail:
            warn_models.append(
                f"<strong>{esc(dname(r))}</strong>: {esc(_infrastructure_summary(r))}"
            )
        lines.append(f"""    <div class="mb-4 rounded-3xl bg-amber-100 border border-amber-300 text-amber-950 px-6 py-4 shadow-sm">
      <div class="flex items-start gap-3">
        <i class="fa-solid fa-triangle-exclamation text-amber-600 mt-0.5"></i>
        <div>
          <div class="font-semibold">Invalid or infrastructure-failed run detected</div>
          <div class="text-sm mt-1">{"<br>".join(warn_models)}</div>
          <div class="text-sm mt-2 text-amber-800">A 0.0 score here reflects server/connection failures, not model capability. Re-run the affected benchmark before drawing conclusions.</div>
        </div>
      </div>
    </div>""")

    lines.append(f"""    <div class="mb-8 rounded-3xl bg-emerald-700 text-white px-6 py-4 flex items-center gap-4 shadow-sm">
      <div class="flex-1">
        <div class="flex items-center gap-x-2">
          <i class="fa-solid fa-check-circle text-emerald-300"></i>
          <span class="font-semibold text-lg">Winner: <span class="font-display">{esc(wdn)}</span></span>
        </div>
        <div class="text-white/90 text-sm mt-0.5">{vp_text}</div>
      </div>
      <div class="text-right text-sm font-bold px-5 py-1 bg-white/30 rounded-2xl text-white">{sign(int(diff_score))} pts</div>
    </div>""")

    # ─── KEY METRICS ───
    lines.append('    <div class="mb-8">')
    lines.append(
        '      <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('        <i class="fa-solid fa-tachometer-alt text-slate-600"></i>')
    lines.append("        <span>Key Metrics</span>")
    lines.append("      </div>")
    lines.append('      <div class="light-card rounded-3xl overflow-hidden">')
    lines.append('        <table class="w-full text-sm">')
    lines.append("          <thead>")
    lines.append(f"""            <tr class="bg-slate-200 border-b-2 border-slate-400">
              <th class="text-left py-3 px-6 font-semibold w-1/3">Metric</th>
              <th class="text-center py-3 px-4 font-semibold text-emerald-700">{wl} (Winner)</th>
              <th class="text-center py-3 px-4 font-semibold">{rl}</th>
              <th class="text-center py-3 px-4 font-semibold w-20">\u0394</th>
            </tr>""")
    lines.append("          </thead>")
    lines.append('          <tbody class="divide-y divide-slate-200 text-sm">')

    def km(label, wv, rv, dv, dc="diff-positive", tc="", wx=""):
        tc_attr = f' class="{tc}"' if tc else ""
        w_class = wx + " " if wx else ""
        w_class += "font-extrabold text-emerald-800 tabular-nums"
        lines.append(f'''            <tr{tc_attr}>
              <td class="py-3 px-6 font-semibold">{esc(label)}</td>
              <td class="py-3 px-4 text-center"><span class="{w_class}">{wv}</span></td>
              <td class="py-3 px-4 text-center"><span class="font-extrabold tabular-nums text-slate-800">{rv}</span></td>
              <td class="py-3 px-4 text-center"><span class="{dc}">{esc(dv)}</span></td>
            </tr>''')

    km(
        "Mean Score",
        f"{w['mean_score']:.1f}",
        f"{r['mean_score']:.1f}",
        f"+{diff_score:.1f}",
        wx="text-lg",
    )
    std_delta = round(w["std_score"] - r["std_score"], 1)
    std_delta_text = f"{'-' if std_delta < 0 else '+'}{abs(std_delta)}"
    km(
        "Std Dev",
        f"\u00b1{w['std_score']:.1f}",
        f"\u00b1{r['std_score']:.1f}",
        std_delta_text,
        "diff-positive" if w["std_score"] <= r["std_score"] else "diff-negative",
    )
    km("Mean Points", f"{w['mean_points']:.1f}", f"{r['mean_points']:.1f}", f"+{diff_pts:.1f}")
    deploy_sign = sign(dd)
    deploy_cls = "diff-positive" if dd >= 0 else "diff-negative"
    km(
        "Deployability (\u03b1=0.7)",
        f"{w_deploy} / 100",
        f"{r_deploy} / 100",
        deploy_sign,
        deploy_cls,
    )
    km("Quality", f"{w_quality} / 100", f"{r_quality} / 100", sign(qd))
    km("Responsiveness", f"{w_resp} / 100", f"{r_resp} / 100", sign(rd_))
    km("Median Turn Time", mt_g, mt_r, mt_d, mt_cls, "bg-emerald-100")

    # Safety row
    w_safe_str = str(w_safe_max)
    r_safe_str = f"{r_safe_max} max" if r_safe_max > 0 else str(r_safe_max)
    sv = "Winner" if w_safe_max <= r_safe_max else rl
    wsc = "text-emerald-600" if w_safe_max == 0 else "text-rose-600"
    rsc = "text-rose-600" if r_safe_max > 0 else "text-emerald-600"
    lines.append(f"""            <tr>
              <td class="py-3 px-6 font-semibold">Safety Warnings (max)</td>
              <td class="py-3 px-4 text-center"><span class="font-semibold {wsc}">{w_safe_str}</span></td>
              <td class="py-3 px-4 text-center"><span class="font-semibold {rsc}">{r_safe_str}</span></td>
              <td class="py-3 px-4 text-center"><span class="text-emerald-600 font-medium">{sv}</span></td>
            </tr>""")

    # Reliability row
    if w_pass8 is not None and r_pass8 is not None:
        reliability_delta = w_pass8 - r_pass8
        if reliability_delta > 0:
            reliability_delta_text = f"+{reliability_delta:.1f}"
        elif reliability_delta < 0:
            reliability_delta_text = f"{reliability_delta:.1f}"
        else:
            reliability_delta_text = "\u2014"
        reliability_cls = pct_cls(w_pass8, r_pass8) if w_pass8 != r_pass8 else "text-slate-500"
        lines.append(f'''            <tr>
              <td class="py-3 px-6 font-semibold">Reliability (Pass\u2078)</td>
              <td class="py-3 px-4 text-center"><span class="font-semibold text-emerald-700">{w_pass8:.1f}%</span></td>
              <td class="py-3 px-4 text-center"><span class="font-semibold">{r_pass8:.1f}%</span></td>
              <td class="py-3 px-4 text-center"><span class="{reliability_cls}">{reliability_delta_text}</span></td>
            </tr>''')

    lines.append("          </tbody>")
    lines.append("        </table>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── RELIABILITY SECTION ───
    lines.append('    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">')

    # Reliability card
    lines.append("      <div>")
    lines.append(
        '        <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('          <i class="fa-solid fa-shield-alt text-slate-600"></i>')
    lines.append("          <span>Reliability &amp; Safety</span>")
    lines.append("        </div>")
    lines.append('        <div class="light-card rounded-3xl p-5 space-y-4">')
    lines.append(f"""          <div class="flex justify-between items-center">
            <div>
              <div class="text-sm font-medium">Reliability floor (Pass\u2078)</div>
            </div>
            <div class="text-right">
              <div class="font-semibold tabular-nums">{wl}: {_pct_or_dash(w.get("pass_8"))}%</div>
              <div class="text-xs text-slate-500">{rl}: {_pct_or_dash(r.get("pass_8"))}%</div>
            </div>
          </div>""")

    if w_gap is not None or r_gap is not None:
        lines.append(f"""          <div class="flex justify-between items-center">
            <div>
              <div class="text-sm font-medium">Reliability gap (Pass@\u2088 \u2212 Pass\u2078)</div>
            </div>
            <div class="text-right">
              <div class="font-semibold tabular-nums">{wl}: {_pp_or_dash(w_gap)}</div>
              <div class="text-xs text-slate-500">{rl}: {_pp_or_dash(r_gap)}</div>
            </div>
          </div>""")

    lines.append(f'''          <div>
            <div class="flex justify-between text-sm mb-1">
              <div class="font-medium">Safety-critical failures (max per trial)</div>
            </div>
            <div class="flex items-center gap-x-3">
              <div class="flex-1">
                <div class="{"text-emerald-700" if w_safe_max == 0 else "text-rose-600"} text-sm font-semibold">{wl}: <span class="font-normal">{w_safe_max}</span></div>
              </div>
              <div class="flex-1">
                <div class="{"text-rose-600" if r_safe_max > 0 else "text-emerald-700"} text-sm font-semibold">{rl}: <span class="font-normal">{r_safe_max}</span></div>
              </div>
            </div>
          </div>''')

    lines.append("        </div>")
    lines.append("      </div>")

    # Stability card
    lines.append("      <div>")
    lines.append(
        '        <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('          <i class="fa-solid fa-chart-bar text-slate-600"></i>')
    lines.append("          <span>Stability &amp; Consistency</span>")
    lines.append("        </div>")
    lines.append('        <div class="light-card rounded-3xl p-5 space-y-4">')

    w_var_count = sum(
        1
        for c in w["categories"]
        if "zero" not in c.get("variance", "").lower()
        and "variance" not in c.get("variance", "").lower()
    )
    r_var_count = sum(
        1
        for c in r["categories"]
        if "zero" not in c.get("variance", "").lower()
        and "variance" not in c.get("variance", "").lower()
    )

    lines.append(f"""          <div>
            <div class="text-sm font-medium mb-2">Categories with variance</div>
            <div class="flex items-center gap-x-3">
              <div class="flex-1">
                <div class="text-sm">{wl}: <strong>{w_var_count}</strong> / {len(w["categories"])}</div>
              </div>
              <div class="flex-1">
                <div class="text-sm">{rl}: <strong>{r_var_count}</strong> / {len(r["categories"])}</div>
              </div>
            </div>
          </div>""")

    lines.append("""          <div class="pt-1 border-t text-xs text-slate-600">
            Lower variance categories indicate more consistent performance across trials.
          </div>""")

    lines.append("        </div>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── CATEGORY SCORES ───
    lines.append('    <div class="mb-8">')
    lines.append(
        '      <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('        <i class="fa-solid fa-layer-group text-slate-600"></i>')
    lines.append("        <span>Category Mean Scores</span>")
    lines.append("      </div>")
    lines.append('      <div class="light-card rounded-3xl overflow-hidden">')
    lines.append('        <table class="w-full text-sm comparison-table">')
    lines.append("          <thead>")
    lines.append(f"""            <tr class="bg-slate-200 border-b-2 border-slate-400">
              <th class="text-left py-3 px-6">Category</th>
              <th class="px-3 py-3 text-center" style="width:13rem">{wl}</th>
              <th class="px-3 py-3 text-center" style="width:13rem">{rl}</th>
              <th class="px-3 py-3 text-center w-14">Diff</th>
            </tr>""")
    lines.append("          </thead>")
    lines.append('          <tbody class="divide-y divide-slate-200 text-[13.5px]">')

    for row in cat_rows:
        wc = pct_cls(int(row["w"].rstrip("%")), int(row["r"].rstrip("%")))
        rc = pct_cls(int(row["r"].rstrip("%")), int(row["w"].rstrip("%")))
        extra = ' class="bg-emerald-100"' if row["name"] == "Hard Mode" else ""
        wdisp = f'<span class="{wc}">{esc(row["w"])}</span>' if wc else esc(row["w"])
        rdisp = f'<span class="{rc}">{esc(row["r"])}</span>' if rc else esc(row["r"])
        lines.append(f'''            <tr{extra}>
              <td class="py-2.5 px-6 font-medium">{esc(row["name"])}</td>
              <td class="text-center">{wdisp}</td>
              <td class="text-center">{rdisp}</td>
              <td class="text-center"><span class="{row["dc"]}">{esc(row["diff"])}</span></td>
            </tr>''')

    lines.append("          </tbody>")
    lines.append("        </table>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── NEVER PASSES / FLAKY / PARTIALS ───
    lines.append('    <div class="mb-8">')
    lines.append(
        '      <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('        <i class="fa-solid fa-exclamation-triangle text-slate-600"></i>')
    lines.append("        <span>Scenario Reliability Analysis</span>")
    lines.append("      </div>")
    lines.append('      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">')

    # Winner failures
    w_np_list = w.get("never_passes", [])
    w_flaky_list = w.get("flaky", [])
    w_cp_list = w.get("consistent_partials", [])
    lines.append(f"""        <div class="light-card rounded-3xl p-5">
          <div class="uppercase text-xs font-semibold tracking-widest mb-2 text-emerald-700">{wl} ({w["mean_score"]:.0f} mean)</div>
          <ul class="text-sm space-y-[5px]">""")
    if w_np_list:
        lines.append(
            f"""            <li class="text-xs font-semibold text-rose-600 mt-2">Never passes ({len(w_np_list)}):</li>"""
        )
        for s in w_np_list[:4]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-rose-500 mt-px">\u2715</span> <span><strong>{esc(s["id"])}</strong> {esc(s["issue"][:80])}</span></li>"""
            )
    if w_flaky_list:
        lines.append(
            f"""            <li class="text-xs font-semibold text-amber-600 mt-2">Flaky ({len(w_flaky_list)}):</li>"""
        )
        for s in w_flaky_list[:4]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-amber-500 mt-px">\u21c4</span> <span><strong>{esc(s["id"])}</strong></span></li>"""
            )
    if w_cp_list:
        lines.append(
            f"""            <li class="text-xs font-semibold text-amber-600 mt-2">Consistent partials ({len(w_cp_list)}):</li>"""
        )
        for s in w_cp_list[:4]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-amber-500 mt-px">\u26a0</span> <span><strong>{esc(s["id"])}</strong> {esc(s["issue"][:80])}</span></li>"""
            )
    if not w_np_list and not w_flaky_list and not w_cp_list:
        lines.append(
            """            <li class="text-slate-500">No reliability issues detected.</li>"""
        )
    lines.append("          </ul>")
    lines.append("        </div>")

    # Runner failures
    r_np_list = r.get("never_passes", [])
    r_flaky_list = r.get("flaky", [])
    r_cp_list = r.get("consistent_partials", [])
    border_r = " border border-rose-300" if r_np_list else ""
    lines.append(f"""        <div class="light-card rounded-3xl p-5{border_r}">
          <div class="uppercase text-xs font-semibold tracking-widest mb-2 text-rose-700">{rl} ({r["mean_score"]:.0f} mean)</div>
          <ul class="text-sm space-y-[5px]">""")
    if r_np_list:
        lines.append(
            f"""            <li class="text-xs font-semibold text-rose-600 mt-2">Never passes ({len(r_np_list)}):</li>"""
        )
        for s in r_np_list[:5]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-rose-500 mt-px">\u2715</span> <span><strong>{esc(s["id"])}</strong> {esc(s["issue"][:80])}</span></li>"""
            )
    if r_flaky_list:
        lines.append(
            f"""            <li class="text-xs font-semibold text-amber-600 mt-2">Flaky ({len(r_flaky_list)}):</li>"""
        )
        for s in r_flaky_list[:4]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-amber-500 mt-px">\u21c4</span> <span><strong>{esc(s["id"])}</strong></span></li>"""
            )
    if r_cp_list:
        lines.append(
            f"""            <li class="text-xs font-semibold text-amber-600 mt-2">Consistent partials ({len(r_cp_list)}):</li>"""
        )
        for s in r_cp_list[:4]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-amber-500 mt-px">\u26a0</span> <span><strong>{esc(s["id"])}</strong> {esc(s["issue"][:80])}</span></li>"""
            )
    if not r_np_list and not r_flaky_list and not r_cp_list:
        lines.append(
            """            <li class="text-slate-500">No reliability issues detected.</li>"""
        )
    lines.append("          </ul>")
    lines.append("        </div>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── STRENGTHS & WEAKNESSES ───
    lines.append('    <div class="mb-8">')
    lines.append(
        '      <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('        <i class="fa-solid fa-balance-scale text-slate-600"></i>')
    lines.append("        <span>Winner vs. Runner-up: Strengths &amp; Weaknesses</span>")
    lines.append("      </div>")
    lines.append('      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">')

    w_str = [c for c in cat_rows if c["w_b"]]
    lines.append(f"""        <div class="light-card rounded-3xl p-5">
          <div class="flex items-center gap-x-2 mb-3">
            <i class="fa-solid fa-tachometer-alt text-emerald-600"></i>
            <span class="font-semibold text-emerald-800">{wl} Strengths</span>
          </div>
          <ul class="space-y-2 text-sm">""")
    for s in w_str[:5]:
        lines.append(
            f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span><strong>Superior {esc(s["name"].lower())}</strong> ({s["w"]} vs {s["r"]})</span></li>"""
        )
    if w_pass8 is not None and r_pass8 is not None and w_pass8 > r_pass8:
        lines.append(
            f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span><strong>Higher reliability floor</strong> ({w_pass8:.1f}% vs {r_pass8:.1f}%)</span></li>"""
        )
    if wmt is not None and rmt is not None and wmt < rmt:
        lines.append(
            f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span><strong>Faster inference</strong> ({wmt:.1f}s median vs {rmt:.1f}s)</span></li>"""
        )
    if w_safe_max == 0 and r_safe_max > 0:
        lines.append(
            """            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span><strong>Stronger safety</strong> \u2014 no safety-critical failures</span></li>"""
        )
    if not w_str and w_pass8 is None:
        lines.append(
            """            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span>On par or better across categories.</span></li>"""
        )
    lines.append("          </ul>")
    lines.append("        </div>")

    r_str = [c for c in cat_rows if c["r_b"]]
    lines.append(f"""        <div class="light-card rounded-3xl p-5">
          <div class="flex items-center gap-x-2 mb-3">
            <i class="fa-solid fa-exclamation text-amber-500"></i>
            <span class="font-semibold text-amber-700">{wl} Weaknesses vs {rl}</span>
          </div>
          <ul class="space-y-2 text-sm">""")
    for s in r_str[:5]:
        lines.append(
            f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-minus text-amber-600 mt-1 text-xs"></i> <span><strong>Lower {esc(s["name"].lower())}</strong> ({s["w"]} vs {s["r"]})</span></li>"""
        )
    if not r_str:
        lines.append(
            """            <li class="flex items-start gap-x-2"><i class="fa-solid fa-minus text-amber-600 mt-1 text-xs"></i> <span>No significant weaknesses \u2014 wins or ties in every category.</span></li>"""
        )
    if r_np_list and not w_np_list:
        lines.append(
            f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-minus text-amber-600 mt-1 text-xs"></i> <span>Has never-pass scenarios ({len(r_np_list)} vs 0).</span></li>"""
        )
    lines.append("          </ul>")
    lines.append("        </div>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── CONCLUSION ───
    if r_safe_max > 0:
        conc = f"The {rl} model was outmatched in safety and reliability testing."
    else:
        conc = f"The {rl} model showed competitive scores but lagged in consistency."

    lines.append(f"""    <div class="light-card rounded-3xl p-6">
      <div class="font-semibold text-lg mb-2">Conclusion</div>
      <div class="text-[15px] leading-relaxed text-slate-700">
        The <span class="font-semibold text-emerald-700">{esc(wdn)}</span> is the clear winner across {w["trials"]} trials.
        It delivers better mean scores with {("lower" if w["std_score"] < r["std_score"] else "comparable")} variance,
        {"higher" if w_pass8 is not None and r_pass8 is not None and w_pass8 > r_pass8 else "competitive"} reliability,
        and {"fewer" if w_safe_max < r_safe_max else "comparable"} safety concerns.
      </div>
      <div class="mt-4 text-sm text-slate-700">
        {esc(conc)}
      </div>
      <div class="text-xs mt-4 pt-3 border-t text-emerald-700 font-medium flex items-center gap-x-1.5">
        <i class="fa-solid fa-info-circle"></i>
        <span>Both models use the same backend configuration, temperature {esc(temp)}, and thinking {esc(think)}.</span>
      </div>
    </div>""")

    # ─── FOOTER ───
    lines.append(f"""    <div class="mt-8 text-center text-[10px] text-slate-500">
      Generated comparison \u2022 Light theme \u2022 Cross-trial summaries from tool-eval-bench runs {esc(date_str)}
    </div>""")

    lines.append("  </div>")
    lines.append("""  <script>
    function initializeTailwind() {
      document.documentElement.style.setProperty('--accent', '#0ea47a');
    }
    initializeTailwind();
  </script>""")
    lines.append(HTML_FOOTER)

    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary comparison HTML generated: {out}")


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_summary.py <summary_a.md> <summary_b.md> <output_html>")
        sys.exit(1)
    fa, fb, out = sys.argv[1], sys.argv[2], sys.argv[3]
    for p in (fa, fb):
        if not Path(p).exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)
    da = parse_summary(fa)
    db = parse_summary(fb)
    print(f"Summary A: {da['model_name']}  (mean: {da['mean_score']})")
    print(f"Summary B: {db['model_name']}  (mean: {db['mean_score']})")
    generate_html(da, db, out)


if __name__ == "__main__":
    main()
