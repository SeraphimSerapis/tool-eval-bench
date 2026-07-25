#!/usr/bin/env python3
"""Generate a head-to-head comparison HTML page from two tool-eval-bench markdown results.

Usage:
    python compare_tool_eval.py <model_a.md> <model_b.md> <output.html>
"""

import html
import re
import sys
from pathlib import Path
from typing import Any

# ─── Parser ─────────────────────────────────────────────────────────────────


def parse_md(fp: str) -> dict:
    txt = Path(fp).read_text(encoding="utf-8")
    d: dict = {}

    d["model_name"] = _r(r"^# Tool-Call Benchmark — (.+)$", txt, fl=re.M) or "Unknown"
    d["run_id"] = _r(r"\*\*Run ID\*\*:\s*`(.+?)`", txt)
    d["date"] = _r(r"\*\*Date\*\*:\s*`(.+?)`", txt)
    d["date_short"] = d["date"][:10] if d["date"] else ""
    d["tool_eval_version"] = _r(r"\*\*tool-eval-bench\*\*:\s*`(.+?)`", txt)
    d["final_score"] = int(_r(r"\*\*Final Score\*\*:\s*\*\*(\d+)\*\*", txt) or 0)
    pts = re.search(r"\*\*Total Points\*\*:\s*(\d+)\s*/\s*(\d+)", txt)
    d["total_points_earned"] = int(pts.group(1)) if pts else 0
    d["total_points_max"] = int(pts.group(2)) if pts else 0
    d["total_points_str"] = f"{d['total_points_earned']} / {d['total_points_max']}" if pts else ""
    d["rating"] = _r(r"\*\*Rating\*\*:\s*(.+?)$", txt, fl=re.M) or ""

    d["deployability"] = int(_r(r"\*\*Deployability\*\*:\s*\*\*(\d+)\*\*", txt) or 0)
    d["quality"] = int(_r(r"\*\*Quality\*\*:\s*(\d+)\s*/\s*100", txt) or 0)
    d["responsiveness"] = int(_r(r"\*\*Responsiveness\*\*:\s*(\d+)\s*/\s*100", txt) or 0)
    d["median_turn_time"] = _r(r"median turn:\s*([\d.]+)s", txt, fl=re.I) or ""

    d["backend"] = _tv("Backend", txt)
    d["model_api"] = _tv("Model (API)", txt, strip_bt=True)
    d["model_root"] = _tv("Model (Root)", txt, strip_bt=True)
    d["temperature"] = _tv("Temperature", txt)
    d["thinking"] = _tv("Thinking", txt)

    d["categories"] = _parse_cats(txt)
    d["scenarios"] = _parse_scenarios(txt)
    d["difficulties"] = _parse_diffs(txt)

    # Back-fill tier order from whichever summary has more data
    all_tiers = ["Trivial", "Easy", "Moderate", "Hard", "Very Hard"]
    d["tier_order"] = all_tiers

    d["safety_critical"] = []
    d["safety_critical_count"] = 0
    warn = re.search(r"> \[!WARNING\]\n(.*?)(?=\n## |\n\n(?!\s*>))", txt, re.S)
    if warn:
        wt = warn.group(0)
        cm = re.search(r"\*\*(\d+) safety-critical", wt)
        if cm:
            d["safety_critical_count"] = int(cm.group(1))
        for sm in re.finditer(
            r"> - \*\*(TC-\d+)\*\*\s*\((.+?)\):\s*(.*?)(?=\n> - \*\*TC-|\n## |\Z)", wt, re.S
        ):
            d["safety_critical"].append(
                {
                    "id": sm.group(1),
                    "type": sm.group(2),
                    "desc": sm.group(3).strip(),
                }
            )
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


def _parse_cats(txt):
    rows: list[dict[str, Any]] = []
    block = re.search(r"## Category Scores\n\n(.+?)(?=\n## |\Z)", txt, re.S)
    if not block:
        return rows
    for line in block.group(1).strip().split("\n"):
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 5 or not cols[1] or cols[1].startswith("-") or cols[1] == "Category":
            continue
        rows.append(
            {
                "name": cols[1],
                "earned": cols[2],
                "max": cols[3],
                "percent": float(cols[4].rstrip("%")),
            }
        )
    return rows


def _parse_scenarios(txt):
    rows: list[dict[str, Any]] = []
    block = re.search(r"## Scenario Results\n\n(.+?)(?=\n## |\Z)", txt, re.S)
    if not block:
        return rows
    for line in block.group(1).split("\n"):
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 2 or not cols[1].startswith("TC-"):
            continue
        if all(c in ("---", "") for c in cols):
            continue
        sr = cols[4] if len(cols) > 4 else ""
        if "pass" in sr or "✅" in sr:
            status = "pass"
        elif "partial" in sr or "⚠" in sr:
            status = "partial"
        else:
            status = "fail"
        pm = re.search(r"(\d+)/(\d+)", cols[5] if len(cols) > 5 else "")
        rows.append(
            {
                "id": cols[1],
                "title": cols[2],
                "diff": cols[3].count("★") if len(cols) > 3 else 0,
                "status": status,
                "points_earned": int(pm.group(1)) if pm else 0,
                "points_max": int(pm.group(2)) if pm else 0,
                "summary": cols[6] if len(cols) > 6 else "",
            }
        )
    return rows


def _parse_diffs(txt):
    rows: list[dict[str, Any]] = []
    block = re.search(r"## Performance by Difficulty\n\n(.+?)(?=\n## |\Z)", txt, re.S)
    if not block:
        return rows
    for line in block.group(1).split("\n"):
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 5 and cols[1] in ("Trivial", "Easy", "Moderate", "Hard", "Very Hard"):
            try:
                rate = int(float(cols[4].rstrip("%")))
            except (ValueError, TypeError):
                rate = 0
            rows.append(
                {
                    "tier": cols[1],
                    "scenarios": cols[2],
                    "passed": cols[3],
                    "rate": rate,
                }
            )
    return rows


# ─── Helpers ────────────────────────────────────────────────────────────────


def short_label(name: str, api: str) -> tuple:
    combined = f"{name} {api}".lower()
    if "nvfp4" in combined or ("nvidia" in combined and "fp4" in combined):
        return "NVFP4", "NVIDIA FP4 optimized (vLLM)"
    if "gguf" in combined or "q8" in combined:
        return "GGUF Q8", "Q8_K_XL GGUF quantization"
    short = api.split("/")[-1] if api else name
    for suf in ("-UD-Q8_K_XL.gguf", "-Q8_K_XL.gguf", ".gguf"):
        if short.endswith(suf):
            short = short[: -len(suf)]
            return short, "GGUF quantized"
    return short, short


def dname(d: dict) -> str:
    return d["model_api"] or d["model_name"]


def esc(s: str) -> str:
    """Escape a value for HTML text or double-quoted attribute context.

    Every string that originates in a parsed Markdown report must pass through
    here: the reports are shared between people, so an attacker-authored report
    must not be able to inject markup into the comparison page.
    """
    return html.escape(str(s), quote=True)


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


# ─── HTML Generation ────────────────────────────────────────────────────────

HTML_HEAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tool-Eval Bench Comparison \u2022 {title}</title>
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
    # Determine winner and runner-up
    w, r = (da, db) if da["final_score"] >= db["final_score"] else (db, da)
    wl_raw, wd = short_label(w["model_name"], w["model_api"])
    rl_raw, rd = short_label(r["model_name"], r["model_api"])
    wl = esc(wl_raw)
    rl = esc(rl_raw)
    wdn, rdn = dname(w), dname(r)

    dates = sorted(set(d["date_short"] for d in (da, db) if d["date_short"]))
    if not dates:
        date_str = ""
    elif len(dates) == 1:
        date_str = dates[0]
    else:
        date_str = f"{dates[0]} \u2014 {dates[-1]}"
    vs = sorted(set(d["tool_eval_version"] for d in (da, db) if d["tool_eval_version"]))
    ver = " / ".join(vs) if len(vs) > 1 else (vs[0] if vs else "")

    sd = w["final_score"] - r["final_score"]
    pd_ = w["total_points_earned"] - r["total_points_earned"]
    qd = w["quality"] - r["quality"]
    dd = w["deployability"] - r["deployability"]

    try:
        wmt = float(w["median_turn_time"]) if w["median_turn_time"] else None
        rmt = float(r["median_turn_time"]) if r["median_turn_time"] else None
    except (ValueError, TypeError):
        wmt = rmt = None

    mt_g, mt_r, mt_d, mt_cls = turn_time_display(
        wmt, rmt, w["median_turn_time"], r["median_turn_time"]
    )

    # Category rows
    r_cats = {c["name"]: c for c in r["categories"]}
    cat_rows = []
    for wc in w["categories"]:
        rc = r_cats.get(wc["name"], {})
        wp = wc["percent"]
        rp = rc.get("percent", 0)
        dv, dc = diff_display(wp, rp)
        cat_rows.append(
            {
                "name": wc["name"],
                "w": f"{int(wp)}%",
                "r": f"{int(rp)}%",
                "w_b": wp > rp,
                "r_b": wp < rp,
                "diff": dv,
                "dc": dc,
            }
        )

    # Difficulty rows
    w_di = {d["tier"]: d for d in w["difficulties"]}
    r_di = {d["tier"]: d for d in r["difficulties"]}
    diff_rows = []
    tier_order = w.get("tier_order") or r.get("tier_order") or []
    for tier in tier_order:
        wd2 = w_di.get(tier, {})
        rd3 = r_di.get(tier, {})
        wr = wd2.get("rate", 0) if wd2 else 0
        rr = rd3.get("rate", 0) if rd3 else 0
        diff_rows.append(
            {
                "tier": tier,
                "w_pct": wr,
                "r_pct": rr,
                "w_disp": (
                    f"{wd2.get('passed', '?')} / {wd2.get('scenarios', '?')}" if wd2 else "\u2014"
                ),
                "r_disp": (
                    f"{rd3.get('passed', '?')} / {rd3.get('scenarios', '?')}" if rd3 else "\u2014"
                ),
            }
        )

    # Failures
    w_fails = [s for s in w["scenarios"] if s["status"] == "fail"]
    r_fails = [s for s in r["scenarios"] if s["status"] == "fail"]
    w_parts = [s for s in w["scenarios"] if s["status"] == "partial"]
    r_parts = [s for s in r["scenarios"] if s["status"] == "partial"]
    wc_cnt, rc_cnt = w["safety_critical_count"], r["safety_critical_count"]
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
            <p class="text-slate-600 text-sm -mt-0.5">Head-to-head comparison</p>
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
            <div class="text-4xl font-bold tabular-nums stat-value text-slate-800">{r["final_score"]}</div>
            <div class="text-xs uppercase font-bold text-slate-500 -mt-1">/ 100</div>
          </div>
        </div>
        <div class="mt-4 text-xs flex items-center gap-x-2">
          <div class="px-2.5 py-0.5 bg-slate-200 rounded-full text-slate-700 font-semibold">{esc(r["total_points_str"])}</div>
          <div class="px-2.5 py-0.5 bg-rose-200 text-rose-800 rounded-full text-[10px] font-bold">{esc(r["rating"])}</div>
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
            <div class="text-4xl font-bold tabular-nums stat-value text-emerald-800">{w["final_score"]}</div>
            <div class="text-xs uppercase font-bold text-emerald-700 -mt-1">/ 100</div>
          </div>
        </div>
        <div class="mt-4 text-xs flex items-center gap-x-2">
          <div class="px-2.5 py-0.5 bg-emerald-200 text-emerald-800 rounded-full font-semibold">{esc(w["total_points_str"])}</div>
          <div class="px-2.5 py-0.5 bg-emerald-300 text-emerald-900 rounded-full text-[10px] font-bold">{esc(w["rating"])}</div>
        </div>
      </div>
    </div>""")

    # ─── VERDICT BANNER ───
    vp = [f"+{sd} points overall"]
    hw = next((d for d in diff_rows if d["tier"] == "Hard"), {})
    hw_p = hw.get("w_pct", 0)
    hr_p = hw.get("r_pct", 0)
    if hw_p > hr_p:
        vp.insert(1, "Stronger hard-mode performance")
    if wmt is not None and rmt is not None and wmt < rmt:
        vp.append("Faster")
    if rc_cnt > 0 and wc_cnt == 0:
        vp.append("Safer")
    vp_text = " \u2022 ".join(esc(v) for v in vp)

    lines.append(f"""    <div class="mb-8 rounded-3xl bg-emerald-700 text-white px-6 py-4 flex items-center gap-4 shadow-sm">
      <div class="flex-1">
        <div class="flex items-center gap-x-2">
          <i class="fa-solid fa-check-circle text-emerald-300"></i>
          <span class="font-semibold text-lg">Winner: <span class="font-display">{esc(wdn)}</span></span>
        </div>
        <div class="text-white/90 text-sm mt-0.5">{vp_text}</div>
      </div>
      <div class="text-right text-sm font-bold px-5 py-1 bg-white/30 rounded-2xl text-white">+{sd} pts</div>
    </div>""")

    # ─── QUALITY-ONLY NOTE ───
    qword = "point" if abs(qd) == 1 else "points"
    lines.append(f"""    <div class="mb-8 -mt-4 text-sm text-slate-500 flex items-center justify-end gap-x-1.5">
      <i class="fa-solid fa-info-circle text-slate-400"></i>
      <span>Quality-only (excluding speed): {wl} <strong>{w["quality"]}</strong> vs {rl} <strong>{r["quality"]}</strong> \u2014 {sign(qd)} {qword} lead</span>
    </div>""")

    # ─── KEY METRICS TABLE ───
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

    km("Final Score", str(w["final_score"]), str(r["final_score"]), f"+{sd}", wx="text-lg")
    km("Total Points", w["total_points_str"], r["total_points_str"], f"+{pd_}")
    kms = sign(dd)
    kmc = "diff-positive" if dd >= 0 else "diff-negative"
    km(
        "Deployability (\u03b1=0.7)",
        f"{w['deployability']} / 100",
        f"{r['deployability']} / 100",
        kms,
        kmc,
    )
    km("Quality", f"{w['quality']} / 100", f"{r['quality']} / 100", sign(qd))
    rd_dv, rd_dc = diff_display(w["responsiveness"], r["responsiveness"])
    km(
        "Responsiveness",
        f"{w['responsiveness']} / 100",
        f"{r['responsiveness']} / 100",
        rd_dv,
        rd_dc,
    )
    km("Median Turn Time", mt_g, mt_r, mt_d, mt_cls, "bg-emerald-100")

    # Safety row
    ws = str(wc_cnt)
    rs = f"{rc_cnt} critical" if rc_cnt > 0 else str(rc_cnt)
    sv = "Winner" if wc_cnt <= rc_cnt else rl
    wsc = "text-emerald-600" if wc_cnt == 0 else "text-rose-600"
    rsc = "text-rose-600" if rc_cnt > 0 else "text-emerald-600"
    lines.append(f"""            <tr>
              <td class="py-3 px-6 font-semibold">Safety Warnings</td>
              <td class="py-3 px-4 text-center"><span class="font-semibold {wsc}">{ws}</span></td>
              <td class="py-3 px-4 text-center"><span class="font-semibold {rsc}">{rs}</span></td>
              <td class="py-3 px-4 text-center"><span class="text-emerald-600 font-medium">{sv}</span></td>
            </tr>""")

    lines.append("          </tbody>")
    lines.append("        </table>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── CATEGORY SCORES ───
    lines.append('    <div class="mb-8">')
    lines.append(
        '      <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('        <i class="fa-solid fa-layer-group text-slate-600"></i>')
    lines.append("        <span>Category Scores</span>")
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
        if row["name"] == "Hard Mode" and row["w_b"]:
            hd_extra = " font-extrabold text-base"
        else:
            hd_extra = ""
        wdisp = f'<span class="{wc}{hd_extra}">{esc(row["w"])}</span>' if wc else esc(row["w"])
        rdisp = f'<span class="{rc}">{esc(row["r"])}</span>' if rc else esc(row["r"])
        if row["name"] == "Hard Mode" and not wc:
            wdisp = esc(row["w"])
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

    # ─── DIFFICULTY + RELIABILITY GRID ───
    lines.append('    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">')

    # Difficulty table
    lines.append("      <div>")
    lines.append(
        '        <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('          <i class="fa-solid fa-chart-bar text-slate-600"></i>')
    lines.append("          <span>Performance by Difficulty</span>")
    lines.append("        </div>")
    lines.append('        <div class="light-card rounded-3xl p-1">')
    lines.append('          <table class="w-full text-sm">')
    lines.append("            <thead>")
    lines.append(f"""              <tr class="text-slate-600 text-xs font-semibold">
                <th class="py-2 px-4 text-left">Tier</th>
                <th class="py-2 px-2 text-center">{wl}</th>
                <th class="py-2 px-2 text-center">{rl}</th>
              </tr>""")
    lines.append("            </thead>")
    lines.append('            <tbody class="text-[13px]">')

    for dr in diff_rows:
        bg = ' class="border-t bg-emerald-100"' if dr["tier"] == "Hard" else ' class="border-t"'
        wh = " font-semibold" if dr["tier"] == "Hard" else ""
        if dr["w_pct"] > dr["r_pct"]:
            wcc = " text-emerald-700 font-medium"
        elif dr["w_pct"] < dr["r_pct"]:
            wcc = " text-rose-600"
        else:
            wcc = ""
        if dr["r_pct"] > dr["w_pct"]:
            rcc = " text-emerald-700 font-medium"
        elif dr["r_pct"] < dr["w_pct"]:
            rcc = " text-rose-600"
        else:
            rcc = ""
        if dr["tier"] == "Hard":
            if "emerald" in wcc:
                wcc = " font-bold text-emerald-700"
            if "rose" in rcc:
                rcc = " font-bold text-rose-600"

        lines.append(f"""            <tr{bg}>
              <td class="py-2 px-4 font-medium{wh}">{esc(dr["tier"])}</td>
              <td class="py-2 px-2 text-center{wcc}">{dr["w_pct"]}% ({esc(dr["w_disp"])})</td>
              <td class="py-2 px-2 text-center{rcc}">{dr["r_pct"]}% ({esc(dr["r_disp"])})</td>
            </tr>""")

    lines.append("            </tbody>")
    lines.append("          </table>")
    lines.append("        </div>")
    lines.append("      </div>")

    # Reliability & Safety
    lines.append("      <div>")
    lines.append(
        '        <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('          <i class="fa-solid fa-shield-alt text-slate-600"></i>')
    lines.append("          <span>Reliability &amp; Safety</span>")
    lines.append("        </div>")
    lines.append('        <div class="light-card rounded-3xl p-5 space-y-4">')
    wcs = "text-emerald-700" if wc_cnt == 0 else "text-rose-600"
    rcs = "text-rose-600" if rc_cnt > 0 else "text-emerald-700"
    r_detail = f" ({rc_cnt} {'warning' if rc_cnt == 1 else 'warnings'})" if rc_cnt > 0 else " (0)"
    lines.append(f'''          <div>
            <div class="flex justify-between text-sm mb-1">
              <div class="font-medium">Safety-critical failures</div>
            </div>
            <div class="flex items-center gap-x-3">
              <div class="flex-1">
                <div class="{wcs} text-sm font-semibold">{wl}: <span class="font-normal">{wc_cnt}</span></div>
              </div>
              <div class="flex-1">
                <div class="{rcs} text-sm font-semibold">{rl}: <span class="font-normal">{rc_cnt}{esc(r_detail)}</span></div>
              </div>
            </div>
          </div>''')

    all_safe = r.get("safety_critical", []) + w.get("safety_critical", [])
    if all_safe:
        items = [f"{sc['id']} ({sc['type']})" for sc in all_safe]
        lines.append(f"""          <div class="pt-1 border-t text-xs text-slate-600">
            {"; ".join(esc(i) for i in items)}
          </div>""")
    elif rc_cnt > 0:
        r_crit_ids = [
            s["id"] for s in r_fails if s["id"] in [x["id"] for x in r.get("safety_critical", [])]
        ]
        if r_crit_ids:
            lines.append(f"""          <div class="pt-1 border-t text-xs text-slate-600">
            {rl} failed in safety-critical scenarios: {", ".join(esc(i) for i in r_crit_ids)}.
          </div>""")

    lines.append("        </div>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── NOTABLE SCENARIO OUTCOMES ───
    lines.append('    <div class="mb-8">')
    lines.append(
        '      <div class="section-header font-semibold mb-3 px-1 flex items-center gap-x-2 text-slate-800">'
    )
    lines.append('        <i class="fa-solid fa-exclamation-triangle text-slate-600"></i>')
    lines.append("        <span>Notable Scenario Outcomes</span>")
    lines.append("      </div>")
    lines.append('      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">')

    # Winner failures
    wlabel2 = "Consistent Issues" if w_fails else ("Minor Gaps" if w_parts else "Clean Run")
    lines.append(f"""        <div class="light-card rounded-3xl p-5">
          <div class="uppercase text-xs font-semibold tracking-widest mb-2 text-emerald-700">{wl} ({w["final_score"]}) \u2014 {wlabel2}</div>
          <ul class="text-sm space-y-[5px]">""")
    if w_fails:
        for f in w_fails[:6]:
            lines.append(
                f"""            <li class="flex gap-x-2"><span class="text-rose-500 mt-px">\u2715</span> <span><strong>{esc(f["id"])}</strong> {esc(f["summary"])}</span></li>"""
            )
    elif not w_parts:
        lines.append("""            <li class="text-slate-500">No failures detected.</li>""")
    if w_parts:
        pid = ", ".join(esc(p["id"]) for p in w_parts[:5])
        como = f" ({len(w_parts)} total)" if len(w_parts) > 5 else ""
        lines.append(
            f"""            <li class="text-xs text-slate-600 mt-2 pl-5">Partials on {pid}{como}</li>"""
        )
    lines.append("          </ul>")
    lines.append("        </div>")

    # Runner failures
    rlabel2 = (
        "Critical Weaknesses" if (r_fails or rc_cnt) else ("Minor Gaps" if r_parts else "Clean Run")
    )
    border_r = " border border-rose-300" if (r_fails or rc_cnt) else ""
    lines.append(f"""        <div class="light-card rounded-3xl p-5{border_r}">
          <div class="uppercase text-xs font-semibold tracking-widest mb-2 text-rose-700">{rl} ({r["final_score"]}) \u2014 {rlabel2}</div>
          <ul class="text-sm space-y-[5px]">""")
    safe_ids = set(s["id"] for s in r.get("safety_critical", []))
    if rc_cnt > 0:
        for f in r_fails:
            if f["id"] in safe_ids:
                lines.append(
                    f"""            <li class="flex gap-x-2"><span class="text-rose-500 mt-px">\u2715</span> <span><strong>{esc(f["id"])}</strong> {esc(f["summary"])} <span class="font-semibold text-rose-600">(safety-critical)</span></span></li>"""
                )
    other_fails = [f for f in r_fails if f["id"] not in safe_ids]
    for f in other_fails[:4]:
        lines.append(
            f"""            <li class="flex gap-x-2"><span class="text-rose-500 mt-px">\u2715</span> <span><strong>{esc(f["id"])}</strong> {esc(f["summary"])}</span></li>"""
        )
    if r_parts:
        pid = ", ".join(esc(p["id"]) for p in r_parts[:4])
        lines.append(
            f"""            <li class="text-xs text-slate-600 mt-2 pl-5">Partials on {pid}</li>"""
        )
    if not r_fails and not r_parts:
        lines.append("""            <li class="text-slate-500">No failures detected.</li>""")
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

    # Winner strengths
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
    if wmt is not None and rmt is not None and wmt < rmt:
        lines.append(
            f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span><strong>Better responsiveness</strong> \u2014 faster inference ({wmt:.1f}s median vs {rmt:.1f}s)</span></li>"""
        )
    if wc_cnt == 0 and rc_cnt > 0:
        lines.append(
            """            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span><strong>Stronger safety</strong> \u2014 no safety-critical failures</span></li>"""
        )
    if not w_str and wmt is None and wc_cnt == 0:
        lines.append(
            """            <li class="flex items-start gap-x-2"><i class="fa-solid fa-check text-emerald-600 mt-1 text-xs"></i> <span>No clear weaknesses \u2014 on par or better across categories.</span></li>"""
        )
    lines.append("          </ul>")
    lines.append("        </div>")

    # Winner weaknesses vs runner
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
            """            <li class="flex items-start gap-x-2"><i class="fa-solid fa-minus text-amber-600 mt-1 text-xs"></i> <span>No significant weaknesses identified \u2014 wins or ties in every category.</span></li>"""
        )
    if r_fails:
        if len(r_fails) > len(w_fails):
            lines.append(
                f"""            <li class="flex items-start gap-x-2"><i class="fa-solid fa-minus text-amber-600 mt-1 text-xs"></i> <span>More outright failures than runner-up ({len(r_fails)} vs {len(w_fails)}).</span></li>"""
            )
    lines.append("          </ul>")
    lines.append("        </div>")
    lines.append("      </div>")
    lines.append("    </div>")

    # ─── CONCLUSION ───
    faster_word = "markedly" if (wmt is not None and rmt is not None and wmt < rmt) else "also"
    if rc_cnt > 0:
        conc = f"The {rl} model was outmatched in hard-mode and safety testing."
    else:
        conc = f"The {rl} model showed solid performance across basic categories but lagged in complex multi-step scenarios."

    lines.append(f"""    <div class="light-card rounded-3xl p-6">
      <div class="font-semibold text-lg mb-2">Conclusion</div>
      <div class="text-[15px] leading-relaxed text-slate-700">
        The <span class="font-semibold text-emerald-700">{esc(wdn)}</span> is the clear winner.
        It delivers better performance on complex tasks (especially Hard and Very Hard scenarios),
        shows strong safety posture, and is {esc(faster_word)} faster in interactive use.
      </div>
      <div class="mt-4 text-sm text-slate-700">
        {esc(conc)} The {wl}-optimized model appears better suited for production tool-use workloads.
      </div>
      <div class="text-xs mt-4 pt-3 border-t text-emerald-700 font-medium flex items-center gap-x-1.5">
        <i class="fa-solid fa-info-circle"></i>
        <span>Both models use the same backend configuration, temperature {esc(temp)}, and thinking {esc(think)}.</span>
      </div>
    </div>""")

    # ─── FOOTER ───
    lines.append(f"""    <div class="mt-8 text-center text-[10px] text-slate-500">
      Generated comparison \u2022 Light theme \u2022 Data from tool-eval-bench runs {esc(date_str)}
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
    print(f"Comparison HTML generated: {out}")


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_tool_eval.py <md_file_a> <md_file_b> <output_html>")
        sys.exit(1)
    fa, fb, out = sys.argv[1], sys.argv[2], sys.argv[3]
    for p in (fa, fb):
        if not Path(p).exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)
    da = parse_md(fa)
    db = parse_md(fb)
    print(f"Model A: {da['model_name']}  (score: {da['final_score']})")
    print(f"Model B: {db['model_name']}  (score: {db['final_score']})")
    generate_html(da, db, out)


if __name__ == "__main__":
    main()
