"""Build the short summary page (index.html) of the token-drop Pareto artifact, plus a compacted copy of the full
generated page (details.html).

The full page (setA_pareto_page.html, rebuilt by make_pareto_section / make_pertask_section / make_inflight_section)
stays the source of every table; this script only adds a one-screen front page on top of it:

  * one compact chart pair from results/pareto_points.json: 4-task mean / worst-cell delta vs dense at matched
    training FLOPs, for a handful of key recipes (seed copies averaged);
  * curated rows for results that live in other sessions' records (11-task floor, warmup-then-freeze, 64k, CPT),
    each naming its source.

usage: python3 make_summary_page.py --points results/pareto_points.json --page <setA_pareto_page.html> --out-dir <dir>
"""
import argparse
import html
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pareto_section import build  # noqa: E402

# (arm, label, colour group) -- the recipes the front-page chart tracks
KEY = [
    ("otr10M7", "router, keep 10% after block 7", "router"),
    ("otr10M8", "router, keep 10% after block 8", "router"),
    ("otr30M8", "router, keep 30% after block 8", "router"),
    ("otr10M7f16", "router + 16-token floor per doc", "router"),
    ("otr10M7de0", "router, no full-context steps", "aggr"),
    ("fa30M7", "router + freeze early blocks after 25%", "aggr"),
    ("frz0M7", "router + early blocks frozen from step 0", "aggr"),
    ("rnd10M8", "random 10% after block 8", "random"),
    ("tier67a50m64", "best fixed keep rule", "rule"),
]
COL = {"router": "#7b4fd6", "aggr": "#db2777", "random": "#8a8f8c", "rule": "#0f8b8d"}


def key_points(d):
    groups = {}
    for r in build(d, "mt4"):
        if r["xkey"] != "pf" or not r.get("complete") or "mean_delta" not in r:
            continue
        for arm, label, grp in KEY:
            if r["arm"] == arm:
                groups.setdefault((arm, r["rows"]), []).append(r)
    out = []
    for (arm, rows), rs in groups.items():
        label, grp = next((l, g) for a, l, g in KEY if a == arm)
        out.append(dict(arm=arm, label=label, grp=grp, rows=rows, n=len(rs),
                        pf=sum(r["x"] for r in rs) / len(rs),
                        md=sum(r["mean_delta"] for r in rs) / len(rs),
                        wd=sum(r["worst_delta"] for r in rs) / len(rs)))
    return sorted(out, key=lambda p: p["pf"])


def chart(pts, field, title, ylo, yhi):
    W, H, L, R, T, B = 420, 230, 40, 10, 10, 30
    xlo, xhi = math.log(60), math.log(800)
    xs = lambda v: L + (math.log(v) - xlo) / (xhi - xlo) * (W - L - R)
    ys = lambda v: T + (yhi - max(ylo, min(yhi, v))) / (yhi - ylo) * (H - T - B)
    g = [f'<rect x="{L}" y="{ys(0.03):.1f}" width="{W - L - R}" height="{ys(-0.03) - ys(0.03):.1f}" class="band"/>']
    y = ylo
    while y <= yhi + 1e-9:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(y):.1f}" y2="{ys(y):.1f}" class="{"zero" if abs(y) < 1e-9 else "grid"}"/>'
                 f'<text x="{L - 4}" y="{ys(y) + 3:.1f}" text-anchor="end">{y:+.2f}</text>')
        y = round(y + 0.05, 4)
    for v in (100, 200, 400, 800):
        g.append(f'<text x="{xs(v):.1f}" y="{H - 14}" text-anchor="middle">{v}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{H - 2}" text-anchor="middle">training PF (log)</text>')
    for p in pts:
        v = p[field]
        tip = (f'{p["label"]}, {p["rows"]} rows/task, {p["pf"]:.0f} PF: mean Δ {p["md"]:+.3f}, worst Δ {p["wd"]:+.3f}'
               + (f' ({p["n"]} runs averaged)' if p["n"] > 1 else ""))
        clip = "▼" if v < ylo else ""
        g.append(f'<circle cx="{xs(p["pf"]):.1f}" cy="{ys(v):.1f}" r="4" fill="{COL[p["grp"]]}" opacity=".9">'
                 f'<title>{html.escape(tip)}</title></circle>'
                 + (f'<text x="{xs(p["pf"]):.1f}" y="{ys(v) - 6:.1f}" text-anchor="middle">{clip}</text>' if clip else ""))
    return (f'<figure class="fig"><figcaption>{title}</figcaption><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{html.escape(title)}">{"".join(g)}</svg></figure>')


def best_table(pts):
    pick = {}
    for p in pts:  # most-trained point per recipe
        if p["arm"] not in pick or p["rows"] > pick[p["arm"]]["rows"]:
            pick[p["arm"]] = p
    rows = []
    for arm, label, grp in KEY:
        if arm in pick:
            p = pick[arm]
            cls = lambda v: "pos" if v >= 0.02 else ("neg" if v <= -0.03 else "tie")
            rows.append(f'<tr><td><span class="dot" style="background:{COL[grp]}"></span>{html.escape(label)}</td>'
                        f'<td class="n">{p["rows"]}</td><td class="n">{p["pf"]:.0f}</td>'
                        f'<td class="n {cls(p["md"])}">{p["md"]:+.3f}</td><td class="n {cls(p["wd"])}">{p["wd"]:+.3f}</td>'
                        f'<td class="n">{p["n"]}</td></tr>')
    return "\n".join(rows)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Drop Pareto</title>
<style>
:root { --bg:#fbfbf9; --surface:#fff; --fg:#1c1d1f; --muted:#62666d; --rule:#e2e2dc; --pos:#127a3e; --neg:#b42318; --band:#eef0ea;
  --body: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; --mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141517; --surface:#1c1e21; --fg:#e7e8ea; --muted:#9a9ea6;
  --rule:#2e3136; --pos:#4cc38a; --neg:#f07a6e; --band:#23262a; } }
:root[data-theme="dark"] { --bg:#141517; --surface:#1c1e21; --fg:#e7e8ea; --muted:#9a9ea6; --rule:#2e3136; --pos:#4cc38a; --neg:#f07a6e; --band:#23262a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 var(--body); }
.wrap { max-width: 860px; margin: 0 auto; padding: 24px 16px 56px; display: grid; gap: 22px; }
h1 { font-size: 1.5rem; margin: 0; } h2 { font-size: 1.1rem; margin: 0 0 8px; }
.sub { color: var(--muted); margin: 4px 0 0; font-size: .9rem; }
p { margin: 0 0 8px; max-width: 72ch; } ul { margin: 0; padding-left: 20px; } li { margin-bottom: 5px; max-width: 72ch; }
.box { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.fig { margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; max-width: 430px; }
.fig figcaption { font-size: .82rem; font-weight: 600; margin: 0 0 4px; }
svg { width: 100%; height: auto; display: block; }
svg text { fill: var(--muted); font: 10px var(--mono); }
svg .grid { stroke: var(--rule); } svg .zero { stroke: var(--muted); stroke-dasharray: 3 3; } svg .band { fill: var(--band); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .82rem; color: var(--muted); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; font-size: .85rem; background: var(--surface); width: 100%; }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .78rem; }
td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pos { color: var(--pos); } .neg { color: var(--neg); } .tie { color: var(--fg); }
.note { color: var(--muted); font-size: .82rem; }
a { color: inherit; }
</style></head><body><div class="wrap">

<header>
  <h1>Train-time token drop vs dense: where it stands</h1>
  <p class="sub">Goal: one train-time token/layer-drop method that beats dense at matched FLOPs (and memory) on every task, then on long-document CPT. Qwen3.5-4B SFT on CTC tasks unless noted. Updated __UPDATED__ · in progress · <a href="details.html">full tables and every run →</a></p>
</header>

<section class="box">
  <h2>Bottom line</h2>
  <ul>
    <li><b>The mid-stack online router is the method.</b> Every token runs the first 7 of 32 blocks. A linear router, refit each step to predict each token's gradient norm, keeps 10% of each document's tokens for the remaining 25 blocks. On the 4-task mix with 1,464–1,952 rows/task, all 11 runs (2 split depths, up to 3 seeds) are within noise of dense at matched FLOPs in every cell (worst −0.002 to −0.030) and ahead on average (+0.02 to +0.065), at about 0.39× dense's FLOPs. The best fixed keep rule only ties on average and misses a cell by −0.06.</li>
    <li><b>Cheaper still: freeze the early blocks halfway.</b> Train every block for the first half, then freeze blocks 0–6 and drop their optimizer state, with no full-context steps. That costs 0.24× dense's FLOPs and 12 GiB less memory per GPU after the freeze, and is level with dense on contradiction/nq and ahead on oolong/outlier (one seed). Freezing from step 0, pyramids and merging lose.</li>
    <li><b>The 11-task mix works once each document keeps at least 16 tokens.</b> strmatch's ~25-token documents were starved by a 10% cut. The floor fixes it (strmatch 0.79 → 0.91–0.92, 2 seeds). Mean over 9 tasks: 0.730 vs dense 0.741 at 0.49×; with the freeze, 0.712 at 0.36×. Grouping@8k is still behind.</li>
    <li><b>Not yet:</b> at 64k with 10× data, the router beats dense at the same LR but trails dense at its better LR (5e-6); the router at 5e-6 is training. On long-document CPT (OLMo 3 7B) every router variant loses to dense on dev CE. The early-exit loss on dropped tokens is the fix being screened there and here.</li>
  </ul>
</section>

<section>
  <h2>4-task mix: gap to dense at matched FLOPs</h2>
  <div class="legend"><span><span class="dot" style="background:#7b4fd6"></span>router</span><span><span class="dot" style="background:#db2777"></span>aggressive (no full steps / freeze)</span><span><span class="dot" style="background:#8a8f8c"></span>random tokens</span><span><span class="dot" style="background:#0f8b8d"></span>fixed rule</span><span>grey band = ±0.03 noise; above 0 = beats dense</span></div>
  <div class="figs">__CHARTS__</div>
  <p class="note">Each dot is one recipe at one data size (points left to right = more rows/task); hover for details. Dense reference per cell = the better of dense LR 2e-6 and 5e-6, interpolated in log PF. 8 cells = contradiction, nq, outlier, oolong × 8k, 2k; eval_size 500 each, no-cot. ▼ = below the axis.</p>
  <div class="tbl"><table>
    <tr><th>recipe (most data run)</th><th class="n">rows/task</th><th class="n">PF</th><th class="n">mean Δ</th><th class="n">worst Δ</th><th class="n">runs</th></tr>
    __BEST__
  </table></div>
</section>

<section>
  <h2>Cheaper variants (FLOPs and memory)</h2>
  <div class="tbl"><table>
    <tr><th>variant</th><th class="n">FLOPs × dense</th><th class="n">peak GiB/GPU</th><th>result</th></tr>
    <tr><td>router (full-context step every 8)</td><td class="n">0.39</td><td class="n">60</td><td>reference; ≥ dense at 1,464+ rows</td></tr>
    <tr><td>no full-context steps</td><td class="n">0.31</td><td class="n">60</td><td>free at 488 rows (worst −0.057 vs −0.081)</td></tr>
    <tr><td><b>freeze blocks 0–6 halfway</b> (1,464 rows)</td><td class="n">0.24</td><td class="n">47.7</td><td><b>level with dense on contra/nq, ahead on oolong/outlier</b>; −0.03 contra vs router</td></tr>
    <tr><td>freeze blocks 0–6 from step 0 / no-grad</td><td class="n">0.19</td><td class="n">47.7 / 53</td><td class="neg">loses: contra −0.24 vs router</td></tr>
    <tr><td>pyramid (50% at 3, 10% at 7, 3% at 16)</td><td class="n">0.17</td><td class="n">53</td><td class="neg">loses contradiction</td></tr>
    <tr><td>merge dropped tokens into a summary</td><td class="n">0.28</td><td class="n">60</td><td>no better than dropping</td></tr>
    <tr><td>early-exit LM loss on dropped tokens</td><td class="n">≈0.31</td><td class="n">≤60</td><td>training (CTC on Beaker, CPT on cubbins)</td></tr>
  </table></div>
  <p class="note">488 rows/task unless noted; one seed; from the router session's record (records/online-token-router.md, "Screen results"). Peak memory on H100.</p>
</section>

<section>
  <h2>Other tracks</h2>
  <div class="tbl"><table>
    <tr><th>track</th><th>status</th></tr>
    <tr><td>11-task mix, 720 rows/task</td><td>router + 16-token floor: mean9 0.730 (2 seeds) vs dense 0.741 at 0.49×; strmatch fixed; grouping@8k still −0.08. Floor + freeze: 0.712 at 0.36×.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n">64k, 10× data</a></td><td>Router (¼ epoch, 1,872 PF, LR 2e-6) vs dense at the same LR: ahead on 6 of 7 tasks scored on that line (strmatch −0.04). Vs dense at its better LR: −0.006 to −0.05 (strmatch −0.09). Router at LR 5e-6 is training; it decides.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk">Long-document CPT (OLMo 3 7B)</a></td><td class="neg">Every router variant loses on dev CE. Best (early blocks forward-only): 1.6975 at 1,278 PF; dense needs ≈356 PF for that. Median memory −37%, peak unchanged. Early-exit loss screen running (~4 h).</td></tr>
    <tr><td><a href="https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ">Gradient composition</a></td><td>Done. Blocks 0–6 hold 0.42–0.51 of gradient energy at the base, 0.13–0.25 after training (why freezing early helps late, hurts early). Token importance is smooth in position; rescoring at block 16 adds ≈ +0.08 of gradient mass.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1">Router session page</a></td><td>Design, fidelity probes, all ablations.</td></tr>
  </table></div>
</section>

<section>
  <h2>Training now</h2>
  <ul>
    <li>Beaker: no-full-step router and + early-exit loss (W 0.3 / 1.0) on the 4-task mix at 488 and 1,464 rows/task; 11-task 960-row dense ×2 / router ×3 evals.</li>
    <li>Router session: seed repeats of freeze-halfway and freeze + floor; freeze-point sweep at 1,464 rows.</li>
    <li>CPT: early-exit loss (W 1.0 / 0.3, with and without full steps), then early-blocks-forward-only with 50%-keep periodic steps.</li>
    <li>64k: router at LR 5e-6.</li>
  </ul>
  <p class="note">Code: branch prasann/flopdrop_clean (research/token_routing/ds64_fast8k/), router claude/online-router-1bc01a. Status: ops/experiments/tokdrop_lr, learned_router, mt11_64k, olmo3_cptdrop. This page is private.</p>
</section>
</div></body></html>
"""

DETAILS_CSS = """<style id="compact">
.charts { grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)) !important; gap: 10px !important; }
.charts3 svg { min-width: 0 !important; } .charts3 .panel { max-width: 640px; }
.panel svg { max-height: 230px; }
</style>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updated", default="")
    a = ap.parse_args()
    d = json.load(open(a.points))
    pts = key_points(d)
    charts = (chart(pts, "md", "Mean Δ vs dense (8 cells)", -0.10, 0.10)
              + chart(pts, "wd", "Worst-cell Δ vs dense", -0.15, 0.05))
    page = (PAGE.replace("__CHARTS__", charts).replace("__BEST__", best_table(pts))
            .replace("__UPDATED__", a.updated or "today"))
    open(os.path.join(a.out_dir, "index.html"), "w").write(page)
    det = open(a.page).read()
    det = re.sub(r'<style id="compact">.*?</style>', "", det, flags=re.S)
    det = re.sub(r"^\s*<!doctype html>\s*<meta[^>]*>\s*<meta[^>]*>", "", det, flags=re.I)
    det = det.replace("</head>", DETAILS_CSS + "</head>", 1) if "</head>" in det else DETAILS_CSS + det
    if not det.lstrip().lower().startswith("<!doctype"):  # served as written, so it needs its own skeleton
        det = ('<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n'
               + det)
    if 'href="index.html"' not in det:
        det = re.sub(r"(<header[^>]*>)", r'\1<p class="sub"><a href="index.html">← summary</a></p>', det, count=1)
    open(os.path.join(a.out_dir, "details.html"), "w").write(det)
    for p in pts:
        print(f'{p["arm"]:14s} rows={p["rows"]:5d} n={p["n"]} PF={p["pf"]:6.0f} md={p["md"]:+.3f} wd={p["wd"]:+.3f}')


if __name__ == "__main__":
    main()
