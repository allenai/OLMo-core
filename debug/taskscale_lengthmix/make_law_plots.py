"""Render the fitted scaling laws as SVG: measured points vs extrapolated ones, drawn differently.

Two families of panel, both from the SAME fits the prediction tables use:

  A. Data laws -- score against training tokens, one line per rung. The curve is SOLID across the
     span we actually ran and DASHED beyond it, and measured points are filled circles. Anything
     dashed is the fit talking, not the data.
  B. Length laws -- tokens needed for a target score against context length, log-log. Filled
     circles are budgets derived from measured rungs; hollow squares at 64k/128k/256k/1M are the
     extrapolation. The dashed segment carries the fitted exponent beta.

Usage: python make_law_plots.py points.json out.html [--target 0.5]
"""
import argparse
import json
import math

import numpy as np
from scipy.optimize import curve_fit

RUNG_TOK = {"2k": 2048, "3k": 3072, "4k": 4096, "8k": 8192, "16k": 16384,
            "32k": 32768, "64k": 65536}
OUT_L = [65536, 131072, 262144, 1048576]
RUNG_COLOR = {"2k": "#2E7D4F", "3k": "#2E7D4F", "4k": "#5E7370",
              "8k": "#2B5FB8", "16k": "#A6690F", "32k": "#C0442A", "64k": "#7A3E9D"}
W, H = 300, 200
PAD_L, PAD_B, PAD_T, PAD_R = 44, 30, 12, 10


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


def fit_rung(bs, sc, ceiling=1.0):
    B, f = np.asarray(bs, float), np.asarray(sc, float)
    if len(B) >= 3:
        p0 = [min(max(max(f) * 1.05, 0.06), 1.04), 1.0, float(np.median(B))]
        p, _ = curve_fit(hill, B, f, p0=p0, bounds=([0.05, 0.1, 1e5], [1.05, 4.0, 1e11]),
                         maxfev=400000)
        return p
    p, _ = curve_fit(lambda x, g, K: hill(x, ceiling, g, K), B, f, p0=[1.0, float(np.median(B))],
                     bounds=([0.1, 1e5], [4.0, 1e11]), maxfev=400000)
    return np.array([ceiling, p[0], p[1]])


def budget_for(p, t):
    fmax, g, K = p
    return None if t >= fmax else K * (t / (fmax - t)) ** (1.0 / g)


def sx(v, lo, hi):
    return PAD_L + (math.log10(v) - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def sy(v, lo, hi):
    return H - PAD_B - (v - lo) / (hi - lo) * (H - PAD_B - PAD_T)


def axes(xlo, xhi, ylab, yticks, ylo, yhi, xticks):
    out = [f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}"/>',
           f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}"/>']
    for v, lab in yticks:
        y = sy(v, ylo, yhi)
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>')
        out.append(f'<text x="{PAD_L - 5}" y="{y + 3:.1f}" text-anchor="end">{lab}</text>')
    for v, lab in xticks:
        x = sx(v, xlo, xhi)
        out.append(f'<text x="{x:.1f}" y="{H - PAD_B + 13}" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="6" y="{PAD_T + 6}" style="font-size:9px">{ylab}</text>')
    return "".join(out)


def data_panel(task, variant, rungs):
    fits, allB = {}, []
    for lab, pts in rungs.items():
        if lab not in RUNG_TOK or len(pts) < 2:
            continue
        bs = sorted(float(b) for b in pts)
        fits[lab] = (fit_rung(bs, [pts[f"{int(b)}"] for b in bs]), bs,
                     [pts[f"{int(b)}"] for b in bs])
        allB += bs
    if not fits:
        return ""
    xlo, xhi = math.log10(min(allB) / 2), math.log10(max(allB) * 12)
    ylo, yhi = 0.0, 1.0
    body = [axes(xlo, xhi, "score", [(0, "0"), (.5, ".5"), (1, "1")], ylo, yhi,
                 [(10**e, f"{10**e / 1e6:.0f}M") for e in (7, 8, 9) if xlo < e < xhi])]
    for lab, (p, bs, scores) in sorted(fits.items(), key=lambda kv: RUNG_TOK[kv[0]]):
        col = RUNG_COLOR.get(lab, "#888")
        grid = np.logspace(math.log10(min(bs)), math.log10(max(bs)), 40)
        solid = " ".join(f"{sx(b, xlo, xhi):.1f},{sy(hill(b, *p), ylo, yhi):.1f}" for b in grid)
        body.append(f'<polyline points="{solid}" fill="none" stroke="{col}" stroke-width="1.6"/>')
        gx = np.logspace(math.log10(max(bs)), xhi, 30)
        dash = " ".join(f"{sx(b, xlo, xhi):.1f},{sy(hill(b, *p), ylo, yhi):.1f}" for b in gx)
        body.append(f'<polyline points="{dash}" fill="none" stroke="{col}" stroke-width="1.2" '
                    f'stroke-dasharray="3 3" opacity=".8"/>')
        for b, s_ in zip(bs, scores):
            body.append(f'<circle cx="{sx(b, xlo, xhi):.1f}" cy="{sy(s_, ylo, yhi):.1f}" r="3" '
                        f'fill="{col}"/>')
    legend = " ".join(
        f'<span style="color:{RUNG_COLOR.get(l, "#888")}">&#9679; {l}</span>'
        for l in sorted(fits, key=lambda x: RUNG_TOK[x]))
    return (f'<figure><figcaption><b>{task} &middot; {variant}</b> &mdash; {legend}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(body)}</svg></figure>')


def length_panel(task, variant, rungs, target):
    pts = []
    for lab, budgets in rungs.items():
        if lab not in RUNG_TOK or len(budgets) < 2:
            continue
        bs = sorted(float(b) for b in budgets)
        p = fit_rung(bs, [budgets[f"{int(b)}"] for b in bs])
        need = budget_for(p, target)
        if need is not None and need <= 20 * max(bs):
            pts.append((RUNG_TOK[lab], need))
    if len(pts) < 2:
        return "", None
    pts.sort()
    L = np.log(np.array([x[0] for x in pts], float))
    B = np.log(np.array([x[1] for x in pts], float))
    beta, c = np.polyfit(L, B, 1)
    preds = [(t, math.exp(c + beta * math.log(t))) for t in OUT_L]
    xs = [x[0] for x in pts] + [t for t, _ in preds]
    ys = [x[1] for x in pts] + [v for _, v in preds]
    xlo, xhi = math.log10(min(xs) / 1.4), math.log10(max(xs) * 1.4)
    ylo, yhi = math.log10(min(ys) / 3), math.log10(max(ys) * 3)

    def py(v):
        return sy(math.log10(v), ylo, yhi)
    yt = [(10**e, f"{10**e / 1e9:.0f}B" if e >= 9 else f"{10**e / 1e6:.0f}M")
          for e in range(int(ylo), int(yhi) + 1)]
    body = [axes(xlo, xhi, "tokens", [(math.log10(v), lab) for v, lab in yt], ylo, yhi,
                 [(v, f"{v // 1024}k") for v in (2048, 8192, 32768, 131072, 1048576)
                  if xlo < math.log10(v) < xhi])]
    col = "#2B5FB8" if variant == "dense" else "#C0442A"
    seg = " ".join(f"{sx(x, xlo, xhi):.1f},{py(y):.1f}" for x, y in pts)
    body.append(f'<polyline points="{seg}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    ext = [pts[-1]] + preds
    seg2 = " ".join(f"{sx(x, xlo, xhi):.1f},{py(y):.1f}" for x, y in ext)
    body.append(f'<polyline points="{seg2}" fill="none" stroke="{col}" stroke-width="1.4" '
                f'stroke-dasharray="4 3"/>')
    for x, y in pts:
        body.append(f'<circle cx="{sx(x, xlo, xhi):.1f}" cy="{py(y):.1f}" r="3.4" fill="{col}"/>')
    for x, y in preds:
        body.append(f'<rect x="{sx(x, xlo, xhi) - 3:.1f}" y="{py(y) - 3:.1f}" width="6" height="6" '
                    f'fill="var(--panel)" stroke="{col}" stroke-width="1.4"/>')
    cap = (f'<b>{task} &middot; {variant}</b> &mdash; &beta;={beta:.2f}, '
           f'{len(pts)} measured rungs &#9679; &rarr; 4 extrapolated &#9633;')
    return (f'<figure><figcaption>{cap}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(body)}</svg></figure>'), beta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--target", type=float, default=0.5)
    a = ap.parse_args()
    data = json.load(open(a.points))

    dpanels, lpanels = [], []
    for task in sorted(data):
        for variant in ("dense", "sparse"):
            rungs = data[task].get(variant, {})
            if not rungs:
                continue
            d = data_panel(task, variant, rungs)
            if d:
                dpanels.append(d)
            lp, _ = length_panel(task, variant, rungs, a.target)
            if lp:
                lpanels.append(lp)

    css = """<style>
:root{--paper:#F7F8F6;--panel:#FFF;--ink:#1A2129;--ink-soft:#4A5560;--hairline:#D9DED9;--teal:#5E7370;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12171C;--panel:#1A2129;
 --ink:#DEE4E0;--ink-soft:#9BA8A3;--hairline:#2C353D;--teal:#8FA5A0;}}
:root[data-theme="dark"]{--paper:#12171C;--panel:#1A2129;--ink:#DEE4E0;--ink-soft:#9BA8A3;
 --hairline:#2C353D;--teal:#8FA5A0;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font-family:Charter,'Iowan Old Style',Georgia,serif;font-size:16px;line-height:1.55}
main{max-width:1040px;margin:0 auto;padding:40px 20px 80px}
h1{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:2rem;font-weight:600;margin:.2em 0}
h2{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:1.25rem;margin:2em 0 .3em}
p{max-width:70ch;color:var(--ink)}
.sub{color:var(--ink-soft);max-width:70ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--hairline);border-radius:6px;padding:10px}
figcaption{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:.78rem;
 color:var(--ink-soft);padding-bottom:4px}
svg text{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:9px;fill:var(--ink-soft)}
svg .axis{stroke:var(--hairline);stroke-width:1}
svg .grid{stroke:var(--hairline);stroke-width:.5;stroke-dasharray:2 3}
.key{background:var(--panel);border:1px solid var(--hairline);border-left:3px solid var(--teal);
 border-radius:0 4px 4px 0;padding:12px 16px;margin:1.2em 0;font-size:.9rem;max-width:70ch}
</style>"""
    html = [f"<title>Scaling-Law Fits: Measured vs Extrapolated</title>{css}<main>",
            "<h1>Scaling-Law Fits</h1>",
            '<p class="sub">Every curve below is the fit behind the published budget tables. '
            'Solid line and filled markers cover the range we actually trained; dashed line and '
            'hollow markers are extrapolation.</p>',
            '<div class="key"><b>How to read it.</b> Panel set A is the data law per rung: score '
            'against training tokens. Panel set B is the length law: tokens needed for '
            f'score={a.target} against context length, log-log, with the slope being the length '
            'exponent &beta;. A rung only appears in B if the target is reachable within 20&times; '
            'the budgets we ran &mdash; otherwise the "budget" would be the fit running away rather '
            'than a measurement.</div>',
            "<h2>A. Data laws — score vs training tokens</h2>",
            f'<div class="grid">{"".join(dpanels)}</div>',
            f"<h2>B. Length laws — tokens for score={a.target} vs context length</h2>",
            f'<div class="grid">{"".join(lpanels)}</div>',
            "</main>"]
    open(a.out, "w").write("\n".join(html))
    print(f"wrote {a.out}: {len(dpanels)} data panels, {len(lpanels)} length panels")


if __name__ == "__main__":
    main()
