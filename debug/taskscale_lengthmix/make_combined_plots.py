"""One panel per task: every rung and both variants together, on tokens and on training hours.

This replaces the earlier per-(task, variant) and per-(task, rung) panel sets, which spread ~50
small charts over three pages and made the two comparisons that matter -- across rungs, and dense
vs sparse -- into a paging exercise. Here each task is a single chart:

    colour   = eval rung (2k / 4k / 8k / 16k / 32k)
    solid    = dense, filled circle
    dashed   = sparse-landmark, hollow circle

so the length trend reads down the colours and the architecture gap reads between the two lines of
one colour. Two sections plot the same fits against different x-axes: training tokens, and measured
training wall-clock (dense 0.00552 h per M tokens, sparse 0.00332 -- so the sparse curve slides
left by 1.66x relative to its token position, which is where several of the gaps close).
"""
import argparse
import json
import math

import numpy as np
from scipy.optimize import curve_fit

H_PER_MTOK = {"dense": 0.00552, "sparse": 0.00332}
RUNG_TOK = {"2k": 2048, "3k": 3072, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
            "64k": 65536}
RUNG_COLOR = {"2k": "#2E7D4F", "3k": "#2E7D4F", "4k": "#5E7370", "8k": "#2B5FB8",
              "16k": "#A6690F", "32k": "#C0442A", "64k": "#7A3E9D"}
W, H = 400, 250
PAD_L, PAD_B, PAD_T, PAD_R = 46, 34, 14, 12


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


def fit(xs, ys):
    X, Y = np.asarray(xs, float), np.asarray(ys, float)
    if len(X) >= 3:
        p0 = [min(max(max(Y) * 1.05, 0.06), 1.04), 1.0, float(np.median(X))]
        p, _ = curve_fit(hill, X, Y, p0=p0, bounds=([0.05, 0.1, 1e-6], [1.05, 4.0, 1e14]),
                         maxfev=400000)
        return p
    p, _ = curve_fit(lambda x, g, K: hill(x, min(max(max(Y) * 1.3, .1), 1.0), g, K), X, Y,
                     p0=[1.0, float(np.median(X))], bounds=([0.1, 1e-6], [4.0, 1e14]), maxfev=400000)
    return np.array([min(max(max(Y) * 1.3, .1), 1.0), p[0], p[1]])


def sx(v, lo, hi):
    return PAD_L + (math.log10(v) - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def sy(v):
    return H - PAD_B - max(min(v, 1.0), 0.0) * (H - PAD_B - PAD_T)


def panel(task, task_data, axis, err=True):
    series, allx = [], []
    for rung in sorted(set(task_data.get("dense", {})) | set(task_data.get("sparse", {})),
                       key=lambda r: RUNG_TOK.get(r, 0)):
        for variant in ("dense", "sparse"):
            pts = task_data.get(variant, {}).get(rung)
            if not pts:
                continue
            scale = 1.0 if axis == "tokens" else H_PER_MTOK[variant] / 1e6
            xs = sorted(float(b) * scale for b in pts)
            ys = [pts[f"{int(float(b))}"] for b in sorted(pts, key=lambda b: float(b))]
            allx += xs
            series.append((rung, variant, xs, ys, fit(xs, ys) if len(xs) >= 2 else None))
    if not series:
        return ""
    xlo, xhi = math.log10(min(allx) / 2.2), math.log10(max(allx) * 2.2)
    body = [f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}"/>',
            f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}"/>']
    for v in (0, .25, .5, .75, 1.0):
        body.append(f'<line class="grid" x1="{PAD_L}" y1="{sy(v):.1f}" x2="{W - PAD_R}" '
                    f'y2="{sy(v):.1f}"/>')
        body.append(f'<text x="{PAD_L - 5}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:g}</text>')
    for e in range(int(math.floor(xlo)), int(math.ceil(xhi)) + 1):
        x = 10.0**e
        if not (xlo < e < xhi):
            continue
        lab = (f"{x / 1e6:g}M" if axis == "tokens" else
               (f"{x:g}h" if x >= 1 else f"{x:.2g}h"))
        body.append(f'<text x="{sx(x, xlo, xhi):.1f}" y="{H - PAD_B + 13}" '
                    f'text-anchor="middle">{lab}</text>')
    body.append(f'<text x="{PAD_L}" y="{H - 4}" style="font-size:9px">'
                + ("training tokens" if axis == "tokens" else "training node-hours (8xH100)")
                + "</text>")
    for rung, variant, xs, ys, p in series:
        col = RUNG_COLOR.get(rung, "#888")
        dash = ' stroke-dasharray="4 3"' if variant == "sparse" else ""
        if p is not None:
            grid = np.logspace(math.log10(min(xs)), math.log10(max(xs)), 40)
            body.append('<polyline points="' + " ".join(
                f"{sx(x, xlo, xhi):.1f},{sy(hill(x, *p)):.1f}" for x in grid)
                + f'" fill="none" stroke="{col}" stroke-width="1.7"{dash}/>')
        for x, y in zip(xs, ys):
            if err:
                se = (max(y, 0.0) * (1 - max(min(y, 1.0), 0.0)) / 500) ** 0.5
                body.append(f'<line x1="{sx(x, xlo, xhi):.1f}" y1="{sy(y + se):.1f}" '
                            f'x2="{sx(x, xlo, xhi):.1f}" y2="{sy(y - se):.1f}" stroke="{col}" '
                            f'stroke-width="1" opacity=".65"/>')
            if variant == "dense":
                body.append(f'<circle cx="{sx(x, xlo, xhi):.1f}" cy="{sy(y):.1f}" r="3.1" '
                            f'fill="{col}"/>')
            else:
                body.append(f'<circle cx="{sx(x, xlo, xhi):.1f}" cy="{sy(y):.1f}" r="3.1" '
                            f'fill="var(--panel)" stroke="{col}" stroke-width="1.6"/>')
    rungs_here = sorted({r for r, _, _, _, _ in series}, key=lambda r: RUNG_TOK.get(r, 0))
    legend = " ".join(f'<span style="color:{RUNG_COLOR.get(r, "#888")}">&#9679;{r}</span>'
                      for r in rungs_here)
    return (f'<figure><figcaption><b>{task}</b> &nbsp; {legend}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(body)}</svg></figure>')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points")
    ap.add_argument("out")
    a = ap.parse_args()
    data = json.load(open(a.points))
    order = ["contradiction", "nq", "oolong", "outlier", "qdmatch_nq", "xabsence", "absence",
             "grouping", "reorder", "textgroups"]
    tasks = [t for t in order if t in data] + [t for t in sorted(data) if t not in order]

    tok = "".join(panel(t, data[t], "tokens") for t in tasks)
    wall = "".join(panel(t, data[t], "hours") for t in tasks)
    css = """<style>
:root{--paper:#F7F8F6;--panel:#FFF;--ink:#1A2129;--ink-soft:#4A5560;--hairline:#D9DED9;--teal:#5E7370;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12171C;--panel:#1A2129;
 --ink:#DEE4E0;--ink-soft:#9BA8A3;--hairline:#2C353D;--teal:#8FA5A0;}}
:root[data-theme="dark"]{--paper:#12171C;--panel:#1A2129;--ink:#DEE4E0;--ink-soft:#9BA8A3;
 --hairline:#2C353D;--teal:#8FA5A0;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font-family:Charter,'Iowan Old Style',Georgia,serif;font-size:16px;line-height:1.55}
main{max-width:1120px;margin:0 auto;padding:40px 20px 80px}
h1{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:2rem;font-weight:600;margin:.2em 0}
h2{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:1.2rem;margin:2.2em 0 .4em}
p{max-width:74ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--hairline);border-radius:6px;padding:10px}
figcaption{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:.78rem;
 color:var(--ink-soft);padding-bottom:4px}
svg text{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:9px;fill:var(--ink-soft)}
svg .axis{stroke:var(--hairline);stroke-width:1}
svg .grid{stroke:var(--hairline);stroke-width:.5;stroke-dasharray:2 3}
.key{background:var(--panel);border:1px solid var(--hairline);border-left:3px solid var(--teal);
 border-radius:0 4px 4px 0;padding:12px 16px;margin:1.2em 0;font-size:.92rem;max-width:74ch}
</style>"""
    html = [f"<title>Task Scaling Ladders</title>{css}<main>",
            "<h1>Task Scaling Ladders</h1>",
            '<div class="key"><b>Reading a panel.</b> Colour is the eval rung, so the length trend '
            'reads down the colours. <b>Solid line + filled dot = dense; dashed line + hollow dot = '
            'sparse-landmark</b>, so the architecture gap is the distance between the two lines of '
            'the same colour. Vertical bars are &plusmn;1 binomial SE at eval_size 500 '
            '(&plusmn;.022 near .5) &mdash; a gap smaller than two of them is not a result. Curves '
            'span only the budgets actually trained; nothing here is extrapolated.</div>',
            "<h2>Against training tokens</h2>",
            f'<div class="grid">{tok}</div>',
            "<h2>Against measured training wall-clock</h2>",
            '<p>Same fits, same points, x rescaled by each variant\'s measured cost per token '
            '(dense 0.00552, sparse 0.00332 node-hours per million). Sparse slides <b>1.66&times; '
            'left</b> relative to its token position, which is enough to reverse the outlier 16k '
            'comparison and to close much of the oolong gap.</p>',
            f'<div class="grid">{wall}</div>',
            "</main>"]
    open(a.out, "w").write("\n".join(html))
    print(f"wrote {a.out}: {len(tasks)} tasks x 2 axes = {len(tasks) * 2} panels")


if __name__ == "__main__":
    main()
