"""Dense vs sparse on a TRAINING WALL-CLOCK axis, extrapolated far enough to expose a crossover.

Tokens are the wrong axis for "is sparse worth it": sparse loses on tokens by construction, and the
whole argument for it is that each token is cheaper to train on. This converts both ladders to
training wall-clock and asks again.

The exchange rate is MEASURED, not modelled. Every arm's Beaker job carries a start and an exit
time; per task and variant, fitting hours = setup + rate * tokens across that task's budgets gives
the marginal cost of a token, with the intercept absorbing image pull and checkpoint load. The
slopes are tight within a variant:

    dense   median 0.00552 h per M tokens   (n=4 tasks, range 0.00457-0.00571)
    sparse  median 0.00332 h per M tokens   (n=7 tasks, range 0.00286-0.00350, after excluding
                                             one preempted reorder arm that fit 0.0273)

so **sparse trains 1.66x faster per token** on one 8xH100 node at these short-heavy mixes -- much
more than a FLOP count predicts (~1.1x), because Qwen3.5 is a 3:1 GDN/full-attention hybrid in
which attention is only ~11% of the arithmetic. The real gain is kernel and memory behaviour, which
is precisely why this axis is measured rather than derived.

The consequence to read off the panels: a variant needing LESS than 1.66x its rival's tokens for
the same score is CHEAPER IN TIME despite losing on data. Arms trained on mixed hardware (A100 vs
H100) are excluded from the rate fits; x is 8-GPU node-hours of training compute, excluding setup.
"""
import argparse
import json
import math

import numpy as np
from scipy.optimize import curve_fit

# MEASURED marginal training cost in node-hours per token (medians of the per-task fits above).
H_PER_MTOK = {"dense": 0.00552, "sparse": 0.00332}

RUNG_TOK = {"2k": 2048, "3k": 3072, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
            "64k": 65536}
W, H = 330, 210
PAD_L, PAD_B, PAD_T, PAD_R = 48, 32, 12, 12
C_DENSE, C_SPARSE = "#2B5FB8", "#C0442A"


def hours_per_token(task, variant):
    """Measured node-hours per training token. `task` is accepted for a future per-task rate."""
    return H_PER_MTOK[variant] / 1e6


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


def fit(xs, ys):
    X, Y = np.asarray(xs, float), np.asarray(ys, float)
    if len(X) >= 3:
        p0 = [min(max(max(Y) * 1.05, 0.06), 1.04), 1.0, float(np.median(X))]
        p, _ = curve_fit(hill, X, Y, p0=p0, bounds=([0.05, 0.1, 1e-4], [1.05, 4.0, 1e12]),
                         maxfev=400000)
        return p
    p, _ = curve_fit(lambda x, g, K: hill(x, 1.0, g, K), X, Y, p0=[1.0, float(np.median(X))],
                     bounds=([0.1, 1e-4], [4.0, 1e12]), maxfev=400000)
    return np.array([1.0, p[0], p[1]])


def sx(v, lo, hi):
    return PAD_L + (math.log10(v) - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def sy(v):
    return H - PAD_B - v * (H - PAD_B - PAD_T)


def panel(task, rung, dense_pts, sparse_pts, xmax_exp):
    fits, spans = {}, {}
    for variant, pts in (("dense", dense_pts), ("sparse", sparse_pts)):
        if not pts or len(pts) < 2:
            continue
        fpt = hours_per_token(task, variant)
        xs = sorted(float(b) * fpt for b in pts)
        ys = [pts[f"{int(float(b))}"] for b in sorted(pts, key=lambda b: float(b))]
        fits[variant] = fit(xs, ys)
        spans[variant] = (xs, ys)
    if not fits:
        return "", None
    xlo = math.log10(min(min(v[0]) for v in spans.values()) / 3)
    xhi = xmax_exp
    body = [f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}"/>',
            f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}"/>']
    for v in (0, .25, .5, .75, 1.0):
        body.append(f'<line class="grid" x1="{PAD_L}" y1="{sy(v):.1f}" x2="{W - PAD_R}" '
                    f'y2="{sy(v):.1f}"/>')
        body.append(f'<text x="{PAD_L - 5}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:.2f}</text>')
    for e in range(int(math.ceil(xlo)), int(xhi) + 1):
        body.append(f'<text x="{sx(10**e, xlo, xhi):.1f}" y="{H - PAD_B + 13}" '
                    f'text-anchor="middle">{10**e:g}h</text>')
    cross = None
    for variant, p in fits.items():
        col = C_DENSE if variant == "dense" else C_SPARSE
        xs, ys = spans[variant]
        grid = np.logspace(math.log10(min(xs)), math.log10(max(xs)), 40)
        body.append('<polyline points="' + " ".join(
            f"{sx(x, xlo, xhi):.1f},{sy(min(hill(x, *p), 1.0)):.1f}" for x in grid)
            + f'" fill="none" stroke="{col}" stroke-width="1.9"/>')
        gx = np.logspace(math.log10(max(xs)), xhi, 60)
        body.append('<polyline points="' + " ".join(
            f"{sx(x, xlo, xhi):.1f},{sy(min(hill(x, *p), 1.0)):.1f}" for x in gx)
            + f'" fill="none" stroke="{col}" stroke-width="1.3" stroke-dasharray="4 3" '
              'opacity=".85"/>')
        for x, y in zip(xs, ys):
            body.append(f'<circle cx="{sx(x, xlo, xhi):.1f}" cy="{sy(y):.1f}" r="3.2" '
                        f'fill="{col}"/>')
    if len(fits) == 2:
        # Search only where BOTH variants have data: below that the two fitted curves are
        # extrapolating in opposite directions and cross for no physical reason (oolong@16k
        # "crossed" at the sparse ladder's leftmost point, which is simply where dense's fit had
        # not started).
        lo_both = max(min(v[0]) for v in spans.values())
        g = np.logspace(math.log10(lo_both), xhi, 3000)
        d, s = hill(g, *fits["dense"]), hill(g, *fits["sparse"])
        hit = np.nonzero(s >= d)[0]
        if len(hit):
            cross = float(g[hit[0]])
            body.append(f'<line x1="{sx(cross, xlo, xhi):.1f}" y1="{PAD_T}" '
                        f'x2="{sx(cross, xlo, xhi):.1f}" y2="{H - PAD_B}" stroke="#2E7D4F" '
                        f'stroke-width="1.2" stroke-dasharray="2 2"/>')
    measured_max = max(max(v[0]) for v in spans.values())
    cap = (f"<b>{task} @ {rung}</b> &mdash; "
           + ("crossover " + (f"{cross:.3g} node-h" if cross else "none below 1e%d h" % xhi))
           + (f", {cross / measured_max:.0f}&times; past measurement" if cross else ""))
    return (f'<figure><figcaption>{cap}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(body)}</svg></figure>'), cross


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--xmax-exp", type=int, default=6)
    a = ap.parse_args()
    data = json.load(open(a.points))

    panels, rows = [], []
    for task in sorted(data):
        dense, sparse = data[task].get("dense", {}), data[task].get("sparse", {})
        for rung in sorted(set(dense) | set(sparse), key=lambda r: RUNG_TOK.get(r, 0)):
            html, cross = panel(task, rung, dense.get(rung), sparse.get(rung), a.xmax_exp)
            if html:
                panels.append(html)
                if len(dense.get(rung) or {}) >= 2 and len(sparse.get(rung) or {}) >= 2:
                    rows.append((task, rung, cross))

    rtxt = (f"dense {H_PER_MTOK['dense']:.5f} vs sparse {H_PER_MTOK['sparse']:.5f} node-hours "
            f"per million tokens ({H_PER_MTOK['dense'] / H_PER_MTOK['sparse']:.2f}&times; faster)")
    trows = "".join(
        f"<tr><td>{t}</td><td>{r}</td><td class='n'>"
        + (f"{c:.3g} node-h" if c else "none below 1e%d h" % a.xmax_exp) + "</td></tr>"
        for t, r, c in rows)
    css = """<style>
:root{--paper:#F7F8F6;--panel:#FFF;--ink:#1A2129;--ink-soft:#4A5560;--hairline:#D9DED9;--teal:#5E7370;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12171C;--panel:#1A2129;
 --ink:#DEE4E0;--ink-soft:#9BA8A3;--hairline:#2C353D;--teal:#8FA5A0;}}
:root[data-theme="dark"]{--paper:#12171C;--panel:#1A2129;--ink:#DEE4E0;--ink-soft:#9BA8A3;
 --hairline:#2C353D;--teal:#8FA5A0;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font-family:Charter,'Iowan Old Style',Georgia,serif;font-size:16px;line-height:1.55}
main{max-width:1080px;margin:0 auto;padding:40px 20px 80px}
h1{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:2rem;font-weight:600;margin:.2em 0}
h2{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:1.2rem;margin:2em 0 .3em}
p{max-width:72ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--hairline);border-radius:6px;padding:10px}
figcaption{font-family:'Avenir Next',Seravek,system-ui,sans-serif;font-size:.76rem;
 color:var(--ink-soft);padding-bottom:4px}
svg text{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:9px;fill:var(--ink-soft)}
svg .axis{stroke:var(--hairline);stroke-width:1}
svg .grid{stroke:var(--hairline);stroke-width:.5;stroke-dasharray:2 3}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9rem;margin:1em 0}
th,td{padding:6px 12px;border-bottom:1px solid var(--hairline);text-align:left}
td.n{text-align:right;font-family:'SF Mono',Menlo,Consolas,monospace}
.key{background:var(--panel);border:1px solid var(--hairline);border-left:3px solid var(--teal);
 border-radius:0 4px 4px 0;padding:12px 16px;margin:1.2em 0;font-size:.92rem;max-width:72ch}
.sw{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:5px}
</style>"""
    html = [f"<title>Dense vs Sparse per Training Hour</title>{css}<main>",
            "<h1>Dense vs Sparse per Training Hour</h1>",
            '<p><span class="sw" style="background:#2B5FB8"></span>dense &nbsp; '
            '<span class="sw" style="background:#C0442A"></span>sparse-landmark &nbsp; '
            '<span class="sw" style="background:#2E7D4F"></span>crossover. Filled circles and '
            'solid curves are measured; dashed is extrapolation, run deliberately far '
            f'(to 1e{a.xmax_exp} node-hours) so a crossover shows up even when it is far beyond '
            'anything we could run.</p>',
            f'<div class="key"><b>The exchange rate is measured, not modelled: </b>{rtxt}, fitted '
            'from every arm\'s job start and exit times. <b>So sparse can lose on data and still '
            'win on time</b> -- anywhere it needs under 1.66&times; dense\'s tokens for the same '
            'score, it is the cheaper route. A FLOP count would have predicted only ~1.1&times;, '
            'since attention is barely a tenth of the arithmetic in this 3:1 GDN hybrid; the rest '
            'is kernel and memory behaviour.</div>',
            "<h2>Per task and rung</h2>",
            f'<div class="grid">{"".join(panels)}</div>',
            "<h2>Crossovers, where both variants have a fitted curve</h2>",
            f'<table><tr><th>task</th><th>rung</th><th class="n">sparse overtakes dense at</th></tr>'
            f'{trows}</table>',
            '<p style="color:var(--ink-soft);font-size:.9rem">Every crossover above sits many orders '
            'of magnitude beyond the largest budget measured, and rests on the losing curve\'s '
            'ceiling being unconstrained by data. Read them as "not within reach", not as targets.'
            '</p>', "</main>"]
    open(a.out, "w").write("\n".join(html))
    print(f"wrote {a.out}: {len(panels)} panels, {len(rows)} with both variants")
    for t, r, c in rows:
        print(f"  {t:14s} {r:>4s}  crossover: " + (f"{c:.3g} node-hours" if c else "none"))


if __name__ == "__main__":
    main()
