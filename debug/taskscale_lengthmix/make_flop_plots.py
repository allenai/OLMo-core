"""Dense vs sparse on a FLOP axis, extrapolated far enough to expose any crossover.

Tokens are the wrong axis for "is sparse worth it": sparse loses on tokens by construction, and the
whole argument for it is that each token is cheaper. This converts both ladders to training FLOPs
and asks the question again.

FLOP model (Qwen3.5-4B, from TransformerConfig.qwen3_5_4B):

    per token = 6N  +  f * 12 * n_full * d_attn * L_eff
                ^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                params  attention over the training mix's token-weighted mean length

  * N = 4.0e9, and 6N is the usual fwd+bwd parameter term.
  * n_full = 8, NOT 32. Qwen3.5 interleaves Gated DeltaNet and full attention 3:1, so only a
    quarter of the layers pay a quadratic attention cost. This is the single most important number
    here: it makes attention a MINORITY of compute at the lengths we trained, which caps how much
    sparsity can possibly save.
  * d_attn = n_heads * head_dim = 16 * 256 = 4096.
  * f = 1 for dense, 1/64 for sparse-landmark (the campaign's throughput model: a landmark block
    attends within a block of 64 plus the landmark row).
  * L_eff is the TRAINING mix's token-weighted mean length -- a property of the arm, not of the
    eval rung, since every rung of a task is scored from the same checkpoint.

The consequence, visible in every panel: at our training lengths sparse buys ~10% of FLOPs, because
attention is ~11% of the bill. Sparsity only becomes a large lever when L_eff is big enough for
attention to dominate, which is far outside anything we trained -- so the extrapolation runs to
1e26 FLOPs to show where the curves would cross IF the fitted trends held. They almost certainly do
not hold that far; the dashed regions are there to show the shape of the argument, not to predict.
"""
import argparse
import json
import math

import numpy as np
from scipy.optimize import curve_fit

N_PARAMS = 4.0e9
N_FULL_ATTN_LAYERS = 8          # 32 layers, GDN:full = 3:1
D_ATTN = 16 * 256

# Token-weighted mean training length per task, from the measured rung medians and the
# short-heavy shares (61.6/21.9/11.0/5.5, renormalized when a task has three rungs).
L_EFF = {
    "oolong": 6519, "contradiction": 6659, "nq": 6659, "qdmatch_nq": 6659, "outlier": 6659,
    "xabsence": 8362, "absence": 9359, "reorder": 3960, "textgroups": 3960, "grouping": 6659,
}
RUNG_TOK = {"2k": 2048, "3k": 3072, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
            "64k": 65536}
W, H = 330, 210
PAD_L, PAD_B, PAD_T, PAD_R = 48, 32, 12, 12
C_DENSE, C_SPARSE = "#2B5FB8", "#C0442A"


def flops_per_token(task, variant):
    f = 1.0 if variant == "dense" else 1.0 / 64.0
    return 6 * N_PARAMS + f * 12 * N_FULL_ATTN_LAYERS * D_ATTN * L_EFF.get(task, 6659)


def hill(B, fmax, g, K):
    return fmax * B**g / (B**g + K**g)


def fit(xs, ys):
    X, Y = np.asarray(xs, float), np.asarray(ys, float)
    if len(X) >= 3:
        p0 = [min(max(max(Y) * 1.05, 0.06), 1.04), 1.0, float(np.median(X))]
        p, _ = curve_fit(hill, X, Y, p0=p0, bounds=([0.05, 0.1, 1e10], [1.05, 4.0, 1e30]),
                         maxfev=400000)
        return p
    p, _ = curve_fit(lambda x, g, K: hill(x, 1.0, g, K), X, Y, p0=[1.0, float(np.median(X))],
                     bounds=([0.1, 1e10], [4.0, 1e30]), maxfev=400000)
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
        fpt = flops_per_token(task, variant)
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
    for e in range(int(math.ceil(xlo)), int(xhi) + 1, 3):
        body.append(f'<text x="{sx(10**e, xlo, xhi):.1f}" y="{H - PAD_B + 13}" '
                    f'text-anchor="middle">1e{e}</text>')
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
        g = np.logspace(math.log10(min(min(v[0]) for v in spans.values())), xhi, 3000)
        d, s = hill(g, *fits["dense"]), hill(g, *fits["sparse"])
        hit = np.nonzero(s >= d)[0]
        if len(hit):
            cross = float(g[hit[0]])
            body.append(f'<line x1="{sx(cross, xlo, xhi):.1f}" y1="{PAD_T}" '
                        f'x2="{sx(cross, xlo, xhi):.1f}" y2="{H - PAD_B}" stroke="#2E7D4F" '
                        f'stroke-width="1.2" stroke-dasharray="2 2"/>')
    measured_max = max(max(v[0]) for v in spans.values())
    cap = (f"<b>{task} @ {rung}</b> &mdash; "
           + ("crossover " + (f"{cross:.0e} FLOPs" if cross else "none below 1e%d" % xhi))
           + (f", {cross / measured_max:.0f}&times; past measurement" if cross else ""))
    return (f'<figure><figcaption>{cap}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" width="100%">{"".join(body)}</svg></figure>'), cross


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--xmax-exp", type=int, default=26)
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

    ratio = {t: flops_per_token(t, "dense") / flops_per_token(t, "sparse") for t in L_EFF}
    rtxt = ", ".join(f"{t} {ratio[t]:.2f}&times;" for t in sorted(ratio))
    trows = "".join(
        f"<tr><td>{t}</td><td>{r}</td><td class='n'>"
        + (f"{c:.0e}" if c else "none below 1e%d" % a.xmax_exp) + "</td></tr>"
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
    html = [f"<title>Dense vs Sparse on a FLOP Axis</title>{css}<main>",
            "<h1>Dense vs Sparse per FLOP</h1>",
            '<p><span class="sw" style="background:#2B5FB8"></span>dense &nbsp; '
            '<span class="sw" style="background:#C0442A"></span>sparse-landmark &nbsp; '
            '<span class="sw" style="background:#2E7D4F"></span>crossover. Filled circles and '
            'solid curves are measured; dashed is extrapolation, run deliberately far '
            f'(to 1e{a.xmax_exp} FLOPs) so a crossover shows up even when it is nowhere near '
            'anything we could run.</p>',
            f'<div class="key"><b>Why the FLOP axis barely moves the answer.</b> Qwen3.5 '
            'interleaves Gated DeltaNet and full attention <b>3:1</b>, so only 8 of 32 layers pay a '
            'quadratic cost. At our training mixes attention is roughly a tenth of the compute, and '
            f'sparse-landmark\'s 1/64 attention discount buys only: {rtxt} fewer FLOPs per token. '
            'Sparsity is a large lever only when attention dominates the bill, which needs a much '
            'longer training mix than any arm here used. So a curve that loses on tokens loses on '
            'FLOPs too, by almost the same margin.</div>',
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
        print(f"  {t:14s} {r:>4s}  crossover: " + (f"{c:.2e} FLOPs" if c else "none"))


if __name__ == "__main__":
    main()
