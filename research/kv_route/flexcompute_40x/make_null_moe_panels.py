"""Regenerate the 'Null rungs versus narrow rungs' panels with a standard-MoE series added.

Reads `ffn_null_rung_{scale}_{T}.json` (narrow-only / full-or-null / ladder+null) and
`ffn_moe_arm_{scale}_{T}.json` (top-k MoE sweep + real Qwen3.5 geometry anchors) and writes the
four <figure> panels plus their legend to stdout, in the same SVG dialect as the rest of
`research/kv_route/viz/ffn_routing_linearity.html`.

    python research/kv_route/flexcompute_40x/make_null_moe_panels.py --tokens 32768 > /tmp/panels.html
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = ["0.8B", "4B", "9B", "27B"]

X0, X1, Y0, Y1 = 56.0, 334.0, 16.0, 218.0
LOGC_MIN, LOGC_MAX = -4.0, 1.0

SERIES = [
    ("narrow_ms", "narrow_c", "s1", "narrow-only"),
    ("null_ms", "null_c", "s2", "full-or-null"),
    ("ln_ms", "ln_c", "s3", "ladder + null"),
]


def xpos(c):
    return X0 + (math.log10(c) - LOGC_MIN) / (LOGC_MAX - LOGC_MIN) * (X1 - X0)


def ypos(ms, lo, hi):
    return Y1 - (math.log10(ms) - lo) / (hi - lo) * (Y1 - Y0)


def fmt_tick(v):
    return f"{v:g}"


def panel(scale, tokens, moe_key="gemm_ms"):
    null = json.load(open(os.path.join(HERE, f"ffn_null_rung_{scale}_{tokens}.json")))
    moe = json.load(open(os.path.join(HERE, f"ffn_moe_arm_{scale}_{tokens}.json")))
    rows = null["rows"]

    moe_sweep = [r for r in moe["rows"] if "anchor" not in r and r["k"] == 8]
    moe_sweep.sort(key=lambda r: r["c"])
    moe_k2 = [r for r in moe["rows"] if "anchor" not in r and r["k"] == 2]
    moe_k2.sort(key=lambda r: r["c"])
    anchors = [r for r in moe["rows"] if "anchor" in r]

    vals = []
    for mk, _, _, _ in SERIES:
        vals += [r[mk] for r in rows if r[mk] == r[mk]]
    vals += [r[moe_key] for r in moe_sweep + moe_k2 + anchors if r[moe_key] == r[moe_key]]
    lo = math.floor(math.log10(min(vals)))
    hi = math.ceil(math.log10(max(vals)))
    if hi - lo < 1:
        hi = lo + 1

    out = []
    for dec in range(int(lo), int(hi) + 1):
        y = ypos(10**dec, lo, hi)
        out.append(f'<line class="grid" x1="{X0:g}" y1="{y:.1f}" x2="{X1:g}" y2="{y:.1f}"/>')
        out.append(
            f'<text class="tick" x="{X0 - 7:g}" y="{y + 3:.1f}" text-anchor="end">'
            f"{fmt_tick(10**dec)}</text>"
        )
    for e in range(-4, 2):
        x = xpos(10**e)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{Y0:g}" x2="{x:.1f}" y2="{Y1:g}"/>')
        out.append(
            f'<text class="tick" x="{x:.1f}" y="233" text-anchor="middle">{10.0**e:g}</text>'
        )
    xd = xpos(1.0)
    out.append(f'<line class="capline" x1="{xd:.1f}" y1="{Y0:g}" x2="{xd:.1f}" y2="{Y1:g}"/>')
    out.append(f'<text class="guidelab" x="{xd + 3:.1f}" y="{Y0 + 8:g}">dense</text>')
    out.append(f'<line class="axis" x1="{X0:g}" y1="{Y1:g}" x2="{X1:g}" y2="{Y1:g}"/>')
    out.append(f'<line class="axis" x1="{X0:g}" y1="{Y0:g}" x2="{X0:g}" y2="{Y1:g}"/>')
    out.append('<text class="axlab" x="195" y="256" text-anchor="middle">mean FFN compute c (log)</text>')
    out.append('<text class="axlab" x="4" y="9">ms</text>')

    def draw(points, cls, label, marker="circle", extra=None):
        pts = [(xpos(c), ypos(t, lo, hi)) for c, t, *_ in points if t == t and c > 0]
        if not pts:
            return
        d = " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<path class="line {cls}" d="M{d}"/>')
        for (x, y), pt in zip(pts, [p for p in points if p[1] == p[1] and p[0] > 0]):
            c, t = pt[0], pt[1]
            title = f"{label} — c={c:.5f}, {t:.3f} ms"
            if extra and len(pt) > 2:
                title += f" ({extra}: {pt[2]:.3f} ms)"
            if marker == "square":
                out.append(
                    f'<rect class="dot {cls}" x="{x - 3:.1f}" y="{y - 3:.1f}" width="6" '
                    f'height="6" rx="1"><title>{title}</title></rect>'
                )
            else:
                out.append(
                    f'<circle class="dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="3.4">'
                    f"<title>{title}</title></circle>"
                )

    for mk, ck, cls, label in SERIES:
        draw([(r[ck], r[mk]) for r in rows], cls, label)
    draw(
        [(r["c"], r[moe_key], r["full_ms"]) for r in moe_sweep],
        "s5",
        "standard MoE, top-8",
        marker="square",
        extra="with dispatch",
    )

    draw(
        [(r["c"], r[moe_key], r["full_ms"]) for r in moe_k2],
        "s5 dashed",
        "standard MoE, top-2",
        marker="square",
        extra="with dispatch",
    )

    # Real shipped Qwen3.5 MoE geometry: one hollow marker each, no connecting line.
    for r in anchors:
        if r[moe_key] != r[moe_key]:
            continue
        x, y = xpos(r["c"]), ypos(r[moe_key], lo, hi)
        out.append(
            f'<rect class="dot hollow s5" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
            f'rx="1"><title>{r["anchor"]} geometry — {r["E"]}x{r["We"]}, {r["k"]} live — '
            f'c={r["c"]:.3f}, {r[moe_key]:.3f} ms (with dispatch: {r["full_ms"]:.3f} ms)'
            f"</title></rect>"
        )

    # headline: how much cheaper is full-or-null than the MoE sweep at the MoE's own best point?
    best_moe = min((r for r in moe_sweep if r[moe_key] == r[moe_key]), key=lambda r: r[moe_key])
    best_null = min(r["null_ms"] for r in rows if r["null_ms"] == r["null_ms"])
    adv = best_moe[moe_key] / best_null

    body = "\n".join(out)
    return (
        f'<figure class="panel"><figcaption><span class="tok">{scale}</span>'
        f'<span class="vd">null vs MoE {adv:.1f}&times;</span></figcaption>'
        f'<svg viewBox="0 0 350 262" role="img" aria-label="log-log chart">\n{body}\n</svg></figure>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--moe-key", default="gemm_ms", choices=["gemm_ms", "full_ms"])
    args = ap.parse_args()
    panels = "".join(panel(s, args.tokens, args.moe_key) for s in SCALES)
    legend = (
        '<div class="legend">'
        '<span class="lg"><span class="key s1"></span>narrow-only</span>'
        '<span class="lg"><span class="key s2"></span>full-or-null</span>'
        '<span class="lg"><span class="key s3"></span>ladder + null</span>'
        '<span class="lg"><span class="key sq s5"></span>standard MoE, top-8 (the shipped k)</span>'
        '<span class="lg"><span class="key dash s5"></span>standard MoE, top-2</span>'
        '<span class="lg"><span class="key sq hollow s5"></span>shipped Qwen3.5 expert geometry</span>'
        "</div>"
    )
    print(legend + '<div class="grid3">' + panels + "</div>")
    print(table(args.tokens, args.moe_key))




def table(tokens, moe_key="gemm_ms"):
    """The same four panels as a table: best FFN time each family reaches, and where."""
    out = [
        '<div class="tablewrap"><table><thead><tr><th>scale</th><th>dense FFN</th>'
        "<th>best narrow-only</th><th>best full-or-null</th><th>best MoE top-8</th>"
        "<th>best MoE top-2</th><th>null vs MoE top-8</th></tr></thead><tbody>"
    ]
    for scale in SCALES:
        null = json.load(open(os.path.join(HERE, f"ffn_null_rung_{scale}_{tokens}.json")))
        moe = json.load(open(os.path.join(HERE, f"ffn_moe_arm_{scale}_{tokens}.json")))
        rows, mrows = null["rows"], [r for r in moe["rows"] if "anchor" not in r]
        dense = [r["narrow_ms"] for r in rows if r["c"] == 1.0][0]

        def best(cands, key):
            ok = [r for r in cands if r[key] == r[key]]
            return min(ok, key=lambda r: r[key])

        bn, bz = best(rows, "narrow_ms"), best(rows, "null_ms")
        b8 = best([r for r in mrows if r["k"] == 8], moe_key)
        b2 = best([r for r in mrows if r["k"] == 2], moe_key)

        def cell(ms, c):
            return (
                f'<td class="num">{ms:.2f} <span class="num muted">'
                f"({dense / ms:.1f}&times;, c={c:.3f})</span></td>"
            )

        out.append(
            f'<tr><td class="tok-cell">{scale}</td><td class="num">{dense:.2f}</td>'
            + cell(bn["narrow_ms"], bn["narrow_c"])
            + cell(bz["null_ms"], bz["null_c"])
            + cell(b8[moe_key], b8["c"])
            + cell(b2[moe_key], b2["c"])
            + f'<td class="num strong">{b8[moe_key] / bz["null_ms"]:.1f}&times;</td></tr>'
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


if __name__ == "__main__":
    main()
