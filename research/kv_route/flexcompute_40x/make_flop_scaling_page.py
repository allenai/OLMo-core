"""Render `scale_ceiling.json` as research/kv_route/viz/flop_scaling_ceilings.html.

Two log-log families: achievable step speedup against MODEL SCALE (one panel per context) and
against CONTEXT LENGTH (one panel per scale), with the cumulative router stack drawn as four
curves and the untouchable-floor Amdahl cap as a dashed rule. Filled markers are cells whose
partition was measured on hardware; hollow markers are cells the calibrated model fills in.

    python research/kv_route/flexcompute_40x/make_flop_scaling_page.py
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "viz", "flop_scaling_ceilings.html")

SCALE_ORDER = ["0.8B", "4B", "9B", "27B"]
PANEL_CTX = [8192, 16384, 131072]  # 8k and 16k carry the measured anchors; 128k is the far end
BEST = "staticFFN+bskipK-0.25+headgate"
BEST_PACKED = "staticFFN+bskipC-0.25+headgate"
WITHKV = "staticFFN+kv0.25+bskipC-0.25+headgate"
SERIES = [
    ("head", "head gating", "s1"),
    ("head_static", "+ static FFN deletion", "s2"),
    ("head_static_skip", "+ compacted block skip", "s4"),
    ("head_static_skip_kv", "+ KV router (loses)", "s3"),
]

# Plot geometry, matching research/kv_route/viz/ffn_routing_linearity.html.
W, H = 350, 262
X0, X1, Y0, Y1 = 56, 334, 16, 218


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def decades(lo: float, hi: float):
    """Decade gridlines spanning [lo, hi], padded outward to whole decades."""
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    if b - a < 1:
        b = a + 1
    return a, b, [10**k for k in range(a, b + 1)]


def fmt_tick(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.0f}"
    return f"{v:g}"


def svg_panel(
    title: str,
    verdict: str,
    verdict_bad: bool,
    xs_label: str,
    ys_label: str,
    xticks,
    curves,
    cap=None,
    legend="",
    marks=(),
):
    """One log-log panel.

    ``curves`` is a list of dicts: {label, cls, pts=[(x, y, measured)]}. ``cap`` is a list of
    (x, y) for the dashed Amdahl rule.
    """
    allx = [p[0] for c in curves for p in c["pts"]] + [t for t, _ in xticks]
    ally = [p[1] for c in curves for p in c["pts"]] + ([y for _, y in cap] if cap else [])
    ally += [m[1] for m in marks]
    xa, xb, _ = decades(min(allx), max(allx))
    ya, yb, ydec = decades(min(ally), max(ally))

    def px(v):
        return X0 + (math.log10(v) - xa) / (xb - xa) * (X1 - X0)

    def py(v):
        return Y1 - (math.log10(v) - ya) / (yb - ya) * (Y1 - Y0)

    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="log-log chart">',
    ]
    for v in ydec:
        y = py(v)
        out.append(f'<line class="grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>')
        out.append(
            f'<text class="tick" x="{X0-7}" y="{y+3:.1f}" text-anchor="end">{fmt_tick(v)}x</text>'
        )
    for v, lab in xticks:
        x = px(v)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y1}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{Y1+15}" text-anchor="middle">{lab}</text>')
    out.append(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
    out.append(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}"/>')
    out.append(
        f'<text class="axlab" x="{(X0+X1)/2}" y="{H-6}" text-anchor="middle">{xs_label}</text>'
    )
    out.append(f'<text class="axlab" x="9" y="20">{ys_label}</text>')

    if cap:
        d = " ".join(
            ("M" if i == 0 else "L") + f"{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(cap)
        )
        out.append(f'<path class="capline" d="{d}"/>')
        cx, cy = cap[-1]
        out.append(
            f'<text class="guidelab" x="{px(cx)-3:.1f}" y="{py(cy)-5:.1f}" '
            f'text-anchor="end">floor cap</text>'
        )

    for c in curves:
        pts = c["pts"]
        d = " ".join(
            ("M" if i == 0 else "L") + f"{px(x):.1f},{py(y):.1f}" for i, (x, y, _) in enumerate(pts)
        )
        out.append(f'<path class="line {c["cls"]}" d="{d}"/>')
        for x, y, meas in pts:
            tip = f'{c["label"]} &mdash; {c["tip"](x)}, {y:.2f}x' + (
                " (measured partition)" if meas else " (modelled)"
            )
            cls = f'dot {c["cls"]}' if meas else f'dot hollow {c["cls"]}'
            out.append(
                f'<circle class="{cls}" cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4">'
                f"<title>{tip}</title></circle>"
            )
    for mx, my, tip in marks:
        out.append(
            f'<rect class="meas-mark" x="{px(mx)-4.2:.1f}" y="{py(my)-4.2:.1f}" width="8.4" '
            f'height="8.4" transform="rotate(45 {px(mx):.1f} {py(my):.1f})">'
            f"<title>{tip}</title></rect>"
        )
    out.append("</svg>")
    vclass = "vd bad" if verdict_bad else "vd"
    return (
        '<figure class="panel"><figcaption>'
        f'<span class="tok">{esc(title)}</span><span class="{vclass}">{verdict}</span>'
        "</figcaption>" + "\n".join(out) + f'<div class="legend panel-legend">{legend}</div>'
        "</figure>"
    )


COMP = [
    ("floor", "floor (untouchable)", "var(--muted)"),
    ("head_waste", "head waste", "var(--s1)"),
    ("ffn", "FFN", "var(--s2)"),
    ("attn", "attention", "var(--s3)"),
    ("gdn", "GDN", "var(--s4)"),
]


def comp_bars(data, scales, contexts):
    """100%-stacked wall-clock composition, one row per (scale, context).

    This is the mechanism behind both panel families above: a router can only claim the band it
    owns, and the grey band at the left is the one nothing in this family can claim.
    """
    rowh, gap, lab = 15, 4, 74
    ctxs = [c for c in contexts if c in (2048, 8192, 32768, 131072)]
    rows = [(s, c) for s in scales for c in ctxs]
    height = len(rows) * (rowh + gap) + 26
    x0, x1 = lab, 560
    out = [f'<svg viewBox="0 0 {x1+8} {height}" role="img" aria-label="step composition">']
    for i, (s, c) in enumerate(rows):
        y = i * (rowh + gap) + 4
        cell = data[s]["by_context"][str(c)]
        meas = cell["basis"] == "measured"
        out.append(
            f'<text class="tick" x="{lab-8}" y="{y+rowh-4}" text-anchor="end">'
            f"{s} @ {c//1024}k</text>"
        )
        x = x0
        for key, label, colour in COMP:
            w = cell["wall_shares"][key] * (x1 - x0)
            op = "1" if meas else "0.45"
            out.append(
                f'<rect x="{x:.1f}" y="{y}" width="{max(w,0.4):.1f}" height="{rowh}" '
                f'fill="{colour}" opacity="{op}">'
                f"<title>{s} @ {c:,} tokens &mdash; {label} "
                f'{100*cell["wall_shares"][key]:.1f}%'
                f'{"" if meas else " (modelled)"}</title></rect>'
            )
            x += w
    out.append(
        f'<text class="tick" x="{x0}" y="{height-6}">0%</text>'
        f'<text class="tick" x="{x1}" y="{height-6}" text-anchor="end">100% of step wall-clock'
        "</text>"
    )
    out.append("</svg>")
    return "".join(out)


def main():
    data = json.load(open(os.path.join(HERE, "scale_ceiling.json")))
    scales = [s for s in SCALE_ORDER if s in data]
    anchors, val = data["_anchors"], data["_validation"]
    art = data["_basis_artefact"]
    params = {s: data[s]["params_b"] for s in scales}

    def cell(s, T):
        return data[s]["by_context"][str(T)]

    contexts = sorted(int(k) for k in data[scales[0]]["by_context"])

    legend = (
        "".join(
            f'<span class="lg"><span class="key {cls}"></span>{lab}</span>'
            for _, lab, cls in SERIES
        )
        + '<span class="lg"><span class="key dash"></span>floor cap</span>'
        + '<span class="lg"><span class="key hollow"></span>modelled</span>'
    )
    legend_meas = (
        legend + '<span class="lg"><span class="key diamond"></span>MEASURED best stack</span>'
    )

    # ---- panels: speedup vs model scale, one per context ---------------------------------
    scale_panels = []
    for T in PANEL_CTX:
        curves = []
        for key, label, cls in SERIES:
            pts = [
                (params[s], cell(s, T)["stages"][key], cell(s, T)["basis"] == "measured")
                for s in scales
            ]
            curves.append(
                {"label": label, "cls": cls, "pts": pts, "tip": lambda x: f"{x:.2f}B params"}
            )
        cap = [(params[s], cell(s, T)["amdahl_cap"]) for s in scales]
        hits = [(sc, a) for sc, a in anchors.items() if a["ctx"] == T and sc in scales]
        marks = [
            (
                params[sc],
                a["arms"][BEST],
                f'{sc} @ {T:,} MEASURED {a["arms"][BEST]:.2f}x'
                + (" (activation-checkpointed basis)" if a["ac_basis"] else ""),
            )
            for sc, a in hits
        ]
        # How far below the measured anchor the modelled curve sits at the same point. The model
        # never credits compaction with shortening the GDN scan or cutting launches, so it is a
        # floor, and the gap widens with model size.
        gap = max(
            (a["arms"][BEST] / cell(sc, T)["stages"]["head_static_skip"] for sc, a in hits),
            default=0.0,
        )
        scale_panels.append(
            svg_panel(
                f"{T:,} tokens",
                (f"model is {gap:.2f}x low vs measured" if marks else "modelled floor"),
                bool(marks),
                "model params, B (log)",
                "step speedup",
                [(params[s], s) for s in scales],
                curves,
                cap,
                legend_meas if marks else legend,
                marks,
            )
        )

    # ---- panels: speedup vs context, one per scale ---------------------------------------
    ctx_panels = []
    for s in scales:
        curves = []
        for key, label, cls in SERIES:
            pts = [
                (T, cell(s, T)["stages"][key], cell(s, T)["basis"] == "measured") for T in contexts
            ]
            curves.append(
                {"label": label, "cls": cls, "pts": pts, "tip": lambda x: f"{int(x):,} tokens"}
            )
        cap = [(T, cell(s, T)["amdahl_cap"]) for T in contexts]
        h = [cell(s, T)["stages"]["head"] for T in contexts]
        ctx_panels.append(
            svg_panel(
                f"{s} ({params[s]:.2f}B)",
                f"head gating {h[0]:.1f}x &rarr; {h[-1]:.2f}x",
                True,
                "context, tokens (log)",
                "step speedup",
                [(T, f"{T//1024}k") for T in contexts if T in (2048, 8192, 32768, 131072)],
                curves,
                cap,
                legend,
            )
        )

    # ---- table ---------------------------------------------------------------------------
    rows = []
    for s in scales:
        for i, T in enumerate(contexts):
            c = cell(s, T)
            st = c["stages"]
            first = (
                f'<td rowspan="{len(contexts)}" class="tok-cell">{s}<br>'
                f'<span class="sub">{params[s]:.2f}B</span></td>'
                if i == 0
                else ""
            )
            rows.append(
                f"<tr>{first}"
                f'<td class="num">{T:,}</td>'
                f'<td class="src {"meas" if c["basis"] == "measured" else "model"}">'
                f'{c["basis"]}</td>'
                f'<td class="num">{c["step_s"]:.2f}</td>'
                f'<td class="num">{100*c["wall_shares"]["floor"]:.1f}%</td>'
                f'<td class="num">{st["head"]:.2f}x</td>'
                f'<td class="num">{st["head_static"]:.2f}x</td>'
                f'<td class="num strong">{st["head_static_skip"]:.2f}x</td>'
                f'<td class="num bad">{st["head_static_skip_kv"]:.2f}x</td>'
                f'<td class="num muted">{c["amdahl_cap"]:.1f}x</td>'
                "</tr>"
            )

    # ---- measured best per scale, put on one basis ----------------------------------------
    bestrows = []
    for s, a in anchors.items():
        T, is_ac = a["ctx"], a["ac_basis"]
        f = art[s][str(T)]
        equiv = a["arms"][BEST] / f if is_ac else a["arms"][BEST]
        bestrows.append(
            f'<tr><td class="tok-cell">{s}</td>'
            f'<td class="num">{T:,}</td>'
            f'<td>{"activation-checkpointed" if is_ac else "plain"}</td>'
            f'<td class="num">{a["arms"]["headgate"]:.3f}x</td>'
            f'<td class="num">{a["arms"]["ffn-static"]:.3f}x</td>'
            f'<td class="num strong">{a["arms"][BEST]:.3f}x</td>'
            f'<td class="num bad">{a["arms"][WITHKV]:.3f}x</td>'
            f'<td class="num">{equiv:.2f}x</td></tr>'
        )

    valrows = []
    for s, d in val.items():
        for i, (k, v) in enumerate(d.items()):
            if not v.get("measured"):
                continue
            lead = f'<td class="tok-cell">{s}</td>' if i == 0 else "<td></td>"
            bad = " bad" if abs(v["rel_err"]) > 0.15 else ""
            valrows.append(
                f"<tr>{lead}"
                f'<td class="mono">{esc(k)}</td>'
                f'<td class="num">{v["measured"]:.3f}x</td>'
                f'<td class="num">{v["predicted"]:.3f}x</td>'
                f'<td class="num{bad}">{100*v["rel_err"]:+.1f}%</td></tr>'
            )

    # ---- headline cards -------------------------------------------------------------------
    a4, a27 = anchors["4B"]["arms"], anchors["27B"]["arms"]
    pk = data["_packed"]["arms"]
    lb = data["_launch_bound"]
    cards = [
        (
            f"{a27[BEST]:.2f}x",
            "MEASURED on a 27B step at 8k tokens, the best result anywhere in this study: head "
            "gating, static deletion of the always-null FFN layers, and <b>compacted</b> block "
            "skip at keep=0.25. A 13.1x FLOP cut converting at 0.44. 4B/16k reaches "
            f"{a4[BEST]:.2f}x, reproduced across five separate jobs (5.590&ndash;5.693x).",
        ),
        (
            f"{a27[WITHKV]:.2f}x",
            "the same 27B stack with the KV router added. It is <b>redundant, not merely "
            "expensive</b>: compacted block skip already removes a skipped token as a key by "
            "gathering rather than masking, so the KV router re-does the work and charges for it. "
            f"4B behaves the same way ({a4[BEST]:.2f} &rarr; {a4[WITHKV]:.2f}x).",
        ),
        (
            f"{pk[BEST_PACKED]:.2f}x",
            "the same 4B recipe when the sequence is <b>packed</b> into 32 documents &mdash; "
            f"against {a4[BEST]:.2f}x unpacked. Packing already makes attention cheap, so it "
            "roughly halves the prize. Plan real runs against this number.",
        ),
        (
            f'{lb["gpu_idle_best_arm_pct"]:.1f}%',
            f'GPU idle on the best arm, against {lb["gpu_idle_dense_pct"]:.1f}% on a dense step. '
            "Past roughly 5x the step stops being GPU-bound and becomes launch-bound, which makes "
            f'CUDA graphs worth about {lb["cuda_graph_headroom"]:.2f}x &mdash; the next lever, and '
            "the one that decides whether this line goes further.",
        ),
    ]

    css = """
:root {
  --bg:#F4F6F8; --surface:#FFFFFF; --ink:#141D26; --ink-2:#41505E; --muted:#6B7A88;
  --rule:#D7DDE3; --rule-soft:#E8ECEF; --s1:#1B6CA8; --s2:#C4691B; --s3:#7A4FA3;
  --s4:#0F7B8A; --bad:#B33A26; --ok:#1E6B4E; --chip:#EDF1F4;
}
@media (prefers-color-scheme:dark) {
  :root:not([data-theme="light"]) {
    --bg:#11161B; --surface:#171E25; --ink:#E4EAEF; --ink-2:#B3C0CB; --muted:#8494A1;
    --rule:#2A343D; --rule-soft:#212A32; --s1:#4A98CF; --s2:#C97B34; --s3:#A47AC9;
    --s4:#4FB0C0; --bad:#E0765C; --ok:#5FB98F; --chip:#1D262E;
  }
}
:root[data-theme="dark"] {
  --bg:#11161B; --surface:#171E25; --ink:#E4EAEF; --ink-2:#B3C0CB; --muted:#8494A1;
  --rule:#2A343D; --rule-soft:#212A32; --s1:#4A98CF; --s2:#C97B34; --s3:#A47AC9;
  --s4:#4FB0C0; --bad:#E0765C; --ok:#5FB98F; --chip:#1D262E;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:Charter,"Iowan Old Style","Source Serif Pro",Georgia,serif;font-size:16.5px;line-height:1.6}
main{max-width:1080px;margin:0 auto;padding:56px 24px 96px;display:flex;flex-direction:column;gap:30px}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.eyebrow{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.7rem;
 letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
h1{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:2.05rem;font-weight:600;
 letter-spacing:-.02em;margin:.15em 0 0;text-wrap:balance;line-height:1.15}
h2{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:1.02rem;font-weight:600;
 letter-spacing:.01em;margin:0 0 2px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
p{margin:0;max-width:70ch;color:var(--ink-2)}
section{display:flex;flex-direction:column;gap:14px}
.lede{font-size:1.1rem;color:var(--ink-2);max-width:66ch}
.answer{display:flex;flex-wrap:wrap;gap:14px}
.card{flex:1 1 250px;background:var(--surface);border:1px solid var(--rule-soft);
 border-radius:3px;padding:15px 17px;display:flex;flex-direction:column;gap:5px}
.card .big{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:1.55rem;
 font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.card .cap{font-size:.9rem;color:var(--muted);line-height:1.45}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.panel{margin:0;background:var(--surface);border:1px solid var(--rule-soft);border-radius:3px;padding:10px 10px 4px}
figcaption{display:flex;justify-content:space-between;align-items:baseline;gap:8px;
 padding:2px 4px 8px;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.74rem}
.tok{font-weight:600;letter-spacing:.02em}
.vd{color:var(--ok)} .vd.bad{color:var(--bad)}
svg{display:block;width:100%;height:auto}
.grid{stroke:var(--rule-soft);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.tick,.axlab{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:8.5px;fill:var(--muted)}
.line{fill:none;stroke-width:2;stroke-linejoin:round}
.capline{fill:none;stroke:var(--ink-2);stroke-width:1.1;stroke-dasharray:5 3;opacity:.55}
.guidelab{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:8px;fill:var(--muted);opacity:.9}
.meas-mark{fill:none;stroke:var(--ink);stroke-width:1.8}
.key.diamond{background:none;border:1.8px solid var(--ink);width:9px;height:9px;transform:rotate(45deg)}
.dot{stroke:var(--surface);stroke-width:1.6}
.dot.hollow{fill:var(--surface);stroke-width:2}
.s1{stroke:var(--s1)} .s2{stroke:var(--s2)} .s3{stroke:var(--s3)} .s4{stroke:var(--s4)}
circle.s1{fill:var(--s1)} circle.s2{fill:var(--s2)} circle.s3{fill:var(--s3)} circle.s4{fill:var(--s4)}
circle.hollow.s1{fill:var(--surface);stroke:var(--s1)}
circle.hollow.s2{fill:var(--surface);stroke:var(--s2)}
circle.hollow.s3{fill:var(--surface);stroke:var(--s3)}
circle.hollow.s4{fill:var(--surface);stroke:var(--s4)}
.legend{display:flex;flex-wrap:wrap;gap:18px;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 font-size:.76rem;color:var(--ink-2)}
.panel-legend{gap:11px;font-size:.64rem;padding:7px 4px 3px;border-top:1px solid var(--rule-soft);
 margin-top:4px;color:var(--muted)}
.panel-legend .key{width:9px;height:9px}
.lg{display:inline-flex;align-items:center;gap:7px}
.key{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none}
.key.s1{background:var(--s1)} .key.s2{background:var(--s2)} .key.s3{background:var(--s3)}
.key.s4{background:var(--s4)}
.key.hollow{background:var(--surface);border:2px solid var(--muted)}
.key.dash{background:none;border-top:2px dashed var(--ink-2);height:0;border-radius:0;width:14px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--rule-soft);border-radius:3px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:7px 12px;border-bottom:1px solid var(--rule-soft);text-align:left;vertical-align:middle}
th{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.68rem;letter-spacing:.07em;
 text-transform:uppercase;color:var(--muted);font-weight:500;border-bottom:1px solid var(--rule)}
td{font-family:Charter,"Iowan Old Style",Georgia,serif}
.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;text-align:right;font-variant-numeric:tabular-nums}
.num.strong{font-weight:600;color:var(--ink)}
.num.muted{color:var(--muted)}
.src{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.72rem;
 letter-spacing:.04em;text-transform:uppercase}
.src.meas{color:var(--ok)} .src.model{color:var(--muted)}
.tok-cell{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-weight:600;
 border-right:1px solid var(--rule-soft);vertical-align:top;padding-top:10px}
.tok-cell .sub{font-weight:400;color:var(--muted);font-size:.78rem}
.callout{background:var(--chip);border-left:2px solid var(--s2);padding:13px 17px;border-radius:0 3px 3px 0;
 font-size:.97rem;color:var(--ink-2);max-width:78ch}
.note{display:flex;gap:9px;align-items:baseline;font-size:.93rem;color:var(--ink-2);max-width:76ch}
footer{border-top:1px solid var(--rule);padding-top:14px;color:var(--muted);font-size:.83rem;max-width:80ch}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.85em;background:var(--chip);
 padding:1px 5px;border-radius:2px}
"""

    meas_note = "; ".join(
        f"{s}: "
        + (
            ", ".join(
                f"{int(t)//1024}k"
                for t, b in data[s]["basis_by_context"].items()
                if b == "measured"
            )
            or "none (activation checkpointing only)"
        )
        for s in scales
    )

    complegend = "".join(
        f'<span class="lg"><span class="key" style="background:{col}"></span>{lab}</span>'
        for _, lab, col in COMP
    )
    compbars = comp_bars(data, scales, contexts)

    html = f"""<title>Where the Router Speedup Runs Out</title>
<style>{css}</style>
<main>
<header>
  <div class="eyebrow">Qwen3.5 &middot; one H200 &middot; head gating + three compute routers &middot; all curves log&ndash;log</div>
  <h1>Where the Router Speedup Runs Out</h1>
  <p class="lede">The 40x question priced in <em>wall-clock</em> rather than FLOPs, across four
  model sizes and seven context lengths. Times are measured medians of a real forward+backward
  step; FLOP counts are calculated; the cells the measurement does not reach are filled by a
  calibrated model and drawn hollow.</p>
</header>

<section><h2>The answer</h2><div class="answer">
{"".join(f'<div class="card"><div class="big">{b}</div><div class="cap">{c}</div></div>' for b, c in cards)}
</div>
<div class="callout"><b>The FLOP cut is not the objective, and following it picks the wrong
stack.</b> Adding the KV router raises the FLOP cut from 13.1x to 13.6x and <em>lowers</em> the
measured speed. Ordering the levers by what the hardware pays for gives: head gating, then
<b>static deletion</b> of the FFN layers the trained router leaves permanently null, then
<b>compacted</b> block skip. Two implementation fixes produced most of the gain and neither changed
a FLOP: block skip became a <em>gather</em> instead of a mask (0.99&rarr;2.27x at keep=0.25), and a
keep-all flash fast path plus split full/partial block-mask tiles removed the kept-key mask's 1.3x
entry cost (kv keep=1.0: 0.753&rarr;0.993x). What is left is bounded by a floor &mdash; 4&ndash;10%
of every step is embedding lookups, residual adds, the final norm, <code>zero_grad</code>,
inter-kernel gaps and the part of the LM head that survives gating &mdash; which nothing here
touches. That is the dashed rule in every panel.</div></section>

<section><h2>Measured, per scale</h2>
<p>Three end-to-end router matrices, one per scale, each interleaved with a dense control
re-measured every round (drift 1.001&ndash;1.016x). <b>Only 4B is on the plain basis</b>: 9B does
not fit on one H200 without activation checkpointing past 8k and 27B does not fit at all, so their
matrices ran under checkpointing, which runs each block's forward twice and depresses the measured
speedup. The <span class="mono">plain-equiv</span> column removes that using the per-cell basis
factor this page's model computes for the same recipe (0.94&ndash;0.97x).</p>
<div class="tablewrap"><table>
<thead><tr><th>model</th><th class="num">ctx</th><th>basis</th><th class="num">head gate</th>
<th class="num">static FFN</th><th class="num">best stack</th><th class="num">+ KV router</th>
<th class="num">plain-equiv</th></tr></thead>
<tbody>{"".join(bestrows)}</tbody></table></div>
<p><b>27B is the best scale, not the worst.</b> An earlier revision of this page said the opposite,
and that was a bug rather than a finding: under activation checkpointing the model's
<code>CheckpointWrapper</code> forwards attribute reads to the wrapped block but not writes, so
assigning to <code>wrapper.feed_forward</code> registered a new submodule nothing read and the
static-FFN arm silently measured <em>nothing</em> at 9B and 27B. Fixed, 27B's static-FFN arm goes
1.000x &rarr; 1.604x and its best stack 3.79x &rarr; 5.72x.</p>
<p>The head-gating mechanism survives that correction and still runs opposite to the routers: it
saves a roughly <em>fixed</em> 0.23&ndash;0.36&nbsp;s (the fused-linear-CE backward touches a full
<code>d_model &times; vocab</code> fp32 accumulator regardless of how many rows survive), so it is
worth 1.29x at 4B and 1.10x at 27B. It simply no longer decides the ordering. The routers' own
return &mdash; best stack divided by head gating alone &mdash; is 4.39x at 4B, 3.54x at 9B and
<b>5.22x at 27B</b>, and now points the same way as the total. The FFN is 46% of a 27B step against
22% of a 4B one, and that is what the recipe is buying.</p></section>

<section><h2>Two caveats that change what you would plan</h2>
<div class="grid3" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">
<div class="card"><div class="big">{pk[BEST_PACKED]:.2f}x <span class="cap">packed</span></div>
<div class="cap">Same 4B model, same 16k context, same recipe &mdash; but the sequence packed into
32 documents, which is how real training runs. Dense step falls 1417&nbsp;ms &rarr; 1121&nbsp;ms
because packing already makes attention cheap, and the recipe's prize falls with it:
{a4[BEST]:.2f}x unpacked &rarr; {pk[BEST_PACKED]:.2f}x packed, conversion 0.54 &rarr; 0.38. Head
gating goes the other way ({a4["headgate"]:.2f}x &rarr; {pk["headgate"]:.2f}x), because the head is
untouched by packing while everything around it got cheaper. <b>Plan against the packed
number.</b></div></div>
<div class="card"><div class="big">{lb["gpu_idle_best_arm_pct"]:.1f}% <span class="cap">GPU idle</span></div>
<div class="cap">Past roughly 5x the step is no longer GPU-bound. Idle goes
{lb["gpu_idle_dense_pct"]:.1f}% on a dense step &rarr; {lb["gpu_idle_best_arm_pct"]:.1f}% on the
best arm: the levers removed enough GPU work that the CPU cannot issue kernels fast enough to keep
up. That makes CUDA graphs worth about {lb["cuda_graph_headroom"]:.2f}x and puts them next in line
&mdash; the constraint has moved from FLOPs to launches, and no further FLOP removal will pay until
it moves back.</div></div>
</div></section>

<section><h2>The anchors, and what the model gets wrong</h2>
<p>Three coefficients are read off those matrices &mdash; what static deletion removes, what the
compacted gather/scatter costs, and what the KV router does when added to the winning stack &mdash;
and the composition is then checked against every arm at every anchor. It is <b>conservative
everywhere</b>, and increasingly so with model size: &minus;3.7% at 9B, &minus;17% at 4B/16k,
&minus;28% at 27B. The model never credits compaction with shortening the GDN scan or cutting
kernel launches, and at 27B both matter. Treat the modelled grid below as a floor and quote the
measured anchors above as the result &mdash; the diamonds on the scale panels show the gap
directly.</p>
<div class="tablewrap"><table>
<thead><tr><th>model</th><th>measured arm</th><th class="num">measured</th>
<th class="num">this model</th><th class="num">error</th></tr></thead>
<tbody>{"".join(valrows)}</tbody></table></div></section>

<section><h2>Against model scale</h2>
<p>Each panel is one context length; the four curves stack the routers cumulatively, in the order
they were built. A curve that rises to the right is a lever that gets <em>better</em> with model
size. Filled markers are scales where the step partition was measured on hardware.</p>
<div class="grid3">{"".join(scale_panels)}</div></section>

<section><h2>Against context length</h2>
<p>The same stack read the other way. Head gating is a fixed number of milliseconds &mdash; the
fused-linear-CE backward touches a full <code>d_model &times; vocab</code> fp32 accumulator whether
one row survives or a million &mdash; so it is a large multiple of a short step and almost nothing
on a long one. The FFN router runs the opposite way: its own launch floor and per-layer prologue
dominate until there are enough tokens to amortise them. Notice that the <span class="mono">+KV</span>
curve sits <em>below</em> <span class="mono">+FFN</span> at short context on every scale: that is
the measured masking overhead exceeding the FLOPs the router removes, not a plotting error.</p>
<div class="grid3">{"".join(ctx_panels)}</div></section>

<section><h2>What the step is actually made of</h2>
<p>Every number above follows from this one picture. A router can only ever claim the band it
owns; the grey band on the left is the one nothing in this family can claim, and it is why the
dashed cap sits where it does. Faded rows are modelled rather than measured.</p>
<div class="legend">{complegend}</div>
<div class="panel">{compbars}</div></section>

<section><h2>The table</h2>
<p>Operating point throughout: head gating, static deletion of the always-null FFN layers
(65% of the FFN component, measured), compacted block skip at keep=0.25. The
<span class="mono">+KV router</span> column is shown in red because it is where the stack gets
<em>slower</em>. The <span class="mono">partition</span> column says how that row's step was
obtained: <span class="mono">measured</span> on hardware without activation checkpointing;
<span class="mono">converted</span> measured under checkpointing and converted onto the plain basis
by a measured per-component ratio; <span class="mono">modelled</span> neither. The rightmost column
is the cap that nothing in this family can pass.</p>
<div class="tablewrap"><table>
<thead><tr><th>model</th><th class="num">ctx</th><th>partition</th><th class="num">step s</th>
<th class="num">floor</th><th class="num">head gate</th><th class="num">+static FFN</th>
<th class="num">+block skip</th><th class="num">+KV router</th><th class="num">cap</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></section>

<footer>Generated by <code>research/kv_route/flexcompute_40x/make_flop_scaling_page.py</code> from
<code>scale_ceiling.json</code>. Measured inputs: <code>bench_step_partition.py</code> (cumulative
ablation of a real step, summing to 100% by construction), and three end-to-end router matrices
&mdash; <code>router_matrix_4B_16384_v4.json</code> (plain),
<code>router_matrix_9B_8192_ac_v3.json</code> and <code>router_matrix_27B_8192_ac_v2.json</code>
(activation-checkpointed), p10/median/p90 over 15 iterations with a dense control re-measured every round (drift 1.001&ndash;1.016x); the packed comparison is <code>router_matrix_4B_16384_v3_packed32.json</code>. Layer-pruning
evidence from <code>rungs_v11-t05</code> / <code>rungs_v12-t02</code>: 13 of 36 and 22 of 36 layers
send &gt;99% of tokens to the null rung. Partition measured without activation checkpointing at
&mdash; {esc(meas_note)}; every other cell was measured under checkpointing and converted onto the
plain basis by a measured per-component ratio, or modelled. <b>27B has no plain-basis cell at
all</b>: it does not fit on one 141&nbsp;GB H200 without checkpointing, and with it peaks at
104&ndash;124&nbsp;GiB and stops fitting past 32k, so every 27B row here is extrapolated onto the
plain basis. 9B stops fitting without checkpointing past 8k and 4B past 16k. Batch 1, bf16, no
compile. Head gating is exactly loss-neutral; static FFN deletion and block skip are mechanism
limits whose task quality has not been established.</footer>
</main>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"# wrote {OUT} ({len(html)/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
