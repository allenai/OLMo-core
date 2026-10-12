#!/usr/bin/env python3
"""
Rebuild the two published flexcompute artifact pages from the measured JSON in
``research/kv_route/flexcompute_40x/`` after the ``--ffn-start-layer`` fix (every FFN layer routed).

Outputs (overwritten in place, same artifact URLs):
  research/kv_route/viz/ffn_routing_linearity.html   "Does Routed FFN Compute Buy Speed?"
  research/kv_route/viz/flop_scaling_ceilings.html   "Where the Router Speedup Runs Out"

Sources of truth: research/kv_route/flexcompute_40x/README.md (newest section),
joint_sweep_4B_16384_v1.json / _packed32.json, joint_sweep_27B_8192_v1.json,
joint_sweep_4B_16384_shell.log (per-component percentages only), graph_probe_*.json.

Nothing in research/kv_route/flexcompute_40x/ is written by this script.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Iterable, Sequence

KERNEL_OPEN = "<!--KERNEL-SECTIONS-->"
KERNEL_CLOSE = "<!--/KERNEL-SECTIONS-->"

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.join(HERE, "..", "flexcompute_40x")  # measured result JSONs live here (not in git; see README)


def load(name: str) -> dict:
    with open(os.path.join(DBG, name)) as f:
        return json.load(f)


J4 = load("joint_sweep_4B_16384_v1.json")
J4P = load("joint_sweep_4B_16384_packed32.json")
J27 = load("joint_sweep_27B_8192_v1.json")


def rows(sweep: dict) -> dict:
    return {r["arm"]: r for r in sweep["matrix"]["rows"]}


R4, R4P, R27 = rows(J4), rows(J4P), rows(J27)


def dense_ms(sweep: dict) -> float:
    """The dense arm's own median step, which is what every ``wall_speedup`` divides."""
    return 1000.0 * rows(sweep)["dense"]["median"]


DENSE4, DENSE4P, DENSE27 = dense_ms(J4), dense_ms(J4P), dense_ms(J27)

# ---------------------------------------------------------------- palette / shell

CSS = """
/* Categorical series slots validated with the dataviz skill's
   scripts/validate_palette.js: s1..s3 pass --pairs all in BOTH modes; the
   4th hue is only ever used in the stacked composition bar, where the
   adjacent pairlist applies and it passes there in both modes. --sn is a
   deliberate neutral for the residual bucket, not a categorical hue. */
:root {
  --bg:#F4F6F8; --surface:#FFFFFF; --ink:#141D26; --ink-2:#41505E; --muted:#6B7A88;
  --rule:#D7DDE3; --rule-soft:#E8ECEF;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --sn:#6B7A88;
  --o1:#86b6ef; --o2:#2a78d6; --o3:#104281;
  --on1:#FFFFFF; --on2:#3A1400; --on3:#04301F; --on4:#3A2B00; --onn:#FFFFFF;
  --bad:#B33A26; --ok:#1E6B4E; --chip:#EDF1F4;
}
@media (prefers-color-scheme:dark) {
  :root:not([data-theme="light"]) {
    --bg:#11161B; --surface:#171E25; --ink:#E4EAEF; --ink-2:#B3C0CB; --muted:#8494A1;
    --rule:#2A343D; --rule-soft:#212A32;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --sn:#8494A1;
    --o1:#9ec5f4; --o2:#3987e5; --o3:#1c5cab;
    --on1:#FFFFFF; --on2:#FFFFFF; --on3:#04301F; --on4:#241800; --onn:#11161B;
    --bad:#E0765C; --ok:#5FB98F; --chip:#1D262E;
  }
}
:root[data-theme="dark"] {
  --bg:#11161B; --surface:#171E25; --ink:#E4EAEF; --ink-2:#B3C0CB; --muted:#8494A1;
  --rule:#2A343D; --rule-soft:#212A32;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --sn:#8494A1;
  --o1:#9ec5f4; --o2:#3987e5; --o3:#1c5cab;
  --on1:#FFFFFF; --on2:#FFFFFF; --on3:#04301F; --on4:#241800; --onn:#11161B;
  --bad:#E0765C; --ok:#5FB98F; --chip:#1D262E;
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
.card{flex:1 1 228px;background:var(--surface);border:1px solid var(--rule-soft);
 border-radius:3px;padding:15px 17px;display:flex;flex-direction:column;gap:5px}
.card .big{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:1.55rem;
 font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.card .cap{font-size:.9rem;color:var(--muted);line-height:1.45}
.card .tagline{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.62rem;
 letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.card .tagline.meas{color:var(--ok)} .card .tagline.calc{color:var(--s3)}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
/* a lone panel must not stretch to 1080px -- the SVG is authored at ~350px wide and
   scales its type with it, which is what made the labels collide. */
.solo{max-width:560px}
.panel{margin:0;background:var(--surface);border:1px solid var(--rule-soft);border-radius:3px;padding:10px 10px 4px}
figcaption{display:flex;justify-content:space-between;align-items:baseline;gap:8px;
 padding:2px 4px 8px;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.74rem}
.tok{font-weight:600;letter-spacing:.02em}
.vd{color:var(--ok)} .vd.bad{color:var(--bad)} .vd.neutral{color:var(--muted)}
svg{display:block;width:100%;height:auto}
.grid{stroke:var(--rule-soft);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.tick,.axlab{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:8.5px;fill:var(--muted)}
.vallab{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:8px;fill:var(--ink-2);
 font-variant-numeric:tabular-nums}
.line{fill:none;stroke-width:2;stroke-linejoin:round}
.line.dashed{stroke-dasharray:5 3}
.capline{fill:none;stroke:var(--ink-2);stroke-width:1.1;stroke-dasharray:5 3;opacity:.55}
/* used by the retained kernel-level sections */
.actfloor{stroke:var(--ink-2);stroke-width:1;stroke-dasharray:5 3;opacity:.5}
.guide{stroke:var(--muted);stroke-width:1;stroke-dasharray:1 3;opacity:.75}
.guidelab{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:8px;fill:var(--muted);opacity:.9}
.dot{stroke:var(--surface);stroke-width:1.6}
.dot.hollow{fill:var(--surface);stroke-width:2}
.s1{stroke:var(--s1)} .s2{stroke:var(--s2)} .s3{stroke:var(--s3)} .s4{stroke:var(--s4)}
.sn{stroke:var(--sn)} .o1{stroke:var(--o1)} .o2{stroke:var(--o2)} .o3{stroke:var(--o3)}
.ink{stroke:var(--ink-2)}
circle.s1{fill:var(--s1)} circle.s2{fill:var(--s2)} circle.s3{fill:var(--s3)} circle.s4{fill:var(--s4)}
circle.sn{fill:var(--sn)}
circle.o1{fill:var(--o1)} circle.o2{fill:var(--o2)} circle.o3{fill:var(--o3)}
circle.ink{fill:var(--ink-2)}
circle.hollow.s1{fill:var(--surface);stroke:var(--s1)}
circle.hollow.s2{fill:var(--surface);stroke:var(--s2)}
circle.hollow.s3{fill:var(--surface);stroke:var(--s3)}
circle.hollow.s4{fill:var(--surface);stroke:var(--s4)}
circle.hollow.sn{fill:var(--surface);stroke:var(--sn)}
circle.hollow.ink{fill:var(--surface);stroke:var(--ink-2)}
circle.hollow.o1{fill:var(--surface);stroke:var(--o1)}
circle.hollow.o2{fill:var(--surface);stroke:var(--o2)}
circle.hollow.o3{fill:var(--surface);stroke:var(--o3)}
rect.s1{fill:var(--s1)} rect.s2{fill:var(--s2)} rect.s3{fill:var(--s3)} rect.s4{fill:var(--s4)}
rect.sn{fill:var(--sn)} rect.neutral{fill:var(--sn)}
.onbar{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:8px;
 font-variant-numeric:tabular-nums;font-weight:600;stroke:none}
.onbar.s1{fill:var(--on1)} .onbar.s2{fill:var(--on2)} .onbar.s3{fill:var(--on3)}
.onbar.s4{fill:var(--on4)} .onbar.sn{fill:var(--onn)}
.bar{stroke:var(--surface);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:18px;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 font-size:.76rem;color:var(--ink-2)}
.panel-legend{gap:11px;font-size:.64rem;padding:7px 4px 3px;border-top:1px solid var(--rule-soft);
 margin-top:4px;color:var(--muted)}
.panel-legend .key{width:9px;height:9px}
.lg{display:inline-flex;align-items:center;gap:7px}
.key{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none}
.key.s1{background:var(--s1)} .key.s2{background:var(--s2)} .key.s3{background:var(--s3)}
.key.s4{background:var(--s4)} .key.sn,.key.neutral{background:var(--sn)}
.key.o1{background:var(--o1)} .key.o2{background:var(--o2)} .key.o3{background:var(--o3)}
.key.dash{background:none;border-top:2px dashed var(--ink-2);height:0;border-radius:0;width:14px}
.key.dash.s1{border-top-color:var(--s1)} .key.dash.s2{border-top-color:var(--s2)}
.key.dash.s3{border-top-color:var(--s3)} .key.dash.s4{border-top-color:var(--s4)}
.key.dash.o1{border-top-color:var(--o1)} .key.dash.o3{border-top-color:var(--o3)}
.key.dash.ink{border-top-color:var(--ink-2)}
.key.hollow{background:var(--surface);border:2px solid var(--muted)}
.key.hollow.s1{border-color:var(--s1)} .key.hollow.s3{border-color:var(--s3)}
.note{display:flex;gap:9px;align-items:baseline;font-size:.93rem;color:var(--ink-2);max-width:76ch}
.note .key{position:relative;top:1px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--rule-soft);border-radius:3px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:7px 12px;border-bottom:1px solid var(--rule-soft);text-align:left;vertical-align:middle}
th{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.68rem;letter-spacing:.07em;
 text-transform:uppercase;color:var(--muted);font-weight:500;border-bottom:1px solid var(--rule)}
td{font-family:Charter,"Iowan Old Style",Georgia,serif}
.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;text-align:right;font-variant-numeric:tabular-nums}
.num.strong{font-weight:600;color:var(--ink)}
.num.muted{color:var(--muted)}
.num.was{color:var(--muted);text-decoration:line-through}
.src{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.72rem;
 letter-spacing:.04em;text-transform:uppercase}
.src.meas{color:var(--ok)} .src.calc{color:var(--s3)} .src.model{color:var(--muted)}
.tok-cell{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-weight:600;
 border-right:1px solid var(--rule-soft);vertical-align:top;padding-top:10px}
.tok-cell .sub{font-weight:400;color:var(--muted);font-size:.78rem}
.bad{color:var(--bad);font-weight:600}
.pill{display:inline-block;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.62rem;
 letter-spacing:.08em;text-transform:uppercase;padding:1px 6px;border-radius:2px;
 border:1px solid var(--rule);color:var(--muted);white-space:nowrap}
.pill.train{border-color:var(--ok);color:var(--ok)}
.pill.deploy{border-color:var(--s2);color:var(--s2)}
.callout{background:var(--chip);border-left:2px solid var(--s2);padding:13px 17px;border-radius:0 3px 3px 0;
 font-size:.97rem;color:var(--ink-2);max-width:78ch}
.callout.warn{border-left-color:var(--bad)}
footer{border-top:1px solid var(--rule);padding-top:14px;color:var(--muted);font-size:.83rem;max-width:80ch}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.85em;background:var(--chip);
 padding:1px 5px;border-radius:2px}
"""


def page(title: str, eyebrow: str, h1: str, lede: str, body: str, footer: str) -> str:
    return (
        f"<title>{title}</title>\n<style>{CSS}</style>\n<main>\n<header>\n"
        f'  <div class="eyebrow">{eyebrow}</div>\n  <h1>{h1}</h1>\n'
        f'  <p class="lede">{lede}</p>\n</header>\n{body}\n'
        f"<footer>{footer}</footer>\n</main>\n"
    )


# ---------------------------------------------------------------- svg primitives

PW, PH = 350, 262
PX0, PX1, PY0, PY1 = 58, 334, 16, 218


def _lin(v, a, b, o0, o1):
    return o0 + (v - a) / (b - a) * (o1 - o0)


class LogLog:
    """Log-x / log-y panel with the same grammar as the earlier pages."""

    def __init__(self, xmin, xmax, ymin, ymax, xlab, ylab, w=PW, h=PH, logx=True):
        self.lx0, self.lx1 = math.log10(xmin), math.log10(xmax)
        self.ly0, self.ly1 = math.log10(ymin), math.log10(ymax)
        self.logx = logx
        self.xmin, self.xmax = xmin, xmax
        self.w, self.h = w, h
        self.x1 = w - 16
        self.parts: list[str] = []
        self.xlab, self.ylab = xlab, ylab

    def X(self, v):
        if self.logx:
            return _lin(math.log10(v), self.lx0, self.lx1, PX0, self.x1)
        return _lin(v, self.xmin, self.xmax, PX0, self.x1)

    def Y(self, v):
        return _lin(math.log10(v), self.ly0, self.ly1, PY1, PY0)

    def frame(self, xticks, yticks):
        for v, lab in yticks:
            y = self.Y(v)
            self.parts.append(
                f'<line class="grid" x1="{PX0}" y1="{y:.1f}" x2="{self.x1}" y2="{y:.1f}"/>'
            )
            self.parts.append(
                f'<text class="tick" x="{PX0 - 7}" y="{y + 3:.1f}" text-anchor="end">{lab}</text>'
            )
        for v, lab in xticks:
            x = self.X(v)
            self.parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{PY0}" x2="{x:.1f}" y2="{PY1}"/>'
            )
            self.parts.append(
                f'<text class="tick" x="{x:.1f}" y="{PY1 + 15}" text-anchor="middle">{lab}</text>'
            )
        self.parts.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{self.x1}" y2="{PY1}"/>')
        self.parts.append(f'<line class="axis" x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}"/>')
        mid = (PX0 + self.x1) / 2
        self.parts.append(
            f'<text class="axlab" x="{mid:.0f}" y="{self.h - 6}" text-anchor="middle">{self.xlab}</text>'
        )
        self.parts.append(f'<text class="axlab" x="4" y="9">{self.ylab}</text>')
        return self

    def series(self, pts, cls, name, hollow=False, dashed=False, r=3.4, fmt="{:.3g}"):
        if not pts:
            return self
        d = " ".join(
            ("M" if i == 0 else "L") + f"{self.X(x):.1f},{self.Y(y):.1f}"
            for i, (x, y, _) in enumerate(pts)
        )
        dsh = " dashed" if dashed else ""
        self.parts.append(f'<path class="line {cls}{dsh}" d="{d}"/>')
        hol = " hollow" if hollow else ""
        for x, y, tip in pts:
            self.parts.append(
                f'<circle class="dot{hol} {cls}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}">'
                f"<title>{name} &mdash; {tip}</title></circle>"
            )
        return self

    def hrule(self, y, label, cls="s2", anchor="start", below=False):
        yy = self.Y(y)
        self.parts.append(
            f'<line class="line {cls} dashed" x1="{PX0}" y1="{yy:.1f}" x2="{self.x1}" y2="{yy:.1f}"'
            f' stroke-width="1.4" opacity="0.85"/>'
        )
        x = PX0 + 5 if anchor == "start" else self.x1 - 3
        dy = 10.5 if below else -4
        self.parts.append(
            f'<text class="guidelab" x="{x:.0f}" y="{yy + dy:.1f}" text-anchor="{anchor}">{label}</text>'
        )
        return self

    def label(self, x, y, text, dx=0, dy=-8, anchor="middle"):
        self.parts.append(
            f'<text class="vallab" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}"'
            f' text-anchor="{anchor}">{text}</text>'
        )
        return self

    def svg(self, aria):
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{aria}">\n'
            + "\n".join(self.parts)
            + "\n</svg>"
        )


def legend(items: Sequence[tuple[str, str]], panel=False) -> str:
    cls = "legend panel-legend" if panel else "legend"
    inner = "".join(f'<span class="lg"><span class="key {k}"></span>{v}</span>' for k, v in items)
    return f'<div class="{cls}">{inner}</div>'


def figure(cap_left: str, cap_right: str, verdict_cls: str, svg: str, leg: str = "") -> str:
    return (
        f'<figure class="panel"><figcaption><span class="tok">{cap_left}</span>'
        f'<span class="vd {verdict_cls}">{cap_right}</span></figcaption>{svg}{leg}</figure>'
    )


def table(head: Sequence[str], body_rows: Iterable[str]) -> str:
    th = "".join(
        f'<th class="num">{h[1:]}</th>' if h.startswith("#") else f"<th>{h}</th>" for h in head
    )
    return (
        f'<div class="tablewrap"><table>\n<thead><tr>{th}</tr></thead>\n<tbody>'
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


# ---------------------------------------------------------------- shared measured data

FFN_BUCKET_4 = J4["buckets"]["ffn"]
FFN_BUCKET_27 = J27["buckets"]["ffn"]


def mean_ffn_compute(sweep: dict, arm: str) -> float:
    """Mean FFN compute per token as a fraction of the dense FFN, from the FLOP ledger."""
    r = rows(sweep)[arm]
    null = rows(sweep)["ffn-fnull1.00"]["flops_per_token"]
    ffn = sweep["buckets"]["ffn"]
    return (r["flops_per_token"] - null) / ffn


FNULL_ARMS = ["ffn-fnull0.00", "ffn-fnull0.50", "ffn-fnull0.75", "ffn-fnull0.90"]

# what the two published pages said before the --ffn-start-layer fix (12 of 32 FFNs dense)
WAS_4 = {
    "ffn-fnull0.00": 1.101,
    "ffn-fnull0.50": 1.129,
    "ffn-fnull0.75": 1.140,
    "ffn-fnull1.00": 1.177,
    "ffn-static": 1.200,
    "stack": 5.650,
}
WAS_27 = {
    "ffn-fnull0.00": 1.269,
    "ffn-fnull0.50": 1.344,
    "ffn-fnull0.75": 1.377,
    "ffn-fnull1.00": 1.524,
    "ffn-static": 1.604,
    "stack": 5.720,
}

# 4B step partition WITH the block-shell stage: percentages read off
# joint_sweep_4B_16384_shell.log (its own floor=/cap_from_here= fields are stale and unused).
PART4 = [
    # arm, label, step_ms, head, ffn, attn, gdn
    ("dense", "dense", 1418.5, 22.5, 26.3, 21.8, 20.8),
    ("J-f0.50-k0.5000", "trainable &middot; keep 0.5", 919.8, 34.1, 24.8, 11.3, 16.4),
    ("J-f1.00-k0.2500", "trainable &middot; keep 0.25", 663.9, 48.1, 18.9, 6.2, 14.0),
    ("J-f1.00-k0.0625", "trainable &middot; keep 1/16", 538.8, 59.4, 9.7, 1.9, 17.1),
    (
        "staticFFN+bskipK-0.25+headgate",
        "static FFN &middot; keep 0.25",
        540.5,
        59.4,
        -0.1,
        7.7,
        17.2,
    ),
    (
        "staticFFN+bskipK-0.0625+headgate",
        "static FFN &middot; keep 1/16",
        494.4,
        64.3,
        0.8,
        3.3,
        18.8,
    ),
]
PART27 = []
for r in J27["partition"]["rows"]:
    lab = {
        "dense": "dense",
        "J-f0.50-k0.5000": "trainable &middot; keep 0.5",
        "J-f1.00-k0.2500": "trainable &middot; keep 0.25",
        "J-f1.00-k0.0625": "trainable &middot; keep 1/16",
        "staticFFN+bskipK-0.25+headgate": "static FFN &middot; keep 0.25",
    }[r["arm"]]
    PART27.append(
        (
            r["arm"],
            lab,
            r["t_full_s"] * 1000.0,
            r["head_waste_pct"],
            r["ffn_pct"],
            r["attn_pct"],
            r["gdn_pct"],
        )
    )

KEEP_OF = {
    "dense": 1.0,
    "J-f0.50-k0.5000": 0.5,
    "J-f1.00-k0.2500": 0.25,
    "J-f1.00-k0.0625": 0.0625,
    "staticFFN+bskipK-0.25+headgate": 0.25,
    "staticFFN+bskipK-0.0625+headgate": 0.0625,
}


def part_derived(part, dense_step):
    """floor ms, cap-from-dense and per-component ms for a partition table."""
    out = []
    for arm, lab, step, head, ffn, attn, gdn in part:
        resid = 100.0 - (head + ffn + attn + gdn)
        floor = step * resid / 100.0
        out.append(
            dict(
                arm=arm,
                label=lab,
                step=step,
                head=head,
                ffn=ffn,
                attn=attn,
                gdn=gdn,
                resid=resid,
                floor=floor,
                cap=dense_step / floor,
                ffn_ms=step * ffn / 100.0,
                attn_ms=step * attn / 100.0,
                gdn_ms=step * gdn / 100.0,
                keep=KEEP_OF[arm],
            )
        )
    return out


P4 = part_derived(PART4, PART4[0][2])
P27 = part_derived(PART27, PART27[0][2])

FLOOR4 = J4["floor"]["rows"]
HEADROWS4 = J4["floor"]["head_rows"]
FIT4 = J4["floor"]["fits"]

GRAPH4 = {r["arm"]: r for r in J4["graph"]["rows"]}

ROUTED_FFN_ARMS = [
    "ffn-overhead",
    "ffn-fnull0.00",
    "ffn-fnull0.50",
    "ffn-fnull0.75",
    "ffn-fnull0.90",
    "ffn-fnull1.00",
]


def worst_launch_ratio(R: dict) -> float:
    """Most kernel launches any routed-FFN arm issues, as a multiple of dense. MEASURED."""
    d = R["dense"]["launches"]
    return max(R[a]["launches"] for a in ROUTED_FFN_ARMS) / d


def conv_range(R: dict) -> tuple[float, float]:
    """Min/max FLOP-to-wall conversion across the f_null sweep. CALCULATED from the ledger."""
    v = [R[a]["conversion"] for a in FNULL_ARMS + ["ffn-fnull1.00"]]
    return min(v), max(v)


LR4, LR27 = worst_launch_ratio(R4), worst_launch_ratio(R27)
CV4, CV27 = conv_range(R4), conv_range(R27)

# The framing caveat both pages must carry above the fold: these are pinned-operating-point
# costs, not speedups a training run reaches on its own.
PINNED = (
    '<div class="callout warn"><b>These are pinned-operating-point costs, not achieved training '
    "speedups.</b> Every number on this page is what the hardware does when the token "
    "distribution is <em>forced</em> to a given sparsity. A parallel real-training run shows that a "
    "learned router under a budget hinge does <b>not</b> reach its target sparsity inside a short "
    "(31-step) SFT epoch, so a training job that merely switches these routers on will not realise "
    "these numbers. Read them as the wall-clock price of an operating point &mdash; an upper bound "
    "on what routing could pay if a router got there &mdash; and never as a measured training "
    "speedup.</div>"
)


# ================================================================ page 1


def build_ffn_page() -> str:
    src = os.path.join(HERE, "ffn_routing_linearity.html")
    with open(src) as f:
        old = f.read()
    # Keep the four kernel-level sections verbatim: they are per-FFN-kernel measurements
    # and are untouched by the --ffn-start-layer bug (they never went through the layer
    # selector). Fenced with a marker so re-running this script is idempotent -- without
    # it the second run would also pick up the model-level sections written by the first.
    if KERNEL_OPEN in old:
        kernel_sections = old.split(KERNEL_OPEN, 1)[1].split(KERNEL_CLOSE, 1)[0]
    else:
        keep = re.findall(r"<section>.*?</section>", old, flags=re.S)
        assert len(keep) == 4, f"expected 4 kernel sections, found {len(keep)}"
        kernel_sections = "".join(keep)
    assert kernel_sections.count("<section>") == 4, "kernel sections lost"
    # Same one-line axis-label nudge applied to the new panels: at x=9,y=20 the y-axis label
    # sits on top of the topmost y tick. Move it above the plot area instead.
    kernel_sections = kernel_sections.replace(
        '<text class="axlab" x="9" y="20">', '<text class="axlab" x="4" y="9">'
    )
    kernel_sections = KERNEL_OPEN + kernel_sections + KERNEL_CLOSE

    # --- corrected model-level cards
    f1_4, f1_27 = R4["ffn-fnull1.00"], R27["ffn-fnull1.00"]
    st_4, st_27 = R4["ffn-static"], R27["ffn-static"]
    ov_4, ov_27 = R4["ffn-overhead"], R27["ffn-overhead"]
    cards = (
        '<div class="answer">'
        f'<div class="card"><div class="tagline meas">measured &middot; trainable</div>'
        f'<div class="big">{f1_4["wall_speedup"]:.3f}x</div><div class="cap">4B at 16k, every token on the'
        f" null rung, <b>all 32 of 32 FFN layers routed</b>. The published page said"
        f' {WAS_4["ffn-fnull1.00"]:.3f}x, measured when <code>--ffn-start-layer</code> still defaulted to 12 and'
        f' a third of the FFN was never routed. 27B at 8k goes {WAS_27["ffn-fnull1.00"]:.3f}x &rarr;'
        f' <b>{f1_27["wall_speedup"]:.3f}x</b>.</div></div>'
        f'<div class="card"><div class="tagline calc">measured &middot; deployment only</div>'
        f'<div class="big">{st_4["wall_speedup"]:.3f}x</div><div class="cap">the same FLOP cut taken by'
        f" <b>deleting</b> the always-null FFN layers instead of routing them. Not available during training"
        f" &mdash; the router has to exist for the layers to become deletable. It wins because it carries no"
        f' per-layer prologue and no host sync: {st_4["launches"]:,.0f} kernel launches against'
        f' {f1_4["launches"]:,.0f}.</div></div>'
        f'<div class="card"><div class="tagline meas">measured</div>'
        f'<div class="big">{ov_4["wall_speedup"]:.3f}x</div><div class="cap">the router switched on with every'
        f" token still taking the <em>full</em> rung: routing all 32 layers costs"
        f' {(1 - ov_4["wall_speedup"]) * 100:.0f}% at 4B and'
        f' {(1 - ov_27["wall_speedup"]) * 100:.0f}% at 27B before it saves anything. That entry fee is what the'
        f" FFN axis has to earn back.</div></div>"
        f'<div class="card"><div class="tagline meas">measured</div><div class="big">{CV4[0]:.2f}&ndash;{CV4[1]:.2f}</div>'
        f'<div class="cap">FLOP-to-wall conversion on the FFN axis at 4B &mdash; calculated from the'
        f" FLOP ledger &mdash; and it is <b>flat</b> from"
        f' <span class="mono">f_null=0</span> to <span class="mono">f_null=1</span>'
        f" ({CV27[0]:.2f}&ndash;{CV27[1]:.2f} at 27B)."
        f" The routed FFN is not where the conversion breaks down &mdash; that happens later, once block"
        f" skipping is stacked on top.</div></div>"
        "</div>"
    )

    # --- corrected arm table
    def arm_row(arm, label, was4, was27, pill):
        r4, r27 = R4[arm], R27[arm]
        return (
            f'<tr><td class="mono">{label}</td><td>{pill}</td>'
            f'<td class="num">{r4["flop_reduction"]:.2f}x</td>'
            f'<td class="num was">{was4:.3f}x</td><td class="num strong">{r4["wall_speedup"]:.3f}x</td>'
            f'<td class="num muted">{r4["conversion"]:.2f}</td>'
            f'<td class="num was">{was27:.3f}x</td><td class="num strong">{r27["wall_speedup"]:.3f}x</td>'
            f'<td class="num muted">{r27["conversion"]:.2f}</td></tr>'
        )

    TRAIN = '<span class="pill train">trainable</span>'
    DEPLOY = '<span class="pill deploy">deployment only</span>'
    arm_rows = [
        arm_row(
            "ffn-fnull0.00", "ffn-fnull0.00", WAS_4["ffn-fnull0.00"], WAS_27["ffn-fnull0.00"], TRAIN
        ),
        arm_row(
            "ffn-fnull0.50", "ffn-fnull0.50", WAS_4["ffn-fnull0.50"], WAS_27["ffn-fnull0.50"], TRAIN
        ),
        arm_row(
            "ffn-fnull0.75", "ffn-fnull0.75", WAS_4["ffn-fnull0.75"], WAS_27["ffn-fnull0.75"], TRAIN
        ),
        arm_row(
            "ffn-fnull1.00", "ffn-fnull1.00", WAS_4["ffn-fnull1.00"], WAS_27["ffn-fnull1.00"], TRAIN
        ),
        arm_row("ffn-static", "ffn-static", WAS_4["ffn-static"], WAS_27["ffn-static"], DEPLOY),
    ]
    # fnull0.90 had no published predecessor
    r4, r27 = R4["ffn-fnull0.90"], R27["ffn-fnull0.90"]
    arm_rows.insert(
        3,
        f'<tr><td class="mono">ffn-fnull0.90</td><td>{TRAIN}</td>'
        f'<td class="num">{r4["flop_reduction"]:.2f}x</td><td class="num muted">&mdash;</td>'
        f'<td class="num strong">{r4["wall_speedup"]:.3f}x</td><td class="num muted">{r4["conversion"]:.2f}</td>'
        f'<td class="num muted">&mdash;</td><td class="num strong">{r27["wall_speedup"]:.3f}x</td>'
        f'<td class="num muted">{r27["conversion"]:.2f}</td></tr>',
    )
    stack4, stack27 = R4["staticFFN+bskipK-0.25+headgate"], R27["staticFFN+bskipK-0.25+headgate"]
    arm_rows.append(
        f'<tr><td class="mono">staticFFN+bskipK-0.25+headgate</td><td>{DEPLOY}</td>'
        f'<td class="num">{stack4["flop_reduction"]:.1f}x</td>'
        f'<td class="num was">{WAS_4["stack"]:.3f}x</td>'
        f'<td class="num strong">{stack4["wall_speedup"]:.3f}x</td>'
        f'<td class="num muted">{stack4["conversion"]:.2f}</td>'
        f'<td class="num was">{WAS_27["stack"]:.3f}x</td>'
        f'<td class="num strong">{stack27["wall_speedup"]:.3f}x</td>'
        f'<td class="num muted">{stack27["conversion"]:.2f}</td></tr>'
    )
    arm_table = table(
        [
            "arm",
            "availability",
            "#FLOP cut (4B)",
            "#4B was",
            "#4B now",
            "#conv",
            "#27B was",
            "#27B now",
            "#conv",
        ],
        arm_rows,
    )

    # --- chart: wall speedup vs mean FFN compute, log-log
    def conv_panel(sweep, R, label, verdict):
        p = LogLog(
            0.013, 1.35, 0.85, 2.1, "mean FFN compute c, fraction of dense (log)", "step speedup"
        )
        p.frame(
            [
                (0.015625, "1/64"),
                (0.03125, "1/32"),
                (0.0625, "1/16"),
                (0.125, "1/8"),
                (0.25, "1/4"),
                (0.5, "1/2"),
                (1.0, "1"),
            ],
            [(0.9, "0.9x"), (1.0, "1x"), (1.25, "1.25x"), (1.5, "1.5x"), (2.0, "2x")],
        )
        pts = []
        for arm in ["ffn-overhead"] + FNULL_ARMS:
            c = 1.0 if arm == "ffn-overhead" else mean_ffn_compute(sweep, arm)
            r = R[arm]
            pts.append(
                (
                    c,
                    r["wall_speedup"],
                    f'{arm} &mdash; c={c:.4f}, {r["wall_speedup"]:.3f}x measured',
                )
            )
        pts.sort(key=lambda t: -t[0])
        p.series(pts, "s1", "routed FFN (trainable)")
        p.hrule(
            R["ffn-fnull1.00"]["wall_speedup"],
            f'routed, every token null (c=0) &mdash; {R["ffn-fnull1.00"]["wall_speedup"]:.3f}x',
            "s2",
            anchor="end",
            below=True,
        )
        p.hrule(
            R["ffn-static"]["wall_speedup"],
            f'static deletion (c=0, deployment only) &mdash; {R["ffn-static"]["wall_speedup"]:.3f}x',
            "s3",
            anchor="end",
        )
        p.label(
            1.0, R["ffn-overhead"]["wall_speedup"], "router on, no cut", dx=-6, dy=12, anchor="end"
        )
        return figure(
            label,
            verdict,
            "neutral",
            p.svg("wall-clock speedup against mean routed FFN compute, log-log"),
            legend(
                [
                    ("s1", "routed FFN, f_null sweep"),
                    ("dash s2", "all-null routed (c=0)"),
                    ("dash s3", "static deletion (c=0)"),
                ],
                panel=True,
            ),
        )

    conv_charts = (
        '<div class="grid2">'
        + conv_panel(
            J4, R4, "4B &middot; 16,384 tokens &middot; plain", "1.18x &rarr; 1.31x over the sweep"
        )
        + conv_panel(
            J27,
            R27,
            "27B &middot; 8,192 tokens &middot; checkpointed",
            "1.36x &rarr; 1.75x over the sweep",
        )
        + "</div>"
    )

    # --- chart: launches
    def launch_panel(R, dense_arm_launches, label, verdict):
        bars = [
            ("dense", dense_arm_launches, "sn"),
            ("router on", R["ffn-overhead"]["launches"], "s2"),
            ("f_null=0", R["ffn-fnull0.00"]["launches"], "s1"),
            ("f_null=1", R["ffn-fnull1.00"]["launches"], "s1"),
            ("static", R["ffn-static"]["launches"], "s3"),
        ]
        w, h = 350, 200
        x0, x1b, ytop, ybase = 20, 340, 30, 160
        top = max(b[1] for b in bars) * 1.12
        n = len(bars)
        slot = (x1b - x0) / n
        bw = slot * 0.62
        parts = []
        for i, (lab, v, cls) in enumerate(bars):
            cx = x0 + slot * (i + 0.5)
            bh = (v / top) * (ybase - ytop)
            parts.append(
                f'<rect class="bar {cls}" x="{cx - bw / 2:.1f}" y="{ybase - bh:.1f}" width="{bw:.1f}"'
                f' height="{bh:.1f}" rx="2"><title>{re.sub("<[^>]+>", "", lab)}: {v:,.0f} launches per step</title></rect>'
            )
            parts.append(
                f'<text class="vallab" x="{cx:.1f}" y="{ybase - bh - 5:.1f}" text-anchor="middle">{v:,.0f}</text>'
            )
            parts.append(
                f'<text class="tick" x="{cx:.1f}" y="{ybase + 13:.0f}" text-anchor="middle">{lab}</text>'
            )
            if cls != "sn":
                parts.append(
                    f'<text class="tick" x="{cx:.1f}" y="{ybase + 25:.0f}" text-anchor="middle">'
                    f"{v / dense_arm_launches:.2f}&times;</text>"
                )
        parts.append(f'<line class="axis" x1="{x0}" y1="{ybase}" x2="{x1b}" y2="{ybase}"/>')
        parts.append('<text class="axlab" x="10" y="18">kernel launches per training step</text>')
        svg = (
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="kernel launches per step">\n'
            + "\n".join(parts)
            + "\n</svg>"
        )
        return figure(
            label,
            verdict,
            "bad",
            svg,
            legend(
                [
                    ("sn", "dense baseline"),
                    ("s2", "router on, no FLOPs cut"),
                    ("s1", "routed FFN (trainable)"),
                    ("s3", "static deletion (deployment only)"),
                ],
                panel=True,
            ),
        )

    launch_charts = (
        '<div class="grid2">'
        + launch_panel(
            R4, R4["dense"]["launches"], "4B &middot; 16,384 tokens", f"up to {LR4:.2f}x dense"
        )
        + launch_panel(
            R27, R27["dense"]["launches"], "27B &middot; 8,192 tokens", f"up to {LR27:.2f}x dense"
        )
        + "</div>"
    )

    body = f"""
<section><h2>Correction: a third of the FFN was never routed</h2>
<div class="callout warn"><b>Every FFN and stacked number this page previously carried was
understated.</b> <code>--ffn-start-layer</code> defaulted to <b>12</b>, so on Qwen3.5-4B (32 blocks)
twelve of thirty-two FFNs stayed dense. The arm nominally called &ldquo;every token takes the null
rung&rdquo; was in fact cutting the FFN only to 0.375 of its own share. Both defaults are now 0 and
both benchmark scripts print the routed-layer list at startup
(<span class="mono"># FFN routed layers (32/32)</span>), so it cannot recur silently. The dense
control was re-measured every round; drift 1.004x at 4B and 1.009x at 27B. The kernel-level curves
further down are unaffected &mdash; they never went through the layer selector.</div>
{cards}
{PINNED}
<p>Both scales, one arm per row. <span class="mono">4B was</span> / <span class="mono">27B was</span>
are the published pre-fix numbers; the bold column is the re-measurement with all layers routed.
<b>conv</b> is FLOP cut converted to wall-clock &mdash; how much of the arithmetic saving the clock
actually paid out. Availability distinguishes arms a model can be <em>trained</em> with from
post-training deletion, which is a deployment lever only.</p>
{arm_table}</section>

<section><h2>Does the FFN axis convert? Yes, and at a flat rate</h2>
<p>Measured wall-clock speedup of a whole forward+backward step against the mean FFN compute the
router leaves per token, both axes logarithmic. The curve is nearly flat because the FFN is only
26% of a 4B step (46% at 27B): even taking <span class="mono">c</span> from 0.20 to 0.02 &mdash;
a 10&times; further cut &mdash; buys 6%. What the panel really shows is that the conversion rate does
not decay along this axis: it sits at {CV4[0]:.2f}&ndash;{CV4[1]:.2f} (4B) the whole way down. The two rules mark the
<span class="mono">c=0</span> arms, which cannot be drawn on a log axis: routing every token to the
null rung, and deleting those layers outright.</p>
{conv_charts}
<p>The gap between the two rules is the whole argument for static deletion: identical FLOPs,
{R4["ffn-static"]["wall_speedup"] / R4["ffn-fnull1.00"]["wall_speedup"] - 1:.1%} more speed at 4B and
{R27["ffn-static"]["wall_speedup"] / R27["ffn-fnull1.00"]["wall_speedup"] - 1:.1%} more at 27B, because
deletion carries no per-layer router prologue, no dispatch histogram and no host synchronisation.</p>
</section>

<section><h2>What the router actually costs: kernel launches</h2>
<p>Measured launch count for one training step. The routed FFN issues up to
<b>{LR4:.2f}&times;</b> as many kernels as dense at 4B and <b>{LR27:.2f}&times;</b> at 27B, for a
fraction of the work. Static deletion sits <em>below</em> dense
({R4["ffn-static"]["launches"]:,.0f} against {R4["dense"]["launches"]:,.0f} at 4B). That difference
&mdash; not FLOPs &mdash; is what separates the best trainable arm from the best deployment arm once
the levers are stacked.</p>
{launch_charts}
<div class="callout">On the full stack this shows up directly: at 4B/16k the best trainable
configuration measures <b>6.46x</b> and the same operating point with the FFN layers deleted instead
of routed measures <b>8.39x</b>. Where that ends, and why GDN rather than the FFN is what stops it,
is the subject of the companion page.</div>
</section>

{kernel_sections}"""

    footer = (
        "Model-level arms: <code>bench_joint_sweep.py</code> &rarr; "
        "<code>joint_sweep_4B_16384_v1.json</code> (4B/16k, plain) and "
        "<code>joint_sweep_27B_8192_v1.json</code> (27B/8k, activation-checkpointed); "
        "p10/median/p90 over 12&ndash;15 iterations, batch 1, bf16, no compile, one H200, with a "
        "dense control re-measured every round. Kernel-level curves (below the fold) from "
        "<code>bench_ffn_linearity.py</code>, <code>ffn_floor_probe.py</code>, "
        "<code>ffn_null_rung_sweep.py</code>, <code>ffn_mass_shift.py</code>: median of 11&ndash;14 "
        "iterations after warmup, forward+backward, contiguous rung indices (the scattered-assignment "
        "penalty was measured separately at 4&ndash;6%). Times are measured; FLOP counts, conversion "
        "ratios and mean-compute values are calculated from the FLOP ledger. Static FFN deletion is a "
        "post-training lever and its task quality has not been established."
    )

    return page(
        "Does Routed FFN Compute Buy Speed?",
        "Qwen3.5 &middot; H200 &middot; nested-ladder FFN router &middot; all 32/32 layers routed "
        "&middot; log&ndash;log",
        "Does Routed FFN Compute Buy Speed?",
        "Re-measured after a bug left twelve of thirty-two FFN layers dense. Model-level arms first, "
        "then the kernel-level curves that explain them &mdash; every one on the same x-axis, mean FFN "
        'compute <span class="mono">c</span>. Times are measured medians on one H200; FLOP and byte '
        "counts are calculated.",
        body,
        footer,
    )


# ================================================================ page 2


def build_ceiling_page() -> str:
    best_train4 = R4["J-f0.50-k0.0625"]
    safe_train4 = R4["J-f0.50-k0.2500"]
    best_static4 = R4["staticFFN+bskipK-0.0625+headgate"]
    stack4 = R4["staticFFN+bskipK-0.25+headgate"]
    best_train27 = R27["J-f0.50-k0.0625"]
    best_static27 = R27["staticFFN+bskipK-0.0625+headgate"]
    stack27 = R27["staticFFN+bskipK-0.25+headgate"]
    best_train_p = R4P["J-f1.00-k0.0625"]
    best_static_p = R4P["staticFFN+bskipK-0.0625+headgate"]

    cap_static_lowkeep = P4[-1]["cap"]

    cards = (
        '<div class="answer">'
        f'<div class="card"><div class="tagline meas">measured &middot; trainable</div>'
        f'<div class="big">{best_train4["wall_speedup"]:.2f}x</div><div class="cap">the best'
        f" <b>all-learned</b> configuration, 4B at 16k against a {DENSE4:.0f}&nbsp;ms dense control:"
        f" head gating, routed FFN and compacted block skip at keep=1/16. The defensible operating point"
        f' &mdash; keep=0.25 &mdash; is {safe_train4["wall_speedup"]:.2f}x. 27B at 8k reaches'
        f' {best_train27["wall_speedup"]:.2f}x.</div></div>'
        f'<div class="card"><div class="tagline calc">measured &middot; deployment only</div>'
        f'<div class="big">{best_static4["wall_speedup"]:.2f}x</div><div class="cap">the same operating'
        f" point with the always-null FFN layers <b>deleted</b> rather than routed. This is not available"
        f" during training. It is faster only because it stops issuing router kernels:"
        f' {best_static4["launches"]:,.0f} launches against'
        f' {R4["J-f1.00-k0.0625"]["launches"]:,.0f}. 27B: {best_static27["wall_speedup"]:.2f}x.</div></div>'
        f'<div class="card"><div class="tagline meas">measured &middot; packed</div>'
        f'<div class="big">{best_train_p["wall_speedup"]:.2f}x</div><div class="cap">the same 4B recipe'
        f" with the sequence <b>packed into 32 documents</b>, which is how real training runs. Packing"
        f" already makes attention cheap, so it roughly halves the prize:"
        f' {best_train4["wall_speedup"]:.2f}x &rarr; {best_train_p["wall_speedup"]:.2f}x trainable,'
        f' {best_static4["wall_speedup"]:.2f}x &rarr; {best_static_p["wall_speedup"]:.2f}x with deletion.'
        f" <b>Plan against this number.</b></div></div>"
        f'<div class="card"><div class="tagline">verdict</div><div class="big">20x is not reachable</div>'
        f'<div class="cap">the honest best is {best_static4["wall_speedup"]:.2f}x. Zeroing GDN entirely'
        f" gives 17.0x; zeroing GDN <em>and</em> attention gives 21.1x; the un-routable floor at that keep"
        f" is {cap_static_lowkeep:.1f}x. So 20x needs both mixers at zero, at a keep of 1/16 that is not a"
        f" trainable model.</div></div>"
        "</div>"
    )

    # ---- correction table
    def corr_row(label, w4, n4, w27, n27):
        return (
            f'<tr><td class="mono">{label}</td>'
            f'<td class="num was">{w4:.3f}x</td><td class="num strong">{n4:.3f}x</td>'
            f'<td class="num muted">+{(n4 / w4 - 1) * 100:.1f}%</td>'
            f'<td class="num was">{w27:.3f}x</td><td class="num strong">{n27:.3f}x</td>'
            f'<td class="num muted">+{(n27 / w27 - 1) * 100:.1f}%</td></tr>'
        )

    corr_table = table(
        ["arm", "#4B was", "#4B now", "#&Delta;", "#27B was", "#27B now", "#&Delta;"],
        [
            corr_row(
                "ffn-fnull1.00",
                WAS_4["ffn-fnull1.00"],
                R4["ffn-fnull1.00"]["wall_speedup"],
                WAS_27["ffn-fnull1.00"],
                R27["ffn-fnull1.00"]["wall_speedup"],
            ),
            corr_row(
                "ffn-static",
                WAS_4["ffn-static"],
                R4["ffn-static"]["wall_speedup"],
                WAS_27["ffn-static"],
                R27["ffn-static"]["wall_speedup"],
            ),
            corr_row(
                "staticFFN+bskipK-0.25+headgate",
                WAS_4["stack"],
                stack4["wall_speedup"],
                WAS_27["stack"],
                stack27["wall_speedup"],
            ),
        ],
    )

    # ---- chart 1: the ceiling rises with keep
    ceil = LogLog(0.05, 1.25, 9.5, 26, "block-skip keep (log)", "dense / floor")
    ceil.frame(
        [(0.0625, "1/16"), (0.125, "1/8"), (0.25, "1/4"), (0.5, "1/2"), (1.0, "dense")],
        [(10, "10x"), (15, "15x"), (20, "20x"), (25, "25x")],
    )
    c4 = [
        (
            p["keep"],
            p["cap"],
            f'{p["label"]} &mdash; floor {p["floor"]:.1f} ms, cap {p["cap"]:.1f}x',
        )
        for p in P4
        if not p["arm"].startswith("staticFFN")
    ]
    c4s = [
        (
            p["keep"],
            p["cap"],
            f'{p["label"]} &mdash; floor {p["floor"]:.1f} ms, cap {p["cap"]:.1f}x',
        )
        for p in P4
        if p["arm"].startswith("staticFFN")
    ]
    c27 = [
        (
            p["keep"],
            p["cap"],
            f'{p["label"]} &mdash; floor {p["floor"]:.1f} ms, cap {p["cap"]:.1f}x',
        )
        for p in P27
        if not p["arm"].startswith("staticFFN")
    ]
    ceil.series(sorted(c4, key=lambda t: -t[0]), "s1", "4B / 16k plain")
    ceil.series(sorted(c27, key=lambda t: -t[0]), "s2", "27B / 8k checkpointed")
    ceil.series(
        sorted(c4s, key=lambda t: -t[0]),
        "s1",
        "4B / 16k, static-FFN arms",
        hollow=True,
        dashed=True,
    )
    # Only the 4B curve is value-labelled: the two curves run within ~1x of each other and
    # cross, so labelling both collides. The 27B figures are in the prose below and in every
    # point's tooltip.
    cap27 = {p["keep"]: p["cap"] for p in P27 if not p["arm"].startswith("staticFFN")}
    for k, v, _ in sorted(c4, key=lambda t: -t[0]):
        other = cap27.get(k)
        ceil.label(k, v, f"{v:.1f}x", dy=(-9 if other is None or v >= other else 14))
    ceiling_fig = figure(
        "the un-routable floor, as a share of a dense step",
        "11.6x at dense &rarr; 22.4x at keep 1/16",
        "bad",
        ceil.svg("ceiling on speedup against block-skip keep, both scales"),
        legend(
            [
                ("s1", "4B / 16k, plain"),
                ("s2", "27B / 8k, checkpointed"),
                ("hollow", "4B static-FFN arms"),
            ],
            panel=True,
        ),
    )

    # ---- chart 2: joint sweep, log-log FLOPx vs wall
    def sweep_panel(sweep, R, dense, label, verdict, vcls, keeps=(0.5, 0.25, 0.125, 0.0625)):
        p = LogLog(
            4, 130, 1.6, 10.5, "total FLOP cut vs dense, calculated (log)", "measured step speedup"
        )
        p.frame(
            [(5, "5x"), (10, "10x"), (20, "20x"), (50, "50x"), (100, "100x")],
            [(2, "2x"), (3, "3x"), (5, "5x"), (8, "8x"), (10, "10x")],
        )
        for cls, fn in (("o1", "0.50"), ("o2", "0.90"), ("o3", "1.00")):
            pts = []
            for k in keeps:
                arm = f"J-f{fn}-k{k:.4f}"
                r = R[arm]
                pts.append(
                    (
                        r["flop_reduction"],
                        r["wall_speedup"],
                        f'{arm} &mdash; {r["flop_reduction"]:.1f}x FLOPs, '
                        f'{r["median"] * 1000:.0f} ms, {r["wall_speedup"]:.3f}x, conv {r["conversion"]:.2f}',
                    )
                )
            p.series(pts, cls, f"f_null={fn} (trainable)")
        spts = []
        for arm in ("staticFFN+bskipK-0.25+headgate", "staticFFN+bskipK-0.0625+headgate"):
            r = R[arm]
            spts.append(
                (
                    r["flop_reduction"],
                    r["wall_speedup"],
                    f'{arm} &mdash; {r["flop_reduction"]:.1f}x FLOPs, {r["median"] * 1000:.0f} ms, '
                    f'{r["wall_speedup"]:.3f}x',
                )
            )
        p.series(spts, "s2", "static FFN deletion (deployment only)", dashed=True)
        for x, y, _ in spts:
            p.label(x, y, f"{y:.2f}x", dy=-9)
        kv = R.get("J-f1.00-k0.2500+kv0.25")
        if kv:
            p.parts.append(
                f'<circle class="dot hollow s3" cx="{p.X(kv["flop_reduction"]):.1f}"'
                f' cy="{p.Y(kv["wall_speedup"]):.1f}" r="4.2"><title>+ KV router &mdash; '
                f'{kv["flop_reduction"]:.1f}x FLOPs but only {kv["wall_speedup"]:.3f}x, '
                f'down from {R["J-f1.00-k0.2500"]["wall_speedup"]:.3f}x</title></circle>'
            )
            p.label(
                kv["flop_reduction"],
                kv["wall_speedup"],
                "+KV router: slower",
                dx=8,
                dy=3,
                anchor="start",
            )
        return figure(
            label,
            verdict,
            vcls,
            p.svg("measured step speedup against calculated FLOP cut, log-log"),
            legend(
                [
                    ("o1", "f_null=0.50"),
                    ("o2", "f_null=0.90"),
                    ("o3", "f_null=1.00"),
                    ("dash s2", "static FFN deletion (deployment only)"),
                    ("hollow s3", "+ KV router"),
                ],
                panel=True,
            ),
        )

    sweep_charts = (
        '<div class="grid2">'
        + sweep_panel(
            J4,
            R4,
            DENSE4,
            f"4B &middot; 16k &middot; plain &middot; dense {DENSE4:.0f} ms",
            "saturates at 6.5x",
            "bad",
        )
        + sweep_panel(
            J27,
            R27,
            DENSE27,
            f"27B &middot; 8k &middot; AC &middot; dense {DENSE27:.0f} ms",
            "saturates at 5.1x",
            "bad",
        )
        + sweep_panel(
            J4P,
            R4P,
            DENSE4P,
            f"4B &middot; 16k &middot; packed &times;32 &middot; dense {DENSE4P:.0f} ms",
            "packing halves it",
            "bad",
        )
        + "</div>"
    )

    # ---- chart 3: step composition stacked bars
    def compose_svg(part, title_prefix):
        w = 604
        rowh, gap = 21, 4
        h = 34 + len(part) * (rowh + gap)
        x0, x1b = 168.0, 546.0
        cats = [
            ("head waste", "s1", "head"),
            ("FFN", "s2", "ffn"),
            ("attention", "s3", "attn"),
            ("GDN", "s4", "gdn"),
            ("routing + block shell + floor", "sn", "resid"),
        ]
        parts = []
        for i, p in enumerate(part):
            y = 24 + i * (rowh + gap)
            parts.append(
                f'<text class="tick" x="{x0 - 8:.0f}" y="{y + rowh / 2 + 3:.0f}" text-anchor="end">'
                f'{p["label"]}</text>'
            )
            cursor = x0
            for cname, cls, key in cats:
                frac = max(p[key], 0.0) / 100.0
                bw = frac * (x1b - x0)
                if bw <= 0.2:
                    continue
                parts.append(
                    f'<rect class="bar {cls}" x="{cursor:.1f}" y="{y}" width="{bw:.1f}" height="{rowh}">'
                    f'<title>{title_prefix} {p["label"]} &mdash; {cname} {p[key]:.1f}% of a '
                    f'{p["step"]:.0f} ms step</title></rect>'
                )
                if bw > 34:
                    parts.append(
                        f'<text class="onbar {cls}" x="{cursor + bw / 2:.1f}"'
                        f' y="{y + rowh / 2 + 3:.0f}" text-anchor="middle">{p[key]:.0f}%</text>'
                    )
                cursor += bw
            parts.append(
                f'<text class="vallab" x="{x1b + 6:.0f}" y="{y + rowh / 2 + 3:.0f}">{p["step"]:.0f} ms</text>'
            )
        parts.append(f'<text class="tick" x="{x0:.0f}" y="16">0%</text>')
        parts.append(
            f'<text class="tick" x="{x1b:.0f}" y="16" text-anchor="end">100% of step wall-clock</text>'
        )
        return (
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="step composition per arm">\n'
            + "\n".join(parts)
            + "\n</svg>"
        )

    comp_legend = legend(
        [
            ("s1", "head waste"),
            ("s2", "FFN"),
            ("s3", "attention"),
            ("s4", "GDN"),
            ("sn", "routing + block shell + floor"),
        ]
    )

    # ---- chart 4: component ms against keep
    def comp_panel(part, label, verdict):
        p = LogLog(0.05, 1.25, 1.2, 2000, "block-skip keep (log)", "component, ms (log)")
        p.frame(
            [(0.0625, "1/16"), (0.125, "1/8"), (0.25, "1/4"), (0.5, "1/2"), (1.0, "dense")],
            [(2, "2"), (10, "10"), (100, "100"), (1000, "1000")],
        )
        rows_ = [r for r in part if not r["arm"].startswith("staticFFN")]
        for cls, key, name in (
            ("s2", "ffn_ms", "FFN"),
            ("s3", "attn_ms", "attention"),
            ("s1", "gdn_ms", "GDN"),
            ("ink", "floor", "floor"),
        ):
            pts = [
                (r["keep"], max(r[key], 1.3), f'{name} at {r["label"]} &mdash; {r[key]:.1f} ms')
                for r in sorted(rows_, key=lambda z: -z["keep"])
            ]
            p.series(pts, cls, name, hollow=(cls == "ink"), dashed=(cls == "ink"))
        by_keep = {r["keep"]: r for r in rows_}
        lo, hi = by_keep[0.0625]["gdn_ms"], by_keep[0.25]["gdn_ms"]
        if abs(lo / hi - 1) < 0.05:
            p.label(0.125, max(lo, hi), "GDN flat", dy=-11)
        return figure(
            label,
            verdict,
            "bad",
            p.svg("component wall-clock against block-skip keep, log-log"),
            legend(
                [
                    ("s2", "FFN"),
                    ("s3", "attention"),
                    ("s1", "GDN"),
                    ("dash ink", "un-routable floor"),
                ],
                panel=True,
            ),
        )

    def gdn_verdict(part) -> str:
        """MEASURED GDN wall-clock: dense, then keep 0.25 and keep 1/16."""
        by = {r["keep"]: r for r in part if not r["arm"].startswith("staticFFN")}
        return (
            f'GDN {by[1.0]["gdn_ms"]:.0f} &rarr; {by[0.25]["gdn_ms"]:.0f} &rarr; '
            f'{by[0.0625]["gdn_ms"]:.0f} ms'
        )

    comp_charts = (
        '<div class="grid2">'
        + comp_panel(P4, "4B &middot; 16k &middot; plain", gdn_verdict(P4))
        + comp_panel(P27, "27B &middot; 8k &middot; checkpointed", gdn_verdict(P27))
        + "</div>"
    )

    # ---- partition tables
    def part_table(part, dense_step):
        body = []
        for p in part:
            strong = ' class="num strong"' if p is part[0] else ' class="num"'
            body.append(
                f'<tr><td class="mono">{p["label"]}</td><td class="num">{p["step"]:.1f}</td>'
                f'<td class="num">{p["head"]:.1f}%</td><td class="num">{p["ffn"]:.1f}%</td>'
                f'<td class="num">{p["attn"]:.1f}%</td><td class="num strong">{p["gdn"]:.1f}%</td>'
                f'<td class="num muted">{p["resid"]:.1f}%</td>'
                f'<td class="num">{p["floor"]:.1f}</td><td{strong}>{p["cap"]:.1f}x</td></tr>'
            )
        return table(
            [
                "arm",
                "#step ms",
                "#head waste",
                "#FFN",
                "#attn",
                "#GDN",
                "#routing+shell+floor",
                "#floor ms",
                "#cap from dense",
            ],
            body,
        )

    # ---- floor section
    floor_rows = []
    for r in FLOOR4:
        if r["tokens"] == 4096 and r["batch"] == 1 and r is not FLOOR4[1]:
            continue  # the repeat B=1/T=4096 drift check
        rest = r["base_s"] * 1000
        floor_rows.append(
            f'<tr><td class="num">{r["batch"]}</td><td class="num">{r["tokens"]:,}</td>'
            f'<td class="num">{r["total_tokens"]:,}</td><td class="num">{r["dense_s"] * 1000:.1f}</td>'
            f'<td class="num strong">{r["floor_s"] * 1000:.2f}</td>'
            f'<td class="num muted">{r["norm_s"] * 1000:.2f}</td>'
            f'<td class="num muted">{r["head_s"] * 1000:.2f}</td>'
            f'<td class="num muted">{r["embed_s"] * 1000:.2f}</td>'
            f'<td class="num muted">{r["zero_grad_s"] * 1000:.2f}</td>'
            f'<td class="num">{rest:.2f}</td>'
            f'<td class="num">{r["floor_pct_of_dense"]:.2f}%</td>'
            f'<td class="num">{r["base_gpu_busy_s"] * 1000:.1f} '
            f'({r["base_gpu_busy_s"] / r["base_s"]:.0%})</td>'
            f'<td class="num muted">{r["base_launches"]:,.0f}</td></tr>'
        )
    floor_table = table(
        [
            "#B",
            "#T",
            "#total tok",
            "#dense ms",
            "#floor ms",
            "#final norm",
            "#gated head",
            "#embed",
            "#zero_grad",
            "#block shell",
            "#floor % of dense",
            "#GPU-busy in the block shell",
            "#launches",
        ],
        floor_rows,
    )

    # floor scaling chart
    fl = LogLog(1700, 20000, 12, 210, "total tokens in the step (log)", "ms (log)")
    fl.frame(
        [(2048, "2k"), (4096, "4k"), (8192, "8k"), (16384, "16k")],
        [(15, "15"), (30, "30"), (60, "60"), (120, "120"), (200, "200")],
    )
    b1 = [r for r in FLOOR4 if r["batch"] == 1]
    seen = set()
    b1u = []
    for r in b1:
        if r["tokens"] in seen:
            continue
        seen.add(r["tokens"])
        b1u.append(r)
    b1u.sort(key=lambda r: r["tokens"])
    fl.series(
        [
            (
                r["total_tokens"],
                r["floor_s"] * 1000,
                f'B=1, {r["total_tokens"]:,} tokens &mdash; floor {r["floor_s"] * 1000:.2f} ms '
                f'({r["floor_pct_of_dense"]:.2f}% of dense)',
            )
            for r in b1u
        ],
        "s1",
        "floor, measured",
    )
    fl.series(
        [
            (
                r["total_tokens"],
                r["base_gpu_busy_s"] * 1000,
                f'GPU-busy inside the block shell &mdash; {r["base_gpu_busy_s"] * 1000:.1f} ms, '
                f'{r["base_gpu_busy_s"] / r["base_s"]:.0%} of it',
            )
            for r in b1u
        ],
        "s2",
        "GPU-busy in the block shell",
    )
    fix, per = FIT4["floor_s"]["fixed_s"] * 1000, FIT4["floor_s"]["per_token_s"] * 1000
    fitpts = [
        (t, fix + per * t, f"fit: {fix:.2f} ms + {per * 1000:.3f} ms per 1k tokens")
        for t in (2048, 16384)
    ]
    fl.series(fitpts, "s3", "least-squares fit", dashed=True, r=0)
    b2 = [r for r in FLOOR4 if r["batch"] == 2][0]
    fl.parts.append(
        f'<circle class="dot hollow s1" cx="{fl.X(b2["total_tokens"]):.1f}"'
        f' cy="{fl.Y(b2["floor_s"] * 1000):.1f}" r="5"><title>B=2, T=4096 &mdash; same 8,192 tokens, '
        f'floor {b2["floor_s"] * 1000:.2f} ms, identical to B=1/T=8192 to three digits</title></circle>'
    )
    floor_fig = figure(
        "4B &middot; the floor with every block replaced by Identity",
        "86% per-token, 14% fixed",
        "bad",
        fl.svg("floor wall-clock against total tokens, log-log"),
        legend(
            [
                ("s1", "floor, measured"),
                ("s2", "GPU-busy in the block shell"),
                ("dash s3", "fit a + b&middot;tokens"),
                ("hollow", "B=2 / T=4096"),
            ],
            panel=True,
        ),
    )

    # head-cost table
    head_rows = "".join(f'<td class="num">{h["head_s"] * 1000:.2f}</td>' for h in HEADROWS4)
    head_hdr = "".join(f'<th class="num">{h["n_sup"]:,}</th>' for h in HEADROWS4)
    head_table = (
        '<div class="tablewrap"><table><thead><tr><th>supervised rows</th>'
        + head_hdr
        + "</tr></thead><tbody><tr><td>gated head cost, ms</td>"
        + head_rows
        + "</tr></tbody></table></div>"
    )

    # ---- launch-bound table
    LB = [
        ("dense", GRAPH4["dense"]),
        ("staticFFN+bskipK-0.25", GRAPH4["staticFFN+bskipK-0.25"]),
        ("J-f1.00-k0.2500", GRAPH4["J-f1.00-k0.2500"]),
        ("staticFFN+bskipK-0.25+headgate", GRAPH4["staticFFN+bskipK-0.25+headgate"]),
    ]
    lb_rows = []
    for name, g in LB:
        lb_rows.append(
            f'<tr><td class="mono">{name}</td><td class="num">{g["eager_median_s"] * 1000:.1f}</td>'
            f'<td class="num">{g["gpu_busy_s"] * 1000:.1f}</td>'
            f'<td class="num strong">{g["idle_frac"]:.1%}</td>'
            f'<td class="num">{g["launches"]:,.0f}</td>'
            f'<td class="num muted">{g["graph_upper_bound"]:.2f}x</td></tr>'
        )
    bs = R4["staticFFN+bskipK-0.0625+headgate"]
    lb_rows.append(
        f'<tr><td class="mono">staticFFN+bskipK-0.0625+headgate</td>'
        f'<td class="num">{bs["median"] * 1000:.1f}</td><td class="num">{bs["gpu_busy_s"] * 1000:.1f}</td>'
        f'<td class="num strong bad">{bs["idle_frac"]:.1%}</td><td class="num">{bs["launches"]:,.0f}</td>'
        f'<td class="num muted">{1 / bs["gpu_busy_frac"]:.2f}x</td></tr>'
    )
    lb_table = table(
        ["arm", "#step ms", "#GPU-busy ms", "#idle", "#launches", "#graph upper bound"], lb_rows
    )

    capture_rows = [
        ("<span class='mono'>matmul</span> (sanity)", "captured, 1.04x", "&mdash;", True),
        ("one block, forward", "captured, <b>3.90x</b>", "&mdash;", True),
        ("one block, fwd+bwd", "captured, <b>4.38x</b>", "&mdash;", True),
        ("whole model forward, no head", "captured, 3.32x", "&mdash;", True),
        ("whole model fwd+bwd, no head", "captured, <b>2.65x</b>", "captured, 1.003x", True),
        (
            "&hellip; with <span class='mono'>budget_attach</span> off",
            "captured, 2.67x",
            "&mdash;",
            True,
        ),
        (
            "whole step, <b>unfused</b> cross entropy",
            "captured, <b>2.09x</b>",
            "OOM (16k &times; 248k logits)",
            True,
        ),
        ("whole step, <b>fused-linear</b> CE (our trainer)", "<b>FAILS</b>", "<b>FAILS</b>", False),
        ("whole step, head gating as shipped", "&mdash;", "<b>FAILS</b>", False),
        (
            "whole step, <b>fixed-capacity</b> head gate (prototype)",
            "&mdash;",
            "<b>captured</b>, 1.006x",
            True,
        ),
        ("&hellip; on any routed arm", "&mdash;", "<b>FAILS</b>", False),
    ]
    cap_table = table(
        ["case", "0.8B / 2k", "4B / 16k"],
        [
            f'<tr><td>{c}</td><td class="{"" if ok else "bad"}">{a}</td>'
            f'<td class="{"" if ok else "bad"}">{b}</td></tr>'
            for c, a, b, ok in capture_rows
        ],
    )

    # ---- the 20x ladder
    p_static_low = P4[-1]
    gdn_zero_ms = p_static_low["floor"] + p_static_low["attn_ms"] + max(p_static_low["ffn_ms"], 0)
    both_zero_ms = p_static_low["floor"] + max(p_static_low["ffn_ms"], 0)
    ladder = [
        ("dense control", "dense", DENSE4, "measured", "sn"),
        (
            "best all-learned, trainable",
            "J-f0.50-k0.0625",
            best_train4["median"] * 1000,
            "measured",
            "s1",
        ),
        (
            "best trainable at a defensible keep",
            "J-f0.50-k0.2500",
            safe_train4["median"] * 1000,
            "measured",
            "s1",
        ),
        (
            "best with static FFN deletion",
            "staticFFN+bskipK-0.0625+headgate",
            best_static4["median"] * 1000,
            "measured",
            "s2",
        ),
        (
            "... if GDN cost nothing",
            "floor + attention + FFN at the keep-1/16 static arm",
            gdn_zero_ms,
            "calculated",
            "s3",
        ),
        (
            "... if GDN and attention both cost nothing",
            "floor + FFN at the same arm",
            both_zero_ms,
            "calculated",
            "s3",
        ),
        (
            "the un-routable floor at that keep",
            "step - (head waste + FFN + attention + GDN)",
            p_static_low["floor"],
            "calculated",
            "sn",
        ),
        ("the 20x target", "for reference", DENSE4 / 20.0, "target", "sn"),
    ]
    lw, rowh, gap = 568, 20, 5
    lh = 30 + len(ladder) * (rowh + gap)
    lx0, lx1 = 316.0, 500.0
    maxsp = 23.0
    lparts = []
    for i, (lab, note, ms, kind, cls) in enumerate(ladder):
        y = 20 + i * (rowh + gap)
        sp = DENSE4 / ms
        bw = (sp / maxsp) * (lx1 - lx0)
        lparts.append(
            f'<text class="tick" x="{lx0 - 8:.0f}" y="{y + rowh / 2 + 3:.0f}" text-anchor="end">'
            f"{lab}</text>"
        )
        op = ' opacity="0.55"' if kind != "measured" else ""
        dash = ' stroke-dasharray="3 2"' if kind != "measured" else ""
        lparts.append(
            f'<rect class="bar {cls}" x="{lx0:.1f}" y="{y}" width="{bw:.1f}" height="{rowh}" rx="2"'
            f"{op}{dash}><title>{lab} ({note}) &mdash; {ms:.1f} ms, {sp:.2f}x, {kind}</title></rect>"
        )
        lparts.append(
            f'<text class="vallab" x="{lx0 + bw + 6:.1f}" y="{y + rowh / 2 + 3:.0f}">'
            f"{sp:.2f}x &middot; {ms:.1f} ms</text>"
        )
    x20 = lx0 + (20.0 / maxsp) * (lx1 - lx0)
    lparts.append(f'<line class="capline" x1="{x20:.1f}" y1="14" x2="{x20:.1f}" y2="{lh - 14}"/>')
    lparts.append(f'<text class="guidelab" x="{x20 - 3:.1f}" y="12" text-anchor="end">20x</text>')
    ladder_svg = (
        f'<svg viewBox="0 0 {lw} {lh}" role="img" aria-label="ladder from dense to the floor">\n'
        + "\n".join(lparts)
        + "\n</svg>"
    )

    comp_svg4 = compose_svg(P4, "4B/16k")
    part_table4 = part_table(P4, P4[0]["step"])
    part_table27 = part_table(P27, P27[0]["step"])

    body = f"""
<section><h2>The answer</h2>{cards}
{PINNED}
<div class="callout warn"><b>Read keep &le; 0.125 as a mechanism probe, not an operating point.</b>
Every routed block is forced to skip the <em>same</em> tokens, so at keep=1/16 the 93.75% that are
skipped do no mixing in any block &mdash; they ride the residual stream from embedding to head. The
wall-clock is honest (the compacted path's cost depends on how many tokens are kept, not which), but
it is not a model anyone would train. The defensible trainable point is keep=0.25:
<b>{safe_train4["wall_speedup"]:.2f}x</b> at 4B/16k.</div>
<div class="callout"><b>What changed since the published version of this page.</b>
<code>--ffn-start-layer</code> defaulted to 12, leaving twelve of thirty-two FFNs dense on
Qwen3.5-4B, so every FFN and stacked number here was understated. Re-measured with all layers
routed, the dense control re-taken each round (drift 1.004x at 4B, 1.009x at 27B):</div>
{corr_table}
</section>

<section><h2>The ceiling is not a fixed number &mdash; it rises as you skip harder</h2>
<p>The earlier reading of this study was that a floor of roughly 8.7% of every step caps the family
at about 11.5x. That is correct <em>at dense</em>, and it reproduces independently at both scales.
It is <b>not</b> a ceiling on an aggressively skipped model, because compaction removes skipped
tokens from the block shell too &mdash; the two RMSNorms and two residual adds each block runs over
the whole activation stream. Cut the kept fraction and the floor itself shrinks, so
<span class="mono">dense / floor</span> climbs.</p>
<div class="solo">{ceiling_fig}</div>
<p>All measured, by the same cumulative-ablation method on each arm at its own operating point:
<b>11.6x</b> at dense and keep=0.5, <b>16.7x</b> at keep=0.25, <b>22.4x</b> at keep=1/16 for 4B/16k.
27B/8k traces the same curve from a completely different component mix: 11.6 / 12.3 / 17.8 /
20.9x.</p></section>

<section><h2>The joint sweep: every lever at once</h2>
<p>Head gating on throughout, block skip compacted and fixed-capacity, KV router excluded except for
one arm that keeps it to re-confirm the verdict. Measured wall-clock against the calculated FLOP cut,
both axes logarithmic. Three things read straight off the panels: the trainable curves
<b>saturate</b> &mdash; keep 0.25 &rarr; 0.125 buys 1.44x, 0.125 &rarr; 1/16 buys 1.09x for another
2&times; less work; the three <span class="mono">f_null</span> curves sit almost on top of each other
at low keep, so the FFN router stops mattering once block skip is aggressive (2.7% at keep=1/16, and
in the <em>wrong</em> direction); and the static-deletion arms sit above every trainable one at the
same FLOPs.</p>
{sweep_charts}
<div class="callout"><b>The KV router loses for the third independent time.</b> At 4B it takes
{R4["J-f1.00-k0.2500"]["wall_speedup"]:.3f}x down to
{R4["J-f1.00-k0.2500+kv0.25"]["wall_speedup"]:.3f}x while <em>raising</em> the FLOP cut from
{R4["J-f1.00-k0.2500"]["flop_reduction"]:.1f}x to
{R4["J-f1.00-k0.2500+kv0.25"]["flop_reduction"]:.1f}x; at 27B,
{R27["J-f1.00-k0.2500"]["wall_speedup"]:.3f}x &rarr;
{R27["J-f1.00-k0.2500+kv0.25"]["wall_speedup"]:.3f}x. Compacted block skip already removes a skipped
token as a key by gathering rather than masking, so the KV router re-does that work and charges for
it. Keep it out of the stack.</div>
<p><b>Packing roughly halves the prize.</b> Same 4B, same 16k, sequence packed into 32 documents
&mdash; how real training runs. The dense step falls {DENSE4:.0f} ms &rarr; {DENSE4P:.0f} ms because
packing already makes attention cheap, and the recipe falls with it: best trainable
{best_train4["wall_speedup"]:.2f}x &rarr; <b>{best_train_p["wall_speedup"]:.2f}x</b>, best with static
deletion {best_static4["wall_speedup"]:.2f}x &rarr; <b>{best_static_p["wall_speedup"]:.2f}x</b>, with
GPU busy at the best arm down to {best_static_p["gpu_busy_frac"]:.1%}.</p></section>

<section><h2>The bottleneck moved: it is GDN now, not the FFN</h2>
<p>The step partition re-measured <em>on the routed model</em>, with the routers live at each arm's
operating point. One stage is new relative to earlier versions of this study: <b>delete the block
itself</b>. Replacing a block's mixer and FFN leaves the block <em>shell</em> running &mdash; two
RMSNorms and two residual adds per block over the whole activation stream, forward and backward,
32 times. That is per-token GPU work no component ablation ever removed, and it is essentially the
entire floor (8.4% of a dense 4B step, with the true base measuring 0.0% underneath it).</p>
<div class="panel">{comp_svg4}</div>
{comp_legend}
<p>Routing overhead and the block shell are reported as one bucket on purpose: the ablation turns the
routers off before it deletes the shell, so it measures the shell at <em>full</em> sequence length and
books the difference as negative routing overhead. Their sum is the honest quantity; the split
between them is an artifact.</p>
{part_table4}
<p>As the levers bite, attention collapses (21.8% &rarr; 3.3%) and the FFN collapses
(26.3% &rarr; 0.8%), but <b>GDN barely moves in absolute terms</b>: 295 ms dense &rarr; 93 ms, and
then flat from keep=0.25 to keep=1/16 (93.0 &rarr; 92.9 ms). Compaction shortens the scan, but the
delta rule is memory-bound at roughly 20 FLOP/byte and gets <em>less</em> efficient on shorter
sequences, so a 4&times; shorter sequence is not 4&times; cheaper.</p>
{comp_charts}
<p><b>At the best 4B arm, once the head-gating waste it removes is taken out, the step is GDN 53%,
floor 36%, attention 9%, FFN 2%.</b> 27B reproduces every part of this on a completely different
component mix &mdash; the FFN is 46.5% of a dense 27B step against 26.3% at 4B &mdash; and lands in
the same place: GDN goes from 23.0% of a dense step to 34.8% of the best routed one and becomes the
largest single component, overtaking that 46.5% FFN.</p>
{part_table27}</section>

<section><h2>The floor, decomposed</h2>
<p>Every block set to Identity, then token count and batch size swept with the same
alternating-direction rounds. This is the quantity the ceiling is made of, so it is worth knowing
exactly what is in it.</p>
{floor_table}
<div class="solo">{floor_fig}</div>
<div class="answer">
<div class="card"><div class="tagline calc">calculated from measured points</div>
<div class="big">86% per-token</div><div class="cap">least-squares fit over the B=1 rows:
{fix:.2f} ms fixed + {per * 1000:.3f} ms per 1k tokens. At 16k only {fix:.1f} of
{FLOOR4[3]["floor_s"] * 1000:.1f} ms is amortisable &mdash; <b>1.2% of the dense step</b>. A bigger
step cannot buy back more than that.</div></div>
<div class="card"><div class="tagline meas">measured</div><div class="big">batch does not help</div>
<div class="cap">B=2/T=4096 and B=1/T=8192 are the same 8,192 tokens and give the same floor to three
digits ({FLOOR4[2]["floor_s"] * 1000:.2f} vs {FLOOR4[5]["floor_s"] * 1000:.2f} ms). The floor tracks
<em>total tokens</em>, full stop &mdash; and 27B behaves identically.</div></div>
<div class="card"><div class="tagline meas">measured</div><div class="big">98% GPU-busy</div>
<div class="cap">it is real GPU work, not launch overhead. The launch count in the floor
configuration is constant at ~1,786 regardless of token count, while GPU-busy rises linearly and
reaches 98% of the floor's wall-clock at 16k. There is nothing here for CUDA graphs to
recover.</div></div>
</div>
<p><b>The fused-CE head's fixed cost is not hiding in the floor.</b> Measured head cost against the
number of supervised rows, 4B/16k:</p>
{head_table}
<p>Flat at about 2.1 ms below ~128 rows, so the N-independent term exists and is <b>0.15% of the
dense step</b>. The old worry from the isolated-head work &mdash; &ldquo;13 rows still costs
70 ms&rdquo; &mdash; does not apply at model level, because <code>LMHead</code> drops the fused kernel
below <span class="mono">sparse_loss_unfused_max_rows=4096</span> and never allocates its
<span class="mono">d_model &times; vocab</span> fp32 accumulator. At our loss density (0.0064, so 104
rows at 16k) the gated head costs 2.1 ms. Embeddings, the final norm, <code>zero_grad</code> and the
gated head together are {small_terms:.1f} ms of a {FLOOR4[3]["floor_s"] * 1000:.1f} ms floor; the rest
is the block shell.</p></section>

<section><h2>Launch-bound, and why CUDA graphs cannot be cashed in yet</h2>
<p>Measured idle fraction and launch count from the profiler. The launch-bound regime is real but
arrives late: the <span class="mono">graph upper bound</span> column is wall divided by GPU-busy
&mdash; what a <em>perfect</em> graph could recover, not a prediction.</p>
{lb_table}
<p>Whether a training step can be captured at all, one case per process (a failed capture poisons
every later capture in the same CUDA context):</p>
{cap_table}
<p>The blockers are host synchronisations and value-dependent shapes, and they are enumerable: the
fused-linear CE kernel (which falls away once the head is gated below the unfused threshold); head
gating as shipped (<code>lm_head.py:286</code> <span class="mono">bool(keep.any())</span> plus
<span class="mono">x[keep]</span>, whose output shape depends on tensor <em>values</em> &mdash;
replacing it with a fixed row budget chosen by <span class="mono">topk</span> makes the whole
4B/16k step capturable and is exact for capacity &ge; supervised rows); the block-skip holder's
bookkeeping, three host syncs in <code>block_skip.py</code>, which is what still blocks every routed
arm; and the FFN router's <span class="mono">bincount(&hellip;).tolist()</span>. None is load-bearing
numerics &mdash; they are budget statistics and dispatch counts that could live on device and be read
once every N steps. <b>That is the concrete engineering item</b>, and by the idle column it is worth
about 1.19x at the recommended operating point.</p></section>

<section><h2>So: is 20x reachable?</h2>
<p><b>Not in practice.</b> All against the same {DENSE4:.1f} ms dense control, 4B at 16k, batch 1,
one H200, in one process. Solid bars are measured; the faded, dash-outlined bars are calculated
by subtracting a measured component from a measured step.</p>
<div class="panel">{ladder_svg}</div>
{legend([("sn", "dense control / floor / target"), ("s1", "measured, trainable"),
         ("s2", "measured, deployment only"), ("s3", "calculated")])}
<p><b>What stops us is GDN.</b> It is 53% of the best arm's step at 4B and 34.8% at 27B, and it is
flat from keep=0.25 to keep=1/16 &mdash; skipping tokens stops helping it. Every GDN lever tried in
this study is already refuted: q/k/v fusion 1.02x, chunk size 1&ndash;2%, packing 0.91x,
<span class="mono">expand_v</span> about 1.4% of a step, more batch and head parallelism refuted.
Even a <em>perfect</em> GDN gives 17.0x. Reaching 20x needs GDN at zero <em>and</em> attention at
zero, at a keep of 1/16 that is not a trainable model.</p>
<p><b>Plan against 4&ndash;6.5x trainable and 5&ndash;8.4x with post-training FFN deletion, halved by
packing</b> ({best_train_p["wall_speedup"]:.2f}x / {best_static_p["wall_speedup"]:.2f}x measured at
<span class="mono">--docs 32</span>). The next real 1.19x is CUDA graphs, blocked by four enumerable
host syncs rather than by anything fundamental.</p></section>"""

    footer = (
        "Measured on one H200 (sneetches), batch 1, bf16, no compile, forward+backward, with a dense "
        "control re-measured every round (drift 1.003&ndash;1.017x). Arms and profiling: "
        "<code>bench_joint_sweep.py</code> &rarr; <code>joint_sweep_4B_16384_v1.json</code> (plain), "
        "<code>joint_sweep_4B_16384_packed32.json</code> (<code>--docs 32</code>), "
        "<code>joint_sweep_27B_8192_v1.json</code> (activation-checkpointed); graph capture from "
        "<code>graph_probe.py</code> &rarr; <code>graph_probe_*.json</code>. The 4B step partition "
        "including the block-shell stage is <code>joint_sweep_4B_16384_shell.log</code> rather than a "
        "JSON &mdash; that run filled the <code>/accounts</code> quota at its final write, so its stdout "
        "is the record; only its per-component percentages are used here, and the floor and cap columns "
        "are recomputed as step &minus; (head waste + FFN + attention + GDN), because the log's own "
        '<span class="mono">floor=</span> fields predate the fix that moved the block shell into the '
        "floor sum. 27B's partition ran before that stage existed, so its residual bucket is the same "
        "recomputation. Times are measured; FLOP cuts, conversion ratios, the floor fit and the two "
        "&ldquo;if X cost nothing&rdquo; rows are calculated. Head gating is exactly loss-neutral; "
        "static FFN deletion and block skip are mechanism limits whose task quality has not been "
        "established, and static deletion is not available during training."
    )

    return page(
        "Where the Router Speedup Runs Out",
        "Qwen3.5 &middot; one H200 &middot; head gating + compute routers &middot; all 32/32 FFN "
        "layers routed &middot; log&ndash;log",
        "Where the Router Speedup Runs Out",
        "The 40x question priced in <em>wall-clock</em> rather than FLOPs, re-measured after a bug "
        "left a third of the FFN unrouted. Every number below is a real forward+backward step on one "
        "H200; the trainable ceiling and the deployment-only ceiling are kept apart throughout.",
        body,
        footer,
    )


# ---------------------------------------------------------------- small derived text values

small_terms = sum(FLOOR4[3][k] * 1000 for k in ("norm_s", "head_s", "embed_s", "zero_grad_s"))
fix = FIT4["floor_s"]["fixed_s"] * 1000
per = FIT4["floor_s"]["per_token_s"] * 1000


def main() -> None:
    out1 = os.path.join(HERE, "ffn_routing_linearity.html")
    out2 = os.path.join(HERE, "flop_scaling_ceilings.html")
    html1 = build_ffn_page()
    html2 = build_ceiling_page()
    with open(out1, "w") as f:
        f.write(html1)
    with open(out2, "w") as f:
        f.write(html2)
    print(f"wrote {out1} ({len(html1):,} bytes)")
    print(f"wrote {out2} ({len(html2):,} bytes)")


if __name__ == "__main__":
    main()
