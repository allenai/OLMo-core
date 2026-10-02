"""Render the dev-loss grid artifact from ``grid.json`` (collect_grid.py) + ``manifest.json``.

Columns = tasks (low-CTC group, high-CTC group, then cpt80), rows = schemes, cells = answer CE with
dCE vs full and compaction underneath. Four views (2k / 8k / 32k / mean over rungs). Cell shading
is a one-hue sequential ramp on dCE (light = at parity, dark = far from it) so a scheme that works
everywhere reads as a light stripe. Missing cells say why (dropped / not run yet).

    python debug/devloss_grid/render_grid.py [--grid grid.json] [--out .../devloss_grid.html]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = ("/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/"
       "b25b0e58-1d93-45d2-a076-a44e278feae6/scratchpad/devloss_grid.html")
# default row order: the five rows Prasann reads first (2026-09-22), then their random-position twin, then the rest
SCHEMES = ["full", "gold_fl20p8", "gold_fl20p8_noslot", "gold_fl20p8_skipodd", "gold_first20", "gold_rand20p8_noslot", "k0", "first16", "first64", "fl16", "fl64", "idf16", "idf64", "fl20", "fl20p8", "rand33", "gold_rand33", "gold_fl64", "gold_fl20", "gold_k0", "first20", "fl20p8_noslot", "rand20", "first128", "rule20", "grad20", "attnrow20", "attntop10_noslot", "attntop20_noslot", "attntop30_noslot", "gold_fl20p8_skipeven", "gold_fl20p8_skipgdn2"]
SCHEME_DESC = {
    "full": "uncompacted, plain causal (reference)",
    "k0": "keep 0 docs; one cent_cmean slot per doc",
    "first16": "first 16 body tokens real per doc + slot",
    "first64": "first 64 body tokens real + slot",
    "fl16": "first 8 + last 8 real + slot",
    "fl64": "first 32 + last 32 real + slot",
    "idf16": "top-16 body tokens by IDF real + slot",
    "idf64": "top-64 by IDF real + slot",
    "rand33": "random whole docs, keep 1/3; plain slot (scheme B)",
    "gold_rand33": "gold docs real + random to 1/3; plain slot (scheme A)",
    "gold_fl64": "gold docs real + fl64 elsewhere (scheme G)",
    "fl20": "first_last 20% of EVERY document body (per-doc budget); cent_cmean",
    "gold_fl20": "gold docs real + fl20 elsewhere",
    "fl20p8": "first 8 tokens real per document + first_last 20% of the rest; cent_cmean",
    "gold_fl20p8": "gold docs real + fl20p8 elsewhere",
    "gold_k0": "gold docs real, every other document a bare slot",
    "first20": "first 20% of every document body (contiguous prefix); cent_cmean",
    "gold_first20": "gold docs real + first20 elsewhere",
    "fl20p8_noslot": "fl20p8 kept tokens, pooled remainder DROPPED (no slot)",
    "gold_fl20p8_noslot": "gold docs real + fl20p8 elsewhere, no slots",
    "gold_rand20p8_noslot": "gold docs real + 8 id tokens + a random 20% of the rest of each body, no slots",
    "gold_fl20p8_skipodd": "gold_fl20p8 + non-kept docs skip blocks 1,3,…,31 (every attention block)",
    "gold_fl20p8_skipeven": "gold_fl20p8 + non-kept docs skip blocks 0,2,…,30 (GDN only)",
    "gold_fl20p8_skipgdn2": "gold_fl20p8 + non-kept docs skip every other GDN block (attention intact)",
}
# learned linear token router (debug/learned_router/, 2026-09-23): slot-less, doc-level keep none,
# markers kept; rows appear right after gold_rand20p8_noslot, only when grid.json has them
ROUTER_SCHEMES = [f"router_{v}l{lam}{sfx}" for lam in ("0.05", "0.2", "0.5", "1.0", "2.0") for v in ("", "nogold_", "noemb_") for sfx in ("", "_samp")] + ["router_l0.2_nomark"]
ROUTER_SCHEMES += [f"router_diff_tau{t}" for t in ("0.05", "0.1", "0.2", "0.4")]
ROUTER_SCHEMES += [f"router_e2e_rho{r}" for r in ("0.05", "0.1", "0.2", "0.3", "0.5")] + [f"router_e2e_tau{t}" for t in ("0.05", "0.1", "0.2")]
for _r in ("0.05", "0.1", "0.2", "0.3", "0.5"):
    SCHEME_DESC[f"router_e2e_rho{_r}"] = (f"END-TO-END linear router, target keep ρ={_r} (CoFi Lagrangian, CE+KL, relaxed exact removal, "
                                          "no rule init); eval = exact hard removal, p>0.5, no slots, markers kept")
for _t in ("0.05", "0.1", "0.2"):
    SCHEME_DESC[f"router_e2e_tau{_t}"] = f"END-TO-END router, ρ selected on val: most compact with val ΔCE ≤ max({_t}·CE_full, 0.02)"
for _t in ("0.05", "0.1", "0.2", "0.4"):
    SCHEME_DESC[f"router_diff_tau{_t}"] = (f"DIFFERENTIABLE linear router (relaxed removal, hard-concrete), trained to max compaction s.t. "
                                         f"mean ΔCE ≤ {_t}·CE_full (floor 0.005 nats); eval = exact hard removal, keep p>0.5, no slots, markers kept")
for _p in (10, 20, 30):
    SCHEME_DESC[f"attntop{_p}_noslot"] = (f"gold-blind, slot-less: top {_p}% of ALL document body tokens by layer-3 attention mass "
                                         "from the last 32 prompt positions, then a second compacted forward; markers kept")
SCHEME_DESC["gold_only_noslot"] = "gold docs real, every other document dropped outright (no slot, no markers)"
for _s in ROUTER_SCHEMES:
    if _s.startswith(("router_diff_", "router_e2e_")):
        continue  # described above
    _v = "no gold feature (embedding + position)" if "nogold_" in _s else ("no embedding (position + gold)" if "noemb_" in _s else "embedding + position + gold")
    _lam = re.search(r"l([0-9.]+)", _s[len("router_"):]).group(1)
    SCHEME_DESC[_s] = (f"learned linear router ({_v}), λ={_lam}, " + ("seeded Bernoulli sample" if _s.endswith("_samp") else "keep p>0.5")
                       + "; body tokens dropped outright, no slots, markers kept"
                       + (" — eval with markers of partial docs DROPPED (noslot-baseline semantics)" if _s.endswith("_nomark") else ""))
GOLD_TWIN = {"gold_rand33": "rand33", "gold_fl64": "fl64", "gold_fl20": "fl20", "gold_fl20p8": "fl20p8", "gold_k0": "k0", "gold_first20": "first20", "gold_fl20p8_noslot": "fl20p8_noslot", "gold_rand20p8_noslot": "rand20p8_noslot", "gold_fl20p8_skipodd": "fl20p8 + skipodd", "gold_fl20p8_skipeven": "fl20p8 + skipeven", "gold_fl20p8_skipgdn2": "fl20p8 + skipgdn2", "gold_only_noslot": "k0 (no gold set: every document a slot)"}
RUNGS = ["2k", "8k", "32k"]
VIEWS = RUNGS + ["mean"]
PARITY = 0.02
# tasks left out of the per-row mean/median dCE columns: xabsence is VOID (paraphrase-era checkpoint
# scored on EXACT rows, manifest ckpt_note) and cpt80 is continued pretraining, not a CTC task
AGG_EXCLUDE = {"xabsence", "cpt80"}
# dCE bins -> shade class b0..b6 (sequential, one hue; b0 = at parity)
BINS = [(-math.inf, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.80), (0.80, math.inf)]


GRIP = ('<button type="button" class="grip" aria-label="Move row (drag, or focus and use the arrow keys)" title="Drag to reorder">'
        '<svg viewBox="0 0 10 16" aria-hidden="true"><g fill="currentColor"><circle cx="3" cy="3" r="1.4"/><circle cx="7" cy="3" r="1.4"/>'
        '<circle cx="3" cy="8" r="1.4"/><circle cx="7" cy="8" r="1.4"/><circle cx="3" cy="13" r="1.4"/><circle cx="7" cy="13" r="1.4"/></g></svg></button>')


def shade(dce: float | None) -> str:
    if dce is None or (isinstance(dce, float) and math.isnan(dce)):
        return "bx"
    for i, (lo, hi) in enumerate(BINS):
        if lo <= dce < hi:
            return f"b{i}"
    return "b6"


def load(grid_path: str, manifest_path: str):
    g = json.load(open(grid_path))
    m = json.load(open(manifest_path))
    cells = defaultdict(dict)  # (task, rung) -> scheme -> rec
    for r in g["records"]:
        cells[(r["task"], r["rung"])][r["scheme"]] = r
    return g, m, cells


def task_columns(m: dict):
    """[(task, class, dropped)] : low rows, high rows, cpt80; dropped rows at the end of their class."""
    low, high, cpt = [], [], []
    for name, t in m["tasks"].items():
        if name == "cpt80":
            cpt.append((name, "cpt", False)); continue
        (low if t.get("ctc_class") == "low" else high).append((name, t.get("ctc_class"), bool(t.get("dropped"))))
    key = lambda x: (x[2], x[0])  # noqa: E731
    return sorted(low, key=key) + sorted(high, key=key) + cpt


def flop_ratio(rec) -> float:
    """Forward FLOPs of the construction / FLOPs of the full row, from collect_grid.flops_ratio
    (per-row lengths, per-layer Qwen3.5-4B costs incl. attention's causal T^2 term, skipped
    layers, selector cost). Falls back to the linear estimate for records collected before it."""
    if rec.get("flops") is not None:
        return rec["flops"]
    return rec["compaction"] * (1.0 - (rec.get("layer_frac") or 0.0)) + (rec.get("sel_cost") or 0.0)


def cell_view(cells, task, scheme, view):
    """Return dict(ce, dce, dce_se, compaction, eval_size, req, degenerate, n_rungs, notes) or None."""
    if view != "mean":
        rec = cells.get((task, view), {}).get(scheme)
        if rec is None:
            return None
        return dict(ce=rec["ce"], dce=rec["dce"], dce_se=rec["dce_se"], comp=rec["compaction"],
                    eval_size=rec["eval_size"], req=rec["rows_requested"], degenerate=rec["gold_degenerate"],
                    flops=flop_ratio(rec), n_rungs=1, rung_note=rec["rung_note"], ckpt_note=rec["ckpt_note"])
    recs = [cells[(task, r)][scheme] for r in RUNGS if scheme in cells.get((task, r), {})]
    if not recs:
        return None
    n = len(recs)
    return dict(ce=sum(r["ce"] for r in recs) / n, dce=sum(r["dce"] for r in recs) / n,
                dce_se=math.sqrt(sum(r["dce_se"] ** 2 for r in recs)) / n, comp=sum(r["compaction"] for r in recs) / n,
                flops=sum(flop_ratio(r) for r in recs) / n,
                eval_size=min(r["eval_size"] for r in recs), req=recs[0]["rows_requested"],
                degenerate=any(r["gold_degenerate"] for r in recs), n_rungs=n,
                rung_note="; ".join(sorted({r["rung_note"] for r in recs if r["rung_note"]})), ckpt_note=recs[0]["ckpt_note"])


def render_cell(cells, task, scheme, view, dropped):
    if dropped:
        return '<td class="na"><span class="miss">dropped</span></td>'
    v = cell_view(cells, task, scheme, view)
    if v is None:
        return '<td class="na"><span class="miss">not run yet</span></td>'
    flags = []
    if v["req"] and v["eval_size"] < v["req"]:
        flags.append(f"eval_size {v['eval_size']} < {v['req']} requested")
    elif view == "mean" and v["n_rungs"] < 3:
        flags.append(f"{v['n_rungs']}/3 rungs")
    if v["degenerate"]:
        flags.append(f"no gold set → = {GOLD_TWIN.get(scheme, '?')}")
    if v["rung_note"] and "stand" in v["rung_note"]:
        flags.append(v["rung_note"])
    title = f"{task} · {scheme} · {view}: CE {v['ce']:.4f}, ΔCE {v['dce']:+.4f} ± {v['dce_se']:.4f}, compaction {v['comp']:.3f}, eval_size {v['eval_size']}"
    if scheme == "full":
        body = (f'<span class="ce">{v["ce"]:.3f}</span><span class="sub">reference · eval_size {v["eval_size"]}</span>')
        cls = "ref"
    else:
        body = (f'<span class="ce">{v["ce"]:.3f}</span>'
                f'<span class="sub">Δ {v["dce"]:+.3f} <span class="se">±{v["dce_se"]:.3f}</span> · ×{v["comp"]:.2f}</span>')
        cls = shade(v["dce"])
    fl = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in flags)
    return f'<td class="{cls}" data-v="{v["dce"]:.5f}" title="{html.escape(title)}">{body}{fl}</td>'


def summary_block(cells, cols, view):
    """Per scheme: #tasks at parity (dCE <= 0.02), #tasks scored, worst task by dCE, mean compaction."""
    rows = []
    for s in SCHEMES:
        if s == "full":
            continue
        vals = []
        for task, _, dropped in cols:
            if dropped:
                continue
            v = cell_view(cells, task, s, view)
            if v is not None:
                vals.append((task, v["dce"], v["comp"]))
        if not vals:
            rows.append(f'<tr data-s="{s}"><th>{GRIP}{s}</th><td class="num">—</td><td class="num">—</td><td>—</td><td class="num">—</td></tr>')
            continue
        ok = sum(1 for _, d, _ in vals if d <= PARITY)
        worst = max(vals, key=lambda x: x[1])
        med = sorted(d for _, d, _ in vals)[len(vals) // 2]
        comp = sum(c for _, _, c in vals) / len(vals)
        cls = "b0" if ok == len(vals) else ""
        rows.append(f'<tr class="{cls}" data-s="{s}"><th>{GRIP}{s}</th><td class="num" data-v="{ok / len(vals):.4f}"><b>{ok}</b> / {len(vals)}</td>'
                    f'<td class="num" data-v="{med:.5f}">{med:+.3f}</td><td data-v="{worst[1]:.5f}">{html.escape(worst[0])} <span class="num">{worst[1]:+.3f}</span></td>'
                    f'<td class="num" data-v="{comp:.4f}">×{comp:.2f}</td></tr>')
    hdr = "".join(f'<th><button type="button" class="sortby" data-col="{i}" data-dir="{d}">{lab}<span class="arr" aria-hidden="true"></span></button></th>'
                  for i, (lab, d) in enumerate([("tasks at parity (ΔCE ≤ 0.02)", "desc"), ("median ΔCE", "asc"), ("worst task", "asc"), ("mean compaction", "asc")], 1))
    return ('<div class="tblwrap small"><table class="grid summ"><thead><tr><th>scheme</th>' + hdr +
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def agg_cells(cells, cols, scheme, view):
    """Two cells: mean and median dCE over the tasks this scheme has a value for in ``view``
    (dropped tasks and AGG_EXCLUDE left out), with the task count so partial rows read as partial."""
    ds, cs, fs = [], [], []
    for task, _, dropped in cols:
        if dropped or task in AGG_EXCLUDE:
            continue
        v = cell_view(cells, task, scheme, view)
        if v is not None:
            ds.append(v["dce"])
            cs.append(v["comp"])
            fs.append(v["flops"])
    n_all = sum(1 for t, _, d in cols if not d and t not in AGG_EXCLUDE)
    if not ds:
        return '<td class="na agg"><span class="miss">no cells</span></td>' * 3 + '<td class="na agg last"><span class="miss">no cells</span></td>'
    mean = sum(ds) / len(ds)
    srt = sorted(ds)
    med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
    cov = f"{len(ds)}/{n_all} tasks"
    out = []
    for lab, val in (("mean", mean), ("median", med)):
        cls = "ref" if scheme == "full" else shade(val)
        title = f"{scheme} · {view}: {lab} ΔCE over {cov} (excl. {', '.join(sorted(AGG_EXCLUDE))})"
        out.append(f'<td class="{cls} agg" data-v="{val:.5f}" title="{html.escape(title)}"><span class="ce">{val:+.3f}</span>'
                   f'<span class="sub">{cov}</span></td>')
    comp = sum(cs) / len(cs)
    title = f"{scheme} · {view}: mean compaction (compacted / full length) over {cov}"
    out.append(f'<td class="{"ref" if scheme == "full" else "bx"} agg" data-v="{comp:.4f}" title="{html.escape(title)}">'
               f'<span class="ce">×{comp:.2f}</span><span class="sub">{cov}</span></td>')
    fl = sum(fs) / len(fs)
    title = (f"{scheme} · {view}: mean forward-FLOP ratio vs full over {cov}: per-row Qwen3.5-4B cost incl. the"
             f" causal attention T² term, skipped layers and selector cost")
    out.append(f'<td class="{"ref" if scheme == "full" else "bx"} agg last" data-v="{fl:.4f}" title="{html.escape(title)}">'
               f'<span class="ce">×{fl:.2f}</span><span class="sub">{"" if fl <= 0 or scheme == "full" else (f"{1 / fl:.1f}× fewer · " if fl < 1 else f"{fl:.1f}× more · ")}{cov}</span></td>')
    return "".join(out)


def grid_table(cells, cols, view):
    groups = []
    for name, cls, _ in cols:
        g = {"low": "low-CTC (O(N))", "high": "high-CTC (O(N²)+)", "cpt": "continued pretraining"}[cls if cls in ("low", "high") else "cpt"]
        if not groups or groups[-1][0] != g:
            groups.append([g, 0])
        groups[-1][1] += 1
    ghdr = '<th colspan="4" class="grp agg last">across tasks</th>' + "".join(f'<th colspan="{n}" class="grp">{g}</th>' for g, n in groups)
    thdr = "".join(f'<th class="dropped">{html.escape(t)}</th>' if d else
                   f'<th><button type="button" class="sortby" data-col="{j}" data-dir="asc" data-name="{html.escape(t)}">{html.escape(t)}<span class="arr" aria-hidden="true"></span></button></th>'
                   for j, (t, _, d) in enumerate(cols, 5))
    thdr = ('<th class="agg"><button type="button" class="sortby" data-col="1" data-dir="asc" data-name="mean ΔCE">mean Δ<span class="arr" aria-hidden="true"></span></button></th>'
            '<th class="agg"><button type="button" class="sortby" data-col="2" data-dir="asc" data-name="median ΔCE">median Δ<span class="arr" aria-hidden="true"></span></button></th>'
            '<th class="agg"><button type="button" class="sortby" data-col="3" data-dir="asc" data-name="mean compaction">mean ×c<span class="arr" aria-hidden="true"></span></button></th>'
            '<th class="agg last"><button type="button" class="sortby" data-col="4" data-dir="asc" data-name="mean FLOPs">mean FLOPs<span class="arr" aria-hidden="true"></span></button></th>' + thdr)
    body = []
    for s in SCHEMES:
        tds = agg_cells(cells, cols, s, view) + "".join(render_cell(cells, t, s, view, d) for t, _, d in cols)
        body.append(f'<tr data-s="{s}"><th class="m">{GRIP}<span class="lab">{s}</span><span class="desc">{html.escape(SCHEME_DESC.get(s, s))}</span></th>{tds}</tr>')
    return (f'<div class="tblwrap"><table class="grid"><thead><tr><th class="m" rowspan="2">scheme</th>{ghdr}</tr>'
            f'<tr>{thdr}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def page(g, m, cells):
    cols = task_columns(m)
    n_done = len({k for k in cells})
    n_tasks_scored = len({t for t, _ in cells})
    kept = [t for t, _, d in cols if not d]
    views = "".join(f'<button aria-pressed="{"true" if v == "32k" else "false"}" data-v="{v}">{"mean over rungs" if v == "mean" else v}</button>' for v in VIEWS)
    sections = "".join(f'<div id="v{v}"{"" if v == "32k" else " hidden"}>{summary_block(cells, cols, v)}{grid_table(cells, cols, v)}</div>' for v in VIEWS)
    dropped = [t for t, _, d in cols if d]
    notes = m.get("notes", [])
    return f'''<title>Dev-Loss Scheme Grid</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F6F7F5;--ink:#1A1E1C;--mute:#66706B;--rule:#D6DBD7;--soft:#ECEFEC;--acc:#1F6F5B;--accbg:#E3F0EA;--warn:#9A5F0B;--bad:#A63D3D;--ref:#EEF0F2;--code:#F0F2EF;
 --s0:#e4eefb;--s1:#cde2fb;--s2:#9ec5f4;--s3:#6da7ec;--s4:#3987e5;--s5:#256abf;--s6:#0d366b;--t0:#1A1E1C;--t1:#1A1E1C;--t2:#1A1E1C;--t3:#1A1E1C;--t4:#ffffff;--t5:#ffffff;--t6:#ffffff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15181A;--ink:#E4E7E4;--mute:#98A19C;--rule:#2E3437;--soft:#1D2224;--acc:#6CC7A6;--accbg:#173129;--warn:#D9A23E;--bad:#E08585;--ref:#1C2124;--code:#1D2224;
 --s0:#1b2530;--s1:#0d366b;--s2:#184f95;--s3:#256abf;--s4:#3987e5;--s5:#6da7ec;--s6:#9ec5f4;--t0:#E4E7E4;--t1:#E4E7E4;--t2:#ffffff;--t3:#ffffff;--t4:#ffffff;--t5:#0d1a2b;--t6:#0d1a2b}}}}
:root[data-theme="dark"]{{--bg:#15181A;--ink:#E4E7E4;--mute:#98A19C;--rule:#2E3437;--soft:#1D2224;--acc:#6CC7A6;--accbg:#173129;--warn:#D9A23E;--bad:#E08585;--ref:#1C2124;--code:#1D2224;
 --s0:#1b2530;--s1:#0d366b;--s2:#184f95;--s3:#256abf;--s4:#3987e5;--s5:#6da7ec;--s6:#9ec5f4;--t0:#E4E7E4;--t1:#E4E7E4;--t2:#ffffff;--t3:#ffffff;--t4:#ffffff;--t5:#0d1a2b;--t6:#0d1a2b}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:15px;line-height:1.5;margin:0}}
.wrap{{max-width:1600px;margin:0 auto;padding:40px 28px 80px}}
h1{{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:34px;line-height:1.1;margin:0 0 6px;text-wrap:balance}}
h2{{font-family:Fraunces,Georgia,serif;font-weight:500;font-size:22px;margin:0 0 4px;text-wrap:balance}}
.sub{{color:var(--mute);max-width:80ch;margin:0}}
.meta{{display:flex;flex-wrap:wrap;gap:8px 22px;color:var(--mute);font-size:13px;margin:16px 0 0;padding:12px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.meta b{{color:var(--ink);font-weight:600}}
.meta span{{overflow-wrap:anywhere;min-width:0}}
section{{margin-top:36px}}
.lede{{max-width:80ch;margin:6px 0 14px}}
.tblwrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;margin-top:14px}}
.tblwrap.small{{max-width:760px}}
table.grid{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
table.grid th,table.grid td{{padding:7px 9px;vertical-align:top;border-bottom:1px solid var(--rule);text-align:left;border-right:1px solid var(--rule)}}
table.grid thead th{{font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--mute);background:var(--soft);font-weight:600;white-space:nowrap}}
table.grid thead th.grp{{font-family:Fraunces,Georgia,serif;text-transform:none;letter-spacing:0;font-size:13.5px;font-weight:500;color:var(--ink);text-align:center}}
table.grid thead th.dropped{{color:var(--mute);font-weight:400;text-decoration:line-through}}
table.grid th.m{{min-width:190px;font-weight:400;background:var(--bg);position:sticky;left:0;z-index:1}}
.lab{{display:block;font-weight:600;font-family:"JetBrains Mono",monospace;font-size:13px}}
.desc{{display:block;color:var(--mute);font-size:11.5px;line-height:1.3;margin-top:1px}}
td .ce{{display:block;font-family:"JetBrains Mono",monospace;font-size:15px;font-weight:500;line-height:1.2}}
td .sub{{color:inherit;display:block;font-family:"JetBrains Mono",monospace;font-size:10.5px;opacity:.85;white-space:nowrap}}
td .se{{opacity:.7}}
td .flag{{display:block;font-size:10.5px;font-style:italic;opacity:.85;max-width:16ch;line-height:1.25;margin-top:2px}}
td .miss{{font-size:11.5px;color:var(--mute);font-style:italic}}
td.na{{background:var(--bg)}}
td.ref{{background:var(--ref)}}
td.bx{{background:var(--soft)}}
td.b0{{background:var(--s0);color:var(--t0)}} td.b1{{background:var(--s1);color:var(--t1)}} td.b2{{background:var(--s2);color:var(--t2)}}
td.b3{{background:var(--s3);color:var(--t3)}} td.b4{{background:var(--s4);color:var(--t4)}} td.b5{{background:var(--s5);color:var(--t5)}} td.b6{{background:var(--s6);color:var(--t6)}}
table.summ th{{font-weight:600;font-family:"JetBrains Mono",monospace;font-size:13px}}
table.summ td.num,table.summ .num{{font-family:"JetBrains Mono",monospace}}
table.summ tr.b0 th{{color:var(--acc)}}
.views{{display:flex;gap:6px;margin:14px 0 0;flex-wrap:wrap}}
.views button{{font:inherit;font-size:13px;font-weight:600;padding:5px 14px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);border-radius:3px;cursor:pointer}}
.views button[aria-pressed="true"]{{background:var(--ink);color:var(--bg);border-color:var(--ink)}}
.views button:focus-visible{{outline:2px solid var(--acc);outline-offset:2px}}
.legend{{display:flex;gap:4px;align-items:center;font-size:12px;color:var(--mute);margin-top:10px;flex-wrap:wrap}}
.legend span.sw{{display:inline-block;width:34px;height:14px;border:1px solid var(--rule)}}
.foot{{font-size:12.5px;color:var(--mute);max-width:90ch;margin-top:10px}}
.caution{{border-left:3px solid var(--warn);padding:8px 14px;background:var(--soft);max-width:80ch;margin:14px 0}}
ul.notes{{max-width:90ch;font-size:13.5px;color:var(--mute)}}
table.grid td.agg.last,table.grid th.agg.last{{border-right:3px double var(--rule)}}
.grip{{all:unset;display:inline-flex;align-items:center;justify-content:center;width:18px;height:26px;margin:-2px 6px 0 -4px;float:left;color:var(--mute);cursor:grab;touch-action:none;border-radius:3px}}
.grip svg{{width:10px;height:16px}}
.grip:hover{{color:var(--ink);background:var(--soft)}}
.grip:focus-visible{{outline:2px solid var(--acc);outline-offset:1px;color:var(--ink)}}
table.summ th .grip{{height:18px;margin-top:0}}
tr.dragging > *{{background:var(--accbg)!important;color:var(--ink)!important}}
tr.dragging .grip{{cursor:grabbing;color:var(--acc)}}
body.is-dragging{{user-select:none;-webkit-user-select:none;cursor:grabbing}}
button.sortby{{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:4px;border-radius:2px}}
button.sortby:hover{{color:var(--ink)}}
button.sortby:focus-visible{{outline:2px solid var(--acc);outline-offset:2px}}
button.sortby .arr{{display:inline-block;width:8px;font-size:10px}}
button.sortby[data-active="asc"] .arr::after{{content:"▲"}}
button.sortby[data-active="desc"] .arr::after{{content:"▼"}}
button.sortby[data-active]{{color:var(--acc)}}
.ordbar{{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin:12px 0 0;font-size:13px;color:var(--mute)}}
.ordbar b{{color:var(--ink);font-weight:600}}
.ordbar button{{font:inherit;font-size:12.5px;font-weight:600;padding:3px 12px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);border-radius:3px;cursor:pointer}}
.ordbar button:disabled{{opacity:.45;cursor:default}}
.ordbar button:focus-visible{{outline:2px solid var(--acc);outline-offset:2px}}
.sr{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}}
</style>
<div class="wrap">
<h1>Dev-Loss Scheme Grid</h1>
<p class="sub">Answer-token cross-entropy of a frozen dense checkpoint under general token-selection schemes. Columns are the olmo-eval CTC rows (plus continued pretraining), rows are schemes, cells are dev loss — lighter is closer to full attention.</p>
<div class="meta"><span><b>Model</b> per-task dense Qwen3.5-4B suite checkpoints; cpt80 on the repaired 4B base</span><span><b>Metric</b> teacher-forced CE on the gold answer tokens (cpt80: last 20% of the document)</span><span><b>Rows scored</b> {html.escape(str(sorted({r["rows_requested"] for r in g["records"] if r["rows_requested"]})))} per rung requested ⚠ small sets, see SE</span><span><b>Cells</b> {n_done} task×rung done · {n_tasks_scored}/{len(kept)} tasks touched</span><span><b>Source</b> {html.escape(g["results_root"])} via <code>collect_grid.py</code></span></div>

<div class="caution"><b>Δ is paired.</b> ΔCE = mean over rows of (CE<sub>scheme</sub> − CE<sub>full</sub>) on the same rows, ± its standard error. Parity threshold used for the summary: ΔCE ≤ 0.02. Compaction ×c = compacted length / full length. eval sizes are 32–64 rows per rung, so CE is a noisy screen, not a result: read rows against each other within a column.</div>

<section id="grid">
<h2>Scheme × task</h2>
<div class="views" role="tablist">{views}</div>
<div class="legend">ΔCE vs full: <span class="sw" style="background:var(--s0)"></span>≤0.02 <span class="sw" style="background:var(--s1)"></span>0.05 <span class="sw" style="background:var(--s2)"></span>0.10 <span class="sw" style="background:var(--s3)"></span>0.20 <span class="sw" style="background:var(--s4)"></span>0.40 <span class="sw" style="background:var(--s5)"></span>0.80 <span class="sw" style="background:var(--s6)"></span>&gt;0.80</div>
<div class="ordbar"><span>Row order: <b id="ordstate">default</b></span><button type="button" id="ordreset" disabled>Reset order</button><span>Drag a row by its grip (or focus the grip and press ↑/↓). Click a task or summary header to sort by it; click again to flip, a third time to return to your own order. One order applies to every table and view.</span></div>
<div class="sr" aria-live="polite" id="ordlive"></div>
{sections}
<p class="foot">All schemes are header-free (K counts from each document's first token), gold-blind and keep 0 whole documents with a <code>cent_cmean</code> slot unless the name says otherwise; gold schemes keep the task's own gold document set real and fall back to their gold-blind twin where a task has no gold subset (oolong, reorder, grouping, cpt80 — flagged in-cell). Dropped this pass (no 4B dense checkpoint): {html.escape(", ".join(dropped)) or "none"}. obliq is scored on the <code>obliq_retrieval</code> checkpoint; absence and reorder ladders top out at 16k, which stands in for the 32k column. The <b>mean FLOPs</b> column is the forward cost as a fraction of the full row, costed per row at its real length on Qwen3.5-4B: every block pays 2 × its parameters per token, GatedDeltaNet blocks add their delta-rule state update, and the 8 attention blocks add the causal QKᵀ + AV term 2·16·256·T², which is what makes compaction pay off more than linearly at long rungs (at 32k it exceeds an attention block's linear cost). Skipped columns drop out of the layers they skip, as queries and keys, and selector cost is added in full-forward equivalents (grad20 3, attnrow20 1). The skip savings assume an implementation that actually drops the work; the eval computes it and discards it. The <b>mean Δ / median Δ / mean ×c</b> columns summarise each row's paired ΔCE and compaction over the tasks it has a cell for in the current view (count shown under each value), leaving out xabsence (void checkpoint) and cpt80 (not a CTC task); a row with fewer tasks is not directly comparable to a full one.</p>
<div class="caution" id="layerskip"><b>Layer skip (added 2026-09-22).</b> The three <code>gold_fl20p8_skip*</code> rows keep <code>gold_fl20p8</code>'s token selection, then let the tokens of non-gold documents bypass whole blocks. That removes 26–45% of the token×layer work at 2k–32k (19–34% for <code>skipgdn2</code>). Qwen3.5-4B stacks blocks as gdn, gdn, gdn, attn. <b>Skipping the odd blocks, which include every attention block, is nearly free:</b> its median extra ΔCE over <code>gold_fl20p8</code> is +0.004 / +0.001 / +0.015 at 2k / 8k / 32k. <b>Skipping GDN blocks alone is not:</b> <code>skipeven</code> costs +0.04 / +0.16 / +0.26 and <code>skipgdn2</code> +0.06 / +0.12 / +0.23. The xabsence column reads below full under every skip scheme; it is void, because a paraphrase-era checkpoint is scored on EXACT rows. Eval sizes are 16/16/8 rows per rung ⚠, so read these rows against <code>gold_fl20p8</code> within each column.</div>
</section>

{cpt_section()}
<section id="notes">
<h2>Provenance</h2>
<ul class="notes">{"".join(f"<li>{html.escape(n)}</li>" for n in notes)}<li>Driver <code>debug/devloss_grid/ctc_devloss_grid.py</code>: rows rendered with the suite's own chat-template + per-document marker layout (the layout the 4B checkpoints trained and were graded on), CE at answer positions through <code>Transformer._compact_pooled_soft_tokens</code>.</li></ul>
</section>
</div>
<script>
document.querySelectorAll('.views button').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('.views button').forEach(x=>x.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true');
  {json.dumps(VIEWS)}.forEach(k=>{{document.getElementById('v'+k).hidden=(k!==b.dataset.v);}});
}}));
{ORDER_JS}
</script>
'''


# ---- CPT dev-loss vs training FLOPs (soft-detach CPT campaign, 2026-09-21/22) ----------------
# sd20 is split by training shard (records/softdetach-cpt-crossover.md A.5): <=128M on cpt_u128M,
# the 1B-shard series is a true prefix family (128M-s1B, 256M, 512M, 1B).
CPT_CURVES = {
    "dense": [(196, 1.314, "4M"), (392, 1.303, "8M"), (759, 1.296, "16M"), (1518, 1.283, "32M"), (3011, 1.256, "64M"), (5998, 1.245, "128M")],
    "sd20": [(97, 1.309, "16M"), (193, 1.291, "32M"), (383, 1.279, "64M"), (763, 1.272, "128M")],
    "sd20_1B": [(763, 1.2812, "128M"), (1523, 1.2725, "256M"), (3043, 1.2634, "512M"), (5942, 1.2585, "1B")],
    "sfl20": [(172, 1.308, "32M"), (340, 1.295, "64M"), (678, 1.288, "128M")],
    "lslot20": [(193, 1.320, "32M"), (383, 1.320, "64M"), (763, 1.320, "128M")],
    "sd20mix": [(1945, 1.2668, "256M"), (5122, 1.2527, "512M")],
    "sd20p32": [(2076, 1.2655, "256M"), (4148, 1.2575, "512M")],
}
CPT_BASE = 1.320
CPT_DESC = {
    "dense": "full attention, every token",
    "sd20": "random 20% of 512-token blocks whole, rest one detached slot; 128M-token shard",
    "sd20_1B": "same arm trained on the 1B-token shard (its own prefix family)",
    "sfl20": "first_last 20% per block",
    "lslot20": "learned slot, backbone frozen",
    "sd20mix": "sd20 + uncompressed-row curriculum, p(full) 0 → 0.5 over the run",
    "sd20p32": "sd20 + first 32 real tokens kept in every pooled block",
}
CPT_LABEL = {"sd20_1B": "sd20 · 1B shard"}
# position-binned full-attention dev CE (eval_cpt_devloss.py full_pos_*; weka softdetach_cpt/devloss_pos/)
CPT_POS_BINS = ["0–2k", "2–8k", "8–16k", "16–32k", "32k–64k", "all"]
CPT_POS = [
    ("dense", "32M", 1518, [1.462, 1.314, 1.260, 1.226, 1.301, 1.283]),
    ("dense", "64M", 3011, [1.387, 1.281, 1.234, 1.200, 1.277, 1.256]),
    ("dense", "128M", 5998, [1.369, 1.268, 1.223, 1.190, 1.266, 1.245]),
    ("sd20", "64M", 383, [1.399, 1.305, 1.257, 1.223, 1.301, 1.279]),
    ("sd20", "256M", 1523, [1.390, 1.293, 1.250, 1.216, 1.295, 1.2725]),
    ("sd20mix", "256M", 1945, [1.386, 1.290, 1.245, 1.211, 1.289, 1.2668]),
    ("sd20p32", "256M", 2076, [1.3832, 1.2878, 1.2414, 1.2097, 1.2883, 1.2655]),
    ("sd20", "512M", 3043, [1.375, 1.282, 1.239, 1.208, 1.287, 1.2634]),
    ("sd20p32", "512M", 4148, [1.3710, 1.2785, 1.2339, 1.2014, 1.2808, 1.2575]),
    ("sd20mix", "512M", 5122, [1.367, 1.274, 1.230, 1.197, 1.275, 1.2527]),
    ("sd20", "1B", 5942, [1.369, 1.275, 1.234, 1.202, 1.283, 1.2585]),
]
# downstream raw-CPT nq (retrieval f1, eval_size 500 per rung, --prompt-format raw)
CPT_NQ = [
    ("dense", "16M", [0.734, 0.464, 0.180, 0.060]),
    ("dense", "32M", [0.484, 0.242, 0.104, 0.044]),
    ("dense", "64M", [0.642, 0.368, 0.172, 0.066]),
    ("dense", "128M", [0.374, 0.242, 0.098, 0.040]),
    ("sd20", "64M", [0.730, 0.404, 0.158, 0.050]),
    ("sd20", "128M", [0.642, 0.356, 0.118, 0.046]),
    ("sfl20", "64M", [0.222, 0.136, 0.064, 0.030]),
]


def _dense_at(pf: float, col: int) -> float | None:
    """Log-linear interpolation of the dense position-binned CE at ``pf`` (anchors 1518/3011/5998)."""
    anchors = [(p, v[col]) for a, _, p, v in CPT_POS if a == "dense"]
    for (p0, v0), (p1, v1) in zip(anchors, anchors[1:]):
        if p0 <= pf <= p1:
            return v0 + (v1 - v0) * math.log(pf / p0) / math.log(p1 / p0)
    for p, v in anchors:
        if abs(math.log(pf / p)) < 0.02:
            return v
    return None


def cpt_pos_table() -> str:
    head = "".join(f"<th>{b}</th>" for b in CPT_POS_BINS)
    rows = []
    for arm, tok, pf, vals in sorted(CPT_POS, key=lambda r: r[2]):
        cells = []
        for i, v in enumerate(vals):
            if arm == "dense":
                cells.append(f"<td>{v:.3f}</td>")
                continue
            ref = _dense_at(pf, i)
            note = ""
            if ref is None:  # sd20-64M sits at 1/4 of dense-32M's PF: the parity comparison
                ref, note = CPT_POS[0][3][i], "†"
            d = v - ref
            cls = "neg" if d <= -0.005 else ("pos" if d >= 0.005 else "")
            cells.append(f'<td>{v:.3f}<span class="dd {cls}">{d:+.3f}{note}</span></td>')
        lab = f"{arm}-{tok}"
        rows.append(f'<tr class="{"dense" if arm == "dense" else ""}"><td>{lab}</td><td>{pf}</td>{"".join(cells)}</tr>')
    return (f'<div class="tblwrap"><table class="ct pos"><thead><tr><th>run</th><th>PF</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def cpt_nq_table() -> str:
    rows = "".join(f"<tr><td>{a}-{t}</td>{''.join(f'<td>{v:.3f}</td>' for v in vals)}</tr>" for a, t, vals in CPT_NQ)
    return (f'<div class="tblwrap small"><table class="ct"><thead><tr><th>run</th><th>3k</th><th>8k</th><th>16k</th><th>32k</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


CPT_HTML = """
<section id="cpt" class="cptchart">
<style>
.cptchart{--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300;--surf:var(--bg);--neg:#1F6F5B;--pos:#A63D3D}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .cptchart{--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300;--neg:#6CC7A6;--pos:#E08585}}
:root[data-theme="dark"] .cptchart{--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300;--neg:#6CC7A6;--pos:#E08585}
.cptchart .chartwrap{position:relative;max-width:900px}
.cptchart svg{width:100%;height:auto;display:block;font-family:"Source Sans 3",system-ui,sans-serif}
.cptchart .axis text,.cptchart .lbl{fill:var(--mute);font-size:12px}
.cptchart .grid line{stroke:var(--rule);stroke-width:1}
.cptchart .axis line,.cptchart .axis path{stroke:var(--rule)}
.cptchart .base{stroke:var(--mute);stroke-dasharray:4 4;stroke-width:1.5}
.cptchart .ser path{fill:none;stroke-width:2}
.cptchart .ser circle{stroke:var(--surf);stroke-width:2}
.cptchart .hit{fill:transparent;cursor:crosshair}
.cptchart .tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:6px 9px;border-radius:3px;white-space:nowrap;transform:translate(-50%,-115%);display:none;font-variant-numeric:tabular-nums}
.cptchart .cleg{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:6px 22px;font-size:13px;margin:8px 0 4px;max-width:900px}
.cptchart .cleg svg{display:inline-block;width:22px;height:8px;vertical-align:middle;margin-right:6px}
.cptchart .cleg b{font-weight:600;font-family:"JetBrains Mono",monospace;font-size:12.5px}
.cptchart .cleg span{color:var(--mute)}
.cptchart table.ct{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12.5px}
.cptchart table.ct th,.cptchart table.ct td{padding:4px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
.cptchart table.ct th:first-child,.cptchart table.ct td:first-child{text-align:left;font-family:"JetBrains Mono",monospace}
.cptchart table.ct thead th{background:var(--soft);color:var(--mute);font-weight:600;font-size:11.5px;letter-spacing:.04em}
.cptchart table.pos tr.dense td{background:var(--ref)}
.cptchart .dd{display:block;font-size:11px;font-family:"JetBrains Mono",monospace;color:var(--mute)}
.cptchart .dd.neg{color:var(--neg);font-weight:600}
.cptchart .dd.pos{color:var(--pos);font-weight:600}
.cptchart h3{font-family:Fraunces,Georgia,serif;font-weight:500;font-size:17px;margin:26px 0 2px}
.cptchart .tblwrap{display:inline-block;max-width:100%;vertical-align:top}
.cptchart details summary{cursor:pointer;color:var(--mute);font-size:13px;margin-top:8px}
.cptchart .verdict{border-left:3px solid var(--acc);background:var(--accbg);padding:10px 14px;max-width:80ch;margin:14px 0 4px}
</style>
<h2>CPT: dev loss vs training FLOPs</h2>
<p class="lede">Continued pretraining of the marker-repaired Qwen3.5-4B base on dolma3+longmino (64k rows), dense vs soft-detached pooling, scored with full attention on 32 held-out 64k rows ⚠ small dev set: seed-to-seed spread is ±0.004, so steps under ~0.005 are not resolved. The x-axis is the FLOP meter's actual petaFLOPs, log scale. Dashed orange is the same sd20 arm on the larger 1B-token shard: it runs ~0.009 above the 128M-shard series at equal tokens, so the two are drawn as separate lines rather than one curve.</p>
<div class="verdict"><b>Updated 2026-09-22.</b> sd20 matches dense-32M at ¼ of its FLOPs, but dense overtakes it near 2k PF, and neither fix tried at the crossover holds a margin. The uncompressed-row curriculum (sd20mix) and the 32-token real prefix (sd20p32) both land <i>on</i> the dense curve: −0.006 and −0.005 near 2k PF, +0.005 and +0.007 at 4–5k PF. Soft-detached CPT is a cheap-end technique, a ~4× saving up to roughly 1.5k PF on this model, not a dense replacement at scale.</div>
<div class="cleg" id="cleg"></div>
<div class="chartwrap"><svg id="cptsvg" viewBox="0 0 900 430" role="img" aria-label="Dev loss versus training petaFLOPs for dense and soft-detached CPT arms"></svg><div class="tip" id="cpttip"></div></div>
<p class="foot">Δ in the hover and table views is against dense log-linearly interpolated at the same PF. sd20mix costs 1.3–1.7× the PF of plain sd20 at the same tokens because each uncompressed row is ~7.7× a compacted one. Every soft arm's saving is training-only: under its own compressed input sd20 reads 1.47–1.53. Runs: wandb group <code>sdcpt-q35-4b</code>. Records: <code>records/softdetach-cpt-plan.md</code>, <code>records/softdetach-cpt-crossover.md</code>.</p>
<details><summary>Table view</summary><div class="tblwrap"><table class="ct" id="cpttable"></table></div></details>

<h3>Where the difference lives, by position in the 64k row</h3>
<p class="lede">Full-attention dev CE binned by the target token's position. The small figure under each soft-arm value is Δ against dense interpolated at the same PF (green = soft arm better, red = worse). sd20's whole advantage sits in the first 2k tokens, and its deficit grows with position and with budget. The two fixes recover the long-range bins only as far as the dense curve.</p>
__CPT_POS_TABLE__
<p class="foot">† sd20-64M is compared with dense-32M at 4× its PF: this is the parity-at-¼-FLOPs point. Dense anchors are 32M/64M/128M; points between them are interpolated per bin in log PF.</p>

<h3>Downstream: raw-CPT nq retrieval</h3>
<p class="lede">The same checkpoints on the CTC nq ladder, raw prompt format, eval_size 500 per rung (SE ≈ ±0.02 at f1 0.5). The dense column is not monotone in tokens, so nq cannot rank the arms: dense-16M beats dense-128M at every rung. Contradiction reads ~0 for every raw checkpoint. Oolong scores were not captured because its logs print <code>score=</code>, which the collector misses, and its per-example lines show no parsed prediction. The three base-model anchor jobs failed.</p>
__CPT_NQ_TABLE__
<script>
(function(){
  const D = __CPT_JSON__;
  const base = __CPT_BASE__;
  const desc = __CPT_DESC__;
  const lab = __CPT_LABEL__;
  const name = s => lab[s] || s;
  const order = ["dense","sd20","sd20_1B","sfl20","lslot20","sd20mix","sd20p32"];
  const col = {dense:"var(--c1)", sd20:"var(--c2)", sd20_1B:"var(--c2)", sfl20:"var(--c3)", lslot20:"var(--c4)", sd20mix:"var(--c5)", sd20p32:"var(--c6)"};
  const dash = {sd20_1B:"6 4"};
  const W=900,H=430,ml=64,mr=110,mt=18,mb=46;
  const x0=80,x1=9000,y0=1.24,y1=1.33;
  const sx=v=>ml+(Math.log10(v)-Math.log10(x0))/(Math.log10(x1)-Math.log10(x0))*(W-ml-mr);
  const sy=v=>mt+(y1-v)/(y1-y0)*(H-mt-mb);
  const NS="http://www.w3.org/2000/svg";
  const svg=document.getElementById("cptsvg");
  const el=(n,a,p)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e;};
  const grid=el("g",{class:"grid"}), axis=el("g",{class:"axis"});
  for(let v=y0;v<=y1+1e-9;v+=0.02){el("line",{x1:ml,x2:W-mr,y1:sy(v),y2:sy(v)},grid);const t=el("text",{x:ml-8,y:sy(v)+4,"text-anchor":"end"},axis);t.textContent=v.toFixed(2);}
  for(const v of [100,200,500,1000,2000,5000]){el("line",{x1:sx(v),x2:sx(v),y1:H-mb,y2:H-mb+5},axis);const t=el("text",{x:sx(v),y:H-mb+19,"text-anchor":"middle"},axis);t.textContent=v>=1000?(v/1000)+"k":v;}
  el("path",{d:`M${ml},${H-mb}H${W-mr}`,fill:"none"},axis);
  let t=el("text",{x:(ml+W-mr)/2,y:H-8,"text-anchor":"middle",class:"lbl"});t.textContent="training compute (actual petaFLOPs, log)";
  t=el("text",{x:14,y:mt+(H-mt-mb)/2,transform:`rotate(-90 14 ${mt+(H-mt-mb)/2})`,"text-anchor":"middle",class:"lbl"});t.textContent="dev CE (32 rows × 64k, full attention)";
  el("line",{x1:ml,x2:W-mr,y1:sy(base),y2:sy(base),class:"base"});
  t=el("text",{x:W-mr+6,y:sy(base)+4,class:"lbl"});t.textContent="frozen base "+base.toFixed(3);
  const denseAt=pf=>{const d=D.dense;for(let i=0;i+1<d.length;i++){const a=d[i],b=d[i+1];if(a[0]<=pf&&pf<=b[0])return a[1]+(b[1]-a[1])*Math.log(pf/a[0])/Math.log(b[0]/a[0]);}return null;};
  const pts=[];
  for(const s of order){
    const g=el("g",{class:"ser"});
    const d=D[s].map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+","+sy(p[1]).toFixed(1)).join("");
    const pa={d,stroke:col[s]}; if(dash[s]) pa["stroke-dasharray"]=dash[s];
    el("path",pa,g);
    for(const p of D[s]){el("circle",{cx:sx(p[0]),cy:sy(p[1]),r:4,fill:dash[s]?"var(--surf)":col[s],stroke:dash[s]?col[s]:"var(--surf)"},g);pts.push({s,p});}
  }
  const leg=document.getElementById("cleg");
  for(const s of order){const d=document.createElement("div");d.innerHTML=`<svg viewBox="0 0 22 8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" stroke="${col[s]}" stroke-width="2.5" ${dash[s]?`stroke-dasharray="${dash[s]}"`:""}/></svg><b>${name(s)}</b> <span>${desc[s]}</span>`;leg.appendChild(d);}
  const tip=document.getElementById("cpttip"), hit=el("rect",{x:ml,y:mt,width:W-ml-mr,height:H-mt-mb,class:"hit"});
  const fmtD=(s,p)=>{if(s==="dense")return "";const r=denseAt(p[0]);return r===null?"":`  (dense at same PF ≈ ${r.toFixed(3)}, Δ ${(p[1]-r>=0?"+":"")+(p[1]-r).toFixed(3)})`;};
  hit.addEventListener("mousemove",ev=>{
    const r=svg.getBoundingClientRect();const mx=(ev.clientX-r.left)*W/r.width, my=(ev.clientY-r.top)*H/r.height;
    let best=null,bd=1e9;for(const q of pts){const dx=sx(q.p[0])-mx,dy=sy(q.p[1])-my;const dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=q;}}
    if(!best||bd>40*40){tip.style.display="none";return;}
    tip.style.display="block";tip.style.left=(sx(best.p[0])*r.width/W)+"px";tip.style.top=(sy(best.p[1])*r.height/H)+"px";
    tip.textContent=`${name(best.s)}-${best.p[2]} · ${best.p[0].toFixed(0)} PF · CE ${best.p[1].toFixed(4)}`+fmtD(best.s,best.p);
  });
  hit.addEventListener("mouseleave",()=>{tip.style.display="none";});
  const tb=document.getElementById("cpttable");
  tb.innerHTML="<thead><tr><th>arm</th><th>tokens</th><th>PF</th><th>dev CE</th><th>dense at same PF</th><th>Δ vs dense</th></tr></thead><tbody>"+order.flatMap(s=>D[s].map(p=>{const r=s==="dense"?null:denseAt(p[0]);return `<tr><td>${name(s)}</td><td>${p[2]}</td><td>${p[0]}</td><td>${p[1].toFixed(4)}</td><td>${r===null?"—":r.toFixed(3)}</td><td>${r===null?"—":((p[1]-r>=0?"+":"")+(p[1]-r).toFixed(3))}</td></tr>`;})).join("")+"</tbody>";
})();
</script>
</section>
"""
def cpt_section() -> str:
    import json as _j
    return (CPT_HTML.replace("__CPT_JSON__", _j.dumps(CPT_CURVES)).replace("__CPT_BASE__", repr(CPT_BASE))
            .replace("__CPT_DESC__", _j.dumps(CPT_DESC)).replace("__CPT_LABEL__", _j.dumps(CPT_LABEL))
            .replace("__CPT_POS_TABLE__", cpt_pos_table()).replace("__CPT_NQ_TABLE__", cpt_nq_table()))


# Row ordering: one global scheme order shared by every grid/summary table in every view. Custom
# order (drag, arrow keys) is remembered per viewer in localStorage; a header click is a transient
# sort on top of it. `full` stays pinned first in the grid (it is the reference row).
ORDER_JS = r"""
(function(){
  const KEY="devloss-grid-order-v2";
  const bodies=[...document.querySelectorAll("#grid table.grid tbody")].filter(tb=>tb.querySelector("tr[data-s]"));
  const DEFAULT=[...bodies[bodies.length-1].querySelectorAll("tr[data-s]")].map(r=>r.dataset.s);
  let custom=null; try{custom=JSON.parse(localStorage.getItem(KEY)||"null");}catch(e){}
  if(!Array.isArray(custom)) custom=null;
  let sort=null; // {btn, col, dir, table}
  const state=document.getElementById("ordstate"), reset=document.getElementById("ordreset"), live=document.getElementById("ordlive");
  const baseOrder=()=>{const o=(custom||DEFAULT).filter(s=>DEFAULT.includes(s));for(const s of DEFAULT)if(!o.includes(s))o.push(s);return o;};
  function apply(order){
    for(const tb of bodies){
      const rows=new Map([...tb.querySelectorAll("tr[data-s]")].map(r=>[r.dataset.s,r]));
      const seq=[...order]; if(rows.has("full")&&!tb.closest("table").classList.contains("summ")){seq.splice(seq.indexOf("full"),1);seq.unshift("full");}
      for(const s of seq){const r=rows.get(s); if(r) tb.appendChild(r);}
    }
  }
  function label(){
    if(sort){const n=sort.btn.dataset.name||sort.btn.textContent.trim(); const v=sort.btn.closest("[id^=v]").id.slice(1);
      state.textContent=`sorted by ${n} (${v==="mean"?"mean over rungs":v}) ${sort.dir==="asc"?"ascending":"descending"}`;}
    else state.textContent=custom?"your order":"default";
    reset.disabled=!custom&&!sort;
    document.querySelectorAll("button.sortby").forEach(b=>{if(!sort||b!==sort.btn)b.removeAttribute("data-active");else b.dataset.active=sort.dir;});
  }
  function sortedOrder(){
    const tb=sort.btn.closest("table").querySelector("tbody"); const col=+sort.btn.dataset.col; const sign=sort.dir==="asc"?1:-1;
    const base=baseOrder(); const val=new Map();
    for(const r of tb.querySelectorAll("tr[data-s]")){const c=r.children[col]; const v=c&&c.dataset.v!==undefined?parseFloat(c.dataset.v):NaN; val.set(r.dataset.s,v);}
    return [...base].sort((a,b)=>{const va=val.get(a),vb=val.get(b);const na=!Number.isFinite(va),nb=!Number.isFinite(vb);
      if(na||nb) return na-nb || base.indexOf(a)-base.indexOf(b); return sign*(va-vb) || base.indexOf(a)-base.indexOf(b);});
  }
  function refresh(){apply(sort?sortedOrder():baseOrder());label();}
  function save(order){custom=order; try{localStorage.setItem(KEY,JSON.stringify(order));}catch(e){}}
  // freeze whatever is on screen as the custom order (used when a drag starts from a sorted view)
  function currentOrder(tb){const shown=[...tb.querySelectorAll("tr[data-s]")].map(r=>r.dataset.s); for(const s of baseOrder()) if(!shown.includes(s)) shown.push(s); return shown;}
  document.querySelectorAll("button.sortby").forEach(b=>b.addEventListener("click",()=>{
    if(sort&&sort.btn===b){ if(sort.dir===b.dataset.dir) sort.dir=(b.dataset.dir==="asc"?"desc":"asc"); else sort=null; }
    else sort={btn:b,dir:b.dataset.dir};
    refresh(); live.textContent=state.textContent;
  }));
  reset.addEventListener("click",()=>{custom=null;sort=null;try{localStorage.removeItem(KEY);}catch(e){} refresh(); live.textContent="Row order reset";});
  function commit(tb){const o=currentOrder(tb); sort=null; save(o); apply(o); label();}
  // pointer drag (mouse, pen, touch)
  let drag=null;
  document.addEventListener("pointerdown",ev=>{
    const g=ev.target.closest(".grip"); if(!g||ev.button>0) return;
    const row=g.closest("tr[data-s]"); if(row.dataset.s==="full"&&!row.closest("table").classList.contains("summ")) return;
    ev.preventDefault(); g.setPointerCapture(ev.pointerId);
    drag={row,tb:row.parentElement,g}; row.classList.add("dragging"); document.body.classList.add("is-dragging");
  });
  document.addEventListener("pointermove",ev=>{
    if(!drag) return;
    const el=document.elementFromPoint(ev.clientX,ev.clientY); const over=el&&el.closest("tr[data-s]");
    if(!over||over===drag.row||over.parentElement!==drag.tb) return;
    if(over.dataset.s==="full"&&!over.closest("table").classList.contains("summ")) return;
    const r=over.getBoundingClientRect(); const after=ev.clientY>r.top+r.height/2;
    drag.tb.insertBefore(drag.row, after?over.nextSibling:over);
  });
  const end=()=>{ if(!drag) return; drag.row.classList.remove("dragging"); document.body.classList.remove("is-dragging");
    const tb=drag.tb, s=drag.row.dataset.s; drag=null; commit(tb); live.textContent=`Moved ${s}`; };
  document.addEventListener("pointerup",end); document.addEventListener("pointercancel",end);
  // keyboard
  document.addEventListener("keydown",ev=>{
    const g=ev.target.closest&&ev.target.closest(".grip"); if(!g||(ev.key!=="ArrowUp"&&ev.key!=="ArrowDown")) return;
    ev.preventDefault(); const row=g.closest("tr[data-s]"), tb=row.parentElement;
    const sib=ev.key==="ArrowUp"?row.previousElementSibling:row.nextElementSibling;
    if(!sib||(sib.dataset.s==="full"&&!tb.closest("table").classList.contains("summ"))) return;
    if(ev.key==="ArrowUp") tb.insertBefore(row,sib); else tb.insertBefore(sib,row);
    const s=row.dataset.s; commit(tb);
    const again=tb.querySelector(`tr[data-s="${CSS.escape(s)}"] .grip`); if(again) again.focus();
    const pos=[...tb.querySelectorAll("tr[data-s]")].indexOf(tb.querySelector(`tr[data-s="${CSS.escape(s)}"]`))+1;
    live.textContent=`${s} moved to position ${pos}`;
  });
  refresh();
})();
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.join(HERE, "grid.json"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    g, m, cells = load(a.grid, a.manifest)
    present = {r["scheme"] for r in g["records"]}
    for x in sorted(present):
        if x.startswith("router_xtask_") and x not in SCHEME_DESC:
            nm = x[len("router_xtask_"):]
            cal = nm.endswith("_cal")
            src = ("all 17 tasks, 16 rows each (pooled upper bound)" if nm.startswith("17t")
                   else "nq + contradiction + scifact, 16 rows each" if nm.startswith("3t") else "a few tasks")
            SCHEME_DESC[x] = (f"CROSS-TASK learned router (one weight vector, trained end to end at 2k on {src}); "
                              + ("label-free per-task cutoff calibrated on unlabeled val rows to gold_fl20p8_noslot's 2k T2/T" if cal
                                 else "one shared global cutoff"))
            ROUTER_SCHEMES.append(x)
    for x in sorted(present):
        mm = re.match(r"^router_(v\d+)@(exact|val|pair)$", x)
        if mm and x not in SCHEME_DESC:
            SCHEME_DESC[x] = (f"learned linear router, canonical recipe {mm.group(1)} (records/learned-token-router.md §9): end to end, "
                              "no embedding, routed markers, rho = the bar's routed keep, 3 val-selected restarts + hard-loss polish, "
                              "keep-dropout 0.15; " + ("per row, as many kept tokens as gold_fl20p8_noslot keeps on that row (paired budget)" if mm.group(2) == "pair" else "per-row top-k at exactly gold_fl20p8_noslot's T2/T" if mm.group(2) == "exact"
                                                       else "label-free global threshold matched on val rows to gold_fl20p8_noslot's T2/T"))
            ROUTER_SCHEMES.append(x)
    for x in sorted(present):
        # layer-skip on top of the token router (debug/learned_router/layerskip, records §10): FLOPs come from the
        # file's own per-scheme field (per-(token, layer) skips)
        if x == "router_v6a+ls" and x not in SCHEME_DESC:
            SCHEME_DESC[x] = ("v6a token router (paired budget) + learned per-(token, layer) SKIP router (shared, keep 0.75 / 0.5 "
                              "warm-started); layer keep = the most aggressive whose VAL dCE vs token-only is within max(SE, 0.005)")
            ROUTER_SCHEMES.append(x)
        mt = re.match(r"^router_frontier_t([0-9.]+)$", x)
        if mt and x not in SCHEME_DESC:
            SCHEME_DESC[x] = (f"compute frontier, tier dCE <= {mt.group(1)}: the cheapest (token budget f x bar, layer keep) combo "
                              "whose VAL dCE vs full meets the tier (token router v6a + layer-skip router)")
            ROUTER_SCHEMES.append(x)
    rs = [x for x in ROUTER_SCHEMES if x in present]
    if rs:
        i = SCHEMES.index("gold_rand20p8_noslot") + 1
        SCHEMES[i:i] = [x for x in (["gold_only_noslot"] if "gold_only_noslot" in present else []) + rs if x not in SCHEMES]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(page(g, m, cells))
    print(f"[render] {len(cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
