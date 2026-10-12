"""Render the method x task grid artifact from grid.json (built by build_grid.py)."""
import json, html, os
G = json.load(open("research/token_routing/ds64/scheme_grid/grid.json"))
TASKS = ["contradiction", "nq", "oolong", "outlier"]
OUT = os.environ.get("SCHEME_GRID_OUT", "visualizations/token_routing/soft_token_scheme_grid.html")  # relative to the repo root (cwd)

# general schemes: (label, rule description, flags, {task: arm or None}, note)
SCHEMES = [
 ("A", "Gold documents forced real + random whole documents to keep 1/3; plain mean slot; header real where the task renders one",
  "--st-keep-frac 0.3333 --st-keep-mode gold_plus_random [+ header stop id]",
  {"contradiction": "hdr33", "nq": "kv33", "oolong": "ohdr33", "outlier": "kv33"},
  {"oolong": "no gold set exists — gold-blind by construction (same arm as B)"}),
 ("B", "Gold-blind, random whole documents to keep 1/3; plain mean slot; header real where rendered",
  "--st-gold-blind --st-keep-prob 0.3333 [+ header stop id]",
  {"contradiction": None, "nq": "kvgb", "oolong": "ohdr33", "outlier": "kvgb"},
  {"contradiction": "not run at 1/3 (gold-blind 1/6 → 0.053, handoff §1)"}),
 ("C", "Gold-blind, keep 0 documents; header real; cent_cmean slot, no real body tokens",
  "--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id … --st-slot-mode cent_cmean",
  {"contradiction": None, "nq": None, "oolong": "occ00", "outlier": "cc00"}, {}),
 ("D", "Gold-blind, keep 0; header + first 32 body tokens real per document; cent_cmean slot",
  "… --st-header-extra-tokens 32 --st-slot-mode cent_cmean",
  {"contradiction": "chk32", "nq": "nqk32", "oolong": None, "outlier": "ck32"},
  {"oolong": "not run — occ00 (K=0) is this family's oolong point"}),
 ("E", "Gold-blind, keep 0; header + first 64 body tokens real per document; cent_cmean slot",
  "… --st-header-extra-tokens 64 --st-slot-mode cent_cmean",
  {"contradiction": "chk64", "nq": "nqk64", "oolong": None, "outlier": "ck64"},
  {"oolong": "not run"}),
 ("F", "Gold-blind, keep 0; header + first 32 and last 32 body tokens real; cent_cmean slot",
  "… --st-keep-token-rule first_last --st-keep-token-k 64 --st-slot-mode cent_cmean",
  {"contradiction": None, "nq": None, "oolong": None, "outlier": "flk64"},
  {"contradiction": "not run", "nq": "not run", "oolong": "not run"}),
]
BEST = {  # task-specific frontier per budget: arm shown in the summary row
 "16M": {"contradiction": "chk64", "nq": "kv33", "oolong": "ohdr08", "outlier": "flk64"},
 "32M": {"contradiction": "chk32", "nq": "kv33", "oolong": "occ08", "outlier": "flk64"},
 "64M": {"contradiction": "chk16", "nq": "kv33", "oolong": "ohdr17", "outlier": "flk64"},
}
BEST_WHY = {
 "16M": {"contradiction": "beats dense-16M outright, 2.2×", "nq": "−0.006 vs dense at matched PF*; no other nq arm is closer", "oolong": "3.3×; occ00 ties it 7% cheaper", "outlier": "3.9× over dense-16M"},
 "32M": {"contradiction": "= dense-32M at 2.4×", "nq": "2.4× over dense-16M", "oolong": "= dense-32M at 3.3× (occ00 3.55×, −0.001)", "outlier": "4.0× over dense-32M, +0.05"},
 "64M": {"contradiction": "−0.01 vs dense-64M at 3.5×", "nq": "2.5× over dense-32M", "oolong": "−0.001 vs dense-64M at 3.1×", "outlier": "+0.32 at matched PF (headline)"},
}

def cell(task, arm, budget, best=False):
    if arm is None:
        return '<td class="na">—</td>'
    b = G[task].get(arm, {}).get(budget)
    if b is None:
        return f'<td class="na"><span class="arm">{arm}</span><span class="miss">no {budget} run</span></td>'
    if arm == "dense":
        return f'<td class="ref"><span class="f1">{b["f1"]:.3f}</span><span class="pf">@ {b["pf"]:.0f} PF</span><span class="arm">dense, 8 GPU</span></td>'
    d = b["delta"]; cls = "up" if d >= 0.02 else "down" if d <= -0.02 else "flat"
    ex = "*" if b["extrap"] else ""
    dom = f'<span class="dom">{b["dom_ratio"]:.1f}× ≥ dense-{b["dom_budget"]}</span>' if b.get("dom_ratio") and b["dom_ratio"] >= 1.0 else ""
    return (f'<td class="{ "best" if best else ""}"><span class="arm">{arm}</span>'
            f'<span class="f1">{b["f1"]:.3f}</span><span class="pf">@ {b["pf"]:.0f} PF</span>'
            f'<span class="delta {cls}">Δ {d:+.3f}{ex} vs dense</span>{dom}</td>')

def grid(budget):
    rows = []
    rows.append('<tr class="grp"><th colspan="6">Reference</th></tr>')
    rows.append('<tr><th class="m"><span class="lab">Dense</span><span class="desc">Full attention, every token real</span></th><td class="flags">—</td>' +
                "".join(cell(t, "dense", budget) for t in TASKS) + "</tr>")
    rows.append('<tr class="grp"><th colspan="6">Task-specific frontier (best arm per task at this budget)</th></tr>')
    rows.append('<tr class="bestrow"><th class="m"><span class="lab">Per-task best</span><span class="desc">Each column tuned on its own task</span></th><td class="flags">per task</td>' +
                "".join(cell(t, BEST[budget][t], budget, best=True).replace('</td>', f'<span class="why">{BEST_WHY[budget][t]}</span></td>') for t in TASKS) + "</tr>")
    rows.append('<tr class="grp"><th colspan="6">General schemes — one rule applied to every task</th></tr>')
    for key, desc, flags, arms, notes in SCHEMES:
        tds = []
        for t in TASKS:
            a = arms[t]
            c = cell(t, a, budget)
            if t in notes:
                c = c.replace("</td>", f'<span class="note">{html.escape(notes[t])}</span></td>')
            tds.append(c)
        rows.append(f'<tr><th class="m"><span class="lab">Scheme {key}</span><span class="desc">{desc}</span></th><td class="flags"><code>{html.escape(flags)}</code></td>' + "".join(tds) + "</tr>")
    rows.append('<tr><th class="m"><span class="lab">Scheme G <em>proposed</em></span><span class="desc">Gold documents forced real (uniformly, all tasks) + first_last 64 on every other document; cent_cmean slot</span></th><td class="flags"><code>--st-keep-mode gold_plus_random --st-keep-frac 0 --st-keep-token-rule first_last --st-keep-token-k 64 --st-slot-mode cent_cmean</code></td>'
                + "".join('<td class="na"><span class="miss">not run</span></td>' for _ in TASKS) + "</tr>")
    hdr = "".join(f"<th>{t}</th>" for t in TASKS)
    return f'''<div class="tblwrap"><table class="grid"><thead><tr><th class="m">Method</th><th>Flags</th>{hdr}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'''

CE = [  # (task, arm, ckpt, [(rung, size, ce_full, ce_soft, f1_full, f1_soft, compact)])
 ("contradiction", "hdr33", "64M", [("8k", 48, .023, .004, .965, 1.000, .469), ("32k", 48, .082, .025, .826, .938, .457)]),
 ("contradiction", "chk32", "32M", [("8k", 240, .022, .034, .965, .924, .790), ("32k", 120, .077, .117, .853, .736, .785)]),
 ("nq", "kv33", "64M", [("8k", 48, .035, .009, .931, .965, .363), ("32k", 48, .081, .033, .917, .927, .341)]),
 ("oolong", "occ00", "16M", [("8k", 240, .354, .353, .583, .562, .485), ("32k", 120, .590, .502, .375, .375, .466)]),
 ("oolong", "ohdr08", "16M", [("8k", 240, .355, .357, .562, .562, .526), ("32k", 120, .599, .537, .333, .375, .513)]),
 ("outlier", "ck64", "32M", [("8k", 240, .046, .073, .743, .688, .493), ("32k", 120, .190, .232, .161, .042, .488)]),
]
def ce_table():
    by_arm = {}
    for task, arm, ck, rows in CE:
        by_arm.setdefault(arm, {})[task] = (ck, rows)
    out = []
    for arm, tasks in by_arm.items():
        tds = []
        for t in TASKS:
            if t not in tasks:
                tds.append('<td class="na">—</td>'); continue
            ck, rows = tasks[t]
            parts = []
            for rung, n, cf, cs, ff, fs, comp in rows:
                d = cs - cf; cls = "up" if d <= -0.01 else "down" if d >= 0.01 else "flat"
                parts.append(f'<div class="cerow"><span class="rung">{rung}</span><span class="ce">{cf:.3f} → <b>{cs:.3f}</b></span><span class="delta {cls}">Δ {d:+.3f}</span><span class="pf">f1 {ff:.2f}→{fs:.2f} · eval_size {n}{" ⚠" if n < 500 else ""}</span></div>')
            tds.append(f'<td><span class="arm">{arm} · {ck} ckpt</span>{"".join(parts)}</td>')
        out.append(f'<tr><th class="m"><span class="lab">{arm}</span></th>' + "".join(tds) + "</tr>")
    hdr = "".join(f"<th>{t}</th>" for t in TASKS)
    return f'<div class="tblwrap"><table class="grid ce"><thead><tr><th class="m">Arm</th>{hdr}</tr></thead><tbody>{"".join(out)}</tbody></table></div>'

FROZEN = [
 ("cc00 — keep nothing (floor)", "+0.454", "+0.688", "+0.778", "0.103 / 0.058 / 0.043", ""),
 ("rand8 — 8 random body tokens", "+0.362", "+0.651", "+0.717", "0.153 / 0.109 / 0.087", ""),
 ("first8", "+0.268", "+0.546", "+0.591", "0.153 / 0.109 / 0.087", ""),
 ("first16", "+0.152", "+0.366", "+0.442", "0.204 / 0.160 / 0.132", "the rule the trained ck* arms use"),
 ("idf16 — top-16 by corpus IDF", "+0.113", "+0.198", "+0.269", "0.204 / 0.160 / 0.132", "no forward pass, no fitting"),
 ("rule16 — 6-feature ridge fit to gradient saliency", "+0.082", "+0.201", "+0.288", "0.204 / 0.160 / 0.132", "deployable"),
 ("attn16 — top-16 by answer→token attention (oracle)", "+0.117", "+0.198", "+0.249", "0.204 / 0.160 / 0.132", "needs the full forward"),
 ("grad16 — top-16 by gradient saliency (oracle)", "+0.066", "+0.162", "+0.203", "0.204 / 0.160 / 0.132", "ceiling for uniform per-doc"),
 ("gradrow16 — same tokens, budget allocated row-wide (oracle)", "+0.179", "+0.146", "+0.131", "0.204 / 0.160 / 0.132", "best ΔCE anywhere at 32k"),
 ("attnrow8 — layer-3 attention sets a per-doc budget, first-k inside", "+0.281", "+0.274", "+0.169", "0.153 / 0.109 / 0.087", "cheapest good cell; needs 4 dense layers"),
 ("prev4_k8 — 4-layer preview, layer-3 attention picks tokens", "+0.299", "+0.589", "+0.649", "0.259 / 0.220 / 0.201", "worse than first8 at same cost"),
 ("prev4_first8 — 4-layer preview + first 8", "+0.250", "+0.530", "+0.575", "0.259 / 0.220 / 0.201", "preview buys nothing over first8"),
 ("grad16_swap — control: gold's kept tokens swapped with a random doc's", "+0.910", "+1.272", "+1.212", "0.204 / 0.160 / 0.132", "the model is reading the kept tokens"),
]
def frozen_table():
    rows = "".join(f'<tr><th class="m"><span class="lab">{n}</span></th><td class="num">{a}</td><td class="num">{b}</td><td class="num">{c}</td><td class="num">{f}</td><td class="note">{note}</td></tr>' for n, a, b, c, f, note in FROZEN)
    return f'<div class="tblwrap"><table class="grid fz"><thead><tr><th class="m">Selector (outlier, frozen dense-64M)</th><th>ΔCE 2k</th><th>ΔCE 8k</th><th>ΔCE 32k</th><th>FLOP fraction 2k / 8k / 32k</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>'

page = f'''<title>Soft-Token Scheme Grid</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F6F7F5;--ink:#1A1E1C;--mute:#66706B;--rule:#D6DBD7;--soft:#ECEFEC;--acc:#1F6F5B;--accbg:#E3F0EA;--warn:#9A5F0B;--bad:#A63D3D;--badbg:#F5E6E6;--ref:#EEF0F2;--code:#F0F2EF}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15181A;--ink:#E4E7E4;--mute:#98A19C;--rule:#2E3437;--soft:#1D2224;--acc:#6CC7A6;--accbg:#173129;--warn:#D9A23E;--bad:#E08585;--badbg:#3A2323;--ref:#1C2124;--code:#1D2224}}}}
:root[data-theme="dark"]{{--bg:#15181A;--ink:#E4E7E4;--mute:#98A19C;--rule:#2E3437;--soft:#1D2224;--acc:#6CC7A6;--accbg:#173129;--warn:#D9A23E;--bad:#E08585;--badbg:#3A2323;--ref:#1C2124;--code:#1D2224}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:15px;line-height:1.5;margin:0}}
.wrap{{max-width:1180px;margin:0 auto;padding:40px 28px 80px}}
h1{{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:34px;line-height:1.1;margin:0 0 6px;text-wrap:balance}}
h2{{font-family:Fraunces,Georgia,serif;font-weight:500;font-size:22px;margin:0 0 4px;text-wrap:balance}}
.sub{{color:var(--mute);max-width:70ch;margin:0}}
.meta{{display:flex;flex-wrap:wrap;gap:8px 22px;color:var(--mute);font-size:13px;margin:16px 0 0;padding:12px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.meta b{{color:var(--ink);font-weight:600}}
section{{margin-top:44px}}
.lede{{max-width:72ch;margin:6px 0 16px}}
.tblwrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px}}
table.grid{{border-collapse:collapse;width:100%;min-width:980px;font-variant-numeric:tabular-nums}}
table.grid th,table.grid td{{padding:10px 12px;vertical-align:top;border-bottom:1px solid var(--rule);text-align:left}}
table.grid thead th{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);background:var(--soft);font-weight:600}}
table.grid th.m{{width:250px;font-weight:400}}
table.grid td.flags{{width:190px;font-size:11.5px;color:var(--mute)}}
table.grid td.flags code{{font-family:"JetBrains Mono",monospace;font-size:11px;white-space:normal;word-break:break-word;background:var(--code);padding:2px 4px;border-radius:3px;display:inline-block}}
tr.grp th{{background:var(--soft);font-family:Fraunces,Georgia,serif;font-size:15px;font-weight:500;padding:8px 12px;color:var(--ink)}}
.lab{{display:block;font-weight:600}}
.lab em{{font-style:normal;font-weight:400;color:var(--warn);font-size:12px;margin-left:6px}}
.desc{{display:block;color:var(--mute);font-size:12.5px;line-height:1.35;margin-top:2px}}
td .arm{{display:block;font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--mute)}}
td .f1{{display:block;font-family:"JetBrains Mono",monospace;font-size:19px;font-weight:500;line-height:1.15;margin-top:2px}}
td .pf{{display:block;font-size:12px;color:var(--mute)}}
td .delta{{display:block;font-family:"JetBrains Mono",monospace;font-size:12px;margin-top:3px}}
.up{{color:var(--acc)}} .down{{color:var(--bad)}} .flat{{color:var(--mute)}}
td .dom{{display:inline-block;margin-top:5px;font-size:11.5px;font-weight:600;color:var(--acc);background:var(--accbg);padding:1px 7px;border-radius:10px}}
td .why,td .note,td .miss{{display:block;font-size:11.5px;color:var(--mute);margin-top:5px;font-style:italic;max-width:22ch}}
td.na{{color:var(--mute)}}
td.ref{{background:var(--ref)}}
td.best{{box-shadow:inset 3px 0 0 var(--acc)}}
tr.bestrow td .arm{{color:var(--acc)}}
table.ce td{{min-width:200px}}
.cerow{{display:grid;grid-template-columns:28px 1fr;column-gap:8px;margin-top:6px;font-size:12.5px}}
.cerow .rung{{font-family:"JetBrains Mono",monospace;color:var(--mute)}}
.cerow .ce{{font-family:"JetBrains Mono",monospace}}
.cerow .delta,.cerow .pf{{grid-column:2;margin:0}}
table.fz td.num{{font-family:"JetBrains Mono",monospace;white-space:nowrap}}
table.fz td.note{{font-size:12.5px;color:var(--mute);font-style:italic}}
.budgets{{display:flex;gap:6px;margin:14px 0 10px}}
.budgets button{{font:inherit;font-size:13px;font-weight:600;padding:5px 14px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);border-radius:3px;cursor:pointer}}
.budgets button[aria-pressed="true"]{{background:var(--ink);color:var(--bg);border-color:var(--ink)}}
.budgets button:focus-visible{{outline:2px solid var(--acc);outline-offset:2px}}
.foot{{font-size:12.5px;color:var(--mute);max-width:80ch;margin-top:10px}}
.reading{{max-width:76ch}}
.reading li{{margin-bottom:10px}}
.reading b{{font-weight:600}}
.caution{{border-left:3px solid var(--warn);padding:8px 14px;background:var(--soft);max-width:76ch;margin:14px 0}}
</style>
<div class="wrap">
<h1>Soft-Token Scheme Grid</h1>
<p class="sub">Rows are token-selection methods, columns are the four ds64 tasks. Task-specific frontiers first, then the same rule applied uniformly to every task.</p>
<div class="meta"><span><b>Model</b> Qwen3.5-4B, short-heavy 2k–64k mix</span><span><b>Metric</b> ladder mean f1, eval_size 500 per rung, 5 rungs (outlier: 4 — its 64k file is broken)</span><span><b>Cost</b> <code>flop_meter/actual_pflops</code> for the training run</span><span><b>Source</b> results/ds64/results.csv, 2026-09-16; arms of the <code>-b128f3-</code> generation</span></div>

<div class="caution"><b>There is no dev-loss grid.</b> The ladder is generation-graded and training logs carry only <code>train/CE</code>; no in-loop dev loss was run. Eval-time CE on trained checkpoints exists for six arms only (<a href="#ce">below</a>). Getting a full CE row set means one <code>trained_parity_check.py</code> pass per checkpoint — the checkpoints are on weka.</div>

<section id="grid">
<h2>Method × task at matched training tokens</h2>
<p class="lede">Each cell: arm, mean f1, training PF, and Δ against the dense curve interpolated (log-PF) to the same cost. A chip marks the most expensive dense point the arm matches or beats and the FLOP ratio to it. <code>*</code> = the arm sits below dense's cheapest measured point, so Δ is an extrapolation (outlier has 4M/8M anchors, the others start at 16M).</p>
<div class="budgets" role="tablist"><button aria-pressed="false" data-b="16M">16M tokens</button><button aria-pressed="true" data-b="32M">32M tokens</button><button aria-pressed="false" data-b="64M">64M tokens</button></div>
<div id="g16M" hidden>{grid("16M")}</div>
<div id="g32M">{grid("32M")}</div>
<div id="g64M" hidden>{grid("64M")}</div>
<p class="foot">Header stop ids are a rendering detail, not a task-specific rule: 25 (<code>:</code>) ×1 on contradiction's <code>Claim N:</code>, 25 ×3 on oolong's <code>Date/User/Instance</code>, 5491 (<code>]:</code>) ×1 on outlier's <code>Document [N]:</code>, none on nq (K counts from the document's first token). Every soft arm trained at 4 GPU / micro 2 except <code>hdr33</code>, <code>kv33</code>, <code>kv17</code>, <code>kvgb</code>-16M, <code>ohdr33/17</code> at 8 GPU / micro 4 — compaction arms' padding compute differs by up to 1.44× between the two, so treat cross-layout PF as ±30%.</p>
</section>

<section id="ce">
<h2>Dev loss where it was measured</h2>
<p class="lede">Eval-time answer CE on the trained checkpoint, scored on full real text (what the ladder feeds) → on the arm's own soft construction. Negative Δ means the compressed input is <i>easier</i> for the model it trained. From <code>records/trained-parity-winning-arms.md</code>; eval sizes are under the 500 bar (⚠) and are confirmatory only.</p>
{ce_table()}
<p class="foot">Pattern: the gold-forced whole-document arms (hdr33, kv33) and oolong's keep-0 arms are at or below full-text CE on their own construction; the first-k arms (chk32, ck64) are slightly above it and the gap widens with length — their Pareto wins are full-attention-eval results and the pooled remainder is used imperfectly at 32k.</p>
</section>

<section id="frozen">
<h2>General selectors, frozen-model probe (outlier only)</h2>
<p class="lede">The one place several data-agnostic selectors were compared head-to-head at equal token budget: a frozen dense-64M outlier checkpoint, ΔCE vs full attention (paired, eval_size 240/240/120 ⚠), from <code>records/outlier-saliency-preview-probe.md</code>. Lower is better; nothing reaches parity (full CE ≈ 0.06). The frozen probe under-predicts trained first-k by ~2.5×, so read this as a ranking.</p>
{frozen_table()}
<p class="foot">Contradiction and oolong were probed with different constructions (header-real, layer-L dense prefix) and cannot be put in the same columns; on contradiction the 4-layer dense prefix reached CE parity, on outlier it bought nothing — the one preview result that is itself task-dependent.</p>
</section>

<section id="reading" class="reading">
<h2>Reading</h2>
<ul>
<li><b>Scheme E (first-64, gold-blind, keep 0) is the closest thing to a uniform rule with results on three tasks.</b> It is the task-specific best on outlier and contradiction, and on nq it trails the tuned kv33 by −0.05 at 32M (0.802 @ 296 vs 0.849 @ 286) and −0.06 at 64M — the price of one config. Oolong has no first-k arm; its keep-0 point (occ00) already ties dense at 3.55×, so a K=64 run there is a check, not a rescue.</li>
<li><b>nq is the single holdout, and gold is why.</b> Every gold-blind nq arm sits below dense at matched FLOPs; the gold-forced kv33 is above it. nq's answer is the text of one document, so pooling that document 2/3 of the time removes the supervision signal. Gold-forcing is also harmless on contradiction (hdr33 above dense) and moot on oolong.</li>
<li><b>Scheme G is the missing arm.</b> Gold forced real uniformly + first_last 64 on everything else. It is the only uniform rule that could plausibly match every column, and the one open question is outlier: whole-document gold-forcing collapsed there (kv33 → 0.005) because the gold document's identity was the only complete document in the row; with every document carrying a real 64-token prefix that completeness cue no longer exists, but this is untested. Four arms at 32M decide it.</li>
<li><b>first_last ≥ first at identical cost</b> on every outlier budget (flk64 vs ck64: +0.008 / +0.031 / +0.021), so G should use it; it has not been run on any other task.</li>
<li><b>Layer-based selection helps as a per-document budget, not as a token picker.</b> In the frozen probe layer-3 attention's top tokens inside a document lose to first-k (prev4_k8 +0.589 vs prev4_first8 +0.530 at 8k), while the same attention used to size each document's budget is the cheapest good cell at 32k (attnrow8 +0.169 at 0.087). On this hybrid the first attention layer is layer 3, so a preview costs 4 dense layers.</li>
</ul>
</section>
</div>
<script>
document.querySelectorAll('.budgets button').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('.budgets button').forEach(x=>x.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true');
  ['16M','32M','64M'].forEach(k=>{{document.getElementById('g'+k).hidden=(k!==b.dataset.b);}});
}}));
</script>
'''
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
open(OUT, "w").write(page)
print(OUT, len(page))
