"""Render the CTC-suite artifact page from `suite_table.json`.

The page has one job: show, per task, what the 4B dense-vs-chunked numbers currently are and
exactly which cells are still empty. Numbers are never typed by hand here -- they come from the
harvested grade JSONs, so a re-harvest + re-render is the whole update path.
"""
import json, datetime, html

RUNGS = [2048, 4096, 8192, 16384, 32768]
RUNG_LABEL = {2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k"}

TABLE = json.load(open("debug/ctc_final_suite/suite_table.json"))

LABEL = {
    "fiqa": "fiqa", "nq": "nq", "hpqa": "hpqa",
    "qdmatch_fiqa": "qdmatch fiqa", "qdmatch_nq": "qdmatch nq", "qdmatch_hpqa": "qdmatch hpqa",
    "outlier_amazon": "outlier amazon", "outlier_scalek": "outlier scale-k",
    "outlier_fixk": "outlier fix-k", "oolong": "OOLONG", "grouping": "grouping",
    "absence": "absence", "xabsence": "xabsence", "rerank": "rerank", "msmarco": "msmarco",
    "reorder": "re-order", "obliq": "obliq", "niah_contra": "niah contra",
    "contra_real": "contradiction (realistic)",
    "strmatch": "strmatch", "textgroups": "textgroups", "scifact": "scifact",
}

# status: ok | evals-owed | no-ladder | partial ; note is shown under the row label
STATUS = {
    "fiqa":           ("ok",         "Ladder built and both arms evaluated 2026-08-13."),
    "nq":             ("ok",         None),
    "hpqa":           ("ok",         "Dense backfilled 2026-08-12 after the corrupt S3 distcp."),
    "qdmatch_fiqa":   ("ok",         "Trained on Beaker and evaluated 2026-08-13, closing the last "
                                     "blank row. Scored on the q61 rung family, matching the trained "
                                     "shard (the generator emitted two query-count ladders)."),
    "qdmatch_nq":     ("ok",         "Backfilled from the model-scale 4B runs 2026-08-13."),
    "qdmatch_hpqa":   ("ok",         "Rung labels run 0.74-0.82x of the stated tokens."),
    "outlier_amazon": ("ok",         "Rung labels run 1.12-1.24x."),
    "outlier_scalek": ("ok",         None),
    "outlier_fixk":   ("partial",    "32k (n=220) deliberately not shown: the row is a smooth "
                                     "decline in n, not a cliff at the last rung. See the n-curve "
                                     "below."),
    "oolong":         ("ok",         "Retrain landed. Both arms are ctcms-oolong-*-4b-vsl on the "
                                     "rebuilt, decontaminated, query-after shard, graded query-after."),
    "grouping":       ("ok",         "Regraded after the parser fix; the old 0.439/0.358 row is dead."),
    "absence":        ("partial",    "32k rung never built; dense 16k ungraded. Labels run 3.0-3.6x."),
    "xabsence":       ("retraining", "SUPERSEDED. Training CE sat on the 0.78 flatline, so these "
                                     "numbers are not a trained model. Four learnability probes now "
                                     "reach CE 0.23-0.33; their evals decide the row."),
    "rerank":         ("ok",         "Ladder rebuilt token-accurate and fully regraded 2026-08-12."),
    "msmarco":        ("ok",         "Ladder rebuilt token-accurate and fully regraded 2026-08-12."),
    "reorder":        ("ok",         "32k capped by rung policy -- deliberate, not a gap."),
    "obliq":          ("partial",    "Twitter rebuild. Table-only numbers, eval_size=126."),
    "niah_contra":    ("partial",    "Dense 2k conflicts: 0.164 on disk vs 0.988 in the July table."),
    "contra_real":    ("ok",         "PubMed multi-claim, scored on the IID realistic-mode ladder that "
                                     "matches the training generator. 2k column is the n=56 / rung_2560 "
                                     "file; the other four rungs are n-identical to contradiction_clean."),
    "strmatch":       ("partial",    "Dense 32k ungraded. Labels run 0.60x."),
    "textgroups":     ("ok",         "Labels drift 0.89x to 1.55x across the ladder."),
    "scifact":        ("ok",         "eval_size=300. The old dense-collapse verdict is stale."),
}

CLASS_LABEL = {"N": "O(N)", "NM": "O(NM)", "N2": "O(N²)"}


def cell_html(c):
    if c is None:
        return '<td class="cell empty"><span class="sr">no number</span></td>'
    v, se, n = c["value"], c["se"], c["eval_size"]
    void = c["source"] == "superseded"
    carried = c["source"] == "table-2026-07-27"
    small = n is not None and n < 500
    cls = "cell" + (" void" if void else " carried" if carried else "") + ("" if void or not small else " small")
    bar = "" if void else f'<span class="bar" style="--w:{v * 100:.1f}%"></span>'
    tip = f"eval_size={n}"
    if carried:
        tip += " · carried from the 2026-07-27 table"
    if void:
        tip += " · superseded by the in-flight rebuild -- do not quote"
    return (f'<td class="{cls}" title="{html.escape(tip)}">{bar}'
            f'<span class="v">{v:.3f}</span>'
            f'<span class="se">±{se:.3f}</span></td>')


rows_html = []
for e in TABLE:
    status, note = STATUS[e["row"]]
    tds = []
    for arm in ("dense", "chunked"):
        for i, r in enumerate(RUNGS):
            c = e["cells"][arm].get(str(r))
            td = cell_html(c)
            if arm == "chunked" and i == 0:
                td = td.replace('class="cell', 'class="armstart cell', 1)
            tds.append(td)
    # gap at the deepest rung where both arms have a number
    gap_txt, gap_cls = "—", "gap none"
    for r in [] if status == "retraining" else reversed(RUNGS):
        d = e["cells"]["dense"].get(str(r))
        k = e["cells"]["chunked"].get(str(r))
        if d and k:
            g = d["value"] - k["value"]
            gap_txt = f"{g:+.3f}"
            gap_cls = "gap " + ("wide" if g >= 0.15 else "flat" if abs(g) < 0.05 else "mid")
            gap_txt += f'<span class="at">{RUNG_LABEL[r]}</span>'
            break
    note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
    rows_html.append(
        f'<tr class="r-{status}">'
        f'<th scope="row"><span class="tname">{html.escape(LABEL[e["row"]])}</span>'
        f'<span class="tmeta">{CLASS_LABEL[e["class"]]} · {html.escape(e["metric"])}</span>'
        f'{note_html}</th>'
        f'<td class="st"><span class="dot {status}"></span></td>'
        + "".join(tds) +
        f'<td class="{gap_cls}">{gap_txt}</td></tr>'
    )

n_dense = sum(e["n_dense"] for e in TABLE)
n_chunk = sum(e["n_chunked"] for e in TABLE)
n_owed = sum(1 for e in TABLE if STATUS[e["row"]][0] == "evals-owed") * 10
n_void = sum(e["n_void"] for e in TABLE)
built = sum(1 for e in TABLE if STATUS[e["row"]][0] != "no-ladder")
n_tasks = len(TABLE)
per_arm = n_tasks * len(RUNGS)
stamp = datetime.datetime.now().strftime("%Y-%m-%d")

HEAD_RUNGS = "".join(
    f'<th class="rung{" armstart" if (arm == "chunked" and i == 0) else ""}">{RUNG_LABEL[r]}</th>'
    for arm in ("dense", "chunked") for i, r in enumerate(RUNGS))

page = f"""<title>CTC Suite Grid</title>
<style>
:root {{
  --paper:#F6F7F7; --panel:#FFFFFF; --ink:#12191C; --ink-2:#4A5860; --ink-3:#7A8790;
  --rule:#DCE2E3; --rule-2:#EDF1F1;
  --dense:#0E6E70; --dense-soft:#0E6E7022;
  --chunk:#6A4A86; --chunk-soft:#6A4A8622;
  --warn:#9C6B12; --bad:#A93E37; --ok:#3F7A4E;
  --hatch:#C9D2D3;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0E1416; --panel:#151D20; --ink:#E6ECEC; --ink-2:#9FB0B3; --ink-3:#71858A;
    --rule:#263134; --rule-2:#1C2528;
    --dense:#4FC3BC; --dense-soft:#4FC3BC26;
    --chunk:#B69BD8; --chunk-soft:#B69BD826;
    --warn:#D9A340; --bad:#E08078; --ok:#77C48D;
    --hatch:#2C393C;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0E1416; --panel:#151D20; --ink:#E6ECEC; --ink-2:#9FB0B3; --ink-3:#71858A;
  --rule:#263134; --rule-2:#1C2528;
  --dense:#4FC3BC; --dense-soft:#4FC3BC26;
  --chunk:#B69BD8; --chunk-soft:#B69BD826;
  --warn:#D9A340; --bad:#E08078; --ok:#77C48D;
  --hatch:#2C393C;
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:48px 24px 96px; display:flex; flex-direction:column; gap:40px; }}

header {{ display:flex; flex-direction:column; gap:14px; }}
.eyebrow {{
  font-family:var(--mono); font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--ink-3);
}}
h1 {{
  font-family:var(--mono); font-weight:600; font-size:clamp(26px,4vw,38px);
  letter-spacing:-.02em; margin:0; text-wrap:balance;
}}
.lede {{ margin:0; max-width:66ch; color:var(--ink-2); }}
.lede strong {{ color:var(--ink); font-weight:600; }}

.stats {{ display:flex; flex-wrap:wrap; gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:3px; overflow:hidden; }}
.stat {{ flex:1 1 150px; background:var(--panel); padding:14px 16px; display:flex; flex-direction:column; gap:2px; }}
.stat .k {{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-3); }}
.stat .n {{ font-family:var(--mono); font-size:24px; font-weight:600; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.stat .n small {{ font-size:14px; font-weight:400; color:var(--ink-3); }}
.stat.d .n {{ color:var(--dense); }}
.stat.c .n {{ color:var(--chunk); }}
.stat.w .n {{ color:var(--warn); }}

section {{ display:flex; flex-direction:column; gap:14px; }}
h2 {{
  font-family:var(--mono); font-size:12px; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-2); margin:0;
  padding-bottom:8px; border-bottom:1px solid var(--rule);
}}

.scroller {{ overflow-x:auto; border:1px solid var(--rule); border-radius:3px; background:var(--panel); }}
table {{ border-collapse:separate; border-spacing:0; width:100%; min-width:940px; }}

thead th {{
  position:sticky; top:0; z-index:2; background:var(--panel);
  font-family:var(--mono); font-size:10px; font-weight:600; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-3); text-align:center;
  padding:10px 0 8px; border-bottom:1px solid var(--rule); white-space:nowrap;
}}
thead tr.arms th {{ padding-top:12px; padding-bottom:4px; border-bottom:none; }}
thead tr.arms th.a-dense {{ color:var(--dense); }}
thead tr.arms th.a-chunk {{ color:var(--chunk); }}
th.rowhead {{ text-align:left; padding-left:16px; }}

th.armstart, td.armstart {{ border-left:1px solid var(--rule); }}

tbody th {{
  text-align:left; font-weight:400; padding:9px 14px 9px 16px; vertical-align:top;
  border-bottom:1px solid var(--rule-2); background:var(--panel);
  position:sticky; left:0; z-index:1; min-width:212px;
}}
.tname {{ display:block; font-family:var(--mono); font-size:13px; font-weight:600; letter-spacing:-.01em; }}
.tmeta {{ display:block; font-family:var(--mono); font-size:10px; color:var(--ink-3); letter-spacing:.04em; }}
.note {{ display:block; font-size:11.5px; line-height:1.4; color:var(--ink-2); margin-top:3px; max-width:30ch; }}

tbody td {{ border-bottom:1px solid var(--rule-2); }}
td.st {{ width:22px; text-align:center; }}
.dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; }}
.dot.ok {{ background:var(--ok); }}
.dot.partial {{ background:var(--warn); }}
.dot.evals-owed {{ background:var(--chunk); }}
.dot.no-ladder {{ background:var(--bad); }}
.dot.retraining {{ background:var(--bad); box-shadow:0 0 0 2px color-mix(in srgb,var(--bad) 30%,transparent); }}

td.cell {{
  position:relative; text-align:right; padding:9px 12px 9px 8px; width:74px;
  font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap;
}}
td.cell .bar {{
  position:absolute; left:0; bottom:0; height:2px; width:var(--w);
  background:var(--dense); opacity:.55;
}}
td.armstart ~ td.cell .bar, td.cell.armstart .bar {{ background:var(--chunk); }}
td.cell .v {{ display:block; font-size:13px; font-weight:500; letter-spacing:-.01em; }}
td.cell .se {{ display:block; font-size:9.5px; color:var(--ink-3); letter-spacing:-.02em; }}
td.cell.carried .v {{ color:var(--ink-2); font-style:italic; }}
td.cell.void {{ background:repeating-linear-gradient(135deg, transparent 0 5px, var(--hatch) 5px 6px); }}
td.cell.void .v {{ color:var(--ink-3); text-decoration:line-through; font-weight:400; }}
td.cell.void .se {{ visibility:hidden; }}
td.cell.small .se {{ color:var(--warn); }}
td.empty {{
  background:repeating-linear-gradient(135deg, transparent 0 5px, var(--hatch) 5px 6px);
}}
.sr {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }}

td.gap {{
  text-align:right; padding:9px 16px 9px 10px; font-family:var(--mono);
  font-variant-numeric:tabular-nums; font-size:12.5px; white-space:nowrap;
  border-left:1px solid var(--rule);
}}
td.gap .at {{ display:block; font-size:9.5px; color:var(--ink-3); letter-spacing:.06em; }}
td.gap.wide {{ color:var(--bad); font-weight:600; }}
td.gap.mid {{ color:var(--warn); }}
td.gap.flat {{ color:var(--ink-3); }}
td.gap.none {{ color:var(--ink-3); }}

.legend {{ display:flex; flex-wrap:wrap; gap:8px 22px; font-size:12px; color:var(--ink-2); }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.swatch {{ width:16px; height:10px; border:1px solid var(--rule); display:inline-block; }}
.swatch.hatch {{ background:repeating-linear-gradient(135deg, transparent 0 5px, var(--hatch) 5px 6px); }}

ol.todo {{ margin:0; padding-left:0; list-style:none; counter-reset:t; display:flex; flex-direction:column; gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:3px; overflow:hidden; }}
ol.todo li {{ counter-increment:t; background:var(--panel); padding:12px 16px 12px 46px; position:relative; }}
ol.todo li::before {{
  content:counter(t,decimal-leading-zero); position:absolute; left:16px; top:12px;
  font-family:var(--mono); font-size:11px; color:var(--ink-3); letter-spacing:.06em;
}}
ol.todo b {{ font-weight:600; }}
ol.todo .cost {{ font-family:var(--mono); font-size:11px; color:var(--ink-3); margin-left:6px; }}

.method {{ display:flex; flex-direction:column; gap:10px; font-size:13px; color:var(--ink-2); max-width:74ch; }}
.method p {{ margin:0; }}
.method h3 {{ margin:14px 0 0; font-size:14px; letter-spacing:-0.01em; color:var(--ink); }}
/* The aux tables carry a handful of columns, not the 11-column grid, so they must not inherit the
   940px min-width that forces the main table to scroll on every screen. */
table.aux {{ min-width:0; width:auto; font-size:13px; }}
table.aux th, table.aux td {{
  padding:5px 12px; border-bottom:1px solid var(--rule-2); text-align:right; white-space:nowrap;
  font-family:var(--mono); color:var(--ink);
}}
table.aux th {{ font-family:inherit; color:var(--ink-3); font-weight:600; }}
table.aux td:first-child, table.aux th:first-child {{ text-align:left; font-family:inherit; }}
.method code {{ font-family:var(--mono); font-size:12px; background:var(--rule-2); padding:1px 5px; border-radius:2px; color:var(--ink); }}
.flag {{ border-left:2px solid var(--warn); padding-left:14px; }}
.flag b {{ color:var(--ink); }}

footer {{ font-family:var(--mono); font-size:11px; color:var(--ink-3); letter-spacing:.04em; border-top:1px solid var(--rule); padding-top:16px; }}
@media (max-width:640px) {{ .wrap {{ padding:32px 14px 64px; }} .note {{ display:none; }} }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Qwen3.5-4B · full attention vs document-chunked · frozen {stamp}</div>
  <h1>The final {n_tasks}-task CTC suite</h1>
  <p class="lede">Every task in the figure, on the 2k–32k token ladder, with both arms and a
  binomial error bar on each number. <strong>{n_dense + n_chunk} of {per_arm * 2} cells hold a number that is
  current</strong>; {n_void} more are struck through, superseded by a rebuild still in flight. The
  hatched cells are the whole to-do list.</p>
</header>

<div class="stats">
  <div class="stat"><span class="k">Tasks</span><span class="n">{n_tasks}</span></div>
  <div class="stat"><span class="k">Ladders built</span><span class="n">{built}<small>/{n_tasks}</small></span></div>
  <div class="stat d"><span class="k">Dense cells</span><span class="n">{n_dense}<small>/{per_arm}</small></span></div>
  <div class="stat c"><span class="k">Chunked cells</span><span class="n">{n_chunk}<small>/{per_arm}</small></span></div>
  <div class="stat w"><span class="k">One eval away</span><span class="n">{n_owed}</span></div>
  <div class="stat"><span class="k">Superseded</span><span class="n">{n_void}</span></div>
</div>

<section>
  <h2>The grid</h2>
  <div class="scroller">
    <table>
      <thead>
        <tr class="arms">
          <th class="rowhead" rowspan="2">Task</th><th rowspan="2"></th>
          <th class="a-dense" colspan="5">Dense</th>
          <th class="a-chunk armstart" colspan="5">Chunked</th>
          <th rowspan="2">Gap</th>
        </tr>
        <tr>{HEAD_RUNGS}</tr>
      </thead>
      <tbody>
        {chr(10) + (chr(10) + "        ").join(rows_html)}
      </tbody>
    </table>
  </div>
  <div class="legend">
    <span><span class="dot ok"></span> ladder + both arms complete</span>
    <span><span class="dot partial"></span> partial or flagged</span>
    <span><span class="dot evals-owed"></span> trained, evals owed</span>
    <span><span class="dot no-ladder"></span> no eval ladder</span>
    <span><span class="dot retraining"></span> retraining — numbers void</span>
    <span><span class="swatch hatch"></span> no number</span>
    <span><em>italic</em> = carried from the 2026-07-27 table, no grade JSON on disk</span>
  </div>
</section>

<section>
  <h2>Beyond the grid</h2>
  <div class="method">
    <p>Four measurements from 2026-08-13 that the 2k–32k grid cannot express — a length axis, a
      model-scale axis, a document-count axis, and a harness audit — but that change how some of its
      columns should be read.</p>

    <h3>Length generalization past the 32k training ceiling</h3>
    <p>4B dense, evaluated on natively-generated 64k and 128k rungs. eval_size 500 everywhere.</p>
    <div class="scroller"><table class="aux">
      <thead><tr><th>Task</th><th>32k (in-ladder)</th><th>~64k</th><th>~128k</th></tr></thead>
      <tbody>
        <tr><td>oolong</td><td>0.573</td><td>0.573</td><td>—</td></tr>
        <tr><td>oolong <em>(chunked)</em></td><td>0.554</td><td>0.554</td><td>—</td></tr>
        <tr><td>rerank</td><td>0.986</td><td>0.963</td><td>0.768</td></tr>
        <tr><td>msmarco</td><td>0.954</td><td>0.953</td><td>0.647</td></tr>
        <tr><td>hpqa</td><td>0.943</td><td>0.718</td><td>0.408</td></tr>
        <tr><td>nq</td><td>0.860</td><td>0.490</td><td>0.304</td></tr>
        <tr><td>contra (realistic)</td><td>0.946</td><td>0.278</td><td>0.061</td></tr>
        <tr><td>qdmatch nq</td><td>0.686</td><td>0.223</td><td>0.017</td></tr>
        <tr><td>qdmatch nq <em>(chunked)</em></td><td>0.104</td><td>0.030</td><td>0.000</td></tr>
      </tbody>
    </table></div>
    <p><b>oolong is flat on BOTH arms</b> — dense 0.5727 → 0.5733 and chunked 0.5539 → 0.5544 from
      32k to 64k, one full octave beyond anything either was trained on, with the dense-chunked gap
      holding constant at ~0.019. So the chunked mask is not what limits length generalization here;
      whatever caps oolong caps both arms equally. rerank and msmarco lose little at 64k and then
      fall at 128k. The O(N²) rows fall first and hardest: contradiction drops 0.946 → 0.278 in a
      single octave.</p>
    <div class="flag"><p><b>Chunked eval above 32k is memory-bound, and the knobs do not transfer.</b>
      The chunked 64k run first died allocating a <b>74 GiB</b> boolean attention mask. The fallback
      mask costs O(num_actual_tokens x total_cache_tokens) with
      <code>total_cache_tokens = max_model_len x CHUNK_SEQ_HEADROOM</code>, so the fast-eval knobs
      (seqs 16 / headroom 18, tuned and validated at rung ≤ 8192) are ~4x too expensive at 64k on top
      of the mask's own quadratic growth in rung length. <code>gpu_eval_task_chunked.sh</code> gates
      those knobs to rung ≤ 8192 for exactly this reason; the oolong launcher sets
      <code>CHUNK_FAST_MAX_RUNG=99999999</code> to defeat the gate because they are part of the
      varlen recipe — but varlen did not engage on this model, so the run got the varlen config's
      memory cost with the fallback path's memory profile. Re-run at the default 8/10 it completes in
      45 min. More GPUs would not have helped: the mask is a [query x key] tensor shared across
      heads, and tensor parallelism shards heads, not positions.</p></div>
    <div class="flag"><p><b>Only injection-safe tasks appear above.</b> The long rungs are built by
      injecting filler documents, which is sound when gold is defined by matching a query or a
      partner (<code>query_match</code>, <code>pairwise</code>) and unsound when gold is defined by
      what is <em>absent</em> or by global structure (<code>absence</code>,
      <code>structural</code>) — there, injected fillers silently create unlabeled golds. So the
      64k/128k numbers for outlier, outlier amazon, cycle, textgroups, absence and xabsence are
      <b>retracted</b> and deliberately not shown; they need natively-generated rungs. The rule is
      encoded as <code>gold_semantics</code> in
      <code>debug/ctc_modelscale/expand_ctc_rung.py</code>.</p></div>

    <h3>Model scale: 0.8B → 2B → 4B</h3>
    <p>Five of the tasks were retrained at 0.8B and 2B on the identical shards and scored on the
      identical ladders, so the grid's 4B column extends into a scale axis. Two things fall out, and
      they are different kinds of result.</p>
    <div class="scroller"><table class="aux">
      <thead><tr><th>Task</th><th>Arm</th><th>0.8B @32k</th><th>2B @32k</th><th>4B @32k</th>
        <th>gap @deepest</th></tr></thead>
      <tbody>
        <tr><td>hpqa</td><td>dense</td><td>0.940</td><td>0.940</td><td>0.943</td><td>−0.002</td></tr>
        <tr><td>hpqa</td><td>chunked</td><td>0.945</td><td>0.896</td><td>0.945</td><td></td></tr>
        <tr><td>contra (realistic)</td><td>dense</td><td>0.861</td><td>0.910</td><td>0.946</td><td>+0.247</td></tr>
        <tr><td>contra (realistic)</td><td>chunked</td><td>—</td><td>0.589</td><td>0.699</td><td></td></tr>
        <tr><td>fiqa</td><td>dense</td><td>0.287</td><td>—</td><td>0.310</td><td>+0.030</td></tr>
        <tr><td>fiqa</td><td>chunked</td><td>0.244</td><td>—</td><td>0.280</td><td></td></tr>
        <tr><td>qdmatch nq</td><td>dense</td><td><b>0.000</b></td><td>0.663</td><td>0.686</td><td>+0.582</td></tr>
        <tr><td>qdmatch nq</td><td>chunked</td><td><b>0.000</b></td><td>0.179</td><td>0.104</td><td></td></tr>
        <tr><td>reorder <em>(@16k)</em></td><td>dense</td><td><b>0.000</b></td><td><b>0.000</b></td><td>0.047</td><td>+0.047</td></tr>
        <tr><td>reorder <em>(@16k)</em></td><td>chunked</td><td>—</td><td><b>0.000</b></td><td>0.000</td><td></td></tr>
      </tbody>
    </table></div>
    <p><b>Two tasks have a capability threshold, and they sit at different places.</b> qdmatch nq is
      at zero on <em>both</em> arms at 0.8B — 0.039 and 0.033 at 2k, decaying to 0.000 — and then
      works at 2B (0.995 dense at 2k). That is not a chunking effect and not underfitting: the same
      0.8B base learned contradiction to 0.962 and hpqa to 0.995 on the same wave of runs, while
      qdmatch's training CE plateaued at 0.78 against 2B's 0.06. reorder's threshold is one step
      higher: it is at chance at 0.8B <em>and</em> 2B (|kendall_tau| &lt; 0.01 at every rung) and only
      becomes a task at 4B, where it reaches 0.747 at 2k.</p>
    <p><b>Scaling does not close the CTC gap — on the hardest task it widens it.</b> hpqa (O(N)
      retrieval) is saturated at 0.8B and has no gap at any scale. But qdmatch nq goes from
      no-measurable-gap at 0.8B (both arms dead) to +0.484 at 2B to <b>+0.582</b> at 4B, because the
      dense arm keeps improving while the chunked arm gets <em>worse</em> from 2B to 4B
      (0.179 → 0.104). Contradiction narrows slightly, +0.321 → +0.247. So a bigger model is not a
      substitute for the attention pattern.</p>
    <div class="flag"><p><b>Four cells are missing, and two of them for an unexplained reason.</b>
      fiqa was only run at 0.8B, so it has no 2B column. But <b>contradiction and reorder have no
      0.8B chunked arm</b>: those two runs OOM at 0.8B × chunked-mix × seq_len 40960 on both saturn
      and jupiter, while the <em>same</em> config at 2B and 4B trains fine and 0.8B at 33792 also
      trains fine. Four hypotheses (activation checkpointing, sharding, architecture shape, shard
      length) were each tested and refuted. Until that is understood, read the 0.8B column as dense
      only — do not infer a gap from its absence.</p></div>

    <h3>The outlier fix-k “32k cliff” is not a cliff — which is why 32k is not in the grid</h3>
    <p>The ladder held ~0.95 to 16k and then read 0.319 at 32k on both arms, which looked like a
      length limit or a training-support boundary (training sampled <code>n ~ U[14,220]</code>, and
      the 32k rung <em>is</em> n=220). Filling the 111 → 220 gap shows neither: it is a smooth
      monotone decline that continues past the training support. Rendered as the last column of a
      2k–32k ladder that single point reads as “collapses at 32k”, which is the one thing it does
      not do, so the fix-k row stops at 16k and the real shape is shown here against n:</p>
    <div class="scroller"><table class="aux">
      <thead><tr><th>n docs</th><th>111</th><th>140</th><th>170</th><th>190</th><th>220</th><th>300</th><th>440</th></tr></thead>
      <tbody><tr><td>set_f1</td><td>0.982</td><td>0.882</td><td>0.610</td><td>0.566</td><td>0.391</td>
        <td>0.171</td><td>0.060</td></tr></tbody>
    </table></div>
    <p>parse_rate 1.0 and eval_size 500 at every point, 0% overlap with the training set, and n=300
      and n=440 sit outside <code>U[14,220]</code> yet continue the same curve rather than falling
      off it. The apparent cliff was rung spacing: the ladder steps 111 → 220 in one jump, and the
      only place a cliff could hide was inside it.</p>

    <h3>The stop-token bug is real, and it costs nothing</h3>
    <p><code>stop_token_ids</code> ships only <code>&lt;|endoftext|&gt;</code> (248044), but our SFT
      targets end the assistant turn with <code>&lt;|im_end|&gt;</code> (248046), so vLLM never
      stops — the model answers and then repeats itself to the token cap. Tasks graded with
      <code>stop="newline"</code> are protected by an accidental text-level cut; the seven on
      <code>stop="eos"</code> are not. That looked like it could invalidate rows: reorder's
      parse_rate decays 0.95 @4k → 0.654 @8k → 0.16 @16k, and a score that falls while parse_rate
      falls with it cannot be told apart from a length effect.</p>
    <p>It does not invalidate anything. A/B on <em>identical prefills</em>, changing only the stop
      token, at 4B:</p>
    <div class="scroller"><table class="aux">
      <thead><tr><th>Task</th><th>Rung</th><th>shipped</th><th>+&lt;|im_end|&gt;</th><th>Δ</th>
        <th>gen len p50</th></tr></thead>
      <tbody>
        <tr><td>outlier</td><td>2k</td><td>0.9820</td><td>0.9820</td><td>0.0000</td><td>1045 → 24</td></tr>
        <tr><td>outlier</td><td>8k</td><td>0.8765</td><td>0.8765</td><td>0.0000</td><td>1275 → 26</td></tr>
        <tr><td>outlier</td><td>16k</td><td>0.6789</td><td>0.6789</td><td>0.0000</td><td>1412 → 26</td></tr>
        <tr><td>outlier</td><td>32k</td><td>0.4299</td><td>0.4292</td><td>−0.0007</td><td>1388 → 28</td></tr>
        <tr><td>grouping</td><td>2k</td><td>0.8419</td><td>0.8419</td><td>0.0000</td><td>2407 → 73</td></tr>
        <tr><td>grouping</td><td>8k</td><td>0.5305</td><td>0.5303</td><td>−0.0002</td><td>2332 → 670</td></tr>
        <tr><td>grouping</td><td>16k</td><td>0.2850</td><td>0.2862</td><td>+0.0012</td><td>2357 → 1435</td></tr>
        <tr><td>reorder</td><td>8k</td><td>0.2591</td><td>0.2606</td><td>+0.0015</td><td>1021 → 248</td></tr>
        <tr><td>reorder</td><td>16k</td><td>0.0473</td><td>0.0503</td><td>+0.0030</td><td>1021 → 476</td></tr>
      </tbody>
    </table></div>
    <p>Generations shrink by up to <b>50×</b> and the metrics are unchanged — three of the outlier
      cells are bit-identical. The parsers were already cutting the ramble. So every
      <code>stop="eos"</code> number in the grid stands as measured, and the decaying parse_rate on
      reorder is a real property of the model rather than a harness artifact. The fix is still worth
      keeping for wall-clock (a 50× shorter generation is a much cheaper eval), not for accuracy.</p>
    <div class="flag"><p><b>Retracted: an earlier version of this page claimed the fix moved outlier
      fix-k by +0.0722 at n=220.</b> That comparison was confounded and is withdrawn. The 0.3191 came
      from the original ladder file and the 0.3913 from a <em>freshly generated</em> n=220 rung in
      the extension job — same n, different sample — so the delta mixed eval-set variation with the
      stop token and was attributed entirely to the latter. The controlled A/B above, on identical
      prefills, is the measurement that should be used. The fix-k n-curve itself is unaffected: it
      was generated end to end with the fix applied, so it is internally consistent.</p></div>
  </div>
</section>

<section>
  <h2>What is missing, in the order worth doing it</h2>
  <ol class="todo">
    <li><s><b>Relay, export and eval qdmatch fiqa.</b></s> <b>Done 2026-08-13</b> — both arms
      trained on Beaker/jupiter at 4B (<code>01KZWZ8VQVYCNE9CVKGPKQ7GD0</code>,
      <code>01KZWZ9QGN2GEDC8XCWJNBYC3T</code>), relayed, exported and evaluated. Every row in the
      grid now has numbers in both arms.</li>
    <li><s><b>Re-eval the <code>stop="eos"</code> rows with <code>&lt;|im_end|&gt;</code>
      added.</b></s> <b>Done — no re-eval needed.</b> The controlled A/B found zero metric change
      across outlier, grouping and reorder despite generations shrinking up to 50×, so those cells
      stand as measured. See the stop-token section above.</li>
    <li><b>Build native 64k/128k rungs for the injection-unsafe tasks</b> (outlier, outlier amazon,
      cycle, textgroups, absence, xabsence) so their length-generalization numbers can be
      un-retracted.</li>
    <li><b>Fill three one-off holes:</b> reorder 32k and absence 32k (neither rung was ever built),
      and niah dense 2k — the on-disk 0.164 predates the pipeline fix and the two 64k/128k niah
      builds disagree with each other (0.736 vs 0.002), so that ladder needs a rebuild before any of
      it is quoted.<span class="cost">3 cells</span></li>
    <li><b>Replace the xabsence row.</b> It is struck through because its numbers never came from a
      model that learned the task — training CE sat on the 0.78 flatline. Four learnability probes
      (data scale 20k, small P=4, a 50-example memorization control, and a Gutenberg-corpus variant
      matching the sibling <em>absence</em> task) now reach CE 0.23–0.33; their generation-level
      evals are in flight and decide whether the row can be rebuilt at all. The obliq row still
      needs its grade JSONs persisted — those five cells exist only as prose.</li>
    <li><b>Rebuild absence, niah, strmatch and qdmatch hpqa token-accurate</b>, the way msmarco and
      rerank already were. Their rung labels are off by 0.5× to 3.6×, which mislabels the
      x-axis rather than the metric.</li>
  </ol>
</section>

<section>
  <h2>Reading the numbers</h2>
  <div class="method">
    <p><b>Error bars.</b> Each ± is the binomial standard error
      <code>sqrt(p(1-p)/eval_size)</code>. For a per-example metric valued in [0,1] the Bernoulli
      variance is the maximum, so this is an <em>upper bound</em> on the true SE, not an estimate of
      it — the graders do not persist per-example scores. Run-to-run seed variation is on top of
      this and is not measured here.</p>
    <p><b>eval_size is 500</b> everywhere except scifact (300, ±.019 at f1≈0.93) and obliq
      (126, ±.042). Those two are flagged in amber in the grid and no sub-0.05 difference on
      them should be read as real.</p>
    <p><b>Gap</b> is dense minus chunked at the deepest rung where both arms have a number, with that
      rung noted underneath. It is the CTC quantity: flat across the ladder for O(N) tasks, widening
      for O(NM) and O(N²).</p>
    <div class="flag"><p><b>Two different contradiction rows.</b> Row 18 is
      <em>NIAH</em>-contradiction, a retrieval task. Row 19, <em>contradiction (realistic)</em>, is
      the PubMed multi-claim task — a different task, added here as the 22nd row.</p>
      <p><b>The realistic row is scored on the IID ladder, and that choice is the whole result.</b>
      Training is <code>realistic</code>-mode gold pairs; every earlier contradiction number was
      scored on a <code>both</code>-mode eval, a different generator whose gold pairs are 38%
      near-duplicates. On that mismatched ladder the same dense checkpoint reads 0.843→0.559 and
      chunked 0.402→0.191. Swapping in the mode-matched ladder, with nothing retrained, gives the
      row shown: dense 0.990→0.946 and chunked 0.861→0.699. So dense does not collapse with corpus
      size — 13.6× more documents costs 0.043 f1 — and the dense-chunked gap <em>widens</em> with n
      (0.128→0.247) instead of narrowing. Do not mix the two ladders in one figure; see
      <code>records/contradiction-train-eval-non-iid.md</code>.</p>
      <p>Its 2k column is the <code>rung_2560</code> / n=56 file: n=44 sat below the training
      minimum of 52, so the IID rebuild starts one step up. The other four rungs are n-identical to
      the old ladder (92 / 187 / 379 / 762), which is what isolates the mode as the only change.</p></div>
    <p><b>Provenance.</b> Numbers are harvested from the node-local
      <code>ctc_suite_vllm_results{{,_chunked}}</code> and <code>ctc_newrung_results</code> trees by
      <code>debug/ctc_final_suite/harvest_grades.py</code>; when a cell exists on more than one node
      the newest file wins. Two rows come from elsewhere: <b>outlier fix-k</b> and <b>qdmatch nq</b>
      were backfilled by the model-scale drivers and are read from
      <code>debug/ctc_modelscale/results_4b/</code>, filtered to 4B by run name so that no 0.8B or 2B
      cell can leak into this grid. Roster and coverage notes live in
      <code>records/ctc-final-suite.md</code>.</p>
    <p><b>The scale table is built against the grid, not beside it.</b>
      <code>build_scale_table.py</code> reads the 0.8B/2B cells from
      <code>debug/ctc_modelscale/results/</code> and takes the 4B column by reading
      <code>suite_table.json</code> back, so the scale figure and the grid cannot disagree about what
      4B scored. Every scale cell was verified to sit on the <em>same</em> ladder as its 4B row from
      the <code>eval_data</code> path in its grade JSON — contradiction on
      <code>eval_rungs/contradiction_iid</code>, not the both-mode ladder, whose copies are kept in a
      separate <code>__BOTHMODE</code> directory and excluded by the run-name pattern. A scale plot
      built across two different eval sets would read as a capability curve.</p>
    <div class="flag"><p><b>oolong is pinned to the retrain, not to “newest”.</b> Newest-file-wins
      resolves duplicate grades of the same model; it cannot resolve two different models sharing a
      task directory, and oolong has exactly that — <code>ctc-4b-oolong-*</code> (original) and
      <code>ctcms-oolong-*-4b-vsl</code> (the retrain that fixed it). Left alone the grid rendered
      oolong as dense 0.232 vs chunked 0.628 at 2k, i.e. the dense arm losing badly, purely because
      the dense arm was the broken original and the chunked arm was not. The row now shows the
      retrained pair on both sides (0.710 vs 0.691) and is pinned by run name, so a later re-grade of
      the original cannot quietly take it back.</p></div>
  </div>
</section>

<footer>Rendered {stamp} from suite_table.json · {n_dense + n_chunk} graded cells · {n_tasks} tasks</footer>
</div>
"""

out = "debug/ctc_final_suite/ctc_suite_grid.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)} bytes)")
