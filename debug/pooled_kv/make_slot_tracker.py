"""
Render results/pooled_kv/slot_probe_results.csv (collect_slot_results.py) into the "Slot Construction
Tracker" page: a cross-task leaderboard (mean answer-CE gap to full attention over the 4 tasks at
32k on the dense ladder checkpoints), per-task detail, and the train-time speedup / activation
memory saving each construction implies. Re-run after collect_slot_results.py; publish the HTML.

    python debug/pooled_kv/make_slot_tracker.py  -> results/pooled_kv/slot_tracker.html
"""

import csv
import html
from collections import defaultdict
from datetime import date

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
CSV = f"{REPO}/results/pooled_kv/slot_probe_results.csv"
OUT = f"{REPO}/results/pooled_kv/slot_tracker.html"
TASKS = ["contradiction", "nq", "outlier", "oolong"]
KEEPS = [0.3333, 0.1667, 0.0833, 0.0]


def keep_label(k):
    return {0.3333: "1/3", 0.1667: "1/6", 0.0833: "1/12", 0.0: "gold only"}.get(round(k, 4), f"{k:.3f}")


def fmt(v, p=3):
    return "—" if v in (None, "") else f"{float(v):.{p}f}"


def main():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ("keep", "answer_ce", "full_ce", "delta_ce", "top1_agree", "kl", "exact_answer", "compaction", "train_speedup_x", "act_memory_saving_pct"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["G"] = int(r["G"]); r["rows"] = int(r["rows"]); r["ceiling_only"] = int(r["ceiling_only"])
    # canonical 32k / dense-ladder / beaker rows (one per task x config); prefer the most-rows source
    main = [r for r in rows if r["rung"] == "32k" and r["checkpoint"].startswith("dense") and r["construction"] != "full attention"]
    key = lambda r: (r["construction"], r["bias"], r["G"], round(r["keep"], 4))
    best = {}
    for r in main:
        k = (r["task"],) + key(r)
        if k not in best or (r["rows"], r["source"] == "beaker") > (best[k]["rows"], best[k]["source"] == "beaker"):
            best[k] = r
    full_ce = {}
    for r in rows:
        if r["construction"] == "full attention" and r["rung"] == "32k" and r["checkpoint"].startswith("dense"):
            if r["task"] not in full_ce or r["rows"] > full_ce[r["task"]][1]:
                full_ce[r["task"]] = (r["full_ce"] or r["answer_ce"], r["rows"])
    configs = defaultdict(dict)
    for (task, *k), r in best.items():
        configs[tuple(k)][task] = r
    lb = []
    for k, per in configs.items():
        deltas = [per[t]["delta_ce"] for t in TASKS if t in per and per[t]["delta_ce"] is not None]
        if not deltas:
            continue
        comp = sum(per[t]["compaction"] for t in per) / len(per)
        rows_min = min(per[t]["rows"] for t in per)
        lb.append({"key": k, "per": per, "mean_delta": sum(deltas) / len(deltas), "n_tasks": len(deltas), "comp": comp, "rows_min": rows_min,
                   "ceiling": any(per[t]["ceiling_only"] for t in per)})
    lb.sort(key=lambda x: (x["ceiling"], -x["n_tasks"], x["mean_delta"]))

    css = """
<style>
:root{--bg:#F6F7F5;--surface:#FFFFFF;--ink:#1B2226;--muted:#5C6871;--accent:#0E6B66;--accent-ink:#0A4F4B;--tint:#E3EFEC;--warn:#B9552A;--warn-tint:#F7E9E1;--rule:#D6DBD7;--rule-soft:#E7EBE8;--code-bg:#EEF1EF;--good:#DDF1E4;--bad:#F9E3DC}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12191B;--surface:#182022;--ink:#E4E9E6;--muted:#93A0A5;--accent:#52BDB4;--accent-ink:#8ED9D2;--tint:#16302F;--warn:#E6915F;--warn-tint:#3A241A;--rule:#2A363A;--rule-soft:#212B2E;--code-bg:#1F292C;--good:#1B3A2A;--bad:#3E2420}}
:root[data-theme="dark"]{--bg:#12191B;--surface:#182022;--ink:#E4E9E6;--muted:#93A0A5;--accent:#52BDB4;--accent-ink:#8ED9D2;--tint:#16302F;--warn:#E6915F;--warn-tint:#3A241A;--rule:#2A363A;--rule-soft:#212B2E;--code-bg:#1F292C;--good:#1B3A2A;--bad:#3E2420}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.5}
.page{max-width:1180px;margin:0 auto;padding:36px 22px 80px}
h1,h2,h3{font-family:"Bricolage Grotesque","IBM Plex Sans",Arial,sans-serif;text-wrap:balance;line-height:1.15;margin:0}
h1{font-size:2.2rem;font-weight:700}h2{font-size:1.4rem;font-weight:600;margin-top:44px;padding-top:14px;border-top:2px solid var(--ink)}h3{font-size:1.05rem;font-weight:600;margin-top:22px}
p{margin:10px 0;max-width:80ch}.eyebrow{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:1.05rem;color:var(--muted);max-width:76ch}
.status{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.chip{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.74rem;padding:4px 10px;border-radius:999px;border:1px solid var(--rule);color:var(--muted);background:var(--surface)}.chip.ok{border-color:var(--accent);color:var(--accent-ink);background:var(--tint)}.chip.run{border-color:var(--warn);color:var(--warn);background:var(--warn-tint)}
.tablewrap{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--rule-soft);vertical-align:top;white-space:nowrap}th{font-family:"IBM Plex Mono",Menlo,monospace;font-weight:500;font-size:.7rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule);position:sticky;top:0;background:var(--bg)}
td.num,th.num{text-align:right;font-family:"IBM Plex Mono",Menlo,monospace;font-size:.82rem}td.good{background:var(--good)}td.bad{background:var(--bad)}tr.ceil td{color:var(--muted);font-style:italic}tr.best td{font-weight:600}
.note{background:var(--tint);border-left:3px solid var(--accent);padding:9px 13px;margin:14px 0;max-width:84ch;font-size:.93rem}.note.warn{background:var(--warn-tint);border-left-color:var(--warn)}
code{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.85em;background:var(--code-bg);padding:1px 5px;border-radius:3px}
ul{padding-left:20px;max-width:80ch}li{margin:5px 0}details{margin:10px 0}summary{cursor:pointer;color:var(--accent-ink);font-weight:600}
</style>"""
    n_running = sum(1 for x in lb if x["rows_min"] < 24)
    h = [f"""<title>Slot Construction Tracker</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
{css}
<div class="page">
<span class="eyebrow">OLMo-core · soft-KV eval-side probes · updated {date.today().isoformat()}</span>
<h1>Slot Construction Tracker</h1>
<p class="lede">Every way of standing in for a pooled document, scored on the dense-trained Qwen3.5-4B ladder checkpoints over 24 held-out rows per task at 32k. The number that matters is the answer-token cross-entropy gap to full attention, averaged over the four tasks. Each row also carries the train-time speedup and activation-memory saving the construction implies at that keep fraction.</p>
<div class="status"><span class="chip ok">{len(lb)} constructions × keep settings</span><span class="chip ok">4 tasks</span><span class="chip {'run' if n_running else 'ok'}">{n_running} still accumulating rows</span><span class="chip">full attention CE: {' · '.join(f"{t} {fmt(full_ce[t][0])}" for t in TASKS if t in full_ce)}</span></div>
<div class="note"><b>How to read.</b> Δ CE = answer cross-entropy minus full attention's on the same rows (lower is better; 0 means the construction is invisible). Speedup ≈ 1 / compaction ratio and memory saving ≈ 1 − compaction, both measured per row and validated by the throughput benchmark (3.5× at compaction 0.36, 11.9× at 0.10). Italic rows are ceilings: they use the dense forward's own keys and values and cannot be trained with as-is. Bold marks the best trainable construction at each keep.</div>
<h2>Cross-task leaderboard, 32k</h2>
<div class="tablewrap"><table><tr><th>construction</th><th>slot bias</th><th class=num>slots/doc</th><th class=num>keep</th><th class=num>mean Δ CE</th>""" + "".join(f"<th class=num>Δ {t}</th>" for t in TASKS) + """<th class=num>speedup</th><th class=num>act. mem saving</th><th class=num>rows</th></tr>"""]
    best_trainable = {}
    for x in lb:
        if not x["ceiling"]:
            kk = round(x["key"][3], 4)
            if kk not in best_trainable or x["mean_delta"] < best_trainable[kk]["mean_delta"]:
                best_trainable[kk] = x
    for x in lb:
        cons, bias, G, keep = x["key"]
        cls = "ceil" if x["ceiling"] else ""
        if not x["ceiling"] and best_trainable.get(round(keep, 4)) is x:
            cls += " best"
        cells = "".join(f"<td class=num>{fmt(x['per'][t]['delta_ce']) if t in x['per'] else '—'}</td>" for t in TASKS)
        h.append(f"<tr class='{cls}'><td>{html.escape(cons)}</td><td>{html.escape(bias)}</td><td class=num>{G}</td><td class=num>{keep_label(keep)}</td><td class=num>{fmt(x['mean_delta'])}</td>{cells}<td class=num>{x['comp'] and fmt(1/x['comp'],1)}×</td><td class=num>{100*(1-x['comp']):.0f}%</td><td class=num>{x['rows_min']}</td></tr>")
    h.append("</table></div>")
    # per-task detail
    h.append("<h2>Per task, 32k, dense ladder checkpoints</h2>")
    for t in TASKS:
        trs = sorted([r for (task, *k), r in best.items() if task == t], key=lambda r: (r["ceiling_only"], round(r["keep"], 4) * -1, r["answer_ce"]))
        h.append(f"<h3>{t} <span class='eyebrow'>full attention CE {fmt(full_ce.get(t, (None,))[0])}</span></h3><div class='tablewrap'><table><tr><th>construction</th><th>bias</th><th class=num>slots/doc</th><th class=num>keep</th><th class=num>answer CE</th><th class=num>Δ CE</th><th class=num>top-1 = full</th><th class=num>exact answer</th><th class=num>compaction</th><th class=num>speedup</th><th class=num>mem saving</th><th>source</th><th class=num>rows</th></tr>")
        for r in trs:
            h.append(f"<tr class='{'ceil' if r['ceiling_only'] else ''}'><td>{html.escape(r['construction'])}</td><td>{html.escape(r['bias'])}</td><td class=num>{r['G']}</td><td class=num>{keep_label(r['keep'])}</td><td class=num>{fmt(r['answer_ce'])}</td><td class=num>{fmt(r['delta_ce'])}</td><td class=num>{fmt(r['top1_agree'])}</td><td class=num>{fmt(r['exact_answer'],2)}</td><td class=num>{fmt(r['compaction'])}</td><td class=num>{fmt(r['train_speedup_x'],1)}×</td><td class=num>{r['act_memory_saving_pct']:.0f}%</td><td>{r['source']}</td><td class=num>{r['rows']}</td></tr>")
        h.append("</table></div>")
    # other rungs / checkpoints
    other = [r for r in rows if not (r["rung"] == "32k" and r["checkpoint"].startswith("dense")) and r["construction"] != "full attention"]
    if other:
        h.append("<h2>Other rungs and checkpoints</h2><details><summary>8k rung on the dense checkpoints (sneetches) and the local 2k-trained contradiction checkpoint (horton)</summary><div class='tablewrap'><table><tr><th>task</th><th>rung</th><th>checkpoint</th><th>construction</th><th>bias</th><th class=num>G</th><th class=num>keep</th><th class=num>answer CE</th><th class=num>full CE</th><th class=num>Δ CE</th><th class=num>compaction</th><th class=num>rows</th></tr>")
        for r in sorted(other, key=lambda r: (r["task"], r["rung"], r["checkpoint"], r["ceiling_only"], -r["keep"], r["answer_ce"])):
            h.append(f"<tr class='{'ceil' if r['ceiling_only'] else ''}'><td>{r['task']}</td><td>{r['rung']}</td><td>{html.escape(r['checkpoint'])}</td><td>{html.escape(r['construction'])}</td><td>{html.escape(r['bias'])}</td><td class=num>{r['G']}</td><td class=num>{keep_label(r['keep'])}</td><td class=num>{fmt(r['answer_ce'])}</td><td class=num>{fmt(r['full_ce'])}</td><td class=num>{fmt(r['delta_ce'])}</td><td class=num>{fmt(r['compaction'])}</td><td class=num>{r['rows']}</td></tr>")
        h.append("</table></div></details>")
    h.append("""<h2>Ideas log</h2>
<ul>
<li><b>Slot logit bias</b> (none / +log L / +log L + c / constant c): calibration is task-dependent; retrieval tasks want ≤ 0, oolong wants ≥ log L + 2; no single bias helps all four. <i>Done, all four tasks, 24 rows.</i></li>
<li><b>Oracle mean K/V</b> (the dense forward's own per-layer means): equals the default soft token within 0.01 at every keep on contradiction; ceiling for a MEAN slot is the default itself. <i>Done, 24 rows on all four tasks.</i></li>
<li><b>Fitted log-mass slot</b> (k*, v*, c per document and layer, fitted on the row's queries): the ceiling for ONE static slot per document. <i>Running.</i></li>
<li><b>G slots per document</b> (2 / 4 / 8 contiguous pieces) for the soft token, the mean-K/V oracle and the fitted slot. <i>Running.</i></li>
<li>Next candidates: per-task calibrated constant learned from a handful of rows; slot count proportional to document length; soft token from a small trained pooler on the dense checkpoint's own keys (trainable version of the fitted slot); keep policy by document salience rather than at random.</li>
</ul>
<p class="eyebrow">source: results/pooled_kv/slot_probe_results.csv · debug/pooled_kv/{eval_side_slot_probe,oracle_meankv_probe,slot_ceiling_probe,collect_slot_results,make_slot_tracker}.py</p>
</div>""")
    open(OUT, "w").write("\n".join(h))
    print("wrote", OUT, "leaderboard rows", len(lb))


if __name__ == "__main__":
    main()
