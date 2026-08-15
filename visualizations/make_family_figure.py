#!/usr/bin/env python3
"""Render the cross-family contradiction/hotpotqa figure as a self-contained HTML page.

    python visualizations/make_family_figure.py

Numbers are read from the grade JSONs, never typed here -- the one exception is documented
inline (Qwen3.5-4B, see SOURCES below) and is itself read from a harvested JSON.

── WHY EACH SERIES COMES FROM WHERE IT DOES ───────────────────────────────────────────────────
OLMo-3-7B and Llama-3.2-3B are directly comparable: both were scored by the NATIVE olmo-core
evaluator (``run_rung_eval``) against the same ``eval_rungs/contradiction`` and
``eval_rungs/hotpotqa`` ladders, at the same 2k/4k/8k/16k rungs, 500 examples per cell.

Qwen3.5-4B is not, and the reason matters:

* Its native contradiction results under ``results/ctc_suite/contradiction/qwen3.5-4b_*`` are
  SUPERSEDED. They were produced 2026-07-20/23 with ``max_length: 6144``, which is far below the
  real prompt length -- contradiction rungs hold a fixed corpus but wildly varying claim lengths,
  so at "rung 4096" the median prompt is ~6,969 tokens and the longest is 23,796. Prompts over the
  limit are skipped and scored 0 while ``parse_rate`` still reads 1.0. That is why those files fall
  0.8649 -> 0.2185 -> 0.0380: it is truncation, not length generalization. OLMo kept explicit
  ``-maxlen-truncated`` copies of its equivalent; the Qwen ones were never renamed, so they look
  current. DO NOT USE THEM.
* The Qwen series plotted here therefore comes from the vLLM harness (``harvested_grades.json``),
  on the same clean ladder. This is a HARNESS difference, annotated on the figure.

How much that harness difference is worth, measured rather than assumed: on hotpotqa the two
harnesses overlap at rung 2048 and agree to four decimals (native 0.9980 vs vLLM 0.9980). There is
no overlapping contradiction point, which is why native Qwen contradiction re-runs are worth doing
before this figure goes in a paper.

── THE LLAMA NUMBERS ARE THE RE-SCORED ONES ───────────────────────────────────────────────────
Llama-3.2-3B emits no EOS on this task and rambles to the token cap in 94-99% of examples, which
collapses precision (raw f1 0.359 at 2k). ``regrade_truncated.json`` re-scores the saved
predictions truncated at the first ``]]``. Those are the numbers here; the raw ones are the figures
still quoted in paper-v2-todo-status.md and they understate Llama by roughly 2x.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNGS = [2048, 4096, 8192, 16384]
RUNG_LABEL = {2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k"}

#: family -> (display, colour token suffix, harness note)
FAMILIES = ["Qwen3.5-4B", "OLMo-3-7B", "Llama-3.2-3B"]


def from_rung_dir(path):
    """Read {rung: metric} from a directory of ``rung_N.json`` grade files.

    :param path: Directory holding ``rung_<N>.json``.
    :returns: Mapping of rung to metric value, restricted to :data:`RUNGS`.
    """
    out = {}
    for r in RUNGS:
        f = os.path.join(REPO, path, f"rung_{r}.json")
        if os.path.exists(f):
            out[r] = json.load(open(f))["metric_value"]
    return out


def from_harvested(task, arm):
    """Read {rung: metric} for the 4B vLLM suite run out of ``harvested_grades.json``.

    :param task: Harvested task key.
    :param arm: ``dense`` or ``chunked``.
    :returns: Mapping of rung to metric value.
    """
    rows = json.load(open(os.path.join(REPO, "debug/ctc_final_suite/harvested_grades.json")))
    return {
        r["rung"]: r["metric_value"]
        for r in rows
        if r.get("task") == task and r.get("arm") == arm and r.get("rung") in RUNGS
    }


def llama_contradiction(arm):
    """Read the ``]]``-truncated Llama contradiction re-score.

    :param arm: ``full`` or ``chunked-mix``.
    :returns: Mapping of rung to truncated f1.
    """
    rows = json.load(open(os.path.join(REPO, "results/ctc_suite_llama/contradiction/regrade_truncated.json")))
    return {r["rung"]: r["truncated"]["f1"] for r in rows if r["arm"] == arm and r["rung"] in RUNGS}


SERIES = {
    "contradiction": {
        ("Qwen3.5-4B", "dense"): from_harvested("contradiction", "dense"),
        ("Qwen3.5-4B", "chunked"): from_harvested("contradiction", "chunked"),
        ("OLMo-3-7B", "dense"): from_rung_dir("results/ctc_suite/contradiction/olmo3-7b_full"),
        ("OLMo-3-7B", "chunked"): from_rung_dir("results/ctc_suite/contradiction/olmo3-7b_chunked-mix"),
        ("Llama-3.2-3B", "dense"): llama_contradiction("full"),
        ("Llama-3.2-3B", "chunked"): llama_contradiction("chunked-mix"),
    },
    "hotpotqa": {
        ("Qwen3.5-4B", "dense"): from_harvested("hotpotqa", "dense"),
        ("Qwen3.5-4B", "chunked"): from_harvested("hotpotqa", "chunked"),
        ("OLMo-3-7B", "dense"): from_rung_dir("results/ctc_suite_olmo3_hpqa/retrieval/olmo3-7b_full"),
        ("OLMo-3-7B", "chunked"): from_rung_dir("results/ctc_suite_olmo3_hpqa/retrieval/olmo3-7b_chunked-mix"),
        ("Llama-3.2-3B", "dense"): from_rung_dir("results/ctc_suite_llama/retrieval/llama3.2-3b_full"),
        ("Llama-3.2-3B", "chunked"): from_rung_dir("results/ctc_suite_llama/retrieval/llama3.2-3b_chunked-mix"),
    },
}

METRIC = {"contradiction": "set F1", "hotpotqa": "gold-id F1"}
CLASS = {"contradiction": "O(N&#178;) &mdash; pair finding", "hotpotqa": "O(N) &mdash; retrieval"}

# Plot geometry (user units; the SVG scales via viewBox).
W, H = 460, 330
PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 16, 44


def x_of(i):
    """X pixel for rung index *i* (rungs are evenly spaced, i.e. log-2 spacing)."""
    return PAD_L + i * (W - PAD_L - PAD_R) / (len(RUNGS) - 1)


def y_of(v):
    """Y pixel for a metric value in [0, 1]."""
    return PAD_T + (1 - v) * (H - PAD_T - PAD_B)


def panel(task):
    """Emit the SVG for one task panel.

    :param task: ``contradiction`` or ``hotpotqa``.
    :returns: SVG markup string.
    """
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{task} across model families">']
    # horizontal gridlines + y labels
    for v in (0, .2, .4, .6, .8, 1.0):
        y = y_of(v)
        p.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}"/>')
        p.append(f'<text class="tick ytick" x="{PAD_L-9}" y="{y+3.5:.1f}">{v:.1f}</text>')
    # x labels
    for i, r in enumerate(RUNGS):
        p.append(f'<text class="tick" x="{x_of(i):.1f}" y="{H-PAD_B+19}">{RUNG_LABEL[r]}</text>')
    p.append(f'<text class="axis-title" x="{(PAD_L+W-PAD_R)/2:.1f}" y="{H-6}">context rung</text>')
    p.append(f'<text class="axis-title" transform="translate(13,{(PAD_T+H-PAD_B)/2:.1f}) rotate(-90)" x="0" y="0">{METRIC[task]}</text>')

    for fam in FAMILIES:
        for arm in ("dense", "chunked"):
            pts = SERIES[task][(fam, arm)]
            xy = [(x_of(i), y_of(pts[r])) for i, r in enumerate(RUNGS) if r in pts]
            if len(xy) < 2:
                continue
            d = " ".join(("M" if k == 0 else "L") + f"{x:.1f},{y:.1f}" for k, (x, y) in enumerate(xy))
            cls = f"ln f-{FAMILIES.index(fam)}" + ("" if arm == "dense" else " dash")
            p.append(f'<path class="{cls}" d="{d}"/>')
            for x, y in xy:
                if arm == "dense":
                    p.append(f'<circle class="pt f-{FAMILIES.index(fam)}" cx="{x:.1f}" cy="{y:.1f}" r="3.4"/>')
                else:
                    p.append(f'<rect class="pt f-{FAMILIES.index(fam)}" x="{x-2.9:.1f}" y="{y-2.9:.1f}" width="5.8" height="5.8"/>')
    p.append("</svg>")
    return "\n".join(p)


def table(task):
    """Emit the numeric table backing one panel."""
    rows = []
    for fam in FAMILIES:
        for arm, lab in (("dense", "dense"), ("chunked", "chunked-mix")):
            pts = SERIES[task][(fam, arm)]
            cells = "".join(
                f'<td class="num">{pts[r]:.3f}</td>' if r in pts else '<td class="num na">&mdash;</td>'
                for r in RUNGS
            )
            gap = ""
            if arm == "chunked":
                d = SERIES[task][(fam, "dense")]
                common = [r for r in RUNGS if r in d and r in pts]
                if common:
                    r = common[-1]
                    gap = f'<td class="num gap">&minus;{d[r]-pts[r]:.3f}</td>'
            if not gap:
                gap = '<td class="num"></td>'
            rows.append(
                f'<tr><td class="rowlab"><span class="swatch f-{FAMILIES.index(fam)}"></span>{fam}</td>'
                f'<td class="rowlab arm">{lab}</td>{cells}{gap}</tr>'
            )
    return "\n".join(rows)


HTML = f"""<title>Chunking Across Three Families</title>
<style>
:root{{
  --paper:#FAF9F7; --surface:#FFFFFF; --surface-2:#F2F1ED;
  --ink:#1A1D23; --ink-soft:#454A52; --neutral:#6C7280;
  --rule:#DEDCD6; --rule-soft:#EBE9E4;
  --accent:#2F5F80;
  --f0:#2F5F80;  /* Qwen3.5  - steel blue */
  --f1:#2E7D6B;  /* OLMo-3   - teal */
  --f2:#A8622F;  /* Llama    - ochre */
  --warn:#9A3B2C; --warn-bg:#F6E2DE;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --paper:#14171C; --surface:#1A1E24; --surface-2:#21252C;
    --ink:#E9EAED; --ink-soft:#B7BBC3; --neutral:#8B919B;
    --rule:#2C313A; --rule-soft:#242932;
    --accent:#83B5D7;
    --f0:#7FB2D8; --f1:#63C0A9; --f2:#DE9А61;
    --warn:#E08B7B; --warn-bg:#2E1E1B;
  }}
}}
:root[data-theme="dark"]{{
  --paper:#14171C; --surface:#1A1E24; --surface-2:#21252C;
  --ink:#E9EAED; --ink-soft:#B7BBC3; --neutral:#8B919B;
  --rule:#2C313A; --rule-soft:#242932;
  --accent:#83B5D7;
  --f0:#7FB2D8; --f1:#63C0A9; --f2:#DE9A61;
  --warn:#E08B7B; --warn-bg:#2E1E1B;
}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);margin:0;padding:0 20px 80px;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.6}}
.wrap{{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:40px}}
header{{padding:52px 0 0;display:flex;flex-direction:column;gap:14px}}
h1{{font-family:ui-serif,"Iowan Old Style",Georgia,serif;font-weight:600;
  font-size:clamp(2rem,4.4vw,2.85rem);line-height:1.12;letter-spacing:-.015em;margin:0;text-wrap:balance}}
.standfirst{{margin:0;font-size:1.05rem;color:var(--ink-soft);max-width:66ch}}
.stamp{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.71rem;
  letter-spacing:.12em;text-transform:uppercase;color:var(--neutral)}}
.legend{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;padding:2px 0}}
.lg{{display:flex;align-items:center;gap:7px;font-size:.85rem}}
.swatch{{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:-1px}}
.swatch.f-0{{background:var(--f0)}} .swatch.f-1{{background:var(--f1)}} .swatch.f-2{{background:var(--f2)}}
.lg .key{{width:26px;height:0;border-top:2.5px solid var(--neutral)}}
.lg .key.dash{{border-top-style:dashed}}
.panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:26px}}
.panel{{border:1px solid var(--rule);background:var(--surface);border-radius:2px;padding:16px 14px 8px;
  display:flex;flex-direction:column;gap:6px}}
.panel h2{{margin:0;font-family:ui-serif,"Iowan Old Style",Georgia,serif;font-size:1.22rem;font-weight:600}}
.panel .cls{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;
  letter-spacing:.06em;color:var(--neutral);text-transform:uppercase}}
svg{{width:100%;height:auto;display:block;overflow:visible}}
.grid{{stroke:var(--rule-soft);stroke-width:1}}
.tick{{fill:var(--neutral);font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;text-anchor:middle}}
.ytick{{text-anchor:end}}
.axis-title{{fill:var(--neutral);font-size:11px;text-anchor:middle;
  font-family:system-ui,sans-serif;letter-spacing:.04em}}
.ln{{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}}
.ln.dash{{stroke-dasharray:5 4;stroke-width:2}}
.f-0{{stroke:var(--f0)}} .f-1{{stroke:var(--f1)}} .f-2{{stroke:var(--f2)}}
circle.pt.f-0,rect.pt.f-0{{fill:var(--f0);stroke:none}}
circle.pt.f-1,rect.pt.f-1{{fill:var(--f1);stroke:none}}
circle.pt.f-2,rect.pt.f-2{{fill:var(--f2);stroke:none}}
.scroll{{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:.87rem}}
caption{{text-align:left;padding:11px 15px;font-size:.79rem;color:var(--neutral);border-bottom:1px solid var(--rule-soft)}}
th,td{{padding:8px 13px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule-soft)}}
thead th{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.71rem;letter-spacing:.07em;
  text-transform:uppercase;color:var(--neutral);font-weight:500;background:var(--surface-2);border-bottom:1px solid var(--rule)}}
td.rowlab,th.rowlab{{text-align:left}}
td.arm{{color:var(--neutral);font-size:.8rem}}
td.num{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}}
td.na{{color:var(--neutral)}}
td.gap{{color:var(--warn)}}
tbody tr:last-child td{{border-bottom:none}}
h3{{margin:0;font-size:1.02rem}}
.note{{border-left:2px solid var(--accent);background:var(--surface);padding:13px 17px;
  font-size:.9rem;color:var(--ink-soft);max-width:74ch}}
.note.warn{{border-left-color:var(--warn)}}
.note strong{{color:var(--ink)}}
section{{display:flex;flex-direction:column;gap:14px}}
.eyebrow{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.71rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);display:flex;align-items:center;gap:10px}}
.eyebrow::after{{content:"";flex:1;height:1px;background:var(--rule)}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;
  background:var(--surface-2);padding:1px 5px;border-radius:2px}}
footer{{border-top:1px solid var(--rule);padding-top:18px;font-size:.82rem;color:var(--neutral);max-width:74ch}}
</style>

<div class="wrap">
<header>
  <p class="stamp">CTC suite &middot; cross-family &middot; 500 examples per cell</p>
  <h1>Chunking Across Three Families</h1>
  <p class="standfirst">Document-chunked attention costs almost nothing on retrieval and a great deal on pair-finding &mdash; and that holds in every model family tested, at three different sizes and three different backbones. Solid lines are dense attention; dashed are chunked-mix.</p>
  <div class="legend">
    <span class="lg"><span class="swatch f-0"></span>Qwen3.5-4B</span>
    <span class="lg"><span class="swatch f-1"></span>OLMo-3-7B</span>
    <span class="lg"><span class="swatch f-2"></span>Llama-3.2-3B</span>
    <span class="lg"><span class="key"></span>dense</span>
    <span class="lg"><span class="key dash"></span>chunked-mix</span>
  </div>
</header>

<div class="panels">
  <div class="panel">
    <span class="cls">{CLASS['contradiction']}</span>
    <h2>Contradiction</h2>
    {panel('contradiction')}
  </div>
  <div class="panel">
    <span class="cls">{CLASS['hotpotqa']}</span>
    <h2>HotpotQA</h2>
    {panel('hotpotqa')}
  </div>
</div>

<section>
  <p class="eyebrow">the numbers</p>
  <div class="scroll">
    <table>
      <caption>Contradiction, <code>set_f1</code>. Gap column is dense minus chunked at the longest shared rung.</caption>
      <thead><tr><th class="rowlab">Family</th><th class="rowlab">Arm</th><th>2k</th><th>4k</th><th>8k</th><th>16k</th><th>gap @16k</th></tr></thead>
      <tbody>{table('contradiction')}</tbody>
    </table>
  </div>
  <div class="scroll">
    <table>
      <caption>HotpotQA, <code>gold_id_f1</code>.</caption>
      <thead><tr><th class="rowlab">Family</th><th class="rowlab">Arm</th><th>2k</th><th>4k</th><th>8k</th><th>16k</th><th>gap @16k</th></tr></thead>
      <tbody>{table('hotpotqa')}</tbody>
    </table>
  </div>
</section>

<section>
  <p class="eyebrow">reading it</p>
  <p class="note">Every family loses ground to chunking on contradiction and essentially none on HotpotQA. The absolute level differs a lot by family &mdash; OLMo-3-7B is the strongest dense model here and Llama-3.2-3B the weakest &mdash; but the <strong>shape</strong> is invariant: retrieval stays near ceiling out to 16k under a chunk mask, pair-finding falls away. That is the claim the suite rests on, and it does not depend on the Qwen backbone it was originally measured in.</p>
  <p class="note warn"><strong>One harness caveat.</strong> OLMo-3 and Llama were both scored by the native olmo-core evaluator on the same ladders; the Qwen series comes from the vLLM harness. Where the two overlap &mdash; HotpotQA at 2k &mdash; they agree to four decimals (0.9980 vs 0.9980), but there is no overlapping contradiction point, so the Qwen contradiction line carries an unquantified harness difference. Native Qwen contradiction re-runs would remove it.</p>
</section>

<section>
  <p class="eyebrow">provenance</p>
  <div class="note">
    <p style="margin:0 0 9px"><strong>Llama numbers are the re-scored ones.</strong> Llama-3.2-3B never emits EOS on contradiction and rambles to the token cap in 94&ndash;99% of examples, which destroys precision. Scored raw it reads 0.359 at 2k; truncated at the first <code>]]</code> it reads 0.743. The raw figures are still what <code>paper-v2-todo-status.md</code> quotes.</p>
    <p style="margin:0"><strong>Two Qwen result trees look current and one is not.</strong> <code>results/ctc_suite/contradiction/qwen3.5-4b_*</code> was produced with <code>max_length: 6144</code>, well under the real prompt lengths, so long prompts were silently skipped and scored 0 while <code>parse_rate</code> stayed 1.0 &mdash; it reads 0.865 / 0.219 / 0.038 and is truncation, not length generalization. OLMo kept explicit <code>-maxlen-truncated</code> copies of its equivalent; the Qwen ones were never renamed. They are not used here.</p>
  </div>
</section>

<footer>
  <p>Generated by <code>visualizations/make_family_figure.py</code> directly from the grade JSONs. Binomial SE at 500 examples is roughly &plusmn;0.021 near F1 0.70 and &plusmn;0.010 near 0.95; Llama trained on a 2k&ndash;16k context mix, the other two on 2k&ndash;32k.</p>
</footer>
</div>
"""

out = os.path.join(REPO, "visualizations/ctc_family_figure.html")
with open(out, "w") as f:
    f.write(HTML)
print("wrote", out)
for task in SERIES:
    for k, v in SERIES[task].items():
        missing = [r for r in RUNGS if r not in v]
        print(f"  {task:14s} {k[0]:14s} {k[1]:8s} " + " ".join(f"{r}:{v[r]:.4f}" for r in RUNGS if r in v)
              + (f"   MISSING {missing}" if missing else ""))
