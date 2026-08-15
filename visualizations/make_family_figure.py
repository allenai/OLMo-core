#!/usr/bin/env python3
"""Render the cross-family contradiction figure as a self-contained HTML page.

    python visualizations/make_family_figure.py

── ONE LADDER, FOUR FAMILIES ──────────────────────────────────────────────────────────────────
Every series is on ``contradiction_iid`` -- the ladder matching the ``contradiction_train`` shard
all four families fine-tuned on, and where the Qwen3.5-4B reference lives. That uniformity IS the
figure. The first attempt had Qwen on ``contradiction_clean`` and OLMo/Llama on ``contradiction``,
which are three distinct corpora with different rung counts and file sizes, and would have compared
four models on up to three different inputs while looking entirely reasonable.

── TWO SERIES CARRY CAVEATS, BOTH VISIBLE ON THE PAGE ─────────────────────────────────────────
* Llama numbers are the ``]]``-TRUNCATED re-score. Llama emits no EOS and rambles to the token cap
  on 87-100% of examples, which destroys precision: dense reads 0.625 raw and 0.970 truncated at
  2.5k. The raw figures are what paper-v2-todo-status.md still quotes.
* Olmo-Hybrid's CHUNKED arm is omitted, not missing. It scored 0.113/0.062/0.021/0.006, but its
  final training CE was 0.958 against OLMo-3's 0.171 on identical task, data and 1109 steps -- it
  never fit the training data. The same model's qdmatch chunked arm (healthy CE 0.156) reaches
  0.931 at 2k, so the backbone handles a chunk mask fine and this is an optimization failure of one
  run. Plotting it would read as a backbone result and be wrong.

── WHAT IS DELIBERATELY NOT HERE ──────────────────────────────────────────────────────────────
No HotpotQA panel. OLMo-3's retrieval results were scored against
``/data/prasann/ctc_olmo3/eval_rungs/rung_2048.jsonl`` -- a flat path -- while Llama and Qwen used
``.../eval_rungs/hotpotqa/rung_2048.jsonl``. Those may well be the same bytes, but after the
three-ladder contradiction mess an unverified "probably the same" is exactly the assumption worth
refusing. Verify the ladders, then add the panel.
"""


import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNGS = [2560, 4096, 8192, 16384]
RUNG_LABEL = {2560: "2.5k", 4096: "4k", 8192: "8k", 16384: "16k"}

#: family -> (display, colour token suffix, harness note)
FAMILIES = ["Qwen3.5-4B", "OLMo-3-7B", "Olmo-Hybrid-7B", "Llama-3.2-3B"]


def from_rung_dir(path):
    """Read {rung: metric} from a directory of ``rung_N.json`` grade files.

    :param path: Repo-relative directory holding ``rung_<N>.json``.
    :returns: Mapping of rung to metric value, restricted to :data:`RUNGS`.
    """
    out = {}
    for r in RUNGS:
        f = os.path.join(REPO, path, f"rung_{r}.json")
        if os.path.exists(f):
            out[r] = json.load(open(f))["metric_value"]
    return out


def from_s3_sync(local_dir):
    """Read {rung: metric} from a locally-synced S3 result directory."""
    return from_rung_dir(local_dir)


def llama_iid(arm):
    """Read the ``]]``-truncated Llama contradiction_iid re-score.

    Llama emits no EOS and rambles to the token cap on 87-100% of examples, which destroys
    precision; the raw ladder understates it by roughly a factor of two (dense 0.625 raw vs
    0.970 truncated at 2.5k). These are the truncated numbers.

    :param arm: ``full`` or ``chunked-mix``.
    :returns: Mapping of rung to truncated f1.
    """
    rows = json.load(open(os.path.join(
        REPO, "results/ctc_suite_llama_iid/contradiction/regrade_truncated.json")))
    return {r["rung"]: r["truncated"]["f1"] for r in rows if r["arm"] == arm and r["rung"] in RUNGS}


#: Every series below is on contradiction_iid -- the ladder matching the contradiction_train shard
#: all four families fine-tuned on. That uniformity is the whole point: the first attempt at this
#: figure had Qwen on contradiction_clean and OLMo/Llama on contradiction, three distinct corpora,
#: and would have compared four models on up to three different inputs.
SERIES = {
    "contradiction": {
        ("Qwen3.5-4B", "dense"):   {2560: .9895, 4096: .9843, 8192: .9760, 16384: .9652},
        ("Qwen3.5-4B", "chunked"): {2560: .8613, 4096: .8337, 8192: .8039, 16384: .7586},
        ("OLMo-3-7B", "dense"):    {2560: .9801, 4096: .9772, 8192: .9689, 16384: .9389},
        ("OLMo-3-7B", "chunked"):  {2560: .9719, 4096: .9545, 8192: .8768, 16384: .6612},
        ("Olmo-Hybrid-7B", "dense"):   {2560: .9340, 4096: .9211, 8192: .8869, 16384: .8299},
        # chunked arm deliberately absent -- see the note on the page.
        ("Olmo-Hybrid-7B", "chunked"): {},
        ("Llama-3.2-3B", "dense"):   llama_iid("full"),
        ("Llama-3.2-3B", "chunked"): llama_iid("chunked-mix"),
    },
}

METRIC = {"contradiction": "set F1"}
CLASS = {"contradiction": "O(N&#178;) &mdash; pair finding"}

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
  --f2:#7A5EA8;  /* Olmo-Hybrid - violet */
  --f3:#A8622F;  /* Llama       - ochre */
  --warn:#9A3B2C; --warn-bg:#F6E2DE;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --paper:#14171C; --surface:#1A1E24; --surface-2:#21252C;
    --ink:#E9EAED; --ink-soft:#B7BBC3; --neutral:#8B919B;
    --rule:#2C313A; --rule-soft:#242932;
    --accent:#83B5D7;
    --f0:#7FB2D8; --f1:#63C0A9; --f2:#A98FD6; --f3:#DE9A61;
    --warn:#E08B7B; --warn-bg:#2E1E1B;
  }}
}}
:root[data-theme="dark"]{{
  --paper:#14171C; --surface:#1A1E24; --surface-2:#21252C;
  --ink:#E9EAED; --ink-soft:#B7BBC3; --neutral:#8B919B;
  --rule:#2C313A; --rule-soft:#242932;
  --accent:#83B5D7;
  --f0:#7FB2D8; --f1:#63C0A9; --f2:#A98FD6; --f3:#DE9A61;
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
.swatch.f-0{{background:var(--f0)}} .swatch.f-1{{background:var(--f1)}} .swatch.f-2{{background:var(--f2)}} .swatch.f-3{{background:var(--f3)}}
.lg .key{{width:26px;height:0;border-top:2.5px solid var(--neutral)}}
.lg .key.dash{{border-top-style:dashed}}
.panels{{display:grid;grid-template-columns:minmax(0,1fr);gap:26px;max-width:620px}}
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
.f-0{{stroke:var(--f0)}} .f-1{{stroke:var(--f1)}} .f-2{{stroke:var(--f2)}} .f-3{{stroke:var(--f3)}}
circle.pt.f-0,rect.pt.f-0{{fill:var(--f0);stroke:none}}
circle.pt.f-1,rect.pt.f-1{{fill:var(--f1);stroke:none}}
circle.pt.f-2,rect.pt.f-2{{fill:var(--f2);stroke:none}}
circle.pt.f-3,rect.pt.f-3{{fill:var(--f3);stroke:none}}
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
  <p class="standfirst">Four model families fine-tuned on the same data and scored on the same ladder. Dense attention is nearly family-invariant here &mdash; every backbone lands between 0.87 and 0.97 at 16k. What the backbone changes is how <em>chunked</em> attention degrades with length, and it changes it a lot. Solid lines are dense; dashed are chunked-mix.</p>
  <div class="legend">
    <span class="lg"><span class="swatch f-0"></span>Qwen3.5-4B <span style="color:var(--neutral)">3:1 GDN</span></span>
    <span class="lg"><span class="swatch f-1"></span>OLMo-3-7B <span style="color:var(--neutral)">3:1 sliding</span></span>
    <span class="lg"><span class="swatch f-2"></span>Olmo-Hybrid-7B <span style="color:var(--neutral)">3:1 linear</span></span>
    <span class="lg"><span class="swatch f-3"></span>Llama-3.2-3B <span style="color:var(--neutral)">dense</span></span>
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

</section>

<section>
  <p class="eyebrow">reading it</p>
  <p class="note"><strong>The chunked curves cross; the dense ones do not.</strong> OLMo-3's chunked arm starts 0.11 above Qwen's at 2.5k (0.972 vs 0.861) and finishes 0.10 below it at 16k (0.661 vs 0.759), while all four dense arms sit inside a 0.13 band the whole way. So the backbone is close to irrelevant for dense attention on this task and decisive for how chunked attention decays: the Gated-DeltaNet hybrid starts worse and degrades gently, the sliding-window hybrid starts near-dense and falls off a cliff. A single "chunked gap" per family would erase exactly the thing worth reporting.</p>
  <p class="note">Llama-3.2-3B is the smallest model here at 3B and its dense arm is competitive (0.875 at 16k against Qwen-4B's 0.965), but its chunked arm is the weakest at every rung. Read alongside the model-scale ladder, where the chunked gap narrows monotonically with size, that fits: chunked attention is where capacity gets spent.</p>
  <p class="note warn"><strong>Olmo-Hybrid has no chunked line, and that is deliberate.</strong> It scored 0.113 / 0.062 / 0.021 / 0.006, but its chunked arm reached a final training CE of 0.958 where OLMo-3's reached 0.171 on identical task, data and 1109 steps &mdash; it never fit the training data. The same model's <em>qdmatch</em> chunked arm trained to a healthy CE of 0.156 and scores 0.931 at 2k, so the linear-attention backbone handles a chunk mask perfectly well. Plotting that series would read as "GDN backbones cannot do chunked pair-finding", which is not what was measured. It needs a longer or re-tuned run.</p>
  <p class="note warn"><strong>A harness split remains.</strong> Qwen's series comes from the vLLM harness; OLMo-3, Olmo-Hybrid and Llama from the native olmo-core evaluator. Where the two overlap elsewhere they agree to four decimals, but there is no overlapping contradiction point, so the Qwen line carries an unquantified harness difference. Native Qwen contradiction re-runs would close it.</p>
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
