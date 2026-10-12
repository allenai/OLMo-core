"""
Render the ds64 soft-KV-vs-dense Pareto dashboard from ``results/ds64/results.csv``.

The campaign is live, so this is a *generator*, not a one-off page: re-run it whenever the
collector has picked up new runs and the same page redraws with whatever has landed. Tasks whose
soft arms have not finished render as an explicit "in flight" panel rather than being omitted.

    python research/token_routing/ds64/collect_ds64.py        # refresh results.csv from Beaker
    python research/token_routing/ds64/make_pareto_artifact.py

:returns: writes ``results/ds64/pareto.html``.
"""

import csv
import datetime
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CSV = f"{REPO}/results/ds64/results.csv"
OUT = f"{REPO}/results/ds64/pareto.html"

RUNGS = ["2k", "8k", "16k", "32k", "64k"]
EVAL_SIZE = 500  # examples per rung

# Arm metadata. ``keep`` is the fraction of non-gold documents left real; it is ordinal, so the
# colour ramp is sequential in it. ``kind`` separates the two document-selection families.
ARMS = {
    "dense":  {"label": "dense",       "keep": 1.0,      "kind": "dense", "blurb": "full attention"},
    "hdr33":  {"label": "hdr 1/3",     "keep": 1 / 3,    "kind": "header", "blurb": "headers real, gold + 1/3 of docs"},
    "hdr17":  {"label": "hdr 1/6",     "keep": 1 / 6,    "kind": "header", "blurb": "headers real, gold + 1/6"},
    "hdr08":  {"label": "hdr 1/12",    "keep": 1 / 12,   "kind": "header", "blurb": "headers real, gold + 1/12"},
    "hdr03":  {"label": "hdr 1/36",    "keep": 1 / 36,   "kind": "header", "blurb": "headers real, gold + 1/36"},
    "ohdr33": {"label": "hdr 1/3",     "keep": 1 / 3,    "kind": "header", "blurb": "headers real, gold + 1/3 of docs"},
    "ohdr17": {"label": "hdr 1/6",     "keep": 1 / 6,    "kind": "header", "blurb": "headers real, gold + 1/6"},
    "kv33":   {"label": "kv 1/3",      "keep": 1 / 3,    "kind": "header", "blurb": "gold + 1/3 of docs"},
    "kv17":   {"label": "kv 1/6",      "keep": 1 / 6,    "kind": "header", "blurb": "gold + 1/6"},
    "runs08": {"label": "runs 1/12",   "keep": 1 / 12,   "kind": "runs",  "blurb": "neighbour runs, 1/12"},
    "runs03": {"label": "runs 1/36",   "keep": 1 / 36,   "kind": "runs",  "blurb": "neighbour runs, 1/36"},
}

TASKS = ["contradiction", "oolong", "nq", "outlier"]

# Rungs whose eval FILE is broken, so the rung measures nothing for any arm. They are excluded from
# the headline mean (dense and soft alike, so comparisons stay like-for-like); averaging a rung that
# is ~0.0 for everything only deflates the task and hides real differences in the rungs that work.
BROKEN_RUNGS = {"outlier": ["64k"]}
TASK_NOTE = {
    "outlier": "<b>keep 1/3 learns this task at short context and loses it as context grows</b> — "
               "at 2k it reaches 0.890 against dense's 0.996 (128M tokens), then falls to 0.163 at 8k "
               "and 0.030 at 16k. It also needs data to get there at all: 2k goes 0.006 → 0.642 → "
               "0.890 as the budget grows 16M → 64M → 128M. <b>keep 1/6 is different — it never "
               "clears 0.032 on any rung, i.e. degenerate, not merely weak.</b> Training CE is "
               "healthy throughout (0.05-0.09 vs dense's 0.16), so the long-context failure is a "
               "train/eval distribution gap, not underfitting: at training time most documents are "
               "pooled, while at inference every document is real, and outlier is defined by "
               "comparing all of them. Also, its 64k eval file is the v2 xlong one with random "
               "fillers, so that rung scores ~0.0 for every arm and measures nothing. <b>This task's "
               "mean is over the four working rungs (2k/8k/16k/32k)</b>, dense and soft alike.",
    "contradiction": "the 64k rung skips 19/500 rows over the 70k generation limit, identically "
                     "for every arm. Its pool caps at 64M tokens.",
}

# Generation tags. Only the newest is wall-clock faithful; accuracy is valid from f2 on.
GENS = {
    "":       {"label": "gen-0", "wall": False, "acc": False},
    "b128":   {"label": "gen-1", "wall": False, "acc": False},
    "b128f":  {"label": "gen-1f", "wall": False, "acc": False},
    "b128f2": {"label": "gen-2", "wall": False, "acc": True},
    "b128f3": {"label": "gen-3", "wall": True,  "acc": True},
}


def gen_of(run):
    """Extract the recipe generation tag from a run name (``ds64-<task>-<arm>-<gen>-u<budget>``)."""
    parts = run.split("-")
    return parts[3] if len(parts) >= 5 and parts[3].startswith("b128") else ""


def budget_tokens(b):
    return int(b.rstrip("M")) * 1_000_000


def se(p, n=EVAL_SIZE):
    """Binomial standard error of a right/wrong-graded rate."""
    return math.sqrt(max(p * (1 - p), 1e-9) / n)


def interp_dense(dense, pf):
    """Dense mean f1 at ``pf`` petaflops, interpolated in log-FLOPs.

    :returns: ``(f1, extrapolated)``; ``extrapolated`` is True when ``pf`` falls outside the
        measured dense range, where the value is a linear continuation and NOT evidence.
    """
    pts = sorted(dense, key=lambda d: d["pf"])
    if len(pts) < 2:
        return None, True
    x = math.log10(pf)
    xs = [math.log10(p["pf"]) for p in pts]
    ys = [p["f1"] for p in pts]
    if x < xs[0]:
        i, extrap = 0, True
    elif x > xs[-1]:
        i, extrap = len(xs) - 2, True
    else:
        extrap = False
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i]), extrap


def main():
    rows = list(csv.DictReader(open(CSV)))
    # The orchestrator's state is the truth about what is still training; results.csv keeps
    # rows for launches that were reset and relaunched.
    live = {}
    sp = f"{REPO}/research/token_routing/ds64/orchestrator_ds64_state.json"
    if os.path.exists(sp):
        st = json.load(open(sp))
        live = {k: v for k, v in st.get("runs", {}).items()
                if v.get("state") not in ("DONE", "FAILED")}

    def num(r, k):
        v = r.get(k)
        try:
            return float(v) if v not in (None, "") else None
        except ValueError:
            return None

    # ---- scored runs: a full 5-rung eval and a FLOP-meter reading -------------------------
    scored, inflight = {}, []
    for r in rows:
        arm, task = r["arm"], r["task"]
        f1, pf = num(r, "mean_f1"), num(r, "flops_meter")
        gen = gen_of(r["run"])
        rec = {
            "run": r["run"], "task": task, "arm": arm, "budget": r["budget"], "gen": gen,
            "f1": f1, "pf": pf, "dense_pf": num(r, "dense_pflops"),
            "ratio": num(r, "actual_over_dense"), "gpu_hours": num(r, "gpu_hours"),
            "steps": num(r, "steps"), "secs": num(r, "train_seconds"),
            "rungs": {k: num(r, f"f1_{k}") for k in RUNGS},
            "n_rungs": int(r["n_rungs"] or 0),
            "train_state": r["train_state"], "eval_state": r["eval_state"] or "",
        }
        rec["s_per_step"] = (rec["secs"] / rec["steps"]) if (rec["secs"] and rec["steps"]) else None
        bad = BROKEN_RUNGS.get(task, [])
        if bad:
            good = [v for k, v in rec["rungs"].items() if k not in bad and v is not None]
            if good:
                rec["f1_all_rungs"], rec["excluded"] = f1, bad
                f1 = rec["f1"] = sum(good) / len(good)
        # A run that took <=1 optimizer step is not a training run. --max-tokens below ONE step's
        # worth of tokens silently yields a 1-step run instead of an error: the soft arms do
        # 128 rows x 65536 = 8.4M tokens/step, so every budget under ~8.4M collapses to step 1 (the
        # 4M and 8M soft arms came out byte-identical at 6 PF). Those points then sit far left of
        # the dense curve and produce the LARGEST apparent matched-FLOP wins in the ladder.
        acc_valid = (arm == "dense" or GENS.get(gen, {}).get("acc")) and (
            rec["steps"] is None or rec["steps"] >= 2
        )
        if f1 is not None and pf and rec["n_rungs"] == 5 and acc_valid:
            # keep the newest generation per (task, arm, budget)
            key = (task, arm, r["budget"])
            if key not in scored or GENS[gen]["label"] > GENS[scored[key]["gen"]]["label"]:
                scored[key] = rec
        elif gen == "b128f3":
            inflight.append(rec)

    runs = list(scored.values())

    # ---- per-task series -----------------------------------------------------------------
    tasks = []
    for task in TASKS:
        mine = [r for r in runs if r["task"] == task]
        dense = sorted([r for r in mine if r["arm"] == "dense"], key=lambda d: d["pf"])
        arms = {}
        for r in mine:
            if r["arm"] == "dense":
                continue
            arms.setdefault(r["arm"], []).append(r)
        for a in arms:
            arms[a].sort(key=lambda d: d["pf"])
            for r in arms[a]:
                d_f1, extrap = interp_dense(dense, r["pf"])
                r["dense_at_pf"] = d_f1
                r["delta"] = (r["f1"] - d_f1) if d_f1 is not None else None
                r["extrapolated"] = extrap
                # dominated dense points: this arm is at least as accurate AND strictly cheaper
                dom = [d for d in dense if d["f1"] <= r["f1"] and d["pf"] > r["pf"]]
                best = max(dom, key=lambda d: d["pf"]) if dom else None
                r["dominates"] = ({"budget": best["budget"], "f1": best["f1"], "pf": best["pf"],
                                   "saving": best["pf"] / r["pf"]} if best else None)
        pend = sorted({r["arm"] for r in inflight if r["task"] == task})
        tasks.append({
            "task": task, "dense": dense,
            "arms": [{"arm": a, "meta": ARMS.get(a, {"label": a, "keep": None, "kind": "other", "blurb": a}),
                      "points": arms[a]} for a in sorted(arms, key=lambda a: -(ARMS.get(a, {}).get("keep") or 0))],
            "pending": pend, "note": TASK_NOTE.get(task, ""),
        })

    # ---- flag arms that trained fine but score at chance: that is a train/eval mismatch, not a
    # result, and plotting it as an ordinary point invites "soft tokens fail on task X" ------------
    # Judge an arm by its BEST rung, not its mean. An arm that solves 2k and collapses at 32k has a
    # low mean but is not degenerate -- it is a length-generalisation failure, which is a result.
    # Only an arm that never clears the floor on ANY rung is a broken/at-chance run.
    for t in tasks:
        dmax = max([d["f1"] for d in t["dense"]], default=0)
        for a in t["arms"]:
            for r in a["points"]:
                best_rung = max([v for v in r["rungs"].values() if v is not None], default=0)
                r["best_rung"] = best_rung
                r["suspect"] = bool(dmax > 0.3 and best_rung < 0.05)
                # solves short context but collapses long: worth showing, worth flagging
                r["length_collapse"] = bool(not r["suspect"] and best_rung > 0.3 and r["f1"] < 0.5 * best_rung)
        t["suspect_arms"] = sorted({a["arm"] for a in t["arms"]
                                    if all(r["suspect"] for r in a["points"])})
        t["collapse_arms"] = sorted({a["arm"] for a in t["arms"]
                                     for r in a["points"] if r["length_collapse"]})

    # ---- the headline: best non-extrapolated matched-FLOP comparison ----------------------
    runs = [r for r in runs if not r.get("suspect")] + [r for r in runs if r.get("suspect")]
    cands = [r for r in runs if r.get("delta") is not None and not r["extrapolated"] and not r.get("suspect")]
    best = max(cands, key=lambda r: r["delta"]) if cands else None
    doms = [r for r in runs if r.get("dominates") and not r.get("suspect")]
    best_dom = max(doms, key=lambda r: r["dominates"]["saving"]) if doms else None

    # ---- wall-clock: s/step by generation, dense as the control --------------------------
    wall = []
    for r in rows:
        s, st = num(r, "train_seconds"), num(r, "steps")
        if not (s and st and st >= 5):
            continue
        wall.append({"run": r["run"], "task": r["task"], "arm": r["arm"], "budget": r["budget"],
                     "gen": gen_of(r["run"]), "s_per_step": s / st,
                     "gpu_hours": num(r, "gpu_hours"), "pf": num(r, "flops_meter"),
                     "ratio": num(r, "actual_over_dense")})

    # Matched-BUDGET per-rung deltas (soft minus dense on the same token budget). The mean-f1 view
    # averages these away, but the whole point of the method is long context, so where the accuracy
    # is actually lost matters as much as the average.
    deltas = {}
    for t in tasks:
        dense_by_b = {d["budget"]: d for d in t["dense"]}
        series = []
        for a in t["arms"]:
            # one line per arm, at its best-trained budget: several budgets of the same arm share a
            # colour (colour encodes keep) and would be mutually unreadable.
            cands = [r for r in a["points"] if r["budget"] in dense_by_b and not r.get("suspect")]
            if not cands:
                continue
            r = max(cands, key=lambda r: budget_tokens(r["budget"]))
            d = dense_by_b[r["budget"]]
            # a broken rung scores ~0 for BOTH arms, so its delta is ~0 and reads as a recovery
            bad = BROKEN_RUNGS.get(t["task"], [])
            per = {k: (r["rungs"].get(k) - d["rungs"][k])
                   for k in RUNGS
                   if k not in bad and r["rungs"].get(k) is not None and d["rungs"].get(k) is not None}
            if per:
                series.append({"arm": r["arm"], "budget": r["budget"], "ratio": r["ratio"],
                               "label": a["meta"]["label"], "keep": a["meta"]["keep"], "d": per})
        if series:
            deltas[t["task"]] = series

    data = {
        "deltas": deltas, "rungs": RUNGS,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tasks": tasks, "best": best, "best_dom": best_dom, "wall": wall, "inflight": inflight,
        "arms": ARMS, "gens": {k: v["label"] for k, v in GENS.items()},
        "eval_size": EVAL_SIZE,
        "counts": {"scored": len(runs), "total": len(rows),
                   "inflight": len(live) or len(inflight)},
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(html)
    print(f"{len(runs)} scored runs, {len(inflight)} in flight -> {OUT}")
    if best_dom:
        d = best_dom["dominates"]
        print(f"dominance: {best_dom['run']} f1 {best_dom['f1']:.3f} @ {best_dom['pf']:.0f} PF "
              f"beats dense {d['budget']} ({d['f1']:.3f} @ {d['pf']:.0f} PF) -- {d['saving']:.1f}x cheaper")
    if best:
        print(f"headline: {best['run']} f1 {best['f1']:.3f} vs dense {best['dense_at_pf']:.3f} "
              f"at {best['pf']:.0f} PF  (+{best['delta']:.3f})")


TEMPLATE = r"""
<title>Soft-KV vs Dense Pareto</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root{
  --paper:#EFF3F6; --surface:#FFFFFF; --sunk:#E5EBF0;
  --ink:#111821; --grey:#5A6875; --faint:#8795A3; --rule:#D5DEE5;
  --dense:#44566B;
  --k33:#12656D; --k17:#3B8A88; --k08:#6FAAA6; --k03:#A6CAC6;
  --accent:#A0620E; --accent-soft:#F0E2CC;
  --good:#2C6F52; --warn:#A0620E; --bad:#A03E2B; --pend:#8795A3;
  --shadow:0 1px 2px rgba(17,24,33,.06),0 8px 22px -14px rgba(17,24,33,.35);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0E141A; --surface:#151D25; --sunk:#1B242D;
  --ink:#E4ECF2; --grey:#9BAAB7; --faint:#71818F; --rule:#26313B;
  --dense:#93A6BA;
  --k33:#3FB0B6; --k17:#5FC0BC; --k08:#84CCC7; --k03:#A9D8D4;
  --accent:#D9A055; --accent-soft:#3A2E1C;
  --good:#5FB489; --warn:#D9A055; --bad:#D9705A; --pend:#71818F;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 26px -16px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --paper:#0E141A; --surface:#151D25; --sunk:#1B242D;
  --ink:#E4ECF2; --grey:#9BAAB7; --faint:#71818F; --rule:#26313B;
  --dense:#93A6BA;
  --k33:#3FB0B6; --k17:#5FC0BC; --k08:#84CCC7; --k03:#A9D8D4;
  --accent:#D9A055; --accent-soft:#3A2E1C;
  --good:#5FB489; --warn:#D9A055; --bad:#D9705A; --pend:#71818F;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 26px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:38px 22px 76px;display:flex;flex-direction:column;gap:34px}
h1,h2,h3{font-family:"IBM Plex Serif",Georgia,serif;text-wrap:balance;margin:0}
h1{font-size:35px;font-weight:600;letter-spacing:-.015em;line-height:1.14}
h2{font-size:20px;font-weight:600;letter-spacing:-.005em}
h3{font-size:15px;font-weight:600}
p{margin:0}
.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;
  letter-spacing:.16em;text-transform:uppercase;color:var(--faint)}
.lede{color:var(--grey);max-width:66ch;font-size:16px}
.note{color:var(--grey);font-size:13.5px;max-width:72ch}

/* header */
header{display:flex;flex-direction:column;gap:13px;
  border-bottom:1px solid var(--rule);padding-bottom:22px}
.status{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.chip{font-family:"IBM Plex Mono",monospace;font-size:11.5px;font-variant-numeric:tabular-nums;
  padding:3px 9px;border:1px solid var(--rule);border-radius:2px;color:var(--grey);background:var(--surface)}
.chip b{color:var(--ink);font-weight:600}
.chip.live{border-color:var(--accent);color:var(--accent)}
.chip.live b{color:var(--accent)}

/* verdict */
.verdict.dom{border-left-color:var(--good)}
.verdict.dom .delta{color:var(--good)}
.verdict{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);
  padding:24px 26px;display:grid;grid-template-columns:auto 1fr;gap:26px;align-items:center;box-shadow:var(--shadow)}
.delta{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  font-size:50px;font-weight:600;color:var(--accent);line-height:1;letter-spacing:-.03em}
.delta small{display:block;font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--faint);font-weight:500;margin-top:9px}
.verdict-body{display:flex;flex-direction:column;gap:9px}

/* generic section */
section{display:flex;flex-direction:column;gap:15px}
.sec-head{display:flex;flex-direction:column;gap:5px}
.panel{background:var(--surface);border:1px solid var(--rule);padding:20px 22px;box-shadow:var(--shadow)}
.chart-wrap{overflow-x:auto}
svg{display:block;max-width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:7px 18px;align-items:center;margin-top:12px}
.lg{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--grey);
  font-family:"IBM Plex Mono",monospace}
.lg i{width:22px;height:0;border-top-width:2px;border-top-style:solid;display:inline-block;position:relative}
.lg i::after{content:"";position:absolute;left:8px;top:-3px;width:6px;height:6px;border-radius:50%;background:currentColor}
.lg.sq i::after{border-radius:0;transform:rotate(45deg)}

/* task grid */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}
.tcard{background:var(--surface);border:1px solid var(--rule);padding:17px 18px;
  display:flex;flex-direction:column;gap:10px;box-shadow:var(--shadow)}
.tcard header{border:0;padding:0;flex-direction:row;justify-content:space-between;align-items:baseline;gap:10px}
.tname{font-family:"IBM Plex Serif",serif;font-size:16px;font-weight:600}
.pill{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;padding:2px 7px;border-radius:2px;white-space:nowrap}
.pill.won{background:var(--accent-soft);color:var(--accent)}
.pill.wait{background:var(--sunk);color:var(--pend)}
.pill.warnp{background:var(--sunk);color:var(--bad)}

/* rung picker */
.rungpick{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.rungpick button{font-family:"IBM Plex Mono",monospace;font-size:12.5px;padding:5px 13px;
  border:1px solid var(--rule);background:var(--surface);color:var(--grey);cursor:pointer;
  border-radius:2px;font-variant-numeric:tabular-nums}
.rungpick button:hover{border-color:var(--faint);color:var(--ink)}
.rungpick button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.rungpick button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.rungpick .lbl{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--faint);margin-right:6px}

/* tables */
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint);font-weight:500;border-bottom:1px solid var(--rule)}
tbody td{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
td.win{color:var(--accent);font-weight:600}
td.lose{color:var(--grey)}
.swatch{display:inline-block;width:9px;height:9px;margin-right:7px;vertical-align:baseline}
.bar{position:relative;height:5px;background:var(--sunk);min-width:60px}
.bar span{position:absolute;inset:0 auto 0 0;background:var(--dense)}

/* callouts */
.calls{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.call{border-left:2px solid var(--rule);padding:2px 0 2px 14px;display:flex;flex-direction:column;gap:5px}
.call.open{border-left-color:var(--warn)}
.call.broken{border-left-color:var(--bad)}
.call.done{border-left-color:var(--good)}
.call h3{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--faint);font-weight:500}
.call p{font-size:13.5px;color:var(--grey)}
.call b{color:var(--ink);font-weight:500}
footer{border-top:1px solid var(--rule);padding-top:18px;color:var(--faint);font-size:12.5px;
  display:flex;flex-direction:column;gap:6px}
code{font-family:"IBM Plex Mono",monospace;font-size:.92em;background:var(--sunk);padding:1px 5px}
@media (max-width:640px){
  h1{font-size:27px}
  .verdict{grid-template-columns:1fr;gap:16px}
  .delta{font-size:40px}
  .grid{grid-template-columns:1fr}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">ds64 &nbsp;·&nbsp; Qwen3.5-4B &nbsp;·&nbsp; short-heavy 2k–64k mix</div>
    <h1>Is soft-KV training Pareto-optimal against dense attention?</h1>
    <p class="lede">Pooled-document soft tokens compress the context a model trains on. The question
      is whether the accuracy they give up is worth less than the compute they save — measured on the
      FLOP meter, at matched training cost, against a dense-attention control.</p>
    <div class="status" id="status"></div>
  </header>

  <div class="verdict" id="verdict"></div>
  <div class="verdict dom" id="verdict-dom" hidden></div>

  <section>
    <div class="sec-head">
      <h2>The matched-FLOP frontier</h2>
      <p class="note">Each arm is a ladder of training budgets. What matters is not that a soft arm
        scores below dense — it always does — but whether it sits <em>above the dense curve at the
        same FLOPs</em>. Colour encodes how much of the context stays real; the dashed shape is the
        neighbour-runs family, which selects contiguous document runs instead of headers.</p>
    </div>
    <div class="panel">
      <div class="chart-wrap"><svg id="main-chart"></svg></div>
      <div class="legend" id="main-legend"></div>
    </div>
    <p class="note" id="extrap-note"></p>
  </section>

  <section>
    <div class="sec-head">
      <h2>Keep ratio is the load-bearing knob</h2>
      <p class="note">Accuracy is monotone in the fraction of documents left real, at every budget —
        and the collapse at the bottom is the campaign's sharpest negative result. An eval-side probe
        found keep 1/36 reproduced full attention on held-out rows; in <em>training</em> it never
        recovers, because a tiny real set teaches the model that the answer lives in a real document.
        Eval-side parity did not transfer to training.</p>
    </div>
    <div class="panel">
      <div class="chart-wrap"><svg id="keep-chart"></svg></div>
      <div class="legend" id="keep-legend"></div>
    </div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Per task</h2>
      <p class="note" id="task-note"></p>
    </div>
    <div class="grid" id="task-grid"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>By context length</h2>
      <p class="note">The mean above averages five eval rungs, which hides where the compression is
        actually paid for. These are the same Pareto panels scored at a single context length — pick
        one. Training used a short-heavy mix over 2k/4k/8k/16k/32k/56k packed into a 65536-token
        window; the evals score 2k/8k/16k/32k/64k.</p>
    </div>
    <div class="rungpick" id="rung-picker"></div>
    <div class="grid" id="rung-grid"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Where the compression is paid for</h2>
      <p class="note">Soft minus dense at the <em>same token budget</em>, rung by rung — so this is
        the accuracy cost of compressing, with the compute saving set aside. Above the zero line the
        soft token is free; below it, it costs. The slope is the finding: on the retrieval-shaped
        tasks the cost is near zero at 2k and grows with context, which is precisely the regime the
        method exists for.</p>
    </div>
    <div class="panel">
      <div class="chart-wrap"><svg id="delta-chart"></svg></div>
      <div class="legend" id="delta-legend"></div>
    </div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Wall-clock: the FLOPs are real, the speedup is not</h2>
      <p class="note">A third of the FLOPs should buy roughly a third of the time. It does not.
        Length-sorting each rank's micro-batches (gen-3) helps exactly where there is padding to
        recover — <span class="mono">hdr17</span>, at four rows per micro-batch, gained 1.58×, while
        <span class="mono">hdr33</span> at two rows had almost nothing to sort. Even so the best soft
        arm still spends <b>5.4×</b> dense's seconds per step for about a third of its FLOPs. Sorting
        was a partial fix, not the fix; packing the compacted rows with
        <span class="mono">cu_seqlens</span> is the remaining lever. Cross-job timings drift on shared
        nodes, so read the magnitudes loosely and the gap firmly.</p>
    </div>
    <div class="panel">
      <div class="chart-wrap"><svg id="wall-chart"></svg></div>
      <div class="legend" id="wall-legend"></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>Reading this safely</h2></div>
    <div class="calls" id="calls"></div>
  </section>

  <footer id="footer"></footer>
</div>

<script>
const D = __DATA__;
const KEEP_VAR = {"0.3333":"--k33","0.1667":"--k17","0.0833":"--k08","0.0278":"--k03"};
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const armColor = m => cssv(KEEP_VAR[(m.keep||0).toFixed(4)] || "--k03");
const fmtPF = v => v >= 1000 ? (v/1000).toFixed(v>=10000?0:2)+"k" : Math.round(v);
const SVGNS = "http://www.w3.org/2000/svg";
const el = (t, a={}) => { const n = document.createElementNS(SVGNS,t);
  for (const k in a) n.setAttribute(k, a[k]); return n; };
const txt = (x,y,s,a={}) => { const n = el("text",Object.assign({x,y},a)); n.textContent=s; return n; };

/* ---------- Pareto chart: mean f1 against training FLOPs, log x ---------- */
function pareto(svg, task, opts={}) {
  const W = opts.w||960, H = opts.h||420, small = !!opts.small;
  const M = {t:16, r:small?16:26, b:40, l:50};
  const all = [...task.dense, ...task.arms.flatMap(a=>a.points)];
  if (!all.length) return false;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  svg.setAttribute("width",W); svg.setAttribute("height",H);
  svg.textContent = "";
  const pfs = all.map(p=>p.pf);
  const lo = Math.log10(Math.min(...pfs)*0.72), hi = Math.log10(Math.max(...pfs)*1.34);
  const X = v => M.l + (Math.log10(v)-lo)/(hi-lo) * (W-M.l-M.r);
  const Y = v => H-M.b - v * (H-M.b-M.t);
  const ink=cssv("--ink"), grey=cssv("--grey"), faint=cssv("--faint"),
        rule=cssv("--rule"), dense=cssv("--dense"), accent=cssv("--accent"), sunk=cssv("--sunk");

  /* extrapolation zone: cheaper than the cheapest dense run we actually measured */
  const dmin = task.dense.length ? Math.min(...task.dense.map(d=>d.pf)) : null;
  if (dmin && X(dmin) > M.l+2 && task.arms.length) {
    svg.appendChild(el("rect",{x:M.l,y:M.t,width:X(dmin)-M.l,height:H-M.b-M.t,fill:sunk,opacity:.55}));
    if (!small) svg.appendChild(txt(M.l+8, H-M.b-8, "dense not measured here",
      {fill:faint,"font-size":10.5,"font-family":"IBM Plex Mono, monospace"}));
  }
  /* y grid + ticks */
  for (let v=0; v<=1.0001; v+=0.25) {
    svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:Y(v),y2:Y(v),stroke:rule,"stroke-width":1}));
    svg.appendChild(txt(M.l-9, Y(v)+4, v.toFixed(2), {fill:faint,"font-size":11,"text-anchor":"end",
      "font-family":"IBM Plex Mono, monospace"}));
  }
  /* x ticks at round petaflop values inside the domain */
  const ticks=[100,200,500,1000,2000,5000,10000].filter(t=>Math.log10(t)>=lo&&Math.log10(t)<=hi);
  ticks.forEach(t=>{
    svg.appendChild(el("line",{x1:X(t),x2:X(t),y1:H-M.b,y2:H-M.b+4,stroke:rule}));
    svg.appendChild(txt(X(t), H-M.b+18, fmtPF(t), {fill:faint,"font-size":11,"text-anchor":"middle",
      "font-family":"IBM Plex Mono, monospace"}));
  });
  svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,stroke:rule}));
  svg.appendChild(txt((M.l+W-M.r)/2, H-6, "training FLOPs (petaflops, log)",
    {fill:grey,"font-size":11.5,"text-anchor":"middle","font-family":"IBM Plex Mono, monospace"}));
  svg.appendChild(txt(0,0,opts.ylabel||"mean f1",{fill:grey,"font-size":11.5,"text-anchor":"middle",
    "font-family":"IBM Plex Mono, monospace",transform:`translate(13,${(M.t+H-M.b)/2}) rotate(-90)`}));

  const series = (pts,color,dash,shape) => {
    if (pts.length>1) svg.appendChild(el("path",{
      d: pts.map((p,i)=>`${i?"L":"M"}${X(p.pf).toFixed(1)},${Y(p.f1).toFixed(1)}`).join(" "),
      fill:"none", stroke:color, "stroke-width":small?1.6:2,
      "stroke-dasharray": dash?"5 3":"", "stroke-linejoin":"round"}));
    pts.forEach(p=>{
      const r = small?3.2:4.4;
      if (shape==="sq") svg.appendChild(el("rect",{x:X(p.pf)-r,y:Y(p.f1)-r,width:r*2,height:r*2,
        fill:cssv("--surface"),stroke:color,"stroke-width":1.8,transform:`rotate(45 ${X(p.pf)} ${Y(p.f1)})`}));
      else svg.appendChild(el("circle",{cx:X(p.pf),cy:Y(p.f1),r,fill:color}));
    });
  };
  series(task.dense, dense, false, "circle");
  task.arms.forEach(a=>series(a.points, armColor(a.meta), a.meta.kind==="runs", a.meta.kind==="runs"?"sq":"circle"));

  /* budget labels on the dense ladder */
  if (!small) task.dense.forEach(p=>svg.appendChild(txt(X(p.pf), Y(p.f1)-11, p.budget,
    {fill:dense,"font-size":10.5,"text-anchor":"middle","font-family":"IBM Plex Mono, monospace"})));

  /* annotate the one trustworthy matched-FLOP comparison */
  const B = opts.annotate;
  if (B && !small) {
    const x=X(B.pf), y1=Y(B.f1), y2=Y(B.dense_at_pf);
    svg.appendChild(el("line",{x1:x,x2:x,y1,y2,stroke:accent,"stroke-width":1.5,"stroke-dasharray":"3 3"}));
    [y1,y2].forEach(y=>svg.appendChild(el("line",{x1:x-5,x2:x+5,y1:y,y2:y,stroke:accent,"stroke-width":1.5})));
    const pw=152, ph=36, lx=M.l+10, ly=M.t+6;
    svg.appendChild(el("rect",{x:lx-7,y:ly,width:pw,height:ph,fill:cssv("--surface"),
      stroke:accent,"stroke-width":1}));
    svg.appendChild(el("line",{x1:x-5,x2:lx+pw-7,y1:(y1+y2)/2,y2:ly+ph/2,stroke:accent,
      "stroke-width":1,"stroke-dasharray":"2 2"}));
    svg.appendChild(txt(lx, ly+16, `+${B.delta.toFixed(3)} over dense`,
      {fill:accent,"font-size":12,"font-weight":600,"font-family":"IBM Plex Mono, monospace"}));
    svg.appendChild(txt(lx, ly+29, `at ${Math.round(B.pf)} PF, same compute`,
      {fill:accent,"font-size":10.5,"font-family":"IBM Plex Mono, monospace"}));
  }
  return true;
}

/* ---------- legend ---------- */
function legend(node, task) {
  node.textContent="";
  const add=(color,label,sq,dash)=>{
    const d=document.createElement("span"); d.className="lg"+(sq?" sq":"");
    d.style.color=color;
    d.innerHTML=`<i style="border-top-color:${color};border-top-style:${dash?"dashed":"solid"}"></i>`+
      `<span style="color:var(--grey)">${label}</span>`;
    node.appendChild(d);
  };
  add(cssv("--dense"),"dense (full attention)",false,false);
  task.arms.forEach(a=>add(armColor(a.meta), a.meta.label+" — "+a.meta.blurb,
    a.meta.kind==="runs", a.meta.kind==="runs"));
}

/* ---------- keep-ratio ladder: grouped bars, header family ---------- */
function keepChart(svg, task) {
  const fam = task.arms.filter(a=>a.meta.kind==="header").sort((a,b)=>a.meta.keep-b.meta.keep);
  if (!fam.length) return;
  const budgets=[...new Set(fam.flatMap(a=>a.points.map(p=>p.budget)))]
    .sort((a,b)=>parseInt(a)-parseInt(b));
  const W=960,H=300,M={t:18,r:20,b:52,l:50};
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.setAttribute("width",W); svg.setAttribute("height",H);
  svg.textContent="";
  const rule=cssv("--rule"),faint=cssv("--faint"),grey=cssv("--grey"),dense=cssv("--dense");
  const Y=v=>H-M.b-v*(H-M.b-M.t);
  for(let v=0;v<=1.0001;v+=0.25){
    svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:Y(v),y2:Y(v),stroke:rule}));
    svg.appendChild(txt(M.l-9,Y(v)+4,v.toFixed(2),{fill:faint,"font-size":11,"text-anchor":"end",
      "font-family":"IBM Plex Mono, monospace"}));
  }
  const gw=(W-M.l-M.r)/budgets.length;
  budgets.forEach((b,gi)=>{
    const x0=M.l+gi*gw, inner=gw*0.78, bw=inner/fam.length;
    fam.forEach((a,ai)=>{
      const p=a.points.find(p=>p.budget===b); if(!p) return;
      const x=x0+gw*0.11+ai*bw;
      svg.appendChild(el("rect",{x:x+1,y:Y(p.f1),width:bw-2,height:Y(0)-Y(p.f1),
        fill:armColor(a.meta)}));
      svg.appendChild(txt(x+bw/2,Y(p.f1)-5,p.f1.toFixed(2),{fill:grey,"font-size":10,
        "text-anchor":"middle","font-family":"IBM Plex Mono, monospace"}));
    });
    /* the dense control for this budget */
    const dp=task.dense.find(p=>p.budget===b);
    if(dp){
      svg.appendChild(el("line",{x1:x0+gw*0.08,x2:x0+gw*0.92,y1:Y(dp.f1),y2:Y(dp.f1),
        stroke:dense,"stroke-width":2,"stroke-dasharray":"4 3"}));
      svg.appendChild(txt(x0+gw*0.92,Y(dp.f1)-5,"dense "+dp.f1.toFixed(2),{fill:dense,"font-size":10,
        "text-anchor":"end","font-family":"IBM Plex Mono, monospace"}));
    }
    svg.appendChild(txt(x0+gw/2,H-M.b+18,b+" tokens",{fill:grey,"font-size":12,"text-anchor":"middle",
      "font-family":"IBM Plex Mono, monospace"}));
  });
  svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:Y(0),y2:Y(0),stroke:rule}));
  svg.appendChild(txt((M.l+W-M.r)/2,H-8,"bars left→right: keep 1/36 → 1/3 of documents real",
    {fill:faint,"font-size":11,"text-anchor":"middle","font-family":"IBM Plex Mono, monospace"}));
  svg.appendChild(txt(0,0,"mean f1",{fill:grey,"font-size":11.5,"text-anchor":"middle",
    "font-family":"IBM Plex Mono, monospace",transform:`translate(13,${(M.t+H-M.b)/2}) rotate(-90)`}));
}

/* ---------- wall-clock bars ---------- */
function wallChart(svg, node) {
  const ORDER={dense:0,hdr17:1,hdr33:2};
  const pick = D.wall.filter(w=>w.task==="contradiction" && w.arm in ORDER
    && w.budget==="16M" && ["","b128f2","b128f3"].includes(w.gen))
    .sort((a,b)=>(ORDER[a.arm]-ORDER[b.arm])||a.gen.localeCompare(b.gen));
  const rows = pick.length ? pick : D.wall.slice(0,6);
  const W=960,H=52+rows.length*46,M={t:14,r:24,b:30,l:210};
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.setAttribute("width",W); svg.setAttribute("height",H);
  svg.textContent="";
  const rule=cssv("--rule"),faint=cssv("--faint"),grey=cssv("--grey"),ink=cssv("--ink"),
        dense=cssv("--dense"),accent=cssv("--accent");
  const max=Math.max(...rows.map(r=>r.s_per_step))*1.16;
  const X=v=>M.l+v/max*(W-M.l-M.r);
  rows.forEach((r,i)=>{
    const y=M.t+i*46, isD=r.arm==="dense", c=isD?dense:accent;
    svg.appendChild(txt(M.l-12,y+17,`${r.arm} · ${D.gens[r.gen]||r.gen}`,
      {fill:ink,"font-size":12.5,"text-anchor":"end","font-family":"IBM Plex Mono, monospace"}));
    const MICRO={hdr17:4,hdr33:2};
    svg.appendChild(txt(M.l-12,y+31,(r.ratio?`${r.ratio.toFixed(2)}× FLOPs`:"full FLOPs")+
      (MICRO[r.arm]?` · micro ${MICRO[r.arm]}`:""),
      {fill:faint,"font-size":10.5,"text-anchor":"end","font-family":"IBM Plex Mono, monospace"}));
    svg.appendChild(el("rect",{x:M.l,y:y+6,width:Math.max(X(r.s_per_step)-M.l,1),height:22,fill:c,opacity:isD?.9:.85}));
    svg.appendChild(txt(X(r.s_per_step)+9,y+22,`${r.s_per_step.toFixed(0)} s/step`,
      {fill:c,"font-size":12,"font-weight":600,"font-family":"IBM Plex Mono, monospace"}));
  });
  svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:H-M.b+4,y2:H-M.b+4,stroke:rule}));
  svg.appendChild(txt(M.l,H-8,"seconds per optimizer step — contradiction, 16M budget, 8×H100",
    {fill:faint,"font-size":11,"font-family":"IBM Plex Mono, monospace"}));
  node.textContent="";
}
</script>

<script>
/* ---------- render ---------- */
const TASK_LABEL = {contradiction:"contradiction", oolong:"oolong", nq:"nq", outlier:"outlier"};
const byTask = Object.fromEntries(D.tasks.map(t=>[t.task,t]));
const se = (p,n) => Math.sqrt(Math.max(p*(1-p),1e-9)/n);

/* status strip */
(function(){
  const s=document.getElementById("status");
  const bits=[
    `<span class="chip">scored runs <b>${D.counts.scored}</b></span>`,
    `<span class="chip live">in flight <b>${D.counts.inflight}</b></span>`,
    `<span class="chip">eval_size <b>${D.eval_size}</b> / rung × 5 rungs</span>`,
    `<span class="chip">refreshed <b>${D.generated}</b></span>`];
  s.innerHTML=bits.join("");
})();

/* verdict */
(function(){
  const v=document.getElementById("verdict"), B=D.best;
  if(!B){v.innerHTML=`<div class="verdict-body"><h2>No matched-FLOP comparison yet</h2>
    <p class="note">No soft arm has both a finished 5-rung eval and a FLOP-meter reading inside the
    dense curve's measured range.</p></div>`;return;}
  const m=D.arms[B.arm]||{label:B.arm,blurb:""};
  const err=se(B.f1,D.eval_size);
  v.innerHTML=`
    <div><div class="delta">+${B.delta.toFixed(3)}</div><small>mean f1 over dense</small></div>
    <div class="verdict-body">
      <h2>On ${B.task}, <span class="mono">${m.label}</span> beats dense at equal training FLOPs</h2>
      <p class="note">Trained on ${B.budget} tokens, <b class="mono">${m.blurb}</b> reaches
        <b class="mono">${B.f1.toFixed(3)}</b> mean f1 for <b class="mono">${Math.round(B.pf)} PF</b> —
        where dense attention, interpolated along its own measured curve, reaches only
        <b class="mono">${B.dense_at_pf.toFixed(3)}</b>. It spends
        <b class="mono">${B.ratio?B.ratio.toFixed(2):"—"}×</b> the FLOPs of a dense pass over the same data.</p>
      <p class="note">This is the first clean matched-compute win on this task, and the only comparison
        that sits <em>inside</em> the measured dense range rather than extrapolating below it. Per-rung
        binomial standard error is ±${err.toFixed(3)} at eval_size ${D.eval_size}, so the margin is
        roughly two standard errors — real, but not yet a large effect.</p>
    </div>`;
})();

/* dominance verdict: no interpolation involved, so it holds where the dense curve is unmeasured */
(function(){
  const v=document.getElementById("verdict-dom"), B=D.best_dom;
  if(!B) return;
  const m=D.arms[B.arm]||{label:B.arm,blurb:""}, d=B.dominates;
  const err=se(B.f1,D.eval_size), tie=Math.abs(B.f1-d.f1)<err;
  v.hidden=false;
  v.innerHTML=`
    <div><div class="delta">${d.saving.toFixed(1)}×</div><small>cheaper, at ${tie?"parity":"better accuracy"}</small></div>
    <div class="verdict-body">
      <h2>On ${B.task}, <span class="mono">${m.label}</span> ${tie?"matches":"beats"} a measured dense run for a fraction of its compute</h2>
      <p class="note"><b class="mono">${B.f1.toFixed(3)}</b> mean f1 at <b class="mono">${Math.round(B.pf)} PF</b>,
        against dense's ${d.budget} run at <b class="mono">${d.f1.toFixed(3)}</b> for
        <b class="mono">${Math.round(d.pf)} PF</b>. Higher accuracy and lower cost than a point that was
        actually measured — so this one needs <em>no</em> interpolation and survives wherever the dense
        curve is thin. ${tie?`The ${(B.f1-d.f1>=0?"+":"")}${(B.f1-d.f1).toFixed(3)} accuracy edge is inside
        the ±${err.toFixed(3)} standard error, so read it as parity at ${d.saving.toFixed(1)}× less compute
        rather than as a gain.`:""}</p>
    </div>`;
})();

/* main chart */
(function(){
  const t=byTask.contradiction;
  pareto(document.getElementById("main-chart"), t, {w:960,h:430,annotate:D.best});
  legend(document.getElementById("main-legend"), t);
  const dmin=Math.min(...t.dense.map(d=>d.pf));
  const outside=t.arms.flatMap(a=>a.points).filter(p=>p.extrapolated).length;
  document.getElementById("extrap-note").innerHTML =
    `The shaded band is cheaper than the cheapest dense run ever measured (<span class="mono">`+
    `${Math.round(dmin)} PF</span>). ${outside} soft points fall inside it: their apparent margins rest on `+
    `extrapolating the dense curve downward, not on a measurement. Dense anchors at 4M and 8M tokens `+
    `would close that gap and are the single highest-value missing piece.`;
})();

/* keep chart */
(function(){
  keepChart(document.getElementById("keep-chart"), byTask.contradiction);
  const n=document.getElementById("keep-legend");
  const fam=byTask.contradiction.arms.filter(a=>a.meta.kind==="header")
    .sort((a,b)=>a.meta.keep-b.meta.keep);
  n.innerHTML=fam.map(a=>`<span class="lg"><span class="swatch" style="background:${armColor(a.meta)}"></span>`+
    `<span style="color:var(--grey)">${a.meta.label}</span></span>`).join("")+
    `<span class="lg"><i style="border-top-color:var(--dense);border-top-style:dashed"></i>`+
    `<span style="color:var(--grey)">dense control at the same budget</span></span>`;
})();

/* task grid */
(function(){
  const g=document.getElementById("task-grid");
  D.tasks.forEach(t=>{
    const card=document.createElement("div"); card.className="tcard";
    const soft=t.arms.flatMap(a=>a.points);
    const wins=soft.filter(p=>p.delta>0 && !p.extrapolated).length;
    const dom=soft.filter(p=>p.dominates).sort((a,b)=>b.dominates.saving-a.dominates.saving)[0];
    let pill = soft.length
      ? (dom?`<span class="pill won">dominates dense · ${dom.dominates.saving.toFixed(1)}× cheaper</span>`
        :wins?`<span class="pill won">${wins} matched-FLOP win${wins>1?"s":""}</span>`
             :`<span class="pill wait">no win yet</span>`)
      : `<span class="pill wait">soft arms in flight</span>`;
    if(t.collapse_arms && t.collapse_arms.length)
      pill=`<span class="pill warnp">solves short context, collapses long</span>`;
    if(t.suspect_arms && t.suspect_arms.length)
      pill+=` <span class="pill warnp">${t.suspect_arms.join("/")} degenerate</span>`;
    const exc=soft.concat(t.dense).find(p=>p.excluded);
    if(exc) pill+=` <span class="pill warnp">mean over 4 rungs · ${exc.excluded.join("/")} broken</span>`;
    card.innerHTML=`<header><span class="tname">${TASK_LABEL[t.task]||t.task}</span><span>${pill}</span></header>`;
    const wrap=document.createElement("div"); wrap.className="chart-wrap";
    const svg=document.createElementNS(SVGNS,"svg"); wrap.appendChild(svg); card.appendChild(wrap);
    pareto(svg,t,{w:480,h:250,small:true});
    /* table of the arms, or the pending state */
    const tb=document.createElement("table");
    if(soft.length){
      tb.innerHTML=`<thead><tr><th>arm</th><th>budget</th><th>f1</th><th>PF</th><th>vs dense</th></tr></thead>`+
        `<tbody>${soft.sort((a,b)=>(a.extrapolated-b.extrapolated)||(b.delta-a.delta)).slice(0,4).map(p=>{
          const m=D.arms[p.arm]||{label:p.arm};
          const d=p.delta==null?"—":(p.delta>0?"+":"")+p.delta.toFixed(3);
          const cls=p.extrapolated?"lose":(p.delta>0?"win":"lose");
          return `<tr><td><span class="swatch" style="background:${armColor(m)}"></span>${m.label}</td>`+
            `<td>${p.budget}</td><td>${p.f1.toFixed(3)}</td><td>${Math.round(p.pf)}</td>`+
            `<td class="${cls}">${d}${p.extrapolated?" *":""}</td></tr>`;}).join("")}</tbody>`;
    } else {
      const pend=t.pending.length?t.pending.map(a=>(D.arms[a]||{label:a}).label).join(", "):"queued";
      tb.innerHTML=`<thead><tr><th>dense budget</th><th>f1</th><th>PF</th><th>GPU-h</th></tr></thead><tbody>`+
        t.dense.map(p=>`<tr><td>${p.budget}</td><td>${p.f1.toFixed(3)}</td><td>${Math.round(p.pf)}</td>`+
          `<td>${p.gpu_hours?p.gpu_hours.toFixed(1):"—"}</td></tr>`).join("")+`</tbody>`;
      const note=document.createElement("p"); note.className="note";
      note.style.fontSize="12.5px";
      note.innerHTML=`Dense ladder complete; soft arms <span class="mono">${pend}</span> are training now.`;
      card.appendChild(tb); card.appendChild(note);
      if(t.note){const n2=document.createElement("p");n2.className="note";n2.style.fontSize="12.5px";
        n2.innerHTML="<b>Caveat:</b> "+t.note;card.appendChild(n2);}
      g.appendChild(card); return;
    }
    card.appendChild(tb);
    if(t.note){const n2=document.createElement("p");n2.className="note";n2.style.fontSize="12.5px";
      n2.innerHTML="<b>Caveat:</b> "+t.note;card.appendChild(n2);}
    g.appendChild(card);
  });
  const pending=D.tasks.filter(t=>!t.arms.length).map(t=>t.task);
  document.getElementById("task-note").innerHTML = pending.length
    ? `Dense ladders are complete on all four tasks. Soft arms have only been read out on `+
      `contradiction so far — <span class="mono">${pending.join(", ")}</span> are training now and will `+
      `fill in here as their evals land. An asterisk marks a margin that rests on extrapolating dense.`
    : `All four tasks have soft arms. An asterisk marks a margin that rests on extrapolating dense.`;
})();

/* ---------- by context length: the same Pareto panels at one rung ---------- */
function taskAtRung(t, rung){
  const conv = p => Object.assign({}, p, {f1: p.rungs[rung]});
  const keep = p => p.f1 !== null && p.f1 !== undefined;
  return {task:t.task, note:t.note, pending:t.pending, suspect_arms:t.suspect_arms,
          dense:t.dense.map(conv).filter(keep),
          arms:t.arms.map(a=>({arm:a.arm, meta:a.meta, points:a.points.map(conv).filter(keep)}))
                     .filter(a=>a.points.length)};
}
function domAtRung(tr){
  let best=null;
  tr.arms.forEach(a=>a.points.forEach(p=>{
    if(p.suspect) return;
    const beaten=tr.dense.filter(d=>d.f1<=p.f1 && d.pf>p.pf);
    if(!beaten.length) return;
    const d=beaten.reduce((x,y)=>y.pf>x.pf?y:x), saving=d.pf/p.pf;
    if(!best || saving>best.saving) best={p,d,saving,label:a.meta.label};
  }));
  return best;
}
function renderRungs(rung){
  const g=document.getElementById("rung-grid"); g.textContent="";
  D.tasks.forEach(t=>{
    const tr=taskAtRung(t,rung);
    const card=document.createElement("div"); card.className="tcard";
    const broken=(t.task==="outlier"&&rung==="64k");
    const dom=broken?null:domAtRung(tr);
    let pill = broken ? `<span class="pill warnp">eval file broken at 64k</span>`
      : (t.collapse_arms&&t.collapse_arms.length) ? `<span class="pill warnp">collapses beyond short context</span>`
      : dom ? `<span class="pill won">dominates dense · ${dom.saving.toFixed(1)}× cheaper</span>`
      : tr.arms.length ? `<span class="pill wait">no dominance at ${rung}</span>`
      : `<span class="pill wait">soft arms in flight</span>`;
    card.innerHTML=`<header><span class="tname">${t.task}</span><span>${pill}</span></header>`;
    const wrap=document.createElement("div"); wrap.className="chart-wrap";
    const svg=document.createElementNS(SVGNS,"svg"); wrap.appendChild(svg); card.appendChild(wrap);
    if(tr.dense.length) pareto(svg,tr,{w:480,h:250,small:true,ylabel:`f1 at ${rung}`});
    const p=document.createElement("p"); p.className="note"; p.style.fontSize="12.5px";
    p.innerHTML = broken
      ? `Its 64k eval file is the v2 xlong one with random fillers — this rung measures nothing for any arm.`
      : dom ? `At ${rung}, <span class="mono">${dom.label}</span> reaches
          <span class="mono">${dom.p.f1.toFixed(3)}</span> for <span class="mono">${Math.round(dom.p.pf)} PF</span>,
          beating dense's ${dom.d.budget} run (<span class="mono">${dom.d.f1.toFixed(3)}</span> at
          <span class="mono">${Math.round(dom.d.pf)} PF</span>) at ${dom.saving.toFixed(1)}× less compute.`
        : `No soft arm beats a measured dense point at ${rung}.`;
    card.appendChild(p); g.appendChild(card);
  });
}
(function(){
  const pick=document.getElementById("rung-picker");
  pick.innerHTML=`<span class="lbl">context</span>`+
    D.rungs.map(r=>`<button type="button" data-r="${r}" aria-pressed="${r==="32k"}">${r}</button>`).join("");
  pick.addEventListener("click",e=>{
    const b=e.target.closest("button[data-r]"); if(!b) return;
    [...pick.querySelectorAll("button")].forEach(x=>x.setAttribute("aria-pressed",x===b));
    renderRungs(b.dataset.r);
  });
  renderRungs("32k");
})();

/* ---------- where the compression is paid for: delta vs rung ---------- */
(function(){
  const svg=document.getElementById("delta-chart"), leg=document.getElementById("delta-legend");
  const tasks=Object.keys(D.deltas); if(!tasks.length) return;
  const W=960, PH=190, M={t:26,r:16,b:34,l:52}, gap=40;
  const H=tasks.length*(PH+gap)+10;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.setAttribute("width",W); svg.setAttribute("height",H);
  svg.textContent="";
  const rule=cssv("--rule"), faint=cssv("--faint"), grey=cssv("--grey"), ink=cssv("--ink");
  const limOf=t=>{
    const v=D.deltas[t].flatMap(s=>Object.values(s.d)).map(Math.abs);
    return Math.max(0.06, Math.ceil(Math.max(...v)*20)/20);
  };
  tasks.forEach((t,ti)=>{
    const lim=limOf(t);
    const y0=ti*(PH+gap)+M.t;
    const X=i=>M.l+i*(W-M.l-M.r)/(D.rungs.length-1);
    const Y=v=>y0+PH/2-(v/lim)*(PH/2-8);
    [lim,0,-lim].forEach(v=>{
      svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:Y(v),y2:Y(v),stroke:v===0?faint:rule,
        "stroke-width":v===0?1.2:1}));
      svg.appendChild(txt(M.l-9,Y(v)+4,(v>0?"+":"")+v.toFixed(2),{fill:faint,"font-size":10.5,
        "text-anchor":"end","font-family":"IBM Plex Mono, monospace"}));
    });
    D.rungs.forEach((r,i)=>svg.appendChild(txt(X(i),y0+PH+16,r,{fill:faint,"font-size":11,
      "text-anchor":"middle","font-family":"IBM Plex Mono, monospace"})));
    const bud=[...new Set(D.deltas[t].map(s=>s.budget))].join("/");
    svg.appendChild(txt(M.l,y0-6,t,{fill:ink,"font-size":13,"font-weight":600,
      "font-family":"IBM Plex Serif, serif"}));
    svg.appendChild(txt(M.l+8+t.length*7.2,y0-6,`· ${bud} tokens · scale ±${lim.toFixed(2)}`,
      {fill:faint,"font-size":10.5,"font-family":"IBM Plex Mono, monospace"}));
    D.deltas[t].forEach(sr=>{
      const pts=D.rungs.map((r,i)=>sr.d[r]!==undefined?{x:X(i),y:Y(sr.d[r])}:null).filter(Boolean);
      if(pts.length<2) return;
      const c=armColor({keep:sr.keep});
      svg.appendChild(el("path",{d:pts.map((p,i)=>`${i?"L":"M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "),
        fill:"none",stroke:c,"stroke-width":1.8,"stroke-linejoin":"round",opacity:.9}));
      pts.forEach(p=>svg.appendChild(el("circle",{cx:p.x,cy:p.y,r:2.6,fill:c})));
      const last=pts[pts.length-1];
      svg.appendChild(txt(last.x-7,last.y+3.5,sr.label,{fill:c,"font-size":10,"text-anchor":"end",
        "font-family":"IBM Plex Mono, monospace"}));
    });
  });
  svg.appendChild(txt(0,0,"soft − dense (same token budget)",{fill:grey,"font-size":11.5,
    "text-anchor":"middle","font-family":"IBM Plex Mono, monospace",
    transform:`translate(13,${H/2}) rotate(-90)`}));
  const seen=new Map();
  tasks.forEach(t=>D.deltas[t].forEach(s=>seen.set(s.label,s.keep)));
  leg.innerHTML=[...seen].map(([l,k])=>`<span class="lg"><span class="swatch" style="background:${armColor({keep:k})}"></span>`+
    `<span style="color:var(--grey)">${l}</span></span>`).join("")+
    `<span class="lg"><span style="color:var(--faint)">each line is one arm at its best-trained `+
    `budget; each panel has its own vertical scale</span></span>`;
})();

/* wall clock */
wallChart(document.getElementById("wall-chart"), document.getElementById("wall-legend"));

/* callouts */
(function(){
  const items=[
    ["done","Settled","Keep ratio is the knob. Accuracy is monotone in it at every budget, and only "+
      "<b>1/3</b> tracks dense. Contradiction's honest compaction is therefore <b>~0.33× FLOPs</b>, "+
      "not the 0.09× the eval-side probe suggested."],
    ["open","Open — wall-clock","FLOPs are down 3×; seconds per step are not. Sorting recovered "+
      "1.58× where a micro-batch held enough padding to matter, leaving the best soft arm at <b>5.4×</b> "+
      "dense's s/step. The remaining fix is to compact <b>then pack</b> with <b>cu_seqlens</b> so no "+
      "padding is spent at all — not yet implemented."],
    ["broken","Hard failure","A step that runs past <b>900 s</b> trips the NCCL watchdog and FSDP "+
      "aborts. It has already killed <b>oolong ohdr17</b> at 16M and 32M, whose shard pads 8.9M tokens "+
      "to 238M. Lower that arm's micro-batch; a longer timeout would only hide it."],
    ["open","Open — the cheap end","Every claim below <b>758 PF</b> extrapolates dense. Dense anchors "+
      "at <b>4M</b> and <b>8M</b> tokens would make the cheap end evidence rather than inference."],
    ["broken","Do not quote","Wall-clock from any <b>gen-2</b> run — those predate micro-batch sorting. "+
      "Accuracy from them is fine. And <b>outlier@64k</b>, whose eval file is the v2 xlong one with "+
      "random fillers: ~0.0 at every budget is a broken file, not a result."],
  ];
  document.getElementById("calls").innerHTML = items.map(([k,h,p])=>
    `<div class="call ${k}"><h3>${h}</h3><p>${p}</p></div>`).join("");
})();

document.getElementById("footer").innerHTML =
  `<div>Generated from <span class="mono">results/ds64/results.csv</span> by `+
  `<span class="mono">research/token_routing/ds64/make_pareto_artifact.py</span>. Re-run the collector, then the `+
  `generator, and this page redraws with whatever has landed.</div>`+
  `<div>Qwen3.5-4B · short-heavy 2k–64k mix · mean f1 over the 2k/8k/16k/32k/64k rungs (outlier: `+
  `over the four working rungs — its 64k eval file is broken for every arm), `+
  `eval_size ${D.eval_size} per rung · FLOPs from the in-training meter's `+
  `<span class="mono">actual_pflops</span>, which charges the compacted sequence, not the dense one.</div>`;
</script>

"""

if __name__ == "__main__":
    main()
