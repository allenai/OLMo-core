"""
Collect the fast2k screening loop into ``debug/ds64_fast2k/results.csv``.

One row per training run:

  run, arm, budget, steps, gpus, gpu_hours,
  actual_pflops / dense_pflops / actual_over_dense   (the FLOP METER -- never the trainer's
      ``throughput/total petaflops``, which charges every token the DENSE per-token cost and reads
      ~10x high on a compacted arm),
  ce_1 / ce_5 / ce_10 / ce_20 / ce_final,  ce_floor,  at_floor_step10,
  f1_2k, eval_size, se, dense_f1_at_same_flops, matched_flop_delta, extrapolated.

THE TWO EARLY SIGNALS, and what they mean:

* ``ce_floor`` is ``ln C(n, k) / answer_tokens`` for the shard (computed by ``fast2k_stats.py`` at
  build time and cached in ``debug/ds64_fast2k/fast2k_stats.json``). It is the cross-entropy a model
  that has learned only the OUTPUT FORMAT cannot beat. ``at_floor_step10`` / ``at_floor_final`` fire
  when the CE has NOT dropped below ``ce_floor - --floor-tol``: that is the signature of the xhdr collapse
  (``debug/ds64/xhdr_collapse_diagnosis.md``) -- "sample k ids from the visible list" -- and it is
  readable ~3 minutes into a run instead of after a 500-row eval. A run that fires it is dead; a run
  that does NOT fire it still has to clear the matched-FLOP bar.
* ``matched_flop_delta`` = this arm's f1 minus the DENSE f1 interpolated (in log FLOPs) to this
  arm's measured FLOPs. Positive = beats dense at the same compute. ``extrapolated=True`` means the
  arm sits outside the measured dense range, where ds64 has already been burned once
  (records/ds64-handoff.md section 1: dense at ~195 PF was projected 0.07-0.15 and measured 0.317).
  Treat an extrapolated delta as a hypothesis, not a result.

    python debug/ds64_fast2k/collect_fast2k.py
    python debug/ds64_fast2k/collect_fast2k.py --stats-from-job <data-build-experiment-id>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/ds64_fast2k"
STATE = f"{D}/state.json"
STATS = f"{D}/fast2k_stats.json"
CACHE = f"{D}/logs"
OUT = f"{D}/results.csv"
ENV = dict(os.environ, PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""))

METER_RE = {k: re.compile(r"wandb:\s+flop_meter/%s\s+([\d.]+)" % k)
            for k in ("actual_pflops", "dense_pflops", "actual_over_dense")}
STEP_RE = re.compile(r"\[step=(\d+)/")
CE_RE = re.compile(r"train/CE loss=([0-9.]+)")
# the multirung evaluator's own result line
LADDER_RE = re.compile(r"\[ladder:(\w+)@(\d+k)\] (?:f1|score)=([0-9.]+) \(n=(\d+)")


def beaker_log(ex, kind, final=False):
    """Fetch a Beaker log; cache it only when the job is FINALIZED (a mid-run cache freezes
    partial FLOP meters and partial rung sets -- ds64 trap 6)."""
    os.makedirs(CACHE, exist_ok=True)
    p = f"{CACHE}/{kind}_{ex}.log"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return open(p).read()
    try:
        out = subprocess.run(["beaker", "experiment", "logs", ex], env=ENV,
                             capture_output=True, text=True, timeout=900).stdout
    except Exception:  # noqa: BLE001
        return ""
    if final and out:
        open(p, "w").write(out)
    return out


def ce_curve(log_text):
    """{step: train CE} from the console logger's `[step=N/M]` + indented `train/CE loss=` lines."""
    curve, step = {}, None
    for ln in log_text.splitlines():
        m = STEP_RE.search(ln)
        if m:
            step = int(m.group(1))
        m2 = CE_RE.search(ln)
        if m2 and step is not None:
            curve[step] = float(m2.group(1))
    return curve


def train_facts(ex, final):
    out = beaker_log(ex, "train", final=final)
    meter = {}
    for k, rx in METER_RE.items():
        v = rx.findall(out)
        if v:
            meter[k] = float(v[-1])
    stamps = []
    for ln in out.splitlines():
        m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z .*\[step=(\d+)/", ln)
        if m:
            stamps.append((datetime.fromisoformat(m.group(1)), int(m.group(2))))
    secs = (stamps[-1][0] - stamps[0][0]).total_seconds() if len(stamps) > 1 else None
    steps = stamps[-1][1] if stamps else None
    return meter, steps, secs, ce_curve(out)


def eval_f1(ex, final):
    out = beaker_log(ex, "eval", final=final)
    for _task, rung, val, n in LADDER_RE.findall(out):
        if rung == "2k":
            return float(val), int(n)
    return None, None


def log_interp(anchors, pf):
    """Dense f1 at ``pf`` petaflops by linear interpolation in log FLOPs.

    :param anchors: [(pflops, f1)] sorted -- the DENSE runs.
    :returns: (f1, extrapolated?) or (None, None) if there are fewer than 2 anchors.
    """
    if len(anchors) < 2 or not pf:
        return None, None
    xs = [math.log(a) for a, _ in anchors]
    ys = [b for _, b in anchors]
    x = math.log(pf)
    if x <= xs[0]:
        i, ext = 0, True
    elif x >= xs[-1]:
        i, ext = len(xs) - 2, True
    else:
        ext = False
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i]), ext


def stats_for(run, stats):
    """fast2k_stats.json entry for a run: keyed by shard arm name (`<task>_f<budget>`)."""
    if not stats:
        return {}
    if "ce_floor" in stats:          # single-shard file
        return stats
    for k, v in stats.items():
        if run.endswith(k.split("_f")[-1]):
            return v
    return next(iter(stats.values()), {})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor-tol", type=float, default=0.03,
                    help="a CE at or above ce_floor - tol counts as 'parked at the floor'")
    ap.add_argument("--stats-from-job", default="",
                    help="data-build experiment id: pull the fast2k_stats.json blocks out of its log")
    args = ap.parse_args()

    if args.stats_from_job:
        txt = beaker_log(args.stats_from_job, "data", final=True)
        blocks, cur, depth = {}, [], 0
        for ln in txt.splitlines():
            body = ln.split(" ", 1)[-1] if ln[:4].isdigit() else ln
            if body.strip() == "{":
                depth, cur = 1, [body]
            elif depth:
                cur.append(body)
                depth += body.count("{") - body.count("}")
                if depth == 0:
                    try:
                        j = json.loads("\n".join(cur))
                        if "ce_floor" in j:
                            blocks[os.path.basename(j["shard_dir"])] = j
                    except Exception:  # noqa: BLE001
                        pass
                    cur = []
        json.dump(blocks, open(STATS, "w"), indent=2)
        print(f"{len(blocks)} shard stats -> {STATS}")
        for k, v in blocks.items():
            print(f"  {k}: ce_floor={v['ce_floor']:.4f} n_docs~{v['n_docs_mean']:.1f} "
                  f"ans_tok={v['mean_answer_tokens']:.1f} f1_guess={v['f1_uniform_guess']:.3f} "
                  f"overlap={v.get('example_overlap', '?')}")

    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}
    stats = json.load(open(STATS)) if os.path.exists(STATS) else {}
    rows = []
    for run, r in st["runs"].items():
        meter, steps, secs, ce = train_facts(r["ex"], r.get("state") == "DONE") if r.get("ex") else ({}, None, None, {})
        e = st["evals"].get(run, {})
        f1, esz = eval_f1(e["ex"], e.get("state") == "DONE") if e.get("ex") else (None, None)
        sdict = stats_for(run, stats)
        floor = sdict.get("ce_floor")
        ce10 = ce.get(10) or ce.get(max([s for s in ce if s <= 10], default=0))
        se = math.sqrt(f1 * (1 - f1) / esz) if (f1 is not None and esz) else None
        rows.append({
            "run": run, "task": r["task"], "arm": r["arm"], "budget": r["budget"],
            "train_state": r.get("state"), "eval_state": e.get("state"),
            "steps": steps, "steps_expected": r.get("steps_expected"), "gpus": r.get("gpus"),
            "train_seconds": secs,
            "gpu_hours": (r.get("gpus") * secs / 3600) if (secs and r.get("gpus")) else None,
            "actual_pflops": meter.get("actual_pflops"), "dense_pflops": meter.get("dense_pflops"),
            "actual_over_dense": meter.get("actual_over_dense"),
            "ce_1": ce.get(1), "ce_5": ce.get(5), "ce_10": ce.get(10), "ce_20": ce.get(20),
            "ce_final": ce[max(ce)] if ce else None,
            "ce_floor": floor,
            # "at the floor" = the CE has NOT gone meaningfully BELOW ln C(n,k)/answer_tokens, i.e.
            # the run has learned the output format and nothing else. (An earlier version tested
            # `ce - floor <= tol`, which fires for every HEALTHY run -- a model that learns the task
            # drives CE far below the floor. The test is `>=`, not `<=`.)
            "at_floor_step10": (None if (floor is None or ce10 is None) else bool(ce10 >= floor - args.floor_tol)),
            # The step-10 flag is an EARLY HINT only: a slow-starting but healthy soft arm can still
            # be at the floor at step 10 (kvgb50 measured 0.377 at step 10 vs a 0.390 floor, then
            # finished at 0.296). at_floor_final is the decision.
            "at_floor_final": (None if (floor is None or not ce) else bool(ce[max(ce)] >= floor - args.floor_tol)),
            "f1_2k": f1, "eval_size": esz, "se": se,
            "train_ex": r.get("ex"), "eval_ex": e.get("ex"),
        })

    anchors = sorted([(x["actual_pflops"], x["f1_2k"]) for x in rows
                      if x["arm"] == "dense" and x["actual_pflops"] and x["f1_2k"] is not None
                      and (x["steps"] or 0) > 2])
    for x in rows:
        d, ext = log_interp(anchors, x["actual_pflops"])
        x["dense_f1_at_same_flops"] = d
        x["matched_flop_delta"] = (x["f1_2k"] - d) if (d is not None and x["f1_2k"] is not None) else None
        x["extrapolated"] = ext
    cols = list(rows[0].keys()) if rows else ["run"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} runs -> {OUT}   dense anchors: " +
          ", ".join(f"{a:.0f}PF/{b:.3f}" for a, b in anchors))
    hdr = f"{'run':32} {'st':5} {'steps':>6} {'PF':>8} {'x':>6} {'CE10':>6} {'CEfin':>6} {'floor':>6} {'f1_2k':>7} {'delta':>7}"
    print(hdr + "\n" + "-" * len(hdr))
    for x in sorted(rows, key=lambda z: (z["arm"], z["budget"])):
        def g(k, f="{:.3f}"):
            return f.format(x[k]) if x[k] is not None else "-"
        flag = (" FLOOR" if x["at_floor_final"] else ("  ~f10" if x["at_floor_step10"] else ""))
        ext = "*" if x["extrapolated"] else ""
        print(f"{x['run']:32} {str(x['train_state'])[:5]:5} {str(x['steps'] or '-'):>6} "
              f"{g('actual_pflops','{:.1f}'):>8} {g('actual_over_dense','{:.2f}'):>6} "
              f"{g('ce_10','{:.3f}'):>6} {g('ce_final','{:.3f}'):>6} {g('ce_floor','{:.3f}'):>6} "
              f"{g('f1_2k'):>7} {g('matched_flop_delta','{:+.3f}'):>6}{ext}{flag}")
    print("\n* = matched-FLOP delta EXTRAPOLATED beyond the measured dense range -- hypothesis only.")
    print("f1 SE is the binomial approximation at eval_size=500 (~0.021 at f1 0.7, ~0.010 at 0.95);")
    print("outlier's per-example set-F1 is not Bernoulli, so read it as a resolution, not a test.")


if __name__ == "__main__":
    main()
