"""
Collect the fast8k screening loop into ``debug/ds64_fast8k/results.csv`` (one row per training run)
and ``debug/ds64_fast8k/parity_rows.csv`` (one row per run x parity-condition x rung).

Reuses ``debug/ds64_fast2k/collect_fast2k.py``'s log fetch, CE-curve parse and log-FLOP
interpolation verbatim -- only the directory it caches into is overridden -- and adds the three
things fast8k needs that fast2k does not:

1. **EVAL-TIME CE PARITY, which is the primary metric here.** ``CE_full`` vs ``CE_soft`` is the SAME
   trained checkpoint run on two inputs: full real text (what the ladder eval feeds it) and its own
   soft construction. A ladder f1 alone cannot tell "this checkpoint cannot do outlier" from "this
   checkpoint can do outlier but its compaction throws the answer away"; ``dCE`` -- and especially
   ``dCE`` on the answer DIGITS, which drops the shared output format -- does. Parity is
   ``dCEdig ~ 0`` AND ``dF1 ~ 0``. ⚠ A large ``dCEdig`` with a small ``|dF1|`` on a WEAK ``F1_full``
   is not parity; it is two ways of being wrong, so always read ``dF1`` beside ``F1_full``.

2. **Two rungs, and the 8k one decides.** ``f1_8k`` is the screen's verdict; ``f1_2k`` is carried
   only for continuity with the fast2k table. The matched-FLOP delta is computed against the DENSE
   anchors' **8k** f1.

3. **A second CE floor for the ``gold_pooled_random`` arms.** ``ce_floor`` is the shard's
   ``ln C(n, k) / answer_tokens`` -- what a model that learned only the output format cannot beat.
   But a ``cpi<p>`` arm pools every gold document, so a model can learn the TRAINING-ONLY
   regularity "the answer is among the pooled documents" and guess inside that smaller set:
   ``ce_floor_pooled = ln C(k + (1-p)(n-k), k) / answer_tokens``. That shortcut transfers to
   nothing (eval has no pooled subset), so it can only flatter the training CE. Read a cpi arm's CE
   descent against ``ce_floor_pooled``, not ``ce_floor``.

    python debug/ds64_fast8k/collect_fast8k.py --stats-from-job <data-build-experiment-id>
    python debug/ds64_fast8k/collect_fast8k.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/ds64_fast8k"
sys.path.insert(0, f"{REPO}/debug/ds64_fast2k")

import collect_fast2k as C  # noqa: E402

C.CACHE = f"{D}/logs"  # the ONLY fast2k global this collector repoints

STATE = f"{D}/state.json"
STATS = f"{D}/fast8k_stats.json"
OUT = f"{D}/results.csv"
PARITY_OUT = f"{D}/parity_rows.csv"

#: the multirung evaluator's own per-rung result line
LADDER_RE = re.compile(r"\[ladder:(\w+)@(\d+k)\] (?:f1|score)=([0-9.]+) \(n=(\d+)")
#: one row of outlier_slot_probe.parity_report's table:
#:   rung eval_size | CE_full CE_soft dCE | CEdig_full CEdig_soft dCEdig | F1_full F1_soft dF1 | compact
PARITY_RE = re.compile(
    r"(\d+k)\s+(\d+)\s*\|"
    r"\s*([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s*\|"
    r"\s*([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s*\|"
    r"\s*([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s*\|"
    r"\s*([-\d.]+)"
)
#: the keep FRACTION of each cpi arm, for the pooled-guess floor
CPI_FRAC = {"cpi17": 1.0 / 6.0, "cpi33": 1.0 / 3.0, "cpi50": 0.5}


def eval_f1(ex, final):
    """{rung: (f1, eval_size)} from a multirung eval job's log."""
    out = C.beaker_log(ex, "eval", final=final)
    return {rung: (float(v), int(n)) for _t, rung, v, n in LADDER_RE.findall(out)}


def parity_rows(ex, final):
    """[{rung, eval_size, ce_full, ce_soft, dce, ...}] from a parity probe job's log."""
    out = C.beaker_log(ex, "parity", final=final)
    rows = []
    for m in PARITY_RE.findall(out):
        rows.append({
            "rung": m[0], "parity_eval_size": int(m[1]),
            "ce_full": float(m[2]), "ce_soft": float(m[3]), "dce": float(m[4]),
            "cedig_full": float(m[5]), "cedig_soft": float(m[6]), "dcedig": float(m[7]),
            "genf1_full": float(m[8]), "genf1_soft": float(m[9]), "dgenf1": float(m[10]),
            "compaction": float(m[11]),
        })
    return rows


def pooled_floor(sd, frac):
    """``ln C(k + (1-p)(n-k), k) / answer_tokens`` -- the floor a gold_pooled_random arm can reach
    by learning "the answer is among the POOLED documents" and guessing inside that set."""
    n, k, a = sd.get("n_docs_mean"), sd.get("k_gold_mean"), sd.get("mean_answer_tokens")
    if not (n and k and a):
        return None
    k = int(round(k))
    m = max(k, int(round(k + (1.0 - frac) * (n - k))))
    return math.log(math.comb(m, k)) / a


def extract_stats(job_id):
    """Pull every ``fast8k_stats*.json`` block out of the data-build job's log.

    Keyed ``<shard>::<eval file>`` -- unlike fast2k, fast8k prints TWO blocks per shard (the 8k
    rung and the 2k rung), and keying by shard alone silently keeps only the last one.
    """
    txt = C.beaker_log(job_id, "data", final=True)
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
                        key = (os.path.basename(j["shard_dir"]) + "::"
                               + os.path.basename(j.get("eval_jsonl", "none")))
                        blocks[key] = j
                except Exception:  # noqa: BLE001
                    pass
                cur = []
    return blocks


def stats_for(budget, stats, rung_file="rung_8192.jsonl"):
    for k, v in stats.items():
        shard, _, ev = k.partition("::")
        if shard.endswith(f"_g{budget}") and ev == rung_file:
            return v
    for k, v in stats.items():  # any eval file, if the 8k one is missing
        if k.partition("::")[0].endswith(f"_g{budget}"):
            return v
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor-tol", type=float, default=0.03)
    ap.add_argument("--stats-from-job", default="")
    args = ap.parse_args()

    if args.stats_from_job:
        blocks = extract_stats(args.stats_from_job)
        json.dump(blocks, open(STATS, "w"), indent=2)
        print(f"{len(blocks)} shard x eval-file stats -> {STATS}")
        for k, v in sorted(blocks.items()):
            print(f"  {k}: ce_floor={v['ce_floor']:.4f} n_docs~{v['n_docs_mean']:.1f} "
                  f"k={v['k_gold_mean']:.1f} ans_tok={v['mean_answer_tokens']:.1f} "
                  f"max_len={v['shard_max_example_len']} rows={v['shard_num_instances']} "
                  f"dropped={v['shard_num_dropped']} f1_guess={v['f1_uniform_guess']:.3f} "
                  f"example_overlap={v.get('example_overlap', '?')}")

    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}, "parity": {}}
    stats = json.load(open(STATS)) if os.path.exists(STATS) else {}

    # --- parity long form -----------------------------------------------------------------------
    prows = []
    for key, p in sorted(st.get("parity", {}).items()):
        if not p.get("ex"):
            continue
        for row in parity_rows(p["ex"], p.get("state") == "DONE"):
            prows.append(dict(run=p["run"], cond=p["cond"], state=p.get("state"),
                              parity_ex=p["ex"], **row))
    if prows:
        with open(PARITY_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
            w.writeheader()
            w.writerows(prows)

    rows = []
    for run, r in st["runs"].items():
        meter, steps, secs, ce = (C.train_facts(r["ex"], r.get("state") == "DONE")
                                  if r.get("ex") else ({}, None, None, {}))
        e = st["evals"].get(run, {})
        f1s = eval_f1(e["ex"], e.get("state") == "DONE") if e.get("ex") else {}
        sd = stats_for(r["budget"], stats)
        floor = sd.get("ce_floor")
        frac = CPI_FRAC.get(r["arm"].replace("-warm", ""))
        floor_pooled = pooled_floor(sd, frac) if (frac is not None and sd) else None
        # the floor the arm can actually reach in TRAINING (a cpi arm's pooled-guess shortcut)
        eff_floor = floor_pooled if floor_pooled is not None else floor
        ce10 = ce.get(10) or ce.get(max([s for s in ce if s <= 10], default=0))
        f1_8k, esz = f1s.get("8k", (None, None))
        f1_2k, _ = f1s.get("2k", (None, None))
        se = math.sqrt(f1_8k * (1 - f1_8k) / esz) if (f1_8k is not None and esz) else None
        row = {
            "run": run, "task": r["task"], "arm": r["arm"], "budget": r["budget"],
            "train_state": r.get("state"), "eval_state": e.get("state"),
            "steps": steps, "steps_expected": r.get("steps_expected"), "gpus": r.get("gpus"),
            "train_seconds": secs,
            "gpu_hours": (r.get("gpus") * secs / 3600) if (secs and r.get("gpus")) else None,
            "actual_pflops": meter.get("actual_pflops"), "dense_pflops": meter.get("dense_pflops"),
            "actual_over_dense": meter.get("actual_over_dense"),
            "ce_1": ce.get(1), "ce_5": ce.get(5), "ce_10": ce.get(10), "ce_20": ce.get(20),
            "ce_final": ce[max(ce)] if ce else None,
            "ce_floor": floor, "ce_floor_pooled": floor_pooled,
            # "at the floor" = the CE never went meaningfully BELOW the floor the arm can reach,
            # i.e. the run learned the output format and nothing else.
            "at_floor_step10": (None if (eff_floor is None or ce10 is None)
                                else bool(ce10 >= eff_floor - args.floor_tol)),
            "at_floor_final": (None if (eff_floor is None or not ce)
                               else bool(ce[max(ce)] >= eff_floor - args.floor_tol)),
            "f1_8k": f1_8k, "f1_2k": f1_2k, "eval_size": esz, "se": se,
            "train_ex": r.get("ex"), "eval_ex": e.get("ex"),
        }
        # primary parity condition = the FIRST one launched for this run
        mine = [p for p in prows if p["run"] == run]
        for rung in ("8k", "32k"):
            hit = next((p for p in mine if p["rung"] == rung), None)
            for k in ("cond", "ce_full", "ce_soft", "dce", "cedig_full", "cedig_soft", "dcedig",
                      "genf1_full", "genf1_soft", "dgenf1", "compaction", "parity_eval_size"):
                row[f"{k}_{rung}"] = hit[k] if hit else None
        rows.append(row)

    anchors = sorted([(x["actual_pflops"], x["f1_8k"]) for x in rows
                      if x["arm"] == "dense" and x["actual_pflops"] and x["f1_8k"] is not None
                      and (x["steps"] or 0) > 2])
    for x in rows:
        d, ext = C.log_interp(anchors, x["actual_pflops"])
        x["dense_f1_8k_at_same_flops"] = d
        x["matched_flop_delta"] = (x["f1_8k"] - d) if (d is not None and x["f1_8k"] is not None) else None
        x["extrapolated"] = ext
    if not rows:
        print("no runs in state.json yet")
        return
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} runs -> {OUT}   ({len(prows)} parity rows -> {PARITY_OUT})")
    print("dense 8k anchors: " + ", ".join(f"{a:.0f}PF/{b:.3f}" for a, b in anchors))
    hdr = (f"{'run':34} {'st':5} {'steps':>5} {'PF':>7} {'x':>5} {'CE10':>6} {'CEfin':>6} "
           f"{'floor':>6} {'f1_8k':>6} {'f1_2k':>6} {'dFLOP':>7} | "
           f"{'CEd_f':>6} {'CEd_s':>6} {'dCEd':>6} {'gF1_f':>6} {'gF1_s':>6} {'cond':>6}")
    print(hdr + "\n" + "-" * len(hdr))
    for x in sorted(rows, key=lambda z: (z["arm"], str(z["budget"]))):
        def g(k, f="{:.3f}"):
            return f.format(x[k]) if x.get(k) is not None else "-"
        flag = (" FLOOR" if x["at_floor_final"] else ("  ~f10" if x["at_floor_step10"] else ""))
        ext = "*" if x["extrapolated"] else ""
        fl = x["ce_floor_pooled"] if x["ce_floor_pooled"] is not None else x["ce_floor"]
        print(f"{x['run']:34} {str(x['train_state'])[:5]:5} {str(x['steps'] or '-'):>5} "
              f"{g('actual_pflops','{:.1f}'):>7} {g('actual_over_dense','{:.2f}'):>5} "
              f"{g('ce_10'):>6} {g('ce_final'):>6} "
              f"{('%.3f' % fl) if fl is not None else '-':>6} "
              f"{g('f1_8k'):>6} {g('f1_2k'):>6} {g('matched_flop_delta','{:+.3f}'):>6}{ext} | "
              f"{g('cedig_full_8k'):>6} {g('cedig_soft_8k'):>6} {g('dcedig_8k','{:+.3f}'):>6} "
              f"{g('genf1_full_8k'):>6} {g('genf1_soft_8k'):>6} {str(x.get('cond_8k') or '-'):>6}{flag}")
    print("\nPARITY (the primary signal) = dCEd ~ 0 AND gF1_s ~ gF1_f. A big dCEd with a small dF1 on")
    print("a WEAK gF1_f is not parity -- it is two ways of being wrong.")
    print("'floor' is ce_floor_pooled for the cpi arms (they can guess inside the POOLED set at")
    print("train time only) and ce_floor otherwise. * = matched-FLOP delta EXTRAPOLATED.")


if __name__ == "__main__":
    main()
