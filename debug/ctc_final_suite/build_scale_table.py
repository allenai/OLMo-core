"""Merge the 0.8B / 2B model-scale cells with the 4B suite row into one scale table.

Emits `scale_table.json`: per task, per arm, per scale, the 2k-32k ladder plus the dense-chunked gap
at the deepest shared rung. The 4B column is not re-harvested -- it is READ BACK from
`suite_table.json`, so the scale figure and the grid can never disagree about what 4B scored.

⚠ EVERY CELL HERE IS ON THE SAME LADDER AS THE 4B GRID. Verified from the `eval_data` path in each
grade JSON: contradiction on `eval_rungs/contradiction_iid` (NOT the both-mode ladder -- those live
in a separate `__BOTHMODE` directory and are skipped by the run-name regex), fiqa on `eval_rungs/fiqa`,
and hotpotqa / qdmatch_nq / reorder on their own. A scale plot built across two different eval sets
would read as a capability curve; this is the check that it is not one.

    python debug/ctc_final_suite/build_scale_table.py
"""
import collections
import glob
import json
import math
import os
import re

RUNGS = [2048, 4096, 8192, 16384, 32768]

# model-scale task dir -> (suite row in suite_table.json, display label, metric)
TASKS = [
    ("hotpotqa",      "hpqa",        "hpqa",             "gold_id_f1"),
    ("contradiction", "contra_real", "contra (realistic)", "set_f1"),
    ("fiqa",          "fiqa",        "fiqa",             "gold_id_f1"),
    ("qdmatch_nq",    "qdmatch_nq",  "qdmatch nq",       "pair_f1"),
    ("reorder",       "reorder",     "reorder",          "kendall_tau"),
]
SCALES = ["0.8b", "2b", "4b"]
# Result dir names are ctcms-<task>-<arm>-<scale>; the __BOTHMODE suffix marks a deliberately
# retained wrong-ladder copy and must not match.
RUN_RE = re.compile(r"^ctcms-(?P<task>.+)-(?P<arm>full|cmix)-(?P<scale>08b|2b)$")
SCALE_OF = {"08b": "0.8b", "2b": "2b"}

# Same alias the grid applies: the realistic contradiction ladder's bottom rung is 2560, not 2048
# (n=44 sits below the training minimum of 52, so the IID rebuild starts at n=56). Without this the
# scale rows lose their 2k column and contradiction looks like it has one fewer rung than 4B.
RUNG_ALIAS = {"contradiction": {2560: 2048}}


def se(p, n):
    if p is None or not n:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)


cells = collections.defaultdict(dict)   # (task, arm, scale) -> {rung: cell}
for p in sorted(glob.glob("debug/ctc_modelscale/results/*/grade_*.json")):
    run = os.path.basename(os.path.dirname(p))
    m = RUN_RE.match(run)
    if not m:
        continue
    r = re.search(r"(\d+)\.json$", os.path.basename(p))
    if not r:
        continue
    d = json.load(open(p))
    arm = "dense" if m.group("arm") == "full" else "chunked"
    v, n = d.get("metric_value"), d.get("eval_size")
    task = m.group("task")
    rung = RUNG_ALIAS.get(task, {}).get(int(r.group(1)), int(r.group(1)))
    cells[(task, arm, SCALE_OF[m.group("scale")])][rung] = {
        "value": round(v, 4) if v is not None else None,
        "eval_size": n,
        "se": round(se(v, n), 4) if se(v, n) is not None else None,
        "parse_rate": d.get("parse_rate"),
    }

# 4B comes from the grid, not from a second harvest.
suite = {e["row"]: e for e in json.load(open("debug/ctc_final_suite/suite_table.json"))}

out = []
for task_dir, suite_row, label, metric in TASKS:
    entry = {"task": task_dir, "row": suite_row, "label": label, "metric": metric, "scales": {}}
    for scale in SCALES:
        for arm in ("dense", "chunked"):
            if scale == "4b":
                got = dict(suite.get(suite_row, {}).get("cells", {}).get(arm, {}))
                # The grid marks superseded/dropped cells; carry that through rather than
                # silently plotting a struck-through number as if it were live.
                got = {int(k): v for k, v in got.items() if v.get("source") != "superseded"}
            else:
                got = cells.get((task_dir, arm, scale), {})
            entry["scales"].setdefault(scale, {})[arm] = {str(k): got[k] for k in sorted(got)}
    # Gap at the deepest rung where this scale has BOTH arms -- the CTC quantity, as a function of
    # model size. Reported per scale so the reader can see whether scaling closes it.
    entry["gap"] = {}
    for scale in SCALES:
        d_, c_ = entry["scales"][scale]["dense"], entry["scales"][scale]["chunked"]
        shared = sorted((int(k) for k in d_ if k in c_), reverse=True)
        entry["gap"][scale] = (
            {"rung": shared[0],
             "value": round(d_[str(shared[0])]["value"] - c_[str(shared[0])]["value"], 4)}
            if shared else None
        )
    out.append(entry)

json.dump(out, open("debug/ctc_final_suite/scale_table.json", "w"), indent=1)

print(f"{'task':20s} {'arm':8s} {'scale':6s} {'ladder':>34s}   gap")
for e in out:
    for scale in SCALES:
        for arm in ("dense", "chunked"):
            c = e["scales"][scale][arm]
            lad = " ".join(f"{c[str(r)]['value']:.3f}" if str(r) in c else "  --  " for r in RUNGS)
            g = e["gap"][scale]
            gs = f"{g['value']:+.3f}@{g['rung'] // 1024}k" if g and arm == "dense" else ""
            print(f"{e['label'][:20]:20s} {arm:8s} {scale:6s} {lad:>34s}   {gs}")
missing = [(e["label"], s, a) for e in out for s in SCALES for a in ("dense", "chunked")
           if not e["scales"][s][a]]
print(f"\n{len(missing)} empty (task, scale, arm) combinations:")
for lab, s, a in missing:
    print(f"  {lab} / {s} / {a}")
