"""Fetch the CTC-suite sweep's per-job metrics from Beaker and build the 4:1 vs 7:1 table.

One job per (arm, task); each writes /results/<tag>/metrics.json. This pulls every result dataset,
keys by (arm, task, rung), and prints the comparison plus a per-rung ladder. Reports parse rate
beside every score -- on these checkpoints a low score at low parse rate is a formatting failure,
not a capability one, and the two have already been confused once in this project.

Missing or failed jobs are listed explicitly rather than silently dropped: a table that quietly
covers 12 of 16 arms reads as complete.
"""
import json, os, subprocess, sys
from collections import defaultdict

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
IDS = f"{REPO}/debug/hybridish_sft/sweep_ids.txt"
OUT = f"{REPO}/debug/hybridish_sft/results/ctc_suite_sweep.json"
CACHE = f"{REPO}/debug/hybridish_sft/results/_raw"
BK = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/beaker"

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout

os.makedirs(CACHE, exist_ok=True)
rows, failed, missing = defaultdict(dict), [], []
for line in open(IDS):
    if not line.strip():
        continue
    tag, exp = line.split()
    arm, task = tag.split("_", 1)
    info = sh(BK, "experiment", "get", exp, "--format", "json")
    try:
        j = json.loads(info)[0]["jobs"][-1]
    except Exception:
        missing.append(tag); continue
    code = j.get("status", {}).get("exitCode")
    if code != 0:
        failed.append((tag, code)); continue
    ds = j.get("result", {}).get("beaker") or j.get("resultDatasetId")
    dest = f"{CACHE}/{tag}"
    if not os.path.exists(f"{dest}/{tag}/metrics.json"):
        sh(BK, "dataset", "fetch", ds, "-o", dest)
    mp = None
    for root, _, files in os.walk(dest):
        if "metrics.json" in files:
            mp = os.path.join(root, "metrics.json"); break
    if not mp:
        missing.append(tag); continue
    m = json.load(open(mp))
    entries = m if isinstance(m, list) else m.get("results", m.get("metrics", []))
    if isinstance(entries, dict):
        entries = [dict(v, task=k) for k, v in entries.items()]
    for e in entries:
        name = str(e.get("task") or e.get("task_name") or "")
        if ":" not in name:
            continue
        t, rung = name.split(":")[-2:] if name.count(":") > 1 else name.split(":")
        score = next((e[k] for k in ("primary_score", "score", "f1:ctc", "metric_value")
                      if k in e and isinstance(e[k], (int, float))), None)
        if score is None:
            mets = e.get("metrics", {})
            score = next((v for k, v in mets.items() if isinstance(v, (int, float))
                          and "parse" not in k), None)
            parse = next((v for k, v in mets.items() if "parse" in k), None)
        else:
            parse = e.get("ctc_parse_ok", e.get("parse_rate"))
        rows[(t.replace("ctc_", ""), rung)][arm] = (score, parse,
                                                    e.get("num_instances", e.get("n")))

json.dump({f"{t}|{r}": v for (t, r), v in rows.items()}, open(OUT, "w"), indent=2, default=str)

RUNGS = ["r2k", "r4k", "r8k", "r16k", "r32k"]
tasks = sorted({t for t, _ in rows})
print(f"\n{'task':<16}{'rung':<7}{'4:1':>9}{'7:1':>9}{'delta':>9}   parse 4:1 / 7:1")
print("-" * 72)
for t in tasks:
    for r in RUNGS:
        v = rows.get((t, r))
        if not v:
            continue
        a, b = v.get("4to1"), v.get("7to1")
        fa = f"{a[0]:.4f}" if a and a[0] is not None else "  --  "
        fb = f"{b[0]:.4f}" if b and b[0] is not None else "  --  "
        d = f"{b[0]-a[0]:+.4f}" if (a and b and a[0] is not None and b[0] is not None) else "   --"
        pa = f"{a[1]:.2f}" if a and a[1] is not None else "-"
        pb = f"{b[1]:.2f}" if b and b[1] is not None else "-"
        print(f"{t:<16}{r:<7}{fa:>9}{fb:>9}{d:>9}   {pa} / {pb}")
    print()
if failed:
    print(f"\n⚠ FAILED jobs ({len(failed)}): " + ", ".join(f"{t}(exit {c})" for t, c in failed))
if missing:
    print(f"⚠ NO METRICS ({len(missing)}): " + ", ".join(missing))
if not failed and not missing:
    print("all 16 jobs produced metrics")
print(f"\nwrote {OUT}")
