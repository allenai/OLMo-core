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
BK = "/accounts/projects/berkeleynlp/prasann/.local/bin/beaker"

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout

os.makedirs(CACHE, exist_ok=True)
rows, failed, missing, running = defaultdict(dict), [], [], []
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
    st = j.get("status", {})
    if "exitCode" not in st:
        # Still in flight. Reporting it as FAILED would be a lie, and reporting nothing would make
        # a partial table look complete -- so it gets its own bucket.
        running.append(tag); continue
    if st["exitCode"] != 0:
        failed.append((tag, st["exitCode"])); continue
    ds = j.get("result", {}).get("beaker") or j.get("resultDatasetId")
    # Fetch ONLY metrics.json. The full result dataset is ~130 MB of saved predictions apiece;
    # sixteen of those exhausted the /accounts quota, after which beaker's mkdir failed and the
    # harvester's own json.dump left a 0-byte file -- a silent loss that looked like a parse bug.
    dest = f"{CACHE}/{tag}"
    if not os.path.exists(f"{dest}/{tag}/metrics.json"):
        sh(BK, "dataset", "fetch", ds, "--prefix", f"{tag}/metrics.json", "-o", dest)
    mp = None
    for root, _, files in os.walk(dest):
        if "metrics.json" in files:
            mp = os.path.join(root, "metrics.json"); break
    if not mp:
        missing.append(tag); continue
    m = json.load(open(mp))
    for e in m.get("tasks", []):
        name = e.get("task", "")            # e.g. "ctc_strmatch:r16k"
        if ":" not in name:
            continue
        tname, rung = name.rsplit(":", 1)
        mets = e.get("metrics", {})
        # primary_metric is "<family>:<scorer>", e.g. "f1:ctc" -- index rather than guess
        fam, _, scorer = e.get("primary_metric", "").partition(":")
        score = mets.get(fam, {}).get(scorer)
        parse = mets.get("parse_rate", {}).get("ctc_parse_ok")
        rows[(tname.replace("ctc_", ""), rung)][arm] = (score, parse, e.get("num_instances"))

payload = json.dumps({f"{t}|{r}": v for (t, r), v in rows.items()}, indent=2, default=str)
with open(OUT + ".tmp", "w") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())          # a truncated write on a full disk must fail here, not silently
os.replace(OUT + ".tmp", OUT)
assert os.path.getsize(OUT) > 2, "results file is empty -- refusing to report from it"

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
if running:
    print(f"\n… STILL RUNNING ({len(running)}): " + ", ".join(running))
if not failed and not missing and not running:
    print("all 16 jobs produced metrics")
else:
    print(f"\nTABLE IS PARTIAL: {16-len(failed)-len(missing)-len(running)}/16 arms present")
print(f"\nwrote {OUT}")
