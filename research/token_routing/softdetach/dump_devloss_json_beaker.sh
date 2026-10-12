#!/bin/bash
# CPU gantry job: print every dev-loss JSON's summary (means + SEs) from weka, since weka is not
# mounted on the login node. Output is read back with `beaker experiment logs`.
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/softdetach_cpt/devloss
read -r -d '' WORK <<EOF
/opt/conda/bin/python - <<'PY'
import glob, json, os, math
D = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob("$WEKA/*.json"))}
def paired(a, b):
    diff = [x - y for x, y in zip(a["per_row"]["full_ce"], b["per_row"]["full_ce"])]
    m = sum(diff) / len(diff); se = (sum((x - m) ** 2 for x in diff) / (len(diff) - 1)) ** 0.5 / math.sqrt(len(diff))
    return m, se
dense = {n: d for n, d in D.items() if d["arm"] == "dense"}
for name, d in D.items():
    s = d["summary"]
    line = " ".join(f"{k}={v:.4f}" for k, v in s.items() if v is not None)
    # PAIRED per-row deltas vs EVERY dense point (the between-row SE ~0.045 is row difficulty, not
    # noise); the reader picks the dense point at equal PF from the collector table
    for dn, dd in sorted(dense.items()):
        if len(dd["per_row"]["full_ce"]) == len(d["per_row"]["full_ce"]):
            m, se = paired(d, dd)
            line += f" d_vs_{dn.split('-u')[-1]}={m:+.4f}({se:.4f})"
    print("DEVLOSS", name, d["arm"], d["eval_size"], line)
PY
EOF
gantry run --name "sdcpt-dump-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --cpus 2 --memory 8GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --branch prasann/landmark --allow-dirty --no-logs \
  --weka oe-training-default:/weka/oe-training-default --timeout 0 --yes -- bash -c "$WORK"
