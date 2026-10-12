#!/usr/bin/env bash
# Paired per-row Δ full-attention CE between the sd20 budget points (they share the 32 dev rows):
# is the flat 128M -> 256M step (old shard -> 1B shard) noise or real? records/softdetach-cpt-crossover.md
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
names = [n for n in D if D[n]["arm"] == "sd20"] + [n for n in D if D[n]["arm"] == "dense"]
for a in names:
    for b in names:
        if a < b and len(D[a]["per_row"]["full_ce"]) == len(D[b]["per_row"]["full_ce"]):
            m, se = paired(D[a], D[b]); print(f"PAIR {a} - {b} = {m:+.4f} (se {se:.4f}, {m/se if se else 0:+.1f} sigma)")
PY
EOF
gantry run --name "sdcpt-pairs-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --cpus 2 --memory 8GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --branch prasann/landmark --allow-dirty \
  --weka oe-training-default:/weka/oe-training-default --timeout 0 --yes -- bash -c "$WORK"
