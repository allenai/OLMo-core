#!/bin/bash
# Harvest the ds64 runs' flops.json/config.json (FLOP meter, compaction stats) and realized example
# lengths of the ds64 shards from weka to S3 (ds64/harvest), for collect_ds64.py.
set -uo pipefail
CK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
SH=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ds64/shards
S3=s3://ai2-llm/checkpoints/prasanns/ds64/harvest
CMD='AWS=$(command -v aws || ls /opt/conda/bin/aws 2>/dev/null || true); '
CMD+='if [ -z "$AWS" ]; then pip install -q awscli && AWS=$(command -v aws); fi; '
CMD+='[ -n "$AWS" ] || { echo FATAL_NO_AWSCLI; exit 127; }; '
CMD+='mkdir -p ~/.aws && echo "$AWS_CREDS" > ~/.aws/credentials && echo "$AWS_CFG" > ~/.aws/config; export AWS_PROFILE=S3; '
CMD+="mkdir -p /tmp/h/runs /tmp/h/arms; for d in $CK/ds64-*/; do r=\$(basename \$d); mkdir -p /tmp/h/runs/\$r; for f in flops.json provenance.json config.json; do [ -f \$d/\$f ] && cp \$d/\$f /tmp/h/runs/\$r/; done; done; echo runs=\$(ls /tmp/h/runs | wc -l); "
LEN_PY='import numpy as np, glob, json, os
out="/tmp/h/arms"
for a in glob.glob("/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ds64/shards/*_u*"):
    name = os.path.basename(a); f = f"{out}/{name}_lengths.json"
    if not os.path.exists(a+"/metadata.json"): continue
    m = json.load(open(a+"/metadata.json")); eos = m.get("eos_token_id") or 248044
    lens = []
    for part in sorted(glob.glob(a+"/token_ids_part_*.npy")):
        t = np.memmap(part, dtype=np.dtype(m.get("dtype", "uint32")), mode="r")
        e = np.flatnonzero(np.asarray(t) == eos); lens += np.diff(np.concatenate([[-1], e])).tolist()
    json.dump({"arm": name, "eos": eos, "n": len(lens), "lengths": lens, "metadata": m}, open(f, "w")); print(name, len(lens), "examples", flush=True)
'
CMD+="python -c '$LEN_PY' || echo LENGTHS_FAILED; "
CMD+="\$AWS s3 sync /tmp/h/ $S3/ --only-show-errors && echo HARVEST_DONE"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
exec gantry run --name "ds64-harvest-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --cluster ai2/neptune-cirrascale --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --allow-dirty --install false --timeout 0 --yes -- bash -c "$CMD" 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1 | sed 's#ex/#SUBMITTED id=#'
