#!/usr/bin/env bash
# Stage the ladder's dense-trained Qwen3.5-4B checkpoints (contradiction 56M, oolong 80M) from weka
# to S3 as MODEL-ONLY bf16 distcp (~9.6 GB each instead of ~58 GB with AdamW state), for the local
# eval-side slot probes (records/pooled-doc-kv-attention.md, 2026-09-08). Recipe from the
# weka-s3-checkpoint-transfer memory: gantry CPU job with weka mounted + AWS secrets; then locally
#   AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/<name>/ /data/prasann/dense_ckpts/<name>/
# on the target node (node-local /data, never /scratch).
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
NAMES="${NAMES:-tsl-full-contradiction-s56M-4b-20260831T235942-0700 tsl-full-oolong-s80M-4b-20260901T085004-0700}"
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 RANK=0 WORLD_SIZE=0 LOCAL_RANK=0
for NAME in $NAMES; do
  CK=$(ls -d $WEKA/$NAME*/model_and_optim $WEKA/$NAME*/step*/model_and_optim 2>/dev/null | tail -1)
  [ -n "$CK" ] || { echo "!!! no model_and_optim for $NAME"; continue; }
  RUN=$(dirname $CK); OUT=/tmp/transfer/$NAME; mkdir -p $OUT
  echo "--- $NAME: $CK -> $OUT $(date +%T)"
  python - <<PY
import json, torch, torch.distributed as dist, os
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
os.environ["WORLD_SIZE"] = "1"
dist.init_process_group("gloo", rank=0, world_size=1)
run, out = "$RUN", "$OUT"
cfg = json.load(open(run + "/config.json"))
m = TransformerConfig.from_dict(cfg["model"]).build(init_device="cpu")
load_model_and_optim_state("$CK", m)
m = m.to(torch.bfloat16)
save_model_and_optim_state(out + "/model_and_optim", m, save_overwrite=True)
json.dump(cfg, open(out + "/config.json", "w"))
print("resaved", out, sum(p.numel() for p in m.parameters()) / 1e9, "B params")
PY
  AWS_PROFILE=S3 aws s3 sync $OUT s3://ai2-llm/checkpoints/prasanns/_transfer/$NAME/ --only-show-errors && echo "SYNCED $NAME $(du -sh $OUT | cut -f1)"
  rm -rf $OUT
done
echo TRANSFER_DONE
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "transfer-dense-ckpts-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale --cluster ai2/neptune-cirrascale \
  --gpus 0 --cpus 16 --memory 160GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env NAMES="$NAMES" --env WEKA="$WEKA" \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror" | head -3
