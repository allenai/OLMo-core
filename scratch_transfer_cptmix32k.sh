#!/bin/bash
set -euo pipefail
export PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH"

CK="/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-cptmix-5task-32k-1n/step122"
NAME="q4b-cptmix-5task-32k-step122-modelonly"
S3="s3://ai2-llm/checkpoints/prasanns/_transfer/${NAME}"

PY_B64=$(cat <<'PY' | base64 -w0
import json, os
import torch, torch.distributed as dist
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
dist.init_process_group("gloo")
ck = os.environ["CK"]; out = os.environ["OUT"]
cfg = json.load(open(ck + "/config.json"))
m = TransformerConfig.from_dict(cfg["model"]).build(init_device="cpu").to(torch.bfloat16)
load_model_and_optim_state(ck + "/model_and_optim", m)
os.makedirs(out, exist_ok=True)
save_model_and_optim_state(out + "/model_and_optim", m, save_overwrite=True)
json.dump(cfg, open(out + "/config.json", "w"))
print("RESAVE_OK", out, flush=True)
PY
)

gantry run \
  --name "${NAME}" \
  --workspace ai2/flex2 \
  --budget ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan --cluster ai2/saturn \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --cpus 8 \
  --priority urgent \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS \
  --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env CK="${CK}" \
  --env OUT="/tmp/${NAME}" \
  --env PY_B64="${PY_B64}" \
  --env S3="${S3}" \
  --allow-dirty \
  --timeout 0 \
  --yes \
  -- bash -lc 'set -euo pipefail; mkdir -p ~/.aws; printf "%s" "$AWS_CREDS" > ~/.aws/credentials; printf "%s" "$AWS_CFG" > ~/.aws/config; pip install -q awscli || true; ls -la "$CK" || { echo MISSING_CK "$CK"; exit 3; }; export MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 RANK=0 WORLD_SIZE=1; echo "$PY_B64" | base64 -d > /tmp/resave.py; python /tmp/resave.py; AWS_PROFILE=S3 aws s3 sync "$OUT" "$S3" --only-show-errors; echo SYNC_DONE'
