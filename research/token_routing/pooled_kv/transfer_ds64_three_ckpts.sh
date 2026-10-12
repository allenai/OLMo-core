#!/usr/bin/env bash
# Stage three more ds64 checkpoints from weka to S3 as MODEL-ONLY bf16 distcp:
#   ds64-oolong-occ00-b128f3-u16M, ds64-contradiction-hdr33-b128f3-u64M, ds64-nq-kv33-b128f3-u64M
# Same recipe as transfer_ds64_outlier_dense.sh / transfer_dense_ckpts_gantry.sh (see
# ckpt_transfer_log.md): gantry CPU job, weka mount + AWS secrets, NO --install flag (gantry's
# default full `pip install -e .` path -- `--install true` breaks torch import on this image).
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
NAMES="${NAMES:-ds64-oolong-occ00-b128f3-u16M ds64-contradiction-hdr33-b128f3-u64M ds64-nq-kv33-b128f3-u64M}"
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 RANK=0 WORLD_SIZE=0 LOCAL_RANK=0
for NAME in $NAMES; do
  echo "--- resolving checkpoint dir for $NAME"
  ls -la $WEKA/$NAME* 2>&1
  CK=$(ls -d $WEKA/$NAME*/model_and_optim $WEKA/$NAME*/step*/model_and_optim 2>/dev/null | sort -V | tail -1)
  [ -n "$CK" ] || { echo "!!! no model_and_optim for $NAME"; continue; }
  RUN=$(dirname $CK); OUT=/tmp/transfer/$NAME; mkdir -p $OUT
  echo "--- $NAME: $CK -> $OUT $(date +%T)"
  python - <<PY
import json, os, sys, torch, torch.distributed as dist, traceback
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
os.environ["WORLD_SIZE"] = "1"
try:
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    run, out = "$RUN", "$OUT"
    cfg_path = None
    d = run
    for _ in range(3):
        cand = os.path.join(d, "config.json")
        if os.path.exists(cand):
            cfg_path = cand
            break
        d = os.path.dirname(d)
    if cfg_path is None:
        raise FileNotFoundError(f"no config.json found near {run} (searched up 2 levels)")
    cfg = json.load(open(cfg_path))
    m = TransformerConfig.from_dict(cfg["model"]).build(init_device="cpu")
    load_model_and_optim_state("$CK", m)
    m = m.to(torch.bfloat16)
    save_model_and_optim_state(out + "/model_and_optim", m, save_overwrite=True)
    json.dump(cfg, open(out + "/config.json", "w"))
    print("resaved", out, sum(p.numel() for p in m.parameters()) / 1e9, "B params")
except Exception:
    print("!!! PYFAIL $NAME", flush=True)
    traceback.print_exc()
    sys.exit(1)
PY
  if [ $? -ne 0 ]; then echo "!!! FAILED $NAME (python step)"; rm -rf $OUT; continue; fi
  AWS_PROFILE=S3 aws s3 sync $OUT s3://ai2-llm/checkpoints/prasanns/_transfer/$NAME/ --only-show-errors && echo "SYNCED $NAME $(du -sh $OUT | cut -f1)" || echo "!!! FAILED $NAME (s3 sync)"
  rm -rf $OUT
done
echo TRANSFER_DONE
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "transfer-ds64-three-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale --cluster ai2/neptune-cirrascale \
  --gpus 0 --cpus 16 --memory 160GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env NAMES="$NAMES" --env WEKA="$WEKA" \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | tee "${GANTRY_LOG:-/tmp/gantry_ds64_three_launch.log}" | grep -E "beaker.org/ex|rror" | head -3
