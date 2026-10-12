#!/usr/bin/env bash
# Stage the ds64 outlier DENSE checkpoint (ds64-outlier-dense-u64M) from weka to S3 as MODEL-ONLY
# bf16 distcp, plus the outlier eval rung jsonls (rung_2048/8192/16384/32768) used by the ds64
# multirung eval. Same recipe as transfer_dense_ckpts_gantry.sh (see ckpt_transfer_log.md):
# gantry CPU job with weka mounted + AWS secrets, NO --install flag (must take gantry's default
# full `pip install -e .` path so torch/olmo_core are importable -- `--install true` breaks
# torch import on this baked image). Then locally:
#   AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/ds64-outlier-dense-u64M/ /data/prasann/ckpts/ds64-outlier-dense-u64M/
#   AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/ds64-outlier-eval-rungs/ /data/prasann/ds64_eval/outlier/
# on the target node (node-local /data, never /scratch/tmp).
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
NAME="ds64-outlier-dense-u64M"
RUNG_DIR=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/outlier_lengthmix/eval_rungs/outlier
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 RANK=0 WORLD_SIZE=0 LOCAL_RANK=0

echo "--- resolving checkpoint dir for $NAME"
ls -la $WEKA/$NAME* 2>&1
CK=$(ls -d $WEKA/$NAME*/model_and_optim $WEKA/$NAME*/step*/model_and_optim 2>/dev/null | sort -V | tail -1)
[ -n "$CK" ] || { echo "!!! no model_and_optim found for $NAME"; exit 1; }
RUN=$(dirname $CK); OUT=/tmp/transfer/$NAME; mkdir -p $OUT
echo "--- $NAME: $CK -> $OUT $(date +%T)"
python - <<PY
import json, os, sys, torch, torch.distributed as dist, traceback
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
os.environ["WORLD_SIZE"] = "1"
try:
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
if [ $? -ne 0 ]; then echo "!!! FAILED $NAME (python step)"; rm -rf $OUT; exit 1; fi
AWS_PROFILE=S3 aws s3 sync $OUT s3://ai2-llm/checkpoints/prasanns/_transfer/$NAME/ --only-show-errors && echo "SYNCED $NAME $(du -sh $OUT | cut -f1)" || echo "!!! FAILED $NAME (s3 sync)"
rm -rf $OUT

echo "--- syncing outlier eval rungs from $RUNG_DIR"
ls -la $RUNG_DIR 2>&1
AWS_PROFILE=S3 aws s3 sync $RUNG_DIR s3://ai2-llm/checkpoints/prasanns/_transfer/ds64-outlier-eval-rungs/ \
  --exclude "*" --include "rung_2048.jsonl" --include "rung_8192.jsonl" --include "rung_16384.jsonl" --include "rung_32768.jsonl" \
  --only-show-errors && echo "SYNCED eval-rungs" || echo "!!! FAILED eval-rungs sync"

echo TRANSFER_DONE
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "transfer-ds64-outlier-dense-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale --cluster ai2/neptune-cirrascale \
  --gpus 0 --cpus 16 --memory 160GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env NAME="$NAME" --env WEKA="$WEKA" --env RUNG_DIR="$RUNG_DIR" \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | tee "${GANTRY_LOG:-/tmp/gantry_ds64_outlier_launch.log}" | grep -E "beaker.org/ex|rror" | head -3
