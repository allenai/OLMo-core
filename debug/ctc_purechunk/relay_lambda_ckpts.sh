#!/usr/bin/env bash
# Move the two lambda-trained pure-chunked checkpoints (outlier, reorder) onto weka so they can be
# evaluated by the same Beaker vLLM pipeline that graded every other cell in this comparison.
#
#   bash debug/ctc_purechunk/relay_lambda_ckpts.sh            # both
#   TASKS=outlier bash debug/ctc_purechunk/relay_lambda_ckpts.sh
#
# ── WHY THE THREE-HOP ROUTE ───────────────────────────────────────────────────────────────────
# lambda is air-gapped: no internet, so it cannot push to S3 itself. Berkeley cannot mount weka.
# So the only path is lambda -> Berkeley -> S3 -> weka, and each hop needs a different tool.
#
# ── WHY /tmp AND NOT THE REPO OR /scratch ─────────────────────────────────────────────────────
# These are 19G apiece. The repo lives on NFS (`/accounts`) and `/scratch` is the same NFS pool at
# ~5MB/s -- 38G through either is hours of wall-clock and it is exactly the access pattern that
# parks readers in nfs_wait_bit_killable. The login node `radagast` has a real local disk mounted
# at /tmp (/dev/sdc1, 185G free); that is the correct staging area, and it is staging, not output.
#
# ── WHY NOT JUST EVALUATE ON LAMBDA ───────────────────────────────────────────────────────────
# lambda has A100s and could run the native olmo-core evaluator with no transfer at all. But the
# cmix baselines these two arms get compared against were graded by the Beaker vLLM pipeline, and a
# metric difference that is really a harness difference is indistinguishable from the result being
# measured. The transfer is the cheaper mistake.
set -uo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

TASKS="${TASKS:-outlier reorder}"
SUFFIX="${SUFFIX:-purechunk-lambda-r3}"
LAMBDA_ROOT=/accounts/projects/sewonm/prasann/ctc_suite/ckpts
STAGE="${STAGE:-/tmp/pcrelay}"
S3=s3://ai2-llm/checkpoints/prasanns/_transfer
mkdir -p "$STAGE"

for T in $TASKS; do
  NAME="ctc-4b-${T}-${SUFFIX}"
  echo "=== [$T] $(date '+%F %T') rsync lambda -> $STAGE/$NAME"
  rsync -a --info=progress2 "lambda:$LAMBDA_ROOT/$NAME/" "$STAGE/$NAME/" || { echo "[$T] FATAL rsync"; continue; }

  # ⚠ VERIFY BEFORE UPLOADING. lambda's /accounts quota silently truncated this exact checkpoint
  # twice: the job reported COMPLETED rc=0 while the distcp write had died on
  # "OSError: [Errno 122] Disk quota exceeded". A short .metadata or a thin shard directory is the
  # only visible trace, and pushing one to weka means the eval reads an untrained model and
  # returns a number instead of an error.
  MD="$STAGE/$NAME/model_and_optim/.metadata"
  NSHARD=$(ls "$STAGE/$NAME/model_and_optim" 2>/dev/null | wc -l)
  MDSIZE=$(stat -c %s "$MD" 2>/dev/null || echo 0)
  echo "[$T] .metadata=${MDSIZE}B shards=$NSHARD"
  if [ ! -f "$STAGE/$NAME/config.json" ] || [ "$MDSIZE" -lt 700000 ] || [ "$NSHARD" -lt 120 ]; then
    echo "[$T] FATAL: incomplete checkpoint (config.json / .metadata / shard count) -- NOT uploading"
    continue
  fi

  echo "=== [$T] $(date '+%F %T') s3 sync -> $S3/$NAME"
  AWS_PROFILE=S3 aws s3 sync "$STAGE/$NAME/" "$S3/$NAME/" --only-show-errors || { echo "[$T] FATAL s3"; continue; }
  echo "=== [$T] $(date '+%F %T') S3 DONE: $S3/$NAME"
done
echo "=== relay finished $(date '+%F %T') ==="
