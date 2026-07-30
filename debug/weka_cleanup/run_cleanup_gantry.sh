#!/bin/bash
# Two-stage weka checkpoint cleanup, run on a weka node via gantry (weka isn't mounted at
# Berkeley; 0 GPUs so it schedules immediately regardless of jupiter congestion).
#
#   bash run_cleanup_gantry.sh plan  phase1   # discovery: writes an explicit manifest to S3
#   bash run_cleanup_gantry.sh apply phase1   # deletes ONLY the exact paths in that manifest
#
# Stage 1 (plan) never deletes. Stage 2 (apply) does no globbing -- it reads literal paths from
# the manifest and re-validates each against the filesystem before removing it. The manifest is
# persisted to S3, so the exact list of what was removed stays auditable after the fact.
#
# phase1 = ctc_suite/ckpts, MODE=modelonly -- drops the full training checkpoints (model+optimizer
#          +train state) and keeps <run>/model_and_optim/, the model-only final save that eval
#          loads. A run without a valid final model is refused, so nothing ever loses its weights.
# phase2 = top-level per-run dirs, MODE=keepfinal -- these have no model-only save, so the highest
#          step<N>/ is kept and only earlier ones are listed.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

STAGE="${1:?usage: $0 plan|apply phase1|phase2}"
PHASE="${2:?usage: $0 plan|apply phase1|phase2}"
PRASANNS=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
S3M="s3://ai2-llm/checkpoints/prasanns/_inventory/manifest_${PHASE}.txt"

case "$PHASE" in
  phase1) ROOT="$PRASANNS/ctc_suite/ckpts"; MODE=modelonly ;;
  phase2) ROOT="$PRASANNS";                 MODE=keepfinal ;;
  *) echo "FATAL: phase must be phase1 or phase2" >&2; exit 2 ;;
esac
case "$STAGE" in plan|apply) ;; *) echo "FATAL: stage must be plan or apply" >&2; exit 2 ;; esac

# Ship the workers inline (base64) rather than from the cloned repo: gantry clones the last PUSHED
# commit, so an unpushed edit would silently run a stale version.
B64P="$(base64 -w0 debug/weka_cleanup/plan_deletions.sh)"
B64A="$(base64 -w0 debug/weka_cleanup/apply_deletions.sh)"

COMMON='
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG"   > ~/.aws/config
command -v aws >/dev/null 2>&1 || python -m pip install -q awscli
AWS=$(command -v aws)
echo "B64P_SUB" | base64 -d > /tmp/plan_deletions.sh
echo "B64A_SUB" | base64 -d > /tmp/apply_deletions.sh
echo "### free space BEFORE ###"; df -h /weka/oe-training-default | tail -1
'

if [ "$STAGE" = plan ]; then
  WORK="$COMMON"'
export ROOT="ROOT_SUB" MODE="MODE_SUB" OUT=/tmp/manifest.txt
bash /tmp/plan_deletions.sh
echo "### manifest head ###"; head -5 /tmp/manifest.txt
AWS_PROFILE=S3 "$AWS" s3 cp /tmp/manifest.txt "S3M_SUB" --only-show-errors && echo "manifest -> S3M_SUB"
'
else
  WORK="$COMMON"'
AWS_PROFILE=S3 "$AWS" s3 cp "S3M_SUB" /tmp/manifest.txt --only-show-errors || exit 1
echo "### applying manifest ($(wc -l < /tmp/manifest.txt) paths) ###"
export MANIFEST=/tmp/manifest.txt APPLY=1 FRESH_MIN="FRESH_SUB" MODE="MODE_SUB"
bash /tmp/apply_deletions.sh
echo "### free space AFTER ###"; df -h /weka/oe-training-default | tail -1
'
fi

# In-flight window for the apply stage. Default 90m matches the plan stage. Lower it ONLY to
# resume an interrupted apply, where the plan stage has already certified the runs idle over the
# full 90m and the only writer since has been this tooling itself.
FRESH_MIN="${FRESH_MIN:-90}"

WORK="${WORK//FRESH_SUB/$FRESH_MIN}"
WORK="${WORK//B64P_SUB/$B64P}"; WORK="${WORK//B64A_SUB/$B64A}"
WORK="${WORK//ROOT_SUB/$ROOT}"; WORK="${WORK//MODE_SUB/$MODE}"; WORK="${WORK//S3M_SUB/$S3M}"

echo "stage=$STAGE phase=$PHASE root=$ROOT mode=$MODE"
if [ "$STAGE" = apply ]; then
  echo ">>> APPLY: this DELETES the paths in $S3M. Ctrl-C within 10s to abort."; sleep 10
fi

gantry run --name "weka-$STAGE-$PHASE-$(date +%m%d-%H%M%S)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
