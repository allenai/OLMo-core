#!/bin/bash
# Launch olmo3_cptdrop arm families on Beaker jupiter (8 x H100 each) via gantry, from the PUSHED HEAD.
#   bash research/token_routing/olmo3_cptdrop/launch_beaker.sh C120 C240 C480 IFL
# Prints one "FAMILY EXPERIMENT_ID" line per launch; register each with ops (cluster beaker).
# --allow-dirty: the shared checkout carries other sessions' uncommitted files; gantry ships the pushed HEAD.
set -euo pipefail
GANTRY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry
for FAM in "$@"; do
  "$GANTRY" run --name "o3cd-$FAM-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
    --cluster ai2/jupiter-cirrascale-2 --gpus 8 --shared-memory 256GiB --priority urgent --min-runtime 1h \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install true \
    --weka oe-training-default:/weka/oe-training-default \
    --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
    --env-secret WANDB_API_KEY=PRASANNS_WANDB_API_KEY \
    --allow-dirty --timeout 0 --yes -- bash research/token_routing/olmo3_cptdrop/beaker_job.sh "$FAM" 2>&1 | tee /dev/stderr \
    | grep -oE "beaker.org/ex/[0-9A-Z]+" | head -1 | sed "s|beaker.org/ex/|$FAM |"
done
