#!/usr/bin/env bash
# Second half of the staging two-step: S3 -> weka, so Beaker jobs can read the checkpoints.
# Pushing to S3 alone does nothing for a Beaker job; its absence shows up as a MISSING path at step 0.
set -euo pipefail
S3=s3://ai2-llm/checkpoints/prasanns/ctc_hybridish_sft
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft
/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry run --name sync-hybridish-sft-ckpts --workspace ai2/flex2 --budget ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --weka oe-training-default:/weka/oe-training-default \
  --cpus 4 --priority urgent --no-python --allow-dirty --timeout 0 --yes \
  -- bash -c "
set -euo pipefail
for c in sft_4to1_ml_hf sft_7to1_ml_hf; do
  mkdir -p '$WEKA'/\$c
  aws s3 sync '$S3'/\$c '$WEKA'/\$c --only-show-errors
  echo \"[weka] \$c: \$(du -sh '$WEKA'/\$c | cut -f1)\"
  python -c \"import json;print('  scalable_softmax=',json.load(open('$WEKA/'+'\$c'+'/config.json')).get('scalable_softmax'))\"
done"
