#!/bin/bash
# Pull the study's small artifacts from weka to S3 (gantry job) so collect_results.py can read them
# here: every fs-* run's flops.json + provenance.json + config.json (no weights) and every fs-*
# multirung eval JSON. Incremental; rerun any time.   bash debug/flop_scaling/harvest_to_s3.sh
set -uo pipefail
CK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
EV=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results
S3=s3://ai2-llm/checkpoints/prasanns/flop_scaling/harvest
CMD='AWS=$(command -v aws || ls /opt/conda/bin/aws 2>/dev/null || true); '
CMD+='if [ -z "$AWS" ]; then pip install -q awscli && AWS=$(command -v aws); fi; '
CMD+='[ -n "$AWS" ] || { echo FATAL_NO_AWSCLI; exit 127; }; '
CMD+='mkdir -p ~/.aws && echo "$AWS_CREDS" > ~/.aws/credentials && echo "$AWS_CFG" > ~/.aws/config; export AWS_PROFILE=S3; '
CMD+="mkdir -p /tmp/h/runs /tmp/h/evals; for d in $CK/fs-*/; do r=\$(basename \$d); mkdir -p /tmp/h/runs/\$r; for f in flops.json provenance.json config.json; do [ -f \$d/\$f ] && cp \$d/\$f /tmp/h/runs/\$r/; done; done; "
CMD+="cp $EV/fs-*multirung*.json /tmp/h/evals/ 2>/dev/null; echo runs=\$(ls /tmp/h/runs | wc -l) evals=\$(ls /tmp/h/evals | wc -l); "
CMD+="\$AWS s3 sync /tmp/h/ $S3/ --only-show-errors && echo HARVEST_DONE"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
exec gantry run --name "fs-harvest-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$CMD"
