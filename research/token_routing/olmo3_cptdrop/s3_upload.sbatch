#!/bin/bash
# Mirror the olmo3_cptdrop shards and converted bases from the build node to the S3 hub
# (s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/) for weka (Beaker) and Lambda node /tmp.
#   sbatch -w <build node> research/token_routing/olmo3_cptdrop/s3_upload.sbatch
#SBATCH --job-name=o3cd-s3up
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -euo pipefail
R=/data/prasann/olmo3_cptdrop
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cp "$REPO/research/token_routing/olmo3_cptdrop/s3_sync.py" "$R/s3_sync.py"
cd "$R"
OPS_RUN_ID="o3cd-s3up-${SLURM_JOB_ID}" OPS_EXPERIMENT=olmo3_cptdrop OPS_KIND=data \
OPS_EXPECTED_MINUTES=60 OPS_STALL=1800 \
bash "$REPO/recipes/env/ops_job.sh" bash -c "
  set -e
  for d in dev32k short4k train32k c8k; do $PY -u s3_sync.py up $R/data/\$d data/\$d; done
  for b in stage2-step47684 stage3-step11921; do $PY -u s3_sync.py up $R/base/\$b base/\$b; done"
