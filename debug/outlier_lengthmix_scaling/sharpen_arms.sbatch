#!/bin/bash
#SBATCH --job-name=lm-sharpen-arms
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/lm_sharpen_arms_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
$PY -u /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/outlier_lengthmix_scaling/sharpen_build.py || exit 1
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms
for A in qmix_s64M qmix_s160M qmix_s320M q32k_16000; do
  aws s3 sync "/data/prasann/qdmatch_lengthmix/arms_tokenized/$A" "$S3/$A" --only-show-errors || exit 1
done
for A in nmix_s16M nmix_s32M nmix_s48M nqD32k_4000 nqD64k_2000; do
  aws s3 sync "/data/prasann/nq_lengthmix/arms_tokenized/$A" "$S3/$A" --only-show-errors || exit 1
done
aws s3 sync "/data/prasann/outlier_lengthmix/arms_tokenized/p32k_4000" "$S3/p32k_4000" --only-show-errors || exit 1
echo "=== sharpen arms DONE $(date '+%F %T') ==="
