#!/bin/bash
#SBATCH --job-name=stage-dense-4b
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/stage_dense_4b_%j.log
set -uo pipefail
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/_transfer
DST=/data/prasann/ctc_suite/ckpts_4b

# logical_name -> exact S3 dirname (dense/-full arm; harvested by ctc-harvest-all-4b).
declare -A CK=(
  [absence_gutenberg]=ctc-4b-absence_gutenberg-full-07200501-20260720T050249-0700
  [cycle]=ctc-4b-cycle-full-07200501-20260720T050324-0700
  [fiqa]=ctc-4b-fiqa-full-07200501-20260720T050213-0700
  [grouping]=ctc-4b-grouping-full-20260719T225805-0700
  [grouping_labeled]=ctc-4b-grouping_labeled-full-07200501-20260720T050236-0700
  [helmet_qa]=ctc-4b-helmet_qa-full-07200501-20260720T050154-0700
  [helmet_summ]=ctc-4b-helmet_summ-full-07200501-20260720T050201-0700
  [mathmatch]=ctc-4b-mathmatch-full-07200501-20260720T050312-0700
  [msmarco]=ctc-4b-msmarco-full-07200501-20260720T050219-0700
  [niah]=ctc-4b-niah-full-07200501-20260720T050152-0700
  [nq]=ctc-4b-nq-full-07200501-20260720T050342-0700
  [obliq_retrieval]=ctc-4b-obliq_retrieval-full-07200501-20260720T050336-0700
  [outlier_amzn]=ctc-4b-outlier_amzn-full-07200501-20260720T050230-0700
  [outlier]=ctc-4b-outlier-full-20260719T225805-0700
  [qdmatch_hpqa]=ctc-4b-qdmatch_hpqa-full-07200501-20260720T050300-0700
  [reorder]=ctc-4b-reorder-full-07200501-20260720T050317-0700
  [rerank]=ctc-4b-rerank-full-07200501-20260720T050225-0700
  [scifact]=ctc-4b-scifact-full-07200501-20260720T050207-0700
  [strmatch]=ctc-4b-strmatch-full-07200501-20260720T050254-0700
  [textgroups]=ctc-4b-textgroups-full-07200501-20260720T050242-0700
)

echo "=== HOST=$(hostname) START=$(date '+%F %T') staging ${#CK[@]} dense ckpts ==="
FAIL=0
for name in "${!CK[@]}"; do
  src="$S3/${CK[$name]}"
  dst="$DST/ctc-4b-${name}-full"
  if [ -f "$dst/model_and_optim/.metadata" ]; then echo "  [$name] already staged, skip"; continue; fi
  echo "=== [$name] $src -> $dst  $(date '+%T') ==="
  mkdir -p "$dst"
  aws s3 sync "$src/model_and_optim" "$dst/model_and_optim" --only-show-errors || FAIL=1
  aws s3 cp  "$src/config.json"      "$dst/config.json"      --only-show-errors || FAIL=1
done
echo "=== STAGE DONE $(date '+%F %T') FAIL=$FAIL ==="
ls -d "$DST"/*/ 2>/dev/null