#!/bin/bash
# Score the EXISTING CTC contradiction checkpoints on the IID (realistic-mode) ladder.
#
# Nothing is retrained. Both arms are the same checkpoints that produced the `both`-mode numbers:
#   dense   ctc-4b-contradiction-full  ->  0.843 / 0.803 / 0.744 / 0.664 / 0.559   (n 44/92/187/379/762)
#   chunked ctc-4b-contradiction-cmix  ->  0.402 / 0.331 / 0.277 / 0.234 / 0.191
# The four upper rungs here hold n identical to that ladder, so any movement is attributable to the
# perturbation mode alone (train realistic, jaccard 0.306; old eval both, 0.375). The bottom rung
# moves n=44 -> 56 because 44 sat below the training minimum of 52.
#
# RUNG_TASK=contradiction_iid, so the driver reads eval_rungs/contradiction_iid/rung_*.jsonl and
# writes to ctc_suite_vllm_results{,_chunked}/contradiction_iid/.
#SBATCH --job-name=contra-iid-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/contra_iid_eval_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

VDIR=/data/prasann/repo/OLMo-core/debug/ctc_vllm_validation/node_local
RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid
mkdir -p /data/prasann/joblogs

echo "=== [contra-iid-eval] HOST=$(hostname) START=$(date '+%F %T') ==="

# Pull the rungs from S3 if this node does not have them yet (they are generated on mooney).
if [ ! -d "$RUNGDIR" ] || [ -z "$(ls -A "$RUNGDIR" 2>/dev/null)" ]; then
  echo "[stage] syncing rungs from S3 $(date '+%F %T')"
  export HOME=/data/prasann/home
  export AWS_PROFILE=S3
  mkdir -p "$RUNGDIR"
  aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/contra_iid_rungs "$RUNGDIR" \
    --only-show-errors || { echo "!!! s3 sync FAILED"; exit 1; }
fi
ls -la "$RUNGDIR" || { echo "!!! rungs not staged at $RUNGDIR"; exit 1; }
[ -n "$(ls -A "$RUNGDIR"/rung_*.jsonl 2>/dev/null)" ] || { echo "!!! no rung files"; exit 1; }

MODE_ARG="${MODE:-dense}"
RUNGS="${RUNGS:-2560 4096 8192 16384 32768}"

# The driver derives BOTH the serving copy and the output dir from $TASK, so running this as
# TASK=contradiction would write grade_4096.json et al. straight over the existing `both`-mode
# results (and push over their S3 roster entries). Run as TASK=contradiction_iid instead -- which
# needs a serving-copy alias, since no ctc-4b-contradiction_iid-* checkpoint exists. EVAL_TASK
# stays `contradiction` so the prefill builder and grader use the contradiction scorer.
# Same checkpoint, separate results namespace.
ln -sfn /data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-contradiction-full \
        /data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-contradiction_iid-full
ln -sfn /data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction-cmix \
        /data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-contradiction_iid-cmix

if [ "$MODE_ARG" = "chunked" ]; then
  TASK=contradiction_iid EVAL_TASK=contradiction MODE=chunked RUNG_TASK=contradiction_iid \
    RUNGS="$RUNGS" KEEP_RESPONSES=1 MAX_NEW_TOKENS=512 \
    bash "$VDIR/gpu_eval_task_chunked.sh"
  RESDIR=/data/prasann/ctc_suite_vllm_results_chunked/contradiction_iid
else
  TASK=contradiction_iid EVAL_TASK=contradiction RUNG_TASK=contradiction_iid \
    RUNGS="$RUNGS" KEEP_RESPONSES=1 MAX_NEW_TOKENS=512 \
    bash "$VDIR/gpu_eval_task.sh"
  RESDIR=/data/prasann/ctc_suite_vllm_results/contradiction_iid
fi

echo "=== IID ladder ($MODE_ARG) ==="
for R in $RUNGS; do
  F="$RESDIR/grade_$R.json"
  if [ -f "$F" ]; then
    python -c "
import json;d=json.load(open('$F'));m=d.get('metrics',{})
print(f\"  $R f1={d.get('metric_value'):.4f} prec={m.get('precision',0):.3f} rec={m.get('recall',0):.3f} em={m.get('exact_match',0):.3f} parse={m.get('parse_rate',0):.2f} eval_size={d.get('eval_size')}\")"
  else
    echo "  $R MISSING"
  fi
done
echo "=== [contra-iid-eval] DONE $(date '+%F %T') ==="
