#!/bin/bash
# Sanity-check the v3 (iid) contradiction rungs by scoring an EXISTING checkpoint on them.
#
# WHAT THIS DOES AND DOES NOT PROVE.
# The hub's own 5-task checkpoints are weka-only (not on S3, not on any node here), so this scores
# `ctc-4b-contradiction-full` -- Qwen3.5-4B, CTC suite. That is a different model family from the
# hub's Qwen3-4B runs, BUT both trained on the SAME contradiction source jsonl
# (contradiction_train_pubmed_realistic_n50-950_k3.jsonl; the hub shard's metadata names it, the CTC
# shard's instance count 19366+748 dropped = 20114 matches the file exactly). So:
#   PROVES     -- the v3 rung FILES are correctly built and iid with that training distribution.
#   DOES NOT   -- say anything about the hub's Qwen3-4B checkpoint specifically.
#
# EXPECTED. On the CTC iid ladder this checkpoint scored 0.9843 at n=92 and 0.9760 at n=187, against
# 0.803 / 0.744 on the `both`-mode ladder. v3's n=100 and n=190 are a few docs from those, so a
# correct v3 build should land ~0.98 at n=100 and ~0.97 at n=190. Anything near the `both` numbers
# means the rungs are wrong, not that the finding failed.
#
# Rungs are labelled by n, not by a token budget: the hub's own "2k" label is n=100 which measures
# ~4450 tokens, and propagating that mislabel into a new artifact would be repeating a known bug.
#SBATCH --job-name=contra-v3-verify
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/contra_v3_verify_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

VDIR=/data/prasann/repo/OLMo-core/debug/ctc_vllm_validation/node_local
RUNGDIR=/data/prasann/ctc_suite_staged/eval_rungs/contradiction_v3
mkdir -p /data/prasann/joblogs "$RUNGDIR"

echo "=== [contra-v3-verify] HOST=$(hostname) START=$(date '+%F %T') ==="

# Pull the published v3 bundle and rename to the driver's rung_<label>.jsonl convention.
export HOME=/data/prasann/home
export AWS_PROFILE=S3
STAGE=/data/prasann/contra_v3_dl
mkdir -p "$STAGE"
aws s3 sync s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3/contra "$STAGE" \
  --only-show-errors || { echo "!!! s3 sync FAILED"; exit 1; }
for n in 100 190 385 765; do
  src="$STAGE/contradiction_eval_pubmed_realistic_n${n}_k3.jsonl"
  [ -f "$src" ] || { echo "!!! missing n=$n in the published bundle"; exit 1; }
  cp -f "$src" "$RUNGDIR/rung_${n}.jsonl"
done
ls -la "$RUNGDIR"

# Separate results namespace so this cannot overwrite the contradiction_iid / contradiction results.
ln -sfn /data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-contradiction-full \
        /data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-contradiction_v3-full

TASK=contradiction_v3 EVAL_TASK=contradiction RUNG_TASK=contradiction_v3 \
  RUNGS="100 190 385 765" KEEP_RESPONSES=1 MAX_NEW_TOKENS=512 \
  bash "$VDIR/gpu_eval_task.sh"

echo ""
echo "=== v3 ladder (dense, ctc-4b-contradiction-full) ==="
echo "    reference: CTC iid ladder n=92 -> 0.9843, n=187 -> 0.9760, n=379 -> 0.9652, n=762 -> 0.9463"
echo "    reference: both-mode ladder  n=92 -> 0.803,  n=187 -> 0.744,  n=379 -> 0.664,  n=762 -> 0.559"
for R in 100 190 385 765; do
  F=/data/prasann/ctc_suite_vllm_results/contradiction_v3/grade_$R.json
  if [ -f "$F" ]; then
    python -c "
import json;d=json.load(open('$F'));m=d.get('metrics',{})
print(f\"  n=$R  f1={d.get('metric_value'):.4f} em={m.get('exact_match',0):.3f} parse={m.get('parse_rate',0):.2f} eval_size={d.get('eval_size')}\")"
  else
    echo "  n=$R MISSING"
  fi
done
echo "=== [contra-v3-verify] DONE $(date '+%F %T') ==="
