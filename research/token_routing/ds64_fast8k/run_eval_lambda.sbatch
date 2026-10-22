#!/bin/bash
#SBATCH --job-name=f8k-eval-lambda
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/logs/f8k_eval_%x_%j.log
# fast8k ladder eval ON LAMBDA (tokdrop_lr): the same on-node runner the Beaker evals use
# (recipes/eval/legacy_native/run_beaker_multirung_eval.sh, docchunk ladder, fast8k's arguments), pointed at
# node-local /tmp checkpoints and eval files. Submit with explicit gres and node, e.g.
#   sbatch --gres=gpu:A100:2 --qos=preemptive_high -w lambda-hyperplane02 --job-name=<run>-ev<task> \
#     --export=ALL,RUN=<run>,TASK=<contradiction|nq|outlier|oolong>,ROOT=/tmp/prasann/tdlr run_eval_lambda.sbatch
# Results: $ROOT/ckpts/<run>/eval/<key>_multirung.json (+ predictions), copied to $LROOT/eval_results/.
set -o pipefail   # no -u: it kills the conda-pack activate
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
REPO=$LROOT/OLMo-core
ROOT="${ROOT:-/tmp/prasann/tdlr}"
RUN="${RUN:?set RUN}"; TASK="${TASK:?set TASK}"
NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-2}}"
case "$TASK" in
  contradiction) KEY=contra; R8=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl
                 R2=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl ;;
  nq)      KEY=nq;      R8=outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl; R2=outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl ;;
  outlier) KEY=outlier; R8=outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl; R2=outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl ;;
  oolong)  KEY=oolong;  R8=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl
           R2=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl ;;
  *) echo "unknown TASK=$TASK"; exit 2 ;;
esac
RUN_DIR=$ROOT/ckpts/$RUN
CKPT=$(for d in $(find "$RUN_DIR" -name .metadata -path '*model_and_optim*' 2>/dev/null); do dirname "$(dirname "$d")"; done | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "FATAL: no complete model_and_optim under $RUN_DIR"; exit 3; }
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
source "$ENV/bin/activate"; conda-unpack 2>/dev/null || true
export PATH="$ENV/bin:$PATH"   # torchrun from the node-visible env, never a system one
export HF_HOME="${HF_HOME_OVERRIDE:-/accounts/projects/sewonm/prasann/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p "$TRITON_CACHE_DIR"
# the runner cd's into $WEKA_LLM/checkpoints/prasanns/_eval_bundle (rung files below are absolute)
mkdir -p "$ROOT/weka_stub/checkpoints/prasanns/_eval_bundle"
# scipy/scikit-learn (outlier/rerank metrics) were installed offline from wheels on 2026-10-20;
# lambda has no pypi, so the runner's own pip fallback cannot fetch them
"$ENV/bin/python" -c "import scipy, sklearn" || { echo "FATAL: scipy/sklearn missing from $ENV"; exit 4; }
# Batched greedy decode (eval_lc_native_docchunk_ladder.py --gen-batch, 2026-10-22): validated against
# the bs=1 loop on contradiction@8k (f1 0.641 vs 0.639 over 500 rows, 22/24 identical generations,
# 2.6x faster at 98k tokens/batch). LADDER_GEN_BATCH=1 restores the bs=1 loop.
export LADDER_GEN_BATCH="${LADDER_GEN_BATCH:-32}" LADDER_GEN_BATCH_TOKENS="${LADDER_GEN_BATCH_TOKENS:-196608}"
FILES="{\"$TASK\": {\"8k\": \"$ROOT/eval/$R8\", \"2k\": \"$ROOT/eval/$R2\"}}"
echo "=== LAMBDA fast8k eval host=$(hostname) RUN=$RUN TASK=$TASK CKPT=$CKPT NGPU=$NGPU $(date '+%F %T') ==="
cd "$REPO"
RUN="$RUN" TASK="$KEY" VARIANT=docchunk STEP='' CKPT="$CKPT" EVAL_OUT_DIR="$RUN_DIR/eval" \
  PROMPT_FORMAT=chat QUERY_POSITION=after MAX_TEST=500 MAX_LENGTH=20000 BATCH_SIZE=8 NGPU="$NGPU" \
  LADDER_XLONG=0 COT_MODE=none LADDER_VERSION=v2 TOKENIZER=Qwen/Qwen3.5-0.8B-Base \
  DC_RUNG_FILES="$FILES" DC_RUNGS=8k,2k WEKA_LLM="$ROOT/weka_stub" RESULTS="$ROOT/eval_results" REPO="$REPO" \
  bash recipes/eval/legacy_native/run_beaker_multirung_eval.sh
rc=$?
# a ladder whose rung files were not found writes an EMPTY result with rc=0 (2026-10-20: the
# override was keyed "contra" while the ladder looks up "contradiction"); fail loudly instead
"$ENV/bin/python" - "$RUN_DIR/eval/${KEY}_multirung.json" <<'PY' || rc=5
import json, sys
d = json.load(open(sys.argv[1]))
n = d.get("_meta", {}).get("eval_size") or {}
print("eval_size per rung:", n)
sys.exit(0 if n and all(v for v in n.values()) else 1)
PY
mkdir -p "$LROOT/eval_results/$RUN" && cp "$RUN_DIR"/eval/*.json "$RUN_DIR"/eval/*_responses.jsonl "$LROOT/eval_results/$RUN/" 2>/dev/null
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
