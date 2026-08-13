#!/bin/bash
# ROUND 5e (2026-08-13): CONTENTION CONTROL. sacct shows a co-tenant job holding
# gres/gpu=8 on cubbins since 01:50 while every round-A/B/C/D bench ran there — the
# c1_varlen "regression" (134.9s) reproduced at 15.4s under d0 with the same config,
# and the a4/a5 "fp+cache interaction" may be the same phantom. Rerun the key arms
# 3x back-to-back, report ALL times, judge by MIN. nvidia-smi logs who shares my GPU.
#
#SBATCH --job-name=chunk-varlen-rerun
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/chunk_varlen_rerun_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

ACC=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
R2=/data/prasann/repo_r2/OLMo-core
mkdir -p "$R2/src" "$R2/debug/ctc_vllm_validation" /data/prasann/joblogs
rsync -a --delete /data/prasann/repo/OLMo-core/src/ "$R2/src/"
rsync -a --delete /data/prasann/repo/OLMo-core/debug/ctc_vllm_validation/ "$R2/debug/ctc_vllm_validation/"
cp "$ACC/src/corpus_reasoning/lib/vllm_chunked_patch.py" "$R2/src/corpus_reasoning/lib/vllm_chunked_patch.py"
cp "$ACC/debug/ctc_vllm_validation/run_vllm_eval.py"     "$R2/debug/ctc_vllm_validation/run_vllm_eval.py"

VDIR=$R2/debug/ctc_vllm_validation
GDIR=$VDIR/general
source /data/prasann/node_local_env.sh 2>/dev/null || true
export HOME=/data/prasann/home
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONPATH=$R2/src
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
GRADE_PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python

WORK=/data/prasann/chunk_smallscale
mkdir -p "$WORK"
SERVE=/data/prasann/ctc_suite/vllm_serving/ctc-smoke-contra-3ep
PREFILLS=$WORK/prefills_2560_first100.json

echo "=== GPU assigned: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>/dev/null | head -20

SUMMARY=$WORK/SUMMARY_varlen_rerun.tsv
echo -e "arm\trep\tgen_s\tsteps\tvarlen\tfallback\tcotenant_procs" > "$SUMMARY"

run_arm() {
  local ARM=$1 REP=$2; shift 2
  local RESP="$WORK/resp_rr_${ARM}_${REP}.json"
  # count compute processes on our GPU right before the run
  local COPROCS
  COPROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  echo ""
  echo "=== [rerun] ARM=$ARM rep=$REP cotenant_procs=$COPROCS $(date '+%F %T') ==="
  env "$@" $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode chunked --task contradiction \
    --max-new-tokens 512 --max-model-len 3072 --gpu-mem-util 0.85 \
    --out "$RESP" 2>&1 | grep -E "gen_seconds|generated |patch debug state|Traceback|RuntimeError" | tail -6
  $GRADE_PY - "$ARM" "$REP" "$WORK" "$SUMMARY" "$COPROCS" <<'PYEOF'
import json, sys
arm, rep, work, summary, cop = sys.argv[1:6]
r = json.load(open(f"{work}/resp_rr_{arm}_{rep}.json"))
pd = r.get("patch_debug") or {}
row = (f"{arm}\t{rep}\t{r['gen_seconds']:.1f}\t{pd.get('calls','-')}\t"
       f"{pd.get('varlen','-')}\t{pd.get('fallback','-')}\t{cop}")
open(summary, "a").write(row + "\n")
print(f"RESULT {row}")
PYEOF
}

STOP="CHUNK_EXTRA_STOP=]]"
BASE="CHUNK_MAX_BATCHED_TOKENS=4096 CHUNK_MAX_NUM_SEQS=16 CHUNK_SEQ_HEADROOM=18"

for rep in 1 2 3; do
  run_arm c0_simple "$rep" "$STOP" $BASE
  run_arm c1_varlen "$rep" "$STOP" $BASE CHUNK_VARLEN_PREFILL=1
  run_arm c2_varlen64 "$rep" "$STOP" CHUNK_MAX_BATCHED_TOKENS=4096 CHUNK_MAX_NUM_SEQS=64 \
                       CHUNK_SEQ_HEADROOM=66 CHUNK_VARLEN_PREFILL=1
done

echo ""
echo "=== [rerun] SUMMARY (judge by MIN per arm) ==="
column -t -s $'\t' "$SUMMARY"
echo "=== [rerun] DONE $(date '+%F %T') ==="
