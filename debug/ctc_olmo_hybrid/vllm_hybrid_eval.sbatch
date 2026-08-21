#!/bin/bash
# vLLM ladder eval for an **Olmo-Hybrid-7B** (3:1 GDN:full-attention) checkpoint.
#
# The hybrid twin of vllm_pair_eval_anynode.sbatch, which covers only the full-attention (noswa)
# arm. Same three-step pipeline per rung (prefills -> vLLM -> grade) and the same rung tree; the
# only differences are the exporter (export_hybrid_to_hf.py, which emits model_type=olmo_hybrid
# instead of mapping onto the Olmo2 class) and the OLMo marker ids, which are passed explicitly.
#
#   EVAL_TASK=xabsence LADDER=xabsence SCALE=7b-hybrid \
#   CKPT=/data/prasann/ctc_pair_eval_ckpts/<run> \
#   [RUNGS="2048 4096 8192 16384 32768"] [DEST=...] \
#   sbatch -w <node> --qos=preemptive_high debug/ctc_olmo_hybrid/vllm_hybrid_eval.sbatch
#
# ⚠ CKPT must already be on the target node's /data. Unlike the noswa script this one does not
# stage from another host: a 28GB distcp pulled over /net is the NFS deadlock local_cluster.md
# warns about. Relay weka -> S3 -> node-local first (relay_and_eval_noswa_local.sh's phase 2 is
# the pattern), or pick the node that already holds the checkpoint.
#
# ⚠ EVAL_TASK is the TASK_CFG key and it is not cosmetic: `absence` and `xabsence` share a config
# row but build different prompts (evaluate.build_prompt branches on the name), so using the wrong
# one silently mis-prompts every example. Match the key the native run recorded.
#SBATCH --job-name=vllm-hybrid-eval
#SBATCH --partition=jsteinhardt
#SBATCH --account=site
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/vllm_hybrid_eval_%j.log

set -uo pipefail
: "${EVAL_TASK:?}" "${LADDER:?}" "${CKPT:?}" "${SCALE:?}"
RUNGS="${RUNGS:-2048 4096 8192 16384 32768}"

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CPUENV=/data/prasann/conda/envs/corpus-reasoning-olmo
VLLMPY=/data/prasann/ctc_vllm_venv/bin/python
# Both interpreters MUST be node-local; the NFS copies work, just 100x slower, which reads as a
# hung job (local_cluster.md).
case "$CPUENV" in /data/*) ;; *) echo "FATAL: cpu env not node-local"; exit 2;; esac
[ -x "$CPUENV/bin/python" ] || { echo "FATAL: no cpu env on $(hostname) /data"; exit 2; }
[ -x "$VLLMPY" ] || { echo "FATAL: no vllm venv on $(hostname) /data"; exit 2; }

export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH   # GDN JIT
export HOME=/data/prasann/home_cache TMPDIR=/data/prasann/tmp FLASHINFER_CACHE_DIR=/data/prasann/flashinfer
mkdir -p "$TMPDIR" "$FLASHINFER_CACHE_DIR" "$HOME" /data/prasann/hf_exports

[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "FATAL: no distcp at $CKPT"; exit 3; }

# ── HF export on node-local /data (skipped when already present) ──
HF_MODEL=${HF_MODEL:-/data/prasann/hf_exports/$(basename "$CKPT")-hf}
if [ ! -f "$HF_MODEL/model.safetensors" ]; then
  echo "=== exporting $CKPT -> $HF_MODEL $(date '+%F %T') ==="
  "$CPUENV/bin/python" -u "$REPO/debug/ctc_olmo_hybrid/export_hybrid_to_hf.py" \
    "$CKPT" "$HF_MODEL" --max-seq-len 40960 --dtype bfloat16 \
    || { echo "FATAL: HF export failed"; exit 4; }
fi

# The decode budget is per task (oolong wants 256, most want 200) and it is NOT cosmetic: a
# vLLM run that truncates where the native run did not is a scoring difference, not a speedup.
# Read it from the same TASK_CFG row the native evaluator uses rather than hardcoding.
MAX_NEW=$("$CPUENV/bin/python" -c "
from corpus_reasoning.eval.eval_lc_native_docchunk import TASK_ALIASES, TASK_CFG
t='$EVAL_TASK'; print(TASK_CFG[TASK_ALIASES.get(t, t)]['max_new'])")
[ -n "$MAX_NEW" ] || { echo "FATAL: could not resolve max_new for $EVAL_TASK"; exit 5; }
echo "[cfg] task=$EVAL_TASK max_new=$MAX_NEW"

TOK=/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk
RUNGDIR=/scratch/users/prasann/ctc_suite_staged/eval_rungs/$LADDER
WORK=/data/prasann/hybrid_vllm_work/${LADDER}__${SCALE}
DEST=${DEST:-/data/prasann/ctc_olmo3ns_results/${LADDER}/${SCALE}_vllm_full}
mkdir -p "$WORK" "$DEST"

for R in $RUNGS; do
  EV="$RUNGDIR/rung_${R}.jsonl"
  [ -f "$EV" ] || { echo "skip rung $R (no file)"; continue; }
  echo "=== rung $R $(date '+%F %T') ==="
  T0=$(date +%s)
  if [ ! -f "$WORK/prefills_${R}.jsonl" ]; then
    "$CPUENV/bin/python" -u "$REPO/debug/ctc_vllm_validation/build_prefills_generic.py" \
      --task "$EVAL_TASK" --eval-data "$EV" --tokenizer "$TOK" \
      --doc-start-id 100266 --doc-end-id 100267 --eos-token-id 100257 \
      --out "$WORK/prefills_${R}.jsonl" || { echo "prefills rc=$? rung $R"; continue; }
  fi
  "$VLLMPY" -u "$REPO/debug/ctc_vllm_validation/run_vllm_eval_generic.py" \
    --hf-model "$HF_MODEL" --prefills "$WORK/prefills_${R}.jsonl" \
    --mode full --max-new-tokens "$MAX_NEW" --max-model-len 40960 \
    --out "$WORK/responses_${R}.jsonl" || { echo "vllm rc=$? rung $R"; continue; }
  "$CPUENV/bin/python" -u "$REPO/debug/ctc_vllm_validation/grade_responses_generic.py" \
    --responses "$WORK/responses_${R}.jsonl" --eval-data "$EV" --task "$EVAL_TASK" \
    --out "$DEST/rung_${R}.vllm.json" || { echo "grade rc=$? rung $R"; continue; }
  echo "rung $R wall-clock $(( $(date +%s) - T0 ))s"
  "$CPUENV/bin/python" -c "
import json,sys; print('  metrics:', json.load(open(sys.argv[1]))['metrics'])" "$DEST/rung_${R}.vllm.json"
done
echo "=== VLLM-HYBRID-EVAL DONE $(date '+%F %T') -> $DEST ==="
