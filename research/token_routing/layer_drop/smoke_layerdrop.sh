#!/bin/bash
# Layer-drop wall-clock smoke (2026-10-22): Qwen3.5-4B repaired base, fast8k mt4s488 shard, 2 GPUs,
# the tokdrop_lr geometry (seq 10240, gb 16, micro 2, unpacked, no compile). Arms run back to back
# in ONE allocation so they share the node:
#   dense | dense + LayerDrop p=0.25 | tier67a50m32 | tier67a50m32 + LayerDrop p=0.25
# Run on the node with the code staged under $ROOT (git archive | srun tar -x), e.g.
#   srun --partition=jsteinhardt --qos=preemptive -w sneetches --gres=gpu:2 ... bash smoke_layerdrop.sh
# Per arm: logs/<arm>.log (trainer step times + flop_meter metrics). MAX_STEPS (default 12).
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
D=${D:-/data/prasann/layerskip}
PY=${PY:-/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python}
MAX_STEPS=${MAX_STEPS:-12}
ARMS=${ARMS:-"dense dense_ld25 tier67 tier67_ld25"}
export PYTHONPATH="$ROOT/src" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export HOME=/data/prasann/home HF_HUB_CACHE=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p "$D/logs" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$ROOT"
RULE="--st-keep-mode gold_plus_random --st-keep-frac 0.6667 --st-keep-token-rule first_last --st-keep-token-k 0.05 --st-kept-doc-k 0.5 --st-keep-token-min 32 --st-header-extra-tokens 8 --st-drop-slots --attn-backend flash_2"
LD="--layer-drop-p 0.25"
for arm in $ARMS; do
  case $arm in
    dense) V=full; X="" ;;
    dense_ld25) V=full; X="$LD" ;;
    tier67) V=softtoken; X="$RULE" ;;
    tier67_ld25) V=softtoken; X="$RULE $LD" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  run=layerdrop-smoke-$arm
  rm -rf "$D/save/$run" "$D/work/$run"
  echo "=== $arm: --variant $V $X" | tee "$D/logs/$arm.log"
  /usr/bin/time -f "[wall] %e s" "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    recipes/train/research_trainer/train_ctc_suite.py \
    --task mt4 --variant $V --model-scale 4b --model-family qwen3_5 --no-wandb \
    --run-name "$run" --base-checkpoint "$D/base/q35-4b-base-markerfix/model_and_optim" \
    --data "$D/shards/mt4s488" --work-dir "$D/work/$run" --save-folder "$D/save/$run" \
    --seq-len 10240 --global-batch 16 --micro-batch-instances 2 --epochs 1 --lr 5e-6 --seed 0 \
    --max-steps "$MAX_STEPS" --no-final-checkpoint --no-compile $X >> "$D/logs/$arm.log" 2>&1
  echo "=== $arm exit $?" | tee -a "$D/logs/$arm.log"
  [ -f "$D/save/$run/flops.json" ] && cat "$D/save/$run/flops.json" >> "$D/logs/$arm.log"
done
