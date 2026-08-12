#!/bin/bash
# Task #26: close the train -> eval loop on real files.
#
# One job, two halves, because the second cannot run without the first:
#
#   1. TRAIN 30 steps through the migrated `sft.py`, exactly as job 3438664 did, but with the
#      `config_saver` callback's config actually SET. Without that, no checkpoint this repo trains
#      carries `config.json`, and `TransformerGenerationModule.from_checkpoint` reads that file to
#      rebuild the model -- so the native eval backend cannot load the checkpoint at all. The
#      3438664 checkpoints on this node are unloadable for exactly that reason.
#
#   2. GRADE it with `ctc-eval` five ways, to exercise the format-fingerprint guard in BOTH
#      directions on real files:
#        A  accept   --attn chunked --query-position after   (the format it was trained on)
#        B  reject   --attn chunked --query-position both    (a knob the guard compares)
#        C  reject   --attn landmark                         (a different TOKEN stream)
#        D  override --ignore-format-fingerprint on B's mismatch; must run and record UNVERIFIED
#        E  accept   --attn full --query-position after      (same token stream, plain causal)
#
# The model is a 0.6B trained for 30 steps on synthetic data. The SCORE is expected to be poor;
# this tests the machinery.
#
# mooney because the base, the shards and the node-local conda env are all here already, and
# jsteinhardt/preemptive_high is the QOS that can actually get a card (see debug/train_smoke).
#SBATCH --job-name=ctc-eval-loop
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-eval-loop-%j.log

set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_eval_loop
SMOKE=/data/prasann/ctc_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
TOKSRC=/net/horton/data/prasann/hf-cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots
TOK=$WORK/tok
SAVE=$WORK/ckpt
RES=$WORK/results
DATA=$WORK/data/mathmatch/eval_2k.jsonl

export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/triton
export HF_HOME=$WORK/hf
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$HF_HOME" "$TOK" "$RES" "$WORK/data/mathmatch" \
    /data/prasann/joblogs

echo "=== node $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}  $(date +%T) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# --- staging: one NFS hop here, none during the run -------------------------------------------
# -L dereferences: an HF cache snapshot is symlinks into ../../blobs, and a plain -a copies the
# dangling links. model.safetensors is excluded -- only the tokenizer is wanted, and the weights
# come from the checkpoint.
echo "=== staging tokenizer ==="
rsync -aL --exclude 'model.safetensors' "$TOKSRC"/*/ "$TOK"/
ls "$TOK"
cp "$REPO/debug/eval_loop_close/data/mathmatch/eval_2k.jsonl" "$DATA"
wc -l "$DATA"

cd "$REPO"
export PYTHONPATH=src:ctc/src

echo "=== interpreter ==="; $PY -c "import sys, torch; print(sys.executable, torch.__version__)"

# --- 1. train ----------------------------------------------------------------------------------
# A save folder holding a checkpoint is loaded FIRST and silently wins over --base, and the step
# counter then continues from it -- so start clean rather than resume by accident.
rm -rf "$SAVE"
echo "=== marker audit on the staged base ==="
$PY src/scripts/ctc/fix_marker_embeddings.py --base "$SMOKE/base_fixmark/model_and_optim" \
    --check-only --model qwen3_0_6B --tokenizer qwen3

echo "=== SFT: 30 steps, 1 GPU, chunked mask, marker-repaired base  $(date +%T) ==="
$PY -m torch.distributed.run --nproc-per-node=1 --master_port=29583 \
    src/scripts/ctc/train/sft.py ctc-eval-loop \
    --data "$SMOKE/shards/mathmatch_qwen3" \
    --base "$SMOKE/base_fixmark/model_and_optim" \
    --arch chunked \
    --model qwen3_0_6B \
    --tokenizer qwen3 \
    --seq-len 2048 \
    --max-steps 30 \
    --gpus-per-node 1 \
    --lr 1e-5 \
    --save-folder "$SAVE" \
    --save-interval 30 \
    --ephemeral-save-interval 10 \
    --wandb-project ""
echo "=== SFT EXIT=$? ==="

echo "=== checkpoint tree ==="
find "$SAVE" -maxdepth 2
echo "=== config.json present? (the whole point of the run.py fix) ==="
test -f "$SAVE/step30/config.json" && echo "YES $SAVE/step30/config.json" || echo "NO -- eval will fail to load"
echo "=== fingerprint ==="
cat "$SAVE/step30/ctc_format_fingerprint.json"

# --- 2. grade ----------------------------------------------------------------------------------
# From here a non-zero exit is a RESULT, not a failure: B and C are supposed to be refused. So drop
# -e for this section and report each arm's status explicitly.
set +e
CKPT=$SAVE/step30
EVAL="$PY debug/eval_loop_close/ctc_eval_mathmatch.py --data $DATA --ckpt $CKPT --tasks mathmatch \
    --backend native --tokenizer $TOK --max-length 4096 --out $RES"

run_arm () {  # name, expectation, then ctc-eval args
    local name=$1 expect=$2; shift 2
    echo ""
    echo "############ ARM $name  (expected: $expect)  $(date +%T) ############"
    $EVAL "$@"
    echo "############ ARM $name EXIT=$? ############"
}

run_arm A "runs, produces a number" --attn chunked --query-position after --tag accept
run_arm B "REFUSED: query_position"  --attn chunked --query-position both  --tag rejectqp
run_arm C "REFUSED: chunk_layout"    --attn landmark --query-position after --tag rejectlandmark
run_arm D "runs, records UNVERIFIED" --attn chunked --query-position both --tag ignoreqp \
    --ignore-format-fingerprint --limit 50
run_arm E "runs, produces a number" --attn full --query-position after --tag full

echo ""
echo "=== results written ==="
find "$RES" -maxdepth 1 -name '*.json'
echo "=== DONE $(date +%T) ==="
