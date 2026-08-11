#!/usr/bin/env bash
# Port validation: reproduce a known contradiction @2k number with the ctc package.
#
# TARGET (results/ctc_suite/all_results.jsonl, 2026-07-19, git e57aaea6c):
#   ckpt      ctc-t0-contra-n20-full  (root; provenance ckpt_path, not a step dir)
#   data      contradiction_eval_pubmed_both_n20_k3.jsonl   (488 lines, verified)
#   arm       full (plain causal)
#   f1        0.9583333333333334      exact_match 0.8770491803278688
#   parse_rate 1.0                    eval_size   488
#   max_length 4096                   max_new     512
#
# This is a REPRODUCTION check, not a science run: the question is only whether the ported code
# gets the same number from the same inputs. If the FEVER filler leak affects this eval file it
# changes what 0.9583 MEANS, not whether we reproduce it.
#
# Runs on SNEETCHES. The artifacts were produced on cubbins, which is now `drng` (Prolog error)
# and cannot allocate, so both were staged to sneetches' local /data beforehand as a CPU-only copy
# -- 9.3G should not cross the network while a GPU sits idle.
#
# Everything the job reads is node-local: the checkpoint, the data, AND the repo source. Importing
# olmo_core from /accounts costs minutes of NFS stall and looks exactly like a hung model load.
#
# Expected outcomes, and what each would mean:
#   f1 within ~0.001           -> the port is faithful.
#   f1 moves, parse_rate 1.0   -> most likely the stop rule. This runs stop="pairs" (terminate at
#                                 the closing "]]"), whereas the original ran stop_rule="eos",
#                                 whose should_stop returns False unconditionally -- so it decoded
#                                 to the budget and scored whatever the parser made of the tail.
#                                 Re-run with --stop-preset eos to separate the two.
#   parse_rate < 1.0           -> prompt assembly or tokenization differs; read the generations.
#
#SBATCH --job-name=ctc-validate-contra2k
#SBATCH --nodelist=sneetches
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_validate_contra2k_%j.log

set -euo pipefail

STAGE=/data/prasann/ctc_validate
REPO=$STAGE/repo
# The results row's provenance names the checkpoint ROOT, not a step subdirectory:
# config.json and model_and_optim both live there. step750/ has weights but no config.
CKPT=$STAGE/ckpts/ctc-t0-contra-n20-full
DATA=$STAGE/contradiction_eval_pubmed_both_n20_k3.jsonl
OUT=$STAGE/contra_2k_full.json

mkdir -p /data/prasann/joblogs "$STAGE"

# Node-local caches: the GDN JIT leaks into TMPDIR, and flashinfer/triton caches on NFS wedge
# concurrent jobs in nfs_wait_bit_killable.
export TMPDIR=$STAGE/tmp
export TRITON_CACHE_DIR=$STAGE/triton
export FLASHINFER_CACHE_DIR=$STAGE/flashinfer
export XDG_CACHE_HOME=$STAGE/cache
export HF_HOME=$STAGE/cache/huggingface
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR" "$XDG_CACHE_HOME"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH

# The conda env supplies torch/transformers; PYTHONPATH points olmo_core and ctc at the NEW repo,
# overriding the editable install that resolves to the old one.
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src:$REPO/ctc/src"

# Tokenizer and EOS are NOT defaults -- the suite driver passes Qwen/Qwen3.5-0.8B-Base and an
# explicit eos of 248044, a reserved Qwen3.5 marker id rather than the tokenizer's own EOS. Using
# the tokenizer's value would mean generation never terminates on its own. The local
# Qwen3.5-4B-Base directory is used because tokenizer.json is shared across Qwen3.5 sizes and the
# node needs no network that way.
TOKENIZER=/data/prasann/hf_models/Qwen3.5-4B-Base

for p in "$PY" "$CKPT" "$DATA" "$TOKENIZER" "$REPO/src/olmo_core" "$REPO/ctc/src/ctc"; do
  [[ -e "$p" ]] || { echo "MISSING on $(hostname): $p" >&2; exit 1; }
done

echo "[validate] host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "[validate] target f1=0.9583333333333334 em=0.8770491803278688 eval_size=488"
"$PY" -c "import olmo_core,os;print('[validate] olmo_core ->',os.path.dirname(olmo_core.__file__))"

exec "$PY" -m ctc.eval.validate \
  --ckpt "$CKPT" \
  --task contradiction \
  --rung 2k \
  --data "$DATA" \
  --attn full \
  --tokenizer "$TOKENIZER" \
  --eos-token-id 248044 \
  --max-length 4096 \
  --max-new-tokens 512 \
  --out "$OUT" \
  --expect-f1 0.9583333333333334 \
  --expect-eval-size 488
