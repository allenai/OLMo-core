#!/bin/bash
#SBATCH --job-name=prefill-parity
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/prefill_parity/logs/parity_%j.log
#
# Does `ctc-eval --share-prefix` change anything but the cost?
#
# One GPU, one checkpoint, two arms per configuration:
#   * `parity_probe.py`  -- last-prompt-token logits and generated text, plain vs reused KV, in one
#     process. This is the strong evidence: a text difference could in principle come from anywhere,
#     an identical logit vector cannot.
#   * `ctc-eval` twice   -- the actual command, once with --share-prefix and once without, `--tag`
#     keeping the two result files apart because the filename encodes neither.
#
# Everything node-local: /data for logs, caches, staged data and results. /accounts and /scratch are
# both NFS at ~5 MB/s and deadlock concurrent readers.
set -uo pipefail

PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/prefill_parity
CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b
TOK=Qwen/Qwen3.5-0.8B-Base

export HOME=/data/prasann/hometmp
export TMPDIR=/data/prasann/hometmp
export TRITON_CACHE_DIR=/data/prasann/triton_cache
export HF_HOME=/data/prasann/hf-cache
export PYTHONPATH="$REPO/src:$REPO/ctc/src"

mkdir -p "$WORK"/{logs,data,bundle/contra,bundle/nq,results,out}
cd "$REPO"

echo "=== host $(hostname)  $(date '+%F %T') ==="
echo "interpreter: $PY"
$PY -c "import sys, torch; print(sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── stage the two shared-corpus files node-local ─────────────────────────────────────────────────
# Built by src/scripts/ctc/build_fast_bundle.py and kept in the repo under debug/fast_bundle/out.
# The fast bundle proper lives on weka, which this cluster does not mount.
CONTRA=$WORK/data/contradiction_rung_2048_tail50.jsonl
NQ=$WORK/data/nq_rung_8192_mux.jsonl
[ -s "$CONTRA" ] || cp "$REPO/debug/fast_bundle/out/contradiction/rung_2048_tail50.jsonl" "$CONTRA"
[ -s "$NQ" ]     || cp "$REPO/debug/fast_bundle/out/nq/rung_8192_mux.jsonl" "$NQ"

# `--bundle <path>` is treated as a reliable bundle carrying the BASE ladder, so the staged files
# have to sit at the base ladder's relative paths. The names below are the ladder's, the contents
# are the shared-corpus rungs -- which is fine here because both arms read the same file and the
# question is whether the two prefill paths agree, not what the number means.
ln -sf "$CONTRA" "$WORK/bundle/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"
ln -sf "$NQ"     "$WORK/bundle/nq/nq_validation_k50_600.jsonl"

# ── 1. logit-level parity, contradiction (50% shared prefix at query_position=both) ──────────────
echo; echo "=== probe: contradiction, attn full ==="
$PY debug/prefill_parity/parity_probe.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --groups 3 --rows-per-group 8 \
    --query-positions both --attn full \
    --out "$WORK/out/probe_contradiction_full.json"
echo "probe exit=$?"

# ── 2. logit-level parity, nq multiplexed: 'both' shares ~nothing, 'after' shares ~everything ────
echo; echo "=== probe: nq mux, attn full, both + after ==="
$PY debug/prefill_parity/parity_probe.py \
    --ckpt "$CKPT" --data "$NQ" --spec retrieval --tokenizer "$TOK" \
    --max-length 16384 --groups 3 --rows-per-group 8 \
    --query-positions both,after --attn full \
    --out "$WORK/out/probe_nq_full.json"
echo "probe exit=$?"

# ── 3. same, under the chunked mask -- the suffix forward gets only the suffix's tokens, so if the
#       mask's chunk-id reconstruction needs the whole stream this is where it shows ──────────────
echo; echo "=== probe: contradiction, attn chunked ==="
$PY debug/prefill_parity/parity_probe.py \
    --ckpt "$CKPT" --data "$CONTRA" --spec contradiction --tokenizer "$TOK" \
    --max-length 16384 --groups 2 --rows-per-group 6 \
    --query-positions both --attn chunked \
    --out "$WORK/out/probe_contradiction_chunked.json"
echo "probe exit=$?"

# ── 4/5. the real command, both ways, full 500 rows ─────────────────────────────────────────────
for arm in plain shared; do
  FLAG=""
  [ "$arm" = shared ] && FLAG="--share-prefix"
  echo; echo "=== ctc-eval contradiction 2k  arm=$arm ==="
  /usr/bin/time -f "WALL %e s" $PY -m ctc.eval.cli \
      --ckpt "$CKPT" --tasks contradiction --rungs 2k \
      --bundle "$WORK/bundle" --out "$WORK/results" --tag "$arm" \
      --backend native --attn full --query-position both \
      --max-length 16384 --tokenizer "$TOK" --ignore-format-fingerprint $FLAG
  echo "ctc-eval exit=$?"
done

# ── 6. and on the multiplexed construction under `after`, where reuse is ~99% ────────────────────
for arm in plain shared; do
  FLAG=""
  [ "$arm" = shared ] && FLAG="--share-prefix"
  echo; echo "=== ctc-eval nq 8k mux query_position=after  arm=$arm ==="
  /usr/bin/time -f "WALL %e s" $PY -m ctc.eval.cli \
      --ckpt "$CKPT" --tasks nq --rungs 8k \
      --bundle "$WORK/bundle" --out "$WORK/results" --tag "after_$arm" \
      --backend native --attn full --query-position after \
      --limit 200 --max-length 16384 --tokenizer "$TOK" --ignore-format-fingerprint $FLAG
  echo "ctc-eval exit=$?"
done

echo; echo "=== done $(date '+%F %T') ==="
ls -la "$WORK/results" "$WORK/out"
