#!/bin/bash
# Eval a 128k contradiction checkpoint via vLLM -- the 128k extension of eval_2k.sbatch.
# Native olmo-core eval is bs=1 at 128k and far too slow, so this uses the validated vLLM path
# (memory: qwen35-4b-vllm-load-recipe). Handles BOTH arms of the qwen3-vs-qwen3.5 comparison.
#
#   # dense (Qwen3-4B, NTK-extended) -- checkpoint comes off Beaker/weka
#   sbatch -w horton --export=ALL,MODEL=qwen3,CKPT=/data/prasann/ctc_suite/ckpts/q3-4b-contra-128kmix eval_128k.sbatch
#   # hybrid (Qwen3.5-4B GDN, native 262k)
#   sbatch -w sneetches --export=ALL,MODEL=qwen35,CKPT=/data/prasann/ctc_suite/ckpts/q35-4b-contra-128k-local eval_128k.sbatch
#
# ALWAYS do a timing probe first (memory: eval-wallclock-sanity-check) -- 500 x 128k prefill is
# ~64M prefill tokens and the cost is NOT linear in rung:
#   sbatch ... --export=ALL,...,MAXSAMP=20,TIMEONLY=1 eval_128k.sbatch
# then scale --time from the measured per-example seconds before running the full 500.
#SBATCH --job-name=contra128k-eval
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/contra128k_eval_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
VDIR=$REPO/debug/ctc_vllm_validation
# Prefer node-local copies of the transformers shadow and the HF cache. Both were on /scratch,
# and the first 128k run wedged the export in `nfs_wait_bit_killable` (D state, 0.4% CPU, 12 min,
# no output) -- the same NFS failure mode as the flashinfer cache, just via a different path.
# HF_HOME must point at a cache that actually HAS the base snapshot, because the hybrid export
# calls snapshot_download() under HF_HUB_OFFLINE=1; a missing cache is a hard failure, not a
# fallback, so the first EXISTING candidate wins rather than a hardcoded one.
TFSHADOW=/data/prasann/tf_qwen35_target
[ -d "$TFSHADOW" ] || TFSHADOW=/scratch/users/prasann/tf_qwen35_target
export PYTHONPATH="$TFSHADOW:$REPO/src"
for C in /data/prasann/hf_cache /data/prasann/huggingface-cache /scratch/users/prasann/huggingface-cache; do
  [ -d "$C" ] && { export HF_HOME=$C; break; }
done
export HF_HOME="${HF_HOME:-/scratch/users/prasann/huggingface-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.8
# flashinfer/triton JIT caches MUST be node-local -- $HOME is /accounts NFS and concurrent evals
# deadlock in nfs_wait_bit_killable (memory: flashinfer-cache-HOME-NFS-wedge).
export HOME=/data/prasann
export FLASHINFER_CACHE_DIR=/data/prasann/flashinfer_cache
export TRITON_CACHE_DIR=/data/prasann/triton_cache
mkdir -p "$FLASHINFER_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true

# Node-local env first: running the interpreter itself off /scratch drags every .so through NFS,
# which is what stalled the export (see the HF_HOME note above). run_ctc_local.sbatch already
# does this; the eval path had not caught up.
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python             # export/tokenize/grade
[ -x "$PY" ] || PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python                          # generation only
MODEL="${MODEL:?set MODEL=qwen3|qwen35}"
CKPT="${CKPT:?set CKPT=<olmo save folder containing step<N>/ or model_and_optim>}"
MAXSAMP="${MAXSAMP:-100000}"
TIMEONLY="${TIMEONLY:-}"
RUNG=131072
STAGE=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction
RUNG_JSONL=$STAGE/rung_${RUNG}.jsonl
[ -f "$RUNG_JSONL" ] || { echo "FATAL: missing $RUNG_JSONL (build via build_xlong_rungs.py)"; exit 2; }

OUT=/data/prasann/ctc_suite/contra128k
mkdir -p "$OUT"
HF_EXPORT=$OUT/hf_${MODEL}
SERVE=$OUT/serve_${MODEL}
PREFILLS=$OUT/prefills_${MODEL}_${RUNG}.json
RESP=$OUT/responses_${MODEL}_${RUNG}.json
GRADE=$OUT/grade_${MODEL}_${RUNG}.json

if [ "$MODEL" = "qwen35" ]; then
  BASE_ID="Qwen/Qwen3.5-4B-Base"
  BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
  [ -d "$BASE_SNAP" ] || BASE_SNAP=/scratch/users/prasann/hf_models/Qwen3.5-4B-Base
  IDS="--doc-start-id 248049 --doc-end-id 248050 --eos-token-id 248044"
  FAMILY=qwen3_5
  # Hybrid is NATIVE 262k -- no NTK theta override, only the context bound. Do NOT clamp this to
  # 131072: the rung tokenizes to ~142k, run_vllm_eval raises max_model_len to cover the longest
  # prefill, and vLLM then refuses to start because max_model_len > max_position_embeddings. That
  # is what killed the first two attempts -- and the engine-init failure leaves the DRIVER polling
  # forever, so the job burns its whole time limit looking like a slow load. Use the native bound.
  MAXPOS=${MAXPOS:-262144}
  ROPE_ARGS="--max-position-embeddings $MAXPOS"
else
  BASE_ID="Qwen/Qwen3-4B-Base"
  BASE_SNAP=/data/prasann/hf_models/Qwen3-4B-Base
  [ -d "$BASE_SNAP" ] || BASE_SNAP=/scratch/users/prasann/hf_models/Qwen3-4B-Base
  IDS="--doc-start-id 151648 --doc-end-id 151649 --eos-token-id 151643"
  FAMILY=qwen3
  # CRITICAL: the dense arm TRAINED with NTK --rope-theta 4e6. export_olmo_to_hf.py builds its
  # HF config from the BASE, so without this the export would claim rope_theta=1e6 /
  # max_position_embeddings=32768 and rebuild inv_freq from the stale theta -> the model evals
  # with RoPE it never trained on and looks like it collapsed at long context. Must match the
  # run's --rope-theta exactly.
  # The rung tokenizes to ~147k over the full 500 (the 20-example probe only reached 142k, so do
  # not size this from a probe), and 131072 is BELOW the actual prompt -- vLLM would refuse to
  # start. 163840 leaves real headroom; 147456 cleared the observed requirement by 106 tokens,
  # which is not a margin. ⚠ This is genuine extrapolation past the θ=4e6 design point on top of an
  # already-4× extension -- see the trap section in HANDOFF_128k.md before quoting a dense number.
  MAXPOS=${MAXPOS:-163840}
  ROPE_ARGS="--rope-theta 4e6 --max-position-embeddings $MAXPOS"
fi

# Export from the LOCAL snapshot dir when we have one. The hub id would send export_olmo_to_hf
# through snapshot_download() under HF_HUB_OFFLINE=1, and no HF cache on these nodes holds
# Qwen3.5-4B-Base -- the 2k eval never hit this because it evaluated an already-exported model.
EXPORT_BASE="$BASE_ID"
[ -d "$BASE_SNAP" ] && EXPORT_BASE="$BASE_SNAP"

echo "=== eval MODEL=$MODEL family=$FAMILY rung=$RUNG ckpt=$CKPT maxsamp=$MAXSAMP $(date '+%F %T') ==="
echo "    rope args: $ROPE_ARGS"
echo "    PY=$PY  HF_HOME=$HF_HOME  export base=$EXPORT_BASE"

# ---- 1) olmo distcp -> HF (idempotent) ----
if [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "=== export -> HF $(date '+%F %T') ==="
  $PY "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$CKPT" --hf-out "$HF_EXPORT" --base-model "$EXPORT_BASE" $ROPE_ARGS \
    || { echo "!!! EXPORT FAILED"; exit 3; }
else
  echo "HF export exists: $HF_EXPORT"
fi
# Fail loudly if the exported config did not take the overrides -- this is the silent-corruption
# guard, not a formality.
$PY - "$HF_EXPORT" "$MODEL" <<'EOF' || exit 3
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
c = cfg.get("text_config", cfg)          # qwen3_5 exports nest the text config
theta, maxpos = c.get("rope_theta"), c.get("max_position_embeddings")
print(f"[check] exported rope_theta={theta} max_position_embeddings={maxpos}")
if maxpos is None or maxpos < 131072:
    sys.exit(f"FATAL: max_position_embeddings={maxpos} < 131072 -> vLLM will refuse 128k prompts")
if sys.argv[2] == "qwen3" and theta != 4e6:
    sys.exit(f"FATAL: dense rope_theta={theta}, expected 4e6 -> RoPE mismatch vs training")
print("[check] exported config OK")
EOF

# ---- 2) vLLM serving copy (qwen3.5 VL-wrapper recipe only; dense serves the export directly) ----
if [ "$FAMILY" = "qwen3_5" ]; then
  if [ ! -f "$SERVE/preprocessor_config.json" ]; then
    echo "=== building vLLM serving copy $(date '+%F %T') ==="
    $PY "$VDIR/make_vllm_serving_copy.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out "$SERVE" \
      || { echo "!!! SERVING COPY FAILED"; exit 4; }
    $PY "$VDIR/make_vl_weights.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE" \
      || { echo "!!! VL WEIGHTS FAILED"; exit 4; }
  else
    echo "serving copy exists: $SERVE"
  fi
  # make_vllm_serving_copy rebuilds the wrapper config from the BASE snapshot, which would undo
  # the max_position_embeddings override inside text_config. Re-stamp it on the serving copy.
  $PY - "$SERVE" "$MAXPOS" <<'EOF' || exit 4
import json, sys
p = f"{sys.argv[1]}/config.json"; want = int(sys.argv[2]); cfg = json.load(open(p))
tc = cfg.get("text_config", cfg)
if tc.get("max_position_embeddings", 0) < want:
    tc["max_position_embeddings"] = want
    json.dump(cfg, open(p, "w"))
    print(f"[check] re-stamped serving-copy max_position_embeddings -> {want}")
else:
    print(f"[check] serving copy max_position_embeddings={tc.get('max_position_embeddings')} OK")
EOF
else
  SERVE=$HF_EXPORT
  echo "dense arm: serving the HF export directly (no VL wrapper): $SERVE"
fi

# ---- 3) prefills ----
echo "=== build prefills $(date '+%F %T') ==="
$PY -u "$VDIR/build_prefills.py" \
  --tokenizer "$BASE_SNAP" --contra-data "$RUNG_JSONL" \
  --max-test-samples "$MAXSAMP" --cot-mode none $IDS --out "$PREFILLS" \
  || { echo "!!! PREFILLS FAILED"; exit 5; }
$PY -c "
import json,sys; p=json.load(open('$PREFILLS')); r=p['rows']
L=[len(x['prefill']) for x in r]
need=max(L)+512+256
print(f'[prefills] eval_size={len(r)} tok min={min(L)} med={sorted(L)[len(L)//2]} max={max(L)}')
print(f'[prefills] vLLM will need max_model_len>={need}; config allows $MAXPOS')
if need > $MAXPOS:
    sys.exit(f'FATAL: longest prefill {max(L)} + 512 gen + 256 = {need} > max_position_embeddings $MAXPOS. '
             'vLLM refuses to start AND leaves the driver polling a dead engine forever (burns the '
             'whole time limit looking like a slow load). Raise MAXPOS or shorten the rung.')
" || exit 5

# ---- 4) generate ----
# max-model-len is a FLOOR: run_vllm_eval bumps it to (longest prefill + gen + 256) automatically.
# gpu-mem-util high because a 128k KV cache is the whole cost here (dense Qwen3-4B ~147KB/token
# -> ~19GB for ONE 128k sequence; the GDN hybrid is ~4x cheaper, only 1/4 layers are full-attn).
echo "=== vLLM generate (mode=full, family=$FAMILY) $(date '+%F %T') ==="
T0=$(date +%s)
PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode full --model-family "$FAMILY" \
  --max-new-tokens 512 --max-model-len 131072 --gpu-mem-util 0.92 \
  --out "$RESP" \
  || { echo "!!! VLLM EVAL FAILED"; exit 6; }
T1=$(date +%s); NEX=$($PY -c "import json;print(len(json.load(open('$PREFILLS'))['rows']))")
echo "=== generate took $((T1-T0))s for $NEX examples ($(( (T1-T0) / (NEX>0?NEX:1) ))s/example) ==="
[ -n "$TIMEONLY" ] && { echo "TIMEONLY set -> stopping before grade. Scale --time from the rate above."; exit 0; }

# ---- 5) grade ----
$PY -u "$VDIR/grade_responses.py" \
  --responses "$RESP" --contra-data "$RUNG_JSONL" \
  --max-test-samples "$MAXSAMP" --out "$GRADE" || { echo "!!! GRADE FAILED"; exit 7; }
echo "--- rung=$RUNG MODEL=$MODEL grade ---"
$PY -c "
import json; d=json.load(open('$GRADE'))['contradiction']; n=$NEX
import math
se=math.sqrt(max(d['f1'],1e-9)*(1-d['f1'])/n) if n else float('nan')
warn='' if n>=500 else f'  !! eval_size={n} <500 -- FLAG THIS INLINE'
print(f\"f1={d['f1']:.3f} p={d['precision']:.3f} r={d['recall']:.3f} em={d['exact_match']:.3f} parse={d['parse_rate']:.3f} eval_size={n} SE=+/-{se:.3f}{warn}\")
if d['parse_rate'] < 0.5:
    print('WARNING: parse_rate<0.5 -- DUMP GENERATIONS before believing this number')
"
echo "=== DONE rc=$? $(date '+%F %T') ==="
