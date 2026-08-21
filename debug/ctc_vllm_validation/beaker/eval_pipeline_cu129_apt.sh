#!/bin/bash
# Real CTC-suite eval fan-out on Beaker/jupiter, built on the PROVEN cu129+apt-toolkit vLLM
# load recipe (run_pipeline_cu129_apt.sh -- verified: flashinfer GDN JIT-compiles against a
# coherent apt cuda-toolkit-12-8, torch/torchvision/torchaudio triad pinned via PIP_CONSTRAINT,
# vllm.LLM loads the 4B GDN-hybrid and generates coherently). Install+serving-copy section is
# reused VERBATIM; only the tail (smoke generate) is replaced with the real
# build_prefills_any -> run_vllm_eval -> grade_any -> S3-push chain, one (TASK, RUNG) per job,
# mirroring node_local/gpu_eval_task.sh's proven per-rung flow (same S3 grade prefix:
# ctc_dense_results/<task>/rung_<N>.json).
#
# Required env (passed via `gantry run --env`):
#   TASK        logical task name, e.g. grouping, absence_gutenberg, outlier, reorder, strmatch
#   EVAL_TASK   catalog --task spelling for build_prefills_any/run_vllm_eval/grade_any
#               (defaults to TASK; differs for the alias-corrected tasks, e.g.
#               absence_gutenberg -> absence)
#   RUNG        token-budget rung, e.g. 16384 or 32768
#   CKPT_NAME   exact checkpoint dirname shared by weka (ctc_suite/ckpts/<name>) and S3
#               (_transfer/<name>), e.g. ctc-4b-grouping-full-20260719T225805-0700
set -uo pipefail
echo "=== HOST=$(hostname) START=$(date '+%F %T') TASK=${TASK:?} EVAL_TASK=${EVAL_TASK:=$TASK} RUNG=${RUNG:?} CKPT_NAME=${CKPT_NAME:?} ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
df -h / 2>/dev/null | tail -1

# --- locate the gantry-cloned repo (works whether or not `pip install -e .` already ran) ---
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
if [ -z "$REPO" ]; then
  echo "FATAL: could not locate cloned OLMo-core repo (pyproject.toml) under /"; find / -maxdepth 2 2>/dev/null; exit 1
fi
echo "REPO=$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# --- materialize AWS creds from the injected beaker secrets (AWS_CREDS/AWS_CFG env vars ->
# ~/.aws/{credentials,config}) -- needed for every S3 pull/push below (eval code tarball, rung
# jsonl, checkpoint fallback, grade push). Same pattern as
# src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh. ---
mkdir -p ~/.aws
printf '%s\n' "${AWS_CREDS:?set via --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS}" > ~/.aws/credentials
printf '%s\n' "${AWS_CFG:?set via --env-secret AWS_CFG=PRASANNS_AWS_CONFIG}" > ~/.aws/config
command -v aws >/dev/null 2>&1 || pip install --quiet awscli
echo "aws cli: $(aws --version 2>&1)"

WORK=/root/vllm_beaker_eval_${TASK}_${RUNG}
mkdir -p "$WORK"
VENV="$WORK/venv"

# --- PyPI only publishes a cu13-linked vllm==0.25.1 wheel (needs driver >=~580), but vLLM's
# GitHub release also ships a vllm-0.25.1+cu129 wheel (CUDA 12.9). jupiter's driver reports
# "CUDA Version: 12.8" -- same MAJOR version (12.x) as cu129, so NVIDIA's built-in *minor
# version compatibility* (driver >= the 12.x baseline, no forward-compat package needed)
# should let this run natively, unlike the cu13 (major-version) jump.
CU129_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
echo "=== building vllm venv (cu129 wheel from GitHub releases) $(date '+%F %T') ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$CU129_WHEEL_URL" 2>&1 | tail -60
echo "=== vllm install done $(date '+%F %T') ==="
"$VENV/bin/python" -c "import vllm, transformers; print('vllm', vllm.__version__); print('transformers', transformers.__version__)"

echo "=== checking torch build pulled in by the cu129 wheel $(date '+%F %T') ==="
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
TORCH_CUDA_TAG=$("$VENV/bin/python" -c "import torch; print(torch.__version__.split('+')[-1])")
echo "TORCH_CUDA_TAG=$TORCH_CUDA_TAG"
if [ "$TORCH_CUDA_TAG" != "cu129" ]; then
  echo "torch is not cu129 (got $TORCH_CUDA_TAG) -- forcing the matching cu129 torch+torchvision+torchaudio TRIAD"
  TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")
  "$VENV/bin/pip" install --quiet --force-reinstall \
    "torch==$TORCH_VER" "torchvision==0.26.0" "torchaudio==$TORCH_VER" \
    --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -60
  "$VENV/bin/python" -c "
import torch, torchvision, torchaudio
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
"
fi

echo "=== verifying CUDA is usable (cu129 on the 12.8 driver, via minor-version compat) $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda.is_available()', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    x = torch.randn(4, 4, device='cuda') @ torch.randn(4, 4, device='cuda')
    print('matmul OK', x.shape)
" 2>&1 | tee "$WORK/cuda_cu129_verify.log"
if ! grep -q "matmul OK" "$WORK/cuda_cu129_verify.log"; then
  echo "FATAL: CUDA still not usable with the cu129 wheel -- stopping here (see cuda_cu129_verify.log above)"
  exit 1
fi
echo "=== also verify vllm's own compiled extension imports clean (the actual failure point before) $(date '+%F %T') ==="
"$VENV/bin/python" -c "from vllm.platforms import current_platform; print('vllm platform import OK:', current_platform)"

echo "=== verifying transformers + vllm import cleanly on the pinned cu129 triad $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import torch, torchvision, torchaudio
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'torchaudio', torchaudio.__version__)
from transformers import Qwen3_5ForCausalLM
print('Qwen3_5ForCausalLM import OK')
import vllm
print('vllm', vllm.__version__, 'import OK')
" 2>&1 | tee "$WORK/torchvision_fix_verify.log"
if ! grep -q "Qwen3_5ForCausalLM import OK" "$WORK/torchvision_fix_verify.log"; then
  echo "FATAL: Qwen3_5ForCausalLM still fails to import on the pinned cu129 triad"; exit 1
fi

# --- LOCK the triad for every pip install from here on (see run_pipeline_cu129_apt.sh for the
# two drift incidents this fixes). ---
cat > "$WORK/pin-triad.constraints.txt" <<EOF
torch==2.11.0+cu129
torchvision==0.26.0+cu129
torchaudio==2.11.0+cu129
EOF
export PIP_CONSTRAINT="$WORK/pin-triad.constraints.txt"
echo "PIP_CONSTRAINT=$PIP_CONSTRAINT locking torch/torchvision/torchaudio for all remaining installs"

# --- torchcodec: the FOURTH member of the triad, and the one that is not in it ---------------
# vLLM 0.25.1 imports torchcodec UNCONDITIONALLY on the way to `LLM`
# (vllm.multimodal.media.connector -> vllm.multimodal.video), even with
# limit_mm_per_prompt image/video = 0. It is not pinned by the constraints file above because it
# is not part of the torch/torchvision/torchaudio triad, so pip resolves it fresh -- and its
# current default wheel is CUDA-13-linked. On jupiter's 12.8 driver that dies at
# `torch.ops.load_library` with "libnvrtc.so.13: cannot open shared object file", which is the
# same cu13-vs-cu12 incoherence the whole cu129 recipe exists to avoid, arriving through a
# dependency that drifted after the recipe was validated in July.
# Reinstall it from the SAME cu129 index as the triad; if that is not resolvable, drop it entirely
# (we never decode a video) and let the hard check below decide whether that was enough.
echo "=== pinning torchcodec to the cu129 index $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet --force-reinstall torchcodec \
  --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -20 || \
  echo "(cu129 torchcodec not resolvable -- the fallback below will handle it)"

# --- HARD CHECK: `from vllm import LLM`, not `import vllm` ------------------------------------
# ⚠ `import vllm` IS NOT A REAL CHECK. vllm/__init__.py resolves its public names through a lazy
# module __getattr__, so a bare `import vllm` succeeds while the entire engine import chain is
# still broken. That is exactly how the libnvrtc.so.13 failure got past the verification block
# above and only surfaced ~14 minutes later, AFTER the 19 GB checkpoint sync, the distcp->HF
# export and the three-script serving-copy build had all been paid for. Forcing the real import
# here turns a 14-minute burn into a 5-minute one.
vllm_llm_import_ok() {
  "$VENV/bin/python" -c "from vllm import LLM, SamplingParams, TokensPrompt; print('vllm LLM import OK')" 2>&1
}
OUT=$(vllm_llm_import_ok); echo "$OUT" | tail -20
if ! echo "$OUT" | grep -q "vllm LLM import OK"; then
  echo "--- first LLM import failed; dropping torchcodec (unused: this is a TEXT eval) and retrying ---"
  "$VENV/bin/pip" uninstall -y torchcodec 2>&1 | tail -5
  OUT=$(vllm_llm_import_ok); echo "$OUT" | tail -20
fi
if ! echo "$OUT" | grep -q "vllm LLM import OK"; then
  echo "FATAL: 'from vllm import LLM' fails -- stopping BEFORE the expensive checkpoint/serving stages"
  exit 1
fi

echo "=== installing flashinfer (GDN prefill kernel backend) $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13" 2>&1 | tail -40
"$VENV/bin/python" -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo "=== CUDA_HOME / nvcc: REAL apt system toolkit (coherent nvcc+headers), NOT pip metapackage $(date '+%F %T') ==="
. /etc/os-release; UBU_TAG="ubuntu${VERSION_ID//./}"
echo "distro=$UBU_TAG"
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates >/dev/null
if ! dpkg -l cuda-keyring >/dev/null 2>&1; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU_TAG}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb >/dev/null
fi
apt-get update -qq
apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8 cuda-nvrtc-dev-12-8 2>&1 | tail -60
CUDA_HOME=/usr/local/cuda-12.8
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "FATAL: apt cuda-toolkit-12-8 did not produce $CUDA_HOME/bin/nvcc"
  ls -d /usr/local/cuda* 2>/dev/null
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME  nvcc=$(which nvcc)"
"$CUDA_HOME/bin/nvcc" --version
echo "=== coherence check: nvcc __CUDACC_VER vs cuda.h CUDA_VERSION $(date '+%F %T') ==="
echo | "$CUDA_HOME/bin/nvcc" -x cu -E -dM - 2>/dev/null | grep -E "CUDACC_VER"
grep -E "CUDA_VERSION" "$CUDA_HOME/include/cuda.h" | head -3

echo "=== light olmo_core/corpus_reasoning deps $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet dataclass-extensions cached-path filelock packaging pyyaml rich \
  safetensors importlib_resources bettermap pandas huggingface_hub scikit-learn 2>&1 | tail -20
echo "=== installing flash-linear-attention (fla) -- required for GatedDeltaNet construction $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet "flash-linear-attention==0.4.1" 2>&1 | tail -60

# --- pull the untracked debug/ctc_vllm_validation tree (build_prefills_any.py, grade_any.py,
# run_vllm_eval.py) + a refreshed src/scripts/eval/ctc_suite/run_rung_eval.py, overlaid onto the
# gantry-cloned $REPO -- same tarball node_local/prep_node.sh uses on cubbins, so this job runs
# the EXACT code the overnight roster's dense 2k/4k/8k sweep already validated. ---
# CODE_TARBALL selects which overlay to apply. The default `ctc_eval_code.tar.gz` is the JULY
# snapshot the dense fan-out was validated against -- keep it as the default so those results stay
# reproducible. ⚠ It PREDATES `--query-position`, so a run that needs query_position=after must
# pass CODE_TARBALL=ctc_eval_code_2026-08-13.tar.gz (or later); with the July overlay the emitter
# silently ignores the flag and renders "both". The rendered-value check after build_prefills_any
# is what turns that into a hard failure instead of a plausible wrong number.
CODE_TARBALL="${CODE_TARBALL:-ctc_eval_code.tar.gz}"
echo "=== pulling $CODE_TARBALL (untracked eval scripts) $(date '+%F %T') ==="
export AWS_PROFILE=S3
aws s3 cp "s3://ai2-llm/checkpoints/prasanns/_transfer/${CODE_TARBALL}" "$WORK/code.tar.gz" --only-show-errors
tar xzf "$WORK/code.tar.gz" -C "$REPO"
rm -f "$WORK/code.tar.gz"
ls "$REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" "$REPO/debug/ctc_vllm_validation/general/grade_any.py" "$REPO/debug/ctc_vllm_validation/run_vllm_eval.py"

echo "=== import verification $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import olmo_core, corpus_reasoning, vllm
from transformers import Qwen3_5ForCausalLM
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
print('ALL IMPORTS OK: olmo_core=', olmo_core.__file__)
print('corpus_reasoning=', corpus_reasoning.__file__)
print('has_fla()=', has_fla())
assert has_fla(), 'fla still not importable after pip install'
" || { echo "FATAL: import verification failed"; exit 1; }

# --- env for HF downloads (network needed for the base Qwen/Qwen3.5-4B-Base config+weights) ---
export HF_HOME="$WORK/hf_home"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
unset TRANSFORMERS_CACHE

CKPT_WEKA="/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/${CKPT_NAME}"
HF_EXPORT="$WORK/hf_export"
# ⚠ MUST MATCH THE CHECKPOINT'S MODEL SCALE.
# export_olmo_to_hf builds the olmo-core model from --base-model and then loads the distcp into it,
# so a scale mismatch is a shape mismatch: a 0.8B checkpoint against the 4B default dies with
#   ValueError: Size mismatch between saved torch.Size([248320, 1024]) and current:
#   torch.Size([248320, 2560]) for model.embeddings.weight
# That is the good case -- it fails loudly. This was hardcoded to 4B because the Beaker pipeline had
# only ever evaluated the 4B suite (the model-scale cells were graded by the node-local driver
# instead), so the first 0.8B job through here lost ~28 minutes of venv build before dying.
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3.5-4B-Base}"

# ⚠ EXPORTER SELECTS THE CHECKPOINT->HF PATH. It is an explicit caller-set switch, NOT sniffed
# from the checkpoint: guessing is what turns "wrong family" into a plausible number instead of an
# error, and the caller always knows which family it launched.
#   qwen  (default, unchanged): src/corpus_reasoning/train/export_olmo_to_hf.py. Its
#         resolve_olmo_model() supports ONLY model_type qwen3 / qwen3_5 and RAISES otherwise, so an
#         OLMo checkpoint cannot go through it at all.
#   noswa: debug/ctc_olmo_hybrid/export_noswa_to_hf.py -- a sliding-window-FREE olmo3 model
#         (--model-scale 7b-noswa) mapped onto the Olmo2 HF class, which IS the full-attention
#         arch. Validated locally: /scratch/users/prasann/ctc_olmo3ns_results/*-noswa-cpt-vllm_full/
#         (oolong 2k gen_seconds=25.0 vs 2191s native).
# NOT a path for GDN/hybrid (ctc-olmohyb-*) checkpoints: get_hf_config routes those to
# save_hf_hybrid_model and vLLM has no loader for the result. Those stay on the native evaluator.
EXPORTER="${EXPORTER:-qwen}"
case "$EXPORTER" in qwen|noswa) ;; *) echo "FATAL: EXPORTER must be qwen|noswa, got '$EXPORTER'"; exit 2 ;; esac

# ⚠ MARKER IDS AND TOKENIZER ARE NOW PASSED EXPLICITLY TO build_prefills_any (below).
# They used to be left at that script's defaults, which are the Qwen3.5 values. Pointed at an OLMo
# checkpoint those defaults DO NOT CRASH -- they emit a mis-tokenized prompt wrapped in the wrong
# marker ids and the run reports a plausible number. The values here are read off
# olmo_core.data.document_chunk_landmark.RESERVED_IDS; the qwen defaults below are bit-identical to
# build_prefills_any's own defaults (RESERVED_IDS['qwen3_5'] = 248049/248050/248044), so the Qwen
# path is unchanged -- they are spelled out only so the log records which set was used.
if [ "$EXPORTER" = "noswa" ]; then
  DOC_START_ID="${DOC_START_ID:-100266}"   # <|box_start|> (renamed <|extra_id_1|>)
  DOC_END_ID="${DOC_END_ID:-100267}"       # <|box_end|>   (renamed <|extra_id_2|>)
  EOS_ID="${EOS_ID:-100257}"               # <|endoftext|> -- what the dolma2 SFT shards stop on
  # The PATCHED dolma2 marker tokenizer (weka copy; the Berkeley /scratch copy does not exist here).
  # Must be the same copy the shards were tokenized with -- stock allenai/dolma2-tokenizer has no
  # <|box_start|>/<|box_end|> at all.
  TOKENIZER_DIR="${TOKENIZER_DIR:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3/tokenizer}"
else
  DOC_START_ID="${DOC_START_ID:-248049}"
  DOC_END_ID="${DOC_END_ID:-248050}"
  EOS_ID="${EOS_ID:-248044}"
fi

if [ -f "$CKPT_WEKA/config.json" ] || [ -d "$CKPT_WEKA/model_and_optim" ]; then
  CKPT="$CKPT_WEKA"
  echo "=== using weka checkpoint: $CKPT ==="
else
  echo "=== weka path $CKPT_WEKA not found/usable -- falling back to S3 sync $(date '+%F %T') ==="
  CKPT="$WORK/ckpt_from_s3"
  mkdir -p "$CKPT/model_and_optim"
  aws s3 sync "s3://ai2-llm/checkpoints/prasanns/_transfer/${CKPT_NAME}/model_and_optim" \
    "$CKPT/model_and_optim" --only-show-errors
  aws s3 cp "s3://ai2-llm/checkpoints/prasanns/_transfer/${CKPT_NAME}/config.json" \
    "$CKPT/config.json" --only-show-errors
  echo "=== using S3-synced checkpoint: $CKPT ==="
fi

if [ "$EXPORTER" = "noswa" ]; then
  [ -d "$TOKENIZER_DIR" ] || { echo "FATAL: marker tokenizer not found at $TOKENIZER_DIR (is --weka mounted?)"; exit 1; }
  # NOSWA_MAX_SEQ_LEN -> max_position_embeddings in the export. 40960 is the --seq-len these runs
  # trained at. It does NOT bound vLLM: with YaRN (factor 8, original_max_position_embeddings 8192)
  # vLLM derives max_model_len = 8192*8 = 65536, and run_vllm_eval auto-bumps its own
  # --max-model-len to the longest actual prompt. A rung whose prompts exceed 65536 needs a config
  # patch, not this knob.
  echo "=== export olmo3-noswa distcp -> HF (Olmo2 arch) $(date '+%F %T') ==="
  # NOSWA_DTYPE=bfloat16 matches the native evaluator, which builds at DType("bfloat16"). Leaving
  # it fp32 (the distcp's master-weight dtype) makes vLLM's dtype="auto" downcast to FLOAT16 --
  # a different rounding mode than the number we are trying to reproduce -- and writes a 29GB
  # export instead of a 15GB one.
  "$VENV/bin/python" "$REPO/debug/ctc_olmo_hybrid/export_noswa_to_hf.py" \
    "$CKPT" "$HF_EXPORT" \
    --tokenizer "$TOKENIZER_DIR" --max-seq-len "${NOSWA_MAX_SEQ_LEN:-40960}" \
    --dtype "${NOSWA_DTYPE:-bfloat16}"
  echo "--- disk after export (a 7B bf16 export is ~15GB on top of the ~15GB venv) ---"
  df -h "$WORK" 2>/dev/null | tail -1
else
echo "=== export olmo distcp -> HF text $(date '+%F %T') ==="
"$VENV/bin/python" "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
  --save-folder "$CKPT" --ckpt "$CKPT" \
  --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
fi
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "FATAL: export step failed (rc=$rc) or $HF_EXPORT/config.json missing"; exit 1
fi
echo "=== export done $(date '+%F %T') ==="

if [ "$EXPORTER" = "noswa" ]; then
# An Olmo2 export is a PLAIN single-arch causal LM: no multimodal wrapper, no vision tower, no
# key rename. The three-script serving-copy recipe below exists solely to serve the Qwen3.5 VL
# wrapper text-only, and every piece of it is wrong here (make_vl_weights would also assume a
# single model.safetensors, which a 7B export is not -- it shards). Serve the export directly.
SERVE="$HF_EXPORT"
echo "=== noswa: serving straight from the HF export (no VL wrapper) $(date '+%F %T') ==="
ls "$SERVE"
"$VENV/bin/python" -c "
import json; c=json.load(open('$SERVE/config.json'))
print('[serve] model_type=%s arch=%s vocab=%s max_pos=%s rope_scaling=%s' % (
    c.get('model_type'), c.get('architectures'), c.get('vocab_size'),
    c.get('max_position_embeddings'), c.get('rope_scaling')))
assert c.get('model_type') == 'olmo2', 'expected an Olmo2 (full-attention) export, got %r -- a sliding-window or hybrid checkpoint has no vLLM path here' % c.get('model_type')
" || { echo "FATAL: noswa export is not a plain Olmo2 config"; exit 1; }
else
echo "=== resolve base VL snapshot (reuses export's HF cache) $(date '+%F %T') ==="
BASE_SNAP=$("$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$BASE_MODEL_ID'))
")
echo "BASE_SNAP=$BASE_SNAP"
ls "$BASE_SNAP" | head -20

mkdir -p "$WORK/scripts"

cat > "$WORK/scripts/make_vllm_serving_copy.py" <<'PYEOF'
import argparse, glob, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-export", required=True)
    ap.add_argument("--base-snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text_cfg = json.load(open(os.path.join(args.hf_export, "config.json")))
    base_cfg = json.load(open(os.path.join(args.base_snapshot, "config.json")))
    wrapper = dict(base_cfg)
    wrapper["architectures"] = ["Qwen3_5ForCausalLM"]
    wrapper["text_config"] = text_cfg
    wrapper["tie_word_embeddings"] = bool(text_cfg.get("tie_word_embeddings", False))
    os.makedirs(args.out, exist_ok=True)
    for f in glob.glob(os.path.join(args.hf_export, "*")):
        name = os.path.basename(f)
        if name == "config.json":
            continue
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
        src = os.path.join(args.base_snapshot, name)
        dst = os.path.join(args.out, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(wrapper, f, indent=2)
    print(f"[serving-copy] {args.out}: tie_word_embeddings={text_cfg.get('tie_word_embeddings')}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

cat > "$WORK/scripts/make_vl_weights.py" <<'PYEOF'
import argparse, json, os
from safetensors.torch import load_file, save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-export", required=True)
    ap.add_argument("--base-snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    state = load_file(os.path.join(args.hf_export, "model.safetensors"))
    out = {}
    for k, v in state.items():
        if k.startswith("model."):
            out["model.language_model." + k[len("model."):]] = v
        else:
            out[k] = v
    idx_path = os.path.join(args.base_snapshot, "model.safetensors.index.json")
    n_vis = 0
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))["weight_map"]
        shards = sorted({f for k, f in idx.items() if k.startswith("model.visual.")})
        for shard in shards:
            shard_path = os.path.join(args.base_snapshot, shard)
            if not os.path.exists(shard_path):
                print(f"[vl-weights] missing shard {shard}; skipping", flush=True)
                continue
            st = load_file(shard_path)
            for k, v in st.items():
                if k.startswith("model.visual."):
                    out[k] = v
                    n_vis += 1
    else:
        print(f"[vl-weights] no index.json at {idx_path}; skipping vision graft", flush=True)
    dst = os.path.join(args.out_dir, "model.safetensors")
    if os.path.islink(dst):
        os.unlink(dst)
    save_file(out, dst, metadata={"format": "pt"})
    print(f"[vl-weights] wrote {len(out)} tensors ({n_vis} visual) -> {dst}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

cat > "$WORK/scripts/add_dummy_visual.py" <<'PYEOF'
import argparse, json, os
import torch
from safetensors.torch import load_file, save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.serve, "config.json")))
    vcfg = cfg.get("vision_config")
    assert vcfg is not None, "serving config has no vision_config"
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    vision_config = Qwen3_5VisionConfig(**vcfg)
    torch.manual_seed(0)
    with torch.device("cpu"):
        vision = Qwen3_5VisionModel(vision_config).to(torch.bfloat16)
    vstate = {f"visual.{k}": v.contiguous() for k, v in vision.state_dict().items()}
    print(f"[dummy-visual] built {len(vstate)} visual.* tensors", flush=True)
    model_path = os.path.join(args.serve, "model.safetensors")
    state = load_file(model_path)
    n_text = len(state)
    added = 0
    for k, v in vstate.items():
        if k not in state:
            state[k] = v
            added += 1
    print(f"[dummy-visual] text={n_text} visual_added={added} total={len(state)}", flush=True)
    if os.path.islink(model_path):
        os.unlink(model_path)
    save_file(state, model_path, metadata={"format": "pt"})
    print(f"[dummy-visual] wrote {model_path}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

SERVE="$WORK/serving"
echo "=== building vLLM serving copy $(date '+%F %T') ==="
"$VENV/bin/python" "$WORK/scripts/make_vllm_serving_copy.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out "$SERVE"
"$VENV/bin/python" "$WORK/scripts/make_vl_weights.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE"

HAS_VISUAL=$("$VENV/bin/python" -c "
from safetensors import safe_open
f = safe_open('$SERVE/model.safetensors', framework='pt')
print(any(k.startswith('visual.') for k in f.keys()))
")
echo "HAS_VISUAL=$HAS_VISUAL"
if [ "$HAS_VISUAL" != "True" ]; then
  echo "=== no real vision weights grafted -> synthesizing dummy visual.* $(date '+%F %T') ==="
  "$VENV/bin/python" "$WORK/scripts/add_dummy_visual.py" --serve "$SERVE"
fi
TOKENIZER_DIR="$BASE_SNAP"
fi

# =====================================================================================
# REAL EVAL CHAIN (replaces the smoke test): pull rung data -> build_prefills_any ->
# run_vllm_eval --mode full --task EVAL_TASK -> grade_any -> push grade json to S3
# under the SAME prefix cubbins's dense fan-out uses.
# =====================================================================================
# RUNG_TREE selects the S3 rung prefix, mirroring the node-local driver's split. Default
# `ctc_eval_rungs` is the SHIPPED 2k-32k ladder. The length-generalization study passes
# `ctc_eval_rungs_lengthgen`, which is a separate prefix precisely so a 64k/128k rung can never be
# confused with a shipped one -- they are built by different generators against different budgets,
# and a lengthgen rung silently graded as a shipped one would be published as ladder coverage.
RUNG_TREE="${RUNG_TREE:-ctc_eval_rungs}"
echo "=== pulling eval rung data from S3 (tree=$RUNG_TREE) $(date '+%F %T') ==="
RUNG_JSONL="$WORK/rung_${RUNG}.jsonl"
aws s3 cp "s3://ai2-llm/checkpoints/prasanns/_transfer/${RUNG_TREE}/${TASK}/rung_${RUNG}.jsonl" \
  "$RUNG_JSONL" --only-show-errors
if [ ! -f "$RUNG_JSONL" ]; then
  echo "FATAL: rung data not found at ${RUNG_TREE}/${TASK}/rung_${RUNG}.jsonl"; exit 1
fi
echo "RUNG_JSONL=$RUNG_JSONL size=$(wc -c < "$RUNG_JSONL")"

PREFILLS="$WORK/prefills.json"
# QUERY_POSITION / COT_MODE: unset reproduces this script's original behavior (the emitter's
# "both" / "plan" defaults), which is what the July fan-out was scored under. They are OPT-IN
# because a shard tokenized with query_position="after" is scored as garbage under "both": the
# question gets prepended a second time into the FREE prefix, which for a chunked arm attends
# globally, i.e. the question broadcast to every chunk in a layout the model never trained on.
# That is the contradiction 0.559-vs-0.946 failure. Any run whose shard metadata.json says
# "after" MUST pass QUERY_POSITION=after here, and BOTH arms must pass the same value or the
# dense-vs-chunked delta is a prompt-layout artifact rather than a result.
echo "=== build_prefills_any tokenizer=$TOKENIZER_DIR ids=$DOC_START_ID/$DOC_END_ID/$EOS_ID query_position=${QUERY_POSITION:-<default both>} cot=${COT_MODE:-<default plan>} $(date '+%F %T') ==="
"$VENV/bin/python" "$REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
  --tokenizer "$TOKENIZER_DIR" --task "$EVAL_TASK" --eval-data "$RUNG_JSONL" \
  --doc-start-id "$DOC_START_ID" --doc-end-id "$DOC_END_ID" --eos-token-id "$EOS_ID" \
  ${COT_MODE:+--cot-mode "$COT_MODE"} \
  ${QUERY_POSITION:+--query-position "$QUERY_POSITION"} \
  --max-test-samples 100000 --out "$PREFILLS"
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$PREFILLS" ]; then
  echo "FATAL: build_prefills_any failed (rc=$rc)"; exit 1
fi
# Verify the flag TOOK EFFECT rather than trusting it was passed -- build_prefills_any records the
# value it actually rendered with, so this compares intent against the artifact. On Beaker the way
# this silently regresses is the ctc_eval_code.tar.gz overlay above: a tarball built before
# --query-position existed restores an emitter that ignores the flag and exits 0.
ACTUAL_QP=$("$VENV/bin/python" -c "import json;print(json.load(open('$PREFILLS')).get('query_position'))")
echo "[$TASK] rung=$RUNG rendered query_position=$ACTUAL_QP (requested='${QUERY_POSITION:-<default both>}')"
if [ -n "${QUERY_POSITION:-}" ] && [ "$ACTUAL_QP" != "$QUERY_POSITION" ]; then
  echo "FATAL: query_position MISMATCH: wanted '$QUERY_POSITION' got '$ACTUAL_QP' -- stale ctc_eval_code.tar.gz overlay?"
  exit 1
fi

echo "=== run_vllm_eval --mode full --task $EVAL_TASK $(date '+%F %T') ==="
echo "CUDA_HOME=$CUDA_HOME nvcc=$(which nvcc)"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
RESP="$WORK/responses.json"
# EXTRA_STOP_TOKEN_IDS is read from the environment by run_vllm_eval.py; echo it so the artifact
# records whether it was set. For OOLONG it is LOAD-BEARING: the prefill pack's eos is 248044
# (`<|endoftext|>`) but our SFT assistant turns end with `<|im_end|>` = 248046, so without 248046
# the model answers, signals end-of-turn, then rambles to max_new_tokens. OOLONG's answer is a bare
# number/label with no newline, so the text-level truncation rescue that silently saves the other
# tasks has nothing to cut on and the grader scores the whole ramble -- measured 0.0055 at rung
# 2048. The job still exits 0, so this shows up as a "result", not a failure.
# TP shards the KV cache across GPUs (see node_local/gpu_eval_task.sh for the head-count
# divisibility rule). Must match --gpus on the gantry call; TP>1 with fewer GPUs dies at init.
# ⚠ MODE MUST MATCH THE MASK THE CHECKPOINT WAS TRAINED WITH.
# This script used to hardcode `--mode full`, which is correct ONLY for a dense-trained arm.
# Scoring a document-chunked checkpoint under dense attention hands it a layout it never saw in
# training, and the failure is TOTAL rather than partial: the pure-chunked contradiction arm read
# 0.0007 / 0.0013 / 0.0000 set_f1 at 2560/4096/8192 while its training CE had descended normally
# to ~1.05. A CE-healthy model scoring exactly zero is an eval artifact, and if it had been
# believed it would have been published as "mask-mixing is worth +0.86", which is not a thing the
# data says. The chunked-mix BASELINE these numbers get compared against was itself graded with
# mode=chunked, so anything but a matching mode is not a comparison.
# MODE defaults to full so every existing caller is unchanged.
MODE="${MODE:-full}"
# ⚠ EXPORTER=noswa + MODE=chunked IS REFUSED, and not for a portability reason.
# run_vllm_eval's document-chunk patch masks EVERY attention layer, but the olmo3 chunked arms
# (olmo3_7B_ctc_swa / the noswa variants) apply the mask to the full-attention layers only. Scoring
# one with the other is not the model that trained, and -- like every other mask mismatch in this
# file -- it would return a well-formed number rather than an error. The flex kv-block size below
# is also tuned to the Qwen3.5 GDN page size (528), which an Olmo2 model does not have.
if [ "$EXPORTER" = "noswa" ] && [ "$MODE" = "chunked" ]; then
  echo "FATAL: MODE=chunked is not supported for EXPORTER=noswa (the in-process patch masks all"
  echo "       layers; the olmo3 chunked arms mask only the full-attention layers). Use the native"
  echo "       evaluator (debug/ctc_crossfamily/eval_olmo_beaker.sh) for chunked OLMo arms."
  exit 2
fi
# The chunked path rebuilds a FlexAttention block mask per step and is ~18x slower without the
# varlen prefill plan; turn it on by default for chunked mode (validated: +0.0014/-0.0020 metric
# movement at eval_size 500) and keep concurrency pinned at the measured-best 16/18.
# ⚠ CHUNK_VARLEN_PREFILL DEFAULTS OFF *ON BEAKER*, unlike the node-local driver.
# The varlen plan routes prefill through flash_attn_varlen_func, and the flash-attn that ships
# with the cu129 vLLM wheel carries PTX the jupiter/ceres 12.8 driver refuses:
#   torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported
#   toolchain  (cudaErrorUnsupportedPtxVersion)  at vllm_chunked_patch._varlen_forward
# It dies on the very first prefill step ("varlen step #1: prefill_reqs=1"), so the whole rung is
# lost. Dense mode never enters that kernel, which is why the same image evaluates a full-attention
# checkpoint fine. Leaving it off falls back to the FlexAttention block-mask path -- correct, but
# ~18x slower, so expect long wall-clock on the high rungs. Set CHUNK_VARLEN_PREFILL=1 explicitly
# only on an image whose flash-attn matches the driver.
if [ "$MODE" = "chunked" ]; then
  export CHUNK_VARLEN_PREFILL="${CHUNK_VARLEN_PREFILL:-0}"
  if [ "$CHUNK_VARLEN_PREFILL" = "1" ]; then
    export CHUNK_FAST_MAX_RUNG="${CHUNK_FAST_MAX_RUNG:-99999999}"
    export CHUNK_MAX_NUM_SEQS_FAST="${CHUNK_MAX_NUM_SEQS_FAST:-16}"
    export CHUNK_SEQ_HEADROOM_FAST="${CHUNK_SEQ_HEADROOM_FAST:-18}"
  fi
fi
echo "=== run_vllm_eval MODE=$MODE TP=${TP:-1} stop_ids=+${EXTRA_STOP_TOKEN_IDS:-<none>} gpu_mem_util=${GPU_MEM_UTIL:-0.85} varlen=${CHUNK_VARLEN_PREFILL:-0} ==="
# --model-family is run_vllm_eval's switch for the Qwen3.5-VL serving overrides
# (hf_overrides={"architectures":["Qwen3_5ForCausalLM"]} + limit_mm_per_prompt=0). "qwen3" means
# PLAIN DENSE -- no VL wrapper, no arch override, use the config's own architecture -- which is
# exactly what an Olmo2 export needs. ⚠ The name is a misnomer here and cannot be fixed from this
# repo: run_vllm_eval.py is OVERWRITTEN by the $CODE_TARBALL overlay untarred over $REPO above, so
# adding an "olmo2" choice would require rebuilding that S3 tarball. "qwen3" is the existing
# plain-dense branch; naming is the only thing wrong with it.
[ "$EXPORTER" = "noswa" ] && VLLM_MODEL_FAMILY=qwen3 || VLLM_MODEL_FAMILY=qwen3_5
echo "=== run_vllm_eval --model-family $VLLM_MODEL_FAMILY (exporter=$EXPORTER) ==="
"$VENV/bin/python" -u "$REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode "$MODE" --task "$EVAL_TASK" \
  --model-family "$VLLM_MODEL_FAMILY" \
  --max-new-tokens 256 --max-model-len 4096 --gpu-mem-util "${GPU_MEM_UTIL:-0.85}" \
  --tensor-parallel-size "${TP:-1}" \
  --out "$RESP"
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$RESP" ]; then
  echo "FATAL: run_vllm_eval failed (rc=$rc)"; exit 1
fi

echo "=== grade_any $(date '+%F %T') ==="
GRADE="$WORK/grade.json"
"$VENV/bin/python" "$REPO/debug/ctc_vllm_validation/general/grade_any.py" \
  --responses "$RESP" --eval-data "$RUNG_JSONL" --task "$EVAL_TASK" \
  --max-test-samples 100000 --out "$GRADE"
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$GRADE" ]; then
  echo "FATAL: grade_any failed (rc=$rc)"; exit 1
fi

echo "--- [$TASK] rung=$RUNG GRADE ---"
cat "$GRADE"
echo ""

# ⚠ REFUSE TO PUBLISH AN UNMASKED "CHUNKED" RESULT.
# The document-chunk mask is an IN-PROCESS monkey-patch that relies on
# VLLM_ENABLE_V1_MULTIPROCESSING=0 keeping the model in the driver process. Anything that forces
# vLLM to spawn workers -- tensor_parallel_size > 1 above all -- leaves the model unpatched, and
# the run then completes normally and writes a well-formed grade that is simply the DENSE number
# wearing a chunked label. Nothing downstream can distinguish it. Measured: fiqa cmix at rung 2048
# under MODE=chunked TP=2 returned gold_id_f1=0.9165619047619048, bit-identical to the same
# checkpoint scored dense, with patch_debug.calls=0.
# run_vllm_eval prints a warning for this, but the repo copy is shadowed by the S3 code tarball
# that is untarred over $REPO above, so the enforcement has to live here to be reliable.
if [ "$MODE" = "chunked" ]; then
  APPLIED=$("$VENV/bin/python" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print((d.get('patch_debug') or {}).get('applied', 0))
" "$GRADE" 2>/dev/null || echo 0)
  echo "chunk-mask applied count: $APPLIED"
  if [ "${APPLIED:-0}" -eq 0 ] 2>/dev/null; then
    echo "FATAL: MODE=chunked but the chunk mask was applied 0 times -- this run is UNMASKED."
    echo "       Its metric is the dense number mislabelled as chunked; refusing to publish it."
    echo "       Most likely cause: TP>1. Re-run chunked mode with TP=1."
    exit 4
  fi
fi

echo "=== pushing grade to S3 $(date '+%F %T') ==="
# ⚠ THE DESTINATION MUST CARRY MODEL IDENTITY, NOT JUST $TASK. Keyed on the task alone, every
# checkpoint evaluated on a task overwrites the previous one in place -- on 2026-08-12 that is how
# a 2B model-scale run destroyed the 4B suite's contradiction ladder, leaving grade files with the
# suite's directory name and the 2B model's numbers. Worse than losing them: the result looks like
# a 4B reference and is not one. RESULT_DIR defaults to the historical flat path so existing
# callers are unchanged; every new caller should pass one that names the checkpoint.
RESULT_DIR="${RESULT_DIR:-ctc_dense_results/${TASK}}"
aws s3 cp "$GRADE" \
  "s3://ai2-llm/checkpoints/prasanns/_transfer/${RESULT_DIR}/rung_${RUNG}.json" \
  --only-show-errors
echo "grade -> s3://ai2-llm/checkpoints/prasanns/_transfer/${RESULT_DIR}/rung_${RUNG}.json"
rc=$?
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
