#!/bin/bash
# On-node half of the CTC fast-suite cost benchmark, for Beaker/jupiter (H100).
#
# Measures what a (task, rung) cell costs through vLLM so the eval_size policy for a ~1h /
# 8-GPU full-suite run is chosen from data rather than guessed. 2k-32k first; the xlong rungs
# are a separate launch because 256k needs a YaRN serving copy.
#
# Model: the STOCK public Qwen/Qwen3.5-4B. Architecture (and therefore prefill cost) is identical
# to our SFT checkpoints, and it needs none of the VL-serving-copy recipe an olmo export does --
# so this isolates throughput from checkpoint plumbing. Decode length IS checkpoint-dependent, so
# the script reports generated tokens per example separately from prefill; read the decode half as
# an upper bound (an SFT checkpoint emits EOS earlier).
#
# Data: the public HF dataset PrasannSinghal/ctc-suite-eval -- the same rungs olmo-eval reads.
#
# The install block is the validated Beaker Qwen3.5 stack from
# debug/ctc_vllm_validation/beaker/run_pipeline.sh (see the beaker-qwen35-vllm-cracked record):
# vllm 0.25.1 is cu13-linked while jupiter's driver is 12.8, so CUDA-13 forward-compat libs go
# FIRST on LD_LIBRARY_PATH, and the toolkit comes from the coherent `cuda-toolkit` metapackage.
set -uo pipefail
echo "=== HOST=$(hostname) START=$(date -u '+%F %T')Z ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
[ -n "$REPO" ] || { echo "FATAL: no cloned repo"; exit 1; }
echo "REPO=$REPO"

WORK=/root/ctcfast
VENV=$WORK/venv
mkdir -p "$WORK"
export HF_HOME=$WORK/hf
export TMPDIR=$WORK/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

echo "=== CUDA-13 forward-compat $(date -u '+%T')Z ==="
. /etc/os-release; UBU_TAG="ubuntu${VERSION_ID//./}"
apt-get update -qq && apt-get install -y -qq wget gnupg ca-certificates >/dev/null
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU_TAG}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
  && dpkg -i /tmp/cuda-keyring.deb >/dev/null && apt-get update -qq \
  && apt-get install -y -qq cuda-compat-13-0
COMPAT_DIR=$(dpkg -L cuda-compat-13-0 2>/dev/null | grep 'libcuda\.so' | head -1 | xargs -r dirname)
[ -n "$COMPAT_DIR" ] && export LD_LIBRARY_PATH="$COMPAT_DIR:${LD_LIBRARY_PATH:-}"
echo "COMPAT_DIR=${COMPAT_DIR:-none}"

echo "=== venv $(date -u '+%T')Z ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
  "vllm==0.25.1" "transformers==5.14.1" \
  "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13" \
  "cuda-toolkit==13.0.2" "datasets" 2>&1 | tail -20
"$VENV/bin/python" -c "
import torch, vllm, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('vllm', vllm.__version__, 'transformers', transformers.__version__)
from transformers import Qwen3_5ForCausalLM; print('Qwen3_5ForCausalLM OK')
" || { echo "FATAL: stack did not import"; exit 1; }

# CUDA_HOME must point at the pip cuda-toolkit's OWN nvcc, or vLLM's JIT dies at engine init with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist" -- after the model has
# already loaded, which makes it read like a model problem. It must specifically be the
# nvidia/cuda_nvcc component: another nvidia-* package can ship a MISMATCHED nvcc whose paired
# cuda.h declares a different CUDA_VERSION, and flashinfer's coherence guard then rejects it.
# See the beaker-qwen35-vllm-cracked record.
echo "=== CUDA_HOME $(date -u '+%T')Z ==="
NVCC_PATH=$(find "$VENV/lib" -path '*nvidia/cuda_nvcc*' -iname nvcc 2>/dev/null | head -1)
if [ -z "$NVCC_PATH" ]; then
  echo "nvidia/cuda_nvcc not found; all nvcc under the venv:"; find "$VENV" -iname nvcc 2>/dev/null
  NVCC_PATH=$(find "$VENV" -iname nvcc 2>/dev/null | head -1)
fi
[ -n "$NVCC_PATH" ] || { echo "FATAL: no nvcc under the venv"; exit 1; }
export CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"; "$CUDA_HOME/bin/nvcc" --version | tail -2

# The vendored ctc spec code (prompt/parse/score) rides along in the olmo-eval clone; fetch it
# shallow so the benchmark grades with byte-identical logic to the harness it is sizing.
echo "=== olmo-eval vendor tree $(date -u '+%T')Z ==="
git clone --depth 1 --branch "${OE_BRANCH:-prasann/ctc-suite-grader-fixes}" \
  https://github.com/allenai/olmo-eval.git "$WORK/olmo-eval" 2>&1 | tail -3
VENDOR=$WORK/olmo-eval/src/olmo_eval/evals/tasks/ctc_suite/_vendor
[ -d "$VENDOR/ctc" ] || { echo "FATAL: no vendored ctc at $VENDOR"; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
CELLS="${CELLS:?set CELLS}"
LIMIT="${LIMIT:-100}"
MAXLEN="${MAXLEN:-40960}"
OUT=/results/bench_${TAG:-run}.json

echo "=== bench MODEL=$MODEL LIMIT=$LIMIT MAXLEN=$MAXLEN $(date -u '+%T')Z ==="
"$VENV/bin/python" -u "$REPO/debug/ctc_fast_suite/bench_vllm_rungs.py" \
  --model "$MODEL" --data-root hf --ctc-vendor "$VENDOR" \
  --cells "$CELLS" --limit "$LIMIT" --max-model-len "$MAXLEN" --out "$OUT"
rc=$?
echo "--- $OUT ---"; cat "$OUT" 2>/dev/null
echo "=== DONE rc=$rc $(date -u '+%F %T')Z ==="
exit $rc
