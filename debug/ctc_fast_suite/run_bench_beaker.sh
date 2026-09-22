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
# A REAL system toolkit from the same repo. The image ships /usr/local/cuda-13.0 with only the
# compat libs (no bin/nvcc), and every pip toolkit in this stack is incoherent, so without this
# there is nothing for flashinfer's GDN kernels to compile with. Installing nvcc and the cudart
# headers as a matched apt pair is what makes __CUDACC_VER__ and cuda.h agree by construction.
# The FULL toolkit, not a hand-picked subset. Installing nvcc + cudart-dev alone got the GDN
# kernels compiling and then died on `curand.h: No such file` from flashinfer's sampling kernels --
# and behind curand sit cublas, cusolver and the rest, each a separate apt package and a separate
# failed job to discover. The metapackage is a couple of GB and about two minutes; guessing the
# closure is neither.
apt-get install -y -qq cuda-toolkit-13-0 2>&1 | tail -3
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

# CUDA_HOME has to name a COHERENT toolkit: nvcc's __CUDACC_VER__ and the cuda.h it resolves must
# agree, because flashinfer's bundled cccl guard (cuda_toolkit.h:41) hard-errors when they differ --
# "CUDA compiler and CUDA toolkit headers are incompatible". That error surfaces as ~60 per-layer
# "GDN prefill kernel warmup failed" WARNINGS, so the engine appears to come up and then dies at
# first inference, which is the worst possible place to learn about it.
#
# Do not pick a toolkit by path and hope. The only honest test is to compile the guard, so that is
# what this does: try each candidate, keep the first whose test compile succeeds, fail loudly if
# none do. Candidate order puts REAL system toolkits first (the beaker-qwen35-vllm-cracked record's
# universal fix) and the pip ones after -- `cuda-toolkit==13.0.2` used to land a coherent nvcc at
# nvidia/cuda_nvcc, but now resolves to nvidia/cu13, whose nvcc/cuda.h pair is NOT coherent.
echo "=== CUDA_HOME: probing for a coherent toolkit $(date -u '+%T')Z ==="
CCCL=$("$VENV/bin/python" -c "import flashinfer,os;print(os.path.join(os.path.dirname(flashinfer.__file__),'data','cccl'))" 2>/dev/null)
cat > "$WORK/guard.cu" <<'CUEOF'
#include <cuda.h>
#include <cuda/std/__cccl/cuda_toolkit.h>
__global__ void k() {}
int main() { return 0; }
CUEOF
CUDA_HOME=""
CANDIDATES=$(ls -d /usr/local/cuda /usr/local/cuda-* 2>/dev/null;              find "$VENV/lib" -maxdepth 4 -path '*nvidia/cuda_nvcc' -type d 2>/dev/null;              find "$VENV/lib" -maxdepth 4 -path '*nvidia/cu*' -type d 2>/dev/null)
for cand in $CANDIDATES; do
  [ -x "$cand/bin/nvcc" ] || continue
  if "$cand/bin/nvcc" -std=c++20 --expt-relaxed-constexpr         -I"$CCCL/libcudacxx/include" -I"$CCCL/cub" -I"$CCCL/thrust"         -isystem "$cand/include" -c "$WORK/guard.cu" -o "$WORK/guard.o" 2>"$WORK/guard.err"; then
    CUDA_HOME="$cand"
    echo "  COHERENT: $cand  ($("$cand/bin/nvcc" --version | tail -1))"
    break
  fi
  echo "  incoherent, skipping: $cand  ($(grep -m1 -oE 'error:.*' "$WORK/guard.err" || echo 'compile failed'))"
done
if [ -z "$CUDA_HOME" ]; then
  echo "FATAL: no coherent CUDA toolkit. Candidates tried:"; echo "$CANDIDATES"
  echo "Last compile error:"; cat "$WORK/guard.err"
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"

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
  --cells "$CELLS" --limit "$LIMIT" --max-model-len "$MAXLEN" --out "$OUT" \
  ${SAVE_GENERATIONS:+--save-generations}
rc=$?
echo "--- $OUT ---"; cat "$OUT" 2>/dev/null
echo "=== DONE rc=$rc $(date -u '+%F %T')Z ==="
exit $rc
