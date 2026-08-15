#!/bin/bash
# One-off environment probe for the Beaker/jupiter vLLM feasibility check.
# Answers: nvcc/CUDA_HOME availability, baked-env transformers/fla/olmo_core state,
# network reachability to HF hub, weka checkpoint presence, disk space.
set -uo pipefail
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "--- disk ---"
df -h / /root /tmp 2>/dev/null
echo "--- cuda toolkit search ---"
which nvcc || echo "no nvcc on PATH"
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"
echo "--- weka checkpoint ---"
CKPT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/ctc-4b-grouping-full-20260719T225805-0700
ls -la "$CKPT" 2>&1 | head -20
ls -la "$CKPT/model_and_optim" 2>&1 | head -10
cat "$CKPT/config.json" 2>&1 | head -5
echo "--- network: huggingface.co ---"
curl -sI --max-time 10 https://huggingface.co/Qwen/Qwen3.5-4B-Base/resolve/main/config.json | head -5
echo "--- baked default env ---"
which python python3
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)" 2>&1
python -c "import transformers; print('transformers', transformers.__version__)" 2>&1
python -c "from transformers import Qwen3_5ForCausalLM; print('Qwen3_5ForCausalLM OK')" 2>&1 | tail -3
python -c "import flash_linear_attention; print('fla OK', flash_linear_attention.__file__)" 2>&1 | tail -3
python -c "import olmo_core; print('olmo_core OK', olmo_core.__file__)" 2>&1 | tail -3
python -c "import corpus_reasoning" 2>&1 | tail -3
echo "--- building throwaway vllm venv to check its transformers ---"
python3 -m venv /root/vllm_probe_venv
/root/vllm_probe_venv/bin/pip install --quiet --upgrade pip
/root/vllm_probe_venv/bin/pip install --quiet vllm==0.25.1 2>&1 | tail -30
/root/vllm_probe_venv/bin/python -c "
import vllm, transformers
print('vllm', vllm.__version__)
print('transformers(vllm-venv)', transformers.__version__)
from transformers import Qwen3_5ForCausalLM
print('Qwen3_5ForCausalLM OK in vllm venv')
"
/root/vllm_probe_venv/bin/pip install --quiet nvidia-cuda-nvcc-cu12 2>&1 | tail -10
find /root/vllm_probe_venv -iname 'nvcc' 2>/dev/null
echo "=== DONE $(date '+%F %T') ==="
