# Vision-MoE evaluation

These runners load the native OLMo-core distributed checkpoint directly. They do not
require an HF conversion and preserve the s002 model's EP8 topology.

## Recreate the uv environment

From the repository root:

```bash
mkdir -p /weka/oe-training-default/rustin/.cache/{cuda,tmp,torchinductor,triton,xdg}
UV_CACHE_DIR=/weka/oe-training-default/rustin/.cache/uv \
UV_PYTHON_INSTALL_DIR=/weka/oe-training-default/rustin/.uv-python \
  uv venv /weka/oe-training-default/rustin/envs/vision-moe-eval --python 3.12
UV_CACHE_DIR=/weka/oe-training-default/rustin/.cache/uv \
  uv pip sync requirements/vision-moe-eval.txt \
  --python /weka/oe-training-default/rustin/envs/vision-moe-eval/bin/python \
  --torch-backend cu128
```

The pinned lmms-eval commit has an upstream packaging omission: its wheel excludes
MMMU-Pro's extensionless default template. Install the repository-owned copy after every
`uv pip sync`:

```bash
install -m 0644 requirements/lmms-eval-overrides/mmmu_pro/_default_template_yaml \
  /weka/oe-training-default/rustin/envs/vision-moe-eval/lib/python3.12/site-packages/lmms_eval/tasks/mmmu_pro/_default_template_yaml
install -m 0644 requirements/lmms-eval-overrides/mmmu_pro/reasoning/_default_template_yaml \
  /weka/oe-training-default/rustin/envs/vision-moe-eval/lib/python3.12/site-packages/lmms_eval/tasks/mmmu_pro/reasoning/_default_template_yaml
```

Transformer Engine 2.9 does not publish a Torch 2.10 CXX11-ABI binding wheel. Build only
its Python binding after the sync (the base and CUDA packages remain lock-pinned):

```bash
CPATH=/weka/oe-training-default/rustin/envs/vision-moe-eval/lib/python3.12/site-packages/nvidia/cudnn/include \
LIBRARY_PATH=/weka/oe-training-default/rustin/envs/vision-moe-eval/lib/python3.12/site-packages/nvidia/cudnn/lib \
LD_LIBRARY_PATH=/weka/oe-training-default/rustin/envs/vision-moe-eval/lib/python3.12/site-packages/nvidia/cudnn/lib:/weka/oe-training-default/rustin/envs/vision-moe-eval/lib/python3.12/site-packages/transformer_engine/wheel_lib \
TMPDIR=/weka/oe-training-default/rustin/.cache/tmp \
XDG_CACHE_HOME=/weka/oe-training-default/rustin/.cache/xdg \
UV_CACHE_DIR=/weka/oe-training-default/rustin/.cache/uv \
  uv pip install --python /weka/oe-training-default/rustin/envs/vision-moe-eval/bin/python \
  --no-build-isolation --no-deps transformer-engine-torch==2.9.0
```

`uv pip check` reports one expected platform warning for `transformer-engine-cu12`: NVIDIA's
library-only 2.9 wheel is tagged CPython 3.10 even though the library is Python-ABI independent.
The separately built `transformer-engine-torch` binding is CPython 3.12. Verify the effective
runtime with `python -c 'import transformer_engine.pytorch'`; the native EP8 smoke and full
baseline also exercise this binding.

## Native s002 LLM baseline

```bash
PATH=/weka/oe-training-default/rustin/envs/vision-moe-eval/bin:$PATH \
HF_HOME=/weka/oe-training-default/rustin/hf-cache \
XDG_CACHE_HOME=/weka/oe-training-default/rustin/.cache/xdg \
TMPDIR=/weka/oe-training-default/rustin/.cache/tmp \
CUDA_CACHE_PATH=/weka/oe-training-default/rustin/.cache/cuda \
TORCHINDUCTOR_CACHE_DIR=/weka/oe-training-default/rustin/.cache/torchinductor \
OLMO_TRITON_CACHE_BASE=/weka/oe-training-default/rustin/.cache/triton \
OLMO_SYMM_VDEV2D_AUTO_BUILD=1 \
torchrun --standalone --nproc-per-node=8 src/scripts/eval/s002_downstream.py \
  --checkpoint /weka/oe-training-default/robertb/s002-step125500 \
  --task-group fast
```

Use the same command with a Stage-1 checkpoint to measure language-capability retention.
The runner pins 8,192 evaluator tokens per EP-DP rank by default; keep that batch size
unchanged for the pre/post comparison because reduced-precision loss metrics can vary slightly
with batch shape.
The output records whether the checkpoint was detected as a pretrained LM or multimodal
Stage-1 model, all task names, topology, limits, git revision, and dirty-tree state.

## Native Stage-1 MMMU-Pro

The base s002 checkpoint has no vision tower or connector and is deliberately rejected by
this runner. Use it after Stage-1 produces a multimodal checkpoint:

```bash
PATH=/weka/oe-training-default/rustin/envs/vision-moe-eval/bin:$PATH \
HF_HOME=/weka/oe-training-default/rustin/hf-cache \
XDG_CACHE_HOME=/weka/oe-training-default/rustin/.cache/xdg \
TMPDIR=/weka/oe-training-default/rustin/.cache/tmp \
CUDA_CACHE_PATH=/weka/oe-training-default/rustin/.cache/cuda \
TORCHINDUCTOR_CACHE_DIR=/weka/oe-training-default/rustin/.cache/torchinductor \
OLMO_TRITON_CACHE_BASE=/weka/oe-training-default/rustin/.cache/triton \
OLMO_SYMM_VDEV2D_AUTO_BUILD=1 \
torchrun --standalone --nproc-per-node=8 src/scripts/eval/s002_mmmu_pro.py \
  --checkpoint /weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/STAGE1_CHECKPOINT \
  --tasks mmmu_pro
```

For a harness smoke test, add `--limit 1 --max-new-tokens 4`. A run with either option is
partial and must not be reported as the benchmark score. The full protocol uses both
`mmmu_pro_vision` and `mmmu_pro_standard`, greedy decoding, the task's 256-token generation
limit, and no KV cache. Image features are cached once per request; LM decoding recomputes a
fixed, bucket-padded sequence because the custom multimodal FlexAttention mask does not yet
support cached decoding.
