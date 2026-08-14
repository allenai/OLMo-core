# Standalone OLMoE3 model references

This directory contains portable implementations and benchmarks for the smallest
and largest OLMoE3 ladder configurations. The files can be copied together and
used independently of the scaling-ladders repository.

## Files

- `standalone_model.py` is a readable, unfused PyTorch reference implementation
  of the 3.5B-active model. It includes document-aware KDA, attention, EMo and
  standard routers, latent MoE, parameter accounting, and OLMo-style
  initialization. Use it for architecture inspection and small-shape correctness
  checks, not performance measurements.
- `fused_model.py` builds the canonical `30m` smoke or `3p5b` target config using
  OLMo-core's production FLA KDA, FlashAttention 4, and fused MoE v2 paths. Its
  default execution validates the config and prints parameter counts without
  constructing the model.
- `distributed_fused_benchmark.py` is the runnable CUDA benchmark. It applies
  DDP and, when requested, expert parallelism before materializing parameters,
  then measures compiled BF16 forward/backward iterations. AdamW can optionally
  be included in the timed region.

## Unfused reference setup

Python 3.12+ and PyTorch are sufficient:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
python standalone_model.py
```

The full model is instantiated on the `meta` device, so running the script prints
its structure and counts without allocating 65B parameters.

## Fused setup

Use a Linux CUDA environment. The OLMo-core commit is pinned because the scripts
depend on its current KDA, EMo, global-balancing, and fused MoE v2 interfaces:

```bash
python -m venv .venv
source .venv/bin/activate
pip install \
  'ai2-olmo-core[fa4,fla] @ git+https://github.com/allenai/OLMo-core.git@f2cf93839a823b88955e94a851c808829c5201ba'
```

Transformer Engine is required even though full attention uses FA4: at the pinned
OLMo-core commit, the fused MoE path uses TE's `moe_permute` and `moe_unpermute`
operators. OLMo-core does not declare TE in its `fa4` or `fla` extras. The
single-GPU smoke benchmark supplies differentiable PyTorch permutation
operations and repairs any misaligned grouped-MM operand at the kernel boundary,
so TE and NVCC are not required for that test.

The multi-GPU EP benchmark does require TE. Install a compatible prebuilt wheel,
or run the following in a CUDA development image that provides `nvcc`:

```bash
pip install --no-build-isolation 'transformer-engine[pytorch]'
```

Verify that both its Python and compiled extensions load:

```bash
python -c \
  'import transformer_engine.pytorch; import transformer_engine_torch; print("TE available")'
```

PyTorch supplies its compatible Triton build. Multi-GPU rowwise expert
parallelism additionally requires the NVSHMEM environment expected by OLMo-core.

Inspect either fused configuration without allocating model weights:

```bash
python fused_model.py --model-size 30m
python fused_model.py --model-size 3p5b --no-emo --global-load-balancing
```

EMo and global load balancing are independent switches:

```text
--emo / --no-emo
--global-load-balancing / --no-global-load-balancing
```

## Single-GPU smoke benchmark

The `30m` rung exercises the fused forward and backward paths without EP or
NVSHMEM:

```bash
torchrun --standalone --nproc-per-node=1 \
  distributed_fused_benchmark.py \
  --model-size 30m \
  --ep-degree 1 \
  --sequence-length 512 \
  --microbatch-sequences 1 \
  --warmup 1 \
  --iterations 2 \
  --no-compile
```

The Transformer Engine-free fallback is intentionally run without the outer
`torch.compile` wrapper. It also makes a differentiable copy of a grouped-MM
operand only when its storage pointer is not 16-byte aligned; PyTorch rejects
such operands in both compiled and eager execution. This does not disable the
fused FLA, FlashAttention, or grouped-MM kernels. If Transformer Engine is
installed, the fallback is not selected and compiled single-GPU execution can
be used.

## Multi-GPU target benchmark

The following uses all eight launched ranks for the routed-expert shard. Dense
parameters follow the ladder's replicated DDP policy; the routed expert bank is
EP-sharded through rowwise NVSHMEM.

```bash
torchrun --standalone --nproc-per-node=8 \
  distributed_fused_benchmark.py \
  --model-size 3p5b \
  --ep-degree -1 \
  --sequence-length 8192 \
  --microbatch-sequences 1 \
  --warmup 2 \
  --iterations 5
```

Add `--include-optimizer-step` to time the configured AdamW update and allocate
its optimizer state. Compiled execution is the multi-GPU benchmark default.

The benchmark reports synchronized iteration time, global tokens per second,
estimated TFLOP/s per GPU, peak allocated GPU memory, and exact active/total
parameter counts.

## Canonical counts

| Config | Total parameters | Active parameters | Active non-embedding |
| --- | ---: | ---: | ---: |
| `30m` | 32,323,588 | 29,964,292 | 17,119,236 |
| `3p5b` | 65,342,371,200 | 3,479,664,000 | 3,299,833,216 |
