# Standalone partner model family — September 2026

This directory updates `akshitab/standalone` with our current **0.794B-active hero
architecture as Tiny**, the previous **3.781B-active Large as Small**, and two
proposed approximately 4x-active rungs above Small. Copy the files
together; no scaling-ladders repository is needed. These names are partner-family
names: **Small here is not the existing 0.8B hero**, and Medium here is not our
existing 2.388B production candidate.

| Config | Active params | Total params | Layers | d_model / latent | Expert hidden | Q / KV heads | KDA / FA | Active multiplier |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| Tiny | 0.794233472B | 12.496341632B | 16 | 1024 / 512 | 1024 | 8 / 4 | 14 / 2 | — |
| Small | 3.780515200B | 72.237847936B | 40 | 1536 / 768 | 1536 | 16 / 8 | 35 / 5 | 4.760x |
| Medium | 15.421227520B | 322.601566720B | 64 | 2560 / 1280 | 2560 | 24 / 12 | 56 / 8 | 4.079x |
| Large | 62.133719296B | 1310.163554560B | 80 | 4608 / 2304 | 4608 | 48 / 24 | 70 / 10 | 4.029x |

Reusing the old Large makes the first jump 4.76x, not 4x.
The next jumps are within 2% and 1% of 4x while retaining aligned widths and
eight-layer attention periods. Medium/Large increase both depth and width;
we do not change sparsity, latent compression or GQA to hit the targets.

**Large stores about 1.310 trillion parameters.** Keeping the expert count/top-k
fixed while growing active capacity makes these much larger than their active
labels suggest. BF16 weights alone are about 2.62 TB for Large, before master
weights, gradients, optimizer state, activations or communication buffers.

Tiny is the trained hero architecture; Small reuses the previous Large candidate.
Medium/Large are best guesses: parameter counts and configuration structure
are checked, but no GPU speed, memory, convergence or kernel-shape qualification
has been run for them. No learning rate or deployment topology is implied.

## Shared architecture

- 512 routed experts, top-16, one always-active shared expert; first block is dense
  with an 8*d_model SwiGLU hidden width. Later shared and routed hidden widths are d_model.
- Latent width is **exactly d_model/2**. Main, latent and expert widths are multiples
  of 256. Head dimension remains **128**, deliberately preserving the trained geometry.
  Alignment is a design constraint, not a TPU/GPU performance guarantee.
- Seven KDA blocks per gated full-attention block; zero-based FA indices 7,15,23,...
  KDA uses expand_v=2, negative eigenvalues, and bias-free causal convolution size 4.
- GQA is always 2:1; KDA head count equals query head count. Head counts are multiples
  of eight, but projected Q width / d_model is **not** constant across rungs.
- NoPE full attention, elementwise output gating, scalable softmax, and independent
  per-head Q/K RMSNorm gains (the updated hero architecture).
- Vocabulary 100352; untied embedding/output weights; embedding and four per-block
  RMSNorms; sqrt(d_model) embedding scale. Norm epsilon 1e-6, KDA output norm 1e-5.
- FP32 softmax router; selected top-k weights are normalized to sum 1, then scaled
  by 16 (`restore_weight_scale=True`). Router auxiliary loss weights 0.01 and 1e-5.
- EMO and global load balancing remain independently switchable. Default EMO pool
  is 16–512 experts/document during training, 512 at evaluation.

Active counts include the full embedding and untied output tables, shared experts,
attention, router, latent projections and top-16 routed experts per MoE block.
They are conventional active-parameter counts, not literal embedding rows read per token.

## Files

- `model_configs.py` is the dependency-free shared dimension/count table.
- `standalone_model.py` is a readable, unfused PyTorch reference implementation
  of all four rungs. It includes document-aware KDA, attention, EMo and
  standard routers, latent MoE, parameter accounting, and OLMo-style
  initialization. Use it for architecture inspection and small-shape correctness
  checks, not performance measurements.
- `fused_model.py` builds `tiny`, `small`, `medium`, `large` or a separate `30m` smoke using
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
python standalone_model.py --model-size tiny
python standalone_model.py --model-size large
```

The requested model is instantiated on the `meta` device, so these commands check
counts without allocating parameter storage. Add `--show-model` to print its structure.
Importing the module does not instantiate a model.

## Fused setup

Use a Linux CUDA environment. The OLMo-core commit is pinned because the scripts
depend on its KDA, EMO, global-balancing, per-head QK-gain and fused MoE v2 interfaces.
Install this pin separately: the older core sources underneath the standalone branch
do not support the updated QK flag. Do not replace this install with `pip install -e .`
from the standalone branch's repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install \
  'ai2-olmo-core[fa4,fla] @ git+https://github.com/allenai/OLMo-core.git@2610a90ced51542c10848a7d82e9534f3ef65923'
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

Inspect configurations without allocating model weights:

```bash
python fused_model.py --model-size 30m
python fused_model.py --model-size tiny --no-emo --global-load-balancing
python fused_model.py --model-size large
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

## Example Tiny benchmark (hardware qualification still required)

The following uses all eight launched ranks for the routed-expert shard. Dense
parameters follow the ladder's replicated DDP policy; the routed expert bank is
EP-sharded through rowwise NVSHMEM.

```bash
torchrun --standalone --nproc-per-node=8 \
  distributed_fused_benchmark.py \
  --model-size tiny \
  --ep-degree -1 \
  --sequence-length 8192 \
  --microbatch-sequences 1 \
  --warmup 2 \
  --iterations 5
```

Add `--include-optimizer-step` to time the configured AdamW update and allocate
its optimizer state. Compiled execution is the multi-GPU benchmark default.

This is the original synthetic benchmarking harness with updated model geometry,
not the full hero training launcher. It does not reproduce the complete hero
kernel/metrics optimization stack, global batch accumulation, data, schedule or
checkpointing. Do not compare its forward/backward-only TPS directly to hero TPS.
The example is not a memory guarantee, especially with optimizer state enabled.
The harness has DDP+EP only, no PP/FSDP; **do not run the larger rungs unsharded
or assume that eight GPUs can hold them**. Large will require a separate
parallelism and memory plan.

The benchmark reports synchronized iteration time, global tokens per second,
estimated TFLOP/s per GPU, peak allocated GPU memory, and exact active/total
parameter counts.

## Validation and provenance

Based on standalone branch commit `00ea561418c54968ea5043279e9450a8c45b97e7`.
Tiny geometry comes from the hero model, including its per-head QK-gain change.
The old `3p5b` target is replaced, not renamed: its head dimension, latent ratio,
attention ratio and normalization did not describe our current hero family.

The separate installation smoke retains its five-layer/32-expert toy geometry;
with scalable softmax it has 32,323,589 total / 29,964,293 active parameters. It is
not performance-representative and is not one of the four partner rungs.

Local verification checks all four PyTorch meta-module counts against native
OLMo-core config counts, both EMO choices, and the updated attention's packed
document isolation/gradients. This is structural validation, **not** end-to-end
numerical equivalence or distributed GPU qualification.

Native fused module construction could not be exercised in the local CPU-only
environment because the required FLA runtime is not installed there. Config-only
validation passed for all five geometries and all four EMO/global-balancing flag
combinations. Tiny also matches the canonical core model configuration after the
QK-gain update, apart from the documented portable FLA versus custom-CuTe backend
selection. No training or benchmark jobs were launched for this partner update.
