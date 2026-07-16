# Tech-report MoE infrastructure benchmark

This directory contains matched eight-layer throughput entry points:

- `moe_8l_ddp.py`: OLMo-owned DDP with configurable EP degree
  (EP8 for the composed-system run; degree 1 for the no-EP run).
- `dense_8l_ddp.py`: shared-only dense blocks under the same OLMo-owned DDP
  train module and optimizer, with a 20,480-wide active MLP.
- `moe_8l_fsdp.py`: the OLMo3-style FSDP-based HSDP path with no EP.

Both use the same macro architecture and parameter-count target:

| Setting | Value |
| --- | ---: |
| Layers | 8 (layer 0 is dense; layers 1--7 are MoE) |
| Model / attention width | 4096 / 4096 |
| Attention heads / KV heads | 32 / 8 |
| Routed experts / top-k | 8, 32, 48, 64, or 128 / 4 |
| Routed / shared expert hidden size | 4096 / 4096 |
| Sequence length | 8192 |
| Default global batch | 4,194,304 tokens |
| Default rank microbatch | 2 sequences |
| Parameter count | Derived from the selected expert count and vocabulary |

By default, both variants use random top-k routing for this systems benchmark. Learned
routing is still evaluated so its operations remain in the autograd graph, but
random scores determine the selected experts. Model checkpoint saving is
disabled; activation checkpointing is disabled by default.

## Important interpretation boundary

This is the closest supported comparison, not a one-line parallelism toggle.
The fused-v2 `OLMoDDPModel` explicitly rejects FSDP wrapping, and the generic
Transformer stack explicitly rejects OLMo-owned DDP. Consequently:

- the DDP run uses the fused-v2 block, v2 router, OLMo multi-group reducer,
  OLMo optimizer, and rowwise NVSHMEM EP;
- the HSDP run uses the generic hybrid-MoE block, v1 router, generic optimizer,
  and FSDP2 wrapping.

The EP8 DDP comparison bundles the weight-lifetime choice with expert placement. Set
`TECH_REPORT_PARALLEL_DEGREE=1` to isolate DDP versus no-EP FSDP when
the full model fits. The dimensions, active experts, dense-first-layer pattern, capacity factor,
attention backend, data shape, and measurement window are matched. Router
details, shared-branch mixing, optimizer implementation, and sparse kernels are
not identical. Report this as a **DDP+EP stack versus FSDP/HSDP stack**
comparison, not as an isolated causal estimate of the DDP/FSDP API alone.

## Example launches

Single 8-GPU node:

```bash
torchrun --rdzv-endpoint=localhost:18396 --nproc-per-node 8 \
  /workspace/OLMo-core/src/scripts/train/tech_report/moe_8l_ddp.py \
  train tech-report-ddp localhost

TECH_REPORT_PARALLEL_DEGREE=1 torchrun \
  --rdzv-endpoint=localhost:18395 --nproc-per-node 8 \
  /workspace/OLMo-core/src/scripts/train/tech_report/moe_8l_ddp.py \
  train tech-report-ddp-noep localhost

torchrun --rdzv-endpoint=localhost:18394 --nproc-per-node 8 \
  /workspace/OLMo-core/src/scripts/train/tech_report/dense_8l_ddp.py \
  train tech-report-dense-ddp localhost

torchrun --rdzv-endpoint=localhost:18397 --nproc-per-node 8 \
  /workspace/OLMo-core/src/scripts/train/tech_report/moe_8l_fsdp.py \
  train tech-report-fsdp localhost
```

The existing remote launcher can run either script after its script-selection
logic is pointed at the desired entry point. Keep node allocation, GPU clocks,
global batch, rank microbatch, compile settings, warmup exclusion, and timed
step window identical.

Useful environment controls:

```bash
export TECH_REPORT_WANDB=1
export TECH_REPORT_WANDB_GROUP=moe-8l-run1
export TECH_REPORT_MAX_STEPS=100
export TECH_REPORT_PARALLEL_DEGREE=8
export TECH_REPORT_NUM_EXPERTS=64
export TECH_REPORT_GLOBAL_BATCH_SIZE=$((4 * 1024 * 1024))
export TECH_REPORT_PROFILE=0

# Optional ablations; defaults preserve the standard benchmark.
export TECH_REPORT_EP_PATH=auto # auto, sync_1d, or rowwise_nvshmem
export TECH_REPORT_ROWWISE_GET_NBLOCKS=256
export TECH_REPORT_ROWWISE_PUT_NBLOCKS=256
export TECH_REPORT_ROWWISE_WEIGHTED_PUT_NBLOCKS=128
export TECH_REPORT_MXFP8_MLP=0
export TECH_REPORT_MXFP8_ATTN_QKV=0
export TECH_REPORT_MXFP8_ATTN_OUT=0
export TECH_REPORT_MXFP8_ATTN_SAVE_QKV=0
export TECH_REPORT_RECOMPUTE_EACH_BLOCK=0
export TECH_REPORT_TWO_BATCH_OVERLAP=0
export TECH_REPORT_UNIFORM_ROUTING=0
export TECH_REPORT_PROFILE_START=20
export TECH_REPORT_PROFILE_END=25
```

`TECH_REPORT_MXFP8_MLP=1` selects the native rowwise MXFP8 path for the
routed, shared, and dense-first MLPs and therefore requires rowwise NVSHMEM EP.
The attention toggles are independent so the report can measure MLP-only,
attention-only, and combined MXFP8 configurations. Block recompute also makes
the rowwise dispatch/combine scratch buffers shared across blocks, matching the
production recompute memory policy.

`TECH_REPORT_TWO_BATCH_OVERLAP=1` selects the existing rowwise-NVSHMEM TBO
schedule, gives each rowwise scratch buffer two slots, and splits each rank
microbatch into two equal sequence batches. It therefore requires rowwise
NVSHMEM EP, an even `TECH_REPORT_RANK_MICROBATCH_SEQUENCES`, and no per-block
recompute. `TECH_REPORT_UNIFORM_ROUTING=1` replaces the default seeded random
top-k assignment with the benchmark router's deterministic uniform assignment.
Both controls default to zero, so existing launches retain the current normal
schedule and random routing.

The supported global batches are 128, 256, 512, 1024, 2048, 4096, 8192,
16384, and 32768 Ki tokens. The
dedicated multinode wrapper accepts bare script names and sweep values:

```bash
bash /workspace/beaker-toolbox/run_script_remotely.sh \
  /workspace/hostfile2 \
  /workspace/beaker-toolbox/node_cmd-tech-report-fsdp-vs-ddp.sh \
  "moe_8l_ddp 64 4096 8"
```

For the paper, report steady-state maximum-rank step time, useful tokens per
second per GPU, useful-token MFU, peak allocated/reserved memory, route drops,
and the exact number of microbatches. Do not include initialization, compile,
or prewarm steps in the steady-state average.
