# OLMoE3 small/medium profiling handoff

This branch is based on `allenai/OLMo-core:akshitab/moe-v2-core` at
`a19a19ec32ffeb597f3ccd3ff90e623ba5ad01c7`. It adds only the runtime features
and configuration needed to reproduce the current small and medium systems
tests: EMO, CuTe KDA PR 837, scalable softmax, Dolma 3.5, and the PP1 rowwise-EP
buffer initialization fixes.

The model definitions are in `olmoe3_small_medium_models.py`. The runnable
100-step Beaker harness is `olmoe3_small_medium_profile.py`.

## Model configurations

| Setting | Small | Medium |
|---|---:|---:|
| Active parameters | 794,230,912 | 2,387,524,992 |
| Active non-embedding parameters | 691,470,464 | 2,233,384,320 |
| Total parameters | 12,496,339,072 | 42,759,798,144 |
| Layers | 16 | 24 |
| `d_model` / latent dimension | 1024 / 512 | 1536 / 768 |
| Q heads / KV heads | 8 / 4 | 16 / 8 |
| KDA / full-attention layers | 14 / 2 | 21 / 3 |
| Full-attention layer indices | 7, 15 | 7, 15, 23 |

Shared architecture:

- 512 routed experts, top-16, with routed hidden size equal to `d_model`.
- One shared expert per sparse block, also with hidden size equal to `d_model`.
- Layer 0 is dense KDA with an `8 * d_model` shared MLP.
- KDA uses 128-wide heads, `expand_v=2`, negative eigenvalues, convolution size
  4, and the CuTe PR 837 kernel.
- Full attention uses GQA, FlashAttention 4, scalable softmax, elementwise
  gating, and head-wise QK norm. It does not use `FusedAttentionV2`.
- EMO document pools grow from 16 to 512 experts; evaluation uses all 512.
  Global load balancing uses the local-batch auxiliary loss.
- Latent dimension is exactly `d_model / 2`; vocabulary size is 100,352;
  input embeddings and the LM head are untied.
- Fused-v2 MoE blocks, peri-norm, BF16 training, and no activation or block
  recomputation. MXFP8 and two-batch overlap are disabled.

## Current profiling topologies

| Preset | GPUs | Nodes | PP | EP | Rank microbatch | Global batch | Grad accumulation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `small-64g` | 64 B300 | 8 | 1 | 1 | 4 sequences / 32,768 tokens | 8 Mi tokens | 4 |
| `medium-128g` | 128 B300 | 16 | 1 | 8 | 2 sequences / 16,384 tokens | 8 Mi tokens | 4 |

Both use ordinary DDP for dense parameters, compiled model and optimizer,
FP32 gradient accumulation/reduction, and rowwise NVSHMEM for routed-expert
transport. Reduce-scatter and shared EP output buffers are disabled. Runtime
symmetric allocation is forbidden after the initialization prewarm so a
buffer-sizing regression fails loudly instead of adding synchronization.

The harness reads Dolma 3.5 from `s3://ai2-llm`, uses sequence length 8192,
runs WSD with a two-step warmup and two-step decay, and defaults to 100 steps.
Its `8e-4` LR is the systems-test placeholder, not a final scientific LR. The
separate small critical-batch run currently uses `1.3e-3`, an 8 Mi-token batch,
and a 2,000-step warmup; a final medium training LR has not been selected.

## Run it

Install the checkout in the usual OLMo-core development environment, then
render each recipe before launching it:

```bash
export OLMOE3_BEAKER_WORKSPACE=ai2/olmo3p5-training
export OLMOE3_BEAKER_PRIORITY=urgent

export OLMOE3_PROFILE_PRESET=small-64g
python src/examples/olmo_ddp/olmoe3_small_medium_profile.py \
  dry_run profile-small-64g ai2/holmes
python src/examples/olmo_ddp/olmoe3_small_medium_profile.py \
  launch profile-small-64g ai2/holmes

export OLMOE3_PROFILE_PRESET=medium-128g
python src/examples/olmo_ddp/olmoe3_small_medium_profile.py \
  dry_run profile-medium-128g ai2/holmes
python src/examples/olmo_ddp/olmoe3_small_medium_profile.py \
  launch profile-medium-128g ai2/holmes
```

The launcher expects the standard per-user Beaker secrets:
`<username>_GITHUB_TOKEN`, `<username>_BEAKER_TOKEN`,
`<username>_WANDB_API_KEY`, AWS access/config secrets, and Google credentials.
The target workspace must contain them.

Useful controls:

- `OLMOE3_PROFILE_MAX_STEPS=20` changes the short-run length.
- `OLMOE3_ALLOWED_HOSTNAMES=node-a,node-b` pins an exact host set.
- `OLMOE3_WANDB_PROJECT=...` changes the W&B project.
- OLMo-core's built-in profiler callback is present but disabled. Enable a
  short rank-0 trace by appending
  `--trainer.callbacks.profiler.enabled=true`
  `--trainer.callbacks.profiler.skip_first=20`
  `--trainer.callbacks.profiler.wait=1`
  `--trainer.callbacks.profiler.warmup=2`
  `--trainer.callbacks.profiler.active=5` to a command. Do not use all-rank
  tracing by default; it can generate a very large result dataset.

Always use a run name unique to the preset. The script records the complete
model size, topology, EMO policy, kernel, precision, batch, and recomputation
state in W&B tags and notes.
