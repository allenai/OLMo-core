# Aggressive MXFP8 ladder plan

## Status and objective

The 275M LR sweep was submitted on 2026-07-27. Larger-model configurations
are audited and launchable in the trainer, but no larger MXFP8 jobs have been
launched.

The immediate experiment is a full 275M Cx1/Cx2/Cx4/Cx8 LR sweep using the
aggressive OLMo-core MXFP8 recipe demonstrated by
`src/examples/olmo_ddp/OLMoE3-dev-u001.py`. The goal is to determine whether
this precision recipe preserves pretraining loss, stability, and downstream
quality before testing whether its systems benefit appears at larger model
sizes. The completed 275M single-B300 throughput smoke is a functional
qualification, not evidence that MXFP8 cannot improve throughput at scale.

This experiment is intentionally different from Kimi K3's post-training QAT
recipe. Kimi K3 quantizes routed-expert weights to MXFP4 and routed-expert
activations to MXFP8 while keeping shared experts, attention, and routers in
higher precision. Our experiment uses MXFP8 much more broadly during
pretraining.

## Exact 275M candidate

Parent: `geometry_275m_kda_ev2_neg_nope_gated`.

Candidate: `geometry_275m_kda_ev2_neg_nope_gated_mxfp8_672`.

The residual-stream model dimension is **not** changed: `d_model` remains 640.
Only the expert hidden dimension changes from 664 to 672, which also changes
the dense-first FFN from 5,976 to 6,048. These changes make every expert
projection dimension divisible by the 32-element MXFP8 block size.

| Quantity | BF16 parent | MXFP8 candidate | Delta |
|---|---:|---:|---:|
| `d_model` | 640 | 640 | 0 |
| Expert hidden width | 664 | 672 | +8 (+1.205%) |
| Dense-first FFN width | 5,976 | 6,048 | +72 (+1.205%) |
| Active parameters | 290,503,488 | 291,885,888 | +1,382,400 (+0.476%) |
| Active non-embedding parameters | 226,278,208 | 227,660,608 | +1,382,400 (+0.611%) |
| Stored parameters | 3,136,035,648 | 3,171,701,568 | +35,665,920 (+1.137%) |

Everything else remains fixed: ten layers; KDA at layers 0--3 and 5--8;
gated NoPE full attention at layers 4 and 9; `expand_v=2`; negative
eigenvalues enabled; 8 Q / 4 KV heads; one dense-first FFN; the routed/shared
expert counts; initialization; tokenizer; data mix; and optimizer recipe.

## Precision and kernel configuration

Match the aggressive example configuration on every sweep point:

- `TransformerType.moe_fused_v2` and `TransformerBlockType.moe_fused_v2`;
- `AttentionType.fused_v2` with FlashAttention-4 on layers 4 and 9;
- MXFP8 QKV and output-projection GEMMs on both full-attention layers;
- `mxfp8_save_qkv_for_backward=False`;
- rowwise MXFP8 for every routed and shared FFN, including the dense-first
  single shared-expert representation;
- `OLMO_MXFP8_SCALE_MODE=rceil` set before importing OLMo-core; and
- KDA recurrence, routers, norms, embeddings, LM head, attention core, and
  other elementwise operations remain in their current higher-precision path.

Use EP=1 through the now-qualified no-EP rowwise-MXFP8 path. The qualification
completed 50 compiled MB16 optimizer steps with finite loss and zero skipped
updates. Its isolated patch is commit `8fb79223c` on
`codex/ep1-mxfp8`, based directly on `akshitab/moe-v2-core`.

## Planned 275M LR sweep

Use the established four-point grid at every data multiple:
`4e-4`, `8e-4`, `1.6e-3`, and `3.2e-3`. This brackets the observed optima of
the existing 275M KDA/GDN families while preserving direct comparability with
their U-plots.

Token budgets use `20 * active_non_embedding_params * Cx` for the new
227,660,608-parameter non-embedding count. Step counts below round up to a
complete optimizer batch.

| Cx | Target tokens | Global batch | GPUs | Rank MB | Accum | Steps | Jobs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,553,212,160 | 262,144 | 2 | 16 | 1 | 17,370 | 4 |
| 2 | 9,106,424,320 | 393,216 | 2 | 12 | 2 | 23,159 | 4 |
| 4 | 18,212,848,640 | 524,288 | 2 | 16 | 2 | 34,739 | 4 |
| 8 | 36,425,697,280 | 786,432 | 2 | 16 | 3 | 46,318 | 4 |

This resource-conscious layout requests 32 B300s if all 16 jobs run
concurrently. MB16 was already qualified with the full aggressive precision
configuration; Cx2's MB12 is smaller. The older master settings table lists
MB8/accumulation 3 for two-GPU Cx2, while the later geometry/KDA manifests use
the already validated MB12 shape. MB12/accumulation 2 was approved for this
sweep and is now the canonical master setting. At the qualified one-GPU rate, a
two-GPU job would ideally finish Cx1/Cx2/Cx4/Cx8 in approximately
3/6/12/24 hours. Allow roughly 20% uncertainty until we measure the smaller
production batches on two GPUs. A higher-GPU wall-clock layout can be chosen
before launch, but it must preserve the global batches above.

Use 8,192-token sequences, `OLMo_mix_0925`, the Dolma 2 tokenizer, 10% token
warmup, and cosine decay to 10% of peak LR. Save rolling ephemeral checkpoints
every 500 steps with `remove=ephemeral_only` plus the permanent final
checkpoint. Disable all in-loop and on-finish evaluators; run the full
validation suite from final checkpoints afterward.

Proposed run-name prefix:
`pt-275m-kda-ev2-neg-nope-gated-mxfp8-672`.

The 16 jobs were submitted as urgent unallocated Holmes work in
[Beaker experiment `01KYJPTZ3J4VHGBH0FSVAQRDGC`](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJPTZ3J4VHGBH0FSVAQRDGC).
At the latest post-submission check, all 16 tasks had completed initial kernel
compilation and reached real optimizer steps; none had failed.

## Audited larger-size configurations

The larger candidate variant is
`geometry_matched_kda_ev2_neg_nope_gated_mxfp8_aligned`. It uses the same KDA
`expand_v=2`, negative-eigenvalue, gated-NoPE architecture as the existing
BF16 KDA family, then rounds only the expert hidden width to the nearest
multiple of 32. All residual-stream widths, depths, heads, attention/KDA
placements, expert counts, and initialization remain unchanged.

| Size | `d_model` | Layers | Q / KV | KDA / full | Old expert | MXFP8 expert | Dense-first | Active params | Active delta vs old KDA | Stored params |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 480M | 768 | 15 | 8 / 4 | 12 / 3 | 840 | 832 | 7,488 | 496,253,280 | -2,488,320 (-0.499%) | 7,151,827,296 |
| 810M | 1,024 | 15 | 16 / 8 | 12 / 3 | 1,032 | 1,024 | 9,216 | 835,921,856 | -3,317,760 (-0.395%) | 11,757,889,472 |
| 1.2B | 1,280 | 20 | 16 / 8 | 16 / 4 | 952 | 960 | 8,640 | 1,256,992,512 | +5,529,600 (+0.442%) | 18,627,309,312 |

Exact construction checks prove that every shared/routed expert input and
hidden dimension is divisible by 32, layer 0 remains dense-first, all expected
KDA and full-attention layers are present, the model/block type is
`moe_fused_v2`, full attention is `fused_v2` plus FlashAttention-4, every FFN
has rowwise MXFP8 enabled, and both full-attention projection pairs have MXFP8
enabled. No larger job should be launched before size-specific MB/EP/GPU
smokes, because the 275M throughput result does not determine the larger
systems optimum.

## Proposed KDA continuation strategy

If the 275M aggressive-MXFP8 curves remain stable and competitive, make the
aligned aggressive-MXFP8 configuration the coherent continuation family for
480M, 810M, and 1.2B rather than launching another complete BF16 KDA wave.
Retain all completed BF16 KDA results as architectural evidence and allow
already-running BF16 jobs to finish; do not duplicate them merely to obtain an
exact 32-aligned BF16 control yet.

Before the larger full wave:

1. select only observed, bracketed 275M MXFP8 LRs;
2. decide whether larger LRs should use the existing wide-transfer values or
   a measured per-Cx MXFP8 adjustment from the 275M sweep;
3. smoke 480M/810M/1.2B with the aggressive precision settings, starting from
   the largest plausible MB and testing EP1 first where it fits;
4. test rowwise EP only where memory or measured throughput requires it,
   especially for 1.2B; and
5. record same-setting throughput, memory, skipped steps, and checkpoint
   save/load before launching the full transferred-LR cells.

## Analysis and promotion gates

Produce one intervention-only U-plot containing all four Cx curves and an
observed-best summary. Use the final-250M-token training CE and the existing
bracketing rules; never promote a fitted-but-unobserved LR. Backfill C4 and the
full validation suite for each selected final checkpoint.

An exact BF16 control with the 672 expert width, fused attention, and FA4 is
deferred to save compute. Therefore comparisons with the existing 664-wide
BF16 KDA family are approximate and must be labeled as such: the precision
change is confounded with a +0.476% active-parameter increase and the
attention implementation/backend change. This first sweep can establish that
aggressive MXFP8 is healthy and competitive, but it cannot attribute a small
quality difference solely to precision.

Before promoting the recipe to larger sizes, require:

1. four finished points and a bracketed curve at every Cx;
2. no systematic nonfinite/skipped-step behavior;
3. training and validation results that are competitive with the nearby KDA
   and wide-integration trajectories, accounting for normal run noise; and
4. successful final-checkpoint save/load plus a plan for fused-QKV and KDA HF
   conversion. Strict HF/logit parity can follow before downstream serving;
   it does not need to block the pretraining sweep itself.

After the 275M quality gate, smoke the chosen aggressive configuration at
480M, 810M, and 1.2B before selecting larger-size GPU/EP/MB layouts. Larger
models are the meaningful test of whether MXFP8 improves training throughput
or enables larger microbatches.
