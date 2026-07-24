# 275M throughput findings

## Scope

The matched test matrix covers the gated-RoPE geometry model with SWA, GDN1,
and GDN2 on B300s at 8,192-token context. Each mixer has 18 finalized 50-step
cells spanning 2/4 Mi-token global batches, 1/2/4/8 GPUs, EP1, code-default full
EP, and reduce-scatter. All 54 cells completed with zero skipped optimizer
updates. Raw tokens/s is the appropriate wall-clock comparison because the
three mixers do different amounts of modeled work per token.

## Headline results

All values below use MB16 and EP1/all-reduce. TPS/GPU is in thousands.

| Global batch | Mixer | TFLOPs/GPU | TPS/GPU | MFU | Peak active / reserved |
|---:|---|---:|---:|---:|---:|
| 2 Mi | SWA | 603.7 | 406.5k | 26.83% | 200.8 / 203.7 GiB |
| 2 Mi | GDN1 | 453.7 | 298.8k | 20.16% | 221.5 / 226.0 GiB |
| 2 Mi | GDN2 | 420.4 | 260.3k | 18.68% | 253.7 / 256.8 GiB |
| 4 Mi | SWA | 570.5 | 384.1k | 25.35% | 200.8 / 203.9 GiB |
| 4 Mi | GDN1 | 445.3 | 293.3k | 19.79% | 221.5 / 226.1 GiB |
| 4 Mi | GDN2 | 413.9 | 256.3k | 18.40% | 253.7 / 257.0 GiB |

At 8 GPUs and a 4 Mi batch, aggregate throughput is 3.04M tokens/s for SWA,
2.26M for GDN1, and 2.00M for GDN2. Across all matched cells, GDN2 is
11.1--13.4% slower in raw TPS than GDN1 and 32.1--36.0% slower than SWA. Its
TFLOPs/GPU is only 5.5--7.9% below GDN1 because its modeled FLOPs/token is about
6.4% higher. Roughly half of the raw GDN2/GDN1 gap is therefore additional
model work and half is lower kernel/memory efficiency.

## Capacity and parallelism

- MB16 is the largest formally qualified GDN2 microbatch. It fits narrowly on
  one B300; the 4 Mi case reserves 257.0 GiB. The production sweep therefore
  uses normal backward recomputation and MB8/12/16 according to its data scale.
- Retaining GDN2 forward intermediates (`disable_recompute=True`) does not fit
  at MB16: the diagnostic exhausted real device capacity while attempting a
  24.5 GiB allocation, rather than failing from allocator fragmentation.
- EP1/all-reduce is the production default. Full EP costs 12.6--17.6% relative
  to EP1 at the same world size. Reduce-scatter ranges from 1.1% slower to 0.7%
  faster and provides no consistent benefit.
- More GPUs reduce wall time but generally lower per-GPU efficiency for this
  small model. Use the fewest GPUs compatible with the desired job duration.

## 1:1 attention-ratio follow-up

The 2026-07-24 follow-up changes only depth and mixer placement. Every model
strictly alternates a recurrent/local mixer on even layers with gated global
RoPE attention on odd layers. Model width, head geometry, `expand_v`, MoE
dimensions/counts, dense-first placement, initialization, and all other mixer
settings are unchanged. Layer count is selected to come as close as possible
to the original GDN1 speed model's 292,092,800 active parameters while retaining
an exact 1:1 ratio.

| Mixer | Layers | Recurrent/local | Global attention | Active params | Active non-embedding | Total params | Active delta vs old GDN1 |
|---|---:|---|---|---:|---:|---:|---:|
| SWA | 12 | 0/2/4/6/8/10 | 1/3/5/7/9/11 | 295,500,032 | 231,274,752 | 3,773,372,672 | +1.17% |
| GDN1 | 10 | 0/2/4/6/8 | 1/3/5/7/9 | 284,148,560 | 219,923,280 | 3,129,680,720 | -2.72% |
| GDN2 | 10 | 0/2/4/6/8 | 1/3/5/7/9 | 292,960,040 | 228,734,760 | 3,138,492,200 | +0.30% |

The 12-layer SWA model has more *total* parameters because adding depth adds
two full routed-expert banks; active parameters are the matching axis used for
the ladder and throughput comparison.

Each model ran two urgent, unallocated, single-B300 Holmes cells: 2 Mi and 4 Mi
global batches, MB16, EP1/all-reduce, 8,192-token sequences, 50 optimizer
steps, compile enabled, and no checkpoints or evals. All six cells finished
successfully with zero skipped optimizer updates. Results below use the same
final-ten medians and memory checks as the original matrix. TPS/GPU is in
thousands.

| Global batch | Mixer | TFLOPs/GPU | TPS/GPU | MFU | Peak active / reserved |
|---:|---|---:|---:|---:|---:|
| 2 Mi | SWA 1:1 | 592.15 | 325.1k | 26.32% | 234.2 / 237.6 GiB |
| 2 Mi | GDN1 1:1 | 499.00 | 311.4k | 22.18% | 215.4 / 219.1 GiB |
| 2 Mi | GDN2 1:1 | 479.35 | 288.2k | 21.31% | 240.0 / 243.0 GiB |
| 4 Mi | SWA 1:1 | 573.30 | 314.7k | 25.48% | 234.2 / 237.7 GiB |
| 4 Mi | GDN1 1:1 | 500.00 | 312.0k | 22.23% | 215.4 / 219.6 GiB |
| 4 Mi | GDN2 1:1 | 472.95 | 284.4k | 21.02% | 240.0 / 243.5 GiB |

Relative to the original 4:1 mixer/full-attention controls, the 1:1 GDN1
model improves raw TPS by 4.2% at 2 Mi and 6.4% at 4 Mi; GDN2 improves by
10.7% and 11.0%. GDN2 remains 7.4--8.9% slower in raw TPS than GDN1 at 1:1,
narrower than the original 11.1--13.4% gap. The deeper active-matched SWA
model is 20.0% and 18.1% slower than its ten-layer 4:1 control. Consequently,
SWA is only 4.4% faster than GDN1 at 2 Mi and 0.9% faster at 4 Mi in this
active-matched 1:1 comparison.

- [GDN1 1:1 work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8YWHCPGVB40W51W7G69VM5)
- [GDN2 1:1 work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8YWJQEKHCBKQS9QYBHCY7T)
- [SWA 1:1 work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8YWKZE3QCZ4SCXD224D5RY)
- [Machine-readable 1:1 results](275m_gdn_gdn2_swa_1to1_single_gpu.csv)

## Artifacts

- [Machine-readable 54-cell results](275m_gdn_gdn2_swa_large_batch_parallelism.csv)
- [GDN2 capacity work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8N6V09MWSNSY51BBWD4X33)
- [GDN2 parallelism work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8NMMR9AETXTGWR11Y51QWQ)
- [GDN2 no-recompute diagnostic](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8RRSFNTBCVHJCEC439GW11)
