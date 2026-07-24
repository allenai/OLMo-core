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

## Artifacts

- [Machine-readable 54-cell results](275m_gdn_gdn2_swa_large_batch_parallelism.csv)
- [GDN2 capacity work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8N6V09MWSNSY51BBWD4X33)
- [GDN2 parallelism work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8NMMR9AETXTGWR11Y51QWQ)
- [GDN2 no-recompute diagnostic](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8RRSFNTBCVHJCEC439GW11)
