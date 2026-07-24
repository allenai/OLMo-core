# Gated DeltaNet 2 integration

GDN2 is implemented as a separate OLMo-core sequence mixer. Existing GDN1
configs, checkpoints, imports, and image references are unchanged.

## Dependency isolation

GDN2 currently runs through an ephemeral Python overlay installed by
`src/scripts/train/jacobm_olmoe3_gdn2_beaker.sh`. The wrapper installs exactly:

```text
flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@cbb0a72efb55c18ca0ef4f298298317573ad2cb3
```

It uses `pip --target /tmp/... --no-deps --no-build-isolation`, prepends that
temporary directory only for the GDN2 process, asserts FLA version 0.5.2 and
the presence of `fla.ops.gdn2`, and then calls the ordinary hybrid wrapper.
Normal GDN1 jobs do not set the overlay variable and continue to use the
existing image environment. A derived immutable image can replace this
temporary mechanism after the architecture is selected.

## Audited 275M candidate

The first variant is `geometry_275m_gdn2_ev2_rope_gated`. Relative to the
current gated-RoPE geometry model it changes only the eight GDN1 mixers to
GDN2. It retains:

- 640 model width, ten layers, and GDN layers 0--3 and 5--8;
- global gated RoPE attention at layers 4 and 9;
- eight QK/value heads, 128-dimensional keys, and `expand_v=2`;
- `allow_neg_eigval=true`, four-token short convolutions, all MoE dimensions,
  dense-first FFN placement, initialization, data, and optimization settings.

GDN2 adds channel-wise decay and independent channel-wise erase/write gates,
plus the canonical low-rank decay and sigmoid output-gate projections. This
adds 1,762,296 parameters per recurrent mixer.

| Variant | Active params | Active non-embedding | Total params |
|---|---:|---:|---:|
| GDN1 gated-RoPE geometry | 292,092,800 | 227,867,520 | 3,137,624,960 |
| GDN2 gated-RoPE geometry | 306,191,168 | 241,965,888 | 3,151,723,328 |
| Difference | +14,098,368 | +14,098,368 | +14,098,368 |

## Functional gate and throughput results

The compiled one-GPU MB1 smoke passed its 8,192-token forward/backward dry run
and six real optimizer steps with finite loss and gradients, zero skipped
updates, and exit code zero. Steady active/reserved memory was approximately
64.4/65.4 GiB. The result-bearing work is
[01KY8MNNDVT0BMFD20F7ZSS95P](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8MNNDVT0BMFD20F7ZSS95P).

The full systems comparison followed the completed GDN1/SWA protocol:

1. qualify MB8 and MB16 on one GPU at both 2 Mi and 4 Mi optimizer batches;
2. if MB16 passes, run the exact 16-cell 2/4/8-GPU EP1, code-default EP, and
   reduce-scatter matrix for 50 steps with compile enabled;
3. save no checkpoints or evals and record final-ten median TFLOPs/GPU,
   TPS/GPU, aggregate TPS, step time, MFU, peak active/reserved memory, and
   skipped-step count alongside the GDN1 and SWA rows.

The four-cell capacity work
[01KY8N6V09MWSNSY51BBWD4X33](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8N6V09MWSNSY51BBWD4X33)
passed 4/4 tasks. MB16 fit at both optimizer batches, peaking at approximately
253.7 GiB active and 257.0 GiB reserved memory. The full matrix work
[01KY8NMMR9AETXTGWR11Y51QWQ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8NMMR9AETXTGWR11Y51QWQ)
passed 16/16 tasks. All 20 jobs completed with zero skipped optimizer updates.

The table reports final-ten median TFLOPs/GPU and thousands of tokens/s. All
cells use MB16.

| Batch | 1 GPU EP1 | 2 GPU EP1 | 4 GPU EP1 | 4 GPU RS | 8 GPU EP1 | 8 GPU RS |
|---:|---:|---:|---:|---:|---:|
| 2 Mi | 420.4 / 260.3 | 407.0 / 252.0 | 404.3 / 250.3 | 402.6 / 249.3 | 385.1 / 238.4 | 387.8 / 240.1 |
| 4 Mi | 413.9 / 256.3 | 413.7 / 256.2 | 407.5 / 252.3 | 409.0 / 253.3 | 404.0 / 250.1 | 399.5 / 247.3 |

Across these matched cells, GDN2 processes 11.1--13.4% fewer tokens/s than
GDN1 and 32.1--36.0% fewer than SWA. Its reported TFLOPs/GPU are 5.5--7.9%
below GDN1, a smaller gap than raw throughput because GDN2 performs more
modeled work per token. Code-default full EP is 12.6--17.6% slower than EP1 at
the same world size. Reduce-scatter ranges from 1.1% slower to 0.7% faster, so
it does not offer a consistent benefit. EP1/all-reduce remains the default.

The complete machine-readable GDN1/GDN2/SWA results, including aggregate TPS,
step time, MFU, memory, Beaker job IDs, and W&B links, are in
`results/throughput/275m_gdn_gdn2_swa_large_batch_parallelism.csv`. Re-run
`results/throughput/collect_275m_throughput.py` to merge finalized Beaker work
idempotently.

### Backward recomputation diagnostic

FLA's opt-in `disable_recompute=True` path retains forward WY/state
intermediates to avoid reconstructing them in the backward. The exact one-GPU,
2 Mi-token, MB16 A/B work
[01KY8RRSFNTBCVHJCEC439GW11](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8RRSFNTBCVHJCEC439GW11)
did not fit. After compiling the first dry-run microbatch, the second attempted
a 24.50 GiB allocation with only 17.70 GiB free; PyTorch already held 245.55
GiB allocated. This was a real capacity failure, not fragmentation (only 0.37
GiB was reserved but unallocated).

The earlier MB20 fit boundary belonged to GDN1. GDN2 with normal recomputation
already peaks at 253.7 GiB active / 257.0 GiB reserved at MB16. Keep
`disable_recompute=False` for production and the first LR sweep. The opt-in
flag remains available for a separate smaller-MB kernel diagnostic but must
not be enabled implicitly.

## 275M LR sweep

The first GDN2 quality experiment mirrors the completed 275M geometry-matched
GDN1 gated-NoPE sweep. It was submitted as 16 urgent, unallocated Holmes tasks
in [work 01KY8TKEBSZHYBZYEC5NFB92YK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8TKEBSZHYBZYEC5NFB92YK).
It uses normal backward recomputation, EP1, compile, no in-loop evals, and the
established four-LR grid `4e-4`, `8e-4`, `1.6e-3`, `3.2e-3` at every Cx.
The model has NoPE on global-attention layers 4 and 9; every other model and
training setting matches the audited GDN2 gated-RoPE systems recipe.

The initially submitted gated-RoPE work
[01KY8SY728GAJN9MZ5B9VGZNP2](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8SY728GAJN9MZ5B9VGZNP2)
was canceled during compilation/dry-run, before its first optimizer step or
checkpoint, and is superseded by the gated-NoPE work above.

| Cx | Target tokens | Approx. steps | Global batch | GPUs | Rank MB | Approx. wall time/run |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.839B | 18,461 | 262,144 | 4 | 8 | 1.4--1.6 h |
| 2 | 9.679B | 24,615 | 393,216 | 4 | 12 | 2.7--3.0 h |
| 4 | 19.357B | 36,922 | 524,288 | 4 | 16 | 5.4--5.8 h |
| 8 | 38.715B | 49,229 | 786,432 | 8 | 12 | 5.6--6.0 h |

This is 16 jobs and 80 concurrent GPUs, approximately 340--350 GPU-hours if
all cells run cleanly. Token budgets follow the usual `20 * active
non-embedding parameters * Cx` rule and are therefore about 6.2% larger than
the GDN1 gated-NoPE budgets. Use the ordinary 10%-of-tokens warmup followed by
cosine decay to 0.1x peak LR, the established ephemeral-checkpoint policy, and
out-of-loop validation. Plot one GDN2 U-curve per Cx and compare the observed
optima against wide integration, first hybrid, and GDN1 gated-NoPE geometry.
