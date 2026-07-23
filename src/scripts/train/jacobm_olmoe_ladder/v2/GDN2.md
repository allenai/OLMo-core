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

## Functional gate and throughput plan

The compiled one-GPU MB1 smoke passed its 8,192-token forward/backward dry run
and six real optimizer steps with finite loss and gradients, zero skipped
updates, and exit code zero. Steady active/reserved memory was approximately
64.4/65.4 GiB. The result-bearing work is
[01KY8MNNDVT0BMFD20F7ZSS95P](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8MNNDVT0BMFD20F7ZSS95P).

The full systems comparison follows the completed GDN1/SWA protocol:

1. qualify MB8 and MB16 on one GPU at both 2 Mi and 4 Mi optimizer batches;
2. if MB16 passes, run the exact 16-cell 2/4/8-GPU EP1, code-default EP, and
   reduce-scatter matrix for 50 steps with compile enabled;
3. save no checkpoints or evals and record final-ten median TFLOPs/GPU,
   TPS/GPU, aggregate TPS, step time, MFU, peak active/reserved memory, and
   skipped-step count alongside the GDN1 and SWA rows.

The four-cell capacity work is
[01KY8N6V09MWSNSY51BBWD4X33](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8N6V09MWSNSY51BBWD4X33).
