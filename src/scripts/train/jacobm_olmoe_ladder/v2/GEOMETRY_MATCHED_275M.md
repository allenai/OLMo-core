# Geometry-matched 275M design

Dense-ladder source: `scaling-ladders` commit `aaeeca3`,
`ladders/mainline/workloads/arch.py`. The candidate builder is
[`models/geometry_matched_275m.py`](models/geometry_matched_275m.py).

## Shared geometry

Both candidates train from initialization and adopt the dense 275M rung's
backbone geometry while retaining the tested MoE recipe:

| Field | Current 275M hybrid | Geometry-matched candidate |
|---|---:|---:|
| Model dimension | 768 | 640 |
| Layers | 12 | 10 |
| Query heads | 8 | 8 |
| Head dimension | 128 | 128 |
| GDN `expand_v` | 1 | 2 |
| GDN layers | 2, 4, 6, 8, 10 | 0, 1, 2, 3, 5, 6, 7, 8 |
| Full-attention layers | 0, 1, 3, 5, 7, 9, 11 | 4, 9 |
| Dense FFN layers | 0 | 0 |
| MoE layers | 1–11 | 1–9 |
| Experts / top-k / shared | 256 / 8 / 1 | 256 / 8 / 1 |

Layer 0 uses GDN plus the dense-first FFN. Layers 1–9 use MoE FFNs; full
attention replaces GDN at layers 4 and 9. The experiment adopts the dense
hybrid's `expand_v=2` while retaining RoPE and `init_std=0.01`, so NoPE and
initialization remain separate interventions.

## Full-attention choice

The dense ladder is not uniformly GQA. Its 275M and 450M rungs use 8 query
heads and 8 KV heads (MHA); 810M and larger use 2:1 GQA. Our current 275M uses
8 query heads and 4 KV heads. Dense full-attention blocks also project a
1,024-element sigmoid gate from each token representation and multiply it into
the attention output before the output projection.

The builder therefore exposes four audited profiles:

| Profile | KV heads | Attention gate | Expert hidden | Dense-first hidden | Active params | Delta vs current hybrid active | Active non-embedding | Total params |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `geometry_only` | 4 | no | 664 | 5,976 | 290,782,080 | +0.898% | 226,556,800 | 3,136,314,240 |
| `geometry_nope` | 4 | no | 664 | 5,976 | 290,782,080 | +0.898% | 226,556,800 | 3,136,314,240 |
| `geometry_nope_gated` | 4 | elementwise | 664 | 5,976 | 292,092,800 | +1.353% | 227,867,520 | 3,137,624,960 |
| `dense_attention` | 8 | elementwise | 648 | 5,832 | 290,638,720 | +0.848% | 226,413,440 | 3,067,603,840 |

The primary `geometry_only` profile changes exactly the agreed geometry bundle
plus `expand_v=2`. Its 290.78M active parameters are 2.59M, or 0.898%, above
the current 288.19M hybrid. Its active non-embedding count is 7.31% higher,
because reducing `d_model` from 768 to 640 removes 12.85M embedding parameters
while the extra GDN capacity adds non-embedding parameters. Accordingly, the
Cx-derived token budgets are also 7.31% larger than the current hybrid's.

The optional `dense_attention` profile is not part of this first launch. It
exists for a later exact-KV-head/attention-gate alignment test.

The isolated NoPE-plus-gating test uses `geometry_nope_gated`: it keeps the
NoPE control's GQA ratio and expert widths fixed and adds only the dense
ladder's elementwise, full-precision sigmoid gate. Its exact rationale and
remaining dense-ladder differences are recorded in
[`ATTENTION_GATING_275M.md`](ATTENTION_GATING_275M.md).

The smoke and sweep launchers live under `launchers/pretraining/`. Smokes test
the largest legal per-Cx microbatches without writing checkpoints. The full
sweep uses the original hybrid's four learning rates at every Cx only after
those capacity tests pass.
