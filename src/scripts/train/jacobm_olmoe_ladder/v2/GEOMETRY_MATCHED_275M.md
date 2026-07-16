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
| GDN layers | 2, 4, 6, 8, 10 | 0, 1, 2, 3, 5, 6, 7, 8 |
| Full-attention layers | 0, 1, 3, 5, 7, 9, 11 | 4, 9 |
| Dense FFN layers | 0 | 0 |
| MoE layers | 1–11 | 1–9 |
| Experts / top-k / shared | 256 / 8 / 1 | 256 / 8 / 1 |

Layer 0 uses GDN plus the dense-first FFN. Layers 1–9 use MoE FFNs; full
attention replaces GDN at layers 4 and 9. The experiment retains RoPE,
`expand_v=1`, and `init_std=0.01` so NoPE, `expand_v=2`, and initialization
remain separate interventions.

## Full-attention choice

The dense ladder is not uniformly GQA. Its 275M and 450M rungs use 8 query
heads and 8 KV heads (MHA); 810M and larger use 2:1 GQA. Our current 275M uses
8 query heads and 4 KV heads. Dense full-attention blocks also project a
1,024-element sigmoid gate from each token representation and multiply it into
the attention output before the output projection.

The builder therefore exposes two audited profiles:

| Profile | KV heads | Attention gate | Expert hidden | Dense-first hidden | Active params | Active non-embedding | Delta vs current hybrid non-embedding | Total params |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `geometry_only` | 4 | no | 664 | 5,976 | 275,019,648 | 210,794,368 | -0.156% | 3,120,551,808 |
| `dense_attention` | 8 | elementwise | 648 | 5,832 | 274,876,288 | 210,651,008 | -0.224% | 3,051,841,408 |

The strict `dense_attention` profile is 0.224% below the dense model's
275,493,760 active parameters and 0.292% below its 211,268,480 active
non-embedding parameters. The `geometry_only` profile is the cleaner isolated
test of the previously planned width/depth/mixer-ratio bundle. KV-head geometry
and the attention gate can then be added together or tested separately.

No training launcher is attached yet. Choose the profile before constructing
the LR-sweep manifest.
