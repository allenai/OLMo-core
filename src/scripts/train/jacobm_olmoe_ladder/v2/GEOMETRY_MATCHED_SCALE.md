# Geometry-matched hybrid scale design

Dense-ladder source: `scaling-ladders` commit `aaeeca3`,
`ladders/mainline/workloads/arch.py`. The canonical builder is
[`models/geometry_matched_scale.py`](models/geometry_matched_scale.py), selected
from the scale trainer with `geometry_matched_gdn_ev2`.

No training jobs have been launched from these larger-size configurations.
Future smokes and training runs return to allocated Holmes capacity; the
unallocated exception applied only to the first 275M geometry sweep and the
branch-comparison experiments.

## Scaling rule

Each rung adopts the corresponding dense ladder geometry while keeping the
established MoE recipe:

- 480M maps to dense 450M geometry, 810M maps to dense 810M geometry, and
  1.2B maps to dense 1.4B geometry.
- Full attention is every fifth layer (layers 4, 9, 14, and 19 when present);
  all other layers use GatedDeltaNet with `expand_v=2`.
- Layer 0 retains the dense-first FFN. All other layers retain 256 routed
  experts, top-k 8, and one shared expert.
- Expert widths are 8-aligned and chosen to match the corresponding trained
  `integration_wide_gdn_ev1` model's total active parameters as closely as
  possible.
- As in the primary 275M `geometry_only` experiment, full attention retains
  the current model's GQA ratio, RoPE, and no output gate. Initialization
  remains `init_std=0.01`. Exact dense KV geometry/gating, NoPE, and the
  `0.02` initialization are still separate interventions.

## Exact configurations

| Named size | Dense geometry rung | `d_model` | Layers | Q / KV heads | GDN / full layers | Expert hidden | Dense-first hidden |
|---|---|---:|---:|---:|---:|---:|---:|
| 275M | 275M | 640 | 10 | 8 / 4 | 8 / 2 | 664 | 5,976 |
| 480M | 450M | 768 | 15 | 8 / 4 | 12 / 3 | 840 | 7,560 |
| 810M | 810M | 1,024 | 15 | 16 / 8 | 12 / 3 | 1,032 | 9,288 |
| 1.2B | 1.4B | 1,280 | 20 | 16 / 8 | 16 / 4 | 952 | 8,568 |

The 810M query-head count changes from 12 to 16 as part of matching dense
geometry; its 2:1 GQA ratio is unchanged. The 480M model retains 8 Q / 4 KV
instead of adopting the dense 450M rung's 8 Q / 8 KV, exactly paralleling the
primary 275M geometry-only control.

## Parameter matching

| Named size | Existing hybrid active | Geometry active | Delta | Active non-embedding | Non-embedding delta | Stored params |
|---|---:|---:|---:|---:|---:|---:|
| 275M | 288,194,512 | 290,782,080 | +0.898% | 226,556,800 | +7.310% | 3,136,314,240 |
| 480M | 501,228,784 | 501,137,856 | -0.018% | 424,067,520 | +6.424% | 7,220,707,776 |
| 810M | 859,400,792 | 858,237,056 | -0.135% | 755,476,608 | +3.355% | 11,865,532,544 |
| 1.2B | 1,288,662,592 | 1,289,441,280 | +0.060% | 1,160,990,720 | +2.333% | 18,515,005,440 |

The token budgets will increase by the non-embedding deltas because the
pretraining trainer derives Chinchilla tokens from active non-embedding
parameters. This is intentional and matches the established 275M experiment:
quality should be compared at the rung's derived Cx budget and with explicit
active-parameter/FLOP accounting.

Run the local structural and exact-count audit without creating external work:

```bash
PYTHONPATH=src uv run python \
  src/scripts/train/jacobm_olmoe_ladder/v2/models/geometry_matched_scale.py
```
