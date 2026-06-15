# Qwen3-Like MoE Geometry

Plan for Qwen3-30B-A3B-inspired MoE geometry experiments on the current
active-parameter ladder.

## Goal

Compare the current OLMoE3 ladder MoE shape against Qwen3-like sparse FFN
shapes while keeping the non-MoE backbone fixed per size for the first pass.
The first wave is 275M-only: it is an LR-transfer/geometry test, not yet a full
multi-size promotion.

Qwen3-like means:

- all FFN layers are sparse MoE layers;
- no shared expert;
- no dense prefix layer;
- many smaller routed experts;
- top-8 routing over 128 experts.

Do not include Qwen3.6 in this first pass. Its hybrid token mixer and gated
shared expert make it a different ablation.

## Baseline Ladder Rows

Use these labels in plans and run names:

| Size | `d_model` | `d_attn` | Baseline layers |
| --- | ---: | ---: | ---: |
| 275m | 768 | 1024 | 12 |
| 480m | 1024 | 1024 | 16 |
| 810m | 1280 | 1536 | 20 |
| 1p2b | 1536 | 2048 | 22 |

## Implemented Variants

Implemented in `../experiments/qwen3_like/qwen3_like_ladder.py` as a dedicated
script rather than adding another combined axis to `../moe_a0_ladder.py`.

### `active_matched` / `q3am128e8k`

Qwen3-style routing and no-shared semantics, but active FFN width matched to the
current OLMoE3 ladder baseline.

| Size | Layers | Experts | `top_k` | `moe_hidden_size` | Shared experts | Dense prefix | Active routed width | Total routed width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275m | 12 | 128 | 8 | 432 | 0 | 0 | 3456 = 4.5d | 55296 = 72d |

This is the cleaner active-budget comparison, but it is not literal Qwen3 total
capacity because total routed expert width increases from 48d to 72d.

### `true_3d_depth_matched` / `q3td128e8k`

Literal Qwen3 routed width ratio with extra depth at 275M to keep active
parameter count closer to the baseline.

| Size | Layers | Experts | `top_k` | `moe_hidden_size` | Shared experts | Dense prefix | Active routed width | Total routed width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275m | 16 | 128 | 8 | 288 | 0 | 0 | 2304 = 3.0d | 36864 = 48d |

This matches Qwen3-30B-A3B's routed capacity ratio:
`expert_hidden = 0.375 * d_model`, `top_k / num_experts = 8 / 128`, and
`num_experts * expert_hidden = 48 * d_model`. The 275M version uses 16 layers
instead of 12 to make the active-parameter comparison less underpowered.

## 275M LR Search

Start from the current 275M baseline-centered LR triples and test transfer.
Use only observed results for later summary plots.

| Cx | LR grid | Global batch seq | Tokens/step | GPUs | Microbatch | Grad accum |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `1e-3`, `2e-3`, `4e-3` | 32 | 262,144 | 2 | 8 | 2 |
| 2 | `9e-4`, `1.8e-3`, `3.6e-3` | 48 | 393,216 | 2 | 8 | 3 |
| 4 | `8e-4`, `1.6e-3`, `3.2e-3` | 64 | 524,288 | 2 | 8 | 4 |
| 8 | `8e-4`, `1.6e-3`, `3.2e-3` | 96 | 786,432 | 4 | 8 | 3 |

If run concurrently, this is 30 GPUs per variant and 60 GPUs for both variants.
All jobs are intended for Holmes low-priority/preemptible with compilation on.

## Smoke Before Full Grid

Launch one short Cx0.02 center-LR smoke for each variant before queueing the
full grid:

| Variant | Run tag | Cx | LR | Global batch seq | GPUs | Microbatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `active_matched` | `q3am128e8k` | 0.02 | `2e-3` | 32 | 2 | 8 |
| `true_3d_depth_matched` | `q3td128e8k` | 0.02 | `2e-3` | 32 | 2 | 8 |

Proceed to `../experiments/qwen3_like/launch_275m_full_ladder.sh` only after the
smokes show that the configs compile, fit, and train without obvious runtime
failures.

## Dry-Run Checklist

Before launch, dry-run and record:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- active/total percentage;
- routed active hidden units;
- routed total hidden units;
- dense/MoE layer counts.

## Dry-Run Parameter Counts

Local config dry-run on 2026-06-15 with `--model-size=275m`, Cx0.02 smoke
batch settings, and LR `2e-3` produced:

| Variant | Tag | Layers | Total params | Active params | Active non-embedding params | Active/total | Active width |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `active_matched` | `q3am128e8k` | 12 | 1,712,497,152 | 279,224,832 | 202,154,496 | 16.305% | 3456 = 4.5d |
| `true_3d_depth_matched` | `q3td128e8k` | 16 | 1,552,471,552 | 278,451,712 | 201,381,376 | 17.936% | 2304 = 3.0d |

## Launchers

- `../experiments/qwen3_like/launch_smoke.sh`: two Cx0.02 center-LR smokes.
- `../experiments/qwen3_like/launch_275m_full_ladder.sh`: full 275M Cx1/2/4/8 LR grid.

Both launchers check that `global_batch_size_seq` divides cleanly by
`nodes * gpus * microbatch` before submitting Beaker jobs.
