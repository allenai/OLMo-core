# Qwen3-Like MoE Geometry

Stub plan for Qwen3-30B-A3B-inspired MoE geometry experiments on the current
active-parameter ladder.

## Goal

Compare the current OLMoE3 ladder MoE shape against a Qwen3-like sparse FFN
shape while keeping the non-MoE backbone fixed per size.

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

| Size | `d_model` | `d_attn` | Layers |
| --- | ---: | ---: | ---: |
| 275m | 768 | 1024 | 12 |
| 480m | 1024 | 1024 | 16 |
| 810m | 1280 | 1536 | 20 |
| 1p2b | 1536 | 2048 | 22 |

## First Variants

### `qwen3_like_literal`

Literal Qwen3 MoE geometry ratios.

| Size | Experts | `top_k` | `moe_hidden_size` | Shared experts | Dense prefix | Active routed width | Total routed width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275m | 128 | 8 | 288 | 0 | 0 | 2304 = 3.0d | 36864 = 48d |
| 480m | 128 | 8 | 384 | 0 | 0 | 3072 = 3.0d | 49152 = 48d |
| 810m | 128 | 8 | 480 | 0 | 0 | 3840 = 3.0d | 61440 = 48d |
| 1p2b | 128 | 8 | 576 | 0 | 0 | 4608 = 3.0d | 73728 = 48d |

This matches Qwen3-30B-A3B's routed capacity ratio:
`expert_hidden = 0.375 * d_model`, `top_k / num_experts = 8 / 128`, and
`num_experts * expert_hidden = 48 * d_model`.

### `qwen3_like_active_matched`

Qwen3-style routing and no-shared semantics, but active FFN width matched to the
current OLMoE3 ladder baseline.

| Size | Experts | `top_k` | `moe_hidden_size` | Shared experts | Dense prefix | Active routed width | Total routed width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275m | 128 | 8 | 432 | 0 | 0 | 3456 = 4.5d | 55296 = 72d |
| 480m | 128 | 8 | 576 | 0 | 0 | 4608 = 4.5d | 73728 = 72d |
| 810m | 128 | 8 | 720 | 0 | 0 | 5760 = 4.5d | 92160 = 72d |
| 1p2b | 128 | 8 | 864 | 0 | 0 | 6912 = 4.5d | 110592 = 72d |

This is the cleaner active-budget comparison, but it is not literal Qwen3
capacity because total routed expert width increases from 48d to 72d.

## Implementation Sketch

Add a new independent geometry axis, tentatively:

```text
--qwen3-like {none,literal,active_matched}
```

When enabled, override:

- `NUM_EXPERTS = 128`
- `TOP_K = 8`
- `NUM_SHARED_EXPERTS = 0`
- `SHARED_MLP_HIDDEN_SIZE = 0`
- dense block overrides: none
- `MOE_HIDDEN_SIZE = round_to_valid_multiple(d_model * hidden_mult)`

Use `hidden_mult = 0.375` for `literal` and `0.5625` for `active_matched`.

Keep attention, layer count, normalization, RoPE, sliding-window schedule, data,
optimizer, and evaluation settings unchanged for the first pass.

## First Wave

Start with 275m Cx1 and Cx4:

| Variant | Cx | Purpose |
| --- | ---: | --- |
| `qwen3_like_literal` | 1, 4 | Tests the actual Qwen3 sparse geometry ratio. |
| `qwen3_like_active_matched` | 1, 4 | Tests Qwen3-style routing at matched active FFN width. |

Before launching, dry-run and record:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- active/total percentage;
- routed active hidden units;
- routed total hidden units;
- dense/MoE layer counts.

## Open TODOs

- Decide exact CLI/API shape after inspecting the current ladder code.
- Decide whether hidden sizes need divisibility rounding for kernels.
- Replace this stub with exact dry-run parameter counts.
- Add launch scripts only after the 275m smoke passes.
