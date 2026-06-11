# Remaining V0 Ablation Plan

This is the compact plan for the three independent V0 ablations we want to run
on top of the MoE A0 baseline before doing an integrated "best of each axis"
run.

The core rule is: run each ablation independently against the baseline shape
first. Do not combine wins across axes until each axis has enough evidence to
name a candidate winner. LR selection uses final-window training CE only;
validation/eval metrics remain observational.

## Axis 1: Expert Granularity

Status: active.

Runbook: `expert_granularity.md`

Question: at fixed routed active capacity and fixed routed total capacity, do we
prefer fewer larger experts or more smaller experts?

Serious variants:

| Variant | Experts | `top_k` | `moe_hidden_size` |
| --- | ---: | ---: | ---: |
| `coarse_24e_top2` | 24 | 2 | `2 * d_model` |
| baseline `48e_top4` | 48 | 4 | `d_model` |
| `fine_96e_top8` | 96 | 8 | `d_model / 2` |

Diagnostic extremes:

| Variant | Experts | `top_k` | `moe_hidden_size` |
| --- | ---: | ---: | ---: |
| `extreme_192e_top16` | 192 | 16 | `d_model / 4` |
| `ultra_384e_top32` | 384 | 32 | `d_model / 8` |

Current plan:

- Complete the serious 275M ladder through Cx8, and Cx16 if capacity allows.
- Treat extreme variants as Cx1 diagnostics until a result justifies promotion.
- Promote the strongest serious variant to 810M Cx1/Cx4 once the 275M evidence
  is clear.

## Axis 2: Total Sparsity At Fixed Active Compute

Status: planned next.

Runbook: `total_sparsity_fixed_active.md`

Question: if each token activates the same routed compute, does adding inactive
expert capacity help?

Keep fixed:

- `top_k = 4`
- `moe_hidden_size = d_model`
- shared expert and dense prefix
- attention schedule

Vary only total expert count.

First-wave variants:

| Variant | Experts | `top_k` | Approx active / total | Role |
| --- | ---: | ---: | ---: | --- |
| baseline `sp48e4k` | 48 | 4 | 25% at 275M, 17% at 810M, 16% at 1.2B | Current control. |
| `sp96e4k` | 96 | 4 | 14% at 275M, 9% at 810M, 8% at 1.2B | More total capacity. |
| `sp192e4k` | 192 | 4 | 7% at 275M, 4% at 810M, 4% at 1.2B | Aggressive sparse point. |

Do not run `sp24e4k` in the first wave. It is less sparse than the baseline and
is reserved for a later diagnostic only if we need to map the full curve.

Initial 275M scope:

- Cx1/Cx4 for `sp96e4k` and `sp192e4k`, three LRs per rung.
- If either variant is promising, extend to 275M Cx8.
- Promote the best sparsity variant to 810M Cx1/Cx4.

Before launch:

- Implement `--total-sparsity`.
- Dry-run exact active params, total params, active/total percentage, and router
  params.
- Smoke `sp96e4k` and `sp192e4k`; higher total capacity may increase optimizer
  memory even though active compute stays fixed.

## Axis 3: Dense Layers And Shared Expert

Status: planned after splitting the original mixed-axis runbook.

Runbook: `shared_expert_dense_schedule.md`

Question: how much dense computation do we need, and is the shared expert useful
independently of dense prefix depth?

Split the old mixed design into two cleaner sub-experiments.

### 3A. Dense Count With Shared Expert Fixed

Hold fixed:

- `num_shared_experts = 1`
- `shared_mlp_hidden_size = d_model / 2`
- `moe_hidden_size = d_model`
- 48 experts, top-4 routing

Vary dense prefix depth:

| Variant | Dense layers | Shared expert | Role |
| --- | --- | --- | --- |
| `dense0_shared` | none | yes | Is the dense prefix needed? |
| baseline `dense1_shared` | `[0]` | yes | Current control. |
| `dense2_shared` | `[0, 1]` | yes | Does more dense prefix help? |
| `dense4_shared` | `[0, 1, 2, 3]` | yes | Larger dense-prefix sentinel. |

### 3B. Shared Expert Ablation With Dense Count Fixed

Hold dense prefix fixed at baseline `[0]`.

Compare:

| Variant | Dense layers | Shared expert | `moe_hidden_size` | Role |
| --- | --- | --- | ---: | --- |
| baseline `dense1_shared` | `[0]` | yes | `d_model` | Current control. |
| `dense1_no_shared_am` | `[0]` | none | `9/8 * d_model` | Active-matched quality comparison. |
| `dense1_no_shared_unmatched` | `[0]` | none | `d_model` | Optional cheaper/simpler model comparison. |

First-wave recommendation:

- Run 275M Cx1/Cx4 for `dense0_shared`, `dense2_shared`, `dense4_shared`, and
  `dense1_no_shared_am`.
- Add `dense1_no_shared_unmatched` if compute is abundant or if the practical
  "remove shared expert and save params/compute" question becomes important.
- Defer alternating dense/MoE schedules until dense count and shared expert are
  understood separately.

Before launch:

- Implement `--dense-schedule`.
- Generalize layer construction beyond a single dense prefix layer.
- Dry-run active/total params and dense/MoE layer counts.
- Smoke all first-wave variants.

## Integration Run

After the three axes each have a candidate winner:

1. Define an integrated candidate that combines the best setting from expert
   granularity, total sparsity, and dense/shared.
2. Run a small 275M confirmation ladder, starting with Cx1/Cx4 and three LRs
   centered by transferred baseline/variant multipliers.
3. Compare the integrated candidate against the independent-axis winners to
   detect non-additive interactions.
4. Promote to 810M only if the integrated candidate is cleanly better or
   strategically important.
