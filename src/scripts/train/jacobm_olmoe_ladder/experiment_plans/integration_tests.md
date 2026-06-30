# MoE Integration Test Plan

Last refreshed: 2026-06-30.

This doc stages the next integration tests after the independent baseline,
expert-granularity, total-sparsity, shared-expert, dense-schedule, and
Qwen-like ladders. The goal is to combine the strongest non-Qwen-like signals
without losing attribution completely.

Current evidence:

- Expert granularity: `96E/top8` beats baseline `48E/top4` across the full
  275M/480M/810M/1.2B Cx1/2/4/8 grid.
- Total sparsity: increasing total experts at fixed active routed compute wins
  strongly. `192E/top4` beats `96E/top4`, which beats baseline, especially by
  810M.
- Shared expert: removing the shared expert is not a larger-scale win. Keep the
  shared expert for the first integration.
- Dense schedule: `dense0` is competitive, but `dense2`/`dense4` do not look
  like wins. Keep the baseline one dense prefix layer for the first integration.
- Qwen-like: depth-expanded `3.0d` and active-matched `4.5d` both beat baseline,
  but they combine geometry/depth/no-shared/no-dense. Treat depth as a companion
  integration axis, not as part of the clean first integration.

## Candidate Configs

### `int-wide-256e8k`

Primary integration candidate. This keeps the baseline depth and active FFN
budget while combining fine granularity with higher total routed capacity.

| Setting | Value |
| --- | --- |
| Routed experts | 256 |
| Top-k | 8 |
| Routed expert hidden | `0.5 * d_model` |
| Shared experts | 1 |
| Shared expert hidden | `0.5 * d_model` |
| Dense prefix | layer 0 only |
| Active FFN width | `8 * 0.5d + 0.5d = 4.5d` |
| Total routed width | `256 * 0.5d = 128d` |
| Depth | baseline depth per size |

This sits between the completed total-sparsity settings in total routed width:

| Variant | Total routed width | Active FFN width |
| --- | ---: | ---: |
| baseline `48E/top4` | `48d` | `4.5d` including shared |
| fine `96E/top8` | `48d` | `4.5d` including shared |
| huge sparsity `192E/top4` | `192d` | `4.5d` including shared |
| `int-wide-256e8k` | `128d` | `4.5d` including shared |

Dry-run parameter counts from local config construction:

| Size | Layers | Total params | Active params | Active non-emb params | Active/total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 275M | 12 | 2,693,767,680 | 280,207,872 | 203,137,536 | 10.40% |
| 480M | 16 | 6,337,402,880 | 486,348,800 | 383,588,352 | 7.67% |
| 810M | 20 | 12,403,781,120 | 823,569,920 | 695,119,360 | 6.64% |
| 1.2B | 22 | 19,655,832,064 | 1,225,011,712 | 1,070,871,040 | 6.23% |

### `int-deep-256e8k`

Companion depth test. This keeps the same total expert count and top-k as the
wide integration, narrows routed and shared experts to Qwen-like `0.375d`, keeps one
dense prefix layer, and adds depth only enough to active-match the baseline
ladder.

| Setting | Value |
| --- | --- |
| Routed experts | 256 |
| Top-k | 8 |
| Routed expert hidden | `0.375 * d_model` |
| Shared experts | 1 |
| Shared expert hidden | `0.375 * d_model` |
| Dense prefix | layer 0 only |
| Active FFN width | `8 * 0.375d + 0.375d = 3.375d` |
| Total routed width | `256 * 0.375d = 96d` |
| Depth | active-matched schedule below |

Do not use the full Qwen-like true-3D depth schedule (`16/22/27/29`) after
adding same-width shared+dense back in; it overshoots active params. The proposed
active-matched depth schedule is:

| Size | Baseline layers | Proposed deep layers |
| --- | ---: | ---: |
| 275M | 12 | 14 |
| 480M | 16 | 20 |
| 810M | 20 | 24 |
| 1.2B | 22 | 27 |

Dry-run parameter counts:

| Size | Layers | Total params | Active params | Active non-emb params | Active/total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 275M | 14 | 2,414,664,704 | 275,373,056 | 198,302,720 | 11.40% |
| 480M | 20 | 6,047,882,240 | 489,380,864 | 386,620,416 | 8.09% |
| 810M | 24 | 11,323,400,704 | 809,787,904 | 681,337,344 | 7.15% |
| 1.2B | 27 | 18,340,753,152 | 1,226,419,968 | 1,072,279,296 | 6.69% |

## 275M LR Search

Qwen-like LR transfer was close to baseline, but total sparsity shifted optimal
LRs lower, especially for `192E/top4` at Cx2/Cx4/Cx8. Because the integration
uses more total routed capacity than `96E/top8` and less than `192E/top4`, center
slightly below the baseline/Qwen-like centers and bracket by a factor of two.

Use the same LR grid for both `int-wide-256e8k` and `int-deep-256e8k` unless the
first completed Cx1/Cx2 curves show a clear variant-specific shift.

Observed 275M baseline reference:

| Cx | Observed baseline best LR | Baseline fit LR | Baseline best avg250M |
| ---: | ---: | ---: | ---: |
| 1 | `2e-3` | `2.13e-3` | 2.7767 |
| 2 | `1.8e-3` | `1.78e-3` | 2.6541 |
| 4 | `1.5e-3` | `1.46e-3` | 2.5611 |
| 8 | `1.6e-3` | `1.35e-3` | 2.4864 |

| Cx | Baseline best observed | 192E/top4 fitted/best signal | Proposed integration grid | Rationale |
| ---: | ---: | ---: | --- | --- |
| 1 | `2e-3` | fit `1.57e-3`, best `2e-3` | `8e-4`, `1.6e-3`, `3.2e-3` | Center near sparsity-shifted fit while retaining hot-side coverage. |
| 2 | `1.8e-3` | fit `7.67e-4`, best `9e-4` | `8e-4`, `1.6e-3`, `3.2e-3` | Keeps 192E/top4 best observed on the cold side while preserving baseline/96E hot-side coverage. |
| 4 | `1.5e-3` | fit `1.05e-3`, best `8e-4` | `8e-4`, `1.6e-3`, `3.2e-3` | Brackets 192E best observed and baseline/96E observed bests. |
| 8 | `1.6e-3` | fit `9.99e-4`, best `8e-4` | `8e-4`, `1.6e-3`, `3.2e-3` | Brackets 192E best observed and baseline/96E observed bests. |

If a completed curve has the best point on an edge, extend once in the improving
direction before promoting larger sizes. For first-night queueing, launch all
three LRs for Cx1/2/4/8 for one or both variants depending on available GPUs.

## 275M Training Settings

Use optimizer-batch-compatible settings. Keep EP=1 for comparability and because
the completed 275M sparsity runs fit without expert parallelism.

| Cx | Batch tokens | `global_batch_size_seq` | GPUs/job | EP | Microbatch | Grad accum | Jobs/variant | Concurrent GPUs/variant |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 262,144 | 32 | 2 | 1 | 8 | 2 | 3 | 6 |
| 2 | 393,216 | 48 | 2 | 1 | 8 | 3 | 3 | 6 |
| 4 | 524,288 | 64 | 4 | 1 | 8 | 2 | 3 | 12 |
| 8 | 786,432 | 96 | 8 | 1 | 4 | 3 | 3 | 24 |

Total concurrent demand:

| Scope | Jobs | GPUs |
| --- | ---: | ---: |
| One variant, full 275M LR grid | 12 | 48 |
| Both variants, full 275M LR grid | 24 | 96 |

Cluster note: these configs include a shared expert, so prefer the known-good
Titan/old-workspace compile-on path unless Holmes shared-expert compile behavior
has been revalidated. If using Holmes/B300 low-priority, run one smoke first for
each variant with the exact compile-on path before flooding the queue.

## Promotion Policy

After 275M Cx1/2/4/8 curves are bracketed:

1. Pick one LR per size/Cx from the best observed 275M multiplier relative to
   baseline, unless the fitted optimum is clearly more stable.
2. Promote `int-wide-256e8k` to 480M and 810M first.
3. Promote `int-deep-256e8k` in parallel only if the 275M result beats or matches
   wide on at least Cx2/Cx4/Cx8, or if compute is abundant enough that the depth
   question is worth paying for immediately.
4. Hold 1.2B until 480M/810M confirm that the combined gains transfer.

## Open Decisions Before Launch

- Confirm whether to run both variants tonight or start with `int-wide-256e8k`.
- Confirm cluster/workspace choice for shared-expert compile-on jobs.
- Decide whether to add a fourth hot/cold LR for Cx2 if compute is abundant; the
  current three-point grid is intentionally cold-shifted and should bracket if
  the integration behaves between `96E/top8` and `192E/top4`.
- Implement the dedicated integration ladder script or extend the existing
  combined ladder carefully enough that the semantic run names remain stable.
