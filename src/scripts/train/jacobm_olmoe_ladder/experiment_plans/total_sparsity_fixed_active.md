# Total Sparsity At Fixed Active Compute Experiment Plan

This runbook specifies the total-sparsity ablation: change the total routed
expert capacity while keeping active routed compute fixed. It complements
`expert_granularity.md`, which keeps total routed capacity fixed while changing
expert granularity.

## Status

Launch-ready after a small launcher refresh. The code supports
`--total-sparsity`; the stale part is mostly launch orchestration and old Cx2
batch assumptions in early notes/scripts.

Run after the current expert-granularity queue drains, or alongside it if new
compute appears. Treat this as an independent architecture ablation on top of
the baseline 48E/top4 geometry: keep `top_k=4` and `moe_hidden_size=d_model`,
change only total expert count, and do not stack it on the expert-granularity
winner until a later integration run.

The first wave should focus on *more* total expert capacity, not less. Our
baseline is already relatively high-active compared with recent sparse MoEs, so
the low-total / less-sparse variant is a future diagnostic rather than an
initial run target. Do not launch `low_total_24e_top4` / `sp24e4k` in the first
wave.

## Goal

Test whether adding inactive expert capacity improves pretraining loss at fixed
active compute. This directly asks how sparse the model should be: do more total
parameters help when each token still activates the same amount of expert MLP?

## Hypothesis

More total expert capacity should give the router more specialization options
and may improve loss, especially at higher data scales. Too much total capacity
may hurt through worse routing, thinner per-expert data, load-balancing stress,
memory pressure, or lower throughput.

## Matching Policy

Keep fixed:

- `top_k = 4`
- `moe_hidden_size = d_model`
- active routed hidden units: `top_k * moe_hidden_size = 4 * d_model`
- dense backbone
- shared expert
- dense prefix
- attention schedule

Vary:

- `num_experts`
- therefore total routed expert hidden units:
  `num_experts * moe_hidden_size`

This changes total parameters while keeping active parameters approximately
fixed. Router parameters also change with `num_experts`; record that as a real
part of the design.

## Variants

### Core Family

| Variant | Experts | `top_k` | `moe_hidden_size` | Total routed capacity | Active routed capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_48e_top4` | 48 | 4 | `d_model` | `48 * d_model` | `4 * d_model` |
| `high_total_96e_top4` | 96 | 4 | `d_model` | `96 * d_model` | `4 * d_model` |
| `huge_total_192e_top4` | 192 | 4 | `d_model` | `192 * d_model` | `4 * d_model` |

Future low-total diagnostic, not in the first wave:

| Variant | Experts | `top_k` | `moe_hidden_size` | Total routed capacity | Active routed capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low_total_24e_top4` | 24 | 4 | `d_model` | `24 * d_model` | `4 * d_model` |

The compact run-name tags are:

| Variant | Tag | First wave? |
| --- | --- | --- |
| `low_total_24e_top4` | `sp24e4k` | No; less sparse than baseline. |
| `baseline_48e_top4` | `sp48e4k` | Control only. |
| `high_total_96e_top4` | `sp96e4k` | Yes. |
| `huge_total_192e_top4` | `sp192e4k` | Yes. |

### Approximate Active Fraction

These estimates use the measured baseline active/total counts and assume the
non-expert static parameter block stays fixed while routed expert total
capacity scales linearly. Exact dry-run counts must replace these before final
launch decisions. Always report active params, total params, and active/total
percentage next to loss and throughput for this experiment.

| Size | Variant | Approx active params | Approx total params | Active / total |
| --- | --- | ---: | ---: | ---: |
| 275M | `low_total_24e_top4` | 0.28B | 0.67B | 42% |
| 275M | `baseline_48e_top4` | 0.28B | 1.13B | 25% |
| 275M | `high_total_96e_top4` | 0.28B | 2.06B | 14% |
| 275M | `huge_total_192e_top4` | 0.28B | 3.92B | 7% |
| mid_480m | `low_total_24e_top4` | 0.48B | 1.43B | 34% |
| mid_480m | `baseline_48e_top4` | 0.48B | 2.56B | 19% |
| mid_480m | `high_total_96e_top4` | 0.48B | 4.83B | 10% |
| mid_480m | `huge_total_192e_top4` | 0.48B | 9.37B | 5% |
| 810M | `low_total_24e_top4` | 0.82B | 2.69B | 30% |
| 810M | `baseline_48e_top4` | 0.82B | 4.93B | 17% |
| 810M | `high_total_96e_top4` | 0.82B | 9.42B | 9% |
| 810M | `huge_total_192e_top4` | 0.82B | 18.39B | 4% |
| 1.2B | `low_total_24e_top4` | 1.22B | 4.19B | 29% |
| 1.2B | `baseline_48e_top4` | 1.22B | 7.76B | 16% |
| 1.2B | `high_total_96e_top4` | 1.22B | 14.90B | 8% |
| 1.2B | `huge_total_192e_top4` | 1.22B | 29.17B | 4% |

### Exact 275M Dry-Run Counts

Verified locally on 2026-06-11 by building the 275M config with
`--total-sparsity`.

| Variant | Active params incl emb/head | Total params | Active / total |
| --- | ---: | ---: | ---: |
| `high_total_96e_top4` / `sp96e4k` | 278,856,192 | 2,069,561,856 | 13.47% |
| `huge_total_192e_top4` / `sp192e4k` | 279,667,200 | 3,938,935,296 | 7.10% |

## Exact Configs

For each model size, keep all baseline fields fixed except `num_experts`.

Baseline size fields:

| Size | `d_model` | `d_attn` | Layers | Heads | KV heads | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 768 | 1024 | 12 | 8 | 4 | 4 | 768 | 384 | 3456 |
| 480M (`mid_480m` code alias) | 1024 | 1024 | 16 | 8 | 4 | 4 | 1024 | 512 | 4608 |
| 810M | 1280 | 1536 | 20 | 12 | 6 | 4 | 1280 | 640 | 5760 |
| 1.2B | 1536 | 2048 | 22 | 16 | 8 | 4 | 1536 | 768 | 6912 |

The 480M row uses the implemented `mid_480m` code alias in
`tiny_275m.py` / `moe_a0_ladder.py`.

## Implemented Code Support

The training script already supports this axis:

```text
--total-sparsity {baseline_48e_top4,high_total_96e_top4,huge_total_192e_top4,low_total_24e_top4}
```

Implementation notes:

1. `MODEL_SIZE_SPECS` remains the size/backbone table.
2. `TOTAL_SPARSITY_SPECS` overrides `num_experts` only.
3. `top_k`, `moe_hidden_size`, shared expert, dense prefix, and attention stay
   unchanged.
4. Compact tags are `sp24e4k`, `sp48e4k`, `sp96e4k`, and `sp192e4k`.
5. `--expert-geometry` and `--total-sparsity` are independent ablation axes;
   do not combine them in a first-wave run.

Do not change existing baseline run names retroactively.

## Parameter Check

Before launch, instantiate each variant locally and record:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- active/total percentage;
- router parameter count if easy to extract;
- routed active hidden units;
- routed total hidden units.

Keep the approximate table above as planning context, but replace it with exact
dry-run counts for 480M, 810M, and 1.2B before launching those sizes.

## Launch-Ready Run Plan, 2026-06-13

This is the current source of truth for what to queue when sparsity becomes the
next active experiment. It supersedes older partial-wave notes and old Cx2
`b512k` references.

### Scientific Conditions

Run the two serious higher-total variants:

| Variant | Tag | Experts | `top_k` | `moe_hidden_size` | First-wave role |
| --- | --- | ---: | ---: | ---: | --- |
| `high_total_96e_top4` | `sp96e4k` | 96 | 4 | `d_model` | Main 2x-total-capacity point. |
| `huge_total_192e_top4` | `sp192e4k` | 192 | 4 | `d_model` | Main 4x-total-capacity point. |

Do not launch `low_total_24e_top4` / `sp24e4k` in the first real wave. It is a
less-sparse diagnostic, not part of the initial decision surface.

The baseline comparison is the existing `baseline_48e_top4` ladder at the same
model size and Cx. Do not create new baseline runs under `sp48e4k` names unless
we explicitly need a resume-stable control rerun.

### Current Historical Sparsity Runs

There was a partial early total-sparsity wave. Treat it as noncanonical except
where explicitly useful for sanity checking:

- `sp96e4k` 275M Cx1 `1e-3` completed and now counts as the low-LR
  high-total Cx1 point.
- Most other tracked total-sparsity Cx1/Cx4 r1 jobs were manually canceled;
  stopped Cx1 points were relaunched as r2 on 2026-06-13.
- Cx1 and repaired `b384k` Cx2 LR-transfer checks were first queued on 2026-06-13, then stopped and requeued on 2026-06-14 as `gpu2-ep1mb8` after memory plots suggested enough headroom. The stable run names and checkpoint folders were retained; see `RUNS.md` for replacement Beaker IDs and old stopped IDs.
- No complete tracked Cx8 sparsity result should be used as a final point.

For the real experiment, continue the full matrix below with repaired Cx2 and
stable semantic names.

### LR Policy

At 275M, run full three-point U-curves for both variants and every real Cx:
Cx1/Cx2/Cx4/Cx8. Center on the current 275M baseline best/fitted LR and use
factor-of-two spacing. Use repaired `b384k` for Cx2.

After 275M completes, fit each variant's local LR multiplier per Cx if the
curve brackets cleanly:

```text
m_variant(Cx) = lr*_variant_275m(Cx) / lr*_baseline_275m(Cx)
lr_variant(size, Cx) = m_variant(Cx) * lr_baseline(size, Cx)
```

If the multipliers are effectively 1, use the baseline best observed LR for
larger-size single-point promotions. That keeps the comparison directly
interpretable: same size, same Cx, same LR, only total expert capacity changes.
If a 275M curve lands on an edge, launch one bounded extension before promoting
that Cx to larger sizes.

### 275M Full LR-Tuning Matrix

Queue these when compute is available. This is 24 jobs total: two variants,
four Cx values, three LRs each.

| Model | Variant | Cx | LRs | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 275M | `sp96e4k` | 1 | `1e-3`, `2e-3`, `4e-3` | 262,144 | 32 | 4 | 1 | 4 | Queued/covered; `1e-3-r1` finished, `2e-3/4e-3` relaunched as r2. |
| 275M | `sp192e4k` | 1 | `1e-3`, `2e-3`, `4e-3` | 262,144 | 32 | 4 | 1 | 4 | Queued as r2 on 2026-06-13 after old r1 jobs were stopped. |
| 275M | `sp96e4k` | 2 | `9e-4`, `1.8e-3`, `3.6e-3` | 393,216 | 48 | 4 | 1 | 4 | Queued on 2026-06-13 with repaired canonical `b384k`. |
| 275M | `sp192e4k` | 2 | `9e-4`, `1.8e-3`, `3.6e-3` | 393,216 | 48 | 4 | 1 | 4 | Queued on 2026-06-13 with repaired canonical `b384k`. |
| 275M | `sp96e4k` | 4 | `8e-4`, `1.6e-3`, `3.2e-3` | 524,288 | 64 | 4 | 1 | 4 | Matches existing launcher shape. |
| 275M | `sp192e4k` | 4 | `8e-4`, `1.6e-3`, `3.2e-3` | 524,288 | 64 | 4 | 1 | 4 | Matches existing launcher shape. |
| 275M | `sp96e4k` | 8 | `8e-4`, `1.6e-3`, `3.2e-3` | 786,432 | 96 | 8 | 1 | 4 | Existing Cx8 launcher shape. |
| 275M | `sp192e4k` | 8 | `8e-4`, `1.6e-3`, `3.2e-3` | 786,432 | 96 | 8 | 1 | 4 | Existing Cx8 launcher shape. |

Cx16 is intentionally excluded. Reserve it for important or unclear cases after
Cx1/Cx2/Cx4/Cx8 are interpreted.

### Larger-Size Promotion Matrix

Once 275M multipliers are known, run single-point promotions for both variants
at all four model sizes and Cx1/Cx2/Cx4/Cx8. For 480M/810M/1.2B, do not run
full LR sweeps unless the 275M transfer clearly fails or a promoted point looks
surprisingly bad.

If the 275M multipliers are effectively 1, the default LR table is:

| Model | Cx1 LR | Cx2 LR | Cx4 LR | Cx8 LR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 480M | `1.2e-3` | `9e-4` | `8e-4` | `8e-4` | Use repaired `b384k` Cx2 and completed 480M baseline best observed LRs. |
| 810M | `6e-4` | `5.6e-4` | `4e-4` | `4e-4` | Use repaired `b384k` Cx2; Cx8 baseline completed on 8 GPUs. |
| 1.2B | `4e-4` | TBD | `3e-4` | TBD | Cx2 and one-node Cx8 baselines are still settling; fill in from best observed before launch. |

Provisional 1.2B notes:

- Cx2 baseline candidates are currently `1.5e-4`, `3e-4`, `6e-4`; choose the
  best observed after the baseline jobs finish.
- Cx8 should use the one-node `gpu8-ep1mb4` baseline family once the `2e-4` and
  `8e-4` replacements finish. The old 4-node `4e-4` result is useful but should
  not silently define the canonical systems setting if the one-node sweep lands
  differently.

### Promotion Queue Order

If compute arrives in a burst, use this order:

1. Smoke `sp96e4k` and `sp192e4k` at the largest memory-pressure cell we plan to
   launch immediately. For 275M-only launch, smoke 275M Cx1/Cx2 is enough; for
   broad launch, smoke 480M or 810M too.
2. Queue the full 275M LR-tuning matrix above. Cx1/Cx2 were requeued on
   2026-06-14 as `gpu2-ep1mb8`; Cx4/Cx8 remain next when compute allows. This is the most
   important sparsity work because it calibrates LR transfer.
3. If 275M Cx1/Cx4 complete first and look normal, queue 480M Cx1/Cx4 single
   points for both variants while waiting for 275M Cx2/Cx8.
4. After all 275M real Cx values finish, queue the full 480M and 810M
   Cx1/Cx2/Cx4/Cx8 single-point matrices for both variants.
5. Queue 1.2B Cx1 as soon as 275M transfer looks sane; queue 1.2B Cx2/Cx4/Cx8
   after the baseline anchors are settled.

### Launch Names And Tags

Use semantic, resume-stable names. Do not encode GPU count, node count, EP, or
microbatch in the run name; put systems details in W&B tags and in `RUNS.md`.

Name pattern:

```text
sp-{model}-cx{cx}-{sp_tag}-{lr_tag}-r{attempt}
```

Examples:

```text
sp-275m-cx2-sp96e4k-lr1.8e-3-r1
sp-480m-cx8-sp192e4k-lr8e-4-r1
sp-810m-cx2-sp96e4k-lr5.6e-4-r1
sp-1p2b-cx1-sp192e4k-lr4e-4-r1
```

Required W&B tags:

```text
exp_total_sparsity
sp96e4k or sp192e4k
{model}
cx{cx}
batch tag: b256k, b384k, b512k, or b768k
lr tag
baseline-transferred for 275M tuning points
baseline-best-observed or predicted-lr for larger promotions
nodes{n}, gpu{g}, ep1, mb{microbatch}
```

### Systems Settings

Use EP=1 throughout unless a smoke test proves we need a fallback. Keep the
optimizer-relevant batch fixed exactly as below.

| Model | Cx | Batch tokens | Batch seqs | Default GPUs | EP | Default microbatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 1 | 262,144 | 32 | 4 | 1 | 4 |
| 275M | 2 | 393,216 | 48 | 4 | 1 | 4 |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 4 |
| 275M | 8 | 786,432 | 96 | 8 | 1 | 4 |
| 480M | 1 | 262,144 | 32 | 4 | 1 | 4 |
| 480M | 2 | 393,216 | 48 | 4 | 1 | 4 |
| 480M | 4 | 524,288 | 64 | 4 | 1 | 4 |
| 480M | 8 | 786,432 | 96 | 8 | 1 | 4 |
| 810M | 1 | 262,144 | 32 | 8 | 1 | 4 |
| 810M | 2 | 393,216 | 48 | 8 | 1 | 2 |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 |
| 810M | 8 | 786,432 | 96 | 8 | 1 | 4 |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 |
| 1.2B | 2 | 393,216 | 48 | 8 | 1 | 2 |
| 1.2B | 4 | 524,288 | 64 | 8-16 | 1 | 2 |
| 1.2B | 8 | 786,432 | 96 | 8 | 1 | 4 |

These settings are intentionally compute-efficient and optimizer-batch-matched.
If a run OOMs, first lower microbatch if divisibility allows; if that is not
enough, increase GPU count while preserving the same global batch and run name.
Record the systems change in tags and `RUNS.md`, not by renaming the run.

### Launcher Work Needed Before Big Queue

The current scripts are useful but not sufficient for the new full plan:

- `experiments/total_sparsity/launch_275m_cx1_cx4.sh` now covers Cx1/Cx2/Cx4
  and uses the repaired `b384k` Cx2 LR grid.
- `experiments/total_sparsity/launch_275m_cx8.sh` covers Cx8 only.
- Add 480M, 810M, and 1.2B promotion launchers that use the semantic names and
  systems tags above.

Checkpoint/eval settings for all runs:

```text
--ladder-evals
--eval-task-set=fast
--eval-interval=2000
--save-interval=999999999
--ephemeral-save-interval=500
--no-pre-train-checkpoint
```

Validation/eval metrics remain observational. LR choice and promotion decisions
use final-window training CE until we explicitly change that policy.

## Monitoring

Check:

- W&B run appears.
- loss finite and decreasing;
- skipped steps remain zero;
- throughput and memory relative to baseline;
- router load balance and dead/overused experts;
- checkpoint path matches run name.

For high-total variants, explicitly record wall-clock and memory pressure: the
scientific win is less useful if total capacity slows training dramatically.

## Analysis

Primary metric:

- final-window training CE `avg250M`.

Also report:

- `avg100M`;
- `avg500M`;
- final step loss;
- throughput;
- active params;
- total params;
- active/total percentage;
- eval metrics as observational context only.

Recommended plots:

- per-Cx U-plots by sparsity variant;
- selected-LR baseline-vs-variant comparison across Cx;
- loss vs active/total percentage;
- loss improvement per added total parameter.

Plot outputs should live under:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/total_sparsity_fixed_active/
```

## Promotion Criteria

Promote if:

- higher total capacity beats baseline after LR adjustment;
- the advantage grows with Cx;
- throughput/memory cost is acceptable;
- routing diagnostics do not show severe collapse.

Do not promote if:

- it loses clearly at Cx1 and Cx4;
- it is too LR-sensitive;
- it has severe load imbalance or dead experts;
- the wall-clock cost dominates the loss improvement.

## Documentation Requirements

After launch:

- append runs to `RUNS.md`;
- add rationale and LR grids to `ANALYSIS.md`;
- commit code/script changes before relying on launchers.

After completion:

- refresh W&B cache for this run family;
- update final-window summaries;
- regenerate experiment-specific plots;
- push docs and plots.

## Open TODOs

- Implement `--total-sparsity` or decide to fold this into the existing
  `--expert-geometry` table.
- Replace approximate active/total percentages with exact dry-run counts.
- Smoke high-total variants for memory and throughput.
- Decide whether to run total-sparsity variants on top of baseline 48E/top4 or
  the winning expert-granularity geometry.
