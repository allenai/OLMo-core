# Settings Audit

Working audit for the MoE A0 baseline ladder and the first ablation runs.
Created 2026-06-12 after the expert-granularity Cx2 curves exposed a batch
setting mismatch.

This file records the settings we actually launched or used for comparison.
It is intentionally separate from the forward-looking resource policy in
`CURRENT_PLAN.md`, because some historical baseline runs predate that policy.

Unless noted otherwise:

- sequence length is 8192 tokens;
- total batch tokens = `global_batch_size_seq * 8192`;
- expert parallelism is EP=1;
- the data root is `s3://ai2-llm`;
- LR selection uses training loss only.

## Canonical Settings Going Forward

These are the settings to use for new comparable baseline and ablation runs
unless explicitly overridden. The 275M Cx2 row was updated after comparing our
old settings against the external `olmo-hybrid-pe` workspace: old baseline Cx2
used 262,144 tokens, while expert-granularity Cx2 used 524,288 tokens. The new
canonical repair is the smoother midpoint 393,216 tokens / 48 sequences.

| Model | Cx | Total batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 275M | 1 | 262,144 | 32 | 1 | 1 | 16 | Fine-grained EG variants may need lower mbz for memory. |
| 275M | 2 | 393,216 | 48 | 2 | 1 | 8 | Canonical Cx2 repair; rerun baseline, `eg24e2k`, and `eg96e8k`. |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 | Fine-grained EG variants may need lower mbz for memory. |
| 275M | 8 | 786,432 | 96 | 8 | 1 | 4 | Use 8 GPUs going forward; historical baseline used 4 GPUs / mb8. |
| 275M | 16 | 1,048,576 | 128 | 8 | 1 | 16 | Keep unless throughput or queue pressure says otherwise. |
| mid_480m | 1 | 262,144 | 32 | 4 | 1 | 8 | Baseline policy. |
| mid_480m | 2 | 524,288 | 64 | 4 | 1 | 8 | Baseline policy. |
| mid_480m | 4 | 524,288 | 64 | 4 | 1 | 8 | Baseline policy. |
| mid_480m | 8 | 786,432 | 96 | 8 | 1 | 4 | Baseline policy. |
| 810M | 1 | 262,144 | 32 | 8 | 1 | 4 | Some completed pilots used 4 GPUs. Prefer 8 GPUs going forward. |
| 810M | 2 | 524,288 | 64 | 8 | 1 | 4 | Baseline policy. |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 | Baseline policy. |
| 810M | 8 | 786,432 | 96 | 16 | 1 | 4 | Forward policy; historical completed runs used 8 GPUs. |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 | Baseline policy. |
| 1.2B | 2 | 524,288 | 64 | 8 | 1 | 2 | Planned. |
| 1.2B | 4 | 524,288 | 64 | 16 | 1 | 2 | Forward policy; historical completed runs used 8 GPUs. |
| 1.2B | 8 | 786,432 | 96 | 8 | 1 | 4 | Preferred replacement policy after 32 GPU / mb1 underperformed. |

## Baseline: Actual Settings Used

| Model | Cx | Total batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 275M | 1 | 262,144 | 32 | 1 | 1 | 16 | Current canonical family. Some names say `gpu2`, but the tracker records 1 GPU for the canonical Cx1 relaunch family. |
| 275M | 2 | 262,144 | 32 | 2 | 1 | 16 | Current canonical family. This is the outlier relative to the later Cx2 plan and the EG Cx2 jobs. |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 | Current canonical family. |
| 275M | 8 | 786,432 | 96 | 4 | 1 | 8 | Current completed canonical family. Later policy would prefer 8 GPUs. |
| 275M | 16 | 1,048,576 | 128 | 8 | 1 | 16 | Current completed canonical family. |
| mid_480m | 1 | 262,144 | 32 | 4 | 1 | 8 | Midpoint baseline. |
| mid_480m | 2 | 524,288 | 64 | 4 | 1 | 8 | Midpoint baseline. |
| mid_480m | 4 | 524,288 | 64 | 4 | 1 | 8 | Midpoint baseline. |
| mid_480m | 8 | 786,432 | 96 | 8 | 1 | 4 | Midpoint baseline. |
| 810M | 1 | 262,144 | 32 | 4 / 8 | 1 | 4 | Main sweep used 4 GPUs; one cold sentinel used 8 GPUs. Same optimizer batch and microbatch. |
| 810M | 2 | 524,288 | 64 | 8 | 1 | 4 | Baseline Cx2 sweep. |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 | Baseline Cx4 sweep. |
| 810M | 8 | 786,432 | 96 | 8 | 1 | 4 | Current completed baseline. Forward policy later moved this row to 16 GPUs. |
| 810M | 16 | 1,048,576 | 128 | 8 | 1 | 4 | Started and intentionally stopped; not a completed canonical baseline. |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 | Baseline Cx1 sweep. |
| 1.2B | 2 | 524,288 | 64 | 8 | 1 | 2 | Planned/not launched as of this audit. |
| 1.2B | 4 | 524,288 | 64 | 8 | 1 | 2 | Actual baseline runs. Forward policy later moved this row to 16 GPUs. |
| 1.2B | 8 | 786,432 | 96 | 32 | 1 | 1 | Initial 4-node jobs; poor throughput. One 4e-4 job was kept as a systems comparison. |
| 1.2B | 8 | 786,432 | 96 | 8 | 1 | 4 | Replacement policy for 2e-4 and 8e-4. Preferred current setting unless later evidence says otherwise. |

## Forward Resource Policy

This is the policy we had written down before this audit. It should be revised
after we settle the dense-ladder comparison and decide whether 275M Cx2 should
be `b256k` or `b512k`.

| Model | Cx1 | Cx2 | Cx4 | Cx8 |
| --- | ---: | ---: | ---: | ---: |
| 275M | 1-2 GPUs | 2 GPUs | 4 GPUs | 8 GPUs |
| mid_480m | 4 GPUs | 4 GPUs | 4 GPUs | 8 GPUs |
| 810M | 8 GPUs | 8 GPUs | 8 GPUs | 16 GPUs |
| 1.2B | 8 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |

## Expert Granularity: Actual Launched Settings

Variants:

- `eg24e2k`: 24 experts, top-2, larger experts.
- `eg96e8k`: 96 experts, top-8, smaller experts.
- `eg192e16k`: 192 experts, top-16, Cx1 exploratory extreme.
- `eg384e32k`: 384 experts, top-32, Cx1 exploratory ultra-extreme.

| Variant(s) | Cx | Total batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `eg24e2k` | 1 | 262,144 | 32 | 1 | 1 | 16 | Cx1 transfer probe. |
| `eg96e8k` | 1 | 262,144 | 32 | 1 | 1 | 8 | Cx1 transfer probe; mb16 OOMed in smoke. |
| `eg192e16k` | 1 | 262,144 | 32 | 1 | 1 | 4 | Exploratory Cx1 extreme. |
| `eg384e32k` | 1 | 262,144 | 32 | 1 | 1 | 2 | Exploratory Cx1 ultra-extreme. |
| `eg24e2k` | 2 | 524,288 | 64 | 2 | 1 | 16 | This does not match 275M baseline Cx2, which used 262,144 tokens. |
| `eg96e8k` | 2 | 524,288 | 64 | 2 | 1 | 8 | This does not match 275M baseline Cx2, which used 262,144 tokens. |
| `eg24e2k` | 4 | 524,288 | 64 | 4 | 1 | 16 | Batch-matched to 275M baseline Cx4. |
| `eg96e8k` | 4 | 524,288 | 64 | 4 | 1 | 8 | Batch-matched to 275M baseline Cx4. |
| `eg24e2k`, `eg96e8k` | 8 | 786,432 | 96 | 8 | 1 | 4 | Batch-matched to 275M baseline Cx8, but not systems-matched: baseline used 4 GPUs / mb8. |

## Total Sparsity: Actual Started Settings

Variants:

- `sp96e4k`: 96 experts, top-4, higher total params than baseline.
- `sp192e4k`: 192 experts, top-4, much higher total params than baseline.

These jobs are currently paused/cancelled pending the plan reset.

| Variant(s) | Cx | Total batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sp96e4k`, `sp192e4k` | 0.02 smoke | 262,144 | 32 | 4 | 1 | 4 | Smoke tests. |
| `sp96e4k`, `sp192e4k` | 1 | 262,144 | 32 | 4 | 1 | 4 | Cx1 jobs were started; only one finished before pause. |
| `sp96e4k`, `sp192e4k` | 4 | 524,288 | 64 | 4 | 1 | 4 | Cx4 jobs were started/queued and then paused. |
| `sp96e4k`, `sp192e4k` | 8 | 786,432 | 96 | 8 | 1 | 4 | Script exists but these should be treated as planned, not part of the completed started set. |

## Drift / Mismatch Notes

1. The most important mismatch is 275M Cx2:
   - baseline canonical family: 262,144 tokens, `global_batch_size_seq=32`;
   - expert-granularity Cx2: 524,288 tokens, `global_batch_size_seq=64`.
2. The 275M Cx8 baseline and expert-granularity runs are optimizer-batch
   matched but systems-different:
   - baseline: 4 GPUs, mb8;
   - EG: 8 GPUs, mb4.
3. The 1.2B Cx8 baseline has mixed systems settings:
   - initial jobs: 32 GPUs, mb1, poor throughput;
   - replacement jobs: 8 GPUs, mb4.
4. The forward policy table differs from some historical completed baselines:
   - 275M Cx8 policy says 8 GPUs, but completed baseline used 4 GPUs;
   - 810M Cx8 policy says 16 GPUs, but completed baseline used 8 GPUs;
   - 1.2B Cx4 policy says 16 GPUs, but actual completed runs used 8 GPUs.
5. The experiment-plan generic table said 275M EG Cx8 mb8, but the actual
   EG Cx8 launcher uses mb4.

## Batch Settings Comparison

This table uses baseline as the reference. Differences in experiment columns
are marked with **DIFF**. Empty experiment cells mean that experiment has not
started for that model/Cx pair.

| Model | Cx | Baseline batch | Baseline GPUs / EP / mbz | EG batch | EG GPUs / EP / mbz | Sparsity batch | Sparsity GPUs / EP / mbz | Difference notes |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 275M | 1 | 262,144 tok / 32 seq | 1 / 1 / 16 | 262,144 tok / 32 seq | `eg24e2k`: 1 / 1 / 16; `eg96e8k`: 1 / 1 / 8; `eg192e16k`: 1 / 1 / 4; `eg384e32k`: 1 / 1 / 2 | 262,144 tok / 32 seq | 4 / 1 / 4 | Same optimizer batch. EG and sparsity differ in systems/memory settings. |
| 275M | 2 | 262,144 tok / 32 seq | 2 / 1 / 16 | **DIFF:** 524,288 tok / 64 seq | `eg24e2k`: 2 / 1 / 16; `eg96e8k`: 2 / 1 / 8 |  |  | EG Cx2 is not batch-matched to baseline Cx2. |
| 275M | 4 | 524,288 tok / 64 seq | 4 / 1 / 16 | 524,288 tok / 64 seq | `eg24e2k`: 4 / 1 / 16; `eg96e8k`: 4 / 1 / 8 | 524,288 tok / 64 seq | 4 / 1 / 4 | Same optimizer batch. EG fine and sparsity differ in microbatch. |
| 275M | 8 | 786,432 tok / 96 seq | 4 / 1 / 8 | 786,432 tok / 96 seq | **DIFF systems:** 8 / 1 / 4 | planned only: 786,432 tok / 96 seq | planned: 8 / 1 / 4 | Same optimizer batch, but EG uses different GPUs/mbz from baseline. |
| 275M | 16 | 1,048,576 tok / 128 seq | 8 / 1 / 16 | planned only | planned |  |  | EG doc intends full 275M ladder, but no Cx16 EG launch yet in this audit. |
| mid_480m | 1 | 262,144 tok / 32 seq | 4 / 1 / 8 |  |  |  |  | Baseline only. |
| mid_480m | 2 | 524,288 tok / 64 seq | 4 / 1 / 8 |  |  |  |  | Baseline only. |
| mid_480m | 4 | 524,288 tok / 64 seq | 4 / 1 / 8 |  |  |  |  | Baseline only. |
| mid_480m | 8 | 786,432 tok / 96 seq | 8 / 1 / 4 |  |  |  |  | Baseline only. |
| 810M | 1 | 262,144 tok / 32 seq | 4 / 1 / 4; one sentinel 8 / 1 / 4 |  |  |  |  | Baseline only. |
| 810M | 2 | 524,288 tok / 64 seq | 8 / 1 / 4 |  |  |  |  | Baseline only. |
| 810M | 4 | 524,288 tok / 64 seq | 8 / 1 / 4 |  |  |  |  | Baseline only. |
| 810M | 8 | 786,432 tok / 96 seq | 8 / 1 / 4 |  |  |  |  | Baseline only; forward policy later suggested 16 GPUs. |
| 810M | 16 | 1,048,576 tok / 128 seq | 8 / 1 / 4 |  |  |  |  | Started/stopped, not canonical complete. |
| 1.2B | 1 | 262,144 tok / 32 seq | 8 / 1 / 2 |  |  |  |  | Baseline only. |
| 1.2B | 2 | planned: 524,288 tok / 64 seq | planned: 8 / 1 / 2 |  |  |  |  | Not launched as of this audit. |
| 1.2B | 4 | 524,288 tok / 64 seq | 8 / 1 / 2 |  |  |  |  | Baseline only; forward policy later suggested 16 GPUs. |
| 1.2B | 8 | 786,432 tok / 96 seq | initial 32 / 1 / 1; replacements 8 / 1 / 4 |  |  |  |  | Same optimizer batch, mixed systems settings. |

## Dense-Ladder Comparison To Fill In

We still need the dense-ladder source-of-truth batch table before making new
launch decisions. W&B runs I found in `ai2-llm/hybrid-small-suite` include
450M Cx2 runs with a 4,194,304-token global batch and older 1.4B Cx2 runs with
a 2,097,152-token global batch, but those may not be the dense ladder rows Jacob
is looking for. Treat those as observations, not settled policy.

## External Reference: `ai2-llm/olmo-hybrid-pe`

Queried W&B project `ai2-llm/olmo-hybrid-pe` on 2026-06-12. These are actual
config values from the runs, grouped by W&B model-size tag and Cx tag. The
project uses 8192-token sequences, so `global batch seqs` is
`data_loader.global_batch_size / 8192`, and `rank microbatch seqs` is
`train_module.rank_microbatch_size / 8192`.

| Model tag | Cx | Runs | States | Global batch tokens | Global batch seqs | Rank microbatch tokens | Rank microbatch seqs | GPUs | Nodes | Max duration tokens |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 260m | 1 | 10 | 8 finished, 2 failed | 196,608 | 24 | 24,576 | 3 | 8 | 1 | ~3.91B |
| 260m | 2 | 8 | 8 finished | 327,680 | 40 | 40,960 | 5 | 8 | 1 | ~7.82B |
| 260m | 4 | 9 | 8 finished, 1 crashed | 458,752 | 56 | 57,344 | 7 | 8 | 1 | ~15.64B |
| 260m | 6 | 10 | 6 finished, 2 failed, 2 crashed | 589,824 | 72 | 24,576 | 3 | 8 | 1 | ~23.46B |
| 260m | 8 | 6 | 6 finished | 655,360 | 80 | 40,960 | 5 | 8 | 1 | ~31.28B |
| 709m | 1 | 6 | 6 finished | 393,216 | 48 | 24,576 | 3 | 8 | 1 | ~12.64B |
| 709m | 2 | 7 | 6 finished, 1 failed | 524,288 | 64 | 16,384 | 2 | 8 | 1 | ~25.28B |
| 709m | 4 | 18 | 5 finished, 13 failed | 786,432 | 96 | 24,576 | 3 | 8 | 2 | ~50.56B |
| 709m | 6 | 10 | 6 finished, 4 failed | 1,048,576 | 128 | 16,384 | 2 | 8 | 2 | ~75.84B |
| 709m | 8 | 12 | 4 finished, 8 failed | 1,179,648 | 144 | 24,576 | 3 | 8 | 2 | ~101.12B |
| 1.3B | 1 | 9 | 4 finished, 4 failed, 1 crashed | 524,288 | 64 | 16,384 | 2 | 8 | 2 | ~23.35B |
| 1.3B | 2 | 9 | 1 finished, 6 failed, 2 crashed | 786,432 | 96 | 16,384 | 2 | 8 | 2 | ~46.71B |
| 1.3B | 4 | 6 | 4 failed, 2 crashed | 1,048,576 | 128 | 16,384 | 2 | 8 | 4 | ~93.42B |
| 1.3B | 6 | 4 | 1 failed, 3 crashed | 1,310,720 | 160 | 8,192 | 1 | 8 | 4 | ~140.12B |

The 260M Cx2 row is the most relevant immediate comparison: this external
workspace used a 327,680-token global batch, not our current 275M baseline
Cx2 value of 262,144 tokens and not the later 524,288-token setting used for
our expert-granularity Cx2 runs.

## 275M Cx2 Repair Runs

On 2026-06-12 we decided to rerun the three Cx2 comparison curves at the new
canonical Cx2 setting:

- total batch: 393,216 tokens / 48 sequences;
- systems: 2 GPUs, EP=1, microbatch=8;
- predicted LR center: about `1.75e-3` to `1.8e-3` from completed 275M
  Cx1/Cx4/Cx8/Cx16 fits, excluding the old mismatched Cx2;
- canonical LRs: `9e-4`, `1.8e-3`, `3.6e-3`;
- families: baseline A0, `eg24e2k`, `eg96e8k`;
- launcher: `launch_275m_cx2_b384k_comparison.sh`.

Until these finish, treat the old Cx2 curves as diagnostic rather than
canonical for baseline-vs-experiment comparisons.
