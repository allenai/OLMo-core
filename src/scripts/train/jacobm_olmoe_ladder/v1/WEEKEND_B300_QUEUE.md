# Weekend B300 Queue Plan

Status: refreshed 2026-06-15 23:30 UTC after the Qwen3-like 275M launch and
plot refresh.

This note is the short-term launch surface for the low-priority, preemptible
B300 allocation. It complements `RUNS.md` and should be refreshed from Beaker
before launching a large wave.

## B300 Launch Policy

Use the working B300 runtime path from the successful Holmes smoke:

- Cluster: `ai2/holmes`
- Workspace: `ai2/holmes-testing` unless Jacob switches to the main workspace
- Priority: low
- Preemptible: yes
- Image: `tianhuat/olmo-core-torch212-2404-cu130`
- Python mode: launch with the image as-is; skip Gantry Python setup with
  `--no-python`
- Python binary: `/opt/conda/bin/python`
- `PYTHONPATH`: include `/gantry-runtime/src:/workspace/OLMo-core/src`

Do not put node count, GPU count, EP, microbatch, cluster, workspace, image, or
priority in run names. Put those in W&B/Beaker tags/config so systems-only
changes can resume the same checkpoint.

## Current Queue Snapshot

Live Beaker snapshot across the recently tracked launch surface in `RUNS.md`,
`CURRENT_PLAN.md`, and this file:

- 43 started jobs using 168 GPUs.
- 9 created/queued jobs requesting 40 GPUs.
- 110 finalized jobs.
- 1 stale tracker entry is a Beaker job id rather than an experiment id:
  `01KTSB2H1TMF7Z1T2MY40J2QM0`.

The finalized nonzero-exit entries are known historical failures, intentional
cancellations, or replaced attempts: Holmes compile failures, legal-microbatch
relaunches, and Titan reroutes. The current live surface did not show new failed
or canceled jobs in this status pass.

Still open:

- Qwen3-like 275M: all 24 full-grid jobs are started on Holmes; the two smoke
  tests finalized.
- Shared expert promoted ladder: 10 jobs are started on Holmes across 480M,
  810M, and 1.2B. The legal-microbatch retries for 480M Cx2 and 810M Cx2 are
  the active versions.
- Expert granularity: 1.2B Cx4 fine and 1.2B Cx8 coarse are started; 1.2B Cx8
  fine is still created/queued.
- Dense schedule: six Cx4 Titan reroutes are started; eight Cx8 Titan reroutes
  are still created/queued.
- 1.2B baseline: the Titan `Cx2 b384k lr1.2e-3` hot-side run is started.

Plots were refreshed after this status pass with stale-cache updates and running
jobs excluded from LR/summary decisions.

## Experiment Catalog

### 1. Finish Current Runs

Let the current non-shared jobs finish unless they fail:

- 275M total-sparsity Cx4/Cx8 tail.
- 810M expert-granularity Cx8.
- 1.2B expert-granularity Cx1/Cx4.
- 1.2B baseline Cx8 jobs already in flight.

### 2. Expert Granularity Completion

Serious variants:

- `eg24e2k` / `coarse_24e_top2`
- `eg96e8k` / `fine_96e_top8`

Already covered or in flight:

- 275M Cx1/Cx2/Cx4/Cx8.
- 480M Cx1/Cx2/Cx4/Cx8.
- 810M Cx1/Cx2/Cx4, with Cx8 currently running/queued.
- 1.2B Cx1/Cx4 currently running/queued.

Remaining to queue after baseline result refresh:

- `eg-1p2b-cx2-eg24e2k-...`
- `eg-1p2b-cx2-eg96e8k-...`
- `eg-1p2b-cx8-eg24e2k-...`
- `eg-1p2b-cx8-eg96e8k-...`

Use best observed baseline LRs at the same size/Cx unless the just-refreshed
plots show a clear reason to use a fitted value instead.

### 3. Total Sparsity Full Ladder

Serious variants:

- `sp96e4k` / `high_total_96e_top4`
- `sp192e4k` / `huge_total_192e_top4`

Current 275M LR verification grid is mostly complete or running:

- Cx1: complete for both variants.
- Cx2: complete for both variants.
- Cx4/Cx8: tail still running/queued.

After 275M Cx4/Cx8 finish and plots are refreshed, promote to larger sizes with
predicted or best-observed transferred LRs:

| Size | Cx | Variants | Jobs |
| --- | ---: | --- | ---: |
| 480M | 1, 2, 4, 8 | `sp96e4k`, `sp192e4k` | 8 |
| 810M | 1, 2, 4, 8 | `sp96e4k`, `sp192e4k` | 8 |
| 1.2B | 1, 2, 4, 8 | `sp96e4k`, `sp192e4k` | 8 |

Before launching 480M/810M/1.2B sparsity, run exact dry parameter checks for
those sizes and record active params, total params, and active/total fraction.

### 4. Shared Expert / Dense Schedule

Deferred for this weekend unless Jacob explicitly promotes it again. The
no-shared matched-active probes were canceled before start. When resumed, the
first useful wave is still likely:

- `1dense_no_shared_am`
- `2dense_no_shared_am`
- `alternating_no_shared_am`

Each should start at 275M Cx1/Cx4 with three LRs centered on baseline optima.

### 5. Later Axes

Do not spend weekend flood compute here until expert granularity and total
sparsity are clearer:

- Width/depth geometry: needs dry parameter checks and possibly adjusted head/KV
  shapes.
- SWA/full-attention ratio: needs concrete launch scripts and candidate ratios.
- Integration runs: wait until we know which EG and sparsity settings win.

## Suggested First B300 Wave

Start smaller than the allocation ceiling:

1. Launch one real low-priority/preemptible B300 job from the next approved
   family to re-confirm the no-python runtime on a full-length job path.
2. If healthy, run the four remaining 1.2B expert-granularity jobs, because EG
   is closest to completing its full ladder.
3. Once the 275M sparsity tail finishes and plots are refreshed, queue 480M
   sparsity for both variants across Cx1/Cx2/Cx4/Cx8.
4. If 480M sparsity is healthy and the allocation is still large, queue the
   810M sparsity ladder, then 1.2B sparsity.

With 64 GPUs, the practical first wave is the EG completion jobs plus some or
all 480M sparsity. With 256+ GPUs, we can queue the entire 480M+810M sparsity
ladder after the 275M fit check. With 576 GPUs, we can add 1.2B sparsity too,
but only after a quick status/plot refresh so we do not scale a bad LR transfer.
