# Drop-CPT: random null-FFN continued pretraining before routed-FFN SFT

Started 2026-09-29 (Prasann). Code: `src/olmo_core/nn/ffn_token_drop.py`, launcher
`src/scripts/train/memexpress/cpt/ffndrop/`.

## Why
Routed FFN (nested-width router + null rung) was never compute-optimal against dense SFT on Qwen3.5-4B
(`records/flop-scaling-report-2026-09-02.md`: multipliers 0.76/0.81/0.91 at 0.8B/2B/9B; only the joint
FFN+attention flex-c70 reached 1.15). The losses look like adaptation, not a ceiling: all-layer routing
collapsed on 30-150-step SFT runs and recovered with horizon (outlier stage 2 at 320M: 0.70 at 8k vs
0.00-0.05 at <=56M), with losses concentrated on the 32k rung. Drop-CPT moves that adaptation into a
one-time CPT (the LayerDrop / stochastic-depth argument). Ceiling: FFN ~57% of training FLOPs, so a
lossless router is worth at most ~1.4x (layers 12+) to ~2.3x (all layers).

## Accounting (decided 2026-09-29)
The drop-CPT checkpoint is a new BASE: its CPT is amortized, not charged to each task's SFT FLOPs
(like the dense base's own pretraining). Report the break-even too: SFT tokens of saving that pay back
the CPT.

## Design
Arms (CPT, 1B tokens, `softdetach_cpt/shards/cpt_u1B`, base `q35-4b-base-markerfix`, 8 x 64k rows/step,
lr 3e-5, 1 node x 8 H100, jupiter urgent unallocated; mid-CPT saves every 480 steps ~ 250M tokens):

| CPT arm | then SFT | tells us |
|---|---|---|
| dense | dense | the baseline to beat |
| dense | routed | routing without drop-CPT (controls for the CPT data itself) |
| drop75l12 | routed | **the question** |
| drop75l12 | dense | drop-CPT didn't hurt the base |

`drop75l12`: per row r ~ U[0, 0.75] shared across layers; each token skips each FFN of layers >= 1 with
prob r; each (row, layer) drops the whole row's FFN with prob 0.125 (trained routers mostly prune whole
layers). No 1/(1-p) rescale (a routed null token gets exactly zero). Kept tokens run on a compacted
gather, so the drop is a real FLOP saving; draws seeded by (seed, forward call, rank, layer) so AC
recompute reproduces them.

SFT stage (next): the fs35 recipe (`debug/flop_scaling/launch_grid35.py` / `orchestrate35.py`) with
the base swapped to each CPT export, on contradiction / oolong / nq / outlier, dense and ffnmoe arms.

## Checks
- Dev loss on `cpt_dev` (full FFN) for both CPT arms, and CE under random drop at r = 0.25/0.5/0.75 for
  base vs dense-CPT vs drop-CPT: the drop-robustness curve must flatten for the SFT stage to have a chance.
- FLOP meter charges dense FFN cost on the drop arm (CPT is amortized, so this is only a wall-clock note).

## Status
- 2026-10-03 05:30: **SFT results** (routed `ffnmoe-t10`, L12+, two-sided target 0.10 -- both drop-CPT runs
  converged to it like the base-model runs; meter actual/dense 0.900/0.885 vs base-model 0.880/0.876 on the
  same packed pricing, so ~1-2% more FLOPs; real-length PF below scaled from the results35 numbers).
  Eval sets = fs35 ladders (contra eval_size 500/rung, nq 600/rung), routing ON at eval.

  | setting | run | 2k | 8k | 16k | 32k | mean | ~PF |
  |---|---|---|---|---|---|---|---|
  | contra 28M | base -> routed | .935 | .881 | .790 | .627 | .808 | 515 |
  | contra 28M | **drop-CPT -> routed** | .924 | .871 | .814 | .687 | **.824** | ~541 |
  | contra | dense 14M / 28M | .909/.967 | .836/.932 | .751/.873 | .592/— | .772/.924* | 337/674 |
  | nq 32M | base -> routed | .977 | .915 | .818 | .700 | .853 | 572 |
  | nq 32M | **drop-CPT -> routed** | .972 | .908 | .857 | .778 | **.879** | ~586 |
  | nq | dense 16M / 32M | .977/.977 | .917/.930 | .857/.885 | .760/.837 | .878/.907 | 379/758 |

  (*dense-28M contra lacks its 32k rung.) Read: drop-CPT lifts routed SFT exactly where it used to lose --
  16k/32k (contra 32k +.06, nq 32k +.08, nq 16k +.04; 2k/8k within noise) -- but neither point reaches the
  dense frontier at matched FLOPs: nq ~.017 below log-interpolated dense at ~586 PF (base routed was ~.04
  below; drop-CPT routed matches dense-16M's .878 at ~1.55x its FLOPs); contra on the 3 rungs dense-28M has
  (2k/8k/16k) .870 vs ~.895 interpolated dense (base routed .869 -- no change there; the gain is all 32k).
  **Confound:** without the dense-CPT control, the 16k/32k gains may be the long-context CPT data itself,
  which would lift dense SFT too and move the frontier. Cheapest disambiguation: dense SFT from the drop-CPT
  base on the same two settings (`FS35_BASE=... FS35_BASE_TAG=-bfdrop launch_grid35.py --arms dense`).
- 2026-10-03 04:33: **drop-CPT finished (exit 0) and its probe is in** -- same dev rows, same drop pattern as
  the base probe (⚠ eval_size=32 rows): CE r=0 1.264, r=0.25 1.346 (+0.08), r=0.5 1.467 (+0.20),
  r=0.75 1.656 (+0.39) vs base +0.58/+2.73/+6.87 -- the null-FFN penalty shrank ~7-17x. Dense quality at r=0
  is 1.264 vs base 1.320; for scale, the sdcpt dense-CPT points on this dev set are 1.283 (32M) / 1.245
  (128M), so drop-CPT keeps dense CE roughly at dense-CPT level (no dense-1B reference: control dropped).
  First launches failed (another session's unpushed HEAD -> gantry UnpushedChangesError); pushed, relaunched
  04:26: SFT 01M40R8PBE5C192A0TFS7PE9NY (contra 28M), 01M40R9FN39TNRVGPJKXV3CM2S (nq 32M).
- 2026-10-02 15:50: Beaker capacity outage (jupiter/ceres/titan heavily cordoned) held both jobs ~2.5 days.
  **Base drop-robustness probe** (q35-4b-base-markerfix, cpt_dev, 32 held-out 64k rows ⚠ eval_size=32 rows,
  ~2M tokens; paired deltas): CE r=0 1.320, r=0.25 1.898 (+0.58), r=0.5 4.046 (+2.73), r=0.75 8.187 (+6.87)
  -- the untrained base is extremely fragile to null FFNs. **Drop-CPT** started 13:21 PDT; at step 870/1923
  (~10 s/step) train CE under drop (realized mean frac 0.48 on rank 0, expected 0.45) fell 4.36 (steps 1-10)
  -> 1.87 (50-100) -> 1.68 (800-870). ETA ~19:00 PDT.
- 2026-09-30 01:00: login-node `cpt/ffndrop/pipeline.py` (log `debug/ffndrop_cpt/pipeline.log`) waits for the
  drop-CPT run, then submits its dev-loss eval + routed SFT (`ffnmoe-t10`) from the export on contradiction 28M
  and nq 32M (base-model t10: 0.808 vs dense 0.924; 0.853 @572 PF vs dense-16M 0.878 @379 PF), then scores
  them with the fs35 native ladder evaluator. Runs `fs35r2-*-ffnmoe-t10-s*-bfdrop`, wandb group fdcpt-q35-4b-sft.
- 2026-09-30 00:47: dense-CPT control CANCELED before it started (Prasann: not needed). SFT
  comparisons use the existing fs35 dense points from the unmodified base; caveat: a drop-CPT gain then
  mixes the drop effect with the CPT data itself (drop-CPT -> dense SFT is the partial check).
- 2026-09-29 ~23:30: CPU tests pass (`src/test/nn/ffn_token_drop_test.py`); both CPT arms launched
  (ledger `src/scripts/train/memexpress/cpt/ffndrop/LAUNCH_LEDGER.tsv`),
  wandb https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/fdcpt-q35-4b
