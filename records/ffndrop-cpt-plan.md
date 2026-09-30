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
- 2026-09-29 ~00:00: CPU tests pass (`src/test/nn/ffn_token_drop_test.py`); both CPT arms launched
  (ledger `src/scripts/train/memexpress/cpt/ffndrop/LAUNCH_LEDGER.tsv`),
  wandb https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/fdcpt-q35-4b
