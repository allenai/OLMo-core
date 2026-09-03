# Routed-FFN speed, theoretical vs measured (A100-80GB, gantry conda env, bf16, batch 1)

Forced one-hot routing on EVERY layer at a fixed cost c (fraction of the FFN's intermediate width). `theory` = model-wide FLOP speedup at that seq len (attention counted at its real quadratic cost). `full` = measured on the whole model where it fits on one GPU; `probe` = per-layer time from 2- vs 4-layer builds times the full depth plus the measured embed/head overhead. Cost 1 is the routed code path at full width (the routing overhead itself is the difference between cost 1 and the un-routed model, not shown here).

| model | L | shape | FFN FLOP share | theory x @1/16 | measured x @1/16 | theory x @null | measured x @null |
|---|---|---|---|---|---|---|---|
| q35-0.8B | 2048 | train | 0.39 | 1.57 | 1.00 (full) | 1.63 | 1.01 (full) |
| q35-0.8B | 2048 | prefill | 0.39 | 1.57 | 1.00 (full) | 1.63 | 1.19 (full) |
| q35-0.8B | 8192 | train | 0.32 | 1.42 | 1.08 (full) | 1.46 | 1.09 (full) |
| q35-0.8B | 8192 | prefill | 0.32 | 1.42 | 1.18 (full) | 1.46 | 1.20 (full) |
| q35-0.8B | 32768 | train | 0.18 | 1.21 | 1.15 (full) | 1.23 | 1.16 (full) |
| q35-0.8B | 32768 | prefill | 0.18 | 1.21 | 1.18 (full) | 1.23 | 1.20 (full) |
| q35-2B | 2048 | train | 0.54 | 2.02 | 1.03 (full) | 2.16 | 1.03 (full) |
| q35-2B | 2048 | prefill | 0.54 | 2.02 | 1.21 (full) | 2.16 | 1.18 (full) |
| q35-2B | 8192 | train | 0.49 | 1.86 | 1.22 (full) | 1.97 | 1.24 (full) |
| q35-2B | 8192 | prefill | 0.49 | 1.86 | 1.43 (full) | 1.97 | 1.46 (full) |
| q35-2B | 32768 | train | 0.37 | 1.53 | nan (full) | 1.59 | nan (full) |
| q35-2B | 32768 | prefill | 0.37 | 1.53 | 1.39 (full) | 1.59 | 1.43 (full) |
| q35-4B | 2048 | train | 0.62 | 2.36 | 1.13 (full) | 2.60 | 1.13 (full) |
| q35-4B | 2048 | prefill | 0.62 | 2.36 | 1.37 (full) | 2.60 | 1.33 (full) |
| q35-4B | 8192 | train | 0.55 | 2.08 | 1.32 (full) | 2.24 | 1.36 (full) |
| q35-4B | 8192 | prefill | 0.55 | 2.08 | 1.51 (full) | 2.24 | 1.57 (full) |
| q35-4B | 32768 | train | 0.40 | 1.59 | 1.31 (probe) | 1.66 | 1.34 (probe) |
| q35-4B | 32768 | prefill | 0.40 | 1.59 | 1.44 (full) | 1.66 | 1.49 (full) |
| q35-9B | 2048 | train | 0.69 | 2.83 | 1.28 (full) | 3.22 | 1.31 (full) |
| q35-9B | 2048 | prefill | 0.69 | 2.83 | 1.59 (full) | 3.22 | 1.58 (full) |
| q35-9B | 8192 | train | 0.65 | 2.57 | nan (full) | 2.87 | nan (full) |
| q35-9B | 8192 | prefill | 0.65 | 2.57 | 1.78 (full) | 2.87 | 1.87 (full) |
| q35-9B | 32768 | train | 0.54 | 2.01 | 1.49 (probe) | 2.15 | 1.56 (probe) |
| q35-9B | 32768 | prefill | 0.54 | 2.01 | 1.68 (full) | 2.15 | 1.78 (full) |
| q35-27B | 2048 | train | 0.77 | 3.54 | 1.46 (probe) | 4.26 | 1.61 (probe) |
| q35-27B | 2048 | prefill | 0.77 | 3.54 | 1.89 (probe) | 4.26 | 2.31 (probe) |
| q35-27B | 8192 | train | 0.73 | 3.13 | 1.83 (probe) | 3.65 | 1.96 (probe) |
| q35-27B | 8192 | prefill | 0.73 | 3.13 | 1.96 (probe) | 3.65 | 2.17 (probe) |
| q35-27B | 32768 | train | 0.60 | 2.30 | 1.64 (probe) | 2.51 | 1.72 (probe) |
| q35-27B | 32768 | prefill | 0.60 | 2.30 | 1.70 (probe) | 2.51 | 1.80 (probe) |
| q35like-70B | 2048 | train | 0.85 | 4.97 | 2.16 (probe) | 6.75 | 2.51 (probe) |
| q35like-70B | 2048 | prefill | 0.85 | 4.97 | 2.47 (probe) | 6.75 | 2.88 (probe) |
| q35like-70B | 8192 | train | 0.83 | 4.44 | 2.51 (probe) | 5.77 | 2.80 (probe) |
| q35like-70B | 8192 | prefill | 0.83 | 4.44 | 2.65 (probe) | 5.77 | 2.97 (probe) |
| q35like-70B | 32768 | train | 0.74 | 3.26 | - | 3.84 | - |
| q35like-70B | 32768 | prefill | 0.74 | 3.26 | 2.35 (probe) | 3.84 | 2.51 (probe) |

Reading: measured speedups saturate by cost 1/16 (the FFN is already off the critical path there); the remaining gap to theory is the non-FFN work (GatedDeltaNet, attention, embed/head) running at lower utilization than the wide FFN GEMMs, so its wall-clock share exceeds its FLOP share. The gap widens with FFN FLOP share: at 70B geometry, 2k tokens, theory 6.8x vs 2.5x measured.
