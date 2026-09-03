# Routed-FFN speed, theoretical vs measured (NVIDIA H100 80GB HBM3, olmo-core training image, bf16, batch 1)

Forced one-hot routing on EVERY layer at cost c. `theory` = model-wide FLOP speedup at that seq len. `model measured` = whole model where it fits on one GPU, else per-layer probe x depth + measured embed/head. `FFN-only` = one routed FFN sub-block timed alone (forward+backward on L tokens), i.e. the routing implementation against the pure width ratio.

| model | L | FFN FLOP share | theory @1/16 | model measured @1/16 (train) | model measured @null (train) | FFN-only x @1/16 | FFN-only x @1/H | FFN-only x @null | FFN-only ms @full |
|---|---|---|---|---|---|---|---|---|---|
| q35-0.8B | 2048 | 0.39 | 1.57 | 1.00 (full) | 1.02 (full) | 1.1 | 1.0 | 1.4 | 1.38 |
| q35-0.8B | 8192 | 0.32 | 1.42 | 1.05 (full) | 1.07 (full) | 1.8 | 1.6 | 2.1 | 2.25 |
| q35-0.8B | 32768 | 0.18 | 1.21 | 1.10 (full) | 1.12 (full) | 3.4 | 3.7 | 4.9 | 6.56 |
| q35-2B | 2048 | 0.54 | 2.02 | 1.00 (full) | 1.03 (full) | 1.3 | 1.3 | 1.6 | 1.70 |
| q35-2B | 8192 | 0.49 | 1.86 | 1.14 (full) | 1.16 (full) | 3.0 | 3.2 | 4.1 | 4.34 |
| q35-2B | 32768 | 0.37 | 1.53 | nan (full) | nan (full) | 4.8 | 5.5 | 7.6 | 15.39 |
| q35-4B | 2048 | 0.62 | 2.36 | 1.06 (full) | 1.07 (full) | 1.8 | 1.8 | 2.3 | 2.45 |
| q35-4B | 8192 | 0.55 | 2.08 | 1.22 (full) | 1.25 (full) | 4.2 | 4.5 | 6.0 | 6.97 |
| q35-4B | 32768 | 0.40 | 1.59 | 1.23 (probe) | 1.24 (probe) | 6.2 | 7.7 | 11.3 | 26.31 |
| q35-9B | 2048 | 0.69 | 2.83 | 1.17 (full) | 1.20 (full) | 2.6 | 2.5 | 3.3 | 3.73 |
| q35-9B | 8192 | 0.65 | 2.57 | nan (full) | nan (full) | 5.6 | 6.8 | 9.2 | 13.16 |
| q35-9B | 32768 | 0.54 | 2.01 | 1.41 (probe) | 1.45 (probe) | 8.2 | 10.9 | 15.9 | 52.97 |
| q35-27B | 2048 | 0.77 | 3.54 | 1.38 (probe) | 1.48 (probe) | 4.0 | 4.3 | 6.0 | 6.32 |
| q35-27B | 8192 | 0.73 | 3.13 | 1.61 (probe) | 1.67 (probe) | 6.9 | 10.2 | 12.8 | 22.41 |
| q35-27B | 32768 | 0.60 | 2.30 | 1.44 (probe) | 1.57 (probe) | 9.3 | 15.2 | 22.4 | 88.88 |
| q35like-70B | 2048 | 0.85 | 4.97 | 1.87 (probe) | 2.02 (probe) | 5.9 | 6.9 | 9.8 | 14.33 |
| q35like-70B | 8192 | 0.83 | 4.44 | 2.23 (probe) | 2.43 (probe) | 9.4 | 16.8 | 21.7 | 54.28 |
| q35like-70B | 32768 | 0.74 | 3.26 | - | - | 11.4 | 24.2 | 36.0 | 219.02 |

Reading (H100, training image):
- Whole-model wall-clock speedups are smaller on H100 than on the A100 run (4B/8k train 1.22x vs 1.32x): the FFN GEMMs are the part Hopper accelerates most, so the non-FFN share (GatedDeltaNet, attention, head) grows and caps the gain. Model-level gains reach 1.2x (4B) / 1.4x (9B) / 1.6x (27B) / 2.2-2.4x (70B geometry) at 8k against FLOP theory of 2.1x / 2.6x / 3.1x / 4.4x.
- FFN-only: the routing implementation delivers the width ratio only where the GEMMs are large. At cost 1/16 (theory 16x): 1.1x for 0.8B at 2k, 4.2x for 4B at 8k, 8.2x for 9B at 32k, 11.4x for 70B geometry at 32k. The floor is the per-token routing work (router, sort, gather/scatter, gain scaling), roughly 1.3-2 ms per 32k-token layer regardless of width.
- Widths below 1/16 buy little in wall-clock at small sizes (0.8B: 1/16 3.4x vs 1/H 3.7x at 32k) but keep paying at large ones (70B geometry: 11.4x vs 24.2x at 32k), because there the width-1 GEMMs are still non-trivial at 1/16.
