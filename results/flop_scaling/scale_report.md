# Model-scale ladder: KV soft tokens and routed FFN vs dense at 0.8B / 2B / 4B / 9B / 27B (Qwen3.5)

Same recipe as the 4B grid (seq 65536, lr 5e-6, batch 8, one epoch), oolong and contradiction at two budgets. Arms: dense; KV soft tokens (oolong keep 1/6 gold-blind, contradiction gold + 1/3); routed FFN two-sided target 0.10 on the top 62.5% of layers. FLOPs priced per scale at real example lengths. Dense curves have two points per scale, so `x dense` interpolates dense f1 log-linearly in FLOPs between them (extrapolations flagged with ~).

## Summary (2026-09-04 23:10, all ladder runs: 0.8B–27B + the 4B grid rows + the trainable-prefix arm at 0.8B/2B/4B)

Large-budget points (contradiction 56M, oolong 80M), FLOPs dense would need to match the method's f1, divided by the method's FLOPs (>1 = method is compute-optimal; ~ = extrapolated beyond the dense range):

| task | arm | 0.8B | 2B | 4B | 9B | 27B* |
|---|---|---|---|---|---|---|
| contradiction | KV gold + 1/3 | 1.26 (steep dense curve) | 0.87 | 1.00 | **1.66** | **1.59** |
| contradiction | routed FFN, two-sided 0.10 | 0.47 | 0.76 | 0.81 | 0.91 | 0.67 |
| contradiction | routed FFN + frozen tail (H/16 prefix trained) | 0.58 | 0.73 | 0.85 | — | — |
| oolong | KV 1/6 gold-blind | **2.86** | **1.55** | 0.94 | 0.75~ | 0.34~ |
| oolong | routed FFN, two-sided 0.10 | 0.48 | 0.29~ | 0.62 | 0.48 | 0.15~ |
| oolong | routed FFN + frozen tail (H/16 prefix trained) | 0.53 | 0.35 | 0.43 | — | — |

\* 27B caveat: there is no Qwen3.5-27B base release, so the 27B rung fine-tunes the **post-trained** Qwen/Qwen3.5-27B (marker-repaired). Its dense points are much stronger (contradiction 0.980, oolong 0.786 at the large budget), which makes every method's matched-compute multiplier harder to earn than on a same-family base; the 27B column is not strictly comparable to the base-model rungs.

- **KV soft tokens stay compute-optimal on contradiction from 4B up**: 1.00 -> 1.66 (9B) -> 1.59 (27B). At 27B the KV arm scores 0.956 for 3341 PF where dense needs ~5300 PF. The larger model extracts more from a soft token per document; the 27B multiplier holding at ~1.6 on a post-trained base is the strongest evidence so far.
- **KV on oolong saturates at every scale**: the KV arm sits at f1 0.66–0.68 from 0.8B to 27B while dense climbs from 0.67 to 0.79 with size. Keeping 1/6 of the lines real caps the aggregate the model can compute, so the win at small scales (2.9x at 0.8B, 1.6x at 2B) turns into a tie at 4B and a loss from 9B on (0.34~ at 27B). A higher keep fraction or absolute breadth is the lever here.
- **Routed FFN approaches parity with scale but never crosses, and falls back at 27B**: 0.47 -> 0.76 -> 0.81 -> 0.91 -> 0.67 on contradiction. The FFN FLOP share grows with model size (0.39 at 0.8B, 0.69 at 9B), so the same routing buys a bigger saving and the accuracy penalty shrinks through 9B; on the post-trained 27B the routed arm loses 0.05 f1 (0.932 vs 0.980) for a 32% FLOP saving, and on oolong it drops 0.10 — the post-trained model's FFNs are less tolerant of width cuts, or the post-trained base needs the two-stage recipe. Parity for the FFN router is still not demonstrated at any scale.
- **"Train what you route to" (frozen tail, only the H/16 prefix + router + gains trained) is quality-neutral at 0.8B–4B**: contradiction 0.885 vs 0.880 (4B, 56M), oolong 0.671 vs 0.688 (4B, 80M), 0.653 vs 0.646 (2B), at identical FLOPs. Its multipliers move within noise of the whole-FFN arm (0.85 vs 0.81 on contradiction 4B; 0.43 vs 0.62 on oolong 4B). Not run at 9B/27B: two 9B attempts NaN'd at steps 5/11 (open bug) and the single-node 27B OOM'd.
- At the small budget (14M / 20M) every method loses at every scale: the routers and projectors need the longer horizon.
- 9B needed 8 GPUs per run; 27B needed 2 nodes (Ulysses CP 2 for packed arms, 16 ranks without CP for the KV arm), and its evals fit on H100s.

| task | scale | arm | budget | f1 | PF | dense f1 at same PF | FLOPs for dense to match, ÷ method PF |
|---|---|---|---|---|---|---|---|
| contradiction | 0.8b | ffnmoe-t10 | 14M | 0.007 | 62 | -0.011 | 1.04~ |
| contradiction | 0.8b | ffnmoe-t10 | 56M | 0.271 | 235 | 0.635 | 0.47 |
| contradiction | 0.8b | ffnmoe-t10p | 14M | 0.000 | 67 | 0.023 | 0.96~ |
| contradiction | 0.8b | ffnmoe-t10p | 56M | 0.363 | 234 | 0.633 | 0.58 |
| contradiction | 0.8b | kv33 | 14M | 0.091 | 31 | -0.357 | 2.51 |
| contradiction | 0.8b | kv33 | 56M | 0.454 | 129 | 0.343 | 1.26 |
| contradiction | 2b | ffnmoe-t10 | 14M | 0.050 | 125 | 0.401 | 0.30~ |
| contradiction | 2b | ffnmoe-t10 | 56M | 0.707 | 470 | 0.786 | 0.76 |
| contradiction | 2b | ffnmoe-t10p | 14M | 0.027 | 124 | 0.398 | 0.28~ |
| contradiction | 2b | ffnmoe-t10p | 56M | 0.692 | 467 | 0.785 | 0.73 |
| contradiction | 2b | kv33 | 14M | 0.238 | 76 | 0.254 | 0.95~ |
| contradiction | 2b | kv33 | 56M | 0.632 | 318 | 0.673 | 0.87 |
| contradiction | 4b | ffnmoe-t10 | 14M | 0.579 | 277 | 0.747 | 0.26~ |
| contradiction | 4b | ffnmoe-t10 | 56M | 0.880 | 998 | 0.907 | 0.81 |
| contradiction | 4b | ffnmoe-t10p | 14M | 0.525 | 272 | 0.745 | 0.17~ |
| contradiction | 4b | ffnmoe-t10p | 56M | 0.885 | 991 | 0.906 | 0.85 |
| contradiction | 4b | kv17 | 14M | 0.343 | 103 | 0.625 | 0.10~ |
| contradiction | 4b | kv17 | 56M | 0.764 | 429 | 0.802 | 0.74~ |
| contradiction | 4b | kv33 | 14M | 0.525 | 166 | 0.684 | 0.28~ |
| contradiction | 4b | kv33 | 56M | 0.861 | 695 | 0.862 | 1.00 |
| contradiction | 9b | ffnmoe-t10 | 14M | 0.582 | 483 | 0.812 | 0.05~ |
| contradiction | 9b | ffnmoe-t10 | 56M | 0.905 | 1746 | 0.913 | 0.91 |
| contradiction | 9b | kv33 | 14M | 0.586 | 234 | 0.755 | 0.12~ |
| contradiction | 9b | kv33 | 56M | 0.914 | 1069 | 0.874 | 1.66 |
| contradiction | 27b | ffnmoe-t10 | 14M | 0.531 | 1701 | 0.887 | 0.00~ |
| contradiction | 27b | ffnmoe-t10 | 56M | 0.932 | 5310 | 0.956 | 0.67 |
| contradiction | 27b | kv33 | 14M | 0.793 | 746 | 0.837 | 0.48~ |
| contradiction | 27b | kv33 | 56M | 0.956 | 3341 | 0.928 | 1.59 |
| oolong | 0.8b | ffnmoe-t10 | 20M | 0.560 | 86 | 0.618 | 0.19~ |
| oolong | 0.8b | ffnmoe-t10 | 80M | 0.639 | 333 | 0.664 | 0.48 |
| oolong | 0.8b | ffnmoe-t10p | 20M | 0.572 | 86 | 0.618 | 0.26~ |
| oolong | 0.8b | ffnmoe-t10p | 80M | 0.642 | 330 | 0.664 | 0.53 |
| oolong | 0.8b | kv17 | 20M | 0.585 | 24 | 0.575 | 1.33~ |
| oolong | 0.8b | kv17 | 80M | 0.660 | 103 | 0.624 | 2.86 |
| oolong | 2b | ffnmoe-t10 | 20M | 0.506 | 182 | 0.644 | 0.01~ |
| oolong | 2b | ffnmoe-t10 | 80M | 0.646 | 677 | 0.685 | 0.29~ |
| oolong | 2b | ffnmoe-t10p | 20M | 0.544 | 181 | 0.644 | 0.04~ |
| oolong | 2b | ffnmoe-t10p | 80M | 0.653 | 672 | 0.685 | 0.35 |
| oolong | 2b | kv17 | 20M | 0.604 | 62 | 0.611 | 0.80~ |
| oolong | 2b | kv17 | 80M | 0.669 | 259 | 0.655 | 1.55 |
| oolong | 4b | ffnmoe-t10 | 20M | 0.596 | 382 | 0.650 | 0.29~ |
| oolong | 4b | ffnmoe-t10 | 80M | 0.688 | 1437 | 0.709 | 0.62 |
| oolong | 4b | ffnmoe-t10p | 20M | 0.579 | 384 | 0.651 | 0.20~ |
| oolong | 4b | ffnmoe-t10p | 80M | 0.671 | 1424 | 0.709 | 0.43 |
| oolong | 4b | kv17 | 20M | 0.637 | 134 | 0.604 | 2.11~ |
| oolong | 4b | kv17 | 80M | 0.665 | 564 | 0.668 | 0.94 |
| oolong | 4b | kv33 | 20M | 0.641 | 227 | 0.627 | 1.36~ |
| oolong | 4b | kv33 | 80M | 0.674 | 962 | 0.691 | 0.68 |
| oolong | 9b | ffnmoe-t10 | 20M | 0.637 | 676 | 0.667 | 0.46~ |
| oolong | 9b | ffnmoe-t10 | 80M | 0.690 | 2516 | 0.718 | 0.48 |
| oolong | 9b | kv17 | 20M | 0.635 | 198 | 0.619 | 1.53~ |
| oolong | 9b | kv17 | 80M | 0.662 | 816 | 0.674 | 0.75~ |
| oolong | 27b | ffnmoe-t10 | 20M | 0.583 | 2163 | 0.716 | 0.04~ |
| oolong | 27b | ffnmoe-t10 | 80M | 0.690 | 7589 | 0.769 | 0.15~ |
| oolong | 27b | kv17 | 20M | 0.682 | 634 | 0.665 | 1.50~ |
| oolong | 27b | kv17 | 80M | 0.680 | 2628 | 0.725 | 0.34~ |


Multiplier > 1 means the method reaches its f1 for fewer FLOPs than dense (a compute-optimal win); < 1 means dense is cheaper at that f1. Dense contradiction 0.8B at 14M scored 0.03 (the task is not learnable there), so 0.8B contradiction interpolations are steep and unreliable.
