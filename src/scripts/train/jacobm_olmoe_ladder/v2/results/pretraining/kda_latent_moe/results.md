# KDA LatentMoE paper-matched pretraining

Generated: `2026-08-01T18:57:50.125920+00:00`

Metric: final `250M`-token mean training CE. The best-of plot uses observed finished points only after a curve is bracketed.

| Variant | Cx | Coverage | Curve | Observed best | Predicted fit (visual only) |
|---|---:|---:|---|---|---|
| LatentMoE L=2 (paper-matched) | Cx1 | 4/4 | bracketed | 2.659602 @ 0.0016 | 2.658280 @ 0.0013 |
| LatentMoE L=2 (paper-matched) | Cx2 | 4/4 | bracketed | 2.541918 @ 0.0016 | 2.541502 @ 0.0014 |
| LatentMoE L=2 (paper-matched) | Cx4 | 4/4 | bracketed | 2.443058 @ 0.0016 | 2.442457 @ 0.0014 |
| LatentMoE L=2 (paper-matched) | Cx8 | 4/4 | bracketed | 2.362515 @ 0.0016 | 2.360705 @ 0.0012 |
| LatentMoE L=4 (1,000 experts/top-32) | Cx1 | 4/4 | bracketed | 2.663658 @ 0.0016 | 2.663635 @ 0.0016 |
| LatentMoE L=4 (1,000 experts/top-32) | Cx2 | 4/4 | bracketed | 2.555501 @ 0.0016 | 2.555500 @ 0.0016 |
| LatentMoE L=4 (1,000 experts/top-32) | Cx4 | 4/4 | bracketed | 2.462333 @ 0.0016 | 2.461737 @ 0.0014 |
| LatentMoE L=4 (1,000 experts/top-32) | Cx8 | 4/4 | bracketed | 2.383977 @ 0.0008 | 2.381851 @ 0.0011 |

## Larger-size transferred-LR results

| Variant | Model | Cx | LR | Final-250M CE | Tokens | W&B |
|---|---|---:|---:|---:|---:|---|
| LatentMoE L=2 (paper-matched) | 480m | Cx1 | 0.0012 | 2.463192 | 8.654B | [meyce8po](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/meyce8po) |
| LatentMoE L=2 (paper-matched) | 480m | Cx2 | 0.0009 | 2.361694 | 17.307B | [quugqq1o](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/quugqq1o) |
| LatentMoE L=2 (paper-matched) | 480m | Cx4 | 0.0008 | 2.270144 | 34.615B | [feop0y0x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/feop0y0x) |
| LatentMoE L=2 (paper-matched) | 480m | Cx8 | 0.0008 | 2.203397 | 69.230B | [2m4xllen](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2m4xllen) |
| LatentMoE L=2 (paper-matched) | 810m | Cx1 | 0.0006 | 2.320513 | 15.097B | [3pf9w1p5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3pf9w1p5) |
| LatentMoE L=2 (paper-matched) | 810m | Cx2 | 0.00056 | 2.223199 | 30.193B | [uet3vyje](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uet3vyje) |
| LatentMoE L=2 (paper-matched) | 810m | Cx4 | 0.0004 | 2.139766 | 60.386B | [mao1l8wg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mao1l8wg) |
| LatentMoE L=4 (1,000 experts/top-32) | 810m | Cx4 | 0.0004 | 2.152349 | 60.359B | [1fv6x7z0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1fv6x7z0) |
| LatentMoE L=2 (paper-matched) | 810m | Cx8 | 0.0004 | 2.074273 | 120.773B | [cxwvqg3r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cxwvqg3r) |
| LatentMoE L=2 (paper-matched) | 1p2b | Cx1 | 0.0004 | 2.213475 | 23.208B | [gh7zj1n8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gh7zj1n8) |
| LatentMoE L=2 (paper-matched) | 1p2b | Cx2 | 0.0006 | 2.131599 | 46.415B | [5n8ngak5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5n8ngak5) |
| LatentMoE L=2 (paper-matched) | 1p2b | Cx4 | 0.0003 | 2.051686 | 92.830B | [upmkyxzv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/upmkyxzv) |

Uninitialized planned W&B runs: `0`.
