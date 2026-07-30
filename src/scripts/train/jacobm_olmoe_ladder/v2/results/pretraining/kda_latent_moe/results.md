# KDA LatentMoE paper-matched pretraining

Generated: `2026-07-30T04:27:50.838932+00:00`

Metric: final `250M`-token mean training CE. The best-of plot uses observed finished points only after a curve is bracketed.

| Variant | Cx | Coverage | Curve | Observed best | Predicted fit (visual only) |
|---|---:|---:|---|---|---|
| LatentMoE L=2 (paper-matched) | Cx1 | 4/4 | bracketed | 2.659602 @ 0.0016 | 2.658280 @ 0.0013 |
| LatentMoE L=2 (paper-matched) | Cx2 | 4/4 | bracketed | 2.541918 @ 0.0016 | 2.541502 @ 0.0014 |
| LatentMoE L=2 (paper-matched) | Cx4 | 4/4 | bracketed | 2.443058 @ 0.0016 | 2.442457 @ 0.0014 |
| LatentMoE L=2 (paper-matched) | Cx8 | 0/4 | — | — | — |
| LatentMoE L=4 (1,000 experts/top-32) | Cx1 | 4/4 | bracketed | 2.663658 @ 0.0016 | 2.663635 @ 0.0016 |
| LatentMoE L=4 (1,000 experts/top-32) | Cx2 | 4/4 | bracketed | 2.555501 @ 0.0016 | 2.555500 @ 0.0016 |
| LatentMoE L=4 (1,000 experts/top-32) | Cx4 | 0/4 | — | — | — |
| LatentMoE L=4 (1,000 experts/top-32) | Cx8 | 0/4 | — | — | — |

## Larger-size transferred-LR results

| Model | Cx | LR | Final-250M CE | Tokens | W&B |
|---|---:|---:|---:|---:|---|
| 480m | Cx1 | 0.0012 | 2.463192 | 8.654B | [meyce8po](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/meyce8po) |

Uninitialized planned W&B runs: `0`.
