# Midtraining Validation Results

Generated: 2026-07-07 17:41 UTC

Interpretation: lower is better for CE loss, PPL, Z loss, router Z loss, and load-balancing loss; higher is better for MFU/TPS. Accuracy-style validation metrics are higher-is-better when present.

Note: the current 275M midtraining grid was launched with `--eval-task-set=fast`, but `midtraining_ladder.py` sets `ladder_evals=False`, so no `eval/*` validation metrics are present in W&B for these runs yet. The tables below therefore show final or latest training/progress metrics and will include validation metrics automatically once future midtraining runs log them.

Settings: 100B midtraining tokens, sequence length 8192, global batch seq 128 (1,048,576 tokens), 1 node, 4 GPUs, EP1, microbatch 8, fresh optimizer state, 2000-step warmup then constant LR.

| source | finished | best finished LR | best CE | best PPL | still running |
| --- | --- | --- | --- | --- | --- |
| Cx1 | 3/4 | 2e-4 | 1.6064 | 4.9848 | 1.6e-3 |
| Cx8 | 3/4 | 2e-4 | 1.5642 | 4.7789 | 1.6e-3 |

## Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.6064 | 4.9848 | 0.00083 | 0.00013 | 0.11042 | 29.30 | 363861.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cm8ww646) [Beaker](https://beaker.org/ex/01KWWM1043JEC9MC3PV7PXQ745) |
| 4e-4 | finished | 100.00B | 1.6162 | 5.0340 | 0.00095 | 0.00020 | 0.11038 | 29.36 | 364560.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r6ts032g) [Beaker](https://beaker.org/ex/01KWWM1AVQXMJ3JBJQ1W2G8YAV) |
| 8e-4 | finished | 100.00B | 1.6446 | 5.1791 | 0.00110 | 0.00027 | 0.11048 | 29.01 | 360250.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lfydkxv4) [Beaker](https://beaker.org/ex/01KWWM1N0EQDMCFER90NVP9QW0) |
| 1.6e-3 | running | 91.27B | 1.7903 | 5.9914 | 0.00120 | 0.00038 | 0.11058 | 28.75 | 356934.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w3vof8b9) [Beaker](https://beaker.org/ex/01KWWM1ZXN5R5XWK00GH0WA36G) |

## Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.5642 | 4.7789 | 0.00084 | 0.00015 | 0.11033 | 29.36 | 364568.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4amcsbx6) [Beaker](https://beaker.org/ex/01KWWM10ANMMW2YTNN6RKJBGE7) |
| 4e-4 | finished | 100.00B | 1.5924 | 4.9157 | 0.00101 | 0.00021 | 0.11038 | 29.35 | 364491.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8jdqtmgg) [Beaker](https://beaker.org/ex/01KWWM1AK89SKDA1KGCX5D8SMM) |
| 8e-4 | finished | 100.00B | 1.6283 | 5.0952 | 0.00111 | 0.00028 | 0.11047 | 29.19 | 362440.4 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/drm1ceit) [Beaker](https://beaker.org/ex/01KWWM1P9TXRSMDV5DH9EF4KXM) |
| 1.6e-3 | running | 90.66B | 1.7400 | 5.6975 | 0.00122 | 0.00037 | 0.11068 | 28.95 | 359487.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/edljug2e) [Beaker](https://beaker.org/ex/01KWWM213KMG47KADPK5Q67GJP) |

## Validation Metrics

No `eval/*` validation metrics were found for this midtraining grid.
