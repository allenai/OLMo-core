# Midtraining Validation Results

Generated: 2026-07-13 20:20 UTC

Interpretation: lower is better for CE loss, PPL, Z loss, router Z loss, and load-balancing loss; higher is better for MFU/TPS. Accuracy-style validation metrics are higher-is-better when present.

Selection rule: use only `eval/*` validation metrics for midtraining checkpoint/LR selection. Training loss on the midtraining mixture is shown only as run-health metadata and must not be used to choose LRs.

Backfill note: the first 275M grid did not run in-loop evals during training, so final-checkpoint eval-only backfills are required. Once those eval jobs finish and `copy_eval_backfills_to_wandb.py` copies their metrics back, this table will populate the `eval/*` section.

Settings: 100B midtraining tokens, sequence length 8192, global batch seq 128 (1,048,576 tokens), 1 node, 4 GPUs, EP1, microbatch 8, fresh optimizer state, 2000-step warmup then constant LR.

Exception: `275M integration wide top16 Cx4` is the high-active diagnostic trained on the same token budget as the 275M wide Cx8 pretraining run. It uses 1 node, 8 GPUs, EP1, microbatch 4, and the same global batch seq 128 to fit the larger active parameter count.

| source | training finished | eval metrics present | LRs with evals | still running |
| --- | --- | --- | --- | --- |
| 275M baseline Cx1 | 4/4 | 4/4 | 2e-4, 4e-4, 8e-4, 1.6e-3 |  |
| 275M integration deep Cx1 | 1/1 | 1/1 | 2e-4 |  |
| 275M integration wide Cx1 | 1/1 | 1/1 | 2e-4 |  |
| 275M baseline Cx2 | 1/1 | 1/1 | 1.8e-4 |  |
| 275M integration deep Cx2 | 1/1 | 1/1 | 1.8e-4 |  |
| 275M integration wide Cx2 | 1/1 | 1/1 | 1.8e-4 |  |
| 275M baseline Cx4 | 1/1 | 1/1 | 1.5e-4 |  |
| 275M integration deep Cx4 | 1/1 | 1/1 | 1.5e-4 |  |
| 275M integration wide Cx4 | 1/1 | 1/1 | 1.5e-4 |  |
| 275M integration wide top16 Cx4 | 1/1 | 1/1 | 8e-5 |  |
| 275M baseline Cx8 | 4/4 | 4/4 | 2e-4, 4e-4, 8e-4, 1.6e-3 |  |
| 275M integration deep Cx8 | 1/1 | 1/1 | 1.6e-4 |  |
| 275M integration wide Cx8 | 1/1 | 1/1 | 1.6e-4 |  |
| 480M baseline Cx1 | 1/1 | 1/1 | 1.2e-4 |  |
| 480M baseline Cx8 | 1/1 | 1/1 | 8e-5 |  |
| 480M integration deep Cx8 | 1/1 | 1/1 | 8e-5 |  |
| 480M integration wide Cx8 | 1/1 | 1/1 | 8e-5 |  |
| 810M baseline Cx1 | 1/1 | 1/1 | 6e-5 |  |
| 810M baseline Cx8 | 1/1 | 1/1 | 4e-5 |  |
| 810M integration deep Cx8 | 1/1 | 1/1 | 4e-5 |  |
| 810M integration wide Cx8 | 1/1 | 1/1 | 4e-5 |  |
| 1.2B baseline Cx1 | 1/1 | 1/1 | 4e-5 |  |
| 1.2B baseline Cx8 | 1/1 | 1/1 | 4e-5 |  |
| 1.2B integration deep Cx8 | 1/1 | 1/1 | 4e-5 |  |
| 1.2B integration wide Cx8 | 1/1 | 1/1 | 4e-5 |  |

## Eval Win Summary

Wins are computed separately within each source checkpoint group. Raw counts include every logged eval metric. De-duplicated counts collapse `v2`/non-`v2` repeats for the same task and score family, preferring `v2` when both are present. Ties, if any, count for every tied LR.

### 275M baseline Cx1 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 2e-4 | 66/178 | 44/100 |
| 4e-4 | 52/178 | 26/100 |
| 8e-4 | 32/178 | 16/100 |
| 1.6e-3 | 34/178 | 17/100 |

| category | dedup metrics | 2e-4 | 4e-4 | 8e-4 | 1.6e-3 |
| --- | --- | --- | --- | --- | --- |
| arc | 12 | 4 | 5 | 1 | 2 |
| basic_skills | 30 | 8 | 6 | 11 | 6 |
| codex | 2 | 2 | 0 | 0 | 0 |
| copycolors | 5 | 3 | 1 | 2 | 1 |
| hellaswag | 1 | 0 | 1 | 0 | 0 |
| lm | 22 | 22 | 0 | 0 | 0 |
| minerva | 1 | 0 | 1 | 0 | 0 |
| mmlu | 24 | 4 | 10 | 2 | 8 |
| mt | 3 | 1 | 2 | 0 | 0 |

### 275M integration deep Cx1 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 2e-4 | 178/178 | 100/100 |

| category | dedup metrics | 2e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration wide Cx1 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 2e-4 | 178/178 | 100/100 |

| category | dedup metrics | 2e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M baseline Cx2 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.8e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.8e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration deep Cx2 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.8e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.8e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration wide Cx2 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.8e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.8e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M baseline Cx4 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.5e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.5e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration deep Cx4 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.5e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.5e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration wide Cx4 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.5e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.5e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration wide top16 Cx4 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 8e-5 | 178/178 | 100/100 |

| category | dedup metrics | 8e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M baseline Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 2e-4 | 74/178 | 47/100 |
| 4e-4 | 35/178 | 18/100 |
| 8e-4 | 30/178 | 15/100 |
| 1.6e-3 | 45/178 | 23/100 |

| category | dedup metrics | 2e-4 | 4e-4 | 8e-4 | 1.6e-3 |
| --- | --- | --- | --- | --- | --- |
| arc | 12 | 5 | 0 | 5 | 2 |
| basic_skills | 30 | 5 | 7 | 9 | 10 |
| codex | 2 | 2 | 0 | 0 | 0 |
| copycolors | 5 | 0 | 3 | 1 | 3 |
| hellaswag | 1 | 1 | 0 | 0 | 0 |
| lm | 22 | 22 | 0 | 0 | 0 |
| minerva | 1 | 1 | 0 | 0 | 0 |
| mmlu | 24 | 9 | 7 | 0 | 8 |
| mt | 3 | 2 | 1 | 0 | 0 |

### 275M integration deep Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.6e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.6e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 275M integration wide Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.6e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.6e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 480M baseline Cx1 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 1.2e-4 | 178/178 | 100/100 |

| category | dedup metrics | 1.2e-4 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 480M baseline Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 8e-5 | 178/178 | 100/100 |

| category | dedup metrics | 8e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 480M integration deep Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 8e-5 | 178/178 | 100/100 |

| category | dedup metrics | 8e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 480M integration wide Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 8e-5 | 178/178 | 100/100 |

| category | dedup metrics | 8e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 810M baseline Cx1 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 6e-5 | 178/178 | 100/100 |

| category | dedup metrics | 6e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 810M baseline Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 810M integration deep Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 810M integration wide Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 1.2B baseline Cx1 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 1.2B baseline Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 1.2B integration deep Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

### 1.2B integration wide Cx8 Win Counts

| LR | raw wins / 178 | dedup wins / 100 |
| --- | --- | --- |
| 4e-5 | 178/178 | 100/100 |

| category | dedup metrics | 4e-5 |
| --- | --- | --- |
| arc | 12 | 12 |
| basic_skills | 30 | 30 |
| codex | 2 | 2 |
| copycolors | 5 | 5 |
| hellaswag | 1 | 1 |
| lm | 22 | 22 |
| minerva | 1 | 1 |
| mmlu | 24 | 24 |
| mt | 3 | 3 |

## 275M baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.6064 | 4.9848 | 0.00083 | 0.00013 | 0.11042 | 29.30 | 363861.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cm8ww646) [Beaker](https://beaker.org/ex/01KWWM1043JEC9MC3PV7PXQ745) |
| 4e-4 | finished | 100.00B | 1.6162 | 5.0340 | 0.00095 | 0.00020 | 0.11038 | 29.36 | 364560.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r6ts032g) [Beaker](https://beaker.org/ex/01KWWM1AVQXMJ3JBJQ1W2G8YAV) |
| 8e-4 | finished | 100.00B | 1.6446 | 5.1791 | 0.00110 | 0.00027 | 0.11048 | 29.01 | 360250.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lfydkxv4) [Beaker](https://beaker.org/ex/01KWWM1N0EQDMCFER90NVP9QW0) |
| 1.6e-3 | finished | 100.00B | 1.6966 | 5.4554 | 0.00121 | 0.00036 | 0.11070 | 29.20 | 362582.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w3vof8b9) [Beaker](https://beaker.org/ex/01KWWM1ZXN5R5XWK00GH0WA36G) |

## 275M integration deep Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.5416 | 4.6722 | 0.00087 | 0.00075 | 0.13036 | 24.63 | 292554.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/aklwxdeh) [Beaker](https://beaker.org/ex/01KX1X0QVHZ7X0HSF0YNJWEBBH) |

## 275M integration wide Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.5402 | 4.6657 | 0.00084 | 0.00073 | 0.11032 | 25.66 | 316761.3 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o4ufyagk) [Beaker](https://beaker.org/ex/01KX1WZ551YCZYTHXRQEDX1WK1) |

## 275M baseline Cx2 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.8e-4 | finished | 100.00B | 1.5903 | 4.9054 | 0.00089 | 0.00013 | 0.11041 | 29.14 | 361891.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kpkhplj7) [Beaker](https://beaker.org/ex/01KWZ8T9BZ3B869VZ878FNNQ8T) |

## 275M integration deep Cx2 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.8e-4 | finished | 100.00B | 1.5247 | 4.5939 | 0.00087 | 0.00070 | 0.13028 | 24.62 | 292505.1 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jnp3nrqb) [Beaker](https://beaker.org/ex/01KX1X14K49GJN4KS9QQ3JEQ1E) |

## 275M integration wide Cx2 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.8e-4 | finished | 100.00B | 1.5262 | 4.6008 | 0.00088 | 0.00067 | 0.11012 | 25.59 | 315907.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gindbthc) [Beaker](https://beaker.org/ex/01KX1WZGKE5K5GF4SZ98G7SVJD) |

## 275M baseline Cx4 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5e-4 | finished | 100.00B | 1.5736 | 4.8240 | 0.00087 | 0.00013 | 0.11039 | 29.14 | 361830.2 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u508l6x8) [Beaker](https://beaker.org/ex/01KWZ8T9DZN8Y4VD63AP1AN387) |

## 275M integration deep Cx4 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5e-4 | finished | 100.00B | 1.5030 | 4.4952 | 0.00083 | 0.00064 | 0.13009 | 24.77 | 294265.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eojex5a0) [Beaker](https://beaker.org/ex/01KX1X1GPC3N51C7E8AW2YXN63) |

## 275M integration wide Cx4 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5e-4 | finished | 100.00B | 1.5082 | 4.5187 | 0.00081 | 0.00061 | 0.11020 | 25.32 | 312596.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vzpxeard) [Beaker](https://beaker.org/ex/01KX1WZXCBX8VFNFFZK2DT50JB) |

## 275M integration wide top16 Cx4 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8e-5 | finished | 100.00B | 1.4481 | 4.2552 | 0.00077 | 0.00057 | 0.11095 | 23.20 | 227983.8 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kmxhtpxp) [Beaker](https://beaker.org/ex/01KXBH2HBN9W4M46H33CRGHVJS) |

## 275M baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.5642 | 4.7789 | 0.00084 | 0.00015 | 0.11033 | 29.36 | 364568.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4amcsbx6) [Beaker](https://beaker.org/ex/01KWWM10ANMMW2YTNN6RKJBGE7) |
| 4e-4 | finished | 100.00B | 1.5924 | 4.9157 | 0.00101 | 0.00021 | 0.11038 | 29.35 | 364491.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8jdqtmgg) [Beaker](https://beaker.org/ex/01KWWM1AK89SKDA1KGCX5D8SMM) |
| 8e-4 | finished | 100.00B | 1.6283 | 5.0952 | 0.00111 | 0.00028 | 0.11047 | 29.19 | 362440.4 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/drm1ceit) [Beaker](https://beaker.org/ex/01KWWM1P9TXRSMDV5DH9EF4KXM) |
| 1.6e-3 | finished | 100.00B | 1.6843 | 5.3887 | 0.00129 | 0.00037 | 0.11067 | 29.06 | 360889.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/edljug2e) [Beaker](https://beaker.org/ex/01KWWM213KMG47KADPK5Q67GJP) |

## 275M integration deep Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.6e-4 | finished | 100.00B | 1.4901 | 4.4376 | 0.00083 | 0.00065 | 0.13012 | 24.77 | 294227.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/whozy6wn) [Beaker](https://beaker.org/ex/01KX1X203AABJ29GHKST48J5E9) |

## 275M integration wide Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.6e-4 | finished | 100.00B | 1.4948 | 4.4583 | 0.00086 | 0.00062 | 0.11007 | 25.35 | 312996.3 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fgl0o34z) [Beaker](https://beaker.org/ex/01KX1X0AJAMCEE5MSXBP33PHNG) |

## 480M baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.2e-4 | finished | 100.00B | 1.5472 | 4.6984 | 0.00094 | 0.00009 | 0.15040 | 32.02 | 233360.8 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ro38bhjl) [Beaker](https://beaker.org/ex/01KWZARWH7XAS4MD2238VRKP0Y) |

## 480M baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8e-5 | finished | 100.00B | 1.4834 | 4.4081 | 0.00101 | 0.00008 | 0.15011 | 31.94 | 232761.3 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6pd01ptm) [Beaker](https://beaker.org/ex/01KWZARZ2T7FZDH42Q2VS92WXN) |

## 480M integration deep Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8e-5 | finished | 100.00B | 1.4206 | 4.1394 | 0.00097 | 0.00034 | 0.18980 | 26.81 | 181384.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2qobflhw) [Beaker](https://beaker.org/ex/01KX6K41XG75V1DQ9NWZB988GQ) |

## 480M integration wide Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8e-5 | finished | 100.00B | 1.4293 | 4.1758 | 0.00092 | 0.00034 | 0.14989 | 27.89 | 202028.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/in1ztv0o) [Beaker](https://beaker.org/ex/01KX6K2NTZZRVE98HCQWE2DR64) |

## 810M baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6e-5 | finished | 100.00B | 1.4171 | 4.1251 | 0.00060 | 0.00007 | 0.19036 | 31.43 | 125162.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7ylgav9r) [Beaker](https://beaker.org/ex/01KWZAT4PF87PYA96JT7ZXSQKX) |

## 810M baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.3399 | 3.8186 | 0.00068 | 0.00007 | 0.19004 | 31.51 | 125499.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zeia5b0a) [Beaker](https://beaker.org/ex/01KWZAT4PQ0NWD20VS0XT12ZF5) |

## 810M integration deep Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.2851 | 3.6149 | 0.00066 | 0.00022 | 0.22968 | 25.75 | 98221.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rge9kicc) [Beaker](https://beaker.org/ex/01KX6K4GKDHQS05727Q9DH0NF4) |

## 810M integration wide Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.2855 | 3.6166 | 0.00071 | 0.00022 | 0.18969 | 26.23 | 103922.4 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sq79ify2) [Beaker](https://beaker.org/ex/01KX6K330J3Z5CSDY518BGJ05B) |

## 1.2B baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.3877 | 4.0056 | 0.00082 | 0.00007 | 0.21038 | 34.41 | 90037.1 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5utsk31j) [Beaker](https://beaker.org/ex/01KWZAVZFS1FMR59ASRVH7VD4X) |

## 1.2B baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.2874 | 3.6235 | 0.00083 | 0.00007 | 0.21003 | 34.37 | 89934.1 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xgswsn5t) [Beaker](https://beaker.org/ex/01KWZAV9AGDSD3PTS5RAM7G5N2) |

## 1.2B integration deep Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.2682 | 3.5545 | 0.00078 | 0.00043 | 0.25971 | 23.36 | 57427.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k25pw5nl) [Beaker](https://beaker.org/ex/01KX703FE8T96HJM4EFG90KHVF) |

## 1.2B integration wide Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.2618 | 3.5317 | 0.00075 | 0.00038 | 0.20970 | 25.10 | 65370.1 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5stly1t0) [Beaker](https://beaker.org/ex/01KX7032K1GXDKQR2D9FAA0CBF) |

## Validation Metrics

| metric | direction | 275M baseline Cx1 2e-4 | 275M baseline Cx1 4e-4 | 275M baseline Cx1 8e-4 | 275M baseline Cx1 1.6e-3 | 275M integration deep Cx1 2e-4 | 275M integration wide Cx1 2e-4 | 275M baseline Cx2 1.8e-4 | 275M integration deep Cx2 1.8e-4 | 275M integration wide Cx2 1.8e-4 | 275M baseline Cx4 1.5e-4 | 275M integration deep Cx4 1.5e-4 | 275M integration wide Cx4 1.5e-4 | 275M integration wide top16 Cx4 8e-5 | 275M baseline Cx8 2e-4 | 275M baseline Cx8 4e-4 | 275M baseline Cx8 8e-4 | 275M baseline Cx8 1.6e-3 | 275M integration deep Cx8 1.6e-4 | 275M integration wide Cx8 1.6e-4 | 480M baseline Cx1 1.2e-4 | 480M baseline Cx8 8e-5 | 480M integration deep Cx8 8e-5 | 480M integration wide Cx8 8e-5 | 810M baseline Cx1 6e-5 | 810M baseline Cx8 4e-5 | 810M integration deep Cx8 4e-5 | 810M integration wide Cx8 4e-5 | 1.2B baseline Cx1 4e-5 | 1.2B baseline Cx8 4e-5 | 1.2B integration deep Cx8 4e-5 | 1.2B integration wide Cx8 4e-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | **0.90040** | 0.91077 | 0.92087 | 0.94430 | **0.89796** | **0.86775** | **0.88624** | **0.85221** | **0.87512** | **0.86619** | **0.83814** | **0.82362** | **0.81067** | **0.86890** | 0.88767 | 0.91593 | 0.90970 | **0.84198** | **0.81024** | **0.82377** | **0.79155** | **0.74177** | **0.75515** | **0.78635** | **0.76221** | **0.70931** | **0.70934** | **0.76853** | **0.69597** | **0.68661** | **0.70270** |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | **0.98279** | 0.99730 | 1.0085 | 1.0332 | **0.98215** | **0.94654** | **0.96654** | **0.93061** | **0.95523** | **0.94651** | **0.91351** | **0.89859** | **0.88508** | **0.94960** | 0.96898 | 0.99863 | 0.99320 | **0.91554** | **0.88084** | **0.89998** | **0.86322** | **0.80937** | **0.82514** | **0.85740** | **0.83583** | **0.77414** | **0.77387** | **0.84017** | **0.76088** | **0.74593** | **0.76401** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0399 | **1.0123** | 1.0441 | 1.0601 | **0.97895** | **1.0031** | **1.0012** | **1.0079** | **0.96135** | **1.0415** | **1.0431** | **0.96765** | **0.95556** | 1.0584 | 1.0015 | **0.99425** | 1.0389 | **0.97235** | **0.96545** | **0.93554** | **0.84440** | **0.82961** | **0.82441** | **0.95086** | **0.76888** | **0.68461** | **0.74396** | **0.83973** | **0.59855** | **0.58759** | **0.53589** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0798 | **2.0246** | 2.0882 | 2.1202 | **1.9579** | **2.0063** | **2.0023** | **2.0158** | **1.9227** | **2.0829** | **2.0862** | **1.9353** | **1.9111** | 2.1167 | 2.0029 | **1.9885** | 2.0778 | **1.9447** | **1.9309** | **1.8711** | **1.6888** | **1.6592** | **1.6488** | **1.9017** | **1.5378** | **1.3692** | **1.4879** | **1.6795** | **1.1971** | **1.1752** | **1.0718** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.72079 | **0.70174** | 0.72373 | 0.73480 | **0.67856** | **0.69534** | **0.69393** | **0.69866** | **0.66639** | **0.72185** | **0.72294** | **0.67076** | **0.66232** | 0.73360 | 0.69418 | **0.68922** | 0.72010 | **0.67399** | **0.66918** | **0.64843** | **0.58530** | **0.57508** | **0.57139** | **0.65909** | **0.53299** | **0.47450** | **0.51566** | **0.58207** | **0.41489** | **0.40729** | **0.37140** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4416 | **1.4035** | 1.4475 | 1.4696 | **1.3571** | **1.3907** | **1.3879** | **1.3973** | **1.3328** | **1.4437** | **1.4459** | **1.3415** | **1.3246** | 1.4672 | 1.3884 | **1.3784** | 1.4402 | **1.3480** | **1.3384** | **1.2969** | **1.1706** | **1.1502** | **1.1428** | **1.3182** | **1.0660** | **0.94900** | **1.0313** | **1.1641** | **0.82977** | **0.81459** | **0.74280** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | **0.30290** | 0.28584 | 0.26962 | 0.29608 | **0.35495** | **0.36945** | **0.34215** | **0.34983** | **0.36092** | **0.34215** | **0.32765** | **0.40444** | **0.41468** | **0.31997** | 0.30461 | 0.29181 | 0.30717 | **0.35239** | **0.37287** | **0.39164** | **0.46758** | **0.48805** | **0.49659** | **0.40870** | **0.54608** | **0.60324** | **0.54266** | **0.46502** | **0.65614** | **0.67321** | **0.69625** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | **0.30290** | 0.28584 | 0.26962 | 0.29608 | **0.35495** | **0.36945** | **0.34215** | **0.34983** | **0.36092** | **0.34215** | **0.32765** | **0.40444** | **0.41468** | **0.31997** | 0.30461 | 0.29181 | 0.30717 | **0.35239** | **0.37287** | **0.39164** | **0.46758** | **0.48805** | **0.49659** | **0.40870** | **0.54608** | **0.60324** | **0.54266** | **0.46502** | **0.65614** | **0.67321** | **0.69625** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4398 | -1.4002 | -1.4455 | **-1.4664** | **-1.3556** | **-1.3888** | **-1.3857** | **-1.3945** | **-1.3303** | **-1.4401** | **-1.4437** | **-1.3365** | **-1.3232** | **-1.4652** | -1.3858 | -1.3746 | -1.4380 | **-1.3458** | **-1.3368** | **-1.2951** | **-1.1683** | **-1.1483** | **-1.1410** | **-1.3170** | **-1.0647** | **-0.94799** | **-1.0300** | **-1.1631** | **-0.82821** | **-0.81372** | **-0.74199** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4398 | -1.4002 | -1.4455 | **-1.4664** | **-1.3556** | **-1.3888** | **-1.3857** | **-1.3945** | **-1.3303** | **-1.4401** | **-1.4437** | **-1.3365** | **-1.3232** | **-1.4652** | -1.3858 | -1.3746 | -1.4380 | **-1.3458** | **-1.3368** | **-1.2951** | **-1.1683** | **-1.1483** | **-1.1410** | **-1.3170** | **-1.0647** | **-0.94799** | **-1.0300** | **-1.1631** | **-0.82821** | **-0.81372** | **-0.74199** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.27198 | 0.26455 | **0.26440** | 0.26812 | **0.29520** | **0.31964** | **0.27855** | **0.29069** | **0.29421** | **0.30237** | **0.28556** | **0.31697** | **0.35691** | 0.28740 | 0.27147 | **0.26320** | 0.26703 | **0.29178** | **0.31053** | **0.33108** | **0.36825** | **0.39749** | **0.39440** | **0.35374** | **0.45134** | **0.53353** | **0.47385** | **0.37856** | **0.55666** | **0.57460** | **0.60544** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.27198 | 0.26455 | **0.26440** | 0.26812 | **0.29520** | **0.31964** | **0.27855** | **0.29069** | **0.29421** | **0.30237** | **0.28556** | **0.31697** | **0.35691** | 0.28740 | 0.27147 | **0.26320** | 0.26703 | **0.29178** | **0.31053** | **0.33108** | **0.36825** | **0.39749** | **0.39440** | **0.35374** | **0.45134** | **0.53353** | **0.47385** | **0.37856** | **0.55666** | **0.57460** | **0.60544** |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | **0.67326** | 0.68598 | 0.71772 | 0.74546 | **0.67113** | **0.66264** | **0.67496** | **0.64684** | **0.66671** | **0.65778** | **0.62958** | **0.63828** | **0.59570** | **0.66515** | 0.68166 | 0.69719 | 0.71477 | **0.63267** | **0.61775** | **0.61555** | **0.59136** | **0.52206** | **0.53849** | **0.55607** | **0.53494** | **0.50146** | **0.49747** | **0.56164** | **0.49543** | **0.48522** | **0.50962** |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | **0.73274** | 0.74696 | 0.78173 | 0.81192 | **0.73027** | **0.72156** | **0.73409** | **0.70317** | **0.72523** | **0.71558** | **0.68543** | **0.69557** | **0.64858** | **0.72352** | 0.74132 | 0.75811 | 0.77633 | **0.68805** | **0.67207** | **0.66956** | **0.64288** | **0.56666** | **0.58414** | **0.60403** | **0.58121** | **0.54357** | **0.53944** | **0.60992** | **0.53735** | **0.52572** | **0.55263** |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 0.99662 | **0.99125** | 1.0231 | 1.0287 | **0.89255** | **0.83624** | **0.93367** | **0.90322** | **0.86288** | **0.93560** | **0.99609** | **0.77363** | **0.71490** | 0.96440 | 0.98397 | **0.95077** | 0.99402 | **0.89941** | **0.79174** | **0.72319** | **0.64723** | **0.57726** | **0.58750** | **0.75618** | **0.47433** | **0.37492** | **0.42179** | **0.64226** | **0.33480** | **0.28913** | **0.25678** |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 1.9932 | **1.9825** | 2.0461 | 2.0575 | **1.7851** | **1.6725** | **1.8673** | **1.8064** | **1.7258** | **1.8712** | **1.9922** | **1.5473** | **1.4298** | 1.9288 | 1.9679 | **1.9015** | 1.9880 | **1.7988** | **1.5835** | **1.4464** | **1.2945** | **1.1545** | **1.1750** | **1.5124** | **0.94866** | **0.74984** | **0.84358** | **1.2845** | **0.66961** | **0.57826** | **0.51356** |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.69082 | **0.68713** | 0.70916 | 0.71306 | **0.61867** | **0.57967** | **0.64717** | **0.62605** | **0.59810** | **0.64853** | **0.69042** | **0.53622** | **0.49550** | 0.66847 | 0.68201 | **0.65906** | 0.68902 | **0.62345** | **0.54878** | **0.50129** | **0.44861** | **0.40014** | **0.40727** | **0.52421** | **0.32877** | **0.25985** | **0.29238** | **0.44518** | **0.23208** | **0.20042** | **0.17801** |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.3816 | **1.3743** | 1.4183 | 1.4261 | **1.2373** | **1.1593** | **1.2943** | **1.2521** | **1.1962** | **1.2971** | **1.3808** | **1.0724** | **0.99100** | 1.3369 | 1.3640 | **1.3181** | 1.3780 | **1.2469** | **1.0976** | **1.0026** | **0.89722** | **0.80027** | **0.81453** | **1.0484** | **0.65754** | **0.51970** | **0.58475** | **0.89036** | **0.46415** | **0.40084** | **0.35601** |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | **0.33333** | 0.30051 | 0.27988 | 0.32113 | **0.44571** | **0.49663** | **0.41035** | **0.46170** | **0.45665** | **0.42172** | **0.37247** | **0.54714** | **0.57492** | **0.42130** | 0.37416 | 0.37247 | 0.32912 | **0.43561** | **0.52441** | **0.57534** | **0.63173** | **0.68098** | **0.66120** | **0.54630** | **0.74032** | **0.79840** | **0.76389** | **0.63510** | **0.81987** | **0.85017** | **0.87037** |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | **0.33333** | 0.30051 | 0.27988 | 0.32113 | **0.44571** | **0.49663** | **0.41035** | **0.46170** | **0.45665** | **0.42172** | **0.37247** | **0.54714** | **0.57492** | **0.42130** | 0.37416 | 0.37247 | 0.32912 | **0.43561** | **0.52441** | **0.57534** | **0.63173** | **0.68098** | **0.66120** | **0.54630** | **0.74032** | **0.79840** | **0.76389** | **0.63510** | **0.81987** | **0.85017** | **0.87037** |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3799 | -1.3699 | -1.4161 | **-1.4234** | **-1.2357** | **-1.1577** | **-1.2921** | **-1.2493** | **-1.1941** | **-1.2942** | **-1.3791** | **-1.0681** | **-0.98982** | -1.3346 | -1.3613 | -1.3143 | **-1.3757** | **-1.2450** | **-1.0961** | **-1.0012** | **-0.89529** | **-0.79884** | **-0.81305** | **-1.0475** | **-0.65647** | **-0.51884** | **-0.58375** | **-0.88958** | **-0.46284** | **-0.40027** | **-0.35526** |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3799 | -1.3699 | -1.4161 | **-1.4234** | **-1.2357** | **-1.1577** | **-1.2921** | **-1.2493** | **-1.1941** | **-1.2942** | **-1.3791** | **-1.0681** | **-0.98982** | -1.3346 | -1.3613 | -1.3143 | **-1.3757** | **-1.2450** | **-1.0961** | **-1.0012** | **-0.89529** | **-0.79884** | **-0.81305** | **-1.0475** | **-0.65647** | **-0.51884** | **-0.58375** | **-0.88958** | **-0.46284** | **-0.40027** | **-0.35526** |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.28917 | **0.27068** | 0.27513 | 0.28151 | **0.33428** | **0.41875** | **0.30396** | **0.35699** | **0.33208** | **0.36635** | **0.31482** | **0.41032** | **0.49003** | 0.33045 | 0.30209 | 0.27870 | **0.27773** | **0.32815** | **0.40233** | **0.44801** | **0.50308** | **0.55870** | **0.54735** | **0.45862** | **0.63998** | **0.72710** | **0.67066** | **0.50694** | **0.73312** | **0.76457** | **0.79105** |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.28917 | **0.27068** | 0.27513 | 0.28151 | **0.33428** | **0.41875** | **0.30396** | **0.35699** | **0.33208** | **0.36635** | **0.31482** | **0.41032** | **0.49003** | 0.33045 | 0.30209 | 0.27870 | **0.27773** | **0.32815** | **0.40233** | **0.44801** | **0.50308** | **0.55870** | **0.54735** | **0.45862** | **0.63998** | **0.72710** | **0.67066** | **0.50694** | **0.73312** | **0.76457** | **0.79105** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 0.65010 | 0.72478 | **0.55652** | 0.70821 | **0.49844** | **0.51014** | **0.68047** | **0.54328** | **0.52849** | **0.63290** | **0.49546** | **0.44611** | **0.48403** | 0.67539 | 0.64967 | **0.52547** | 0.53748 | **0.46089** | **0.47416** | **0.48405** | **0.53418** | **0.44153** | **0.42639** | **0.44202** | **0.54788** | **0.32558** | **0.36705** | **0.48320** | **0.36090** | **0.35313** | **0.24808** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.0304 | 1.1536 | **0.89431** | 1.1286 | **0.78732** | **0.81013** | **1.0807** | **0.85881** | **0.83887** | **1.0033** | **0.78946** | **0.70415** | **0.76525** | 1.0851 | 1.0427 | **0.83556** | 0.86121 | **0.73354** | **0.75283** | **0.77106** | **0.84875** | **0.70318** | **0.68149** | **0.69579** | **0.85682** | **0.50594** | **0.57318** | **0.76554** | **0.57471** | **0.55911** | **0.38745** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.45056 | 0.50242 | **0.38574** | 0.49085 | **0.34551** | **0.35352** | **0.47168** | **0.37662** | **0.36631** | **0.43863** | **0.34346** | **0.30922** | **0.33545** | 0.46814 | 0.45029 | **0.36423** | 0.37252 | **0.31945** | **0.32864** | **0.33549** | **0.37028** | **0.30608** | **0.29555** | **0.30633** | **0.37978** | **0.22560** | **0.25443** | **0.33491** | **0.25017** | **0.24476** | **0.17194** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 0.71425 | 0.79946 | **0.61992** | 0.78232 | **0.54575** | **0.56153** | **0.74898** | **0.59530** | **0.58145** | **0.69535** | **0.54719** | **0.48807** | **0.53037** | 0.75220 | 0.72279 | **0.57914** | 0.59696 | **0.50851** | **0.52180** | **0.53454** | **0.58835** | **0.48750** | **0.47237** | **0.48231** | **0.59396** | **0.35066** | **0.39726** | **0.53062** | **0.39827** | **0.38761** | **0.26853** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.77555 | 0.75072 | **0.79083** | 0.77650 | **0.82999** | **0.82713** | **0.74785** | **0.79370** | **0.81184** | **0.75549** | **0.83190** | **0.84145** | **0.81280** | 0.75549 | 0.78032 | **0.81471** | 0.80898 | **0.82139** | **0.82808** | **0.82330** | **0.79752** | **0.83381** | **0.83859** | **0.83381** | **0.80325** | **0.86915** | **0.85960** | **0.80611** | **0.84814** | **0.85291** | **0.89971** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.77555 | 0.75072 | **0.79083** | 0.77650 | **0.82999** | **0.82713** | **0.74785** | **0.79370** | **0.81184** | **0.75549** | **0.83190** | **0.84145** | **0.81280** | 0.75549 | 0.78032 | **0.81471** | 0.80898 | **0.82139** | **0.82808** | **0.82330** | **0.79752** | **0.83381** | **0.83859** | **0.83381** | **0.80325** | **0.86915** | **0.85960** | **0.80611** | **0.84814** | **0.85291** | **0.89971** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -0.74648 | **-0.79550** | -0.64098 | -0.78717 | **-0.55702** | **-0.53675** | **-0.80240** | **-0.61797** | **-0.60502** | **-0.77952** | **-0.54978** | **-0.50647** | **-0.56427** | **-0.76073** | -0.72979 | -0.58688 | -0.61183 | **-0.53850** | **-0.55269** | **-0.55049** | **-0.61065** | **-0.53165** | **-0.47580** | **-0.51026** | **-0.63253** | **-0.42026** | **-0.44407** | **-0.60517** | **-0.45930** | **-0.43547** | **-0.30795** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -0.74648 | **-0.79550** | -0.64098 | -0.78717 | **-0.55702** | **-0.53675** | **-0.80240** | **-0.61797** | **-0.60502** | **-0.77952** | **-0.54978** | **-0.50647** | **-0.56427** | **-0.76073** | -0.72979 | -0.58688 | -0.61183 | **-0.53850** | **-0.55269** | **-0.55049** | **-0.61065** | **-0.53165** | **-0.47580** | **-0.51026** | **-0.63253** | **-0.42026** | **-0.44407** | **-0.60517** | **-0.45930** | **-0.43547** | **-0.30795** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.71496 | 0.70982 | 0.74671 | **0.70340** | **0.76439** | **0.78746** | **0.69931** | **0.74366** | **0.77597** | **0.71053** | **0.79458** | **0.79979** | **0.76393** | **0.71088** | 0.72648 | 0.75770 | 0.74300 | **0.76985** | **0.77324** | **0.78531** | **0.76606** | **0.79478** | **0.79474** | **0.78089** | **0.75776** | **0.84140** | **0.81992** | **0.76285** | **0.82068** | **0.82299** | **0.86986** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.71496 | 0.70982 | 0.74671 | **0.70340** | **0.76439** | **0.78746** | **0.69931** | **0.74366** | **0.77597** | **0.71053** | **0.79458** | **0.79979** | **0.76393** | **0.71088** | 0.72648 | 0.75770 | 0.74300 | **0.76985** | **0.77324** | **0.78531** | **0.76606** | **0.79478** | **0.79474** | **0.78089** | **0.75776** | **0.84140** | **0.81992** | **0.76285** | **0.82068** | **0.82299** | **0.86986** |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | **0.36288** | 0.36813 | 0.41394 | 0.42801 | **0.38178** | **0.31520** | **0.34787** | **0.36922** | **0.30869** | **0.33147** | **0.36937** | **0.35506** | **0.36966** | 0.42875 | 0.38882 | 0.40584 | **0.37626** | **0.35012** | **0.40377** | **0.38452** | **0.36810** | **0.35206** | **0.40080** | **0.38795** | **0.35527** | **0.34645** | **0.29569** | **0.36785** | **0.38293** | **0.35881** | **0.34652** |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | **0.39531** | 0.40115 | 0.45103 | 0.46660 | **0.41670** | **0.34351** | **0.37927** | **0.40338** | **0.33671** | **0.36123** | **0.40274** | **0.38689** | **0.40297** | 0.46679 | 0.42271 | 0.44186 | **0.41018** | **0.38187** | **0.44015** | **0.41974** | **0.40152** | **0.38404** | **0.43718** | **0.42279** | **0.38705** | **0.37762** | **0.32221** | **0.40167** | **0.41748** | **0.39087** | **0.37812** |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | **0.25150** | 0.25519 | 0.28697 | 0.29669 | **0.26467** | **0.21848** | **0.24109** | **0.25589** | **0.21394** | **0.22977** | **0.25602** | **0.24612** | **0.25624** | 0.29715 | 0.26951 | 0.28132 | **0.26080** | **0.24269** | **0.27986** | **0.26654** | **0.25516** | **0.24403** | **0.27780** | **0.26890** | **0.24624** | **0.24017** | **0.20495** | **0.25498** | **0.26545** | **0.24868** | **0.24020** |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | **0.27401** | 0.27806 | 0.31263 | 0.32344 | **0.28882** | **0.23810** | **0.26287** | **0.27960** | **0.23337** | **0.25035** | **0.27916** | **0.26817** | **0.27930** | 0.32356 | 0.29301 | 0.30628 | **0.28429** | **0.26470** | **0.30510** | **0.29092** | **0.27830** | **0.26617** | **0.30303** | **0.29305** | **0.26827** | **0.26176** | **0.22333** | **0.27840** | **0.28940** | **0.27089** | **0.26208** |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.73123 | **0.73221** | 0.67589 | 0.71245 | **0.74901** | **0.73814** | **0.74308** | **0.75000** | **0.75593** | **0.75395** | **0.78557** | **0.76680** | **0.75296** | 0.71542 | **0.72727** | **0.72727** | 0.72036 | **0.79941** | **0.67095** | **0.75593** | **0.77569** | **0.79743** | **0.76976** | **0.75889** | **0.79842** | **0.81324** | **0.82213** | **0.78953** | **0.80040** | **0.82213** | **0.81028** |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.73123 | **0.73221** | 0.67589 | 0.71245 | **0.74901** | **0.73814** | **0.74308** | **0.75000** | **0.75593** | **0.75395** | **0.78557** | **0.76680** | **0.75296** | 0.71542 | **0.72727** | **0.72727** | 0.72036 | **0.79941** | **0.67095** | **0.75593** | **0.77569** | **0.79743** | **0.76976** | **0.75889** | **0.79842** | **0.81324** | **0.82213** | **0.78953** | **0.80040** | **0.82213** | **0.81028** |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -0.95967 | -0.92786 | **-1.2174** | -1.1560 | **-0.93625** | **-0.89640** | **-0.91551** | **-0.82032** | **-0.81302** | **-0.92825** | **-0.74654** | **-0.81995** | **-0.84164** | -1.0867 | -1.1009 | -1.0207 | **-1.1063** | **-0.76196** | **-1.1750** | **-0.84528** | **-0.71162** | **-0.62978** | **-0.74884** | **-0.81769** | **-0.64245** | **-0.61387** | **-0.52453** | **-0.69492** | **-0.63085** | **-0.56427** | **-0.58318** |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -0.95967 | -0.92786 | **-1.2174** | -1.1560 | **-0.93625** | **-0.89640** | **-0.91551** | **-0.82032** | **-0.81302** | **-0.92825** | **-0.74654** | **-0.81995** | **-0.84164** | -1.0867 | -1.1009 | -1.0207 | **-1.1063** | **-0.76196** | **-1.1750** | **-0.84528** | **-0.71162** | **-0.62978** | **-0.74884** | **-0.81769** | **-0.64245** | **-0.61387** | **-0.52453** | **-0.69492** | **-0.63085** | **-0.56427** | **-0.58318** |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.71747 | 0.72129 | **0.67513** | 0.69753 | **0.71795** | **0.72317** | **0.71454** | **0.73815** | **0.73447** | **0.73118** | **0.77447** | **0.74218** | **0.73736** | 0.70050 | 0.70888 | 0.71252 | **0.69943** | **0.76257** | **0.65922** | **0.72890** | **0.75704** | **0.78158** | **0.75141** | **0.74331** | **0.77600** | **0.79272** | **0.80829** | **0.77618** | **0.79058** | **0.81247** | **0.80437** |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.71747 | 0.72129 | **0.67513** | 0.69753 | **0.71795** | **0.72317** | **0.71454** | **0.73815** | **0.73447** | **0.73118** | **0.77447** | **0.74218** | **0.73736** | 0.70050 | 0.70888 | 0.71252 | **0.69943** | **0.76257** | **0.65922** | **0.72890** | **0.75704** | **0.78158** | **0.75141** | **0.74331** | **0.77600** | **0.79272** | **0.80829** | **0.77618** | **0.79058** | **0.81247** | **0.80437** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.52145 | 0.50044 | 0.56032 | **0.45591** | **0.40501** | **0.42236** | **0.61793** | **0.34173** | **0.59887** | **0.45528** | **0.43290** | **0.38457** | **0.33510** | **0.41584** | 0.42808 | 0.46691 | 0.57099 | **0.34481** | **0.44913** | **0.38017** | **0.34771** | **0.27184** | **0.33362** | **0.33961** | **0.29878** | **0.20941** | **0.23238** | **0.31155** | **0.22860** | **0.25569** | **0.26072** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.62514 | 0.60055 | 0.67130 | **0.54775** | **0.48601** | **0.50802** | **0.74099** | **0.40949** | **0.71828** | **0.54682** | **0.52013** | **0.46148** | **0.40297** | **0.49918** | 0.51560 | 0.56114 | 0.68624 | **0.41438** | **0.54011** | **0.45674** | **0.41874** | **0.32789** | **0.39987** | **0.40854** | **0.36014** | **0.25183** | **0.27941** | **0.37425** | **0.27431** | **0.30704** | **0.31332** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.36162 | 0.34710 | 0.38849 | **0.31612** | **0.28082** | **0.29294** | **0.42851** | **0.23700** | **0.41524** | **0.31578** | **0.30030** | **0.26677** | **0.23249** | **0.28835** | 0.29681 | 0.32376 | 0.39590 | **0.23913** | **0.31154** | **0.26379** | **0.24121** | **0.18864** | **0.23153** | **0.23563** | **0.20731** | **0.14539** | **0.16128** | **0.21609** | **0.15867** | **0.17744** | **0.18092** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.43355 | 0.41659 | 0.46550 | **0.37975** | **0.33707** | **0.35240** | **0.51381** | **0.28410** | **0.49813** | **0.37925** | **0.36086** | **0.32013** | **0.27958** | **0.34622** | 0.35752 | 0.38910 | 0.47588 | **0.28735** | **0.37464** | **0.31690** | **0.29051** | **0.22756** | **0.27757** | **0.28349** | **0.24988** | **0.17490** | **0.19398** | **0.25963** | **0.19042** | **0.21314** | **0.21741** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.79460 | **0.80906** | 0.80231 | 0.79942 | **0.83799** | **0.83703** | **0.80906** | **0.85921** | **0.83317** | **0.83510** | **0.86210** | **0.87946** | **0.88621** | 0.83317 | **0.83414** | 0.80810 | 0.75217 | **0.86982** | **0.86017** | **0.84957** | **0.89392** | **0.93443** | **0.91321** | **0.88139** | **0.93925** | **0.94889** | **0.96143** | **0.92382** | **0.94407** | **0.96143** | **0.95661** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.79460 | **0.80906** | 0.80231 | 0.79942 | **0.83799** | **0.83703** | **0.80906** | **0.85921** | **0.83317** | **0.83510** | **0.86210** | **0.87946** | **0.88621** | 0.83317 | **0.83414** | 0.80810 | 0.75217 | **0.86982** | **0.86017** | **0.84957** | **0.89392** | **0.93443** | **0.91321** | **0.88139** | **0.93925** | **0.94889** | **0.96143** | **0.92382** | **0.94407** | **0.96143** | **0.95661** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | **-0.57022** | -0.52961 | -0.53063 | -0.53294 | **-0.47262** | **-0.45263** | **-0.54784** | **-0.37459** | **-0.47022** | **-0.48210** | **-0.41016** | **-0.38375** | **-0.32348** | -0.45539 | -0.49365 | -0.53345 | **-0.64738** | **-0.37097** | **-0.40240** | **-0.40982** | **-0.31237** | **-0.21295** | **-0.25155** | **-0.31913** | **-0.20420** | **-0.17768** | **-0.14437** | **-0.25664** | **-0.17342** | **-0.14672** | **-0.13760** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | **-0.57022** | -0.52961 | -0.53063 | -0.53294 | **-0.47262** | **-0.45263** | **-0.54784** | **-0.37459** | **-0.47022** | **-0.48210** | **-0.41016** | **-0.38375** | **-0.32348** | -0.45539 | -0.49365 | -0.53345 | **-0.64738** | **-0.37097** | **-0.40240** | **-0.40982** | **-0.31237** | **-0.21295** | **-0.25155** | **-0.31913** | **-0.20420** | **-0.17768** | **-0.14437** | **-0.25664** | **-0.17342** | **-0.14672** | **-0.13760** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | **0.68596** | 0.70690 | 0.70220 | 0.71715 | **0.72974** | **0.73275** | **0.69235** | **0.78310** | **0.72664** | **0.73393** | **0.75902** | **0.76868** | **0.79719** | 0.73205 | 0.72526 | 0.70574 | **0.66858** | **0.78265** | **0.76522** | **0.75347** | **0.80345** | **0.86188** | **0.85881** | **0.80686** | **0.86541** | **0.88211** | **0.89834** | **0.83239** | **0.88291** | **0.89718** | **0.90613** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | **0.68596** | 0.70690 | 0.70220 | 0.71715 | **0.72974** | **0.73275** | **0.69235** | **0.78310** | **0.72664** | **0.73393** | **0.75902** | **0.76868** | **0.79719** | 0.73205 | 0.72526 | 0.70574 | **0.66858** | **0.78265** | **0.76522** | **0.75347** | **0.80345** | **0.86188** | **0.85881** | **0.80686** | **0.86541** | **0.88211** | **0.89834** | **0.83239** | **0.88291** | **0.89718** | **0.90613** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29722 | 0.28635 | **0.27329** | 0.28688 | **0.28865** | **0.34992** | **0.30843** | **0.25377** | **0.28537** | **0.29545** | **0.27097** | **0.29769** | **0.25116** | 0.29338 | 0.30529 | **0.25685** | 0.27956 | **0.25675** | **0.29429** | **0.28405** | **0.28000** | **0.22853** | **0.25592** | **0.27437** | **0.26400** | **0.23195** | **0.22136** | **0.27948** | **0.25966** | **0.24341** | **0.22040** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30718 | 0.29591 | **0.28236** | 0.29639 | **0.29831** | **0.36152** | **0.31873** | **0.26229** | **0.29496** | **0.30540** | **0.28004** | **0.30771** | **0.25954** | 0.30316 | 0.31546 | **0.26542** | 0.28891 | **0.26529** | **0.30422** | **0.29366** | **0.28941** | **0.23623** | **0.26458** | **0.28363** | **0.27288** | **0.23982** | **0.22888** | **0.28883** | **0.26842** | **0.25173** | **0.22798** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20603 | 0.19851 | **0.18945** | 0.19886 | **0.20011** | **0.24257** | **0.21381** | **0.17591** | **0.19783** | **0.20479** | **0.18783** | **0.20636** | **0.17411** | 0.20338 | 0.21163 | **0.17807** | 0.19379 | **0.17796** | **0.20402** | **0.19690** | **0.19409** | **0.15843** | **0.17740** | **0.19019** | **0.18302** | **0.16079** | **0.15344** | **0.19374** | **0.17999** | **0.16874** | **0.15278** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21293 | 0.20514 | **0.19573** | 0.20549 | **0.20679** | **0.25060** | **0.22095** | **0.18181** | **0.20445** | **0.21169** | **0.19414** | **0.21332** | **0.17992** | 0.21017 | 0.21869 | **0.18401** | 0.20028 | **0.18388** | **0.21091** | **0.20356** | **0.20062** | **0.16374** | **0.18341** | **0.19661** | **0.18917** | **0.16624** | **0.15865** | **0.20023** | **0.18607** | **0.17452** | **0.15804** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | **0.83721** | 0.80590 | 0.81395 | **0.83721** | **0.83542** | **0.83363** | **0.82648** | **0.85510** | **0.86225** | **0.82648** | **0.87925** | **0.82558** | **0.85420** | 0.82737 | 0.83095 | 0.84526 | **0.88104** | **0.85510** | **0.82826** | **0.88551** | **0.87030** | **0.92129** | **0.91860** | **0.87120** | **0.92844** | **0.96869** | **0.91771** | **0.84705** | **0.95796** | **0.95886** | **0.95707** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | **0.83721** | 0.80590 | 0.81395 | **0.83721** | **0.83542** | **0.83363** | **0.82648** | **0.85510** | **0.86225** | **0.82648** | **0.87925** | **0.82558** | **0.85420** | 0.82737 | 0.83095 | 0.84526 | **0.88104** | **0.85510** | **0.82826** | **0.88551** | **0.87030** | **0.92129** | **0.91860** | **0.87120** | **0.92844** | **0.96869** | **0.91771** | **0.84705** | **0.95796** | **0.95886** | **0.95707** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.45501 | **-0.51743** | -0.48103 | -0.41398 | **-0.42174** | **-0.41726** | **-0.47340** | **-0.39935** | **-0.39761** | **-0.44094** | **-0.29810** | **-0.48459** | **-0.37568** | -0.42338 | **-0.43458** | -0.39605 | -0.34205 | **-0.34218** | **-0.49545** | **-0.33964** | **-0.31669** | **-0.21843** | **-0.22733** | **-0.28563** | **-0.21686** | **-0.12954** | **-0.22825** | **-0.40391** | **-0.13885** | **-0.13612** | **-0.11180** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.45501 | **-0.51743** | -0.48103 | -0.41398 | **-0.42174** | **-0.41726** | **-0.47340** | **-0.39935** | **-0.39761** | **-0.44094** | **-0.29810** | **-0.48459** | **-0.37568** | -0.42338 | **-0.43458** | -0.39605 | -0.34205 | **-0.34218** | **-0.49545** | **-0.33964** | **-0.31669** | **-0.21843** | **-0.22733** | **-0.28563** | **-0.21686** | **-0.12954** | **-0.22825** | **-0.40391** | **-0.13885** | **-0.13612** | **-0.11180** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82881 | **0.81260** | 0.82368 | 0.82590 | **0.83505** | **0.82324** | **0.81456** | **0.83881** | **0.83676** | **0.82875** | **0.87032** | **0.82343** | **0.85008** | **0.83332** | 0.83485 | 0.84300 | 0.86230 | **0.85773** | **0.82618** | **0.86776** | **0.86641** | **0.89918** | **0.89940** | **0.86601** | **0.91116** | **0.94530** | **0.89491** | **0.85133** | **0.93724** | **0.93706** | **0.94179** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82881 | **0.81260** | 0.82368 | 0.82590 | **0.83505** | **0.82324** | **0.81456** | **0.83881** | **0.83676** | **0.82875** | **0.87032** | **0.82343** | **0.85008** | **0.83332** | 0.83485 | 0.84300 | 0.86230 | **0.85773** | **0.82618** | **0.86776** | **0.86641** | **0.89918** | **0.89940** | **0.86601** | **0.91116** | **0.94530** | **0.89491** | **0.85133** | **0.93724** | **0.93706** | **0.94179** |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | **1.0774** | 1.0777 | 1.1267 | 1.1079 | **0.96761** | **0.98676** | **1.1054** | **0.94319** | **1.0315** | **1.0039** | **0.84976** | **0.81133** | **0.76554** | 0.98241 | **0.96241** | 1.0032 | 1.0850 | **0.91387** | **0.76579** | **0.86417** | **0.86737** | **0.88127** | **0.80899** | **0.82876** | **0.82792** | **0.60998** | **0.74306** | **0.76292** | **0.66285** | **0.76380** | **0.73376** |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | **1.7259** | 1.7327 | 1.8180 | 1.7590 | **1.5526** | **1.5765** | **1.7749** | **1.5238** | **1.6444** | **1.6300** | **1.3896** | **1.3069** | **1.2641** | 1.5959 | **1.5595** | 1.6136 | 1.7488 | **1.4974** | **1.2445** | **1.4280** | **1.4171** | **1.4482** | **1.3170** | **1.3711** | **1.3688** | **1.0087** | **1.2337** | **1.2575** | **1.1083** | **1.2732** | **1.2001** |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | **0.78066** | 0.78721 | 0.81873 | 0.80983 | **0.70613** | **0.71961** | **0.80280** | **0.68380** | **0.75013** | **0.73289** | **0.61746** | **0.59281** | **0.56022** | 0.70918 | **0.70760** | 0.73758 | 0.79738 | **0.67163** | **0.56500** | **0.64205** | **0.63064** | **0.64725** | **0.59590** | **0.60626** | **0.60513** | **0.45389** | **0.54028** | **0.56153** | **0.49174** | **0.55881** | **0.53556** |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | **1.2830** | 1.3037 | 1.3565 | 1.3274 | **1.1655** | **1.1825** | **1.3239** | **1.1324** | **1.2302** | **1.2239** | **1.0359** | **0.98275** | **0.95098** | **1.1781** | 1.1851 | 1.2264 | 1.3283 | **1.1357** | **0.95057** | **1.1005** | **1.0562** | **1.0969** | **1.0033** | **1.0323** | **1.0286** | **0.77785** | **0.91909** | **0.95498** | **0.85039** | **0.95723** | **0.90019** |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.62547 | 0.65543 | **0.66667** | 0.65356 | **0.71348** | **0.67228** | **0.63670** | **0.68352** | **0.67041** | **0.67041** | **0.71723** | **0.68539** | **0.71910** | 0.67041 | **0.69288** | 0.67041 | 0.66105 | **0.70974** | **0.72472** | **0.72659** | **0.75094** | **0.74345** | **0.73783** | **0.74906** | **0.76592** | **0.79026** | **0.77903** | **0.76779** | **0.79213** | **0.78090** | **0.77154** |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.62547 | 0.65543 | **0.66667** | 0.65356 | **0.71348** | **0.67228** | **0.63670** | **0.68352** | **0.67041** | **0.67041** | **0.71723** | **0.68539** | **0.71910** | 0.67041 | **0.69288** | 0.67041 | 0.66105 | **0.70974** | **0.72472** | **0.72659** | **0.75094** | **0.74345** | **0.73783** | **0.74906** | **0.76592** | **0.79026** | **0.77903** | **0.76779** | **0.79213** | **0.78090** | **0.77154** |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.94568 | -0.92084 | -0.86213 | **-0.99549** | **-0.79883** | **-0.89031** | **-0.93601** | **-0.82331** | **-0.86938** | **-0.86051** | **-0.72721** | **-0.79067** | **-0.71455** | -0.85807 | -0.88042 | -0.87968 | **-0.91216** | **-0.81234** | **-0.71946** | **-0.73105** | **-0.67990** | **-0.67446** | **-0.63274** | **-0.65656** | **-0.58736** | **-0.52231** | **-0.56021** | **-0.61824** | **-0.53072** | **-0.56258** | **-0.58924** |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.94568 | -0.92084 | -0.86213 | **-0.99549** | **-0.79883** | **-0.89031** | **-0.93601** | **-0.82331** | **-0.86938** | **-0.86051** | **-0.72721** | **-0.79067** | **-0.71455** | -0.85807 | -0.88042 | -0.87968 | **-0.91216** | **-0.81234** | **-0.71946** | **-0.73105** | **-0.67990** | **-0.67446** | **-0.63274** | **-0.65656** | **-0.58736** | **-0.52231** | **-0.56021** | **-0.61824** | **-0.53072** | **-0.56258** | **-0.58924** |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | **0.56627** | 0.57771 | 0.59092 | 0.57766 | **0.63218** | **0.59076** | **0.55178** | **0.61773** | **0.60531** | **0.59990** | **0.64801** | **0.62687** | **0.65465** | 0.59597 | 0.61575 | 0.60133 | **0.59220** | **0.63014** | **0.65437** | **0.63038** | **0.66033** | **0.67317** | **0.66845** | **0.66200** | **0.68168** | **0.72682** | **0.69517** | **0.67184** | **0.71088** | **0.71791** | **0.67791** |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | **0.56627** | 0.57771 | 0.59092 | 0.57766 | **0.63218** | **0.59076** | **0.55178** | **0.61773** | **0.60531** | **0.59990** | **0.64801** | **0.62687** | **0.65465** | 0.59597 | 0.61575 | 0.60133 | **0.59220** | **0.63014** | **0.65437** | **0.63038** | **0.66033** | **0.67317** | **0.66845** | **0.66200** | **0.68168** | **0.72682** | **0.69517** | **0.67184** | **0.71088** | **0.71791** | **0.67791** |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.6057 | 1.5035 | **1.4928** | 1.6324 | **1.5834** | **1.6120** | **1.6248** | **1.3936** | **1.4011** | **1.7586** | **1.2541** | **1.3408** | **1.2426** | 1.4228 | 1.4875 | **1.3942** | 1.4802 | **1.2853** | **1.4924** | **1.3287** | **1.3171** | **1.0541** | **1.1605** | **1.2702** | **1.2018** | **0.99337** | **0.87161** | **1.1253** | **0.92335** | **0.97563** | **0.72252** |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.1960 | 2.0810 | **2.0582** | 2.2539 | **2.1887** | **2.2421** | **2.2318** | **1.9109** | **1.9442** | **2.4283** | **1.7560** | **1.8677** | **1.7475** | 1.9750 | 2.0425 | **1.9355** | 2.0493 | **1.7871** | **2.0442** | **1.8295** | **1.8330** | **1.5110** | **1.6416** | **1.7868** | **1.7095** | **1.4221** | **1.2530** | **1.5630** | **1.3054** | **1.3981** | **1.0436** |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.1130 | 1.0421 | **1.0347** | 1.1315 | **1.0974** | **1.1173** | **1.1262** | **0.96607** | **0.97114** | **1.2189** | **0.86930** | **0.92944** | **0.86127** | 0.98621 | 1.0311 | **0.96640** | 1.0260 | **0.89091** | **1.0345** | **0.92101** | **0.91289** | **0.73062** | **0.80442** | **0.88047** | **0.83303** | **0.68867** | **0.60416** | **0.78010** | **0.63993** | **0.67625** | **0.50077** |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.5221 | 1.4425 | **1.4268** | 1.5623 | **1.5170** | **1.5542** | **1.5469** | **1.3245** | **1.3476** | **1.6830** | **1.2171** | **1.2945** | **1.2112** | 1.3691 | 1.4158 | **1.3417** | 1.4205 | **1.2387** | **1.4169** | **1.2682** | **1.2704** | **1.0474** | **1.1380** | **1.2383** | **1.1850** | **0.98569** | **0.86852** | **1.0832** | **0.90477** | **0.96908** | **0.72339** |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.32075 | **0.32978** | 0.29532 | 0.30025 | **0.37736** | **0.32896** | **0.31091** | **0.33552** | **0.37244** | **0.32896** | **0.37982** | **0.36259** | **0.39869** | 0.31747 | 0.29040 | **0.33306** | 0.32158 | **0.36833** | **0.32650** | **0.36669** | **0.37736** | **0.44217** | **0.41427** | **0.41099** | **0.44135** | **0.49467** | **0.50533** | **0.40197** | **0.49549** | **0.53651** | **0.57096** |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.32075 | **0.32978** | 0.29532 | 0.30025 | **0.37736** | **0.32896** | **0.31091** | **0.33552** | **0.37244** | **0.32896** | **0.37982** | **0.36259** | **0.39869** | 0.31747 | 0.29040 | **0.33306** | 0.32158 | **0.36833** | **0.32650** | **0.36669** | **0.37736** | **0.44217** | **0.41427** | **0.41099** | **0.44135** | **0.49467** | **0.50533** | **0.40197** | **0.49549** | **0.53651** | **0.57096** |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.5834 | -3.5888 | -3.5875 | **-3.6072** | **-3.0228** | **-2.9742** | **-3.4966** | **-2.9964** | **-2.8522** | **-3.8056** | **-2.8037** | **-3.0062** | **-2.6449** | -3.2438 | -3.3550 | -3.1842 | **-3.4066** | **-2.8954** | **-3.1910** | **-2.8807** | **-2.9132** | **-2.1969** | **-2.4078** | **-2.7458** | **-2.2474** | **-1.8714** | **-1.7428** | **-2.4781** | **-1.9744** | **-1.6367** | **-1.2612** |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.5834 | -3.5888 | -3.5875 | **-3.6072** | **-3.0228** | **-2.9742** | **-3.4966** | **-2.9964** | **-2.8522** | **-3.8056** | **-2.8037** | **-3.0062** | **-2.6449** | -3.2438 | -3.3550 | -3.1842 | **-3.4066** | **-2.8954** | **-3.1910** | **-2.8807** | **-2.9132** | **-2.1969** | **-2.4078** | **-2.7458** | **-2.2474** | **-1.8714** | **-1.7428** | **-2.4781** | **-1.9744** | **-1.6367** | **-1.2612** |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.32348 | 0.32609 | **0.30050** | 0.30530 | **0.38860** | **0.34644** | **0.32426** | **0.35003** | **0.38317** | **0.33776** | **0.39102** | **0.36546** | **0.40434** | 0.32756 | **0.31172** | 0.33903 | 0.32783 | **0.38130** | **0.33609** | **0.36253** | **0.38460** | **0.44512** | **0.43428** | **0.42249** | **0.45376** | **0.49929** | **0.51832** | **0.40970** | **0.50986** | **0.55647** | **0.57479** |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.32348 | 0.32609 | **0.30050** | 0.30530 | **0.38860** | **0.34644** | **0.32426** | **0.35003** | **0.38317** | **0.33776** | **0.39102** | **0.36546** | **0.40434** | 0.32756 | **0.31172** | 0.33903 | 0.32783 | **0.38130** | **0.33609** | **0.36253** | **0.38460** | **0.44512** | **0.43428** | **0.42249** | **0.45376** | **0.49929** | **0.51832** | **0.40970** | **0.50986** | **0.55647** | **0.57479** |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | **0.39462** | 0.39990 | 0.40951 | 0.41252 | **0.37299** | **0.37844** | **0.38762** | **0.37539** | **0.37317** | **0.39369** | **0.36867** | **0.36033** | **0.35432** | **0.37988** | 0.38802 | 0.39949 | 0.42595 | **0.35743** | **0.35926** | **0.35574** | **0.33577** | **0.32510** | **0.32810** | **0.33783** | **0.32498** | **0.31390** | **0.30659** | **0.33685** | **0.30613** | **0.29433** | **0.29872** |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | **0.39976** | 0.40503 | 0.41495 | 0.41808 | **0.37817** | **0.38376** | **0.39290** | **0.38034** | **0.37808** | **0.39910** | **0.37358** | **0.36512** | **0.35917** | **0.38485** | 0.39309 | 0.40512 | 0.43172 | **0.36212** | **0.36420** | **0.36042** | **0.34003** | **0.32944** | **0.33245** | **0.34221** | **0.32945** | **0.31827** | **0.31061** | **0.34133** | **0.31034** | **0.29825** | **0.30271** |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | **0.63536** | 0.64263 | 0.63706 | 0.66217 | **0.60414** | **0.59382** | **0.64630** | **0.58565** | **0.60054** | **0.64155** | **0.58593** | **0.61408** | **0.57955** | **0.62432** | 0.64576 | 0.63079 | 0.66491 | **0.58152** | **0.59095** | **0.59422** | **0.56411** | **0.55120** | **0.54163** | **0.53679** | **0.52429** | **0.51253** | **0.50279** | **0.53607** | **0.52131** | **0.48973** | **0.48377** |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | **0.64087** | 0.64825 | 0.64276 | 0.66801 | **0.60945** | **0.59895** | **0.65212** | **0.59065** | **0.60584** | **0.64725** | **0.59104** | **0.61954** | **0.58456** | **0.62940** | 0.65140 | 0.63636 | 0.67071 | **0.58649** | **0.59606** | **0.59945** | **0.56899** | **0.55594** | **0.54639** | **0.54144** | **0.52908** | **0.51700** | **0.50714** | **0.54074** | **0.52595** | **0.49410** | **0.48808** |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.2870 | 2.1814 | **1.9198** | 1.9998 | **1.7423** | **2.2652** | **1.9509** | **1.9304** | **2.0013** | **2.2274** | **2.1135** | **1.8046** | **1.6565** | 2.1948 | **1.9414** | 1.9959 | 2.3464 | **2.1809** | **1.9989** | **2.2674** | **1.6531** | **1.3904** | **1.8228** | **1.7352** | **1.2234** | **1.0992** | **1.4958** | **1.9344** | **0.15896** | **0.65544** | **0.34862** |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.5740 | 4.3628 | **3.8396** | 3.9996 | **3.4845** | **4.5305** | **3.9019** | **3.8608** | **4.0027** | **4.4548** | **4.2270** | **3.6093** | **3.3130** | 4.3895 | **3.8827** | 3.9919 | 4.6928 | **4.3619** | **3.9978** | **4.5347** | **3.3062** | **2.7809** | **3.6455** | **3.4704** | **2.4468** | **2.1984** | **2.9917** | **3.8689** | **0.31792** | **1.3109** | **0.69725** |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5854 | 1.5118 | **1.3303** | 1.3861 | **1.2076** | **1.5701** | **1.3527** | **1.3382** | **1.3870** | **1.5442** | **1.4649** | **1.2506** | **1.1482** | 1.5213 | **1.3460** | 1.3834 | 1.6261 | **1.5116** | **1.3854** | **1.5714** | **1.1458** | **0.96399** | **1.2640** | **1.2029** | **0.84815** | **0.76215** | **1.0369** | **1.3410** | **0.11010** | **0.45417** | **0.24177** |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.1707 | 3.0237 | **2.6605** | 2.7721 | **2.4152** | **3.1402** | **2.7054** | **2.6764** | **2.7739** | **3.0884** | **2.9298** | **2.5013** | **2.2964** | 3.0426 | **2.6920** | 2.7669 | 3.2521 | **3.0232** | **2.7707** | **3.1427** | **2.2916** | **1.9280** | **2.5280** | **2.4059** | **1.6963** | **1.5243** | **2.0739** | **2.6819** | **0.22020** | **0.90835** | **0.48355** |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | **0.10000** | **0.10000** | 0.09000 | **0.10000** | **0.14000** | **0.24000** | **0.09000** | **0.12000** | **0.09000** | **0.16000** | **0.14000** | **0.11000** | **0.28000** | 0.07000 | **0.09000** | **0.09000** | **0.09000** | **0.18000** | **0.08000** | **0.16000** | **0.16000** | **0.36000** | **0.22000** | **0.21000** | **0.47000** | **0.55000** | **0.38000** | **0.26000** | **0.94000** | **0.71000** | **0.91000** |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | **0.10000** | **0.10000** | 0.09000 | **0.10000** | **0.14000** | **0.24000** | **0.09000** | **0.12000** | **0.09000** | **0.16000** | **0.14000** | **0.11000** | **0.28000** | 0.07000 | **0.09000** | **0.09000** | **0.09000** | **0.18000** | **0.08000** | **0.16000** | **0.16000** | **0.36000** | **0.22000** | **0.21000** | **0.47000** | **0.55000** | **0.38000** | **0.26000** | **0.94000** | **0.71000** | **0.91000** |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | **-3.1645** | -3.0153 | -2.6477 | -2.7536 | **-2.3853** | **-3.1236** | **-2.6942** | **-2.6655** | **-2.7621** | **-3.0824** | **-2.9204** | **-2.4912** | **-2.2868** | -3.0386 | -2.6847 | -2.7545 | **-3.2430** | **-3.0123** | **-2.7535** | **-3.1331** | **-2.2792** | **-1.9207** | **-2.5197** | **-2.3765** | **-1.6929** | **-1.5183** | **-2.0587** | **-2.6761** | **-0.21512** | **-0.90639** | **-0.48098** |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | **-3.1645** | -3.0153 | -2.6477 | -2.7536 | **-2.3853** | **-3.1236** | **-2.6942** | **-2.6655** | **-2.7621** | **-3.0824** | **-2.9204** | **-2.4912** | **-2.2868** | -3.0386 | -2.6847 | -2.7545 | **-3.2430** | **-3.0123** | **-2.7535** | **-3.1331** | **-2.2792** | **-1.9207** | **-2.5197** | **-2.3765** | **-1.6929** | **-1.5183** | **-2.0587** | **-2.6761** | **-0.21512** | **-0.90639** | **-0.48098** |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | **0.09619** | 0.09935 | 0.09730 | 0.09728 | **0.10708** | **0.10808** | **0.09898** | **0.10546** | **0.09644** | **0.11748** | **0.10058** | **0.11118** | **0.15113** | 0.10150 | 0.10299 | 0.09943 | **0.09575** | **0.09949** | **0.10212** | **0.11608** | **0.11245** | **0.23868** | **0.13532** | **0.11772** | **0.30636** | **0.37547** | **0.22183** | **0.15517** | **0.85975** | **0.55322** | **0.75442** |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | **0.09619** | 0.09935 | 0.09730 | 0.09728 | **0.10708** | **0.10808** | **0.09898** | **0.10546** | **0.09644** | **0.11748** | **0.10058** | **0.11118** | **0.15113** | 0.10150 | 0.10299 | 0.09943 | **0.09575** | **0.09949** | **0.10212** | **0.11608** | **0.11245** | **0.23868** | **0.13532** | **0.11772** | **0.30636** | **0.37547** | **0.22183** | **0.15517** | **0.85975** | **0.55322** | **0.75442** |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.86836 | **0.86716** | 0.87231 | 0.87845 | **0.83811** | **0.84039** | **0.85810** | **0.83612** | **0.83747** | **0.85003** | **0.82730** | **0.82987** | **0.80003** | **0.84314** | 0.85257 | 0.86222 | 0.88006 | **0.81597** | **0.82019** | **0.80764** | **0.77712** | **0.76471** | **0.76476** | **0.78286** | **0.74488** | **0.72407** | **0.72773** | **0.76184** | **0.72008** | **0.71326** | **0.71440** |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.87793 | **0.87660** | 0.88181 | 0.88806 | **0.84737** | **0.84960** | **0.86765** | **0.84538** | **0.84665** | **0.85945** | **0.83649** | **0.83904** | **0.80884** | **0.85257** | 0.86183 | 0.87166 | 0.88967 | **0.82503** | **0.82920** | **0.81642** | **0.78568** | **0.77327** | **0.77307** | **0.79156** | **0.75314** | **0.73210** | **0.73566** | **0.77018** | **0.72811** | **0.72120** | **0.72226** |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.55037 | **0.53742** | 0.54168 | 0.55648 | **0.51058** | **0.51336** | **0.53579** | **0.50579** | **0.51272** | **0.53467** | **0.49352** | **0.50572** | **0.49957** | **0.52413** | 0.52563 | 0.53092 | 0.56245 | **0.50053** | **0.50900** | **0.50806** | **0.49148** | **0.45555** | **0.46054** | **0.48758** | **0.45435** | **0.42948** | **0.44315** | **0.47495** | **0.44937** | **0.41593** | **0.40648** |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.55216 | **0.53911** | 0.54345 | 0.55831 | **0.51227** | **0.51497** | **0.53752** | **0.50735** | **0.51423** | **0.53634** | **0.49509** | **0.50748** | **0.50130** | **0.52569** | 0.52732 | 0.53261 | 0.56420 | **0.50200** | **0.51061** | **0.50972** | **0.49306** | **0.45696** | **0.46204** | **0.48907** | **0.45587** | **0.43084** | **0.44456** | **0.47644** | **0.45086** | **0.41721** | **0.40770** |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | **0.78296** | 0.79651 | 0.79550 | 0.82352 | **0.76899** | **0.74861** | **0.77905** | **0.74417** | **0.76494** | **0.77105** | **0.72538** | **0.73008** | **0.68962** | **0.75765** | 0.78282 | 0.80396 | 0.81406 | **0.72465** | **0.71369** | **0.71471** | **0.67222** | **0.63954** | **0.64768** | **0.68590** | **0.62223** | **0.58970** | **0.59067** | **0.64692** | **0.58757** | **0.57347** | **0.57509** |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | **0.82267** | 0.83718 | 0.83570 | 0.86566 | **0.80804** | **0.78603** | **0.81906** | **0.78183** | **0.80405** | **0.80982** | **0.76183** | **0.76632** | **0.72411** | **0.79562** | 0.82213 | 0.84453 | 0.85553 | **0.76073** | **0.74879** | **0.75033** | **0.70499** | **0.67049** | **0.67931** | **0.72018** | **0.65195** | **0.61707** | **0.61838** | **0.67845** | **0.61493** | **0.59998** | **0.60125** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0140 | 1.0105 | **1.0037** | 1.0151 | **1.0033** | **1.0012** | **0.99135** | **0.98862** | **0.99053** | **1.0058** | **0.99485** | **0.99424** | **0.97176** | 1.0030 | **1.0026** | 1.0048 | 1.0190 | **0.99241** | **0.98485** | **0.98736** | **0.94989** | **0.93746** | **0.95042** | **0.96155** | **0.90603** | **0.87412** | **0.89229** | **0.93260** | **0.85853** | **0.84639** | **0.83548** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0279 | 2.0210 | **2.0073** | 2.0301 | **2.0065** | **2.0024** | **1.9827** | **1.9772** | **1.9811** | **2.0116** | **1.9897** | **1.9885** | **1.9435** | 2.0061 | **2.0052** | 2.0095 | 2.0381 | **1.9848** | **1.9697** | **1.9747** | **1.8998** | **1.8749** | **1.9008** | **1.9231** | **1.8121** | **1.7482** | **1.7846** | **1.8652** | **1.7171** | **1.6928** | **1.6710** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70285 | 0.70046 | **0.69574** | 0.70366 | **0.69547** | **0.69399** | **0.68723** | **0.68532** | **0.68664** | **0.69722** | **0.68959** | **0.68917** | **0.67363** | 0.69530 | **0.69499** | 0.69655 | 0.70634 | **0.68794** | **0.68267** | **0.68439** | **0.65848** | **0.64980** | **0.65880** | **0.66658** | **0.62802** | **0.60588** | **0.61850** | **0.64644** | **0.59510** | **0.58667** | **0.57912** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4057 | 1.4009 | **1.3915** | 1.4073 | **1.3909** | **1.3880** | **1.3745** | **1.3706** | **1.3733** | **1.3944** | **1.3792** | **1.3783** | **1.3473** | 1.3906 | **1.3900** | 1.3931 | 1.4127 | **1.3759** | **1.3653** | **1.3688** | **1.3170** | **1.2996** | **1.3176** | **1.3332** | **1.2560** | **1.2118** | **1.2370** | **1.2929** | **1.1902** | **1.1733** | **1.1582** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.30372 | **0.30840** | 0.30584 | 0.29692 | **0.31775** | **0.33433** | **0.30606** | **0.32795** | **0.31668** | **0.30925** | **0.31945** | **0.32221** | **0.34793** | **0.31647** | 0.29245 | 0.29734 | 0.27949 | **0.33369** | **0.33369** | **0.32965** | **0.36451** | **0.38895** | **0.36961** | **0.35005** | **0.40659** | **0.43379** | **0.42614** | **0.38193** | **0.44867** | **0.47078** | **0.47333** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.30372 | **0.30840** | 0.30584 | 0.29692 | **0.31775** | **0.33433** | **0.30606** | **0.32795** | **0.31668** | **0.30925** | **0.31945** | **0.32221** | **0.34793** | **0.31647** | 0.29245 | 0.29734 | 0.27949 | **0.33369** | **0.33369** | **0.32965** | **0.36451** | **0.38895** | **0.36961** | **0.35005** | **0.40659** | **0.43379** | **0.42614** | **0.38193** | **0.44867** | **0.47078** | **0.47333** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3651 | -1.3659 | -1.3620 | **-1.3703** | **-1.3494** | **-1.3445** | **-1.3593** | **-1.3473** | **-1.3506** | **-1.3569** | **-1.3523** | **-1.3415** | **-1.3260** | -1.3541 | -1.3649 | -1.3721 | **-1.3816** | **-1.3426** | **-1.3379** | **-1.3409** | **-1.3152** | **-1.2874** | **-1.3076** | **-1.3255** | **-1.2571** | **-1.2163** | **-1.2405** | **-1.2931** | **-1.2001** | **-1.1856** | **-1.1746** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4018 | -1.3968 | -1.3885 | **-1.4028** | **-1.3885** | **-1.3840** | **-1.3720** | **-1.3669** | **-1.3685** | **-1.3925** | **-1.3758** | **-1.3741** | **-1.3451** | -1.3880 | -1.3862 | -1.3872 | **-1.4071** | **-1.3733** | **-1.3631** | **-1.3665** | **-1.3140** | **-1.2975** | **-1.3158** | **-1.3307** | **-1.2535** | **-1.2099** | **-1.2341** | **-1.2917** | **-1.1887** | **-1.1720** | **-1.1567** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.26375 | 0.26213 | 0.26367 | **0.26109** | **0.27150** | **0.27302** | **0.26280** | **0.26902** | **0.26717** | **0.26722** | **0.26715** | **0.27319** | **0.27827** | 0.26822 | 0.26144 | 0.25772 | **0.25517** | **0.27267** | **0.27350** | **0.27227** | **0.27932** | **0.29315** | **0.28394** | **0.27624** | **0.30466** | **0.32291** | **0.31192** | **0.28866** | **0.33106** | **0.33681** | **0.34190** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.27511 | 0.27282 | 0.27513 | **0.27030** | **0.28643** | **0.28914** | **0.27368** | **0.28411** | **0.28038** | **0.27999** | **0.28104** | **0.29007** | **0.29900** | 0.28217 | 0.27108 | 0.26446 | **0.25962** | **0.28879** | **0.29132** | **0.28920** | **0.30256** | **0.32531** | **0.30847** | **0.29627** | **0.34378** | **0.36899** | **0.35506** | **0.31840** | **0.38192** | **0.39247** | **0.40097** |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | **1.0669** | 1.0977 | 1.0900 | 1.1418 | **1.0270** | **1.0239** | **1.0396** | **1.0066** | **1.0198** | **1.0393** | **0.98081** | **1.0069** | **0.91092** | **1.0253** | 1.0647 | 1.0766 | 1.1195 | **0.98509** | **0.96428** | **0.96320** | **0.91150** | **0.86539** | **0.86443** | **0.93405** | **0.85661** | **0.81237** | **0.79654** | **0.87534** | **0.79542** | **0.79131** | **0.78549** |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | **1.1859** | 1.2207 | 1.2095 | 1.2688 | **1.1396** | **1.1381** | **1.1535** | **1.1179** | **1.1315** | **1.1540** | **1.0893** | **1.1184** | **1.0117** | **1.1375** | 1.1842 | 1.1949 | 1.2442 | **1.0942** | **1.0710** | **1.0699** | **1.0132** | **0.96122** | **0.96260** | **1.0396** | **0.95356** | **0.90302** | **0.88611** | **0.97435** | **0.88508** | **0.88060** | **0.87378** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0191 | **0.98980** | 1.0121 | 1.0692 | **0.97898** | **0.97450** | **0.96428** | **0.95416** | **0.94646** | **1.0138** | **0.96706** | **0.97986** | **0.92342** | 0.99128 | **0.96519** | 0.98552 | 0.99369 | **0.97037** | **0.96312** | **0.91320** | **0.86258** | **0.84860** | **0.84803** | **0.89962** | **0.79070** | **0.73637** | **0.76859** | **0.85531** | **0.71001** | **0.68854** | **0.66845** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0382 | **1.9796** | 2.0241 | 2.1384 | **1.9580** | **1.9490** | **1.9286** | **1.9083** | **1.8929** | **2.0276** | **1.9341** | **1.9597** | **1.8468** | 1.9826 | **1.9304** | 1.9710 | 1.9874 | **1.9407** | **1.9262** | **1.8264** | **1.7252** | **1.6972** | **1.6961** | **1.7992** | **1.5814** | **1.4727** | **1.5372** | **1.7106** | **1.4200** | **1.3771** | **1.3369** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70637 | **0.68608** | 0.70151 | 0.74111 | **0.67858** | **0.67549** | **0.66841** | **0.66138** | **0.65604** | **0.70272** | **0.67028** | **0.67915** | **0.64004** | 0.68706 | **0.66904** | 0.68314 | 0.68873 | **0.67263** | **0.66761** | **0.63298** | **0.59789** | **0.58821** | **0.58782** | **0.62359** | **0.54808** | **0.51042** | **0.53274** | **0.59284** | **0.49215** | **0.47730** | **0.46335** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4127 | **1.3722** | 1.4030 | 1.4822 | **1.3572** | **1.3510** | **1.3368** | **1.3228** | **1.3121** | **1.4054** | **1.3406** | **1.3583** | **1.2801** | 1.3741 | **1.3381** | 1.3663 | 1.3775 | **1.3453** | **1.3352** | **1.2660** | **1.1958** | **1.1764** | **1.1756** | **1.2472** | **1.0962** | **1.0208** | **1.0655** | **1.1857** | **0.98430** | **0.95460** | **0.92670** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.33498 | **0.34392** | 0.33159 | 0.30444 | **0.37816** | **0.39636** | **0.36181** | **0.38279** | **0.38680** | **0.35996** | **0.37384** | **0.38217** | **0.43152** | **0.37662** | 0.34762 | 0.34917 | 0.31215 | **0.38310** | **0.40068** | **0.41147** | **0.45558** | **0.48149** | **0.48242** | **0.42844** | **0.52653** | **0.56200** | **0.54226** | **0.46545** | **0.57989** | **0.58914** | **0.61536** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.33498 | **0.34392** | 0.33159 | 0.30444 | **0.37816** | **0.39636** | **0.36181** | **0.38279** | **0.38680** | **0.35996** | **0.37384** | **0.38217** | **0.43152** | **0.37662** | 0.34762 | 0.34917 | 0.31215 | **0.38310** | **0.40068** | **0.41147** | **0.45558** | **0.48149** | **0.48242** | **0.42844** | **0.52653** | **0.56200** | **0.54226** | **0.46545** | **0.57989** | **0.58914** | **0.61536** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3408 | -1.3354 | -1.3423 | **-1.3714** | **-1.3001** | **-1.2818** | **-1.3215** | **-1.2934** | **-1.2947** | **-1.3122** | **-1.3001** | **-1.2891** | **-1.2393** | -1.3115 | -1.3257 | -1.3393 | **-1.3599** | **-1.2973** | **-1.2734** | **-1.2478** | **-1.2161** | **-1.1797** | **-1.1926** | **-1.2353** | **-1.1226** | **-1.0501** | **-1.0972** | **-1.1956** | **-1.0266** | **-1.0082** | **-0.99258** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4107 | -1.3684 | -1.4002 | **-1.4793** | **-1.3553** | **-1.3473** | **-1.3340** | **-1.3201** | **-1.3091** | **-1.4038** | **-1.3385** | **-1.3542** | **-1.2777** | -1.3722 | -1.3349 | -1.3610 | **-1.3726** | **-1.3432** | **-1.3333** | **-1.2646** | **-1.1940** | **-1.1746** | **-1.1739** | **-1.2449** | **-1.0944** | **-1.0191** | **-1.0634** | **-1.1845** | **-0.98241** | **-0.95312** | **-0.92485** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.27897 | 0.27525 | 0.27650 | **0.27200** | **0.29578** | **0.30568** | **0.27856** | **0.29353** | **0.29043** | **0.29570** | **0.29223** | **0.30187** | **0.32039** | 0.29071 | 0.27603 | 0.27227 | **0.26187** | **0.29501** | **0.30829** | **0.31283** | **0.32058** | **0.34173** | **0.33246** | **0.31759** | **0.36524** | **0.40000** | **0.37614** | **0.33205** | **0.40921** | **0.41695** | **0.42272** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.30023 | 0.29589 | 0.29628 | **0.28652** | **0.32837** | **0.34154** | **0.30206** | **0.32664** | **0.32053** | **0.32616** | **0.32293** | **0.33567** | **0.36493** | 0.32001 | 0.29797 | 0.29051 | **0.27234** | **0.32692** | **0.34805** | **0.35522** | **0.37048** | **0.39736** | **0.38498** | **0.36237** | **0.43184** | **0.47611** | **0.44740** | **0.38661** | **0.49217** | **0.50112** | **0.51621** |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | **0.89880** | 0.91745 | 0.92155 | 0.96177 | **0.87940** | **0.85948** | **0.88692** | **0.86359** | **0.85731** | **0.87537** | **0.84055** | **0.84227** | **0.79046** | **0.87903** | 0.90194 | 0.92370 | 0.94024 | **0.84500** | **0.82307** | **0.82976** | **0.78734** | **0.73295** | **0.74372** | **0.79174** | **0.73674** | **0.69692** | **0.68986** | **0.75585** | **0.68824** | **0.67663** | **0.67275** |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | **0.95927** | 0.97894 | 0.98270 | 1.0272 | **0.93907** | **0.91644** | **0.94641** | **0.92226** | **0.91485** | **0.93420** | **0.89757** | **0.89995** | **0.84343** | **0.93871** | 0.96311 | 0.98695 | 1.0029 | **0.90147** | **0.87879** | **0.88665** | **0.84102** | **0.78156** | **0.79366** | **0.84500** | **0.78560** | **0.74294** | **0.73484** | **0.80673** | **0.73394** | **0.72093** | **0.71660** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0384 | **1.0264** | 1.0563 | 1.0903 | **1.0398** | **1.0104** | **0.98032** | **0.97261** | **0.99458** | **1.0375** | **0.99070** | **1.0279** | **0.95280** | 1.0296 | **0.98518** | 0.99769 | 1.0209 | **0.99984** | **0.98392** | **0.94297** | **0.86919** | **0.84204** | **0.87254** | **0.90786** | **0.79273** | **0.74193** | **0.77313** | **0.87363** | **0.70609** | **0.66075** | **0.65703** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0767 | **2.0527** | 2.1126 | 2.1806 | **2.0797** | **2.0209** | **1.9606** | **1.9452** | **1.9892** | **2.0751** | **1.9814** | **2.0557** | **1.9056** | 2.0593 | **1.9704** | 1.9954 | 2.0418 | **1.9997** | **1.9678** | **1.8859** | **1.7384** | **1.6841** | **1.7451** | **1.8157** | **1.5855** | **1.4839** | **1.5463** | **1.7473** | **1.4122** | **1.3215** | **1.3141** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.71973 | **0.71143** | 0.73218 | 0.75573 | **0.72075** | **0.70034** | **0.67950** | **0.67419** | **0.68942** | **0.71917** | **0.68668** | **0.71251** | **0.66045** | 0.71368 | **0.68286** | 0.69157 | 0.70765 | **0.69306** | **0.68205** | **0.65363** | **0.60245** | **0.58362** | **0.60474** | **0.62928** | **0.54950** | **0.51430** | **0.53590** | **0.60558** | **0.48944** | **0.45801** | **0.45538** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4395 | **1.4229** | 1.4644 | 1.5115 | **1.4415** | **1.4007** | **1.3590** | **1.3484** | **1.3788** | **1.4383** | **1.3734** | **1.4250** | **1.3209** | 1.4274 | **1.3657** | 1.3831 | 1.4153 | **1.3861** | **1.3641** | **1.3073** | **1.2049** | **1.1672** | **1.2095** | **1.2586** | **1.0990** | **1.0286** | **1.0718** | **1.2112** | **0.97889** | **0.91602** | **0.91076** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.32792 | **0.33019** | 0.32272 | 0.29802 | **0.35034** | **0.37732** | **0.35912** | **0.37732** | **0.37894** | **0.35944** | **0.36822** | **0.37342** | **0.42801** | 0.35132 | **0.35197** | 0.33117 | 0.30907 | **0.37634** | **0.39097** | **0.38869** | **0.45954** | **0.48391** | **0.46474** | **0.43126** | **0.52779** | **0.56711** | **0.53819** | **0.45044** | **0.59084** | **0.61326** | **0.62821** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.32792 | **0.33019** | 0.32272 | 0.29802 | **0.35034** | **0.37732** | **0.35912** | **0.37732** | **0.37894** | **0.35944** | **0.36822** | **0.37342** | **0.42801** | 0.35132 | **0.35197** | 0.33117 | 0.30907 | **0.37634** | **0.39097** | **0.38869** | **0.45954** | **0.48391** | **0.46474** | **0.43126** | **0.52779** | **0.56711** | **0.53819** | **0.45044** | **0.59084** | **0.61326** | **0.62821** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3612 | -1.3623 | -1.3677 | **-1.3944** | **-1.3405** | **-1.3136** | **-1.3352** | **-1.3111** | **-1.3220** | **-1.3426** | **-1.3236** | **-1.3269** | **-1.2731** | -1.3393 | -1.3378 | -1.3526 | **-1.3753** | **-1.3201** | **-1.3015** | **-1.2822** | **-1.2255** | **-1.1816** | **-1.2119** | **-1.2521** | **-1.1320** | **-1.0536** | **-1.1025** | **-1.2178** | **-1.0223** | **-0.98556** | **-0.96835** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4373 | -1.4192 | -1.4615 | **-1.5086** | **-1.4396** | **-1.3971** | **-1.3565** | **-1.3456** | **-1.3755** | **-1.4368** | **-1.3712** | **-1.4205** | **-1.3193** | **-1.4253** | -1.3623 | -1.3773 | -1.4108 | **-1.3842** | **-1.3621** | **-1.3058** | **-1.2028** | **-1.1657** | **-1.2082** | **-1.2562** | **-1.0975** | **-1.0271** | **-1.0702** | **-1.2101** | **-0.97745** | **-0.91486** | **-0.90932** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.27150 | 0.26815 | 0.27169 | **0.26412** | **0.28542** | **0.29498** | **0.27415** | **0.28635** | **0.28588** | **0.28309** | **0.28346** | **0.29129** | **0.30616** | 0.28283 | 0.27320 | 0.26743 | **0.25966** | **0.28795** | **0.29516** | **0.29690** | **0.31455** | **0.33648** | **0.32408** | **0.30703** | **0.35701** | **0.39601** | **0.37054** | **0.32103** | **0.40969** | **0.42232** | **0.43308** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.28865 | 0.28331 | 0.28766 | **0.27504** | **0.30983** | **0.32632** | **0.29465** | **0.31504** | **0.31368** | **0.30810** | **0.31036** | **0.32131** | **0.34668** | 0.30623 | 0.29280 | 0.28209 | **0.26869** | **0.31590** | **0.32834** | **0.33228** | **0.36270** | **0.39566** | **0.37493** | **0.34807** | **0.42655** | **0.47516** | **0.44506** | **0.37222** | **0.49886** | **0.51836** | **0.53196** |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | **1.2301** | 1.2345 | 1.2375 | 1.2881 | **1.1723** | **1.1623** | **1.2122** | **1.1744** | **1.1577** | **1.2313** | **1.1517** | **1.1543** | **1.1049** | **1.1917** | 1.2220 | 1.2312 | 1.2760 | **1.1514** | **1.1439** | **1.1453** | **1.0799** | **1.0096** | **1.0338** | **1.0894** | **1.0331** | **0.96780** | **0.96033** | **1.0551** | **0.96061** | **0.91132** | **0.91010** |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | **1.5202** | 1.5239 | 1.5253 | 1.5865 | **1.4459** | **1.4340** | **1.4961** | **1.4512** | **1.4240** | **1.5205** | **1.4232** | **1.4247** | **1.3651** | **1.4715** | 1.5064 | 1.5116 | 1.5731 | **1.4218** | **1.4118** | **1.4187** | **1.3357** | **1.2471** | **1.2798** | **1.3467** | **1.2777** | **1.1963** | **1.1864** | **1.3053** | **1.1857** | **1.1199** | **1.1225** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0378 | **1.0118** | 1.0235 | 1.0552 | **1.0106** | **1.0123** | **0.99324** | **1.0068** | **1.0004** | **1.0068** | **1.0133** | **1.0300** | **0.98493** | **0.99968** | 1.0083 | 1.0006 | 1.0242 | **0.99712** | **1.0046** | **0.96969** | **0.94006** | **0.92278** | **0.93720** | **0.96582** | **0.90603** | **0.85889** | **0.90107** | **0.96258** | **0.85123** | **0.81525** | **0.79980** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0756 | **2.0236** | 2.0470 | 2.1105 | **2.0211** | **2.0245** | **1.9865** | **2.0136** | **2.0007** | **2.0136** | **2.0265** | **2.0600** | **1.9699** | **1.9994** | 2.0165 | 2.0011 | 2.0485 | **1.9942** | **2.0092** | **1.9394** | **1.8801** | **1.8456** | **1.8744** | **1.9316** | **1.8121** | **1.7178** | **1.8021** | **1.9252** | **1.7025** | **1.6305** | **1.5996** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.71932 | **0.70134** | 0.70949 | 0.73145 | **0.70048** | **0.70164** | **0.68850** | **0.69788** | **0.69341** | **0.69788** | **0.70233** | **0.71397** | **0.68273** | **0.69292** | 0.69893 | 0.69353 | 0.70996 | **0.69119** | **0.69636** | **0.67214** | **0.65158** | **0.63960** | **0.64962** | **0.66951** | **0.62799** | **0.59536** | **0.62460** | **0.66715** | **0.59004** | **0.56506** | **0.55436** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4386 | **1.4027** | 1.4190 | 1.4629 | **1.4010** | **1.4033** | **1.3770** | **1.3958** | **1.3868** | **1.3958** | **1.4047** | **1.4279** | **1.3655** | **1.3858** | 1.3979 | 1.3871 | 1.4199 | **1.3824** | **1.3927** | **1.3443** | **1.3032** | **1.2792** | **1.2992** | **1.3390** | **1.2560** | **1.1907** | **1.2492** | **1.3343** | **1.1801** | **1.1301** | **1.1087** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.29125 | **0.30020** | 0.29920 | 0.27767 | **0.31610** | **0.32671** | **0.31345** | **0.32140** | **0.32538** | **0.32538** | **0.31677** | **0.30948** | **0.35454** | **0.32074** | 0.30616 | 0.30020 | 0.27005 | **0.33068** | **0.33135** | **0.34294** | **0.37806** | **0.40060** | **0.38370** | **0.34361** | **0.40557** | **0.45096** | **0.42445** | **0.36746** | **0.46289** | **0.49139** | **0.49867** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.29125 | **0.30020** | 0.29920 | 0.27767 | **0.31610** | **0.32671** | **0.31345** | **0.32140** | **0.32538** | **0.32538** | **0.31677** | **0.30948** | **0.35454** | **0.32074** | 0.30616 | 0.30020 | 0.27005 | **0.33068** | **0.33135** | **0.34294** | **0.37806** | **0.40060** | **0.38370** | **0.34361** | **0.40557** | **0.45096** | **0.42445** | **0.36746** | **0.46289** | **0.49139** | **0.49867** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3861 | -1.3745 | -1.3776 | **-1.3959** | **-1.3617** | **-1.3515** | **-1.3632** | **-1.3554** | **-1.3584** | **-1.3601** | **-1.3645** | **-1.3581** | **-1.3294** | -1.3620 | -1.3723 | -1.3712 | **-1.3871** | **-1.3544** | **-1.3485** | **-1.3362** | **-1.3133** | **-1.2803** | **-1.2975** | **-1.3274** | **-1.2658** | **-1.2026** | **-1.2432** | **-1.3119** | **-1.1973** | **-1.1610** | **-1.1462** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4346 | -1.3979 | -1.4156 | **-1.4585** | **-1.3980** | **-1.3973** | **-1.3728** | **-1.3921** | **-1.3815** | **-1.3933** | **-1.4007** | **-1.4209** | **-1.3631** | -1.3830 | -1.3928 | -1.3800 | **-1.4141** | **-1.3797** | **-1.3900** | **-1.3415** | **-1.3001** | **-1.2766** | **-1.2969** | **-1.3354** | **-1.2537** | **-1.1878** | **-1.2467** | **-1.3324** | **-1.1781** | **-1.1276** | **-1.1060** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25687 | 0.25771 | 0.25882 | **0.25535** | **0.26497** | **0.27107** | **0.26035** | **0.26780** | **0.26417** | **0.26519** | **0.26395** | **0.27057** | **0.27858** | 0.26261 | 0.25829 | 0.25701 | **0.25333** | **0.26655** | **0.27153** | **0.27084** | **0.27794** | **0.29395** | **0.28671** | **0.27496** | **0.29964** | **0.32743** | **0.31220** | **0.28407** | **0.33094** | **0.34439** | **0.35150** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.26331 | 0.26518 | 0.26733 | **0.26074** | **0.27716** | **0.28702** | **0.27015** | **0.28348** | **0.27630** | **0.27830** | **0.27597** | **0.28646** | **0.30036** | 0.27395 | 0.26628 | 0.26379 | **0.25651** | **0.28085** | **0.28943** | **0.28851** | **0.30196** | **0.32641** | **0.31503** | **0.29530** | **0.33560** | **0.37494** | **0.35289** | **0.30995** | **0.38375** | **0.40339** | **0.41426** |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.41602 | **0.40476** | 0.41797 | 0.43233 | **0.39978** | **0.39802** | **0.43358** | **0.39989** | **0.39911** | **0.42343** | **0.38365** | **0.40020** | **0.35948** | 0.41970 | **0.41635** | 0.43091 | 0.43435 | **0.39802** | **0.38262** | **0.38996** | **0.36766** | **0.35095** | **0.36118** | **0.36038** | **0.35067** | **0.34141** | **0.31818** | **0.35559** | **0.33180** | **0.30893** | **0.30397** |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.41840 | **0.40708** | 0.42028 | 0.43482 | **0.40197** | **0.40022** | **0.43627** | **0.40225** | **0.40145** | **0.42587** | **0.38574** | **0.40257** | **0.36150** | 0.42216 | **0.41884** | 0.43342 | 0.43684 | **0.40021** | **0.38484** | **0.39224** | **0.36973** | **0.35300** | **0.36328** | **0.36245** | **0.35264** | **0.34341** | **0.32002** | **0.35772** | **0.33375** | **0.31077** | **0.30583** |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.31670 | **0.31553** | 0.33110 | 0.32565 | **0.30698** | **0.30638** | **0.32586** | **0.29704** | **0.29698** | **0.30789** | **0.29934** | **0.29918** | **0.28491** | **0.31954** | 0.32661 | 0.32116 | 0.32629 | **0.31106** | **0.28524** | **0.29051** | **0.27559** | **0.25530** | **0.27748** | **0.28189** | **0.26167** | **0.25505** | **0.24110** | **0.27178** | **0.24845** | **0.23986** | **0.23155** |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.31796 | **0.31677** | 0.33242 | 0.32693 | **0.30819** | **0.30758** | **0.32717** | **0.29818** | **0.29807** | **0.30904** | **0.30047** | **0.30038** | **0.28606** | **0.32082** | 0.32792 | 0.32241 | 0.32758 | **0.31234** | **0.28637** | **0.29159** | **0.27668** | **0.25633** | **0.27856** | **0.28297** | **0.26272** | **0.25605** | **0.24205** | **0.27284** | **0.24939** | **0.24080** | **0.23246** |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | **0.55095** | 0.56554 | 0.56977 | 0.56677 | **0.50553** | **0.51550** | **0.57295** | **0.50371** | **0.52026** | **0.63708** | **0.52946** | **0.53293** | **0.48655** | **0.56326** | 0.59465 | 0.59852 | 0.60544 | **0.50654** | **0.50202** | **0.50293** | **0.46196** | **0.47153** | **0.46899** | **0.48141** | **0.43852** | **0.45513** | **0.41436** | **0.46626** | **0.42784** | **0.40649** | **0.40789** |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | **0.55497** | 0.56944 | 0.57375 | 0.57075 | **0.50915** | **0.51917** | **0.57716** | **0.50721** | **0.52381** | **0.64174** | **0.53329** | **0.53673** | **0.49010** | **0.56714** | 0.59892 | 0.60280 | 0.60976 | **0.50996** | **0.50553** | **0.50653** | **0.46524** | **0.47483** | **0.47232** | **0.48491** | **0.44165** | **0.45847** | **0.41723** | **0.46953** | **0.43097** | **0.40944** | **0.41078** |
| eval/lm/c4_en-validation/CE loss | lower | **3.2356** | 3.2485 | 3.2792 | 3.3284 | **3.1582** | **3.1598** | **3.2094** | **3.1309** | **3.1347** | **3.1811** | **3.0943** | **3.1076** | **3.0023** | **3.1674** | 3.2063 | 3.2482 | 3.3123 | **3.0723** | **3.0815** | **3.0430** | **2.9313** | **2.8505** | **2.8658** | **2.9192** | **2.7853** | **2.7133** | **2.7157** | **2.8370** | **2.6868** | **2.6784** | **2.6735** |
| eval/lm/c4_en-validation/PPL | lower | **25.42** | 25.75 | 26.55 | 27.89 | **23.53** | **23.57** | **24.76** | **22.89** | **22.98** | **24.07** | **22.07** | **22.37** | **20.13** | **23.75** | 24.69 | 25.74 | 27.45 | **21.59** | **21.79** | **20.97** | **18.75** | **17.30** | **17.56** | **18.53** | **16.20** | **15.08** | **15.12** | **17.06** | **14.69** | **14.56** | **14.49** |
| eval/lm/dolma_books-validation/CE loss | lower | **3.1928** | 3.2103 | 3.2560 | 3.3148 | **3.1113** | **3.1079** | **3.1696** | **3.0750** | **3.0807** | **3.1236** | **3.0331** | **3.0466** | **2.9199** | **3.1079** | 3.1654 | 3.2248 | 3.3054 | **3.0007** | **3.0084** | **2.9616** | **2.8301** | **2.7326** | **2.7487** | **2.8150** | **2.6469** | **2.5665** | **2.5594** | **2.7054** | **2.5323** | **2.5262** | **2.5313** |
| eval/lm/dolma_books-validation/PPL | lower | **24.36** | 24.79 | 25.95 | 27.52 | **22.45** | **22.37** | **23.80** | **21.65** | **21.77** | **22.73** | **20.76** | **21.04** | **18.54** | **22.37** | 23.70 | 25.15 | 27.26 | **20.10** | **20.26** | **19.33** | **16.95** | **15.37** | **15.62** | **16.69** | **14.11** | **13.02** | **12.93** | **14.96** | **12.58** | **12.51** | **12.57** |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | **3.3697** | 3.3862 | 3.4172 | 3.4667 | **3.2966** | **3.2963** | **3.3451** | **3.2681** | **3.2743** | **3.3141** | **3.2320** | **3.2404** | **3.1443** | **3.3008** | 3.3388 | 3.3836 | 3.4499 | **3.2107** | **3.2188** | **3.1785** | **3.0702** | **2.9924** | **3.0039** | **3.0563** | **2.9232** | **2.8540** | **2.8551** | **2.9752** | **2.8270** | **2.8190** | **2.8151** |
| eval/lm/dolma_common-crawl-validation/PPL | lower | **29.07** | 29.55 | 30.48 | 32.03 | **27.02** | **27.01** | **28.36** | **26.26** | **26.43** | **27.50** | **25.33** | **25.54** | **23.20** | **27.13** | 28.19 | 29.48 | 31.50 | **24.80** | **25.00** | **24.01** | **21.55** | **19.93** | **20.16** | **21.25** | **18.60** | **17.36** | **17.38** | **19.59** | **16.90** | **16.76** | **16.69** |
| eval/lm/dolma_pes2o-validation/CE loss | lower | **2.3417** | 2.3491 | 2.3714 | 2.4140 | **2.2746** | **2.2808** | **2.3215** | **2.2540** | **2.2609** | **2.2931** | **2.2255** | **2.2352** | **2.1560** | **2.2791** | 2.3093 | 2.3463 | 2.3977 | **2.2033** | **2.2151** | **2.1844** | **2.0952** | **2.0332** | **2.0478** | **2.0874** | **1.9806** | **1.9269** | **1.9329** | **2.0245** | **1.9040** | **1.8974** | **1.8959** |
| eval/lm/dolma_pes2o-validation/PPL | lower | **10.40** | 10.48 | 10.71 | 11.18 | **9.7243** | **9.7846** | **10.19** | **9.5257** | **9.5921** | **9.9056** | **9.2577** | **9.3486** | **8.6364** | **9.7681** | 10.07 | 10.45 | 11.00 | **9.0551** | **9.1624** | **8.8854** | **8.1267** | **7.6388** | **7.7508** | **8.0636** | **7.2472** | **6.8682** | **6.9094** | **7.5727** | **6.7126** | **6.6688** | **6.6583** |
| eval/lm/dolma_reddit-validation/CE loss | lower | **3.5102** | 3.5252 | 3.5447 | 3.5906 | **3.4397** | **3.4372** | **3.4879** | **3.4152** | **3.4167** | **3.4557** | **3.3805** | **3.3867** | **3.2982** | **3.4465** | 3.4852 | 3.5175 | 3.5751 | **3.3577** | **3.3678** | **3.3357** | **3.2377** | **3.1718** | **3.1754** | **3.2255** | **3.1037** | **3.0402** | **3.0438** | **3.1475** | **3.0142** | **3.0091** | **3.0062** |
| eval/lm/dolma_reddit-validation/PPL | lower | **33.45** | 33.96 | 34.63 | 36.25 | **31.18** | **31.10** | **32.72** | **30.42** | **30.47** | **31.68** | **29.39** | **29.57** | **27.06** | **31.39** | 32.63 | 33.70 | 35.70 | **28.72** | **29.01** | **28.10** | **25.47** | **23.85** | **23.94** | **25.17** | **22.28** | **20.91** | **20.98** | **23.28** | **20.37** | **20.27** | **20.21** |
| eval/lm/dolma_stack-validation/CE loss | lower | **1.4795** | 1.4831 | 1.5073 | 1.5466 | **1.4285** | **1.4307** | **1.4662** | **1.4115** | **1.4201** | **1.4501** | **1.3913** | **1.4010** | **1.3381** | **1.4386** | 1.4636 | 1.4942 | 1.5425 | **1.3767** | **1.3838** | **1.3450** | **1.2796** | **1.2267** | **1.2377** | **1.2678** | **1.1794** | **1.1342** | **1.1349** | **1.2115** | **1.1133** | **1.1335** | **1.1314** |
| eval/lm/dolma_stack-validation/PPL | lower | **4.3905** | 4.4068 | 4.5147 | 4.6957 | **4.1723** | **4.1817** | **4.3329** | **4.1020** | **4.1375** | **4.2634** | **4.0200** | **4.0594** | **3.8118** | **4.2148** | 4.3215 | 4.4557 | 4.6764 | **3.9617** | **3.9899** | **3.8383** | **3.5953** | **3.4100** | **3.4478** | **3.5530** | **3.2523** | **3.1086** | **3.1109** | **3.3584** | **3.0445** | **3.1066** | **3.0999** |
| eval/lm/dolma_wiki-validation/CE loss | lower | **2.7348** | 2.7508 | 2.7853 | 2.8424 | **2.6475** | **2.6437** | **2.7069** | **2.6209** | **2.6236** | **2.6759** | **2.5867** | **2.5932** | **2.4987** | **2.6669** | 2.7044 | 2.7577 | 2.8251 | **2.5643** | **2.5691** | **2.5321** | **2.4316** | **2.3426** | **2.3502** | **2.4145** | **2.2942** | **2.2141** | **2.2120** | **2.3480** | **2.2009** | **2.1709** | **2.1638** |
| eval/lm/dolma_wiki-validation/PPL | lower | **15.41** | 15.66 | 16.20 | 17.16 | **14.12** | **14.06** | **14.98** | **13.75** | **13.79** | **14.53** | **13.29** | **13.37** | **12.17** | **14.39** | 14.94 | 15.76 | 16.86 | **12.99** | **13.05** | **12.58** | **11.38** | **10.41** | **10.49** | **11.18** | **9.9162** | **9.1533** | **9.1336** | **10.46** | **9.0332** | **8.7663** | **8.7038** |
| eval/lm/ice-validation/CE loss | lower | **3.2444** | 3.2517 | 3.2919 | 3.3105 | **3.1802** | **3.1770** | **3.2260** | **3.1565** | **3.1707** | **3.1878** | **3.1164** | **3.1204** | **3.0232** | **3.1703** | 3.1970 | 3.2510 | 3.2843 | **3.0762** | **3.1013** | **3.0424** | **2.9562** | **2.8866** | **2.8938** | **2.9221** | **2.7842** | **2.7255** | **2.7431** | **2.8352** | **2.6958** | **2.7155** | **2.7179** |
| eval/lm/ice-validation/PPL | lower | **25.65** | 25.83 | 26.89 | 27.40 | **24.05** | **23.97** | **25.18** | **23.49** | **23.82** | **24.24** | **22.57** | **22.65** | **20.56** | **23.81** | 24.46 | 25.82 | 26.69 | **21.68** | **22.23** | **20.96** | **19.22** | **17.93** | **18.06** | **18.58** | **16.19** | **15.26** | **15.54** | **17.03** | **14.82** | **15.11** | **15.15** |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | **3.1334** | 3.1408 | 3.1564 | 3.1969 | **3.0651** | **3.0656** | **3.1183** | **3.0544** | **3.0527** | **3.1016** | **3.0342** | **3.0375** | **2.9695** | **3.0913** | 3.1138 | 3.1538 | 3.1810 | **3.0154** | **3.0195** | **2.9925** | **2.9291** | **2.8592** | **2.8668** | **2.8938** | **2.8119** | **2.7536** | **2.7585** | **2.8495** | **2.7426** | **2.7206** | **2.7073** |
| eval/lm/m2d2_s2orc-validation/PPL | lower | **22.95** | 23.12 | 23.49 | 24.46 | **21.44** | **21.45** | **22.61** | **21.21** | **21.17** | **22.23** | **20.78** | **20.85** | **19.48** | **22.01** | 22.51 | 23.42 | 24.07 | **20.40** | **20.48** | **19.94** | **18.71** | **17.45** | **17.58** | **18.06** | **16.64** | **15.70** | **15.78** | **17.28** | **15.53** | **15.19** | **14.99** |
| eval/lm/pile-validation/CE loss | lower | **2.4637** | 2.4738 | 2.5014 | 2.5457 | **2.3940** | **2.3949** | **2.4397** | **2.3687** | **2.3765** | **2.4114** | **2.3336** | **2.3476** | **2.2559** | **2.4003** | 2.4338 | 2.4773 | 2.5341 | **2.3142** | **2.3247** | **2.2898** | **2.1911** | **2.1192** | **2.1336** | **2.1793** | **2.0570** | **1.9964** | **2.0002** | **2.1068** | **1.9696** | **1.9650** | **1.9632** |
| eval/lm/pile-validation/PPL | lower | **11.75** | 11.87 | 12.20 | 12.75 | **10.96** | **10.97** | **11.47** | **10.68** | **10.77** | **11.15** | **10.31** | **10.46** | **9.5439** | **11.03** | 11.40 | 11.91 | 12.60 | **10.12** | **10.22** | **9.8728** | **8.9450** | **8.3243** | **8.4451** | **8.8398** | **7.8226** | **7.3626** | **7.3906** | **8.2218** | **7.1680** | **7.1348** | **7.1219** |
| eval/lm/wikitext_103-validation/CE loss | lower | **2.7592** | 2.7723 | 2.8083 | 2.8554 | **2.6666** | **2.6631** | **2.7282** | **2.6371** | **2.6441** | **2.6902** | **2.5884** | **2.6050** | **2.4899** | **2.6786** | 2.7229 | 2.7714 | 2.8426 | **2.5603** | **2.5658** | **2.5485** | **2.3951** | **2.2958** | **2.3192** | **2.3891** | **2.2333** | **2.1446** | **2.1491** | **2.3020** | **2.1266** | **2.0823** | **2.0800** |
| eval/lm/wikitext_103-validation/PPL | lower | **15.79** | 16.00 | 16.58 | 17.38 | **14.39** | **14.34** | **15.30** | **13.97** | **14.07** | **14.73** | **13.31** | **13.53** | **12.06** | **14.57** | 15.22 | 15.98 | 17.16 | **12.94** | **13.01** | **12.79** | **10.97** | **9.9326** | **10.17** | **10.90** | **9.3310** | **8.5383** | **8.5770** | **9.9942** | **8.3861** | **8.0229** | **8.0041** |
