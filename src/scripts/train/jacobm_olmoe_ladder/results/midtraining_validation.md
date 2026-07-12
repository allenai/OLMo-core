# Midtraining Validation Results

Generated: 2026-07-12 02:06 UTC

Interpretation: lower is better for CE loss, PPL, Z loss, router Z loss, and load-balancing loss; higher is better for MFU/TPS. Accuracy-style validation metrics are higher-is-better when present.

Selection rule: use only `eval/*` validation metrics for midtraining checkpoint/LR selection. Training loss on the midtraining mixture is shown only as run-health metadata and must not be used to choose LRs.

Backfill note: the first 275M grid did not run in-loop evals during training, so final-checkpoint eval-only backfills are required. Once those eval jobs finish and `copy_eval_backfills_to_wandb.py` copies their metrics back, this table will populate the `eval/*` section.

Settings: 100B midtraining tokens, sequence length 8192, global batch seq 128 (1,048,576 tokens), 1 node, 4 GPUs, EP1, microbatch 8, fresh optimizer state, 2000-step warmup then constant LR.

| source | training finished | eval metrics present | LRs with evals | still running |
| --- | --- | --- | --- | --- |
| 275M baseline Cx1 | 4/4 | 4/4 | 2e-4, 4e-4, 8e-4, 1.6e-3 |  |
| 275M baseline Cx2 | 1/1 | 1/1 | 1.8e-4 |  |
| 275M baseline Cx4 | 1/1 | 1/1 | 1.5e-4 |  |
| 275M baseline Cx8 | 4/4 | 4/4 | 2e-4, 4e-4, 8e-4, 1.6e-3 |  |
| 480M baseline Cx1 | 1/1 | 1/1 | 1.2e-4 |  |
| 480M baseline Cx8 | 1/1 | 1/1 | 8e-5 |  |
| 810M baseline Cx1 | 1/1 | 1/1 | 6e-5 |  |
| 810M baseline Cx8 | 1/1 | 1/1 | 4e-5 |  |
| 1.2B baseline Cx1 | 1/1 | 1/1 | 4e-5 |  |
| 1.2B baseline Cx8 | 1/1 | 1/1 | 4e-5 |  |

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

## 275M baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.6064 | 4.9848 | 0.00083 | 0.00013 | 0.11042 | 29.30 | 363861.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cm8ww646) [Beaker](https://beaker.org/ex/01KWWM1043JEC9MC3PV7PXQ745) |
| 4e-4 | finished | 100.00B | 1.6162 | 5.0340 | 0.00095 | 0.00020 | 0.11038 | 29.36 | 364560.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r6ts032g) [Beaker](https://beaker.org/ex/01KWWM1AVQXMJ3JBJQ1W2G8YAV) |
| 8e-4 | finished | 100.00B | 1.6446 | 5.1791 | 0.00110 | 0.00027 | 0.11048 | 29.01 | 360250.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lfydkxv4) [Beaker](https://beaker.org/ex/01KWWM1N0EQDMCFER90NVP9QW0) |
| 1.6e-3 | finished | 100.00B | 1.6966 | 5.4554 | 0.00121 | 0.00036 | 0.11070 | 29.20 | 362582.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w3vof8b9) [Beaker](https://beaker.org/ex/01KWWM1ZXN5R5XWK00GH0WA36G) |

## 275M baseline Cx2 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.8e-4 | finished | 100.00B | 1.5903 | 4.9054 | 0.00089 | 0.00013 | 0.11041 | 29.14 | 361891.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kpkhplj7) [Beaker](https://beaker.org/ex/01KWZ8T9BZ3B869VZ878FNNQ8T) |

## 275M baseline Cx4 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5e-4 | finished | 100.00B | 1.5736 | 4.8240 | 0.00087 | 0.00013 | 0.11039 | 29.14 | 361830.2 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u508l6x8) [Beaker](https://beaker.org/ex/01KWZ8T9DZN8Y4VD63AP1AN387) |

## 275M baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.5642 | 4.7789 | 0.00084 | 0.00015 | 0.11033 | 29.36 | 364568.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4amcsbx6) [Beaker](https://beaker.org/ex/01KWWM10ANMMW2YTNN6RKJBGE7) |
| 4e-4 | finished | 100.00B | 1.5924 | 4.9157 | 0.00101 | 0.00021 | 0.11038 | 29.35 | 364491.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8jdqtmgg) [Beaker](https://beaker.org/ex/01KWWM1AK89SKDA1KGCX5D8SMM) |
| 8e-4 | finished | 100.00B | 1.6283 | 5.0952 | 0.00111 | 0.00028 | 0.11047 | 29.19 | 362440.4 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/drm1ceit) [Beaker](https://beaker.org/ex/01KWWM1P9TXRSMDV5DH9EF4KXM) |
| 1.6e-3 | finished | 100.00B | 1.6843 | 5.3887 | 0.00129 | 0.00037 | 0.11067 | 29.06 | 360889.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/edljug2e) [Beaker](https://beaker.org/ex/01KWWM213KMG47KADPK5Q67GJP) |

## 480M baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.2e-4 | finished | 100.00B | 1.5472 | 4.6984 | 0.00094 | 0.00009 | 0.15040 | 32.02 | 233360.8 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ro38bhjl) [Beaker](https://beaker.org/ex/01KWZARWH7XAS4MD2238VRKP0Y) |

## 480M baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8e-5 | finished | 100.00B | 1.4834 | 4.4081 | 0.00101 | 0.00008 | 0.15011 | 31.94 | 232761.3 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6pd01ptm) [Beaker](https://beaker.org/ex/01KWZARZ2T7FZDH42Q2VS92WXN) |

## 810M baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6e-5 | finished | 100.00B | 1.4171 | 4.1251 | 0.00060 | 0.00007 | 0.19036 | 31.43 | 125162.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7ylgav9r) [Beaker](https://beaker.org/ex/01KWZAT4PF87PYA96JT7ZXSQKX) |

## 810M baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.3399 | 3.8186 | 0.00068 | 0.00007 | 0.19004 | 31.51 | 125499.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zeia5b0a) [Beaker](https://beaker.org/ex/01KWZAT4PQ0NWD20VS0XT12ZF5) |

## 1.2B baseline Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.3877 | 4.0056 | 0.00082 | 0.00007 | 0.21038 | 34.41 | 90037.1 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5utsk31j) [Beaker](https://beaker.org/ex/01KWZAVZFS1FMR59ASRVH7VD4X) |

## 1.2B baseline Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-5 | finished | 100.00B | 1.2874 | 3.6235 | 0.00083 | 0.00007 | 0.21003 | 34.37 | 89934.1 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xgswsn5t) [Beaker](https://beaker.org/ex/01KWZAV9AGDSD3PTS5RAM7G5N2) |

## Validation Metrics

| metric | direction | 275M baseline Cx1 2e-4 | 275M baseline Cx1 4e-4 | 275M baseline Cx1 8e-4 | 275M baseline Cx1 1.6e-3 | 275M baseline Cx2 1.8e-4 | 275M baseline Cx4 1.5e-4 | 275M baseline Cx8 2e-4 | 275M baseline Cx8 4e-4 | 275M baseline Cx8 8e-4 | 275M baseline Cx8 1.6e-3 | 480M baseline Cx1 1.2e-4 | 480M baseline Cx8 8e-5 | 810M baseline Cx1 6e-5 | 810M baseline Cx8 4e-5 | 1.2B baseline Cx1 4e-5 | 1.2B baseline Cx8 4e-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | **0.90040** | 0.91077 | 0.92087 | 0.94430 | **0.88624** | **0.86619** | **0.86890** | 0.88767 | 0.91593 | 0.90970 | **0.82377** | **0.79155** | **0.78635** | **0.76221** | **0.76853** | **0.69597** |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | **0.98279** | 0.99730 | 1.0085 | 1.0332 | **0.96654** | **0.94651** | **0.94960** | 0.96898 | 0.99863 | 0.99320 | **0.89998** | **0.86322** | **0.85740** | **0.83583** | **0.84017** | **0.76088** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0399 | **1.0123** | 1.0441 | 1.0601 | **1.0012** | **1.0415** | 1.0584 | 1.0015 | **0.99425** | 1.0389 | **0.93554** | **0.84440** | **0.95086** | **0.76888** | **0.83973** | **0.59855** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0798 | **2.0246** | 2.0882 | 2.1202 | **2.0023** | **2.0829** | 2.1167 | 2.0029 | **1.9885** | 2.0778 | **1.8711** | **1.6888** | **1.9017** | **1.5378** | **1.6795** | **1.1971** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.72079 | **0.70174** | 0.72373 | 0.73480 | **0.69393** | **0.72185** | 0.73360 | 0.69418 | **0.68922** | 0.72010 | **0.64843** | **0.58530** | **0.65909** | **0.53299** | **0.58207** | **0.41489** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4416 | **1.4035** | 1.4475 | 1.4696 | **1.3879** | **1.4437** | 1.4672 | 1.3884 | **1.3784** | 1.4402 | **1.2969** | **1.1706** | **1.3182** | **1.0660** | **1.1641** | **0.82977** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | **0.30290** | 0.28584 | 0.26962 | 0.29608 | **0.34215** | **0.34215** | **0.31997** | 0.30461 | 0.29181 | 0.30717 | **0.39164** | **0.46758** | **0.40870** | **0.54608** | **0.46502** | **0.65614** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | **0.30290** | 0.28584 | 0.26962 | 0.29608 | **0.34215** | **0.34215** | **0.31997** | 0.30461 | 0.29181 | 0.30717 | **0.39164** | **0.46758** | **0.40870** | **0.54608** | **0.46502** | **0.65614** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4398 | -1.4002 | -1.4455 | **-1.4664** | **-1.3857** | **-1.4401** | **-1.4652** | -1.3858 | -1.3746 | -1.4380 | **-1.2951** | **-1.1683** | **-1.3170** | **-1.0647** | **-1.1631** | **-0.82821** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4398 | -1.4002 | -1.4455 | **-1.4664** | **-1.3857** | **-1.4401** | **-1.4652** | -1.3858 | -1.3746 | -1.4380 | **-1.2951** | **-1.1683** | **-1.3170** | **-1.0647** | **-1.1631** | **-0.82821** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.27198 | 0.26455 | **0.26440** | 0.26812 | **0.27855** | **0.30237** | 0.28740 | 0.27147 | **0.26320** | 0.26703 | **0.33108** | **0.36825** | **0.35374** | **0.45134** | **0.37856** | **0.55666** |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.27198 | 0.26455 | **0.26440** | 0.26812 | **0.27855** | **0.30237** | 0.28740 | 0.27147 | **0.26320** | 0.26703 | **0.33108** | **0.36825** | **0.35374** | **0.45134** | **0.37856** | **0.55666** |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | **0.67326** | 0.68598 | 0.71772 | 0.74546 | **0.67496** | **0.65778** | **0.66515** | 0.68166 | 0.69719 | 0.71477 | **0.61555** | **0.59136** | **0.55607** | **0.53494** | **0.56164** | **0.49543** |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | **0.73274** | 0.74696 | 0.78173 | 0.81192 | **0.73409** | **0.71558** | **0.72352** | 0.74132 | 0.75811 | 0.77633 | **0.66956** | **0.64288** | **0.60403** | **0.58121** | **0.60992** | **0.53735** |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 0.99662 | **0.99125** | 1.0231 | 1.0287 | **0.93367** | **0.93560** | 0.96440 | 0.98397 | **0.95077** | 0.99402 | **0.72319** | **0.64723** | **0.75618** | **0.47433** | **0.64226** | **0.33480** |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 1.9932 | **1.9825** | 2.0461 | 2.0575 | **1.8673** | **1.8712** | 1.9288 | 1.9679 | **1.9015** | 1.9880 | **1.4464** | **1.2945** | **1.5124** | **0.94866** | **1.2845** | **0.66961** |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.69082 | **0.68713** | 0.70916 | 0.71306 | **0.64717** | **0.64853** | 0.66847 | 0.68201 | **0.65906** | 0.68902 | **0.50129** | **0.44861** | **0.52421** | **0.32877** | **0.44518** | **0.23208** |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.3816 | **1.3743** | 1.4183 | 1.4261 | **1.2943** | **1.2971** | 1.3369 | 1.3640 | **1.3181** | 1.3780 | **1.0026** | **0.89722** | **1.0484** | **0.65754** | **0.89036** | **0.46415** |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | **0.33333** | 0.30051 | 0.27988 | 0.32113 | **0.41035** | **0.42172** | **0.42130** | 0.37416 | 0.37247 | 0.32912 | **0.57534** | **0.63173** | **0.54630** | **0.74032** | **0.63510** | **0.81987** |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | **0.33333** | 0.30051 | 0.27988 | 0.32113 | **0.41035** | **0.42172** | **0.42130** | 0.37416 | 0.37247 | 0.32912 | **0.57534** | **0.63173** | **0.54630** | **0.74032** | **0.63510** | **0.81987** |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3799 | -1.3699 | -1.4161 | **-1.4234** | **-1.2921** | **-1.2942** | -1.3346 | -1.3613 | -1.3143 | **-1.3757** | **-1.0012** | **-0.89529** | **-1.0475** | **-0.65647** | **-0.88958** | **-0.46284** |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3799 | -1.3699 | -1.4161 | **-1.4234** | **-1.2921** | **-1.2942** | -1.3346 | -1.3613 | -1.3143 | **-1.3757** | **-1.0012** | **-0.89529** | **-1.0475** | **-0.65647** | **-0.88958** | **-0.46284** |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.28917 | **0.27068** | 0.27513 | 0.28151 | **0.30396** | **0.36635** | 0.33045 | 0.30209 | 0.27870 | **0.27773** | **0.44801** | **0.50308** | **0.45862** | **0.63998** | **0.50694** | **0.73312** |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.28917 | **0.27068** | 0.27513 | 0.28151 | **0.30396** | **0.36635** | 0.33045 | 0.30209 | 0.27870 | **0.27773** | **0.44801** | **0.50308** | **0.45862** | **0.63998** | **0.50694** | **0.73312** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 0.65010 | 0.72478 | **0.55652** | 0.70821 | **0.68047** | **0.63290** | 0.67539 | 0.64967 | **0.52547** | 0.53748 | **0.48405** | **0.53418** | **0.44202** | **0.54788** | **0.48320** | **0.36090** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.0304 | 1.1536 | **0.89431** | 1.1286 | **1.0807** | **1.0033** | 1.0851 | 1.0427 | **0.83556** | 0.86121 | **0.77106** | **0.84875** | **0.69579** | **0.85682** | **0.76554** | **0.57471** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.45056 | 0.50242 | **0.38574** | 0.49085 | **0.47168** | **0.43863** | 0.46814 | 0.45029 | **0.36423** | 0.37252 | **0.33549** | **0.37028** | **0.30633** | **0.37978** | **0.33491** | **0.25017** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 0.71425 | 0.79946 | **0.61992** | 0.78232 | **0.74898** | **0.69535** | 0.75220 | 0.72279 | **0.57914** | 0.59696 | **0.53454** | **0.58835** | **0.48231** | **0.59396** | **0.53062** | **0.39827** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.77555 | 0.75072 | **0.79083** | 0.77650 | **0.74785** | **0.75549** | 0.75549 | 0.78032 | **0.81471** | 0.80898 | **0.82330** | **0.79752** | **0.83381** | **0.80325** | **0.80611** | **0.84814** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.77555 | 0.75072 | **0.79083** | 0.77650 | **0.74785** | **0.75549** | 0.75549 | 0.78032 | **0.81471** | 0.80898 | **0.82330** | **0.79752** | **0.83381** | **0.80325** | **0.80611** | **0.84814** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -0.74648 | **-0.79550** | -0.64098 | -0.78717 | **-0.80240** | **-0.77952** | **-0.76073** | -0.72979 | -0.58688 | -0.61183 | **-0.55049** | **-0.61065** | **-0.51026** | **-0.63253** | **-0.60517** | **-0.45930** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -0.74648 | **-0.79550** | -0.64098 | -0.78717 | **-0.80240** | **-0.77952** | **-0.76073** | -0.72979 | -0.58688 | -0.61183 | **-0.55049** | **-0.61065** | **-0.51026** | **-0.63253** | **-0.60517** | **-0.45930** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.71496 | 0.70982 | 0.74671 | **0.70340** | **0.69931** | **0.71053** | **0.71088** | 0.72648 | 0.75770 | 0.74300 | **0.78531** | **0.76606** | **0.78089** | **0.75776** | **0.76285** | **0.82068** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.71496 | 0.70982 | 0.74671 | **0.70340** | **0.69931** | **0.71053** | **0.71088** | 0.72648 | 0.75770 | 0.74300 | **0.78531** | **0.76606** | **0.78089** | **0.75776** | **0.76285** | **0.82068** |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | **0.36288** | 0.36813 | 0.41394 | 0.42801 | **0.34787** | **0.33147** | 0.42875 | 0.38882 | 0.40584 | **0.37626** | **0.38452** | **0.36810** | **0.38795** | **0.35527** | **0.36785** | **0.38293** |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | **0.39531** | 0.40115 | 0.45103 | 0.46660 | **0.37927** | **0.36123** | 0.46679 | 0.42271 | 0.44186 | **0.41018** | **0.41974** | **0.40152** | **0.42279** | **0.38705** | **0.40167** | **0.41748** |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | **0.25150** | 0.25519 | 0.28697 | 0.29669 | **0.24109** | **0.22977** | 0.29715 | 0.26951 | 0.28132 | **0.26080** | **0.26654** | **0.25516** | **0.26890** | **0.24624** | **0.25498** | **0.26545** |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | **0.27401** | 0.27806 | 0.31263 | 0.32344 | **0.26287** | **0.25035** | 0.32356 | 0.29301 | 0.30628 | **0.28429** | **0.29092** | **0.27830** | **0.29305** | **0.26827** | **0.27840** | **0.28940** |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.73123 | **0.73221** | 0.67589 | 0.71245 | **0.74308** | **0.75395** | 0.71542 | **0.72727** | **0.72727** | 0.72036 | **0.75593** | **0.77569** | **0.75889** | **0.79842** | **0.78953** | **0.80040** |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.73123 | **0.73221** | 0.67589 | 0.71245 | **0.74308** | **0.75395** | 0.71542 | **0.72727** | **0.72727** | 0.72036 | **0.75593** | **0.77569** | **0.75889** | **0.79842** | **0.78953** | **0.80040** |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -0.95967 | -0.92786 | **-1.2174** | -1.1560 | **-0.91551** | **-0.92825** | -1.0867 | -1.1009 | -1.0207 | **-1.1063** | **-0.84528** | **-0.71162** | **-0.81769** | **-0.64245** | **-0.69492** | **-0.63085** |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -0.95967 | -0.92786 | **-1.2174** | -1.1560 | **-0.91551** | **-0.92825** | -1.0867 | -1.1009 | -1.0207 | **-1.1063** | **-0.84528** | **-0.71162** | **-0.81769** | **-0.64245** | **-0.69492** | **-0.63085** |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.71747 | 0.72129 | **0.67513** | 0.69753 | **0.71454** | **0.73118** | 0.70050 | 0.70888 | 0.71252 | **0.69943** | **0.72890** | **0.75704** | **0.74331** | **0.77600** | **0.77618** | **0.79058** |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.71747 | 0.72129 | **0.67513** | 0.69753 | **0.71454** | **0.73118** | 0.70050 | 0.70888 | 0.71252 | **0.69943** | **0.72890** | **0.75704** | **0.74331** | **0.77600** | **0.77618** | **0.79058** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.52145 | 0.50044 | 0.56032 | **0.45591** | **0.61793** | **0.45528** | **0.41584** | 0.42808 | 0.46691 | 0.57099 | **0.38017** | **0.34771** | **0.33961** | **0.29878** | **0.31155** | **0.22860** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.62514 | 0.60055 | 0.67130 | **0.54775** | **0.74099** | **0.54682** | **0.49918** | 0.51560 | 0.56114 | 0.68624 | **0.45674** | **0.41874** | **0.40854** | **0.36014** | **0.37425** | **0.27431** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.36162 | 0.34710 | 0.38849 | **0.31612** | **0.42851** | **0.31578** | **0.28835** | 0.29681 | 0.32376 | 0.39590 | **0.26379** | **0.24121** | **0.23563** | **0.20731** | **0.21609** | **0.15867** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.43355 | 0.41659 | 0.46550 | **0.37975** | **0.51381** | **0.37925** | **0.34622** | 0.35752 | 0.38910 | 0.47588 | **0.31690** | **0.29051** | **0.28349** | **0.24988** | **0.25963** | **0.19042** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.79460 | **0.80906** | 0.80231 | 0.79942 | **0.80906** | **0.83510** | 0.83317 | **0.83414** | 0.80810 | 0.75217 | **0.84957** | **0.89392** | **0.88139** | **0.93925** | **0.92382** | **0.94407** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.79460 | **0.80906** | 0.80231 | 0.79942 | **0.80906** | **0.83510** | 0.83317 | **0.83414** | 0.80810 | 0.75217 | **0.84957** | **0.89392** | **0.88139** | **0.93925** | **0.92382** | **0.94407** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | **-0.57022** | -0.52961 | -0.53063 | -0.53294 | **-0.54784** | **-0.48210** | -0.45539 | -0.49365 | -0.53345 | **-0.64738** | **-0.40982** | **-0.31237** | **-0.31913** | **-0.20420** | **-0.25664** | **-0.17342** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | **-0.57022** | -0.52961 | -0.53063 | -0.53294 | **-0.54784** | **-0.48210** | -0.45539 | -0.49365 | -0.53345 | **-0.64738** | **-0.40982** | **-0.31237** | **-0.31913** | **-0.20420** | **-0.25664** | **-0.17342** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | **0.68596** | 0.70690 | 0.70220 | 0.71715 | **0.69235** | **0.73393** | 0.73205 | 0.72526 | 0.70574 | **0.66858** | **0.75347** | **0.80345** | **0.80686** | **0.86541** | **0.83239** | **0.88291** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | **0.68596** | 0.70690 | 0.70220 | 0.71715 | **0.69235** | **0.73393** | 0.73205 | 0.72526 | 0.70574 | **0.66858** | **0.75347** | **0.80345** | **0.80686** | **0.86541** | **0.83239** | **0.88291** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29722 | 0.28635 | **0.27329** | 0.28688 | **0.30843** | **0.29545** | 0.29338 | 0.30529 | **0.25685** | 0.27956 | **0.28405** | **0.28000** | **0.27437** | **0.26400** | **0.27948** | **0.25966** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30718 | 0.29591 | **0.28236** | 0.29639 | **0.31873** | **0.30540** | 0.30316 | 0.31546 | **0.26542** | 0.28891 | **0.29366** | **0.28941** | **0.28363** | **0.27288** | **0.28883** | **0.26842** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20603 | 0.19851 | **0.18945** | 0.19886 | **0.21381** | **0.20479** | 0.20338 | 0.21163 | **0.17807** | 0.19379 | **0.19690** | **0.19409** | **0.19019** | **0.18302** | **0.19374** | **0.17999** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21293 | 0.20514 | **0.19573** | 0.20549 | **0.22095** | **0.21169** | 0.21017 | 0.21869 | **0.18401** | 0.20028 | **0.20356** | **0.20062** | **0.19661** | **0.18917** | **0.20023** | **0.18607** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | **0.83721** | 0.80590 | 0.81395 | **0.83721** | **0.82648** | **0.82648** | 0.82737 | 0.83095 | 0.84526 | **0.88104** | **0.88551** | **0.87030** | **0.87120** | **0.92844** | **0.84705** | **0.95796** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | **0.83721** | 0.80590 | 0.81395 | **0.83721** | **0.82648** | **0.82648** | 0.82737 | 0.83095 | 0.84526 | **0.88104** | **0.88551** | **0.87030** | **0.87120** | **0.92844** | **0.84705** | **0.95796** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.45501 | **-0.51743** | -0.48103 | -0.41398 | **-0.47340** | **-0.44094** | -0.42338 | **-0.43458** | -0.39605 | -0.34205 | **-0.33964** | **-0.31669** | **-0.28563** | **-0.21686** | **-0.40391** | **-0.13885** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.45501 | **-0.51743** | -0.48103 | -0.41398 | **-0.47340** | **-0.44094** | -0.42338 | **-0.43458** | -0.39605 | -0.34205 | **-0.33964** | **-0.31669** | **-0.28563** | **-0.21686** | **-0.40391** | **-0.13885** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82881 | **0.81260** | 0.82368 | 0.82590 | **0.81456** | **0.82875** | **0.83332** | 0.83485 | 0.84300 | 0.86230 | **0.86776** | **0.86641** | **0.86601** | **0.91116** | **0.85133** | **0.93724** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82881 | **0.81260** | 0.82368 | 0.82590 | **0.81456** | **0.82875** | **0.83332** | 0.83485 | 0.84300 | 0.86230 | **0.86776** | **0.86641** | **0.86601** | **0.91116** | **0.85133** | **0.93724** |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | **1.0774** | 1.0777 | 1.1267 | 1.1079 | **1.1054** | **1.0039** | 0.98241 | **0.96241** | 1.0032 | 1.0850 | **0.86417** | **0.86737** | **0.82876** | **0.82792** | **0.76292** | **0.66285** |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | **1.7259** | 1.7327 | 1.8180 | 1.7590 | **1.7749** | **1.6300** | 1.5959 | **1.5595** | 1.6136 | 1.7488 | **1.4280** | **1.4171** | **1.3711** | **1.3688** | **1.2575** | **1.1083** |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | **0.78066** | 0.78721 | 0.81873 | 0.80983 | **0.80280** | **0.73289** | 0.70918 | **0.70760** | 0.73758 | 0.79738 | **0.64205** | **0.63064** | **0.60626** | **0.60513** | **0.56153** | **0.49174** |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | **1.2830** | 1.3037 | 1.3565 | 1.3274 | **1.3239** | **1.2239** | **1.1781** | 1.1851 | 1.2264 | 1.3283 | **1.1005** | **1.0562** | **1.0323** | **1.0286** | **0.95498** | **0.85039** |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.62547 | 0.65543 | **0.66667** | 0.65356 | **0.63670** | **0.67041** | 0.67041 | **0.69288** | 0.67041 | 0.66105 | **0.72659** | **0.75094** | **0.74906** | **0.76592** | **0.76779** | **0.79213** |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.62547 | 0.65543 | **0.66667** | 0.65356 | **0.63670** | **0.67041** | 0.67041 | **0.69288** | 0.67041 | 0.66105 | **0.72659** | **0.75094** | **0.74906** | **0.76592** | **0.76779** | **0.79213** |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.94568 | -0.92084 | -0.86213 | **-0.99549** | **-0.93601** | **-0.86051** | -0.85807 | -0.88042 | -0.87968 | **-0.91216** | **-0.73105** | **-0.67990** | **-0.65656** | **-0.58736** | **-0.61824** | **-0.53072** |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.94568 | -0.92084 | -0.86213 | **-0.99549** | **-0.93601** | **-0.86051** | -0.85807 | -0.88042 | -0.87968 | **-0.91216** | **-0.73105** | **-0.67990** | **-0.65656** | **-0.58736** | **-0.61824** | **-0.53072** |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | **0.56627** | 0.57771 | 0.59092 | 0.57766 | **0.55178** | **0.59990** | 0.59597 | 0.61575 | 0.60133 | **0.59220** | **0.63038** | **0.66033** | **0.66200** | **0.68168** | **0.67184** | **0.71088** |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | **0.56627** | 0.57771 | 0.59092 | 0.57766 | **0.55178** | **0.59990** | 0.59597 | 0.61575 | 0.60133 | **0.59220** | **0.63038** | **0.66033** | **0.66200** | **0.68168** | **0.67184** | **0.71088** |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.6057 | 1.5035 | **1.4928** | 1.6324 | **1.6248** | **1.7586** | 1.4228 | 1.4875 | **1.3942** | 1.4802 | **1.3287** | **1.3171** | **1.2702** | **1.2018** | **1.1253** | **0.92335** |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.1960 | 2.0810 | **2.0582** | 2.2539 | **2.2318** | **2.4283** | 1.9750 | 2.0425 | **1.9355** | 2.0493 | **1.8295** | **1.8330** | **1.7868** | **1.7095** | **1.5630** | **1.3054** |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.1130 | 1.0421 | **1.0347** | 1.1315 | **1.1262** | **1.2189** | 0.98621 | 1.0311 | **0.96640** | 1.0260 | **0.92101** | **0.91289** | **0.88047** | **0.83303** | **0.78010** | **0.63993** |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.5221 | 1.4425 | **1.4268** | 1.5623 | **1.5469** | **1.6830** | 1.3691 | 1.4158 | **1.3417** | 1.4205 | **1.2682** | **1.2704** | **1.2383** | **1.1850** | **1.0832** | **0.90477** |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.32075 | **0.32978** | 0.29532 | 0.30025 | **0.31091** | **0.32896** | 0.31747 | 0.29040 | **0.33306** | 0.32158 | **0.36669** | **0.37736** | **0.41099** | **0.44135** | **0.40197** | **0.49549** |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.32075 | **0.32978** | 0.29532 | 0.30025 | **0.31091** | **0.32896** | 0.31747 | 0.29040 | **0.33306** | 0.32158 | **0.36669** | **0.37736** | **0.41099** | **0.44135** | **0.40197** | **0.49549** |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.5834 | -3.5888 | -3.5875 | **-3.6072** | **-3.4966** | **-3.8056** | -3.2438 | -3.3550 | -3.1842 | **-3.4066** | **-2.8807** | **-2.9132** | **-2.7458** | **-2.2474** | **-2.4781** | **-1.9744** |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.5834 | -3.5888 | -3.5875 | **-3.6072** | **-3.4966** | **-3.8056** | -3.2438 | -3.3550 | -3.1842 | **-3.4066** | **-2.8807** | **-2.9132** | **-2.7458** | **-2.2474** | **-2.4781** | **-1.9744** |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.32348 | 0.32609 | **0.30050** | 0.30530 | **0.32426** | **0.33776** | 0.32756 | **0.31172** | 0.33903 | 0.32783 | **0.36253** | **0.38460** | **0.42249** | **0.45376** | **0.40970** | **0.50986** |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.32348 | 0.32609 | **0.30050** | 0.30530 | **0.32426** | **0.33776** | 0.32756 | **0.31172** | 0.33903 | 0.32783 | **0.36253** | **0.38460** | **0.42249** | **0.45376** | **0.40970** | **0.50986** |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | **0.39462** | 0.39990 | 0.40951 | 0.41252 | **0.38762** | **0.39369** | **0.37988** | 0.38802 | 0.39949 | 0.42595 | **0.35574** | **0.33577** | **0.33783** | **0.32498** | **0.33685** | **0.30613** |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | **0.39976** | 0.40503 | 0.41495 | 0.41808 | **0.39290** | **0.39910** | **0.38485** | 0.39309 | 0.40512 | 0.43172 | **0.36042** | **0.34003** | **0.34221** | **0.32945** | **0.34133** | **0.31034** |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | **0.63536** | 0.64263 | 0.63706 | 0.66217 | **0.64630** | **0.64155** | **0.62432** | 0.64576 | 0.63079 | 0.66491 | **0.59422** | **0.56411** | **0.53679** | **0.52429** | **0.53607** | **0.52131** |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | **0.64087** | 0.64825 | 0.64276 | 0.66801 | **0.65212** | **0.64725** | **0.62940** | 0.65140 | 0.63636 | 0.67071 | **0.59945** | **0.56899** | **0.54144** | **0.52908** | **0.54074** | **0.52595** |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.2870 | 2.1814 | **1.9198** | 1.9998 | **1.9509** | **2.2274** | 2.1948 | **1.9414** | 1.9959 | 2.3464 | **2.2674** | **1.6531** | **1.7352** | **1.2234** | **1.9344** | **0.15896** |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.5740 | 4.3628 | **3.8396** | 3.9996 | **3.9019** | **4.4548** | 4.3895 | **3.8827** | 3.9919 | 4.6928 | **4.5347** | **3.3062** | **3.4704** | **2.4468** | **3.8689** | **0.31792** |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5854 | 1.5118 | **1.3303** | 1.3861 | **1.3527** | **1.5442** | 1.5213 | **1.3460** | 1.3834 | 1.6261 | **1.5714** | **1.1458** | **1.2029** | **0.84815** | **1.3410** | **0.11010** |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.1707 | 3.0237 | **2.6605** | 2.7721 | **2.7054** | **3.0884** | 3.0426 | **2.6920** | 2.7669 | 3.2521 | **3.1427** | **2.2916** | **2.4059** | **1.6963** | **2.6819** | **0.22020** |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | **0.10000** | **0.10000** | 0.09000 | **0.10000** | **0.09000** | **0.16000** | 0.07000 | **0.09000** | **0.09000** | **0.09000** | **0.16000** | **0.16000** | **0.21000** | **0.47000** | **0.26000** | **0.94000** |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | **0.10000** | **0.10000** | 0.09000 | **0.10000** | **0.09000** | **0.16000** | 0.07000 | **0.09000** | **0.09000** | **0.09000** | **0.16000** | **0.16000** | **0.21000** | **0.47000** | **0.26000** | **0.94000** |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | **-3.1645** | -3.0153 | -2.6477 | -2.7536 | **-2.6942** | **-3.0824** | -3.0386 | -2.6847 | -2.7545 | **-3.2430** | **-3.1331** | **-2.2792** | **-2.3765** | **-1.6929** | **-2.6761** | **-0.21512** |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | **-3.1645** | -3.0153 | -2.6477 | -2.7536 | **-2.6942** | **-3.0824** | -3.0386 | -2.6847 | -2.7545 | **-3.2430** | **-3.1331** | **-2.2792** | **-2.3765** | **-1.6929** | **-2.6761** | **-0.21512** |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | **0.09619** | 0.09935 | 0.09730 | 0.09728 | **0.09898** | **0.11748** | 0.10150 | 0.10299 | 0.09943 | **0.09575** | **0.11608** | **0.11245** | **0.11772** | **0.30636** | **0.15517** | **0.85975** |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | **0.09619** | 0.09935 | 0.09730 | 0.09728 | **0.09898** | **0.11748** | 0.10150 | 0.10299 | 0.09943 | **0.09575** | **0.11608** | **0.11245** | **0.11772** | **0.30636** | **0.15517** | **0.85975** |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.86836 | **0.86716** | 0.87231 | 0.87845 | **0.85810** | **0.85003** | **0.84314** | 0.85257 | 0.86222 | 0.88006 | **0.80764** | **0.77712** | **0.78286** | **0.74488** | **0.76184** | **0.72008** |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.87793 | **0.87660** | 0.88181 | 0.88806 | **0.86765** | **0.85945** | **0.85257** | 0.86183 | 0.87166 | 0.88967 | **0.81642** | **0.78568** | **0.79156** | **0.75314** | **0.77018** | **0.72811** |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.55037 | **0.53742** | 0.54168 | 0.55648 | **0.53579** | **0.53467** | **0.52413** | 0.52563 | 0.53092 | 0.56245 | **0.50806** | **0.49148** | **0.48758** | **0.45435** | **0.47495** | **0.44937** |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.55216 | **0.53911** | 0.54345 | 0.55831 | **0.53752** | **0.53634** | **0.52569** | 0.52732 | 0.53261 | 0.56420 | **0.50972** | **0.49306** | **0.48907** | **0.45587** | **0.47644** | **0.45086** |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | **0.78296** | 0.79651 | 0.79550 | 0.82352 | **0.77905** | **0.77105** | **0.75765** | 0.78282 | 0.80396 | 0.81406 | **0.71471** | **0.67222** | **0.68590** | **0.62223** | **0.64692** | **0.58757** |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | **0.82267** | 0.83718 | 0.83570 | 0.86566 | **0.81906** | **0.80982** | **0.79562** | 0.82213 | 0.84453 | 0.85553 | **0.75033** | **0.70499** | **0.72018** | **0.65195** | **0.67845** | **0.61493** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0140 | 1.0105 | **1.0037** | 1.0151 | **0.99135** | **1.0058** | 1.0030 | **1.0026** | 1.0048 | 1.0190 | **0.98736** | **0.94989** | **0.96155** | **0.90603** | **0.93260** | **0.85853** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0279 | 2.0210 | **2.0073** | 2.0301 | **1.9827** | **2.0116** | 2.0061 | **2.0052** | 2.0095 | 2.0381 | **1.9747** | **1.8998** | **1.9231** | **1.8121** | **1.8652** | **1.7171** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70285 | 0.70046 | **0.69574** | 0.70366 | **0.68723** | **0.69722** | 0.69530 | **0.69499** | 0.69655 | 0.70634 | **0.68439** | **0.65848** | **0.66658** | **0.62802** | **0.64644** | **0.59510** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4057 | 1.4009 | **1.3915** | 1.4073 | **1.3745** | **1.3944** | 1.3906 | **1.3900** | 1.3931 | 1.4127 | **1.3688** | **1.3170** | **1.3332** | **1.2560** | **1.2929** | **1.1902** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.30372 | **0.30840** | 0.30584 | 0.29692 | **0.30606** | **0.30925** | **0.31647** | 0.29245 | 0.29734 | 0.27949 | **0.32965** | **0.36451** | **0.35005** | **0.40659** | **0.38193** | **0.44867** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.30372 | **0.30840** | 0.30584 | 0.29692 | **0.30606** | **0.30925** | **0.31647** | 0.29245 | 0.29734 | 0.27949 | **0.32965** | **0.36451** | **0.35005** | **0.40659** | **0.38193** | **0.44867** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3651 | -1.3659 | -1.3620 | **-1.3703** | **-1.3593** | **-1.3569** | -1.3541 | -1.3649 | -1.3721 | **-1.3816** | **-1.3409** | **-1.3152** | **-1.3255** | **-1.2571** | **-1.2931** | **-1.2001** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4018 | -1.3968 | -1.3885 | **-1.4028** | **-1.3720** | **-1.3925** | -1.3880 | -1.3862 | -1.3872 | **-1.4071** | **-1.3665** | **-1.3140** | **-1.3307** | **-1.2535** | **-1.2917** | **-1.1887** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.26375 | 0.26213 | 0.26367 | **0.26109** | **0.26280** | **0.26722** | 0.26822 | 0.26144 | 0.25772 | **0.25517** | **0.27227** | **0.27932** | **0.27624** | **0.30466** | **0.28866** | **0.33106** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.27511 | 0.27282 | 0.27513 | **0.27030** | **0.27368** | **0.27999** | 0.28217 | 0.27108 | 0.26446 | **0.25962** | **0.28920** | **0.30256** | **0.29627** | **0.34378** | **0.31840** | **0.38192** |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | **1.0669** | 1.0977 | 1.0900 | 1.1418 | **1.0396** | **1.0393** | **1.0253** | 1.0647 | 1.0766 | 1.1195 | **0.96320** | **0.91150** | **0.93405** | **0.85661** | **0.87534** | **0.79542** |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | **1.1859** | 1.2207 | 1.2095 | 1.2688 | **1.1535** | **1.1540** | **1.1375** | 1.1842 | 1.1949 | 1.2442 | **1.0699** | **1.0132** | **1.0396** | **0.95356** | **0.97435** | **0.88508** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0191 | **0.98980** | 1.0121 | 1.0692 | **0.96428** | **1.0138** | 0.99128 | **0.96519** | 0.98552 | 0.99369 | **0.91320** | **0.86258** | **0.89962** | **0.79070** | **0.85531** | **0.71001** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0382 | **1.9796** | 2.0241 | 2.1384 | **1.9286** | **2.0276** | 1.9826 | **1.9304** | 1.9710 | 1.9874 | **1.8264** | **1.7252** | **1.7992** | **1.5814** | **1.7106** | **1.4200** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70637 | **0.68608** | 0.70151 | 0.74111 | **0.66841** | **0.70272** | 0.68706 | **0.66904** | 0.68314 | 0.68873 | **0.63298** | **0.59789** | **0.62359** | **0.54808** | **0.59284** | **0.49215** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4127 | **1.3722** | 1.4030 | 1.4822 | **1.3368** | **1.4054** | 1.3741 | **1.3381** | 1.3663 | 1.3775 | **1.2660** | **1.1958** | **1.2472** | **1.0962** | **1.1857** | **0.98430** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.33498 | **0.34392** | 0.33159 | 0.30444 | **0.36181** | **0.35996** | **0.37662** | 0.34762 | 0.34917 | 0.31215 | **0.41147** | **0.45558** | **0.42844** | **0.52653** | **0.46545** | **0.57989** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.33498 | **0.34392** | 0.33159 | 0.30444 | **0.36181** | **0.35996** | **0.37662** | 0.34762 | 0.34917 | 0.31215 | **0.41147** | **0.45558** | **0.42844** | **0.52653** | **0.46545** | **0.57989** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3408 | -1.3354 | -1.3423 | **-1.3714** | **-1.3215** | **-1.3122** | -1.3115 | -1.3257 | -1.3393 | **-1.3599** | **-1.2478** | **-1.2161** | **-1.2353** | **-1.1226** | **-1.1956** | **-1.0266** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4107 | -1.3684 | -1.4002 | **-1.4793** | **-1.3340** | **-1.4038** | -1.3722 | -1.3349 | -1.3610 | **-1.3726** | **-1.2646** | **-1.1940** | **-1.2449** | **-1.0944** | **-1.1845** | **-0.98241** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.27897 | 0.27525 | 0.27650 | **0.27200** | **0.27856** | **0.29570** | 0.29071 | 0.27603 | 0.27227 | **0.26187** | **0.31283** | **0.32058** | **0.31759** | **0.36524** | **0.33205** | **0.40921** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.30023 | 0.29589 | 0.29628 | **0.28652** | **0.30206** | **0.32616** | 0.32001 | 0.29797 | 0.29051 | **0.27234** | **0.35522** | **0.37048** | **0.36237** | **0.43184** | **0.38661** | **0.49217** |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | **0.89880** | 0.91745 | 0.92155 | 0.96177 | **0.88692** | **0.87537** | **0.87903** | 0.90194 | 0.92370 | 0.94024 | **0.82976** | **0.78734** | **0.79174** | **0.73674** | **0.75585** | **0.68824** |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | **0.95927** | 0.97894 | 0.98270 | 1.0272 | **0.94641** | **0.93420** | **0.93871** | 0.96311 | 0.98695 | 1.0029 | **0.88665** | **0.84102** | **0.84500** | **0.78560** | **0.80673** | **0.73394** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0384 | **1.0264** | 1.0563 | 1.0903 | **0.98032** | **1.0375** | 1.0296 | **0.98518** | 0.99769 | 1.0209 | **0.94297** | **0.86919** | **0.90786** | **0.79273** | **0.87363** | **0.70609** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0767 | **2.0527** | 2.1126 | 2.1806 | **1.9606** | **2.0751** | 2.0593 | **1.9704** | 1.9954 | 2.0418 | **1.8859** | **1.7384** | **1.8157** | **1.5855** | **1.7473** | **1.4122** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.71973 | **0.71143** | 0.73218 | 0.75573 | **0.67950** | **0.71917** | 0.71368 | **0.68286** | 0.69157 | 0.70765 | **0.65363** | **0.60245** | **0.62928** | **0.54950** | **0.60558** | **0.48944** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4395 | **1.4229** | 1.4644 | 1.5115 | **1.3590** | **1.4383** | 1.4274 | **1.3657** | 1.3831 | 1.4153 | **1.3073** | **1.2049** | **1.2586** | **1.0990** | **1.2112** | **0.97889** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.32792 | **0.33019** | 0.32272 | 0.29802 | **0.35912** | **0.35944** | 0.35132 | **0.35197** | 0.33117 | 0.30907 | **0.38869** | **0.45954** | **0.43126** | **0.52779** | **0.45044** | **0.59084** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.32792 | **0.33019** | 0.32272 | 0.29802 | **0.35912** | **0.35944** | 0.35132 | **0.35197** | 0.33117 | 0.30907 | **0.38869** | **0.45954** | **0.43126** | **0.52779** | **0.45044** | **0.59084** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3612 | -1.3623 | -1.3677 | **-1.3944** | **-1.3352** | **-1.3426** | -1.3393 | -1.3378 | -1.3526 | **-1.3753** | **-1.2822** | **-1.2255** | **-1.2521** | **-1.1320** | **-1.2178** | **-1.0223** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4373 | -1.4192 | -1.4615 | **-1.5086** | **-1.3565** | **-1.4368** | **-1.4253** | -1.3623 | -1.3773 | -1.4108 | **-1.3058** | **-1.2028** | **-1.2562** | **-1.0975** | **-1.2101** | **-0.97745** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.27150 | 0.26815 | 0.27169 | **0.26412** | **0.27415** | **0.28309** | 0.28283 | 0.27320 | 0.26743 | **0.25966** | **0.29690** | **0.31455** | **0.30703** | **0.35701** | **0.32103** | **0.40969** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.28865 | 0.28331 | 0.28766 | **0.27504** | **0.29465** | **0.30810** | 0.30623 | 0.29280 | 0.28209 | **0.26869** | **0.33228** | **0.36270** | **0.34807** | **0.42655** | **0.37222** | **0.49886** |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | **1.2301** | 1.2345 | 1.2375 | 1.2881 | **1.2122** | **1.2313** | **1.1917** | 1.2220 | 1.2312 | 1.2760 | **1.1453** | **1.0799** | **1.0894** | **1.0331** | **1.0551** | **0.96061** |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | **1.5202** | 1.5239 | 1.5253 | 1.5865 | **1.4961** | **1.5205** | **1.4715** | 1.5064 | 1.5116 | 1.5731 | **1.4187** | **1.3357** | **1.3467** | **1.2777** | **1.3053** | **1.1857** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0378 | **1.0118** | 1.0235 | 1.0552 | **0.99324** | **1.0068** | **0.99968** | 1.0083 | 1.0006 | 1.0242 | **0.96969** | **0.94006** | **0.96582** | **0.90603** | **0.96258** | **0.85123** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0756 | **2.0236** | 2.0470 | 2.1105 | **1.9865** | **2.0136** | **1.9994** | 2.0165 | 2.0011 | 2.0485 | **1.9394** | **1.8801** | **1.9316** | **1.8121** | **1.9252** | **1.7025** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.71932 | **0.70134** | 0.70949 | 0.73145 | **0.68850** | **0.69788** | **0.69292** | 0.69893 | 0.69353 | 0.70996 | **0.67214** | **0.65158** | **0.66951** | **0.62799** | **0.66715** | **0.59004** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4386 | **1.4027** | 1.4190 | 1.4629 | **1.3770** | **1.3958** | **1.3858** | 1.3979 | 1.3871 | 1.4199 | **1.3443** | **1.3032** | **1.3390** | **1.2560** | **1.3343** | **1.1801** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.29125 | **0.30020** | 0.29920 | 0.27767 | **0.31345** | **0.32538** | **0.32074** | 0.30616 | 0.30020 | 0.27005 | **0.34294** | **0.37806** | **0.34361** | **0.40557** | **0.36746** | **0.46289** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.29125 | **0.30020** | 0.29920 | 0.27767 | **0.31345** | **0.32538** | **0.32074** | 0.30616 | 0.30020 | 0.27005 | **0.34294** | **0.37806** | **0.34361** | **0.40557** | **0.36746** | **0.46289** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3861 | -1.3745 | -1.3776 | **-1.3959** | **-1.3632** | **-1.3601** | -1.3620 | -1.3723 | -1.3712 | **-1.3871** | **-1.3362** | **-1.3133** | **-1.3274** | **-1.2658** | **-1.3119** | **-1.1973** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4346 | -1.3979 | -1.4156 | **-1.4585** | **-1.3728** | **-1.3933** | -1.3830 | -1.3928 | -1.3800 | **-1.4141** | **-1.3415** | **-1.3001** | **-1.3354** | **-1.2537** | **-1.3324** | **-1.1781** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25687 | 0.25771 | 0.25882 | **0.25535** | **0.26035** | **0.26519** | 0.26261 | 0.25829 | 0.25701 | **0.25333** | **0.27084** | **0.27794** | **0.27496** | **0.29964** | **0.28407** | **0.33094** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.26331 | 0.26518 | 0.26733 | **0.26074** | **0.27015** | **0.27830** | 0.27395 | 0.26628 | 0.26379 | **0.25651** | **0.28851** | **0.30196** | **0.29530** | **0.33560** | **0.30995** | **0.38375** |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.41602 | **0.40476** | 0.41797 | 0.43233 | **0.43358** | **0.42343** | 0.41970 | **0.41635** | 0.43091 | 0.43435 | **0.38996** | **0.36766** | **0.36038** | **0.35067** | **0.35559** | **0.33180** |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.41840 | **0.40708** | 0.42028 | 0.43482 | **0.43627** | **0.42587** | 0.42216 | **0.41884** | 0.43342 | 0.43684 | **0.39224** | **0.36973** | **0.36245** | **0.35264** | **0.35772** | **0.33375** |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.31670 | **0.31553** | 0.33110 | 0.32565 | **0.32586** | **0.30789** | **0.31954** | 0.32661 | 0.32116 | 0.32629 | **0.29051** | **0.27559** | **0.28189** | **0.26167** | **0.27178** | **0.24845** |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.31796 | **0.31677** | 0.33242 | 0.32693 | **0.32717** | **0.30904** | **0.32082** | 0.32792 | 0.32241 | 0.32758 | **0.29159** | **0.27668** | **0.28297** | **0.26272** | **0.27284** | **0.24939** |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | **0.55095** | 0.56554 | 0.56977 | 0.56677 | **0.57295** | **0.63708** | **0.56326** | 0.59465 | 0.59852 | 0.60544 | **0.50293** | **0.46196** | **0.48141** | **0.43852** | **0.46626** | **0.42784** |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | **0.55497** | 0.56944 | 0.57375 | 0.57075 | **0.57716** | **0.64174** | **0.56714** | 0.59892 | 0.60280 | 0.60976 | **0.50653** | **0.46524** | **0.48491** | **0.44165** | **0.46953** | **0.43097** |
| eval/lm/c4_en-validation/CE loss | lower | **3.2356** | 3.2485 | 3.2792 | 3.3284 | **3.2094** | **3.1811** | **3.1674** | 3.2063 | 3.2482 | 3.3123 | **3.0430** | **2.9313** | **2.9192** | **2.7853** | **2.8370** | **2.6868** |
| eval/lm/c4_en-validation/PPL | lower | **25.42** | 25.75 | 26.55 | 27.89 | **24.76** | **24.07** | **23.75** | 24.69 | 25.74 | 27.45 | **20.97** | **18.75** | **18.53** | **16.20** | **17.06** | **14.69** |
| eval/lm/dolma_books-validation/CE loss | lower | **3.1928** | 3.2103 | 3.2560 | 3.3148 | **3.1696** | **3.1236** | **3.1079** | 3.1654 | 3.2248 | 3.3054 | **2.9616** | **2.8301** | **2.8150** | **2.6469** | **2.7054** | **2.5323** |
| eval/lm/dolma_books-validation/PPL | lower | **24.36** | 24.79 | 25.95 | 27.52 | **23.80** | **22.73** | **22.37** | 23.70 | 25.15 | 27.26 | **19.33** | **16.95** | **16.69** | **14.11** | **14.96** | **12.58** |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | **3.3697** | 3.3862 | 3.4172 | 3.4667 | **3.3451** | **3.3141** | **3.3008** | 3.3388 | 3.3836 | 3.4499 | **3.1785** | **3.0702** | **3.0563** | **2.9232** | **2.9752** | **2.8270** |
| eval/lm/dolma_common-crawl-validation/PPL | lower | **29.07** | 29.55 | 30.48 | 32.03 | **28.36** | **27.50** | **27.13** | 28.19 | 29.48 | 31.50 | **24.01** | **21.55** | **21.25** | **18.60** | **19.59** | **16.90** |
| eval/lm/dolma_pes2o-validation/CE loss | lower | **2.3417** | 2.3491 | 2.3714 | 2.4140 | **2.3215** | **2.2931** | **2.2791** | 2.3093 | 2.3463 | 2.3977 | **2.1844** | **2.0952** | **2.0874** | **1.9806** | **2.0245** | **1.9040** |
| eval/lm/dolma_pes2o-validation/PPL | lower | **10.40** | 10.48 | 10.71 | 11.18 | **10.19** | **9.9056** | **9.7681** | 10.07 | 10.45 | 11.00 | **8.8854** | **8.1267** | **8.0636** | **7.2472** | **7.5727** | **6.7126** |
| eval/lm/dolma_reddit-validation/CE loss | lower | **3.5102** | 3.5252 | 3.5447 | 3.5906 | **3.4879** | **3.4557** | **3.4465** | 3.4852 | 3.5175 | 3.5751 | **3.3357** | **3.2377** | **3.2255** | **3.1037** | **3.1475** | **3.0142** |
| eval/lm/dolma_reddit-validation/PPL | lower | **33.45** | 33.96 | 34.63 | 36.25 | **32.72** | **31.68** | **31.39** | 32.63 | 33.70 | 35.70 | **28.10** | **25.47** | **25.17** | **22.28** | **23.28** | **20.37** |
| eval/lm/dolma_stack-validation/CE loss | lower | **1.4795** | 1.4831 | 1.5073 | 1.5466 | **1.4662** | **1.4501** | **1.4386** | 1.4636 | 1.4942 | 1.5425 | **1.3450** | **1.2796** | **1.2678** | **1.1794** | **1.2115** | **1.1133** |
| eval/lm/dolma_stack-validation/PPL | lower | **4.3905** | 4.4068 | 4.5147 | 4.6957 | **4.3329** | **4.2634** | **4.2148** | 4.3215 | 4.4557 | 4.6764 | **3.8383** | **3.5953** | **3.5530** | **3.2523** | **3.3584** | **3.0445** |
| eval/lm/dolma_wiki-validation/CE loss | lower | **2.7348** | 2.7508 | 2.7853 | 2.8424 | **2.7069** | **2.6759** | **2.6669** | 2.7044 | 2.7577 | 2.8251 | **2.5321** | **2.4316** | **2.4145** | **2.2942** | **2.3480** | **2.2009** |
| eval/lm/dolma_wiki-validation/PPL | lower | **15.41** | 15.66 | 16.20 | 17.16 | **14.98** | **14.53** | **14.39** | 14.94 | 15.76 | 16.86 | **12.58** | **11.38** | **11.18** | **9.9162** | **10.46** | **9.0332** |
| eval/lm/ice-validation/CE loss | lower | **3.2444** | 3.2517 | 3.2919 | 3.3105 | **3.2260** | **3.1878** | **3.1703** | 3.1970 | 3.2510 | 3.2843 | **3.0424** | **2.9562** | **2.9221** | **2.7842** | **2.8352** | **2.6958** |
| eval/lm/ice-validation/PPL | lower | **25.65** | 25.83 | 26.89 | 27.40 | **25.18** | **24.24** | **23.81** | 24.46 | 25.82 | 26.69 | **20.96** | **19.22** | **18.58** | **16.19** | **17.03** | **14.82** |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | **3.1334** | 3.1408 | 3.1564 | 3.1969 | **3.1183** | **3.1016** | **3.0913** | 3.1138 | 3.1538 | 3.1810 | **2.9925** | **2.9291** | **2.8938** | **2.8119** | **2.8495** | **2.7426** |
| eval/lm/m2d2_s2orc-validation/PPL | lower | **22.95** | 23.12 | 23.49 | 24.46 | **22.61** | **22.23** | **22.01** | 22.51 | 23.42 | 24.07 | **19.94** | **18.71** | **18.06** | **16.64** | **17.28** | **15.53** |
| eval/lm/pile-validation/CE loss | lower | **2.4637** | 2.4738 | 2.5014 | 2.5457 | **2.4397** | **2.4114** | **2.4003** | 2.4338 | 2.4773 | 2.5341 | **2.2898** | **2.1911** | **2.1793** | **2.0570** | **2.1068** | **1.9696** |
| eval/lm/pile-validation/PPL | lower | **11.75** | 11.87 | 12.20 | 12.75 | **11.47** | **11.15** | **11.03** | 11.40 | 11.91 | 12.60 | **9.8728** | **8.9450** | **8.8398** | **7.8226** | **8.2218** | **7.1680** |
| eval/lm/wikitext_103-validation/CE loss | lower | **2.7592** | 2.7723 | 2.8083 | 2.8554 | **2.7282** | **2.6902** | **2.6786** | 2.7229 | 2.7714 | 2.8426 | **2.5485** | **2.3951** | **2.3891** | **2.2333** | **2.3020** | **2.1266** |
| eval/lm/wikitext_103-validation/PPL | lower | **15.79** | 16.00 | 16.58 | 17.38 | **15.30** | **14.73** | **14.57** | 15.22 | 15.98 | 17.16 | **12.79** | **10.97** | **10.90** | **9.3310** | **9.9942** | **8.3861** |
