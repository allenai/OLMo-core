# Midtraining Validation Results

Generated: 2026-07-09 01:45 UTC

Interpretation: lower is better for CE loss, PPL, Z loss, router Z loss, and load-balancing loss; higher is better for MFU/TPS. Accuracy-style validation metrics are higher-is-better when present.

Selection rule: use only `eval/*` validation metrics for midtraining checkpoint/LR selection. Training loss on the midtraining mixture is shown only as run-health metadata and must not be used to choose LRs.

Backfill note: the first 275M grid did not run in-loop evals during training, so final-checkpoint eval-only backfills are required. Once those eval jobs finish and `copy_eval_backfills_to_wandb.py` copies their metrics back, this table will populate the `eval/*` section.

Settings: 100B midtraining tokens, sequence length 8192, global batch seq 128 (1,048,576 tokens), 1 node, 4 GPUs, EP1, microbatch 8, fresh optimizer state, 2000-step warmup then constant LR.

| source | training finished | eval metrics present | LRs with evals | still running |
| --- | --- | --- | --- | --- |
| Cx1 | 4/4 | 4/4 | 2e-4, 4e-4, 8e-4, 1.6e-3 |  |
| Cx2 | 1/1 | 0/1 |  |  |
| Cx4 | 1/1 | 0/1 |  |  |
| Cx8 | 4/4 | 4/4 | 2e-4, 4e-4, 8e-4, 1.6e-3 |  |

## Eval Win Summary

Wins are computed separately within each source checkpoint group. Raw counts include every logged eval metric. De-duplicated counts collapse `v2`/non-`v2` repeats for the same task and score family, preferring `v2` when both are present. Ties, if any, count for every tied LR.

### Cx1 Win Counts

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

### Cx2 Win Counts

No `eval/*` validation metrics are present for this source group yet.

### Cx4 Win Counts

No `eval/*` validation metrics are present for this source group yet.

### Cx8 Win Counts

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

## Cx1 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.6064 | 4.9848 | 0.00083 | 0.00013 | 0.11042 | 29.30 | 363861.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cm8ww646) [Beaker](https://beaker.org/ex/01KWWM1043JEC9MC3PV7PXQ745) |
| 4e-4 | finished | 100.00B | 1.6162 | 5.0340 | 0.00095 | 0.00020 | 0.11038 | 29.36 | 364560.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r6ts032g) [Beaker](https://beaker.org/ex/01KWWM1AVQXMJ3JBJQ1W2G8YAV) |
| 8e-4 | finished | 100.00B | 1.6446 | 5.1791 | 0.00110 | 0.00027 | 0.11048 | 29.01 | 360250.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lfydkxv4) [Beaker](https://beaker.org/ex/01KWWM1N0EQDMCFER90NVP9QW0) |
| 1.6e-3 | finished | 100.00B | 1.6966 | 5.4554 | 0.00121 | 0.00036 | 0.11070 | 29.20 | 362582.5 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w3vof8b9) [Beaker](https://beaker.org/ex/01KWWM1ZXN5R5XWK00GH0WA36G) |

## Cx2 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.8e-4 | finished | 100.00B | 1.5903 | 4.9054 | 0.00089 | 0.00013 | 0.11041 | 29.14 | 361891.9 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kpkhplj7) [Beaker](https://beaker.org/ex/01KWZ8T9BZ3B869VZ878FNNQ8T) |

## Cx4 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5e-4 | finished | 100.00B | 1.5736 | 4.8240 | 0.00087 | 0.00013 | 0.11039 | 29.14 | 361830.2 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u508l6x8) [Beaker](https://beaker.org/ex/01KWZ8T9DZN8Y4VD63AP1AN387) |

## Cx8 Source

| LR | state | tokens | train CE | train PPL | Z loss | router Z | load balance | MFU | TPS/GPU | links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-4 | finished | 100.00B | 1.5642 | 4.7789 | 0.00084 | 0.00015 | 0.11033 | 29.36 | 364568.7 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4amcsbx6) [Beaker](https://beaker.org/ex/01KWWM10ANMMW2YTNN6RKJBGE7) |
| 4e-4 | finished | 100.00B | 1.5924 | 4.9157 | 0.00101 | 0.00021 | 0.11038 | 29.35 | 364491.6 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8jdqtmgg) [Beaker](https://beaker.org/ex/01KWWM1AK89SKDA1KGCX5D8SMM) |
| 8e-4 | finished | 100.00B | 1.6283 | 5.0952 | 0.00111 | 0.00028 | 0.11047 | 29.19 | 362440.4 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/drm1ceit) [Beaker](https://beaker.org/ex/01KWWM1P9TXRSMDV5DH9EF4KXM) |
| 1.6e-3 | finished | 100.00B | 1.6843 | 5.3887 | 0.00129 | 0.00037 | 0.11067 | 29.06 | 360889.0 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/edljug2e) [Beaker](https://beaker.org/ex/01KWWM213KMG47KADPK5Q67GJP) |

## Validation Metrics

| metric | direction | Cx1 2e-4 | Cx1 4e-4 | Cx1 8e-4 | Cx1 1.6e-3 | Cx2 1.8e-4 | Cx4 1.5e-4 | Cx8 2e-4 | Cx8 4e-4 | Cx8 8e-4 | Cx8 1.6e-3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | **0.90040** | 0.91077 | 0.92087 | 0.94430 |  |  | **0.86890** | 0.88767 | 0.91593 | 0.90970 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | **0.98279** | 0.99730 | 1.0085 | 1.0332 |  |  | **0.94960** | 0.96898 | 0.99863 | 0.99320 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0399 | **1.0123** | 1.0441 | 1.0601 |  |  | 1.0584 | 1.0015 | **0.99425** | 1.0389 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0798 | **2.0246** | 2.0882 | 2.1202 |  |  | 2.1167 | 2.0029 | **1.9885** | 2.0778 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.72079 | **0.70174** | 0.72373 | 0.73480 |  |  | 0.73360 | 0.69418 | **0.68922** | 0.72010 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4416 | **1.4035** | 1.4475 | 1.4696 |  |  | 1.4672 | 1.3884 | **1.3784** | 1.4402 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | **0.30290** | 0.28584 | 0.26962 | 0.29608 |  |  | **0.31997** | 0.30461 | 0.29181 | 0.30717 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | **0.30290** | 0.28584 | 0.26962 | 0.29608 |  |  | **0.31997** | 0.30461 | 0.29181 | 0.30717 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4398 | -1.4002 | -1.4455 | **-1.4664** |  |  | **-1.4652** | -1.3858 | -1.3746 | -1.4380 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4398 | -1.4002 | -1.4455 | **-1.4664** |  |  | **-1.4652** | -1.3858 | -1.3746 | -1.4380 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.27198 | 0.26455 | **0.26440** | 0.26812 |  |  | 0.28740 | 0.27147 | **0.26320** | 0.26703 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.27198 | 0.26455 | **0.26440** | 0.26812 |  |  | 0.28740 | 0.27147 | **0.26320** | 0.26703 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | **0.67326** | 0.68598 | 0.71772 | 0.74546 |  |  | **0.66515** | 0.68166 | 0.69719 | 0.71477 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | **0.73274** | 0.74696 | 0.78173 | 0.81192 |  |  | **0.72352** | 0.74132 | 0.75811 | 0.77633 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 0.99662 | **0.99125** | 1.0231 | 1.0287 |  |  | 0.96440 | 0.98397 | **0.95077** | 0.99402 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 1.9932 | **1.9825** | 2.0461 | 2.0575 |  |  | 1.9288 | 1.9679 | **1.9015** | 1.9880 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.69082 | **0.68713** | 0.70916 | 0.71306 |  |  | 0.66847 | 0.68201 | **0.65906** | 0.68902 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.3816 | **1.3743** | 1.4183 | 1.4261 |  |  | 1.3369 | 1.3640 | **1.3181** | 1.3780 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | **0.33333** | 0.30051 | 0.27988 | 0.32113 |  |  | **0.42130** | 0.37416 | 0.37247 | 0.32912 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | **0.33333** | 0.30051 | 0.27988 | 0.32113 |  |  | **0.42130** | 0.37416 | 0.37247 | 0.32912 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3799 | -1.3699 | -1.4161 | **-1.4234** |  |  | -1.3346 | -1.3613 | -1.3143 | **-1.3757** |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3799 | -1.3699 | -1.4161 | **-1.4234** |  |  | -1.3346 | -1.3613 | -1.3143 | **-1.3757** |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.28917 | **0.27068** | 0.27513 | 0.28151 |  |  | 0.33045 | 0.30209 | 0.27870 | **0.27773** |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.28917 | **0.27068** | 0.27513 | 0.28151 |  |  | 0.33045 | 0.30209 | 0.27870 | **0.27773** |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 0.65010 | 0.72478 | **0.55652** | 0.70821 |  |  | 0.67539 | 0.64967 | **0.52547** | 0.53748 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.0304 | 1.1536 | **0.89431** | 1.1286 |  |  | 1.0851 | 1.0427 | **0.83556** | 0.86121 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.45056 | 0.50242 | **0.38574** | 0.49085 |  |  | 0.46814 | 0.45029 | **0.36423** | 0.37252 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 0.71425 | 0.79946 | **0.61992** | 0.78232 |  |  | 0.75220 | 0.72279 | **0.57914** | 0.59696 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.77555 | 0.75072 | **0.79083** | 0.77650 |  |  | 0.75549 | 0.78032 | **0.81471** | 0.80898 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.77555 | 0.75072 | **0.79083** | 0.77650 |  |  | 0.75549 | 0.78032 | **0.81471** | 0.80898 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -0.74648 | **-0.79550** | -0.64098 | -0.78717 |  |  | **-0.76073** | -0.72979 | -0.58688 | -0.61183 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -0.74648 | **-0.79550** | -0.64098 | -0.78717 |  |  | **-0.76073** | -0.72979 | -0.58688 | -0.61183 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.71496 | 0.70982 | 0.74671 | **0.70340** |  |  | **0.71088** | 0.72648 | 0.75770 | 0.74300 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.71496 | 0.70982 | 0.74671 | **0.70340** |  |  | **0.71088** | 0.72648 | 0.75770 | 0.74300 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | **0.36288** | 0.36813 | 0.41394 | 0.42801 |  |  | 0.42875 | 0.38882 | 0.40584 | **0.37626** |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | **0.39531** | 0.40115 | 0.45103 | 0.46660 |  |  | 0.46679 | 0.42271 | 0.44186 | **0.41018** |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | **0.25150** | 0.25519 | 0.28697 | 0.29669 |  |  | 0.29715 | 0.26951 | 0.28132 | **0.26080** |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | **0.27401** | 0.27806 | 0.31263 | 0.32344 |  |  | 0.32356 | 0.29301 | 0.30628 | **0.28429** |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.73123 | **0.73221** | 0.67589 | 0.71245 |  |  | 0.71542 | **0.72727** | **0.72727** | 0.72036 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.73123 | **0.73221** | 0.67589 | 0.71245 |  |  | 0.71542 | **0.72727** | **0.72727** | 0.72036 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -0.95967 | -0.92786 | **-1.2174** | -1.1560 |  |  | -1.0867 | -1.1009 | -1.0207 | **-1.1063** |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -0.95967 | -0.92786 | **-1.2174** | -1.1560 |  |  | -1.0867 | -1.1009 | -1.0207 | **-1.1063** |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.71747 | 0.72129 | **0.67513** | 0.69753 |  |  | 0.70050 | 0.70888 | 0.71252 | **0.69943** |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.71747 | 0.72129 | **0.67513** | 0.69753 |  |  | 0.70050 | 0.70888 | 0.71252 | **0.69943** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.52145 | 0.50044 | 0.56032 | **0.45591** |  |  | **0.41584** | 0.42808 | 0.46691 | 0.57099 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.62514 | 0.60055 | 0.67130 | **0.54775** |  |  | **0.49918** | 0.51560 | 0.56114 | 0.68624 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.36162 | 0.34710 | 0.38849 | **0.31612** |  |  | **0.28835** | 0.29681 | 0.32376 | 0.39590 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.43355 | 0.41659 | 0.46550 | **0.37975** |  |  | **0.34622** | 0.35752 | 0.38910 | 0.47588 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.79460 | **0.80906** | 0.80231 | 0.79942 |  |  | 0.83317 | **0.83414** | 0.80810 | 0.75217 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.79460 | **0.80906** | 0.80231 | 0.79942 |  |  | 0.83317 | **0.83414** | 0.80810 | 0.75217 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | **-0.57022** | -0.52961 | -0.53063 | -0.53294 |  |  | -0.45539 | -0.49365 | -0.53345 | **-0.64738** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | **-0.57022** | -0.52961 | -0.53063 | -0.53294 |  |  | -0.45539 | -0.49365 | -0.53345 | **-0.64738** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | **0.68596** | 0.70690 | 0.70220 | 0.71715 |  |  | 0.73205 | 0.72526 | 0.70574 | **0.66858** |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | **0.68596** | 0.70690 | 0.70220 | 0.71715 |  |  | 0.73205 | 0.72526 | 0.70574 | **0.66858** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29722 | 0.28635 | **0.27329** | 0.28688 |  |  | 0.29338 | 0.30529 | **0.25685** | 0.27956 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30718 | 0.29591 | **0.28236** | 0.29639 |  |  | 0.30316 | 0.31546 | **0.26542** | 0.28891 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20603 | 0.19851 | **0.18945** | 0.19886 |  |  | 0.20338 | 0.21163 | **0.17807** | 0.19379 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21293 | 0.20514 | **0.19573** | 0.20549 |  |  | 0.21017 | 0.21869 | **0.18401** | 0.20028 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | **0.83721** | 0.80590 | 0.81395 | **0.83721** |  |  | 0.82737 | 0.83095 | 0.84526 | **0.88104** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | **0.83721** | 0.80590 | 0.81395 | **0.83721** |  |  | 0.82737 | 0.83095 | 0.84526 | **0.88104** |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.45501 | **-0.51743** | -0.48103 | -0.41398 |  |  | -0.42338 | **-0.43458** | -0.39605 | -0.34205 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.45501 | **-0.51743** | -0.48103 | -0.41398 |  |  | -0.42338 | **-0.43458** | -0.39605 | -0.34205 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82881 | **0.81260** | 0.82368 | 0.82590 |  |  | **0.83332** | 0.83485 | 0.84300 | 0.86230 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82881 | **0.81260** | 0.82368 | 0.82590 |  |  | **0.83332** | 0.83485 | 0.84300 | 0.86230 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | **1.0774** | 1.0777 | 1.1267 | 1.1079 |  |  | 0.98241 | **0.96241** | 1.0032 | 1.0850 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | **1.7259** | 1.7327 | 1.8180 | 1.7590 |  |  | 1.5959 | **1.5595** | 1.6136 | 1.7488 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | **0.78066** | 0.78721 | 0.81873 | 0.80983 |  |  | 0.70918 | **0.70760** | 0.73758 | 0.79738 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | **1.2830** | 1.3037 | 1.3565 | 1.3274 |  |  | **1.1781** | 1.1851 | 1.2264 | 1.3283 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.62547 | 0.65543 | **0.66667** | 0.65356 |  |  | 0.67041 | **0.69288** | 0.67041 | 0.66105 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.62547 | 0.65543 | **0.66667** | 0.65356 |  |  | 0.67041 | **0.69288** | 0.67041 | 0.66105 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.94568 | -0.92084 | -0.86213 | **-0.99549** |  |  | -0.85807 | -0.88042 | -0.87968 | **-0.91216** |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.94568 | -0.92084 | -0.86213 | **-0.99549** |  |  | -0.85807 | -0.88042 | -0.87968 | **-0.91216** |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | **0.56627** | 0.57771 | 0.59092 | 0.57766 |  |  | 0.59597 | 0.61575 | 0.60133 | **0.59220** |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | **0.56627** | 0.57771 | 0.59092 | 0.57766 |  |  | 0.59597 | 0.61575 | 0.60133 | **0.59220** |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.6057 | 1.5035 | **1.4928** | 1.6324 |  |  | 1.4228 | 1.4875 | **1.3942** | 1.4802 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.1960 | 2.0810 | **2.0582** | 2.2539 |  |  | 1.9750 | 2.0425 | **1.9355** | 2.0493 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.1130 | 1.0421 | **1.0347** | 1.1315 |  |  | 0.98621 | 1.0311 | **0.96640** | 1.0260 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.5221 | 1.4425 | **1.4268** | 1.5623 |  |  | 1.3691 | 1.4158 | **1.3417** | 1.4205 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.32075 | **0.32978** | 0.29532 | 0.30025 |  |  | 0.31747 | 0.29040 | **0.33306** | 0.32158 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.32075 | **0.32978** | 0.29532 | 0.30025 |  |  | 0.31747 | 0.29040 | **0.33306** | 0.32158 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.5834 | -3.5888 | -3.5875 | **-3.6072** |  |  | -3.2438 | -3.3550 | -3.1842 | **-3.4066** |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.5834 | -3.5888 | -3.5875 | **-3.6072** |  |  | -3.2438 | -3.3550 | -3.1842 | **-3.4066** |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.32348 | 0.32609 | **0.30050** | 0.30530 |  |  | 0.32756 | **0.31172** | 0.33903 | 0.32783 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.32348 | 0.32609 | **0.30050** | 0.30530 |  |  | 0.32756 | **0.31172** | 0.33903 | 0.32783 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | **0.39462** | 0.39990 | 0.40951 | 0.41252 |  |  | **0.37988** | 0.38802 | 0.39949 | 0.42595 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | **0.39976** | 0.40503 | 0.41495 | 0.41808 |  |  | **0.38485** | 0.39309 | 0.40512 | 0.43172 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | **0.63536** | 0.64263 | 0.63706 | 0.66217 |  |  | **0.62432** | 0.64576 | 0.63079 | 0.66491 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | **0.64087** | 0.64825 | 0.64276 | 0.66801 |  |  | **0.62940** | 0.65140 | 0.63636 | 0.67071 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.2870 | 2.1814 | **1.9198** | 1.9998 |  |  | 2.1948 | **1.9414** | 1.9959 | 2.3464 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.5740 | 4.3628 | **3.8396** | 3.9996 |  |  | 4.3895 | **3.8827** | 3.9919 | 4.6928 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5854 | 1.5118 | **1.3303** | 1.3861 |  |  | 1.5213 | **1.3460** | 1.3834 | 1.6261 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.1707 | 3.0237 | **2.6605** | 2.7721 |  |  | 3.0426 | **2.6920** | 2.7669 | 3.2521 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | **0.10000** | **0.10000** | 0.09000 | **0.10000** |  |  | 0.07000 | **0.09000** | **0.09000** | **0.09000** |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | **0.10000** | **0.10000** | 0.09000 | **0.10000** |  |  | 0.07000 | **0.09000** | **0.09000** | **0.09000** |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | **-3.1645** | -3.0153 | -2.6477 | -2.7536 |  |  | -3.0386 | -2.6847 | -2.7545 | **-3.2430** |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | **-3.1645** | -3.0153 | -2.6477 | -2.7536 |  |  | -3.0386 | -2.6847 | -2.7545 | **-3.2430** |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | **0.09619** | 0.09935 | 0.09730 | 0.09728 |  |  | 0.10150 | 0.10299 | 0.09943 | **0.09575** |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | **0.09619** | 0.09935 | 0.09730 | 0.09728 |  |  | 0.10150 | 0.10299 | 0.09943 | **0.09575** |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.86836 | **0.86716** | 0.87231 | 0.87845 |  |  | **0.84314** | 0.85257 | 0.86222 | 0.88006 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.87793 | **0.87660** | 0.88181 | 0.88806 |  |  | **0.85257** | 0.86183 | 0.87166 | 0.88967 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.55037 | **0.53742** | 0.54168 | 0.55648 |  |  | **0.52413** | 0.52563 | 0.53092 | 0.56245 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.55216 | **0.53911** | 0.54345 | 0.55831 |  |  | **0.52569** | 0.52732 | 0.53261 | 0.56420 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | **0.78296** | 0.79651 | 0.79550 | 0.82352 |  |  | **0.75765** | 0.78282 | 0.80396 | 0.81406 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | **0.82267** | 0.83718 | 0.83570 | 0.86566 |  |  | **0.79562** | 0.82213 | 0.84453 | 0.85553 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0140 | 1.0105 | **1.0037** | 1.0151 |  |  | 1.0030 | **1.0026** | 1.0048 | 1.0190 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0279 | 2.0210 | **2.0073** | 2.0301 |  |  | 2.0061 | **2.0052** | 2.0095 | 2.0381 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70285 | 0.70046 | **0.69574** | 0.70366 |  |  | 0.69530 | **0.69499** | 0.69655 | 0.70634 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4057 | 1.4009 | **1.3915** | 1.4073 |  |  | 1.3906 | **1.3900** | 1.3931 | 1.4127 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.30372 | **0.30840** | 0.30584 | 0.29692 |  |  | **0.31647** | 0.29245 | 0.29734 | 0.27949 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.30372 | **0.30840** | 0.30584 | 0.29692 |  |  | **0.31647** | 0.29245 | 0.29734 | 0.27949 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3651 | -1.3659 | -1.3620 | **-1.3703** |  |  | -1.3541 | -1.3649 | -1.3721 | **-1.3816** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4018 | -1.3968 | -1.3885 | **-1.4028** |  |  | -1.3880 | -1.3862 | -1.3872 | **-1.4071** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.26375 | 0.26213 | 0.26367 | **0.26109** |  |  | 0.26822 | 0.26144 | 0.25772 | **0.25517** |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.27511 | 0.27282 | 0.27513 | **0.27030** |  |  | 0.28217 | 0.27108 | 0.26446 | **0.25962** |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | **1.0669** | 1.0977 | 1.0900 | 1.1418 |  |  | **1.0253** | 1.0647 | 1.0766 | 1.1195 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | **1.1859** | 1.2207 | 1.2095 | 1.2688 |  |  | **1.1375** | 1.1842 | 1.1949 | 1.2442 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0191 | **0.98980** | 1.0121 | 1.0692 |  |  | 0.99128 | **0.96519** | 0.98552 | 0.99369 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0382 | **1.9796** | 2.0241 | 2.1384 |  |  | 1.9826 | **1.9304** | 1.9710 | 1.9874 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70637 | **0.68608** | 0.70151 | 0.74111 |  |  | 0.68706 | **0.66904** | 0.68314 | 0.68873 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4127 | **1.3722** | 1.4030 | 1.4822 |  |  | 1.3741 | **1.3381** | 1.3663 | 1.3775 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.33498 | **0.34392** | 0.33159 | 0.30444 |  |  | **0.37662** | 0.34762 | 0.34917 | 0.31215 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.33498 | **0.34392** | 0.33159 | 0.30444 |  |  | **0.37662** | 0.34762 | 0.34917 | 0.31215 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3408 | -1.3354 | -1.3423 | **-1.3714** |  |  | -1.3115 | -1.3257 | -1.3393 | **-1.3599** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4107 | -1.3684 | -1.4002 | **-1.4793** |  |  | -1.3722 | -1.3349 | -1.3610 | **-1.3726** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.27897 | 0.27525 | 0.27650 | **0.27200** |  |  | 0.29071 | 0.27603 | 0.27227 | **0.26187** |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.30023 | 0.29589 | 0.29628 | **0.28652** |  |  | 0.32001 | 0.29797 | 0.29051 | **0.27234** |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | **0.89880** | 0.91745 | 0.92155 | 0.96177 |  |  | **0.87903** | 0.90194 | 0.92370 | 0.94024 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | **0.95927** | 0.97894 | 0.98270 | 1.0272 |  |  | **0.93871** | 0.96311 | 0.98695 | 1.0029 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0384 | **1.0264** | 1.0563 | 1.0903 |  |  | 1.0296 | **0.98518** | 0.99769 | 1.0209 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0767 | **2.0527** | 2.1126 | 2.1806 |  |  | 2.0593 | **1.9704** | 1.9954 | 2.0418 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.71973 | **0.71143** | 0.73218 | 0.75573 |  |  | 0.71368 | **0.68286** | 0.69157 | 0.70765 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4395 | **1.4229** | 1.4644 | 1.5115 |  |  | 1.4274 | **1.3657** | 1.3831 | 1.4153 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.32792 | **0.33019** | 0.32272 | 0.29802 |  |  | 0.35132 | **0.35197** | 0.33117 | 0.30907 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.32792 | **0.33019** | 0.32272 | 0.29802 |  |  | 0.35132 | **0.35197** | 0.33117 | 0.30907 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3612 | -1.3623 | -1.3677 | **-1.3944** |  |  | -1.3393 | -1.3378 | -1.3526 | **-1.3753** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4373 | -1.4192 | -1.4615 | **-1.5086** |  |  | **-1.4253** | -1.3623 | -1.3773 | -1.4108 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.27150 | 0.26815 | 0.27169 | **0.26412** |  |  | 0.28283 | 0.27320 | 0.26743 | **0.25966** |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.28865 | 0.28331 | 0.28766 | **0.27504** |  |  | 0.30623 | 0.29280 | 0.28209 | **0.26869** |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | **1.2301** | 1.2345 | 1.2375 | 1.2881 |  |  | **1.1917** | 1.2220 | 1.2312 | 1.2760 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | **1.5202** | 1.5239 | 1.5253 | 1.5865 |  |  | **1.4715** | 1.5064 | 1.5116 | 1.5731 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0378 | **1.0118** | 1.0235 | 1.0552 |  |  | **0.99968** | 1.0083 | 1.0006 | 1.0242 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0756 | **2.0236** | 2.0470 | 2.1105 |  |  | **1.9994** | 2.0165 | 2.0011 | 2.0485 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.71932 | **0.70134** | 0.70949 | 0.73145 |  |  | **0.69292** | 0.69893 | 0.69353 | 0.70996 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4386 | **1.4027** | 1.4190 | 1.4629 |  |  | **1.3858** | 1.3979 | 1.3871 | 1.4199 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.29125 | **0.30020** | 0.29920 | 0.27767 |  |  | **0.32074** | 0.30616 | 0.30020 | 0.27005 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.29125 | **0.30020** | 0.29920 | 0.27767 |  |  | **0.32074** | 0.30616 | 0.30020 | 0.27005 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3861 | -1.3745 | -1.3776 | **-1.3959** |  |  | -1.3620 | -1.3723 | -1.3712 | **-1.3871** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4346 | -1.3979 | -1.4156 | **-1.4585** |  |  | -1.3830 | -1.3928 | -1.3800 | **-1.4141** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25687 | 0.25771 | 0.25882 | **0.25535** |  |  | 0.26261 | 0.25829 | 0.25701 | **0.25333** |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.26331 | 0.26518 | 0.26733 | **0.26074** |  |  | 0.27395 | 0.26628 | 0.26379 | **0.25651** |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.41602 | **0.40476** | 0.41797 | 0.43233 |  |  | 0.41970 | **0.41635** | 0.43091 | 0.43435 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.41840 | **0.40708** | 0.42028 | 0.43482 |  |  | 0.42216 | **0.41884** | 0.43342 | 0.43684 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.31670 | **0.31553** | 0.33110 | 0.32565 |  |  | **0.31954** | 0.32661 | 0.32116 | 0.32629 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.31796 | **0.31677** | 0.33242 | 0.32693 |  |  | **0.32082** | 0.32792 | 0.32241 | 0.32758 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | **0.55095** | 0.56554 | 0.56977 | 0.56677 |  |  | **0.56326** | 0.59465 | 0.59852 | 0.60544 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | **0.55497** | 0.56944 | 0.57375 | 0.57075 |  |  | **0.56714** | 0.59892 | 0.60280 | 0.60976 |
| eval/lm/c4_en-validation/CE loss | lower | **3.2356** | 3.2485 | 3.2792 | 3.3284 |  |  | **3.1674** | 3.2063 | 3.2482 | 3.3123 |
| eval/lm/c4_en-validation/PPL | lower | **25.42** | 25.75 | 26.55 | 27.89 |  |  | **23.75** | 24.69 | 25.74 | 27.45 |
| eval/lm/dolma_books-validation/CE loss | lower | **3.1928** | 3.2103 | 3.2560 | 3.3148 |  |  | **3.1079** | 3.1654 | 3.2248 | 3.3054 |
| eval/lm/dolma_books-validation/PPL | lower | **24.36** | 24.79 | 25.95 | 27.52 |  |  | **22.37** | 23.70 | 25.15 | 27.26 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | **3.3697** | 3.3862 | 3.4172 | 3.4667 |  |  | **3.3008** | 3.3388 | 3.3836 | 3.4499 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | **29.07** | 29.55 | 30.48 | 32.03 |  |  | **27.13** | 28.19 | 29.48 | 31.50 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | **2.3417** | 2.3491 | 2.3714 | 2.4140 |  |  | **2.2791** | 2.3093 | 2.3463 | 2.3977 |
| eval/lm/dolma_pes2o-validation/PPL | lower | **10.40** | 10.48 | 10.71 | 11.18 |  |  | **9.7681** | 10.07 | 10.45 | 11.00 |
| eval/lm/dolma_reddit-validation/CE loss | lower | **3.5102** | 3.5252 | 3.5447 | 3.5906 |  |  | **3.4465** | 3.4852 | 3.5175 | 3.5751 |
| eval/lm/dolma_reddit-validation/PPL | lower | **33.45** | 33.96 | 34.63 | 36.25 |  |  | **31.39** | 32.63 | 33.70 | 35.70 |
| eval/lm/dolma_stack-validation/CE loss | lower | **1.4795** | 1.4831 | 1.5073 | 1.5466 |  |  | **1.4386** | 1.4636 | 1.4942 | 1.5425 |
| eval/lm/dolma_stack-validation/PPL | lower | **4.3905** | 4.4068 | 4.5147 | 4.6957 |  |  | **4.2148** | 4.3215 | 4.4557 | 4.6764 |
| eval/lm/dolma_wiki-validation/CE loss | lower | **2.7348** | 2.7508 | 2.7853 | 2.8424 |  |  | **2.6669** | 2.7044 | 2.7577 | 2.8251 |
| eval/lm/dolma_wiki-validation/PPL | lower | **15.41** | 15.66 | 16.20 | 17.16 |  |  | **14.39** | 14.94 | 15.76 | 16.86 |
| eval/lm/ice-validation/CE loss | lower | **3.2444** | 3.2517 | 3.2919 | 3.3105 |  |  | **3.1703** | 3.1970 | 3.2510 | 3.2843 |
| eval/lm/ice-validation/PPL | lower | **25.65** | 25.83 | 26.89 | 27.40 |  |  | **23.81** | 24.46 | 25.82 | 26.69 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | **3.1334** | 3.1408 | 3.1564 | 3.1969 |  |  | **3.0913** | 3.1138 | 3.1538 | 3.1810 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | **22.95** | 23.12 | 23.49 | 24.46 |  |  | **22.01** | 22.51 | 23.42 | 24.07 |
| eval/lm/pile-validation/CE loss | lower | **2.4637** | 2.4738 | 2.5014 | 2.5457 |  |  | **2.4003** | 2.4338 | 2.4773 | 2.5341 |
| eval/lm/pile-validation/PPL | lower | **11.75** | 11.87 | 12.20 | 12.75 |  |  | **11.03** | 11.40 | 11.91 | 12.60 |
| eval/lm/wikitext_103-validation/CE loss | lower | **2.7592** | 2.7723 | 2.8083 | 2.8554 |  |  | **2.6786** | 2.7229 | 2.7714 | 2.8426 |
| eval/lm/wikitext_103-validation/PPL | lower | **15.79** | 16.00 | 16.58 | 17.38 |  |  | **14.57** | 15.22 | 15.98 | 17.16 |
