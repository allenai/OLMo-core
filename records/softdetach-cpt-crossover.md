# Why `sd20` stops being Pareto-optimal at high FLOPs — diagnosis + tweaks (2026-09-22)

Companion to `records/softdetach-cpt-plan.md` (the campaign: soft-detached CPT of Qwen3.5-4B on
dolma3+longmino, dense vs `sd20` = random 20% of 512-token blocks whole, rest one detached slot).
Question (Prasann): sd20 is ~0.024 nats below dense at every matched-FLOP point up to ~760 PF,
matches dense-32M at 0.25× its FLOPs, then loses: −0.010 at 1.5k PF, +0.007 at 3k, +0.013 at 6k.
Is that a missing small tweak, and where (by context position) does the deficit live?

Curves (32 held-out 64k rows, full-attention dev CE, x = FLOP-meter actual PF):

| PF | dense | sd20 | Δ |
|---|---|---|---|
| ~196 | 4M 1.314 | 32M 1.291 (193) | −0.023 |
| ~390 | 8M 1.303 | 64M 1.279 (383) | −0.024 |
| ~760 | 16M 1.296 | 128M 1.272 (763) | −0.024 |
| ~1520 | 32M 1.283 | 256M 1.2725 (1523) | −0.010 |
| ~3030 | 64M 1.256 | 512M 1.2634 (3043) | +0.007 |
| ~5970 | 128M 1.245 | 1B 1.2585 (5942) | +0.013 |

⚠ **The ≤128M and ≥256M points come from DIFFERENT training shards.** dense (all budgets), sd20 ≤128M,
sfl20 and lslot20 trained on `cpt_u128M` (source parts 0–7); sd20-256M/512M/1B trained on `cpt_u1B`
(parts 0–27). Both are built by the same script from the same text in file order, but the loader's
row order differs (first-5-step train CE 0.384 on the 1B shard vs 0.440 on the 128M shard), so the
high-end sd20 points are NOT a strict prefix-extension of the low-end ones. Prasann noticed the sd20
curve is flat exactly across that switch — 128M 1.272 (763 PF) → 256M 1.2725 (1523 PF) — before it
resumes (512M −0.009, 1B −0.005). Three cheap checks (§A.5) test whether the flat step is the shard
switch, seed noise, or real.

Read as slopes, not as a "break": dense gains **−0.046 nats per decade of PF** (1.314 → 1.245 over
196 → 5998), sd20 **−0.028 per decade** (1.309 → 1.2585 over 97 → 5942). sd20 starts with a head
start (7.7× more optimizer steps and 1.5× more loss tokens per PF) and dense's steeper slope erodes
it; the lines cross near 2k PF. So the question is why sd20's per-FLOP slope is shallower.

## Part A — diagnosis

### A.1 Loss normalisation / effective LR — NOT the cause

`train_module.py:358` sets `loss_div_factor = batch_num_tokens_for_loss` from the labels **before**
compaction, and `_compact_pooled_soft_tokens` (model.py:1841) drops the pooled blocks' labels, so
the soft arms' summed loss is divided by ~5× too many tokens: the logged `train/CE` and the raw
gradient are ×~0.2. From the Beaker training logs (`optim/total grad norm`, `train/CE loss` per step):

| run | steps | grad-norm mean / median | % steps clipped (>1.0) | train CE first-5 → last-10 |
|---|---|---|---|---|
| dense-16M | 31 | 1.124 / 0.802 | 35.5 % | 1.807 → 1.703 |
| dense-32M | 62 | 0.770 / 0.651 | 14.5 % | 1.786 → 1.664 |
| dense-64M | 123 | 0.766 / 0.699 | 13.8 % | 1.776 → 1.590 |
| dense-128M | 245 | 0.724 / 0.642 | 11.0 % | 1.771 → 1.649 |
| sd20-64M | 123 | 0.289 / 0.254 | 0.8 % | 0.440 → 0.402 (÷0.2 ≈ 2.2 → 2.0) |
| sd20-128M | 245 | 0.275 / 0.251 | 0.4 % | 0.440 → 0.413 |
| sd20-256M | 489 | 0.242 / 0.234 | 0.0 % | 0.384 → 0.374 |
| sd20-512M | 977 | 0.233 / 0.226 | 0.0 % | 0.384 → 0.364 |
| sd20-1B | 1908 | 0.227 / 0.221 | 0.0 % | 0.385 → 0.397 |

The optimizer is `SkipStepAdamW` (β 0.9/0.95, wd 0, `max_grad_norm=1.0`, `LinearWithWarmup`
3 %/→0, peak LR 3e-5 for every arm). Adam is invariant to a constant gradient scale (eps 1e-8 is
irrelevant at these norms), so the ×0.2 does not shrink sd20's steps. The one place the scale
matters is clipping — and it cuts the other way: **dense is clipped on 11–35 % of its steps, sd20
never**, so if anything sd20 gets the larger effective step. Zero skipped steps on either arm.
Verdict: normalisation is a logging/cosmetic bug, not the crossover. (Fixing the divisor would only
change the logged CE and remove clipping-asymmetry in dense's favour.)

### A.2 Where the deficit lives — by target position (full-attention CE, bins of the 64k row)

_(pending: position-binned eval, jobs 01M35FSHGN1E9VGXZP7WHTG5TB … 01M35FVHS6RJRT1CKJXGAWNDK7,
`eval_cpt_devloss.py` now emits `full_pos_*` / `own_pos_*`; results → weka `softdetach_cpt/devloss_pos/`)_

### A.3 Train/eval drift

sd20's own-construction CE keeps improving with budget (1.533 → 1.513 → 1.502 → 1.498 → 1.482 →
1.474 for 32M → 1B) at about the same rate as its full-attention CE (1.291 → 1.2585), while the gap
to dense at equal PF widens monotonically (−0.024 → −0.010 → +0.007 → +0.013). That is not
"specialising to slotted inputs" in the sense of full-attention CE getting worse (it never does); it
is a shallower slope. The mechanism consistent with everything measured: an sd20 loss token attends
to a context that is 80 % slots, so what it can teach is bounded by what a slot conveys; dense's
loss tokens see full context and keep learning long-range structure. At small budgets the extra
loss tokens/steps per FLOP dominate; at large budgets the ceiling does. Prediction for A.2: sd20's
deficit vs dense grows with target position (long-range), not uniformly.

### A.4 Other things checked

- **LR schedule**: every run is a complete warmup→0 linear schedule over its own step count
  (`--max-tokens` sets the horizon), so no arm is caught mid-decay. sd20 does 7.7× more optimizer
  steps per PF (489 vs 62 at ~1.5k PF).
- **Data**: sd20-256M/512M/1B read the 1B shard (source parts 0–27, no second epoch), dense and the
  ≤128M soft arms the 128M shard (parts 0–7); dev = part 28 for everyone. Both shards are built in
  file order by the same script, but the first-5-step train CE differs (0.384 on the 1B shard vs
  0.440 on the 128M shard for sd20), so the loader's row order is not a strict prefix — the two
  shard families are the same text, differently ordered. No arm ever saw a dev token.
- **FLOP accounting**: the meter counts the compacted forward (sd20 ×0.13 of a dense row); the
  actual PF of the 256M/512M/1B runs (1523/3043/5942) land on the dense 32M/64M/128M points
  (1518/3011/5998), so the comparison is at equal measured compute.

### A.5 The flat 128M → 256M step: shard switch, seed, or real? (launched)

| check | run / job | what it decides |
|---|---|---|
| (1) sd20-128M trained on the **1B shard** (`--shard 1B`, run `sdcpt-q35-4b-sd20-u128M-s1B`) | `01M35J4SKD0X5Z3ADWD711B4TV` | a true prefix of the 256M run; if it lands near 1.272 the two segments join and the 256M point is the odd one out; if it lands well above, the 1B shard's row order is harder early and every ≥256M point carries that offset |
| (2) sd20-256M, seed 1 (`--seed 1`, run `sdcpt-q35-4b-sd20-u256M-seed1`) | `01M35J1RBX4KQWXCBD99QC7D30`, eval `01M35J3QVGDSSXMCQA0CRF4NE7` | run-to-run seed variance at the crossover budget |
| (3) paired per-row Δ between sd20-128M/256M/512M/1B (+dense) on the shared 32 dev rows, with SE | `dump_sd20_pairs_beaker.sh` → job `01M35NSJRYB348Y92QBQSHGMK9` (first attempt `01M35J1JYARW6KK7TTD36Y75HR` ran with `--no-logs`, output lost) | whether the flat step is inside ~2σ of paired noise |

_(results pending)_

## Part B — tweaks (launched at the crossover budgets, compared at equal PF)

| arm | what | runs | eval |
|---|---|---|---|
| `sd20mix` | sd20 + compression-mixing curriculum: per-row probability of training UNCOMPRESSED ramps 0 → 0.5 linearly over the run (`--st-mix-start-p 0 --st-mix-end-p 0.5 --st-mix-anneal-frac 1.0`), i.e. late training sees full context; ~0.35× dense FLOPs per token | 256M `01M35G3Z7TP78SPGQNDPP6BR99`, 512M `01M35G8XPPMRKK7FHBFSMME3EB` | `01M35G9TN92AFBZ41MJE0X4EN2`, `01M35GA43JDKN5W0QWCVHAETWV` |
| `sd20p32` | sd20 + first 32 real tokens in every pooled block (~0.17×) — only if A.2 says the deficit is long-range | pending A.2 | |

Loss-normalisation "fix" (B.i) is not run: A.1 shows it cannot move the curve under Adam.

_(results pending)_
