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

### A.2 Where the deficit lives — by target position: **the sd20 win is a SHORT-CONTEXT win, the loss is long-range and grows with budget**

Full-attention dev CE, 32 rows × 64k, binned by the target token's absolute position in the row
(`eval_cpt_devloss.py` `full_pos_*`, weka `softdetach_cpt/devloss_pos/`). Δ = sd20 − dense at
equal PF (rows = the matched pairs from the main table):

| pair (equal PF) | 0–2k | 2–8k | 8–16k | 16–32k | 32k–64k | all |
|---|---|---|---|---|---|---|
| dense-32M (1518) | 1.462 | 1.314 | 1.260 | 1.226 | 1.301 | 1.283 |
| sd20-64M (383, ¼ PF) | 1.399 | 1.305 | 1.257 | 1.223 | 1.301 | 1.279 |
| **Δ sd20-64M − dense-32M** | **−0.063** | −0.009 | −0.002 | −0.003 | 0.000 | −0.004 |
| sd20-256M (1523) | 1.390 | 1.293 | 1.250 | 1.216 | 1.295 | 1.2725 |
| **Δ sd20-256M − dense-32M** | **−0.072** | −0.020 | −0.010 | −0.010 | −0.006 | −0.010 |
| dense-64M (3011) | 1.387 | 1.281 | 1.234 | 1.200 | 1.277 | 1.256 |
| sd20-512M (3043) | 1.375 | 1.282 | 1.239 | 1.208 | 1.287 | 1.2634 |
| **Δ sd20-512M − dense-64M** | −0.012 | +0.001 | +0.005 | +0.008 | **+0.011** | +0.007 |
| dense-128M (5998) | 1.369 | 1.268 | 1.223 | 1.190 | 1.266 | 1.245 |
| sd20-1B (5942) | 1.369 | 1.275 | 1.234 | 1.202 | 1.283 | 1.2585 |
| **Δ sd20-1B − dense-128M** | 0.000 | +0.007 | +0.011 | +0.013 | **+0.017** | +0.013 |

Two facts, both monotone:

1. **sd20's whole advantage sits in the first 2k tokens.** At 383 PF and 1523 PF the 0–2k bin is
   −0.06/−0.07 below dense while every bin past 8k is within ±0.01. A pooled row is, for each real
   token, a *short* real context plus slots, so sd20 sees ~5× more "early-context" prediction
   problems per FLOP and learns that regime fast. Dense catches up on it with tokens: its 0–2k CE
   falls 1.462 → 1.387 → 1.369 across 32M → 128M, and at 128M it equals sd20-1B's 1.369 exactly.
2. **The deficit grows with position and with budget.** At equal PF the 32k–64k bin goes
   0.000 → −0.006 → +0.011 → +0.017 from 383 to 5942 PF, and within each high-end pair Δ rises
   monotonically from the 0–2k bin to the last bin. Long-range use of context is what dense keeps
   learning and sd20 cannot: 80 % of the context an sd20 loss token attends to is slots.

So the crossover is not a tweak-sized artefact of the optimizer; it is the two regimes trading
places. The "small tweak" that could keep sd20 ahead has to put real long-range information back
into the pooled blocks without giving up the compaction — that is Part B (iii) `sd20p32` (every
pooled block keeps its first 32 tokens) and (ii) the dense-mix curriculum.

⚠ The `own_pos_*` bins in the same JSONs are unusable (every bin equals `own_ce`): the own
construction's per-token losses live in compacted coordinates and the bin mask was built in
original coordinates; the full-attention bins above are unaffected. Not fixed (eval-side only).

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

**Results.** (full-attention dev CE; paired per-row Δ on the shared 32 rows from
`dump_sd20_pairs_beaker.sh`, jobs `01M35NSJRYB348Y92QBQSHGMK9` / `01M35TFHKRZ45QV65RGBC7X5EE`)

| point | shard | seed | full CE | paired Δ (se) |
|---|---|---|---|---|
| sd20-128M (763 PF) | 128M | 0 | 1.2720 | — |
| **sd20-128M-s1B** (763 PF) | **1B** | 0 | **1.2812** | +0.0089 (0.0020) vs sd20-128M old shard, +4.5σ; +0.0087 (0.0018) vs sd20-256M, +4.8σ |
| sd20-256M (1523 PF) | 1B | 0 | 1.2725 | −0.0002 (0.0005) vs sd20-128M old shard, −0.4σ |
| **sd20-256M-seed1** | 1B | 1 | **1.2682** | −0.0043 (0.0012) vs seed 0, 3.6σ; −0.0151 (0.0062) vs dense-32M, 2.5σ |
| sd20-512M (3043 PF) | 1B | 0 | 1.2634 | −0.0091 (0.0019) vs sd20-256M, −4.8σ |
| sd20-1B (5942 PF) | 1B | 0 | 1.2585 | −0.0048 (0.0010) vs sd20-512M, −5.1σ |

**Verdict: the plateau is a shard/order artefact, and the seed spread is the same size as the step.**
(1) The true prefix of the 256M run — sd20-128M trained on the 1B shard — lands at **1.2812**, i.e.
0.009 *above* the old-shard 128M point; from there the 1B-shard segment is monotone: 1.2812 →
1.2725 → 1.2634 → 1.2585 (−0.009 / −0.009 / −0.005 per doubling). The two shard families are
offset by ~0.009 at 128M (the 1B shard's early rows are harder / differently ordered), so joining
"old-shard 128M" to "1B-shard 256M" hides one doubling of progress. (2) A second seed at 256M
gives 1.2682, 0.0043 (0.0012) below seed 0 — a paired-significant seed effect of ~0.004, comparable to the
between-budget steps at this end of the curve, so single runs cannot resolve 0.005-size features.
(3) The between-budget paired Δs are all >4σ except the 128M(old)→256M(new) step (−0.4σ), which
is exactly the cross-shard comparison. Consequence for the main table: on a single shard the sd20
slope is ~−0.009 per doubling at 128M–512M, vs dense's −0.013 (1.283 → 1.256 → 1.245 is −0.027,
−0.011 per doubling). The crossover stands — dense-64M vs sd20-512M is −0.0078 (0.0017) paired,
4.6σ in dense's favour; dense-128M vs sd20-1B −0.0139 (0.0011), 12.8σ — but the low-end sd20 curve
(old shard) and the high-end one (1B shard) should not be drawn as one line: the honest sd20 curve
from the 1B shard alone is 1.2812 / 1.2725 / 1.2634 / 1.2585 at 763 / 1523 / 3043 / 5942 PF, which
beats dense at 763 (1.296) and 1523 (1.283, paired −0.0109 (0.0051), 2.2σ) and loses at 3043 and
5942. Same conclusion, crossover still ~2k PF.

## Part B — tweaks (launched at the crossover budgets, compared at equal PF)

| arm | what | runs | eval |
|---|---|---|---|
| `sd20mix` | sd20 + compression-mixing curriculum: per-row probability of training UNCOMPRESSED ramps 0 → 0.5 linearly over the run (`--st-mix-start-p 0 --st-mix-end-p 0.5 --st-mix-anneal-frac 1.0`), i.e. late training sees full context; ~0.35× dense FLOPs per token | 256M `01M35G3Z7TP78SPGQNDPP6BR99`, 512M `01M35G8XPPMRKK7FHBFSMME3EB` | `01M35G9TN92AFBZ41MJE0X4EN2`, `01M35GA43JDKN5W0QWCVHAETWV` |
| `sd20p32` | sd20 + first 32 real tokens in every pooled block (`--st-header-extra-tokens 32`, ~0.17×) — A.2 says long-range, so launched | 256M / 512M (ids in `LAUNCH_LEDGER.tsv`) | position evals queued (`devloss_pos/`) |

Loss-normalisation "fix" (B.i) is not run: A.1 shows it cannot move the curve under Adam.

### Results

| arm | tokens | actual PF | full CE | 0–2k | 2–8k | 8–16k | 16–32k | 32k–64k | vs dense at equal PF |
|---|---|---|---|---|---|---|---|---|---|
| sd20mix | 256M | 1945 | **1.2668** | 1.386 | 1.290 | 1.245 | 1.211 | 1.289 | dense interpolated at 1945 PF ≈ 1.273 (32M 1.283 @1518, 64M 1.256 @3011, log-linear) → **−0.006**; vs sd20-256M (1523 PF) −0.006 at 1.28× the PF |
| sd20mix | 512M | 5122 | **1.2527** | 1.367 | 1.274 | 1.230 | 1.197 | 1.275 | dense interpolated at 5122 PF ≈ 1.248 → **+0.005**; vs sd20-512M (3043 PF) −0.011 at 1.68× the PF |
| sd20p32 | 256M / 512M | — | launched `01M35P03BF9XJVT9SPERRHK5B9` / `01M35P1A4R42PZYJ178VPHCBME`, position evals `01M35P1G5B5DHJPE6GMW2P31PS` / `01M35P1MG0BJV93YEZ9NN2QPJ9` | | | | | | pending |

The mix curriculum (mean p_full 0.25 → the runs cost 1.28×/1.68× the PF of plain sd20 — more than
the nominal 0.35× because uncompressed rows are ~7.7× the FLOPs of a compacted one) buys
−0.006/−0.011 nats over plain sd20 at the same tokens and lands **on the dense curve, not below
it** at its actual PF: −0.006 at 1.9k PF (inside the ~0.005 interpolation/seed noise), +0.005 at
5.1k. Per position it closes the long-range half of the gap (32k–64k bin: 1.289 at 1945 PF vs
plain sd20-256M 1.295, dense-32M 1.301; 1.275 at 5122 PF vs sd20-512M 1.287, dense-64M 1.277,
dense-128M 1.266) at the price of the compute it spends dense. It does not restore a Pareto margin.

## Diagnosis

The crossover is not an optimizer artefact (A.1), not the shard switch (A.5 — the plateau is, but
the crossover survives on a single shard), and not eval-side drift (A.3). It is a **regime trade**:
soft-detached pooling gives each real token a cheap short-context prediction problem, so per FLOP
it sees ~5× more of them and learns the short-range regime fast — its entire advantage at ≤1.5k PF
is the 0–2k-position bin (−0.06 to −0.07 nats), with every bin past 8k at parity. Dense's tokens
see full context and keep learning long-range use of it; once dense has enough tokens to saturate
the short-range regime (its 0–2k CE reaches sd20-1B's 1.369 at 128M tokens) the only thing left to
learn is long-range, and sd20 cannot learn it from a context that is 80 % slots. Every high-end
pair shows the deficit growing monotonically with position (32k–64k: +0.011 at 3k PF, +0.017 at
6k). A "small tweak" that keeps sd20 ahead therefore has to restore long-range signal *inside the
compaction budget*: the dense-mix curriculum restores it but pays dense FLOPs for it and only
reaches the dense curve; the remaining candidate is `sd20p32` (a real 32-token prefix in every
pooled block at ~0.17×), pending. If that does not hold a margin either, the honest statement is
that soft-detached CPT is a **cheap-end** technique — a 4× FLOP saving up to roughly the budget
where dense saturates short context (~1–2k PF here, i.e. 32M dense tokens on this 4B model) — and
not a replacement for dense at scale.
