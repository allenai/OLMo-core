# Outlier length-mix scaling — results snapshot (2026-08-28 ~05:00 PT)

All f1, 600 eval examples (except DIAG rows: 200), chat format, query_position=after.
Rung mapping: 3k=n22, 8k=n55, 16k=n110, 32k=n220. Sparse evals: local mooney, fast decode
(validated == slow at 8k: 0.052 fast vs 0.06 slow on slm-p8k4000-lropt).

## Check #1 — LR (2k data, 5000 ex, f1@3k)
full: 2.5e-6 .737 | 5e-6 .781/.752 (2 seeds, WINNER) | 1e-5 .699 | 2e-5 .703 | 5e-5 .581 | 1.2e-4 .531
slm:  5e-6 .425 | 1e-5 .578 (WINNER) | 2e-5 .529 | 5e-5 .418 | 1.2e-4 .141
Sparse wants ~2x the LR of full.

## Check #2 — mix vs pure at 8k (lr 2e-5 matched, wave-2)
| run | full@3k | full@8k | full@16k | full@32k | slm@3k | slm@8k |
|---|---|---|---|---|---|---|
| m8k_mix (4000x8k+4000x2k) | .735 | .887 | .395 | .068 | .819 | .627 |
| p8k_4000 | .336 | .744 | .273 | .050 | .153 | .052 |
| p8k_5000 (FLOP-matched) | .244 | .878 | .316 | .045 | .110 | .054 |
| p8k_8000 | .251 | .924 | .414 | .070 | .301 | .600 |
DENSE: mix ≈ p8k_5000 at 8k → transfer ratio ρ(2k→8k) ≈ 0.25 (4000x2k ≈ 1000x8k).
SPARSE: mix .627 vs p8k_5000 .054 — mixing QUALITATIVELY unblocks 8k learning (not additive).

## Check #3 — 2k ladder (lr 2e-5, f1@3k)
full: 1250 .670 | 2500 .769 | 5000 .703 | 10000 .658 | 20000 .791  (saturates ~.75 by 1250)
slm:  1250 .138 | 2500 .200 | 5000 .529 | 10000 .529 | 20000 .498  (saturates ~.52 by 5000; asymptote gap ~.25)

## Check #3 — pure-8k ladder (per-arch opt LR: full 5e-6, slm 1e-5; f1@8k)
| examples | full@8k | slm@8k | full@3k | slm@3k |
|---|---|---|---|---|
| 250   | .422 | .051 | .190 | .135 |
| 1000  | .629 | .056 | .201 | .131 |
| 2000  | .720 | .051 | .365 | .111 |
| 4000  | .819 | .052 | .403 | .137 |
| 8000  | .915 | .224 | .392 | .140 |
| 16000 | .952 | .538 | .277 | .115 |
| 32000 | .980 | .923 | .371 | .228 |
full@16k rung (8k-trained): .155/.173/.252/.298/.424/.473/.449 — off-length ceiling ~.45
DENSE: smooth saturating curve, learns from 250 ex.
SPARSE: sigmoid with takeoff at ~8k-16k ex; MATCHES dense ceiling at 32000 (.923 vs .980) — needs ~4x data.

## Check #4 — 32k-context runs (2000 ex @32k; full 5e-6 / slm 1e-5)
full-p32k2000: 3k .248 | 8k .218 | 16k .150 | 32k .209
slm-p32k2000: eval still running.
2000 ex is deep in the data-poor regime at 32k; .209@32k is nonetheless the best 32k-rung
number so far (8k-trained ceiling ~.105).

## qdmatch_nq smoke (rebuilt p10 data; q2k=5000x2k, q8k=4000x8k)
full-q2k5000: 3k .994 | 8k .286      full-q8k4000: 3k .973 | 8k .948
slm-q2k5000:  3k .031 | 8k .000      slm-q8k4000:  3k .000 | 8k .001
Sparse failure is REAL (diag: well-formed [[q,d]] answers, wrong pairs; train CE gap 15x vs
dense's 2x on outlier). Exact query-doc matching doesn't survive landmark compression at this scale.

## Infra notes for the record
- --master_port=0 breaks torchrun (waits 3x600s then dies): fixed everywhere 2026-08-28.
- mooney /data hit 100% (relay pulls incl. optimizer state ~50GB each): freed 339G by deleting
  S3-backed relay copies; eval wrap now deletes its pulled copy post-eval.
- horton sewonm pool was blocked all night by another user's preemptive_high job.

## FINAL (added ~05:45 PT): sparse 32k eval + law fits
slm-p32k2000 (2000 ex @32k): 3k .006 | 8k .007 | 16k .008 | 32k .015 — hard floor, as the
takeoff model predicts (2000 << sparse takeoff threshold at 32k).

Hill fits to the pure-8k ladders, f1@8k = fmax*N^g/(N^g+K^g):
- dense:  fmax≈1.09 (cap ~1.0), K≈572,   gamma≈0.58 — smooth, resid ≤.02.  N(.8)≈3.4k, N(.9)≈8.6k
- sparse: fmax≈1.37 (unbounded — treat as cap 1.0), K≈20.8k, gamma≈1.70 — sharp sigmoid, resid ≤.05.
  N(.8)≈25k, N(.9)≈31k
Sparse/dense data ratio at f1=0.8: ~7.6x at 8k length (2k length: ~4x by inspection).

Cross-length data-need trend (dense, f1≈0.8): 2k ≤1250 (already saturated) | 8k ≈3.4k | 32k: check#4
observed .21@32k at 2000 ex, consistent with N(.8)@32k ≈ 10-15k (extrapolating K ∝ L^~0.9).
Inferred dense needs: 256k ctx ≈ 60-120k examples; 1M ctx ≈ 250-500k examples (order-of-magnitude).
Sparse pure-length at those lengths is impractical (threshold scales faster); mixed-length
curriculum (m8kmix result) is the lever — it collapsed sparse's 8k requirement from ~20k to <8k mixed.

## ANCHOR REFIT (2026-08-28 ~14:30 PT) — measured 16k + 32k anchors
Dense 16k ladder (f1@16k): 250 .170 | 1000 .370 | 4000 .596
Dense p32k_8000: 3k .202 | 8k .188 | 16k .172 | 32k .472  (32k ceiling NOT low — was data starvation)
Sparse p16k_8000: .079@16k (floor). Sparse p16k_16000: .157@16k (first lift) → sparse takeoff at 16k
is ≥16k examples (vs ~10-15k at 8k length): threshold grows with length, roughly ∝ L^0.5-1.

Per-length Hill fits (fmax 1.089): K(8k)=559 γ.57 | K(16k)=2.9k γ.68 | K(32k)=11k γ.84
**N(f1=0.9) ∝ L^1.50** (measured on 3 lengths):
8k 8.7k ex (0.07B tok) | 16k 29k (0.5B) | 32k 70k (2.3B) | 128k ~560k (74B) | 256k ~1.6M (0.4T) |
1M ~12.8M (13T) | 10M ~400M (4200T — infeasible pure-length)
Verdict: between yesterday's scenarios. 0.9 is genuinely reachable through ~128k with big-but-finite
data; ≥1M pure-length is infeasible → curriculum/mixing required (see mix wave).
Sparse mixes (early): mixs16M .233@8k with 16M tokens ≈ pure-8k at 66M tokens (~4x token efficiency);
mixu64M .396@8k/.169@16k beats pure at equal budget.

## Straggler fold-in (2026-08-28 ~12:45 PT)
sparse two-point: t16 .190/.054/.016/.004 | t32 .480/.127/.038/.014 | t64 .563/.185/.144/.071
sparse long-heavy 64M .626/.238/.099/.023 | 128M .800/.542/.355/.186 (takes off late — absolute
short-example threshold, not shape-impossible). NO conclusion change: sparse ordering stays
short-heavy > uniform > two-point ≈ long-heavy(early)/long-heavy(late catches up), and dense
short-heavy/uniform tie stands. Dense mixs32M + mixl16M evals died on transient HF 429 (tokenizer
download rate-limit); resubmitted (01M15FN2..., 01M15FPF...).

## Wave B: qdmatch_nq ladders COMPLETE (2026-08-28 ~20:15 PT)
Dense 2k ladder (f1@3k | f1@8k): 1250 .993|.570  2500 .991|.617  5000 .994|.286  10000 .996|.023  20000 .999|.007
Dense 8k ladder (f1@3k | f1@8k): 1000 .958|.920  2000 .891|.947  4000 .973|.948  8000 .961|.964
Sparse probes: q2k_20000 .031|.000 ; q8k_8000 .000|.002 — NO late takeoff; architectural failure confirmed.
Findings: (1) dense saturates ~1000 ex at 8k (K~200-400, vs outlier 559 at HALF-max — much easier task);
(2) OOD-length transfer NON-MONOTONIC: 2k-trained @8k peaks at 2500 ex (.617) then collapses to floor
by 10k ex — length-overfitting; outlier's ρ-transfer laws don't universally generalize across tasks;
(3) sparse landmark cannot do qdmatch at any tested scale.

## Waves A + C COMPLETE (2026-08-28 ~21:50 PT)
A. Short-heavy budget laws, 5 points each arch (16/32/64/96/160M tokens), seed replicate ±0.03:
  dense @8k rung: .505/.601/.725/.779/.887 -> Hill(fmax<=1.05): K=19M g=.71, B(0.9)=244M tokens
  sparse @8k rung: .233/.278/.467/.645/.754 -> K=70M g=1.07, B(0.9)=370M tokens (band was 0.5-1.4B; now a line)
  mean-over-rungs gap dense/sparse: 1.71x @16M -> 1.17x @160M — sparse CONVERGES to dense under
  short-heavy as budget grows; sparse token penalty at 0.9@8k is only ~1.5x under short-heavy.
B(0.9)@8k dense short-heavy 244M vs earlier 2-point estimate 310M — revised down.
C. NQ ladders complete (11/11) + 3-task generality:
  nq dense 8k ladder @8k: 1000 .908 / 2000 .928 / 4000 .927 / 8000 .950 (saturates ~1k ex, like qdmatch)
  nq 2k ladder @8k rung (OOD): 1250 .895 / 2500 .890 / 5000 .772 / 10000 .643 / 20000 .575 —
  non-monotonic length-overfit REPLICATES (gentler than qdmatch's collapse to .007)
  SPARSE taxonomy: nq@2k WORKS (.927 at 20k ex — first sparse retrieval success); nq@8k floors at
  8k ex (pre-takeoff, consistent w/ threshold model); qdmatch never; outlier late-takeoff.
  K@8k across tasks spans ~5-10x (outlier hardest). Law SHAPE generalizes; constants are per-task.

## SNQ + 16K-ARMS delivered (2026-08-29 ~10:30 PT)
Sparse nq@2k ladder: 1250 .087 | 2500 .087 | 5000 .215 | 10000 .152 | 20000 .927 — takeoff 10-20k ex
(outlier's 2k takeoff was ~2.5-5k): the sparse bootstrap threshold is TASK-dependent, and once crossed
sparse lands near dense (.927 vs .980).
16k task arms (dense, f1@16k): q16k 2000 .875 / 8000 .927 ; nq16k 2000 .895 / 8000 .922.
K(16k): qdmatch ~190-440, nq ~150-460 — FLAT vs their K(8k) (~250-300). vs outlier K: 559→2.9k→11k.
=> The length exponent is a TASK PROPERTY: outlier (cross-document aggregation) N(.9) ∝ L^1.5;
qdmatch/nq (retrieval) β≈0 — length is ~free once the task is learned. Practical: outlier-type
tasks dominate the long-context data bill; retrieval-type tasks need almost nothing extra.
128k relay: template quoting fixed (YAML-validated), 4 fx2 pushes running, serialized lean evals chained.

## mixs64k96M dense: ABANDONED (2026-08-29 ~10:50 PT)
NaN at step 128 (lr 5e-6) and step 127 (lr 4e-6 retry) — same packed data window both times ⇒
data-driven, LR-independent. Suspect: a pathological packed window in the 64k-tail region
(dense --pack has no cu_doc_lens masking; a 64k example + shorts in one causal stream).
Sparse twin (properly masked landmark packing) trained fine — consistent with the packing-mask
asymmetry. D64 readout will use: p64k_1500 profile + sparse mixs64k96M + existing mixs64M
(no 64k tail). Root-cause belongs with the cu_doc_lens packing fix (known open item), not
another retry. 64k-rung comparisons remain undersized regardless (K(64k)≈40-50k examples).

## Sparse ceiling RESOLVED (2026-08-30 ~17:00 PT)
slm p8k_64000 (lr 1e-5): f1@8k = .967 (3k rung .246). vs 32k-ex .923 and dense .980.
=> Sparse landmark reaches the SAME asymptote as dense at 8k length; the 2k plateau gap (~.52 vs
.75) was the anomaly. Refit (fmax<=1.05): fmax .985, K 14.8k, gamma 2.55; N(0.9) ~ 38k examples
(dense 8.7k) — a 4.4x data penalty to 0.9, pure-length.
Extended length laws (in-length, dense, RAW f1 with lexical baseline in parens):
qdmatch: 32k 2000->.735 / 8000->.793 (baseline .55); 64k 1000->.518 / 4000->.656 (baseline .365
on the de-correlated rung) — NOT flat anymore: K and ceiling both degrade once the shortcut dies.
nq deep: 32k-trained 2000->.865 / 8000->.910 (baseline .27); 64k-trained 1000->.788 (baseline .21);
nqD32k8000 scores .863@64k OOD — NQ stays cheap and generalizes; K(32k)~2-4k ex, still tiny in tokens.
nq-18k (clamped arms) models score .89-.91 on the REAL 32k rung — length generalization, not memorized length.

## EXTENSION WAVE COMPLETE (2026-08-30 ~18:00 PT) — final per-task length laws
nqD64k_4000 landed .857@64k, closing the last eval. Final N(0.9) per length (dense, in-length):
outlier 8.7k/29k/70k/~150k† ex @8k/16k/32k/64k† (beta 1.5); nq-deep 0.9k/3k/7.5k/~10k† (beta ~1.1,
task constant ~15x below outlier); qdmatch 0.8k/5k/unreached(.79 max)/unreached(.66 max) — its
apparent beta=0 was the lexical shortcut (baseline .97/.83/.70/.55/.37 across 2k-64k); the real
pairwise task surfaces at 32k+ with rising K AND falling ceiling. († = one-octave extrapolation.)
REVISED HEADLINE: N0.9(L) = C_task * L^beta with beta ~ 1-1.5 for every real task; C spans ~15-50x;
apparent beta=0 = saturation or shortcut artifact. The data bill for a long-context mix is set by
its highest C*L^beta task. Sparse: same ceiling as dense (.967 vs .980 at 8k), 4.4x data to 0.9.
Deep-NQ source regen (10 min GPU) unlocked real 32k/64k NQ + de-correlated qdmatch 64k evals.

## Certainty wave, verdict 2: the qdmatch 32k ceiling is REAL (2026-08-31)

Question: does qdmatch_nq@32k break past f1=.793 given 16x the training data?
Run: lmx-full-q32k32000-qd-4b — q32k_32000 pure-length arm (M=N=172, 32,000 examples,
974.8M tokens, seq 32768 --pack, lr 5e-6, diagnostic exception to the short-heavy recipe).
Eval: held-out qdmatch_nq rungs, eval_size=600 each (binomial SE ~±0.016 at f1≈0.8).

  32k ladder (dense): 2,000 ex -> .735 | 8,000 -> .793 | 32,000 -> .812
  64k transfer:       q64k_1000 -> .518 | q64k_4000 -> .656 | q32k_32000 -> .713 (!)

VERDICT: 4x more data past the previous best buys +.019 (~1.2 SE) — the increments decay ~3x per
4x data, extrapolating to an asymptote of ~.83-.84 at 32k. Combined with the ambiguity audit
(AMBIGUITY_AUDIT.json: resolvable-adjusted ceiling is flat .970-.983 at every rung, and the model
BEATS the uniform-ambiguity ceiling at 8k/16k), label noise does not explain the gap: qdmatch has
a real, mid-band performance ceiling at long context that data cannot buy through. The artifact's
"above ceiling" cells for qdmatch at >=32k are correct and now measured, not extrapolated.

Bonus: the 32k-trained big arm TRANSFERS UP — .713@64k beats every 64k-trained arm (best .656 at
4,000 ex, the pool-feasible max). Upward length transfer with a big in-distribution corpus
outperforms small in-length training, consistent with the outlier catalytic-mass findings.

Caveat noted in-file: the q32k pool extension placed the old 300-example heldout inside the new
train prefix — heldout-CE for this arm is contaminated; the graded rungs above are from the
disjoint validation units and unaffected.

## Certainty wave, verdict 1: NO sparse token-crossover in the measured range (2026-08-31)

Question: does sparse beat dense on TOKENS at 16k/32k inside the measured range, as the earlier
slope-based extrapolation (sparse B∝L^1.2-1.4 vs dense L^1.8-2.0) implied?
New runs: mix_s320M + mix_s640M, both archs, short-heavy, optimal LRs. All rungs eval_size=600.

  @16k: dense .453(64M) .659(160M) .753(320M) .894(640M) | sparse .216(64M) .363(96M) .497(160M) .700(320M) .816(640M)
  @32k: dense .107 .266 .352 .532                        | sparse .054 .102 .186 .254 .386

VERDICT: NO. Sparse never beats dense on tokens at any rung/budget through 640M. The sparse/dense
score ratio converges (~2x -> 0.91@16k, 0.73@32k) then PLATEAUS — the convergence that motivated
the crossover hypothesis stalls short of parity. At matched score sparse needs ~1.5-1.7x tokens
(measured, no fit: dense reaches sparse-640M's .816@16k at ~410M; its .386@32k at ~370M).

Hill refits with the extended range (fmax free <=1.05):
  dense @16k: fmax 1.03 g .85 K 86M  -> B(.8)=371M  B(.9)=834M   (rmse .017)
  sparse@16k: fmax 0.90 g 1.45 K 133M -> B(.8)=557M  B(.9) at/above ceiling (fmax≈target)
  dense @32k: fmax 1.01 g .90 K 586M -> B(.8)=2.6B  B(.9)=6.1B   (rmse .018)
  sparse@32k: reaches only .386 in-range; >=1.7x dense at matched f; ceiling unconstrained.
Seed-replicate error bars (this wave + prior): +-.02-.03 per point — the .078 gap at 640M@16k is ~4 SE.

Artifact updated: 16k row now MEASURED both archs at f=.8; sparse .9@16k flagged ceiling≈.90;
sparse >=64k cells recomputed as ~1.6x dense tokens with wall-clock from the throughput model
(sparse GPU-h parity ~64-128k, decisive win >=256k — a FLOP advantage, not data efficiency).
The old "sparse crossover at 16-32k" paragraph replaced with the measured refutation.

## nq law MEASURED, derived cells retired (2026-09-01)

The nq mix ladder is complete at every rung (short-heavy, eval_size 600/rung):

  2k:  16M .977 | 32M .977 | 48M .973      (saturated, no slope -- 2k DRIFTS -.004 over 3x data)
  8k:  16M .917 | 32M .930 | 48M .933
  16k: 16M .857 | 32M .885 | 48M .910      (+.028 then +.043 per doubling)
  32k: 16M .760 | 32M .837 | 48M .873      (+.077 then +.062 per doubling)

Hill fits: 16k g=.45 K=0.3M -> B(.8)=6M, B(.9)=41M; 32k g=.70 K=3.1M -> B(.8)=22M, B(.9)=71M.
fmax hits its 1.0 bound at both rungs, so the .9 budgets are the soft end. The artifact's derived
cells (25M/190M/0.95B/1.5B for f=.8) were 8-40x PESSIMISTIC -- the same failure mode as the
qdmatch derived cells, and for the same reason: dividing a pure-length need by the target rung's
token share ignores how much the short rungs transfer.

Pure-length anchors: nqD32k_4000 .888@32k, nqD64k_2000 .875@32k / .838@64k. Spending every token
at the scored rung buys ~+.015 over the mix -- the mix is nearly free at length on this task.

SPARSE vs dense at a matched 48M: .912/.728/.608/.248 against .973/.933/.910/.873. Sparse holds
94% of dense at 2k and 28% at 32k; the deficit compounds with length. Two of the three tasks now
measured head-to-head (qdmatch, nq) show that collapse; only outlier shows the constant-factor
penalty that the "sparse just needs more tokens" story assumes.

qdmatch 64k transfer (neither arm trained at 64k): q32k_16000 (487M tok) -> .706,
q32k_32000 (975M) -> .713. +.007 for a doubling, the same flat tail as its 32k ladder.

outlier 32k pure-length ladder completed by p32k_4000: 2,000 ex/66M -> .209, 4,000/131M -> .302,
8,000/262M -> .472. Increments ACCELERATE (+.093 then +.170) -- still in takeoff, unlike qdmatch.
Refit fmax 1.05 g .88 K 339M (rmse .011) -> B(.5)=304M, B(.7)=742M, B(.8)=1.26B, all extrapolated
from inside a rising regime. Its 16k rung reads .119 -- a downward transfer from 32k-only training,
against .453 for the 64M short-heavy mix, which is the clearest single argument for the mix recipe.
