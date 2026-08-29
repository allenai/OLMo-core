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
