# CTC-suite: Qwen3.5-4B dense (full-attn) vs document-chunked

All eval_size=500 unless noted. metric per task. `flag`: ⚠=parse_rate<0.5 (corruption); CHUNKED-ONLY=no dense checkpoint; UNRELIABLE=near-floor in both arms (task-level issue).

| task | metric | arm | 2k | 4k | 8k | 16k | 32k | flag |
|---|---|---|---|---|---|---|---|---|
| absence_gutenberg | set_f1 | dense | 0.964 | 0.976 | 0.986 | 0.932 | — |  |
| | | chunked | 0.949 | 0.961 | 0.981 | 0.945 | — | |
| contradiction | set_f1 | dense | 0.849 | 0.766 | 0.690 | 0.619 | 0.335 |  |
| | | chunked | 0.473 | 0.415 | 0.352 | 0.317 | — | |
| cycle | cycle_f1 | dense | 1.000 | 1.000 | 0.998 | 1.000 | — |  |
| | | chunked | 0.996 | 0.994 | 0.994 | 0.992 | — | |
| grouping | pairwise_f1 | dense | 0.439 | 0.358 | 0.186 | 0.043 | 0.011 |  |
| | | chunked | 0.439 | 0.357 | 0.185 | 0.041 | 0.009 | |
| grouping_labeled | pairwise_f1 | dense | 0.439 | 0.365 | 0.231 | 0.054 | — |  |
| | | chunked | 0.439 | 0.370 | 0.226 | 0.051 | 0.018 | |
| groups4 | cycle_f1 | dense | 0.010 | 0.010 | 0.000 | — | — | DENSE EXCLUDED (-full ckpt const-collapsed under full attn) |
| | | chunked | 0.000 | 0.000 | 0.000 | — | — | |
| helmet_qa | token_f1 | dense | 0.033 | 0.036 | 0.039 | — | — |  |
| | | chunked | 0.067 | 0.071 | 0.071 | 0.071 | 0.075 | |
| helmet_summ | rouge1_f | dense | 0.329 | 0.338 | 0.341 | — | — |  |
| | | chunked | 0.335 | 0.341 | 0.346 | 0.347 | 0.349 | |
| hotpotqa | gold_id_f1 | dense | — | — | — | — | — | CHUNKED-ONLY(no -full ckpt) |
| | | chunked | 0.995 | 0.987 | 0.983 | 0.965 | — | |
| mathmatch | set_f1 | dense | 0.003 | 0.001 | 0.000 | — | — |  |
| | | chunked | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | |
| msmarco | gold_id_f1 | dense | 0.961 | 0.939 | 0.917 | 0.900 | — |  |
| | | chunked | 0.897 | 0.902 | 0.868 | 0.871 | — | |
| niah | gold_id_f1 | dense | 0.988 | 0.988 | 0.976 | 0.984 | 0.966 |  |
| | | chunked | 0.984 | 0.994 | 0.976 | 0.978 | 0.970 | |
| nq | gold_id_f1 | dense | 0.904 | 0.930 | 0.864 | — | — |  |
| | | chunked | 0.964 | 0.942 | 0.896 | — | — | |
| ~~obliq_retrieval~~ | gold_id_f1 | dense | — | — | ~~0.229~~ | — | — | ❌ VOID — degenerate ladder, see below |
| | | chunked | — | — | ~~0.226~~ | — | — | |
| **obliq_twitter** (rebuilt) | gold_id_f1 | dense | 0.740 | 0.659 | 0.608 | 0.277 | 0.327 | ⚠ eval_size=126 (SE ±0.045); NOT comparable to the void row |
| | | chunked | 0.689 | 0.605 | 0.563 | 0.262 | 0.293 | |
| **xabsence** (exact-copy) | set_f1 | dense | 0.609 | 0.545 | 0.528 | 0.505 | 0.518 | ⚠ dense score is HALF-TRIVIAL — see A/B asymmetry below |
| | | chunked | 0.135 | 0.057 | 0.037 | 0.012 | 0.008 | |
| oolong | partial_credit | dense | — | — | — | — | — | CHUNKED-ONLY(no -full ckpt) |
| | | chunked | 0.628 | 0.523 | 0.390 | 0.297 | 0.297 | |
| outlier | set_f1 | dense | 0.982 | 0.956 | 0.877 | 0.679 | 0.428 |  |
| | | chunked | 0.962 | 0.869 | 0.641 | 0.343 | 0.125 | |
| outlier_amzn | set_f1 | dense | 0.921 | 0.891 | 0.896 | 0.870 | 0.864 |  |
| | | chunked | 0.912 | 0.894 | 0.887 | 0.875 | 0.858 | |
| qdmatch_hpqa | pair_f1 | dense | 0.999 | 0.997 | 0.998 | 0.992 | 0.981 |  |
| | | chunked | 0.650 | 0.760 | 0.668 | 0.547 | 0.333 | |
| reorder | kendall_tau | dense | 0.747 | 0.600 | 0.262 | 0.047 | — | 16k=genuine floor (model can't emit exact 116-perm) |
| | | chunked | 0.471 | 0.214 | 0.043 | 0.000 | — | |
| rerank | mrr@10 | dense | 0.989 | 0.971 | 0.960 | 0.952 | — |  |
| | | chunked | 0.980 | 0.966 | 0.950 | 0.949 | — | |
| scifact | gold_id_f1 | dense | 0.271 | 0.087 | 0.058 | — | — | DENSE EXCLUDED (-full ckpt const-collapsed under full attn) |
| | | chunked | 0.976 | 0.956 | 0.933 | 0.886 | 0.840 | |
| strmatch | set_f1 | dense | 0.999 | 0.997 | 0.997 | 0.995 | 0.994 |  |
| | | chunked | 0.003 | 0.003 | 0.000 | 0.001 | 0.000 | |
| textgroups | textgroups_f1 | dense | 0.196 | 0.090 | 0.054 | 0.051 | 0.047 |  |
| | | chunked | 0.087 | 0.021 | 0.019 | 0.007 | 0.001 | |

**Coverage:** dense grades=83, chunked grades=100, tasks=23.

## Overnight session verdicts (2026-07-22) — every blank has a reason

**Filled this session (14 cells, verified):** niah dense 16k/32k=0.984/0.966 (a false 0.096/0.182 from a
stale parser on the re-stage node was caught + corrected), qdmatch_hpqa 16k/32k=0.992/0.981,
outlier_amzn dense 16k/32k=0.870/0.864 + chunked 32k=0.858, strmatch 16k=0.995, rerank 16k=0.952,
msmarco 16k=0.900, cycle 16k=1.0, grouping_labeled 16k/32k=0.054/0.020, textgroups 16k/32k=0.051/0.047.
Plus the **outlier(wiki)** scale-K rebuild (dense 0.982/0.956/0.877/0.679/0.428) is now in.

**Blank = capped by policy (correct, NOT missing):** reorder / cycle / groups4 at 32k (rung policy caps
these at 16k).

## 2026-07-26/27 update — obliq VOIDED and rebuilt; xabsence rebuilt twice

**obliq_retrieval is VOID, not merely "unreliable".** Baselines exposed it: on that ladder a
random-k guess scores **0.663/0.572/0.413/0.237/0.150** — beating BOTH trained arms at every rung —
because random-k's F1 ≈ the gold density, and the ladder's gold density was 65%→17%. The "32k" rung
held a median of **30 documents, 7 of them gold**; at 2k the median was **3 documents** and **43% of
examples had gold == every document**, i.e. answerable by printing all ids without reading. BM25
(0.630) and Gemini embeddings (0.728) also beat the trained model. Root cause:
`build_obliq_token_ladder.py` never drops golds, so shrinking the token budget inflates the gold
fraction instead of adding distractors. The builder's own docstring already called 2k/4k/8k
ill-posed; they were built anyway with `--keep-fitting`.

**obliq_twitter is the rebuild.** Twitter is the only subset whose candidate pool (500 docs) is
large enough to dilute golds — writing/wildchat are capped at 30–50 candidates and are stuck at
27%/53% gold density regardless of budget. Golds are held at a constant set across rungs while only
distractors scale, taking gold density to **1.8%** at 32k and the random baseline from 0.150 → **0.018**.
Train/eval gold counts now match (8.2 vs 7.9 at 2k; 10.5 vs 9.1 at 8k+) — the old data had 2.23 vs
7.23, which made the model emit ~2 ids where ~7 were needed. Bars on this ladder: non-trivial =
predict-ALL+2SE (0.36/0.23/0.13/0.07/0.04); competitive = Gemini (0.44/0.34/0.24/0.16/0.12). Both
arms now clear Gemini at every rung. ⚠ eval_size=126 — treat sub-0.04 differences as noise; enlarging
this eval past 500 is the highest-value remaining fix.

**xabsence: three versions, first two VOID.** (1) trained to 99 docs but evaluated to 669 → the model
never emits a 3-digit index, 40% of golds unreachable at 8k → a fake long-context collapse
(0.566→0.069). (2) index fixed, but orphans were inserted as ORIGINALS into a corpus of LLM
PARAPHRASES → detectable per-document by style, recall **0.98 on B-side vs 0.08 on A-side**, pinning
f1 at ~0.5 FLAT in n. (3) the row above: exact-copy twins over 3-sentence abstract chunks, so an
orphan is textually indistinguishable from a matched document and can only be found by cross-corpus
comparison. Train/eval pools disjoint (17,167 vs 4,655 chunks, 0 overlap); golds fixed at 2 while
docs scale 16→280.

⚠ **The xabsence dense score is half-trivial and must not be read as an all-pairs result.** Documents
are laid out as an A block then a B block, so a B-side orphan is "the one item in block 2 that is not
a repeat" — local, and easy for a causal model (recall **1.000**). An A-side orphan requires looking
FORWARD ("never repeated later"), which causal attention cannot do (recall **0.016** at 32k). Finding
1 of 2 golds gives f1 = 0.5, which is exactly the plateau. The chunked arm fails BOTH sides
(recall_B 0.006 at 32k) — it cannot ask "have I seen this before" at all, which is the real
dense-vs-chunked signal here. Fix for a symmetric task: interleave A and B rather than blocking them.

**obliq_retrieval — ladder BUILT 2026-07-22 (superseded by the above).** Its "rung_8192" was actually the FULL 488-set at native
~27k tokens (mislabeled). Gold footprints span 288–47k tokens (multi-gold, 1–64 golds/example), so a
uniform-doc rung is not a uniform token budget: **2k/4k/8k are ILL-POSED** (49–89% of examples can't
fit their own golds) → NOT built. A gold-preserving per-example token-budget subsampler
(`build_obliq_token_ladder.py`) produced clean **16k (484 ex) / 32k (486 ex)** rungs. Eval: dense
16k/32k=0.248/0.239, chunked 16k=0.239 (⚠ eval_size 484/486). Flat ~0.24 across the whole ladder, both
arms — a coherent data-poverty/mode-collapse signal, not a single mislabeled point.

**Blank = eval ladder never built (blocked):** nq 16k/32k (CE canonical caps ~117 docs → 16k borderline
<500, 32k needs canonical regen), and the 32k tails for msmarco/rerank/absence.

**Low = GENUINE (verified, not a bug):** scifact-dense & groups4-dense (`-full` ckpt training collapse,
checkpoint-specific — chunked arm fine); mathmatch (constant-attractor floor); grouping/textgroups
(real O(N·M) difficulty scaling, dense≈chunked); helmet_qa (retrieval-tuned ckpt on free-form
NarrativeQA → genuine floor, ~2× artifact-depressed by `</think>` truncation but still a floor).

**Low = ARTIFACT (NOT accepted as a result):** groups4-**chunked** 2k/4k/8k=0.000 is a constant-output
collapse (489/500 identical gens) under document-chunked attention — a chunked-attention-internals bug,
not a capability score. Excluded pending dedicated investigation.

**Chunked-only (no usable dense ckpt):** hotpotqa (S3 distcp incomplete — 80/128 shards), oolong (no
-full ckpt).

**Headline CTC pattern (now well-populated at 16k/32k):** dense≈chunked on O(N) retrieval (niah, nq,
rerank, msmarco, cycle, absence, outlier_amzn all ~0.9+ flat); dense≫chunked on O(N·M)/O(N²)
comparison (qdmatch_hpqa dense 0.981 vs chunked 0.333 at 32k; strmatch 0.994 vs 0.000; outlier(wiki)
0.428 vs 0.125) — the gap widens with N, exactly the CTC thesis.
