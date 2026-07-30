# 5-task SFT mix: 2k → 256k, short-skewed (Qwen3.5)

**Goal.** Extend the canonical 5-task mix (contradiction / nq / oolong / rerank / outlier) from its
current 32k ceiling to a 2k–256k ladder, skewed toward short contexts, with matching eval data at
every rung.

**Decisions (user, 2026-07-26).**
- Tokenizer/markers: **Qwen3.5-4B only** — `--marker-set qwen3_5`,
  `--tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base`.
- Skew: **short-heavy**, with a hard floor of **≥300 examples in the 128–256k band**.
- Long end: **expand all five tasks to 256k** (no capping).
- Eval: **rebuild all xlong rungs at eval_size=500**, plus the missing oolong rungs.

---

## 0. TWO shard sets: chunked has markers, standard does not

⚠ **Superseded claim.** This section originally said one marker-wrapped shard set serves both arms
via a train-time `--variant` flag. That is the **CTC-suite** convention and does NOT apply to the
5-task lineage, which keeps the roots separate (`single_task_ladders_v2` = standard vs
`single_task_docchunk_v2` = chunked) and whose plain-SFT converters emit **zero** marker references.
The evidence originally cited was about shared *eval rung files*, which is a weaker claim than
shared *training shards*.

So two sets are built from the same pools:

| | flag | contents |
|---|---|---|
| `shards_chunked/` | (default) | WITH `<\|box_start\|>` / `<\|box_end\|>` |
| `shards_full/` | `--no-doc-markers` | NO markers — plain full attention |

`--no-doc-markers` renders the identical prompt (same `build_prompt`, chat template, tokenizer, EOS)
with the boundary strings empty, so the pair differs ONLY by 2 tokens per document. Verified on one
example: 7811 vs 7465 tokens, delta 346 == 2×173 docs, identical answer spans, 0 marker ids in the
standard build. `metadata.json` records `doc_markers: true|false`.

---

## 1. Bad variants avoided

| Trap | Status |
|---|---|
| **oolong `--item-regex '\|\|'`** — bare `\|\|` is an empty-branch alternation matching *every* line, so instruction/question/header became their own chunks with FREE `\n\n` bridges between them (`CHUNK_LEAK_AUDIT.md`: 2019 inter-chunk FREE tokens, ~5/example) | **FIXED 2026-07-26.** Converter now rejects any `--item-regex` matching the empty string; 4 call sites + BUILD_MATRIX fixed; regression test `src/test/data/document_chunk_item_regex_test.py` (3 passed). Verified: bad→7 chunks/3 bridges, good→3 chunks/0 bridges. **All pre-2026-07-26 oolong shards must be rebuilt.** |
| FREE-token id/title wrapping leak | Fixed in `abccf2837`, confirmed ancestor of HEAD |
| NQ 98%-hard (hn49/hn99/hn199/ladder64k) | Avoided — using p10 pool, audited hard-neg mean **0.097** |
| contradiction gold off-by-one | Audited PASS (see §2 — index base is per-task) |
| outlier shrink / scale-K | Audited PASS, golds in `[0, n-1]` |
| rerank raw BM25 dump | Audited PASS, 300/300 carry `ce_scores` |
| `free_pad_repeat` / `repeat_doc_text` (verbatim-repetition, void) | Left at defaults `0` / `1` |
| `--use-titles` (title shortcuts the task) | Left **OFF** |

## 2. ⚠ `gold_doc_indices` index base is PER TASK

Verified against both the pools (4000 contradiction examples: min gold = 1, **never 0**) and the
renderer (`corpus_reasoning/lib/data_format.py:1083` — *"These are already 1-indexed claim IDs"*,
`json.dumps`ed verbatim):

| task | base | valid range |
|---|---|---|
| **contradiction** | **1-indexed** | `[1, n_docs]` |
| outlier / rerank / nq | 0-indexed | `[0, n_docs-1]` (rendered answer adds 1) |

Applying the 0-indexed convention to contradiction produces a spurious "1.7% out of range" — that
was a false alarm in the first audit pass, not a data defect.

## 3. Qwen3.5 calibration (measured, not assumed)

`debug/xlong_5task/calibrate_and_audit.py`, 300 real examples/task through the same `build_prompt`
the converter and native eval call. Fit `tokens = a + b·n_docs`:

| task | intercept | tok/doc | MAPE | pool n range | Qwen3 tok/doc (old) |
|---|---|---|---|---|---|
| contradiction | 188.1 | **42.41** | 2.5% | 50–949 | 40.91 |
| nq | 26.1 | **156.54** | 1.2% | 25–202 | 157.15 |
| outlier | −5.4 | **144.33** | 3.3% | 14–220 | 146.25 |
| rerank | 13.8 | **85.23** | 3.1% | 20–313 | 85.39 |

**Finding: Qwen3.5 ≈ Qwen3 to within 1–3%.** The existing eval rungs' `n` values are therefore
valid under Qwen3.5 and do *not* need recalibration — only the rebuild to eval_size=500.

(Note: CLAUDE.md quotes a Qwen3.5 contradiction fit of `288.7 + 22.82·n`. That is ~2× off the
measured 42.41 tok/doc on this pool and must refer to a different contradiction variant; the
measurement above is on the actual `pubmed_realistic` pool being built from.)

### `n` per band

| band | contra | nq | outlier | rerank |
|---|---|---|---|---|
| 2k | 44 | 13 | 14 | 24 |
| 4k | 92 | 26 | 28 | 48 |
| 8k | 189 | 52 | 57 | 96 |
| 16k | 382 | 104 | 114 | 192 |
| 32k | 768 | 209 | 227 | 384 |
| 64k | 1541 | 418 | 454 | 769 |
| 128k | 3086 | 837 | 908 | 1538 |
| 256k | 6176 | 1674 | 1816 | 3076 |

Current pool ceilings: contra 949 (~40k), outlier 220 (~32k), rerank 313 (~27k), nq 202 (~32k).
Everything above 32k requires pool expansion. oolong is token-budgeted directly
(`--len-min/--len-max`) and needs `--pool-max-ctx` raised from 131072 → 262144.

## 4. Length distribution (per task, ~20k examples)

| band | examples | ~tokens |
|---|---|---|
| 2–4k | 6000 | 18M |
| 4–8k | 5000 | 30M |
| 8–16k | 4000 | 48M |
| 16–32k | 2700 | 65M |
| 32–64k | 1400 | 67M |
| 64–128k | 600 | 58M |
| **128–256k** | **300** ← floor | 58M |
| **total** | **20000** | **~344M** |

~1.7B tokens across 5 tasks → ~8.6 GB of shards (uint32 ids + bool mask). Counts fall roughly as
1/T², so ~57% of *examples* sit under 8k while the long tail is still thick enough (300) to train
the 256k format.

## 5. Build location

Node-local `/data`, then S3 → weka via gantry (an S3 push alone leaves rungs MISSING while the job
exits 0).

| node | /data free | note |
|---|---|---|
| **mcfuzz** | 14T (1%) | best free space, jsteinhardt |
| thidwick | 54T (5%) | berkeleynlp |
| mooney | 6.8T (52%) | |
| cubbins | 477G (91%) | source pools live here; too tight to build on |

Source pools on cubbins `/data/prasann/ctc_suite_data/{contradiction,outlier,rerank,oolong}_pool/`,
nq at `/scratch/users/prasann/nq_p10_20k/`. Node-local conda at
`/data/prasann/conda/envs/corpus-reasoning-olmo`, HF cache `/data/prasann/hf_cache`.

---

## ⚠ State divergence: local oolong shards are STALE vs weka (2026-07-27)

**Do not sync `/data/prasann/xlong5/shards_*/oolong_train` to weka.** It would overwrite the good
copy with a worse one.

| | shards_chunked/oolong_train | shards_full/oolong_train |
|---|---|---|
| **weka / S3 (authoritative)** | 19,994 inst, 354.7M tok | 20,000 inst, 342.7M tok |
| local (stale) | 19,758 inst, 351.9M tok | 19,764 inst, 340.0M tok |

Why: a prefix-based signature (query + first 200 chars) appeared to show the oolong train pool
overlapping the 8k/16k/32k eval rungs (108/57/18 examples). 236 training examples were dropped and
oolong re-tokenized. A full-body comparison afterwards
(`verify_contamination.py`) showed **0 exact duplicates and 0 shared bodies** — all 183 hits were
prefix collisions, which oolong produces by construction (shared preamble + opening item). There
was never any contamination, so the removal was unnecessary and the pre-removal shards on weka are
the correct ones.

The local re-tokenization `rm -rf`'d its output dir first, so the full-data shards now exist ONLY on
weka. To rebuild them locally, re-run `tok_oolong_clean.sh` against `pools/oolong/` (the original
pool) instead of `pools_oolong_clean/`.
