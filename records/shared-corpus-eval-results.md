# Shared-corpus ("efficient") v2 evals — results

2026-08-07. Companion to `records/shared-corpus-eval-plan.md` (construction) and
`debug/shared_corpus_eval/STATUS.md` (job-level detail). **EVAL ONLY** — no training data, shards
or checkpoints were modified.

## 1. The headline

Across all five tasks, shared-corpus evals **preserve scores only in the configurations that give no
prefill reuse, and give prefill reuse only in the configurations that do not preserve scores.**
Every arm measured this session falls on one side or the other; nothing landed in the corner that
would actually be useful.

| task | configuration | shared docs | reusable *token* prefix | score vs independent | verdict |
|---|---|---|---|---|---|
| nq | multiplexed | 100% | 0.5% | −0.016 (2·SE 0.033) | faithful, **not** cheaper |
| rerank | multiplexed | 100% | 0.9% | −0.002 (2·SE 0.018) | faithful, **not** cheaper |
| outlier | scatter (control) | 95% | ~0% | +0.010 (2·SE 0.042) | faithful, **not** cheaper |
| outlier | prefix+tail | 95% | 94.7% | **+0.215 / +0.261** | cheap, **not** faithful |
| contradiction | prefix+tail | 75–95% | 95.0% | **−0.102 … −0.175** | cheap, **not** faithful |

Harness sanity: the re-run independents reproduce the published grid (nq 0.858 vs 0.864, rerank
0.958 vs 0.960), so none of this is drift.

**oolong is the degenerate case and is listed separately on purpose.** Its shared file is the same
400 rows as the independent one in a different order (the source split already stores 25 questions
per context over 16 contexts), so alignment is exact *by construction* — the native evaluator scores
rows independently and order cannot change the result. There is therefore no alignment to measure,
and no speedup to collect either: oolong's reusable token prefix is 0.1%, so it sits with nq and
rerank in the "faithful, not cheaper" row. The run itself:
`partial_credit=0.048  ⚠ eval_size=400 only (SE ±0.011)`, and it took **29,699 s — 8.25 hours** for
400 examples on one H200. That is the cost problem stated precisely, on the task where none of this
machinery can help. (Not comparable to the grid's oolong 32k cells: those use a different generation
whose question set overlaps this one by only 41/248.)

## 2. Why the two families split

The split is not a coincidence of construction — it has one cause on each side.

**Multiplexed tasks can't be cached because of the prompt format.** These checkpoints train with
`query_position="both"`, which renders `{questions}\n\n{documents}\n\n{questions}`. The per-query
question is emitted *before* the corpus, so a byte-identical document block is not a *token* prefix
and nothing can be reused. Measured longest common token prefix within a corpus group: nq 0.5%,
rerank 0.9%, oolong 0.1%, against contradiction 95.0% and outlier 94.7% — the latter two only
because their question block is empty (contradiction) or a fixed generic string (outlier).

**The only workaround destroys the model.** Moving the question to `query_position="after"` makes
the corpus a true prefix. Probed on the same 500 nq queries (job 3426850):

| run | query_position | gold_id_f1 |
|---|---|---|
| independent | both (as trained) | **0.860** |
| independent | after | **0.074** |
| shared + cache | after | 0.036 |

Not a mild off-distribution cost — a collapse. These checkpoints cannot do the task when the
question appears only after the context. `--query-position` exists in the evaluator now, defaults
to `both`, and no reported number uses anything else.

**Prefix+tail tasks can be cached but the placement changes the task.** For outlier this is
attributed *exactly*, by a control that holds corpus contents fixed and moves only the golds:

| outlier | set_f1 |
|---|---|
| independent | 0.320 |
| **scatter control** — same rebuilt corpus, trio scattered uniformly | **0.330** (aligned) |
| prefix+tail, 5% tail | 0.535 |
| prefix+tail, 25% tail | 0.580 |

The rebuilt corpus is faithful; **the whole +0.22–0.26 comes from putting the golds at the end.**
Note this is *not* the "guess the tail" shortcut the plan anticipated — the 25% tail has a far lower
shortcut floor (0.055 vs 0.273) yet scores *higher*. It is an end-of-context attention effect, and
it is large: it roughly doubles a near-floor score.

For contradiction the same construction pushes the other way, and no amount of sharing fixes it:

| contradiction | corpora | foreign-gold density | set_f1 |
|---|---|---|---|
| independent | 500 | 0% | **0.579** |
| shared, 125 q/corpus | 4 | 53.4% | 0.477 |
| shared, 25 q/corpus | 20 | 13.4% | 0.404 |
| shared, 8 q/corpus | 63 | 4.6% | 0.420 |

Flat and low across a 16× range of sharing, so "share less" is not a fix. Three mechanisms were
tested and refuted: pair separation (recovery is flat across separation quintiles in the
independent run: 0.563/0.582/0.581/0.607/0.538), preferential confusion with other queries'
orphaned half-pairs (false positives touch them 76.9% vs a 78.3% chance rate), and corpus-level
clustering inflating the error (per-corpus means 0.453–0.500; clustered 2·SE 0.020, *below* the
binomial 0.045). The residual conclusion is that the construction itself changes the task.

The direction is at least the safe one: contradiction scores *lower*, so the pair-split
(one member of each pair in the shared prefix, its partner in the tail) did its job — recency never
hands the model a whole answer.

## 3. What is nonetheless usable

- **The shared data for nq / rerank is valid and worth keeping.** It is score-preserving, and it
  creates 32k rungs that the canonical CE-filtered pools cannot reach on their own (they cap at
  48 and 100 documents per query; a pooled corpus has no such cap). It also cuts unique corpora
  ~30×, which matters for staging and storage even though it does not cut compute.
- **The cache machinery works and is verified.** Generations are byte-identical with and without
  `--shared-corpus-cache` (16/16 on the parity smoke test), and where a prefix does exist it
  delivers: **8.1× fewer prompt tokens fed**, wall clock 1,281 s → 698 s.
- **outlier's scatter build is a faithful eval** (0.330 vs 0.320) — just not a cacheable one.

## 4. What would actually unlock this

Recorded as an observation, not a proposal — the request was eval-only and nothing here justifies
retraining on its own:

1. **The binding constraint is a training decision, not an eval one.** `query_position="both"` is
   what puts the question ahead of the corpus. A model trained with the question after the context
   would make multiplexed corpora both faithful *and* ~20× cheaper, since those tasks already
   preserve scores at 100% document sharing.
2. **The document-chunked arm may be cacheable where the full arm is not** — worth checking before
   assuming. Under chunk masking, context↔context edges are restricted; whether a document's KV is
   independent of a *preceding* query block depends on whether context tokens attend to FREE
   tokens. If they do not, the leading question stops mattering and Family A becomes cacheable in
   the chunked arm without any prompt change. This was not tested (every run here is full
   attention) and should not be assumed either way.
3. **Do not ship prefix+tail evals for contradiction or outlier.** Their numbers are not comparable
   to the independent ladder in either direction.

## 5. Bugs found and fixed along the way

- **outlier topic-selection bug (mine).** Selecting the topic bank largest-first — added to stop the
  pinned 4-per-topic minimum overflowing the prefix — pushed the smallest majority topic well above
  4, while the source data always has exactly 4 (`maj_outlier_gap=1`), i.e. a gap of one above the
  3-document trio. That gap *is* the task; widening it inflated outlier from 0.320 to **0.887**.
  Fixed with random topic order plus explicit re-imposition of the gap; the builder now reports
  `min_majority_topic_size` / `gap_above_trio` in every manifest and the first invalid results were
  archived to `debug/shared_corpus_eval/INVALID_outlier_gap_bug/` and removed from
  `all_results.jsonl`.
- **The staged contradiction "32k" rung is really ~64k** (prompt audit p50 63,749, max 214,446) —
  a pre-existing consequence of `[[contra-fever-filler-leak]]` that affects the current grid, not
  just this work. All contradiction numbers here use the recalibrated PubMed-only rebuild (n=762).
  The mislabeled baseline OOMed at `max_length 215,086`; its shared partners were cancelled rather
  than kept as an uninterpretable rung.
- **GDN JIT fills `/tmp`** — ~165 MB of leaked nvcc intermediates per job took horton's 916 GB
  partition to 217 MB free and killed a job with `Errno 28`, which reads as a code bug. Launchers
  now set `TMPDIR` to node-local `/data`. Saved as `[[gdn-jit-tmp-disk-fill]]`.
- **rerank needs `ce_scores` realigned, not dropped** — pooled foreign documents take `None`, which
  is the format's own "unscored random fill" case; dropping the array makes the prompt builder
  raise on the deprecated binary rerank format.
