# Data-generator port: `corpus_reasoning/data/` → `ctc.data`

**Status 2026-08-17: the port is COMPLETE — 22 generators build from this repo, and rungs are
open-ended up to 10M+ tokens per example.** The only specs still without a generator are `qa` and
`summarization`, both dropped from the final suite roster (their HELMET v1 data is defective at
the source — trap 15 — and the 22-row olmo-eval roster carries neither). The port landed in three
waves: `8ad00aa77` (the 5 main + 4 held-out corpus generators), `510a18be6` (hotpotqa, absence,
xabsence, strmatch, redundancy), and `2fcc0fb57` (reorder, qdmatch, grouping_labeled — the last
three). `164dcb3a7` then made the rung ladder open-ended: labels past the calibrated 2k–32k table
extrapolate the task's own least-squares fit, `ladders.CEILINGS` refuses rungs a corpus provably
cannot supply, and the O(N²) hot spots that made a 10M-token example take hours (mathmatch and
textgroups placement, the closest-pair audit probe) are O(N)/O(N log N) with byte-identical
output. §2 below is the *original* gap list, kept for the trap notes each row carried; every row
in it except `qa`/`summarization` has since landed.

The paragraph below is the 2026-08-11 snapshot the section numbering was written against.

**Status, verified against the tree on 2026-08-11: 13 ladders build from this repo; 9 of the 18
registered task specs still have no generator.** The port landed in `8ad00aa77` ("data: port the 5
main + 4 held-out corpus generators"). Everything below was re-checked by reading the code and
importing the registry — not from the commit message.

Old tree: `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core`, branch `prasann/landmark`,
frozen at tag `pre-migration-source` (`d0f9b940a`). `OLD:` paths are relative to that tree's
`src/corpus_reasoning/data/` unless a fuller path is given; `NEW:` paths are relative to this repo.

This document is the *port* record: what crossed over, what did not, and the failure modes that
survive the move. The user-facing "how do I build data" page is `NEW: ctc/src/ctc/data/README.md`;
the paper's data appendix is `records/ctc-data-generation.md`. Read §3 and §4 before any rebuild.

---

## 1. What builds today

### 1.1 The 13 ported ladders

`NEW: ctc/src/ctc/data/generators/base.py:39-56`. A generator is keyed by its **ladder** name, not
its task name, so `ctc-data build --task nq` and `ctc-eval --task nq` mean the same thing; three
ladders (`nq`, `fiqa`, `scifact`) are graded by the one `retrieval` spec, `contra_fever` by the
`contradiction` spec, `outlier_review` by `outlier`. The set is exactly `ctc.eval.bundles.BUNDLE`'s
nine graded rows plus the four synthetics.

| ladder | graded by | corpus module | eval-only | ladder shape |
|---|---|---|---|---|
| `cycle` | cycle | — | | nested shrink |
| `groups4` | groups4 | — | | nested shrink |
| `mathmatch` | mathmatch | — | | nested shrink |
| `textgroups` | textgroups | — | | nested shrink |
| `contradiction` | contradiction | `data/sources/pubmed.py` | | nested shrink |
| `nq` | retrieval | `data/sources/nq.py` | | nested shrink |
| `outlier` | outlier | `data/sources/wiki100w.py` | | **own `build_ladder`** |
| `rerank` | rerank | `data/sources/msmarco.py` | | nested shrink |
| `oolong` | oolong | `data/sources/oolong.py` | | **independent per rung** |
| `fiqa` | retrieval | `data/sources/beir.py` | ✓ | nested shrink |
| `scifact` | retrieval | `data/sources/beir.py` | ✓ | nested shrink |
| `outlier_review` | outlier | `data/sources/amazon.py` | ✓ | nested shrink |
| `contra_fever` | contradiction | `data/sources/fever.py` | ✓ | nested shrink |

Two ladder shapes are deviations and both are declared in code rather than left to an audit:

- **`outlier` cannot use the generic shrink.** Dropping random distractors can leave a majority
  topic smaller than the outlier group, giving the example two correct answers and one label. It
  therefore builds every rung of a row at once, fixing the outlier article and growing only the
  majority (`NEW: ctc/src/ctc/tasks/outlier/generate.py:7-19`, wired at
  `NEW: .../outlier/sources/wiki100w.py:198-200` with `shrink_safe=False`).
- **`oolong` cannot nest at all.** Its gold is recomputed over whichever items were drawn, so no
  shrink preserves the answer and its rungs grade different questions
  (`NEW: ctc/src/ctc/tasks/oolong/generate.py:12-18`). Its rung values are **token budgets**, not
  document counts (`ladders.py:57-58`), and its `scaling_param` is `target_tokens`.

Two documented departures from the shipped pre-migration files, worth knowing before diffing
against one: `outlier_review` builds a nested ladder where the shipped four rungs were generated
independently (`NEW: .../outlier/sources/review.py:15-22`), and `rerank`'s rung counts are
BUILD_MATRIX midpoints flagged `estimated, wide` in `CALIBRATION` — re-measure before quoting a
rerank context length (`NEW: ctc/src/ctc/data/ladders.py:53-56,79`).

### 1.2 The shared layer that replaced the duplication

The old tree had **no shared module inside `data/`** — its `__init__.py` was 0 bytes — and the
sharing that existed was 11 generators importing private helpers out of each other. What replaced
each cluster:

| old state | now | evidence |
|---|---|---|
| 5 competing `save_jsonl`, 8 forked `load_jsonl`, 29 inline `f.write(json.dumps(...))`; **no `open()` in the whole directory passed `encoding=`** | one implementation, `ensure_ascii=True` (what 38 old files already produced), explicit `encoding="utf-8"`, mkdir-on-write, not overridable | `NEW: ctc/src/ctc/data/io.py:36,64-68,80-82` |
| ~24 hand-written copies of gold-index arithmetic in three families plus five one-offs, two incompatible index bases | `shuffle_with_remap` / `remap` / `remap_groups` / `check_indices`, with **`base` a required keyword and no default** | `NEW: ctc/src/ctc/data/gold.py:20,25,41-45,75-100` |
| 49 independent constructions of the example dict across 34 files; 3 document shapes; `_meta` vs bare `meta`; hand-typed `source` at every site | one `make_example` with fixed key order, `meta` always normalised to `_meta`, `source` required and declared once per `Generator` | `NEW: ctc/src/ctc/data/schema.py:24,50-98` |
| every generator owned its `main()`; five train/eval splitters with `--eval-frac` 0.1 vs 0.2 and `int(round(x))` vs `int(x)` | one `build_train` / `build_eval`; train/eval separation is a property of the *pool* (`for_split`) | `NEW: ctc/src/ctc/data/build.py:1-21`, `data/sources/__init__.py:19-21` |
| 2 BM25 wrappers + 3 raw `LuceneSearcher.from_prebuilt_index` bypasses; 3 copies of the same cross-encoder class | one `_bm25.py`, one `_scoring.py` | `NEW: ctc/src/ctc/data/_bm25.py`, `_scoring.py` |
| ~40 loader implementations over 20 corpora; heavy imports at module scope, worked around by duplicating classes | 8 `sources/` modules, pool = plain dataclass, `datasets`/`pyserini`/`torch` imported **inside** the loader | `NEW: ctc/src/ctc/data/sources/__init__.py:5-21` |

The proposed `configs/tasks/<task>.yaml` rung table was **not** built that way. The rung→count table
landed as a Python dict, `LADDERS` at `NEW: ctc/src/ctc/data/ladders.py:34-67`, with a companion
`CALIBRATION` at `:71-85` recording per row whether it was measured or estimated. There is no
`configs/` directory in this repo (see §3.4).

### 1.3 Guards that are build-time, not advisory

`build` refuses to write past a failed audit; `--force` overrides and says so
(`NEW: ctc/src/ctc/data/cli.py:8-10`).

- Eval sets below 500 examples raise, with the reason inline (`build.py:372-378`).
- Held-out ladders refuse to produce training data — a refusal, not a warning, because by the time
  a warning is noticed the checkpoint is trained (`Generator.eval_only`, `base.py:96-97`).
- Train examples reusing an eval example's gold are dropped and counted (`build.py:52-55`).
- Train and eval draw from RNG substreams keyed by `(seed, split, rung)`, so resizing training no
  longer silently moves the eval set (`build.py:8-12`).
- Unknown `-C` keys are a `TypeError`, not a shrug — a typo would otherwise build at the default
  size and label it as what was asked for (`base.py:128-133`).
- Shortcut probes run on every build: `gold_position_bias`, `gold_length_bias`,
  `cycle_frequency_gap`, `closest_pair_is_gold` (`NEW: ctc/src/ctc/data/audit.py:132,158,186,219`),
  plus `check_split_separation`, `check_rung_sizes`, `check_ladder_nesting` (`:299,339,358`).

### 1.4 What validation actually proved

- **Byte-parity, four tasks.** `cycle`, `groups4`, `mathmatch`, `textgroups` have golden fixtures
  captured from the pre-migration tree (`NEW: ctc/tests/data/fixtures/*_golden.json`, captured by
  `fixtures/_capture_from_pre_migration.py`). This is the only check that can catch a lost
  anti-shortcut fix, because such a rewrite produces *valid* data that is trivially solvable
  (`NEW: ctc/tests/data/test_synthetic_parity.py:1-11`).
- **Structural and invariant-based, everything else.** The corpus-backed generators cannot run
  without their corpora, so there is nothing to diff against; they are exercised against fixture
  pools from `ctc/tests/fixtures/pools.py`. Do not describe these as byte-parity-verified.
- `pytest ctc/tests/data ctc/tests/tasks` → **556 passed, 21 skipped** (2026-08-11, no GPU, no
  network).

---

## 2. What is still un-ported

### 2.1 Nine registered specs with no generator

Confirmed by importing the registry: 18 specs registered
(`NEW: ctc/src/ctc/tasks/__init__.py:37-56`), 9 covered by a generator, 9 not.

| spec | old generator to port | why it is not trivial |
|---|---|---|
| `qa` | **not** `generate_helmet_qa_data.py` — use `OLD: debug/ctc_helmet_v2/build_helmet_v2_data.py` | v1 data is defective at the source; trap 15 |
| `summarization` | same v2 builder, not `generate_helmet_summ_data.py` | trap 15; also needs `ctc[rouge]` or the scorer errors by design |
| `grouping_labeled` | `generate_arxiv_grouping_data.py` (678) | pinned OpenAlex snapshot; carry the trap-14 capacity-aware fix at `:282-343` |
| `reorder` | `generate_reorder_data.py` (328) | writes `gold_order` (1-based) and `source_type`, not `gold_doc_indices`/`source`; streaming HF is not reproducible, trap 19 |
| `absence` | `generate_absence_data.py --gutenberg` only (416) | four `source` tags in one file; `--official` is a dropped mode |
| `xabsence` | `generate_xabsence_data.py` (288) | two-phase; `--build-pool` needs an LLM |
| `strmatch` | `generate_strmatch_data.py` (206) | `--vocab-source file` defaults to `/usr/share/dict/words`, trap 18 |
| `qdmatch` | `generate_qdmatch_data.py` (215) | **not a corpus generator** — a transform of built retrieval JSONL (`:137-138`), so it needs a different `build_example` signature; gold is `gold_pairs`, 1-based and **ordered** |
| `redundancy` | `generate_redundancy_data.py` (240) | dropped from the active build; LLM-serving-bound. Port last or not at all |

`hotpotqa` is the other real gap: it is a live source for `retrieval`/`qa`/`qdmatch`
(`generate_hotpotqa_data.py`, 483) and `NEW: ctc/src/ctc/tasks/retrieval/spec.py:130` still lists it
in `sources`, but no `data/sources/hotpotqa.py` exists and no ladder builds it.

### 2.2 Deliberately not ported

Dropped from the suite on 2026-08-03 (`OLD: paperdraft/figures/fig4_cross_task.py:29`): plain
`grouping` (unlabeled), outlier-Amazon *as a suite row* — note the difficulty-matched
`outlier_review` **is** ported as a held-out probe — absence-official, and `redundancy`. Also not
ported and not wanted: qdmatch-ObliQ (§3.3), `ruler`, `matching_ngram`, `musique`,
`generate_wiki_contradiction_data.py` (zero refs; its absence is documented at
`NEW: ctc/src/ctc/tasks/contradiction/generate.py:50-52`), `generate_pubmed_multiclaim.py`
(separate track, trap 17), and the ~23 one-off/dead files in §5.1.

Note there is **no separate grouping generator**: `grouping` and `grouping_labeled` come from the
same `generate_arxiv_grouping_data.py` output split by a convert-time `--task` flag
(`OLD: BUILD_MATRIX.md:338,342`), so dropping `grouping` costs no generator code. The format layer
deliberately retains serializers for `grouping`, `cot_retrieval`, `ruler` and `matching_ngram`
(`NEW: ctc/src/ctc/format/documents.py:54,153-163`) with no registered task, because the golden
fixture pins their instructions. That is correct; do not "clean them up."

### 2.3 Corpora that would have to come with them

Ported: PubMed, FEVER, NQ, BEIR, MS MARCO, wiki100w, Amazon, OOLONG (8 modules under
`NEW: ctc/src/ctc/data/sources/`). Not ported, and each is one module of work before its task can
build: HotpotQA (`hotpotqa/hotpot_qa`, distractor), Gutenberg (`sedthh/gutenberg_english`),
OpenAlex (bulk `.gz` snapshot or REST), the four HELMET corpora (`deepmind/narrativeqa`,
`xinrongzhang2022/InfiniteBench`, `ccdv/govreport-summarization`, `allenai/multi_lexsum`),
AbsenceBench (`harveyfin/AbsenceBench`), and OBLIQ-Bench (`dianetc/OBLIQ-Bench`, raw `requests`).

---

## 3. Defect list

The 2026-08-10 plan opened with three defects in this repo. Two are fixed; the third is open. Three
more, left behind by the port itself, are new.

### 3.1 ✅ FIXED — the contaminated contradiction ladder

`NEW: ctc/src/ctc/tasks/contradiction/spec.py:55` and `NEW: ctc/src/ctc/data/ladders.py:50` both
carry `{"2k": 44, "4k": 92, "8k": 187, "16k": 379, "32k": 762}`, with the derivation recorded in
comments at both sites and a test pinning the spec copy
(`NEW: ctc/tests/tasks/test_task_packages.py:440-442`). The missing-`32k` KeyError is gone.

The retired ladder was `77/167/346/705/1423` (`OLD: build_v2_eval_ladders.py:66`), fit against a
filler pool that was 92–99.6 % FEVER/wiki_mix rather than PubMed. Wikipedia trivia claims tokenize
at ~22.8 tok/doc against real PubMed claim sentences' ~43, so re-running those counts against the
corrected pubmed-only pool overshoots every label by ~1.8× — measured: `n=77` → 3413 tokens, not
2048; `n=1423` → 61461, not 32768 (`OLD: build_v2_eval_ladders.py:87-93`). The corrected row is
`contra_ctc` at `OLD: build_v2_eval_ladders.py:96`, measured at 1925 / 3933 / 8052 / 16074 / 32397
median tokens with 0.00 % FEVER at every rung.

⚠ **It is stored twice.** `LADDERS["contradiction"]` is what the build path reads
(`build.py:399,416,440,469,556`, via `ladders.docs_for_rung`); `spec.CLAIMS_PER_RUNG` has **no build
consumer** and is pinned only by the test above. They agree today. Nothing enforces that they keep
agreeing, and the test guards the copy the builder ignores.

### 3.2 ✅ FIXED — the stale FEVER-leak warning

`NEW: ctc/src/ctc/tasks/contradiction/generate.py:14-35` no longer says the leak "is still open for
the CTC-suite ladder." It now states that the leak is closed for both the xlong and CTC-suite
ladders (rebuilt and re-verified at 0.00 % on 2026-08-04/05) and, more usefully, that it is closed
**structurally**: a pool holds one corpus, and an example's fillers can only come from the pool its
gold came from. The false-negative control is one level deeper — fillers cannot come from an
abstract that contributed gold to *this* example, since that abstract may restate the fact its
contradiction denies (`NEW: .../contradiction/sources/pubmed.py:53-58`).

Two headline reversals from that rebuild are carried in the same docstring and are why this matters
beyond hygiene: the contamination **depressed** scores rather than flattering them (32k went 0.335 →
0.559, roughly 10 SE at eval_size 500), and the published "dense collapses at 32k" was therefore a
ladder artifact — the dense-vs-chunked absolute gap **narrows** with context (0.441 at 2k → 0.369 at
32k) rather than widening. Any contradiction number in
`OLD: results/ctc_suite/dense_vs_chunked_table.md` predates the rebuild.

### 3.3 🟡 STILL OPEN — `qdmatch` and `retrieval` have their ObliQ assignment inverted

`NEW: ctc/src/ctc/tasks/qdmatch/spec.py:113` — `sources=("nq", "hotpotqa", "obliq")`
`NEW: ctc/src/ctc/tasks/retrieval/spec.py:130` — `sources=(…, "beir_scifact", "beir_fiqa", "msmarco")`, no `obliq`

`OLD: src/scripts/data/ctc_suite/BUILD_MATRIX.md:426-430` (roster change, 2026-07-19) says the
opposite of both: qdmatch-ObliQ is **dropped**, and ObliQ instead enters as a standalone
in-context-retrieval row via `generate_obliq_data.py`, not `generate_qdmatch_data.py` — "do not
tokenize/train the `qdmatch_*obliq*` pilot jsonl."

The stakes have **dropped** since this was written: `TaskSpec.sources`
(`NEW: ctc/src/ctc/format/registry.py:114`) has **zero consumers** anywhere in `ctc/` — build
dispatch goes through `GENERATORS`, keyed by ladder name — and neither `qdmatch` nor any ObliQ
ladder has a generator. So it can no longer drive a wrong build. It is now a wrong statement in the
only in-repo record of which corpora a task can be built from, and it will be read as authority by
whoever ports `qdmatch`. Fix it then, at the latest.

### 3.4 🟡 NEW — three cross-references the port left pointing at nothing

None affects a build; all three mislead a reader and two break Sphinx.

1. `NEW: ctc/src/ctc/format/rungs.py:6` still says the rung→count mapping "lives in
   `configs/tasks/<task>.yaml`". **There is no `configs/` directory in this repo.** It lives in
   `NEW: ctc/src/ctc/data/ladders.py:34`.
2. `NEW: ctc/src/ctc/data/ladders.py:18-19` says contradiction's corrected row "currently lives in
   its spec's `extra["claims_per_rung"]` and should move here when that generator is ported." The
   generator is ported and the row *is* here, at `:50`. See §3.1 on the surviving duplicate.
3. `ctc.data.audit.shortcuts` does not exist — the probe functions are top-level in
   `NEW: ctc/src/ctc/data/audit.py`. Three docstrings reference the non-existent submodule:
   `NEW: ctc/src/ctc/tasks/cycle/generate.py:25`, `.../groups4/generate.py:21`,
   `.../mathmatch/generate.py:17`.

Also still outstanding from the plan: `NEW: ctc/src/ctc/tasks/cot_retrieval/` is still an empty
directory holding only a stale `__pycache__`. `cot_retrieval` is a dropped task; the *format*-layer
retention (§2.2) is separate and should stay.

---

## 4. Trap index

The most valuable part of this record. Status re-verified against the current tree where the trap
lives in ported code, and against the frozen old tree where it does not.

| # | trap | lives in | status | rebuild data? |
|---|---|---|---|---|
| 1 | Qwen3 marker embeddings: cosine 1.0000, then wrong norm | `OLD: src/scripts/data/fix_marker_embeddings.py` → **PORTED 2026-08-12** as `NEW: src/scripts/ctc/fix_marker_embeddings.py`, pinned by `ctc/tests/train/test_fix_marker_embeddings.py` — which tests the **norm** half, not just the cosine | FIXED both sides (seeds from real delimiter rows, asserts in-dist norm; `--check-only` exits non-zero so it gates a launcher under `set -e`) | No — checkpoint fix. Re-repair any base from before 2026-07-14 |
| 2 | oolong `--item-regex '\|\|'` matches every line | `OLD: src/scripts/data/convert_unified_to_document_landmark.py:365-382` — **no equivalent here**; conversion in this repo is `NEW: src/scripts/ctc/convert_to_shards.py` | FIXED old-side (startup guard rejects any regex matching `""`) | **Yes** — shards built before 2026-07-27 |
| 3 | oolong preamble train/eval layout mismatch | old converter's oolong prompt construction vs `OLD: src/olmo_core/data/document_chunk_landmark.py:251-272` | see `records/oolong-preamble-trap-investigation.md` for the current reading | Yes, if it is real |
| 4 | FEVER/wiki fillers in PubMed contradiction evals | old `harvest_fillers` glob | **FIXED, and now structurally impossible** — §3.2 | Contaminated rung files exist old-side; training data was always clean |
| 5 | contra 1-based gold read as 0-based; outlier shrink breaks scale-K | `OLD: build_v2_eval_ladders.py:54,205-210,451` | FIXED and carried: `base` is a required argument (`gold.py:25`) and outlier owns its ladder (§1.1) | Rebuilt already |
| 6 | docchunk id/title label leaked as FREE token | `OLD: src/olmo_core/data/document_chunk_landmark.py:317-349` | FIXED old-side; chunk-leak audit 21/22 tasks clean | Yes, shards from before ~2026-07-06 |
| 7 | NQ 98 %-hard pipeline | `OLD: generate_nq_training_data.py:278,307` — `--hard-neg-frac` default **1.0**, `--ce-filter` **off** | **CLOSED in the port by inversion**: `hard_frac=0.1` and `ce_filter=True` are the defaults (`NEW: .../retrieval/sources/nq.py:35,57,66`), with the reason at `:8-12` | Banned old files (`hn49/hn99/hn199/ladder64k`) exist and must never be used |
| 8 | `parse_doc_ids` required both brackets | `OLD: lib/metrics.py:98-106` fixed vs `OLD: src/scripts/ctc_eval/lib/metrics.py:98-100` still buggy | FIXED here: one implementation in `NEW: ctc/src/ctc/format/parsing.py`, pinned by `ctc/tests/format/test_golden_parity.py:94-101` | No — grading bug |
| 9 | per-task gold index base | §5.2 | standing convention; **re-verified 2026-08-11**, all 18 specs agree with the old generators | N/A |
| 10 | eval doc-id digit-range mismatch (train n-max 697 vs eval 1423) | a build discipline, no single file | OPEN by nature; detector at `OLD: debug/length_mix_scaling/check_digit_truncation.py` | Per-build check |
| 11 | goldgrad `--max-length` truncation → empty gen → f1 0.000 | eval side | FIXED and generalised (auto-sizes from the measured prompt distribution, hard-fails otherwise) | No |
| 12 | eval-bundle weka staging: S3 push alone does nothing; a job that skips every rung still exits 0 | process | discipline, two-step gantry sync | N/A |

13. **`cycle` and `groups4` had exploitable frequency shortcuts — fixed, and the fix is fragile.**
    Cycle entities were once drawn from a pool disjoint from background edges, pinning every cycle
    entity's claim frequency at exactly 2 while background frequency grew with N: gold was "the
    three rarest names", and the shortcut got *stronger* as N grew. Groups4 distractors were kept so
    far apart that any close pair was gold. Both fixes are carried over with the reasoning inline
    (`NEW: ctc/src/ctc/tasks/cycle/generate.py:17-27`, `.../groups4/generate.py:11-21`) and both are
    guarded on every build by `cycle_frequency_gap` / `closest_pair_is_gold` (`audit.py:186,219`).
    **A rewrite that reintroduces a separate distractor pool silently restores the shortcut** —
    which is exactly why these four tasks have byte-level golden fixtures (§1.4) rather than a clean
    reimplementation.
14. **`generate_arxiv_grouping_data.py` level/k-density confound — fixed old-side, not yet ported.**
    `sample_k_for_level` picked k as a fraction of `n_docs`, ignoring that OpenAlex L0 has ~19
    top-level fields, so coarse levels silently dropped out as N grew (L0 share 57 % → 0 %) and
    "more docs" got conflated with "finer grouping." The 3-part capacity-aware fix is at
    `OLD: generate_arxiv_grouping_data.py:282-343`. Carry it when `grouping_labeled` is ported.
15. **HELMET v1 data is defective at the source, not just mis-scored.** `generate_helmet_qa_data.py`
    puts raw scrape text (IMSDb HTML chrome, Gutenberg license headers) into context with no length
    filter, so at the 2k rung **only 9.5 % of examples contain the gold answer anywhere in the
    retained context** — and the same defect is baked into the training set the same generator
    builds. The GovReport 16k and 32k rungs are literally the same documents (median context 8765
    both). Port `OLD: debug/ctc_helmet_v2/build_helmet_v2_data.py`, not the v1 generators.
16. **`build_contra_recombined.py` — every validation passed and the task still broke.** Pooling
    gold pairs against a *globally* sampled filler pool instead of each example's co-sampled
    distractors dropped full-attention train f1 from 0.934 to 0.585, because the nearest non-gold
    pair's similarity crashed 0.372 → 0.163 — "pick the two most similar claims" then solves the
    task. Contamination checks, pair-reuse checks and gold-pair-distance matching all passed; none
    asked whether it was still the same task. Scale contradiction by expanding from train, never by
    recombination. The ported filler control (§3.2) is the structural version of this lesson.
17. **`generate_pubmed_multiclaim.py`'s lexical leak is n-dependent and the fix is deferred.**
    A string-overlap baseline runs F1 0.514 at n=21 down to 0.036 at n=397 — strongest at exactly
    the short rungs the ladder varies. Cause: contradictions hold subject/outcome/population/
    timepoint fixed while distractors come from unrelated abstracts, and siblings are capped at
    ~9/example so their share collapses as n grows. The paraphrase-distractor fix is designed and
    **explicitly deferred**; do not port this track as "already fixed."
18. **`--vocab-source file` in `generate_strmatch_data.py:177` defaults to `/usr/share/dict/words`**
    — machine-dependent content. Pin `--vocab-source wiki` or ship a wordlist when `strmatch` is
    ported, or its fixture is not reproducible.
19. **Streaming HF datasets are not seed-reproducible.** `generate_reorder_data.py:192` uses
    `load_dataset(..., streaming=True)`; book order depends on stored shard order and the `datasets`
    version. Its `--local-dir` path (`:208`, `sorted(glob)`) *is* deterministic. Any `reorder`
    fixture must use `--local-dir`.
20. **`generate_n2ified_data.py` is dead code whose output is still live.** `main()` raises at
    `:135`, but `data/n2ified_eval_{nq,hpqa}_q20.jsonl` are read by 5 old eval scripts. Declining to
    port the generator is right; do not delete the JSONL.
21. **`OLD: build_xlong_rungs.py:59` is not importable as a module** — a bare
    `import build_v2_eval_ladders as v2`, with a `:58` comment asserting the opposite. It runs only
    from cwd `data/`; `python -m corpus_reasoning.data.build_xlong_rungs` raises
    `ModuleNotFoundError`. Relevant only if the xlong ladders are rebuilt from the old tree.
22. **Two generators key gold recovery on `id()` object identity** —
    `generate_n2ified_data.py:85-93` and `generate_review_outlier_data.py:133-136`. Correct today
    only because `rng.sample` returns distinct objects; any `dict(d)`, `deepcopy`, or JSON
    round-trip silently produces wrong gold with no error. Both are on the do-not-port list. The
    ported `outlier_review` does not use the pattern, and nothing else should.
23. **No `open()` in the old `data/` directory passes `encoding=`.** Every read and write rode on
    the ambient locale, so "byte-identical" was machine-dependent for any non-ASCII corpus (PubMed,
    Gutenberg, OpenAlex all carry non-ASCII). **Closed for ported code** — `io.py` sets
    `encoding="utf-8"` explicitly on every path (`:36,66,81`). Still true of anything run out of the
    old tree, including a fixture captured there.

**Highest-risk trap for the next rebuild: #13**, because it is the only one that produces data
passing every structural check while being trivially solvable, and the only defence is a fixture a
rewrite is tempted to regenerate. #15 is second: not a grading bug but bad data, and baked into a
training set as well as an eval ladder.

---

## 5. Reference material carried forward

### 5.1 Old-tree inventory: what is worth porting and what is not

81 `.py` files, 21,446 lines in `OLD: src/corpus_reasoning/data/` (counted at the tag), in two
layers: 45 `generate_*.py` (15,147 lines, the port target) and 35 pipeline/audit files (6,299
lines), plus a 0-byte `__init__.py`. L/B =
load-bearing, inferred from repo-wide greps for the filename and for the emitted `source` string.
**→ ported** marks what `8ad00aa77` covers.

| file | LoC | emits `source` | verdict |
|---|---|---|---|
| `generate_pubmed_contradiction_data.py` | 747 | `pubmed_perturbation` | L/B, BUILD_MATRIX row 16 → **ported** (`contradiction`) |
| `generate_fever_contradiction_data.py` | 469 | `fever` | L/B, the OOD probe → **ported** (`contra_fever`) |
| `generate_nq_training_data.py` | 385 | `nq` | L/B row 1 → **ported** (`nq`); trap 7 |
| `generate_wiki_outlier_data.py` | 328 | `wiki_outlier_topic` | L/B row 11 → **ported** (`outlier`) |
| `generate_review_outlier_data.py` | 383 | `review_outlier_{rating,category}` | dropped as a suite row → **ported as the held-out `outlier_review`** |
| `generate_msmarco_trainhn_data.py` | 450 | `msmarco_trainhn` | L/B rows 9+10, one build feeds retrieval and rerank → **ported** (`rerank` + the msmarco source) |
| `generate_beir_ce_data.py` | 203 | `beir_<ds>_ce` | L/B rows 7+8 → **ported** (`scifact`, `fiqa`) |
| `generate_oolong_ladder_data.py` | 428 | `oolong_<group>` | L/B row 4 → **ported** (`oolong`) |
| `generate_cycle_data.py` | 230 | `cycle` | L/B row 25 → **ported**; trap 13 |
| `generate_groups4_data.py` | 223 | `groups4` | L/B row 26 → **ported**; trap 13 |
| `generate_mathmatch_data.py` | 214 | `synth_mathmatch` | L/B row 23 → **ported** |
| `generate_textgroups_data.py` | 448 | `textgroups` | L/B row 15 → **ported** |
| `generate_hotpotqa_data.py` | 483 | `hotpotqa` | **L/B row 2 — not ported**, the largest remaining retrieval gap |
| `generate_arxiv_grouping_data.py` | 678 | `openalex_grouping_{train,eval}` | **L/B rows 13+14 — not ported**; trap 14 |
| `generate_absence_data.py` | 416 | `absence_{pubmed,gutenberg,numerical,official_*}` | **L/B row 18 (`--gutenberg`) — not ported**; `--official` dropped |
| `generate_xabsence_data.py` | 288 | `xabsence_<tag>` | **L/B row 22 — not ported**, two-phase LLM |
| `generate_strmatch_data.py` | 206 | `strmatch` | **L/B row 20 — not ported**; trap 18 |
| `generate_reorder_data.py` | 328 | `source_type: gutenberg` | **L/B row 24 — not ported**; trap 19 |
| `generate_qdmatch_data.py` | 215 | `qdmatch_<tag>` | **L/B rows 21a/21b — not ported**, a transform not a generator |
| `generate_obliq_data.py` | 400 | `obliq_<subset>` | **L/B row 21c — not ported**; §3.3 |
| `generate_niah_contradiction_data.py` | 139 | `niah_contradiction` | L/B row 3 — not ported |
| `generate_helmet_qa_data.py` | 114 | `helmet_<src>` | L/B row 5 — **do not port**, v1 defective (trap 15) |
| `generate_helmet_summ_data.py` | 119 | `helmet_summ_<src>` | L/B row 6 — same |
| `generate_oolong_data.py` | 101 | `oolong_<group>` | the superseded fixed-bucket v1 path |
| `generate_cot.py` | 174 | adds `chain_of_thought` | L/B — CoT enrichment pass, not ported |
| `generate_ruler_data.py` | 478 | `ruler_<subtask>` | outside the 26-row roster; live old eval path, no task package |
| `generate_redundancy_data.py` | 240 | `pubmed_redundancy` | dropped from the build, LLM-bound |
| `generate_pubmed_multiclaim.py` | 1505 | `pubmed_multiclaim` | separate validated track, not in BUILD_MATRIX; trap 17 |
| `generate_obliq_synthetic_data.py` | 591 | `obliq_synth_<subset>` | pilot, validated but never promoted |
| `generate_wiki_contradiction_data.py` | 546 | `wiki_contradiction` | one-off, zero refs |
| `generate_beir_data.py` | 98 | `beir_<ds>` | superseded by `_ce`; still exports the shared `load_beir()` |
| `generate_msmarco_data.py` | 143 | `msmarco` | `main()` raises (`:38`); survives to export `_passage_text` |
| `generate_msmarco_trecdl_data.py` | 177 | `msmarco_trecdl<yr>` | deprecated behind `--force` (`:129`) |
| `generate_n2ified_data.py` | 182 | `n2ified_<tag>` | `main()` raises (`:135`); **output still live**, trap 20 |
| `generate_matching_ngram_data.py` | 147 | `matching_ngram_wiki100w` | superseded by strmatch |
| `generate_hotpotqa_unified_corpus.py` / `generate_musique_unified_corpus.py` | 531 / 426 | | one-off prototypes |
| `generate_wiki_reorder_data.py` / `generate_xabsence_abstracts_data.py` / `generate_retrieval_triplets.py` / `generate_msmarco_helmet_rerank_data.py` / `generate_contradiction_data.py` / `generate_arithmetic_data.py` | 145 / 288 / 273 / 176 / 126 / 82 | | one-offs, zero refs |
| `generate_configs.py` / `generate_examples.py` | 563 / 261 | | dead — predate olmo-core |

Of the 35 pipeline files, 12 were load-bearing: `build_v2_eval_ladders.py` (504, the ladder engine
and the per-task index-base table), `build_xlong_rungs.py` (391, 64k–2M, runtime token calibration),
`build_v2_outlier_ladder.py` (200, the scale-K fix — its logic is now `outlier/generate.py`'s
`nested_ladder`), `build_obliq_token_ladder.py` (126), `build_shared_corpus_evals.py` (654 — ported
counterpart `NEW: ctc/src/ctc/data/shared_corpus.py`), `build_combined_unified.py` (114),
`tokenize_unified_for_olmo.py` (316 — counterpart `NEW: src/scripts/ctc/convert_to_shards.py`),
`verify_v2_eval_ladders.py` (167) and `verify_shared_corpus.py` (161), `subsample_beir_ladder.py`
(83), `expand_obliq_train.py` (117), `mix_obliq_subsets.py` (80). The remaining 23 are conditionally
live (`align_hn_doc_lengths.py`, `convert_unified_to_sft.py`, `dump_suite_examples_full.py`) or
one-off/dead — including `build_contra_recombined.py` (self-flagged "DO NOT USE THIS AS-IS",
trap 16). Do not port them.

### 5.2 Gold index base per task — re-verified 2026-08-11

The old `data_format.py` docstring said "0-indexed positions." **That is wrong for 9 of the 18
tasks.** Printed from the live registry and compared against the old generators:

| base 1 | base 0 |
|---|---|
| `contradiction`, `redundancy`, `strmatch`, `mathmatch`, `cycle`, `groups4`, `textgroups`, `qdmatch` (`gold_pairs`), `reorder` (`gold_order`) | `absence`, `xabsence`, `grouping_labeled`, `outlier`, `retrieval`, `qa`, `rerank`, `summarization`, `oolong` |

Proving lines old-side: `generate_pubmed_contradiction_data.py:412-419`,
`generate_redundancy_data.py:146-148`, `generate_strmatch_data.py:131-134`,
`generate_mathmatch_data.py:148`, `generate_cycle_data.py:180-184`,
`generate_groups4_data.py:168-171`, `generate_textgroups_data.py:359-363`,
`generate_qdmatch_data.py:115-123`, `generate_reorder_data.py:169-171`;
`generate_absence_data.py:254,265`, `generate_xabsence_data.py:161`,
`generate_arxiv_grouping_data.py:492-501`, `generate_wiki_outlier_data.py:48-56`,
`generate_nq_training_data.py:45`, `generate_helmet_qa_data.py:101`,
`generate_msmarco_trainhn_data.py:167`, `generate_helmet_summ_data.py:104`,
`generate_oolong_data.py:78`. **No disagreement found**, then or now. Keep this table as the
regression reference; `gold.check_indices` is what it is checked with.

Three caveats that are not disagreements but will bite:

- `oolong` declares base 0 over a field that is always `[]`. Do not let an audit "verify" it and
  report a pass that means nothing.
- `generate_ruler_data.py:262-272` and `generate_n2ified_data.py:85-94` are 0-based while their
  nearest neighbours in the pair/group family are 1-based. Neither has a task package.
- `OLD: dump_suite_examples.py:32-33`'s `ONE_INDEXED` set omits `textgroups`, `qdmatch` and
  `reorder`, which really are 1-based. It never misfires because it never loads those tasks — but
  it is not the authority. The table above is.

### 5.3 The unified schema: what was normalised, and what a diff will show

The declared contract (`OLD: lib/data_format.py:6-32`) was never enforced: no validator, no
dataclass, no JSON schema. Six inconsistencies had accumulated across 49 construction sites in 34
files. What `NEW: ctc/src/ctc/data/schema.py` settled — **each of these makes new data
non-byte-comparable with old data, deliberately**:

1. **Document shape.** Three variants old-side (bare `{"text":…}` in 19 files, `{"title": None,…}`
   in 13, `{"title": "",…}` in 3), and 11 files read `doc.get("title")`, so absent vs `None` vs `""`
   was a live distinction. Now one `make_document` (`schema.py:35`).
2. **`_meta` vs bare `meta`** (14 files vs 4, no rule). Now always `_meta` (`schema.py:94-95`).
3. **`source` is required and non-empty**, declared once on the `Generator` rather than hand-typed
   at each of 28 distinct literal sites. Not an enum — a required string.
4. **Key order is fixed**, so a diff between two builds shows only real differences.

Not normalised, and still true of the data: `answers` is overloaded four ways (real answers; a
`[""]` placeholder for obliq/beir/msmarco; `[]` for the whole contradiction family; a *rendered
gold-position string* for outlier), so a validator that only checks the key is present proves
nothing. `reorder` still writes `source_type` rather than `source`, and every `source`-keyed
consumer misses it — settle that when `reorder` is ported.

Undocumented keys that remain load-bearing: `ce_scores` (`list[float|None]` parallel to
`documents`; a `None` is read as gain 0 and **drops that document from the Kendall-tau set**, which
is why msmarco scores its random fill at load time —
`NEW: ctc/src/ctc/tasks/rerank/generate.py:8-15`), `hard_neg_indices` (0-based), `cluster_labels`
(not scored), `gold_order` (1-based permutation; `reorder` has no `gold_doc_indices` at all),
`gold_pairs` (1-based and **ordered**), `_task` / `_cot_mode` (per-row dispatch in a multitask
file), `chain_of_thought`, and the shared-corpus trio `corpus_id` / `shared_prefix_len` /
`shared_prefix_sha1`.

### 5.4 Old-tree hazards that only bite the remaining port

- **Four mutually inconsistent answers to "how long is this example?"** survive old-side: a real
  tokenizer at 14 sites with **four different tokenizer ids** (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`,
  `Qwen/Qwen3-4B-Base`, `Qwen/Qwen2.5-0.5B`); `chars // 4` in 5 copies; a fitted linear model per
  task in `build_v2_eval_ladders.py`; and a word-count proxy in `build_obliq_token_ladder.py`. §3.1
  is what happens when two of them disagree. Rung labels are only meaningful once one is chosen —
  `ladders.CALIBRATION` is where that choice is now recorded, per row.
- **Seeding was already uniform, which is why parity fixtures are feasible at all.** 59 files expose
  `--seed`, 63 instantiate `random.Random(...)`, and there are **zero** occurrences of
  `random.seed`, `np.random`, `default_rng`, or bare global `random.shuffle/sample/choice/randint`.
  Four patterns must be preserved verbatim in any remaining port, because each changes the stream:
  `generate_ruler_data.py:464-468` (`seed` train / `seed + 1` eval); `generate_matching_ngram_data.py:78`
  (re-seeds **per example** from a running cursor); the derived-substream constants
  `random.Random(SEED * 1_000_003 + ei)` and `random.Random(SEED * 7 + ei * 101 + n)`, reinvented in
  four and two files respectively; and the four ladder builders' hardcoded `SEED = 1234` with no
  `--seed` flag.
- **7 dead argparse flags** (verified not `dest=` aliases), do not port:
  `generate_wiki_outlier_data.py:212 --eval-frac`, `generate_textgroups_data.py:407 --separation`,
  `generate_retrieval_triplets.py:230 --num-docs`, `generate_absence_data.py:390 --official-split`,
  `generate_obliq_synthetic_data.py:505 --bm25-threads` and `:510 --index-dir`,
  `generate_oolong_data.py:55 --config`.
- **`word_jaccard` has two semantically divergent copies**: `generate_xabsence_abstracts_data.py:81-89`
  returns `1.0` on empty input, `generate_obliq_synthetic_data.py:165-171` returns `0.0` — opposite
  behaviour in an overlap-*reject* filter. Pick one deliberately when `xabsence` is ported.

---

## 6. Not verified

- **No ported generator has been run against its real corpus in this repo.** Every corpus-backed
  claim above is a claim about *code*, checked by reading it and by 556 tests over fixture pools.
  The first real build is still the first real build; run `ctc-data audit` on its output.
- **Trap 3 (oolong preamble) is not localised here.** See
  `records/oolong-preamble-trap-investigation.md`; no reading is trustworthy without re-running the
  chunk-leak validator against current shards.
- **Which contradiction rung files are physically staged where** was not checked. §3.1 rests on the
  code comments and the old-side verification table, both unambiguous; the JSONL was not read.
- **"Load-bearing" in §5.1 is inferred from repo-wide greps** of the frozen old tree (sbatch,
  launchers, `BUILD_MATRIX.md`, `records/`). A generator run only by hand and never committed to a
  launcher would read as one-off.
