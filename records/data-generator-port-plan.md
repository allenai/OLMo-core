# Data-generator port plan: `corpus_reasoning/data/` → `ctc/src/ctc/tasks/*/generate.py`

Status: analysis only, 2026-08-10. Nothing ported yet. Old tree read at
`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core` (branch `prasann/landmark`, tag
`pre-migration-source`); new tree is this repo (`prasann/ctc`).

All `OLD:` paths below are relative to the old repo's `src/corpus_reasoning/data/` unless a fuller
path is given. All `NEW:` paths are relative to this repo.

---

## 0. Read this first — three defects in THIS repo, found while writing this plan

(Three more were found in the old tree: an unimportable sibling import in §3, and traps 21–23 in
§5. Those cost nothing until someone reruns the old code; these three cost the first rebuild.)

### 0.1 🔴 `contradiction/spec.py` carries the CONTAMINATED rung ladder

`NEW: ctc/src/ctc/tasks/contradiction/spec.py:43`

```python
CLAIMS_PER_RUNG = {"2k": 77, "4k": 167, "8k": 346, "16k": 705}
```

Those five numbers are the `contra` entry of `OLD: build_v2_eval_ladders.py:66`. That ladder was
fit against a filler pool that was **92–99.6 % FEVER/wiki_mix**, not PubMed. The old repo's own
code says so, at `OLD: build_v2_eval_ladders.py:87-93` (the `contra_ctc` entry that replaced it):

> the "contra" rungs above (77/167/346/705/1423) were fit against a filler pool that was 92-99.6%
> FEVER/wiki_mix […] whose one-line Wikipedia trivia claims tokenize at ~22.8 tok/doc. Real PubMed
> claim sentences are ~43 tok/doc, so re-running the SAME n against the fixed pubmed-only glob
> overshoots every label by ~1.8x (measured: n=77 -> 3413 tok, not 2048; n=1423 -> 61461, not
> 32768).

The corrected, verified ladder is `contra_ctc` at `OLD: build_v2_eval_ladders.py:96`:
`{"2k": 44, "4k": 92, "8k": 187, "16k": 379, "32k": 762}`, measured at 1925 / 3933 / 8052 / 16074 /
32397 median tokens with 0.00 % FEVER at every rung (`OLD: records/paper-v2-todo-status.md:67-77`).

Why this matters more than a stale constant: `NEW: ctc/src/ctc/tasks/contradiction/generate.py:34`
declares that the not-yet-written generator will pick its claim count *from this field* —
`":param rung: Rung label; selects the claim count from ``SPEC.extra["claims_per_rung"]``"`. So the
first contradiction rebuild in the new repo would bake the ~1.8× overshoot into every rung, on the
suite's flagship O(N²) task, and the prompts would silently be ~1.8× longer than their labels. The
spec's own module docstring already reports the symptom ("measured prompts at the 4k rung spanned
3,457-23,796 tokens") without connecting it to the cause.

Also: `CLAIMS_PER_RUNG` has four entries while `SPEC.rungs` declares five (`"32k"` is missing), so a
32k build would `KeyError`.

**Fix before porting:** `CLAIMS_PER_RUNG = {"2k": 44, "4k": 92, "8k": 187, "16k": 379, "32k": 762}`.

### 0.2 🟡 The FEVER-leak warning in `contradiction/generate.py` is stale

`NEW: ctc/src/ctc/tasks/contradiction/generate.py:14-18` says the leak "is still open for the
CTC-suite ladder." That was true as of `OLD: records/contradiction-data-and-base-hygiene.md`
(2026-07-29). `OLD: records/paper-v2-todo-status.md` (updated 2026-08-10) supersedes it: the clean
`contra_ctc` ladder was built, verified at 0.00 % FEVER on all five rungs, and **both arms were
fully re-measured** (`:148-158`, dense 0.843→0.559, chunked-mix 0.402→0.191 across 2k→32k). The
clean rungs are staged at `cubbins:/data/prasann/ctc_suite_staged/eval_rungs/contradiction_clean/`.

Two things a porter still needs to know, and the docstring should say instead:
- The clean ladder is **not wired as anyone's default** — unlike `_eval_bundle_eval500_v2_clean`,
  which is the default in `OLD: src/scripts/train/memexpress/singletask_ladder/run_beaker_multirung_eval.sh:65`,
  the CTC clean rungs must be passed explicitly via `--eval-jsonl`.
- Every contradiction number in `OLD: results/ctc_suite/dense_vs_chunked_table.md` predates the
  rebuild and is measured on the contaminated ladder.

The headline consequence is worth carrying into the new repo: on the clean ladder the dense/chunked
**absolute gap narrows** (0.441 → 0.369) and only the *ratio* grows (2.1× → 2.9×), so contradiction
**fails** the plan's "gap grows ≥2×" O(N²) criterion. The previously published widening was partly
an artifact of the contaminated ladder collapsing the dense arm at 32k.

### 0.3 🟡 `qdmatch` and `retrieval` have their ObliQ source assignment inverted

`NEW: ctc/src/ctc/tasks/qdmatch/spec.py:113` — `sources=("nq", "hotpotqa", "obliq")`
`NEW: ctc/src/ctc/tasks/retrieval/spec.py:129` — `sources=(… "beir_scifact", "beir_fiqa", "msmarco")`, no `obliq`

`OLD: src/scripts/data/ctc_suite/BUILD_MATRIX.md:426-430` (roster change, 2026-07-19) says the
opposite of both:

> qdmatch-**ObliQ** is DROPPED from the suite. Keep qdmatch NQ (21a) + HPQA (21b). ObliQ instead
> enters as a **standalone in-context-retrieval** row (21c […]) via `generate_obliq_data.py`, NOT
> `generate_qdmatch_data.py`. Do not tokenize/train the `qdmatch_*obliq*` pilot jsonl — discard it.

So `obliq` belongs in `retrieval.sources` and must come out of `qdmatch.sources`. Left as-is, a
`--suite`-driven build would generate exactly the pilot files the matrix says to discard, and would
never generate the ObliQ retrieval row that replaced them.

---

## 1. Inventory

83 `.py` files, **21,446 lines**. Two distinct layers, and the split is not the filename prefix:

| layer | files | lines | what it is |
|---|---|---|---|
| `generate_*.py` — task generation | 45 | 15,147 | source corpus → unified JSONL. The port target. |
| `build_* / convert_* / verify_* / tokenize_* / split_* / mix_* / dump_* / align_* / prepare_* / audit_* / expand_* / subsample_* / finalize_* / rechunk_* / remap_*` | 37 | 6,299 | ladder derivation, shard conversion, audits, one-off probes |
| `__init__.py` | 1 | 0 | empty — there is no shared module inside this directory |

### 1.1 The 45 `generate_*.py`

L/B = load-bearing (named in `BUILD_MATRIX.md`'s active roster, or its output is consumed by a live
eval/launcher). "one-off" = no reference anywhere outside its own file. Evidence for each verdict is
a repo-wide grep for the filename and for the `source` string it emits.

| file | LoC | emits `source` | task pkg | verdict |
|---|---|---|---|---|
| `generate_pubmed_contradiction_data.py` | 747 | `pubmed_perturbation` | contradiction | **L/B** — BUILD_MATRIX row 16, the anchor |
| `generate_fever_contradiction_data.py` | 469 | `fever` | contradiction | **L/B** — the `contra_fever` OOD eval, wired into 4 eval drivers |
| `generate_wiki_contradiction_data.py` | 546 | `wiki_contradiction` | contradiction | one-off — zero refs repo-wide |
| `generate_contradiction_data.py` | 126 | *(none — alpaca schema)* | contradiction | one-off, superseded; SNLI prototype |
| `generate_pubmed_multiclaim.py` | 1505 | `pubmed_multiclaim` | contradiction | separate validated track, **not** in BUILD_MATRIX; 4-phase CLI |
| `generate_niah_contradiction_data.py` | 139 | `niah_contradiction` | retrieval | **L/B** — BUILD_MATRIX row 3 |
| `generate_nq_training_data.py` | 385 | `nq` | retrieval, qa, qdmatch | **L/B** — row 1; see §5.7, dangerous defaults |
| `generate_hotpotqa_data.py` | 483 | `hotpotqa` | retrieval, qa, qdmatch | **L/B** — row 2 |
| `generate_hotpotqa_unified_corpus.py` | 531 | `hotpotqa_unified_corpus` | retrieval | one-off prototype (self-labelled) |
| `generate_musique_unified_corpus.py` | 426 | `musique_unified_corpus` | retrieval | one-off; MuSiQue is not a suite row |
| `generate_beir_ce_data.py` | 203 | `beir_<ds>_ce` | retrieval | **L/B** — rows 7 (scifact) + 8 (fiqa) |
| `generate_beir_data.py` | 98 | `beir_<ds>` | retrieval | superseded by `_ce`; still exports `load_beir()` |
| `generate_msmarco_trainhn_data.py` | 450 | `msmarco_trainhn` | retrieval, rerank | **L/B** — rows 9 + 10, one build feeds both |
| `generate_msmarco_data.py` | 143 | `msmarco` | rerank | `main()` raises `NotImplementedError` (`:38`); survives only to export `_passage_text` |
| `generate_msmarco_trecdl_data.py` | 177 | `msmarco_trecdl<yr>` | rerank | deprecated behind `--force` (`:129`) |
| `generate_msmarco_helmet_rerank_data.py` | 176 | *(HELMET schema)* | rerank | one-off — BUILD_MATRIX:280 says "not needed for our in-tree path" |
| `generate_retrieval_triplets.py` | 273 | *(triplet schema)* | — | one-off, dense-retrieval baseline |
| `generate_helmet_qa_data.py` | 114 | `helmet_<src>` | qa | **L/B** row 5 — but see §5.11, v1 data is defective |
| `generate_helmet_summ_data.py` | 119 | `helmet_summ_<src>` | summarization | **L/B** row 6 — same caveat |
| `generate_oolong_data.py` | 101 | `oolong_<group>` | oolong | **L/B** — the fixed-bucket v1 path |
| `generate_oolong_ladder_data.py` | 428 | `oolong_<group>` | oolong | **L/B** row 4 — continuous-length rebuild, 12 referencing files |
| `generate_wiki_outlier_data.py` | 328 | `wiki_outlier_topic` | outlier | **L/B** row 11 |
| `generate_review_outlier_data.py` | 383 | `review_outlier_{rating,category}` | outlier | **dropped from the suite 2026-08-03** (`OLD: paperdraft/figures/fig4_cross_task.py:29`) |
| `generate_arxiv_grouping_data.py` | 678 | `openalex_grouping_{train,eval}` | grouping_labeled | **L/B** rows 13+14 — one generator, two tasks |
| `generate_textgroups_data.py` | 448 | `textgroups` | textgroups | **L/B** row 15 |
| `generate_absence_data.py` | 416 | `absence_{pubmed,gutenberg,numerical,official_*}` | absence | **L/B** row 18 (`--gutenberg`); `--official` dropped (row 19) |
| `generate_xabsence_data.py` | 288 | `xabsence_<tag>` | xabsence | **L/B** row 22; has its own sbatch |
| `generate_xabsence_abstracts_data.py` | 288 | `xabsence_abstracts_<tag>` | xabsence | one-off granularity pilot |
| `generate_strmatch_data.py` | 206 | `strmatch` | strmatch | **L/B** row 20 |
| `generate_matching_ngram_data.py` | 147 | `matching_ngram_wiki100w` | *(none)* | superseded by strmatch; eval path still live |
| `generate_mathmatch_data.py` | 214 | `synth_mathmatch` | mathmatch | **L/B** row 23 |
| `generate_cycle_data.py` | 230 | `cycle` | cycle | **L/B** row 25 |
| `generate_groups4_data.py` | 223 | `groups4` | groups4 | **L/B** row 26 |
| `generate_redundancy_data.py` | 240 | `pubmed_redundancy` | redundancy | task pkg exists; generator **dropped from the build** (BUILD_MATRIX row ~~17~~), LLM-bound |
| `generate_reorder_data.py` | 328 | `source_type: gutenberg` | reorder | **L/B** row 24 |
| `generate_wiki_reorder_data.py` | 145 | `source_type: wiki100w` | reorder | one-off alt substrate, zero refs |
| `generate_qdmatch_data.py` | 215 | `qdmatch_<tag>` | qdmatch | **L/B** rows 21a/21b |
| `generate_n2ified_data.py` | 182 | `n2ified_<tag>` | qdmatch | `main()` raises unconditionally (`:135`); its **output files are still live eval fixtures** in 5 scripts |
| `generate_obliq_data.py` | 400 | `obliq_<subset>` | retrieval | **L/B** row 21c |
| `generate_obliq_synthetic_data.py` | 591 | `obliq_synth_<subset>` | retrieval | pilot; validated but not promoted |
| `generate_ruler_data.py` | 478 | `ruler_<subtask>` | *(none)* | out of the 26-row roster; live eval path + 6 consumers |
| `generate_arithmetic_data.py` | 82 | *(none — alpaca)* | *(none)* | one-off FT smoke test |
| `generate_cot.py` | 174 | *(adds `chain_of_thought`)* | all | **L/B** — CoT enrichment pass |
| `generate_configs.py` | 563 | *(Axolotl YAML)* | — | dead — predates olmo-core |
| `generate_examples.py` | 261 | *(txt dumps)* | — | dead — superseded by `dump_suite_examples_full.py` |

### 1.2 The 37 pipeline / audit files

**Load-bearing (port or replace deliberately, 11):** `build_v2_eval_ladders.py` (504, the ladder
engine + the per-task index-base table), `build_xlong_rungs.py` (391, 64k–2M, runtime token
calibration), `build_v2_outlier_ladder.py` (200, the outlier scale-K fix), `build_obliq_token_ladder.py`
(126, OBLIQ needs token budgets not doc counts), `build_shared_corpus_evals.py` (654),
`build_combined_unified.py` (114, emits the `_task`/`_cot_mode` per-row dispatch tags),
`tokenize_unified_for_olmo.py` (316, called from `train.py:215,277`),
`verify_v2_eval_ladders.py` (167) and `verify_shared_corpus.py` (161) — the only real validators —
plus `subsample_beir_ladder.py` (83), `expand_obliq_train.py` (117), `mix_obliq_subsets.py` (80).

**Conditionally live (3):** `align_hn_doc_lengths.py` (302, self-deprecated for NQ, still imported
by `generate_hotpotqa_data.py:38-43`), `convert_unified_to_sft.py` (99, the older of two same-named
scripts — `OLD: src/scripts/data/README.md:36-43` says do not build new shards with it),
`dump_suite_examples_full.py` (275, current audit dump for ~25 tasks).

**One-off / dead (23):** `build_contra_recombined.py` (282, self-flagged "DO NOT USE THIS AS-IS",
`:5`), `build_contradiction_hardneg_pairs.py`, `prepare_contradiction_pair_classifier.py`,
`prepare_nq_pair_classifier.py`, `build_mathmatch_pairs.py` (four sentence-pair classifier probes,
none referenced), `audit_fever_contradictions.py`, `split_contradiction_mode_datasets.py`,
`tokenize_contradiction_for_titan.py`, `tokenize_nq_for_olmo.py` (raises at `:20`),
`convert_nq_to_sft.py`, `convert_tulu3_to_sft.py`, `convert_to_retrieval_triplets.py`,
`build_combined_train_alpaca.py`, `build_eval_pkl.py`, `split_train_eval.py`, `remap_cot.py`,
`mix_cot_tags.py`, `dump_suite_examples.py`, `rechunk_gutenberg_100w.py` (zero refs),
`finalize_oolong_v2.py` (forked into `debug/xlong_5task/finalize_oolong_eval.py`),
`generate_configs.py`, `generate_examples.py`, `generate_arithmetic_data.py`.

**Net: ~24 generators and ~11 pipeline files carry the suite. The other ~48 files are probes,
prototypes, superseded versions, and dead code. Do not port them; leave them in the old tree.**

---

## 2. Map to the 18 task packages

`ls ctc/src/ctc/tasks/` gives 18 registered packages (`NEW: ctc/src/ctc/tasks/__init__.py:38-57`).

| new package | generator(s) to port | note |
|---|---|---|
| `contradiction` | `generate_pubmed_contradiction_data.py`, `generate_fever_contradiction_data.py` | already stubbed at `NEW: …/contradiction/generate.py` with `SOURCES = {}`. The stub also names `generate_wiki_contradiction_data.py` as a third source — it has **zero refs**; drop it or mark experimental. |
| `redundancy` | `generate_redundancy_data.py` | generator is **dropped from the active build** (LLM-serving-bound). Port last or not at all. |
| `strmatch` | `generate_strmatch_data.py` | + optional `generate_matching_ngram_data.py` as a superseded ancestor |
| `mathmatch` | `generate_mathmatch_data.py` | pure synthetic |
| `cycle` | `generate_cycle_data.py` | pure synthetic |
| `groups4` | `generate_groups4_data.py` | pure synthetic |
| `textgroups` | `generate_textgroups_data.py` | pure synthetic |
| `absence` | `generate_absence_data.py` | 4 modes in one file; only `--gutenberg` (+ `--from`) are live |
| `xabsence` | `generate_xabsence_data.py` | two-phase: `--build-pool` (LLM) then assemble |
| `retrieval` | `generate_nq_training_data.py`, `generate_hotpotqa_data.py`, `generate_beir_ce_data.py`, `generate_obliq_data.py`, `generate_niah_contradiction_data.py`, `generate_msmarco_trainhn_data.py` | **6 sources — the widest fan-in in the suite** |
| `qa` | `generate_helmet_qa_data.py`; NQ/HPQA re-rendered via `--task qa` | |
| `grouping_labeled` | `generate_arxiv_grouping_data.py` | |
| `reorder` | `generate_reorder_data.py` | |
| `outlier` | `generate_wiki_outlier_data.py` | Amazon variant dropped 2026-08-03 |
| `rerank` | `generate_msmarco_trainhn_data.py` | **same build as `retrieval`'s msmarco source** |
| `oolong` | `generate_oolong_ladder_data.py` (+ `generate_oolong_data.py` v1) | |
| `summarization` | `generate_helmet_summ_data.py` | |
| `qdmatch` | `generate_qdmatch_data.py` | a *transform* of retrieval JSONL, not a corpus generator |

### (a) Generators for DROPPED tasks — do not port

- **plain `grouping`** — dropped. But note there is **no separate grouping generator**: `grouping`
  and `grouping_labeled` come from the *same* `generate_arxiv_grouping_data.py` output, split by a
  convert-time `--task` flag (`OLD: BUILD_MATRIX.md:338,342`). So dropping `grouping` costs nothing
  in generator code — it drops a render mode. The exclusion is recorded at
  `OLD: paperdraft/figures/fig4_cross_task.py:29`: *"outlier (Amazon), grouping (unlabeled),
  absence (official), redundancy -- removed from the suite on request 2026-08-03."*
- **`cot_retrieval`** — dropped; `NEW: ctc/src/ctc/tasks/cot_retrieval/` is an empty directory
  holding only a stale `__pycache__`. Delete it. Note the *format* layer deliberately keeps
  `cot_retrieval` (`NEW: format/documents.py:54`, `NEW: tasks/_retrieval.py:97`) because the golden
  fixture pins its instruction; that is correct and should stay.
- Also dropped and needing no port: outlier-Amazon (`generate_review_outlier_data.py`),
  absence-official (`generate_absence_data.py --official`), redundancy (`generate_redundancy_data.py`),
  qdmatch-ObliQ (see §0.3).
- The format layer also retains serializers for `grouping`, `ruler` and `matching_ngram`
  (`NEW: format/documents.py:152-162`) with no registered task. Same rationale (golden fixture) —
  fine, but worth a one-line comment so nobody "cleans them up."

### (b) Registered tasks with NO generator found

None are missing outright, but two are not what they look like:

- **`qdmatch` has no corpus generator.** `generate_qdmatch_data.py` consumes single-query retrieval
  JSONL via `--from-train/--from-eval` (`:137-138`) and repackages it. Its `generate.py` therefore
  needs a *different* signature from the others: it takes a built retrieval file, not a corpus.
- **`rerank` shares `retrieval`'s msmarco build.** One run of `generate_msmarco_trainhn_data.py`
  produces the JSONL both tasks read (`OLD: BUILD_MATRIX.md:278`). Two `generate.py` files calling
  one shared source module, or `rerank/generate.py` delegating — but not two builds.

### (c) Generators producing several tasks at once

- `generate_msmarco_trainhn_data.py` → `retrieval` + `rerank`
- `generate_arxiv_grouping_data.py` → `grouping_labeled` (+ dropped `grouping`)
- `generate_nq_training_data.py` → `retrieval`, and feeds `qa` and `qdmatch` downstream
- `generate_hotpotqa_data.py` → same three
- `generate_absence_data.py` → four `source` tags from one file
- `generate_ruler_data.py` → 7 subtasks, one file (no task package)

---

## 3. The shared layer

There is **no shared module inside `corpus_reasoning/data/`** — `__init__.py` is 0 bytes. The only
existing sharing is upward, into `corpus_reasoning/lib/`, and it is partial: **24 of the 82 non-empty
files import nothing from `corpus_reasoning` at all.** Adoption of `lib/`, by module:

| `lib/` module | importers in `data/` |
|---|---|
| `io.py` | 40 |
| `data_format.py`, `bm25.py`, `llm_request_client.py` | 8 each |
| `wiki100w_sample.py` | 4 |
| `bm25_local.py`, `prompts.py` | 3 each |
| `wiki100w_pool.py` | 2 |
| `cross_encoder.py` | **1** |

Meanwhile the generators **already import each other 11 times** —
`build_contradiction_hardneg_pairs.py:33` → `generate_pubmed_contradiction_data.load_pubmed_pool`,
`generate_beir_ce_data.py:40` → `generate_beir_data.load_beir`,
`generate_beir_data.py:27` and `generate_msmarco_trecdl_data.py:36` → `generate_obliq_data._build_examples`,
`generate_msmarco_trainhn_data.py:64` → `generate_msmarco_data._passage_text`,
`generate_xabsence_data.py:104` → `generate_pubmed_contradiction_data.{word_jaccard,clean_response}`,
`generate_absence_data.py:178` → `generate_reorder_data`,
`generate_hotpotqa_data.py:38` → `align_hn_doc_lengths`, and four more. That is the strongest
evidence that a `sources/` layer is wanted: it already exists, informally, as generators importing
private helpers out of each other.

🟠 **Bug in that pattern.** `OLD: build_xlong_rungs.py:59` is a bare `import build_v2_eval_ladders
as v2`, with a comment on `:58` claiming "sys.path hack removed: corpus_reasoning is a package on
PYTHONPATH=src". It is not importable that way — with `PYTHONPATH=src`,
`find_spec("build_v2_eval_ladders")` is `None` while
`find_spec("corpus_reasoning.data.build_v2_eval_ladders")` resolves. The script only runs if cwd is
`data/`; `python -m corpus_reasoning.data.build_xlong_rungs` raises `ModuleNotFoundError`.

### 3.1 JSONL I/O — 5 competing implementations, 23 more files with none

40 of 83 files import `corpus_reasoning.lib.io`. Against that:

1. `OLD: lib/io.py:42` `save_jsonl` — `Path(...).parent.mkdir(parents=True, exist_ok=True)`,
   default `ensure_ascii=True`. **38 importers.**
2. `build_shared_corpus_evals.py:54` + `build_v2_eval_ladders.py:198` — byte-identical to each
   other; use `os.makedirs(os.path.dirname(path), ...)`, which **crashes on a bare filename** with
   no directory component.
3. `generate_absence_data.py:242` + `generate_niah_contradiction_data.py:43` — identical to each
   other, **no mkdir at all**.
4. `generate_ruler_data.py:113` — the sole `ensure_ascii=False` in the tree, and no mkdir.
5. **23 further files with no helper at all**, 29 raw `f.write(json.dumps(x) + "\n")` sites
   (`generate_pubmed_multiclaim.py` has 4, `generate_xabsence_data.py` 3).

Reading is the same story: 14 files import `lib.io.load_jsonl`, **8 define a byte-identical local
copy** (`build_shared_corpus_evals.py:49`, `build_v2_eval_ladders.py:193`,
`generate_absence_data.py:237`, `generate_n2ified_data.py:44`,
`generate_niah_contradiction_data.py:38`, `generate_qdmatch_data.py:53`,
`generate_xabsence_data.py:43`, `generate_xabsence_abstracts_data.py:60`), and 18 more inline
`json.loads(line)` at 22 sites.

They do **not** agree on flags. `ensure_ascii` is explicit in exactly one place repo-wide
(`generate_ruler_data.py:116`, `=False`), so ruler's escaping is byte-incompatible with every other
task's. `sort_keys` is never used for row output. **`encoding=` is never passed to any `open()` in
the whole directory** — every read and write rides on the ambient locale, which is a latent
non-determinism across machines. **One `ctc.data.io` with settled `ensure_ascii` and explicit
`encoding="utf-8"` is a prerequisite for byte-for-byte parity testing**; take `lib/io.py`'s
behaviour, because 38 files already produce it.

### 3.2 Seeding — already uniform, and that is the good news for parity

- **59 files expose `--seed`** (default `42` ×57, `0` ×4, `1234` ×1).
- **63 files instantiate `random.Random(...)`.**
- **Zero occurrences of `random.seed(...)`, `np.random`, `numpy.random`, `default_rng`, or bare
  global `random.shuffle/sample/choice/randint`.** Every RNG use goes through an explicitly seeded
  instance. No file needs flagging.

So a golden-fixture parity test is feasible for every pure-synthetic generator. Four patterns must
be preserved *verbatim*, because each changes the stream and a "cleaner" rewrite silently changes
every output byte:

- `generate_ruler_data.py:464-468` — `seed` for train, `seed + 1` for eval.
- `generate_matching_ngram_data.py:78` — re-seeds **per example** from a running cursor
  (`rng = random.Random(cursor)`) instead of threading one stream.
- The derived-substream magic constants, reinvented in 4 files:
  `random.Random(SEED * 1_000_003 + ei)` at `build_v2_eval_ladders.py:377`,
  `build_v2_outlier_ladder.py:94`, `build_xlong_rungs.py:242`,
  `generate_pubmed_contradiction_data.py:563`; and `random.Random(SEED * 7 + ei * 101 + n)` at
  `build_v2_eval_ladders.py:437`, `build_v2_outlier_ladder.py:109`.
- The four ladder builders hardcode `SEED = 1234` as a module constant with **no `--seed` flag**
  (`build_shared_corpus_evals.py:46`, `build_v2_eval_ladders.py:188`,
  `build_v2_outlier_ladder.py:42`, `build_xlong_rungs.py:127`). Give them one during the port.

### 3.3 Corpus loading — 20 files, 20 distinct corpora, ~14 loader implementations

20 files call `load_dataset(...)`. The corpora:

| corpus | HF id / path | loaded by |
|---|---|---|
| PubMedQA | `qiaojin/PubMedQA` (`pqa_artificial`) | `generate_pubmed_contradiction_data.py:116` (the canonical `load_pubmed_pool`, imported by redundancy + xabsence), `generate_pubmed_multiclaim.py` |
| FEVER | `copenlu/fever_gold_evidence` | `generate_fever_contradiction_data.py:407-408` |
| SNLI | `stanfordnlp/snli` | `generate_contradiction_data.py:65` (dead) |
| NQ-open | `nq_open` | `generate_nq_training_data.py` |
| NQ (alt!) | `tilyupo/nq_cqa` | `generate_retrieval_triplets.py` — **a different NQ source from the canonical one** |
| HotpotQA | `hotpotqa/hotpot_qa` (distractor) | 4 files |
| MuSiQue | `dgslibisey/MuSiQue` | `generate_musique_unified_corpus.py` |
| MS MARCO | `BeIR/msmarco`, `BeIR/msmarco-qrels`, `sentence-transformers/msmarco-hard-negatives` | `generate_msmarco_*` ×3 |
| BEIR (any) | `BeIR/{name}` + `-qrels` | `generate_beir_data.py:load_beir()`, reused by `generate_beir_ce_data.py:40` |
| Wikipedia-100w | pyserini Lucene `wikipedia-dpr-100w` + `lib/wiki100w_pool.py`, `lib/wiki100w_sample.py` | 6 files |
| Gutenberg | `sedthh/gutenberg_english` (streaming) or `--local-dir` | `generate_reorder_data.py:192`, `generate_absence_data.py` |
| arXiv abstracts | `gfissore/arxiv-abstracts-2021` | `generate_pubmed_multiclaim.py` |
| OpenAlex | bulk `.gz` snapshot or REST API | `generate_arxiv_grouping_data.py` |
| Amazon Reviews | `McAuley-Lab/Amazon-Reviews-2023` | `generate_review_outlier_data.py` (dropped task) |
| Oolong | `oolongbench/oolong-synth` | `generate_oolong_data.py`, `generate_oolong_ladder_data.py` |
| NarrativeQA | `deepmind/narrativeqa` | `generate_helmet_qa_data.py` |
| ∞Bench | `xinrongzhang2022/InfiniteBench` | helmet qa + summ |
| GovReport | `ccdv/govreport-summarization` | `generate_helmet_summ_data.py` |
| Multi-LexSum | `allenai/multi_lexsum` | `generate_helmet_summ_data.py` |
| AbsenceBench | `harveyfin/AbsenceBench` | `generate_absence_data.py` (dropped mode) |
| OBLIQ-Bench | `dianetc/OBLIQ-Bench` (raw `requests`, cached) | `generate_obliq_data.py`, `generate_obliq_synthetic_data.py` |
| Tulu-3 | `allenai/tulu-3-sft-mixture` | `convert_tulu3_to_sft.py` (one-off) |

`sources/` should hold one module per corpus in the left column, minus the dropped ones —
roughly **14 modules**. Note PubMed and BEIR *already* have a de-facto canonical loader that other
files import; those two are lift-and-shift.

### 3.4 BM25 / cross-encoder — mostly shared, 4 + 3 implementations

- **BM25: 4 implementations, 12 consumer files, no accidental duplication.**
  `lib/bm25.py:210 BM25Searcher` (prebuilt `wikipedia-dpr-100w`; the rich one, with
  `batch_find_gold_and_hard_negs`, `prefetch_random_pool`, `mine_distractors`) — 8 files.
  `lib/bm25_local.py:58 LocalBM25Searcher` (indexes a caller-supplied corpus, caches on `_SUCCESS`)
  — 3 files. `generate_obliq_synthetic_data.py:203-245 SimpleBM25`, a deliberate pure-Python Okapi
  reimplementation (`k1=0.9/b=0.4`) because *"Anserini JVM init hangs on some jsteinhardt compute
  nodes… core-dumps on others"* — keep it and keep the comment. Plus **3 sites that open
  `LuceneSearcher.from_prebuilt_index` directly, bypassing both wrappers**:
  `generate_msmarco_data.py:80`, `generate_msmarco_trainhn_data.py:296`,
  `generate_msmarco_trecdl_data.py:69`. No `rank_bm25` anywhere.
- **Cross-encoder: 3 copies of the same ~30-line class, and the canonical one has 1 importer.**
  `OLD: lib/cross_encoder.py:11-38` is imported only by `generate_nq_training_data.py:341-344`.
  `generate_beir_ce_data.py:63-89` and `generate_msmarco_trainhn_data.py:73-110` re-declare it with
  the same docstring and body; the msmarco copy admits why (`:76-77`: *"duplicated to keep this
  data-gen script's import chain free of the BEIR/BM25 modules"*) and is the only one with
  multi-GPU `DataParallel` (`:91-92`). `cross-encoder/ms-marco-MiniLM-L-6-v2` is hardcoded as a
  default in 4 places. **Collapse to one, keep the `DataParallel` path, keep the import lazy** —
  the duplication existed to avoid a heavy import chain, and `sources/` solves that properly.
- **Hard-negative mining: 4 strategies**, worth keeping distinct rather than unifying — BM25
  top-hits-not-containing-answer (`lib/bm25.py:344,371`, shared by nq/hpqa/musique), BM25
  rank-window over a local corpus (`generate_obliq_data.py:232`, `generate_beir_ce_data.py:141`),
  CE-margin filtering of pre-mined negatives (`generate_msmarco_trainhn_data.py:197,355`), and
  LLM-judged (`generate_obliq_synthetic_data.py:425`). `hard_neg_indices` is emitted by 9 files and
  remapped by 4 ladder scripts.

### 3.5 Gold placement — 3 families, ~24 copies, 2 incompatible index bases

This is the block that decides gold indices, so a subtle difference here is an off-by-one that
reads as a modelling result. **This is the single strongest argument for the shared layer.**

**Family A — tag/shuffle/enumerate, 0-based. 8 sites in 7 files, near-verbatim.** Canonical form
(`generate_nq_training_data.py:38-46`):

```python
tagged = ([(gold_doc, "gold")] + [(d, "hard") for d in hard_negs]
          + [(d, "pool") for d in pool_distractors])
rng.shuffle(tagged)
documents        = [d for d, _ in tagged]
gold_indices     = [i for i, (_, tag) in enumerate(tagged) if tag == "gold"]
hard_neg_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]
```

Copies: `generate_obliq_data.py:163-179` (the original — and `generate_beir_data.py` /
`generate_msmarco_trecdl_data.py` correctly *import* it, proving the abstraction works),
`generate_beir_ce_data.py:43-60`, `generate_msmarco_trainhn_data.py:155-178`,
`generate_nq_training_data.py:36-59`, `generate_hotpotqa_data.py:183-200` **and** `:259-280` (two
in one file), `generate_hotpotqa_unified_corpus.py:381-396`,
`generate_musique_unified_corpus.py:173-192`.

**Family B — `order` + `old_to_new` remap, 1-based. 9 files.**
`generate_contradiction_data.py:43`, `generate_cycle_data.py:180`,
`generate_fever_contradiction_data.py:327`, `generate_groups4_data.py:168`,
`generate_pubmed_contradiction_data.py:414`, `generate_redundancy_data.py:146`,
`generate_strmatch_data.py:131`, `generate_wiki_contradiction_data.py:365`,
`generate_textgroups_data.py:359`. Three of them (pubmed / fever / wiki contradiction) are
**byte-identical**.

**Family C — one-offs, each subtly different and each a latent bug:**

- `generate_mathmatch_data.py:141-149` — builds `perm` **0-based**, adds `+1` at the use site, and
  the comment on `:139` says "1-indexed". Correct today; one edit from wrong.
- `generate_matching_ngram_data.py:79-89` — no remap dict; recovers positions by **string
  equality** (`if s == g`) with `assert len(positions) == 2`. 1-based.
- `generate_n2ified_data.py:85-93` — recovers by **`id()` object identity**. 0-based. Breaks
  silently if any document dict is ever copied.
- `generate_review_outlier_data.py:133-136` — also `id()`-keyed. 0-based.
- `generate_wiki_outlier_data.py:48-56` — bool-tag enumerate, 0-based, then re-rendered 1-based
  into `answers`.
- `generate_arxiv_grouping_data.py:497-501` — `inv` remap, 0-based gold, `answers` JSON 1-based
  (`:506`).
- Ladder re-permutation with a nested `new_gold_pos` closure: `build_v2_eval_ladders.py:436-445`,
  `build_v2_outlier_ladder.py:108-119`, `build_xlong_rungs.py:256`.

**Why this matters concretely:** the graders encode the base per task, by hand, in two places. In
`OLD: src/corpus_reasoning/eval/evaluate.py` there is a `+1` at `:1012`, `:1574`, `:1631`, `:1939`,
`:2077` and raw use at `:1806`, `:2242`; in `OLD: lib/data_format.py` there is `docs[a - 1]` at
`:340,408,437,485,501,514,541,572` versus `docs[g]` / `f"[{g + 1}]"` at `:287,604,613`. Two
structurally identical group tasks land on opposite bases — `generate_arxiv_grouping_data.py` emits
0-based clusters while `generate_cycle_data.py` and `generate_groups4_data.py` emit 1-based — and
that is only safe because `evaluate.py:1459-1461` happens to route them to different scorers.

Two helpers — `place_gold(gold, distractors, rng) -> (items, gold_idx_0based)` and
`remap(indices_or_pairs, perm, *, base)` with **`base` a required argument, never a default** —
retire ~24 hand-written copies of index arithmetic and give each spec's `gold_index_base`
declaration exactly one implementation to be checked against.

### 3.6 Smaller duplications worth folding in

| thing | copies | where |
|---|---|---|
| `_partition_with_min` (Dirichlet-ish split) | 2 | `generate_wiki_outlier_data.py:104-120`, `generate_review_outlier_data.py:187-201` — wiki's docstring says "Mirrors …" |
| `word_jaccard` | 2, **semantically divergent** | `generate_xabsence_abstracts_data.py:81-89` returns `1.0` on empty input; `generate_obliq_synthetic_data.py:165-171` returns `0.0`. Opposite behaviour in an overlap-*reject* filter. |
| `clean_response` (strip LLM prefix/quotes) | 2 | `generate_pubmed_contradiction_data.py:264-278`, `generate_wiki_contradiction_data.py:129-142` |
| `build_expression` (arithmetic to target) | 2 | `generate_mathmatch_data.py:96-123`, `generate_groups4_data.py:48-67` — independent rewrites |
| `_WORD = re.compile(r"[A-Za-z]{2,}")` + wiki-pool dedup loop | 2 | `generate_strmatch_data.py:44,49-62`, `generate_matching_ngram_data.py:31,40-55` |
| `gold_order[orig_idx] = display_pos + 1` + `word_count` | 2 | `generate_reorder_data.py:166-171,87-88`, `generate_wiki_reorder_data.py:57-62,43-44` (byte-identical `word_count`) |
| `load_jsonl` / `infer_tag` | 2 | `generate_qdmatch_data.py:53-55,198-205`, `generate_n2ified_data.py:44-46,165-172` |
| "grow the list until the token budget" loop | 3 in one file, + 3 elsewhere | `generate_ruler_data.py:255-258,298-302,337-341`; `build_eval_pkl.py:132-144`, `convert_tulu3_to_sft.py:212`, `build_obliq_token_ladder.py:64-81` |
| sentence splitting | 1 shared + 5 inline regexes | `lib/wiki100w_sample.py:203`; `generate_absence_data.py:95` (+ an nltk-punkt path at `:73-89`), `generate_retrieval_triplets.py:71`, `generate_pubmed_multiclaim.py:545`, and a stricter `(?<=[.!?])\s+(?=[A-Z])` at `generate_pubmed_contradiction_data.py:132` |
| title/body extraction from `"Title"\n<body>` chunks | 3 | `lib/wiki100w_sample.py:47`, `generate_hotpotqa_unified_corpus.py:87`, `align_hn_doc_lengths.py:40-43` |
| train/eval split | 5, **with different defaults** | `generate_reorder_data.py:269`, `generate_wiki_reorder_data.py:102`, `generate_obliq_data.py:341`, `split_contradiction_mode_datasets.py:26`, `split_train_eval.py:29` — `--eval-frac` 0.1 vs 0.2, `int(round(...))` vs bare `int(...)` |
| LLM client | 2 implementations, 9 consumer files | `lib/llm_request_client.py ParallelResponsesClient` (8 importers, **all lazy inside functions** to keep `openai`/`google.genai` off the train env's import path), and `generate_pubmed_multiclaim.py:65 LocalChatClient` — a stdlib-only `urllib` reimplementation written for exactly that reason (`:59-63`), and the only one with round-robin across comma-separated vLLM replicas |

**"How long is this example?" has four mutually inconsistent answers in-tree:**

1. Real tokenizer, `AutoTokenizer.from_pretrained` — 14 sites, and **the tokenizer id is not
   agreed**: `Qwen/Qwen3-0.6B` (`convert_unified_to_sft.py:58`, `convert_nq_to_sft.py:53`,
   `convert_tulu3_to_sft.py:54`, `generate_oolong_ladder_data.py:297`), `Qwen/Qwen3-4B`
   (`build_xlong_rungs.py:334`, `verify_v2_eval_ladders.py:145`), `Qwen/Qwen3-4B-Base`
   (`tokenize_nq_for_olmo.py:29`, `tokenize_contradiction_for_titan.py:74`), `Qwen/Qwen2.5-0.5B`
   (`generate_ruler_data.py:425`).
2. `chars // 4` — 5 copies (`build_eval_pkl.py:45`, `generate_ruler_data.py:371`,
   `generate_helmet_summ_data.py:24`, `generate_helmet_qa_data.py:28`,
   `convert_to_retrieval_triplets.py:110`).
3. A fitted linear model per task — `build_v2_eval_ladders.py:62,87,111,143`.
4. A word-count proxy — `build_obliq_token_ladder.py:37`.

Rung labels are only meaningful if one of these is chosen. §0.1 is what happens when two of them
disagree.

**7 dead argparse flags** (verified not `dest=` aliases), do not port:
`generate_wiki_outlier_data.py:212 --eval-frac`, `generate_textgroups_data.py:407 --separation`,
`generate_retrieval_triplets.py:230 --num-docs`, `generate_absence_data.py:390 --official-split`,
`generate_obliq_synthetic_data.py:505 --bm25-threads` and `:510 --index-dir`,
`generate_oolong_data.py:55 --config`.

### 3.7 Proposed `sources/` and `common/`

```
ctc/src/ctc/data/
  gold.py          place_gold(), remap(..., base=REQUIRED)   ← highest value; retires §3.5's ~24 copies
  schema.py        one Example constructor + validator: one document shape, one metadata key,
                   `source` from an enum, not a hand-typed literal   ← retires 49 construction sites
  io.py            load_jsonl / save_jsonl — settled ensure_ascii, explicit encoding, mkdir on write
  cli.py           shared parser parents (--seed 42, ONE output-dir name), the
                   `save_jsonl` + `print_dataset_stats` emit, the filename grammar, the split
  tokens.py        ONE length estimator, ONE tokenizer default, lazily imported
  retrieval.py     the 3 BM25 wrappers behind one interface + ONE CrossEncoderScorer
  llm.py           ParallelResponsesClient with LocalChatClient's replica round-robin folded in,
                   provider imports lazy so the 8 defensive in-function imports become normal ones
  sources/
    pubmed.py fever.py wiki100w.py gutenberg.py arxiv_openalex.py
    beir.py msmarco.py nq.py hotpotqa.py obliq.py oolong.py
    helmet.py absencebench.py amazon.py
```

Ordered by how much they retire: `gold.py` (~24 copies, and the only one that can produce a wrong
*number*), `schema.py` (49 construction sites), `io.py` (5 implementations + 29 inline sites),
`cli.py` (77 argparse blocks with a common shape, 23 files ending in the identical
`save_jsonl` + `print_dataset_stats` pair — e.g. `generate_reorder_data.py:322-329`),
`tokens.py` (4 competing estimators), `retrieval.py` (3 CE copies + 3 raw-Lucene bypasses).

`sources/` collapses ~40 loader implementations into ~14: MS MARCO 9→1, HotpotQA 4→1 (it loads the
same dataset twice inside `generate_hotpotqa_unified_corpus.py` alone, at `:170` and `:407`),
HELMET 5→1-with-configs, and 2→1 for each of Gutenberg / OOLONG / PubMedQA / OpenAlex / MuSiQue /
wiki100w. `generate_beir_data.load_beir` is the model to copy — it is the one loader that is already
shared by import.

`sources/` modules must be importable without pyserini/torch/transformers. The old code duplicated
the cross-encoder class *specifically* to keep import chains light
(`generate_msmarco_trainhn_data.py:76-77`), reimplemented an LLM client from stdlib for the same
reason (`generate_pubmed_multiclaim.py:59-63`), and defers the pyserini import to call time
(`generate_obliq_data.py:50-54,332`). `NEW: ctc/README.md` promises `pip install ./ctc` works with
"No GPU, no CUDA, no compiler" — put pyserini and torch behind extras and lazy imports so those
three workarounds become unnecessary rather than being carried forward.

Not a data generator and should not land in `ctc/data/`: `generate_configs.py` (563 lines) emits
Axolotl training YAML.

---

## 4. The unified example schema

### 4.1 The declared contract

`OLD: src/corpus_reasoning/lib/data_format.py:6-32` is the only place it is written down:

```python
{
    "documents": [{"title": "...", "text": "..."}, ...],
    "queries": ["question text"],           # list, even for single-query
    "answers": ["answer text"],             # list, even for single-answer
    "gold_doc_indices": [3],                # 0-indexed positions in documents
    "hard_neg_indices": [4, 7, 12],         # optional: BM25 hard negatives
    "source": "nq|hotpotqa",                # dataset origin
}
```

There is **no validator, no dataclass, no JSON schema.** The closest thing is
`OLD: verify_v2_eval_ladders.py:47-138` (ladder invariants: ≥500 rows, gold text identical across
rungs, indices in range, distractor nesting) and `OLD: build_v2_eval_ladders.py:223-244`
(`sanitize_gold`, a validator that silently *repairs* by dropping out-of-range gold). Writing a real
validator is a deliverable of the port, not a nicety.

**Required in practice** (accessed by bare subscript, never `.get`): `documents`, and the task's
gold field. **Optional** (always `.get`): `queries`, `answers`, `hard_neg_indices`, `ce_scores`,
`meta`/`_meta`.

### 4.2 Undocumented keys that are load-bearing

| key | type | who writes it | meaning |
|---|---|---|---|
| `ce_scores` | `list[float\|None]`, parallel to `documents` | msmarco/BEIR generators | CE relevance. `None` = unscored; the rerank grader treats it as gain 0 and excludes it from the Kendall-tau set (`OLD: build_xlong_rungs.py:113-117`) |
| `hard_neg_indices` | `list[int]`, 0-based | all retrieval generators | mined negatives; remapped by every ladder builder |
| `cluster_labels` | `list[str]`, parallel to gold clusters | `generate_arxiv_grouping_data.py:515` | group names — **not scored** |
| `gold_order` | `list[int]`, **1-based** | reorder generators | full permutation; `reorder` has no `gold_doc_indices` at all |
| `gold_pairs` | `list[[int,int]]`, **1-based, ORDERED** | `generate_qdmatch_data.py:130` | qdmatch has no `gold_doc_indices` |
| `_meta` / `meta` | dict | ~half the generators | task metadata. **Both spellings are in use** — `_meta` for textgroups/cycle/groups4/oolong/absence-gutenberg-adjacent, bare `meta` for outlier and absence-gutenberg |
| `_task` / `_cot_mode` | str | `build_combined_unified.py:96-98` | per-row dispatch in a multitask file; read at `tokenize_unified_for_olmo.py:196-197` |
| `chain_of_thought` | str | `generate_cot.py:112` | optional CoT |
| `corpus_id`, `shared_prefix_len`, `shared_prefix_sha1` | | `build_shared_corpus_evals.py:74-80` | additive, shared-corpus evals only |
| `source_type`, `source_id` | str | reorder generators | reorder writes `source_type`, **not** `source` |
| `num_docs`, `num_pairs`, `num_unmatched`, `num_queries`, `num_relevant`, `layout`, `level`, `k`, `hop_count`, `musique_id`, `split` | | various | per-generator extras at top level |

### 4.2b Six real inconsistencies inside the "unified" format

There are **49 independent constructions of the example dict across 34 files** (key frequencies:
`documents` 58, `source` 55, `queries` 55, `answers` 50, `gold_doc_indices` 49, `hard_neg_indices`
9, `ce_scores` 1). Four files build it more than once — `generate_absence_data.py` and
`generate_ruler_data.py` 4 times each, `generate_hotpotqa_data.py` and
`generate_review_outlier_data.py` 3 each. The drift that accumulated:

1. **The document shape has three variants, and 11 files read `doc.get("title")`** — so absent vs
   `None` vs `""` is a live distinction, not cosmetic:
   - bare `{"text": ...}` — 19 files (all the pair/group/synthetic tasks)
   - `{"title": None, "text": ...}` — 13 files (`generate_beir_ce_data`, `generate_obliq_data`,
     `generate_msmarco_*`, `generate_niah_contradiction_data`, `generate_helmet_*`,
     `generate_reorder_data`, `generate_ruler_data`, `generate_wiki_outlier_data`,
     `build_v2_outlier_ladder`, …)
   - `{"title": "", "text": ...}` — 3 files (`align_hn_doc_lengths`,
     `generate_hotpotqa_unified_corpus`, `generate_musique_unified_corpus`)
2. **`_meta` (14 files) vs `meta` (4 files)** with no rule. Bare `meta`:
   `generate_wiki_outlier_data`, `generate_review_outlier_data`, `generate_absence_data`,
   `build_v2_outlier_ladder`.
3. **`reorder` and `wiki_reorder` write `source_type`, not `source`** — every `source`-keyed
   consumer misses them.
4. **`answers` is overloaded four ways**: real answers (nq, hotpotqa), a `[""]` placeholder
   (obliq, beir, msmarco), `[]` (the whole contradiction family), and a *rendered gold-position
   string* (`generate_wiki_outlier_data.py:93`, `build_v2_outlier_ladder.py:117`). A validator that
   only checks the key is present proves nothing.
5. **28 distinct `source` string literals, hand-typed at each construction site.** Make it an enum.
6. `generate_pubmed_multiclaim.py:1365-1375` omits `answers` entirely.

Normalise all six in `schema.py`, and **record each as a deliberate change** — a run built on the
normalised shape is not byte-comparable with an old one, and that needs to be a stated decision
rather than a surprise during a parity test.

### 4.3 Gold index base, per task — verified against every generator

The `data_format.py` docstring says "0-indexed positions." **That is wrong for 8 of the 18 tasks.**
The authoritative old-side statements are `OLD: build_v2_eval_ladders.py:54` (`index_base: 1` for
contra), `OLD: build_xlong_rungs.py:76,99,107,121`, `OLD: verify_shared_corpus.py:64-65`
(`index_base = 1 if args.task == "contradiction" else 0`) and `OLD: dump_suite_examples.py:32-33`
(a 7-name `ONE_INDEXED` set).

| new task | new `gold_index_base` | old generator, proving line | agree? |
|---|---|---|---|
| contradiction | 1 (`spec.py:163`) | `generate_pubmed_contradiction_data.py:412-419`; docstring `generate_fever_contradiction_data.py:45` | ✅ |
| redundancy | 1 (`_pairs.py:133`) | `generate_redundancy_data.py:19,146-148` | ✅ |
| strmatch | 1 (`_pairs.py:133`) | `generate_strmatch_data.py:18,131-134` | ✅ |
| mathmatch | 1 (`_pairs.py:133`) | `generate_mathmatch_data.py:9,148` | ✅ |
| cycle | 1 (`_cycles.py:112`) | `generate_cycle_data.py:37,180-184` | ✅ |
| groups4 | 1 (`_cycles.py:112`) | `generate_groups4_data.py:32,168-171` | ✅ |
| textgroups | 1 (`_cycles.py:112`) | `generate_textgroups_data.py:42,359-363` | ✅ |
| qdmatch | 1 (`spec.py:100`, field `gold_pairs`) | `generate_qdmatch_data.py:22-29,115-123` | ✅ |
| reorder | 1 (`spec.py:111`, field `gold_order`) | `generate_reorder_data.py:18,169-171` | ✅ |
| absence | 0 (`_absence.py:109`) | `generate_absence_data.py:25,254,265` | ✅ |
| xabsence | 0 (`_absence.py:109`) | `generate_xabsence_data.py:15,161` | ✅ |
| grouping_labeled | 0 (`_grouping.py:141`) | `generate_arxiv_grouping_data.py:29,492-501` | ✅ |
| outlier | 0 (`spec.py:100`) | `generate_wiki_outlier_data.py:24,48-56`; `build_v2_outlier_ladder.py:21-22,112` | ✅ |
| retrieval | 0 (`spec.py:116`) | `generate_nq_training_data.py:45`, `generate_obliq_data.py:170`, `generate_beir_ce_data.py:51`, `generate_niah_contradiction_data.py:89-96` | ✅ |
| qa | 0 (`spec.py:132`) | `generate_helmet_qa_data.py:101` (`[0]`) | ✅ |
| rerank | 0 (`spec.py:142`) | `generate_msmarco_trainhn_data.py:167` | ✅ |
| summarization | 0 (`spec.py:138`) | `generate_helmet_summ_data.py:104` (`[0]`) | ✅ |
| oolong | 0 (`spec.py:153`) | `generate_oolong_data.py:78` — always `[]`, field is vestigial | ✅ (vacuous) |

**Every declared `gold_index_base` in the new repo matches the old generator. No disagreement
found.** That is a genuinely good result and the table above is the evidence; keep it as the
regression reference.

Three caveats that are *not* disagreements but will bite:

- `oolong` declares `gold_index_base=0` over a field that is always `[]`. Harmless, but do not let
  an audit "verify" it and report a pass that means nothing.
- **`generate_ruler_data.py:262-272` and `generate_n2ified_data.py:85-94` are 0-based** while their
  nearest neighbours in the pair/group family are 1-based. Neither has a task package, so this only
  matters if either is ever revived.
- `OLD: dump_suite_examples.py:32-33`'s `ONE_INDEXED` set **omits `textgroups`** (and `qdmatch`,
  `reorder`), which really are 1-based. It is a display tool that never loads those tasks, so it
  never misfires today — but do not copy that set forward as the authority. The table above is.
- `generate_review_outlier_data.py:24` documents `gold_doc_indices` as *"list of list of 0-indexed"*
  while the code at `:136,142` emits a **flat** list. The docstring is wrong. Dropped task, but
  worth not copying.

### 4.4 Rungs are document counts, fit per task — and that table is not ported

`OLD: build_v2_eval_ladders.py` maps a rung *label* to a fixed **document count**, from an offline
linear fit recorded in comments per task (`tokens = 170 + 42.8·n` for contra `:87`,
`66.6 + 113.36·n` for hpqa `:111`, `114.1 + 146.93·n` for outlier `:143`). Not every task reaches
every rung: nq stops at 8k (`:131`), hpqa and rerank at 16k (`:116`, `:161`). `build_xlong_rungs.py`
does better — it calibrates against the real tokenizer at runtime (`:147-159`) precisely because the
offline fits drifted with the filler pool (`:60-89`).

The full 26-row rung→n table lives at `OLD: src/scripts/data/ctc_suite/BUILD_MATRIX.md:106-140`.

`NEW: ctc/src/ctc/format/rungs.py:5-6` says that mapping "lives in `configs/tasks/<task>.yaml`."
**`configs/tasks/` and `configs/suites/` are both empty directories.** Porting that table is a
work item in its own right; without it `ctc-data build --rungs all` cannot resolve a document count
for any task.

---

## 5. Risks

Each trap is tied to the file it lives in. Status verified against the current old-tree code, not
just the record.

| # | trap | lives in | status | rebuild data? |
|---|---|---|---|---|
| 1 | Qwen3 marker embeddings: cosine 1.0000, then wrong norm | `OLD: src/scripts/data/fix_marker_embeddings.py` | FIXED (seeds from real delimiter rows, asserts in-dist norm) | No — checkpoint fix. Re-repair any base from before 2026-07-14 |
| 2 | oolong `--item-regex '\|\|'` matches every line | `OLD: src/scripts/data/convert_unified_to_document_landmark.py:365-382` | FIXED — startup guard rejects any regex matching `""`; test at `src/test/data/document_chunk_item_regex_test.py` | **Yes** — shards before 2026-07-27 |
| 3 | oolong preamble train/eval layout mismatch | converter's oolong prompt construction, vs `OLD: src/olmo_core/data/document_chunk_landmark.py:251-272` | **OPEN** per `records/paper-v2-todo-status.md:465-485` (2026-08-05); code inspection could not localise the live diff | Yes, once fixed |
| 4 | FEVER/wiki fillers in PubMed contradiction evals | `harvest_fillers`, `OLD: build_v2_eval_ladders.py:255-296`, called from `build_xlong_rungs.py:197` | FIXED — glob narrowed to `contradiction_*pubmed*_k3.jsonl` (`:76`, `:89`); CTC ladder rebuilt + revalidated 2026-08-04/05. See §0.1, §0.2 | Contaminated rungs exist; **training data was always clean** |
| 5 | contra 1-based gold read as 0-based; outlier shrink breaks scale-K | `OLD: build_v2_eval_ladders.py:54,205-210,451`; fix vehicle `build_v2_outlier_ladder.py` | FIXED, both paths present | Rebuilt already |
| 6 | docchunk id/title label leaked as FREE token | `OLD: src/olmo_core/data/document_chunk_landmark.py:317-349` | FIXED (contiguous spans absorb labels + separators); confirmed by the chunk-leak audit, 21/22 tasks clean | Yes, shards from before ~2026-07-06; A/B retrain was inconclusive |
| 7 | NQ 98 %-hard pipeline | `OLD: generate_nq_training_data.py` | **OPEN as a code guard.** `--hard-neg-frac` defaults to **1.0** (`:278`) and `--ce-filter` is `store_true`, default off (`:307`). Running with defaults silently reproduces the banned regime | Banned files (`hn49/hn99/hn199/ladder64k`) exist and must never be used |
| 8 | `parse_doc_ids` required both brackets | `OLD: lib/metrics.py:98-106` FIXED (`r'\[?\s*(\d+)\s*\]'`) vs `OLD: src/scripts/ctc_eval/lib/metrics.py:98-100` **STILL BUGGY** | live path fixed; a stale buggy twin remains and is what a porter grepping for "the evaluator" may find first | No — grading bug |
| 9 | per-task gold index base | §4.3 | standing convention; new repo currently correct | N/A |
| 10 | eval doc-id digit-range mismatch (train n-max 697 vs eval 1423) | no single file — a build discipline | OPEN by nature; detector at `OLD: debug/length_mix_scaling/check_digit_truncation.py` | Per-build check |
| 11 | goldgrad `--max-length` truncation → empty gen → f1 0.000 | `OLD: src/scripts/eval/ctc_suite/run_rung_eval.py:466-530` | FIXED and generalised — auto-sizes from the measured prompt distribution and hard-fails unless `--allow-short-max-length` | No |
| 12 | eval-bundle weka staging: S3 push alone does nothing; a job that skips every rung still exits 0 | process | discipline, two-step gantry sync | N/A |

### Traps found in `records/` that are NOT on the usual list

13. **`cycle` and `groups4` had exploitable frequency shortcuts — already fixed in code.**
    `OLD: records/ctc-setting-verification-2026-07-23.md` diagnosed cycle entities pinned at
    frequency 2 while distractor frequency grew with N (so gold was "the 3 rarest names"), and
    groups4 distractors kept so far apart that any close pair was gold. The current
    `generate_cycle_data.py:28-33` and `generate_groups4_data.py:18-27` cite that record and
    describe the fix inline. **A porter reading only the record would re-flag a solved problem, or
    distrust data that is now fine.** The fix is subtle (cycle entities are now full participants in
    background-edge sampling, subject to a no-both-endpoints-in-one-block rule) and a naive rewrite
    would lose it — this is exactly why these two want a golden-fixture parity test, not a clean
    reimplementation.

14. **`generate_arxiv_grouping_data.py` level/k-density confound — also already fixed.** Same
    record: `sample_k_for_level` picked k as a fraction of `n_docs`, ignoring that OpenAlex L0 has
    ~19 top-level fields, so coarse levels silently dropped out as N grew (L0 share 57 % → 0 %) and
    "more docs" got conflated with "finer grouping." The 3-part capacity-aware fix is at `:282-343`.

15. **HELMET v1 data is defective at the source, not just mis-scored.**
    `OLD: records/helmet-narrativeqa-govreport-repair.md`: `generate_helmet_qa_data.py` puts raw
    scrape text (IMSDb HTML chrome, Gutenberg license headers) into context with no length filter,
    so at the 2k rung **only 9.5 % of examples contain the gold answer anywhere in the retained
    context**, and the same defect is baked into the training set the same generator builds. The
    GovReport 16k and 32k rungs are literally the same documents (median context 8765 both). The v2
    fix is `OLD: debug/ctc_helmet_v2/build_helmet_v2_data.py`. **Port v2, not
    `generate_helmet_qa_data.py`/`generate_helmet_summ_data.py`.**

16. **`build_contra_recombined.py` — every validation passed and the task still broke.** Pooling
    gold pairs against a *globally* sampled filler pool (instead of each example's co-sampled
    distractors) dropped full-attention train f1 from 0.934 to 0.585, because the nearest non-gold
    pair's similarity crashed 0.372 → 0.163 — "pick the two most similar claims" then solves the
    task. Contamination checks, pair-reuse checks and gold-pair-distance matching all passed; none
    asked whether it was still the same task. Scale contradiction with
    `generate_pubmed_contradiction_data.py --expand-from-train`, never by recombination.

17. **`generate_pubmed_multiclaim.py`'s lexical leak is n-dependent and the fix is deferred.**
    String-overlap baseline runs F1 0.514 at n=21 down to 0.036 at n=397 — strongest at exactly the
    short rungs the ladder varies. Cause: contradictions hold subject/outcome/population/timepoint
    fixed (forcing entity overlap) while distractors come from unrelated abstracts, and siblings are
    capped at ~9/example so their share collapses as n grows. The paraphrase-distractor fix is
    designed and **explicitly deferred by the user** — do not port this as "already fixed."

18. **`--vocab-source file` in `generate_strmatch_data.py:177` defaults to `/usr/share/dict/words`**
    — machine-dependent content, so parity tests must pin `--vocab-source wiki` or ship a wordlist.

19. **Streaming HF datasets are not seed-reproducible.** `generate_reorder_data.py:192` uses
    `load_dataset(..., streaming=True)`; book order depends on the stored shard order and the
    `datasets` version. Its `--local-dir` path (`:208`, `sorted(glob)`) *is* deterministic. Parity
    fixtures must use `--local-dir`.

20. **`generate_n2ified_data.py` is dead code whose output is still live.** `main()` raises at
    `:135`, but `data/n2ified_eval_{nq,hpqa}_q20.jsonl` are read by 5 current eval scripts. Do not
    delete the JSONL when you decline to port the generator. It also violates the repo's own
    `deprecated/` convention by sitting live and non-functional.

21. **`build_xlong_rungs.py:59` is not importable as a module** — bare
    `import build_v2_eval_ladders as v2`, with a `:58` comment asserting the opposite. It works only
    from cwd `data/`. See §3. Fix during the port rather than reproducing the comment.

22. **Two generators key gold recovery on `id()` object identity** —
    `generate_n2ified_data.py:85-93` and `generate_review_outlier_data.py:133-136`. Correct today
    only because `rng.sample` returns distinct objects. Any refactor that copies a document dict
    (a `dict(d)`, a `deepcopy`, a round-trip through JSON) silently produces wrong gold with no
    error. Both are on the do-not-port list, but the pattern must not be carried anywhere else.

23. **No `open()` in the whole directory passes `encoding=`.** Every read and write depends on the
    ambient locale, so "byte-identical" is machine-dependent for any non-ASCII corpus (PubMed,
    Gutenberg, OpenAlex all carry non-ASCII). Set `encoding="utf-8"` explicitly in `ctc.data.io`
    before building the first parity fixture, or the fixture will pass on one node and fail on
    another.

---

## 6. Sequencing

Ordering principle: **batch by what can be validated, not by task family.** A batch is done when
old and new produce byte-identical JSONL from the same seed and CLI, checked by a fixture test that
lives in `ctc/tests/data/fixtures/` alongside the existing `golden_format.json` pattern
(`NEW: ctc/tests/format/test_golden_parity.py`, whose docstring already states the rule: *"A failure
here means the port changed behaviour. Fix the port. Regenerating the fixture would delete the
evidence instead of the bug."*).

### Batch 0 — foundations (no generators; blocks everything else)

1. `ctc/data/io.py` — one `save_jsonl` (`ensure_ascii=True`, `encoding="utf-8"`, mkdir-on-write,
   otherwise matching `OLD: lib/io.py:42-46`, which 38 files already produce). Retires 5
   implementations + 8 `load_jsonl` forks + 29 inline sites (§3.1) and closes trap 23.
2. `ctc/data/gold.py` — `place_gold` and `remap(..., base=REQUIRED)` (§3.5). Unit-test both index
   bases against the §4.3 table; that table is the regression reference.
3. `ctc/data/schema.py` — one `Example` constructor + the first real validator this pipeline has
   ever had (§4.2, §4.2b). Normalise the document shape, the `meta` key, and `source`.
4. **Fix the three defects in §0** — `CLAIMS_PER_RUNG`, the stale FEVER docstring, the ObliQ source
   inversion. Three edits, cheap now and expensive after the first rebuild.
5. Port `OLD: BUILD_MATRIX.md:106-140` into `configs/tasks/<task>.yaml` (§4.4) — using the
   **`contra_ctc`** row for contradiction — and pick **one** length estimator and **one** tokenizer
   id (§3.6), since the rung labels mean nothing until that is settled.

### Batch 1 — pure synthetic, byte-parity provable, 4 tasks / 1,115 lines

`cycle` (230), `groups4` (223), `mathmatch` (214), `textgroups` (448).

Why first: no corpus, no model, no network; a single `random.Random(seed)` threaded through
(`generate_cycle_data.py:214` and siblings); zero unseeded RNG anywhere in the tree (§3.2). These
four can be generated old-side and new-side and diffed byte-for-byte with no environment. They also
exercise both gold shapes — pairs (`mathmatch`) and variable-length groups (`cycle`, `groups4`,
`textgroups`) — and all four are 1-based, so `placement.remap_pairs` gets proved on the harder base
first. Carry the anti-shortcut fixes of trap 13 across verbatim; the parity test is what guarantees
you did.

**Exit criterion:** 4 golden fixtures, byte-identical, plus a test asserting each spec's
`gold_index_base` against the generated file.

### Batch 2 — synthetic-with-a-frozen-corpus, 3 tasks / 1,022 lines

`strmatch` (206), `reorder` (328) + `rechunk`, `absence --gutenberg` (416, the live mode only).

Parity is still provable but requires pinning the substrate: `--vocab-source wiki` against a fixed
Lucene snapshot (trap 18), and `--local-dir` Gutenberg rather than streaming (trap 19). Do this
batch second precisely to establish the "pinned corpus fixture" pattern on three cheap tasks before
the expensive ones need it. `reorder` and `absence` also force the two schema oddities into the open
early: `gold_order` instead of `gold_doc_indices`, and `source_type` instead of `source` (§4.2).

### Batch 3 — the retrieval spine, 4 tasks / ~1,700 lines

`retrieval` + `rerank` + `qa` + `qdmatch`, from `generate_nq_training_data.py` (385),
`generate_hotpotqa_data.py` (483), `generate_beir_ce_data.py` (203),
`generate_msmarco_trainhn_data.py` (450), `generate_obliq_data.py` (400),
`generate_niah_contradiction_data.py` (139), `generate_qdmatch_data.py` (215).

This is where `sources/` earns its keep: 6 corpora, 8 copies of the tag→shuffle→enumerate idiom
(§3.5), and 3 copies of the cross-encoder (§3.4) all collapse here. Sequence *within* the batch:

- 3a. `sources/{nq,hotpotqa,beir,msmarco,obliq}.py` + `retrieval.py` (BM25 + CE), no task code.
- 3b. `retrieval/generate.py` over all 6 sources. **`generate_nq_training_data.py` must be ported
  with `hard_neg_frac=0.1` and `ce_filter=True` as the DEFAULTS, not opt-ins** (trap 7) — the old
  defaults reproduce the banned 98 %-hard regime and nothing warns you.
- 3c. `rerank/generate.py` delegating to the same msmarco build (§2c).
- 3d. `qa/generate.py`, `qdmatch/generate.py` (a transform, different signature — §2b).

Byte-parity is achievable per-source given a pinned index and a CE-score cache, but expect to fall
back to *distributional* parity (gold count, hard-neg ratio, doc-length quantiles, index-range) for
the CE-filtered paths. Say which one you achieved, per source.

### Batch 4 — contradiction, 2 sources / 1,216 lines

`generate_pubmed_contradiction_data.py` (747) + `generate_fever_contradiction_data.py` (469) into
the already-stubbed `NEW: …/contradiction/generate.py`.

Fourth, not first, despite being the flagship: it needs an LLM endpoint, it carries the most bug
history, and the §0.1 ladder fix must be in place before a single rung is built. Port the
`--expand-from-train` path (the no-LLM resize) at the same time — it is how contradiction scales,
and trap 16 says recombination is not. Do **not** port `generate_wiki_contradiction_data.py`
(zero refs) or `generate_pubmed_multiclaim.py` (separate track, trap 17) in this batch.

Validation is distributional, not byte-level: gold-pair count, 1-based invariant, nearest-non-gold
similarity (the statistic trap 16 shows is the one that matters), and 0.00 % FEVER in the filler
pool. Re-verify the rung token medians against the §0.1 table.

### Batch 5 — the long tail, 5 tasks / ~2,000 lines

`oolong` (`generate_oolong_ladder_data.py`, 428), `outlier`
(`generate_wiki_outlier_data.py`, 328 + `build_v2_outlier_ladder.py`, 200),
`grouping_labeled` (`generate_arxiv_grouping_data.py`, 678 — carry the trap-14 fix),
`xabsence` (`generate_xabsence_data.py`, 288, two-phase LLM),
`summarization` + `qa`-HELMET — **from `OLD: debug/ctc_helmet_v2/build_helmet_v2_data.py`, not the
v1 generators** (trap 15).

Last because each has a bespoke dependency (a pinned OpenAlex snapshot, an article-pool pickle, an
LLM paraphrase pool, the oolong HF set) and none blocks another task. `oolong` additionally should
not be built until trap 3 (preamble layout) is resolved, or its shards will need rebuilding anyway.

### Not ported

`redundancy` (dropped from the build), outlier-Amazon, absence-official, plain `grouping`,
qdmatch-ObliQ, `ruler`, `matching_ngram`, `musique`, and the ~23 one-off/dead files in §1.2. Delete
`NEW: ctc/src/ctc/tasks/cot_retrieval/`. Where the old tree keeps a live artifact from a dead
generator — `n2ified_eval_{nq,hpqa}_q20.jsonl` (trap 20) — keep the artifact.

---

## 7. Known gaps in this plan

- **Trap 3 (oolong preamble) is not localised.** The record says open as of 2026-08-05; reading
  `_wrap_item_lines` suggests it may be resolved. Neither reading is trustworthy without re-running
  `OLD: debug/ctc_vllm_validation/validate_chunk_leak.py` against current shards. Do that before
  Batch 5.
- **I did not verify which contradiction rung files are physically staged where.** §0.1 rests on the
  code comments in `build_v2_eval_ladders.py` and the verification table in
  `records/paper-v2-todo-status.md:67-77`, both of which are unambiguous; I did not read the JSONL.
- **"Load-bearing" is inferred from repo-wide greps** (references from sbatch, launchers,
  `BUILD_MATRIX.md`, `records/`). A generator run only by hand and never committed to a launcher
  would read as one-off here. The 45-row table in §1.1 shows the evidence per row so any single
  verdict can be re-checked cheaply.
- **Byte-parity feasibility for Batch 3 is asserted, not demonstrated.** The CE-filtered paths depend
  on a model and a cache; treat "byte-parity or distributional parity" as a decision to make
  per-source during the batch, not a settled fact.
