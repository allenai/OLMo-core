# CTC Suite — Stage 1 Data Build Matrix

Companion to `records/ctc-suite-scaling-plan.md` (§3 task table, §11 decisions). One block per
task row (~28 rows incl. the 3 qdmatch variants). **Nothing here has been run** — this is the
audited build spec: exact commands, pool locations (checked), per-doc token estimates, and gaps.

Sources audited: every `src/corpus_reasoning/data/generate_*.py` argparse surface, the catalog
docs (`corpus-reasoning/results/task_suite_README.md`, `results/task_suite.md`), the v2 ladder
builders (`corpus-reasoning/scripts/data/build_v2_eval_ladders.py`, `build_xlong_rungs.py`), and
the chunk-by map in `src/scripts/data/convert_docchunk_singletask_v2_local.sbatch`.

---

## 0. Global conventions (read first)

**Requirements update (2026-07-18, supersedes plan §4 "3–5k"):**
- **TRAIN = ~20k examples/task** (FINAL), uniform over n ∈ [n(2k), n(32k)]. **PILOT = 2–3k**
  (same recipe, `--num-train` scaled down). Per-task feasibility vs. pool size is audited below.
- **EVAL sets are FIXED across rungs**: the SAME 500 underlying examples (queries + golds) at
  every rung; only distractor/filler count varies. This is exactly the nested-prefix design of
  `build_v2_eval_ladders.py` (contra/nq/outlier/rerank already covered; `CAP_PER_RUNG=500`,
  row-aligned canonical). For all other tasks the per-row block says how to get there
  (usually: build the eval ONCE at n(32k), then **shrink-derive** the smaller rungs — removing
  distractors can never create a new gold for any task here, so shrink is always safe; expand
  needs a gold-safe filler pool like contra's).

**Uniform-n policy:**
- Generators WITH a per-example continuous-n knob (marked ✔ below): one command,
  `min=n(2k), max=n(32k)`.
- Generators WITHOUT (marked "discrete-uniform"): build at the 5 discrete values
  n(2k), n(4k), n(8k), n(16k), n(32k) with equal counts (FINAL: 5 × 4k = 20k;
  PILOT: 5 × 500 = 2.5k) and `cat` the files. No code change required.

**Eval rungs:** separate files at n(2k), n(4k), n(8k); **16k/32k DEFERRED** per plan §11 Q2
(n values still listed so the extension is a re-run, not a re-derivation). eval_size 500 each;
anything smaller flagged inline. NOTE: the existing v2 ladders use rung labels 3k/8k/16k/32k —
**every task, including the 4 already-covered ones, needs the rung set re-derived to
2k/4k/8k(/16k/32k)** (config edit in `build_v2_eval_ladders.py` + rebuild).

**Downstream converter** (`src/scripts/data/convert_unified_to_document_landmark.py`):

```bash
python src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task <task_key> --chunk-by <document|line> \
  --cot-mode <none|catalog mode> --query-position both \
  --seq-len 40960 --tokenizer <Qwen3.5 HF path> \
  --input-jsonl <unified.jsonl> --out-dir /data/prasann/ctc_suite/shards/<task>
```

- ✅ **B0 RESOLVED (was: marker ids hardcoded Qwen3).** The converter now takes
  `--marker-set qwen3|qwen3_5` (id sets registered in
  `olmo_core.data.document_chunk_landmark.RESERVED_IDS`; `qwen3_5` = 248049/248050/248044,
  landmark/pad 248200/248203 past the real vocab 248077, embedding 248320), plus explicit
  `--doc-start-id/--doc-end-id/--eos-token-id/--landmark-id/--pad-id` overrides. The resolved
  box ids are verified against `--tokenizer` at startup (mismatch = hard error, not silent
  mixing) and recorded in the shard's `metadata.json` (`marker_set` + the id fields).
  **Every Qwen3.5 shard build MUST pass `--marker-set qwen3_5`** (default stays `qwen3`,
  byte-identical to the old behavior; regression-guarded in
  `src/test/data/document_chunk_marker_set_test.py`).
- `--chunk-by`: `line` ONLY for oolong; `document` for everything else
  (matches `convert_docchunk_singletask_v2_local.sbatch`).
  ⚠ **Do NOT pass `--item-regex`.** The converter default is the *escaped* `r"\|\|"`, which is what
  you want. A bare `'||'` is a regex alternation of empty branches, so it matches **every** line —
  the instruction/question/header get wrapped as their own chunks and the blank lines between them
  stay FREE, bridging chunks and mismatching the eval layout (which keeps the preamble FREE). This
  is the oolong leak in `debug/ctc_vllm_validation/CHUNK_LEAK_AUDIT.md` (2019 inter-chunk FREE
  tokens, ~5/example). The converter now rejects any `--item-regex` matching the empty string.
  **Any oolong shard built before 2026-07-26 has this defect and must be rebuilt.**
- CoT is applied at CONVERT time via `--cot-mode` (build_prompt), not at generation — generator
  commands below never carry CoT flags. `lib/data_format.py` `build_prompt` already supports
  every task key in this file (incl. qa/summarization/rerank/qdmatch/xabsence/groups4/...).
- Eval files stay **unified JSONL** (the eval harnesses consume JSONL, not shards); only TRAIN
  is converted to shards.

**Where things live / where to run:**
- Generators run from the corpus-reasoning tree (in-tree copy `src/corpus_reasoning/data/`,
  identical to `/scratch/users/prasann/corpus-reasoning/scripts/data/`). `cd` to
  `/scratch/users/prasann/corpus-reasoning` so relative `data/` + `data/.cache/` defaults work,
  or pass `--output-dir` absolutes.
- **Run on MOONEY** (corrected 2026-07-18, was "horton" — see B4): the 20k joint-uniform train
  files, `wiki100w_article_pool.pkl`, `openalex_compact.jsonl`, and an HF cache
  (`HF_HOME=/data/prasann/hf_cache`) are all **mooney-local `/data/prasann`**; horton has only
  its own HF/pyserini caches. The `wikipedia-dpr-100w` Lucene index is on NFS
  `~/.cache/pyserini` (visible everywhere). Outputs → mooney
  `/data/prasann/ctc_suite_data/<task>` (14T pool, 8.4T free), NOT /scratch (500G quota,
  ~5MB/s).
- LLM-dependent generators (contradiction, redundancy, xabsence pool) take `--base-url` for a
  local vLLM endpoint — budget a GPU + a serving job for those, or Gemini API.

**Head start — these joint-uniform 20k train pools already exist** (⚠ on **MOONEY**
`/data/prasann/single_task_ladders_20k/` — jsteinhardt partition, NOT horton; provenance = the
v2 single-task sbatch): `contra …realistic_n50-950_k3`, `nq …k20-200_combined_aligned`,
`oolong …ladder_train_combined`, `outlier …wiki100w_contin_n14-220_k3_20000`,
`rerank …msmarco_trainhn_train_k20-315_20000`.
**AUDITED 2026-07-18 (line counts, doc-count histograms, schema spot-checks — see
§Stage-1 build log):** contradiction/oolong/outlier/rerank REUSABLE; **nq FAILED the p10
hard-neg audit (ratio 0.986 — the retired 98%-hard setup) → regenerated fresh.**

**Token estimates:** anchored on the v2/xlong runtime calibrations where they exist, else the
suite README (`chars/4`; synthetic-vocab tasks tokenize 1.5–3× higher). Rung budget = rung −
~300 tok query/answer overhead (2k→~1.7k … 32k→~31.7k). Every task gets a tokenizer-measured
histogram in the Stage-1 audit; ⚠ marks estimates corrected vs. the plan §3 table.

---

## n-ladder summary (docs/items per example at each rung)

| # | Task | tok/doc est | n(2k) | n(4k) | n(8k) | n(16k) | n(32k) | vs plan §3 |
|---|------|-----------|-------|-------|-------|--------|--------|-----------|
| 1 | NQ | ~160 | 11 | 23 | 48 | 100 | 200 | ⚠ was 130; 15→240 → **11→200** |
| 2 | HotpotQA | ~155 | 11 | 24 | 50 | 100 | 205 | ⚠ same correction as NQ |
| 3 | NIAH-contra | ~43 | 40 | 86 | 180 | 365 | 740 | ok |
| 4 | OOLONG | token knob | — | — | — | — | — | n/a (`--len-min/max` in tokens) |
| 5 | HELMET NQA | `--lengths` | 2000 | 4000 | 8000 | 16000 | 32000 | n/a |
| 6 | HELMET GovRep | `--lengths` | 2000 | 4000 | 8000 | 16000 | 32000 | n/a |
| 7 | BEIR SciFact | ~365 | 5 | 10 | 21 | 43 | 88 | ⚠ was ~180; 10→175 → **5→88** |
| 8 | BEIR FiQA | ~400 | 4 | 9 | 19 | 40 | 80 | ⚠ was ~130; 15→240 → **4→80** |
| 9 | MS MARCO | ~100–160 | 13–18 | 25–38 | 50–78 | 100–158 | 200–315 | ⚠ was ~80; 25→400 → **~15→~300** (calibrate: v2 says 160/doc, the k20-315 20k file implies ~100/doc) |
| 10 | MSM rerank | ~100–160 | same as 9 | | | | | ⚠ same |
| 11 | outlier wiki | ~140 | 14 | 28 | 57 | 115 | 220 | ok (matches n14-220 file) |
| 12 | outlier Amazon | ~100 | 20 | 40 | 80 | 160 | 320 | ⚠ was 80/doc 25→400 → **20→320** (matches n30/80/160/325 ladders) |
| 13 | grouping OA | ~180 | 10 | 21 | 43 | 88 | 176 | ⚠ 32k was 160 → **~176** (minor) |
| 14 | grouping_labeled | ~180+labels | 9 | 20 | 42 | 85 | 170 | minor |
| 15 | textgroups | ~150 | 11 | 24 | 50 | 103 | 210 | ok |
| 16 | contradiction | ~42 | 40 | 88 | 190 | 385 | 765 | ok (v2-calibrated exactly) |
| ~~17~~ | ~~redundancy~~ | ~~43~~ | — | — | — | — | — | DROPPED for now (user 2026-07-19); LLM-serving-bound generator, revisit later |
| 18 | absence Gutenberg (text-diff) | ~20/sentence (each sentence = 1 doc/chunk) | ~90 | ~180 | ~360 | ~720 | ~1440 | ⚠ REPLACES absence-PubMed (user 2026-07-19). `generate_absence_data.py --gutenberg`; VersionA=N sents, VersionB=−K, gold=first-four-words. n=sentences, calibrate vs Qwen3.5 tok. Existing eval n10/n50/n200 |
| ~~19~~ | ~~absence official~~ | ~~fixed sets~~ | — | — | — | — | — | DROPPED (user 2026-07-19); published AbsenceBench poetry/numerical/github_prs no longer in suite |
| 20 | strmatch | ~45 (synth ×1.5–3) | 38 | 82 | 170 | 350 | 700 | ⚠ was 30→500 → **~38→~700**; calibrate |
| 21a | qdmatch NQ | ~175/(q+doc) unit | q9 | q20 | q42 | q87 | q178 | ⚠ was q8→120 → **q9→178** (q100 ≈ 17K measured) |
| 21b | qdmatch HPQA | ~175 | q9 | q20 | q42 | q87 | q178 | ⚠ same |
| 21c | ObliQ retrieval (standalone) | ~430/doc (long text) | 4 | 9 | 18 | 37 | 74 | ⚠ REPLACES qdmatch-ObliQ (user 2026-07-19: drop qdmatch variant of ObliQ). `generate_obliq_data.py` `--task retrieval`, pick gold by 1-idx ID; doc-count needs final calibration |
| 22 | xabsence | ~95/pair | P18 | P39 | P81 | P165 | P333 | ⚠ was P4→60 → **P18→333** (P48 ≈ 4.5k measured) |
| 23 | mathmatch | ~35 | 48 | 105 | 220 | 450 | 900 | ~ok (plan 60→1000) |
| 24 | reorder Gutenberg | ~135 | 12 | 27 | 57 | 116 | 234 | ok |
| 25 | cycle | ~25–30 (synth) | 60 | 130 | 270 | 550 | 1100 | ⚠ was 45→750 → **~60→~1100**; consider capping at n1000 |
| 26 | groups4 | ~15–20 (synth) | 100 | 210 | 440 | 900 | 1800 | ⚠ was 60→1000 → n(32k)≈**1800**; recommend capping (task is brutal at n20 already — see §Summary) |

---

## Per-task blocks

Legend: ✔n = continuous per-example n knob exists · "discrete-uniform" = build 5 fixed-n files,
equal counts, and concatenate · `GEN=src/corpus_reasoning/data` · `OUT=/data/prasann/ctc_suite`.

### 1. NQ retrieval — **P0** — `--task retrieval` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_nq_training_data.py` (✔n: `--num-docs-min/--num-docs-max`;
  has `--num-train/--num-eval`, sharding, CE filter).
- **Pool:** HF `nq_open` (~79k train q) + pyserini `wikipedia-dpr` Lucene index (horton HF/pyserini
  cache — verify on horton; radagast can't see `/data`). Existing 20k file:
  `/data/prasann/single_task_ladders_20k/nq/nq_train_k20-200_combined_aligned.jsonl`.
- **20k feasible:** YES (79k queries, no reuse needed). ⚠ DIRECTIVE: p10 hard-negs + CE filter
  only (`--hard-neg-frac 0.1 --ce-filter`); audit hard-ratio ≈ 0.10. CE filter costs yield
  (~20–30%) — request 26k to land 20k.
- **TRAIN FINAL:**
  `python $GEN/generate_nq_training_data.py --num-train 26000 --num-eval 0 --num-docs-min 11 --num-docs-max 200 --hard-neg-frac 0.1 --ce-filter --output-dir $OUT/raw/nq --seed 42`
  (PILOT: `--num-train 2500`, drop `--ce-filter` if speed matters — flag in provenance.)
- **EVAL (fixed-500):** already the v2 pattern — extend `build_v2_eval_ladders.py` `nq.rungs`
  to `{2k:11, 4k:23, 8k:48}` (canonical `nq_validation_k200_hn20_600` has 200 docs ⊇ all rungs)
  and rebuild. eval_size 500 ✓.
- **ACTION A1:** none beyond the shared rung-relabel; this is the cleanest task.

### 2. HotpotQA — **P0** — `--task retrieval` (or `cot_retrieval`) — chunk-by `document` — short-cot (`cot_retrieval`)
- **Generator:** `$GEN/generate_hotpotqa_data.py` (fixed `--num-docs`; `--split both --num-eval`;
  bridge/comparison filter; `--align` length-matching on by default).
- **Pool:** HF `hotpotqa/hotpot_qa` distractor (~90k train bridge+comparison) + `wikipedia-dpr-100w`
  index for hard negs. EXISTS (HF download; cache on horton).
- **20k feasible:** YES (90k questions).
- **TRAIN FINAL (discrete-uniform):** for n in 11 24 50 100 205:
  `python $GEN/generate_hotpotqa_data.py --split train --num-examples 4000 --num-docs $n --question-type bridge --num-hard-negatives $((n/10)) --output-dir $OUT/raw/hotpotqa --seed 42` ; cat.
  (PILOT: `--num-examples 500` per rung.)
- **EVAL (fixed-500):** build ONE canonical at n=205 (`--split validation --num-examples 500`),
  then shrink-derive 2k/4k/8k via a new `hpqa` config in `build_v2_eval_ladders.py`
  (`gold_field=gold_doc_indices`, plain indices, shrink mode). eval_size 500 ✓.
- **ACTION A2:** add `--num-docs-min/max` to the generator (copy the NQ implementation) so FINAL
  can be one command; discrete-uniform acceptable otherwise. ACTION A2b: v2 config for hpqa.

### 3. NIAH-contradiction — P2 — `--task retrieval` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_niah_contradiction_data.py` — a TRANSFORM of existing
  contradiction unified JSONL (`--from-train/--from-eval`, `--num-docs` override).
- **Pool:** the contradiction files of row 16 (same PubMed substrate). EXISTS.
- **20k feasible:** tied to row 16's pool (transform is 1:1); YES if 16 reaches 20k.
- **TRAIN FINAL (discrete-uniform):** run row 16 first, then per rung n∈{40,86,180,365,740}:
  `python $GEN/generate_niah_contradiction_data.py --from-train $OUT/raw/contradiction/contradiction_train_pubmed_realistic_n40-740_k3.jsonl --num-docs $n --pairs-per-example 1 --output-dir $OUT/raw/niah`.
- **EVAL (fixed-500):** transform the row-16 fixed-eval rung files 1:1 (the transform preserves
  example identity) — no new machinery. eval_size = 488 ⚠ (inherits contradiction's held-out set).
- **ACTION A3:** `--num-docs` is a single override — no per-example randomization; discrete-uniform.

### 4. OOLONG — **P1** — `--task oolong` — chunk-by **`line`** (no `--item-regex`; default `r"\|\|"`) — short-cot (`plan`)
- **Generator:** `$GEN/generate_oolong_ladder_data.py` (✔n — in TOKENS: `--len-min/--len-max`,
  target sampled per example; `--num-examples`, sharding, tokenizer-budgeted).
- **Pool:** HF `oolongbench/oolong-synth`. EXISTS (already downloaded; ladder files built).
  Existing 20k: `/data/prasann/single_task_ladders_20k/oolong/raw/oolong_ladder_train_combined.jsonl`
  (built at len 300–38000 — already ⊇ 2k–32k; reusable as-is after histogram audit).
- **20k feasible:** YES (`--num-examples 20000` is the default; items resampled combinatorially).
- **TRAIN FINAL:**
  `python $GEN/generate_oolong_ladder_data.py --num-examples 20000 --len-min 1700 --len-max 31700 --tokenizer <Qwen3.5> --output-dir $OUT/raw/oolong --seed 42` — or reuse the existing combined file.
- **EVAL (fixed-500):** ⚠ generator is TRAIN-only (no `--num-eval`/split flag; output name
  hardcodes `train`) and fixed-across-rungs needs NESTED item sets (same query + gold aggregate
  over a growing item window — the gold ANSWER changes when items are added, so naive expansion
  breaks gold). **ACTION A4:** add an eval mode that, per example, samples the 32k item sequence
  once, computes per-rung prefixes AND recomputes the gold per rung (gold is deterministic from
  items — no LLM), emitting rung-aligned files with a shared example id. Until then: per-rung
  independent eval files (flag as resampling noise) or the fixed `oolong_validation_synth_ctx{2048,4096,8192}`
  buckets ⚠ eval_size=100/bucket.
- Metric is partial-credit (plan R5: not for headline gap claims).

### 5. HELMET NarrativeQA — P2 — `--task qa` — chunk-by `document` (single doc) — no-cot
- **Generator:** `$GEN/generate_helmet_qa_data.py` (`--source narrativeqa --lengths` = native
  token-length control, head-kept truncation ~4 chars/tok; `--num-examples`, `--split`).
- **Pool:** HF `deepmind/narrativeqa`. EXISTS (downloads; train split ~32k QA pairs).
- **20k feasible:** YES for train (~32k pairs; one story serves several questions — acceptable).
- **TRAIN FINAL:**
  `python $GEN/generate_helmet_qa_data.py --source narrativeqa --split train --num-examples 4000 --lengths 2000,4000,8000,16000,32000 --output-dir $OUT/raw/helmet_qa --seed 42`
  (writes one file per length = discrete-uniform by construction; 4000 × 5 = 20k).
- **EVAL (fixed-500):** `--split validation --num-examples 500 --lengths 2000,4000,8000` with the
  SAME seed — same (story, question) rows per length, truncation is nested-prefix by construction
  → fixed-across-rungs for free. ⚠ Caveat: at 2k the gold answer may lie beyond the truncation
  (HELMET-inherited; note in results). eval_size 500 ✓ (val pool ~3.5k pairs).
- **ACTION A5:** verify the generator iterates examples in a seed-stable order so the 5 length
  files are row-aligned (needed for the fixed-eval claim); if not, key by (story,question) id.

### 6. HELMET GovReport — P2 — `--task summarization` — chunk-by `document` — no-cot (long-form out)
- **Generator:** `$GEN/generate_helmet_summ_data.py` (`--source govreport --lengths`,
  `--num-examples`, `--split`). Same mechanics as row 5.
- **Pool:** HF `ccdv/govreport-summarization`. EXISTS (train ~17.5k).
- **20k feasible:** ⚠ train split ≈ 17.5k < 20k → build all 17.5k and flag (or top up with
  `multilexsum`; keep single-source for cleanliness — flag 17.5k).
- **TRAIN FINAL:** `--split train --num-examples 3500 --lengths 2000,4000,8000,16000,32000` (=17.5k).
- **EVAL (fixed-500):** `--split validation --num-examples 500 --lengths 2000,4000,8000`, same
  seed ⇒ nested truncation, fixed rows (same A5 row-alignment check). eval_size 500 ✓ (val 973).
- ROUGE metric — plan R5 flag. Lowest priority in P2.

### 7. BEIR SciFact — P2 — `--task retrieval` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_beir_ce_data.py` (`--dataset scifact`; `--num-docs` LIST — one
  file per k = discrete-uniform native; `--hard-frac 0.1`, CE-cleaned).
- **Pool:** BEIR scifact via `load_beir` + BM25 index cache `data/.cache/beir/_indices/scifact`
  (EXISTS on /scratch — `beir_scifact_*` + `ladder_k11..k88` files present). Corpus 5,183 docs.
- **20k feasible:** **NO** — only ~1,100 queries total (300 test + ~800 train). Max ≈ 800 train
  queries × k-rungs = heavy query reuse to fake 20k. **Realistic max ≈ 4k (800 q × 5 rungs,
  each query at every rung).** FLAGGED.
- **TRAIN (what's achievable):**
  `python $GEN/generate_beir_ce_data.py --dataset scifact --split train --num-docs 5 10 21 43 88 --hard-frac 0.1 --output-dir $OUT/raw/scifact --seed 42`
  ⚠ generator has NO train/eval split flag for the test set and NO --num-train — sizes = query
  count (ACTION A7: verify `--split train` is supported by `load_beir` for scifact; the existing
  `beir_scifact_train_k20_809.jsonl` says yes).
- **EVAL (fixed): ⚠ eval_size 299 max** (entire test set; < 500, quote with ±SE ≈ 0.026 at
  f1≈0.7). Fixed-across-rungs: build k88 once, shrink-derive k5/k10/k21 via a new v2 config
  (`extra_index_fields=[hard_neg_indices]`). ACTION A7b.

### 8. BEIR FiQA — **P1** (the BEIR representative) — `--task retrieval` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_beir_ce_data.py --dataset fiqa` (same as row 7).
- **Pool:** BEIR fiqa + index cache (EXISTS — `beir_fiqa_ce_*` files present). Corpus 57k docs,
  648 test queries, ~5,500 train queries.
- **20k feasible:** ⚠ PARTIAL — ~5,500 train queries; 20k ⇒ each query ~4× (once per rung,
  4 of 5 rungs). Acceptable reuse (different distractor pools per rung) but FLAG. Realistic
  no-reuse max ≈ 5.5k; with per-rung reuse ≈ 27k.
- **TRAIN FINAL:** `--dataset fiqa --split train --num-docs 4 9 19 40 80 --hard-frac 0.1 --output-dir $OUT/raw/fiqa --seed 42` (5 files ≈ 5.5k q each → subsample to 4k each = 20k).
- **EVAL (fixed-500):** test 648 ✓ — build k80 canonical (500 rows), shrink-derive k4/k9/k19.
  ACTION A8: v2 config for fiqa (same shape as scifact).

### 9. MS MARCO retrieval — P2 — `--task retrieval` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_msmarco_trainhn_data.py` (✔n: `--num-docs-min/--num-docs-max`;
  `--n-train/--n-eval`; CE-margin cleaned, `--hard-frac 0.1` default, score-all).
- **Pool:** pyserini `msmarco-v1-passage` prebuilt index + HF BeIR/msmarco qrels (~500k train
  queries). EXISTS on horton (`/data/prasann` caches; 20k rerank-source file already built).
- **20k feasible:** YES (500k queries).
- **TRAIN FINAL:**
  `python $GEN/generate_msmarco_trainhn_data.py --num-docs-min 15 --num-docs-max 300 --n-train 20000 --n-eval 0 --output-dir $OUT/raw/msmarco --seed 42`
  ⚠ tok/doc calibration first (100 vs 160/doc — see summary): set max = n(32k) from the audit.
- **EVAL (fixed-500):** v2 already has the `rerank` config on this data (`msmarco_trainhn_eval_k100_500`,
  shrink) — clone it as `msmarco` with rungs {2k,4k,8k}. 32k rung needs a k≈300 CE-scored eval
  pool: **ACTION A9: regenerate eval with `--num-docs 300`** (the "no CE pool >k100" limit in the
  v2 comments is stale — `msmarco_trainhn_*_k500` files exist with score-all CE; verify
  `ce_scores` coverage on them).

### 10. MS MARCO rerank — **P1** — `--task rerank` — chunk-by `document` — no-cot
- **Generator:** same build as row 9 (`generate_msmarco_trainhn_data.py`) — the rerank task
  reads the SAME CE-scored unified files (`--task rerank` at convert;
  `generate_msmarco_helmet_rerank_data.py` only reformats to HELMET-native, not needed for our
  in-tree path). Existing 20k: `/data/prasann/single_task_ladders_20k/rerank/ce_gen/msmarco_trainhn_train_k20-315_20000.jsonl` ✓.
- **20k feasible:** YES (already built).
- **TRAIN FINAL:** reuse the existing k20-315 file (histogram audit → maps to ~2k–32k) — zero
  new compute. PILOT: `--limit`-style subset at convert time (`--limit 2500`).
- **EVAL (fixed-500):** v2 `rerank` config exists — re-rung to {2k,4k,8k} (+16k/32k deferred;
  32k needs the k≈300 pool of A9). eval_size 500 ✓. Metric MRR@10 from `ce_scores`.

### 11. Outlier wiki scale-k — **P0** — `--task outlier` — chunk-by `document` — short-cot (`label`/`template`)
- **Generator:** `$GEN/generate_wiki_outlier_data.py` (✔n: `--min-docs/--max-docs`; mixed-K
  mode auto-grows majority-article count with n = the paper's **scale-k**; `--eval-frac`).
- **Pool:** ⛔ **`data/wiki100w_article_pool.pkl` MISSING on /scratch** (only
  `wiki100w_article_pool_smoke.pkl`). It is rebuilt automatically from the wikipedia-dpr-100w
  Lucene index (`--pool-cache`, `--pool-shards`) — index lives in the horton pyserini cache.
  **BLOCKER B1 until verified on horton** (pool scan is hours-scale). The existing 20k train
  file (`outlier_wiki100w_contin_n14-220_k3_20000.jsonl` on `/data`) proves the pool existed
  there — likely only the /scratch copy is absent.
- **20k feasible:** YES (already done once: n14-220 × 20000).
- **TRAIN FINAL:** reuse existing n14-220 file (audit: n14-220 ≈ exactly 2k→32k at 140/doc) or
  `python $GEN/generate_wiki_outlier_data.py --num-examples 20000 --min-docs 14 --max-docs 220 --num-outliers 3 --eval-frac 0 --pool-cache /data/prasann/ctc_suite/wiki100w_article_pool.pkl --out-dir $OUT/raw/outlier --seed 42`.
- **EVAL (fixed-500):** v2 `outlier` config exists (canonical n220, shrink, answers recomputed
  from gold) — re-rung to {2k:14, 4k:28, 8k:57}. eval_size 500 ✓ (600-row canonical).
- Fixed-M control (repro-gate T3 only): one extra pair at fixed `--mixed-min-k=--mixed-max-k`,
  same commands with `--simple-ratio 0` — note, not part of the main grid.

### 12. Outlier Amazon — **P1** — `--task outlier` — chunk-by `document` — short-cot
- **Generator:** `$GEN/generate_review_outlier_data.py` (fixed `--num-docs` → discrete-uniform;
  no train/eval flag — single `--out` file, split downstream by row slice; per-doc char bounds
  `--min/max-text-chars`).
- **Pool:** HF `McAuley-Lab/Amazon-Reviews-2023` (huge). EXISTS (downloads; horton cache).
- **20k feasible:** YES (reviews effectively unbounded; `--pool-size` per category).
- **TRAIN FINAL (discrete-uniform):** for n in 20 40 80 160 320:
  `python $GEN/generate_review_outlier_data.py --num-examples 4000 --num-docs $n --num-outliers 4 --seed 42 --out $OUT/raw/outlier_amzn/review_outlier_n${n}_train.jsonl` ; cat.
- **EVAL (fixed-500):** build n320 canonical (fresh `--seed 7 --num-examples 500`), shrink-derive
  20/40/80 via a new v2 config (answers = outlier indices, recompute like wiki-outlier).
  ACTION A12: v2 config; ACTION A12b: add `--min/max-docs` + `--num-eval` knobs (nice-to-have).
- ⚠ `--use-titles` OFF at convert (title=category leak — the known wrapping-leak bug family).

### 13. Grouping OpenAlex — **P0** — `--task grouping` — chunk-by `document` — short-cot
- **Generator:** `$GEN/generate_arxiv_grouping_data.py` (fixed `--docs-per-example` →
  discrete-uniform; `--num-train/--num-eval`; temporal eval split `--eval-year-min`).
- **Pool:** ⛔ **BLOCKER B2: `data/openalex_compact.jsonl` MISSING** — only
  `openalex_compact_tiny.jsonl` exists on /scratch. Must either `--preprocess` from an OpenAlex
  snapshot dir (`data/openalex/works/*.gz` — also not present) or `--api-fetch --api-email …`
  (~2000/field, slow but CPU-only). CHECK horton `/data` first; else API fetch is the unblock
  (budget ~hours, polite pool).
- **20k feasible:** ⚠ DEPENDS on pool size. api-fetch at defaults ≈ 2000 works × ~26 fields ≈
  50k abstracts; 20k examples × avg ~90 docs ≈ 1.8M doc-slots ⇒ each abstract reused ~36×
  (different groupings each time — acceptable for grouping, but FLAG; raise `--api-per-field`
  to 5000+ to cut reuse to ~14×). Realistic: 20k with heavy substrate reuse, or ~5k clean.
- **TRAIN FINAL (discrete-uniform):** for n in 10 21 43 88 176:
  `python $GEN/generate_arxiv_grouping_data.py --compact-in $OUT/raw/openalex_compact.jsonl --num-train 4000 --num-eval 0 --docs-per-example $n --out-dir $OUT/raw/grouping --seed 0` ; cat.
- **EVAL (fixed-500):** grouping gold = the partition of the example's docs — shrink-derive is
  NOT gold-preserving (removing a doc changes the partition metric baseline). **ACTION A13:**
  nested-eval mode in the generator: sample each eval example's doc set at n(32k) with a
  per-group nesting order, emit rung views keeping whole groups' prefix subsets + recomputed
  gold. Until then per-rung independent files (flag). eval_size 500 ✓ (`--num-eval 500`,
  temporal split).
- Convert with `--task grouping` (unlabeled). ARI/pairwise-F1.

### 14. Grouping labeled — P2 — `--task grouping_labeled` — chunk-by `document` — short-cot
- **Same generator + SAME raw JSONL as row 13** — the labeled/unlabeled distinction is the
  convert-time task key (`--task grouping_labeled` renders cluster labels; precedent:
  `build_combined_unified.py:52`). Zero extra generation compute; inherits B2 and A13.
- Commands: identical to row 13; convert twice with the two task keys.

### 15. textgroups — P2 — `--task textgroups` — chunk-by `document` — short-cot
- **Generator:** `$GEN/generate_textgroups_data.py` (synthetic, no external pool; fixed
  `--num-docs` → discrete-uniform; `--num-train/--num-eval`).
- **Pool:** none (synthetic). ✓
- **20k feasible:** YES (unbounded).
- **TRAIN FINAL (discrete-uniform):** for n in 11 24 50 103 210:
  `python $GEN/generate_textgroups_data.py --num-docs $n --num-train 4000 --num-eval 0 --num-groups 2 --group-size 3 --target 70 --output-dir $OUT/raw/textgroups --seed 42` ; cat.
- **EVAL (fixed-500):** shrink-safe (gold = the K exact-target triples; removing filler passages
  can't create a new triple at the target — the generator's `--separation` guarantee is
  subset-stable). Build n210 canonical (`--num-eval 500 --seed 7`), shrink via v2 config.
  ACTION A15: v2 config (gold triples = index lists).

### 16. Contradiction PubMed — **P0** (anchor; repro also 0.8B) — `--task contradiction` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_pubmed_contradiction_data.py` (✔n: `--num-docs-min/max`;
  `--num-train/--num-eval`; LLM for fresh gold pairs; **`--expand-from-train/eval`** = gold-reuse
  + filler expansion, no LLM).
- **Pool:** HF PubMedQA `pqa_artificial` (~170k claim sentences at `--pool-abstracts 20000`).
  EXISTS. Existing 20k: `/data/prasann/single_task_ladders_20k/contradiction/gen_full/contradiction_train_pubmed_realistic_n50-950_k3.jsonl` ✓ (n50-950 — WIDER than 2k–32k; audit +
  either filter to n≤765 or accept the tail as free 32k+ coverage).
- **20k feasible:** YES-with-caveat — 20k × k3 = 60k gold pairs. Fresh LLM generation of 60k
  validated pairs is the single most expensive build (~$ + days at 25-concurrent Gemini).
  The existing 20k file already paid this. If regenerating: generate ~5k fresh examples' worth
  of pairs and `--expand-from-train` to 20k (gold reuse across n-variants — flag in provenance).
- **TRAIN FINAL:** reuse existing file (preferred), else
  `python $GEN/generate_pubmed_contradiction_data.py --num-train 20000 --num-eval 0 --num-docs-min 40 --num-docs-max 765 --num-contradictions 3 --mode realistic --model gemini-2.5-flash --output-dir $OUT/raw/contradiction --seed 42`.
- **EVAL (fixed-488 ⚠):** v2 `contra` config exists (expand mode + harvested fillers; 1-indexed
  gold handled) — re-rung to {2k:40, 4k:88, 8k:190}. eval_size 488 (entire held-out; accepted).

### 17. Redundancy — **P1** — `--task redundancy` — chunk-by `document` — short-cot (`template`; `enumerate`=long)
- **Generator:** `$GEN/generate_redundancy_data.py` (fixed `--num-docs` → discrete-uniform;
  `--num-train/--num-eval`; LLM paraphrase gen — `--base-url` vLLM Qwen2.5-14B).
- **Pool:** same PubMedQA pool as row 16 (imports `load_pubmed_pool`). EXISTS.
- **20k feasible:** ⚠ LLM-bound like row 16: 20k × k3 = 60k validated paraphrase pairs. No
  expand mode exists. **Realistic: ~5k fresh; ACTION A17: port `--expand-from-*` from the
  contradiction generator** (mechanically identical: gold pairs + neutral fillers) → then 20k.
- **TRAIN (discrete-uniform), pilot-first:** for n in 40 85 180 365 735:
  `python $GEN/generate_redundancy_data.py --num-docs $n --num-redundancies 3 --hardneg-pairs 6 --num-train 1000 --num-eval 0 --base-url http://localhost:8000/v1 --output-dir $OUT/raw/redundancy --seed 42` (=5k); scale to 4000/rung post-A17.
- **EVAL (fixed-500):** expand-mode v2 config (gold pairs fixed, PubMed fillers — same recipe as
  contra, incl. its filler harvest). ACTION A17b: v2 config (`gold_is_pairs=True`). Build
  canonical 500 at n40 + expand, or at n735 + shrink (shrink simpler: fillers are gold-free).

### 18. Absence Gutenberg (text-diff) — **P1** — `--task absence` — chunk-by `document` — no-cot
- **REPLACES absence-PubMed** (user 2026-07-19): natural-prose text-diff variant, preferred because
  each sentence is its own document/chunk (real chunk structure for the full-vs-chunked contrast)
  and it shares NO substrate with contradiction (removes the PubMed-overlap confound).
- **Generator:** `$GEN/generate_absence_data.py --gutenberg`. A random window of N consecutive
  sentences from a Gutenberg book = Version A; Version B = same window with K sentences removed;
  target = first-four-words of each removed sentence (uniqueness-filtered → unambiguous gold).
  Emits `documents=[{text: sentence} × N]`, `gold_doc_indices=removed positions`, `answers=first-4-words`.
  LLM-free. `--n-sents N --k-remove K --num-train/--num-eval --min-sentence-words 4`.
- **Pool:** HF `sedthh/gutenberg_english` (same book substrate as reorder row 24; cache exists).
  Existing eval sets already on /scratch: `absence_eval_gutenberg_n{10,50,200}_k3.jsonl`; gen job
  `gen_absence_gutenberg.sh`.
- **n = sentences (~20 tok each)** → ladder ≈ {90,180,360,720,1440} for 2k–32k; CALIBRATE against
  the Qwen3.5 tokenizer before freezing (Version A + Version B both in context ≈ 2× factor).
- **20k feasible:** YES (2000-book scan × examples-per-book; resample windows).
- **EVAL (fixed-500):** gold = removed sentence positions/first-four-words. Hold the removed set +
  kept core fixed; grow rungs by adding more PRESENT sentences (expand) or build the largest rung
  and shrink non-gold present sentences (gold-safe). Bump `--num-eval` to ≥500 (default 300).
  ACTION A18: v2 ladder config (gold = removed ids; answers are first-four-words strings →
  answers-from-gold recompute hook, precedent in the outlier config).

### 19. ~~Absence official~~ — **DROPPED** (user 2026-07-19)
- Published `harveyfin/AbsenceBench` (poetry/numerical/github_prs) removed from the suite — it has
  no n knob (fixed sets, cannot ladder) and external-benchmark comparability isn't needed for the
  full-vs-chunked scaling study. One-line revert if wanted back as a fixed transfer eval:
  `python $GEN/generate_absence_data.py --official --output-dir $OUT/eval/absence_official --seed 42`.

### 20. strmatch — **P1** — `--task strmatch` — chunk-by `document` — no-cot (`enumerate`=long option)
- **Generator:** `$GEN/generate_strmatch_data.py` (fixed `--num-docs` → discrete-uniform;
  `--num-train/--num-eval`; wiki-vocab pool `--pool-passages`). LLM-free, CPU-fast.
- **Pool:** wiki passages for vocab (index) or `/usr/share/dict/words`. EXISTS.
- **20k feasible:** YES (synthetic).
- **TRAIN FINAL (discrete-uniform):** for n in 38 82 170 350 700:
  `python $GEN/generate_strmatch_data.py --num-docs $n --num-pairs 3 --span-len 3 --str-len 10 --num-train 4000 --num-eval 0 --output-dir $OUT/raw/strmatch --seed 42` ; cat.
- **EVAL (fixed-500):** shrink-safe (removing strings cannot create a shared run; gold pairs
  preserved). Build n700 canonical (`--num-eval 500 --seed 7`), shrink via v2 config
  (`gold_is_pairs=True`). ACTION A20: v2 config.
- ⚠ Synthetic-vocab ×1.5–3 tokenization — calibrate before freezing n values.

### 21a. qdmatch NQ — **P1** — `--task qdmatch` — chunk-by `document` — no-cot
> **ROSTER CHANGE (user 2026-07-19):** qdmatch-**ObliQ** is DROPPED from the suite. Keep qdmatch
> NQ (21a) + HPQA (21b). ObliQ instead enters as a **standalone in-context-retrieval** row (21c,
> see below) via `generate_obliq_data.py`, NOT `generate_qdmatch_data.py`. Do not tokenize/train
> the `qdmatch_*obliq*` pilot jsonl — discard it.

### 21b. qdmatch HPQA — P2
- **Generator:** `$GEN/generate_qdmatch_data.py` (derives from single-query unified JSONL —
  `--from-train/--from-eval`; fixed `--num-docs`/`--num-queries` → discrete-uniform;
  `--num-relevant 3`; `--layout separate`).
- **Pools (source files):** NQ `nq_train_k20_hn19_2500_aligned.jsonl` + validation files ✓;
  HPQA `hotpotqa_*_bridge_unified_*` ✓; ObliQ `obliq_*_train/test_*` ✓ (all on /scratch data/).
  Source query pools: NQ ~2.5k, HPQA ~2k, ObliQ ~850.
- **20k feasible:** ⚠ COMBINATORIAL REUSE — a q178 example consumes 178 queries; 20k examples
  ⇒ each source query appears in ~1.4k examples (NQ). Combinations differ but gold q→doc pairs
  repeat massively. FLAG: acceptable for a matching task (the pairing is what varies), but
  record reuse factor; ObliQ (850 q) is the worst — cap ObliQ at ~5k train.
- **TRAIN FINAL (discrete-uniform), NQ:** for q in 9 20 42 87 178:
  `python $GEN/generate_qdmatch_data.py --from-train data/nq_train_k20_hn19_2500_aligned.jsonl --num-docs $q --num-queries $q --num-relevant 3 --layout separate --num-train 4000 --num-eval 0 --src-tag nq --output-dir $OUT/raw/qdmatch_nq --seed 42` ; cat.
  HPQA: same with the bridge file, q ∈ same set. ObliQ: q ∈ {4,8,17,36,73}, 1000/rung.
- **EVAL (fixed-500):** hold the k relevant queries + their gold docs fixed; grow by adding
  irrelevant (query, doc) units. Shrink from q_max is gold-safe IF the relevant queries/docs are
  never dropped. **ACTION A21: v2 config with TWO index namespaces (query list + doc list) —
  the current builder only remaps doc indices; qdmatch gold is (q_idx, d_idx) pairs.** Moderate
  (one new remap function). eval_size: NQ/HPQA 500 ✓; ObliQ ⚠ ~300 max from 850-query pool
  (flag inline).

### 21c. ObliQ retrieval (standalone) — **P2** — `--task retrieval` — chunk-by `document` — no-cot
- **REPLACES qdmatch-ObliQ** (user 2026-07-19). This is OBLIQ-Bench in-context retrieval, NOT
  query-doc matching: ~N docs stuffed in the prompt, model picks the gold by 1-indexed ID (same
  `--task retrieval` path as NQ/HPQA/BEIR).
- **Generator:** `$GEN/generate_obliq_data.py` (BM25-mines distractors from the ObliQ subset's own
  corpus, forces all qrels golds, `--num-docs`, `--num-train-perms`, `--train-frac`; combine
  subsets via `mix_obliq_subsets.py` / `expand_obliq_train.py`). Env: `corpus-reasoning-eval`
  (needs pyserini).
- **Pool:** HF `dianetc/OBLIQ-Bench`, 5 subsets (writing/congress/twitter/wildchat + math).
  Existing standalone files already on /scratch (`obliq_<subset>_train/test_*`, plus combined
  `obliq_mix4_train_1797` / `obliq_mix4_test_488`) — reusable substrate; the 488-example mix4 test
  is a clean fixed eval set.
- **Doc length ~430 tok/doc** (long social-media/writing text) → ladder n ≈ {4,9,18,37,74}
  (calibrate against Qwen3.5 tokenizer before freezing, same as row-11/20 procedure).
- **eval_size:** mix4 test = 488 ✓ (accept as-is, same basis as contradiction). Small subsets
  cap lower — prefer the combined mix.

### 22. xabsence — **P1** — `--task xabsence` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_xabsence_data.py` (two-phase: `--build-pool` LLM paraphrases →
  assemble with `--pool`; fixed `--num-pairs` → discrete-uniform; `--num-train/--num-eval`).
- **Pool:** ⛔ **BLOCKER B3: `xabsence_pool_pubmed.jsonl` = 659 pairs only.** P333 examples
  would use half the pool PER EXAMPLE; 20k train is out of the question. Rebuild:
  `python $GEN/generate_xabsence_data.py --build-pool --from <pubmed unified> --pool-size 50000 --base-url http://localhost:8000/v1 --model Qwen2.5-14B-Instruct --pool-out $OUT/raw/xabsence_pool_pubmed_50k.jsonl`
  (LLM paraphrase per pair — GPU/vLLM, ~hours at 64-concurrent; the second LLM-cost item).
- **20k feasible:** only after B3 (50k-pair pool ⇒ ~10–20× pair reuse at avg P~170 — flag).
  Realistic near-term: pilot at small P from the 659-pool; FINAL blocked on rebuild.
- **TRAIN FINAL (discrete-uniform):** for P in 18 39 81 165 333:
  `python $GEN/generate_xabsence_data.py --pool $OUT/raw/xabsence_pool_pubmed_50k.jsonl --num-pairs $P --num-unmatched 3 --num-train 4000 --num-eval 0 --src-tag pubmed --output-dir $OUT/raw/xabsence --seed 42` ; cat.
- **EVAL (fixed-500):** gold = the k unmatched claims — hold unmatched set fixed, add MATCHED
  pairs to grow (expand; matched pairs are gold-free by construction) or shrink from P333.
  ACTION A22: either a `--nested-eval` flag in assemble (easy: assembly is deterministic given
  seed) or a v2 config.

### 23. mathmatch — **P1** — `--task mathmatch` — chunk-by `document` — no-cot
- **Generator:** `$GEN/generate_mathmatch_data.py` (synthetic; fixed `--num-docs` →
  discrete-uniform; NOTE smaller defaults `--num-train 1000/--num-eval 200`).
- **Pool:** none. ✓ · **20k feasible:** YES.
- **TRAIN FINAL (discrete-uniform):** for n in 48 105 220 450 900:
  `python $GEN/generate_mathmatch_data.py --num-docs $n --num-pairs 3 --tolerance 2 --num-train 4000 --num-eval 0 --output-dir $OUT/raw/mathmatch --seed 42` ; cat.
- **EVAL (fixed-500):** shrink-safe (removing expressions can't create a within-tolerance pair;
  generator enforces exactly-K which survives subsetting). Build n900 canonical
  (`--num-eval 500 --seed 7`), shrink via v2 config (pairs). ACTION A23: v2 config.

### 24. Reorder Gutenberg — **P0** — `--task reorder` — chunk-by `document` — short-cot (`successor`)
- **Generator:** `$GEN/generate_reorder_data.py` (fixed `--n-chunks` → discrete-uniform;
  `--num-examples`/`--eval-frac`/`--eval-only`/`--skip-books`; 100w chunks
  `--target-words 100 --out-suffix 100w`).
- **Pool:** HF `sedthh/gutenberg_english` (~48k books). EXISTS (downloads; `download_gutenberg.sh`
  precedent; reorder_gutenberg100w files built).
- **20k feasible:** YES — 20k examples need books with ≥ n×100 words (n234 ⇒ ~24k-word books —
  plentiful); use `--examples-per-book 2` + `--max-books-to-scan 20000` to be safe.
- **TRAIN FINAL (discrete-uniform):** for n in 12 27 57 116 234:
  `python $GEN/generate_reorder_data.py --n-chunks $n --num-examples 4000 --eval-frac 0 --target-words 100 --out-suffix 100w --examples-per-book 2 --max-books-to-scan 20000 --out-dir $OUT/raw/reorder --seed 42` ; cat.
- **EVAL (fixed-500):** the n chunks ARE the gold (permutation) — shrink is NOT gold-preserving
  (dropping chunks changes the target permutation, though a nested version is well-defined:
  same book, same start, first-n contiguous chunks, permutation re-sampled). **ACTION A24:
  nested-eval mode: per example fix (book, start offset, shuffle seed) and emit all rungs**
  (small change in the generator; cannot be done post-hoc in v2 builder because chunk texts are
  the answer substrate). eval_size 500 ✓ (`--eval-only --skip-books` for disjoint books).

### 25. cycle — **P1** — `--task cycle` — chunk-by `document` — short-cot (`template`; `trace`=long)
- **Generator:** `$GEN/generate_cycle_data.py` (synthetic; fixed `--num-docs` →
  discrete-uniform; `--cycle-len`, `--num-cycles`).
- **Pool:** none ✓ · **20k feasible:** YES.
- **TRAIN FINAL (discrete-uniform):** for n in 60 130 270 550 1100:
  `python $GEN/generate_cycle_data.py --num-docs $n --cycle-len 3 --num-cycles 1 --num-train 4000 --num-eval 0 --output-dir $OUT/raw/cycle --seed 42` ; cat.
  ⚠ Calibrate synth tokenization first; consider capping at n1000 (catalog precedent) and
  letting the 32k rung run slightly short.
- **EVAL (fixed-500):** shrink-safe (removing non-cycle edges can't create a cycle; the gold
  loop's docs are kept). Build n_max canonical, shrink via v2 config (gold = loop doc indices).
  ACTION A25: v2 config.

### 26. groups4 — **P1** — `--task groups4` — chunk-by `document` — short-cot (`template`; `sort`=long)
- **Generator:** `$GEN/generate_groups4_data.py` (synthetic; fixed `--num-docs` →
  discrete-uniform; `--group-size 4 --tolerance 5`).
- **Pool:** none ✓ · **20k feasible:** YES.
- ⚠ **Scale-prior warning:** F1 0.14 at n=20 even with sort-CoT (hardest task in the suite) —
  per §11 Q3 (new tasks must pass the short-context full-attention bar first), build the PILOT
  ONLY until the 2k pilot clears the §5 floor. Do not spend the 20k build yet.
- **TRAIN PILOT (discrete-uniform):** for n in 100 210 440 900 1800 (or capped …900):
  `python $GEN/generate_groups4_data.py --num-docs $n --num-groups 1 --group-size 4 --tolerance 5 --num-train 500 --num-eval 0 --output-dir $OUT/raw/groups4 --seed 42` ; cat.
  FINAL: same with `--num-train 4000` after the gate.
- **EVAL (fixed-500):** shrink-safe (tolerance-cluster uniqueness survives subsetting). v2
  config after canonical n_max build. ACTION A26: v2 config.

---

## Summary

### Blockers (must clear before the corresponding build)
- ~~**B0 (cross-cutting, Stage-0):** converter marker ids hardcoded Qwen3~~ **RESOLVED**:
  `--marker-set qwen3_5` on the converter (see the converter section above); default `qwen3`
  unchanged.
- ~~**B1 (outlier-wiki, P0)**~~ **CLEARED 2026-07-18:** `wiki100w_article_pool.pkl` (6.2 GB)
  exists on **mooney** `/data/prasann/single_task_ladders_20k/wiki100w_article_pool.pkl`.
- ~~**B2 (grouping OpenAlex, P0)**~~ **CLEARED 2026-07-18:** `openalex_compact.jsonl`
  (**52,000 works**, 65 MB) exists on **mooney**
  `/data/prasann/scratch_archive/corpus-reasoning/data/openalex_compact.jsonl`. 20k train
  BUILT from it (see build log). 52k works ⇒ the reuse math in row 13 (~36×) stands.
- **B3 (xabsence, P1):** paraphrase pool is 659 pairs (need ~50k for P≤333 × 20k). LLM rebuild
  (vLLM Qwen2.5-14B, GPU-hours).
- **B4 (environment) — CORRECTED 2026-07-18:** the 20k train pools + wiki pool pkl + openalex
  compact live on **mooney** `/data/prasann` (NOT horton; horton `/data/prasann` has only
  HF/pyserini caches and no `single_task_ladders_20k`). Mooney also has
  `hf_cache` (nq_open, pubmedqa, oolong, msmarco) + java 21. The `wikipedia-dpr-100w` Lucene
  index is in `~/.cache/pyserini/indexes` (NFS home — visible from ALL nodes). Slurm:
  `srun -p jsteinhardt --qos preemptive_high -w mooney`. Generation env:
  `corpus-reasoning-eval` (has pyserini 1.5 + datasets 4.8; `corpus-reasoning-olmo` has NO
  pyserini — don't use it for retrieval-pool generators).

### 20k-train feasibility flags (achievable counts)
| Task | Status |
|---|---|
| SciFact | **~4k max** (800 train queries × 5 rungs; 20k impossible without gross reuse) |
| GovReport | 17.5k (source-capped; build all, flag) |
| FiQA | 20k with ~4× query reuse (5.5k clean) — flagged |
| grouping OA | 20k with ~14–36× abstract reuse (raise api-per-field); ~5k clean |
| contradiction | 20k EXISTS (else 60k LLM pairs ≈ the priciest rebuild; expand-mode reuse path) |
| redundancy | ~5k until A17 (expand-mode port); LLM-bound |
| xabsence | blocked on B3 |
| qdmatch NQ/HPQA | 20k with heavy combinatorial query reuse (flag); ObliQ cap ~5k |
| groups4 | hold at pilot until §5 gate passes |
| all others | 20k clean ✓ |

### ACTION items (generator/tooling gaps)
- ~~**A0** parameterize converter marker/eos ids for Qwen3.5 (B0)~~ DONE (`--marker-set`).
- **Shared:** re-derive `build_v2_eval_ladders.py` rung sets to 2k/4k/8k(/16k/32k) for ALL
  tasks incl. the 4 existing configs; add `CAP_PER_RUNG` canonical builds per new task.
- **New v2 shrink configs** (config-only, shrink is gold-safe): hpqa (A2b), scifact (A7b),
  fiqa (A8), msmarco (A9 + regen k≈300 CE eval pool; verify k500 `ce_scores` coverage),
  outlier-amazon (A12), textgroups (A15), redundancy (A17b), absence-pubmed (A18b, needs
  answers-from-gold hook), strmatch (A20), mathmatch (A23), cycle (A25), groups4 (A26).
- **Generator-side nested-eval modes** (post-hoc shrink insufficient): oolong (A4 — rung
  prefixes + gold recompute; also needs an eval/`--num-eval` flag at all), grouping (A13 —
  partition gold), reorder (A24 — same book/offset/seed across rungs), qdmatch (A21 — two-index
  remap), xabsence (A22 — or expand with matched pairs).
- **A2** optional `--num-docs-min/max` for hotpotqa (else discrete-uniform, fine).
- **A5** verify HELMET generators are row-aligned across `--lengths` (fixed-eval relies on it).
- **A17** port `--expand-from-*` gold-reuse to redundancy (unlocks 20k).
- **A18** confirm/add item-count control in absence `--from` mode.
- **Calibration pass (Stage-1 audit):** tokenizer-measured tok/doc for every task with the
  Qwen3.5 tokenizer (xlong-style runtime calibration), esp. MS MARCO (100 vs 160), the
  synthetic-vocab tasks (strmatch/cycle/groups4, ×1.5–3), and absence's 2× A+B factor.

### Eval-size flags (sub-500, quote inline with SE)
contradiction 488 (accepted) · NIAH-contra 488 (inherited) · SciFact **299** ·
qdmatch-ObliQ ~300 · oolong fixed buckets 100/rung until A4 · absence-official 1184/1200/751 ✓
(no ladder).

### Compute estimate (build only, no training)
- **CPU-only, cheap (< 1 day total on horton):** all synthetic tasks (strmatch, mathmatch,
  cycle, groups4, textgroups), reorder, absence (all modes), qdmatch, NIAH transform, oolong
  ladder, HELMET ports, v2 eval-ladder derivations, all conversions/tokenization.
- **CPU + Java/BM25 (hours each):** NQ, HotpotQA, BEIR, MS MARCO, wiki-outlier pool scan (B1),
  OpenAlex api-fetch (B2).
- **GPU (single H200, hours each):** CE scoring (NQ `--ce-filter` 26k, BEIR-CE, MS MARCO
  trainhn if regenerated) — small.
- **LLM-serving (the real cost):** xabsence pool rebuild (B3) and any fresh
  contradiction/redundancy gold (avoidable via existing 20k file + A17 expand-mode).
  With reuse paths taken: **total build ≈ 1–2 days wall on horton + one vLLM GPU for B3; no
  Beaker needed.**
- Not CPU-only overall, but GPU needs are incidental (CE/vLLM), not training-class.

### Disk estimate
20k train × avg ~17k tok ≈ 340M tok/task ⇒ ~1.4 GB raw JSONL + ~0.7 GB uint32 shards per task
(dense emit doubles nothing; masks negligible). ~26 tasks ⇒ **~55–75 GB total** (+ eval files
< 3 GB, + xabsence/openalex pools < 2 GB). Fits trivially on horton `/data` (12T); do NOT put
on /scratch (500G quota, 89% full pool, ~5MB/s NFS).

---

## Stage-1 build log (2026-07-18, P0 raw generation — unified JSONL only, NO shards yet)

All generation on **mooney** (`-p jsteinhardt --qos preemptive_high`), env
`corpus-reasoning-eval`, `PYTHONPATH=<repo>/src`, `HF_HOME=/data/prasann/hf_cache`.
Outputs → **mooney `/data/prasann/ctc_suite_data/<task>/`**. Logs + sbatch scripts →
`/scratch/users/prasann/ctc_suite_logs/`.

| Task | Status | Where / job |
|---|---|---|
| grouping-OpenAlex (13) | ✅ **DONE** — 5 rungs n∈{10,21,43,88,176} × 4000 = 20k train (`openalex_grouping_n*_levels_train_4000.jsonl`), from the 52k-work compact pool. EVAL deferred (A13 nested mode). | mooney `/data/prasann/ctc_suite_data/grouping/` (job 3337678, ~4 min) |
| HotpotQA (2) | 🔄 LAUNCHED — 5 rungs n∈{11,24,50,100,205} × 4000 train (bridge, hn=n/10, BM25 from NFS `~/.cache/pyserini`) + canonical eval 500 @ n205 (shrink-derive later, A2b). | job **3337734** on **CUBBINS** (see trap note) |
| reorder-Gutenberg (24) | 🔄 LAUNCHED — 5 rungs n∈{12,27,57,116,234} × 4000 train (`--num-examples 4001` because `--eval-frac 0` still forces 1 eval row). EVAL deferred: generator confirmed to have NO nested mode (A24). | job **3337735** on **CUBBINS** |
| NQ (1) | ❌ **STILL MISSING** — job 3337736 (6-shard) killed after 3.5h w/ 0 output; 2026-07-19 fast-retry fixed the real per-example BM25 bottleneck (~5x, see below) but the 48-way shard relaunch (3337963/3337964) thread-thrashed and also produced 0 output. Needs a re-run at safe concurrency (~6-way/node) with thread pinning. | see "NQ fast-regen attempt (2026-07-19)" below |

⚠ These three write to **cubbins** `/data/prasann/ctc_suite_data/` (grouping's output is on
**mooney** — consolidate later).

⚠ **TRAP hit twice tonight (jobs 3337676/77/88, then 3337704/05/06 — all killed):** every
HF-datasets-touching generator process wedged in `nfs_wait_bit_killable` (~0% CPU, 0-byte
logs) on **mooney**. Two layers:
1. The login shell exports `HF_DATASETS_CACHE=/scratch/users/prasann/huggingface-cache` and
   sbatch inherits the submit env → HF filelock convoy on shared NFS cache. Fix (kept in the
   scripts): `export HF_DATASETS_CACHE=/data/prasann/hf_cache` +
   `HF_HUB_CACHE=/data/prasann/hf_cache/hub` (node-local) + per-proc logs on `/data`.
2. After that fix the relaunch STILL wedged: **mooney's NFS client is wedged for the
   `/scratch` conda-env inodes** — `import datasets` from `corpus-reasoning-eval` hangs ≥2 min
   at 0.1 s CPU **on mooney**, while the identical import takes 10 s on the login node and 21 s
   (incl. pyserini) on cubbins. The datasets-free grouping job ran fine before the 12-process
   pile-up, so the wedge likely dates from launch 1. → Jobs moved to **cubbins**. If mooney is
   needed again, test `timeout 60 <env>/bin/python -c "import datasets"` there first (or reboot
   the mount / use a node-local env clone).
Sbatch scripts: `/scratch/users/prasann/ctc_suite_logs/gen_{hotpotqa,reorder,nq_p10,grouping}_p0.sbatch`.

### 20k-pool audit verdicts (mooney `/data/prasann/single_task_ladders_20k/`)

| Pool | Lines | doc-count (min/p50/max) | ~tok (min/p50/max) | Verdict |
|---|---|---|---|---|
| contradiction n50-950 | 20,114 | 50 / 501 / 950 | 1.7k / 20k / 40k | **REUSABLE** — 16,071 inside [40,765], 4,043 above (free 32k+ tail, keep or filter), 0 below; schema ✓ (gold pairs). Low end starts n=50 (~2.1k tok) vs target n(2k)=40 — acceptable. |
| nq k20-200 | 20,004 | 1 / 109 / 200 | 0.2k / 17k / 35k | **❌ NOT REUSABLE** — mean hard-neg ratio **0.986** (built at default `--hard-neg-frac 1.0` = the retired 98%-hard setup; p10 DIRECTIVE requires ~0.10). Also 48 degenerate rows < 20 docs (min 1 doc). → fresh p10+CE regen launched (job 3337688). |
| oolong combined | 21,000 | 1 doc (line-chunked) | 0.3k / 15k / 41k | **REUSABLE as-is** — token range ⊇ [1.7k, 31.7k] target (superset tails; optionally filter). Schema ✓. |
| outlier n14-220 | 20,000 | 14 / 118 / 220 | 2.2k / 19k / 42k | **REUSABLE as-is** — 100% inside target range (exact match). Schema ✓ (gold indices + `N; M; K` answers). |
| rerank k20-315 | 20,000 | 20 / 166 / 315 | 1.6k / 16k / 32k | **REUSABLE as-is** — 100% inside [13,315]; `ce_scores` present ✓. |

Notes: hard-ratio semantics confirmed in `generate_nq_training_data.py` (continuous mode DOES
honor `--hard-neg-frac`; the stale comment at ~line 132 claims n_random=0 — the code at ~line
205-212 splits hard/random correctly). Grouping n176 file ≈ 155 KB/example (~38k tok by chars/4)
— slightly over the 32k budget; settle in the tokenizer calibration pass before freezing rung
assignment.

### NQ fast-regen attempt (2026-07-19) — bottleneck fixed, shard scaling FAILED, 0 output

**Goal:** regenerate the p10 NQ pool in ≤10 min wall-clock (job 3337736's 6-shard CPU run had
been killed after 3.5h at ~97% done with **zero** output, because `save_jsonl` only fires once
at the very end of `_generate_split` — a partial/killed run always yields 0 rows, never a
partial file).

**Root cause found (real, fixed in code):** `generate_nq_training_data.py`'s continuous-n path
called `BM25Searcher.sample_random_passages()` **once per random distractor per example** — a
serial Lucene `doc()` lookup each time. At `--hard-neg-frac 0.1` and n up to 200, that's up to
~180 serial lookups/example, and this (not CE scoring, which is batched and cheap) was the
actual bottleneck matching the observed ~2-3s/kept-example. `bm25.py` already has the fix
pattern used elsewhere in this codebase (`generate_musique_unified_corpus.py`,
`generate_matching_ngram_data.py`): `BM25Searcher.prefetch_random_pool()` once + pure-Python
`sample_from_pool()` per example. Ported the same pattern into
`generate_nq_training_data.py` (new `--pool-passages`/`--pool-fetch-threads` flags, opt-in via
`--pool-passages > 0`, old per-example path kept as fallback). **Validated in isolation:**
single process, `--bm25-threads 8 --pool-passages 20000`, on a busy cubbins node: **~2.0
kept-items/s**, vs. the original run's ~0.3-0.5/s — a **~5x** per-process speedup. CE model
already auto-selects CUDA when visible (`CrossEncoderScorer.__init__`); not exercised (see
below).

**Shard-scaling FAILED:** launched 48-way sharding (40 procs on mooney + 8 on cubbins,
`--num-shards 48`, 4 cores / `--bm25-threads 4` / `--pool-fetch-threads 4` each, `--time
00:09:00`, targeting 420 kept/shard ≈ 20,160 total). **Both jobs (3337963 mooney, 3337964
cubbins) ran the full 9 min and TIMED OUT with every one of the 48 per-shard log files at 0
bytes** — none of the 48 concurrent processes even finished printing their first startup line
(pyserini index load) in 9 minutes, vs. ~12s for a single isolated process. A 6-way-concurrent
calibration on cubbins (job 3337950, `--bm25-threads 4`/6 cores each) DID get through startup
fine in that same window. Diagnosis (not yet fix-validated): 40-48 concurrent cold starts of
JVM (pyserini/Lucene) + PyTorch + transformers on one `srun` step, all sharing **one** cgroup —
none of the per-process launches pinned `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/JVM heap, so each
process's intra-op thread pool defaults to the size of the *whole* visible cpuset, not its
1/N share → severe oversubscription/thrash, not a graceful slowdown. **Net result: 0 lines of
NQ output, 0 output files, from this attempt.**

**Status:** NQ pool still MISSING an audited fast regen. The per-process fix (pool-based random
sampling) is real, committed, and ~5x validated — do not revert it. The shard-count strategy
needs `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (or explicit `torch.set_num_threads`) pinned per
shard and a much lower concurrency ceiling (the 6-way calibration is the only concurrency level
actually observed to start cleanly) before it's safe to scale back up. Files:
`src/corpus_reasoning/data/generate_nq_training_data.py` (pool fix),
`/tmp/.../scratchpad/nq_fast_{mooney,cubbins}.sbatch` (the failed 48-way launcher, for reference
only — do not rerun as-is).

---

## Stage-2 build log (2026-07-19 — dense document-chunked shards + fixed 2k/4k/8k eval rungs)

**Pool staging (mooney → cubbins, node-to-node `scp`, NOT through `/scratch`):** contradiction /
oolong / outlier / rerank (from `single_task_ladders_20k`) and grouping (from mooney
`ctc_suite_data/grouping/`) were `scp`'d directly mooney→cubbins in parallel (~6.5 GB total,
minutes, not the ~5 MB/s NFS path). Byte sizes + line counts verified identical post-transfer.
Landed at cubbins `/data/prasann/ctc_suite_data/{contradiction,oolong,outlier,rerank,grouping}_pool/`.
hotpotqa + reorder were already native to cubbins (Stage-1). **nq skipped** (regen in flight
elsewhere per the task brief).

**Conversion:** `src/scripts/data/ctc_suite/convert_ctc_p0_dense_cubbins.sbatch` (new, `--array=0-6`,
job **3337939**, all 7 tasks in parallel on cubbins, 11–24 min each, env `corpus-reasoning-olmo`).
Wraps `convert_unified_to_document_landmark.py --emit dense --marker-set qwen3_5 --tokenizer
Qwen/Qwen3.5-0.8B-Base --seq-len 40960 --cot-mode none`; multi-rung pools (hotpotqa/reorder/grouping)
pass all per-rung files at once via `--input-jsonl` (nargs, tokenized together into one shard set).
Output → cubbins `/data/prasann/ctc_suite/shards/<task>_train/`.

| Task | kept/dropped | max_len | shards | box_start=box_end | avg docs/ex | Status |
|---|---|---|---|---|---|---|
| hotpotqa | 20000 / 0 | 25742 | 4 | 1,560,000 | 78.0 | ✅ OK — matches mean(11,24,50,100,205)=78 exactly |
| reorder | ~~19954 / 46~~ 19944 / 56 | ~~40938~~ 40957 | 6 | ~~189,447~~ **1,770,896** | ~~9.5~~ **88.8** | ~~⚠ BROKEN~~ ✅ **FIXED 2026-07-19 (FIX 1, reconvert job 3338017)** — see below (expected mean(12,27,57,116,234)=89.2) |
| contradiction | 19366 / 748 | 40957 | 9 | 9,338,761 | 482.2 | ✅ OK — matches pool doc-count median (501) |
| oolong | 21000 / 0 | 39971 | 9 | 7,380,056 | 351.4 | ✅ OK (chunk-by line) |
| outlier | 19986 / 14 | 40558 | 7 | 2,342,927 | 117.2 | ✅ OK — matches pool doc-count p50 (118) |
| rerank | 20000 / 0 | 31135 | 7 | 3,330,854 | 166.5 | ✅ OK — matches pool doc-count p50 (166) exactly |
| grouping | 20000 / 0 | 36785 | 6 | 1,352,000 | 67.6 | ✅ OK — matches mean(10,21,43,88,176)=67.6 exactly |

Every shard's `metadata.json` confirms `marker_set: qwen3_5`, `doc_start_id: 248049`,
`doc_end_id: 248050`, `eos_token_id: 248044`; `labels_mask_part_*.npy` present and non-degenerate
(spot-checked contradiction inst 0: 28,230 tokens, 31 loss tokens = exactly the JSON gold-pairs
answer). Decode spot-check confirmed correct `<|box_start|>`/`<|box_end|>` wrapping and prompt
rendering (contradiction).

✅ **FIXED 2026-07-19 — see "FIX 1" below for the root cause / fix / reconvert numbers.** Original
flag text preserved for the record:

⚠ **FLAG (historical, now fixed) — reorder document-chunking was mostly not happening.** The per-instance decode
(shard 0, inst 0, n=12 passages) shows **0** `box_start` tokens — none of the 12 passages got
wrapped; the whole context stayed FREE/unisolated. The conversion log confirms this at scale:
repeated `document text not found verbatim in the rendered prompt; it stays FREE` warnings, and
one example logged `234/234 documents could not be wrapped`. Aggregate coverage is only
9.5 docs/ex wrapped vs. an expected 89.2 (~11%). Root cause (not yet fixed): the reorder prompt
renders passages as `Passage [N]: <text>` and the wrapper does a **verbatim substring match**
against `documents[i].text` to place `<|box_start|>`/`<|box_end|>` — something in how reorder's
100-word chunks get reflowed into the prompt (whitespace/newline handling, most likely) breaks
that match for the large majority of passages. The shards built cleanly (rc=0, all files present,
19954/20000 examples kept) and are **safe to inspect**, but are **NOT safe to train
document-chunked/landmark attention on as-is** — they'd mostly train on an unchunked (FREE) context
with document-chunked labeling. Needs a fix in the reorder prompt renderer (or the wrapper's
match logic) + a reconversion before use. hotpotqa/contradiction/oolong/outlier/rerank/grouping do
NOT have this problem (coverage matches expected doc counts almost exactly in every case).

**Fixed 2k/4k/8k eval rungs (contradiction, hotpotqa, outlier)** — re-rung via
`src/corpus_reasoning/data/build_v2_eval_ladders.py` (edited in place: contra/outlier rungs
3k/8k/16k/32k → 2k/4k/8k per the row-16/row-11 n-ladder values; added a new `hpqa` task config,
canonical = the n=205 cubbins eval file, shrink mode, `gold_field=gold_doc_indices` +
`hard_neg_indices`, ACTION A2b). 16k/32k rungs deferred per policy. All three: exactly 500
examples/rung, row-aligned (same underlying 500 questions across all 3 rungs, nested-prefix
distractors). Output renamed into `rung_{2048,4096,8192}.jsonl` naming:

| Task | rung | doc count (n) | examples | measured mean tok (20-ex sample, doc text only, pre-wrap) |
|---|---|---|---|---|
| contradiction | 2048/4096/8192 | 40/88/190 | 500/500/500 | 776 / — / 3037 |
| hotpotqa | 2048/4096/8192 | 11/24/50 | 500/500/500 | 1173 / — / 5316 |
| outlier | 2048/4096/8192 | 14/28/57 | 500/500/500 | 1951 / — / 8016 |

⚠ The rung labels are the BUILD_MATRIX-prescribed doc counts, not tokenizer-calibrated for
Qwen3.5 — outlier lands close to its label (1951→2048, 8016→8192) but **contradiction and
hotpotqa under-shoot** their nominal label by roughly 2–2.5× (this doesn't include box-marker /
prompt-template overhead, which adds more, but the raw-text gap alone is large). This is the
open "Calibration pass (Stage-1 audit)" action item in this doc, not a new bug — flagging so the
rung labels aren't read as exact context lengths.

Locations: cubbins `/data/prasann/ctc_suite_data/eval_rungs/{contradiction,hotpotqa,outlier}/rung_{2048,4096,8192}.jsonl`
(raw per-doc-count files also kept at `eval_rungs_raw/{contra,hpqa,outlier}/`). Repo copies (<10MB
each) at `src/scripts/data/ctc_suite/eval_rungs/{contradiction,hotpotqa,outlier}/` — **missing**
`hotpotqa/rung_8192.jsonl` (11.2MB) and `outlier/rung_8192.jsonl` (18.4MB), which stay
cubbins-only (path above).

### FIX 1 (2026-07-19) — reorder wrapping bug: root cause + fix + reconvert

**Root cause, confirmed by diff on a real example** (`reorder_gutenberg100w_n12_train_4000.jsonl`,
inst 0, 12/12 docs unmatched): the reorder prompt renderer (`_format_documents` in
`src/corpus_reasoning/lib/data_format.py`, `task == "reorder"` branch) collapses each passage's
internal ``"\n\n"`` (Gutenberg 100-word chunks routinely contain their own paragraph breaks) to
``"\n"`` before embedding it in the prompt — a deliberate normalization (its own comment says why:
so a passage stays one paragraph and doesn't get split by `wrap_documents`'s `\n\n`-splitting in
the *old* corpus-reasoning path). But `_wrap_documents` in
`src/olmo_core/data/document_chunk_landmark.py` does a **verbatim substring search** against the
*raw* (un-collapsed) `documents[i]["text"]` — it never applied the same collapse, so any passage
with an internal blank line fails to match and silently stays FREE. Diff: raw doc 0 body starts
`"hampered by inadequate preparation, or by poverty.  Her husband\n\ntells us that..."`; the
rendered prompt at the same offset reads `"...Her husband\ntells us that..."` (single `\n`) — a
plain `text.find(raw_body)` never matches.

**Fix** (`src/olmo_core/data/document_chunk_landmark.py`): `_wrap_documents` now takes a `task`
param and looks it up in a new `_RENDERER_TEXT_NORMALIZERS` table (`{"reorder": lambda body:
body.replace("\n\n", "\n")}`) before searching — i.e. it mirrors whatever the renderer did, so the
search text matches what's actually in the prompt. `segment_prompt_to_chunks` threads `task`
through to the `chunk_by == "document"` call site. No change to the rendered prompt itself (per
the task brief — evals must match training format); only the *matching* logic changed. Regression
test: `src/test/data/document_chunk_reorder_wrap_test.py` (asserts ≥90% wrap coverage on
paragraph-break-bearing passages, and that the pre-fix path — normalizer disabled — reproduces 0%
coverage, so a future regression fails loud). Verified end-to-end on cubbins with the real Qwen3.5
tokenizer via `segment_prompt_to_chunks` on all 5 reorder n-files (single example each):
**100% box_start coverage at n=12/27/57/116/234** (was 0% before the fix).

**Reconvert verification** (`sbatch src/scripts/data/ctc_suite/reconvert_reorder_cubbins.sbatch`,
same flags as the original array job — `--emit dense --marker-set qwen3_5 --tokenizer
Qwen/Qwen3.5-0.8B-Base --seq-len 40960 --cot-mode none`, all 5 n-rung files, 20000 examples):

| Task | kept/dropped | box_start=box_end | avg docs/ex | expected (mean of 12/27/57/116/234) | coverage | Status |
|---|---|---|---|---|---|---|
| reorder | 19944 / 56 (too long / reserved collision) | 1,770,896 = 1,770,896 (OK) | 88.8 | 89.2 | **99.3%** (1,770,896 / 1,784,000 raw docs) | ✅ **FIXED** — job `3338017`, rc=0, `ALL_OK`, regression assert (≥90%) passed |

Was 9.5 docs/ex (~11% coverage) before the fix; now 88.8 (~99.3%), matching the expected
mean(12,27,57,116,234)=89.2 to within rounding (the residual ~0.7% is legitimate: some passages
are empty/whitespace-only and skipped by design, not a matching failure). Shards rebuilt in place
at `/data/prasann/ctc_suite/shards/reorder_train/` (6 shards, `token_ids_part_*.npy` +
`labels_mask_part_*.npy`, `max_example_len=40957`, `num_loss_tokens=7,544,416`) — now **safe to
train document-chunked/landmark attention on**.

### FIX 2 (2026-07-19) — eval-rung token calibration

**Root cause:** the doc counts above were set from a "doc text only, pre-wrap" token estimate
(concatenated document bodies, no instruction/query/box-marker overhead), which undershot the
actual rendered-prompt token count. Recalibrated by measuring the REAL prefill (everything the
model actually sees before generating: instruction + query + wrapped documents + box markers,
`segment_prompt_to_chunks(..., include_answer=False)`, Qwen3.5 tokenizer via the local
`Qwen3.5-4B-Base` checkout — verified same `tokenizer.json` across Qwen3.5 sizes) on 50 examples
per existing rung (150 examples/task), fit `tokens ≈ a + b·n_docs` (least squares), solved for the
`n_docs` hitting each of 2048/4096/8192, rebuilt via `build_v2_eval_ladders.py` with the new
`rungs` dict (contra/hpqa edited in place, outlier left untouched), then re-measured the rebuilt
rungs the same way to confirm ±10%. The nested/fixed-500-examples property was spot-checked
post-rebuild (same `queries`/`answers` at every rung, only `documents` count differs) — preserved.

| Task | old n (2k/4k/8k) | old measured ratio | new n (2k/4k/8k) | fit (tokens ≈ a + b·n) | new measured median (2048/4096/8192) | new ratio |
|---|---|---|---|---|---|---|
| contradiction | 40/88/190 | 0.59/0.55/0.56 | **77/167/346** | 288.7 + 22.82·n | 2010/4058/8182 | 0.981/0.991/0.999 |
| hotpotqa | 11/24/50 | 0.64/0.68/0.69 | **17/36/72** | 66.6 + 113.36·n | 1954/4124/8240 | 0.954/1.007/1.006 |
| outlier | 14/28/57 | 1.05/1.03/1.04 | **14/28/57 (unchanged)** | 114.1 + 146.93·n | 2158/4230/8490 (unchanged) | 1.05/1.03/1.04 |

All 6 changed/confirmed rungs land within ±10% of their 2048/4096/8192 label. contradiction's 8k
rung (n=346) is an extrapolation beyond the previously-built max (n=190) but stayed well inside
tolerance because contra's `expand` mode draws from a 444,871-doc harvested filler pool (no
ceiling issue). hotpotqa's rungs are all within its n=205 canonical (shrink mode), so no
extrapolation risk there.

**Files updated:**
- `src/corpus_reasoning/data/build_v2_eval_ladders.py`: `contra`/`hpqa` `rungs` dicts edited to
  the calibrated `n` values (outlier untouched, confirmed already in-tolerance).
- cubbins `/data/prasann/ctc_suite_data/eval_rungs/{contradiction,hotpotqa}/rung_{2048,4096,8192}.jsonl`
  replaced with the recalibrated files (old miscalibrated versions backed up to
  `eval_rungs_PRE_FIX2_backup/{contradiction,hotpotqa}/`); `eval_rungs_raw/{contra,hpqa}/`
  refreshed to the new n-named files (old n40/n88/n190 and n11/n24/n50 files removed).
- Repo copies (`src/scripts/data/ctc_suite/eval_rungs/`): `contradiction/rung_2048.jsonl` (4.5MB),
  `contradiction/rung_4096.jsonl` (9.5MB), `hotpotqa/rung_2048.jsonl` (4.1MB),
  `hotpotqa/rung_4096.jsonl` (8.5MB) updated in place. `contradiction/rung_8192.jsonl` grew from
  7.8MB (old, wrong) to 19.5MB (correct) — now **exceeds the <10MB budget**, so the stale repo
  copy was removed rather than left wrong; it stays cubbins-only like `hotpotqa/rung_8192.jsonl`
  (16.9MB) and `outlier/rung_8192.jsonl` (18.4MB) already did.

---

## Stage-3 build log (2026-07-19 — PILOT data for every remaining task)

Goal (per task brief, supersedes the 20k-train/multi-rung machinery above for pilot purposes):
per remaining task, ONE point at `n(2k)` — ~2,500 train examples (mixed ±30% fine) + a 500-example
eval, both converted to a `--marker-set qwen3_5 --seq-len 4096` dense shard. Pilot does **not**
need the nested/fixed-across-rungs eval machinery (A2b/A7b/A12/A13/...) — those stay open ACTION
items for the FINAL 20k/multi-rung build. contradiction/hotpotqa/oolong/outlier/rerank/grouping/
reorder were already DONE from Stage-1/2 and were skipped here.

Generation: 4 sbatch batches on **cubbins** (`≤6 concurrent procs/node, OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`, scripts in `/scratch/users/prasann/ctc_suite_logs/gen_pilot_batch{A,B,C,D}_*.sbatch`)
+ one NQ-specific fix/relaunch. Harvest/consolidation: `harvest_pilot_all.py` (trims every train
file to ≤2500 rows and every eval file to ≤500 rows into canonical
`ctc_suite_data/pilot_train/<task>.jsonl` and `ctc_suite_data/eval_rungs_pilot/<task>/rung_2048.jsonl`).
Conversion: new `src/scripts/data/ctc_suite/convert_ctc_pilot_dense_cubbins.sbatch` (array job,
mirrors the Stage-2 dense converter but `--seq-len 4096`, pilot single-n files) →
`/data/prasann/ctc_suite/shards_pilot/<task>_train/`.

### NQ — fixed and regenerated (unblocks the Stage-1 "STILL MISSING" row)

Applied the task-brief fix directly: 6-way sharded (`--num-shards 6`), `OMP_NUM_THREADS=1
MKL_NUM_THREADS=1` pinned, `--bm25-threads 4 --pool-fetch-threads 8` per shard (vs. the failed
48-way attempt's unpinned threads) — **all 6 shards completed cleanly in ~7 min wall** (job
3338226), no thrash. `--num-docs-min 12 --num-docs-max 18` (k≈15 pilot target per task brief),
`--hard-neg-frac 0.1`, `--ce-filter` dropped for pilot speed. Output: 2,520 train + 510 eval
(420+85/shard × 6) → harvested to 2,500 train / 500 eval. **Hard-neg ratio audited: 0.0987**
(target ≈0.10 — PASSES, unlike the retired 98%-hard `nq_train_k20-200_combined_aligned` pool).
Converted cleanly (`box_start==box_end=37,555`, avg 15.0 docs/ex, matching k12-18). Deviated from
the brief's "mooney+cubbins split" — ran cubbins-only (6 shards were already fast enough to clear
the whole pilot in minutes; mooney carries a known NFS/datasets-import wedge risk documented above
that wasn't worth the risk for a workload this small). **NQ pilot data now unblocked.**

### Generated + converted this pass (16 tasks, all `ALL_OK` — box_start==box_end, non-degenerate)

| Task | train (raw→kept) | eval (raw→kept) | n_docs target | measured (shard) | flags |
|---|---|---|---|---|---|
| niah_contradiction | 2500→2500 | 500→500 | 40 | 100,000 box tok = 40.0/ex exactly | — |
| absence_pubmed | 1500→1499 | 500→500 | 22 | 29,980 box tok = 20.0/ex (source capped at n20) | ⚠ train 1499 < 2500 (source pool `absence_train_pubmed_n20_p01.jsonl` only has 2000 rows total; 500 went to eval, 1 more dropped by seq-len) |
| qdmatch_nq | 2500→2500 | 500→500 | q9/d9 | 18.0 items/ex (9q+9d, as designed) | — |
| qdmatch_hpqa | 2500→2500 | 500→500 | q9/d9 | 18.0 items/ex | — |
| qdmatch_obliq | 2500→1354 | 300→300 | q4/d4 | 8.0 items/ex on survivors | ⚠ 46% dropped by `--seq-len 4096` (ObliQ source docs are long social-media-style text); eval only 300 (850-query pool cap, matches BUILD_MATRIX's own flag) |
| grouping_labeled | 2500→2500 | 500→500 | 10 | 10.0 docs/ex | ⚠ eval is a disjoint SLICE of the same `openalex_grouping_n10_levels_train_4000.jsonl` generation, not an independently-sampled held-out set (A13 nested-eval mode is still generator-side future work) |
| mathmatch | 2500→2500 | 500→500 | 48 | 48.0 docs/ex | fixed: widened `--ans-min/--ans-max` to ±400 (default ±50 couldn't place 48 docs + 3 gold pairs without collision) |
| cycle | 2500→2500 | 500→500 | 60 | 60.0 docs/ex | — |
| groups4 | 2500→2500 | 500→500 | 100 | 100.0 docs/ex | — |
| textgroups | 2500→2500 | 500→500 | 11 | 11.0 docs/ex | — |
| strmatch | 2500→2500 | 500→500 | 38 | 38.0 docs/ex | — |
| outlier_amzn | 2500→2500 | 500→500 | 20 | 20.0 docs/ex | `--use-titles` not applicable (generator has no title leak path used at convert) |
| helmet_qa | 2500→2499 | 500→500 | single-doc | 1.0 doc/ex (narrativeqa, `--lengths 2000`) | `--task qa` |
| helmet_summ | 2500→2500 | 500→500 | single-doc | 1.0 doc/ex (govreport, `--lengths 2000`) | `--task summarization` |
| xabsence | 2000→2000 | 300→300 | P18 | 39.0 docs/ex (2×18 pair-docs + unmatched) | pool-limited (existing 659-pair pool → P18 train/eval already built pre-brief; only staged, not regenerated); eval 300 < 500 (whole pool) |
| nq | 2520→2500 | 510→500 | k12-18 | 15.02 docs/ex, hard-neg ratio 0.0987 | see NQ section above |

Fixes applied (ONE obvious fix each, per task brief):
- **mathmatch**: `RuntimeError: Could not assemble answer set after 200 attempts` at n=48 with the
  default `--ans-min -50 --ans-max 50` (100-value range too narrow for 48 docs + 3 tolerance-gold
  pairs without collision) → widened to `--ans-min -400 --ans-max 400`. Fixed, reran clean.
- **qdmatch_obliq**: `ValueError: relevant gold docs G=14 exceeds N=4` — ObliQ bridge-style
  questions can carry many gold docs per query (unlike NQ/HPQA's 1-2), overflowing `--num-docs 4`
  at `--num-relevant 3`. Added `--max-gold-per-query 1` (NQ-style 1:1 cap) so `G<=k<=N` always
  holds. Fixed, reran clean (yield loss from `--seq-len 4096` truncation is separate, see table).

### Redundancy — DROPPED (user directive, not attempted)

Out of scope for this pilot pass per explicit directive — do not build or unblock. (Was
tentatively BLOCKED on LLM-serving/GPU quota before the directive; not attempted further.)

### BEIR scifact / BEIR fiqa / MS MARCO — DONE (root cause found + fixed)

**msmarco** converted cleanly on the first (CPU) pass: 2,500 train / 500 eval, 15.4 docs/ex
(k13-18 target) — `ALL_OK`, `shards_pilot/msmarco_train` (+ `msmarco_rerank_train` reusing the
same source under `--task rerank`).

**scifact/fiqa root cause (job 3338218, CPU-only batch):** the original `gen_pilot_batchD_retrieval.sbatch`
requested no GPU, so `CrossEncoderScorer` in `generate_beir_ce_data.py` fell back to
`torch.cuda.is_available() → False → "cpu"`. CE-scoring ~600-6,600 queries × ~100-150 BM25
candidates each on CPU ran **1h20m+ with zero output** before being killed — the generator's own
docstring says CE scoring is "cheap on one GPU," and it is: **fix = `--gres=gpu:1`** (MiniLM-L6 is
tiny, one GPU is plenty), `qos=preemptive` (not `_high`, unnecessary for a few-minute job). Rerun
(job 3338277, scifact+fiqa only, `--ce-batch-size 512`) finished **both in ~15 min wall** (scifact
train+test in <9 min, fiqa — the bigger 5,500-query pool — by ~15 min), vs. the CPU path's 80+ min
with nothing to show. **Do not run BEIR/CE generation without a GPU again.**

| Task | train (raw→kept) | eval (raw→kept) | n_docs | shard | flags |
|---|---|---|---|---|---|
| scifact | 809→807 | 300→300 | 5 | `shards_pilot/scifact_train` (807, box_start=4,035=5.0/ex) | ⚠ train 807 < 2500 (entire train-qrels pool, matches the matrix's own known ~800-query BEIR/scifact ceiling — not a bug); eval 300 < 500 (entire test set, matches the matrix's pre-flagged SciFact eval_size=299/300 ceiling) |
| fiqa | 5500→2500 | 648→500 | 4 | `shards_pilot/fiqa_train` (2498, box_start=9,977≈4.0/ex) | clean — 5,500-query pool trimmed to the 2,500 pilot cap by the harvester |

Locations: `/data/prasann/ctc_suite_data/eval_rungs_pilot/{scifact,fiqa}/rung_2048.jsonl` (300 /
500 rows).

### Redundancy — DROPPED, second confirmation (user directive)

Re-confirmed: not built, not unblocked, removed from the readiness/build list. No further action.
