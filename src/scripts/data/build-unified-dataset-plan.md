# Plan — build a single unified instruction-tuning dataset (→ AI2/olmo-core SFT shards)

Goal: turn the per-task suite train JSONL (landed on weka `cr_suite_data`, 184 files / 17.7 GB)
into **one** combined, tagged JSONL → **one** AI2-format tokenized shard dir the unified SFT
scripts read. Ship a **~1k debug build first** to validate the end-to-end pipeline, then the full
build.

## 0. Pipeline

```
weka cr_suite_data/<task>_train_*.jsonl  (per-task, no _task tag)
        │  build_combined_suite_jsonl.py   (NEW) — tag + filter + budget + shuffle
        ▼
suite_combined[_debug].jsonl   (rows carry _task / _cot_mode)   [+ manifest sidecar]
        │  convert_unified_to_sft.py (gantry, CPU) — build_prompt + Qwen3 chat + EOS-doc shards
        ▼
weka suite_it_sft_qwen/combined[_debug]/  token_ids_part_*.npy + labels_mask_*.npy   ← AI2 format
        │  Qwen3-4B-{dense,fast-landmark,sparse-landmark}-unified-SFT.py (DATA_ROOT)
        ▼
train → eval (oe-eval cr_* suite)
```

The tokenize step IS the "AI2 format" step: raw uint32 `token_ids_part_*.npy` + bool
`labels_mask_*.npy`, EOS(151643)-separated documents — exactly what `NumpyDocumentSource`
(`LandmarkPacking` / `ConcatAndChunk`) consumes.

## 1. Roster — what goes in

**IN TRAIN (16 task types):** retrieval, cot_retrieval, qa, ruler, oolong, contradiction,
matching_ngram*, mathmatch, strmatch, qdmatch, absence, xabsence, outlier, grouping_labeled,
reorder, cycle, groups4, summarization. (*matching_ngram has no train files in this drop.)

**HELD OUT — eval only, asserted absent from train:**
- `redundancy` (whole task: 3 train files)
- `beir_scifact_*` (4 retrieval train files incl. `*_splittrain`)
- `beir_fiqa` — already has **no** train files (eval-only by construction)

The build script keeps a `HELD_OUT_GLOBS` denylist and **asserts zero matching rows** land in the
output.

## 2. CoT handling

24 of 184 train files are CoT variants (`*_cotmix_*` / `*_cot.jsonl`); the rest are plain. The data
already encodes the target style; we tag each row's `_cot_mode` from the manifest so
`convert_unified_to_sft.py` builds the matching target. **Default:** include the plain file for each
(task × rung); where a `cotmix` sibling exists, include it too (≈50/50 CoT/plain for that task) so
one model learns both modes and the eval `::direct` / `::cot` prefill selects between them.
**Decision knob:** `--cot {both, plain, cot}` (default `both`).

## 3. Length ladder + per-(task × rung) budget

Each task spans rungs encoded in the filename (`n20…n1000`, `k20…k500`, `q20…q100`, RULER
`L1024…L65536`). To stop the biggest tasks (qdmatch/cot_retrieval/ruler) from dominating, cap a
**target #examples per (task × rung)**. Default: `--per-rung-budget N` (sample first N rows per
file, seeded). Long-context rungs cost more tokens, so optionally balance on tokens later (v2). The
sidecar logs realized per-task / per-rung / per-token counts.

## 4. The build script — `build_combined_suite_jsonl.py` (NEW)

1. Read `suite_manifest.tsv`; select `split==train`, drop `HELD_OUT_GLOBS`, apply `--tasks` filter.
2. For each file: stream rows, attach `_task` (manifest task) + `_cot_mode` (manifest cot_mode),
   take up to `--per-rung-budget` (or `--total-budget` for the debug build), with a seeded RNG.
3. Assert no held-out rows; shuffle (seeded); write `suite_combined[_debug].jsonl`.
4. Sidecar JSON: per-task / per-rung counts, held-out denylist, seed, totals.

Runs on CPU; testable locally on a small downloaded subset.

## 5. Phase 0 — the ~1k debug build (do first)

- `build_combined_suite_jsonl.py --total-budget 1000 --out suite_combined_debug.jsonl`
  → ~1k rows spread across all in-train tasks (≈50–60/task), held-out excluded.
- Tokenize: `convert_unified_to_sft_gantry.sh --input-jsonl '.../suite_combined_debug.jsonl'
  --out-dir '.../suite_it_sft_qwen/combined_debug'` (small, fast CPU job).
- Smoke-train: a unified SFT script pointed at `combined_debug` for a few steps
  (`--trainer.max_duration.value=50 --train_module...` overrides), 1 node, to prove the data
  pipeline + packing + checkpoint path end-to-end.
- Smoke-eval: a couple `cr_*::direct` tasks on the debug checkpoint.
- Only after this is green do the full build.

## 6. Full build

- `build_combined_suite_jsonl.py --per-rung-budget <N>` → `suite_combined.jsonl` (+ sidecar).
- One `convert_unified_to_sft` gantry job → `suite_it_sft_qwen/combined/` (AI2 format).
- Point the 3 unified SFT scripts' `DATA_ROOT` at it; launch dense / fast-landmark / sparse.

## 7. Decisions to confirm (defaults in brackets)

1. **Per-task budget** for the full build: equal #examples per (task×rung) [default], or token-balanced (v2).
2. **CoT** [`both`] vs plain-only vs cot-only.
3. **Held-out set** [`redundancy`, `beir_scifact`, `beir_fiqa`] — confirm.
4. **RULER**: include all 49 train files [yes]; reserve the 64k rung as eval-only? [no — train all].
5. **query_position** per task — must match the eval suite (eval default `both`). [`both`]
6. **CPT mixing** — separate composable source, **v2** (not in this combined JSONL).

## 8. Known issue — rerank train/eval schema mismatch

The rerank **train** files (`msmarco_helmet_rerank_train_*`) are HELMET-native schema
(`{ctxs, qid, query}` → `rerank_helmet`), but the rerank **eval** files (`msmarco_dev_rerank_*`)
and the manifest tag are standard `rerank` (`{documents, gold_doc_indices}`). So training and eval
rerank would use **different prompt formats**. `convert_unified_to_sft.py` now skips rows whose
`build_prompt` fails (logged + counted in `metadata.json` `skipped_build_by_task`), so rerank rows
are dropped rather than crashing the job — meaning **rerank is currently absent from train**. To
include it, reconcile the schemas: either generate standard-`rerank` train data, or tag the helmet
files `rerank_helmet` AND add a `rerank_helmet` eval (the scorer already exists). **Decision needed.**
