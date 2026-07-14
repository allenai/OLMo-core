# Data generation & conversion — the map

All data-gen code lives in this repo (verified against the standalone `/scratch` checkout
2026-07-13: no drift, nothing standalone-only except deliberately-deprecated scripts). It's a
**two-layer pipeline** split across two directories by role:

## Layer 1 — task generation → unified JSONL: `src/corpus_reasoning/data/`

Makes the *tasks themselves* (75 scripts): `generate_<task>_data.py` for
contradiction/oolong/nq/beir/ruler/outlier/rerank/absence/..., corpus builders
(`generate_*_unified_corpus.py`), eval-ladder builders (`build_v2_eval_ladders.py`,
`build_v2_outlier_ladder.py`, `build_xlong_rungs.py`, `subsample_beir_ladder.py`), CoT generation
(`generate_cot.py`), splits/audits. Output: **unified-format JSONL** under
`/scratch/users/prasann/corpus-reasoning/data/` (~22 GB, gitignored — data lives outside git,
code lives here).

## Layer 2 — JSONL → tokenized shards (+ staging): `src/scripts/data/` (this dir)

Converts unified JSONL into olmo-core SFT shards (`token_ids_part_*.npy` uint32 +
`labels_mask_*.npy` bool, EOS-separated, completion-only loss) and stages them where jobs read
them:

- `convert_longctx_tasks_to_sft.py` — the original oolong/contradiction converter (local CPU,
  seconds; see `local_cluster.md` §training). `convert_unified_to_sft.py` generalizes to the full
  task roster. `convert_nq_to_sft`/`convert_rag_tasks_to_sft`/`convert_rlhn_to_sft`,
  `convert_unified_to_document_landmark.py` (landmark/docchunk boundary-token variants),
  `build_gold_sidecar_from_shard.py` (goldgrad sidecars).
- `*_gantry.sh` twins run the same converters on Beaker writing to **weka** (see `beaker.md` —
  baked image trick, multi-cluster CPU). `*.sbatch` = local slurm variants.
- `fix_marker_embeddings.py` — the base-checkpoint marker repair (not data, but part of every
  docchunk/landmark data path — see `records/document-chunked-marker-embeddings.md`).
- `rag_datagen/` — self-contained RAG data generation (own env + README).
- `longctx-data-PROVENANCE.md` — where oolong/contradiction data came from and the one
  non-reproducible link.

## ⚠ Name collision: two `convert_unified_to_sft.py`

They are DIFFERENT scripts. **Use this dir's** (`src/scripts/data/convert_unified_to_sft.py`): it
builds prompts via the vendored `olmo_core.data.corpus_reasoning_prompts.build_prompt` —
byte-identical to what the oe-eval `cr_*` tasks render, so train and eval prompts match.
`src/corpus_reasoning/data/convert_unified_to_sft.py` is the older one using
`corpus_reasoning.lib.data_format.build_prompt`; kept because the corpus_reasoning eval path still
imports that builder, but don't generate new training shards with it.

## Where generated data lives

| Location | What |
|---|---|
| `/scratch/users/prasann/corpus-reasoning/data/` | Layer-1 unified JSONL (source of truth, ~22 GB) |
| `/scratch/users/prasann/longctx_sft_qwen/<name>/` | Layer-2 tokenized shards for LOCAL runs (staged to node `/data` at job start) |
| `weka: .../checkpoints/prasanns/` (`cr_suite_data`, `_eval_bundle_eval500{,_v2}`, ...) | Shards/eval bundles for BEAKER runs — staging is the S3→weka two-step in `beaker.md`; grep job logs for `MISSING` |
| `s3://ai2-llm/checkpoints/prasanns/` | Transfer buffer between Berkeley and weka |

## Rules that keep biting

- **NQ:** only the p10 pipeline (hard-neg ≈ 10% of k + CE filter). Anything named
  hn49/hn99/hn199/ladder64k is the banned 98%-hard generation (its builder is in `deprecated/`).
- The tokenized shards for docchunk/landmark are CORRECT — when marker-dense training flatlines,
  repair the base checkpoint (`fix_marker_embeddings.py`), do NOT rebuild data.
- Eval sets ≥500 examples; emit `eval_size` (never `n`) in result JSON — CLAUDE.md §reporting.
- New eval-ladder files: S3 push alone does nothing for Beaker jobs — gantry-sync to weka.
