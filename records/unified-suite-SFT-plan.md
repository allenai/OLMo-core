# Unified instruction-tuning suite → OLMo-core SFT on Beaker — implementation plan

Status: **PLAN / not yet implemented.** Companion to the corpus-reasoning
`results/instruction_tuning_suite_plan.md` (data-assembly side). This doc covers the
**OLMo-core side**: tokenize the combined unified-JSONL suite and SFT-train on it on
Beaker.

## 0. Decisions locked (from discussion 2026-06-19)

| Axis | Decision |
|---|---|
| Prompt template | **Qwen3 chat template everywhere** (train + eval) |
| Tokenization home | **In OLMo-core** — port corpus-reasoning `build_prompt` + deps |
| Instance packing | **Block-aligned packing** — `LandmarkPackingInstanceSource` (landmark/sparse) / `ConcatAndChunk` (dense). Packing is now correct for landmark (see below). |
| Data assembly | **Stays in corpus-reasoning** (tied to raw-source generators) |
| Model variants | dense, fast-landmark, sparse-landmark (reuse existing SFT scaffolding) |

**Packing update (2026-06-19, commit `cf4fa5b9`):** landmark/sparse attention now
supports intra-document masking — `cu_doc_lens` is accepted by the landmark kernels,
and `LandmarkPackingInstanceSource` inserts landmark tokens **per document** (each doc
padded to whole landmark blocks), greedily packs whole docs into a window, and emits
block-aligned `doc_lens` so a query never attends across a document boundary (RoPE also
resets per doc). So packed landmark training is equivalent to one-example-per-window but
**without wasting padding on short rungs** — this supersedes the earlier
padded-one-per-instance plan. Constraint: packing uses `cu_doc_lens`, which landmark
attention does **not** support with context parallelism, so packed runs use FSDP
(`shard_degree`) instead of CP. The reference scripts are
`Qwen3-4B-{fast,sparse}-landmark-packed-SFT.py`. Set `generate_doc_lengths=False` on the
loader (boundaries come from the packing source, not EOS).

## 1. Data flow (the seam)

```
corpus-reasoning                         OLMo-core (this repo)
─────────────────                        ─────────────────────
build_combined_unified.py                convert_unified_to_sft.py  (NEW, gantry/CPU)
  → suite_it_train_unified.jsonl   ──►      per-row _task/_cot_mode dispatch
  (+ RULER gen, held-out guard,            → build_prompt(use_alpaca=False)
     per-rung budget, manifest)            → Qwen3 chat template + char-offset mask
        │                                   → token_ids_part_*.npy + labels_mask_*.npy
        │ upload JSONL to weka                        │ (weka, prasanns/ namespace)
        ▼                                              ▼
   weka prasanns/suite_jsonl/         Qwen3-4B-{dense,fast-landmark,sparse}-unified-SFT.py
                                          LandmarkPackingInstanceSource (lm) / ConcatAndChunk (dense)
                                          → train on Beaker → checkpoint
                                                      │
                                                      ▼
                                          eval (oe-eval chat path — see §6 GAP)
```

## 2. Component work

### A. Data assembly — corpus-reasoning (stays put)
Per corpus-reasoning `results/instruction_tuning_suite_plan.md` §3,§5:
- Refactor `build_combined_unified.py`: declarative `(task, source_glob, cot_mode,
  per_rung_n)` table, `HELD_OUT` leakage guard (beir_fiqa/beir_scifact/redundancy +
  `*_splittrain.jsonl`), per-(task×rung) budget, manifest sidecar.
- Generate RULER (`generate_ruler_data.py --length-mode target`, 1k–64k).
- Output `suite_it_train_unified.jsonl` (+ manifest) → upload to weka.
- **No change to the per-row tag contract** (`_task`, `_cot_mode`): the OLMo-core
  tokenizer dispatches on exactly these.

### B. Port `build_prompt` into OLMo-core — core tokenization work
Vendor the prompt-construction code so tokenization is self-contained and the prompt
strings stay identical to the corpus-reasoning baselines.

- **Vendor** (verbatim, with a "do not edit casually" provenance header like the
  existing `convert_longctx_tasks_to_sft.py`):
  - `data_format.py` → `build_prompt` (22 task branches) + CoT builders (~1320 L)
  - `prompts.py` (instruction strings, ~488 L)
  - `io.py` helpers (`insert_dummy_tokens`; **skip** `format_alpaca_prompt` — we use
    chat template, not alpaca)
  - **Skip** `chunked_attention.py` (only needed for `--wrap-docs`/packing → deferred)
- **Home:** new module `src/olmo_core/data/corpus_reasoning_prompts/` (or
  `src/scripts/data/_vendored_prompts/`). Pin the source branch+commit in the header.
- **Contract:** call `build_prompt(ex, task=_task, cot_mode=_cot_mode, use_alpaca=False)`
  → `(user_content, answer)`. `use_alpaca=False` returns the **bare** instruction+context
  (no alpaca wrapper); we then render it as the user turn of the Qwen3 chat template.
- ⚠ **Byte-identical verification:** the chat-rendered user content must match the
  eval-time user content. Today the oe-eval task *reimplements* `build_oolong_user_content`
  / `build_contradiction_user_content`. Add a regression test: for oolong+contradiction,
  assert vendored `build_prompt(use_alpaca=False)` == oe-eval's builders == the current
  `convert_longctx_tasks_to_sft.py` output (which is already in production). This
  de-risks the port before touching the other 20 tasks.

### C. Generalize the converter
New `src/scripts/data/convert_unified_to_sft.py` (generalizes
`convert_longctx_tasks_to_sft.py`):
- Read combined JSONL; **per-row** `_task`/`_cot_mode` dispatch via vendored `build_prompt`
  (replaces the hardcoded `--task oolong|contradiction` builders).
- Reuse the existing machinery unchanged: `render_chat` (Qwen3 chat template),
  `tokenize_instance` (char-offset loss mask, EOS=151643 separator, landmark-id guard),
  `ShardWriter` (`token_ids_part_*.npy` + `labels_mask_*.npy`), `--max-seq-len` drop.
- `metadata.json`: per-task / per-cot_mode counts, num_tokens, num_loss_tokens.
- **Held-out assertion** (defense in depth): refuse any row whose `_task`/source is in
  the held-out denylist.
- Gantry wrapper: generalize `convert_longctx_tasks_to_sft_gantry.sh` (CPU Beaker job,
  JSONL from weka, shards → `weka .../prasanns/suite_it_sft_qwen/`).

### D. Unified SFT training script(s)
New `src/scripts/train/sft/Qwen3-4B-{dense,fast-landmark,sparse-landmark}-unified-SFT.py`
(or one parametrized script + run-name dispatch). **Landmark/sparse variants** are based
on the new `Qwen3-4B-{fast,sparse}-landmark-packed-SFT.py`; **dense** on the longctx dense
script:
- **Landmark/sparse (packed):** `LandmarkPackingInstanceSourceConfig` over a
  `NumpyDocumentSourceConfig` of the combined `token_ids_part_*.npy` +
  `labels_mask_*.npy` shards (the converter emits EOS-separated docs = a document source).
  Set `mem_freq`/`mem_id`/`pad_id`; `sequence_length` a multiple of `block_size`.
  `generate_doc_lengths=False`. **No CP** (landmark `cu_doc_lens` ⊥ context parallelism) —
  shard with FSDP (`shard_degree`).
- **Dense (packed):** `ConcatAndChunkInstanceSource` (+ `generate_doc_lengths=True` for
  block-diagonal masking via flash-attn varlen) over the same shards.
- `SEQUENCE_LENGTH` 32k or 64k + YaRN factor 2 → evaluable to 64k. Packing means short-rung
  examples no longer waste a full window (the §4 token-balance concern is resolved).
- `resolve_dataset_path(run_name)` → the combined unified dir (single mix).
- Hyperparams: reuse longctx SFT (lr 5e-5, wd 0, linear decay, 3 epochs). wandb on. Re-tune
  batch/parallelism for the FSDP (no-CP) packed landmark path.

### E. CPT mixing ("longmino") — composable second source (plan Option B) — **v2**
- Tokenize raw CPT via existing `tokenize_dolma3_longmino_sample.py` → a packed
  (`ConcatAndChunk`) source with full-sequence loss.
- Mix task source (PadToLength) + CPT source by **token fraction** (`CPT_TOKEN_FRAC=0.15`)
  via `MixingInstanceSource`.
- ⚠ Caveats (why v2): (1) landmark variants need landmark-token insertion on CPT rows
  too; (2) padded one-per-instance + CPT interacts awkwardly with token-fraction
  accounting. Ship the no-CPT mix first, add CPT after.

### F. Eval — **DONE** (was the biggest gap)
Took option (i): extended oe-eval (branch `prasann/longctx-eval`) with a generic
full-suite task. `corpus_reasoning_suite.py` + the vendored `corpus_reasoning/` package
dispatch prompt-building (`build_prompt`, use_chat_format) and scoring (all 23 task types,
ported from `evaluate.py`) by `task_type`. All **185 eval files auto-register** from a
bundled manifest as `cr_<stem>` (× direct/cot/cr modes). Validated locally: 15 task types
score primary=1.0 feeding gold back. Eval data lands on weka `cr_suite_data` via
`land_suite_eval_to_weka_gantry.sh`; launch like `launch_longctx_task_evals.sh` with
`-t cr_<task>::<mode>`. Remaining: a GPU smoke-eval to confirm native-`olmo_core`
generation end-to-end; `grouping`/`reorder` need sklearn/scipy in the eval venv.

## 3. Execution order (milestones)

1. **Port `build_prompt` + deps** (§B) + regression test vs the in-production
   oolong/contradiction converter → proves the port is byte-identical. *(CPU, local-testable.)*
2. **Generalize converter + gantry** (§C); dry-run tokenize a small combined JSONL
   (CPT_FRAC=0), inspect `metadata.json`, assert held-out leakage = 0.
3. **Unified SFT script** (§D); smoke-train (short run) on the small shard to prove the
   data pipeline + checkpoint path.
4. **corpus-reasoning**: `build_combined_unified` refactor + RULER gen → full combined
   JSONL → weka (§A).
5. **Full tokenize → full SFT**: dense first, then fast-landmark, sparse-landmark (packed).
6. ~~Eval coverage~~ **DONE** (§F) — run a GPU smoke-eval, then the held-out buckets.
7. **CPT mixing** (§E) — v2.

## 4. Risks / open questions

- **byte-identical prompt port (§B)** — chat-template user content must match eval; the
  eval side already uses `build_prompt(use_alpaca=False)`, so the converter must call it
  identically. Regression-test before scaling.
- **Packed landmark needs FSDP, not CP (§D)** — `cu_doc_lens` ⊥ context parallelism for
  landmark, so the packed runs drop CP and shard with FSDP; re-tune batch/parallelism vs
  the old CP=8 longctx scripts. Watch `LandmarkPackingInstanceSource`'s dropped-doc warning
  (docs longer than one window) — raise `sequence_length` if it fires.
- **CPT × landmark interaction (§E)** — CPT rows also need per-doc landmark insertion;
  deferred to v2.
- No local GPU: tokenizer/converter is CPU (testable here); SFT + eval run on Beaker.
- **eval ⊥ training prompt drift** — eval uses chat template + `build_prompt`; the unified
  tokenizer must match (same template, same `query_position` per task).

## 5. File inventory (new / changed)

| File | Repo | New/Changed | Purpose |
|---|---|---|---|
| `build_combined_unified.py` | corpus-reasoning | changed | declarative table, held-out guard, per-rung budget, manifest |
| `src/olmo_core/data/corpus_reasoning_prompts/` | OLMo-core | **new** | vendored `build_prompt` + `prompts` + `io` helpers (chat path) |
| `src/scripts/data/convert_unified_to_sft.py` | OLMo-core | **new** | per-row multitask tokenizer → SFT shards |
| `convert_unified_to_sft_gantry.sh` | OLMo-core | **new** | Beaker CPU wrapper |
| `src/scripts/train/sft/Qwen3-4B-*-unified-SFT.py` | OLMo-core | **new** | dense / fast-landmark / sparse unified SFT |
| `test/.../vendored_prompts_test.py` | OLMo-core | **new** | byte-identical regression vs production converter |
| oe-eval `longctx_corpus_reasoning.py` (or `evaluate.py` chat path) | oe-eval / corpus-reasoning | changed | full-suite chat eval coverage (§F) |

## 6. Not doing now (explicitly deferred)

- `--wrap-docs` / chunked-attention doc markers, gold-doc gradient masking, `standard_mix_prob`.
- CPT mixing (ships in v2 after the no-CPT mix is validated).

## 7. Appendix — packing (now implemented, commit `cf4fa5b9`)

The packing fork is resolved. Landmark attention now accepts `cu_doc_lens`, and
`LandmarkPackingInstanceSource` (`src/olmo_core/data/composable/`) does the three things
this section previously listed as needed: per-document landmark insertion (each doc padded
to whole `mem_freq` blocks), greedy whole-doc packing into `sequence_length` windows with a
block-aligned tail pad, and explicit per-instance `doc_lens` forwarded to the model for
block-diagonal masking + per-doc RoPE reset. Reference SFT scripts:
`Qwen3-4B-{fast,sparse}-landmark-packed-SFT.py`. Caveats carried into §0/§4: packing is
FSDP-only for landmark (no CP), and docs longer than one window are dropped (warned). The
unified SFT just points `LandmarkPackingInstanceSource` at a `NumpyDocumentSource` of the
converter's EOS-separated shards.
