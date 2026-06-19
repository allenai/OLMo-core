# Long-context task data (OOLONG / contradiction) — provenance

Where the OOLONG and contradiction SFT/eval data comes from, and the one link in the
chain that is **not** reproducible from this repo or from Beaker. Companion to
`convert_longctx_tasks_to_sft.py`.

## Chain (newest → oldest)

1. **Consumed by training/eval** — weka
   `oe-training-default/ai2-llm/checkpoints/prasanns/longctx_sft_qwen/`:
   - `eval_jsonl/oolong_test_synth_ctx{N}_spliteval.jsonl` → read directly by the oe-eval task
     `longctx_oolong_synth_ctx{N}` (branch `prasann/longctx-eval`), scored by `score_oolong`.
   - `oolong_mix/` (+ `contradiction_mix/`) → tokenized SFT shards produced by
     `convert_longctx_tasks_to_sft.py` from the `*_splittrain.jsonl` files.
2. **Copied to weka** by the `longctx-eval-data-copy*` Beaker experiments (workspace `ai2/flex2`),
   whose `/input` mount was the dataset in step 3.
3. **Beaker dataset `01KTR39FVDCRESFH6MRFX5Z93F`** — name `longctx-task-jsonl`, author `prasanns`,
   created 2026-06-10. **`sourceExecution: None`** → a **direct local upload** (`beaker dataset
   create`), NOT the output of a tracked Beaker job. Bundles `oolong_test_synth_*` (7 ctx ×
   {splittrain, spliteval}) and the `contradiction_*_pubmed_both_*` files.
4. **Generated from scratch** by `generate_oolong_data.py` in the **corpus-reasoning** repo, branch
   **`task-suite-expansion`** (prompt strings in the convert script are copied verbatim from
   corpus-reasoning `scripts/lib/prompts.py`). OOLONG = the Oolong benchmark, arXiv:2511.02817.
   corpus-reasoning lives on the **hermione** machine.

## Generated from scratch — yes

The data is **synthetic**, not a downloaded static benchmark file. The generator samples real
HuggingFace classification datasets as raw material — **8 of them**: `agnews`, `imdb`, `yahoo`,
`multinli`, `app_reviews`, `formality`, `negation`, `metaphors` — then constructs the synthetic
corpora (lines of `Date: … || User: … || Instance: …`), the aggregate-statistics questions, the
exact gold answers, and the per-context-length renderings. The Oolong paper is the recipe;
`generate_oolong_data.py` synthesized the actual instances.

## Composition (computed from the JSONLs, 2026-06-12)

- Split sizes: **train (`splittrain`) = 320/ctx × 7 = 2240**; **eval (`spliteval`) = 80/ctx × 7 =
  560** → ~80/20 per context length (ctx ∈ {1024,2048,4096,8192,16384,32768,65536}).
- **In-distribution split**: all 8 datasets, all 6 question types (`answer_type` ∈ COMPARISON ~30%,
  NUMERIC ~27%, LABEL ~24%, USER ~16%, DATE ~1%, MONTH_YEAR ~1%), and all 3 task groups
  (counting / timeline / user, ~⅓ each) appear in BOTH train and eval in near-identical
  proportions. **Nothing is held out** by category — the split is random per ctx, not stratified
  (per-dataset eval share wobbles, e.g. yahoo 7.9% vs 13.7% in train). The only held-out axis is
  the specific sampled instances. DATE/MONTH_YEAR are tiny in eval (5 instances total) → per-type
  eval breakdowns for those are low-confidence.

## Reproducibility gap

The **generation step (4) is not captured anywhere in OLMo-core or Beaker**:
- It did not run as a Beaker job (`sourceExecution: None`), so there is no command/log/spec.
- `generate_oolong_data.py` is not in this repo — it is in corpus-reasoning on hermione.

Consequences: the **seed, the exact HF dataset versions/splits sampled, the sampling logic, and how
the 80/20 train/eval cut was drawn** are all off-platform. To truly reproduce the data, or to verify
that train and eval don't overlap at the raw-item level, you must go to the corpus-reasoning repo
(`task-suite-expansion` branch) on hermione. That is the actual "from scratch" origin and the one
hop with no tracked artifact.
