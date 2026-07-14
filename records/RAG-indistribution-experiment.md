# HELMET-RAG in-distribution experiment (FAR vs NEAR)

Goal: measure how much the **in-distribution-ness** of SFT data affects the HELMET **RAG QA**
eval (NQ / TriviaQA / PopQA / HotpotQA, 8k–64k). One axis, two rungs to start, three model
variants. Same base checkpoint + identical hyperparameters per variant — **only the data differs**,
so any HELMET-RAG delta is attributable to in-distribution-ness.

| Rung | Data | Status |
|---|---|---|
| **FAR** | amandab's RLHN doc-ID retrieval SFT (`amandab/rlhn_sft_qwen_63k`) | already trained — just eval |
| **NEAR** | NQ + HotpotQA **RAG-QA** (answer output, HELMET prompt), corpus-reasoning BM25 / `wikipedia-dpr-100w` | build + train (this runbook) |

(EXACT rung — reproduce HELMET's KILT corpus/retriever — deferred.)

Model variants (each gets a FAR eval + a NEAR train→eval): `dense`, `fast-landmark`,
`sparse-landmark`. The NEAR runs reuse the **same** SFT scripts as FAR via `resolve_dataset_path`
(a run name containing `rag` selects the NEAR dataset).

---

## Step 0 — RLHN token count (sets the NEAR data size)

NEAR is sized to match FAR's total training tokens. Read it once:

```bash
cat /weka/oe-training-default/ai2-llm/checkpoints/amandab/rlhn_sft_qwen_63k/metadata.json
#  -> use the "num_tokens" value as RLHN_NUM_TOKENS below
```

## Step 1 — generate NQ + HotpotQA unified JSONL (corpus-reasoning)

Run in the `corpus-reasoning` repo (needs pyserini + the `wikipedia-dpr-100w` BM25 index; the index
auto-downloads on first use). `--num-docs` sets per-instance length (~130 tokens/passage); the SFT
pipeline packs instances to 64k regardless (ConcatAndChunk), so this mainly controls how many
questions land per 64k window. Generate generously — Step 2 caps to the token budget.

```bash
# in corpus-reasoning/
python scripts/data/generate_nq_training_data.py \
    --num-train 20000 --num-eval 500 \
    --num-docs 150 --num-hard-negatives 10 \
    --output-dir data
#  -> data/nq_train_k150_hn10_20000.jsonl

python scripts/data/generate_hotpotqa_data.py \
    --split train --num-examples 20000 \
    --num-docs 150 --question-type bridge --num-hard-negatives 10 \
    --output-dir data
#  -> data/hotpotqa_train_k150_bridge_hn10_20000.jsonl
```

Copy both JSONL onto weka (under your namespace), e.g.:

```bash
mkdir -p /weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_jsonl
cp data/nq_train_k150_hn10_20000.jsonl \
   data/hotpotqa_train_k150_bridge_hn10_20000.jsonl \
   /weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_jsonl/
```

## Step 2 — convert to SFT shards, capped to RLHN's token count

`--target-tokens` makes NEAR match FAR's total tokens; the two input files are read round-robin so
NQ and HotpotQA interleave before the cap. (Commit + push first — gantry ships your committed HEAD.)

```bash
src/scripts/data/convert_rag_to_sft_gantry.sh \
  --input-jsonl '/weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_jsonl/nq_train_*.jsonl' \
                '/weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_jsonl/hotpotqa_train_*.jsonl' \
  --target-tokens <RLHN_NUM_TOKENS> \
  --query-position after \
  --out-dir /weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_sft_qwen/nq_hotpotqa_near
```

Sanity-check `nq_hotpotqa_near/metadata.json` afterward: `num_tokens` ≈ `RLHN_NUM_TOKENS`,
`num_instances_by_source` has both `nq` and `hotpotqa`.

## Step 3 — launch the 3 NEAR SFT runs

Identical to amandab's RLHN launches but with a `rag` run name (→ NEAR dataset). 4 nodes each.

```bash
python src/scripts/train/sft/Qwen3-4B-dense-SFT.py \
    launch q4b-dense-rag-near-sft ai2/jupiter-cirrascale-2
python src/scripts/train/sft/Qwen3-4B-fast-landmark-SFT.py \
    launch q4b-fast-landmark-rag-near-sft ai2/jupiter-cirrascale-2
python src/scripts/train/sft/Qwen3-4B-sparse-landmark-SFT.py \
    launch q4b-sparse-landmark-rag-near-sft ai2/jupiter-cirrascale-2
```

Checkpoints land at `/weka/oe-training-default/ai2-llm/checkpoints/<run-name>/`.

## Step 4 — HELMET RAG eval (FAR + NEAR), all variants

`SKIP_RULER=1` runs HELMET only (RAG is part of the HELMET 8k–64k suite; read the RAG rows out of
the results). Point it at each checkpoint's final `stepN` dir.

```bash
# FAR — amandab's existing RLHN checkpoints
for v in dense fast-landmark sparse-landmark; do
  SKIP_RULER=1 bash ./launch_long_context_evals.sh \
    /weka/oe-training-default/ai2-llm/checkpoints/q4b-$v-sft/stepFINAL
done

# NEAR — the runs from Step 3
for v in dense fast-landmark sparse-landmark; do
  SKIP_RULER=1 bash ./launch_long_context_evals.sh \
    /weka/oe-training-default/ai2-llm/checkpoints/q4b-$v-rag-near-sft/stepFINAL
done
```

Compare FAR vs NEAR HELMET-RAG accuracy within each variant.

---

### Files in this experiment
- `src/scripts/data/convert_rag_tasks_to_sft.py` — NQ/HotpotQA unified-JSONL → Qwen3 SFT shards (NEAR).
- `src/scripts/data/convert_rag_to_sft_gantry.sh` — gantry wrapper for the converter.
- `Qwen3-4B-{dense,fast-landmark,sparse-landmark}-SFT.py` — `resolve_dataset_path` selects FAR vs NEAR by run name.
