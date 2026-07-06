# Vendored RAG-QA data generators (NEAR rung)

These generate the NEAR NQ + HotpotQA training data (corpus-reasoning *unified JSONL*) via BM25
hard negatives over pyserini's `wikipedia-dpr-100w` index — the upstream of
`src/scripts/data/convert_rag_tasks_to_sft.py`.

`generate_nq_training_data.py`, `generate_hotpotqa_data.py`, `align_hn_doc_lengths.py`, and
`lib/{io,bm25}.py` are **vendored verbatim from `PrasannS/corpus-reasoning`** (only the
`scripts.lib.* / scripts.data.*` imports were rewritten to the local `lib.* / sibling` layout).
They live here because Beaker's GitHub token can clone `allenai/*` repos but not the personal
`PrasannS/corpus-reasoning`, so the CPU generation job runs entirely from OLMo-core.

## Run on a CPU Beaker node

```bash
gantry run --name rag-datagen-near \
  --workspace ai2/flex2 --budget ai2/oe-other --cluster ai2/jupiter-cirrascale-2 \
  --cpus 16 --gpus 0 --priority high \
  --weka oe-training-default:/weka/oe-training-default \
  --python-manager conda --conda-file src/scripts/data/rag_datagen/ragdatagen-env.yml \
  --install true --allow-dirty --shared-memory 16GiB --timeout 0 --yes \
  -- bash src/scripts/data/rag_datagen/run_rag_datagen.sh
```

Writes `nq_train_*.jsonl` + `hotpotqa_train_*.jsonl` to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_jsonl/`, then feed them to
`convert_rag_tasks_to_sft.py` (capped to the RLHN token count the run script prints).
