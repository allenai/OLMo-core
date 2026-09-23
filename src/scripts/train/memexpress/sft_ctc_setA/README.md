# sft_ctc_setA/ — dense Qwen3.5-4B SFT on the CTC setA mix, 32k vs 256k

Dense (full-attention) Qwen3.5-4B SFT on **setA**, the 11-task CTC mix built IID with the
olmo-eval CTC suite at every rung from 2k to 256k. It is a controlled pair on maximum training context:

| Arm | Window | Examples | Geometry | Script |
|---|---|---|---|---|
| `32k` | 32,768 | only those ≤32,768 tokens; longer ones **dropped**, not truncated | 1 node, no CP, DP=8, grad-accum 4 | `Qwen3.5-4B-dense-ctc-setA-32k-SFT.py` |
| `256k` | 262,144 | all | 1 node, Ulysses CP=8, DP=1, grad-accum 4 | `Qwen3.5-4B-dense-ctc-setA-256k-SFT.py` |
| `256k-2node` | 262,144 | all | 2 nodes, CP=4, DP=4 (amandab's exact geometry) — fallback | `Qwen3.5-4B-dense-ctc-setA-256k-2node-SFT.py` |

Everything else is shared, from one builder (`_qwen35_setA_common.py`):

- Base checkpoint: `q35-4b-dense-256k-fix/step2385`, loaded weights-only.
- Data files, one epoch, seed 34521.
- 1,048,576 tokens per step at LR 4e-5 with 3% linear warmup.

The 32k arm trains on a strict subset of the 256k arm's data, so it takes fewer steps.

## Reproduce

Every step runs on Beaker and is pinned to a pushed commit. The jobs are unallocated (workspace
`ai2/flex2`, budget `ai2/oe-other`), run at priority `urgent`, and use jupiter H100s.

```bash
L=src/scripts/train/memexpress/sft_ctc_setA/launch_setA_sft.py
python $L tokenize   # CPU: setA JSONL -> marker-free shards on weka (once)
python $L prep       # CPU: shard checks, drops per window, base ckpt, packed windows/steps (once)
python $L 32k        # 1 x 8 H100
python $L 256k       # 1 x 8 H100   (or: python $L 256k-2node)
```

Add `--dry-run` to print the gantry command without submitting.

- Checkpoints go to `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run-name>/`.
- Logs go to wandb `prasanns-allen-institute-for-ai/memory-networks`, grouped by run name.
- Evaluate with the olmo-eval launcher:
  `scripts/ctc_suite/launch_ctc_suite.py --arm full --ckpt <step dir>` (branch
  `prasann/ctc-suite-launcher`). It renders prompts in the chat template (`CTC_SUITE_PROMPT_FORMAT=chat`),
  IID with this data.

## Recipe source

The recipe is **amandab's validated dense 256k SFT**: `sft_xlong256k/`, run
`q35-dense-contra-3ep-256k-min1h-20260914`. This branch (`prasann/ctc-setA-sft`) is cut from hers
(`amandab/contradiction-only-256k-20260914`), which carries CP fixes that `prasann/landmark` lacks:
Ulysses KV replication, contiguous all-to-alls, and glob-expansion truncation.

`_qwen35_setA_common.py` calls her `build_qwen35_xlong5_experiment` and keeps her settings for the
model, optimizer, AC and CP (Ulysses, since GDN rejects ring CP). It replaces only the data, the
budget and the node/CP geometry. Her builder's docstring explains each of those settings.

**One node at 256k.** FSDP shards over the flattened DP×CP mesh (`_get_model_mesh`), so CP=8 on one
node still shards the 4B params and optimizer state 8 ways. Each rank holds 32,768 tokens of
activations, half of what her CP=4 run held. CP=8 is legal: 8 divides the 16 attention heads and the
32 GDN value heads. This geometry is new, so if it misbehaves, `256k-2node` is her exact tested
layout at the same batch and LR.

## Data

`/weka/.../prasanns/ctc_sft_sets/setA_max20_evaliid/shards_qwen35_256k_nomarkers/<task>/` holds 11
tasks:

- nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction;
- xabsence (one-sided exact-copy);
- reorder, rerank, strmatch;
- grouping (unlabeled OpenAlex).

textgroups is excluded because of its shortest-document shortcut. The shards are built by
`src/scripts/data/ctc_sft/build_ctc_sft.py` on `prasann/landmark` (see `records/ctc-fast-suite-eval.md`
for the IID audit against the eval's own rows).

- **No document markers.** The earlier `shards_qwen35_256k` wraps each document in
  `<|box_start|>`/`<|box_end|>` for the chunked/landmark arms. olmo-eval's prompts carry no
  markers, and Qwen's marker embeddings are untrained (see CLAUDE.md), so the dense arms use a
  `--no-doc-markers` re-tokenization. The rest of the rendering is the same: chat template, query
  position `both`, no CoT. `prep` fails if any marker id appears in a shard.
- **Packing.** Whole examples are packed Best-Fit-Decreasing into windows, across all tasks, with
  block-diagonal masking at EOS. `LongDocStrategy.exclude` drops over-window examples, which is how
  the 32k arm caps length.

## Runs

| Arm | Run name | Beaker | Commit |
|---|---|---|---|
| | | | |
