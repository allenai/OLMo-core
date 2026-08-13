# hils_sft — SFT for HiLS-Attention-7B and its Olmo-3 control

**The contrast.** `tencent/HiLS-Attention-7B` vs `allenai/Olmo-3-1025-7B`, SFT'd on the same
5-task + Dolci25 mixture at a 32k window, everything else held fixed. HiLS is a ~50B-token
continued-pretrain *of that Olmo-3 checkpoint*, so this pair isolates what HiLS's chunk-wise sparse
attention (plus its CPT) does to long-context task ability.

**The readout.** The v2 ladder via `eval_lc_native.py --backend hf` (see `../hils_eval/`):
contra/nq/rerank/outlier/oolong at 2k–32k plus the four OOD ladders, f1 / ndcg@10 / task-native.

**Why SFT at all.** Both models are BASE. Measured zero-shot (2026-08-13), they do not produce the
answer formats: contra floors at f1=0.0 because the model continues the prompt instead of answering,
and nq's apparent 0.058→0.000 rung curve is a coincidence rate (`1/n_docs`) collapsing into a
parse-failure rate — see `records/hils-attention-eval-integration.md`. Without SFT the suite
measures instruction-following, not long context, and cannot distinguish the architectures.

## Why not olmo_core

Neither model can use our standard SFT stack:

* HiLS's attention is not implemented in olmo_core, and cannot be without porting the tilelang
  kernels;
* both use the **OLMo-3 vocabulary (100278)**, so every SFT shard we have — all Qwen3/Qwen3.5 — is
  unusable for them.

So the `create-sft-config` skill's central rules (landmark window geometry, `_DENSE_WEIGHTS`,
`LandmarkPackingInstanceSource`) do not apply here. Its *principles* still do and are honored below:
name the contrast, control data or compute explicitly, weights-only strict load, fresh save folder,
urgent priority.

## The three arms

| arm | stack | what it is for |
|---|---|---|
| `hils-7b-sft-5task-dolci25-32k` | veomni | the treatment |
| `olmo3-7b-sft-5task-dolci25-32k` | veomni | the control — **same trainer**, so the contrast is the model |
| `olmo3-7b-sft-5task-dolci25-32k-olmocore` | olmo_core | bridge — ties these numbers to our existing Qwen3.5 ladder and cross-checks the veomni trainer |

The first two share a trainer, a dataset object, a seed, and therefore byte-identical batches. The
third deliberately changes the trainer while holding the model and data fixed, so a veomni-vs-olmo_core
gap is measurable rather than assumed away.

## Files

| File | What | Status |
|---|---|---|
| `sft_shard_dataset.py` | Reads our `token_ids/labels_mask` `.npy` shard pairs as a map-style dataset for veomni | **done**, unit-tested (`src/test/scripts/hils_sft/`) |
| `train_sft_veomni.py` | The SFT task (modeled on the HiLS repo's `tasks/pretrain_with_ruler.py`, which is CPT+RULER — there is no SFT task upstream) | TODO |
| arm configs | three launchers per the table above | TODO |

### Why a shard adapter rather than veomni's chat dataset

Our shards come from `src/scripts/data/convert_unified_to_sft.py`, whose prompts are built by
`build_prompt` and are **byte-identical to what the eval renders**. Routing training through
veomni's own chat dataset would re-render prompts from messages with a second template
implementation, silently reintroducing the train/eval prompt mismatch the converter exists to
prevent. Keeping the shards and adapting the reader is what preserves that property.

## Data: OLMo-3-vocab shards (the prerequisite)

The mixture is our standard one — contra/rerank/outlier/nq/oolong at the `_DENSE_WEIGHTS` shares
plus 25% Dolci-Instruct-SFT — **re-tokenized to the OLMo-3 vocabulary**. Sources on weka:

* 5-task unified JSONL: `.../checkpoints/prasanns/cr_suite_data/*.jsonl`
* **NQ must come from the p10 build** (`single_task_ladders_p10/nq`), never the 98%-hard one —
  standing directive, and everything is *evaluated* on the p10 ladder.
* Dolci: re-run its converter for olmo3; the existing `amandab/dolci-instruct-sft/{qwen3,qwen35}`
  trees are the wrong vocabulary. Note Dolci ships in this same shard format, so the adapter reads
  it unchanged.

Conversion command shape (the `--chat-template` flag was added for exactly this — base-model
tokenizers ship none):

```bash
python src/scripts/data/convert_unified_to_sft.py \
    --tokenizer /weka/oe-training-default/amandab/hf_models/allenai__Olmo-3-1025-7B \
    --eos-token-id 100257 \
    --landmark-token-id -1 \
    --chat-template src/scripts/ctc_eval/lib/chat_templates/olmo3_chatml.jinja \
    --input <cr_suite_data/...jsonl> --out-dir <weka out> --task <task>
```

**Pass the same template file the eval attaches.** A training-time template that differs from the
eval-time one reintroduces the mismatch silently — nothing errors, the numbers are just wrong.

Verified 2026-08-13 with the OLMo-3 tokenizer: prompt is a prefix of the full conversation, the
tokenizer is fast (needed for offset-derived masks), EOS does not occur inside bodies, and the mask
covers exactly the assistant turn.

## Open decisions before launch

* **Token-matched or data-matched?** Both arms read the identical dataset, so a step count is a
  data count and the two coincide *between the veomni arms*. It matters for the olmo_core bridge
  arm, whose packing differs — state which one that arm controls for.
* **Budget.** Mirror the Qwen3.5 arms' ~700M-token SFT, or size from one epoch of the realized mix.
  Log `content_tokens` from `SFTShardDataset.stats()` so the number is auditable.
* **Base checkpoint hygiene.** Both arms load weights-only, strict. Neither uses document markers
  or landmark ids, so the `fix_marker_embeddings.py` repair does **not** apply here.
