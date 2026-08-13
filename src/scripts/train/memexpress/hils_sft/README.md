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

It is a **two-step**, both on a weka-mounted CPU gantry node (`--priority urgent`):

**1. Combine the 5 tasks into one multitask JSONL.** `cr_suite_data/` carries a
`suite_manifest.tsv` (`file <TAB> task <TAB> cot_mode <TAB> split <TAB> bytes`); this reads it,
attaches `_task` / `_cot_mode` per row so the converter dispatches per row, asserts the held-out
tasks (redundancy, beir_scifact, beir_fiqa) never leak, and samples to a budget:

```bash
python src/scripts/data/build_combined_suite_jsonl.py \
    --data-dir /weka/oe-training-default/ai2-llm/checkpoints/prasanns/cr_suite_data \
    --out <weka>/olmo3_5task_combined.jsonl   # + --tasks / budget flags
```

Do not hand-roll the per-task globs instead — the manifest is what carries each task's `cot_mode`,
and the held-out assertion is what keeps scifact/fiqa out of training while they are scored as OOD
ladders.

**2. Tokenize to OLMo-3 vocabulary.** `--task` is omitted deliberately: every row carries `_task`.

```bash
python src/scripts/data/convert_unified_to_sft.py \
    --tokenizer /weka/oe-training-default/amandab/hf_models/allenai__Olmo-3-1025-7B \
    --eos-token-id 100257 \
    --landmark-token-id -1 \
    --chat-template src/scripts/ctc_eval/lib/chat_templates/olmo3_chatml.jinja \
    --input-jsonl <weka>/olmo3_5task_combined.jsonl \
    --out-dir /weka/oe-training-default/amandab/sft_olmo3/5task
```

`--landmark-token-id -1` because OLMo-3 has no landmark id; leaving the Qwen default (151860) is
harmless only by accident (it is outside a 100278 vocab) and would silently become a real-token
drop filter on any vocabulary that does reach it.

Dolci is a separate run of `convert_dolci_instruct_sft_gantry.sh` with the same tokenizer / eos /
chat-template flags, writing to `.../sft_olmo3/dolci`.

**Tokenize each task to its OWN out-dir.** The mixture is applied by `mix_documents()` at dataset
level (see above); a single pre-combined corpus has no mixing stage, so the sampling weights would
silently do nothing.

### Task names: the manifest does not use our eval keys

`suite_manifest.tsv` names tasks by the canonical key, which differs from the eval task names
(`TASK_ALIASES` in `src/scripts/eval/ctc_suite/run_rung_eval.py` is the authority):

| our eval key | manifest task |
|---|---|
| `contra` | `contradiction` |
| `nq` | **`retrieval`** |
| `rerank` | `rerank` |
| `outlier` | `outlier` |
| `oolong` | `oolong` |

⚠ **`retrieval` is overloaded.** hotpotqa, niah_contradiction, msmarco, beir_scifact and beir_fiqa
all alias to it. scifact/fiqa are held out and the combine script asserts they stay out, but
hotpotqa and msmarco would be swept in by a bare `--tasks retrieval` — silently widening the NQ
component into a different task mixture than our Qwen3.5 arms trained on. Filter by *file*, not by
task, for this one.

### Source files (resolved 2026-08-13)

The `hn<N>` suffix is the **hard-negative count**, and p10 means `hn ≈ 10% of k`. The banned
98%-hard build is the `hn98` / `hn198` / `hn498` family, which sits in the same directory under
nearly identical names — this is the trap the NQ directive exists for.

| task | train JSONL | cot_mode |
|---|---|---|
| contra | `contradiction_train_pubmed_both_n*_k3.jsonl` | `template` |
| **nq** | **`nq_train_k100_hn10_2500.jsonl`** — hn10 of k=100 = p10 ✅ | (from manifest) |
| rerank | `msmarco_helmet_rerank_train_k*_2000.jsonl` | `template` |
| outlier | (manifest, task `outlier`) | (from manifest) |
| oolong | (manifest, task `oolong`) | (from manifest) |

The p10 choice is confirmed by the eval set it is scored against:
`nq_validation_k20_hn2_600.jsonl` — hn2 of k=20, the same 10%. Never take `nq_*_hn98*` or the
`hotpotqa_*_hn98*` neighbours.

`beir_scifact_*_splittrain.jsonl` appears in the manifest as `retrieval`/train, but
`HELD_OUT_GLOBS` excludes it — that assertion is load-bearing here, since scifact and fiqa are
scored as OOD ladders and must never enter training.

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
