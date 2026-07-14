# Document-chunk marker embeddings: a base-checkpoint bug that silently zeroes results

**TL;DR — before running any document-chunked / landmark training from a fresh Qwen3 base, repair the
marker embeddings with `src/scripts/data/fix_marker_embeddings.py`. If you skip this, marker-dense
runs produce chance-level results that look like a modeling finding but are an artifact of the base
checkpoint. The data shards are fine; do not rebuild them.**

## The bug

The document-chunked and landmark data paths mark structure with reserved special tokens:

| token | id | source |
|---|---|---|
| `<|box_start|>` | 151648 | `DOC_START_ID`, `src/olmo_core/data/document_chunk_landmark.py` |
| `<|box_end|>` | 151649 | `DOC_END_ID`, same |
| landmark | 151860 | past the real vocab, in the embedding matrix's padding region |
| pad | 151863 | same |

**Qwen3 never trains these rows.** `<|box_start|>` and `<|box_end|>` are *unused* special tokens, so
they keep their shared initialization and their embeddings are **bit-identical**: cosine similarity
**+1.0000**, and off-distribution besides (norm ≈ 0.35 vs ≈ 0.93 for trained rows). All of
151648–151655 are mutually cos = 1.0000. The landmark and pad rows sit past `REAL_VOCAB_SIZE = 151669`
in the padded region and are likewise cos = +1.0000 with each other.

For contrast, genuinely-trained specials `<|im_start|>` / `<|im_end|>` sit at cos = 0.7392, and
ordinary tokens `«` / `»` at 0.6013.

The consequence: **the model cannot tell an "open document" marker from a "close document" one.** It
sees N copies of one identical out-of-distribution vector.

## What was *not* broken

`chunk_ids` are derived from token **ids**, not embeddings, so the chunk roles and the resulting
attention mask were **always correct**. The tokenized shards were always correct. What the bug
destroyed was the model's *perception* of the document structure, not the structure itself. This is
why the failure is so easy to misread as a modeling result.

## How it presents

Scaling in the number of markers, not in the mask:

- **20 documents (40 markers):** no visible effect. The surrounding `Claim N:` text carries enough
  structural signal for the model to route around the degenerate embeddings. Repairing the markers
  moves contradiction f1 by less than one standard error.
- **100 documents (200 markers):** training **flatlines at CE ≈ 0.79** — "learned the output format,
  answers at chance" — **for every attention mask**, including plain causal.

The tell that cracks it open: `cross_doc_mode="random_doc"` with `doc_keep_prob=1.0` is provably
identical to full causal attention, yet it scored ~0. A model with *unrestricted* attention could not
fit its own training data. That is never a mask result — it is broken infrastructure.

Elimination that confirmed it: crossing (dense script × document-chunked module) with (plain shard ×
marker-bearing shard) showed the **shard** determined everything and the attention module was
innocent; and swapping the markers for ordinary tokens (`«` / `»`), or stripping them entirely, made
training converge normally.

## The fix

`src/scripts/data/fix_marker_embeddings.py` repairs the **base checkpoint** (not the data). It keeps
the reserved ids — which is what guarantees they can never collide with real text — and gives each
marker a distinct in-distribution vector: the mean of the trained rows plus per-token noise, renormed
to the scale of a genuinely-trained special token. It asserts every marker pair ends at |cos| < 0.5.

```bash
PYTHONPATH=$PWD/src python src/scripts/data/fix_marker_embeddings.py \
  --base /path/to/base/model_and_optim \
  --out  /path/to/base-fixmark \
  --model-size 0.6B          # or 4B
```

Then point the launcher's base at the repaired directory (`BASE_SRC=...-fixmark` for
`src/scripts/train/sft/run_q06b_attn_explore_mooney.sbatch`).

**No data rebuild is needed.**

Existing repaired base (Qwen3-0.6B):
`/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-fixmark`.
**No repaired 4B base exists yet** — build one before trusting any 4B document-chunked number.

## Validation

Same shard, same script, same steps, same world size — only the four embedding rows differ:

| 100-doc arm | CE (train) | f1 before | f1 after |
|---|---|---|---|
| `standard` (causal control) | 0.0027 | 0.000 | **0.896** |
| `random_doc`, `doc_keep_prob=0.5` | 0.0176 | 0.000 | **0.255** |
| `chunked` | 0.789 | 0.001 | 0.002 |
| `chunked` + one mid full-attention layer | 0.794 | 0.001 | 0.001 |

The causal control is the proof: it went from chance to 0.896 with nothing changed but the embeddings.
The remaining `chunked` collapse is a **real** capability limit, now that it is measurable.

## How to check a base for this bug

```python
import torch.nn.functional as F
cos = F.cosine_similarity(emb[151648][None].float(), emb[151649][None].float()).item()
# ≈ 1.0  -> POISONED, run fix_marker_embeddings.py
# < 0.5  -> repaired
```

## Results this invalidated

Every marker-dense (100-doc) document-chunked number produced before the fix, across `chunked`,
`hierarchical_dilated`, and `random_doc` at all keep-probs, plus every hybrid full/chunked-layer
sweep. Any run whose base was not `-fixmark` should be assumed contaminated until re-run.

---

## ⚠ SEQUEL (2026-07-14): the repair in this document was INCOMPLETE

The fix described above made the markers mutually distinguishable (cos 1.0 → ~0.01) but renormed them
to `<|im_start|>`'s norm — **0.396 against a trained-token median of 1.415**. RMSNorm rescales every
position to the same RMS, so a marker row at ~1/3.6 of a normal token becomes a full-strength *noise*
vector wherever it appears. On the leak-free document-chunked shards (marker immediately before every
`Claim N:` label) that flatlines training at CE ≈ 0.79 for **every** attention mask — including plain
causal — reproducing this document's exact signature and masquerading, again, as a mask result.

`fix_marker_embeddings.py` now seeds each marker from a real trained delimiter row and asserts the
norm is in-distribution. **Any base repaired before 2026-07-14 must be re-repaired.**
Full diagnosis: `records/n100-chunked-marker-position-bug.md`.
