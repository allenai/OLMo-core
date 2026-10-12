# The `free_pad_repeat` probe is confounded — it does not measure FREE-position capacity

**Status:** found 2026-07-14, while re-running the n=100 analysis experiments on the repaired base.

## The intent

`free_pad_repeat` (in `olmo_core/data/document_chunk_landmark.py`) appends filler tokens **after** the
wrapped documents and **before** the answer, so they land outside every `<|box_start|>..<|box_end|>`
span and get the **FREE** role — attending the whole context under every `cross_doc_mode`. The probe
asks: *document-chunked models do all their cross-document comparison at the trailing FREE positions
and collapse at 100 documents — is that a capacity limit of those positions?* Widen the FREE budget and
see if the collapse lifts.

## The problem

The filler is **N copies of one identical sentence**:

```python
FREE_PAD_SENTENCE = "Review the claims above carefully before answering. "
prompt = prompt + "\n" + (FREE_PAD_SENTENCE * free_pad_repeat)
```

Adding it destroys training **regardless of the attention mask** — including plain causal, where the
FREE role is *irrelevant* (the mask is `causal & not_pad`; roles only gate PAD).

n=100 contradiction, leak-free shards, repaired-marker base, 0.6B, final train CE:

| data | chunked mask | plain causal (`standard`) |
|---|---|---|
| `base` (no filler) | **0.224** | **0.0008** |
| `free60` (+481 filler tokens, 60× the *same* sentence) | 0.80 ❌ | **0.81** ❌ |

A model with unrestricted attention cannot fit the data once the filler is present. So the knob is not
isolating "more FREE positions" — it is measuring what a long block of **exactly-repeated text** does to
the model, and that effect swamps whatever FREE capacity would have contributed. The FREE budget it
actually buys is modest (fraction of the answer's visible keys that are filler: 0.039 → 0.127 at
`free60`, 0.318 at `free10x`), far too small to explain a total collapse on its own.

Likely mechanism: 60 near-identical key vectors sitting between the claims and the answer act as a
coherent attention attractor / duplicate-key sink at exactly the positions that must attend back to the
claims. (Not yet proven — see below.)

## Consequences

- **Every `free_pad_repeat` result is void**, including the `free10x` (fpr=244) runs, which flatlined at
  CE 0.79 and read as "widening the FREE budget doesn't help". They flatlined because the filler broke
  training, not because FREE positions saturated.
- The question the probe was built to answer is **still open**.

## The fix

Make the filler **varied** rather than exactly repeated, holding the FREE-token budget fixed. A
varied-filler shard (`contra_n100_v2_free60v`: 60 *distinct*, content-neutral sentences, 813 FREE
tokens — a *larger* budget than `free60`) is built by `debug/build_free_varied.py` and is the control
that separates the two effects:

**RESULT (2026-07-14): it is the REPETITION.** Same base, same mask, n=100, leak-free:

| filler | added FREE tokens | chunked CE | plain-causal CE |
|---|---|---|---|
| none (`base`) | 0 | 0.224 | 0.0008 |
| `free60` — 60× the **same** sentence | 481 | **0.80** ❌ | **0.81** ❌ |
| `free60v` — 60 **distinct** sentences | 813 (*a larger budget*) | **0.48** ✅ | **0.24** ✅ |

A *bigger* FREE budget made of varied text trains; a smaller one made of one repeated sentence
collapses. So `free_pad_repeat` measures repeated-text damage, not FREE capacity.

⚠ **`repeat_doc_text` (the "more within-chunk tokens" control) has the SAME defect** — it duplicates
each claim's text **verbatim**, and `rep2` collapsed identically (CE 0.795, flat). Both knobs are void.

**Action taken:** the two probes were rebuilt with varied, content-neutral, budget-matched filler, so
they differ only in the *role* the added tokens carry:
- `contra_n100_v2_free60v` (`debug/build_free_varied.py`): +813 **FREE** tokens (ctx 4598, FREE 1030)
- `contra_n100_v2_chunkpad` (`debug/build_chunkpad.py`): +1112 **within-chunk** tokens (ctx 5710, FREE 217)

`FREE_PAD_SENTENCE` / `repeat_doc_text` in `olmo_core/data/document_chunk_landmark.py` should be
replaced by varied generators so the knobs measure what they claim to. Until then, do not quote any
`free_pad_repeat` or `repeat_doc_text` number.

Related: [[marker-embedding-norm-bug]] (the other "adding tokens breaks even plain causal" bug found the
same day — same tell: *if unrestricted attention also fails, the cause is upstream of the mask*).
