# The n=100 chunked collapse is a MARKER-EMBEDDING × MARKER-POSITION bug, not a mask result

**Status:** root-caused 2026-07-14. Supersedes the "only dense works at n=100" conclusion, which was
measured on poisoned runs.

## The symptom

Every training run on the leak-free ("v2") document-chunked contradiction shards flatlines at
**CE ≈ 0.79** and evaluates at chance (f1 ≈ 0.001, parse_rate 1.0 — well-formed output, random claim
indices). Critically this happens **for every attention mask, including full/dense and plain causal** —
so it cannot be a mask-capacity result.

| run | data | mask | final CE |
|---|---|---|---|
| `diagB-oldstd-mix` | v1 (leaky) | standard (= plain causal) | **0.0017** ✅ |
| `diagA-v2std-nomix` | v2 (leak-free) | standard (= plain causal) | 0.7883 ❌ |
| `diagC-v2-densepath` | v2 | dense script, full attention | 0.7928 ❌ |
| `v2-std-c100` | v2 | standard | 0.7901 ❌ |
| `v2-chunked-base` | v2 | chunked | 0.7811 ❌ |
| `v2-rand050-base` | v2 | random_doc | 0.7927 ❌ |

An unrestricted-attention model could not even *memorize* v2 in 3 epochs. That is the tell: the
problem is upstream of attention.

## What was ruled out (all verified, don't re-litigate)

- **Data content.** v1 and v2 hold the same 2000 examples, same claim order, same gold. Gold-pair word
  overlap 0.443 vs random-pair 0.048 in *both* (9x) — gold is correctly aligned in v2.
- **Loss masks.** 2000/2000 well-formed masked answers in both; identical stats (59,046 loss tokens,
  mean 29.5/example).
- **chunk_ids / PAD.** 100 chunks in both; PAD starts exactly at the example end.
- **The mask itself.** `cross_doc_mode="standard"` is *provably* plain causal (allowed/causal = 1.0000,
  tensor-equal to `tril`) on both shards.
- **The data loader.** Real batches from `PadToLengthInstanceSource` are correct for both.
- **The marker-embedding bug** (`box_start` vs `box_end` cos = 1.0): already fixed — the `-fixmark`
  base has cos = **+0.0128**. Not a regression of [document-chunked-marker-embeddings.md].

## The actual cause

The only difference between v1 and v2 is **where the box markers sit**:

```
v1:  Claim 1: <|box_start|>The ability to glucuronidate…<|box_end|>\n\nClaim 2: <|box_start|>…
v2:  <|box_start|>\n\nClaim 1: The ability to glucuronidate…<|box_end|><|box_start|>\n\nClaim 2: …
```

v1 wraps only the claim **body**, leaving `Claim N:` outside the chunk → FREE (this is the free-token
leak; FREE tokens attend everything *and* are attended by everything, so the labels bridge chunks —
measured chunked density 0.344 in v1 vs 0.099 in v2). v2 correctly pulls the label inside the chunk.

**Token-surgery bisect** — shards derived from the v2 stream differing in exactly one variable, all
trained with plain causal (so all carry *identical information*):

| arm | change | CE @375 steps |
|---|---|---|
| `v2asis` | none (control) | **0.79** ❌ flat |
| `nomark` | the 200 marker tokens **deleted** | **0.109** ✅ learns |

Deleting the markers — changing nothing else — restores learning. So the marker *tokens* poison
plain-causal training in the v2 layout.

**Why position matters:** in the `-fixmark` base the markers have embedding norm **0.3963** vs a
trained-token median of **1.4152** (they are ~3.6x too small, because `fix_marker_embeddings.py`
renorms them to `<|im_start|>`'s norm). RMSNorm rescales every token to the same RMS, so a tiny,
semantically-OOD embedding is amplified into a full-strength ~random direction. In v1 that noise token
sits *after* the `Claim N:` label, so the index tokens are encoded cleanly and the model can ignore it.
In v2 it sits **immediately before** the label — i.e. an OOD noise token precedes every claim index the
answer has to copy — and training never recovers.

## The fix

Keep the leak-free v2 wrap (do **not** revert to the v1 body-only wrap — that re-introduces the leak).
Instead give the markers **real, trained delimiter embeddings**: initialize `<|box_start|>` /
`<|box_end|>` from the rows of `«` / `»` (norm ≈ 1.46, in-distribution), rather than
"mean-of-trained + noise renormed to `<|im_start|>`".

Base: `/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-trainedmark`.

## Consequences for past results

- Every **n=100 leak-free (v2)** number is void — it measured this bug, not a mask.
- Every **n=100 v1** number (`fix-std-c100` 0.896, `fix-chunked-c100` 0.002, `fix-rand050-c100` 0.255,
  `fix-hier-k50-cyc5-c100` 0.365, the fa-hybrid sweep) is measured on **leaky** data: the FREE claim
  labels bridge chunks, so "chunked" there was never truly isolated (density 0.344, not 0.099).
  The chunked-vs-hierarchical comparison at n=100 must be **re-run** on the leak-free shards with the
  repaired base.
- The claim "at n=100 only dense works" is **not established**.

Related: [[qwen3-marker-embedding-bug]], [[docchunk-wrapping-id-title-leak-bug]],
[[attn-mask-mixing-default]].
