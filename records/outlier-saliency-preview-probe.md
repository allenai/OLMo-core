# Saliency-selected real tokens, and the one-layer preview — can either reach FULL parity on outlier?

**Date** 2026-09-15 · **Status** RUNNING (6 Beaker jobs; table below filled as rungs land) · **Branch** `prasann/landmark`
**Driver** `debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py` (eval only, 1 GPU, frozen `ds64-outlier-dense-u64M`)

## 1. Where this starts

Two records between them close the *slot-vector* axis for outlier:

* `records/outlier-slot-probe.md` — the mean-input-embedding slot is unreadable: `R@gold_pooled`
  sits on the `k/n` uniform-guess floor in every construction, on two checkpoints, two corpora and
  three lengths, and swapping the gold documents' slot vectors with random non-gold ones is
  undetectable.
* `records/outlier-richer-slot-probe.md` — thirteen richer slots (renormalised, content-only,
  centred, `cent_cmean`, idf, `G=2/4` segments, 2- and 4-layer block-local encoders) all leave the
  frozen dense model on the same floor, even though an *oracle* cosine readout recovers 0.73 at 2k
  from `cent_cmean`. Binding constraint: the **readout**, not the representation.

`records/soft-kv-slot-probe-handoff.md` and `records/layer-soft-probe.md` record what *did* reach
frozen-model parity on contradiction and oolong: keep each document's **header** real and pool the
body, and — separately — read densely for L layers and only then compress (`fromL` +
`layer_input_mean`, ΔCE 0.000 at 0.738 of FULL's FLOPs on contradiction @8k). Neither had ever been
run on outlier.

This probe asks whether outlier can be bought back with **more real tokens chosen well** (idea 1) or
with **a few dense layers before compressing** (idea 2). Same frozen checkpoint, same rung files,
same header-real construction (`--st-header-stop-id 5491 --st-header-stop-count 1`), same
`cent_cmean` slot for the pooled remainder, gold-blind throughout (every document gets the same
rule — no gold-aware condition except where flagged).

## 2. What is measured

Per construction × rung: answer CE and the **paired ΔCE vs FULL with its SE**, CE on the answer's
DIGIT tokens (mean answer CE on outlier is ~95 % prose — `records/outlier-slot-probe.md` §4), free-
generation set-F1 over the k document ids and its paired ΔF1, recall of the gold documents (all
pooled in these constructions), real tokens kept per document, compaction, and an estimated FLOP
fraction against FULL.

**FLOP fraction** is computed from the model's own per-block coefficients — `lin_i` = 2 × the
block's matmul parameters, `quad_i` = 2 · 2 · n_heads · head_dim for the **8** softmax layers and 0
for the **24** GatedDeltaNet layers — as
`(Σ_i lin_i·len_i + Σ_i quad_i·len_i²/2) / (same at len_i = T)`, with `len_i = T` for a dense preview
layer and `len_i = T2` (the compacted length) elsewhere. Note the 4B model has **32** layers, not 36,
so a preview of L layers costs `L/32` of the dense linear term plus its share of the quadratic one.

The ORACLE conditions (`grad*`, `attn*`) need a full forward (and, for `grad*`, a backward) to
choose their tokens; that cost is **not** in `flop_frac` — they are upper bounds, not recipes. The
deployable conditions are `rule*` (features only, free) and `prev4_k*` (the selection falls out of a
preview the construction already pays for).

## 3. Constructions

All header-real, gold-blind, remainder pooled into one `cent_cmean` slot per document.

| name | selection of the k real body tokens | deployable? |
|---|---|---|
| `full` | — (plain full attention, the reference) | — |
| `cc00` | none (k = 0) — the known floor | yes |
| `first{k}` | the first k body tokens | yes |
| `idf{k}` | the k highest `-log p(token)` over the training shard | yes |
| `rand{k}` | k uniformly at random — the "any k real tokens" control | yes |
| `grad{k}` | **ORACLE**: top-k by `‖∂(answer CE)/∂e_t‖` from one gradient-checkpointed backward | no |
| `attn{k}` | **ORACLE**: top-k by attention mass received from the answer + last-16 prompt positions, summed over heads and all 8 softmax layers | no |
| `attnlast{k}` | the same over the **last 4** softmax layers only | no |
| `rule{k}` | ridge prediction of the gradient saliency from 6 cheap features, fit on DISJOINT rows | **yes** |
| `grad{k}_swap` | CONTROL: the kept tokens of the gold documents exchanged with random non-gold documents' | — |
| `prev{L}` | no real body tokens; layers `0..L-1` dense over the whole real context, slot = mean of the document's **layer-L** hidden states, layers `L..31` compacted | yes |
| `prev4_k{k}` | `prev4` **plus** the top-k body tokens per document by **layer-3 attention** from the last 32 prompt positions — gradient-free, no second network | **yes** |
| `prev4_first{k}` | `prev4` plus first-k — the selection control for `prev4_k{k}` | yes |
| `prev4_k{k}_swap` | CONTROL on `prev4_k{k}` | — |

Mechanism: a real-token subset is expressed by marking those positions **FREE** in the chunk ids —
exactly what `mark_doc_headers_free` does to a header — so they survive compaction at their original
positions and are excluded from the slot mean. Only layers 0-2 of the 32 are GatedDeltaNet before the
first softmax layer (index 3), so the preview's gradient-free selector needs **L = 4**; at L = 1 or 2
there is no attention matrix to read and none is synthesised (GDN gate/write magnitudes were not
used — stated here rather than silently substituted).

## 4. Results

_(filled in as the jobs land)_

## 5. Verdict

_(pending)_

## 6. Runs

| mode | rung | rows (gen) | experiment | job |
|---|---|---|---|---|
| saliency | 2k | 240 (64) | `01M2K5DNHBD53823V8DM3GPXSD` | `01M2K5DNNS78RKVNDHGCX99H46` |
| saliency | 8k | 240 (48) | `01M2K5EJHP32V2W8JFCJ20BA2N` | `01M2K5EJNWFSN8VPX4CJFK7EJK` |
| saliency | 32k | 120 (32) | `01M2K5FPAT2VS75BVQ8PTHJBNR` | `01M2K5FPEHDAVC21PM5FYS2JYQ` |
| preview | 2k | 240 (64) | `01M2K5GRVA2QGAMREP6GV6T0C3` | `01M2K5GS0RVB7R5RD50XQNQFE1` |
| preview | 8k | 240 (48) | `01M2K5HQBFCKR9EWP5ZRKEKXDE` | `01M2K5HQF2XSRNGB3H6DM7DAJ2` |
| preview | 32k | 120 (24) | `01M2K5JKJKQSKNQVRASNF3CP4S` | `01M2K5JKPZ2A34TYBKTQYWE423` |

All 1 GPU, `urgent`, workspace `ai2/flex2`, budget `ai2/oe-other`, cluster list
`ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2`. JSON is written to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/saliency_preview_<mode>_ds64-outlier-dense-u64M_<rung>_<tag>.json`
and the same table is printed to stdout, so `beaker job logs <job>` is enough to read a run.

⚠ **Every eval here is under the 500-example bar** (240 rows at 2k/8k, 120 at 32k; free generation on
64/48/32 of them). Paired SEs are printed next to every ΔCE and ΔF1 and are the numbers to judge by.

Reproduce (1 GPU):

```
python debug/flop_scaling/beaker_bench_launch.py \
  --cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2" \
  --script debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py \
  --extra "--mode saliency --rung 8k --rows 240 --fit-rows 30 --gen-rows 48 --work /results/w --tag sal8k"
```

## 7. Traps

* **The compaction memo must be invalidated between the preview and the scored forward.** The probe
  compacts once itself and the model compacts again, so the result is memoised on
  `(condition, input data_ptr, length)`. The preview that produces the layer-L states runs with the
  HEADER-ONLY chunk ids, and the scored forward for the same condition runs with a *different*
  override at the same pointer and length — without clearing the memo the second forward is silently
  served the first one's compaction.
* **The attention backend is called with a single positional `(q, k, v)` tuple**
  (`attention/__init__.py:1183`), not three arguments, so a pre-hook that reads `args[0]` as `q`
  gets the tuple. Also `F.scaled_dot_product_attention` never materialises the probabilities, so
  attention mass has to be re-computed for the selected query rows.
* **A document's tokens are not contiguous once headers are FREE** (marker, free header, body), so
  per-document buffers must be indexed by ordinal within the document, never `pos − first[doc]`.
* **RoPE sizes its sin/cos buffer from the cache, not from the `position_ids` it is handed**
  (`rope.py:563-571`), and free generation on a compacted row walks positions past the row's own
  length — warm every RoPE module before the loop or the failure surfaces as an async index assert
  in an unrelated kernel.
