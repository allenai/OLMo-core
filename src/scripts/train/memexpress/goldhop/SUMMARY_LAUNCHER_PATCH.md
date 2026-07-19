# Patch: `summary_attention` flags for the docchunk mask-mix launcher

**Target:** `src/scripts/train/memexpress/attn_explore/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py`
**Status:** NOT applied — that file is held by the coordinator. Anchors below are exact.

Adds `--cross-doc-mode summary_attention` plus `--summary-every-k` / `--summary-bandwidth` /
`--summary-relay`, threaded into `TransformerConfig`. The config layer already accepts all three
(`AttentionConfig` -> `DocumentChunkedAttention` -> `AttentionPattern`); this only exposes them.

Style: written in the file's **hand style** (hanging-indent `ap.add_argument("--x", type=int, default=N,`
+ aligned `help=`) as it exists at **HEAD** — NOT the black-26 layout currently on disk (that layout is
churn from my `make style` run; see "Note on the churn" at the end).

⚠ **This patch does not touch `mix_total_forwards`, the world_size / micro_batch_instances division, or
the curriculum pre-flight assert.** Every anchor is in the `cross_doc_mode` / argparse regions only.

---

## 1. Widen the `--cross-doc-mode` choices

**Anchor (unique)** — replace this exact line:

```python
        choices=["standard", "chunked", "hierarchical_dilated", "random_doc"],
```

**With:**

```python
        choices=["standard", "chunked", "hierarchical_dilated", "random_doc", "summary_attention"],
```

---

## 2. New argparse flags

**Anchor (unique)** — insert **immediately after** this exact line:

```python
    ap.add_argument("--random-doc-seed", type=int, default=42, help="random_doc: RNG seed")
```

**Insert:**

```python
    ap.add_argument("--summary-every-k", type=int, default=10,
                    help="summary_attention: CELL SIZE -- documents per cell. MUST equal the shard's "
                         "--summary-every-k, and the shard must actually carry one summary span per "
                         "cell: the mask identifies a span purely as (chunk_id %% (k+1)) == k, so a "
                         "mismatch silently rebinds every chunk role rather than erroring.")
    ap.add_argument("--summary-bandwidth", type=int, default=0,
                    help="summary_attention: RELAY BANDWIDTH b -- how many of each earlier summary "
                         "span's leading tokens a later chunk may attend. The ladder's dose knob; the "
                         "DATA is identical across rungs, only visibility changes. b=0 removes the "
                         "relay entirely and reproduces a pure cell-blocks mask BIT-EXACTLY (the "
                         "ladder's floor control). Ladder: 0 / 1 / 4 / 16 / 42. b MUST be <= the "
                         "SHORTEST span (42 tokens, measured) or cell 0 silently exposes fewer keys "
                         "than the other cells and the dose axis is not uniform.")
    ap.add_argument("--no-summary-relay", dest="summary_relay", action="store_false",
                    help="summary_attention: PLACEBO -- forbid each summary span from reading its own "
                         "cell. The span keeps its position, its tokens and every edge into it, but "
                         "reads nothing, so it provably carries ZERO document content (verified in "
                         "src/test/nn/attention/summary_attention_test.py). Run at the TOP rung: it "
                         "must land on b=0; if it does not, information-free keys move the metric on "
                         "their own and every rung must be read against it. NOT a hop-inf control.")
    ap.set_defaults(summary_relay=True)
```

---

## 3. Thread into `qwen_kwargs`

**Anchor (unique)** — insert **immediately after** this exact line:

```python
        qwen_kwargs["random_doc_per_example"] = opts.random_doc_per_example
```

**Insert** (a new `elif` continuing the existing `if / elif` chain):

```python
    elif opts.cross_doc_mode == "summary_attention":
        # summary_attention: documents are grouped into CELLS of summary_every_k docs, each followed by
        # its own SUMMARY SPAN chunk. Within a cell attention is full; the span reads its whole cell (it
        # sits at the cell END, so causally it has already seen every doc of the cell) and is then
        # attendable by every LATER cell -> any two docs in different cells are exactly 2 hops apart,
        # with no gold-aware term anywhere. summary_bandwidth throttles how many of each span's leading
        # tokens are visible: that is the ladder's dose axis, and it needs no separate shard.
        qwen_kwargs["summary_every_k"] = opts.summary_every_k
        qwen_kwargs["summary_bandwidth"] = opts.summary_bandwidth
        qwen_kwargs["summary_relay"] = opts.summary_relay
```

---

## 4. Extend the config print (recommended)

**Anchor (unique)** — replace this exact line (last `f"..."` of the `[docchunk-mix]` print):

```python
        f"doc_keep_prob={opts.doc_keep_prob if opts.cross_doc_mode=='random_doc' else None}",
```

**With:**

```python
        f"doc_keep_prob={opts.doc_keep_prob if opts.cross_doc_mode=='random_doc' else None} "
        f"summary_every_k={opts.summary_every_k if opts.cross_doc_mode=='summary_attention' else None} "
        f"summary_bandwidth={opts.summary_bandwidth if opts.cross_doc_mode=='summary_attention' else None} "
        f"summary_relay={opts.summary_relay if opts.cross_doc_mode=='summary_attention' else None}",
```

---

## 5. Record the knobs in the run metadata (optional)

**Anchor (unique)** — insert **immediately after** this exact line:

```python
                    "cross_doc_mode": opts.cross_doc_mode,
```

**Insert:**

```python
                    "summary_every_k": opts.summary_every_k,
                    "summary_bandwidth": opts.summary_bandwidth,
                    "summary_relay": opts.summary_relay,
```

---

## Invocation

One shard (built with `--summary-every-k 10`) serves EVERY rung: bandwidth/relay are model-config knobs,
restored from `config.json` at eval, so no rung needs its own data and no eval flag carries them.

```bash
# pre-flight gate (MANDATORY, run FIRST -- records/multihop-gold-routing-experiment.md §Approach C)
--cross-doc-mode standard                        # must reach CE <~ 0.01 on the summary shard

# the bandwidth ladder
--cross-doc-mode summary_attention --summary-every-k 10 --summary-bandwidth 0    # floor == cell_blocks
--cross-doc-mode summary_attention --summary-every-k 10 --summary-bandwidth 1
--cross-doc-mode summary_attention --summary-every-k 10 --summary-bandwidth 4
--cross-doc-mode summary_attention --summary-every-k 10 --summary-bandwidth 16
--cross-doc-mode summary_attention --summary-every-k 10 --summary-bandwidth 42   # full-span top rung

# placebo (top rung, relay severed) -- predicted to land EXACTLY on b=0
--cross-doc-mode summary_attention --summary-every-k 10 --summary-bandwidth 42 --no-summary-relay
```

Eval: `eval_lc_native_docchunk_contra.py --summary-every-k 10` (must match the shard).

## Checks that apply to every arm

- ⚠ **The `p_standard` anneal must reach 0**, and this ladder is unusually sensitive to it. If the
  curriculum ends at `p_standard=0.601`, ~60% of forwards run **plain causal** -- which hands every
  document a direct cross-cell edge, bypassing the bandwidth gate entirely and making **every rung look
  the same**. A flat ladder is exactly the "genuine null" signature, so this failure would not look like
  a bug; it would look like the result. Your `mix_total_forwards` fix (divide by world_size AND
  micro_batch_instances) + the pre-flight assert must be live for all six arms.
- ⚠ **`--summary-every-k` must match the shard.** The mask only sees `chunk_ids`, so a mismatch is
  silent -- it rebinds every chunk role rather than erroring.
- ⚠ **MAXLEN >= 8192 at eval**; dump generations if any trained arm evals at exactly 0.

## Note on the churn (my fault, your call)

The on-disk launcher currently carries ~66 lines of black-26 reformatting from my `make style` run
(`mix_keys = dict(` split, `ap.add_argument` restructured). Your functional fix survived it.

All five anchors above are **churn-independent** -- each is a single line black did not touch, so they
match verbatim whether or not you revert the churn first. Only the *inserted* text's style would differ from
its neighbours if you keep the churn. Say the word and I will hand you a revert patch as a separate file.
