# Summary-attention relay ladder (n=50, Qwen3-0.6B, contradiction)

Does information flow through a **dedicated 2-hop relay** (doc → summary span → later doc), and how much
relay bandwidth does it need? Approach C of `records/multihop-gold-routing-experiment.md`.

**Setup (identical across arms unless stated).** Data `contra_n50_v2_orig_sum10` — the original
generator's 2000 examples, `--summary-every-k 10` (5 cells of 10 docs, each followed by a summary span),
max_example_len 3411, no-cot. Base `q06b-dense-cpt-modelonly-trainedmark`. seq_len 3456, 3 epochs =
**750 steps**, DP=1, micro-batch 8, LR 5e-5, eager (`--no-compile`). Curriculum mask-mixing on,
`p_standard` 0.8 → **0.001 verified** on all 7 arms. Eval `contradiction_eval_pubmed_both_n50_k3`,
**⚠ eval_size = 488** (the entire file — below the 500 convention, SE quoted inline), parse_rate 1.00 on
every arm, no mask-mixing at eval. **Every arm is a single seed**; seed noise adds to the SEs below.

Reference points from `results/hopgold-n50-stage1.md`: chunked floor **0.408**, full ceiling **0.943**.

`b` = summary **bandwidth**: how many leading tokens of each summary span later cells may attend.
`placebo` = b=42 with the relay severed (`--no-summary-relay`): the span is equally visible but cannot
read its own cell, so it carries no content.

## 1. Result

| arm | CE (mean-50) | f1 | ±SE |
|---|---|---|---|
| preflight (plain causal on this shard) | 0.0069 | **0.958** | ±0.009 |
| b=42 **placebo** (span visible, no content) | 0.1350 | **0.554** | ±0.023 |
| b=0 (≡ `cell_blocks`, span invisible) | 0.0868 | 0.629 | ±0.022 |
| b=1 | 0.0821 | 0.636 | ±0.022 |
| b=4 | 0.0856 | 0.636 | ±0.022 |
| b=16 | 0.0818 | 0.634 | ±0.022 |
| b=42 (full span, cell 0) | 0.0632 | **0.728** | ±0.020 |

**Preflight gate passes:** plain causal reaches CE 0.0069 / f1 0.958 on this shard, so the shard and its
summary spans are fully learnable. This is the check that catches marker bugs — the marker-norm bug
flatlined *every* mask at CE ≈ 0.79 (`records/n100-chunked-marker-position-bug.md`).

**The relay carries real information.** `b=42` (0.728) vs `placebo` (0.554) = **+0.174, ~5.7σ**. This is
the load-bearing contrast: identical token count, identical positions, identical index text — the *only*
difference is whether the span could read its cell. Information demonstrably flows doc → span → later
doc. Against `b=0` (no span at all) the gain is **+0.099, ~4.6σ**. (§4b restricts this contrast to the
cross-cell pairs the relay is actually needed for, where it reads **+0.201**.)

⚠ **`placebo` (0.554) scores BELOW `b=0` (0.629)**, by ~2.4σ. A visible but content-free 42-token span
per cell is *actively harmful* versus not seeing it — it dilutes attention with tokens that cannot help.
So `placebo`, not `b=0`, is the correct null for "does the relay carry content": `b=0` differs from
`b=42` in token budget as well as content, while `placebo` differs only in content.

## 2. ⚠ Design bug: 4 of the 5 bandwidth rungs measure the same condition

The bandwidth values were chosen in **token** units without accounting for the span's header. The span is
`"\n\nSummary of claims 1 to 10: [1] [2] ... [10]"`, whose header runs **11 tokens** (12 for two-digit
cells) before the first index slot:

| arm | tokens visible | **index slots relayed** | f1 |
|---|---|---|---|
| b=0 | 0/42 | **0 of 10** | 0.629 |
| b=1 | 1/42 | **0 of 10** | 0.636 |
| b=4 | 4/42 | **0 of 10** | 0.636 |
| b=16 | 16/42 | ~1–2 of 10 | 0.634 |
| b=42 | 42/42 (cell 0) · **42/52** (cells 1–4) | ~8–10 of 10 | **0.728** |

`b=0/1/4` expose only `\n\n`, `Summary`, ` of`, ` claims` — **zero** index slots. Their f1s (0.629 /
0.636 / 0.636) are statistically identical because they are nearly the same condition. So the apparent
"flat, then a threshold at b=42" is an artifact: **the ladder sampled ~0 slots four times and ~9 slots
once**, and never sampled 2–8 slots at all. In slot units the data read 0 → 0.632, ~1.5 → 0.634, ~9 →
0.728 — entirely consistent with a plain dose-response that was never measured in its interesting range.

⚠ Also note `b=42` is **not** "full span" uniformly: cell 0's span is 42 tokens but cells 1–4 are **52**
(two-digit indices tokenize longer), so `b=42` truncates ~10 tokens from 4 of 5 cells. **No arm in this
ladder ever exposed a full span for every cell.**

**A redo must set bandwidth in slot units** (for n=50/k=10: b ≈ 11 / 17 / 26 / 42 / 54 for ~0 / 2 / 5 /
10 / all slots), or make the span header-free so tokens and slots coincide.

## 3. What is confounded, and what is not

`b=16 → b=42` adds **both** ~26 more relay token-positions **and** ~7 more index-bound `[i]` slots. So
two readings survive this data equally:

* **capacity** — the relay needs many vectors; or
* **addressability** — the relay needs index-bound slots that can each bind to one document, and generic
  header tokens cannot serve as relay nodes regardless of how many there are.

Addressability is tempting (16 residual vectors of width 1024 should hold 10 claims' worth of signal, yet
buy nothing, while ~9 index slots buy +0.09) but **this ladder cannot adjudicate it** — a slot-unit redo
with token-count held fixed can.

⚠ **The relay is via ACTIVATIONS, not text.** `summary_span_text` lists only claim *indices*; it contains
no document content. The span's tokens attend to their own cell and later cells read the resulting hidden
states. The span text is scaffolding, not payload — which is why `placebo` (same text, severed relay) is
a valid null.

## 4. ⚠ This ladder does NOT answer "never a direct edge"

At k=10, **18.4%** of gold pairs (269/1464, measured — an earlier draft of this section estimated ~14%)
land in the **same cell** and attend **directly**. So this measures *relay bandwidth*, not *edge
deletion*, and cannot answer the motivating question ("can a model use information if never given a
direct edge between gold documents?"). Approach A's hop ladder deletes the gold edge by construction and
is the arm that answers it literally.

### 4b. Correction — same-cell edges explain LESS THAN HALF of `b=0` over the floor

This section previously said `b=0`'s 0.629 "is bought by exactly those within-cell direct edges." That
is wrong. Splitting every arm's per-gold-pair recall by whether the pair is same-cell (direct edge at
every `b`) or cross-cell (no doc→doc route except the relay) — script
`debug/gold_hop/summary_arms_by_cell.py`, from the per-example dumps in `attn_explore_logs`:

| arm | f1 | same-cell (n_pairs=269) | cross-cell (n_pairs=1195) |
|---|---|---|---|
| full (preflight) | 0.958 | 0.959 ±0.012 | 0.958 ±0.006 |
| b=42 placebo | 0.554 | 0.866 ±0.021 | 0.484 ±0.014 |
| **b=0** | 0.629 | **0.922 ±0.016** | **0.563 ±0.014** |
| b=1 | 0.636 | 0.926 ±0.016 | 0.571 ±0.014 |
| b=4 | 0.636 | 0.903 ±0.018 | 0.576 ±0.014 |
| b=16 | 0.634 | 0.903 ±0.018 | 0.573 ±0.014 |
| b=42 | 0.728 | 0.914 ±0.017 | **0.685 ±0.013** |

**Two mechanisms, not one.** `b=0` is not `chunked` — it is *cell_blocks*, five ten-document islands
with full attention inside each. Decomposing its +0.221 over the 0.408 floor:

```
0.184 × 0.922  +  0.816 × 0.563  =  0.629   ✓ reconstructs the arm
0.184 × 0.922  +  0.816 × 0.408  =  0.503   ← counterfactual: cross-cell pairs at the floor
```

Same-cell direct edges are worth ≈ **+0.095**; the remaining ≈ **+0.126** is that **cross-cell pairs
also lift (0.563 vs 0.408) despite having no doc→doc route at all**. Most likely mechanism: under
cell_blocks each document's hidden state is contextualized by its 9 cell-mates before the trailing FREE
tokens aggregate it — the same FREE-only channel as `chunked`, but with better-conditioned inputs.

⚠ **That 0.408 is cross-shard.** It was measured on `contra_n50_v2_orig` (no summary spans, DP=2,
seq_len 3200); these arms run `contra_n50_v2_orig_sum10` (DP=1, seq_len 3456). The ceiling moved +0.015
between the two shards (0.943 → 0.958), so a small slice of the 0.155 cross-cell gap may be shard rather
than mechanism. **A `chunked` arm on the sum10 shard would settle it and was never run.**

**This sharpens the headline.** The `b=42` gain lands *entirely* on pairs where the relay is the only
possible route — cross-cell 0.685 vs placebo 0.484 = **+0.201 ±0.019**, while same-cell is flat (0.914 vs
0.866 vs 0.922 across b=42 / placebo / b=0), exactly as it must be for pairs that never needed the relay.
The aggregate +0.174 of §1 understates the effect by averaging in the ~18% of pairs the relay is
irrelevant for.

## 5. Verdict, and the tension with the p=0.25 preview

**With a dedicated relay at ~9 slots, multi-hop routing works: +0.174 over a content-free control.**

This sits in apparent tension with `results/hopgold-n50-connectivity-preview.md`, where a `random_doc`
p=0.25 model showed a multi-hop lift of only **+0.052 [−0.039, +0.144]** — i.e. no detectable routing.
The two are reconcilable, and the reconciliation is the interesting hypothesis:

* the p=0.25 model was **never forced to route** (a direct edge was available for ~25% of gold pairs, and
  "use the edge when present, else fall back to FREE" suffices for its 0.558), and its relays were
  **ordinary documents** with no reason to carry a neighbour's content;
* the summary arms **must** route (no cross-cell direct edges beyond the 18.4% same-cell pairs) and are
  given a **dedicated, addressable** relay node.

⇒ **Routing appears to be learnable but not spontaneously learned** — it needs both training pressure and
a place to put the relayed information. Approach A's `hop2` arm tests the first half directly (forced
routing, but through ordinary documents rather than a dedicated span), which discriminates these.
