# Saliency-selected real tokens, and the one-layer preview — can either reach FULL parity on outlier?

**Date** 2026-09-15 · **Status** RUNNING (9 Beaker jobs — 3 diagnostic, 6 parity; tables filled as rungs land) · **Branch** `prasann/landmark`
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

## 2. The saliency DIAGNOSTIC (first deliverable, `--mode diag`)

Before asking any construction to reproduce FULL, this asks **where the answer's gradient actually
goes** on a row the frozen dense model is reading with full attention. One dense forward, one
gradient-checkpointed backward (`‖∂(mean answer CE)/∂e_t‖` per token), one attention-capture forward
(mass received from the answer + last-16 prompt positions, summed over heads, over all 8 softmax
layers and over the last 4), and — on the first `--gen-rows` rows — FULL's own free generation so
every statistic can be split by whether FULL gets the row RIGHT or WRONG.

Reported, per saliency kind (`grad`, `attn`, `attnlast`) and per split (`all` / `full_right` /
`full_wrong`), with an SE on every cell:

| block | columns |
|---|---|
| (a) where the mass sits | share of total saliency on gold docs, non-gold docs, document headers, `<doc_*>` markers, the question/preamble, the answer region |
| (b) concentration | `n50` / `n90` = how many documents carry 50 % / 90 % of the **doc-side** mass, and the same divided by `n_docs` (the n-dependence between 2k and 8k is the question) |
| (c) within document | header share of a document's mass; then, normalised per document so 1.0 = that document's average token, `z_sent1` vs `z_rest`, the four position quartiles `z_q0..z_q3`, and `z_digit` / `z_capital` / `z_stopword` / the four IDF quartiles `idfq0..idfq3` |
| (d) which documents | `AUCgold` — AUC of per-token saliency separating gold from non-gold documents within the row; `rho_amb` — Spearman of per-document saliency against the **ambiguity score** (negated cosine of the document's content-mean embedding to the row centroid, the readout of `records/outlier-richer-slot-probe.md`); and mean saliency per token on gold / hard-negative / easy documents |
| (e) right vs wrong | all of the above, split by FULL's own set-F1 on the row |
| (f) qualitative | the top-20 gradient-salient tokens of a few rows, each decoded with 4 tokens of left context and tagged by role (`gold[d]` / `doc[d]` / `HDR[d]` / `Q` / `ANS`) |

"Hard negative" is defined without a label: the non-gold documents ranked inside the top `2k` most
topically distant from the row centroid — the ones a distance readout would wrongly pick.

## 3. What is measured (parity constructions)

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

## 4. Constructions

All header-real, gold-blind, remainder pooled into one `cent_cmean` slot per document.

| name | selection of the k real body tokens | deployable? |
|---|---|---|
| `full` | — (plain full attention, the reference) | — |
| `cc00` | none (k = 0) — the known floor | yes |
| `first{k}` | the first k body tokens | yes |
| `idf{k}` | the k highest `-log p(token)` over the training shard | yes |
| `rand{k}` | k uniformly at random — the "any k real tokens" control | yes |
| `gradrow{k}` / `attnrow{k}` / `rulerow{k}` | **ROW-LEVEL budget**: the same total `Σ_d min(k, \|body_d\|)` tokens the uniform selector spends, but handed out row-wide by saliency, so an ambiguous document can take many and a clear-majority document none | `rulerow` yes |
| `grad{k}` | **ORACLE**: top-k by `‖∂(answer CE)/∂e_t‖` from one gradient-checkpointed backward | no |
| `attn{k}` | **ORACLE**: top-k by attention mass received from the answer + last-16 prompt positions, summed over heads and all 8 softmax layers | no |
| `attnlast{k}` | the same over the **last 4** softmax layers only | no |
| `rule{k}` | ridge prediction of the gradient saliency from 6 cheap features, fit on DISJOINT rows | **yes** |
| `grad{k}_swap` | CONTROL: the kept tokens of the gold documents exchanged with random non-gold documents' | — |
| `prev{L}` | no real body tokens; layers `0..L-1` dense over the whole real context, slot = mean of the document's **layer-L** hidden states, layers `L..31` compacted | yes |
| `prev4_k{k}` | `prev4` **plus** the top-k body tokens per document by **layer-3 attention** from the last 32 prompt positions — gradient-free, no second network | **yes** |
| `prev4_row{k}` | the same layer-3 attention budget allocated ROW-WIDE instead of per document | **yes** |
| `prev4_first{k}` | `prev4` plus first-k — the selection control for `prev4_k{k}` | yes |
| `prev4_k{k}_swap` | CONTROL on `prev4_k{k}` | — |

Every construction reports how its budget lands: mean real tokens per **gold**, per **hard-negative**
and per **easy** document, plus the fraction of documents left with no real token at all. Uniform vs
row-level is a comparison **at identical total tokens kept**, which is what makes the "does adaptive
allocation close more of the CE gap" question answerable.

Mechanism: a real-token subset is expressed by marking those positions **FREE** in the chunk ids —
exactly what `mark_doc_headers_free` does to a header — so they survive compaction at their original
positions and are excluded from the slot mean. Only layers 0-2 of the 32 are GatedDeltaNet before the
first softmax layer (index 3), so the preview's gradient-free selector needs **L = 4**; at L = 1 or 2
there is no attention matrix to read and none is synthesised (GDN gate/write magnitudes were not
used — stated here rather than silently substituted).

## 5. Results

### 5a. The saliency diagnostic — 2k rung, COMPLETE (eval_size 160; FULL generation on 64)

`ds64-outlier-dense-u64M`, `outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl`, 13.4 documents
per row, k = 3 gold. ⚠ eval_size 160 (< 500). **FULL answers 62 of 64 generation rows exactly**, so
the right/wrong split has `eval_size = 2` on the wrong side and is quoted only to say it is
unusable at this rung.

**(a) Where the saliency mass sits** (fraction of the row's total):

| saliency | gold docs | non-gold docs | headers | `<doc_*>` markers | question/preamble | answer |
|---|---|---|---|---|---|---|
| `grad` | **0.184** | 0.383 | 0.067 | 0.020 | 0.326 | 0.020 |
| `attn` (all 8 softmax layers) | 0.084 | 0.119 | **0.167** | 0.036 | **0.497** | 0.097 |
| `attnlast` (last 4) | 0.093 | 0.105 | 0.169 | 0.048 | 0.483 | 0.103 |

Gold is 3 of 13.4 documents = 22 % of them, and takes **32 % of the doc-body gradient mass and
41 % of the doc-body attention mass** — a 1.5× / 1.9× over-representation. But the biggest single
consumer is the **question/instruction**, at half of all attention and a third of all gradient, and
the **document headers alone take as much attention (0.167) as every document body combined
(0.203)** — `hdr/doc = 0.451`, i.e. 45 % of a document's attention mass is its `Document [N]:`
header, against only 11 % of its gradient mass. Attention is spent naming documents; gradient flows
through their content.

**(b) Concentration — the model consults nearly every document.**

| saliency | n50 | n90 | n50 / n | n90 / n |
|---|---|---|---|---|
| `grad` | 4.84 | 11.37 | 0.363 | **0.852** |
| `attn` | 4.41 | 11.22 | 0.330 | **0.840** |
| `attnlast` | 3.69 | 10.47 | 0.277 | 0.784 |

Half the doc-side mass sits in a third of the documents, but **90 % of it needs 84–85 % of them**.
There is no small set of documents the model reads and a long tail it ignores.

**(d) Saliency does identify the gold documents — and the embedding ambiguity score does not
explain it.**

| saliency | AUC(gold vs non-gold, per-token) | ρ(saliency, ambiguity) | s/token gold | hard-neg | easy |
|---|---|---|---|---|---|
| `grad` | 0.829 | 0.330 | 0.009 | 0.006 | 0.006 |
| `attn` | **0.934** | 0.314 | **0.762** | 0.314 | 0.325 |
| `attnlast` | **0.945** | 0.348 | 0.419 | 0.141 | 0.142 |

Attention mass per token separates gold from non-gold documents at **AUC 0.93–0.95** with no
training and no gradient — the frozen model's attention already *knows* which documents are odd.
Gold documents get **2.4× the per-token attention** of the rest. Hard negatives (non-gold inside the
top-2k by topical distance from the row centroid) get **no more attention than easy documents**
(0.314 vs 0.325): the embedding-distance notion of "hard" is not the model's.

**(c) Within a document** (normalised per document, 1.0 = that document's average token):

| saliency | 1st sentence | rest | q0 | q1 | q2 | q3 | digit | capitalised | stopword | idf q0 (most frequent) | q1 | q2 | q3 (rarest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grad` | 1.36 | 0.96 | 1.18 | 0.98 | 0.93 | 0.92 | **0.78** | 1.27 | 0.92 | 0.94 | 0.90 | 0.98 | **1.18** |
| `attn` | **1.57** | 0.94 | 1.36 | 0.95 | 0.85 | 0.84 | **1.60** | 1.29 | **1.43** | **1.68** | 1.08 | 0.58 | 0.69 |
| `attnlast` | 1.64 | 0.94 | 1.29 | 0.95 | 0.88 | 0.88 | **1.92** | 1.41 | 1.39 | **1.74** | 0.95 | 0.54 | 0.79 |

Both signals are front-loaded (first sentence ≈ 1.4–1.6×, monotone decline across position
quartiles), which is the one thing `first{k}` has going for it. They **disagree completely on token
type**: attention piles onto digits (1.6–1.9×), stopwords (1.4×) and the most *frequent* IDF quartile
(1.7×) — classic attention sinks plus the id digits — while gradient prefers the *rarest* quartile
(1.18×) and actively avoids digits (0.78×). So `idf{k}` is aligned with the gradient and
anti-aligned with attention.

**(f) The top-20 globally salient tokens are the prompt, not the documents.** In the dumped rows the
20 highest gradient-norm tokens are almost entirely instruction tokens (`given`, `reviews`,
`outliers`, `majority`, `attribute`, `1-indexed`, `document`, `IDs`) and chat scaffolding
(`<|im_start|>assistant`, `<think>`), with one or two document tokens. Every construction here
selects **within a document's body**, so the prompt never competes — but it means a global top-k on
raw gradient norm would be useless, and it is why the within-document normalised profile in (c) is
the number to read.

### 5b. The saliency diagnostic — 8k rung, COMPLETE (eval_size 160; generation on 48, ⚠ < 500)

56.3 documents per row, k = 3 gold (5.4 % of documents). **FULL answers 41 of 48 generation rows
exactly** (0.85), so at this rung the right/wrong split is readable (41 vs 7; ⚠ 7 is tiny).

| saliency | gold | non-gold | header | marker | quest | answ | n50/n | n90/n | hdr/doc | **AUCgold** | ρ(ambig) | s/tok G / H / E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grad` | 0.087 | 0.728 | 0.067 | 0.015 | 0.096 | 0.007 | 0.273 | 0.794 | 0.077 | 0.875 | 0.083 | 0.037 / 0.014 / 0.014 |
| `attn` | 0.047 | 0.206 | 0.190 | 0.035 | 0.414 | 0.107 | 0.282 | 0.793 | 0.428 | **0.964** | 0.073 | 0.457 / 0.114 / 0.113 |
| `attnlast` | 0.064 | 0.199 | 0.180 | 0.046 | 0.397 | 0.113 | 0.203 | 0.713 | 0.406 | **0.970** | 0.113 | 0.304 / 0.058 / 0.054 |

Within-document, normalised (1.0 = the document's average token):

| saliency | 1st sent | rest | q0 | q1 | q2 | q3 | digit | capital | stopword | idfq0 | q1 | q2 | q3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grad` | 1.38 | 0.96 | 1.19 | 0.99 | 0.94 | 0.88 | 0.77 | 1.26 | 0.90 | 0.92 | 0.89 | 0.99 | **1.20** |
| `attn` | 1.38 | 0.96 | 1.23 | 0.97 | 0.92 | 0.89 | **1.68** | 1.15 | **1.45** | **1.63** | 1.18 | 0.59 | 0.62 |
| `attnlast` | 1.37 | 0.98 | 1.12 | 0.97 | 0.97 | 0.94 | **1.97** | 1.26 | 1.46 | **1.73** | 1.09 | 0.54 | 0.66 |

Same shape as 2k and **sharper where it matters**. Gold takes 19 % of the doc-body attention mass
while being 5.4 % of documents (**3.5×**, up from 1.9× at 2k), and AUC(gold) *rises* with n to
**0.964 / 0.970**. The ambiguity correlation collapses from 0.31 to 0.07 — at 8k the topical-distance
readout and the model's attention have nothing to do with each other, and hard negatives again get
no more attention than easy documents (0.114 vs 0.113). Meanwhile the **gradient concentrates on
documents rather than the prompt as n grows**: the question's share falls 0.326 → 0.096 and the
non-gold documents' rises 0.383 → 0.728. `n90/n` stays at **0.79**: with 56 documents the model still
spreads 90 % of its doc-side mass over ~45 of them.

**Right vs wrong (41 vs 7 rows, ⚠ tiny).** On rows FULL gets wrong, AUC(gold) drops for attention
(0.970 → 0.931; `attnlast` 0.974 → 0.934) and the gradient shifts *away* from the question
(0.098 → 0.050) and further onto non-gold documents (0.722 → 0.769), with the concentration slightly
flatter (n50/n 0.283 → 0.297). The direction is "when FULL fails, it failed to single out the gold
documents and spread itself over the distractors", but at 7 rows this is a hypothesis, not a result.

### 5b-32k. The saliency diagnostic — 32k rung, COMPLETE (eval_size 100; generation on 12, ⚠ « 500)

~219 documents per row, k = 3 gold (**1.4 %** of documents). FULL itself answers only **6 of 12**
generation rows exactly, so at this rung FULL is near its own ceiling and the right/wrong split
(6 vs 6) is too small to read.

| saliency | gold | non-gold | header | marker | quest | answ | n50/n | n90/n | hdr/doc | **AUCgold** | ρ(ambig) | s/tok G / H / E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grad` | 0.045 | 0.848 | 0.071 | 0.013 | 0.022 | 0.001 | 0.222 | **0.762** | 0.073 | **0.925** | 0.016 | 0.058 / 0.017 / 0.018 |
| `attn` | 0.028 | 0.258 | **0.217** | 0.031 | 0.351 | 0.117 | 0.242 | 0.753 | 0.432 | **0.965** | 0.054 | 0.279 / 0.038 / 0.037 |
| `attnlast` | 0.038 | 0.278 | 0.197 | 0.040 | 0.329 | 0.119 | 0.178 | 0.677 | 0.384 | 0.963 | 0.081 | 0.191 / 0.022 / 0.020 |

Within-document, normalised (1.0 = the document's average token):

| saliency | 1st sent | rest | q0 | q1 | q2 | q3 | digit | capital | stopword | idfq0 | q1 | q2 | q3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grad` | **1.44** | 0.95 | 1.24 | 0.99 | 0.92 | 0.86 | **0.72** | 1.35 | 0.86 | 0.86 | 0.86 | 1.01 | **1.27** |
| `attn` | 1.29 | 0.98 | 1.14 | 0.96 | 0.96 | 0.94 | **1.75** | 1.18 | 1.41 | **1.57** | 1.17 | 0.62 | 0.66 |
| `attnlast` | 1.23 | 0.99 | 1.04 | 0.97 | 1.00 | 0.99 | **2.00** | 1.35 | 1.41 | **1.63** | 1.11 | 0.58 | 0.70 |

The trends from 2k → 8k all continue, and two of them sharpen decisively:

* **Gold separation keeps improving with n.** AUC(gold) `grad` 0.829 → 0.875 → **0.925**, `attn`
  0.934 → 0.964 → **0.965**. Gold documents are 1.4 % of the row but take 5.0 % of the doc-body
  gradient mass and 9.8 % of the doc-body attention mass — **3.6× / 7.0× over-represented** — and get
  **7.4× the per-token attention** of everything else (0.279 vs 0.038 / 0.037).
* **The gradient is now essentially all document** (non-gold 0.848, question 0.022, answer 0.001) and
  its within-document profile is at its sharpest: rarest IDF quartile 1.27×, digits 0.72×, first
  sentence 1.44×.
* **`n90/n` falls only to 0.75.** Even with 219 documents, 90 % of the doc-side mass needs 165 of
  them. This is the number that bounds every construction here.
* **The ambiguity score is now noise** (ρ = 0.016–0.081) and hard negatives are indistinguishable
  from easy ones (0.038 vs 0.037). The topical-distance notion of "hard" can be retired for outlier.
* **Headers take 0.217 of all attention at 32k** — more than every document body combined (0.286 ×
  0.76 excluded) and rising with n (0.167 → 0.190 → 0.217).

### 5f. CONCENTRATION vs n — does saliency sparsify as the corpus grows? **No.**

The hypothesis worth killing explicitly: at 32k the model might consult far fewer of the ~219
documents than the ~80 % it consults at 8k, which would make a "keep a few documents" construction
viable. It does not.

| rung | docs/row | gold share of docs | **n50** | **n50/n** | **n90** | **n90/n** | AUC(gold) attn | AUC(gold) grad | gold share of doc-body mass (attn) | s/tok gold ÷ non-gold (attn) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 13.4 | 22 % | 4.41 | 0.330 | 11.22 | **0.840** | 0.934 | 0.829 | 41 % (1.9×) | 2.4× |
| 8k | 56.3 | 5.4 % | 15.86 | 0.282 | 44.61 | **0.793** | 0.964 | 0.875 | 19 % (3.5×) | 4.0× |
| 32k | ~219 | 1.4 % | 53.08 | 0.242 | 164.96 | **0.753** | 0.965 | 0.925 | 9.8 % (7.0×) | 7.4× |

(`attnlast`, the last 4 softmax layers, is the most concentrated variant and still only reaches
`n90/n` 0.784 → 0.713 → 0.677. `grad` tracks `attn` within 0.01 on both fractions at every rung.)

Read it as two facts pulling in opposite directions:

* **In RELATIVE terms concentration barely moves**: `n90/n` 0.840 → 0.793 → 0.753 across a 16×
  increase in document count. Fitting `n90 ∝ n^α` over the three points gives **α ≈ 0.96** — the
  number of documents carrying 90 % of the mass grows very nearly *linearly* with n (11 → 45 → 165).
  Saliency does not sparsify; it dilutes.
* **In TARGETING terms the model gets much better with n**: gold's over-representation in doc-body
  mass rises 1.9× → 3.5× → **7.0×**, per-token attention on gold vs the rest 2.4× → 4.0× → **7.4×**,
  and AUC(gold) 0.934 → 0.964 → 0.965 for attention and 0.829 → 0.875 → **0.925** for the gradient.

So the frozen model is *increasingly* good at saying **which** documents matter, while needing
*proportionally as many* of them to answer. That is precisely why a per-document real-token budget
(§5d) is the live idea and a document-selection scheme is not — and it is the reason the row-level
allocator underperforms at 2k despite correctly concentrating on gold: concentration is not the
lever, per-document detail is.

**Caveat.** `n50`/`n90` are computed on the saliency *distribution*, which measures where signal
flows in a model answering correctly — not the minimum set that would suffice. A causal ablation
(drop the bottom-mass documents and re-score) would bound the latter; it is not run here. §5g is the
cheap calibration instead: the same measurement on a task whose answer provably depends on 2–3
documents.

### 5c. What this implies for the constructions

1. **A per-document attention BUDGET is the signal, not the per-token identity.** AUC(gold) 0.93–0.97
   says document-level saliency is highly informative; (c) says the individual top-attention tokens
   are sinks (digits, stopwords, most-frequent quartile). That is the case for the **row-level
   budget** variants (`gradrow*`, `attnrow*`, `rulerow*`, `prev4_row*`) over the uniform per-document
   top-k, and it is why they were added.
1b. **The two saliencies are not interchangeable, and the *gradient* is the one that tracks the
   task as n grows** (question share 0.33 → 0.09 from 2k to 8k, non-gold document share 0.38 → 0.73),
   while attention keeps half its mass on the question and the headers at both lengths. But
   attention is the one with the higher AUC(gold). Read `grad*` and `attn*` in the parity tables as
   two different hypotheses, not two estimates of one thing.
2. **`n90/n ≈ 0.79–0.85` is the bad news.** No construction that keeps a few documents can work;
   whatever is kept has to be spread over almost all of them.
3. **Headers are already carrying 45 % of a document's attention** — the header-real baseline
   (`cc00`) is not a small concession, it is most of what attention was doing with that document.
4. **`idf{k}` and `attn{k}` will select nearly disjoint tokens** (IDF quartile profiles are inverted),
   so the `overlap@k with grad` column in the parity tables is the thing to read alongside ΔCE.

### 5d. Parity constructions — 2k rung (INTERIM, eval_size 5; ⚠ shape check ONLY, do not quote)

First table off the `sal2k` job. FULL CE 0.003, genF1 1.000. Read the *ordering*, not the values.

| condition | CE | ΔCE | genF1 | tok/doc real | FLOPfrac | gold / hard / easy tokens | docs with 0 |
|---|---|---|---|---|---|---|---|
| `cc00` (k = 0) | 0.470 | +0.467 | 0.067 | 0 | 0.101 | — | — |
| `rand8` | 0.558 | +0.555 | 0.267 | 8 | 0.151 | 8 / 8 / 8 | 0.00 |
| `first8` | 0.263 | +0.260 | 0.533 | 8 | 0.151 | 8 / 8 / 8 | 0.00 |
| `idf8` | 0.262 | +0.259 | 0.333 | 8 | 0.151 | 8 / 8 / 8 | 0.00 |
| `attn8` | 0.319 | +0.317 | 0.400 | 8 | 0.151 | 8 / 8 / 8 | 0.00 |
| **`grad8`** (oracle) | **0.072** | +0.069 | 0.800 | 8 | 0.151 | 8 / 8 / 8 | 0.00 |
| **`rule8`** (deployable) | **0.071** | +0.068 | **0.867** | 8 | 0.151 | 8 / 8 / 8 | 0.00 |
| `grad16` (oracle) | 0.037 | +0.034 | 0.800 | 16 | 0.202 | 16 / 16 / 16 | 0.00 |
| `rule16` | 0.046 | +0.043 | 0.800 | 16 | 0.202 | 16 / 16 / 16 | 0.00 |
| `gradrow8` (row budget) | 0.300 | +0.297 | 0.467 | 8 | 0.151 | **13.9 / 4.5 / 7.4** | 0.22 |
| `attnrow8` | 0.288 | +0.285 | 0.200 | 8 | 0.151 | **16.9 / 6.1 / 5.1** | 0.01 |
| `rulerow8` | 0.106 | +0.103 | 0.667 | 8 | 0.151 | 6.7 / 8.9 / 8.0 | 0.00 |
| **`grad16_swap`** (control) | 0.911 | +0.909 | **0.000** | 16 | 0.202 | 16 / 16 / 16 | 0.00 |

Three things are already visible and are the reason the full runs are worth waiting for.

* **The swap control fires hard** — `grad16` 0.037 / genF1 0.800 versus `grad16_swap` 0.911 / genF1
  **0.000**. Unlike every slot construction in the two previous records, the model is unambiguously
  *reading* the kept tokens.
* **`rand8` is worse than keeping nothing** (0.558 vs `cc00` 0.470): eight arbitrary real tokens per
  document are an active distraction. Which tokens are kept is the whole effect.
* **The transferable rule matches the oracle** (`rule8` 0.071 vs `grad8` 0.072; `rule16` 0.046 vs
  `grad16` 0.037) from six free features — and it is not doing what attention does (`attn8` 0.319).
* **The row-level budget is behind the uniform one at 2k**, on all three saliencies
  (`gradrow8` 0.300 vs `grad8` 0.072), even though it does concentrate on the gold documents
  (13.9 tokens/gold vs 4.5/hard, 7.4/easy) — because it strands 22 % of documents with no real token
  at all, and (b) of the diagnostic says 90 % of the model's mass needs 84 % of the documents. Whether
  this reverses at 8k/32k, where the per-document budget is the binding constraint, is exactly what
  the running jobs answer.

### 5e. The one-layer preview — 2k rung (INTERIM, eval_size 5; ⚠ shape check ONLY, do not quote)

FULL CE 0.003 / genF1 1.000; `cc00` (no preview, plain `cent_cmean` slot) CE 0.469 at FLOP 0.101.

| condition | CE | ΔCE | genF1 | real tok/doc | compaction | **FLOPfrac** |
|---|---|---|---|---|---|---|
| `cc00` | 0.469 | +0.466 | 0.067 | 0 | 0.103 | 0.101 |
| `prev0` (slot = layer-0 input mean) | 0.456 | +0.453 | 0.067 | 0 | 0.103 | 0.101 |
| `prev1` | 0.514 | +0.511 | 0.067 | 0 | 0.103 | **0.129** |
| `prev2` | 0.560 | +0.557 | 0.067 | 0 | 0.103 | **0.156** |
| `prev4` | 0.471 | +0.468 | 0.133 | 0 | 0.103 | **0.213** |
| `prev4_k4` | 0.446 | +0.443 | 0.133 | 4 | 0.128 | 0.235 |
| `prev4_k8` | 0.444 | +0.441 | 0.333 | 8 | 0.154 | 0.257 |
| `prev4_k16` | 0.078 | +0.075 | 0.800 | 16 | 0.205 | 0.301 |
| `prev4_row8` | 0.433 | +0.431 | 0.333 | 8 (10.7 / 7.5 / 7.6) | 0.154 | 0.257 |
| `prev4_row16` | 0.279 | +0.276 | 0.533 | 16 (20.5 / 13.9 / 15.7) | 0.205 | 0.301 |
| `prev4_first8` | **0.227** | +0.224 | 0.467 | 8 | 0.154 | 0.257 |
| `prev4_k8_swap` (control) | 0.624 | +0.621 | 0.200 | 8 | 0.154 | 0.257 |

**The dense preview buys nothing on outlier, and it is not free.** `prev0` / `prev1` / `prev2` /
`prev4` are all sitting on the `cc00` floor (0.456 / 0.514 / 0.560 / 0.471 against 0.469), with no
monotone improvement in L, while the FLOP fraction climbs 0.101 → 0.129 → 0.156 → **0.213**. So
contextualising a document's slot across the whole real row — the construction that reached CE
parity on contradiction (`records/layer-soft-probe.md` §2, `fromL` + `layer_input_mean` with headers
real) — **does not transfer to outlier**. That is consistent with the diagnostic: outlier's problem
is not that each document's summary lacks context, it is that the answer needs per-document detail
from ~80 % of the documents at once (`n90/n` 0.79–0.85).

**Layer-3 attention is a worse token selector than "the first k".** `prev4_k8` 0.444 vs
`prev4_first8` **0.227** at identical cost. This is the diagnostic's §5c point 1 landing exactly:
attention mass identifies *which document* matters (AUC 0.96) but its top tokens *inside* a document
are sinks — digits 1.7×, stopwords 1.5×, most-frequent IDF quartile 1.6× — so selecting by it picks
uninformative tokens.

**Whatever `prev4_k16` achieves, the real-token route achieves more cheaply.** `prev4_k16` reaches
CE 0.078 / genF1 0.800 at FLOP **0.301**; `grad16` (§5d) reaches 0.037 / 0.800 and `rule16` 0.046 /
0.800 at FLOP **0.202**, with no preview at all. The preview's four dense layers are pure overhead
here.

The swap control fires on the preview path too (`prev4_k8` 0.444 → `prev4_k8_swap` 0.624), so the
model is reading the kept tokens rather than reacting to their presence.

## 6. Verdict

_(pending the full rungs — the interim tables above are 5 rows each)_

Direction of travel at 2k, to be confirmed at 8k/32k:

* **Idea 2 (one-layer preview) looks dead on outlier** — no L improves on `cc00`, and every L costs
  more. Unlike contradiction, there is no fidelity for a dense prefix to recover.
* **Idea 1 is alive, and its deployable form is the winner so far** — `rule{k}`, a ridge over six
  free token features fit on disjoint rows, matches the gradient ORACLE (`rule8` 0.071 vs `grad8`
  0.072; `rule16` 0.046 vs `grad16` 0.037) and beats `first`, `idf`, `attn` and `rand` at equal cost.
* **Uniform per-document top-k beats the row-level budget** at 2k on every saliency, despite the
  row-level one correctly concentrating on the gold documents — stranding ~22 % of documents with no
  real token costs more than the concentration gains, which is what `n90/n ≈ 0.85` predicts.
* **Neither has yet reached parity**: the best 2k cell is ΔCE +0.034 against a FULL CE of 0.003, and
  genF1 0.80–0.87 against 1.00. The question the full runs answer is whether that gap closes with k,
  and what it costs at 8k and 32k.

## 7. Runs

| mode | rung | rows (gen) | experiment | job |
|---|---|---|---|---|
| **diag** | 2k | 160 (64) | `01M2K605TCR7SSTFZZ1AY9TSG2` | `01M2K605Y2ZGZPY8BV343F8N8N` |
| **diag** | 8k | 160 (48) | `01M2K610VSTPHRR5MZZEW64074` | `01M2K610ZEA7K54W0K6VW2H2X0` |
| **diag** | 32k | 100 (12) | `01M2K6242QKWHG30KWQVXPNTC7` | `01M2K6246KNV307TEBB0CG6E4V` |
| saliency | 2k | 240 (64) | `01M2K635MBBZ2N69BZMY8JATA7` | `01M2K635QYRBDP3YEKSQPWTF45` |
| saliency | 8k | 240 (48) | `01M2K64D1ZN2ER88SA0AE8FKYE` | `01M2K64D5N83YNH11C14WFE5XQ` |
| saliency | 32k | 120 (24) | `01M2K65D5DR4KBE5Y1D1QKX9V6` | `01M2K65D9RMC441GQYKQAGY0AG` |
| preview | 2k | 240 (64) | `01M2K671PHM5P5M5DE4Y2QRW6G` | `01M2K671T0M0YNZ58JS7YRQZN0` |
| preview | 8k | 240 (48) | `01M2K6817RT1JHX0SE6GNV6M87` | `01M2K681B9KDA8BTHXGWBEFC2G` |
| preview | 32k | 120 (16) | `01M2K698PJH9C1GSQSGT4T3DZT` | `01M2K698T64HVZV4SANJE14XGJ` |

(An earlier batch of the same six parity jobs, `01M2K5DN…`–`01M2K5JK…`, was cancelled while still
queued so the row-level budget variants could be added before anything ran.)

All 1 GPU, `urgent`, workspace `ai2/flex2`, budget `ai2/oe-other`, cluster list
`ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2`. JSON is written to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/saliency_preview_<mode>_ds64-outlier-dense-u64M_<rung>_<tag>.json`
and the same table is printed to stdout, so `beaker job logs <job>` is enough to read a run.

⚠ **Every eval here is under the 500-example bar** (diag 160/160/100; parity 240 rows at 2k/8k, 120 at 32k; free generation on
64/48/32 of them). Paired SEs are printed next to every ΔCE and ΔF1 and are the numbers to judge by.

Reproduce (1 GPU):

```
python debug/flop_scaling/beaker_bench_launch.py \
  --cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2" \
  --script debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py \
  --extra "--mode saliency --rung 8k --rows 240 --fit-rows 30 --gen-rows 48 --work /results/w --tag sal8k"
```

## 8. Traps

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
