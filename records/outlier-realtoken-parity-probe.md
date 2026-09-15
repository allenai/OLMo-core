# Can a REAL-TOKEN subset per document reach frozen-model parity on outlier?

**Date** 2026-09-15 · **Status** RUNNING (jobs listed in §6) · **Branch** `prasann/landmark`
**Driver** `debug/pooled_kv/outlier_probe/outlier_realtoken_probe.py`

## 1. The question

`records/outlier-slot-probe.md` and `records/outlier-richer-slot-probe.md` between them close the
*slot-vector* route for outlier. Thirteen constructions — `mean`, `meanrn`, `cmean100`, `cmean500`,
`centered`, `cent_cmean`, `idf`, `g2`, `g4`, `g2cent`, `enc2`, `enc4`, `enc4late` — all leave the
frozen dense model's `R@gold_pooled` inside ±0.041 of the `k/n` uniform-guess floor, on two
checkpoints, two corpora and two lengths, with the slot-swap control undetectable. An *oracle*
cosine readout recovers 0.73 of gold from `cent_cmean` at 2k, so the signal is in the vector; the
frozen model simply does not read it and falls back on the id-position prior.

`records/soft-kv-slot-probe-handoff.md` records what *did* reach parity on contradiction and oolong,
and it is not a richer vector: keep each document's **HEADER** real (the exact-match tokens) and pool
only the body — `--st-header-stop-id 25`, ΔCE ≈ 0 on the frozen model, 4.8x/11x compaction on
contradiction and 1.4x on oolong. That is a **real-token subset**.

So this probe asks the same question for outlier with a bigger, still-`O(k)`-per-document real-token
subset:

> keep `k` REAL body tokens per document at their ORIGINAL positions, pool (or drop) the rest.

Goal: the **cheapest** construction under which the frozen dense model reaches FULL-attention parity
— ΔCE ≤ 1 paired SE, and genF1 within noise of FULL — at 2k, 8k and 32k.

## 2. Mechanism (why this is a one-line change, not a new model)

`Transformer._compact_pooled_soft_tokens` builds chunk ids from the token stream, then — when
`header_stop_id` is set — calls `mark_doc_headers_free` to re-label the header **FREE**. FREE tokens
survive compaction **at their original positions** and are **excluded from the pooled slot mean**. So
"keep an arbitrary subset of a document's tokens real" is exactly "mark that subset FREE", and every
candidate here is a chunk-id override plus the trainer's own pooling path. `model.py` imports
`mark_doc_headers_free` *inside* the forward, so patching the module attribute reaches both the
probe's own compaction and the model's (they must agree, or the gathered columns go out of bounds
and CUDA reports it asynchronously inside an unrelated kernel).

Each candidate costs `O(doc tokens)` to build from the document's own tokens plus a corpus
token-frequency table precomputed once per corpus. Nothing per-layer, nothing per-token-per-layer,
no second network.

## 3. Constructions

Gold-blind everywhere unless stated (gold documents get exactly the same treatment as non-gold), and
the `\n\nDocument [N]:` header stays real (stop id 5491, count 1) so the ids remain nameable.

| name | kept real per document | remainder |
|---|---|---|
| `full` | everything | — |
| `cc00` | header only | one `cent_cmean` slot — the known-at-the-floor control |
| `first{k}` | header + first `k` body tokens (`k` = 4, 8, 16, 32, 64) | one `cent_cmean` slot |
| `first{k}d` | the same subset | **DROPPED** (no slot) — does the slot add anything? |
| `idf{k}` | header + the `k` body tokens with the highest `−log p(token)` (non-contiguous) | slot |
| `idfspan{k}` | header + the contiguous span of length `k` with the highest summed idf | slot |
| `fl{k}` | header + first `k/2` + last `k/2` body tokens | slot |
| `sent1` | header + the first sentence (to the first `.`/newline piece, cap 32 tokens) | slot |
| `<cond>_swap` | CONTROL: the kept real tokens of the gold documents exchanged with those of random non-gold documents | unchanged |
| `goldonly` | gold documents **fully** real, no header on the rest | `mean` slot |
| `goldonly_cc` | the same | `cent_cmean` slot |

`goldonly` is gold-**aware** and is not a training recipe (gold-forcing on an id-answer task is the
known shortcut). It is here as the reference for "the frozen model CAN do outlier when only the gold
bodies are real", because `records/soft-kv-slot-probe-handoff.md` (2026-09-08, 24 rows, 32k,
`lmx-full-mixs160M-4b`) claims **answer-CE parity** for it — outlier 1.201 gold-only vs 1.206 full —
while `records/outlier-slot-probe.md` (2026-09-14, ds64 2k, eval_size 240) measures CE 0.194 vs full
0.010 and genF1 0.318 vs 0.979. §5 settles which is right on the canonical pairing.

The `_swap` control is the honest test of "is the model reading the kept tokens?": positions, ids,
headers and slot means are all untouched (the kept tokens are FREE, so they never enter a slot mean),
so only *what is legible in the gold documents* changes. Recall must drop.


## 3b. How an outlier example is actually built (and why "smallest category" is the task)

Read off the generators — `src/corpus_reasoning/data/generate_wiki_outlier_data.py`,
`src/corpus_reasoning/data/build_v2_outlier_ladder.py`, sampling in
`src/corpus_reasoning/lib/wiki100w_pool.py` — this is the structure every construction below is
working against:

* **A "topic" is one Wikipedia article.** Majority documents are contiguous ~100-word chunk runs
  drawn from whole articles picked **uniformly at random** (`ArticlePool.sample_run`,
  `wiki100w_pool.py:235-272`); there is **no topical-coherence constraint** tying the majority
  articles to each other. Titles are stripped (`documents[i]["title"] = None`).
* **The gold documents are `k` = 3 contiguous chunks of one further random article**, excluded only
  by title (`outlier = pool.sample_run(num_outliers, rng=rng, exclude_titles=...)`). So all three
  outliers share ONE topic.
* **Hard negatives are not deliberate.** There is no embedding or BM25 similarity filter, no
  cross-encoder, no near-duplicate control anywhere in the outlier path. The only "margin" in the
  code (`build_v2_outlier_ladder.py:113`) is a *count* gap, not a semantic one. Difficulty is
  purely structural: more documents, more topics, longer context.
* **The outlier topic is by construction the STRICTLY SMALLEST category in the row.** Every majority
  article is required to contribute at least `num_outliers + maj_outlier_gap` (default 3 + 1 = 4)
  documents, while the outlier article contributes exactly 3. This invariant is enforced in both
  generators (`build_v2_outlier_ladder.py:77-97`, `generate_wiki_outlier_data.py:240-248`).
* **Number of majority topics grows with the rung**: measured K ≈ 3 / 7 / 13 / 25 at n = 22 / 55 /
  110 / 220 documents (`generate_wiki_outlier_data.py:195-208`), i.e. at the 2k / 8k / 16k / 32k
  rungs. Outlier positions are shuffled uniformly into the list.
* ~147 tokens per document (`tokens ≈ 114.1 + 146.93·n`).

Two consequences for this probe:

1. **`smallcat` is not a heuristic, it is the task's own generative rule read backwards.** "Keep
   every document in the C smallest topical categories" can in principle have rule recall 1.0 at
   C = 1, and its *only* failure mode is the clustering, not the rule. That is why `rule_recall`
   (the fraction of gold documents the rule keeps) is reported as the construction's ceiling.
2. **A "preserve the hard negatives" story has no generative support here.** Nothing in the builder
   creates near-miss distractors, so `margin{M}` and the oracle `hardneg{M}` are measurements of
   whether *accidental* near-misses matter, not of a designed adversarial set. The measured
   `cos_gold` / `cos_other` margin and `hn_rate` in §5 quantify how often one happens by chance.

## 3c. The category-level keep rules

All computed from the per-document `cent_cmean` vector (mean input embedding over content tokens,
minus the row centroid, normalised) — embedding arithmetic only, `O(doc tokens)`, no per-layer cost.
Documents are clustered by **average-linkage agglomerative clustering on cosine distance**, cut at
the largest relative gap in the merge-distance sequence among cuts leaving 2..12 clusters
(parameter-free; validated on synthetic data to recover the true partition whenever the clusters are
separable).

| name | rule | gold-blind? |
|---|---|---|
| `smallcat{C}` | every document in the **C smallest clusters** kept WHOLE, the rest pooled (one `cent_cmean` slot per pooled document). C = 1, 2, 3, 5 | yes |
| `smallcat{C}cat` | the same, but **one slot per pooled CLUSTER** (mean over the cluster's documents) — fewer slots | yes |
| `smallcat{C}f{k}` | the same, plus the **first k tokens** of every pooled document real | yes |
| `smallcatle{s}` | every cluster of size ≤ s kept whole (s = 2, 3, 5) | yes |
| `smallcat{C}d{D}` | plus **D random LARGE clusters** kept whole, so "kept whole" no longer implies "small" (D = 1, 2) | yes |
| `margin{M}` | the M documents **farthest from a robust centroid** (mean of the 75 % closest, renormalised) **∪** the M documents nearest the k-outlier decision boundary `τ` = midpoint between the 3rd and 4th lowest cosine. M = 3, 6, 10 | yes |
| `margin{M}f16` | the same, plus first-16 tokens on the pooled documents | yes |
| `hardneg{M}` | **ORACLE**: gold documents POOLED, the M non-gold documents with the highest cosine to a gold document kept real | **no** |
| `goldcats{K}` | **GOLD-AWARE**: the gold document's whole category kept real, plus the **K smallest** other categories whole (K = 2, 4); everything else pooled | **no** |
| `goldcats{K}r` | the same with **K random** other categories | **no** |
| `gpr33` | the old `gold_plus_random` construction: gold real + a random 1/3 of the non-gold documents real | **no** |

Categories are kept whole or pooled whole in every `smallcat`/`goldcats` variant — never partially.

`gpr33` carries the **mechanistic diagnostic** for the hypothesis that gold-forcing collapsed
because the gold category was the only one kept WHOLE: for every row the probe measures each
cluster's completeness in the kept set and reports `n_cats_full` (how many categories are fully
real) and `gold_only_full` (the gold category is the only fully-real one). `smallcat{C}d{D}` is the
same idea from the other side — it deliberately keeps large categories whole too.

## 4. Metrics

Per condition × rung: answer CE with the **paired** ΔCE vs FULL and its SE; CE on the **digit**
tokens of the answer (mean answer CE on outlier is ~95% prose — `records/outlier-slot-probe.md` §4 —
so the CE column alone ranks constructions almost backwards); free-generation set-F1 over the k
document ids; recall of gold documents split by whether that document's body was pooled/dropped
(`R@gp`) or real (`R@gr`); real tokens kept per document on the compacted row (header + body + slot);
compaction; and the linear-term FLOP fraction ≈ compaction. For the category rules also
`rule_recall` (the fraction of gold documents the keep rule places in the kept set — the ceiling of
that construction), `kept_docs`, and the per-row corpus structure: number of clusters, smallest /
largest cluster size, `gold_in_smallest`, `cos_gold` vs `cos_other`, `hn_rate` (some non-gold
document sits farther from the centroid than a gold one) and `oracle_cosR` (recall of "the 3
lowest-cosine documents are the outliers").

**Parity rule:** ΔCE ≤ 1 paired SE **and** |ΔgenF1| ≤ 1 paired SE, at 8k and 32k.

## 5. Results

### 5a. Canonical pairing — **COMPLETE**: 2k (eval_size 240) and 8k (eval_size 200)

`ds64-outlier-dense-u64M` (frozen, full-attention-trained) on the ds64 outlier rungs. ⚠ eval_size
240 / 200 (< 500); generation on the first 64 / 48 rows, so the genF1 and `R@gold_pooled` columns
average over **192 / 144 gold documents** (per-document SE ≈ 0.03 / 0.04). Paired SEs printed.

**2k** — 13.4 documents/row, `k/n` guess floor **0.224**. Jobs `01M2K3DHJZW7QV690ES88R0HB6`.

| condition | ΔCE (SE) | ΔCE(digits) (SE) | genF1 | ΔgenF1 (SE) | **R@gold_pooled** | tok/doc | compaction |
|---|---|---|---|---|---|---|---|
| `full` | — | — | **0.979** | — | — | 144 | 1.000 |
| `cc00` (slot only — the old floor) | +0.454 (.008) | +1.676 (.037) | 0.224 | −0.755 (.036) | **0.224** = `k/n` | 6.3 | 0.105 |
| `goldonly` (gold-aware) | +0.184 (.004) | +0.561 (.017) | 0.318 | −0.661 (.055) | — (R@gold_real 0.318) | 34.4 | 0.284 |
| `goldonly_cc` | +0.193 (.004) | +0.592 (.017) | 0.328 | −0.651 (.054) | — (0.328) | 34.4 | 0.284 |
| `first4` | +0.381 (.012) | +1.436 (.049) | 0.364 | −0.615 (.043) | 0.364 | 10.3 | 0.130 |
| `first8` | +0.269 (.013) | +1.006 (.051) | 0.552 | −0.427 (.051) | 0.552 | 14.3 | 0.156 |
| `first16` | +0.152 (.012) | +0.571 (.046) | 0.661 | −0.318 (.054) | 0.661 | 22.3 | 0.207 |
| `first32` | +0.049 (.006) | +0.181 (.024) | 0.818 | −0.161 (.048) | 0.818 | 38.3 | 0.309 |
| `first64` | +0.020 (.005) | +0.073 (.020) | 0.872 | −0.107 (.040) | 0.872 | 70.3 | 0.514 |
| `first32d` (no slot) | +0.049 (.006) | +0.184 (.024) | 0.854 | −0.125 (.044) | 0.854 | 37.3 | 0.303 |
| `first64d` (no slot) | **+0.013** (.004) | **+0.048** (.017) | 0.901 | −0.078 (.039) | 0.901 | 69.3 | 0.508 |
| `idf16` | +0.113 (.009) | +0.411 (.038) | 0.724 | −0.255 (.046) | 0.724 | 22.3 | 0.207 |
| `idfspan16` | +0.130 (.011) | +0.491 (.045) | 0.784 | −0.195 (.045) | 0.784 | 22.3 | 0.207 |
| `fl16` | +0.124 (.009) | +0.463 (.036) | 0.776 | −0.203 (.046) | 0.776 | 22.3 | 0.207 |
| **`fl32`** | **+0.044** (.007) | **+0.162** (.028) | **0.927** | **−0.052** (.031) | **0.927** | 38.3 | **0.309** |
| `sent1` | +0.202 (.013) | +0.762 (.049) | 0.630 | −0.349 (.051) | 0.630 | 23.9 | 0.217 |
| `first8_swap` | +0.651 | +2.477 | 0.156 | −0.823 | 0.156 | 14.3 | 0.156 |
| `first16_swap` | +0.809 | +3.101 | 0.099 | −0.880 | 0.099 | 22.3 | 0.207 |
| `first32_swap` | +0.948 | +3.636 | **0.026** | −0.953 (.021) | 0.026 | 38.3 | 0.309 |
| `first64_swap` | +1.042 | +3.997 | 0.044 | −0.935 | 0.044 | 70.3 | 0.514 |
| `idfspan16_swap` | +0.854 | +3.257 | 0.042 | −0.938 | 0.042 | 22.3 | 0.207 |
| `sent1_swap` | +0.773 | +2.954 | 0.219 | −0.760 | 0.219 | 23.9 | 0.217 |

**8k** — 56.3 documents/row, `k/n` floor **0.053**. Job `01M2K3EE89VP13HPV3MHM682P5`.

| condition | ΔCE (SE) | ΔCE(digits) (SE) | genF1 | ΔgenF1 (SE) | **R@gold_pooled** | tok/doc | compaction |
|---|---|---|---|---|---|---|---|
| `full` | — | — | **0.868** | — | — | 146 | 1.000 |
| `cc00` | +0.693 (.010) | +2.006 (.031) | 0.062 | −0.806 (.048) | **0.062** ≈ floor 0.053 | 6.8 | 0.062 |
| `goldonly` | +0.240 (.008) | +0.648 (.023) | **0.049** | −0.819 (.052) | — (R@gold_real 0.049) | 8.9 | 0.076 |
| `goldonly_cc` | +0.262 (.009) | +0.712 (.026) | 0.056 | −0.812 (.052) | — (0.056) | 8.9 | 0.076 |
| `first8` | +0.549 (.019) | +1.622 (.057) | 0.139 | −0.729 (.060) | 0.139 | 14.8 | 0.116 |
| `first16` | +0.361 (.017) | +1.061 (.051) | 0.208 | −0.660 (.062) | 0.208 | 22.8 | 0.170 |
| `first32` | +0.204 (.015) | +0.592 (.044) | 0.319 | −0.549 (.069) | 0.319 | 38.8 | 0.277 |
| `first64` | **+0.065** (.008) | **+0.190** (.024) | **0.603** | −0.265 (.068) | 0.597 | 70.8 | 0.492 |
| `first16d` | +0.389 (.019) | +1.145 (.057) | 0.264 | −0.604 (.071) | 0.264 | 21.8 | 0.163 |
| `first32d` | +0.195 (.015) | +0.567 (.045) | 0.382 | −0.486 (.069) | 0.382 | 37.8 | 0.271 |
| `idf16` | +0.200 (.014) | +0.588 (.042) | 0.368 | −0.500 (.082) | 0.368 | 22.8 | 0.170 |
| `idfspan16` | +0.303 (.016) | +0.886 (.048) | 0.329 | −0.539 (.073) | 0.326 | 22.8 | 0.170 |
| **`fl32`** | +0.176 (.015) | +0.512 (.044) | 0.444 | −0.424 (.082) | 0.444 | 38.8 | **0.277** |
| `sent1` | +0.397 (.018) | +1.159 (.053) | 0.250 | −0.618 (.071) | 0.250 | 24.3 | 0.180 |
| `first32_swap` | +1.242 (.017) | +3.703 (.057) | **0.007** | −0.861 (.048) | 0.007 | 38.8 | 0.277 |

**The headline.** Where every slot-vector construction sat **on** the `k/n` guess floor with an
undetectable swap control, a real-token subset moves `R@gold_pooled` from **0.224 → 0.927** at 2k
and **0.062 → 0.603** at 8k, and the swap control **collapses it to 0.026 / 0.007**. The frozen
dense model reads these tokens; it never read the slots.

**Cheapest near-parity, 2k: `fl32`** (first 16 + last 16 body tokens, header real, remainder pooled)
— ΔCE +0.044 ± 0.007, ΔCE(digits) +0.162 ± 0.028, genF1 **0.927 vs 0.979** (ΔF1 −0.052 ± 0.031,
1.7 SE) at **compaction 0.309 ≈ 3.2x**. It is not *formal* parity by the ΔCE ≤ 1 SE rule — nothing
here is, at 240 rows the SEs are 0.007–0.03 — but it is the closest, and it beats `first64` at 0.6x
its tokens.

**At 8k nothing reaches parity**: the best is `first64` at ΔCE +0.065 ± 0.008 / genF1 0.603 vs 0.868,
compaction 0.492. The ΔCE-vs-tokens curve is smooth and monotone — 8 / 16 / 32 / 64 tokens per
document give ΔCE +0.549 / +0.361 / +0.204 / +0.065 — so the 8k rung would need ~128 tokens per
document (i.e. essentially the whole document) to close. **Parity gets strictly harder with length**,
which is the opposite of what a compaction recipe needs.

**`goldonly` is settled, and it is not at parity.** 2k: ΔCE +0.184 ± 0.004, ΔCE(digits) +0.561,
genF1 **0.318 vs 0.979**. 8k: ΔCE +0.240 ± 0.008, genF1 **0.049 vs 0.868** — it collapses completely.
And on this corpus the CE column is *not* prose-contaminated (`full` CE ≈ CE(digits) ≈ 0.01), so
this is not the measurement artefact of `records/outlier-slot-probe.md` §4. The 2026-09-08 handoff's
"outlier 1.201 gold-only vs 1.206 full = parity" was measured on `lmx-full-mixs160M-4b` at 24 rows,
where answer CE **is** ~95 % prose; the 2026-09-14 reading is the one that survives. Note
**`first16` (gold-blind, cheaper) already beats `goldonly` on every column at 2k** — 0.661 vs 0.318
genF1 at compaction 0.207 vs 0.284.

**The greedy dumps make the failure mode concrete.** `cc00` emits the constant
`Outliers: [1], [2], [3]` (the position prior) whatever the true ids; `first16`/`first32` emit ids
spread across the row and get them right — row 2 of the 2k run: true `[8, 9, 11]`, `cc00` predicts
`[1, 2, 3]`, `first16` predicts `[8, 9, 11]`.

### 5c. Replicate pairing, 2k — **COMPLETE, 48 rows**, the full construction set

`lmx-full-mixs160M-4b` (frozen dense) on `outlier_wiki100w_n22_k3_eval_600.jsonl`, **eval_size = 48**
(⚠ < 500; the SE of every Δ is printed beside it), 21.4 documents/row, `k/n` floor **0.140**,
generation on the first 8 rows (24 gold documents — ⚠ genF1/R columns have SE ≈ 0.04–0.18).
sneetches job `3552630`.

**Corpus structure measured on these rows** (48 rows):

| quantity | value |
|---|---|
| clusters found per row | 3.73 ± 0.07 (true majority topics ≈ 3 + the outlier topic) |
| smallest / largest cluster | 2.77 ± 0.17 / 8.83 ± 0.19 documents |
| **`gold_cat_purity`** | **1.000 ± 0.000** — all three gold documents always land in the SAME cluster |
| **`gold_in_smallest`** | **0.521 ± 0.073** — but that cluster is the smallest only about half the time |
| cos(gold, centroid) / cos(other, centroid) | **−0.271 ± 0.047** / **+0.087 ± 0.009** |
| **`hn_rate`** | **0.708 ± 0.066** — in 71 % of rows some non-gold document sits farther from the centroid than a gold one |
| `oracle_cosR` | 0.396 ± 0.063 — "the 3 lowest-cosine documents are the outliers" recovers 40 % of gold |

So accidental hard negatives are **common** (71 % of rows) even though the generator never makes them
on purpose, and the cosine margin between gold and non-gold is real but far from separating.

| condition | CE | ΔCE (SE) | CE(dig) | ΔCE(dig) (SE) | top1 | genF1 | ΔF1 (SE) | rule_R | kept docs | tok/doc | compaction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 1.804 | — | 0.053 | — | 1.000 | **0.825** | — | — | 21.4 | 146 | 1.000 |
| `goldonly` | 1.396 | −0.400 (.024) | 0.280 | **+0.259** (.026) | 0.909 | **0.250** | −0.575 (.182) | 1.000 | 3.0 | 21.5 | 0.184 |
| `goldonly_cc` | 1.328 | −0.483 | 0.301 | +0.296 (.027) | 0.908 | 0.375 | −0.450 (.165) | 1.000 | 3.0 | 21.5 | 0.184 |
| `gpr33` (gold + 1/3 random) | 1.645 | −0.165 | 0.564 | +0.605 (.073) | 0.934 | 0.350 | −0.475 (.164) | 1.000 | 8.9 | 64.5 | 0.466 |
| `cc00` (control) | 1.481 | −0.281 | 1.514 | +1.536 (.055) | 0.786 | 0.083 | −0.742 (.127) | 0.000 | 0 | 6.6 | 0.085 |
| `smallcat1` | 1.595 | −0.193 | 1.323 | +1.212 (.067) | 0.839 | 0.208 | −0.617 (.165) | **0.528** | 2.6 | 24.8 | 0.205 |
| `smallcat2` | 1.632 | −0.201 | 0.677 | +0.587 (.094) | 0.920 | 0.292 | −0.533 (.182) | 0.826 | 7.6 | 55.6 | 0.407 |
| `smallcat3` | 1.661 | −0.128 | 0.145 | +0.213 (.067) | 0.979 | 0.500 | −0.325 (.181) | **0.951** | 14.9 | 106 | 0.739 |
| `smallcat3cat` (slot/cluster) | 1.708 | −0.073 | 0.196 | +0.222 (.069) | 0.975 | 0.500 | −0.325 (.181) | 0.951 | 14.9 | 106 | 0.738 |
| **`smallcat3f16`** | 1.703 | −0.090 | **0.063** | **+0.064** (.030) | 0.996 | **0.825** | **+0.000** (.038) | 0.951 | 14.9 | 111 | 0.770 |
| `smallcatle3` | 1.558 | −0.227 | 1.062 | +1.047 (.063) | 0.871 | 0.208 | −0.617 (.165) | 0.708 | 2.9 | 26.1 | 0.214 |
| `smallcat5`, `smallcat5cat`, `smallcat3d1/d2`, `goldcats4` | — | +0.000 | — | +0.000 | 1.000 | 0.825 | +0.000 | 1.000 | 21.4 | 146 | **1.000 (degenerate: they keep the whole row at n = 22)** |
| `goldcats2` | 1.675 | −0.129 | 0.118 | +0.153 (.052) | 0.979 | 0.500 | −0.325 (.181) | 1.000 | 15.5 | 109 | 0.756 |
| `goldcats2r` (random cats) | 1.752 | −0.064 | 0.114 | +0.039 (.018) | 0.993 | 0.775 | −0.050 (.050) | 1.000 | 18.9 | 130 | 0.901 |
| `margin6` | 1.732 | −0.101 | 1.329 | +1.173 (.123) | 0.873 | 0.271 | −0.554 (.157) | 0.618 | 6.0 | 47.6 | 0.355 |
| `margin6f16` | 1.639 | −0.188 | 0.278 | +0.271 (.047) | 0.947 | 0.637 | −0.188 (.137) | 0.618 | 6.0 | 59.1 | 0.431 |
| `hardneg6` (ORACLE) | 1.946 | **+0.144** | 2.275 | +2.234 (.090) | 0.810 | 0.217 | −0.608 (.122) | 0.000 | 6.0 | 45.8 | 0.343 |
| `first16` | 1.521 | −0.292 | 0.488 | +0.467 (.057) | 0.906 | 0.458 | −0.367 (.175) | — | 0 | 22.6 | 0.191 |
| **`first32`** | 1.513 | −0.237 | 0.207 | +0.262 (.050) | 0.940 | **0.787** | **−0.038** (.080) | — | 0 | 38.6 | **0.296** |
| `first64` | 1.593 | −0.164 | 0.120 | +0.170 (.055) | 0.962 | 0.775 | −0.050 (.050) | — | 0 | 70.6 | 0.507 |
| `first64d` (no slot) | 1.632 | −0.147 | 0.179 | +0.172 (.050) | 0.972 | 0.738 | −0.087 (.058) | — | 0 | 69.6 | 0.501 |
| `fl16` | 1.461 | −0.294 | 0.468 | +0.513 (.075) | 0.906 | 0.537 | −0.287 (.160) | — | 0 | 22.6 | 0.191 |
| **`fl32`** | 1.522 | −0.283 | 0.267 | +0.204 (.052) | 0.950 | **0.800** | **−0.025** (.045) | — | 0 | 38.6 | **0.296** |
| `sent1` | 1.540 | −0.265 | 0.833 | +0.705 (.090) | 0.870 | 0.617 | −0.208 (.131) | — | 0 | 24.1 | 0.201 |
| `idfspan16` | 1.483 | −0.319 | 0.523 | +0.469 (.067) | 0.906 | 0.479 | −0.346 (.147) | — | 0 | 22.6 | 0.191 |
| `smallcat3_swap` | 2.547 | +0.764 | 2.970 | +3.110 (.143) | 0.812 | **0.050** | −0.775 (.133) | 0.951 | 14.9 | 106 | 0.739 |
| `first32_swap` | 2.439 | +0.637 | 3.223 | +3.203 (.126) | 0.790 | **0.000** | −0.825 (.122) | — | 0 | 38.6 | 0.296 |

⚠ **Read the CE(digits) column, not CE.** On this corpus the answer is a prose sentence wrapping the
ids, so mean answer CE is ~95 % prose and every construction "beats" `full` on it — the driver's own
`PARITY` verdict therefore mislabels anything with negative ΔCE as `ce-ok`. The discriminating
columns are ΔCE(digits) and ΔgenF1. (Fixed for future runs; the printed verdict line in these logs
should be re-read with that substitution.)

**1. A flat per-document token budget beats every category rule at equal cost.** At compaction
0.296, `first32` and `fl32` reach genF1 0.787 / 0.800 against `full`'s 0.825 — ΔF1 −0.038 ± 0.080
and −0.025 ± 0.045, i.e. **parity within noise**. `smallcat3` needs compaction 0.739 to reach only
0.500, and `goldcats2` 0.756 for 0.500. Keeping categories WHOLE spends the whole budget on a few
documents and leaves the rest unreadable.

**2. The one place the category structure helps is as a *floor*, not a *replacement*.**
`smallcat3f16` — whole small categories **plus** 16 real tokens on every pooled document — is the
only construction that matches `full` on **both** ΔCE(digits) (+0.064 ± 0.030) and genF1 (0.825,
ΔF1 +0.000) — but at compaction 0.770 it saves almost nothing. `margin6f16` shows the same shape
more cheaply and less well (0.637 at 0.431).

**3. The clustering, not the rule, is what limits `smallcat`.** `gold_cat_purity` is **1.000** — the
three gold documents always land in one cluster — but `gold_in_smallest` is only **0.521**, so
`smallcat1` has rule recall 0.528 and `smallcatle3` 0.708. Only at C = 3 does rule recall reach
0.951, and by then the rule is keeping 15 of 21 documents. The generator's guarantee ("the outlier
topic is strictly the smallest") is true of the *articles*; it is not recoverable from a
mean-embedding clustering at this accuracy.

**4. The gold-forcing signature is real and large.** Under the old `gpr33` construction the gold
document's category is the **ONLY fully-real category** in **0.708** of rows (`goldonly`: 0.708
likewise). Under `smallcat3` it is **0.000** — three categories are always complete — and under
`smallcat1` 0.521. So the user's mechanism is confirmed: gold-forcing hands the model a
"whole category ⇒ gold" shortcut in 7 of 10 rows, and the category rules remove it by construction.
`smallcat3d1/d2` (decoys) would remove it too, but at n = 22 they degenerate to keeping the whole
row.

**5. The oracle `hardneg6` is the only construction that is WORSE than `cc00` on ΔCE** (+0.144 vs
−0.281 on CE, +2.234 vs +1.536 on digits). Keeping the six non-gold documents most similar to gold,
with gold itself pooled, is actively harmful — consistent with §3b: the generator makes no
deliberate hard negatives, so "preserve the near-misses" is not the binding constraint.

**6. Both swap controls fire.** `first32` 0.787 → `first32_swap` **0.000**; `smallcat3` 0.500 →
`smallcat3_swap` **0.050**, with CE(digits) blowing up to 3.1–3.2 — worse than seeing nothing at
all. The frozen model is genuinely reading the kept tokens, which is exactly what no slot-vector
construction ever achieved.

### 5b. Canonical 32k — INTERIM at row 50/100 (⚠ eval_size 50, 96 gold documents, SE ≈ 0.05)

`rung_32768.jsonl`, 220 documents/row, `k/n` floor **0.014**. Job `01M2K3F8JZYHR1HJNNMY8N77MZ`.

| condition | ΔCE | ΔCE(digits) | genF1 | **R@gold_pooled** | tok/doc | compaction |
|---|---|---|---|---|---|---|
| `full` | — | — | **0.490** | — (R@gold_real 0.490) | 147 | 1.000 |
| `cc00` | +0.749 | +1.828 | 0.010 | **0.010** ≈ floor 0.014 | 7.5 | 0.055 |
| `goldonly` | +0.135 | +0.291 | 0.062 | — (0.062) | 3.0 | 0.025 |
| `goldonly_cc` | +0.177 | +0.388 | 0.042 | — (0.042) | 3.0 | 0.025 |
| `first16` | +0.440 | +1.036 | 0.021 | 0.021 | 23.5 | 0.164 |
| `first32` | +0.258 | +0.607 | 0.094 | 0.094 | 39.5 | 0.273 |
| `first64` | **+0.104** | **+0.243** | **0.229** | 0.229 | 71.5 | 0.490 |
| `first32d` | +0.281 | +0.670 | 0.083 | 0.083 | 38.5 | 0.266 |
| `idfspan16` | +0.413 | +0.983 | 0.031 | 0.031 | 23.5 | 0.164 |
| `sent1` | +0.472 | +1.122 | 0.010 | 0.010 | 25.1 | 0.175 |
| `first32_swap` | +1.186 | +2.939 | 0.021 | 0.021 | 39.5 | 0.273 |

⚠ **`full` itself scores only genF1 0.490 at 32k** — the frozen checkpoint is already half wrong, so
"parity" here is a weak statement. Nothing comes close regardless: the best (`first64`, 64 real
tokens/document, compaction 0.490) recovers 0.229 of 0.490.

### 5d. Replicate pairing, 8k and 32k — the category rules at length (⚠ interim, 16 rows)

`lmx-full-mixs160M-4b` on `outlier_wiki100w_n55` (8k) and `n220` (32k), sneetches `3552660` /
`3552682`. **On this checkpoint `full` itself scores genF1 0.000 at both rungs**, so only
CE(digits), top-1 and the rule diagnostics are informative here.

**8k (n = 55)** — `full` CE(digits) 0.138:

| condition | ΔCE(dig) | top1 | **rule_recall** | kept docs | compaction |
|---|---|---|---|---|---|
| `cc00` | +1.926 | 0.730 | — | 0 | 0.063 |
| `smallcat1` | +1.745 | 0.774 | **0.396** | 2.0 | 0.098 |
| `smallcat2` | +0.720 | 0.890 | 0.917 | 5.4 | 0.158 |
| **`smallcat3`** | +0.380 | 0.928 | **0.938** | 10.1 | **0.240** |
| **`smallcat3cat`** (one slot per pooled cluster) | +0.382 | 0.914 | 0.938 | 10.1 | **0.235** |
| **`smallcat3f16`** | **+0.116** | 0.958 | 0.938 | 10.1 | 0.328 |
| `smallcatle3` | +0.905 | 0.870 | 0.896 | 4.3 | **0.137** |
| `smallcat5` | +0.116 | 0.965 | 1.000 | 24.3 | 0.482 |
| `smallcat3d1` / `d2` (decoys) | +0.289 / +0.242 | 0.928 / 0.948 | 0.938 | 20.0 / 31.3 | 0.414 / 0.610 |
| `goldcats2` / `goldcats4` | +0.188 / +0.106 | 0.941 / 0.973 | 1.000 | 11.7 / 26.2 | 0.267 / 0.515 |
| `goldcats2r` (random cats) | **−0.004** | 0.965 | 1.000 | 20.1 | 0.414 |
| `margin6` / `margin6f16` | +2.381 / +0.767 | 0.751 / 0.896 | **0.042** | 6.5 | 0.184 / 0.279 |
| `hardneg6` (ORACLE) | +2.703 | 0.704 | 0.000 | 6.0 | 0.165 |
| `first16` / `first32` / `first64` | +0.723 / +0.186 / +0.097 | 0.870 / 0.942 / 0.969 | — | 0 | 0.171 / 0.279 / 0.495 |
| `fl32` | +0.170 | 0.938 | — | 0 | 0.279 |
| `smallcat3_swap` / `first32_swap` | +2.276 / +3.283 | 0.783 / 0.753 | | | |

**8k is where the category rule is worth something.** After the cluster-cut fix (§3c, `nover8`),
`smallcat3` lands in the target region — **rule recall 0.938 at compaction 0.240** — and
`smallcat3f16` (small categories whole **+** 16 real tokens on every pooled document) gives
**ΔCE(digits) +0.116 at compaction 0.328**, the best value in the ≤ 0.33 band and half the cost of
`smallcat5` (+0.116 at 0.482). `margin{M}` and the oracle `hardneg{M}` are the **worst** rows in the
table (rule recall 0.042 / 0.000, ΔCE(digits) +2.4 / +2.7) — worse than keeping nothing.

**32k (n = 220)** — `full` CE(digits) 0.545. **The clustering collapses**: `smallcat1`/`smallcat3`
rule recall **0.021**, `smallcat5`/`smallcatle3` 0.312, so every gold-blind category rule is at or
below `cc00` (ΔCE(digits) +1.41 to +1.72). Only the gold-aware `goldcats2`/`goldcats4`
(+0.074 / +0.046 at compaction 0.102 / 0.120) and the flat token budgets
(`first64` +0.111 at 0.490, `fl32` +0.158 at 0.273) survive. A mean-embedding clustering cannot
separate 26 Wikipedia topics at n = 220.

## 5e. Verdict

**1. Real tokens work where slot vectors did not — this is the first construction in this line that
moves `R@gold_pooled` off the `k/n` floor.** Canonical pairing: 0.224 → **0.927** at 2k and
0.062 → **0.603** at 8k, with the swap control collapsing both to 0.026 / 0.007. Thirteen slot
vectors, two checkpoints and three lengths of prior work never moved it at all.

**2. The cheapest near-parity recipe is `fl32` — first 16 + last 16 body tokens per document, header
real, remainder pooled (or simply dropped) — at 2k only.** ΔCE +0.044 ± 0.007, ΔCE(digits)
+0.162 ± 0.028, genF1 0.927 vs 0.979 (ΔF1 −0.052 ± 0.031), **compaction 0.309 ≈ 3.2x**. No
construction meets the strict ΔCE ≤ 1 SE bar at any rung — the SEs at 240 rows are 0.007–0.03 — so
this is *near*-parity, not parity.

**3. Parity gets harder with length, monotonically — which is fatal for the recipe's purpose.** The
tokens/document needed to reach a given ΔCE roughly doubles per rung: at 2k, 32 tokens gives
ΔCE +0.049; at 8k, 32 gives +0.204 and 64 gives +0.065; at 32k, 64 gives +0.104 and genF1 is still
0.229 against `full`'s 0.490. Extrapolated, 8k needs ~128 tokens/document and 32k more than that —
i.e. the whole document. **A compaction recipe that needs more of each document as the context grows
is not a compaction recipe.**

**4. The slot is dead weight.** `first{k}d` — the same tokens with the pooled remainder **dropped
entirely**, no soft token at all — matches or beats `first{k}` at 2k (`first64d` ΔCE +0.013 vs
+0.013.. +0.020, genF1 0.901 vs 0.872) and costs one token per document less. Whatever the model is
using, it is not the slot.

**5. Selection rule: position ≫ frequency, and both ends ≫ one.** At a fixed budget, `fl` > `first`
≈ `idfspan` > `idf` > `sent1`. The one candidate that needed a corpus statistic (idf) is the worst.

**6. `goldonly` is not at parity, and the 2026-09-08 claim does not survive.** 2k genF1 0.318 vs
0.979; 8k 0.049 vs 0.868. `first16` — gold-blind and cheaper — beats it on every column.

**7. The gold-forcing shortcut is confirmed, and it is large.** Under the old `gold_plus_random`
keep at 1/3, the gold document's category is the **only fully-real category** in **70.8 %** of rows
(48 rows, replicate 2k). `smallcat3` drives that to **0.000**. So the user's mechanism is right —
but removing the shortcut does not by itself buy accuracy.

**8. The category rules are a length-limited idea.** They are worth something at 8k
(`smallcat3f16`: ΔCE(digits) +0.116 at compaction 0.328, rule recall 0.938) and worthless at 32k
(rule recall **0.021**), because average-linkage clustering of mean-embedding vectors cannot
separate ~26 topics. At 2k they are degenerate (C = 3 of ~3.7 clusters keeps the whole row). The
binding constraint is the **clustering**, not the rule: `gold_cat_purity` is 1.000 (the three gold
documents always land together) but `gold_in_smallest` is only 0.521.

**9. "Preserve the hard negatives" is not the explanation.** The generator makes no deliberate
near-misses (§3b); accidental ones are common (`hn_rate` 0.708 at 2k) but the constructions built to
keep them — `margin{M}` and the ORACLE `hardneg{M}` — are the **worst** rows measured, below
keeping nothing at all.

## 6. Runs

Canonical pairing: the ds64 dense arm `ds64-outlier-dense-u64M` (frozen, full-attention-trained) on
the ds64 outlier rung files `outlier_lengthmix/eval_rungs/outlier/rung_*.jsonl`. 1-GPU Beaker jobs,
urgent + unallocated (workspace `ai2/flex2`, budget `ai2/oe-other`),
`--cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2"`. JSON to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/realtoken_*.json`.

| rung | rows | gen rows | conditions | where | experiment / job | state |
|---|---|---|---|---|---|---|
| 2k (smoke) | 6 | 3 | core | beaker | `01M2K3CDRXS61AS60WKV2Q4CY6` / `01M2K3CDZS4TX7VS9VQ999NJ6P` | **done** — mechanism validated |
| 2k | 240 | 64 | all (30) | beaker | `01M2K3DHB5WKVM9YGC8SHGV3A5` / `01M2K3DHJZW7QV690ES88R0HB6` | **running** (§5a is its row-25 snapshot) |
| 8k | 200 | 48 | core (15) | beaker | `01M2K3EE19NB5QX1PFF4YZWP57` / `01M2K3EE89VP13HPV3MHM682P5` | running (ETA ~5 h) |
| 32k | 100 | 32 | lean (11) | beaker | `01M2K3F8F74F0KER9VSHJWB11T` / `01M2K3F8JZYHR1HJNNMY8N77MZ` | **running** (§5b is its row-5 snapshot) |
| 32k | 100 | 32 | cat32k (17) | beaker | `01M2K6C1RCE97F0QP2SVD4974N` / `01M2K6C1VTGGF4ZNB69W91Z3EF` | queued (clusters 100 % full) |
| 8k | 200 | 48 | cat (29) | beaker | `01M2K6D0C2M8SYWXBACZ0JT01X` / `01M2K6D0FGXB6QJK88R719WW7V` | queued |
| 2k | 240 | 64 | cat (29) | beaker | `01M2K6DXSWB0S74A49PP9ZQJS7` / `01M2K6DXXG6R7ZSWPWSC5781RT` | queued |
| 2k | 200 | 64 | all (35) | sneetches | slurm `3552283` | running — **replicate pairing** `lmx-full-mixs160M-4b` on `outlier_wiki100w_n22_k3` |
| 2k | 200 | 64 | cat (29) | sneetches | slurm `3552438` | running — replicate pairing, category rules |
| 32k | 100 | 32 | 21 conds | horton | slurm `3552570` | queued behind a full node; waits for the staged checkpoint |
| 8k | 200 | 48 | 33 conds | horton | slurm `3552571` | queued |
| 2k | 240 | 64 | 33 conds | horton | slurm `3552572` | queued |

The horton jobs read a node-local copy of the frozen checkpoint
(`/data/prasann/ckpts/ds64-outlier-dense-u64M`) and the staged rung files
(`/data/prasann/ds64_eval/outlier/rung_*.jsonl`); the launcher
`/scratch/users/prasann/outlier_probe/run_realtoken_horton.sbatch` blocks until the checkpoint
appears. They run from a second detached worktree `/scratch/users/prasann/outlier_probe/wt2`.
Because weka is not mounted locally, those runs build the `cent_cmean` stop set and the idf table
from the **eval rows** rather than the ds64 training shard — a small construction difference from
the Beaker runs, noted wherever the two are compared.

⚠ **Every eval here is < 500 rows** (240 / 200 / 100), and the generation metrics average over fewer
rows still (64 / 48 / 32). Quote them with the SE printed next to them.

Reproduce (1 GPU):

```
python debug/flop_scaling/beaker_bench_launch.py \
  --cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2" \
  --script debug/pooled_kv/outlier_probe/outlier_realtoken_probe.py \
  --extra "--rung 2k --rows 240 --gen-rows 64 --conditions all --work /results/rt_work \
           --ckpt-name ds64-outlier-dense-u64M"
```

## 7. Traps

* **A document's tokens are not contiguous once header-freeing has run** (and even less so once an
  arbitrary subset is freed). Index per-document buffers by **ordinal within the document**, never by
  `pos − first[doc]`; the latter runs past the end and dies as an async device-side assert.
* **RoPE sizes its absolute-position sin/cos buffer from the CACHE, not from the `position_ids` it is
  handed** (`rope.py:563-571`). A compacted row is short but carries ORIGINAL positions, and free
  generation walks past the row's own length. Warm every RoPE module to `max(row) + gen budget`
  first. The `drop` conditions hit this too, because they pass original positions explicitly.
* **The probe compacts once itself** (to map answer positions to compacted columns) **and the forward
  compacts again.** Memoize on the exact input tensor + condition so the two cannot disagree.
* `mark_doc_headers_free` is imported *inside* the forward, which is what makes the chunk-id override
  possible — but it also means a stale `sys.modules` entry would silently un-patch it. The driver
  asserts the patch is installed before it loads the model.
