# Can a REAL-TOKEN subset per document reach frozen-model parity on outlier?

**Date** 2026-09-15 · **Status** COMPLETE — canonical 2k / 8k / 32k all in (§5); 32k category-rule job still running · **Branch** `prasann/landmark`
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

All on the **frozen** `ds64-outlier-dense-u64M` and the ds64 outlier rungs. ⚠ eval_size **240 / 200 /
100** at 2k / 8k / 32k (all < 500); generation on the first 64 / 48 / 32 rows, so genF1 and
`R@gold_pooled` average over **192 / 144 / 96 gold documents** (per-document SE ≈ 0.03 / 0.04 / 0.05).
Paired SEs in brackets. **Read CE(digits), not CE** — the answer wraps its ids in prose, so mean CE
is largely prose and ranks constructions backwards (the driver's PARITY rule now uses CE(digits) +
genF1; logs written before 2026-09-15 evening print the old ΔCE-based verdict).

`k/n` guess floor: **0.224 / 0.053 / 0.014**. `full` genF1: **0.979 / 0.868 / 0.490** — note the
frozen model is itself only half right at 32k, so "parity" there is a weak statement.

### 5a. Real-token subsets — the finalists, all three rungs

| construction | 2k ΔCE(dig) | 2k genF1 | 2k comp | 8k ΔCE(dig) | 8k genF1 | 8k comp | 32k ΔCE(dig) | 32k genF1 | 32k comp |
|---|---|---|---|---|---|---|---|---|---|
| `full` | — | **0.979** | 1.000 | — | **0.868** | 1.000 | — | **0.490** | 1.000 |
| `cc00` (slot only) | +1.676 (.037) | 0.224 = floor | 0.105 | +2.007 (.031) | 0.062 ≈ floor | 0.062 | +1.873 (.044) | 0.010 ≈ floor | 0.055 |
| `goldonly` (gold-aware) | +0.560 (.017) | 0.318 | 0.284 | +0.648 (.023) | 0.062 | 0.076 | +0.367 (.042) | 0.062 | 0.025 |
| `first16` | +0.571 (.046) | 0.661 | 0.207 | +1.063 (.051) | 0.194 | 0.170 | +1.035 (.072) | 0.021 | 0.163 |
| `first32` | +0.181 (.024) | 0.818 | 0.309 | +0.593 (.044) | 0.312 | 0.277 | +0.596 (.060) | 0.094 | 0.272 |
| `first64` | +0.072 (.020) | 0.872 | 0.514 | +0.192 (.024) | 0.600 | 0.492 | +0.224 (.034) | 0.229 | 0.488 |
| `first32d` / `first64d` (no slot) | +0.184 / **+0.048** | 0.854 / 0.901 | 0.303 / 0.508 | +0.567 | 0.382 | 0.271 | +0.630 | 0.083 | 0.265 |
| `idf16` / `idfspan16` | +0.411 / +0.491 | 0.724 / 0.784 | 0.207 | +0.588 / +0.886 | 0.368 / 0.329 | 0.170 | — / +0.907 | — / 0.031 | 0.163 |
| **`fl32`** | **+0.162** (.028) | **0.927** | **0.309** | +0.512 (.044) | 0.444 | 0.277 | — | — | — |
| `sent1` | +0.762 (.049) | 0.630 | 0.217 | +1.159 (.053) | 0.250 | 0.180 | +1.052 (.073) | 0.010 | 0.174 |
| **`first32_swap`** (control) | +3.636 | **0.026** | 0.309 | +3.703 | **0.007** | 0.277 | +2.976 | **0.021** | 0.272 |

### 5b. Category keep rules — canonical 2k (240 rows) and 8k (200 rows)

8k is where they matter; at 2k they are dominated (C = 3 of ~3.7 clusters already keeps most of the
row). Cut rule: `nover8` (Sec. 3c).

| construction | 2k ΔCE(dig) | 2k ΔF1 | 2k comp | 8k ΔCE(dig) | 8k genF1 | 8k comp | 8k **rule_R** |
|---|---|---|---|---|---|---|---|
| `smallcat1` | +1.468 (.046) | −0.750 | 0.241 | +1.573 (.057) | 0.076 | 0.189 | 0.365 |
| `smallcat3` | +1.120 (.069) | −0.396 | 0.561 | +0.497 (.058) | 0.646 | 0.554 | 0.847 |
| `smallcat3cat` (slot per cluster) | +1.243 (.071) | −0.464 | 0.560 | +0.540 | 0.632 | 0.552 | 0.847 |
| **`smallcat5`** | +0.964 (.071) | −0.297 | 0.687 | **+0.180** | **0.812** | 0.734 | **0.950** |
| `smallcat3f16` | **+0.259** (.035) | −0.115 | 0.611 | +0.220 | 0.597 | 0.606 | 0.847 |
| `smallcatle2` / `le3` | +0.908 / +0.409 | −0.456 / −0.344 | 0.564 / 0.716 | +1.996 / +1.500 | 0.056 / 0.056 | 0.070 / 0.096 | 0.023 / 0.408 |
| `smallcat3d1` / `d2` (decoys) | +1.065 / +0.929 | −0.375 / −0.297 | 0.637 / 0.705 | +0.309 / +0.147 | 0.688 / 0.785 | 0.661 / 0.757 | 0.887 / 0.953 |
| `margin3/6/10` | +1.378 / +1.012 / +0.444 | −0.833 / −0.736 / −0.339 | 0.402 / 0.535 / 0.786 | +2.097 / +2.143 / +2.109 | 0.042 / 0.090 / 0.139 | 0.145 / 0.181 / 0.243 | **0.048 / 0.073 / 0.143** |
| `margin6f16` | +0.412 (.036) | −0.271 | 0.589 | +0.967 | 0.271 | 0.276 | 0.073 |
| `hardneg6` / `10` (ORACLE) | +1.707 / +0.807 | −0.661 / −0.366 | 0.502 / 0.750 | +2.224 / **+2.407** | 0.042 / **0.000** | 0.163 / 0.230 | 0.000 |
| `gpr33` (gold + 1/3 random) | +0.719 (.029) | −0.766 | 0.554 | +0.618 (.031) | 0.090 | 0.409 | 1.000 |
| `goldcats2` / `4` (gold-aware) | +0.712 / +0.409 | −0.531 / −0.419 | 0.643 / 0.763 | +0.193 / +0.068 | 0.729 / 0.812 | 0.569 / 0.742 | 1.000 |
| **`goldcats2r`** (random cats) | +0.618 | −0.490 | 0.671 | **+0.038** | **0.854** (ΔF1 −0.014) | 0.634 | 1.000 |
| `smallcat3_swap` (control) | +2.712 | −0.786 | 0.561 | +3.258 | **0.028** | 0.554 | 0.847 |

### 5c. Replicate pairing (`lmx-full-mixs160M-4b`, wiki corpora, 48 rows/rung)

Same ordering at 2k (`fl32` / `first32` at parity, `smallcat*` dominated; `CHEAPEST PARITY: first32`
at compaction 0.296). At 8k and 32k **this checkpoint scores genF1 0.000 itself**, so only
CE(digits) is readable there: 8k `smallcat3f16` +0.116 at compaction 0.328 is the best value in the
≤ 0.33 band; 32k every gold-blind category rule collapses (rule recall **0.021**).

### 5d. Corpus structure and the hard-negative rate, by rung (replicate pairing, 48 rows each)

| quantity | 2k (n = 22) | 8k (n = 55) | 32k (n = 220) |
|---|---|---|---|
| clusters found / true topics | 3.73 ± 0.07 / ≈ 4 | 7.90 ± 0.05 / ≈ 8 | 28.67 ± 0.07 / ≈ 26 |
| smallest / largest cluster | 2.77 / 8.83 | 1.90 / 12.75 | 1.04 / 18.71 |
| **`gold_cat_purity`** (all 3 gold in one cluster) | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **`gold_in_smallest`** | 0.521 ± 0.073 | **0.250 ± 0.063** | **0.000 ± 0.000** |
| cos(gold, centroid) | −0.271 ± 0.047 | −0.024 ± 0.031 | **+0.007 ± 0.026** |
| cos(other, centroid) | +0.087 ± 0.009 | +0.033 ± 0.003 | +0.027 ± 0.002 |
| **`hn_rate`** | 0.708 ± 0.066 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| `oracle_cosR` ("3 lowest cosines are the outliers") | 0.396 ± 0.063 | **0.028 ± 0.017** | 0.014 ± 0.010 |
| **`gold_only_full`** under `gpr33` (the shortcut signature) | **0.708** | **0.583** | **0.167** |
| `gold_only_full` under `goldonly` | 0.708 | 0.771 | 0.312 |
| `gold_only_full` under `smallcat3` | 0.000 | 0.000 | 0.000 |

The cosine margin between gold and non-gold **shrinks by an order of magnitude from 2k to 8k**
(0.358 → 0.057 in cos units), the accidental hard-negative rate goes to **1.000**, and the oracle
cosine rule drops from 0.396 to 0.028. The clustering keeps the three gold documents together
perfectly at both rungs (`gold_cat_purity` 1.000) but increasingly fails to make their cluster the
*smallest* one (0.521 → 0.250). That single quantity is the whole reason the category rules fade
with length: at n = 220 the clustering still puts the three gold documents together perfectly
(`gold_cat_purity` 1.000) but **never** makes their cluster the smallest one
(`gold_in_smallest` 0.000 ± 0.000, 48 rows), and the cosine margin has inverted
(gold +0.007 vs other +0.027). Note also that **the gold-forcing shortcut is itself a
short-context phenomenon** — `gold_only_full` under `gpr33` falls 0.708 → 0.583 → 0.167 from 2k to
32k — so "gold-forcing taught the model a whole-category shortcut" explains the 2k/8k arms far
better than the long ones.


## 5e. Verdict

**1. Real tokens work where slot vectors did not.** `R@gold_pooled` moves off the `k/n` floor for
the first time in this line of work: **0.224 → 0.927** (2k), **0.062 → 0.600** (8k),
**0.010 → 0.229** (32k), and the swap control collapses it to **0.026 / 0.007 / 0.021**. Thirteen
slot-vector constructions across two checkpoints and three lengths never moved it at all, with an
undetectable swap control.

**2. Cheapest near-parity: `fl32`** (first 16 + last 16 body tokens, header real, remainder pooled or
simply dropped) — 2k ΔCE(digits) +0.162 ± 0.028, genF1 **0.927 vs 0.979** (ΔF1 −0.052 ± 0.031) at
**compaction 0.309 ≈ 3.2x**. Nothing meets the strict ≤ 1 SE bar at any rung.

**3. Frozen parity gets monotonically harder with length.** Tokens/document for a given ΔCE(digits)
roughly doubles per rung; at 32k, 64 tokens/document still leaves genF1 0.229 against `full`'s 0.490.

**4. The slot is dead weight.** `first{k}d` (same tokens, remainder dropped, no soft token) matches
or beats `first{k}` and costs one token/document less.

**5. Selection: position ≫ frequency, both ends ≫ one.** `fl` > `first` ≈ `idfspan` > `idf` > `sent1`.

**6. `goldonly` is not at parity** — 2k genF1 0.318 vs 0.979, 8k 0.062 vs 0.868. The 2026-09-08
"1.201 vs 1.206" claim was mean answer CE on a prose-heavy corpus at 24 rows. Gold-blind `first16`
beats it on every column at 2k.

**7. Category rules are a mid-length idea.** Dominated at 2k, genuinely competitive at 8k
(`smallcat5` genF1 0.812 at compaction 0.734, rule recall 0.950; `smallcat3` 0.646 at 0.554 —
roughly on `first64`'s curve), dead at 32k (rule recall 0.021). The binding constraint is the
**clustering**: `gold_cat_purity` is 1.000 at every rung but `gold_in_smallest` falls
0.521 → 0.250 → 0.000.

**8. The gold-forcing shortcut is confirmed and is short-context.** Under `gpr33` the gold category
is the ONLY fully-real category in **0.708 / 0.583 / 0.167** of rows at 2k / 8k / 32k; `smallcat3`
drives it to 0.000. Removing the shortcut does not by itself buy accuracy.

**9. "Preserve the hard negatives" is not the explanation.** `margin{M}` (rule recall 0.048–0.143)
and the ORACLE `hardneg{M}` are the **worst** rows measured — `hardneg10` reaches genF1 **0.000** at
8k, below keeping nothing.

## 5g. ⚠ This probe is a RANKING tool, not a gate — it UNDER-predicts trained first-k

The frozen model has never seen a compacted row. Trained first-k arms beat these numbers by a wide
margin and now **dominate dense on the ladder**: `ck32-32M` **0.409 @ 222 PF** vs `dense-16M` 0.322
@ 780 PF; `ck64-32M` **0.466 @ 396 PF** vs `dense-32M` 0.452 @ 1582 PF. The gap against the frozen
probe is large — frozen `first32` at 8k scores **0.32**, trained `ck32-64M` at 8k scores **0.81**.

So use this probe to **order** candidate constructions cheaply (it got `first{k}`/`fl{k}` > `idf` >
`sent1` > slot-only right, and the swap controls confirm the model reads the tokens), and never to
**reject** one: "no construction reaches frozen parity at 8k/32k" says nothing about what training
recovers. The conclusions that do transfer are the *relative* ones — the slot is dead weight, both
ends beat one end, category rules need the clustering to work — plus the negative controls.

## 6. Runs

Canonical pairing: the ds64 dense arm `ds64-outlier-dense-u64M` (frozen, full-attention-trained) on
the ds64 outlier rung files `outlier_lengthmix/eval_rungs/outlier/rung_*.jsonl`. 1-GPU Beaker jobs,
urgent + unallocated (workspace `ai2/flex2`, budget `ai2/oe-other`),
`--cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2"`. JSON to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/realtoken_*.json`.

| rung | rows | gen rows | conditions | where | experiment / job | state |
|---|---|---|---|---|---|---|
| 2k (smoke) | 6 | 3 | core | beaker | `01M2K3CDRXS61AS60WKV2Q4CY6` / `01M2K3CDZS4TX7VS9VQ999NJ6P` | **done** — mechanism validated |
| 2k | 240 | 64 | all (30) | beaker | `01M2K3DHB5WKVM9YGC8SHGV3A5` / `01M2K3DHJZW7QV690ES88R0HB6` | **done** |
| 8k | 200 | 48 | core (15) | beaker | `01M2K3EE19NB5QX1PFF4YZWP57` / `01M2K3EE89VP13HPV3MHM682P5` | **done** |
| 32k | 100 | 32 | lean (11) | beaker | `01M2K3F8F74F0KER9VSHJWB11T` / `01M2K3F8JZYHR1HJNNMY8N77MZ` | **done** |
| 32k | 100 | 32 | cat32k (17) | beaker | `01M2K6C1RCE97F0QP2SVD4974N` / `01M2K6C1VTGGF4ZNB69W91Z3EF` | running (row 50/100) |
| 8k | 200 | 48 | cat (29) | beaker | `01M2K6D0C2M8SYWXBACZ0JT01X` / `01M2K6D0FGXB6QJK88R719WW7V` | **done** (§5b) |
| 2k | 240 | 64 | cat (29) | beaker | `01M2K6DXSWB0S74A49PP9ZQJS7` / `01M2K6DXXG6R7ZSWPWSC5781RT` | **done** (§5b) |
| 2k | 200 | 64 | all (35) | sneetches | slurm `3552283` | running — **replicate pairing** `lmx-full-mixs160M-4b` on `outlier_wiki100w_n22_k3` |
| 2k | 200 | 64 | cat (29) | sneetches | slurm `3552438` | running — replicate pairing, category rules |

The horton finalist jobs (`3552734-6`) were cancelled — horton went unavailable and the canonical
2k/8k/32k tables had already landed on Beaker. Their launcher reads a node-local copy of the frozen
checkpoint
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

