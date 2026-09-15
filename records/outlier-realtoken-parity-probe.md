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

_(pending — jobs in §6)_

## 6. Runs

Canonical pairing: the ds64 dense arm `ds64-outlier-dense-u64M` (frozen, full-attention-trained) on
the ds64 outlier rung files `outlier_lengthmix/eval_rungs/outlier/rung_*.jsonl`. 1-GPU Beaker jobs,
urgent + unallocated (workspace `ai2/flex2`, budget `ai2/oe-other`),
`--cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2"`. JSON to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/realtoken_*.json`.

| rung | rows (eval_size) | gen rows | conditions | experiment | job | state |
|---|---|---|---|---|---|---|
| 2k (smoke) | 6 | 3 | core | `01M2K3CDRXS61AS60WKV2Q4CY6` | `01M2K3CDZS4TX7VS9VQ999NJ6P` | — |
| 2k | 240 | 64 | all (30) | `01M2K3DHB5WKVM9YGC8SHGV3A5` | `01M2K3DHJZW7QV690ES88R0HB6` | — |
| 8k | 200 | 48 | core (15) | `01M2K3EE19NB5QX1PFF4YZWP57` | `01M2K3EE89VP13HPV3MHM682P5` | — |
| 32k | 100 | 32 | lean (11) | `01M2K3F8F74F0KER9VSHJWB11T` | `01M2K3F8JZYHR1HJNNMY8N77MZ` | — |

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
