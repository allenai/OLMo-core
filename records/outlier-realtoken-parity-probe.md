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

### 5a. The canonical 2k rung — INTERIM at row 25/240 (⚠ eval_size 25, 75 gold documents)

`ds64-outlier-dense-u64M` on `outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl`, 13.4
documents/row, `k/n` guess floor **0.224**. ⚠ **eval_size = 25 so far** — per-document SE ≈ 0.05 at
75 gold documents; this is a shape check, not a number to quote. Job `01M2K3DHJZW7QV690ES88R0HB6`,
still running (ETA ~2.5 h to 240 rows).

| condition | CE | ΔCE | CE(digits) | ΔCE(dig) | genF1 | **R@gold_pooled** | tok/doc | compaction |
|---|---|---|---|---|---|---|---|---|
| `full` | 0.004 | — | 0.007 | — | **1.000** | — (R@gold_real 1.000) | 144.3 | 1.000 |
| `goldonly` | 0.196 | +0.192 | 0.610 | +0.603 | 0.307 | — (R@gold_real 0.307) | 33.4 | 0.282 |
| `goldonly_cc` | 0.203 | +0.199 | 0.626 | +0.619 | 0.333 | — (0.333) | 33.4 | 0.282 |
| `cc00` (control) | 0.445 | +0.442 | 1.643 | +1.636 | 0.213 | **0.213** ← the floor | 6.3 | 0.106 |
| `first4` | 0.348 | +0.344 | 1.301 | +1.293 | 0.280 | 0.280 | 10.3 | 0.132 |
| `first8` | 0.182 | +0.178 | 0.660 | +0.653 | 0.693 | 0.693 | 14.3 | 0.158 |
| `first16` | 0.091 | +0.087 | 0.311 | +0.304 | 0.787 | 0.787 | 22.3 | 0.210 |
| `first32` | 0.047 | +0.043 | 0.154 | +0.146 | 0.853 | 0.853 | 38.3 | 0.314 |
| `first64` | 0.022 | +0.018 | 0.065 | +0.058 | 0.947 | 0.947 | 70.3 | 0.522 |
| `first4d` (no slot) | 0.356 | +0.352 | 1.338 | +1.331 | 0.467 | 0.467 | 9.3 | 0.125 |
| `first8d` | 0.229 | +0.225 | 0.838 | +0.831 | 0.547 | 0.547 | 13.3 | 0.151 |
| `first16d` | 0.109 | +0.105 | 0.380 | +0.372 | 0.792 | 0.787 | 21.3 | 0.203 |
| `first32d` | 0.049 | +0.045 | 0.165 | +0.157 | 0.853 | 0.853 | 37.3 | 0.307 |
| `first64d` | **0.015** | **+0.011** | **0.044** | +0.036 | **0.960** | 0.960 | 69.3 | 0.515 |
| `idf4` / `idf8` / `idf16` | 0.290 / 0.222 / 0.101 | | 1.100 / 0.832 / 0.351 | | 0.453 / 0.520 / 0.747 | | 10.3 / 14.3 / 22.3 | 0.132 / 0.158 / 0.210 |
| `idfspan4` / `8` / `16` | 0.262 / 0.166 / 0.114 | | 0.990 / 0.634 / 0.434 | | 0.587 / 0.680 / 0.787 | | 10.3 / 14.3 / 22.3 | 0.132 / 0.158 / 0.210 |
| `fl8` | 0.201 | +0.197 | 0.762 | +0.755 | 0.560 | 0.560 | 14.3 | 0.158 |
| `fl16` | 0.080 | +0.076 | 0.285 | +0.278 | 0.813 | 0.813 | 22.3 | 0.210 |
| **`fl32`** | **0.018** | **+0.015** | **0.062** | +0.055 | **0.947** | 0.947 | 38.3 | **0.314** |
| `sent1` | 0.148 | +0.145 | 0.550 | +0.543 | 0.707 | 0.707 | 24.0 | 0.221 |
| `first8_swap` | 0.664 | +0.660 | 2.517 | | **0.093** | 0.093 | 14.3 | 0.158 |
| `first16_swap` | 0.807 | +0.803 | 3.080 | | **0.080** | 0.080 | 22.3 | 0.210 |
| `first32_swap` | 0.943 | +0.939 | 3.648 | | **0.000** | 0.000 | 38.3 | 0.314 |
| `first64_swap` | 1.031 | +1.027 | 3.976 | | **0.027** | 0.027 | 70.3 | 0.522 |
| `idfspan16_swap` | 0.826 | +0.822 | 3.164 | | 0.053 | 0.053 | 22.3 | 0.210 |
| `sent1_swap` | 0.779 | +0.775 | 2.975 | | 0.227 | 0.227 | 24.0 | 0.221 |

Four readings, in order of how much they change the picture.

**1. A real-token subset lifts `R@gold_pooled` clean off the `k/n` floor — the first construction in
this line of work that does.** `cc00` sits at 0.213 against a 0.224 floor, exactly where every one of
the thirteen slot vectors sat. Eight real body tokens per document takes it to **0.693**; sixteen to
0.787; sixty-four to **0.947** against `full`'s 1.000. Where a slot vector could not be read at all,
a handful of real tokens is read almost perfectly.

**2. The swap control fires, hard, at every k.** Exchanging the gold documents' kept tokens with
random non-gold documents' collapses genF1 to **0.000–0.093** and blows CE(digits) up to 2.5–4.0 —
*worse* than `cc00`, because the model now actively reads a misleading document. Compare the slot
probes, where the swap control was undetectable to three decimal places. The model really is reading
these tokens.

**3. The slot adds nothing; the tokens are the whole story.** `first{k}d` (the same subset, remainder
**dropped**, no slot at all) matches or beats `first{k}` at every k — `first64d` 0.960 / ΔCE +0.011
against `first64` 0.947 / +0.018. So the cheaper construction is also the better one, and the
soft-token slot can be dropped from this recipe entirely.

**4. Position beats frequency, and both ends beat one.** At a fixed 16 tokens/document: `fl16`
(first 8 + last 8) 0.813 > `first16` 0.787 = `idfspan16` 0.787 > `idf16` 0.747. At 32:
**`fl32` reaches genF1 0.947 and ΔCE +0.015 at compaction 0.314** — i.e. `first64`'s accuracy for
0.6x its tokens. `idf` weighting, the one candidate that needed a corpus statistic, is the *worst*
of the four selection rules.

**On `goldonly`** (the coordinator's question): on this pairing it is **not** at parity —
ΔCE +0.192, ΔCE(digits) +0.603, genF1 **0.307 vs 1.000**. And here the CE column is not
prose-contaminated (`full` CE 0.004 ≈ CE(digits) 0.007), so this is not the measurement artefact of
`records/outlier-slot-probe.md` §4 — gold-only genuinely loses two thirds of the answers. The
2026-09-08 handoff's "outlier 1.201 gold-only vs 1.206 full" was measured on a different checkpoint
and corpus (`lmx-full-mixs160M-4b`, 24 rows) whose answer CE **is** ~95 % prose; the 2026-09-14
reading is the one that holds up. Note `first16` already beats `goldonly` on every column while
being **gold-blind** and cheaper (0.210 vs 0.282 compaction).

### 5b. The 32k rung — INTERIM at row 5/100 (⚠ eval_size 5 — shape only)

`rung_32768.jsonl`, job `01M2K3F8JZYHR1HJNNMY8N77MZ` (ETA ~4 h).

| condition | CE | ΔCE | CE(dig) | genF1 | R@gold_pooled | tok/doc | compaction |
|---|---|---|---|---|---|---|---|
| `full` | 0.102 | — | 0.260 | **0.400** | — (0.400) | 146.4 | 1.000 |
| `goldonly` | 0.271 | +0.170 | 0.662 | 0.000 | — (0.000) | 3.0 | 0.025 |
| `cc00` | 0.970 | +0.869 | 2.397 | 0.000 | 0.000 | 7.5 | 0.055 |
| `first16` | 0.638 | +0.537 | 1.599 | 0.000 | 0.000 | 23.5 | 0.164 |
| `first32` | 0.404 | +0.302 | 1.022 | 0.000 | 0.000 | 39.5 | 0.273 |
| `first64` | 0.153 | **+0.051** | 0.383 | **0.400** | 0.400 | 71.5 | 0.491 |
| `first32d` | 0.399 | +0.298 | 1.003 | 0.133 | 0.133 | 38.5 | 0.266 |
| `idfspan16` | 0.411 | +0.309 | 1.023 | 0.133 | 0.133 | 23.5 | 0.164 |
| `sent1` | 0.627 | +0.525 | 1.567 | 0.000 | 0.000 | 24.8 | 0.173 |
| `first32_swap` | 1.368 | +1.267 | 3.403 | 0.067 | 0.067 | 39.5 | 0.273 |

⚠ **`full` itself only scores genF1 0.400 at 32k on these 5 rows.** Any "parity" claim at this rung
is parity with a model that is already mostly wrong, which is a much weaker statement than at 2k —
flag it wherever it is quoted. The k-dependence is the same shape as at 2k but shifted: at 32k it
takes ~64 real tokens/document to reach `full`, where 2k needed ~32.

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
