# Can a cheaper-than-training richer slot make a pooled outlier readable? (eval-side)

**Date** 2026-09-14/15 · **Status** DRAFT — model-readout runs in flight; the oracle readout is
complete · **Branch** `prasann/landmark`
**Drivers** `debug/pooled_kv/outlier_probe/outlier_richer_slot_probe.py` (can the frozen dense model
read it?), `debug/pooled_kv/outlier_probe/slot_separability.py` (is it in the vector at all?)

## 1. Where this starts

`records/outlier-slot-probe.md` established, on two checkpoints, two corpora and three lengths, that
the **mean-input-embedding slot is unreadable for outlier**: `R@gold_pooled` sits exactly on the
`k/n` uniform-guess floor in every construction, swapping the gold documents' slot vectors with
random non-gold ones is undetectable, and gold documents whose **body** stayed real are recovered at
0.71–0.84 in the same rows. Its recommendation was to change the **slot**, not the keep policy or
the header.

This probe does exactly that and nothing else: same checkpoint, same rung files, same header-real
construction (`--st-header-stop-id 5491 --st-header-stop-count 1`, so document ids stay readable),
same metrics, same swap control. Only the slot vector changes.

The task's own structure sets the bar low: outlier asks "which k of n documents are topically odd",
so the slot only has to carry **topic**, not detail. Every candidate is therefore a cheap topical
summary, and each one respects the two standing constraints from
`records/ds64-overnight-2026-09-14.md`:

* **FLOP-optimal** — `O(doc tokens)` to build, nothing per-token-per-layer on the main stack;
* **trainable at very long lengths** — at most a few extra tokens per document on the compacted row.

## 2. The two questions, separated

The earlier probe's single measurement conflates two different failures, and they have opposite
consequences:

* **the vector carries no outlier signal** → no reader recovers it, and the only route left is a
  *trained summarizer* whose output is not a mean;
* **the vector carries it, but this model cannot read it** → a readout problem, and a trained reader
  is the route (cheap, since the slot itself stays free).

So each construction is measured twice:

| measurement | script | what it answers |
|---|---|---|
| **oracle readout** `R@k` | `slot_separability.py` | rank the row's documents by cosine distance from the slot centroid (and, separately, by "has no near neighbour"), take the top `k`, score against gold. No model forward, no training. A construction at the `k/n` floor here is dead for everyone. |
| **model readout** `R@gold_pooled` | `outlier_richer_slot_probe.py` | the frozen dense model's own free generation, exactly as in the previous probe, with the swap control. |

## 3. Candidates and their cost

`mean` is the training construction. Everything except `enc*` is embedding arithmetic over the
document's own tokens: no matmuls, no extra tokens unless `G > 1`, and the corpus statistics
(token frequencies, corpus-mean embedding) are precomputed once per corpus, so nothing here scales
with sequence length.

| slot | construction | extra tokens/doc | extra FLOPs |
|---|---|---|---|
| `mean` | mean input embedding of the body (**baseline**) | 0 | 0 |
| `meanrn` | the same, renormalised to the mean real-token embedding norm | 0 | 0 |
| `cmean100` / `cmean500` | mean over CONTENT tokens: drop punctuation/whitespace + the top-K corpus token ids (K = 100 / 500) | 0 | 0 |
| `centered` | `mean − corpus mean`, renormalised | 0 | 0 |
| `cent_cmean` | `cmean100 − content-weighted corpus mean`, renormalised | 0 | 0 |
| `idf` | tokens weighted by `−log p(token)`, renormalised | 0 | 0 |
| `g2` / `g4` | G contiguous segment means, each at its own centre position | +1 / +3 | 0 |
| `g2cent` / `g2cc` | the centred / centred-content version of `g2` | +1 | 0 |
| `enc2` / `enc4` | mean hidden state after a BLOCK-LOCAL forward through the frozen model's own first k layers (attention restricted to the document), renormalised to the input-embedding scale | 0 | `k/32` of a dense forward over the pooled tokens: **0.058 / 0.115 of dense**, i.e. **0.68× / 1.36× the compacted forward itself** at keep 0 |
| `enc4late` | the k = 4 block-local mean injected as the slot's RESIDUAL at layer 4 (layers 0–3 see the plain mean) | 0 | as `enc4` |

The `enc*` cost is the one that matters: at keep 0 the compacted forward is ~8–10% of dense, so
adding 5.8% (k = 2) or 11.5% (k = 4) of dense **doubles or more than doubles** the arm's cost. That
ratio gets *worse* with length, because compaction's quadratic-attention saving grows while the
block-local encode stays linear in pooled tokens.

## 4. Result A — the oracle readout: the signal IS there, and the plain mean is the thing hiding it

`lmx-full-mixs160M-4b` (dense) on the two local outlier corpora, **eval_size = 200 rows each**
(⚠ < 500; the per-row SE is printed with every number). k = 3.

**2k rung** (`outlier_wiki100w_n22_k3`, 21.4 documents/row, `k/n` floor **0.140**):

| slot | oracle R@k (centroid) | oracle R@k (no-neighbour) | gold rank pct | cos(gold, centroid) | cos(other, centroid) |
|---|---|---|---|---|---|
| `mean` (baseline) | 0.230 ± 0.018 | 0.198 ± 0.018 | 0.384 | 0.9349 | 0.9444 |
| `meanrn` | 0.250 ± 0.019 | 0.198 ± 0.018 | 0.373 | 0.9341 | 0.9449 |
| `cmean100` | 0.427 ± 0.020 | 0.298 ± 0.022 | 0.207 | 0.6968 | 0.7540 |
| `cmean500` | 0.408 ± 0.022 | 0.253 ± 0.021 | 0.232 | 0.6892 | 0.7476 |
| `centered` | 0.263 ± 0.021 | 0.215 ± 0.018 | 0.315 | 0.2490 | 0.4303 |
| **`cent_cmean`** | **0.550 ± 0.024** | 0.267 ± 0.023 | **0.155** | 0.2088 | 0.3962 |
| `idf` | 0.295 ± 0.018 | 0.197 ± 0.018 | 0.326 | 0.8321 | 0.8601 |
| `g2` | 0.183 ± 0.016 | 0.172 ± 0.017 | 0.440 | 0.8850 | 0.8921 |
| `g4` | 0.172 ± 0.015 | 0.167 ± 0.015 | 0.466 | 0.7801 | 0.7892 |
| `g2cent` | 0.242 ± 0.022 | 0.203 ± 0.017 | 0.372 | 0.0989 | 0.2411 |
| `g2cc` | 0.440 ± 0.022 | 0.238 ± 0.020 | 0.216 | 0.1375 | 0.2707 |
| `enc2` | 0.405 ± 0.022 | 0.293 ± 0.022 | 0.250 | 0.9922 | 0.9941 |
| `enc4` | 0.440 ± 0.023 | 0.290 ± 0.022 | 0.221 | 0.9702 | 0.9788 |
| `lexidf` (reference: lexical TF-IDF, no embeddings) | 0.373 ± 0.023 | 0.233 ± 0.021 | 0.273 | 0.4410 | 0.5047 |

**~8k rung** (`outlier_wiki100w_n55_k3`, 55 documents/row, `k/n` floor **0.055**):

| slot | oracle R@k (centroid) | oracle R@k (no-neighbour) | gold rank pct |
|---|---|---|---|
| `mean` | 0.077 ± 0.011 | 0.073 ± 0.011 | 0.459 |
| `meanrn` | 0.080 ± 0.011 | 0.073 ± 0.011 | 0.448 |
| `cmean100` | 0.120 ± 0.015 | 0.162 ± 0.016 | 0.334 |
| `cmean500` | 0.130 ± 0.017 | 0.143 ± 0.016 | 0.353 |
| `centered` | 0.132 ± 0.017 | 0.088 ± 0.012 | 0.359 |
| **`cent_cmean`** | **0.240 ± 0.020** | 0.128 ± 0.016 | **0.201** |
| `idf` | 0.092 ± 0.012 | 0.083 ± 0.011 | 0.419 |
| `g2` / `g4` | 0.055 / 0.062 | 0.073 / 0.070 | 0.485 / 0.484 |
| `g2cc` | 0.187 ± 0.017 | 0.103 ± 0.013 | 0.249 |
| `enc2` / `enc4` | 0.140 / 0.142 | 0.157 / 0.168 | 0.381 / 0.371 |
| `lexidf` | 0.125 ± 0.015 | 0.105 ± 0.015 | 0.368 |

Four readings, in order of how much they change the picture.

**1. The plain mean is dominated by common-word mass, and that is why it looks empty.** Every
document's mean embedding is cosine **0.93–0.94** from the centroid — gold and non-gold alike
(0.9349 vs 0.9444). The documents' topics are a rounding error on top of a shared "English prose"
vector. Renormalising (`meanrn`) does nothing for the same reason: it is a pure rescale, and the
directions were never the problem — the *shared component* was.

**2. Removing that component is free and worth 2.4–3.9× the floor.** Dropping punctuation and the
top-100 corpus token ids (`cmean100`) moves cos(gold) − cos(other) from 0.009 to 0.057 and oracle
recall from 0.230 to 0.427; also subtracting the corpus-mean embedding (`cent_cmean`) moves it to
0.188 and recall to **0.550 at 2k / 0.240 at 8k — 3.9× and 4.4× the `k/n` floor**, at zero extra
tokens and zero extra FLOPs. So **the outlier signal is present in a mean-embedding slot**; the
previous probe's "the slot carries nothing" was a statement about the *plain* mean, and it is the
common-word mass, not the averaging, that destroys it.

**3. Neither of the expensive ideas earns its cost.** `enc4` (0.440 at 2k, 0.142 at 8k) is *below*
the free `cent_cmean` while costing 11.5% of a dense forward, and `enc2` is below both. The
block-local hidden states are, if anything, even more dominated by a shared component
(cos ≈ 0.99 for everything). `G > 1` segment means are *worse* than one mean at both lengths:
splitting a document gives each half a noisier estimate and the min-over-segments readout picks up
that noise, so the extra token per document buys a loss. The `G > 1` path is closed for outlier.

**4. The ceiling is real but it is not dense.** The best free slot recovers 0.550 of gold at 2k
where full attention scores ~0.98. A one-vector-per-document summary does not make outlier a solved
task — it makes it a **4× better-than-chance** task. Note also that a purely lexical TF-IDF vector
(`lexidf`, 0.373 / 0.125) sits between the plain mean and `cent_cmean`: the embedding mean, once
decontaminated, is a *better* topical summary than bag-of-words, which is the encouraging direction.

## 5. Result B — the model readout (frozen dense model)

(runs in flight; table to follow — canonical `ds64-outlier-dense-u64M` at 2k and 8k, replicate
`lmx-full-mixs160M-4b` at 2k and ~8k, each candidate with its `_swap` control)

## 6. Verdict

(to follow)

## 7. Runs

| what | where | id | rows | state |
|---|---|---|---|---|
| oracle readout, replicate 2k | sneetches | `3549805` | 200 | done (§4) |
| oracle readout, replicate ~8k (n55) | sneetches | `3549806` | 200 | done (§4) |
| oracle readout, canonical ds64 2k | beaker ceres/saturn | exp `01M2HNZQEQ2QY1JN4FD662X55R` | 240 | running |
| oracle readout, canonical ds64 8k | beaker ceres/saturn | exp `01M2HP0GBPEVDFYH6SY6JJM5N1` | 240 | running |
| model readout, canonical ds64 2k, all 13 candidates + swaps | beaker ceres/saturn | exp `01M2HNFEC7P67NP33F8F3DN5Q3`, job `01M2HNFEG0PWX212YS68NJX61W` | 240 (64 gen) | running |
| model readout, canonical ds64 8k, 4 candidates + swaps | beaker ceres/saturn | exp `01M2HP2PDCFV3F6RMVS9D7GRN0` | 240 (48 gen) | running |
| model readout, replicate 2k, all candidates + swaps | sneetches | `3549785` | 200 (64 gen) | running |
| model readout, replicate ~8k, 4 candidates + swaps | sneetches | `3549832` | 200 (48 gen) | running |

Reproduce (1 GPU):

```
python debug/pooled_kv/outlier_probe/slot_separability.py --rung 2k --rows 240 \
  --ckpt-name ds64-outlier-dense-u64M --work /results/sep_work
python debug/pooled_kv/outlier_probe/outlier_richer_slot_probe.py --rung 2k --rows 240 \
  --gen-rows 64 --ckpt-name ds64-outlier-dense-u64M --work /results/probe_work
```

## 8. Traps hit building this

* **A document's tokens are not contiguous once `mark_doc_headers_free` has run.** The
  `<|doc_start|>` marker keeps the document's chunk id, the header tokens after it become FREE, and
  the body follows — so the document's positions have a hole in the middle. Any per-document buffer
  indexed by `pos − first[doc]` (the obvious way to pack documents into a padded batch) then runs
  past the end and dies as an async device-side index assert. Count **ordinals within the document**
  instead (`_ordinal_in_doc`).
* Scoring a `G > 1` construction by averaging its segment means back into one document vector
  measures nothing — it reconstructs the plain mean. Score the oddest segment.
