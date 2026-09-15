# Can a cheaper-than-training richer slot make a pooled outlier readable? (eval-side)

**Date** 2026-09-14/15 · **Status** ANSWERED — no candidate makes the slot readable by a frozen
dense model, but an oracle readout shows the signal is there and free to expose · **Branch**
`prasann/landmark`
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

### Canonical pairing: `ds64-outlier-dense-u64M` on the ds64 2k rung

`outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl`, **eval_size = 240** (⚠ < 500; per-row SE
printed with every number), 13.4 documents/row, k = 3, `k/n` floor **0.225** — the same floor the
`xhdr*` training arms sat on.

| slot | oracle R@k (centroid) | oracle R@k (no-neighbour) | gold rank pct | cos(gold, c) | cos(other, c) | extra cost |
|---|---|---|---|---|---|---|
| `mean` (the training slot) | 0.390 ± 0.019 | 0.307 ± 0.020 | 0.343 | 0.9375 | 0.9503 | — |
| `meanrn` | 0.422 ± 0.019 | 0.307 ± 0.020 | 0.325 | 0.9362 | 0.9512 | free |
| `cmean100` | 0.674 ± 0.020 | 0.382 ± 0.022 | 0.186 | 0.7125 | 0.7852 | free |
| `cmean500` | 0.625 ± 0.021 | 0.376 ± 0.023 | 0.210 | 0.7073 | 0.7796 | free |
| `centered` | 0.468 ± 0.022 | 0.331 ± 0.020 | 0.277 | 0.3027 | 0.5226 | free |
| **`cent_cmean`** | **0.731 ± 0.021** | 0.374 ± 0.023 | **0.167** | 0.2825 | 0.4955 | **free** |
| `idf` | 0.479 ± 0.019 | 0.318 ± 0.020 | 0.277 | 0.8400 | 0.8771 | free |
| `g2` | 0.299 ± 0.017 | 0.265 ± 0.016 | 0.420 | 0.8877 | 0.8980 | +1 tok/doc |
| `g4` | 0.261 ± 0.016 | 0.251 ± 0.016 | 0.468 | 0.7838 | 0.7943 | +3 tok/doc |
| `g2cent` | 0.386 ± 0.021 | 0.301 ± 0.018 | 0.348 | 0.1436 | 0.3239 | +1 tok/doc |
| `g2cc` | 0.603 ± 0.023 | 0.336 ± 0.020 | 0.227 | 0.1941 | 0.3463 | +1 tok/doc |
| `enc2` | 0.646 ± 0.021 | 0.406 ± 0.021 | 0.201 | 0.9926 | 0.9952 | 0.058 dense / 0.68x compacted |
| `enc4` | 0.726 ± 0.020 | 0.414 ± 0.021 | 0.171 | 0.9727 | 0.9842 | 0.115 dense / 1.36x compacted |
| `lexidf` (reference: lexical TF-IDF) | 0.571 ± 0.022 | 0.324 ± 0.022 | 0.258 | 0.4727 | 0.5559 | free |

**`cent_cmean` (free) and `enc4` (0.115 of a dense forward) are tied at 0.73 — 3.2x the `k/n`
floor — so the block-local encoder buys nothing that decontaminating the mean does not.**

### Canonical pairing, 8k rung

`rung_8192.jsonl`, eval_size = 240, 56.6 documents/row, `k/n` floor **0.053**:

| slot | oracle R@k (centroid) | oracle R@k (no-neighbour) | gold rank pct | cos(gold, c) | cos(other, c) |
|---|---|---|---|---|---|
| `mean` | 0.075 ± 0.010 | 0.076 ± 0.010 | 0.463 | 0.9341 | 0.9371 |
| `meanrn` | 0.081 ± 0.011 | 0.076 ± 0.010 | 0.452 | 0.9333 | 0.9375 |
| `cmean100` | 0.107 ± 0.013 | 0.156 ± 0.015 | 0.345 | 0.6937 | 0.7202 |
| `cmean500` | 0.126 ± 0.014 | 0.139 ± 0.015 | 0.367 | 0.6829 | 0.7107 |
| `centered` | 0.124 ± 0.015 | 0.090 ± 0.011 | 0.359 | 0.1783 | 0.3091 |
| **`cent_cmean`** | **0.239 ± 0.019** | 0.128 ± 0.014 | **0.204** | 0.1288 | 0.2462 |
| `idf` | 0.090 ± 0.011 | 0.087 ± 0.011 | 0.422 | 0.8253 | 0.8385 |
| `g2` / `g4` | 0.058 / 0.064 | 0.075 / 0.072 | 0.487 / 0.484 | | |
| `g2cc` | 0.190 ± 0.016 | 0.100 ± 0.012 | 0.249 | 0.0715 | 0.1597 |
| `enc2` / `enc4` | 0.128 / 0.133 | 0.150 / 0.165 | 0.384 / 0.370 | | |
| `lexidf` | 0.129 ± 0.014 | 0.107 ± 0.013 | 0.370 | 0.4087 | 0.4402 |

**`cent_cmean` is 4.5x the floor at 8k** (0.239 vs 0.053) — the margin over `k/n` grows with
length even though the absolute number falls, and `enc4` is now clearly behind it (0.133).

### Replicate: `lmx-full-mixs160M-4b` on the local wiki corpora

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
(0.9375 vs 0.9503 on ds64 2k; 0.9349 vs 0.9444 on the replicate). The documents' topics are a rounding error on top of a shared "English prose"
vector. Renormalising (`meanrn`) does nothing for the same reason: it is a pure rescale, and the
directions were never the problem — the *shared component* was.

**2. Removing that component is free and worth 1.9–3.9× the plain mean.** Dropping punctuation and
the top-100 corpus token ids (`cmean100`) moves cos(gold) − cos(other) from 0.013 to 0.073 on ds64
(0.009 → 0.057 on the replicate) and oracle recall from 0.390 to 0.674 (0.230 → 0.427); also
subtracting the corpus-mean embedding (`cent_cmean`) takes it to **0.731 ± 0.021 on the canonical
ds64 2k rung against a 0.225 floor**, and on the replicate to
0.188 and recall to **0.550 at 2k / 0.240 at 8k — 3.9× and 4.4× the `k/n` floor**, at zero extra
tokens and zero extra FLOPs. So **the outlier signal is present in a mean-embedding slot**; the
previous probe's "the slot carries nothing" was a statement about the *plain* mean, and it is the
common-word mass, not the averaging, that destroys it.

**3. Neither of the expensive ideas earns its cost.** `enc4` ties `cent_cmean` on the canonical
rung (0.726 ± 0.020 vs 0.731 ± 0.021) and is *below* it on the replicate (0.440 vs 0.550 at 2k,
0.142 vs 0.240 at ~8k) — while costing 11.5% of a dense forward, i.e. **1.36× the compacted forward
it is attached to**. `enc2` is below both everywhere. The
block-local hidden states are, if anything, even more dominated by a shared component
(cos ≈ 0.99 for everything). `G > 1` segment means are *worse* than one mean at both lengths:
splitting a document gives each half a noisier estimate and the min-over-segments readout picks up
that noise, so the extra token per document buys a loss. The `G > 1` path is closed for outlier.

**3b. Why this was never learned away in training.** The ds64 soft-token arms run with
`detach_soft_kv=True` (the default; `--st-no-detach-soft-kv` was never passed), and under it the
POOLED slots are injected **detached** -- `model.py` concatenates `soft_vecs[:n_slots].detach()`
before writing them into `h`. So the projector receives no LM gradient for pooled slots and the slot
stays the **raw mean input embedding for the whole run**. Centring and content-filtering are exactly
the kind of thing a trained projector could absorb (subtracting a constant is a bias), but on this
path it never gets the chance -- which is why substituting a better mean is a drop-in change to the
recipe rather than something training would have found on its own.

**4. The ceiling is real but it is not dense.** The best free slot recovers 0.550 of gold at 2k
where full attention scores ~0.98. A one-vector-per-document summary does not make outlier a solved
task — it makes it a **4× better-than-chance** task. Note also that a purely lexical TF-IDF vector
(`lexidf`, 0.373 / 0.125) sits between the plain mean and `cent_cmean`: the embedding mean, once
decontaminated, is a *better* topical summary than bag-of-words, which is the encouraging direction.

## 5. Result B — the model readout: NOTHING lifts it off the floor

The frozen dense model's own free generation, same metric and same swap control as
`records/outlier-slot-probe.md`. The decision rule set for this probe was **≥ 2 SE above `k/n`
AND ≥ +0.10**, and the swap control must DROP.

### Canonical: `ds64-outlier-dense-u64M`, ds64 2k rung (generation complete: 64 rows, 192 pooled
gold documents, `k/n` floor **0.225**, per-document SE **0.030**; CE columns are the running mean at
row 70/240)

| condition | CE | CE(digits) | genF1 | **R@gold_pooled** | swap | Δ(cond − swap) | \|slot\| | compaction | extra FLOPs (dense) |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 0.014 | 0.051 | **0.979** | — (R@gold_real 0.979) | — | — | — | 1.000 | 0 |
| `mean` (baseline) | 0.463 | 1.676 | 0.234 | **0.234 ± 0.031** | 0.229 | +0.005 | 0.221 | 0.105 | 0 |
| `meanrn` | 0.479 | 1.724 | 0.224 | 0.224 ± 0.030 | 0.229 | −0.005 | 0.687 | 0.105 | 0 |
| `cmean100` | 0.486 | 1.747 | 0.234 | 0.234 ± 0.031 | 0.224 | +0.010 | 0.145 | 0.105 | 0 |
| `cmean500` | 0.474 | 1.719 | 0.266 | **0.266 ± 0.032** | 0.234 | +0.032 | 0.160 | 0.105 | 0 |
| `centered` | 0.486 | 1.765 | 0.224 | 0.224 ± 0.030 | 0.229 | −0.005 | 0.687 | 0.105 | 0 |
| `cent_cmean` (best oracle) | 0.471 | 1.711 | 0.234 | 0.234 ± 0.031 | 0.234 | +0.000 | 0.687 | 0.105 | 0 |
| `idf` | 0.498 | 1.777 | 0.229 | 0.229 ± 0.030 | 0.229 | +0.000 | 0.687 | 0.105 | 0 |
| `g2` | 0.468 | 1.691 | 0.229 | 0.229 ± 0.030 | 0.234 | −0.005 | 0.229 | 0.111 | 0 |
| `g4` | 0.451 | 1.646 | 0.229 | 0.229 ± 0.030 | 0.229 | +0.000 | 0.242 | 0.124 | 0 |
| `g2cent` | 0.483 | 1.745 | 0.224 | 0.224 ± 0.030 | 0.234 | −0.010 | 0.687 | 0.111 | 0 |
| `enc2` | 0.469 | 1.706 | 0.229 | 0.229 ± 0.030 | 0.229 | +0.000 | 0.687 | 0.105 | 0.056 |
| `enc4` | 0.463 | 1.684 | 0.229 | 0.229 ± 0.030 | 0.229 | +0.000 | 0.687 | 0.105 | 0.113 |
| `enc4late` | 0.446 | 1.620 | 0.229 | 0.229 ± 0.030 | 0.229 | +0.000 | 0.221 | 0.105 | 0.113 |

Every construction is inside **±0.041 of the `k/n` floor of 0.225**, i.e. inside 1.3 SE. The best
cell (`cmean500`, 0.266) fails both halves of the rule: +0.041 is under +0.10 and under 2 SE, and
its swap control sits at 0.234, a Δ of +0.032 against an SE-of-difference of 0.044. **No candidate
passes.**

Gold-blind keep 1/6, same run (154 pooled / 38 real gold documents, same 0.225 floor):

| condition | CE | CE(digits) | genF1 | **R@gold_pooled** | **R@gold_real** | pred_pooled | compaction | extra FLOPs (dense) |
|---|---|---|---|---|---|---|---|---|
| `mean` | 0.503 | 1.843 | 0.323 | 0.227 ± 0.034 | 0.711 | 0.727 | 0.252 | 0 |
| `meanrn` | 0.520 | 1.900 | 0.307 | 0.208 ± 0.033 | 0.711 | 0.699 | 0.252 | 0 |
| `cmean100` | 0.518 | 1.879 | 0.331 | 0.221 ± 0.033 | 0.763 | 0.705 | 0.252 | 0 |
| `cmean500` | 0.514 | 1.867 | 0.363 | 0.266 ± 0.036 | 0.737 | 0.688 | 0.252 | 0 |
| `centered` | 0.498 | 1.841 | 0.286 | 0.201 ± 0.032 | 0.632 | 0.779 | 0.252 | 0 |
| `cent_cmean` | 0.483 | 1.775 | 0.339 | 0.253 ± 0.035 | 0.684 | 0.801 | 0.252 | 0 |
| `idf` | 0.556 | 2.018 | 0.305 | 0.201 ± 0.032 | 0.711 | 0.626 | 0.252 | 0 |
| `g2` | 0.521 | 1.909 | 0.286 | 0.182 ± 0.031 | 0.711 | 0.710 | 0.257 | 0 |
| `g4` | 0.536 | 1.969 | 0.305 | 0.195 ± 0.032 | 0.737 | 0.693 | 0.268 | 0 |
| `g2cent` | 0.510 | 1.871 | 0.318 | 0.266 ± 0.036 | 0.526 | 0.831 | 0.257 | 0 |
| `enc2` | 0.475 | 1.759 | 0.318 | 0.240 ± 0.034 | 0.632 | 0.751 | 0.252 | 0.047 |
| `enc4` | 0.482 | 1.780 | 0.318 | 0.234 ± 0.034 | 0.658 | 0.751 | 0.252 | 0.094 |
| `enc4late` | 0.475 | 1.750 | 0.339 | 0.247 ± 0.035 | 0.711 | 0.754 | 0.252 | 0.094 |

Pooled recall 0.182–0.266 against the 0.225 floor — nothing passes here either — while recall on
gold whose BODY stayed real is 0.53–0.76 and is essentially unmoved by the slot construction. That
is the previous probe's 3–6× real-vs-pooled gap, reproduced with every richer slot.

### Canonical 8k rung (COMPLETE: eval_size 240, 48 generation rows, 144 pooled gold documents, floor **0.053**, SE 0.020)

| condition | CE | CE(digits) | genF1 | **R@gold_pooled** | swap | \|slot\| | compaction | extra FLOPs (dense) |
|---|---|---|---|---|---|---|---|---|
| `full` | 0.028 | 0.076 | **0.868** | — (R@gold_real 0.868) | — | — | 1.000 | 0 |
| `mean` | 0.678 | 1.966 | 0.062 | 0.062 ± 0.020 | 0.062 | 0.222 | 0.062 | 0 |
| `cmean100` | 0.747 | 2.173 | 0.056 | 0.056 ± 0.019 | 0.076 | 0.144 | 0.062 | 0 |
| `cent_cmean` | 0.723 | 2.089 | 0.062 | 0.062 ± 0.020 | 0.062 | 0.687 | 0.062 | 0 |
| `enc4` | 0.691 | 2.003 | 0.062 | 0.062 ± 0.020 | 0.062 | 0.687 | 0.062 | 0.118 |

Identical to the floor, identical to the swap controls, identical to each other. At keep 1/6
(119 pooled / 25 real gold) pooled recall is 0.017–0.034 while real-body recall is 0.12–0.24.

### Replicate: `lmx-full-mixs160M-4b`, local ~8k corpus (n = 55, 144 pooled gold, floor **0.055**, eval_size 200)

| condition | CE | CE(digits) | genF1 | **R@gold_pooled** | swap |
|---|---|---|---|---|---|
| `full` | 1.785 | 0.187 | 0.762 | — (R@gold_real 0.729) | — |
| `mean` | 1.692 | 2.171 | 0.049 | 0.049 ± 0.018 | 0.049 |
| `cmean100` | 1.620 | 2.240 | 0.049 | 0.049 ± 0.018 | 0.049 |
| `cent_cmean` | 1.677 | 2.042 | 0.049 | 0.049 ± 0.018 | 0.049 |
| `enc4` | 1.591 | 2.050 | 0.049 | 0.049 ± 0.018 | 0.049 |

**Every construction and every swap control gives the identical 0.049** — the model emits the same
id set whatever is in the slots, which is the position prior the previous probe dumped verbatim.

## 6. Verdict

**No candidate lifts `R@gold_pooled` — but the reason is now known, and it is not the slot.**

1. **Model readout: nothing passes, on any construction, at either length, on either checkpoint.**
   The best cell in 40 conditions is +0.041 over `k/n` (1.3 SE) and its swap control eats most of
   that. Renormalising, decontaminating, centring, idf-weighting, splitting into G segments and
   running a 4-layer block-local encoder all leave the frozen dense model exactly on the guess
   floor, with the swap control undetectable — identical to the plain mean.
2. **Oracle readout: the signal is there, and it is cheap.** On the same rows and the same slots, a
   cosine-distance-to-centroid rule recovers **0.731 ± 0.021 of the gold documents at 2k (floor
   0.225) and 0.239 ± 0.019 at 8k (floor 0.053)** from `cent_cmean` — 3.2× and 4.5× the floor, at
   **zero extra tokens and zero extra FLOPs**. The plain mean gives 0.390 / 0.075.
3. So the binding constraint is the **readout, not the representation**. A model that was trained
   with full attention and has never seen a soft token does not compare slot vectors to each other;
   it falls back on the id-position prior, and it does that no matter how good the vectors are. The
   previous record's verdict — "(B) hard limit, the slot does not carry the signal" — holds for the
   *plain mean* but is **too strong as stated**: the plain mean is bad because a document's mean
   embedding is 94% shared common-word mass, and removing that mass is free.
4. **What this does NOT show.** It does not show that a *trained* reader would find the signal: the
   oracle uses an explicit all-pairs comparison the network would have to learn to perform in
   attention. It only removes "there is nothing to find" as the explanation.

**Ranked recommendation (by cost):**

| rank | change | cost | evidence |
|---|---|---|---|
| 1 | Replace the training slot with **`cent_cmean`** (drop punctuation + top-100 corpus ids, subtract the corpus-mean embedding, renormalise to the real-token norm) | **free** — one extra vector and a token-id mask, both precomputed per corpus; `O(doc tokens)`; no extra tokens on the row; nothing per-layer | oracle 0.731 vs 0.390 for the plain mean at 2k; 0.239 vs 0.075 at 8k |
| 2 | `cmean100` alone if the corpus-mean vector is inconvenient | free | oracle 0.674 / 0.107 |
| — | `G > 1` segment means | +1–3 tokens/doc | **rejected**: oracle 0.299/0.261 at 2k, *below* the plain mean |
| — | `enc2` / `enc4` block-local encoder | 0.056 / 0.113 of a dense forward = **0.68× / 1.36× the compacted forward** | **rejected**: ties `cent_cmean` at 2k (0.726 vs 0.731), loses at 8k (0.133 vs 0.239), for a cost that roughly doubles the arm |
| — | a trained summarizer | a second network, trained | **not needed as the next step**: its premise ("the mean carries nothing") is false |

The concrete change is one expression: `_compact_pooled_soft_tokens` in
`src/olmo_core/nn/transformer/model.py` builds `doc_means` by `index_add` over
`self.embeddings(input_ids)`; weighting that sum by a precomputed content mask and subtracting a
precomputed corpus-mean vector is the whole edit. It matters because `detach_soft_kv=True` (the
default in `train_ctc_suite.py`, and what every ds64 arm ran) gives the projector **no LM gradient
on pooled slots**, so the slot is the raw mean for the entire run and training cannot fix it.

**Caveat on scope:** everything here is measured on a model that never trained on slots. The claim
supported is about the *information content* of the slot, not about what training recovers. The
honest next experiment is a cheap trained one: the `fast2k` harness with `cent_cmean` slots at keep
1/6, watching train CE against the `ln(C(n,k))/T` format floor — if the slot is readable at all, a
content-grounded run drops below the floor early (`records/ds64-overnight-2026-09-14.md` decision
rules), and if it parks on the floor the "readout" explanation is dead too and a trained summarizer
is the only route left.

## 7. Runs

| what | where | id | rows | state |
|---|---|---|---|---|
| oracle readout, replicate 2k | sneetches | `3549805` | 200 | done (§4) |
| oracle readout, replicate ~8k (n55) | sneetches | `3549806` | 200 | done (§4) |
| oracle readout, canonical ds64 2k | beaker ceres/saturn | exp `01M2HNZQEQ2QY1JN4FD662X55R`, job `01M2HNZQJM6MWERSTE1CDVBWFK` | 240 | **done** (§4) |
| oracle readout, canonical ds64 8k | beaker ceres/saturn | exp `01M2HP0GBPEVDFYH6SY6JJM5N1`, job `01M2HP0GFGZGWB2DA155RGP4C2` | 240 | **done** (§4) |
| model readout, canonical ds64 2k, all 13 candidates + swaps (41 conditions) | beaker ceres/saturn | exp `01M2HNFEC7P67NP33F8F3DN5Q3`, job `01M2HNFEG0PWX212YS68NJX61W` | 240 (64 gen) | **generation complete** (§5); CE columns still accumulating |
| model readout, canonical ds64 8k, 4 candidates + swaps | beaker ceres/saturn | exp `01M2HP2PDCFV3F6RMVS9D7GRN0`, job `01M2HP2PME4RND9J7EGEJYDH3Y` | 240 (48 gen) | **DONE, all 240 rows** (§5); weka `_eval_results/outlier_slot_probe/outlier_richer_slot_ds64-outlier-dense-u64M_8k_8ksel.json` |
| model readout, replicate 2k, all candidates + swaps | sneetches | `3549785` | 200 (64 gen) | running (slowest; generation completes ~row 64) |
| model readout, replicate ~8k, 4 candidates + swaps | sneetches | `3549832` | 200 (48 gen) | **done** (§5), JSON `/data/prasann/outlier_probe/richer_local_n55.json` |

All four probe runs write JSON; the Beaker ones also write to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/`. Every
number above is quoted with its per-document SE and its row count; **every eval here is < 500 rows**
(240 / 200), and the recall columns average over 144–192 gold documents.

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
