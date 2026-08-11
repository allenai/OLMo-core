# Preview: does `random_doc` p=0.25 route multi-hop? (zero-GPU, observational)

Conditions the **already-run** `random_doc` p=0.25 arm (f1 0.558, `results/hopgold-n50-stage1.md`) on
the connectivity each gold pair *happened* to be dealt, to preview Approach A's question before any
hop-ladder training exists. No GPU, no new run — replays each eval example's segmentation to recover
its true per-example graph and joins to the per-example grading.

Script: `debug/gold_connectivity/stratify_rand025_n50.py`. Data: `contra_n50_v2_orig`, eval
`contradiction_eval_pubmed_both_n50_k3`, **eval_size 488** (1464 gold pairs — pair-grain, so the
per-bucket n below are pairs).

Reference points: chunked floor **0.408**, full ceiling **0.943**.

## 1. Raw conditioning — and two checks that validate the replay

| gold pair's realized connectivity | n_pairs | recall | ±SE |
|---|---|---|---|
| **direct** edge | 330 | **0.879** | ±0.018 |
| 2-hop path | 615 | 0.483 | ±0.020 |
| 3+-hop path | 170 | 0.494 | ±0.038 |
| **unreachable** | 349 | **0.415** | ±0.026 |

Two independent checks say the replay is right:

* **Unreachable pairs score 0.415 ≈ the chunked floor 0.408.** This is exactly what must happen: a gold
  pair with no doc→doc path has only the FREE query/answer channel left, which *is* the chunked
  condition. The floor is reproduced from a completely different direction.
* **Direct-edge pairs score 0.879, near the 0.943 full ceiling**, and the realized direct fraction is
  0.225 against a nominal p=0.246. Buckets sum to 1464 = 488×3.

## 2. ⚠ The naive multi-hop lift is a distance artifact

Reading "2-hop 0.483 vs unreachable 0.415" as a **+0.068 multi-hop lift** is **wrong**. Connectivity and
gold-pair distance are entangled *by construction*: under `random_doc`, a pair at distance `d` has only
`d-1` candidate intermediaries (the mask is causal, so a relay must sit positionally between the pair),
so close pairs are structurally starved of paths.

| bucket | n | mean gold-pair distance | frac dist ≤ 3 |
|---|---|---|---|
| direct | 330 | 16.3 | 0.136 |
| 2-hop | 615 | **22.3** | 0.010 |
| 3+-hop | 170 | 18.6 | 0.000 |
| unreachable | 349 | **6.3** | **0.378** |

Unreachable pairs sit at mean distance **6.3**; 2-hop pairs at **22.3**. And distance moves recall on its
own (U-shaped: 0.585 / 0.507 / 0.548 / 0.592 over dist 1-3 / 4-10 / 11-20 / 21-50). So the two buckets
are not comparable as drawn.

**Distance-stratified, inverse-variance pooled** (only strata with ≥10 pairs in *both* arms carry
weight — the rest are unidentified and are reported as excluded):

| contrast | delta | ±SE | strata support |
|---|---|---|---|
| direct vs unreachable | **+0.472** | ±0.035 | 568/679 pairs |
| direct vs multi-hop | **+0.407** | ±0.026 | 1107/1115 pairs |
| **multi-hop vs unreachable** | **+0.052** | **±0.045** | 740/1134 pairs |

At bin width 10 the multi-hop contrast reads **+0.031 ±0.042**. Both are ~1σ — **statistically
indistinguishable from zero**. The direct-edge effect is ~13σ.

### 2b. The pair-grain SEs are honest (checked, not assumed)

The buckets above are **pairs** (1464), not examples (488) — 3 gold pairs share each context, prompt and
forward, so binomial-on-pairs SEs could have been optimistic. Measured: the intra-example correlation of
correctness is **ICC = −0.088** (design effect **0.82**, SE ×0.91) — slightly *negative*, so the quoted
SEs are mildly **conservative**, not overstated. Mechanically sensible: the model emits exactly 3 pairs,
so pairs compete for slots rather than rise together.

A **cluster bootstrap resampling whole examples** (2000 draws) confirms both headlines:

| contrast | delta | cluster-boot SE | 95% CI |
|---|---|---|---|
| direct vs unreachable | +0.472 | ±0.036 | **[+0.406, +0.546]** |
| multi-hop vs unreachable | +0.052 | ±0.047 | **[−0.039, +0.144]** |

The multi-hop CI straddles zero; the direct-edge CI is nowhere near it.

⚠ **The binding limitation is support, not sample size.** The distance-stratified multi-vs-unreachable
contrast draws on only 740/1134 pairs: past distance ~30 there are **no unreachable pairs at all** (4 at
21–25, 1 at 26–30, 0 beyond), because a distant pair almost always picks up *some* path by chance. That
regime is **unidentified observationally at any eval size** — more examples cannot fix it. Approach A's
`hop_inf` arm constructs those cases by intervention, which is a second thing the ladder buys that
conditioning cannot.

## 3. What this does and does not say

**Does say:** a model trained with `random_doc` p=0.25 solves gold pairs *essentially only when handed
a direct edge*. Once distance is controlled, having a 2-hop or 3-hop path available is worth no
detectable amount over having no path at all. Channel (b) is doing the work; channel (c) is not
visibly in use.

**Does NOT say that multi-hop routing is unlearnable.** This is **observational, not interventional**,
and the confound is not distance — it is *training pressure*. This model was trained with a direct edge
available for ~25% of gold pairs, so it was never forced to route: a policy of "use the edge when it's
there, fall back to the FREE channel otherwise" is sufficient to reach 0.558 and is exactly what the
table shows. The multi-hop machinery may simply never have been learned, rather than being unusable.

**This is why Approach A is necessary, not merely nice.** Its `hop2` arm deletes the direct edge for
**every** gold pair at train time and guarantees a 2-hop path, so routing is the *only* way to win.
That is an intervention on the training distribution, which no amount of conditioning on this run can
substitute for. The sharp prediction:

* `hop2` ≫ 0.415 ⇒ multi-hop routing **is** learnable when forced ⇒ cross-doc attention need not be
  quadratic (the whole point).
* `hop2` ≈ 0.415 (pins to the chunked floor) ⇒ channel (c) is not usable at this scale, and the
  `hop_inf` control should confirm it.

Note the ladder is well-powered for this: `hop2` vs `hop_inf` is a between-arm contrast at
eval_size 488 (SE ≈ 0.022 each), and the gap to detect is against a floor that we now know three
separate ways (chunked arm 0.408, unreachable pairs 0.415, hier-K10's collapse).

## 4. Caveat on the graph replayed

The eval **prefill** is prompt-only (no answer, no EOS, no padding), so the per-example
`random_doc_per_example` nonce — which hashes the token-level `chunk_ids` — differs between train and
eval for the same example. That is a property of the arm, not of this analysis: the graph replayed here
is the one the model's mask is actually built from **at eval**, which is the right conditioner for eval
recall. It does mean an example's train-time graph and eval-time graph are different draws, which is
*intended* (per-example randomization is what denies the model a memorizable fixed graph).
