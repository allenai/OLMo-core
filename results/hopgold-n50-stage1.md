# Stage 1 — the n=50 floor / ceiling / base-graph gates (Qwen3-0.6B, contradiction)

Setup for the multi-hop gold-routing experiment (`records/multihop-gold-routing-experiment.md`):
measure the n=50 floor and ceiling before building any hop ladder.

**Setup (identical across arms unless stated).** Data `contra_n50_v2_orig` — the **original**
generator's examples (`contradiction_train_pubmed_both_n50_k3.jsonl`, 2000 ex), v2 leak-free layout,
no-cot. Base `q06b-dense-cpt-modelonly-trainedmark` (norm-repaired markers). seq_len 3200, 3 epochs =
**750 steps**, global batch 8 sequences, DP=2, LR 5e-5, eager (`--no-compile`). Curriculum mask-mixing
on, `p_standard` 0.8 → **0.0 verified** (0.001 at the last step). Eval
`contradiction_eval_pubmed_both_n50_k3`, **eval_size = 488** (the entire file), parse_rate 1.00, no
mixing at eval. Binomial SE quoted; **every arm is a single seed**, and seed noise adds on top.

`CE` = mean of the last 50 logged steps (the last single step is noisy — `standard` reads 0.0019 at the
last step vs 0.0134 averaged, a 7× difference).

## 1. Result

| arm | cross-doc connectivity | CE (mean-50) | f1 | ±SE |
|---|---|---|---|---|
| `standard` (plain causal) | full | 0.0134 | **0.943** | ±0.011 |
| `random_doc` p=0.25, per-example | random 25% | 0.1369 | **0.558** | ±0.022 |
| `chunked` (pure) | **none** | 0.2207 | **0.408** | ±0.022 |

**Gate 1 — headroom: PASSES decisively.** `standard − chunked` = **0.535**, larger than n=100's 0.493.
n=50 is a better-resolved setting than expected, at roughly half the sequence length (3151 vs ~6144
tokens).

**Gate 2 — can a p=0.25 base route at all? PASSES.** `random_doc` p=0.25 scores **0.558**, clearing
the chunked floor by **0.150 (~7 SE)** and sitting ~28% of the way up the headroom. This is the gate
that mattered: `hier-K10` had **0% unreachable pairs and max 3 hops and still died at the floor**
(0.505 vs a 0.441 floor at n=100), so a sparse base can be structurally connected yet untrainable.
p=0.25 (out-degree 6.02, close to hier-K10's 6.55) does **not** die. Approach A's hop ladder is
therefore resolvable on this base and does **not** need to move to p=0.5.

⚠ **The 0.558 is not yet evidence of multi-hop routing.** At p=0.25 exactly **24.6%** of gold pairs
get a direct edge by coin flip, which is suspiciously close to where the arm lands in the headroom. The
0.558 is consistent with *either* "the model routes multi-hop" *or* "the model solves only the ~25% of
pairs it was handed a direct edge for". Separating those is precisely Approach A's job — and can be
previewed with no GPU by conditioning per-pair recall on each pair's realized connectivity
(`debug/gold_connectivity/stratify_rand025_n50.py`).

**A consistency check on the §2 data diagnosis.** On the poisoned `n50_v2_7k` data the same three arms
read `full` 0.585 / `rand0.25` 0.518 / `chunked` 0.267 — i.e. a 25%-connected graph nearly *matched*
full attention (−0.067). That is what a similarity shortcut looks like: "pick the two most similar
claims" needs almost no cross-doc routing, so sparsifying the graph costs almost nothing. On correct
data the same contrast is **−0.385**. The arms only separate when the task actually requires routing,
which is independent corroboration that the recombined data was measuring the wrong thing.

Sanity check against the n=100 reference (`results/masks-n100.md`): `standard` 0.943 here vs 0.934
there, `chunked` 0.408 vs 0.441. Both sit where they should — n=50 is the *easier* corpus (the eval
rungs n6…n250 are the **same 488 examples and 1464 pairs**, differing only in distractor count), so a
slightly higher ceiling is expected.

## 2. ⚠ The first attempt measured a shortcut, not the masks

Stage 1 was first run on `contra_n50_v2_7k` — 7,421 zero-reuse examples built by recombining the 22,265
LLM-authored contradiction pairs with distractors drawn from a 386k global filler pool. **Every arm was
uniformly and badly depressed:**

| arm | recombined data | original data | delta |
|---|---|---|---|
| `standard` | 0.585 | **0.943** | **−0.358** (~15 SE) |
| `chunked` | 0.267 | **0.408** | −0.141 (~6 SE) |

The tell was **`standard` = 0.585 at n=50 while n=100 scores 0.934** — impossible, since n=50 is
strictly easier. A uniform depression across every arm indicts the *data*, not the masks.

**Cause: the recombination destroyed the hard negatives.** Word-overlap Jaccard over 120 examples:

| dataset | gold-pair sim | **best NON-gold sim** | margin |
|---|---|---|---|
| recombined (global filler pool) | 0.461 | **0.163** | **0.30** — trivial |
| official eval n50 | 0.445 | **0.372** | 0.07 — hard |
| original train n50 | 0.454 | **0.333** | 0.12 |

Gold pairs are equally similar in all three (~0.45); the **distractors** are what changed. The real
generator co-samples distractors *with* the gold pair from a related pool, so the nearest non-gold pair
is nearly as similar as the gold one and the model must genuinely detect contradiction. A global filler
pool drops the nearest distractor to 0.163, so **"pick the two most similar claims" solves training** —
a shortcut that collapses on the topically-coherent eval.

⚠ **Every validation of that dataset passed**: 0 eval contamination (14 poisoned pairs found and
dropped), 0 pair reuse, gold-pair distance matched to eval (17.0 vs 16.7), 0 duplicate docs, 0
filler-secretly-gold. The checks verified the properties someone thought to check; none asked whether
the task was still the same task. **Difficulty is a property of the joint (gold, distractor) sample and
cannot be reconstructed by recombination.** Diagnostic signature: **normal train CE (0.0144, matching a
good n=100 run) with far-below-expected eval f1** — that gap is a distribution mismatch, not
undertraining, which would have shown up in CE. See `records/contradiction-data-and-base-hygiene.md`
§3b.

## 3. ⚠ A second silent invalidation, caught before it reached a conclusion

An earlier pass of these arms ran with `p_standard` ending at **0.601 instead of 0.0** — ~60% of
forwards still plain causal at the end of training, so the masks were never fully in force. Cause: the
`--micro-batch-instances 4` throughput knob cut each rank's forward count 4× while
`mix_total_forwards` still assumed 1 instance/forward (927 real forwards vs an assumed 3710 ⇒ 25% of
the way down the anneal). Same class as the known NGPU anneal bug, via a second divisor.

It reported `rc=0`, step 927/927, and a **healthier** chunked CE (0.069 vs the correct 0.304) —
precisely *because* it was training unmasked. Nothing looked wrong. The launcher now re-derives
forwards-per-rank and **hard-fails** if the curriculum would miss `mix_end_p` (verified: fires on the
0.601 config, passes at 0.0009).

## 4. Verdict

n=50 on the **original** data is a well-resolved setting: floor **0.408**, ceiling **0.943**, headroom
**0.535** — better than n=100's, at half the sequence length. Gate 1 is passed. Gate 2 (does
`random_doc` p=0.25 clear the floor, or pin to it like `hier-K10` did?) is pending.

**Use `contra_n50_v2_orig`. Do not use `contra_n50_v2_7k` for anything.**
