# Stage 0: does hier-K50 beat random-k0.5 because of *gold-pair* connectivity? (analytic)

**Question.** `hierarchical_dilated` K=50 (f1 **0.831**) beats `random_doc` k=0.5 (f1 **0.735**) at
n=100. Hypothesis: hier wins because its fixed strided pattern connects the **gold contradiction
pairs** more directly than random's coin-flip, not because dilation is a better inductive bias.

**Method.** Both masks are deterministic functions of chunk indices (`random_doc_per_example=False`,
seed 42), so the doc→doc graph is identical across all 488 eval examples. Built it once from the
**real** `build_chunked_allowed_mask` (hier: union over 28 layers = cross-layer residual reachability;
random: the single seeded graph), then looked up each example's `gold_doc_indices`. No model forward.
Eval set `contradiction_eval_pubmed_both_n100_k3` (488 ex, 1464 gold pairs). Script +
per-example scores: `debug/gold_connectivity/`.

Context↔context edges only (FREE query/answer tokens bridge every doc in *all* masks — that is how
pure `chunked` still reaches 0.441 — so the connectivity hier/random *add* is the in-context doc↔doc
mixing before the trailing FREE aggregation).

## Result 1 — hier does connect gold pairs more directly (premise holds)

| under mask | gold-pair direct 1-hop | reachable at all | mean hops (reachable) |
|---|---|---|---|
| HIER K=50 | **0.874** | 1.000 | **1.13** (max 2) |
| RANDOM k=0.5 | 0.498 | 0.970 | 1.49 (max 3) |

Hier connects the gold pair directly ~1.75× as often, always within 2 hops; random leaves ~3% of gold
pairs with **no** in-context path at all (FREE-bridge only).

## Result 2 — but it is NOT gold-specific (the hypothesis, as stated, is refuted)

Distance-matched **random non-gold** pairs (same chunk-distance distribution as gold) are connected
*identically*:

| under mask | GOLD direct / hops | CONTROL (non-gold, matched dist) direct / hops |
|---|---|---|
| HIER K=50 | 0.874 / 1.13 | **0.874 / 1.13** |
| RANDOM k=0.5 | 0.498 / 1.49 | **0.530 / 1.46** |

Connectivity here is a pure function of chunk-distance, and gold pairs sit at ordinary distances
(mean 32.6 vs ~33.3 for a uniform random pair). Hier connects *every* pair better, gold or not —
there is no gold-targeted alignment. ⇒ You cannot move gold-pair connectivity independently of
general connectivity under these gold-agnostic masks; the original "ablate gold connectivity" framing
does not isolate a distinct mechanism.

## Result 3 — and it is NOT raw per-layer density (hier wins while per-layer SPARSER)

| mask | per-layer avg out-degree | union (28-layer) out-degree |
|---|---|---|
| HIER K=50 (stride rotates 1,2,4,8,16) | **19.4** | **43.2** |
| RANDOM k=0.5 (fixed graph, all layers) | 25.3 | 25.3 |

Per single layer hier is **sparser** than random (19.4 < 25.3), yet it wins. Its edge budget is spent
on a *rotating* strided pattern, so across depth the union reaches more distinct pairs **and** gives
every pair a ≤2-hop route; random reuses one fixed graph every layer (redundant, longer paths, 3%
unreachable). random k≈0.85 matches hier's union out-degree (42.1).

## Verdict

The win tracks **general cross-doc reachability / short path length**, achieved *cheaply per layer* by
hier's layer-rotation — **not** gold-pair-specific alignment, and **not** raw per-layer edge count.
The user's instinct ("it's connectivity to the relevant docs, not dilation magic") is right that it is
connectivity, but it is *global* short-path connectivity; the gold pair is connected only as well as
any other pair at its distance.

⚠ This is analytic *potential* information flow, not proof the model uses it. Confirm with Part B
(stratify per-example eval f1 by the per-example connectivity scores already written to
`debug/gold_connectivity/per_example_connectivity.json`) — needs a GPU eval re-run that dumps
per-example correctness, which the current aggregate-only eval JSONs do not have.

## Part B — per-example eval f1 DOES track gold-pair connectivity (the model uses it)

Re-evaled the three checkpoints with a per-example dump (patched `eval_lc_native_docchunk_contra.py
--per-example-out`; in-tree code, aggregate f1 reproduced exactly: 0.831 / 0.735 / 0.505, parse 1.00)
and joined with the analytic connectivity. `debug/gold_connectivity/{eval_perex,stratify.py}`.

**Within `random_doc` k=0.5** — a near-randomized experiment: random's per-pair keep is a
distance-independent coin flip, so whether a given example's gold pairs are connected is as-good-as-
random w.r.t. difficulty. f1 tracks it strongly:

| this example's gold pairs, under random | f1 | n |
|---|---|---|
| all directly connected (frac=1.0) | **0.860** ±0.024 | 62 |
| partially | 0.726 ±0.012 | 360 |
| none directly connected (frac=0) | **0.662** ±0.031 | 66 |

+0.198 (≈5 SE) from none→all-direct. When the coin flip links the gold pair, random solves it like
hier; when it misses, random falls toward the chunked floor.

**Cross-model (the decisive test).** The entire hier-K50 > random win lives in the examples where
random FAILS to connect the gold pair; condition on random already connecting it and the advantage
vanishes (reverses, within noise):

| split by how well RANDOM connects the gold pairs | hier-K50 f1 | random f1 | gap | n |
|---|---|---|---|---|
| random connects NONE | 0.828 | 0.662 | **+0.167** | 66 |
| random connects SOME | 0.834 | 0.726 | +0.108 | 360 |
| random connects ALL 1-hop | 0.817 | 0.860 | **−0.043** | 62 |
| overall | 0.831 | 0.735 | +0.097 | 488 |

**hier-K50** shows no within-model gradient (all-direct 0.822 vs partial 0.849) — it is
connectivity-*saturated* (87% of gold pairs 1-hop, 100% ≤2 hops), so almost no poorly-connected tail.
**Caveat — hier-K10 is flat** (all-direct 0.515 vs none 0.503): stuck at ~0.505 ≈ the chunked floor,
it apparently never learned to route through its sparse strided edges, so per-pair connectivity is
irrelevant there. Connectivity predicts success only for models that learned to use cross-doc edges
(random, hier-K50), not for one operating at the FREE-only floor.

### Part B verdict
The dilated win **is** a gold-pair-connectivity effect after all — just not a gold-*targeted* one.
hier doesn't aim at gold (it can't; Part A), but its denser, better-distributed reach connects the
gold pair more often, and **conditioning on gold-pair connectivity erases the hier advantage**
(random ties/beats hier where it connects the pair). The user's original instinct is vindicated in a
precise form: it's connectivity to the relevant docs, not dilation as an inductive bias — random
attention does just as well whenever it happens to wire up the gold pair.

## Reframed clean experiment (for the interventional stage)

1. **Reachability-matched hier vs random** (kills the density/topology confound): hier-K50 vs
   `random_doc` k≈0.85 (union-density-matched). Plus the free lever from existing data — hier already
   wins with *fewer per-layer edges* (19.4 vs 25.3), so per-layer density is not the driver.
2. **f1-vs-connectivity curve**: sweep `dilation_n` and `doc_keep_prob` to trace f1 against union
   reachability for both families. One shared curve ⇒ pure connectivity, dilation structure
   irrelevant. Hier's curve above random's at matched reachability ⇒ the rotating short-path structure
   matters beyond edge count.
3. **Path-length-to-relevant-pair placement probe** (the salvaged, reframed gold-placement idea):
   place gold pairs at distances hier routes in 1 hop vs distances it reaches only in 2 hops, under
   the fixed gold-agnostic mask, and test whether 1-hop-gold examples are solved better. Now a
   "does routing *depth* to the relevant docs matter" question, not "gold vs dilation".
