# Multi-hop gold routing: can a model use two documents that never attend to each other?

**Status:** proposal, awaiting refinement. Nothing trained yet.

## The question

Cross-document information can reach the answer by three channels:

- **(a) via FREE tokens** — the query/answer positions attend to everything, so every mask bridges
  every doc pair at the trailing FREE block. This is why pure `chunked` still scores 0.441 at n=100.
- **(b) direct doc→doc edge** at some layer.
- **(c) a multi-hop path across layers** — doc `b` reads doc `m` at layer L, doc `m` already read doc
  `a` at layer L-1.

If **(c)** works, cross-document attention does not need to be quadratic: it suffices that the gold
docs be *connected*, not *adjacent*. This experiment forbids (b) for the gold pair and asks whether
(c) recovers the loss — i.e. whether the model beats the (a)-only `chunked` floor.

## Why this is worth running now

`results/gold-connectivity-stage0.md` established **observationally** that gold-pair connectivity
drives f1: within `random_doc` k=0.5, examples whose gold pair happened to get a direct edge scored
**0.860** vs **0.662** when it did not — and conditioning on that erases the entire hier-K50 > random
advantage. But every one of those "not directly connected" examples still had a 2–3-hop path (97%
reachable), and it scored 0.662 — well above the chunked model's 0.441. **That is a hint that (c)
already works**, but it is cross-model and observational. This proposal makes it interventional.

## Corpus size: n=50

n=20 is too easy to measure on: pure `chunked` is ≈0.7 there, against a full-attention ceiling of
≈0.93 — only ~0.23 of headroom, and the effect we want to detect lives inside it. n=100 has the clean
spread (chunked **0.441** → full **0.934**) but costs 6144-token sequences. **n=50 is the compromise**:
expected chunked floor ≈0.55–0.60, ceiling ≈0.93, so ~0.35 of headroom at roughly half the sequence
length. The n=50 eval set already exists (`contradiction_eval_pubmed_both_n50_k3`, **eval_size=488**,
the entire file — accepted as-is per CLAUDE.md).

⚠ The chunked/full floor+ceiling at n=50 are **estimates interpolated from n=20 and n=100**. Stage 1
measures them before we commit to the design; if the gap is <0.2 the design is underpowered and we
should fall back to n=100.

## Data (Stage 1) — 7,421 zero-reuse examples, no LLM calls

**The pairs already exist.** Scanning every `contradiction_*pubmed*.jsonl`:

| pool | unique pairs | note |
|---|---|---|
| `both` | 6000 | `--mode both` = 50/50 simple+subtle — matches the eval |
| `subtle` | 11679 | same style family as half of `both` |
| `simple` | 4641 | same style family as the other half |
| `realistic` | 4737 | **different** style ("fully rephrases, no near-duplicate tells") — exclude |
| **style-matched clean pool** | **22,265** | `both ∪ simple ∪ subtle` − all eval pairs |

22,265 pairs ⇒ **7,421 examples at zero pair reuse** — the chosen size. (10k would need ~1.35× reuse;
not worth it, and zero-reuse removes any memorization confound outright.) No gemini calls needed.

⚠ **Contamination found and must be fixed: 14 train pairs appear verbatim in the eval sets** (13 from
`simple`, 1 from `subtle`). The `both`-only pool is clean (0 overlap), which is why existing results
are safe — but the moment we pool `simple`/`subtle` we inherit those 14. The builder drops every pair
that appears in *any* eval file.

**New code required.** The existing `--expand-from-train` is 1 source example → 1 output example, so
it cannot exceed 2000. We need a **recombination** builder (`build_contra_n50_10k.py`):

1. Pool the 22,265 style-matched pairs, drop the 14 eval-contaminated ones.
2. Per example: sample 3 disjoint pairs, each used exactly once across the whole set.

   ⚠ **Zero reuse and eval-matched style are in tension — pick one.** The eval is `both` = **50/50
   simple:subtle**. The clean pool is ~7,628 simple-family vs ~14,678 subtle-family = **34/66**. So:
   - **(i) 7,421 examples, zero reuse, 34/66 skew** — train is *harder* than eval (subtle-heavy).
     Shared identically by every arm, so the `hop2 − hop∞` contrast is unaffected; only absolute f1
     comparability to prior `both`-mode n=100 numbers suffers. **Default per your call.**
   - **(ii) 5,085 examples, zero reuse, exact 50/50** — caps subtle to match simple, discarding ~7k
     pairs.

   Recommend **(i)**: the contrast is what carries the claim, and 7,421 > 5,085 examples matters more
   than absolute-f1 comparability to a different corpus size.
3. Fill to 50 docs with distractors drawn from non-gold docs, **excluding every text that appears in
   any gold pair anywhere** — otherwise a filler could silently be one half of another example's
   contradiction and create an unlabeled positive (label noise).
4. Shuffle positions, emit the standard schema (`documents`, `queries`, `answers`,
   `gold_doc_indices` 1-indexed pairs, `source`).
5. Shuffle positions to **match the eval's gold-pair distance distribution** (mean 16.7, median 15,
   min 1, max 48) — see "placement" below. Do **not** impose a minimum distance.
6. Tokenize with `convert_unified_to_document_landmark.py --emit dense --emit-gold-sidecar` (**v2
   leak-free layout — `Claim N:` inside the chunk**) → `contra_n50_v2_7k`.

Note the existing n=20 shards are **v1-era**, i.e. they carry the free-token leak that made goldgrad
"replicate" on v1 and collapse on clean v2. Building fresh v2 data is not optional.

**Gold sidecar must preserve pairs.** `gold_fingerprints.json` is a *flat set* — it cannot say which
doc contradicts which, which is exactly the defect that invalidated the first goldgrad arms. The label
span literally is the pair list (`[[a,b],[c,d],[e,f]]`), so `gold_pairs.json` is recoverable exactly
(`build_gold_sidecar_from_shard.py` does the regex + validation). This experiment is **pair-critical**:
"delete the gold edge" is meaningless without knowing the partner.

## The design

One dataset, fixed gold placement. **Arms differ only in the mask**, so position/content are held
exactly constant — no data-side confound.

| arm | gold↔gold direct edge | gold path exists | channels | role |
|---|---|---|---|---|
| `full` (`standard`) | yes (all edges) | — | a+b+c | ceiling |
| `hop1` | **forced present** | 1 hop | a+b+c | upper reference |
| `hop2` | **deleted** | forced 2-hop | a+c | **the test** |
| `hop3` | **deleted** | forced 3-hop | a+c | depth of routing |
| `hop∞` | **deleted** | **no path** | a only | **leak-matched control** |
| `chunked` | none (no doc edges at all) | none | a only | floor |

**Read:** `hop2 > hop∞` ⇒ channel (c) is real and the model learns to route. `hop2 ≈ hop∞` ⇒ it cannot,
and everything above the chunked floor is just the FREE bridge.

### The confound that decides this experiment: the mask leaks the answer

A gold-aware mask is a **train-time signal that encodes the answer**. If gold docs are the only pair
that never attend to each other, then "the doc I am *not* allowed to see" *is* "my contradiction
partner" — the model can score f1 without reading anything. This is not hypothetical: goldgrad's
`gpr8` arm, which merely over-represented gold in the *backward* pass, collapsed to f1 0.002.

Two mitigations, both in the design:

1. **Camouflage.** Build on a gold-agnostic sparse base graph (`random_doc` **p=0.25**,
   **per-example**), where 75% of doc pairs are already non-adjacent by coin flip. Deleting the gold
   edge is then statistically invisible — it looks like an ordinary miss. See §"Base graph" for why
   p=0.25 and why per-example is mandatory.
2. **`hop∞` is the control that neutralizes it.** `hop∞` and `hop2` have **identical** gold-edge-absent
   structure and therefore *identical* leak. They differ only in whether a path exists. So the
   `hop2 − hop∞` contrast is leak-cancelled by construction — the leak inflates both arms equally.
   This is why `hop∞`, not `chunked`, is the comparison that carries the claim.

⚠ `chunked` is **not** leak-matched (it has no doc edges at all, so no hole to read). Use it as a
scale reference, never as the control for `hop2`.

### Base graph: `random_doc`, p=0.25, per-example, layer-invariant

**Layer-invariant is a feature here.** `random_doc` hashes `(query_doc, key_doc, seed)` with **no layer
term** (`chunked_mask.py:429`), so one fixed graph serves every layer; only `hierarchical_dilated`
reads `layer_idx`. That makes "hops" unambiguous — a single graph, path length = hop count, traversed
across depth (layer L: `m` reads `a`; layer L+1: `b` reads `m`). A layer-rotating mask would make "number
of hops" ill-defined, which is why we do **not** build this on `hierarchical_dilated`.

⚠ **`random_doc_per_example=True` is mandatory — a deviation from every prior random_doc run.** Prior
runs (masks-n100, Stage 0) used `random_doc_per_example=False, seed=42`: *one* graph shared across all
layers **and all 488 examples**. With a graph that fixed, the model can memorize it — and then a
*missing* edge is a flashing indicator that the doc is gold, amplifying the very leak we camouflage.
Per-example resampling makes each example's absent set fresh.

Measured on the real builder at n=50 (one token per doc ⇒ the `(S,S)` mask *is* the doc graph):

| p | out-deg (of 24.5) | direct | 2-hop | unreachable | path survives deleting direct edge |
|---|---|---|---|---|---|
| 0.5 | 12.0 | 0.49 | 0.43 | 0.064 | 0.905 |
| **0.25** | 6.0 | 0.25 | 0.38 | **0.312** | **0.625** |

p=0.25 **halves the residual leak**: excluding attended docs shrinks the partner candidate set only
~1.14× (49→43), vs ~1.32× (49→37) at p=0.5.

⚠ **But p=0.25 risks the `hier-K10` trap.** At out-degree 6 with 31% of pairs unreachable at *any* hop,
the base is close to the regime where hier-K10 "never learned to route through its sparse strided
edges" and sat at the chunked floor — where per-pair connectivity became **irrelevant** (Stage 0 Part
B). In that regime `hop2` and `hop∞` would both pin to the floor and the null would be meaningless.
Hence the Stage 1 gate below. Fallback: p=0.5, whose larger leak `hop∞` cancels anyway.

### The causality constraint (easy to miss)

The mask is causal: doc `b` attends doc `a` only if `a < b`. So a 2-hop path `b→m→a` requires
`a < m < b` — **the intermediary must sit positionally between the gold docs**. An adjacent gold pair
(`b = a+1`) has *no* possible intermediary and is silently forced into `hop∞` no matter what the mask
says.

**Measured gold-pair distances** (the eval file is fixed, so it sets the terms):

| split | mean | median | min | dist=1 (unroutable) | dist<4 |
|---|---|---|---|---|---|
| eval n50 | 16.7 | 15 | 1 | **3.96%** | 12.5% |
| train n50 (existing) | 16.9 | 15 | 1 | 4.03% | 12.1% |

⚠ **Do not impose a minimum gold distance in the training data** (an earlier draft of this doc did).
It would guarantee routability, but 12.5% of *eval* pairs sit below any threshold ≥4 and the eval set
cannot be rebuilt — so the model would never train on the close pairs it is tested on. A train/eval
placement mismatch is a worse problem than unroutability. Instead: **match the eval distribution
exactly** and accept that ~4% of gold pairs are adjacent and therefore unroutable.

Consequence, stated plainly: in the `hop2` arm ~4% of gold pairs are *silently* `hop∞`, so `hop2`'s f1
is a 96/4 mixture. This **dilutes `hop2` toward `hop∞`** — i.e. it biases against our own hypothesis,
which is the safe direction. Stage 4 stratifies eval f1 by gold-pair distance and reports `hop2`
restricted to the routable subset (distance ≥ 2) as the secondary, mixture-free number.

### Depth is not the binding constraint

Qwen3-0.6B has 28 layers and the doc graph is layer-invariant for every pattern except
`hierarchical_dilated`, so L layers afford up to L hops. A 2–3-hop path has ~10× the depth it needs.
If `hop2` fails it is an **optimization/learnability** result, not a capacity one — worth saying out
loud, since that is the interesting reading.

### Open design tension (for refinement)

Forcing shortest-path *exactly* h means deleting every shorter path. On a dense base that means
deleting many edges, so density drifts across arms — trading the leak confound for a density confound.
Options:

- **(A) Recommended.** Sparse camouflage base (p≈0.5) + minimal targeted edits (~3 edges out of ~600
  per example — negligible density drift), accept that `hop2` means "shortest path = 2" and `hop3`
  requires a sparser base to be constructible at all.
- **(B)** Sparse structured base (`doc_window` k=1 chain) where path length ≡ gold distance, so hops
  are set by *placement* not mask edits. Density identical across arms, but gold placement then
  differs across arms — swaps the confound for a position confound.
- **(C)** User's Approach 2 — landmark hierarchy: everything reaches a landmark in 1 hop, landmarks
  interconnect, so all pairs sit at a uniform 2–3 hops. Most "natural", least knob-precise.

I recommend **(A)** for the primary ladder, with `hop3` possibly dropped if the base cannot support it
cleanly, and **(C)** as the follow-up that tests whether the finding survives in a realistic
architecture.

## New code required

Nothing existing does this — **every mask in the tree is gold-agnostic** (`grep gold
src/olmo_core/nn/attention/` hits only `gold_grad_mask.py`, which is a *backward*-graph pruner, not a
mask). Needed:

1. `build_contra_n50_10k.py` — the recombination data builder (above).
2. **A gold-aware mask.** Reuse `gold_grad_mask.py`'s proven mechanism verbatim: a `forward_pre_hook`
   that fingerprints `input_ids` (SHA1 to first EOS) and looks up gold in a sidecar — gold identity
   never enters the token stream. Instead of detaching K/V, it emits a per-example doc→doc adjacency
   override that `DocumentChunkedAttention` consumes. New `AttentionPattern` name, e.g.
   `gold_hop_controlled`, with knobs `hops ∈ {1,2,3,inf}`, `base_keep_prob`, `seed`.
3. Extend `Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py` with `--gold-pairs` and `--hops`.

Two inherited constraints: **`--no-compile`** (per-forward Python fingerprint is not
compile-capturable) and the **`.contiguous()`** on edited K/V (flash-attn's backward reads out of
bounds and SIGSEGVs at seq 6144 without it — observed deterministically).

**Verification before any GPU time:** a CPU unit test asserting, per example, that (i) no gold→gold
edge survives in hop2/hop3/hop∞, (ii) shortest gold path is exactly h (∞ for hop∞), (iii) per-example
edge count is matched across arms within ~1%, (iv) FREE-token rows are untouched.

## Training config

Standing recipe from `attn_explore/`, held identical across every arm:

| knob | value |
|---|---|
| model | Qwen3-0.6B, `document_chunked=True` |
| base | `/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-trainedmark` (**norm-repaired markers — mandatory**) |
| data | `contra_n50_v2_7k` (7,421 ex, zero pair reuse) |
| epochs | **1** (7,421 fresh zero-reuse examples ≫ 2000×3 repeated) |
| seq_len | ~3584 (n=50 ⇒ ~½ of n=100's 6144; set from the real max after tokenization) |
| LR / sched | 5e-5, `LinearWithWarmup` |
| global batch | 8 sequences (`grad_accum`×`seq_len`, world-size independent) |
| steps | **927** (7,421 ÷ global batch 8, 1 epoch) |
| compile | **OFF** (`--no-compile`) — Stage 2+'s gold-aware mask cannot be compiled, so every arm stays eager and on one code path |
| dataloader | **`--num-workers 0`** — 2 workers deadlock when several torchrun jobs share a node |
| GPUs | **2/arm, arms run concurrently** (global batch is 8 instances regardless of world size, so 8 GPUs would give 1 instance/rank and comms would dominate) |
| micro-batch | `--micro-batch-instances 4` — the rank's whole share in one forward; accumulation is exact so this is mathematically identical |
| timing | ⚠ *provisional*: 0.56 s/step, but measured at steps 2–5 of a **single isolated** job — i.e. warmup, and without the contention of 3 concurrent arms. Re-measure at steady state before quoting. |
| mask mixing | **ON** — curriculum, `p_standard` 0.8 → **0.0** (see below); **OFF at eval** |
| eval | `contradiction_eval_pubmed_both_n50_k3`, eval_size=488, no mixing, `--eos-token-id 151643`, **MAXLEN ≥ 8192** |
| seeds | 2 per arm on the decisive `hop2`/`hop∞` contrast |

**Mask-mixing stays ON (project default), and that is defensible — because the anneal reaches zero.**
The concern was that mixing runs a large fraction of forwards under *plain causal*, handing the model
the exact direct gold edge this experiment forbids. Verified in code: the world-size fix is in place
(`mix_total_forwards = (n_examples * epochs) // world_size`,
`Qwen3-0.6B-docchunk-mask-mix-...-local.py:100-101`), so `p_standard` genuinely anneals **0.8 → 0.0**
rather than stalling at `0.8*(1-1/NGPU)`. It is therefore a *curriculum* that terminates in a pure
target-mask phase, not a persistent leak.

⚠ **Re-verify the anneal actually hits 0 in the logs of every arm** (`mix_log_interval=5`). Under the
old NGPU bug it floored at ~0.4 with DP=2, meaning 40% of forwards would keep plain causal *forever* —
which would silently void this experiment entirely. This is the single highest-risk config item here.

Two honest caveats, neither fatal:
- **Interpretation.** The claim becomes "trained under a full-attention curriculum annealed to zero
  direct-edge exposure, then evaluated with no direct edge" — not "never saw a direct edge in its
  life." The `hop2 − hop∞` contrast is unaffected: both arms get identical mixing, so it cancels.
- **Suggested refinement.** Scale `mix_total_forwards` to ~0.8× the run so `p_standard` reaches 0 at
  80% of training, leaving a clean final fifth under the pure target mask. As-is the anneal only
  touches 0 at the very last step.

**Eval is always unmixed** — no `standard` collapse, no direct gold edge. This is already the default
(`mask_mix` is train-only), but assert it per arm rather than trusting it.

⚠ **MAXLEN ≥ 8192 at eval.** Truncating the prompt yields an empty generation and f1 exactly 0.000 at
parse_rate 1.0 for a *perfect* model — this already faked a "goldgrad doesn't replicate" conclusion
once. Dump generations whenever a trained arm evals at 0.

## Power

At f1≈0.6, binomial SE on 488 examples is ±0.022, so a `hop2 − hop∞` gap needs **≳0.07** (~3 SE
combined) to be real — and that ignores seed noise, which is why the decisive contrast gets 2 seeds.
Expected floor→ceiling spread at n=50 is ~0.35, so the design can resolve an effect that recovers
≳20% of the chunked→full gap. **A null is only interpretable if `hop1` lands near `full` and `hop∞`
near `chunked`** — that bracket is what proves the ladder has resolution at all. If the bracket
collapses, the experiment is uninformative regardless of what `hop2` does.

## Stage plan

- **Stage 1 (now, CPU + 3 GPU runs).** Build `contra_n50_v2_7k`; train **`chunked`**, **`full`**, and
  **gold-agnostic `random_doc p=0.25` (per-example)** for 1 epoch. Two gates:
  1. **Headroom:** `full − chunked` ≥ 0.2, else fall back to n=100.
  2. **Base-graph routability:** `random_doc p=0.25` must land meaningfully **above** the chunked
     floor. If it pins to the floor we are in the `hier-K10` regime where connectivity is irrelevant
     and no hop ladder can resolve anything — fall back to p=0.5.
- **Stage 2.** Implement + CPU-verify the `gold_hop_controlled` mask.
- **Stage 3.** Train `hop1`/`hop2`/`hop∞` (+`hop3` if constructible), 2 seeds on `hop2`/`hop∞`.
- **Stage 4.** Stratify per-example f1 by realized path length (reuse
  `debug/gold_connectivity/stratify.py` + `--per-example-out`), which converts the headline into a
  dose-response curve instead of a single contrast.

## Approach A — the gold-aware hop-controlled mask (IMPLEMENTED)

**Status:** implemented + CPU-verified 2026-07-16. Nothing trained. Code:
`src/olmo_core/nn/attention/gold_hop_mask.py`, the `"gold_hop_controlled"` `AttentionPattern` in
`chunked_mask.py`, tests `src/test/nn/attention/gold_hop_mask_test.py` (**147 CPU tests, all green**),
measurement harness `debug/gold_hop/measure_arms.py`, launcher patch
`src/scripts/train/memexpress/attn_explore/GOLDHOP_LAUNCHER_PATCH.md`.

### The sidecar exists now

`build_gold_sidecar_from_shard.py --emit pairs` writes `gold_pairs.json`. Built for
`contra_n50_v2_orig`: **2000/2000 examples resolve, 6000 pairs**, and the flattened pairs reproduce the
existing flat `gold_fingerprints.json` **exactly** (0 mismatches). The two parses are independent (inner
`[a, b]` groups vs the flat `\d+` scan) and are required to agree per example, so that is a checked
property, not a comment. Independently reproduces this document's gold-pair distances: mean **16.91**,
median 15, min 1, max 49, **4.03% adjacent**.

### Measured: the arms are structurally matched (`debug/gold_hop/measure_arms.py`, 400 examples)

Base = gold-agnostic `random_doc` **per-example**, seed 42.

| arm | out-deg (p=0.25) | out-deg (p=0.5) | **edge drift** | realized hops | unroutable |
|---|---|---|---|---|---|
| `hop1` | 6.126 | 12.251 | **0 (max abs 0)** | 1: 1.000 | 0.00% |
| `hop2` | 6.126 | 12.251 | **0 (max abs 0)** | 2: 0.965, inf: 0.035 | **3.50%** |
| `hop3` | 6.126 | 12.251 | **0 (max abs 0)** | 3: 0.924, inf: 0.076 | **7.58%** |
| `hop_inf` | 6.126 | 12.251 | **0 (max abs 0)** | inf: 1.000 | 0.00% |

**Edge drift is exactly 0 on every example, at both densities** — better than the "~1%" the design hoped
for. This is not luck: the base graph is a pure function of `(seed, keep_prob, nonce)`, so the *same*
example gets the *identical* base graph in every arm, and the edit is compensated back to that graph's
own edge count with safe random non-gold edges (each candidate validated against the arm's invariants
before it is kept). ⇒ §"Open design tension" is resolved: option **(A)** works, and `hop3` **is**
constructible on the sparse base. `doc_keep_prob` is a knob, not a constant — the p=0.5 fallback is a
flag change.

The unroutable fractions are exactly the causality prediction (3.50% = dist==1, 7.58% = dist<=2 in this
400-example slice), i.e. the ladder's dilution is measured, not assumed.

### Three things the design got wrong, found by contact with code

1. **⚠ A distance-2 gold pair silently sat at 2 hops INSIDE the hop3 arm.** Deleting the direct edge is
   not enough for a pair causality leaves unroutable: at distance 2 the path `b → a+1 → a` can survive,
   so the rung leaks into the one below it. Unroutable pairs must be **cut to hop_inf**, which is the
   honest label. 7.58% of pairs. My first test missed this — at p=0.25 the short path often fails to
   exist *by luck*; the test now pins `keep_prob=1.0` so the cut must be real.
2. **⚠ The obvious hop_inf cut is wrong with multiple gold pairs** (and there are 3 per example).
   Cutting `a`'s in-edges from everything reachable from `b` deletes edges another pair's forced path
   routes through; re-forcing recreates them and the two edits **ping-pong forever**. Fix: force every
   path *first*, then cut, and cut by repeatedly removing one **unforced** edge from a surviving path.
3. **⚠ Tracking forced edges in one accumulating set inverts the protection.** Re-forcing picks fresh
   intermediaries, so the union keeps every *abandoned* attempt marked load-bearing — it grew to 21
   edges on real data and the 2-path kill started deleting the path it had just built. Forced edges are
   now tracked per pair and replaced.

All three are the same shape as this project's recurring failure: they do not crash, they produce a
plausible graph, and they quietly answer a different question.

### ~~Leak posture (unchanged, and it is why hop_inf is the control)~~ — ⚠ I WROTE THIS, AND THEN MEASURED IT FALSE

> *Original text, kept as the record of the error:* "`hop2` and `hop_inf` have **identical
> gold-edge-absent structure** and identical edge counts, and now provably identical base graphs per
> example. They differ only in whether a path exists ⇒ the leak inflates both equally and cancels in
> `hop2 − hop_inf`."

Every clause of that is true **except the conclusion**. Identical *edge-absent structure* does not imply
identical *leak*, because the two arms' signatures have wildly different base rates: "shortest==2" is
ordinary at long distance (66% of non-gold pairs), "unreachable" is not (2.2%). Measured: `hop2` 10x
vs `hop_inf` **66x**. See §"THE LEAK IS REAL" below — that section supersedes this one.

What does survive: `chunked` is a scale reference, never the control; and
`--random-doc-per-example` is **enforced at launch** (a graph fixed across examples is memorizable, and
then the missing edge is a gold beacon).

### ⚠ THE LEAK IS REAL, IT IS ASYMMETRIC, AND IT REFUTES THIS DOC'S "leak-cancelled by construction"

**Measured, no GPU, no training** (`debug/gold_hop/leak_probe.py`): a logistic regression on
**graph-only** features (direct edge, shortest path, #2-paths, in/out degrees, distance) -- **no text
at all** -- fit on 150 real eval examples and scored out-of-sample on 150 more. It picks 3 pairs per
example, exactly as the task demands. **Chance precision@3 = 3/1225 = 0.245%.**

| arm | AUC | **precision@3** | lift | with `gold_decoys=12` | lift |
|---|---|---|---|---|---|
| gold-agnostic **base graph** | 0.4982 | 0.44% | **1.8x = chance** (2 hits/450, Poisson noise on 1.1) | — | — |
| `hop1` | 0.875 | 0.89% | 3.6x | 0.89% | 3.6x |
| `hop2` | 0.850 | 2.44% | **10.0x** | 1.78% | **7.3x** |
| `hop3` | 0.955 | 6.44% | **26.3x** | 4.44% | 18.1x |
| `hop_inf` | 0.942 | **16.22%** | **66.2x** | **2.00%** | **8.2x** |

**The base graph leaks nothing** (AUC 0.4982 = chance) — the camouflage works for the *base*, exactly as
designed. **Every arm leaks, and by wildly different amounts.**

**Why — and the record's own number was the wrong null.** §"Base graph" quotes *31.2% unreachable* at
p=0.25 as reassurance that "the gold pair is unreachable" is unremarkable. That aggregate is dominated
by SHORT pairs. Measured `P(unreachable | non-gold)` by distance: **0.756** (d=1), 0.663 (2-4), 0.394
(5-12), 0.108 (13-25), **0.022** (26-49). Gold sits at mean distance ~17, and `hop_inf` makes it
unreachable with probability 1. So at distance 26-49 an unreachable pair is **45x** over-represented
among gold, and `P(gold | unreachable) = 9.7%` vs a 0.238% base rate. **An unreachable *distant* pair is
a beacon.**

⇒ **This doc's central claim — "`hop2` and `hop_inf` have identical leak, so the contrast is
leak-cancelled by construction" — is FALSE as measured.** Undecoyed, `hop_inf` is **6.6x more
identifiable** than `hop2` (66x vs 10x), and the leak *favours the control*. Consequences:

- `hop2 > hop_inf` remains **trustworthy, and is now conservative**: the leak inflates `hop_inf`, so it
  works against the hypothesis.
- `hop2 ≈ hop_inf` becomes **uninterpretable**: "routing does not work" and "routing works, but
  `hop_inf`'s leak made up the difference" predict the same result. ⚠ Given the Stage-1 preview
  (multi-hop path worth only +0.052), **the null is the expected outcome** — i.e. undecoyed, the ladder
  would most likely return exactly the unreadable case.

**The fix: `gold_decoys` (distance-matched camouflage).** Apply the arm's identical edit to `n` non-gold
pairs **at each gold pair's own distance**, so the signature names `1+n` candidates instead of naming
gold. At `gold_decoys=12` the `hop_inf` leak drops **16.2% → 2.0%** (66x → 8.2x), and — the point —
`hop2` (7.3x) and `hop_inf` (8.2x) become **leak-MATCHED**, which is what the contrast actually needs.
Structure survives intact: **150/150 gold pairs converge, decoys 100% realized, edge drift exactly 0**
for `hop1`/`hop2`/`hop_inf`.

**Absolute scale, for honesty:** even undecoyed, `hop_inf`'s graph-only precision@3 of 16.2% ⇒ a
text-blind reader scores f1 ≈ 0.16, far *below* the chunked floor of 0.408. So the leak cannot
manufacture a headline on its own; the danger is a model *combining* it with text, and the asymmetry
across arms. Exploiting it also requires computing reachability — itself multi-hop graph reasoning,
which is amusingly the very capability under test. That lowers the risk but is not a defence I would
rely on.

**Three caveats on the fix, all measured:**
1. **`hop3` is incompatible with decoys.** Shortest==3 requires killing every 2-path; 36 treated pairs
   all doing that over-constrains the graph. Gold convergence falls to 147/150 (drift +3.9) at
   `gold_decoys=12` and **112/150** (drift +15.5) at 24. ⇒ **run `hop3` with `gold_decoys=0` and its 26x
   leak declared, or drop it** (this doc already flags `hop3` as droppable).
2. **Short pairs get no decoys** (`_MIN_DECOY_DISTANCE = 5`), for two measured reasons: there is no leak
   to hide (lift 1.26 at d=1), and 12 decoys at distance 1 forbid 12 direct edges in one neighbourhood,
   which makes the **gold** pairs stop converging.
3. **The decoy pool shrinks with distance** — only `n_docs - d` pairs exist at distance `d` (just 2 at
   d=48). Long distances are simultaneously where the leak is worst and where the disguise is thinnest;
   that is why `hop_inf` still measures 6.06x lift at 26-49 after decoys rather than 1.0x.

**Recommendation: run the ladder with `gold_decoys=12`, and read `hop2` vs `hop_inf` only.**

### Eval-side plumbing (the launch blocker) — CLOSED

- **`build_gold_pairs_for_eval.py`** builds the sidecar keyed by the eval **prefill** fingerprint, by
  calling the eval's own `build_eval_prefill` (now module-level) rather than a copy — a fingerprint is a
  hash, so two renderings would silently diverge. Built for the n50 eval: **488/488 unique fingerprints,
  1464 pairs**, distance mean 16.72 / 3.96% adjacent (matching this doc's eval figures).
- **The premise is confirmed, not assumed:** the training sidecar and the eval sidecar share **0** keys
  (2000 vs 488, intersection 0). The training `gold_pairs.json` would have hit **0/488** and scored
  every arm near the ceiling.
- **Hard hit-rate assert**, demonstrated firing: pre-flight (keys, before any generation) + post-hoc
  (`require_full_hit_rate`, proving the hook actually masked). Fed the training sidecar → `SystemExit`
  at 0.0%; fed the right one → 100%, passes.
- **⚠ The hit-rate denominator counts only document-bearing rows.** KV-cached decode feeds ONE token,
  which carries no documents and cannot match by construction. Counting those would read **12/372 =
  3.2%** on a *healthy* eval and fire a false alarm.
- **`doc_keep_prob` / `random_doc_seed` / `gold_decoys` are read from the checkpoint's `config.json`**,
  never from CLI flags — they define the graph, so a mismatched flag would evaluate the model on a mask
  it never trained under, silently and with a plausible f1.

### The KV-cache decode path — checked, and there is NO hole

The worry: the answer (`[[a, b], ...]`) is generated exactly where the cross-doc evidence is needed, so
if decode reverted to plain causal the mask would be off for the tokens that matter. **It does revert to
plain causal — and that is correct, not a fallback.** Under `allowed = causal & not_pad & (context_ok |
q_free | kv_free)`, a **FREE** query attends everything causally in *every* arm, and a generated token
is FREE. So plain causal **is** the gold-hop mask for that row.

The load-bearing question is whether the CONTEXT rows the answer reads keep their edit. They live in the
KV cache, written during the masked prefill and never recomputed. **Asserted on the real cache tensor**
(`test_kv_cache_retains_the_gold_edit_from_prefill`): layer 0's cached K is identical across arms (its
K/V come straight from the embeddings — the mask cannot have acted yet), and **layer 1's differs**. The
edit reaches the cache.

### What remains before it can be trained

1. **Apply the launcher patch** (`GOLDHOP_LAUNCHER_PATCH.md`) — flags, config, the hook install, the
   realized-hops dump, and `--gold-decoys`. All anchors verified against the live file.
2. ~~**⚠ Eval plumbing does not exist and is the real gap.**~~ **DONE — see §"Eval-side plumbing".** For
   the record, the original text of this item was: A gold-aware mask needs the *eval* set's gold
   pairs at eval time, keyed by the fingerprint of the **prefill** token layout — which differs from the
   training shard's rows, so `contra_n50_v2_orig/gold_pairs.json` will NOT hit. `eval_lc_native_docchunk_contra.py`
   needs `--gold-pairs` + a hook install, and the sidecar must be built from the eval prefill layout
   itself. Until then no arm can be scored. (This is legitimate, not a leak: the mask is the
   architecture under test, and `hop_inf` is what neutralizes its train/eval leak.)
   ⚠ Note the miss-path degrades to **unmasked** = plain causal, so a fingerprint mismatch at eval
   would silently score every arm as `standard` — near the 0.943 ceiling, for every arm. Assert the
   hit rate at eval; do not trust it.
3. ~~**Stage 1 Gate 2 still decides the base.**~~ **PASSED — the base is decided.** `random_doc` p=0.25
   scored f1 **0.558 ±0.022** vs the 0.408 floor and 0.943 ceiling (eval_size 488), so it does NOT pin
   to the floor: **stay at `--doc-keep-prob 0.25`**; the p=0.5 fallback is not needed. (My measured base
   out-degree 6.126 at p=0.25 matches the 6.02 in §"Base graph".)
4. **Decide on `gold_decoys`** (recommend 12) and on `hop3` (recommend dropping it, or running it at
   `gold_decoys=0` with its 26x leak declared) — see §"THE LEAK IS REAL".

## Approach C — summary attention (design)

**Status:** design only, nothing implemented. Every number below is **measured** — from the real
`build_chunked_allowed_mask` for the existing patterns (one token per doc, so the `(S,S)` mask *is* the
doc graph), and from a prototype of the `summary_attention` predicate written in the exact form it
would take in `chunked_mask.py`. The harness reproduces this document's own `random_doc` figures to 3
decimals (p=0.25 → 6.02 / 0.246 / 0.378 / 0.312), which is the check that it measures what Stage 0
measured. Scripts: `<scratchpad>/{summary_tok,vacuity,bandwidth}.py`.

⚠ `random_doc` **p=0.5 is retained only as a measured scale reference** — that Stage-1 arm has been
**dropped**, so nothing in this design may depend on it existing.

### ⚠ Naming: "summary attention" — unrelated to the landmark code

This design has **nothing to do with `landmark_document.py` / `landmark_fast.py` /
`landmark_document_compressive.py`**. This is the only note that mentions them; "landmark" appears
nowhere else in this section, deliberately:

| | the existing landmark code | **summary attention (this design)** |
|---|---|---|
| granularity | **fixed token blocks** — `is_mem = (arange(T) % block_size) == (block_size - 1)`, driven by `mem_freq`; blind to document boundaries | **document cells** — one span per group of `k` documents |
| math | **grouped two-level softmax** — restructures the softmax so a query attends its own block fully and earlier blocks *gated by* their landmark token | **a boolean visibility mask** — plain `allowed = causal & not_pad & (context_ok \| q_free \| kv_free)`, ordinary SDPA |
| what it changes | how attention mass is *distributed* | which keys a query *may see* |

Those modules take `cross_doc_mode` as a pluggable **visibility** policy and feed
`build_chunked_allowed_mask`'s boolean output *into* their softmax — i.e. they are orthogonal
consumers of the very machinery this design extends. Approach C is purely a visibility statement, so
it belongs in `chunked_mask.py` as a new `AttentionPattern` named **`summary_attention`**, consumed by
`DocumentChunkedAttention` exactly like `random_doc`.

### ⚠ CORRECTION: the previous draft's "degree-matched hop-∞ control" was WRONG — it is VACUOUS

The previous draft claimed `summary_first` / `summary_severed` were "exactly degree-matched hop-∞
controls" with "zero density drift". **That claim is retracted. It matched edge COUNT but not edge
INFORMATION, which is the thing that matters.**

A summary span that never reads its cell carries **no document content at all** — and neither do the
earlier summary spans it may attend, for the same reason. So the relay chain is broken at the *first*
link, not the last, and every later doc attending it receives a near-constant vector (its own span
text + position + the FREE instruction prefix). A model will simply learn to ignore keys that carry
nothing, so the density match is **cosmetic**.

**Measured** (transitive closure: how many DOC chunks' content can reach each summary span, k=10, n=50):

| arm | docs whose content reaches each summary span | vacuous? | mean doc-content reaching a *doc* |
|---|---|---|---|
| `summary_attention` (relay) | 10, 20, 30, 40, 50 | **no** | **24.50 of 24.5 — ALL earlier docs** |
| `severed` (relay off) | 0, 0, 0, 0, 0 | **YES** | 4.50 |
| `summary_first` (span at cell start) | 0, 0, 0, 0, 0 | **YES** | 4.50 |

The 4.50 is *exactly* the own-cell count — i.e. **both "controls" collapse to `cell_blocks` plus
information-free decoration.** The user's objection is correct and is confirmed above.

**Consequence, stated plainly: Approach C is NOT cleaner than Approach A in the way the previous draft
argued.** A's `hop2` vs `hop∞` contrast has genuinely matched information content — both arms carry
real doc→doc content, and only the gold path differs. C's would-be contrast reduces to
**summary-relay vs `cell_blocks`**, which is a floor comparison, not a matched control. C's real
advantage is narrower and different: **it is gold-blind, and it has a bandwidth knob A cannot have.**

**And there is no non-vacuous hop-∞ control to find — this is a theorem, not a failure of imagination.**
Any relay visible to a later doc `b` is, by causality, positioned before `b`; so if it carries *real*
document content, that content necessarily came from documents before it — i.e. it creates a real path.
The only way to have no path is to carry nothing. **Measured** on the obvious escape attempt (let the
summary of cell `c` relay a *different* cell via a permutation, so its content is real but "wrong"):
cross-cell doc pairs reachable = **100 / 600 / 400 / 300** across four random permutations — every one
creates real paths. ⇒ **`cell_blocks` (= bandwidth 0) is the control, and the LADDER is the evidence.**

### The pivot: the BANDWIDTH DOSE-RESPONSE is the headline

Dedicated summary spans give Approach C the one thing A cannot do at all: **vary the relay capacity
while holding literally everything else fixed.** A 0-token relay *is* `cell_blocks`. Then 1 / 4 / 16
tokens per span vary **only the bottleneck width** — same data, same positions, same doc-level graph,
same everything. If f1 climbs monotonically with bandwidth, information is provably flowing through the
relay, because nothing else changed. That is a dose-response curve, far more robust than any single
contrast, and it is the natural home for the knob originally asked for.

Dedicated spans also fix the previous draft's dilution problem: **a gold doc is never a relay**, so
every cross-cell gold pair sits at exactly 2 hops (0.816 of pairs at k=10 vs the retired design's
0.754/0.539).

### Layout, roles, and the chunk-id convention

Per cell `c`: `k` documents, then a **summary span** (measured: 42-52 Qwen3 tokens), itself wrapped in
`<|box_start|>…<|box_end|>`:

```
[doc][doc]...[doc] [SUMMARY span]   [doc][doc]...[doc] [SUMMARY span]  ...
\______ cell 0 (k docs) ______/     \______ cell 1 ______________/
```

**No new chunk-id role is needed, and this is the load-bearing simplification.** Because
`build_chunk_ids_from_tokens` assigns ids by *counting `<|box_start|>` occurrences*, a box-wrapped
summary span automatically becomes its own context chunk with the next sequential id. So with stride
`P = k + 1`:

```
cell(id)       = id // P
is_summary(id) = (id % P) == k
```

— pure arithmetic on `chunk_id`, exactly like the current patterns. The `>= 0` doc id / `-1` FREE /
`-2` PAD / `-3` SINK convention is **untouched**: a summary span is an ordinary `>= 0` context chunk.
No new role, no second tensor, no sidecar, `mask_mod`-expressible, FlexAttention- and compile-safe.

⚠ **`k` must divide `n`** so there is no partial final cell to break the `% P` arithmetic. At n=50 that
means **k ∈ {5, 10, 25}**. (The retired draft leaned on a `max_chunk` clamp for partial cells; with a
regular stride that clamp no longer rescues the indexing, so this is a hard constraint.)

**What may attend what:**

- a **doc** attends: its own cell's earlier docs, + the *visible prefix* of the summary spans of
  **strictly earlier** cells (see the bandwidth gate below).
- a **summary span** attends: its own cell's docs (this *is* the relay), + earlier summary spans.
- FREE tokens bridge everything, as always.

**One shard, mask-only arms — the bandwidth is a MASK knob, not a data knob.** The shard always carries
the full summary span per cell (42-52 tokens, measured); the mask throttles how many of each span's tokens a later doc may
attend (`bandwidth = b ≤ B_max`). This preserves the document's own founding principle ("arms differ
only in the mask"): identical tokens, identical positions, identical dataset at every rung. `b = 0`
removes the relay entirely and is *exactly* `cell_blocks`.

### The exact predicate

```python
P = k + 1
both_ctx = (qc >= 0) & (kc >= 0)
cell_q, cell_k   = qc // P, kc // P
is_sum_q, is_sum_k = (qc % P) == k, (kc % P) == k
same_cell = (cell_q == cell_k) & both_ctx

doc_reads_own_cell = same_cell & ~is_sum_q & ~is_sum_k
sum_reads_own_cell = same_cell & is_sum_q & ~is_sum_k & RELAY   # RELAY=False -> the placebo
visible_sum        = is_sum_k & both_ctx & (offset_in_chunk_k < bandwidth)   # <-- the bandwidth gate
reads_earlier_sum  = (cell_k < cell_q) & visible_sum

context_ok = same_chunk | ((doc_reads_own_cell | sum_reads_own_cell | reads_earlier_sum) & (qc >= kc))
```

`offset_in_chunk` is a `(B, S)` tensor precomputed **once per forward** from `chunk_ids` and closed
over by the `mask_mod` — exactly the pattern `random_doc` already uses for `random_doc_nonce`
(`chunked_mask.py:623`), so the mask_mod body stays a pure elementwise function of `(b, q_idx, kv_idx)`
and remains Triton-friendly.

⚠ **`reads_earlier_sum` must test `cell_k < cell_q`, not merely causality** — otherwise a doc could read
its *own* cell's summary, which (a) is a same-cell shortcut that carries the whole cell and (b) makes
the predicate silently placement-dependent.

⚠ **Every token of a span sits after every doc of its cell** (the span is at the cell end), so **every**
visible token has already read the full cell and is a valid relay — the bandwidth gate never exposes a
token that read nothing. **Verified**: `every summary token follows all its cell's docs → True`.

### Causality

Unchanged and still governing: **a relay can only carry information forward, so it must have already
read what it relays.** For `a < b` in different cells, the span `S_{cell(a)}` sits after every doc of
cell `a` and before cell `b` begins, so `a < S_{cell(a)} < b` — the relay always exists and is always
positionally available. Path `b → S_{cell(a)} → a` = **2 hops**. This is why the span goes at the cell
END; at the cell start it reads nothing and the design is vacuous (§CORRECTION above).

### Measured connectivity at n=50

Doc→doc view (the Stage-0-comparable metric). `direct = P(same cell) = (k−1)/49`;
`doc→doc out-deg = (k−1)/2`; cross-cell pairs are **all** at exactly 2 hops. All rows measured.

| mask | doc→doc out-deg | direct | 2-hop | unreach | max hop |
|---|---|---|---|---|---|
| `full` (`standard`) | 24.5 | 1.000 | 0.000 | 0.000 | 1 |
| `random_doc` p=0.5 *(scale ref only — arm dropped)* | 12.02 | 0.491 | 0.426 | 0.064 | 4 |
| `summary_attention` k=25 | 12.0 | 0.490 | 0.510 | 0.000 | 2 |
| `random_doc` p=0.25 | 6.02 | 0.246 | 0.378 | **0.312** | 4 |
| **`summary_attention` k=10, b>0** | **4.5** | 0.184 | **0.816** | **0.000** | **2** |
| **`summary_attention` k=10, b=0** ≡ `cell_blocks` | **4.5** | 0.184 | 0.000 | **0.816** | 1 |
| `summary_attention` k=5, b>0 | 2.0 | 0.082 | **0.918** | 0.000 | 2 |
| `chunked` | 0.0 | 0.000 | 0.000 | 1.000 | — |

⚠ **The `(k−1)/49` same-cell figure assumes uniform gold placement.** The eval's *measured* gold
distances (mean 16.7 vs uniform's 17.0 at n=50) make this a good approximation, but **compute it from
the real distance file before committing** — §"The causality constraint" above notes 12.5% of eval
pairs sit at distance < 4, and those are exactly the ones at risk of landing same-cell.

**The claim that makes this an architecture and not an ablation** — measured: under
`summary_attention`, the mean number of earlier docs whose content can reach a given doc is
**24.50 out of 24.5, i.e. ALL of them**, at a doc→doc out-degree of only **4.5**. Every document is
informationally connected to every earlier document through a 2-hop relay, on an `O(N√N)`-style edge
budget. That is the whole point of the design.

### The bandwidth ladder — THE HEADLINE

`b` is **invisible to the doc-level graph** and controls only how many key *positions* carry a cell
forward. Measured on realistic `chunk_ids` (5 cells × (10 docs + a 16-token span), variable doc
lengths, FREE prefix/tail, PAD tail):

| bandwidth `b` | cross-cell key tokens per doc | doc-graph 2-hop | unreach | ≡ `cell_blocks`? |
|---|---|---|---|---|
| **0** | 0.00 | 0.000 | 0.816 | **True (bit-exact)** |
| **1** | 2.05 | 0.816 | 0.000 | False |
| **4** | 8.20 | 0.816 | 0.000 | False |
| **16** | 32.82 | 0.816 | 0.000 | False |
| **42** | 86.15 | 0.816 | 0.000 | False |

**The doc-level graph is identical for every `b > 0`; only the bottleneck width changes.** And `b=0`
reproduces `cell_blocks` *bit-exactly* — so the ladder's own zero rung is its floor control, with no
separate arm and no data difference. Invariants hold at every rung: FREE rows exactly
causal-over-non-pad, PAD never attended, no fully-masked query row.

**Sizing the rungs (information-theoretic, and it sets `k=10`).** A cell of `k=10` PubMed claims is
~460 tokens (~46 tok/claim). The relay is `b` positions × `d=1024` dims:

| `b` | claims per summary vector | verdict |
|---|---|---|
| 1 | 10 : 1 | ~10× tighter than sentence-embedding density — **plausibly too tight** |
| 4 | 2.5 : 1 | borderline |
| ~10 | **1 : 1** | ≈ sentence-embedding density — **the natural scale** |
| 16 | 0.6 : 1 (1.6 vectors/claim) | above sentence-embedding density — comfortably generous |
| **42** | **0.24 : 1 (4.2 vectors/claim)** | **the entire span — the most generous relay the layout can offer** |

The interesting range brackets **`b ≈ k`**, which is why `k=10` with rungs
**`b ∈ {0, 1, 4, 16, 42}`** is the ladder: it straddles the 1-vector-per-claim scale on both sides and
then tops out at the full span. (Sentence-embedding models routinely compress ~1 sentence into 1 vector
*retrievably*; 10 sentences into 1 vector is well beyond that, so a low `b=1` reading is expected and
informative rather than a failure.)

⚠ **`b = 42` is the MAXIMUM SAFE RUNG, and the bound is not cosmetic.** The realized spans are
**42–52 tokens** and their length **varies per cell** — cell 0 summarizes single-digit claims (42
tokens), later cells double-digit (52). The gate exposes each span's *leading `b`* tokens, so a rung is
uniform only while `b ≤ the SHORTEST span`. At `b = 52` cell 0 would silently contribute **42** keys
while every other cell contributed 52 — a per-cell-varying dose on the axis whose entire purpose is to
be the only thing varying. It would not error; it would just quietly tilt the curve toward the later
cells. Hence **42**, asserted by `test_realized_span_length_supports_the_ladder_rungs` rather than
trusted. (Raising the ceiling means padding every span to a common length, which is not worth it: 42 is
already 4.2 vectors/claim.)

**Why the top rung earns its GPU time:** a flat curve *through the full span* is a far stronger null
than one that stops at `b=16`. At `b=42` the relay has 86 cross-cell key positions per document
(measured) carrying 40 documents, and every document's content is reachable in 2 hops. If f1 still sits
on the `b=0` floor there, "the bottleneck was too narrow" is no longer available as an explanation —
which is precisely the ambiguity §"the dead-zone risk" warns the ladder cannot otherwise resolve.

**Predicted curves — say which is which before running:**

| outcome | shape | reading |
|---|---|---|
| **routing works** | monotone rising from the `b=0` floor, knee near `b ≈ k = 10`, saturating by `b=42` | **information flows through the relay — nothing else changed.** The dose-response *is* the result |
| **routing fails** | flat at the `b=0` floor for every `b`, **including `b=42`** | the relay does not carry, and the FREE bridge is all there is. Credible *because* it survives the full-span rung |
| **mechanical artifact** | jump at `0 → 1` then flat, or non-monotone | a summary key acting as an attention sink / bias, not a relay — **the placebo catches this** |

This is strictly more robust than any single contrast: it needs no matched control arm, because the
*only* thing varying across rungs is the number of relay positions.

### The placebo (one cheap arm, sharp falsifiable prediction)

**`summary_placebo` = `RELAY=False` at the TOP rung `b=42`** — same shard, mask-only, span still at the
cell end, still fully visible, but **not allowed to read its cell**. Verified vacuous above (0 docs
reach it). Run it at the top rung deliberately: that is where the most information-free keys are on
offer (86 cross-cell key positions per document, measured), so any mechanical key-count effect is at its
largest and easiest to detect.

> **Prediction: it lands exactly on `b=0`.** If it does, extra keys *per se* do nothing and the
> bandwidth curve is clean. If it does **not** (helps or hurts), adding information-free keys has a
> mechanical effect — softmax-denominator dilution, an attention sink, a learned bias — that would
> otherwise contaminate the ladder, and every rung must then be read **against the placebo** rather
> than against `b=0`.

⚠ Frame it as exactly this and nothing more: **a placebo that validates the ladder — NOT a
capacity-matched hop-∞ control, and NOT the headline.** `summary_first` is the data-side equivalent and
is strictly worse (needs a second shard, shifts every doc by `B_max` positions) for an identical,
equally vacuous result — **do not build it**.

### The one non-vacuous ∞ contrast that DOES exist: coverage, *within-arm*

No *global* hop-∞ control exists (proved + measured above). But a **within-arm** one does, and it is
clean: let the summary read only the **first `k'` docs of its cell** (`coverage`). The summary still
carries **real document content** (non-vacuous at every `k' > 0`), but only the covered docs are
routable. **Measured** (k=10):

| coverage `k'` | docs seen per span | cross-cell gold pairs with a path | vacuous? |
|---|---|---|---|
| 0 | 0.0 | 0.000 | **yes** |
| 1 | 3.0 | 0.100 | no |
| 3 | 9.0 | 0.300 | no |
| 5 | 15.0 | 0.500 | no |
| 10 (full) | 30.0 | 1.000 | no |

Reachable fraction is exactly `k'/k`. At `k'=5`, **half** the cross-cell gold pairs have a 2-hop path
and half do not — **under one identical mask, with identical summary informativeness and identical
density.** Stratifying that arm's per-example f1 by "is this pair's `a` in the covered slots" is a
genuine `path` vs `no-path` contrast with **zero cross-arm confound**, and it is gold-blind (coverage is
an index rule). This is the closest thing to a real hop-∞ control the design admits. Proposed as a
**secondary axis**, not the headline. ⚠ Covered-ness correlates with within-cell slot, so it is only
as-good-as-random because placement is random — stratify, do not assume.

### ⚠ The dead-zone risk, restated — unchanged as the top risk, and now murkier

The measured `hier-K10` finding stands and still frames everything:

| | per-layer out-deg | union out-deg | unreach | max hop | outcome |
|---|---|---|---|---|---|
| `hier` K=10 @ n=100 | **6.55** | 18.89 | **0.000** | 3 | **died at the floor** |
| `hier` K=50 @ n=100 | 19.98 | 43.25 | 0.000 | 2 | f1 0.831 |
| `random_doc` p=0.5 @ n=100 | 25.33 | 25.33 | 0.028 | 4 | f1 0.735 |

**hier-K10 had 100% reachability and max 3 hops and still died** — so "0 unreachable, max 2 hops" is
*demonstrably not sufficient*. Learners had per-layer out-degree ≥ ~20; the one death was at 6.55; and
nothing was measured in between.

⚠ **The summary design lands at doc→doc out-degree 4.5 (k=10) or 2.0 (k=5) — BELOW the known-dead
6.55.** But the metric no longer transfers cleanly: under `hier-K10` every edge was doc→doc, whereas
here the cross-cell channel *is not a doc→doc edge at all* — each doc has bottlenecked access to **all**
24.5 earlier docs (measured) via `b × (n_cells−1)` summary positions. So the honest statement is:

- by the **only metric with evidence attached** (doc→doc out-degree), this design sits *below* the one
  configuration known to die;
- by the metric that arguably matters (cross-cell key positions / information reachability), it is at
  **100% reachability** and `b` moves the bottleneck freely;
- **which metric predicts the floor is unknown**, and this design does not resolve it.

**But the ladder degrades gracefully where a single arm would not.** If the whole family is dead, every
rung pins to the `b=0` floor and the result is "void, too sparse" — the same verdict a single arm would
give, reached the same way. If it is *not* dead, the ladder returns a curve rather than a point. So the
dead-zone risk costs us the experiment but never *misleads* us — which is exactly what the hier-K10
episode did (it was read as a mask result for months). ⚠ The one thing the ladder cannot do is
distinguish "dead family" from "genuine null" — both are flat. That is the residual risk, and the
`b=16` rung is the mitigation: at 1.6 vectors/claim it is the most generous relay the design can offer,
so a flat curve *through* `b=16` is much stronger evidence of a real null than a single sparse arm would
be.

### ⚠ Special tokens vs a natural-language phrase — the Qwen3 marker bug

New reserved special tokens walk straight into the bug that has already burned this project twice
(`records/document-chunked-marker-embeddings.md`, `records/n100-chunked-marker-position-bug.md`):
Qwen3 never trains its reserved rows (bit-identical, cos 1.0000), and the *first* repair fixed the
cosine but left the norm at 0.396 vs a trained median of 1.415 — which RMSNorm amplifies into
full-strength noise and flatlines training at **CE ≈ 0.79 for every mask, including plain causal**.
That failure looks exactly like "my mask is too restrictive" when the mask is fine.

**RECOMMENDATION: (a), a natural-language phrase built from real, already-trained tokens.** The
decisive evidence is *already in the record* and is direct, not analogical:

1. **`records/document-chunked-marker-embeddings.md` states that swapping the markers for ordinary
   tokens (`«` / `»`) made training converge normally.** Real trained tokens used as structural
   delimiters are a *proven-good* configuration in this exact setting. Option (a) needs no base repair,
   no norm assert, no `-fixmark` discipline, and adds **zero** new embedding-bug surface.
2. **The repetition risk does not apply, and I checked rather than assumed.** The `free60` collapse was
   **60 copies of one 8-token sentence in a single contiguous block** sitting between the claims and
   the answer (`records/free-pad-probe-is-confounded.md` — and 60 *distinct* sentences at a *larger*
   budget trained fine, so it is repetition, not budget). Meanwhile the box markers repeat **100×/example
   at n=100 and are harmless**. So repetition *count* is not the trigger — the markers out-repeat
   `free60` and survive. The distinguishing features are **contiguity** and **span length**, and the
   summary spans are the opposite of `free60`: `n_cells` short spans (5–10), **distributed** across the
   context, one per cell.
3. **Best of all, the phrase can be made varied AND index-bearing for free**, which turns the risk into
   a known *benefit*: render cell `c`'s span as **`Summary of claims 17-24:`**. Every span is then
   textually distinct (different indices), which is exactly the `free60v` fix that resolved the
   repetition confound — and `masks-n100.md` §3 found that in-chunk filler *restating the claim index*
   was **the only intervention that ever helped** chunked (CE 0.224 → 0.198). Index-bearing text is a
   known-good lever here, not merely a safe one.

Option (b) (new reserved tokens) would mandate repairing the base with trained donor rows + the
in-distribution norm assert (`src/scripts/data/fix_marker_embeddings.py`) and pinning every run to the
repaired base — all of that risk, for no benefit this design needs.

**The mask-gated bandwidth removes the repetition question almost entirely, and this is a real win of
the pivot.** There is **one** span text, emitted once per cell; the ladder never
changes the tokens, only how many of them the mask exposes. So:

- **no verbatim repetition across cells** — each span carries its own claim indices:
  `Summary of claims 11 to 20: [11] [12] [13] [14] [15] [16] [17] [18] [19] [20]`. Every span is
  textually distinct, which *is* the `free60v` fix.

  ⚠ **CORRECTED ON CONTACT WITH THE CODE (implementation, 2026-07-16).** This span was *estimated* at
  "≈16 Qwen3 tokens"; **measured** with the real Qwen3 tokenizer it is **42–52 tokens**, and the length
  **varies per cell** (cell 0's claims are single-digit: 42 tokens; later cells 52). Consequences, all
  benign but worth stating rather than discovering later:
  - `B_max` is **not** 16 — which is *why* the ladder tops out at **`b = 42`**, the measured shortest
    span (see §"The bandwidth ladder"). Every rung must be ≤ the **shortest** span, which is the
    condition for the gate to expose the same number of keys for every cell; a rung > 42 would silently
    be non-uniform. `test_realized_span_length_supports_the_ladder_rungs` asserts this rather than
    trusting it.
  - Summary tokens added per example is **~250, not 80** (5 cells × ~50) — ~10% of an n=50 context, and
    not enough to change the seq_len bucket.
  - Tokens beyond the top rung are never visible to a later doc, but they still read the cell and each
    other. That is harmless (they are causally *after* the visible leading tokens, so they cannot affect
    them) and arguably useful: it decouples the relay's **compute** capacity from its **channel** width.
- **no content/length confound in the ladder** — the earlier draft varied span *length* per rung, so
  content covaried with bandwidth. That confound is now **gone**: `b=1` and `b=16` are the same
  sequence of tokens in the same positions.
- **5 short distributed spans**, one per cell — the opposite of `free60`'s single contiguous block of
  60 identical sentences.

⚠ Assert the realized `B_max` from the tokenized shard (tokenizer-dependent) rather than assuming 16,
and pick the phrase so it tokenizes to a round number. ⚠ At low `b` the visible prefix is the span's
*first* tokens (`Summary of…`) rather than the index list — see open question 4 on whether span token
order should be engineered.

⚠ **A mandatory pre-flight gate, cheap and pre-registered.** Both prior marker bugs had the same tell:
*if unrestricted attention also fails, the cause is upstream of the mask.* So **every summary-token
shard must first pass a plain-causal (`standard`) training run reaching CE ≲ 0.01** before any masked
arm is trusted. That single run would have caught both historical bugs immediately, and it costs one
short job.

### ⚠ Eval-side rendering — cheaper than feared, but there is exactly one rule

This is a **data** change, so the eval must render the identical layout or the arm cannot be scored at
all — the gap that reduced the free-pad probe to train-CE-only, with no held-out f1. **Good news, and it
is decisive: train and eval already share one implementation.**
`src/olmo_core/data/document_chunk_landmark.py::segment_prompt_to_chunks` is documented as *"the single
source of truth for both training (`include_answer=True`) and eval prefill (`include_answer=False`) so
their token layouts match exactly"*, and `eval_lc_native_docchunk_contra.py` calls that same function
plus the same `emit_document_chunk_dense` emitter.

**Why the free-pad probe still broke:** it was built in a **`debug/build_free_varied.py` one-off that
bypassed the shared path**, so `segment_prompt_to_chunks` had no idea how to reproduce it. The lesson is
not "layout changes are expensive" — it is **"implement the layout inside the shared path, never in a
`debug/` builder."**

Concretely, the change list is short and touches one code path:

1. **`_wrap_documents`** (`document_chunk_landmark.py:205`) — emit an extra box-wrapped summary chunk
   after every `k`-th document. ⚠ It currently builds **contiguous** chunk spans (each doc's chunk
   starts where the previous ended), so a naively inserted phrase would be *absorbed into the next
   document's chunk* instead of becoming its own. The insertion must deliberately break contiguity.
   ⚠ It also locates each doc body by `text.find(body, cursor)` and silently leaves unmatched docs FREE
   — so insert during the emit pass, **after** the spans are located, never before.
2. **`segment_prompt_to_chunks`** — thread `summary_every_k` (and the span text template) through. It
   already threads `free_pad_repeat` / `repeat_doc_text` exactly this way.
3. **`convert_unified_to_document_landmark.py`** — expose `--summary-every-k`.
4. **`eval_lc_native_docchunk_contra.py`** — add **one** CLI arg, `--summary-every-k`, copying the
   existing `--free-pad-repeat` docstring verbatim: *"MUST match the training shard, or the prefill
   layout differs from training."*
5. **chunk_ids: no change at all.** `build_chunk_ids_from_tokens` counts `<|box_start|>` occurrences, so
   the summary chunk gets its sequential id automatically — at train *and* at eval.

**`bandwidth` and `RELAY` need NO eval plumbing at all** — they are *mask* knobs living in the attention
config, restored from the checkpoint's `config.json` like `cross_doc_mode` already is. Only the **layout**
(`summary_every_k` + the span text) is a data property that must match, and **there is exactly one shard
for every rung of the ladder**, so there is exactly one layout to match and one pre-flight run total.

**Cost verdict, stated plainly since it is decision-relevant:** this is **more expensive than the
retired re-designated-doc version, which was mask-only (zero data work, zero eval work)** — but the
mask-gated bandwidth pulled the cost down a lot from where this revision started: **one** shard (not
three), **one** data flag to match (not two), **one** plain-causal pre-flight (not per-rung), and the
whole ladder + placebo are mask-only arms off that single shard. The irreducible cost is: a data
rebuild, a 3-file parameter thread, one pre-flight run, and a parity test. ⚠ There is **no existing
train/eval layout-parity test**; add one asserting the eval prefill is token-identical to the training
shard's prompt prefix for the same example, or `--summary-every-k` will silently drift exactly as
`--free-pad-repeat` warns it can.

### Proposed arms — ONE shard, every arm mask-only

`k=10`, one span per cell (42-52 tokens, measured). All rungs share one dataset; arms differ **only** in
`bandwidth` / `RELAY` -- both attention-config knobs, so no rung needs its own shard.

| arm | `b` | doc→doc out-deg | cross-cell gold path | role |
|---|---|---|---|---|
| `standard` (plain causal) | — | 24.5 | 1 hop | **ceiling + the MANDATORY pre-flight gate (§below)** |
| **`summary_attention` b=0** ≡ `cell_blocks` | 0 | 4.5 | **none** | **floor = the ladder's zero rung (bit-exact)** |
| **`summary_attention` b=1** | 1 | 4.5 | 2 hops | **ladder** (10:1 compression) |
| **`summary_attention` b=4** | 4 | 4.5 | 2 hops | **ladder** (2.5:1) |
| **`summary_attention` b=16** | 16 | 4.5 | 2 hops | **ladder** (0.6:1) |
| **`summary_attention` b=42** | 42 | 4.5 | 2 hops | **ladder — full span, the most generous relay possible** |
| `summary_placebo` (RELAY off) | **42** | 4.5 | **none** (vacuous) | **placebo — predicted == b=0**; run at the TOP rung, where any mechanical key-count effect is largest |
| `chunked` | — | 0.0 | none | bridge to prior n=50/n=100 numbers |
| `random_doc` p=0.25 | — | 6.02 | mixed (31% none) | measured scale reference (**already a Stage 1 arm**) |

All seven arms come off **one shard** (`--summary-every-k 10`); `bandwidth` / `RELAY` are
attention-config knobs, so no rung needs its own data and eval restores them from `config.json`.
Launcher flags: `src/scripts/train/memexpress/goldhop/SUMMARY_LAUNCHER_PATCH.md`.

**Read:** **monotone f1 in `b`** ⇒ information flows through the relay — the headline, and it needs no
matched control because nothing but the bottleneck width changed. **Flat at the `b=0` floor for every
`b`** ⇒ either a genuine null or a dead family (⚠ the ladder cannot separate these; the `b=16` rung is
what makes a null credible). **Placebo ≠ `b=0`** ⇒ a mechanical key-count artifact; re-read every rung
against the placebo.

⚠ `chunked` here is **not** identical to the established `chunked` floor — this shard carries 80 extra
summary tokens. Use it as a bridge to prior numbers, not as a strict comparison.

**Leak-freedom is C's central advantage and is unchanged:** the mask is a deterministic function of
`(qc, kc)` alone and never sees gold, so given positions its mutual information with gold identity is
**exactly zero** — no edge is deleted, so there is no hole to read. Dedicated spans strictly *improve*
this over the retired draft: **a gold doc can never be a relay**, so every cross-cell gold pair sits at
exactly 2 hops (0.816 of pairs). ⚠ Do **not** constrain gold placement to force cross-cell — that would
make *placement* gold-aware ("the partner is never within ±10"), re-introducing the ~1.14×
candidate-set leak C exists to avoid. Keep placement gold-blind and stratify post-hoc
(`debug/gold_connectivity/stratify.py` + `--per-example-out`).

### ⚠ Two gates that must pass BEFORE any rung is believed

**1. The plain-causal pre-flight on the summary shard — MANDATORY, and it is the first job to run.**
The summary layout adds ~250 new tokens/example. Both historical marker bugs
(`records/document-chunked-marker-embeddings.md`, `records/n100-chunked-marker-position-bug.md`) shared
one tell: **if unrestricted attention also fails, the cause is upstream of the mask.** Both flatlined
training at CE ≈ 0.79 for *every* mask including plain causal, and both read as "the mask is too
restrictive" when the mask was fine.

> **Train `--cross-doc-mode standard` on the summary shard and require CE ≲ 0.01 before trusting any
> masked arm.** One short job. It is a strict superset of the ladder's failure mode: a summary-token
> shard that plain causal cannot fit would make **every rung flat** — i.e. it would forge the exact
> signature of "the relay does not carry", which is the ladder's headline null.

The design's token choice makes this likely to pass (real trained tokens, no reserved rows, no
checkpoint repair — §"Special tokens vs a natural-language phrase"), but "likely" is what both prior
bugs also looked like. Run it anyway.

**2. The `p_standard` anneal must actually reach 0 — this ladder is unusually exposed to it.**
⚠ **Load-bearing, and it has already silently voided three Stage-1 arms.** `mix_total_forwards` must be
divided by **both** `world_size` **and** `micro_batch_instances` (a forward now carries
`micro_batch_instances` instances, so raising it cuts the forward count by that factor). Un-fixed, at
world=2/micro=4 the curriculum ended at **`p_standard = 0.601`** — ~60% of forwards still **plain
causal** at the end of training. The tell was perverse: the loss looked *healthier*, not worse, because
it was training unmasked.

For this experiment specifically that failure is worse than "a bit of leakage": plain causal hands every
document a **direct cross-cell edge**, which bypasses the bandwidth gate entirely and makes **every rung
look the same**. A flat ladder is precisely the "genuine null" signature — so this bug would not present
as a bug, it would present as *the result*. The hard pre-flight assert that re-derives forwards-per-rank
and refuses to start must be live for **all seven arms**.

⚠ MAXLEN ≥ 8192 at eval; dump generations if any trained arm evals at exactly 0. Everything else in
§"Training config" applies unchanged.

### Open questions for refinement

1. **Is the added cost worth it, given what C no longer claims?** Summary attention costs a data
   rebuild + eval plumbing + a parity test + a plain-causal pre-flight per shard. What it buys is the
   **bandwidth dose-response** and gold-blindness — *not* a cleaner matched control than A (that claim is
   retracted). If the bandwidth curve is not the deliverable we want, C's remaining edge over A is thin.
   Worth deciding explicitly before any data work.
2. **`k=5` vs `k=10`.** k=5 gives a bigger 2-hop stratum (0.918 vs 0.816) and more summary positions per
   doc, but drops doc→doc out-degree to **2.0** — deeper below the known-dead 6.55. k=25 is too coarse
   (49% same-cell). k=10 is recommended mainly because it puts `b ≈ k` inside the ladder; is that the
   right thing to optimize?
3. **Does `b` actually behave as bandwidth?** `b` positions × `d=1024` dims is the *nominal* channel, but
   the model may not use it as one. f1 flat in `b` *while above the floor* would mean the bottleneck is
   elsewhere — arguably the most interesting outcome, and one this arm list can detect but not explain.
4. **Content/length confound in the ladder.** With the mask-gated design the *tokens are identical* at
   every rung — only visibility changes — so the classic confound is gone. But the visible prefix at
   `b=1` is the span's first token (`Summary`) rather than a purpose-built single summary vector. Should
   the span be engineered so its first token is the most informative one (e.g. put the index list
   first), or is prefix-order irrelevant since all tokens have read the cell?
5. **Coverage as a second axis.** The `k'` knob (§above) gives the only non-vacuous path/no-path contrast
   available, *within-arm* and confound-free. Run it as a second ladder, or hold it in reserve?
6. **Per-pair vs per-example grading.** Each example has 3 gold pairs that can land in different strata;
   example-level f1 mixes them. Stage 0 Part B binned by "how many of this example's pairs are
   connected" (0/1/2/3) — enough resolution here, or should the eval grade per-pair?
7. **Should C run at all if A returns a clean `hop2 > hop∞`?** C would then be an
   architecture-viability + bandwidth claim rather than a mechanism one. Different paper section,
   possibly still worth it.
