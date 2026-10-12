# n=100 document-chunked: clean results after fixing the marker-embedding bug

**Date:** 2026-07-14. **Model:** Qwen3-0.6B. **Task:** contradiction, n=100 claims, no-CoT.
**Data:** leak-free shards (`contra_n100_v2_*` — the `Claim N:` label is *inside* the chunk, so no FREE
label bridges chunks). **Base:** `q06b-dense-cpt-modelonly-trainedmark` (markers seeded from trained
delimiter rows — see `n100-chunked-marker-position-bug.md`).
**Eval:** held-out `contradiction_eval_pubmed_both_n100_k3`, **eval_size = 488**, parse_rate 1.00
everywhere. Binomial SE quoted; run-to-run seed noise adds more on top.

## 1. Chunked vs hierarchical at 100 documents — the headline

Every one of these previously scored **f1 ≈ 0.001** (flat CE 0.79). They were measuring the marker bug.

| mask | cross-doc connectivity | train CE | held-out f1 | ±SE |
|---|---|---|---|---|
| `standard` (plain causal) | full | 0.0008 | **0.934** | ±0.011 |
| `hierarchical_dilated` K=50, cyc5 | strided, wide | 0.040 | **0.831** | ±0.017 |
| `random_doc` k=0.5 | random 50% | 0.070 | 0.735 | ±0.020 |
| `hierarchical_dilated` K=10, cyc5 | strided, narrow | 0.165 | 0.505 | ±0.023 |
| `chunked` (pure) | **none** | 0.224 | 0.441 | ±0.022 |

**Conclusions:**
1. **"At n=100 only dense attention works" is false.** Pure chunked — documents fully isolated, all
   cross-document work done at the trailing FREE positions — reaches **f1 0.441** on a 100-way task.
2. **Hierarchical beats chunked decisively at n=100: 0.831 vs 0.441** (a ~18 SE gap). This is the
   comparison that previously could not be made, because both arms were pinned at chance.
3. f1 is monotone in cross-document connectivity (chunked < hier-K10 < rand-0.5 < hier-K50 < full), and
   a wide-stride hierarchical mask recovers most of the gap to full attention (0.831 vs 0.934).

⚠ The **old n=100 numbers are not comparable** to these: they were run on the *leaky* v1 shards, where
the FREE `Claim N:` labels bridged chunks (chunked mask density 0.344 vs 0.099 leak-free). "Chunked"
there was never isolated.

## 2. Analysis experiments: more FREE tokens vs more within-chunk tokens

The premise was that chunked collapses at n=100 because all cross-document comparison is funnelled
through the trailing FREE positions, so *widening the FREE budget* should help.

**First, the existing knobs are broken.** `free_pad_repeat` appends N copies of *one identical
sentence*, and `repeat_doc_text` duplicates each claim *verbatim*. Both collapse training **even under
plain causal**, where the FREE role is irrelevant — so they measure repeated-text damage, not capacity.
Full diagnosis in `free-pad-probe-is-confounded.md`.

| filler | added tokens | chunked CE | plain-causal CE |
|---|---|---|---|
| none (`base`) | 0 | 0.224 | 0.0008 |
| `free60` — 60× the **same** sentence | +481 FREE | 0.80 ❌ | **0.81** ❌ |
| `rep2` — each claim duplicated **verbatim** | +~4600 in-chunk | 0.795 ❌ | — |
| `free60v` — 60 **distinct** sentences | +813 FREE | 0.428 ✅ | **0.0003** ✅ |

**Rebuilt with varied, budget-matched, content-neutral filler** (differing only in the *role* the added
tokens carry):

| arm | in-chunk tokens | FREE tokens | chunked train CE |
|---|---|---|---|
| `base` | 4598 | 217 | **0.224** |
| `free60v` — more FREE | 4598 | 1030 (+813) | **0.428** — *much worse* |
| `chunkpad2` — more in-chunk, index-free | 5918 (+1320) | 217 | 0.262 — mild cost |
| `chunkpad` — more in-chunk, filler **restates the claim index** | 5710 (+1112) | 217 | **0.198** — *helps* |

**Conclusions:**
1. **Widening the FREE budget does NOT rescue chunked at n=100 — it makes it substantially worse**
   (0.224 → 0.428), while the *same* filler under plain causal is harmless (0.0008 → 0.0003). The
   chunked collapse is **not** a FREE-position *capacity* limit. That hypothesis is refuted.
2. Adding the same budget of neutral tokens *inside* the chunks costs only a little (0.224 → 0.262) —
   about what "more tokens to encode" should cost. So the damage in (1) is specific to the FREE role,
   not to token count.
3. **In-chunk index redundancy is a real lever.** `chunkpad`'s filler restated each claim's own number
   inside its chunk and *improved* chunked (0.224 → **0.198**), the only intervention here that helped.
   Binding chunk↔index across 100 isolated chunks is exactly chunked's weakness, and this addresses it
   directly and cheaply. Worth pulling on — but note it is not a neutral-filler control (that is
   `chunkpad2`); it is its own finding.

⚠ These filler arms are reported on **train CE only** — no held-out f1. The eval harness builds prompts
via `segment_prompt_to_chunks`, which only knows the (broken) repeat-mode filler, so it cannot reproduce
the varied filler. Threading a varied-filler generator through that single source of truth is the next
step; until then treat this section as directional.

## 3. Gold-gradient (O(1) backward) at n=100 — **does NOT replicate on clean data**

Full attention (`random_doc` @ `doc_keep_prob=1.0`, provably identical to plain causal); only the
*backward* gradient policy differs. `k=3` pairs ⇒ **6 gold docs of 100 (6% base rate)**.

| arm | gradient flows through | gold fraction | train CE | held-out f1 | ±SE |
|---|---|---|---|---|---|
| `full` | all 100 docs | 6% (base) | 0.0014 | **0.931** | ±0.011 |
| `gold_subsample` N_GOLD=1, n_rand=8 | 1 gold + 8 random = 9 | 11% | 0.103 | **0.570** | ±0.022 |
| `random_only` n_rand=8 | 14 random (gold hidden) | ~6% | 0.216 | **0.533** | ±0.023 |
| `gold_plus_random` n_rand=8 | all 6 gold + 8 random = 14 | **43%** | 0.728 | 0.002 ❌ | |

**Two findings, and the second overturns the prior result.**

1. **Gold over-representation is confirmed as the killer variable.** `gold_plus_random` keeps *all* gold
   docs, inflating gold from a 6% base rate to 43% of the gradient — and it collapses to chance (0.002),
   *worse than its own random control*. Consistent with the earlier "75% gold → chance" observation.

2. **Knowing which documents are gold buys nothing.** `gold_subsample` (0.570 ± 0.022) vs `random_only`
   (0.533 ± 0.023): a 0.037 gap against a combined SE of 0.032 — about **1.2 SE, not significant** (and
   seed noise adds more). Sparse backward through ~9–14 of 100 documents lands at **f1 ≈ 0.55 either
   way**, far below full-gradient **0.931**.

⇒ **The claim "gradient through 8–16 of 100 docs ≈ full (f1 ≈ 0.90)" does not hold on leak-free data.**

**Why the old number was inflated (likely):** the previous n=100 goldgrad ran on the **v1 (leaky)**
`*_docdense_nocot_gold` shard, where the `Claim N:` labels sit *outside* every chunk. The KV-detach
detaches whole *document spans*, so those FREE label tokens — which carry exactly the claim indices the
model must emit — were **never detached** and kept gradient in every arm. That leaked a strong,
always-on gradient path that made sparse backward look nearly free. With the labels inside the chunks,
they are detached with their document and the advantage disappears.

Reminder (`goldgrad-o1-backward-experiment`): the KV-detach measured a **1.00x speedup** anyway, so this
was always a *probe* of what gradient signal suffices, not an actual O(1) backward. On clean data, the
answer is: **substantially less than full, and gold identity does not help.**

## Provenance

Runs `n100fix-*` (wandb group `n100fix` / `n100fix-free`) and `q06b-goldgrad-fa-n100-*`.
Shards under `/scratch/users/prasann/longctx_sft_qwen/contra_n100_v2_*`; builders in `debug/`.
