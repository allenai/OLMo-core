# Attention masks at n=100 documents (Qwen3-0.6B, contradiction)

**Setup (identical for every row unless stated).** Data `contra_n100_v2_base` (2000 examples,
leak-free: the `Claim N:` label is INSIDE the chunk, FREE=217 tokens/example). Base
`q06b-dense-cpt-modelonly-trainedmark` (norm-repaired markers — see
`records/n100-chunked-marker-position-bug.md`). seq_len 6144, 3 epochs = **750 steps**, global batch
8 sequences, DP=2. **Curriculum mask-mixing on** (`p_standard` 0.8 → 0.0 over 3000 forwards) — the
project default; it means ~40% of training forwards run under the plain-causal mask, with the
structured mask only fully in force late. Eval: held-out `contradiction_eval_pubmed_both_n100_k3`,
**eval_size = 488** (the entire file), parse_rate 1.00, no mixing at eval. Binomial SE quoted;
run-to-run seed noise adds more on top and **every arm is a single seed**.

`CE` = mean of the last 50 logged steps (the last *single* step is noisy — quoting it was an earlier
mistake here).

## 1. Mask comparison

| mask | cross-doc connectivity | CE (mean-50) | f1 | ±SE |
|---|---|---|---|---|
| `standard` (plain causal) | full | 0.017 | **0.934** | ±0.011 |
| `hierarchical_dilated` K=50, cyc5 | strided, wide | 0.054 | **0.831** | ±0.017 |
| `random_doc` k=0.5 | random 50% | 0.091 | 0.735 | ±0.020 |
| `hierarchical_dilated` K=10, cyc5 | strided, narrow | 0.186 | 0.505 | ±0.023 |
| `chunked` (pure) | **none** | 0.246 | 0.441 | ±0.022 |

1. **"At n=100 only dense attention works" is FALSE.** Pure chunked — documents fully isolated, all
   cross-document work done at the trailing FREE positions — reaches f1 **0.441** on a 100-way task.
2. **Hierarchical beats chunked decisively at n=100: 0.831 vs 0.441** (~18 SE). This comparison could
   not be made before the marker fix, because both arms were pinned at chance.
3. f1 is monotone in cross-document connectivity, and a wide-stride hierarchical mask recovers most of
   the gap to full attention.

## 2. Full-attention-layer hybrids (`--full-attention-layers`)

Chunked mask everywhere except the listed layers, which run plain causal. Qwen3-0.6B has **28 layers**;
`fa14` = layer 14, `fa5mid` = 12–16, `fa10mid` = 9–18.

| arm | full layers | fraction of net | CE (mean-50) | f1 | ±SE |
|---|---|---|---|---|---|
| `chunked` | 0 / 28 | 0% | 0.246 | 0.441 | ±0.022 |
| `fa14` | 1 / 28 | 4% | 0.292 | **0.317** | ±0.021 |
| `fa5mid` | 5 / 28 | 18% | 0.119 | 0.715 | ±0.020 |
| `fa10mid` | 10 / 28 | 36% | 0.023 | **0.923** | ±0.012 |
| `standard` | 28 / 28 | 100% | 0.017 | 0.934 | ±0.011 |

*(`fa0` = first layer, `fam1` = last layer, `fa20mid` = layers 4–23: training/eval in flight.)*

**A single full mid-layer does not help at n=100 — it HURTS.** `fa14` scores 0.317 vs pure chunked's
0.441, a ~4 SE gap, and its CE agrees (0.292 vs 0.246). This is the **opposite** of the n=20 result,
where one mid full layer was the standout.

It is a **dose curve, not a magic layer**, and the f1s confirm it end to end: chunked (0 layers)
**0.441** → `fa14` (1) **0.317** (the disruptive dip) → `fa5mid` (5) **0.715** → `fa10mid` (10)
**0.923** → `standard` (28) **0.934**. One layer is in a bad regime (enough to disrupt the chunked
structure, not enough to do the cross-document work); 5 layers clears hier-K10; and ~10/28 layers
already matches full attention within noise on both f1 (0.923 vs 0.934, ~0.8 SE) and CE (0.023 vs
0.017). (`n100fix-fa5mid` / `n100fix-fa10mid`, eval_size=488, parse_rate 1.0.)

### n=20 reference (for contrast — different corpus size, NOT comparable in absolute terms)

| arm | f1 (eval_size=488) |
|---|---|
| `fa14` (1 mid layer) | 0.925 – 0.948 |
| `fa0` (first layer) | 0.699 |
| `fam1` (last layer) | 0.703 |

## 3. Filler probes: more FREE tokens vs more within-chunk tokens

⚠ **Train CE only — no held-out f1.** The eval harness builds prompts via `segment_prompt_to_chunks`,
which only knows the (broken) repeat-mode filler, so it cannot reproduce the varied filler. Directional.

**The stock knobs are broken.** `free_pad_repeat` appends N copies of *one identical sentence* and
`repeat_doc_text` duplicates each claim *verbatim*. Both collapse training **even under plain causal**,
where the FREE role is irrelevant — so they measure repeated-text damage, not capacity. See
`records/free-pad-probe-is-confounded.md`.

| filler | added tokens | chunked CE | plain-causal CE |
|---|---|---|---|
| none (`base`) | 0 | 0.224 | 0.0008 |
| `free60` — 60× the **same** sentence | +481 FREE | 0.80 ❌ | **0.81** ❌ |
| `rep2` — each claim duplicated **verbatim** | +~4600 in-chunk | 0.795 ❌ | — |
| `free60v` — 60 **distinct** sentences | +813 FREE | 0.428 ✅ | **0.0003** ✅ |

Rebuilt with varied, budget-matched, content-neutral filler (differing only in the *role* the tokens carry):

| arm | in-chunk tokens | FREE tokens | chunked CE |
|---|---|---|---|
| `base` | 4598 | 217 | **0.224** |
| `free60v` — more FREE | 4598 | 1030 (+813) | **0.428** — *much worse* |
| `chunkpad2` — more in-chunk, index-free | 5918 (+1320) | 217 | 0.262 — mild cost |
| `chunkpad` — more in-chunk, filler **restates the claim index** | 5710 (+1112) | 217 | **0.198** — *helps* |

1. **The FREE-capacity hypothesis is REFUTED.** Widening the FREE budget does not rescue chunked — it
   makes it substantially worse (0.224 → 0.428), while the *same* filler under plain causal is harmless
   (0.0008 → 0.0003). The chunked collapse is not a FREE-position capacity limit.
2. Adding the same budget of neutral tokens *inside* chunks costs only a little (0.224 → 0.262), so the
   damage in (1) is specific to the FREE role, not to token count.
3. **In-chunk index redundancy is a real lever.** `chunkpad`'s filler restated each claim's own number
   inside its chunk and *improved* chunked (0.224 → **0.198**) — the only intervention that helped.
   Binding chunk↔index across 100 isolated chunks is exactly chunked's weakness. Worth pulling on.
   (Note it is not a neutral-filler control — that's `chunkpad2`; it is its own finding.)

## Retracted

**All pre-fix n=100 hybrid numbers.** Every n=100 full-attention-layer run before 2026-07-14 scored
f1 0.000–0.012 — they were flatlined by the marker-embedding bug, not by the mask:

| retracted run | f1 | why it's void |
|---|---|---|
| `fa14-c100`, `fa5mid-c100`, `fa20mid-c100`, `fa10mid-c100` | 0.000 – 0.012 | marker-norm bug → CE flat at 0.79 for *every* mask incl. plain causal |

⇒ the old verdict **"mid-layer full attention doesn't help at n=100" was an artifact** — it was
measuring a broken base, not a mask. The rerun above supersedes it (and happens to reach a similar
conclusion for `fa14`, for an entirely different reason).

**All pre-fix n=100 mask numbers** are likewise void, and the *older* leaky-shard numbers are not
comparable either (v1 chunked mask density 0.344 vs 0.099 leak-free).

## Provenance

Runs `n100fix-*` (wandb group `n100fix` / `n100fix-free`). Launcher
`src/scripts/train/memexpress/attn_explore/run_q06b_attn_explore_mooney.sbatch` with
`DATA_SRC_OVERRIDE`, `BASE_SRC`, `FULL_ATTN` (colon-separated!), `NGPU=2`.
Shards under `/scratch/users/prasann/longctx_sft_qwen/contra_n100_v2_*`; builders in `debug/`.
