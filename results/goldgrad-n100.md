# Gold-gradient (sparse backward) at n=100 documents (Qwen3-0.6B, contradiction)

**What goldgrad is.** Keep every document in the **forward** pass (loss identical), but let only a
selected set of documents drive the **weight update**: `k = torch.where(m, k, k.detach())` on the K/V
entering every layer's attention (`olmo_core/nn/attention/gold_grad_mask.py`). FREE tokens
(instruction / query / answer) always keep gradient. Which docs are selected is the `--grad-mode`
policy. The gold set is looked up per example by a content fingerprint against a sidecar, so gold
identity never enters the token stream.

**Setup.** Data `contra_n100_v2_gold` (2000 examples, leak-free). Base
`q06b-dense-cpt-modelonly-trainedmark` (norm-repaired markers). Full attention
(`random_doc @ doc_keep_prob=1.0`, provably identical to plain causal) — **only the backward policy
differs between arms.** seq_len 6144, 3 epochs = **750 steps** (global batch = `grad_accum × seq_len`
= 8 sequences, independent of world size), DP=2. Eval `contradiction_eval_pubmed_both_n100_k3`,
**eval_size = 488** (entire file), parse_rate 1.00. Single seed per arm.

`k=3` pairs ⇒ **6 gold docs of 100 = a 6% base rate**.

## 1. What each arm actually contains

`random_only` draws uniformly from **all** docs, so it contains gold only *by chance*, at the
hypergeometric rate `6 × n_keep / 100`. This is the single most important thing to understand about
these results — it confounds gold count with doc count by construction.

| arm | gradient docs | gold docs (avg) | gold % | complete pair? | f1 | ±SE |
|---|---|---|---|---|---|---|
| `full` | 100 | 6 | 6% | always | **0.931** | ±0.011 |
| `rand15` | 21 | 1.26 | 6% | ~never | 0.710 | ±0.021 |
| `gsub1_8` | 9 | **1** (exact) | 11% | **never** | 0.570 | ±0.022 |
| `gsub1_15` | 16 | **1** (exact) | 6.3% | **never** | 0.554 | ±0.023 |
| `rand8` | 14 | 0.84 | 6% | 5% | 0.533 | ±0.023 |
| `rand2` | 8 | 0.48 | 6% | ~never | 0.456 | ±0.023 |
| `gpr8` | 14 | **6** (all) | **43%** | always | 0.002 ❌ | |

## 2. Findings

**a. There is no O(1) free lunch on clean data.** Sparse backward costs real accuracy, roughly in
proportion to sparsity: 8 → 14 → 21 → 100 gradient docs gives f1 0.456 → 0.533 → 0.710 → 0.931.

**b. Gold COUNT drives f1; doc count barely does.**
- `gsub1_8` (9 docs, 1 gold) = 0.570 vs `gsub1_15` (16 docs, 1 gold) = 0.554 — nearly double the
  gradient budget, same gold, **f1 flat within noise**.
- `rand2` (8 docs, 0.48 gold) = 0.456 vs `gsub1_8` (9 docs, 1 gold) = 0.570 — same doc count, more
  gold, **+0.114 against a combined SE of ~0.032 ≈ 3.5 SE**.

**c. Gold OVER-representation is catastrophic.** `gpr8` keeps all 6 gold, inflating gold from a 6% base
rate to **43%** of the gradient, and collapses to 0.002 — far below its own random control. This is
what `gold_subsample` exists to avoid.

**d. The `random_only` family cannot answer the gold question at all.** Uniform sampling makes
`gold ≈ 0.06 × docs`, so gold count and doc count are perfectly confounded within it. That is why its
budget curve looks so cleanly monotone.

## 3. The pair problem (open, running)

The label is a **relation** — the model must emit `[[9, 28], ...]`. But
`gold_chunks_from_gold_doc_indices` flattens `[[9,28],[10,31],[16,50]]` into an unordered **set**, and
the sidecar was stored flat, so *no keep-policy could ever hold a pair together on purpose*. In every
`gsub1_*` arm the backward pass sees **one half of one pair** and its partner is detached — even though
the forward pass reads both.

⇒ "sparse backward loses accuracy" may really be "**we never gave it an intact pair**."

Fixed: rebuilt the sidecar preserving pairs (`gold_pairs.json`, verified — all 2000/2000 examples
align and reproduce the flat set exactly) and added two pair-aware modes, `gold_pair` and
`gold_halves`. Four arms, all at **exactly 14 gradient docs**:

| arm | docs | gold | complete pair | f1 |
|---|---|---|---|---|
| `nogold8` | 14 | 0.00 | 0% | *pending* |
| `gsub1_13` | 14 | 1.00 | 0% | *pending* |
| `half2_12` | 14 | **2.00** | **0%** | *pending* |
| `pair1_12` | 14 | **2.00** | **100%** | *pending* |

`half2_12` vs `pair1_12` hold gold count, doc count and gold fraction fixed and differ **only** in
whether the two gold docs contradict each other. (Both sit at a 14% gold fraction, above the 6% base
rate — fine for the contrast, but neither is directly comparable to `full`.)

## 4. Why the OLD numbers were so high — RESOLVED (2026-07-15): they are REAL, and v1-specific

The original `famark` (v1 data, `fixmark` base, 2026-07-13) numbers **reproduce exactly** when its own
checkpoints are re-scored with a correctly-configured harness (v1-wrap eval, `--variant full`,
**`--max-length 8192`**, eval_size 488):

| famark arm | reproduced f1 | recorded |
|---|---|---|
| `full` | **0.945** | 0.945 ✓ |
| `gsub1_15` (16 docs, 1 gold) | **0.905** | 0.905 ✓ |
| `rand2` (8 docs, ~0.5 gold) | **0.900** | 0.900 ✓ |

So **sparse backward ≈ full IS a genuine result on the v1 data** — famark's models truly generalize.
The clean **v2** rerun (§1) is *also* real: `full` 0.938, `gsub1_15` 0.554, `rand2` 0.456. Both rows
reproduce. The result holds on v1 and collapses on leak-free v2 ⇒ **"grad through 8–16 of 100 docs ≈
full" is v1-DATA-SPECIFIC, not general.**

**The free-token leak is UN-retracted — it is the leading mechanism.** The KV-detach severs whole
document spans but never the FREE `Claim N:` label tokens (`keep = roles == FREE_CHUNK_ID` is always
true). In the v1 wrap those labels carry the answer-relevant structure, so sparse backward keeps an
always-on gradient path there; in v2 (label inside the chunk) detaching a doc also detaches its label,
and sparse backward genuinely loses ~35 f1. The earlier "v1 and v2 are structurally identical → no
leak" decode was wrong-headed: the SAME `fixmark` base learns v1 (CE 0.0007) but flatlines v2 (CE 0.79)
— proof the two are not equivalent.

**Two bugs produced the false "does not replicate" story:**
1. **Eval truncation.** The n100 prompt is ~6144 tokens; the eval ran `--max-length 4096`, leaving the
   model **zero** generation budget → it emits an EMPTY string → f1 0.000 at parse_rate 1.0 for even a
   perfect model. This silently zeroed most n100 goldgrad evals (incl. the "v2 full = 0.000" anchor and
   the whole apparent "variant flip"). Fixed: launcher scales `MAXLEN` to 8192 for n100.
2. **The `v1fx` / `v1tm` "controls" are BROKEN re-trains, not evidence.** They reach CE ~0.001 on train
   but score 0.000 / 0.066 on held-out — they generate plausible-but-WRONG pairs (e.g. gold
   `[[6,64],[10,76],[48,93]]` → pred `[[12,63],[13,46],[41,78]]`), a train/eval render mismatch (the
   `contradiction_n100_docdense_nocot_gold` shard's wrapping matches neither the v1- nor v2-wrap eval).
   They do NOT reproduce famark and must be discarded, NOT read as "the 0.9 is gone."

**Net:** the O(1)-backward learning finding stands *as a v1 result* but does not survive de-leaking; on
clean data sparse backward costs real accuracy (0.938 → 0.55). The one genuinely open question is why
`v1fx` (a nominal re-run of famark) trains to a non-generalizing model — most likely a different/older
v1 shard with a wrap the eval doesn't match.

## Retracted / corrected

| claim | status |
|---|---|
| "grad through 8–16 of 100 docs ≈ full (f1 ~0.90)" | ✅ **RE-CONFIRMED on v1** — famark reproduces 0.905/0.900. It is v1-data-specific; on leak-free v2 the same arms give 0.554/0.456. |
| "The old number was inflated by the free-token leak" | ✅ **UN-RETRACTED** — leading mechanism; v1≠v2 is proven (same base learns v1, flatlines v2). |
| "v2 sparse doesn't replicate / full = 0.000" | ⚠ **was the MAXLEN-4096 truncation bug.** With `--max-length 8192`: v2 full = **0.938**, sparse gsub1_15 = 0.554, rand2 = 0.456 (validated vs historical). |
| "Gold identity buys nothing" | **RETRACTED** — near-vacuous comparison (arms had near-equal gold); at matched doc count gold clearly helps (§2b). |
| KV-detach gives an O(1) backward | **RETRACTED long-standing** — measured speedup is **1.00x**. It is a *probe* of what gradient signal suffices, not a real O(1) backward. |

## Traps found (cost real time — don't repeat)

- **Eval sbatch hardcoded `CR_SRC` to the `/scratch` standalone checkout**, whose eval script lacks the
  no-cot EOS/truncation fix → a CE-0.0014 model scored **f1 0.000 at parse_rate 1.0**. If a well-trained
  model evals at exactly 0 with parse_rate 1.0, suspect the harness, not the model. Fixed.
- Eval **data** is not in-tree (code is, JSONLs live on `/scratch`) → added `CR_DATA_SRC`.
- `NGPU` defaults to *whatever GPUs are idle* → world size, and with it step count, silently floats
  between runs. **Pin `NGPU`.**
- `global_batch_size = grad_accum × seq_len` = 8 sequences ⇒ `steps = n_examples × epochs / 8`. The 500-
  example v1 shard needs **12 epochs** to reach famark's 750 steps; at 3 epochs it is 187 steps and comes
  back undertrained (f1 0.04). This bit me once and I misdiagnosed it as a world-size effect.
- torchrun rank count vs `--gres` mismatch → rank 1 dies with `invalid device ordinal`.

## Provenance

Runs `q06b-goldgrad-gg3-n100-*` (clean), `q06b-goldgrad-famark-n100-*` (old, v1+fixmark),
`q06b-goldgrad-v1{tm,fx}-n100-*` (controls). Launcher
`src/scripts/train/memexpress/goldgrad/run_q06b_goldgrad_contra.sbatch`.
