---
title: Training gradient with token dropping vs full context (direction, magnitude, which tokens and layers differ; GDN vs full attention)
date: 2026-10-19
updated: 2026-10-20
experiment: learned_router
area: token_routing
kind: result
status: exploratory (eval_size 32-48 per task per model)
superseded_by: 
artifact: https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1
summary: Token-dropped vs full-context answer-CE gradient, base and task-trained Qwen3.5/Qwen3-4B: the lossless bar keeps only cos 0.26-0.76 of the training direction and kills the signal on hard rows; ds64's outlier rules are no better; no GDN effect.
---
# Training gradient with token dropping vs full context

**Question.** If a model is trained on token-dropped contexts, does its gradient look like the full-context
gradient, in direction and in magnitude? Which tokens and layers cause the mismatch? Does the answer change
without GDN?

**Setup.** Each row is a CTC row at 8k (nq, outlier, contradiction; 32 rows per task, ⚠ eval_size=32 per
(model, task), 96 per model). The loss is the teacher-forced answer CE (no-cot). The token-dropped context
hard-deletes document body tokens with the grid semantics: the sequence is compacted and the original RoPE
positions are kept. Markers, prompt, question and answer are never dropped. Drop rules:

* random;
* |dCE/dg| at gate g = 0.5 (the gold-blind rule from [gradprobe-token-importance](gradprobe-token-importance.md));
* gold first, then that score;
* the bar (`gold_fl20p8_noslot`, ≈26% of body tokens).

Keep ratios are ρ ∈ {50, 20, 10, 5, 2}%.

**Models.** The comparison uses BASE checkpoints, since the question is about the training signal from the
initialization:

* `q35-4b-base-modelonly`: Qwen3.5-4B-Base, 32 layers, 24 GDN + 8 full attention.
* `qwen3-4b-base-trainedmark`: Qwen3-4B-Base with repaired markers, 36 full-attention layers.

Two 2-row smoke tests on the SFT checkpoint setA32k (CE ≈ 0, saturated) are mentioned only for contrast.

**Code.** All in `research/token_routing/learned_router/gradprobe/`:

* `grad_similarity.py`: whole-model parameter gradients, global and per-module cosine, norm ratio; `--demean`
  subtracts the leave-one-out mean full gradient.
* `grad_tokens.py`: exact per-token decomposition, described next.
* `gsim_report.py` and `gtok_report.py`: tables and plots.
* `run_gradsim_local.sbatch`: the wrapper, with `PROBE_FAMILY` and `SCRIPT` settings.

Outputs are JSONs on mooney at `/data/prasann/gradprobe/out/{gtok,gsim}_{q35base,q3nt}_*_8k_r32.json`.
Tables and plots are in `gradprobe/results/gtok/` and `gradprobe/results/gsim/` (gitignored).

**Checks.**

* Recomputing the full-context gradient reproduces it: cosine 0.9999–1.0000.
* The compact path with every token kept matches the full gradient: cosine ≥ 0.9999, ΔCE = 0.
* The hooked CE matches the grid CE exactly.

## The decomposition (`grad_tokens.py`)

A linear layer's weight gradient is a sum of per-token outer products, dCE/dW = Σ_t δ_t x_tᵀ. Here x_t is the
layer input at position t and δ_t is dCE/d(output_t). The full-context gradient G therefore splits exactly
into two parts:

* **D**, the part carried by the dropped tokens, which disappears.
* **K**, the part carried by the kept tokens.

The dropped-context gradient is **K′**: the same kept tokens with their new x′ and δ′. To tell activation
drift from signal drift, two hybrids are formed:

* K_fwd = Σ δ_t x′_tᵀ: activations changed only.
* K_bwd = Σ δ′_t x_tᵀ: incoming gradient changed only.

These are measured for each layer's mixer output projection (`attention.w_out`, GDN or softmax) and FFN down
projection (`feed_forward.w2`). Per token, the factorization cos(rank-1) = cos(x, x′) · cos(δ, δ′) is
aggregated by token category and by distance to the nearest dropped token.

## Findings

1. **The dropped tokens' own share hardly rotates the gradient; the kept tokens' new contributions do.**
   * The kept part K points the same way as the full gradient: cos(K, G) = 0.86–1.0 at every depth. This holds
     even when the dropped tokens carried 30–60% of ‖G‖.
   * Changing only activations leaves the kept part nearly unchanged: cos(K_fwd, K) = 0.8–0.98.
   * Changing only the incoming gradient tracks the total: cos(K_bwd, K) ≈ cos(K′, K), at 0.3–0.6 in the lower
     layers.

   This pattern is the same in both families and for all four rules. The mismatch is the backward signal
   δ arriving at the kept tokens being redistributed.
2. **Why δ changes, by token type.** Numbers are FFN w2 at ¼ and ½ depth; the per-category medians are from
   nq.
   * *Kept body tokens.* Under random 10% keep, a kept token has lost its neighbours, so both its activation
     and its δ decorrelate (median cos ≈ 0.1 and ≈ 0.0). It also absorbs far more gradient: the body share of
     the contribution mass rises from 0.04–0.06 to 0.21–0.42. Span-preserving masks (the bar) keep activations
     largely intact (cos 0.67–0.85 next to a drop, 0.83–0.99 two or more tokens away), but δ is still only
     0.03–0.78, rising with distance from the drop and with depth.
   * *Prefix tokens (system prompt and instructions).* They sit upstream of every drop, so their activations
     are identical (cos 1.00). Their δ still changes, because the positions that read them changed.
   * *Question/suffix and answer positions.* Moderate drift (x 0.7–0.96, δ 0.15–0.9).

   In plain terms, the gradient a context token receives is the "read" signal from later positions. Removing
   tokens renormalizes what is read and re-routes that signal, and the few survivors take a larger share of it.
3. **Alignment rises with depth** in both families, from 0.2–0.5 in the first quarter to 0.7–0.95 in the last
   (FFN w2, per-depth tables in `results/gtok/gtok_report.md`). Part of this is structural: near the top only
   the answer positions feed the loss, so the last layer's gradient is entirely answer positions in both
   contexts.
4. **No consistent GDN vs full-attention difference; the sign flips by task.** The table gives Qwen3.5 − Qwen3
   for FFN w2 cos(K′, G), with a row-bootstrap 90% CI:

   | rule | task | first ¼ | middle ½ | last ¼ |
   |---|---|---|---|---|
   | random 10% | nq | +0.14 [+0.10, +0.18] | +0.29 | +0.02 |
   | random 10% | outlier | −0.05 | +0.16 | +0.04 |
   | random 10% | contradiction | −0.27 [−0.31, −0.23] | −0.23 | −0.29 |
   | bar | nq | +0.34 | +0.23 | +0.07 |
   | bar | outlier | −0.20 | −0.07 | −0.07 |
   | bar | contradiction | −0.30 | −0.25 | −0.21 |

   * Qwen3.5 is more aligned on nq; Qwen3 is more aligned on contradiction; outlier is mixed.
   * The ordering does not follow the loss level consistently. Median answer CE is 1.6 / 1.0 / 0.8 for
     Qwen3.5 and 6.8 / 2.9 / 2.1 for Qwen3 (nq / outlier / contradiction), and the alignment trend across
     tasks runs in opposite directions in the two families.
   * Confounds: the families also differ in pretraining data, tokenizer, depth (32 vs 36 layers) and how well
     they handle this format zero-shot.
   * The bootstrap covers row sampling only: with one model per family, it says nothing about variation
     between models.
   * Verdict: same mechanism; the magnitude depends on the model × task combination, not on GDN.

## Whole-model parameter gradient (`grad_similarity.py --demean`)

This measures cosine and norm over all 4B parameters. Two references matter:

* **Raw cosine** is inflated by a large row-independent direction. Two unrelated rows' full gradients have raw
  cosine 0.19–0.75 (median per task).
* **"Centered" cosine** subtracts the leave-one-out mean full-context gradient over the task's 32 rows from
  both sides. Unrelated rows then sit at −0.02 to +0.05, so centered cosine measures the row-specific signal.

Each cell is the median over rows, raw / centered (⚠ eval_size=32 per cell; plot
`results/gsim/gsim_ccos_vs_rho.png`, full tables `results/gsim/gsim_report.md`):

| model | task (median CE) | random 50% | random 10% | \|grad\|@0.5, 50% | \|grad\|@0.5, 10% | bar |
|---|---|---|---|---|---|---|
| Qwen3.5-4B-Base | nq (1.58) | 0.62 / 0.18 | 0.48 / 0.11 | 0.80 / 0.42 | 0.57 / 0.13 | 0.75 / 0.27 |
| Qwen3.5-4B-Base | outlier (1.02) | 0.46 / 0.13 | 0.36 / 0.11 | 0.55 / 0.33 | 0.38 / 0.15 | 0.49 / 0.24 |
| Qwen3.5-4B-Base | contradiction (0.83) | 0.23 / 0.23 | 0.15 / 0.21 | 0.24 / 0.21 | 0.15 / 0.21 | 0.28 / 0.25 |
| Qwen3-4B-Base | nq (6.79) | 0.50 / 0.18 | 0.37 / 0.11 | 0.62 / 0.42 | 0.47 / 0.20 | 0.57 / 0.26 |
| Qwen3-4B-Base | outlier (2.88) | 0.53 / 0.17 | 0.41 / 0.14 | 0.70 / 0.40 | 0.52 / 0.18 | 0.74 / 0.45 |
| Qwen3-4B-Base | contradiction (2.06) | 0.67 / 0.36 | 0.54 / 0.26 | 0.73 / 0.35 | 0.60 / 0.27 | 0.69 / 0.35 |

5. **Direction: the row-specific part of the training signal is mostly not preserved.**
   * Centered cosines are 0.1–0.45 everywhere.
   * Even the best case (keep half the tokens by |grad|@0.5) reaches only 0.33–0.42.
   * At 10% kept, every rule is 0.11–0.27, barely above unrelated rows on nq and outlier.
   * Raw cosines (0.15–0.80) look far better only because both gradients share the common, row-independent
     component (format, answer template), which dropping does not disturb.
   * Gradient-based masks beat random at high ρ (nq 50%: 0.42 vs 0.18). By 10% the rules converge.
6. **Magnitude is preserved.** The median ‖g_drop‖/‖g_full‖ is 10^(−0.04 to −0.17), i.e. 0.68–0.91×. Dropping
   shrinks the gradient slightly; it does not blow it up or kill it.
7. **GDN vs attention, whole model.** On nq the two families are identical (centered 0.18 / 0.11 / 0.42 / 0.13
   vs 0.20 / 0.27 vs 0.26). On outlier and contradiction Qwen3 is somewhat more aligned (contradiction 0.26–0.36
   vs 0.21–0.25; outlier bar 0.45 vs 0.24). This is consistent with finding 4: no GDN advantage, and if
   anything the attention model keeps more of the row-specific signal on two of the three tasks.
8. **Importance scores are not stable under dropping (secondary measurement).** The question is whether the
   per-token gate gradient over the kept tokens, at g = 0.5, keeps its ranking in the dropped context. Median
   Spearman is only 0.10–0.49 in every cell. The cosine of the signed gate gradient is ≈ 0 under random drops
   in both families. Under gradient or bar masks it is 0.36–0.89 for Qwen3.5 on nq and outlier, but only
   0.12–0.37 on contradiction and 0.07–0.26 for Qwen3 on every task. Re-scoring after a drop gives a different
   importance ranking, consistent with the redistribution story in finding 2.

## Task-trained checkpoints, bar-family drops (2026-10-20)

Prasann's request (2026-10-20): run the same analysis on task-trained checkpoints, with the best heuristic,
`gold_fl20p8_noslot` (the bar), instead of "gold first + |grad|".

**Models.** All are Qwen3.5-4B unless marked; each sits at answer CE ≈ 0.0001–0.003:

* the per-task SFT checkpoints `ctc-4b-{nq,outlier,contradiction}-full`, each on its own task;
* setA32k on all three tasks;
* **Qwen3-4B setA** (`q3-4b-dense-ctc-setA-32k-20261013T2244/step886`, the lengen full-attention arm), on all three
  tasks. It was trained on the same 203,156 examples as setA32k, so it gives a loss-matched GDN vs attention pair.
  Its training-time nothink tokenizer was used (answer span identical in layout to Qwen3.5's).

**Rules** (`grad_similarity.py --rule-set barfam`):

* the bar's own family: gold docs + 8 header tokens per doc + first/last x% per doc, x ∈ {0, 5, 10, 20, 40}
  (x = 20 IS the bar, checked mask-for-mask);
* its gold-blind twin;
* random and |grad|@g=0.5, each at that row's matched budget.

**Setup.** 48 rows per (model, task) at 8k (⚠ eval_size=48).

**Centering.** `--demean 2` projects out the leave-one-out mean DIRECTION of the unit full gradients. The old
mean subtraction is useless on saturated checkpoints: per-row gradient norms differ by up to 1000×, so the biggest
rows dominate the mean, both vectors become ≈ −mean, and the cosine reads ≈ 0.99 for any rule. On these checkpoints
unrelated rows' gradients sit at raw cos −0.05 to +0.08, and projected cos equals raw cos to ±0.05. So **raw cosine
is directly readable here**, unlike on the base models.

Plots: `gradprobe/results/gsim_barfam/` (`gsim_cos_vs_rho.png`, `gsim_norm_vs_rho.png`, `gsim_pcos_vs_rho.png`).
Tables: `gsim_barfam_report.md`, `barfam_summary.md`. Per-ρ series: `series.json`.

At the bar's budget (x = 20; keep 26% nq / 29% outlier / 38% contradiction), each cell is median raw cos / median
norm ratio ‖g_drop‖/‖g_full‖ / mean ΔCE (⚠ eval_size=48 per cell):

| model | task | bar | gold-blind twin | \|grad\|@0.5, same k | random, same k |
|---|---|---|---|---|---|
| per-task SFT | nq | 0.75 / 0.25× / −0.03 | 0.44 / 12× / +0.26 | 0.98 / 1.1× / −0.00 | 0.25 / 94× / +1.0 |
| per-task SFT | outlier | 0.26 / 2.3× / +0.04 | 0.04 / 580× / +0.68 | 0.50 / 1.2× / +0.06 | 0.03 / 860× / +1.2 |
| per-task SFT | contradiction | 0.57 / 1.6× / +0.00 | 0.06 / 800× / +0.76 | 0.46 / 2.2× / +0.08 | 0.10 / 1300× / +1.6 |
| setA32k | nq | 0.61 / 0.14× / −0.02 | 0.27 / 79× / +0.39 | 0.83 / 1.1× / −0.00 | 0.04 / 1400× / +2.1 |
| setA32k | outlier | 0.31 / 6.8× / +0.01 | 0.13 / 260× / +0.16 | 0.34 / 3.0× / +0.01 | 0.04 / 1200× / +1.1 |
| setA32k | contradiction | 0.33 / 1.6× / +0.01 | 0.03 / 400× / +0.59 | 0.19 / 8.4× / +0.10 | 0.01 / 610× / +1.6 |
| **Qwen3 setA** | nq | 0.76 / 0.40× / −0.01 | 0.42 / 200× / +0.38 | 0.86 / 1.3× / +0.03 | 0.18 / 1300× / +1.2 |
| **Qwen3 setA** | outlier | 0.34 / 7.3× / +0.02 | 0.15 / 650× / +0.16 | 0.13 / 59× / +0.41 | 0.07 / 980× / +0.77 |
| **Qwen3 setA** | contradiction | 0.31 / 1.1× / −0.00 | 0.04 / 530× / +0.75 | 0.02 / 180× / +0.44 | 0.02 / 480× / +1.3 |

9. **The bar is lossless in CE but keeps only part of the training direction.**
   * nq: cos 0.61–0.76.
   * contradiction: 0.31–0.57.
   * outlier: 0.26–0.34.
   * Cosine rises slowly with x; even x = 40 stays at 0.3–0.85.
10. **Dropping gold destroys the signal.**
    * The gold-blind twin and random drops raise CE by 0.15–2.1 nats.
    * Their gradients are 10²–10³× larger and nearly unrelated to the full one (cos ≈ 0.0–0.15; 0.27–0.44 on nq).
    * The update is then about learning to cope with missing evidence, not about the full-context answer.
11. **On the rows that still carry loss, the bar removes most of the training signal.** These are the rows with
    full-context CE ≥ 0.05: nq per-task 3 rows, outlier per-task 8 rows (⚠ eval_size 3 and 8 only).
    * The bar lowers CE by 0.2–0.4 nats (it removes the distractors, and on outlier it leaks gold identity).
    * Its gradient shrinks to 0.002–0.006× of full at x ≤ 10, and 0.55× at the bar itself on outlier.
    * Training on bar-dropped contexts therefore barely learns from the examples the model still gets wrong.
    * This matters for the `tokdrop_lr` gfl20p8ns arms.
12. **The oracle |grad|@0.5 selector is best on the per-task models, but not on the multi-task ones.**
    * On nq it is at 0.83–0.98 and norm about 1×.
    * On the setA models' outlier and contradiction it is not lossless (ΔCE +0.10 to +0.44) and its gradient is
      unrelated (0.02–0.19); the bar beats it there.
    * A saturated multi-task model's own gate gradient is a poor token scorer at small budgets.
13. **GDN vs attention at matched loss: no difference.** setA32k (Qwen3.5) vs Qwen3 setA, same training data,
    both at CE ≈ 0.0001–0.0002:
    * bar cos 0.61 vs 0.76 (nq), 0.31 vs 0.34 (outlier), 0.33 vs 0.31 (contradiction);
    * x-sweep curves of the same shape.

    This confirms finding 7 (base models) without that run's loss-regime and format confounds.

### Outlier: the bar vs the ds64 frontier heuristics

The question (Prasann, 2026-10-20) was whether the new heuristic is much worse than what drew outlier's Pareto curves.
ds64's outlier frontier arms are `ck64` and `flk64` (`flk64ns` slot-less):

* gold-blind;
* each doc's header up to `Document [N]:` (stop id 5491);
* first 64 (`ck64`) or first 32 + last 32 (`flk64`) body tokens per doc;
* the rest pooled or dropped.

They run gold-blind because on outlier the gold doc's identity is the answer. ds64's record of these arms:
`ck64-32M` dominated dense at 4.0× fewer FLOPs, and `flk64-64M` had the largest matched-FLOP gain. The
`results/ds64/results.csv` behind those numbers is no longer on disk.

Same rows, same checkpoints, 48 rows each (⚠ eval_size=48; `gsimflk_*` JSONs, `outlier_ds64_frontier_vs_bar.md`).
Each cell is median raw cos [row IQR] / norm ratio / ΔCE:

| rule | keep | per-task outlier SFT | setA32k |
|---|---|---|---|
| bar (gold-forced) | 29% | 0.26 [0.13, 0.39] / 2.3× / +0.04 | 0.31 [0.22, 0.42] / 6.8× / +0.01 |
| flk64 | 48% | 0.22 [0.13, 0.37] / 14× / +0.16 | 0.33 [0.24, 0.47] / 9.9× / +0.01 |
| ck64 | 48% | 0.21 [0.09, 0.36] / 17× / +0.20 | 0.30 [0.20, 0.37] / 19× / +0.04 |
| bar, gold-blind twin | 25% | 0.04 / 580× / +0.68 | 0.13 / 260× / +0.16 |
| \|grad\|@0.5 at flk64's k | 48% | 0.71 / 1.0× / −0.00 | 0.52 / 1.0× / −0.00 |
| random at flk64's k | 48% | 0.03 / 970× / +1.06 | 0.03 / 1550× / +1.08 |

14. **The bar is not worse than the ds64 frontier on outlier.**
    * Gradient direction is indistinguishable: 0.21–0.33, overlapping row IQRs.
    * The bar uses about 40% fewer tokens.
    * Its CE is at least as close to lossless.
    * Its norm inflation is no larger: 2–7×, vs 10–19× for flk64/ck64.
    * The bar's caveat on outlier is the gold-shortcut signature on its 8 hard rows: CE goes down (−0.06) and the
      gradient shrinks (0.55×), where flk64 keeps 1.15× and cos 0.14 vs 0.11.
    * Outlier is the task where every heuristic keeps the least of the training direction; even the oracle
      reaches only 0.5–0.7.
    * Scope: these are full-context-trained checkpoints. ds64's flk64 checkpoints were trained WITH that
      construction, which is a different question.

### Dev-loss grid: do the top heuristics transfer to Qwen3?

Both setA models were scored on the dev-loss grid, Qwen3.5 setA32k vs Qwen3 setA:

* 11 setA tasks × 2k/8k/32k, 32 cells (reorder has no 32k);
* 48/48/32 rows, the same rows for both models;
* the artifact's top heuristic schemes;
* the grid driver `ctc_devloss_grid.py` with the new `olmo_st` format, via `run_grid_ckpt.sbatch`.

Comparison: `research/token_routing/devloss_grid/results_family/setA32k_vs_q3setA.md` (`compare_ckpts.py`). JSONs:
sneetches `/data/prasann/devloss_grid/results_{setA32k,q3setA}/`. Both models are trained and read the
context (full CE 0.000–0.6 except oolong, whose CE covers the plan).

Cells at parity, ΔCE ≤ 0.02 | ≤ 0.05 | ≤ 0.10, over 32 (⚠ 32–48 rows per cell):

| scheme | Qwen3.5 setA32k | Qwen3 setA | Qwen3, 32k only (of 10) |
|---|---|---|---|
| `gold_fl20p8_noslot` (the bar) | 25 \| 26 \| 27 | 24 \| 26 \| 28 | 7 (median −0.007) |
| `gold_rand20p8_noslot` | 24 \| 26 \| 26 | 21 \| 23 \| 26 | 7 (median −0.016) |
| `gold_fl20p8` (slot) | 25 \| 26 \| 27 | 16 \| 18 \| 19 | **0** (median +1.27) |
| `gold_first20` (slot) | 24 \| 26 \| 26 | 14 \| 17 \| 18 | **0** (median +1.42) |
| `gold_fl64` (slot) | 29 \| 30 \| 30 | 19 \| 21 \| 21 | **1** (median +0.70) |
| `first128` (gold-blind, slot) | 26 \| 30 \| 30 | 21 \| 22 \| 23 | **1** (median +0.18) |
| `fl20p8_noslot` (gold-blind bar) | 6 \| 9 \| 11 | 7 \| 9 \| 11 | 3 |

15. **The bar transfers to full attention.** Hard-dropping with gold + headers + first/last 20% is at parity on
    24–25 of 32 cells for both models.
    * Shared failures: xabsence (+0.83 at 8k on Qwen3.5) and reorder (+0.57 at 8k on Qwen3).
    * The gold-blind version fails on both alike (+0.3 median).
16. **Slot-based schemes break on Qwen3 at 32k only.** Every `cent_cmean`-slot construction is at parity at 2k and
    8k on both models, but on Qwen3 at 32k it reaches only 0–1 of 10 cells (median ΔCE +0.18 to +1.42;
    contradiction +2.7). The same masks without the slot are fine.
    * Not diagnosed. Candidates: pure attention reads ~200 slot tokens in every layer at 32k (Qwen3.5 has 8
      attention layers), or a 32k slot/position issue on the Qwen3 path.
    * Ruled out: a stale RoPE cache. The driver re-warms it per rung, to 33.6k positions at 32k, for both models.
    * Per-cell evidence: the noslot twin and full context stay at top-1 0.98–1.00, while the slot variants drop
      to 0.42–0.83. Example, contradiction@32k: full 0.080, noslot 0.034, slot 2.80.
    * Use no-slot constructions on Qwen3; this would also matter for any soft-token training on a full-attention
      model.

## Traps found along the way

* **Qwen3 chat template.** The stock Qwen3-4B template puts `<think>\n\n</think>\n\n` inside the assistant
  turn, so the answer loss mask covered the think scaffold. The Qwen3.5 template puts it in the prompt. Fixed
  with `/data/prasann/hf_tokenizers/Qwen3-4B-nothink` on mooney (generation prompt always emits the empty
  think block); the wrapper uses it for `PROBE_FAMILY=qwen3`. Qwen3 base CE barely moved (7.4 → 7.0 on nq),
  so its high CE is genuine.
* **Base-model gradients share a large row-independent direction.** For Qwen3.5 base, two unrelated rows'
  full gradients have raw cosine 0.4–0.9. Raw drop cosines therefore overstate the similarity; read the
  centered ones.
* **SFT checkpoints are the wrong regime.** At answer CE ≈ 0.0002, a random drop moves CE by +2–3 nats and
  the gradient by 30–700× in norm, so the comparison is between a saturated and an unsaturated loss.
  ⚠ setA32k smoke only, eval_size = 2.

## Next steps (not started)

* **Matched-regime GDN test:** setA32k (Qwen3.5) vs `q3-4b-dense-ctc-setA-32k-20261013T2244/step886` (Qwen3,
  the lengen arm). Both were trained on the same 203,156 examples, so the loss regime matches. The Qwen3
  checkpoint is on weka only and needs a gantry model-only resave to S3.
* **Does the mismatch matter for training?** One SGD step on g_drop vs g_full, then compare the full-context
  loss change.

## Does a frozen-checkpoint metric predict which keep rules TRAIN well? (2026-10-21)

Prasann's question after `tokdrop_lr` showed bar (`gold_fl20p8_noslot`, gold-only whole) collapsing in training
while the dev-loss grid ranked it near the top: does gradient similarity predict training outcomes better?

**Setup.** 20 keep rules with MEASURED training outcomes (7 won, 13 lost): ds64 arms in hard-drop form
(contradiction chk16/32/64 W, hdr33 W, hdr17/08/03 L; nq kv33 W, kv17 L, nqk32/64 L; outlier ck64/flk64 W,
ck16 L, cc00 L, kv33 L, kvgb L) plus bar (L on all three, `tokdrop_lr`). Checkpoints: per-task SFT
`ctc-4b-{contradiction,nq,outlier}-full` and setA32k, 8k, ⚠ eval_size=24 rows per cell. New in
`grad_similarity.py`: `--frac kv|hdr|gb`, `--ck K:none`, and per-doc kept counts for mask-only gold-leak scores.
Report: `research/token_routing/learned_router/gradprobe/predict_report.py` (table in
`predict_report_2026-10-21.md`); JSONs mooney `/data/prasann/gradprobe/out/gsimpredq_*_8k_r24.json`.

**AUC separating W from L over rules** (⚠ 7W/13L, ±~0.13):

| metric | per-task | setA32k |
|---|---|---|
| gradient cosine (dropped vs full) | 0.65 | 0.70 |
| dev-loss parity (−ΔCE) | 0.58 | 0.63 |
| leak: AUC−0.5 of "doc kept whole", gold vs non-gold (lower = better) | 0.71 | 0.71 |

15. **Gradient cosine has the dev-loss blind spot.** Bar's cosine matches the winners' (nq 0.61–0.76 vs kv33
    0.66–0.68; outlier 0.26–0.31 vs ck64/flk64 0.21–0.32), and the collapsed gold+few-decoy arms (hdr17/08/03,
    0.26–0.66) match hdr33. Within gold-blind rules it ranks sensibly (outlier winners above ck16 / cc00 / kvgb).
16. **The mask-only leak score sees exactly what the frozen metrics miss.** Leak (whole) tracks the decoy rate:
    bar 0.50, hdr03 0.49, hdr08 0.46, hdr17 0.42, kv17 0.41 (all L) vs hdr33 / kv33 0.33 (W). It is blind to
    gold-blind information loss (ck16, cc00, kvgb ≈ 0), which ΔCE / cosine catch (ΔCE 0.3–1.5).
17. **Post-hoc two-part check** ("leak ≤ 0.33" AND small frozen ΔCE) classifies 18/20 (per-task) and 19/20
    (setA32k). Misses: contradiction ck16 (won with ΔCE 0.19–0.26) and per-task outlier kv33. Thresholds were
    chosen after seeing outcomes: a hypothesis, not a validated predictor.

18. **Correction and the continuous version (same day).** Outlier ck16 was mislabeled L above: at MATCHED
    FLOPs it beats dense (+0.15 at 128M, 0.09x FLOPs); it loses only at the same token budget. Using the
    continuous outcome (best measured ds64 Δf1 vs dense at matched FLOPs) instead of W/L, Spearman ρ over the 17
    ds64 rules: gradient cosine 0.01 (per-task) / 0.21 (setA32k); −ΔCE 0.03 / 0.12; −gold leak 0.58 / 0.58
    (0.68 with bar's step-matched outcome included). Scatter: the `Token-drop Pareto on setA` artifact
    (https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5, section "Which frozen-model metric predicts..."); data
    `research/token_routing/learned_router/gradprobe/predict_points_2026-10-21.json`.
19. **Held-out test (same day).** 17 more ds64 arms with measured outcomes (contradiction runs03/08; nq kvgb, nqcc17/08;
    outlier kv17, kvgb50, xh2k50, cc03/08, ck32, flk32; oolong ohdr33/17/08, occ08/00 on setA32k only) scored with new
    `--frac runs|gbh` masks (⚠ 24 rows each); the combined score's z-scaling frozen on the original 17
    (`heldout_predict.py`, `heldout_points_2026-10-21.json`, mooney `gsimheld_*`). Spearman ρ vs matched-FLOP Δf1 on
    held-out arms only, per-task (12) / setA32k (17): combined 0.44 / 0.63; gold leak 0.45 / 0.37; −ΔCE −0.23 / 0.19;
    gradient cosine −0.46 / 0.44. All 34 on setA32k: combined 0.71. Robust part: every arm with whole-doc leak > 0.4
    (5 held-out gold+few-decoy arms, 6 original, bar) failed to beat dense. Misses on the win side: kvgb50 / xh2k50
    (near parity despite ΔCE 0.5–0.8). Many held-out dense comparisons are extrapolated; ρ at n = 12–17 is ±~0.25.
