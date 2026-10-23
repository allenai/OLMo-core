---
title: Per-token x per-block gradient and activation composition of the answer loss over CTC fine-tuning (Qwen3.5-4B), for router design
date: 2026-10-23
updated: 2026-10-23
experiment: grad_composition
area: token_routing
kind: result
status: exploratory (eval_size 64 rows/task on mt4 rows, 19-23 on mt11 rows; mt4 31/61/122, LR 2e-6 control, mt11 165 + router queued)
superseded_by:
artifact: https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ
summary: Document tokens' gradient share is flat to block ~8 then falls to ~0.1 by block 24; token rankings decorrelate with depth and training (G7~G24 0.57 -> 0.35), so rescore once at ~16; activation targets disagree with the gradient and keep no better than random; blocks 0-6 hold 0.45 -> 0.20-0.33 of the parameter gradient; strmatch is incompressible at 10%/document (short documents), grouping is not explained by composition.
---
# Gradient composition over CTC fine-tuning

**Brief (parent session, 2026-10-22).** Measure how the per-token, per-block gradient (and activation) composition of
the answer loss changes over fine-tuning on CTC tasks, to suggest how train-time token / layer routers should be set
up: where to split, one keep decision vs per-layer, gradient vs activation targets, why strmatch and grouping resist
dropping. Results page: https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ (private). Experiment state:
`ops exp show grad_composition`.

## Setup

* **Probe** (`research/token_routing/grad_composition/probe.py`). Frozen Qwen3.5-4B (olmo-core, bf16), one teacher-forced
  forward + backward of the answer CE (no-cot) per row. Each block runs as two checkpointed halves (mixer, FFN), so the
  residual stream is observed at the block input, between the halves and at the output. Per token t and block b:
  * G = ‖dL/dh_b,t‖ at every block input (b = 0 embedding output … 32 final output);
  * mixer and FFN residual updates;
  * first-order skip attribution ⟨dL/dh, Δh⟩;
  * the token's share of the parameter gradient at each block's output projections (‖δ_t‖·‖x_t‖ at `w_out` and `w2`,
    the exact norm of its rank-1 term);
  * attention mass received at the 8 full-attention blocks (3, 7, …, 31), from the answer-predicting positions, the
    query tokens and all queries;
  * exact per-block parameter-gradient norms, per row and for the task's summed (minibatch) gradient.
* **Token categories.** Prompt, markers, header (the first 8 body tokens of a document), first 32, middle, last 32 —
  gold / non-gold from `gold_fingerprints.json` — query, answer, other.
* **Per-row statistics.**
  * category mass and top-10% composition per block;
  * concentration;
  * cross-block Spearman of body-token rankings;
  * gradient-vs-activation rank agreement;
  * split fidelity: the parameter-gradient energy retained when every token runs blocks < L and each document's top 10%
    by the layer-L score runs the rest;
  * a forward-only ablation (ΔCE after keeping 10% per document before block 7, under 7 rankings);
  * pyramid containment;
  * within-document concentration of the dropped mass.
* **Rows.** Every row set is checked by md5 against every training shard (`results/mt{4,11}_seen.json`).
  * *mt4 rows*: `mt4s1952` rows 1464-1527 of contradiction / nq / outlier / oolong. The mt4 shards are true prefixes, so
    these are unseen by every mt4 run up to s1464. ⚠ eval_size = 64 rows per task.
  * *mt11 rows*: `mt11s720` rows 480-543 of all 11 tasks. The mt11 shards are shuffled supersets: 42 of these 64 per task
    are mt11s480 training rows. Every mt11 comparison therefore uses the 19-23 rows per task that neither mt11s240 nor
    mt11s480 contains. ⚠ eval_size ≈ 22 per task.
  * mt11 rows use the setA prompt format (query both before and after the documents). They are out of format for
    mt4-trained checkpoints (CE 0.47 vs 0.02 on contradiction), which are kept only as a dashed reference.
* **Checkpoints** (Lambda node02 `/tmp/prasann/tdlr/`).
  * Base: `q35-4b-base-markerfix`.
  * mt4 dense, LR 5e-6: 31 / 61 / ~122 / 366 steps (`f8k-mt4s-dense-lrb-…-s31/s61`, `…-lra-lr5e-06`,
    `f8k-mt4s1464-dense-lre-lr5e-06`), plus LR 2e-6 at 366 steps (`…-lri-lr2e-06`).
  * mt11 dense, LR 5e-6: 165 / 330 steps (`f8k-mt11s240/480-dense-lrg`).
  * Online routers: `f8k-mt4s1464-otr10M8-lro-lr2e-06` (keep 10%, drop before block 8) and `f8k-mt11s480-otr20M8-lro`.
* **Compute.** Lambda A100 (`run_probe_lambda.sbatch`), about 3 s per 8k row. Beaker jupiter was backlogged, so the
  first wave of Beaker jobs was cancelled before it started.

## Findings for router design

Numbers are row means. Pooled = the 4 mt4 tasks on mt4 rows unless marked mt11.

1. **Where to split: before block 7-8, and documents can leave by about block 24.** The document-token share of the
   hidden-state gradient energy is flat through block ~8, then falls.

   | checkpoint (mt4 rows, pooled) | b0 | b7 | b12 | b16 | b24 | b28 |
   |---|---|---|---|---|---|---|
   | base | 0.21 | 0.15 | 0.10 | 0.10 | 0.06 | 0.00 |
   | dense 366 | 0.55 | 0.56 | 0.41 | 0.25 | 0.11 | 0.11 |
   | router otr10M8 366 | 0.53 | 0.43 | 0.34 | 0.25 | 0.16 | 0.03 |
   | mt11 dense 330 (mt11 rows, 11 tasks) | 0.57 | 0.56 | 0.41 | 0.30 | 0.14 | 0.04 |

   Fine-tuning *raises* the document share in the early-mid blocks: the query and answer tokens' gradients saturate
   first. Split fidelity with a per-document 10% keep ranked by the true gradient at L (dense 366, pooled):

   | L | retained energy | random keep |
   |---|---|---|
   | 1 | 0.91 | |
   | 4 | 0.93 | |
   | 7 | 0.96 | 0.85 |
   | 12 | 0.99 | |

   There is no step at 7. The router's depth threshold at 7 is therefore more likely about whether h_L can *predict*
   importance than about where the gradient lives.
2. **One decision is not enough after training; rescore once at about block 16.** Spearman of body-token G between
   blocks drops with training:

   | blocks | base | dense 366 |
   |---|---|---|
   | 3 vs 7 | 0.88 | 0.88 |
   | 3 vs 16 | 0.71 | 0.52 |
   | 7 vs 16 | 0.78 | 0.60 |
   | 7 vs 24 | 0.57 | 0.35 |
   | 16 vs 24 | 0.71 | 0.67 |

   Shares of a later block's G² energy held by a per-document keep:

   | keep | target | base | dense 366 |
   |---|---|---|---|
   | top 10% by G7 | G16² | 0.83 | 0.74 |
   | top 50% by G3 | G16² | 0.95 | 0.92 |
   | nested 50%@3 → 10%@7 → 3%@16 | G24² | 0.81 | 0.69 |
   | one-shot top 3% by G16 | G24² | 0.84 | 0.73 |
   | one-shot top 3% by G7 | G24² | 0.75 | 0.61 |
   | random 3% | G24² | 0.03 | 0.03 |

   The final-block gradient is anti-correlated with G7 (−0.19 trained). oolong is the weak task: the nested keep holds
   0.27 of G24² after training.
3. **Gradient vs activation: they disagree everywhere, and the activation target loses.**
   * Spearman(G_b, ‖h_{b+1}−h_b‖) is 0.0-0.2 at every block. Spearman(G7, ‖h7−h0‖) runs −0.37 to +0.15 by task, most
     negative on strmatch, reorder and xabsence.
   * Forward ablation ΔCE, keep 10% per document before block 7:

     | ranking | base | dense 366 | router 366 |
     |---|---|---|---|
     | gradient G7 | +0.13 | +0.30 | +0.02 |
     | activation ‖h7−h0‖ | +0.47 | +0.36 | +0.42 |
     | late answer attention | +0.14 | +0.49 | +0.03 |
     | random | +0.35 | +0.52 | +0.39 |
     | first/last | +0.25 | +0.51 | +0.14 |

   * On nq at the base, the activation keep (+1.03) is worse than random (+0.55). This matches the training-time result
     (dh target ≈ random).
   * Answer-attention mass agrees moderately with the gradient (ρ 0.45-0.65). It is as good as the gradient on the base
     and worse after training (strmatch, mt11 dense 330: +0.91 vs +0.53).
4. **Important-but-learned tokens (Prasann's question): not the failure mode.** After training, the gradient keep
   misses more of the late-attention top 10%: 0.37 → 0.56 of it. But the missed tokens sit at the 64th-82nd percentile
   of G7, not near zero, and keeping by attention instead is worse in the ablation (table above). The risk is the depth
   instability in finding 2, not tokens dropped for being already learned.
5. **Early-block parameter-gradient share** (for `--st-otr-early-nograd`). Blocks 0-6 hold this share of ‖g‖² (blocks +
   LM head, embedding excluded), as a mean of per-row gradients; in brackets, the summed minibatch gradient:

   | checkpoint | blocks 0-6 | block 0 alone |
   |---|---|---|
   | base | 0.45 (0.32-0.53 by task) | 0.17 |
   | mt4 dense 366 | 0.20 (0.29-0.34) | 0.06 |
   | mt11 dense 330 | 0.33 | |
   | mt4 router 366 | 0.32 | |

   Training blocks 0-6 only on full-context steps drops a fifth to a third of the gradient energy, and nearly half early
   in training. The LM head's share grows with training: 0.02 → 0.13 (oolong 0.30).
6. **Why strmatch resists: it is incompressible at 10% per document.**
   * Its rows have 82 documents of ~25 body tokens, so a per-document 10% keep leaves ~3 tokens per document, mostly the
     8-token header.
   * The gradient is spread across all documents rather than gold: gold enrichment is 2.2× at the base and 0.9× trained.
     Each document's top 10% captures 0.68 of block 7's energy, against 0.92 for a row-level top 10% (mt11 dense 330).
   * Its rankings are the least stable with depth: G7~G24 0.26 at the base, 0.38 trained.
   * After 330 mt11 steps even the oracle gradient keep costs ΔCE +0.53, at a full CE of 0.009 (random +0.95,
     ⚠ eval_size = 23). Contradiction (92 documents of ~42 tokens) and xabsence (81 of ~38) behave the same way:
     +0.53 and +0.50.
   * Short documents need a per-document token floor or row-level allocation.
7. **grouping is not explained by gradient composition.** At the oracle level it compresses easily (mt11 dense 330):
   ablation ΔCE −0.00 for the gradient keep and +0.15 for random, split fidelity 0.98. Two things are distinctive: every
   document is gold, and document tokens keep their gradient deep in the stack (0.27 at block 16, 0.14 at 24). The
   training-time loss more likely comes from the learned scorer on all-gold rows, or from the kept tokens' gradients
   being redistributed (the grad-sim finding). Next test: `grad_fidelity.py` on grouping rows with the trained router.
8. **Merge vs drop.** What a 10% keep at block 7 drops is 0.16 (base) to 0.29 (trained) of G7², and it is spread out:
   the top 10% of the dropped tokens hold only 0.26-0.42 of the dropped mass (effective number 0.5-0.7× the count).
   Importance is smooth along the document, though (lag-1 correlation of log G7 between neighbours 0.5-0.85). That
   favours pooling runs of neighbouring dropped tokens over rescuing a few singletons.
9. **The router-trained checkpoint has adapted to its own drop.** Against dense at the same CE (0.118 vs 0.112, mt4
   rows):
   * its forward barely notices a 10% keep at block 7: +0.02 vs +0.30;
   * its gradient stays earlier-heavy and more stable: blocks 0-6 0.32 vs 0.20; the block-7 keep holds 0.84 vs 0.74 of
     G16²; the nested pyramid holds 0.78 vs 0.69.
   * Confound: the router used LR 2e-6, dense 5e-6. The LR 2e-6 dense control is queued.

## Limits

* All rankings are full-context oracles. A router scores from early hidden states and, in a pyramid, on an
  already-dropped context.
* Split fidelity counts gradient energy, not direction.
* The ablation is forward-only at a fixed checkpoint, not a training run.
* ⚠ eval_size is 64 (mt4 rows) or 19-23 (mt11 rows) per task. Share differences under about ±0.03 (mt4) or ±0.05
  (mt11) are noise.

## Files

* Code: branch `claude/grad-composition-6f09c7`, `research/token_routing/grad_composition/`:
  * `probe.py`, `run_probe_lambda.sbatch`, `launch.py` (Beaker);
  * `quick.py` (the router numbers above), `art_data.py`, `build_artifact.py`, `analyze.py`.
* Per-row statistics: `results/<label>/probe.json` + `probe_stats.npz` (gitignored; also on Lambda
  `/accounts/projects/sewonm/prasann/ctc_suite/gradcomp_out/`).
* Full float16 per-token arrays: Lambda node02 `/tmp/prasann/gradcomp/dumps/<label>/`, about 5 MB per row.
