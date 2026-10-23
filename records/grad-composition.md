---
title: Per-token x per-block gradient and activation composition of the answer loss over CTC fine-tuning (Qwen3.5-4B), for router design
date: 2026-10-23
updated: 2026-10-23
experiment: grad_composition
area: token_routing
kind: result
status: complete v2 (eval_size 64 rows/task on mt4 rows, 32-42 on mt11 rows; per-row gradient statistics, not graded evals)
superseded_by:
artifact: https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ
summary: Fine-tuning moves the answer-CE gradient onto document tokens in blocks 0-8, then it fades (0.11 by block 24); rankings decorrelate with depth and training (G7~G24 0.57 -> 0.34), so rescore once at ~16; activation targets keep no better than random; blocks 0-6 hold 0.45 -> 0.23-0.34 of the parameter gradient; strmatch needs deep document processing at >10%/doc; grouping's router deficit is not in teacher-forced CE.
---
# Gradient composition over CTC fine-tuning

**Brief (parent session, 2026-10-22).** Measure how the per-token, per-block gradient (and activation) composition of
the answer loss changes over fine-tuning on CTC tasks, to suggest how train-time token / layer routers should be set
up: where to split, one keep decision vs per-layer, gradient vs activation targets, why strmatch and grouping resist
dropping. Results page (private): https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ. Experiment state:
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
* **Rows.** Each probe row records which training shards contain it (`seen_in`, by md5).
  * *mt4 rows*: `mt4s1952` rows 1464-1527 of contradiction / nq / outlier / oolong. The mt4 shards are true prefixes, so
    these are unseen by every mt4 run up to s1464. ⚠ eval_size = 64 rows per task.
  * *mt11 rows*: `mt11s960` rows 832-959 of all 11 tasks, restricted to the rows no mt11 training shard contains.
    ⚠ eval_size = 32-42 per task, the same subset for every checkpoint.
  * **Trap:** the mt11 shards are shuffled supersets, not prefixes. "Rows past N" of a bigger mt11 shard are mostly
    training rows of the sN run: 42 of `mt11s720` rows 480-543 per task are `mt11s480` rows.
* **Checkpoints** (weka `ctc_suite/ckpts`; Beaker H100 jobs, 10-40 min each).
  * Base: `q35-4b-base-markerfix`.
  * mt4 dense, LR 5e-6: 31 / 61 / 122 / 366 steps (`f8k-mt4-dense-s488st31/st61/s488/s1464-tdc-lr5e-06`), plus LR 2e-6
    at 366.
  * mt11 dense, LR 2e-6: 165 / 330 / 495 steps (`f8k-mt11-dense-s240/480/720-tdc-lr2e-06`).
  * Online router `otr10M8` (keep 10%, drop before block 8): mt4 at 366 steps; mt11 at 330 and 495.
* **Replication.** A first pass on Lambda A100 (`run_probe_lambda.sbatch`, node-local copies) reproduced the base
  exactly (blocks 0-6 share 0.453 vs 0.453, Spearman G7~G24 0.570 vs 0.570). Its separately trained dense-366 run
  differs from the Beaker one by ≤0.03 in shares and 0.08 in ablation ΔCE: that is the run-to-run spread.

## Findings for router design

Pooled = the 4 mt4 tasks on mt4 rows unless marked mt11. Numbers are row means.

1. **Where to split: before block 7-8, and documents can leave by about block 24.** Fine-tuning moves the hidden-state
   gradient onto document tokens in the early-mid blocks (the query and answer tokens saturate first), and that share
   fades past block ~8:

   | checkpoint | doc share of G² at b7 | b16 | b24 |
   |---|---|---|---|
   | base | 0.15 | 0.10 | 0.06 |
   | mt4 dense 31 | 0.54 | 0.32 | 0.22 |
   | mt4 dense 122 | 0.55 | 0.34 | 0.23 |
   | mt4 dense 366 | 0.57 | 0.24 | 0.11 |
   | mt11 dense 495 (mt11 rows) | 0.58 | 0.39 | 0.23 |

   * Split fidelity (dense 366): a 10%-per-document keep by the true gradient at L = 7 retains 0.96 of the
     parameter-gradient energy; a random keep retains 0.85.
   * Retention rises smoothly with L, from 0.91 at L = 1 to 0.99 at L = 12 (Lambda pass). There is no step at 7. The
     router's depth threshold at 7 is about whether h_L can *predict* importance, not about where the gradient lives.
2. **One decision is not enough; rescore once at about block 16.** Spearman of body-token G between blocks 7 and 24
   falls with training: 0.57 / 0.47 / 0.43 / 0.41 / 0.34 at base / 31 / 61 / 122 / 366 mt4 steps. Blocks 3 and 7 stay at
   0.88.

   | checkpoint | top 10% by G7 holds of G16² | nested 50%@3 → 10%@7 → 3%@16 holds of G24² | one-shot 3% by G7 holds of G24² |
   |---|---|---|---|
   | base | 0.83 | 0.81 | 0.75 |
   | mt4 dense 366 | 0.74 | 0.70 | 0.63 |
   | mt11 strmatch, dense 165-495 | | 0.60-0.67 | 0.40-0.52 |

   The 50% stage at block 3 is nearly free: it holds 0.92-0.96 of every later block's energy. Random 3% holds 0.03.
3. **Gradient vs activation: they disagree, and activation keeps no better than random.** Spearman(G_b,
   ‖h_{b+1}−h_b‖) is 0.0-0.3 at every block. Forward ablation, ΔCE after keeping 10% per document before block 7:

   | ranking | base | mt4 dense 366 | mt11 dense 495 (all 11 tasks) |
   |---|---|---|---|
   | gradient G7 | +0.13 | +0.23 | +0.25 |
   | activation ‖h7−h0‖ | +0.47 | +0.50 | +0.56 |
   | late answer attention | +0.14 | +0.33 | +0.29 |
   | random | +0.35 | +0.66 | +0.54 |
   | first/last per document | +0.25 | +0.44 | +0.45 |

   Answer attention ties the gradient on the base but not after training (strmatch, mt11 495: +0.92 vs +0.67). This
   matches the training-time result that the dh target is no better than random.
4. **Important-but-learned tokens (Prasann's question): not the failure mode.** After training, the gradient keep misses
   more of the late-answer-attention top 10%: 0.37 → 0.56 (Lambda pass, mt4 dense 366). But the missed tokens sit at the
   64th-82nd percentile of G7, not near zero, and keeping by attention instead is worse (table above).
5. **Early-block parameter-gradient share** (for `--st-otr-early-nograd`). Share of ‖g‖² in blocks 0-6 (blocks + LM
   head, embedding excluded), mean of per-row gradients:

   | mix | base | after training |
   |---|---|---|
   | mt4 | 0.45 | 0.31 / 0.28 / 0.25 / 0.23 at 31 / 61 / 122 / 366 steps |
   | mt11 | 0.45 | 0.39 / 0.39 / 0.34 at 165 / 330 / 495 steps |

   * The summed minibatch gradient is 0.29-0.34 at mt4 366 (Lambda pass).
   * Block 0 alone: 0.17 at the base, 0.06 trained.
   * The LM head grows from 0.02 to 0.11.
   * Training blocks 0-6 only on full-context steps gives up a fifth to two fifths of the gradient energy, most of it
     early in training.
6. **strmatch needs deep document processing at more than 10% per document.**
   * Its rows have 82 documents of ~25 body tokens, so 10% per document is about 3 tokens each, mostly header.
   * Its documents keep the most gradient deep in the stack: at block 24, dense 0.24-0.29 (nq 0.15-0.23) and the
     router-trained model 0.39-0.45.
   * After mt11 training, even the oracle gradient keep costs ΔCE +0.59 to +0.67 (random +1.0 to +1.1, base +0.13).
     ⚠ eval_size = 37.
   * The router-trained model's teacher-forced strmatch CE stays 2-3× dense's: 0.131 vs 0.066 at 330 steps, 0.111 vs
     0.034 at 495. That matches its eval deficit.
   * Short-document contradiction behaves similarly: oracle keep +0.58 to +0.65 on mt11.
   * Fix candidates: a per-document token floor, or a larger budget for short documents.
7. **grouping's router deficit is not in the loss.**
   * Router-trained and dense checkpoints reach the same teacher-forced grouping CE: 0.339 vs 0.337 at 330 steps, 0.329
     vs 0.325 at 495.
   * Dropping is free: oracle keep ΔCE +0.00 to +0.01; the block-7 keep holds 0.94-0.98 of G16².
   * The 0.05-0.07 eval loss therefore appears only in generation (long structured answers), not in the objective the
     router trains on.
8. **Merge vs drop.** A 10% keep at block 7 drops 0.16-0.26 of G7², and it is spread out: the top 10% of the dropped
   tokens hold 0.26-0.42 of the dropped mass. Importance is smooth along a document, though (lag-1 correlation of log G7
   0.70-0.86). That favours pooling runs of neighbouring dropped tokens over rescuing a few singletons.
9. **The router changes the model, but part of the gap to dense is the LR.** Router-trained vs LR-matched dense (LR
   2e-6), at equal CE (0.115 vs 0.116, mt4 rows):
   * Its forward has adapted to its own drop: a 10% keep at block 7 costs +0.02 vs +0.20 (mt11: +0.02-0.03 vs +0.25).
   * Its gradient is more early-heavy: blocks 0-6 hold 0.31 vs 0.22.
   * Its rankings are only slightly more stable: the block-7 keep holds 0.85 vs 0.83 of G16². Dense at LR 5e-6 holds
     0.74, so most of that gap is the lower LR.

## Limits

* All rankings are full-context oracles. A router scores from early hidden states and, in a pyramid, on an
  already-dropped context.
* Split fidelity counts gradient energy, not direction. Dropping also redistributes the kept tokens' gradients, which
  the grad-sim work found to be the larger effect.
* The ablation is forward-only at a fixed checkpoint, not a training run.
* ⚠ eval_size is 64 (mt4) or 32-42 (mt11) rows per task. Share differences under about ±0.03 (mt4) or ±0.05 (mt11) are
  noise.

## Proposed next

* `grad_fidelity.py` on strmatch and grouping rows with the trained router.
* A short-document token floor in the online router (e.g. at least k tokens per document below ~50 tokens), tested on
  mt11 strmatch.
* A generation-level look at grouping outputs, router vs dense.

## Files

* Code: branch `claude/grad-composition-6f09c7`, `research/token_routing/grad_composition/`:
  * `probe.py`;
  * `launch.py` (Beaker; `GC_STATE=state_beaker.json`, `--prefix`, `--fetch`) and `run_probe_lambda.sbatch` (Lambda);
  * `quick.py`, `art_data.py`, `build_artifact.py`, `analyze.py`.
* Per-row statistics: `results/<label>/probe.json` + `probe_stats.npz` (gitignored). Labels are `bk-mt4rows-*` and
  `bk-mt11rows-*`; the Lambda pass is `mt4rows-*`, `base`, `mt11-st330`, `mt4-*`.
* Full float16 per-token arrays (Lambda pass only): Lambda node02 `/tmp/prasann/gradcomp/dumps/<label>/`.
