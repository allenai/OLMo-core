---
title: Per-token x per-block gradient and activation composition of the answer loss over CTC fine-tuning (Qwen3.5-4B), for router design
date: 2026-10-23
updated: 2026-10-24
experiment: grad_composition
area: token_routing
kind: result
status: complete v2 + CPT addendum (SFT: eval_size 64 rows/task on mt4, 32-42 on mt11; CPT: 48 rows x 32k; per-row gradient statistics, not graded evals)
superseded_by:
artifact: https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ
summary: Fine-tuning moves the answer-CE gradient onto document tokens in blocks 0-8, then it fades (0.11 by block 24); rankings decorrelate with depth and training (G7~G24 0.57 -> 0.34), so rescore once at ~16; activation targets keep no better than random; blocks 0-6 hold 0.45 -> 0.23-0.34 of the parameter gradient; strmatch needs deep document processing at >10%/doc; grouping's router deficit is not in teacher-forced CE. CPT addendum (OLMo 3 7B, 32k): after even 12 dense steps importance is flat (top 10% hold 0.62 of G24^2, a 10% keep at block 7 holds 0.37 vs random 0.11), early blocks hold 0.16-0.20 of the parameter gradient, and blocks 0-12 get 0.6-0.95 of their gradient from serving as context; the CE gain is mostly local copying.
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

## Addendum (2026-10-24): continued pretraining, OLMo 3 7B at 32k

**Brief (parent session via the CPT session, Prasann).** The same analysis on long-context CPT, where token dropping
keeps losing. Five questions:

1. the per-block parameter-gradient share;
2. concentration of per-token ‖dL/dh‖;
3. router fidelity;
4. position structure, and where the long-context gain comes from;
5. how much of each token's gradient comes from its own label vs from serving as context.

**Setup** (`research/token_routing/grad_composition/cpt_probe.py`, `run_cpt_probe.sbatch`; cubbins job 4021638,
8 × H200, ~11 min per checkpoint, 77 GiB peak).
* **Rows.** `dev32k_1024` rows 0-47 (Longmino ≥2^17 buckets, not in training). ⚠ eval_size = 48 rows of 32,768 tokens
  each (1.57M targets).
* **Checkpoints.** All scored at YaRN ×4 on the full-attention layers, except stage 3 at ×8:
  * stage2-step47684, the CPT start;
  * dense Dlr20 (LR 2e-5) after 12 / 48 / 120 / 240 / 480 steps;
  * the online router `OTR10M7lr20-310`;
  * OLMo's own stage3-step11921, as a reference.
* **Passes.** One full forward + backward of the mean next-token CE per row. A second backward with every attention
  layer's keys and values detached gives each token's own-label gradient g_own. The context part is g_full − g_own:
  what the token receives for serving as keys and values to later tokens.
* **Analysis:** `cpt_analyze.py`. Own/context shares are medians over rows; everything else is a row mean.

| checkpoint | loss | blocks 0-6 param-grad share (summed) | LM head | top 10% hold of G24² | top 30% | ρ(G7,G24) | top 10% by G7 hold of G24² | random 10% | nested 50/10/3 | context share @0 / 8 / 16 / 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| base (stage 2) | 1.779 | 0.42 (0.51) | 0.11 | 0.87 | 0.96 | 0.56 | 0.79 | 0.12 | 0.76 | 0.99 / 0.94 / 0.83 / 0.79 |
| dense 12 | 1.636 | 0.20 (0.21) | 0.30 | 0.63 | 0.90 | 0.67 | 0.39 | 0.12 | 0.31 | 0.94 / 0.78 / 0.42 / 0.21 |
| dense 120 | 1.625 | 0.17 (0.14) | 0.31 | 0.62 | 0.89 | 0.66 | 0.37 | 0.11 | 0.28 | 0.94 / 0.79 / 0.42 / 0.21 |
| dense 480 | 1.620 | 0.16 (0.16) | 0.32 | 0.62 | 0.90 | 0.66 | 0.37 | 0.11 | 0.29 | 0.95 / 0.80 / 0.42 / 0.21 |
| router OTR10M7, 310 | 1.645 | 0.18 (0.42) | 0.32 | 0.63 | 0.90 | 0.69 | 0.40 | 0.11 | 0.31 | 0.96 / 0.73 / 0.42 / 0.22 |
| OLMo stage 3 (×8) | 1.759 | 0.58 (0.81) | 0.11 | 0.85 | 0.95 | 0.60 | 0.76 | 0.11 | 0.72 | 0.98 / 0.91 / 0.83 / 0.78 |

1. **The early-stack share falls as fast as in SFT, but from a lower point and with one regime change.** Blocks 0-6
   hold this share of ‖g‖²:

   | | base | dense 12 | dense 480 |
   |---|---|---|---|
   | per-row gradient | 0.42 | 0.20 | 0.16 |
   | summed gradient | 0.51 | 0.21 | 0.16 |

   * Almost all of the change happens in the first 12 steps (LR 2e-5); 12 → 480 is flat on every statistic here.
   * The step-12 change is not a slow drift. In the base, a few recurring tokens dominate the gradient: the top 1% hold
     0.74 of G24² and 0.80 of G7², and in example rows the top 8 tokens, the same few ids at many positions, hold
     0.2-0.7. Norm parameters take 0.29 of the parameter-gradient energy.
   * Dense CPT removes this within 12 steps: top 1% holds 0.24-0.26, norms 0.09-0.11. The LM head's share triples
     (0.11 → 0.30).
   * OLMo's own stage-3 model still has the base pattern (top 1% 0.69, blocks 0-6 0.58, norms 0.36).
   * Freezing blocks 0-6 after the first steps gives up ~1/6 of the gradient energy, similar to SFT (0.20-0.34).
2. **Importance is flat after CPT, so a 10% keep cannot preserve the gradient.** After 12+ dense steps, the top 10% of
   tokens hold 0.62-0.63 of G24² and 0.65-0.76 of G7². The top 30% hold 0.90. In SFT the top 10% at block 7 held
   0.90-0.94 of the body gradient.
3. **The rankings are stable across depth; the problem is the budget.** Spearman(G7, G24) is 0.66 after CPT, higher
   than in SFT (0.34). But a per-row top 10% by G7 holds only 0.37 of G24² (random 0.11), and the nested 50%@3 → 10%@7 →
   3%@16 keep holds 0.29. To hold 0.9 of a late block's gradient, a keep would need ~30% of the tokens.
4. **Position structure.**
   * After CPT, gradient energy roughly tracks token count by position: G24² share 0.19 / 0.12 / 0.23 / 0.45 in the
     0-4k / 4-8k / 8-16k / 16-32k bins, against token shares 0.125 / 0.125 / 0.25 / 0.5. The base over-weights the first
     4k (0.37).
   * Most rows are one or two long documents, so the distance to the document start mirrors position.
   * The dev-CE gain is not long-context-specific. Dense 480 gains +0.169 in the first 4k and +0.167 at 16-32k.
   * Split of the gain at positions ≥ 4k, by repeat category of the target:

     | target category | share of targets | share of the gain | per-token gain |
     |---|---|---|---|
     | near repeat (4-gram last seen ≤ 4,096 back, inside every sliding window) | 0.225 | 0.56-0.66 | +0.35 to +0.41 |
     | far-only repeat (reachable only through the 8 full-attention layers) | 0.033 | 0.06-0.07 | +0.19 to +0.28 |
     | novel | 0.741 | 0.27-0.37 | +0.03 to +0.09 |

   * Stage 3 gains almost nothing over stage 2 (+0.01 to +0.03 overall): +0.12 on far-only targets at 16-32k, and
     −0.08 to −0.18 on near repeats.
5. **Label vs context: early blocks learn from serving as context, late blocks from their own label.** After CPT, the
   context part of a token's hidden-state gradient is:

   | block | 0 | 8 | 12 | 16 | 20 | 24 | 28 |
   |---|---|---|---|---|---|---|---|
   | context share | 0.95 | 0.80 | 0.60 | 0.42 | 0.27 | 0.21 | 0.12 |

   It crosses 0.5 at about block 14. The base and stage 3 are context-dominated at every block (0.67-0.99). Tokens in
   the first 4k get more of their block-16 gradient from context (0.51) than tokens at 16-32k (0.39).
6. **What CPT can tolerate.**
   * **KV-only dropping in late blocks** (tokens keep their own label path but stop serving as keys and values from about
     block 20 on): the most compatible cut. The context share there is ≤ 0.27, and only 6-7% of the long-range CE gain
     needs reach beyond the 4k window.
   * **Early-block freezing**: loses ~1/6 of the gradient energy after the first steps; the CPT session's FRZ arms test
     it directly.
   * **Depth / token routing at 10%** (mid-stack exit at block 7, as in SFT): least compatible. Importance is flat,
     10% holds 0.37 of the downstream gradient, and dropping a token from block 7 on removes 0.6-0.8 of the gradient its
     blocks 7-12 would get as context. Any routed keep needs ≥ 30%.
   * The router-trained checkpoint (`OTR10M7lr20-310`) looks like dense on every statistic (loss 1.645 vs 1.620 for
     dense 480). Unlike the SFT router, it shows no sign of having adapted to its own drop.

**Limits.**
* ⚠ 48 rows. All rankings are full-context oracles.
* The own/context split detaches keys and values only, so a token's gradient through its own attention to itself
  counts as "own".
* The base is scored with YaRN ×4, which it was never trained with. Its sink-dominated gradient may partly reflect
  that; stage 3 at its own ×8 shows the same pattern.

**Files.**
* Per-row statistics: `results/cpt/<label>/cpt.json`, `cpt_stats.npz`, `cpt_ce.npz` (gitignored).
* Full float16 G / G_own / attention maps for 2 rows per checkpoint: cubbins `/data/prasann/grad_composition/cpt/<label>/cpt_dump_*.npz`.
* ops results: `gradcomp_cpt_*`.
