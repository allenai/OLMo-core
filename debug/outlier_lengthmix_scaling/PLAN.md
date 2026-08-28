# Outlier length-mix data-scaling laws: sparse-landmark vs full attention (Qwen3.5)

Goal: fit a predictive law — given architecture and a distribution of training examples over
context-length rungs, predict task performance (and invert: data needed for a target f1) — in the
fewest experiments. Target questions: value of 2k data for 8k performance, 4k data for 16k
performance (both ratio-1/4 transfers, deliberately).

## The law to fit

Per architecture a ∈ {full, sparse_landmark} and target rung L:

    f1_L = g_L(n_eff),   n_eff = n_L + Σ_{ℓ<L} ρ(ℓ→L) · n_ℓ

g_L saturating (fit both `f∞(1−e^{−n/τ})` and logistic-in-log-n; contradiction's rising part fit
the exponential with τ≈11M tokens). ρ estimated as the horizontal shift of mix arms along the pure
curve. Pre-registered hypothesis H2: ρ(2k→8k) ≈ ρ(4k→16k) (ratio-dependence) — if it holds, the
law compresses to one ρ(ratio) curve and generalizes to unseen rung pairs.

Secondary smooth signal: per-rung held-out CE logged in-loop (LMEvaluatorCallback), giving
(run × step) curves for fitting, not just final f1s. CE comparable within-arm only (landmark
streams contain landmark tokens).

## What recon established (changes from the naive plan)

1. **Existing outlier train data is unusable for this.** The weka `single_task_ladders_v2/outlier`
   file is (a) continuous n∈U[14,220] — not rung-binned — and (b) pre-M-axis-fix: K (category
   count) caps at ~10 while eval scales K with n (25 at the 32k rung). The fixed generator
   (`generate_wiki_outlier_data.py --majority-mode articles`, 2026-08-11) has never been retrained.
   → **Generate fresh per-rung pools** with the fixed generator, n-bands matched to the eval rungs
   (2k:n≈22/K3, 4k:n≈44/K5-7, 8k:n≈55-95/K7-13, 16k:n≈110-190/K13), `query_position=after` for new
   runs per standing decision; max_new_tokens ≥512 downstream (answer length scales with K).
2. **The old Qwen3-4B datascale sweep's owed sparse-landmark evals are low-value now** — those arms
   trained on the pre-fix data; a clean comparison needs retraining regardless. Deprioritize.
3. **Qwen3.5 + sparse_landmark SFT needs three bounded infra pieces** (none exist yet):
   a. `train_ctc_suite.py` `--variant sparselandmark`: manual mixer swap on `block["attn"]`
      (exactly as `cpt/Qwen3.5-4B-sparse-landmark-dolma3longmino.py` does; keep the elementwise
      gate) + `LandmarkPackingInstanceSource` in the data path (content_len = seq_len·63/64).
   b. Landmark-embedding repair for the Qwen3.5 base: row 248200 is an untrained embedding row —
      same class as the marker-embedding bug; seed from a real delimiter row à la
      `fix_marker_embeddings.py`, assert in-distribution norm. Both arms then start from the SAME
      repaired plain base (no CPT-base confound; the length_mix experiment validated from-base SFT).
   c. Eval: `eval_lc_native` path (native, architecture-agnostic, bs=1 for landmark) with
      `--tokenizer Qwen/Qwen3.5-4B-Base`; fix the launcher's variant labeling (sparselandmark
      currently silently matches "landmark"). Native backend for BOTH arms — sparse landmark has no
      vLLM path; never mix backends.
4. **id-digit trap not exposed** at rungs ≤16k (n≤220, ids ≤3 digits, train covers eval range).
   Do not extend to 256k+ xlong rungs without an id-range audit (train n=220 vs eval n≥1802).

## THE CHECK THREAD (user protocol 2026-08-27 — supersedes the run matrix below where they differ)

Sparse and full arms run each check IN PARALLEL; all evals on rungs 2k/4k/8k/16k/32k, native
backend, eval_size 600.

* **Check #1 (LR):** best LR for outlier @ 2k ctx, 5k examples — the lmx-* sweep (3 LRs x 2 arch).
* **Check #2 (Data mix):** compare (a) 4000x8k [arm p8k_4000], (b) 4000x8k + 4000x2k [m8k_mix],
  (c) 8000x8k [p8k_8000]. If b≈a: mixing doesn't matter. If b sits predictably between a and c:
  quantify the 2k->8k exchange rate, then run the FLOP-matched test — b (~40M tokens) vs
  p8k_5000 (~40M tokens): does a cheap-mix beat pure-8k at equal FLOPs?
* **Check #3 (Data scaling):** scale the #2-winning 8k mix by 0.25/0.5/1/2/4x; same for pure-2k
  (p2k_{1250,2500,5000,10000,20000}). Fit the saturating law per context length; locate the
  asymptote; fit the cross-length trend (asymptote + tau vs context length) to PREDICT 32k.
* **Check #4 (Length extrapolation):** run 32k-ctx runs (n=220 pool), compare against the #3
  prediction; if it holds, invert the law for 256k/1M data requirements (paper-level inference,
  no 256k runs).
* **Check #5:** all of the above for sparselandmark (runs concurrently from Check #1 on).

Pool sizes to support 4x scaling: n14 -> 24,000; n57 -> 33,000; n220 (32k rung) -> 10,000.
Pools EXTEND by appending (never regenerate): prefixes stay stable so arms remain nested subsets.

## Run matrix

Unit = examples (tokens reported alongside). 1 epoch, LR annealed to 0 per run
(never hard_stop below the epoch — the A4e trap). Seeds: 2 seeds at starred points.

Per architecture:
| stage | arms | runs |
| --- | --- | --- |
| 1a. pure-8k anchors  | n₈ ∈ {250, 1000*, 4000} | 3 |
| 1b. pure-16k anchors | n₁₆ ∈ {250, 1000*, 4000} | 3 |
| 2a. 2k→8k transfer   | n₈=500 + n₂ ∈ {1000, 4000*} | 2 |
| 2b. 4k→16k transfer  | n₁₆=500 + n₄ ∈ {1000, 4000*} | 2 |
| 3. validation mix (pre-registered prediction) | e.g. n₁₆=1000+n₄=2000+n₂=2000 | 1 |

11 + ~4 replicates ≈ 15 per arch, ≈30 total at 4B. Each run ≤1 node-hour (≤65M tokens at
~9-12k tok/s/GPU × 8); evals ~1-2 GPU-h/ckpt (600 examples × 4 rungs, native, bs=1 landmark).

**Small-scale leverage option (recommended):** run the full grid on Qwen3.5-0.8B first
(`--model-scale 08b`, base staged on weka; sparse swap identical; ~10x cheaper, minutes/run),
fit the law, then run 4B at only ~6 points/arch chosen where the 0.8B fit is most informative +
the validation mix. Also tests whether ρ transfers across model scale — a finding in itself.
4B grid shrinks from ~30 to ~14 runs.

## Stage 0.5 — LR sanity check (added 2026-08-27, user request)

Prior lineages disagree: train_ctc_suite (Qwen3.5, from base) defaults to 5e-5; the Qwen3-4B
sparse-landmark singletask-ladder runs used 2e-5 (both LinearWithWarmup, anneal to 0). Before the
fan-out, sweep LR per architecture on a cheap, well-powered probe (user's spec 2026-08-27):
**2k-context (n=14 pool), 5000 examples, 1 epoch** (~625 steps at global_batch 8 — enough steps
for LR differences to express), LR ∈ {2e-5, 5e-5, 1.2e-4} × {full, sparselandmark} = 6 runs
(these double as the smoke tests of the full train→eval chain). Judge on held-out CE (all rungs)
+ f1@2k. Bonus: each winner is also a free pure-2k g_2k(5000) curve point and a zero-mix baseline
for what 2k-only training buys at the 8k rung. The chosen LR is then FIXED per arch for the whole grid
(standard data-scaling protocol; LR×n interaction accepted as a stated limitation). Watch the
sparse arm's early loss for instability from the freshly-seeded landmark embedding row — if its
best LR still shows a rough warmup, add warmup_fraction rather than dropping LR further.

## Execution order

0. Data gen (CPU, mooney/local, overnight): per-rung pools — 2k×8k-ex, 4k×8k-ex, 8k×6k-ex,
   16k×6k-ex + held-out CE shards per rung; tokenize; stage to weka (S3→weka gantry sync).
1. Infra: train_ctc_suite sparselandmark variant (CPU tests), base repair, eval labeling fix,
   LMEvaluatorCallback wiring, compose-arms manifests + fit code (adapt fit_law.py).
2. Smoke + LR sweep (stage 0.5): 6 runs through the full train→eval chain before any fan-out (the length_mix pilot-first
   lesson: every one of its 6 chain failures would have been 10 simultaneous failures).
3. Stage-1 anchors fan-out → fit g_L. 4. Transfer arms → ρ. 5. Replicates + validation run.
6. Ingest to results-hub (self-describing provenance), writeup here.

## Standing-rule compliance
urgent priority; jupiter; wandb links surfaced at launch; loss curves into results JSON;
eval_size=600 stated with SE; no-cot labeling; `n` = corpus size only (n_ℓ here = example counts,
never eval size); native-backend caveat stated next to every absolute number.
