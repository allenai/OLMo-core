---
title: Online train-time token router (and mid-stack token exit) refit every step from the training backward
date: 2026-10-22
updated: 2026-10-22
experiment: learned_router
area: token_routing
kind: plan
status: active (mt4 ablations evaluated: drop layer M1-M12, router target, exploration, de16; per-task runs in progress)
superseded_by: 
artifact: https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1
summary: Online train-time keep router (ridge refit each step on gradient norms) + mid-stack drop; on fresh data the mid-stack router beats dense-s976 on all 4 tasks at 0.61-0.80x its FLOPs (single seed); drop at block 8 (1 and 4 fail), gradient-norm target (activation-strength dh = random), exploration share 0-50% irrelevant.
---
# Online train-time token router

**Goal (Prasann).** A train-time token-drop ALGORITHM, optionally reading a per-document 0/1 gold flag, that is
Pareto-better than dense at matched training FLOPs on all CTC tasks. The best fixed rule so far, `tier67a50m32`
(tokdrop_lr screen), only ties dense at 0.57x FLOPs. Every learned router before this one
(`records/learned-token-router.md`) scored FROZEN checkpoints; this one decides during fine-tuning and learns from
the fine-tuning itself.

**Code.** `src/olmo_core/nn/online_token_router.py` (router), `src/olmo_core/train/callbacks/online_router.py`
(per-step refit + metrics), hooks in `Transformer` (`keep_token_rule="online"`, three small insertions in
`model.py`), FLOP-meter overhead term (`flop_meter.py`), trainer flags `--st-keep-token-rule online --st-otr-*`
(`SoftTokenOptions` fields; `recipes/train/variants_options_test.py` guards them). Tests:
`src/test/nn/online_token_router_test.py`: CPU on horton; GPU on a Beaker H100 (01M5HA8S24PNN2VTZCFFSFMM8C:
79 passed, 0 skipped, including the GDN-hybrid input-prefix and mid-stack modes). Offline probe and launcher:
`research/token_routing/online_router/`.

## Objective (Prasann, 2026-10-22, relayed by the parent session)

"I want the router to be train-time only by the way, and the grad-norm dependence is important as the goal should
be for the truncated forward/backward to be as similar as possible to the un-truncated forward/backward."

* **Train-time only.** The router acts only inside training forwards (`model.training` and soft-token compaction).
  Eval and inference run the full context through every layer.
* **The objective is gradient fidelity.** The truncated pass should reproduce the full pass: answer loss, and above
  all the parameter gradient. The per-token gradient norm is the central target.
* **Measured directly.**
  * Offline: `grad_fidelity.py` gives the cosine, relative error and norm ratio between the truncated and full
    parameter gradients on held-out rows, per rule × budget × drop point.
  * Online: `online_router/fid_mass_*` gives the share of the full-pass squared gradient mass the router's top-k
    keeps.
* **Cost of the full-pass measurement.** It is AMORTISED and CHARGED. Router arms take a full-context step every 8
  steps (`--st-otr-dense-every 8`, the `launch_otr.py` default). These are ordinary dense training steps, so the
  FLOP meter charges them at the full cost. They give unbiased full-pass targets for every token and the
  `fid_mass_*` estimate. On the other steps the targets come from the truncated pass itself: cheap, but measured
  in the compacted context (agreement with the full-pass ranking 0.40–0.66, below).

## Design

**What is routed.** Document body tokens only. Markers, system prompt, query and answer always stay. Dropped
tokens vanish from the row (slot-less compaction, original RoPE positions): the same model path as the
`gold_fl20p8_noslot` / `tier67*` rules, so the comparison with them is like for like.

**Scorer.** Linear, `score_t = w · x_t`, with features

* document structure (58): body offset from start and from end as one-hots 0-15 and log2 buckets, relative
  position, document index, log length, bias;
* lexical (2): the token id occurs in the query; log repeat count in the row;
* hidden state `h` at layer L, RMS-normalised (2,560), and the query interaction `h * q` (2,560) with
  `q` = the mean normalised `h` over the query tokens, plus `cos(h, q)`. With `query_position=after`, document
  tokens never see the query causally, so `h * q` is the only query-aware hidden feature;
* optional gold flag (1), from the shard's gold-fingerprint sidecar (`--st-otr-use-gold`).

Scoring never builds the (T, F) feature matrix: `score = s · w_s + h · (w_h + q * w_hq)`.

**Targets, measured by the training step itself.** For a sample of the kept tokens (all exploration tokens
first), a tensor hook on the compacted embedding output reads `log ||dL/de_t||` (`gnorm`, default) or
`log |<dL/de_t, e_t>|` (`gxh`). Targets are centred per row: only the within-row ranking is fit.

**Fit.** Ridge regression on decayed sufficient statistics (`X^T X`, `X^T y`, decay 0.9 per step), refit before
every optimizer step, after one all-reduce, so every rank holds the same weights. The ridge strength is relative
to the mean Gram diagonal. Default λ = 0.1 (the probe preferred 0.01 at 1M samples; online there are ~30k
samples per step).

**No heuristic init.** Step 1 trains uncompressed (keep-all fallback) and only collects targets over every
token. From step 2 the router routes. Exploration: `explore_frac` (default 10%) of the kept budget is drawn
uniformly at random. These tokens correct the selection bias, since targets exist only for kept tokens, and the
logged held-out rank correlation is measured on them. `--st-otr-dense-every N` adds periodic full-context steps.

**Budget and the gold shortcut.** `keep_frac` of each row's body tokens. The gradient targets are concentrated
on gold documents: on the base model, gold-document AUC of `gnorm` is 0.83-0.94 (below). The parallel
grad-sim analysis (`gradprobe/predict_report_2026-10-21.md`) found that the gradient keep rule has count-leak AUC
≈ 0.95. A router that faithfully imitates gradients would therefore keep gold documents densely and teach the
"dense = gold" shortcut that collapses at full-context eval. Hence `doc_share` (default 1.0): that share of the
budget is spread over documents in proportion to length (each keeps its own top `ceil(doc_share · keep_frac ·
|body|)` tokens by score). The router then decides WHICH tokens of each document stay, not WHICH documents.
Count-leak AUC is 0.5 by construction, up to exploration and rounding. Lower `doc_share` lets the scorer move
budget across documents. `online_router/leak_auc` is logged every routed step whenever the shard has a gold sidecar (it is a diagnostic only; gold is a feature only with `--st-otr-use-gold`).

**Two drop points (the layer-skip half).**

* `--st-otr-drop-layer 0` (input drop): dropped tokens skip every block. Hidden features need a no-grad prefix
  pass through blocks `[0, L)` on the full row (`--st-otr-feature-layer L`). Its cost is
  `(1/3) Σ_{b<L} flops_b(T) · B · T`, about 4% of a dense step at L = 4, and it is charged.
* `--st-otr-drop-layer L` (mid-stack drop): every token runs blocks `[0, L)` on the full row, trained with full
  context. The router then scores the forward's own `h^L` (free, no prefix pass), gathers the kept tokens, and
  the dropped tokens skip blocks `[L, 32)`. This is a two-depth per-(token, layer) skip. Early layers still see
  the whole context: dropped tokens contribute to the kept tokens' representations (and GDN states) below L.
  That keeps the training distribution closer to eval than an input drop at the same downstream budget. The
  cost is `L·Σ_{b<L} f_b(L) + L'·(Σ_{b≥L} f_b(L') + head)` per row, charged as a signed adjustment.
  Composition with the sibling session's whole-block layer drop (`layer_drop.py`) is not wired. Don't enable
  both together.

**FLOP accounting.** The model's compaction counters already give the per-example compacted lengths. Router
overhead goes into `model._flop_extra_actual` and is all-reduced and added to `flop_meter/actual_pflops` by
`FlopMeterCallback`, also logged cumulatively as `flop_meter/overhead_pflops`. That overhead covers the prefix
pass, scoring (2F per body token), the target hook (2D per sampled token), the ridge statistics (2F²N), the
solve (F³/3 + 4F²) and the mid-stack adjustment. The x-axis is therefore honest.

**Diagnostics** (`online_router/*` in wandb and the job log): `fid_mass_router` / `fid_mass_oracle` /
`fid_mass_random` (full-context steps only: share of full-pass squared gradient mass kept by the router's row-level
top-k vs the true top-k vs random), `keep_frac`, `dense`,
`rank_corr_explore` / `rank_corr_kept` (Spearman of the pre-update scores with the measured targets), `leak_auc`,
`fit_samples`, weight norms, `w_in_query`, `w_gold`, `mid_row_frac`.

**Not checkpointed.** The router's statistics and weights live outside the state dict. A resumed run restarts
the router, warmup step included.

## Offline probe (step 2 of the brief): can early-layer features predict the gradient? Yes, partly.

`research/token_routing/online_router/offline_probe.py`, Beaker job 01M5H9AW60673FJG0AXDGC6XMP, commit 5cd8b82.

* **Setup.** Model: the repaired Qwen3.5-4B base (`q35-4b-base-markerfix`, the training start point). Rows: the
  mt4s488 training shard, 48 fit rows per task, then the next 32 per task held out (⚠ eval_size = 32 rows per
  task; per-row metrics averaged). One full-context forward + backward of the answer CE (no-cot, teacher forced)
  per row.
* **Fit.** Ridge on body tokens, pooled over the four tasks. Per-task fits gain only +0.0-0.1 Spearman, so one
  shared router suffices. λ = 0.01 was the best of {0.01, 0.1, 1}.
* **Metrics.** Held-out Spearman / top-10% recall / gradient-mass captured by the predicted top 30%
  (`Σ exp(2·target)`, i.e. squared-gradient-norm share).
* **Baselines.** Random: recall 0.10, mass@30 ≈ 0.30. Oracle mass@30: 0.86-0.95.

Target `gnorm@emb` (what the online router fits), Spearman / recall@10 / mass@30:

| features | contradiction | nq | outlier | oolong |
|---|---|---|---|---|
| structure only | +0.30 / 0.30 / 0.71 | +0.25 / 0.38 / 0.82 | +0.26 / 0.44 / 0.74 | +0.16 / 0.19 / 0.49 |
| + lexical | +0.41 / 0.29 / 0.71 | +0.46 / 0.37 / 0.83 | +0.44 / 0.45 / 0.73 | +0.43 / 0.20 / 0.45 |
| + lexical + gold | +0.46 / 0.42 / 0.81 | +0.49 / 0.42 / 0.85 | +0.51 / 0.53 / 0.80 | +0.43 / 0.20 / 0.45 |
| + embedding h⁰ | +0.44 / 0.30 / 0.62 | +0.58 / 0.40 / 0.85 | +0.55 / 0.41 / 0.78 | +0.53 / 0.28 / 0.50 |
| + h⁴, h⁴·q | +0.51 / 0.34 / 0.69 | +0.57 / 0.42 / 0.85 | +0.61 / 0.50 / 0.79 | +0.57 / 0.32 / 0.52 |
| + h⁸, h⁸·q | +0.54 / 0.36 / 0.71 | +0.59 / 0.44 / 0.86 | +0.63 / 0.54 / 0.81 | +0.60 / 0.34 / 0.58 |
| + h⁸, h⁸·q + gold | same as above (±0.01) | | | |

Target `gnorm@L8` (the mid-stack target) with h⁸ + h⁸·q features: Spearman +0.65 / +0.66 / +0.66 / +0.67, mass@30
0.80 / 0.95 / 0.91 / 0.73 (oracle 0.97 / 0.99 / 0.97 / 0.90).

What it says:

1. **Gradient mass is predictable from cheap features.** The top 30% of predicted tokens carry 70-95% of the
   squared gradient norm (random: 30%). Layer-8 features beat layer 4, which beats the embedding, which beats
   structure. The mid-stack drop at L = 8 gets the best features and the most predictable target for free.
2. **The gold flag is redundant once hidden features are in** (≤ 0.01). It helps structure-only scorers (+0.05
   to +0.07). Hidden features recover most of what the flag says, which is consistent with the gradient
   targets' gold AUC (`gnorm@emb` 0.83-0.88, `gnorm@L8` 0.90-0.94): the gradient itself points at gold.
3. **`gxh` is less predictable than `gnorm`** (Spearman 0.3-0.4) but more concentrated (oracle mass@30 0.96-0.99).
   `gnorm` stays the default.
4. **Online targets track full-context ones.** On a random-30% compacted row, `gnorm` of the kept tokens has
   Spearman +0.40 (contradiction), +0.60 (nq), +0.57 (outlier) and +0.66 (oolong) with their full-context
   `gnorm`. That is a usable but noisy signal, and why exploration and per-row centring matter.
5. **Caveat.** This is the BASE model, where CE is unsaturated (CE 0.8-2.7). Mid-training, rows saturate and
   their targets turn to noise (gradprobe found |grad| at g = 1 is noise on near-certain checkpoints). The
   online `rank_corr_explore` metric is the live check.

Recorded with `ops result add --check` (metric prefix `otrprobe_`, run `otr-probe-base-mt4s488`).

## Gradient fidelity of truncated passes (offline, parameter gradient)

`research/token_routing/online_router/grad_fidelity.py`, Beaker 01M5HBH3XHQ45E55RA7XZ3N77X.

* **Setup.** Base markerfix on mt4s488. Held-out rows 80-91 of each task (⚠ eval_size=12 per task), disjoint from
  every fit row. One full pass per row gives `g_full`. Then one truncated pass per rule × budget {0.1, 0.3, 0.5} ×
  drop point {input, mid8}.
* **Metrics.** `cos(g_drop, g_full)` over all parameters and over the blocks only,
  `||g_drop − g_full|| / ||g_full||`, the norm ratio, and ΔCE. The reference is the cosine between two unrelated
  rows' full gradients.
* **Code and data.** Commit 834c93eef; JSON `research/token_routing/online_router/results/gradfid_base/`; ops results
  `gradfid_*`.

Pooled over the four tasks, median over held-out rows (⚠ eval_size = 48 rows, 12 per task). Cells are
cos(g_drop, g_full) / ‖g_drop − g_full‖ / ‖g_full‖ / ΔCE. **Two unrelated rows' full gradients already have
cos 0.40** (nq 0.67, contradiction 0.28), so on the base model raw cosine is inflated by a shared row-independent
direction. Compare against 0.40, not 0. The gold-aware bar rules are not in this table.

| drop point · rule | ρ = 0.1 | ρ = 0.3 | ρ = 0.5 |
|---|---|---|---|
| input · router, per-document share | 0.42 / 1.02 / +0.37 | 0.54 / 0.95 / +0.07 | 0.73 / 0.74 / −0.01 |
| input · router, row top-k | 0.41 / 1.02 / +0.42 | 0.57 / 0.88 / +0.07 | 0.73 / 0.72 / +0.01 |
| input · oracle, per-document | 0.47 / 1.02 / +0.19 | 0.60 / 0.89 / +0.09 | 0.75 / 0.77 / +0.03 |
| input · oracle, row top-k | 0.45 / 0.99 / −0.04 | 0.71 / 0.76 / −0.02 | 0.83 / 0.58 / −0.01 |
| input · random, per-document | 0.39 / 1.04 / +0.47 | 0.46 / 0.97 / +0.40 | 0.51 / 0.92 / +0.32 |
| input · first/last per document | 0.43 / 0.99 / +0.22 | 0.59 / 0.86 / +0.09 | 0.73 / 0.73 / +0.05 |
| **mid8 · router, per-document** | **0.74 / 0.73 / +0.13** | **0.88 / 0.48 / +0.02** | **0.96 / 0.30 / +0.01** |
| mid8 · oracle, per-document | 0.78 / 0.67 / +0.05 | 0.91 / 0.41 / +0.01 | 0.96 / 0.28 / −0.01 |
| mid8 · random, per-document | 0.60 / 0.86 / +0.40 | 0.66 / 0.81 / +0.25 | 0.72 / 0.77 / +0.13 |

Per task at ρ = 0.3, cos for router vs random vs oracle (per-document):

| | contradiction | nq | outlier | oolong |
|---|---|---|---|---|
| input | 0.29 / 0.26 / 0.25 | 0.73 / 0.47 / 0.77 | 0.45 / 0.44 / 0.50 | 0.77 / 0.76 / 0.81 |
| mid8 | 0.59 / 0.39 / 0.67 | 0.97 / 0.69 / 0.97 | 0.85 / 0.61 / 0.87 | 0.98 / 0.97 / 0.98 |

What it says:

1. **The mid-stack drop is the big lever for gradient fidelity.** Every token runs blocks 0–7 and dropped tokens
   skip 8–31. At ρ = 0.3 the router keeps cos 0.88 with relative error 0.48, against 0.54 / 0.95 for the same
   router dropping at the input. Mid8 at ρ = 0.1 (0.74 / 0.73) already matches the input drop at ρ = 0.5
   (0.73 / 0.74), and it is cheaper.
   * Approximate forward+backward cost is 8/32 + 24/32 × (kept share of the row).
   * Mid8 at ρ = 0.1 keeps ≈ 15–20% of the row (markers, prompt and answer always stay), so it costs ≈ 0.36–0.40 of
     dense. The input drop at ρ = 0.5 costs ≈ 0.55.
   * The trained runs' `actual_pflops` will give the real numbers.
   * This is consistent with the gradient-similarity record: the gradient arriving at kept tokens is redistributed
     when context is removed early. Keeping the early layers full keeps that signal.
2. **Under mid8 the router is near its ceiling.** Router vs oracle with the same per-document allocation: 0.88 vs
   0.91 at ρ = 0.3, 0.74 vs 0.78 at ρ = 0.1. Random is far behind (0.66, 0.60). The router's value is clearest on
   nq and outlier (+0.28 and +0.24 cos over random at ρ = 0.3) and on contradiction (+0.20).
3. **At the input, the router beats random but not the first/last heuristic** (0.54 vs 0.59 at ρ = 0.3, a tie at
   0.5). Even the oracle gains little over first/last there with per-document allocation (0.60). The large headroom
   at the input is in row-level allocation (oracle row 0.71), which is exactly the gold-concentrating, leaky
   direction. For input drops the per-document constraint costs fidelity. Mid-stack recovers most of it without
   moving budget across documents.
4. **ΔCE tracks fidelity, roughly.** Router and first/last are near-lossless in CE at ρ ≥ 0.3; random is not.
5. **Caveats.**
   * This is the BASE model: the first steps of training, not mid-training (a trained-checkpoint rerun is next).
   * Raw cosine includes the shared direction (unrelated rows 0.40), and centred cosine was not computed.
   * Only 12 rows per task.

### The same probe on a TRAINED checkpoint

* **Run.** Beaker 01M5HQKKR1T184608W6SQ0JY5M, commit 908eb0957, ops results `gradfid_trained_*`.
* **Checkpoint.** `f8k-mt4-otr30L4-s976-tdb-lr2e-06` (244 steps of online-router training on mt4s976).
* **Rows.** Unseen rows of mt4s1464 (fit rows 1000-1031, held-out rows 1080-1091 per task; ⚠ eval_size=12 per task,
  48 pooled). Median full CE is 0.17.
* **The parent's dense checkpoints are unusable.** Its tdb runs crashed in `wandb.finish` before saving any
  checkpoint; their run dirs hold only `flops.json`, `provenance.json`, `run_identity.json` and `wandb/`.

**Unrelated rows now have cos 0.02** (0.40 on the base). The shared direction is gone, so these cosines measure
the row-specific direction directly. Pooled cos / rel err:

| drop point · rule | ρ = 0.1 | ρ = 0.3 | ρ = 0.5 |
|---|---|---|---|
| input · router, per-document | 0.09 / 1.49 | 0.45 / 1.07 | 0.60 / 0.98 |
| input · oracle, per-document | 0.12 / 1.53 | 0.50 / 1.12 | 0.69 / 0.89 |
| input · oracle, row top-k | 0.32 / 1.52 | 0.72 / 0.88 | 0.86 / 0.62 |
| input · random, per-document | 0.07 / 1.44 | 0.06 / 2.46 | 0.06 / 3.48 |
| input · first/last per document | 0.06 / 1.81 | 0.28 / 1.65 | 0.53 / 1.10 |
| **mid8 · router, per-document** | 0.22 / 1.93 | **0.78 / 0.80** | **0.89 / 0.53** |
| mid8 · oracle, per-document | 0.42 / 1.41 | 0.83 / 0.62 | 0.93 / 0.40 |
| mid8 · random, per-document | 0.06 / 2.88 | 0.10 / 2.58 | 0.28 / 1.79 |

Per task at ρ = 0.3 (cos), router / random / oracle, per-document:

| | contradiction | nq | outlier | oolong |
|---|---|---|---|---|
| input | 0.06 / 0.03 / – | 0.40 / 0.03 / – | 0.57 / 0.06 / – | 0.75 / 0.67 / – |
| mid8 | 0.13 / 0.06 / 0.32 | 0.62 / 0.15 / 0.74 | 0.83 / 0.07 / 0.84 | 0.88 / 0.84 / 0.93 |

What changes once the model is trained:

1. **WHICH tokens now matter as much as WHERE.** Random selection loses the gradient direction almost entirely
   (cos 0.06-0.10 at ρ = 0.3, relative error 2.5: the dropped-context gradient is larger than the full one and
   points elsewhere). The router keeps 0.45 at the input and 0.78 mid-stack, close to the oracle (0.83). On the
   base model, random mid-stack still kept 0.66.
2. **The router beats the first/last heuristic once trained** (input, ρ = 0.3: 0.45 vs 0.28; ρ = 0.5: 0.60 vs 0.53).
3. **Mid-stack is still the bigger lever**: 0.78 vs 0.45 at ρ = 0.3. But ρ = 0.1 is too aggressive on a trained
   model (0.22).
4. **Contradiction is the weak task.** Even the mid-stack oracle reaches only 0.32, so what contradiction's
   gradient needs is spread over many tokens.
5. **For the training runs:** early training looks like the base-model table (where random mid-stack is fine), and
   later training looks like this one (where only the router keeps the signal). A random or fixed rule that ties
   the router on short budgets should fall behind on longer ones. That is a prediction for the s1464 runs.

## Smoke training (step 5)

`f8k-mt4-otr30L4-s488-otr-lr2e-06` and companions (Beaker ids below).

* **Recipe.** mt4s488, 122 steps, 16 rows/step, 2 GPUs, LR 2e-6 (the rule arms' LR), input drop.
* **Router.** keep 0.3, doc_share 1, explore 0.1, layer-4 prefix features, ridge 0.1, 2,048 samples per row, a
  full-context step every 8 steps.
* **Runs.** Arms otr30L4 01M5HKH25DRTFX95WZE0E84AFW, rnd30 control 01M5HBRN5SC8E4DVBW848BM5C0, otr30M8
  01M5HBD2K8W5772DDZN4RY1H1J. Earlier launches were cancelled while still pending (stale commit; no full-context steps; then the
  wandb-shutdown crash fixed in 0b5b21ace, which killed the parent's tdb runs after their last step). The launch on
  0b5b21ace died in the trainer's argparse (`--st-keep-token-rule online needs --st-keep-token-k > 0`), fixed in
  c5f8106dc together with `check_variant_wiring.py`.
* **Fresh-data setting** (Prasann, 2026-10-22: at matched FLOPs a drop method gets MORE steps on FRESH rows, same
  16-row batch). Shards mt4s976 (3,904 rows, 244 steps) and mt4s1464 (5,856 rows, 366 steps) are first-N supersets
  of mt4s488. Runs are named to parse in `collect_pareto_points.py` (`-tdb`):
  * input drop: f8k-mt4-otr30L4-s976 / -s1464-tdb-lr2e-06 (01M5HK8GC7NBXWZ3A1RT8DX85P / 01M5HK9TCMGBCF9824ND0CJYF0);
  * mid-stack: otr30M8 s976 / s1464 (01M5HKC241SVX9V8J3FVCK57M1 / 01M5HKD7HD0RHZBG6AC6RB5V9Z);
  * random control: rnd30 s976 / s1464 (01M5HKE537S3Y0Y27XHK1ZKV7M / 01M5HKF5CM1ZFBJ2SA9WC5869M).

  * where-vs-which controls (parent session's request): mid-stack random rnd30M8 (01M5HM848904Y7QG4CSQZH0Y9F /
    01M5HM965QX3VJE61BT3410R9X), mid-stack first/last fl30M8 (01M5HMA8KXE5GZ9V1JJW5X77KY / 01M5HMBKT869Z0H66PEYSMFPHF),
    and the mid-stack router at keep 0.1, otr10M8 (01M5HMCSDHQMCG2S86Z2DMV51F / 01M5HME5F5PKNCFJ6ERSRNSTMA).
    `--st-otr-score random|first_last` swaps the scorer for a fixed one, with the same budget, allocation and drop
    point.

  The s488 rnd30 / otr30M8 runs were cancelled while pending, to put the GPUs here. The comparison is against the
  tdb dense curve (s488 st31 / st61 / 122, s976, s1464) and the tdb rule arms on the same fresh shards.
* **Results** (100 of 104 mt4 scores in, 2026-10-22; only the relaunched rnd30-s1464 contradiction / nq are
  pending).
  * Δ is against dense at matched FLOPs, at 8k.
  * Dense is the best LR per cell. Only LR 5e-6 has ≥ 2 points so far: s488 at 31 / 61 / 122 steps on Lambda and
    fresh-data s976 = 244, s1464 = 366. The `lri` LR-2e-6 curve joins automatically when recorded.
  * Interpolated in log PF on our meter.
  * eval_size 500 per cell, single seed (seed-1 repeats are queued); a difference has an SE of ≈ ±0.02-0.03.
  * Full table: `results/matched_flop_table.md` (`collect_otr.py`). ops results experiment `learned_router`.

| run (fresh data) | PF (× dense-eq) | contradiction | nq | oolong | outlier |
|---|---|---|---|---|---|
| **otr30M8-s1464** mid-stack router | 619 (0.54) | **+0.032** | +0.008 | **+0.038** | **+0.225** |
| **otr10M8-s1464** mid-stack router, keep 0.1 | 471 (0.41) | **+0.043** | +0.016 | +0.021 | **+0.238** |
| rnd30M8-s1464 mid-stack random | 619 (0.54) | +0.016 | +0.018 | +0.021 | +0.148 |
| fl30M8-s1464 mid-stack first/last | 619 (0.54) | +0.022 | −0.028 | +0.035 | +0.137 |
| otr30M8-s976 mid-stack router | 417 (0.54) | +0.029 | +0.010 | −0.022 | +0.049 |
| otr10M8-s976 mid-stack router, keep 0.1 | 319 (0.41) | +0.018 | +0.016 | +0.004 | +0.083 |
| rnd30M8-s976 mid-stack random | 417 (0.54) | −0.019 | +0.016 | −0.009 | +0.053 |
| fl30M8-s976 mid-stack first/last | 417 (0.54) | +0.015 | −0.046 | −0.019 | +0.042 |
| otr30L4-s1464 input router | 516 (0.45) | −0.060 | −0.008 | +0.003 | +0.105 |
| otr30L4-s976 input router | 348 (0.45) | −0.100 | +0.004 | −0.019 | +0.046 |
| rnd30-s1464 input random | 516 (0.45) | pending | pending | +0.002 | +0.021 |
| rnd30-s976 input random | 348 (0.45) | **−0.265** | +0.002 | −0.021 | −0.023 |
| otr30L4-s488 (smoke, same rows as dense s488) | 177 (0.45) | −0.229 | −0.013 | −0.030 | −0.028 |

  **Run-vs-run, no interpolation.** Comparing with actual dense runs avoids assuming a shape for the dense curve,
  which matters on outlier, where dense jumps 0.30 / 0.43 / 0.72 at 122 / 244 / 366 steps. Contradiction / nq /
  oolong / outlier at 8k:

  | run | PF | scores |
  |---|---|---|
  | dense-s976 (Lambda, LR 5e-6) | ≈ 773 | 0.931 / 0.916 / 0.584 / 0.432 |
  | otr30M8-s1464 | 619 (0.80×) | 0.951 / 0.918 / 0.633 / 0.615 |
  | otr10M8-s1464 | 471 (0.61×) | 0.947 / 0.918 / 0.629 / 0.577 |
  | dense-s1464 | ≈ 1,160 | 0.956 / 0.936 / 0.630 / 0.719 |

  * Both mid-stack router runs match or beat dense-s976 on all four tasks with less compute. The outlier gaps
    (+0.18 / +0.15) are ≈ 5-6 SE. The others are 0-1.6 SE, single seed.
  * dense-s1464, at 1.9-2.5× the compute, still leads on nq and outlier.
  * `ops result add --check` flagged otr30M8-s1464 outlier@8k = 0.615 as > 3 SE above the experiment's previous
    best. The eval is clean (500 rows scored, none skipped), and the value sits between dense-s976 and
    dense-s1464, so it is a training effect on a steep curve, not an eval artefact.

  ⚠ **Outlier@2k is confounded by LR.** Dense at LR 5e-6 scores ≈ 0.55 there; dense at LR 2e-6 scores 0.83 at
  122 steps. The +0.2 of the drop arms there is LR, not method, so it is excluded.

  What it says:
  1. **The drop point matters most.** Mid-stack dropping (every token runs blocks 0-7, dropped tokens skip
     8-31) removes the contradiction collapse of input dropping, even with random tokens (−0.265 → −0.019 at
     s976).
  2. **With more training, the selection matters too.** At s976 the mid-stack router and random were close. At
     s1464 the router leads random on contradiction (+0.016), oolong (+0.017) and outlier (+0.077), and leads
     first/last on nq (+0.036) and outlier (+0.088). This is what the trained-checkpoint fidelity probe
     predicted: random keeps cos 0.10 of the gradient there vs the router's 0.78.
  3. **Keep 0.1 mid-stack is as good as keep 0.3 for less compute** (0.41× vs 0.54×). Most of the cost is now
     the 8 full blocks plus the full-context step every 8. A shallower drop layer or fewer dense steps would
     lower it further.
  4. The input router trails dense on contradiction at every budget. Input dropping is not the method.
  5. Single seed throughout. The seed-1 repeats of otr30M8 (s976 / s1464) and the 11-task runs are queued.

### Wall-clock (same cluster: Beaker jupiter, 2 × H100, same base / batch / seq)

`research/token_routing/online_router/wallclock.py`, JSON `results/wallclock.json`. Method: median gap between
consecutive `[step=` log lines (steps ≥ 5), and train time from the first to the last step. Dense is the parent's
tdb Beaker runs; they finished training before their shutdown crash.

| run | s/step | routed / full-context steps | train time | FLOPs × dense-eq |
|---|---|---|---|---|
| dense s488 / s976 / s1464 | 9.10 / 9.13 / 9.20 | – | 18.4 / 37.0 / 56.0 min | 1.00 |
| otr30M8 (mid-stack, keep 0.3) s976 / s1464 | 5.40 / 5.43 | 5.38 / 7.57 | 23.4 / 34.8 min | 0.54 |
| otr10M8 (mid-stack, keep 0.1) s976 / s1464 | 4.89 / 4.84 | 4.85 / 7.58 | 21.7 / 32.0 min | 0.41 |
| rnd30M8 / fl30M8 s1464 | 5.43 / 5.62 | 5.40 / 7.63 | 34.8 / 36.4 min | 0.54 |
| otr30L4 (input drop) s976 / s1464 | 4.71 / 4.74 | 4.67 / 7.92 | 21.1 / 31.4 min | 0.45 |

* **Run-vs-run.** otr30M8-s1464 beats dense-s976 on all four tasks in 34.8 vs 37.0 min (0.94× wall-clock);
  otr10M8-s1464 does it in 32.0 min (0.86×).
* **⚠ The dense arm's wall-clock includes padding.** It trains unpacked rows padded to 10,240 tokens; mean real
  rows are ≈ 7.3k. The soft-token path strips padding and sorts micro-batches by length. A full-context step inside
  a router run therefore takes 7.6 s, not 9.1. The FLOP meter charges only real tokens, so the FLOP comparison is
  fair, but against a pad-free dense step (≈ 7.6 s) the routed steps run at 0.71× (keep 0.3) and 0.64× (keep 0.1)
  of dense time, while their FLOP ratios are 0.54 and 0.41.
* **Where the gap goes.** About a third of the FLOP saving does not become time. Fixed costs per step are FSDP
  collectives, the 8 full-row blocks, the router prefix / features, and the gather. Short compacted rows also use
  GPU kernels less efficiently.
* **What can close it.** The keep-0.1 / block-4 / dense-every-16 arms (queued) cut the full-row share; a CUDA-graph
  / fixed-capacity gather (as in `layerskip/fastskip.py`) would cut the rest.

## Follow-up ablations (2026-10-22, later)

All otr10M8-style (keep 0.1, a full-context step every 8, LR 2e-6, mt4s1464 = 366 steps unless noted). Scores are
8k, eval_size 500, no-cot, recorded with `ops result add --check`. Δ = against the best-LR dense curve at the same PF
(Beaker H100, interpolated: at 471 PF contra 0.904, nq 0.902, oolong 0.608, outlier 0.339). Single seed per cell.
Lambda runs (2 × A100, node02, name `-lro-...-lam`) are compared with a same-cluster otr10M8 run (seed 1).

**1. Drop layer: block 8, not earlier.** contra / nq / oolong / outlier:

| run | where | PF | scores |
|---|---|---|---|
| otr10M12 | Lambda | 565 | 0.938 / 0.912 / 0.638 / 0.523 |
| otr10M8-s1 | Lambda | 471 | 0.928 / 0.902 / 0.664 / 0.641 |
| otr10M6 | Lambda | 422 | 0.839 / 0.904 / 0.614 / 0.404 |
| otr10M4 | Lambda | — | 0.781 / 0.884 / 0.613 / 0.372 |
| otr10M1 | Lambda | — | 0.767 / 0.882 / 0.617 / 0.322 |
| otr10M8-s976 | Beaker | 319 | 0.895 (+0.018) / 0.908 / 0.614 / 0.386 |
| otr10M4-s976 | Beaker | 256 | 0.728 (−0.130) / 0.848 / 0.578 / 0.333 |
| otr10M1-s976 | Beaker | 209 | 0.667 (−0.173) / 0.872 / 0.598 / 0.317 |

Routing after block 1 or 4 falls back to input-drop behaviour (contradiction −0.15, outlier −0.27 to −0.32 vs M8).
Full attention sits at blocks 3, 7, 11, … (3:1 GDN). M6 (blocks 0-5, one full-attention layer) is still far behind M8
(contradiction −0.09, outlier −0.24): the SECOND full-attention pass, block 7, is what matters. M12 (+20% PF) matches
M8 on contradiction / nq and is lower on oolong (−0.03) and outlier (−0.12, inside the 0.13 seed noise): later splits
do not keep improving. Against dense at 565 PF (≈ 0.914 / 0.907 / 0.599 / 0.373, interpolated) M12 is ≈ +0.02 / +0.01
/ +0.04 / +0.15; M8 at 471 PF is +0.02 / +0.00 / +0.06 / +0.30. Block 8 is the cheapest split past block 7.

**2. Router target (Prasann: "keep important tokens that are already good").** Targets: `gnorm` log ||dL/dh8||;
`gxh` log |<dL/dh8, h8>|; `dh` log ||h8 − h0||, the residual change through blocks 0-7 (activation strength, no backward,
does not vanish when a token is learned); `mix` the within-row mean of the gnorm and dh ranks. Attention mass was not
used (flash attention exposes no weights). Lambda, all at 471 PF:

| target | contra | nq | oolong | outlier |
|---|---|---|---|---|
| gnorm (seed 1) | 0.928 (+0.024) | 0.902 (+0.000) | 0.664 (+0.056) | 0.641 (+0.302) |
| gxh | 0.924 (+0.020) | 0.918 (+0.016) | 0.618 (+0.010) | 0.622 (+0.283) |
| mix | 0.913 (+0.009) | 0.928 (+0.026) | 0.630 (+0.022) | 0.413 (+0.074) |
| dh | 0.890 (−0.014) | 0.928 (+0.026) | 0.622 (+0.014) | 0.359 (+0.020) |

Gradient fidelity of the same choice on the BASE checkpoint (`grad_fidelity.py`, mid8, cos of the parameter gradient
vs the full pass, medians over ⚠ 48 held-out rows only; unrelated rows 0.40):

| keep | gnorm router / oracle | gxh router / oracle | mix router / oracle | dh router / oracle | random |
|---|---|---|---|---|---|
| 0.1 | 0.739 / 0.775 | 0.751 / 0.754 | 0.669 / 0.587 | 0.584 / 0.574 | 0.597 |
| 0.3 | 0.879 / 0.914 | 0.907 / 0.870 | 0.791 / 0.662 | 0.605 / 0.609 | 0.661 |

The same on the TRAINED otr30L4-s976 export (unseen mt4s1464 rows, ⚠ 48 rows; unrelated rows 0.02): keep 0.3 gnorm
0.776 / 0.823, gxh 0.799 / 0.740, mix 0.659 / 0.271, dh 0.176 / 0.143, random 0.099; keep 0.1 gnorm 0.224 / 0.423, gxh
0.182 / 0.286, mix 0.394 / 0.100, dh 0.098 / 0.126, random 0.057. At 0.3 the ordering holds; at 0.1 every rule loses
most of the direction on this (input-drop-trained) checkpoint and the mix router reads highest, mostly outlier (0.53
vs 0.22, 12 rows) -- against the mix training run's outlier 0.413, so treated as noise until a mid-stack-trained probe
repeats it. JSON `results/gradfid_tgtabl_{base,otr30L4s976}/`, ops `otr-gradfid-tgtabl-*`.

Spearman of the targets with gnorm over body tokens (base): gxh +0.64, mix +0.64, dh −0.165 (trained: +0.66 / +0.66 / −0.10). Fidelity and training agree:
gnorm ≈ gxh > mix > dh ≈ random. The tokens with the largest early residual change are not the ones the gradient
needs (dh is slightly anti-correlated with it), and on outlier, where the needed token is the one that does not fit,
dh throws away nearly all of the router's gain. A fit-free "top-k by dh" rule was queued (dh10M8 / dh10M8de0,
`--st-otr-score dh`) and cancelled at 33 min because its fidelity is below random. Keep gnorm.

**3. Exploration share** (otr10M8, s1464, 471 PF): 0% 0.919 / 0.898 / 0.621 / 0.595; 10% 0.947 / 0.918 / 0.629 /
0.577; 30% 0.929 / 0.906 / 0.623 / 0.644; 50% 0.934 / 0.898 / 0.630 / 0.592; 100% (random) 0.922 / 0.910 / 0.635 /
0.423. 0-50% are within noise; selection bias from exploiting the router does no harm. Random loses ≈ 0.17 on outlier.

**4. Fewer full-context steps (de16) does not pay.** otr10M8de16-s976 (287 PF) 0.877 / 0.884 / 0.601 / 0.340 is
roughly dense-matched, vs otr10M8-s976 (319 PF) +0.018 / +0.016 / +0.004 / +0.083. otr10M4de16 is below dense on
contradiction at both budgets.

**5. Seed noise.** otr30M8sd1-s976 0.917 / 0.916 / 0.619 / 0.494 vs seed 0 0.926 / 0.908 / 0.591 / 0.365: outlier
moves 0.13 between seeds at s976, contradiction / nq stay within 0.01. Single-seed outlier differences below ≈ 0.15
at s976 do not count. (s1464 repeat evaluating.)

## Long-context screen (ds64 setting) -- CANCELLED 2026-10-22

Cancelled before any started (Prasann: reproduce the per-task numbers instead, since the mixed runs give fewer
per-task datapoints). Per-task runs on the fast8k per-task pools (`ds64/fast8k/shards/<task>_g16M`, oolong `_g16M_eg`,
122 steps, query after): `f8k-<task>-otr10M8-16M-tdp-lr2e-06` for all four tasks plus otr30M8 on contradiction and
outlier, evals running. The original ds64 plan, for reference:

`research/token_routing/online_router/launch_ds64_otr.py`, state `state_ds64.json`. The parent session's reason:
* At 8k, otr10M8-s1464 recovers about half of ds64's per-task wins. ds64 wins grow with length (contradiction +0.11
  / +0.18 / +0.19 at 16k / 32k / 64k), and the mid-stack cost falls slowly with length, so the long-context
  comparison is the one that matters.

Setup:
* **Geometry.** ds64's soft-arm geometry: per-task short-heavy 2k-56k pools, UNPACKED rows at seq 65,536, 128
  rows/step, flash_2, LR 5e-6, repaired 4B base. ds64 only packs the dense arm; its soft arms train unpacked, so
  segment mode is not needed.
* **Micro-batch 1.** Mid-stack runs blocks 0-7 on the full row and every 8th step is fully dense.
* **8 GPUs, as in ds64.** A first launch at 4 GPUs was cancelled while pending: 4-GPU requests schedule slowest on jupiter.
* **512 target samples per row.**
* **Evals.** The ds64 orchestrator's 5-rung files (2k / 8k / 16k / 32k / 64k, 500 rows, query after).
* **Reference.** The ds64 dense curves in ops results (experiment `ds64`): contradiction 16M / 32M / 64M, nq 16M /
  32M / 64M / 128M.

Runs:

| run | Beaker |
|---|---|
| ds64-contradiction-otr10M8-b128f-u64M | 01M5JEFZ3YBD22MBQ9SPXWFTVS |
| ds64-contradiction-otr30M8-b128f-u64M | 01M5JEGS0S6MG26DH6J32YTCZJ |
| ds64-nq-otr10M8-b128f-u64M | 01M5JEHJGK8Q230Q5HHPXY3N4F |
| ds64-nq-otr30M8-b128f-u64M | 01M5JEJDDJD79EF03EASST2QCG |

## Proposed run matrix

All on mt4s488 (122 steps) unless noted, LR 2e-6, eval 8k + 2k × 500 per task, compared against the dense curve
(dense 5e-6 at 31 / 61 / 122 steps, tdb campaign on Beaker) at matched `actual_pflops`. Launch with
`research/token_routing/online_router/launch_otr.py --arms <arm>`.

| arm | what it isolates |
|---|---|
| `otr30L4` (smoke) | input drop, within-document routing |
| `rnd30` | same budget, uniformly random tokens per row: the value of the router itself |
| `otr30M8` | mid-stack drop at L8, dropped tokens skip 24/32 blocks: the layer-skip lever |
| `otr30L4-ds05` | half the budget row-level: does cross-document allocation help, and what does it leak? |
| `otr30L4-gold` | + gold flag, doc_share 0.5: the gold-aware variant (veto if `leak_auc` → 1) |
| `otr15L4`, `otr50L4` | budget sweep for the Pareto curve |
| `otr30L4-de4` | a full-context step every 4 (mixing; distribution match to eval) |
| best arm on mt4s976 / mt4s1464 | the "more data at matched FLOPs" setting Prasann set on 2026-10-22 |
| best arm, 3 seeds | run-to-run noise is ±0.02-0.03 f1 at 8k (tokdrop_lr), so single-seed wins don't count |
| best arm on mt11s240, then 32k rows | the 11-task screen, then length generalisation |

**Decision rule.** Promote an arm only if it beats dense at matched PF on all four tasks at 8k by more than the
seed noise, with `leak_auc` near 0.5 (or no worse than the tier67 rule) and `rank_corr_explore` > 0 throughout.
