---
title: Online train-time token router (and mid-stack token exit) refit every step from the training backward
date: 2026-10-22
updated: 2026-10-22
experiment: learned_router
area: token_routing
kind: plan
status: active (offline probe done; smoke training running)
superseded_by: 
artifact: https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1
summary: Design + first evidence for a learned ONLINE keep router (ridge refit each step on per-token gradient norms, no heuristic init) and its mid-stack layer-skip variant; offline probe says early-layer features predict gradient mass (Spearman 0.5-0.67).
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
`src/test/nn/online_token_router_test.py`. Offline probe and launcher: `research/token_routing/online_router/`.

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
* **Results.** _pending_.

## Smoke training (step 5)

`f8k-mt4-otr30L4-s488-otr-lr2e-06` and companions (Beaker ids below).

* **Recipe.** mt4s488, 122 steps, 16 rows/step, 2 GPUs, LR 2e-6 (the rule arms' LR), input drop.
* **Router.** keep 0.3, doc_share 1, explore 0.1, layer-4 prefix features, ridge 0.1, 2,048 samples per row, a
  full-context step every 8 steps.
* **Runs.** Arms otr30L4 01M5HBAQCB9N0ZWN5KMDHM81VG, rnd30 control 01M5HBRN5SC8E4DVBW848BM5C0, otr30M8
  01M5HBD2K8W5772DDZN4RY1H1J. Two earlier launches were cancelled while still pending: one ran on a stale commit,
  the other predated the full-context steps.
* **Fresh-data setting** (Prasann, 2026-10-22: at matched FLOPs a drop method gets MORE steps on FRESH rows, same
  16-row batch). Shards mt4s976 (3,904 rows, 244 steps) and mt4s1464 (5,856 rows, 366 steps) are first-N supersets
  of mt4s488. Runs are named to parse in `collect_pareto_points.py` (`-tdb`):
  * input drop: f8k-mt4-otr30L4-s976 / -s1464-tdb-lr2e-06 (01M5HBY9A2YM6E74V9950H2APH / 01M5HC0D3HR75297V8AFTWPH55);
  * mid-stack: otr30M8 s976 / s1464 (01M5HC2NEG323SKEE60DF3XDX5 / 01M5HC4SYNDB344T66VR8GW5G6);
  * random control: rnd30 s976 / s1464 (01M5HC76FY3TVXA8XWCPATVXK0 / 01M5HC9EJ69DF6DW69W0NCM19J).

  The s488 rnd30 / otr30M8 runs were cancelled while pending, to put the GPUs here. The comparison is against the
  tdb dense curve (s488 st31 / st61 / 122, s976, s1464) and the tdb rule arms on the same fresh shards.
* **Results.** _pending_.

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
