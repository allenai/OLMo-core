# Learned linear token router for slot-less compaction (2026-09-23)

**Question.** Can a *learned* per-token keep/drop decision replace the dev-loss grid's fixed
placement heuristics (`gold_rand20p8_noslot`, `gold_fl20p8_noslot`, `gold_first20`)? The minimal
router is **one linear layer** over [RMS-normed input-embedding row, length-agnostic position
features, gold-document flag], trained per task on the **frozen** dense 4B suite checkpoint with
REINFORCE, then scored on the grid's standard test rows.

**Answer (short).** It works where a single "keep gold, drop the rest" direction is enough, and it
beats the heuristics there:
- **nq:** parity at ×0.22 / 0.17 / 0.16 vs the heuristics' ×0.34 / 0.26 / 0.25;
- **outlier:** parity at ×0.34 / 0.16 / 0.12 vs ×0.45 / 0.30 / 0.26;
- **msmarco:** ×0.21 / 0.16 / 0.15 vs ×0.33 / 0.29 / 0.28, with mean ΔCE ≤ +0.013 against the
  heuristics' −0.006 to +0.001.

On most other tasks REINFORCE is **bistable**: each λ either keeps ≈ everything (parity, no saving)
or collapses to ×0.1–0.2 at a large ΔCE. It never lands on the heuristics' intermediate "gold + a
little of everything" solution. The pilot gate passed (9/12 pilot cells beat `gold_rand20p8_noslot`
at matched compaction, 3 ties), but across all 18 scored tasks the val-selected router beats the
random-placement heuristic on only 3 real tasks (nq, outlier, msmarco: 9 cells), plus 3 cells on the
VOID xabsence checkpoint.

Everything here is ⚠ **eval_size 16 / 16 / 8** test rows per rung (2k / 8k / 32k), with 32 train
+ 16 val rows per task. Read spreads and medians, not third decimals; a ΔCE difference below ~0.02
is inside the noise.

Code: `debug/learned_router/` (README there). Results:
- `debug/devloss_grid/results_router/<task>_<rung>.json`, grid format, with the baselines re-scored
  in the same file so deltas are paired;
- `debug/learned_router/headline.json` / `router_vs_grid.json`;
- `debug/learned_router/runs/<task>/<cfg>.json`: full per-step and per-epoch curves (train and val
  reward, ΔCE, keep_frac, T2/T, w_gold, bias, grad norm);
- `debug/learned_router/weights/<task>/<cfg>.pt`.

## 1. Setup (as run)

* **Router**: `z = b + w_pos·f_pos + w_gold·gold + w_emb·e_rms/√d`, `p = σ(z)`.
  * Every document body token is routed, gold documents included. There is no forced gold keep
    and no forced 8-token id prefix.
  * Markers and everything outside documents are always kept.
  * `f_pos` has 58 features: offset-from-start one-hot 0–15 + 12 log2 buckets, the same from the
    end, relative position in the doc, and the doc's relative index.
  * Init: all zeros (p = 0.5).
* **Drop semantics**: slot-less (Prasann). The path is `_compact_pooled_soft_tokens` with
  `keep_token_rule="custom"`, doc-level keep = none and `drop_slots=True`, at original RoPE positions.
  * **Marker finding (CPU smoke):** under the stock `mark_positions_free` rule, a document's
    markers are freed only if its *whole* body is kept. With `drop_slots`, every partially kept
    document therefore loses its markers, and an all-dropped document vanishes entirely. This is
    also what the `gold_*_noslot` baselines do to every non-gold document.
  * To keep markers as specified, I added a default-off, bit-identical-when-off flag
    `mark_positions_free(free_markers=)` ← `pst["keep_token_mask_markers"]` in
    `chunked_mask.py` / `model.py`. An all-dropped document then leaves an empty marker pair.
  * The eval-only `router_l0.2_nomark` probe measures what the markers are worth. They matter on
    contradiction: ΔCE is +0.11 / +0.06 / +0.03 without them vs 0.00 / 0.00 / −0.03 with them at
    2k / 8k / 32k. Elsewhere they are within noise.
* **Training**: REINFORCE with a leave-one-out baseline.
  * Each row visit draws K = 8 Bernoulli masks, re-drawn every epoch, and runs the batched no-grad
    compacted forwards. Batched CE was asserted equal to one-at-a-time CE, and an all-kept mask
    reproduces CE_full.
  * Reward `-(CE_mask − CE_full) − λ·keep_frac`. The CE part is estimated by REINFORCE; the
    keep_frac part uses its analytic gradient (`E[keep_frac] = mean p`), which is the same
    objective with less variance.
  * Adam: lr 0.1 on bias/position/gold, 0.01 on the embedding; 4 rows per step; up to 40 epochs
    (pilot) or 60 (sweep). Early stopping uses the **val** reward of the sampled policy.
* **Data**: 16 train + 8 val rows per rung (2k, 8k) per task, taken from HF rows 64+.
  * Disjointness is proved by hashing (query, gold-document texts, and (query, answer) for no-gold
    tasks) against the first 64 rows of every staged rung file. Rung files do reuse examples:
    every nq example appears at all three rungs.
  * The check drops 0–45 candidates per task (`split_report.json`). xabsence is short of val rows
    (4 / 3).
* **λ selection** uses val rows only: the cheapest val deterministic T2/T with val ΔCE ≤ 0.02,
  else the lowest-ΔCE λ. Test rows are never used for selection.
* **Eval**: driver schemes `router_<cfg>` (deterministic, p > 0.5) and `router_<cfg>_samp`
  (seeded Bernoulli), `sel_cost = 0`, on the grid's rows and checkpoints.
  * The sampled variants have heavy per-row tails (e.g. nq 2k l0.2_samp max +0.55); read the
    deterministic rows.
* **Infra notes**:
  * `freeze_fla_length_autotune()` is required both in the trainer and in the grid driver. Every
    mask gives a new sequence length, and without it FLA re-autotunes each time: ~20 s per step,
    and a 16-row nq@2k cell took 603 s without it, while contradiction's three rungs took
    4 min with it. It changes speed only.
  * GPUs: horton (berkeleynlp `preemptive_high_sewonm`) and sneetches. mcfuzz and cubbins were
    held by rhys_gould and excluded; balrog is under a MAINT reservation.

## 2. Headline: val-selected router vs the heuristics on the test rows

Cells: compaction T2/T, mean ΔCE / median ΔCE vs `full`. The verdict compares the router with
`gold_rand20p8_noslot` / `gold_fl20p8_noslot`:
- **beats**: parity (within max(0.02, 1 SE)) at ≤ 0.9× their compaction, or lower ΔCE at matched
  compaction;
- **ties**: parity at matched compaction;
- **costlier**: parity at more compaction;
- **tradeoff**: lower ΔCE at more compaction, meaning the heuristic breaks and the router just keeps
  more;
- **loses**: worse ΔCE.

⚠ eval_size 16 / 16 / 8.

| task | rung | n | router (val-selected λ) | gold_rand20p8_noslot | gold_fl20p8_noslot | gold_only_noslot | vs rand20p8 / fl20p8 |
|---|---|---|---|---|---|---|---|
| absence | 2k | 16 | l0.05: ×1.00 +0.000 / +0.000 | ×0.68 +1.075 / +1.083 | ×0.68 +0.890 / +0.977 | ×0.51 +0.876 / +0.786 | tradeoff / tradeoff |
| absence | 8k | 16 | l0.05: ×1.00 +0.000 / +0.000 | ×0.69 +1.223 / +1.119 | ×0.69 +1.256 / +1.176 | ×0.49 +1.252 / +1.222 | tradeoff / tradeoff |
| contradiction | 2k | 16 | l0.2: ×0.48 -0.000 / +0.000 | ×0.48 -0.000 / +0.000 | ×0.48 +0.000 / +0.000 | ×0.20 +0.155 / +0.124 | ties / ties |
| contradiction | 8k | 16 | l0.2: ×0.39 +0.000 / +0.000 | ×0.39 -0.001 / +0.000 | ×0.39 +0.000 / +0.000 | ×0.06 +0.478 / +0.459 | ties / ties |
| contradiction | 32k | 8 | l0.2: ×0.36 -0.033 / +0.000 | ×0.35 -0.033 / +0.000 | ×0.35 -0.032 / +0.000 | ×0.02 +1.173 / +1.188 | ties / ties |
| fiqa | 2k | 16 | l0.5: ×0.97 -0.001 / -0.000 | ×0.46 -0.014 / +0.001 | ×0.46 +0.013 / +0.000 | ×0.30 -0.031 / +0.002 | costlier / costlier |
| fiqa | 8k | 16 | l0.5: ×0.95 +0.014 / +0.003 | ×0.29 +0.016 / +0.000 | ×0.29 -0.100 / -0.008 | ×0.07 -0.284 / -0.114 | costlier / loses |
| fiqa | 32k | 8 | l0.5: ×0.95 -0.005 / -0.001 | ×0.25 +0.066 / +0.103 | ×0.25 +0.065 / +0.027 | ×0.02 +0.233 / -0.007 | costlier / tradeoff |
| grouping | 2k | 16 | l0.05: ×1.00 +0.000 / +0.000 | ×0.29 +0.048 / +0.045 | ×0.29 +0.021 / +0.019 | ×0.07 +0.168 / +0.162 | tradeoff / tradeoff |
| grouping | 8k | 16 | l0.05: ×1.00 +0.000 / +0.000 | ×0.26 +0.054 / +0.065 | ×0.26 -0.002 / +0.001 | ×0.04 +0.622 / +0.494 | tradeoff / costlier |
| grouping | 32k | 8 | l0.05: ×1.00 +0.000 / +0.000 | ×0.27 +0.021 / +0.027 | ×0.27 +0.012 / +0.010 | ×0.04 +0.188 / +0.190 | tradeoff / costlier |
| msmarco | 2k | 16 | l0.05: ×0.21 +0.013 / +0.001 | ×0.33 -0.006 / +0.000 | ×0.33 -0.005 / +0.000 | ×0.08 +0.008 / +0.009 | beats / beats |
| msmarco | 8k | 16 | l0.05: ×0.16 +0.004 / +0.004 | ×0.29 -0.004 / +0.000 | ×0.29 -0.002 / -0.000 | ×0.02 +0.024 / +0.019 | beats / beats |
| msmarco | 32k | 8 | l0.05: ×0.15 +0.001 / +0.001 | ×0.28 +0.001 / +0.001 | ×0.28 +0.001 / +0.001 | ×0.00 +0.149 / +0.125 | beats / beats |
| niah | 2k | 16 | l0.05: ×0.98 +0.000 / +0.000 | ×0.59 +0.000 / -0.000 | ×0.59 +0.000 / -0.000 | ×0.19 +0.648 / +0.541 | costlier / costlier |
| niah | 8k | 16 | l0.05: ×0.98 +0.000 / +0.000 | ×0.51 -0.000 / +0.000 | ×0.51 -0.000 / +0.000 | ×0.05 +0.567 / +0.484 | costlier / costlier |
| niah | 32k | 8 | l0.05: ×0.98 +0.000 / +0.000 | ×0.49 +0.000 / +0.000 | ×0.49 +0.000 / +0.000 | ×0.01 +1.025 / +0.784 | costlier / costlier |
| nq | 2k | 16 | l0.5: ×0.22 -0.001 / +0.000 | ×0.34 -0.002 / +0.000 | ×0.34 -0.002 / +0.000 | ×0.13 +0.006 / +0.003 | beats / beats |
| nq | 8k | 16 | l0.5: ×0.17 -0.033 / -0.001 | ×0.26 -0.036 / -0.002 | ×0.26 -0.036 / -0.002 | ×0.03 -0.025 / +0.005 | beats / beats |
| nq | 32k | 8 | l0.5: ×0.16 +0.005 / +0.009 | ×0.25 -0.006 / -0.000 | ×0.25 -0.006 / -0.000 | ×0.01 +0.124 / +0.020 | beats / beats |
| obliq | 2k | 16 | l0.05: ×0.89 +0.004 / +0.003 | ×0.51 +0.002 / -0.000 | ×0.51 +0.002 / +0.003 | ×0.31 -0.003 / -0.006 | costlier / costlier |
| obliq | 8k | 16 | l0.05: ×0.88 +0.003 / +0.001 | ×0.35 +0.026 / +0.020 | ×0.35 +0.032 / +0.029 | ×0.09 -0.006 / -0.015 | tradeoff / tradeoff |
| obliq | 32k | 8 | l0.05: ×0.88 +0.002 / -0.001 | ×0.31 +0.079 / +0.086 | ×0.31 +0.073 / +0.073 | ×0.02 -0.002 / +0.006 | tradeoff / tradeoff |
| oolong | 2k | 16 | l0.05: ×0.71 +0.006 / +0.000 | ×0.40 +0.063 / +0.000 | ×0.40 +0.077 / -0.000 | ×0.11 +0.079 / +0.002 | tradeoff / tradeoff |
| oolong | 8k | 16 | l0.05: ×0.65 +0.004 / -0.002 | ×0.34 +0.021 / -0.001 | ×0.34 +0.027 / -0.000 | ×0.03 +0.039 / -0.001 | costlier / costlier |
| oolong | 32k | 8 | l0.05: ×0.56 +0.001 / +0.001 | ×0.31 +0.001 / +0.001 | ×0.31 +0.000 / +0.000 | ×0.01 +0.005 / +0.003 | costlier / costlier |
| outlier | 2k | 16 | l0.5: ×0.34 -0.028 / +0.000 | ×0.45 -0.030 / +0.000 | ×0.45 -0.034 / +0.000 | ×0.27 +0.133 / +0.022 | beats / beats |
| outlier | 8k | 16 | l0.5: ×0.16 -0.042 / +0.000 | ×0.30 -0.053 / +0.000 | ×0.30 -0.036 / +0.000 | ×0.07 +0.218 / +0.072 | beats / beats |
| outlier | 32k | 8 | l0.5: ×0.12 -0.163 / -0.090 | ×0.26 -0.079 / -0.024 | ×0.26 -0.040 / -0.001 | ×0.02 +0.286 / +0.227 | beats / beats |
| outlier_amzn | 2k | 16 | l0.05: ×0.99 -0.003 / +0.000 | ×0.44 +0.153 / +0.115 | ×0.44 +0.064 / +0.040 | ×0.25 +0.075 / +0.122 | tradeoff / tradeoff |
| outlier_amzn | 8k | 16 | l0.05: ×0.99 -0.006 / +0.000 | ×0.31 +0.076 / +0.028 | ×0.31 +0.190 / +0.079 | ×0.07 +0.173 / +0.215 | tradeoff / tradeoff |
| outlier_amzn | 32k | 8 | l0.05: ×0.99 -0.000 / +0.000 | ×0.27 +0.159 / +0.142 | ×0.27 +0.115 / +0.089 | ×0.02 +0.697 / +0.739 | tradeoff / tradeoff |
| qdmatch_hpqa | 2k | 16 | l0.5: ×0.99 -0.000 / -0.000 | ×0.77 -0.000 / +0.000 | ×0.77 -0.000 / -0.000 | ×0.67 -0.000 / +0.000 | costlier / costlier |
| qdmatch_hpqa | 8k | 16 | l0.5: ×0.98 +0.003 / +0.000 | ×0.42 -0.008 / +0.000 | ×0.42 -0.009 / -0.000 | ×0.18 +0.001 / +0.005 | costlier / costlier |
| qdmatch_hpqa | 32k | 8 | l0.5: ×0.98 -0.008 / -0.000 | ×0.32 -0.008 / -0.000 | ×0.32 -0.009 / -0.000 | ×0.04 +0.462 / +0.432 | costlier / costlier |
| reorder | 2k | 16 | l0.2: ×0.99 +0.001 / +0.000 | ×0.32 +0.491 / +0.504 | ×0.32 +0.417 / +0.355 | ×0.10 +0.667 / +0.647 | tradeoff / tradeoff |
| reorder | 8k | 16 | l0.2: ×0.99 +0.001 / +0.001 | ×0.27 +0.730 / +0.763 | ×0.27 +0.685 / +0.676 | ×0.04 +1.532 / +1.517 | tradeoff / tradeoff |
| rerank | 2k | 16 | l0.05: ×0.95 +0.018 / +0.020 | ×0.41 +0.094 / +0.079 | ×0.41 +0.031 / +0.051 | ×0.18 +0.502 / +0.537 | tradeoff / costlier |
| rerank | 8k | 16 | l0.05: ×0.94 -0.009 / -0.011 | ×0.33 -0.041 / -0.046 | ×0.33 -0.066 / -0.068 | ×0.08 +0.961 / +0.971 | loses / loses |
| rerank | 32k | 8 | l0.05: ×0.94 -0.003 / -0.004 | ×0.32 +0.212 / +0.204 | ×0.32 +0.121 / +0.114 | ×0.06 +1.300 / +1.311 | tradeoff / tradeoff |
| scifact | 2k | 16 | l0.5: ×0.85 +0.000 / +0.000 | ×0.44 +0.000 / +0.000 | ×0.44 +0.000 / -0.000 | ×0.28 +0.001 / +0.000 | costlier / costlier |
| scifact | 8k | 16 | l0.5: ×0.82 -0.000 / +0.000 | ×0.27 -0.002 / -0.000 | ×0.27 -0.001 / -0.000 | ×0.07 -0.003 / +0.000 | costlier / costlier |
| scifact | 32k | 8 | l0.5: ×0.81 -0.081 / -0.000 | ×0.23 -0.089 / -0.000 | ×0.23 -0.082 / -0.000 | ×0.02 -0.083 / +0.000 | costlier / costlier |
| strmatch | 2k | 16 | l0.05: ×0.40 +1.041 / +1.005 | ×0.63 -0.000 / +0.000 | ×0.63 -0.000 / +0.000 | ×0.33 +0.014 / +0.006 | loses / loses |
| strmatch | 8k | 16 | l0.05: ×0.29 +1.447 / +1.400 | ×0.48 +0.000 / +0.000 | ×0.48 +0.000 / +0.000 | ×0.09 +0.663 / +0.582 | loses / loses |
| strmatch | 32k | 8 | l0.05: ×0.26 +1.588 / +1.540 | ×0.44 -0.000 / +0.000 | ×0.44 +0.000 / +0.000 | ×0.02 +1.649 / +1.793 | loses / loses |
| textgroups | 2k | 16 | l1.0: ×0.99 +0.004 / -0.001 | ×0.61 +0.020 / +0.007 | ×0.61 +0.007 / -0.017 | ×0.49 -0.059 / -0.076 | costlier / costlier |
| textgroups | 8k | 16 | l1.0: ×0.98 +0.015 / +0.010 | ×0.30 +0.161 / +0.141 | ×0.30 +0.099 / +0.065 | ×0.09 -0.009 / -0.057 | tradeoff / tradeoff |
| textgroups | 32k | 8 | l1.0: ×0.97 +0.035 / +0.029 | ×0.25 +0.206 / +0.181 | ×0.25 +0.170 / +0.134 | ×0.02 +0.395 / +0.369 | tradeoff / tradeoff |
| xabsence (VOID ckpt) | 2k | 16 | l0.5: ×0.18 +0.024 / +0.036 | ×0.47 +0.371 / +0.383 | ×0.47 +0.019 / +0.023 | ×0.18 -0.054 / -0.055 | beats / beats |
| xabsence (VOID ckpt) | 8k | 16 | l0.5: ×0.10 -0.575 / -0.617 | ×0.38 +0.277 / +0.256 | ×0.38 +0.012 / -0.018 | ×0.05 -0.570 / -0.576 | beats / beats |
| xabsence (VOID ckpt) | 32k | 8 | l0.5: ×0.08 -0.272 / -0.342 | ×0.36 -0.059 / -0.037 | ×0.36 -0.119 / -0.137 | ×0.01 -0.466 / -0.454 | beats / beats |

**Verdict counts over all cells** (val-selected λ):

* vs `gold_rand20p8_noslot`: {'tradeoff': 17, 'ties': 3, 'costlier': 16, 'beats': 12, 'loses': 4}
* vs `gold_fl20p8_noslot`: {'tradeoff': 15, 'ties': 3, 'costlier': 17, 'loses': 5, 'beats': 12}
* vs `gold_first20`: {'tradeoff': 19, 'costlier': 17, 'loses': 4, 'beats': 12}
* vs `gold_only_noslot`: {'tradeoff': 27, 'costlier': 19, 'loses': 6}

The 12 `beats` cells against `gold_rand20p8_noslot` are nq ×3, outlier ×3, msmarco ×3 and xabsence ×3 (VOID). The 4 `loses` cells are strmatch ×3 and rerank 8k (−0.009 vs the heuristic's −0.041, just outside the tolerance). Absence ran λ = 0.05 only: the λ 0.2 / 0.5 / 1.0 runs were stopped at epochs 19 / 12 / 12 (≈ 2 min per epoch), all at val keep ≥ 0.93 and still rising, so they were converging to keep-all.

Reading by task family:
- **Router wins: nq, outlier, msmarco.** It is 25–55% cheaper than the heuristics at parity.
  Outlier 32k (−0.16 vs −0.08) rests on 8 rows.
- **Ties: contradiction.** The router lands on the heuristics' compaction at ΔCE 0. It never gets
  as cheap as `gold_first20` (×0.38 / 0.27 / 0.23).
- **Keeps everything, so no saving: fiqa, niah, obliq, outlier_amzn, qdmatch_hpqa, rerank, scifact,
  textgroups, grouping, reorder, absence.** These cells read "costlier" or "tradeoff". On reorder,
  rerank, textgroups, outlier_amzn and absence the heuristics themselves break (+0.1 to +1.2), so
  keeping everything is the right answer there, not a win.
- **Fails: strmatch.** Every λ gives val ΔCE +1.26 to +1.33, and w_gold goes negative (−0.5).
  The documents are ≤ 64 tokens and every document's id is load-bearing, which is the same
  id-pooling failure the grid audit found.
- **oolong:** val selection picks λ = 0.05 (×0.56–0.71). At λ ≥ 0.2 the router drops every line
  (×0.04–0.14) at nearly the heuristics' ΔCE, because these oolong test rows barely need the
  context (the grid's `k0` is +0.025). That is a "win" nobody should take seriously.
- **xabsence** is VOID: a paraphrase-era checkpoint scored on EXACT rows (grid manifest).

**Oracle-λ envelope** (λ picked on the test rows, optimistic; `router_vs_grid.json`, 52 cells): against `gold_rand20p8_noslot` the router beats in 25 cells (22 without the VOID xabsence), ties in 4, loses in 20, and in 3 never gets as cheap as the baseline. The extra wins over the val-selected λ come from obliq and rerank (all rungs), reorder 2k, and niah / grouping / outlier_amzn at 32k. On rerank and reorder the heuristic itself breaks; on the others some λ happens to land at an intermediate compaction on the test rows, and 16 val rows do not identify that λ. The "loses" cells on fiqa, qdmatch_hpqa, scifact and textgroups are the bistability again: interpolating between a collapsed point and a keep-all point at the baseline's compaction is worse than the heuristic.

`gold_only_noslot` (new reference scheme: gold whole, every other document dropped outright) is the
rule a gold-aware router can collapse to.
- It reaches parity only on nq 2k / 8k, msmarco 2k, fiqa 2k / 8k, textgroups 2k / 8k, scifact, obliq
  and qdmatch_hpqa 2k / 8k.
- It breaks on niah (+0.57 to +1.03), rerank (+0.5 to +1.3), contradiction (+0.16 to +1.17),
  grouping, strmatch and outlier (+0.13 to +0.29).
- On nq / outlier / msmarco the router sits between the two: it keeps a little of the non-gold
  documents, cheaper than the heuristics and safer than gold-only.

## 3. Ablations (pilot tasks, λ = 0.2, val rows ⚠ 16)

| task | full | noemb | nogold | seed-2 replicate (full) |
|---|---|---|---|---|
| nq | ×0.29, ΔCE −0.027 | ×0.34, −0.028 | ×0.21, **+0.274** (median +0.005) | ×0.12, −0.027 |
| outlier | ×0.94, +0.003 | ×0.85, +0.001 | ×1.00, 0.000 | ×1.00, 0.000 |
| contradiction | ×0.43, 0.000 | ×0.62, 0.000 | ×0.99, 0.000 | ×0.92, +0.001 |
| oolong (no gold) | drops all | drops all | drops all | drops all |

(val deterministic T2/T and mean ΔCE; outlier only escapes the keep-all basin at λ ≥ 0.5.) On the
test rows nogold is as bad: nq ×0.23 / 0.20 / 0.19 at **+0.24 / +0.45 / +1.37**.

* **No gold is bad.** Without the gold flag the router either keeps everything (outlier,
  contradiction) or compacts and pays heavily on the rows whose gold it drops (nq, where the
  median stays small but the mean is driven by a tail of rows at +1 to +3). Nothing gold-blind in
  this family is deployable; the gold flag carries most of the router.
* **The embedding helps a little, within seed noise.** At the same ΔCE, full is somewhat cheaper
  than noemb on nq (×0.29 vs ×0.34) and contradiction (×0.43 vs ×0.62), and dearer on outlier. But
  the seed replicate moves compaction by more than that (nq ×0.29 → ×0.12, contradiction
  ×0.43 → ×0.92): the difference between ablations is smaller than the difference between seeds.

## 4. What the router learned

* **Gold weight**: rises from 0 on every task with a useful gold set. By best epoch it is +1.9 /
  +2.3 / +3.4 on nq (λ 0.05 / 0.2 / 0.5), +5.0 on outlier at λ ≥ 0.5, +4.2 on contradiction at
  λ 0.2, and +4.4 to +5.1 on msmarco. It is **negative on strmatch (−0.5) and niah (−1.5 at
  λ ≥ 0.5)**, the two tasks where every document's id is needed.
* **Position profile**: the per-offset one-hots are mostly noise at about ±2 logits (REINFORCE
  credit over tens of thousands of per-token decisions). What is consistent:
  * `rel_in_doc` is negative on nq, outlier, contradiction and oolong λ ≥ 0.2 (−0.6 to −1.7), so
    the router prefers each document's start (oolong λ = 0.05: +0.36);
  * offsets 3–4 from doc start are positive on nq, outlier and contradiction (+1.4 to +2.9), which
    is where the `Document [N]:` id digits sit;
  * on oolong, the long-range buckets (offset ≥ 64 from either end) are negative (−1 to −5).
* **Vocabulary (s[v] = w_emb·e_rms(v)/√d, token types occurring ≥ 3 times in the train rows,
  `runs/<task>/weights_analysis.json`)**:
  * *outlier* (seed-to-seed correlation +0.71, the most reproducible): keeps **digits,
    punctuation, whitespace and brackets** (`9 7 2 5 … / ' = "`) and drops `the this which that
    their`, **`Document`/`document`** and rare entity pieces. It keeps the id digits of the header
    and not the word "Document".
  * *nq* (seed correlation **+0.29**, mostly noise): seed 0 keeps digits and drops function words
    and month names (corr with IDF +0.18). Seed 2 learned an IDF-like score instead (corr +0.55:
    content words such as `honoured Released toured Elections` up, punctuation and brackets down).
    The only shared piece is "drop function words and punctuation".
  * *oolong* (seed correlation +0.96, but irrelevant at test time because at λ ≥ 0.2 every p is
    < 0.5): keeps rare content words, drops digits and punctuation.
  * The seed-0 configs of one task correlate +0.7 to +0.9 with each other partly because they
    share the RNG stream (same early masks). The two-seed pair is the real noise check.
* **Train/val gap**: small on the tasks that learn (nq l0.5 train ΔCE −0.033 vs val −0.022 at keep
  0.22 / 0.23). With 32 rows the embedding does not visibly memorise row-specific tokens. The
  weights it learns are just weakly determined.

## 5. Why it is bistable, and the method question

REINFORCE learned on the pilot (gate: 9/12 beats, 3 ties), so I **did not switch** to the
differentiable relaxation (GDN `g→g^p`, `β→pβ`; attention `+log p` key bias). The failure on the
extension tasks is optimisation, not expressiveness. Starting from p = 0.5, randomly dropping half
of the gold tokens costs so much CE that:
- at small λ the cheapest move is to raise the bias and keep everything (outlier λ ≤ 0.2,
  qdmatch_hpqa, fiqa, niah λ 0.05);
- at large λ the penalty gradient, which is dense and exact, pushes the bias down faster than the
  noisy REINFORCE signal can raise `w_gold`, and everything collapses (qdmatch_hpqa λ 1.0: val
  ΔCE +0.80 at ×0.13 while `gold_only_noslot` is at parity at ×0.18).

The intermediate solution exists (the heuristics sit on it) but is a narrow basin for this
estimator. Seed replicates show the same thing: contradiction λ 0.2 lands at ×0.43 or ×0.92
depending on the seed.

Next steps, in order of cost:
1. **Warm start**: initialise `w_gold` > 0 and the offset-0–15 weights from the `p8` heuristic, or
   fit the router to a heuristic mask first, then REINFORCE. This contradicts the "gold weight
   starts at 0" sanity check, which was the point of the pilot.
2. **A λ schedule** (anneal from 0 upward) instead of a fixed λ, so `w_gold` rises before the bias
   is pushed down.
3. **The differentiable relaxation**, which removes the credit-assignment noise altogether.
4. More val rows (16 is too few to pick λ on bistable tasks; the oracle envelope wins twice as
   often as the val-selected λ).

## 6. Files

* `debug/learned_router/`: `router_lib.py`, `train_router.py`, `run_router_local.sbatch`,
  `fetch_train_rows.py`, `smoke_cpu.py` (passes), `analyze_router.py`, `analyze_weights.py`,
  `summarize_sweep.py`, `split_report.json`, `headline.json`, `router_vs_grid.json`, `runs/`,
  `weights/`.
* Driver `debug/devloss_grid/ctc_devloss_grid.py`:
  - `sel="router"` schemes (`router_[nogold_|noemb_]l<λ>[_samp]`, `router_l0.2_nomark`);
  - `keep="none"`;
  - the `keep_markers` flag;
  - new reference scheme `gold_only_noslot`;
  - `freeze_fla_length_autotune()`;
  - `route_keep` per-row field.
* `collect_grid.py`: default results roots now include `results_router`. `render_grid.py`: router
  and `gold_only_noslot` rows, shown only when present.
* Shared library (default-off): `mark_positions_free(free_markers=False)` in
  `src/olmo_core/nn/attention/chunked_mask.py`, wired from `pst["keep_token_mask_markers"]` in
  `src/olmo_core/nn/transformer/model.py`.

## 7. Differentiable router with a task-relative tolerance (2026-09-27/28)

**Question.** Does training the same linear router through a *differentiable* removal, with
"maximise compaction s.t. mean ΔCE ≤ τ·mean CE_full", fix REINFORCE's bistability? Specifically:
- does it beat the REINFORCE router and `gold_rand20p8_noslot` at matched compaction;
- does it compact the tasks REINFORCE left at keep-all?

**Answer.** It compacts tasks REINFORCE could not (fiqa, rerank), but on the core pilot tasks it
does not beat REINFORCE or the heuristic. **Gate not passed, so it was not run on the full grid.**
⚠ eval_size 16/16/8 per rung throughout.

### 7.1 Method (as run; code `train_diff_router.py`, `run_diff_local.sbatch`)

* **Relaxation, train time only.** Evaluation is the unchanged exact hard removal (`sel="router"`,
  schemes `router_diff_tau<τ>`). Each routed token carries a hard-concrete gate z in [0, 1], with
  log α = router logit. It is applied without compacting the row through a new default-off
  `soft_keep` kwarg (bit-identical when absent) in `Transformer`, `Attention` and `GatedDeltaNet`:
  * **Attention:** the causal mask plus `log(z + 1e-8)` on every key, through masked SDPA. This is
    exact at z ∈ {0, 1}.
  * **GDN:** β and g (log decay) are scaled by z, so z = 0 gives an identity state step.
  * **GDN short conv, removal-aware** (`CausalConv1d.forward_soft_keep`): gated scan registers
    `R_r(t) = z_t R_{r-1}(t−1) + (1−z_t) R_r(t−1)` hold the previous *kept* inputs. For binary z
    the conv therefore equals the conv of the compacted row, and it is differentiable in between.
    It is chunked with non-positive exponents and activation-checkpointed; 8k fits in 47 GiB.
* **Why the removal-aware conv.** The spec's suggestion (scale token t's conv input by p) left a
  large relaxation gap. On nq, 6 rows at 2k/8k, binary gates at keep 0.8/0.5/0.2, relaxed vs
  exact hard ΔCE had mean |gap| **0.27 nats** against mean |ΔCE| 0.60 (correlation 0.71). One row
  scored +0.99 relaxed vs +0.001 hard. With the removal-aware conv the gap is **0.0008 nats**
  (correlation 0.99999); on outlier it is 0.0032 (correlation 0.99994). `runs/<task>/diff_gap.json`
  is re-measured at every job start.
* **Tests:**
  * `test_relax_cpu.py` (attention-only tiny model): p=1 ≡ full; binary p ≡ hard compaction
    (≤ 5e-7); float64 finite-difference gradcheck of dCE/dp. The conv ≡ compacted conv (2e-6),
    with its own float64 gradcheck.
  * `test_relax_gpu.py` (tiny hybrid, fla kernels): p=1 ≡ full; directional finite differences
    within 1% of scale; binary relaxed ≡ hard compaction (gap ≤ 2e-5).
* **Objective:** minimise E[keep] (expected L0) s.t. ΔCE ≤ ε = max(τ·mean CE_full(train),
  **0.005 nats**). The floor follows the brief's suggestion. It binds on every task whose CE_full is
  ≈ 0 (nq, contradiction, outlier at small τ, niah, strmatch, scifact, qdmatch), so all small τ
  share one run there.
  * Lagrangian `keep_L0 + μ(ΔCE/ε − 1)`.
  * Init p = 0.95; gold and embedding weights 0.
  * β anneals 2/3 → 0.1 over the first 60% of 60 epochs; straight-through binary gates from 75%.
  * Selection: the most compact epoch whose **val** ΔCE (exact hard removal) ≤ ε_val.
* **Four optimisation fixes were needed before it trained at all** (each observed on a pilot run):
  1. **Per-step dual ascent spiked.** One row whose gold got dropped sent μ from 0 to 0.5, and the
     ×μ/ε ≈ ×100 constraint gradient swamped the L0 term under Adam normalisation. Now μ updates
     once per epoch.
  2. **Adam β₂ = 0.999 or 0.95 froze the bias.** The second moment remembered the constraint-phase
     gradients, so once μ = 0 keep stalled at ~0.8 (bias moved ~0.002/step). Now β₂ = 0.8.
  3. **The dual ran on the sampled relaxed ΔCE**, which sits well above the deployed policy's
     (outlier: +0.020 sampled vs −0.001 deterministic). μ over-tightened and sent routers back to
     keep-all. Now the dual runs on the deterministic gate's exact hard ΔCE over all train rows.
  4. **Plain [0,1] clamp killed gates.** oolong and rerank collapsed to keep ≈ 0 and never
     recovered, because dropped gates got zero gradient. Now the clamp is straight-through: a
     dropped token still receives the finite dCE/dz.

### 7.2 Pilot results on the test rows

Each τ cell gives compaction T2/T, mean ΔCE, and whether mean ΔCE ≤ max(τ·CE_full(test rung),
0.005): ✓ holds, ≈ misses by < 1 paired SE, ✗ misses. The REINFORCE router is at its val-selected
λ (§2). All three columns are paired on the same rows.

⚠ eval_size 16/16/8. Where CE_full ≈ 0.001–0.04, the tolerance is the 0.005-nat floor, and a
single row can flip ✓/✗.

| task | rung | n | CE_full | τ=0.05 | τ=0.1 | τ=0.2 | τ=0.4 | REINFORCE (val λ) | gold_rand20p8_noslot |
|---|---|---|---|---|---|---|---|---|---|
| nq | 2k | 16 | 0.003 | ×0.20 +0.020 ✗ | ×0.20 +0.020 ✗ | ×0.34 +0.094 ✗ | ×0.40 +0.099 ✗ | l0.5 ×0.22 -0.001 | ×0.34 -0.002 |
| nq | 8k | 16 | 0.037 | ×0.16 -0.019 ✓ | ×0.16 -0.019 ✓ | ×0.31 +0.102 ✗ | ×0.37 +0.086 ≈ | l0.5 ×0.17 -0.033 | ×0.26 -0.036 |
| nq | 32k | 8 | 0.007 | ×0.15 +0.024 ≈ | ×0.15 +0.024 ≈ | ×0.29 +0.483 ✗ | ×0.36 +0.169 ✗ | l0.5 ×0.16 +0.005 | ×0.25 -0.006 |
| outlier | 2k | 16 | 0.035 | ×0.87 +0.017 ≈ | ×0.87 +0.017 ≈ | ×0.47 -0.034 ✓ | ×0.91 -0.000 ✓ | l0.5 ×0.34 -0.028 | ×0.45 -0.030 |
| outlier | 8k | 16 | 0.057 | ×0.84 -0.022 ✓ | ×0.84 -0.022 ✓ | ×0.34 -0.052 ✓ | ×0.89 -0.005 ✓ | l0.5 ×0.16 -0.042 | ×0.30 -0.053 |
| outlier | 32k | 8 | 0.202 | ×0.84 -0.030 ✓ | ×0.84 -0.030 ✓ | ×0.32 -0.166 ✓ | ×0.89 -0.042 ✓ | l0.5 ×0.12 -0.163 | ×0.26 -0.079 |
| contradiction | 2k | 16 | 0.002 | ×0.79 +0.014 ✗ | ×0.79 +0.014 ✗ | ×0.79 +0.014 ✗ | ×0.87 +0.000 ✓ | l0.2 ×0.48 -0.000 | ×0.48 -0.000 |
| contradiction | 8k | 16 | 0.001 | ×0.77 +0.086 ✗ | ×0.77 +0.086 ✗ | ×0.77 +0.086 ✗ | ×0.85 +0.002 ✓ | l0.2 ×0.39 +0.000 | ×0.39 -0.001 |
| contradiction | 32k | 8 | 0.038 | ×0.77 +0.025 ≈ | ×0.77 +0.025 ≈ | ×0.77 +0.025 ≈ | ×0.85 -0.016 ✓ | l0.2 ×0.36 -0.033 | ×0.35 -0.033 |
| oolong | 2k | 16 | 0.107 | ×1.00 +0.000 ✓ | ×1.00 +0.000 ✓ | ×1.00 +0.000 ✓ | ×0.57 +0.041 ✓ | l0.05 ×0.71 +0.006 | ×0.40 +0.063 |
| oolong | 8k | 16 | 0.082 | ×1.00 +0.000 ✓ | ×1.00 +0.000 ✓ | ×1.00 +0.000 ✓ | ×0.53 -0.001 ✓ | l0.05 ×0.65 +0.004 | ×0.34 +0.021 |
| oolong | 32k | 8 | 0.069 | ×1.00 +0.000 ✓ | ×1.00 +0.000 ✓ | ×1.00 +0.000 ✓ | ×0.52 +0.001 ✓ | l0.05 ×0.56 +0.001 | ×0.31 +0.001 |
| fiqa | 2k | 16 | 0.117 | ×0.44 +0.060 ≈ | ×0.19 +0.157 ✗ | ×0.22 +0.139 ✗ | ×0.26 +0.029 ✓ | l0.5 ×0.97 -0.001 | ×0.46 -0.014 |
| fiqa | 8k | 16 | 0.358 | ×0.36 +0.017 ✓ | ×0.11 +0.120 ≈ | ×0.14 -0.033 ✓ | ×0.18 -0.027 ✓ | l0.5 ×0.95 +0.014 | ×0.29 +0.016 |
| fiqa | 32k | 8 | 0.654 | ×0.35 +0.245 ≈ | ×0.10 +0.332 ✗ | ×0.14 +0.092 ✓ | ×0.17 +0.110 ✓ | l0.5 ×0.95 -0.005 | ×0.25 +0.066 |
| rerank | 2k | 16 | 0.518 | ×0.74 +0.019 ✓ | ×0.14 +0.115 ✗ | ×0.13 +0.169 ✗ | ×0.13 +0.260 ✗ | l0.05 ×0.95 +0.018 | ×0.41 +0.094 |
| rerank | 8k | 16 | 0.998 | ×0.72 -0.031 ✓ | ×0.11 -0.081 ✓ | ×0.11 -0.016 ✓ | ×0.11 -0.042 ✓ | l0.05 ×0.94 -0.009 | ×0.33 -0.041 |
| rerank | 32k | 8 | 1.095 | ×0.72 -0.002 ✓ | ×0.12 +0.027 ✓ | ×0.11 +0.880 ✗ | ×0.11 +0.531 ✗ | l0.05 ×0.94 -0.003 | ×0.32 +0.212 |

`debug/learned_router/headline_diff.json`, `summarize_diff.py`. Over all 72 (task, rung, τ)
cells the tolerance holds out of sample in 40, within one SE in 11, and misses in 21.

### 7.3 Gate verdict

**Beats REINFORCE / the heuristic at matched compaction?** No on the core tasks:
* **nq (τ ≤ 0.1):** ×0.20/0.16/0.15 vs REINFORCE ×0.22/0.17/0.16. It is barely cheaper but
  +0.02 worse at 2k and 32k (medians ≈ 0: a few rows). τ = 0.2/0.4 selected early epochs (w_gold
  ~2) that break at 32k, up to +0.48.
* **outlier:** its best (τ = 0.2) is ×0.47/0.34/0.32 at ΔCE ≤ −0.03, against REINFORCE
  ×0.34/0.16/0.12 and the heuristic ×0.45/0.30/0.26. The other τ stay at ×0.84–0.91.
* **contradiction:** ×0.77–0.87 against ×0.36–0.48 for both baselines, with ΔCE outliers up to
  +0.09.
* **oolong:** τ = 0.4 reaches ×0.53/0.52 at 8k/32k with ΔCE ≈ 0 (2k: ×0.57 at +0.041, within
  its tolerance), cheaper than REINFORCE's ×0.65/0.56 at equal ΔCE. Smaller τ never found a val-satisfying compacted epoch (tolerance 0.0055–0.022 on 16
  noisy rows).

**Compacts what REINFORCE left at keep-all?** Yes:
* **fiqa:** ×0.10–0.44 across τ, against REINFORCE ×0.95. At 8k it beats `gold_rand20p8_noslot`: τ = 0.2
  gives ×0.14 at −0.033 vs ×0.29 at +0.016. At 32k the tolerance holds for τ ≥ 0.2, where the
  heuristic itself is at +0.07. 2k misses at τ = 0.1/0.2.
* **rerank:** τ = 0.05 gives ×0.72–0.74 with ΔCE ≤ +0.02 at every rung, the same as REINFORCE's
  keep-all; the heuristic is +0.09 at 2k
  and +0.21 at 32k. τ = 0.1 gives ×0.11–0.14, holding at 8k and 32k but missing at 2k (+0.115 vs
  ε 0.052). Larger τ break at 32k.

**Does the relative tolerance hold out of sample?**
* For the smallest qualifying τ it mostly does, at 8k in particular.
* It fails in two ways:
  * where the floor is the real tolerance and 16 rows contain one outlier row (contradiction 8k:
    median +0.001, mean +0.086);
  * at 32k for larger τ, a length the router never trained on (nq τ ≥ 0.2, rerank τ ≥ 0.2).
* The 16-row val set is also lenient. At the selected epochs, train deterministic ΔCE is often
  *above* val (outlier +0.023 vs +0.003, contradiction +0.024 vs +0.003). So these misses are
  selection noise rather than the ~2.5k embedding weights memorising the 32 train rows. nq and
  fiqa train/val agree within ~0.01.

### 7.4 What this says

The relaxation itself is sound: exact at binary gates, correct gradients, and the router trains
through it. What decides the outcome is the **start point plus the tolerance**. Starting at keep-all
with a 0.005-nat tolerance, the router has to find a path where every intermediate policy is
within tolerance. On outlier and contradiction, dropping even a few percent of tokens early costs
more than 0.005, so they stall at ×0.8. On tasks where CE_full is large enough that ε is real
(fiqa, rerank, oolong at τ = 0.4) it compacts, including where REINFORCE's p = 0.5 start sent it to
keep-all. The response to τ is also not monotone (nq ×0.19 at τ ≤ 0.1 vs ×0.32/0.38 at τ = 0.2/0.4):
each run's dual trajectory differs, and val selection on 16 rows picks early epochs.

Next steps, not run:
1. **Hybrid init:** start from the REINFORCE router (or `gold_rand20p8`'s mask) and refine with the
   relaxed objective.
2. **A larger val slice** (≥ 64 rows) for selection.
3. **Train at 32k too,** or add a length term; the τ ≥ 0.2 routers break only at 32k.
4. **Run the dual on a trailing mean**, so μ does not overshoot to keep-all within a few epochs.


## 8. End-to-end router with budget calibration vs `gold_fl20p8_noslot` (2026-09-28/29)

**Question.** Trained fully end to end (no initialisation from, or distillation of, any fixed
rule), does the same linear router Pareto-beat the strongest slot-less heuristic,
`gold_fl20p8_noslot`, on every task? Three follow-ups:
- Does one router shared across tasks transfer?
- Does a router trained at 2k transfer to 8k and 32k?
- Can a label-free per-length cutoff fix the length transfer?

**Answer.**
- **Per task at 2k:** it beats the bar on **11 of 17** tasks and matches it on 1. It loses on
  rerank, textgroups, grouping, niah and absence. Cheap retries (more rows, L2 on w_pos, larger ρ)
  fixed none of the five.
- **Cross-task (nq + contradiction + scifact):** it does *not* transfer to held-out tasks. At 2k
  it beats 2, matches 3 and loses 12, and sharing costs ~+0.09 nats paired (median) against the
  per-task routers.
- **Length transfer:** 2k routers keep beating the bar on ~7/17 tasks at 8k and 5/15 at 32k. A
  per-length label-free cutoff does not raise that count, but it repairs strmatch.

⚠ eval_size **16** test rows at 2k/8k and **8** at 32k throughout (the grid's standard test rows).
Every comparison is paired on the same rows against `gold_fl20p8_noslot` re-scored in the same
file.

### 8.1 Method (as run; `train_e2e_router.py`, `run_e2e_local.sbatch`)

* **Relaxed training path:** the same exact-at-binary relaxation as §7 (`soft_keep`: attention
  log-p bias, GDN β/g scaling, removal-aware short conv), with a hard-concrete gate per routed
  token.
  * β anneals 2/3 → 0.1 over 70% of training.
  * Gates are straight-through binary from 85%, with a straight-through [0, 1] clamp throughout.
  * Loss = answer CE + KL(full-model answer distribution ‖ routed), with **w_kl = 1**.
* **Budget calibration instead of a dual.** Each step pools the batch's router logits and bisects
  one scalar offset so that the mean P(z > 0) equals ρ_t. The router therefore learns only the
  *ranking* of tokens, and the ρ_t warmup (1 → ρ over the first 50%) is the only schedule. This
  replaced the Lagrangian of the first e2e attempt, whose multiplier oscillated and wound up.
* **Eval rule:** the global offset hitting ρ on the train rows is folded into b, so the grid's
  `p > 0.5` rule applies unchanged.
* **Label-free matched cutoffs** (`--match-comps`): a bisection on *unlabeled* val rows finds the
  offset whose mean T2/T equals a target. They are saved as `<name>_c<x>.pt` and scored as extra
  schemes. This makes the paired comparison at the bar's own T2/T possible without looking at
  test labels.
* **Data:** 16 train rows per task at 2k, and 64 val rows (32 val + 32 extra drawn from the train
  pool, disjoint from the 16). 40–160 epochs (80 for most tasks), Adam lr 0.1, β₂ 0.8,
  init p = 0.95. There is one ρ per task, set at ≈ 0.7 × the bar's routed-keep fraction; the
  first-pass (r3) runs set it once and did not tune it.
* **Variants:**
  * `nocontpos` drops the two continuous position features (rel_in_doc, doc_rel_idx). The
    contradiction router latched onto them and failed on test.
  * `--route-markers` (strmatch) lets the router drop document markers as well, with its own
    w_marker initialised at p = 0.999. With markers always kept, strmatch cannot reach the bar's
    T2/T by removing text alone.
* **Verdict rule** (`summarize_vsfl.py`). B = the bar at (c_B, d_B); tol = max(paired SE,
  0.005 nats).
  * **beats:** some router point with T2/T ≤ 1.02·c_B and ΔCE ≤ d_B + tol that is strictly
    better on one axis (T2/T ≤ 0.95·c_B, or paired < −tol).
  * **matches:** some point with T2/T ≤ 1.05·c_B and |paired| ≤ tol.
  * **loses:** otherwise.
  * Reported per task: (a) the router calibrated to c_B, and (b) the smallest router T2/T with
    ΔCE ≤ d_B + tol.

### 8.2 Per-task routers at 2k

⚠ eval_size 16 per task. Columns (a) and (b) are as defined in §8.1.

| task | router | bar ×c, ΔCE | (a) router @ c_B: ×c, ΔCE (paired ± SE) | (b) smallest ×c within tol | verdict |
|---|---|---|---|---|---|
| nq | e2ecal2_rho0.1 (full) | ×0.338 −0.002 | ×0.330 +0.002 (+0.003 ± 0.001) | ×0.182 (+0.003) | **beats** |
| contradiction | e2ecal3ncp_rho0.3 | ×0.480 +0.000 | ×0.491 −0.001 (−0.001 ± 0.001) | ×0.385 (+0.005) | **beats** |
| scifact | e2ecal3ncp_rho0.2 | ×0.436 +0.000 | ×0.462 +0.002 (+0.002 ± 0.002) | ×0.265 (+0.001) | **beats** |
| outlier | e2ecal4ncp_rho0.35 | ×0.450 −0.034 | ×0.449 −0.023 (+0.011 ± 0.012) | ×0.311 (−0.001) | **beats** |
| strmatch | e2ecal3rm3_rho0.2 (routed markers) | ×0.627 −0.000 | ×0.623 +0.000 (+0.000 ± 0.000) | ×0.452 (+0.000) | **beats** |
| fiqa | e2ecal3ncp_rho0.32 | ×0.463 +0.013 | ×0.443 +0.067 (+0.055 ± 0.068) | ×0.296 (−0.001) | **beats** |
| msmarco | e2ecal3ncp_rho0.23 | ×0.332 −0.005 | ×0.333 −0.011 (−0.005 ± 0.009) | ×0.235 (+0.001) | **beats** |
| obliq | e2ecal3ncp_rho0.36 | ×0.515 +0.002 | ×0.552 −0.001 (−0.004 ± 0.003) | ×0.236 (−0.003) | **beats** |
| oolong | e2ecal3ncp_rho0.28 | ×0.398 +0.077 | ×0.383 +0.085 (+0.008 ± 0.018) | ×0.148 (+0.079) | **beats** |
| outlier_amzn | e2ecal3ncp_rho0.31 | ×0.443 +0.064 | ×0.447 +0.068 (+0.004 ± 0.035) | ×0.314 (+0.087) | **beats** |
| reorder | e2ecal3ncp_rho0.22 | ×0.319 +0.417 | ×0.299 +0.417 (−0.000 ± 0.052) | ×0.285 (+0.456) | **beats** |
| qdmatch_hpqa | e2ecal3ncp_rho0.54 | ×0.772 −0.000 | ×0.777 −0.000 (+0.000 ± 0.000) | ×0.777 | matches |
| rerank | e2ecal4ncp_rho0.3 | ×0.407 +0.031 | ×0.404 +0.253 (+0.223 ± 0.035) | — | loses |
| textgroups | e2ecal3ncp_rho0.42 | ×0.607 +0.007 | ×0.593 +0.161 (+0.154 ± 0.042) | — | loses |
| grouping | e2ecal3ncp_rho0.2 | ×0.290 +0.021 | ×0.281 +0.076 (+0.055 ± 0.018) | — | loses |
| niah | e2ecal3ncp_rho0.41 | ×0.587 +0.000 | ×0.598 +0.056 (+0.056 ± 0.056) | — | loses |
| absence | e2ecal3ncp_rho0.48 | ×0.685 +0.890 | ×0.679 +1.604 (+0.714 ± 0.121) | — | loses |

- **Most "beats" are on the compaction axis.** The router reaches the bar's ΔCE at 20–60% fewer
  kept tokens: nq ×0.18 vs ×0.34, obliq ×0.24 vs ×0.52, oolong ×0.15 vs ×0.40, scifact ×0.27 vs
  ×0.44.
- **Some wins have wide error bars.** fiqa, oolong, outlier_amzn and reorder win within wide
  paired SEs (0.02–0.07 on 16 rows), so treat them as "at least matches".
- **Floors:** `floors.py` gives the markers-kept and routed-markers floors at 2k. The strmatch
  floor with markers kept is .266, which is why it needed routed markers.

**Why the five losers lose** (`inspect_e2e_rows.py`):
- **rerank:** fails to *train*. Train and val ΔCE stay high at every ρ, because the gold set is a
  large fraction of each row. At ρ = 0.5 (§8.3) val is still +0.29 at ×0.56.
- **textgroups, grouping, niah:** overfit 16 rows. One early start-position offset (start1 or
  start2) goes strongly negative, and on test rows that drops a document-id bracket or digit, which
  the answer copies.
- **absence:** the bar itself is at +0.89, and the router is +0.71 worse again.

### 8.3 Loser retries (1 line each; ⚠ eval_size 16)

The retries use 48 train rows + 32 val rows and L2 on w_pos, with λ ∈ {1e-4, 1e-3} and λ picked
on val. rerank instead uses ρ = 0.5 with no L2, compared at ×0.41.

- **textgroups:** val picks λ = 1e-4 (val +0.16 vs +0.18). On test it gives ×0.58 +0.25
  (paired +0.24 ± 0.09) → **loses**. The summarizer's "beats" at ×0.49 (+0.070 vs +0.007) holds
  only because paired +0.063 sits exactly at 1 SE (0.063) on a non-monotonic curve, so we do not
  count it.
- **grouping:** λ = 1e-3 gives ×0.30 +0.11 (+0.086 ± 0.025); λ = 1e-4 gives +0.125 ± 0.029.
  Both are worse than the 16-row run's +0.055 → **loses**.
- **niah:** λ = 1e-4 gives ×0.63 +0.25 (±0.17); λ = 1e-3 gives ×0.53 +1.00 (±0.35) → **loses**.
- **rerank:** at ×0.41, +0.53 vs the bar's +0.03 (+0.50 ± 0.04); even ×0.55 is +0.41 →
  **loses**.

More rows plus a position L2 does not rescue the overfit tasks: the failure is not just 16 rows.

### 8.4 Cross-task router (`xtask3`: nq + contradiction + scifact, 16 rows each at 2k)

`--xtasks` loads one frozen per-task checkpoint per task; each row runs through its own task's
model.
- **Training (xtask3):** ρ = 0.2, 27 epochs, nocontpos. Pooled val ΔCE is +0.054 (median +0.001)
  at T2/T 0.25; w_gold is +11.4.
- **Schemes:**
  - `router_xtask_3t` uses the one shared global cutoff (p > 0.5).
  - `router_xtask_3t_c<c_B>` (grid row `router_xtask_3t_cal`) uses a label-free per-task cutoff
    matched on unlabeled val rows to the bar's 2k T2/T.
- **8k/32k:** the 2k cutoffs are applied unchanged.
- **Code and results:** `summarize_xtask.py`; `results_router_xtask/`.

| rung | beats | matches | loses | held-in (nq, contradiction, scifact) | held-out beats/matches |
|---|---|---|---|---|---|
| 2k | 2 | 3 | 12 | match, match, loses (+0.006 ± 0.004) | oolong, obliq / outlier |
| 8k | 3 | 1 | 13 | match, loses, loses | obliq, oolong, outlier |
| 32k | 3 | 0 | 12 | all lose | obliq, oolong, outlier |

- **Cost of sharing** at 2k: xtask paired − per-task paired, both at c_B. The median is +0.09
  nats.
  - Worst: absence +1.02, niah +0.64, strmatch +0.30, outlier_amzn +0.25, reorder +0.22.
  - Where the per-task router was itself weak, xtask is *better*: outlier (r3) −0.05 and oolong
    −0.08.
- **The shared cutoff is the wrong one for every task.** p > 0.5 lands at ×0.12–0.65 depending on
  the task (nq ×0.17, strmatch ×0.44, grouping ×0.12), with ΔCE up to +4. A per-task cutoff is
  necessary, and it needs no labels.
- **What does transfer:** "keep gold, drop far-from-question non-gold". It holds on tasks whose
  answer is a gold document (outlier, oolong, obliq).
- **What does not:** tasks that copy doc ids or spans (strmatch, niah, textgroups, reorder,
  absence) need task-specific id-region weights that the three training tasks never exercise.

### 8.5 Pooled 17-task upper bound (`xtask17`)

**Setup.** The same recipe as xtask3, but trained on all 17 tasks: 16 train rows each (272 rows)
plus 32 val rows each, ρ = 0.2, 14 epochs, nocontpos.
- `--xtask-offload` keeps one frozen per-task model on the GPU and parks the others on CPU.
- Rows are visited in task blocks, so each batch comes from one task: 304 swaps, 16 min of a
  ~65 min run.
- Pooled val ΔCE is +0.57 (median +0.31) at T2/T 0.32.
- Results: `results_router_xtask17/`; `summarize_xtask.py --res-dir results_router_xtask17
  --held-in all --name xtask_17t`.

| rung | beats | matches | loses |
|---|---|---|---|
| 2k | 1 (oolong) | 3 (outlier, rerank, scifact) | 13 |
| 8k | 3 (obliq, oolong, outlier_amzn) | 0 | 14 |
| 32k | 4 (obliq, oolong, outlier, rerank) | 1 (niah) | 10 |

**Pooling every task does not recover the per-task routers**, and it is no better than xtask3.
The shortfall is therefore conflict between tasks for one linear vector (at one ρ), not missing
coverage.
- Sharing costs most on contradiction (+0.21), fiqa (+0.29), strmatch (+0.79) and textgroups
  (+0.37).
- The one large gain is **rerank**: +0.016 ± 0.024 pooled vs +0.45 per task. This is consistent
  with its per-task failure being optimisation, not representation (§8.8).

### 8.6 Length transfer of the per-task 2k routers

⚠ eval_size 16 (8k) / 8 (32k). `summarize_lencal.py` → `headline_lencal.json`. Results:
`results_router_e2e_len/` (fixed cutoff), `results_router_e2e_lencal/` (per-length).

Three label-free cutoffs per task × rung, each paired against `gold_fl20p8_noslot` at that rung:
- **fixed:** the 2k thresholds unchanged (the router's own and its `_c<c_B(2k)>`);
- **(i)** `_k2k_<rung>`: an offset on 24–32 unlabeled rows of the rung that keeps the *same
  fraction of routed tokens* as on the 2k val rows;
- **(ii)** `_c<c_B(rung)>_<rung>`: an offset on the same unlabeled rows so that mean T2/T equals the
  bar's T2/T at that rung. Two more-compact targets (×0.85, ×0.7) were also scored.

| rung | fixed: beats/loses | (i) same routed keep | (ii) bar's T2/T |
|---|---|---|---|
| 8k (17 tasks) | 7 / 10 | 3 / 14 | 7 / 10 |
| 32k (15 tasks) | 5 / 10 | 4 / 11 | 5 / 9 (+1 match) |

- **Tasks that beat at 2k, 8k and 32k under the fixed cutoff:** contradiction, obliq, oolong and
  outlier. reorder beats at 8k (it has no 32k rung), fiqa and outlier_amzn at 8k only, and
  scifact at 32k only.
- **Per-length calibration adds no wins overall, but it repairs strmatch.** The fixed cutoff
  over-compacts it (×0.20 at 32k, +0.083); (i) reaches ×0.32 at +0.005 → **beats**, and (ii)
  matches.
  - Its gold tokens stay at 1.0 kept, and its id-region at .50–.52, under both variants at 8k and
    32k.
- **The fixed cutoff over-compacts non-gold text as length grows. Gold keep is length-stable.**
  - nq, fixed cutoff, routed kept .099/.044/.031 at 2k/8k/32k; id-region .091/.025/.006; gold
    .77/.75/.76.
  - (i) restores routed keep to .10 (gold .79/.82, id-region .07/.03).
  - (ii) goes to .25/.24 (gold .86/.89, id-region .16/.13).
- **nq at 32k still loses under (ii)** (+0.19 ± 0.12, one row at +0.97) despite keeping 89% of
  gold. The 32k loss is not a threshold problem.
- **(i) is the weaker rule.** The 2k routed-keep fraction is too high for tasks whose bar compacts
  harder at length, so (i) lands above c_B and fails the T2/T side of the verdict (fiqa, outlier,
  obliq, qdmatch).

### 8.7 What this says

- **Where it wins.** End-to-end training with a calibrated budget is the first learned router
  here that beats the best slot-less heuristic on most tasks (11/17 at 2k). It does so mostly by
  compaction, at equal ΔCE.
- **Where it loses.** All five losses are tasks where the answer copies document ids or spans
  (textgroups, grouping, niah, absence) or where gold is a large fraction of the row (rerank).
  A linear, per-token, position-plus-gold router with 16–48 rows does not learn a robust rule
  there.
- **Sharing.** One router does not transfer from three retrieval-style tasks, and pooling all 17
  tasks does not help (§8.5). One linear vector at one ρ cannot serve every task, so per-task
  routers (or at least per-task ρ and cutoff) are required.
- **Length.** Transfer from 2k is partial. A per-length label-free cutoff is cheap and fixes
  over-compaction (strmatch), but not the tasks that break at 32k for other reasons.

Code: `debug/learned_router/`: `train_e2e_router.py` (`--xtasks`, `--xtask-offload`,
`--match-comps`, `--calib-root`, `--match-keep-2k`), `run_e2e_local.sbatch`,
`summarize_vsfl.py`, `summarize_xtask.py`, `summarize_lencal.py`, `inspect_e2e_rows.py`,
`floors.py`. Weights: `weights/<task>/<name>.pt` plus `_c<x>`, `_c<x>_<rung>` and `_k2k_<rung>`
variants; xtask routers in `weights/xtask3/`, `weights/xtask17/`. wandb group
[router-e2e](https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/router-e2e).

## 9. Stage 2: learned per-(token, layer) SKIP router on top of the frozen token router (2026-09-29)

**Question.** After the token router has dropped tokens, can a second, end-to-end learned router
save more compute by letting each remaining document token skip individual layers? And does it
beat random or fixed-layer skipping at the same budget?

**Answer.**
* **nq:** yes, down to keep 0.5.
* **scifact:** yes at keep 0.75. Keep 0.5 needs a warm start.
* **outlier:** yes at keep 0.75. Keep 0.5 needs a warm start.
* **contradiction:** only at keep 0.75, and only at the level of a static layer mask.
* **Everywhere**, the learned router is far better than uniform random or fixed-rule skipping at the
  same budget.

⚠ Test eval_size 16 per task (2k rung; the grid's test rows). Val is 64 rows, train 16.

### 9.1 Method (code `debug/learned_router/layerskip/`)

* **Semantics.** Skipping block l for token t removes t from that layer only:
  * the residual passes through, so h_out = h_in;
  * at an attention layer it is not a key;
  * at a GDN layer it neither writes nor decays the state;
  * the removal-aware short conv skips it.
* **Relaxation.** Per-layer gate a ∈ [0, 1] and frozen token gate g:
  * h_out = h_in + a·(block(h_in) − h_in), written so that a = 1 and a = 0 are bit-exact;
  * the block's mixer gets `soft_keep = g·a`, which is the existing (B,T) path, unchanged.
* **No shared model file was modified.** `layerskip_lib.forward_layerskip` drives
  `_prepare_inputs` / `blocks` / `lm_head` itself.
* **Tests.**
  * `test_layerskip_cpu.py`, all pass:
    * a (B,T) soft_keep alone, or with an all-ones layer gate, is BIT-identical to
      `model(x, soft_keep=g)`;
    * binary per-layer gates equal an explicit per-layer removal reference
      (`forward_reference`: the block runs on the kept sub-sequence at its original positions),
      with max relative logit diff ≤ 9e-7 on both the full row and the token-compacted row;
    * a token skipped at every layer equals a token-dropped hard compaction (5e-7);
    * float64 gradcheck w.r.t. a and g passes.
  * `test_layerskip_gpu.py` (tiny hybrid, fla kernels):
    * bit-identical as above;
    * binary gates equal the reference with GDN on the sub-sequence (CE gap ≤ 1.6e-4);
    * directional finite differences agree within 2.7%.
* **Rows.** Each row is first compacted exactly by the frozen token router:
  * nq: `e2ecal2_rho0.1_c0.13`;
  * outlier: `e2ecal3full_rho0.35_c0.42`;
  * contradiction: `e2ecal3ncp_rho0.3_c0.38`;
  * scifact: `e2ecal3ncp_rho0.2_c0.245`.

  The eligible tokens are the routed body tokens it kept. They are 58% of the compacted row on nq,
  81% on outlier, 67% on contradiction and 76% on scifact. Prompt, query, answer and markers always
  get every layer. The token-router-only reference reproduces the grid: nq test ΔCE +0.0084 (flash
  compaction path) vs +0.0084 (soft path with all-ones gates).
* **Router variants.** The router reads RMSnorm(h) at each block input, with h detached:
  * `perlayer`: b_l + w_l·x/√d, 82k parameters;
  * `shared` (Prasann's suggestion): one w over [x ; onehot(l) ; is_attn], 2.6k parameters;
  * `shared_tok`: `shared` plus the token router's features (gold flag, 58 position features,
    embedding), 5.2k parameters.
* **Training.** Everything is as in the e2e token router:
  * hard-concrete gates, β 2/3 → 0.1, straight-through from 85%;
  * loss = answer CE + KL to the full model;
  * 40 epochs of 4 rows per step, Adam β₂ 0.8.

  **Size control** is per-step budget calibration: an offset c is solved so that the mean P(z > 0)
  over the step's eligible (token, layer) pairs equals ρ_t, where ρ_t goes from 1 to the target
  over the first half of training. Logits depend on c, because a layer's input depends on the
  earlier gates, so c is found by 2 no-grad fixed-point prepasses with the step's noise.
* **Eval.** A deterministic gate, logit + c* > 0, with one global c* bisected on the val rows to
  hit the target fraction of eligible pairs. It runs on the soft path with binary gates, which is
  exact.
* **Compute.** `collect_grid.forward_flops` extended to per-layer active-column counts (the
  quadratic attention term is included). At 2k, per-token cost is about the same for attention and
  GDN layers, so no cost weighting was needed.
* **Controls at the same budget:**
  * uniform random skip, 3 seeds;
  * skip every attention layer (keep 0.75);
  * skip odd layers (keep 0.5);
  * skip every GDN layer (keep 0.25);
  * a **static mask**: the router's own top round((1 − keep)·32) val-skipped layers, skipped for
    every eligible token.

### 9.2 Results (test, ΔCE vs the token router alone ± paired SE; FLOPs vs the full row)

| task | tok-only | keep | learned router | static mask | random | fixed rule |
|---|---|---|---|---|---|---|
| nq | +0.008, ×0.127 | 0.75 | perlayer −0.007 ± 0.004, shared −0.003 ± 0.002, **×0.109** | −0.003 | +0.30 | attn +0.024 |
| nq | | 0.5 | perlayer −0.006 ± 0.004 (seed 2: −0.006), shared −0.001 ± 0.003, **×0.091** | +0.018 / +0.002 | +0.69 | odd +0.38 |
| nq | | 0.25 | perlayer +0.036 ± 0.018, shared +0.042 ± 0.008, ×0.073 | +0.23 / +0.80 | +0.79 | gdn +0.89 |
| outlier | −0.010, ×0.407 | 0.75 | shared −0.009 ± 0.019, perlayer +0.018 ± 0.045, **×0.323** | +0.004 / +0.20 | +0.59 | attn +0.42 |
| outlier | | 0.5 | default: perlayer +0.34, shared +0.67 → **warm-start shared +0.030 ± 0.028**, ×0.241 | +0.12 | +0.68 | odd +0.53 |
| contradiction | +0.003, ×0.386 | 0.75 | perlayer +0.036 ± 0.023 (shared +0.126), ×0.320 | +0.036 | +0.98 | attn +0.58 |
| contradiction | | 0.5 | perlayer +0.63, shared +0.92; warm start +0.49 / +0.83 | +0.57 to +0.74 | +1.17 | odd +0.81 |
| scifact | +0.007, ×0.256 | 0.75 | perlayer −0.005 ± 0.005, shared −0.004 ± 0.005, **×0.208** | −0.001 | +0.25 | attn +0.14 |
| scifact | | 0.5 | perlayer +0.038 ± 0.030 (median ≈ 0), **warm-start shared +0.018 ± 0.018**, ×0.157 | +0.13 / +0.002 | +0.51 | odd +0.49 |

Keep 0.25 on outlier and contradiction fails for every variant (+0.39 to +1.07). JSONs are in
`runs/<task>/*.json`; `summarize_layerskip.py` prints the tables.

### 9.3 What it learned; what this says

* **Learned ≫ uninformed.** At every budget, random skipping costs +0.2 to +1.1 nats and fixed
  rules cost +0.02 to +1.1. This includes "skip all attention for compressed-doc tokens", which is
  +0.02 on nq but +0.14 to +0.58 on the other three tasks. The learned router sits at or near the
  token router's own ΔCE down to keep 0.5 on nq and scifact, and at 0.75 on outlier.
* **Mostly a layer choice.**
  * At keep 0.75 the router's own static mask is as good as the router on every task. Roughly the
    mask is:
    * nq: layers 8–9, 12 and 27–30;
    * outlier: late layers;
    * contradiction: layers 10, 13, 16–18, 24, 25 and 30;
    * scifact: mid and late layers.
  * Token dependence starts to matter at keep 0.5 on nq (static +0.018 vs learned −0.006, two
    seeds agree) and at keep 0.25 everywhere.
* **Per-layer pattern.** Attention and GDN layers are skipped at similar overall rates. Which layers
  are skipped depends on the task:
  * **nq** skips the early-middle block (layers 7–15) and layers 26–30. It never skips layers 0 and
    4–5, the 16–18 / 20–22 band, or 31.
  * **outlier** skips late layers (keep-0.5 warm start: 0.81 late vs 0.19 early).
  * **contradiction** at keep 0.75 **avoids attention layers** (skip rate 0.03 on attention vs 0.33
    on GDN).
  * No task skips layer 0 or layer 31 much.
* **Keep 0.5 on outlier and contradiction is an optimisation failure, not overfitting:** train ΔCE
  is as bad as test, and the router loses to its own static mask. Two fixes help:
  * a 2× longer anneal (outlier perlayer: +0.34 → +0.07);
  * a warm start from the keep-0.75 router with the target annealed 0.75 → 0.5 (outlier shared
    +0.67 → +0.03, scifact shared +0.19 → +0.02).

  On contradiction nothing tried works at keep 0.5. The warm start from shared began at a
  keep-0.75 router that was already worse than its static mask.
* **shared vs perlayer:** comparable. `shared` is best after a warm start. Its train–val gaps are
  small. `shared_tok` overfits on outlier (keep 0.75: train +0.027 → val +0.138 vs token-only), so
  it was dropped.
* **Scifact oddity:** the layer router *lowers* CE below token-only on train and val (val: token
  router alone +0.103 vs full, perlayer keep 0.5 −0.007). Skipping layers for the kept fragments
  partly undoes damage from the token drops.

### 9.4 Files

* `debug/learned_router/layerskip/`:
  * `layerskip_lib.py`: router variants, `Gater`, `forward_layerskip`, `forward_reference`,
    FLOP model;
  * `train_layerskip.py`: trainer plus in-process eval, baselines, static control, `--eval-only`,
    `--init-router`/`--rho-start` warm start, resume after preemption;
  * `test_layerskip_{cpu,gpu}.py`;
  * `run_layerskip_local.sbatch`;
  * `summarize_layerskip.py`;
  * `runs/`, `weights/`.
* wandb group
  [router-layerskip](https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/router-layerskip).
* No shared file was changed.
* **Infra note:** lorax GPUs IDX 0–1 fail CUDA init ("CUDA driver initialization failed");
  exclude lorax.

## 9. Oracle-then-classifier: iteration log (2026-09-30 →)

### Morning summary

**Status (latest).** Every test verdict below is ⚠ eval_size 16/16/8 (2k/8k/32k).

- **2k.** Recipe v5 had 11 beat / 6 match / 0 lose; table below.
- **Seed replicates.** One task falls outside parity:
  - At parity: rerank, textgroups, grouping.
  - niah is inconsistent: its train/val rows came from a different haystack regime than its test
    rows (fixed, `_stg`).
- **Speed.** Recipe **v6** (10 epochs, fast polish) holds the verdicts on nq and grouping at
  ~10 min/task (v5 ~29 min).
- **Length transfer** of the v5 2k routers (exact T2/T):
  - 8k: 3 beat / 10 match / 3 lose.
  - 32k: 1 beat / 2 match / 11 lose.
- **Feature ladder (textgroups).** The doc-only router (F0) is the only one at parity at 2k, 8k
  and 32k. It fails on niah, which needs every document's id tokens. Span routers don't transfer.
- **Primary matched comparison is now the paired budget** (`@pair`: per row, the bar's own
  realised keep count). Under it, 2 of the earlier "beats" become "matches".
- **niah on matched data.** v6 + 3 polish passes + marker init p 0.95 fixes the train-side
  optimisation (train paired +0.03). Test is still one catastrophic row out of 16 (+1.1 to
  +1.8), so it loses by a hair.

**Phase 2 results (2026-10-01).** All test numbers use the paired budget. ⚠ test64 = 64 rows per
task at 2k (qdmatch 54); testlong = 48 rows at 8k, 56 at 32k.

| router | 2k: beats / matches / loses | notes |
|---|---|---|
| per-task v6a (canonical) | 5 / 9 / 2 | loses niah, outlier_amzn (iteration 47) |
| **one shared router, all 17 tasks (xt17v6a)** | **1 / 13 / 2** | loses niah +1.27, reorder +0.13 (iteration 58) |
| one shared router, 3 tasks (xt3v6a) | 2 / 6 / 8 | iteration 56 |

- **32k transfer of the per-task v6a 2k routers (iteration 54).** nq matches; outlier, scifact
  and rerank beat at 32k. textgroups loses at 32k (+0.35); the doc-only router (F0) beats there
  (−0.097), but F0 loses outlier and rerank at 2k.
- **Markers-follow-doc (iterations 49–53).** Helps niah at 2k (+0.000 / +0.043 vs +0.075 /
  +0.082) and outlier at 2k. It hurts length transfer (outlier 32k +0.20, textgroups 8k +0.21).
  The whole-body variant also loses at 32k. Not adopted.
- **niah** is still open: the needle is cut on several of the 64 rows in every configuration.

**Open.**
1. niah: the needle is cut on several of the 64 test rows in every configuration.
2. textgroups at 32k needs doc-level features, while the id tasks need token-level ones. Shared
   router at 32k: not yet evaluated.
3. Run-to-run variance (GPU nondeterminism).
4. Pooled training with `--xtask-offload` spends 80% of its time on model swaps (polish/restart
   scoring interleaves tasks).

All 17 tasks at 2k, **recipe v5**, seed block 0 (grouping: block 10; niah: the identical v4 +
kd run). Paired Δ vs `gold_fl20p8_noslot` ± SE, ⚠ eval_size 16 per task:
- *exact* = per-row top-k at exactly the bar's T2/T;
- *val* = label-free global threshold matched on val rows.
Verdict rule as above, over all router points. Grid rows: `router_v5@exact`, `router_v5@val`.

| task | bar ×c ΔCE | exact: paired ± SE | val: ×c paired ± SE | verdict |
|---|---|---|---|---|
| rerank | ×0.407 +0.031 | +0.010 ± 0.019 | ×0.406 +0.016 ± 0.020 | **beats** (×0.337 within tol) |
| textgroups | ×0.607 +0.007 | +0.047 ± 0.077 | ×0.570 −0.068 ± 0.012 | **beats** |
| grouping | ×0.290 +0.021 | +0.011 ± 0.032 (block 0: +0.016 ± 0.013) | ×0.280 +0.003 ± 0.032 | **matches** (block 0 borderline) |
| niah | ×0.587 +0.000 | 0.000 ± 0.000 | (overshoots to ×0.786) | **matches** |
| absence | ×0.685 +0.890 | −0.016 ± 0.089 | ×0.676 +0.014 ± 0.096 | **matches** |
| nq | ×0.338 −0.002 | −0.000 ± 0.000 | ×0.278 −0.000 ± 0.000 | **beats** |
| contradiction | ×0.480 +0.000 | +0.001 ± 0.002 | ×0.481 +0.001 ± 0.002 | **beats** |
| scifact | ×0.436 +0.000 | +0.001 ± 0.000 | ×0.456 +0.000 ± 0.000 | **beats** |
| strmatch | ×0.627 −0.000 | +0.000 ± 0.000 | ×0.626 +0.000 ± 0.000 | **beats** |
| outlier | ×0.450 −0.034 | +0.001 ± 0.001 | ×0.396 +0.027 ± 0.027 | **matches** |
| fiqa | ×0.463 +0.013 | −0.003 ± 0.030 | ×0.491 −0.018 ± 0.029 | **beats** |
| msmarco | ×0.332 −0.005 | −0.003 ± 0.002 | ×0.332 −0.002 ± 0.002 | **beats** |
| obliq | ×0.515 +0.002 | −0.000 ± 0.002 | ×0.495 +0.001 ± 0.002 | **beats** |
| oolong | ×0.398 +0.077 | −0.006 ± 0.020 | ×0.392 −0.006 ± 0.020 | **beats** |
| outlier_amzn | ×0.443 +0.064 | +0.058 ± 0.067 | ×0.445 +0.042 ± 0.031 | **matches** |
| reorder | ×0.319 +0.417 | −0.286 ± 0.045 | ×0.301 −0.278 ± 0.045 | **beats** |
| qdmatch_hpqa | ×0.772 −0.000 | −0.000 ± 0.000 | ×0.773 −0.000 ± 0.000 | **matches** |

**Tally: 11 beat, 6 match, 0 lose** (grouping's other block is borderline at 1.2 SE).

**Seed replicates.**
- grouping: two v5 blocks (match / borderline).
- rerank: v2 s0/s1 match/beat; v3, v4, v5 all at parity.
- textgroups: v3 s0/s1 both beat.

**What's next.**
- 8k/32k (not started, per the order).
- Reduce cost: 3 restarts × 40 epochs ≈ 35–40 min/task; absence ~75 min.
- The tie problem of no-embedding logits: a tie-breaking feature, or ranking inside a tie class.

**Current canonical recipe: v5** (= v4 + `--keep-dropout 0.15`) — the e2e router (`train_e2e_router.py`; launcher template
`run_e2e_local.sbatch`):
- `--variant relpos_noemb`: bias, start/end offset one-hots, start/end deciles, doc-length buckets,
  gold flag, is_marker; **no embedding**. `--route-markers --marker-init-p 0.999`.
- `--rho-match-bar`: ρ = the target compaction's routed-keep fraction on the train rows.
- Batch-calibrated budget (bracket fix), CE + KL (w_kl 1), hard-concrete β 2/3 → 0.1, ST from
  85%, Adam lr 0.1 / β₂ 0.8, init p 0.95, 4 rows/step.
- 32 train rows, 40 epochs.
- `--restarts 3`: seeds s..s+2, the lowest val hard ΔCE at the target T2/T is kept.
- `--polish`: coordinate search over w_gold/w_marker shifts and w_pos/w_rel/w_emb scales, by exact
  train hard ΔCE at the target T2/T.
- `--polish-fine`: per-feature shifts of the position profile, on the selected restart.
- Test threshold: label-free, matched on 64 val rows to the target T2/T.
- `--keep-dropout 0.15`: at train time, a random 15% of routed tokens is removed after the
  gates are sampled (added in v5, for niah).
- Primary test comparison: per-row exact T2/T (`router_<name>@c<x>`), plus the val-matched global
  threshold.
- Cost: ~35–40 min per task on one H200 (absence ~75 min: 3k-token rows).

**Canonical recipe (v1).** One pipeline with the same hyperparameters for every task. Only the task
data and the target compaction change. Code: `debug/learned_router/oracle/`.

*Stage 1 — oracle labels* (`oracle.py`):
- Per row, free keep logits θ, one per routed token (body tokens + markers), trained through the
  exact relaxed deletion path of §7. Init p = 0.95 plus N(0, 0.01) noise. Three seeds run as one
  batched forward.
- Hard-concrete gates: β 0.5 → 0.1 within each stage; straight-through for the last 2 steps.
- Per-row budget calibration: the offset puts E[keep] at ρ_t, and ρ_t moves to the stage's ρ over
  the first half of the stage.
- Loss: answer CE + KL(full ‖ routed). Adam lr 0.3, β₂ 0.8.
- ρ stages 0.6 → 0.45 → 0.3 → 0.2 → 0.1, each warm-started from the previous one; **40 steps per
  stage**.
- Per seed, the mask is the smallest ρ whose hard-deletion ΔCE ≤ τ_row = max(0.1·CE_full,
  0.02). If none passes, the top-θ tokens are added back in 5% chunks.
- Soft label = the fraction of seeds that kept each token.
- Labels cover 32 train rows (plus 16 val rows for AUC only).

*Stage 2 — classifier* (`classify.py`):
- `relpos` linear router: bias, offset one-hots, start/end deciles, doc-length buckets, gold flag,
  is_marker, RMS embedding.
- Weighted logistic regression on the soft labels, weight 1 + (w_keep − 1)·label, with
  w_keep ∈ {2, 4} × λ_emb ∈ {1e-3, 1e-2, 1e-1}. LBFGS.
- The config is selected on 64 val rows by hard-deletion ΔCE at the heuristic's T2/T (a
  label-free threshold).

*Test:* the grid's 16 test rows at 2k, paired vs the bar, at the val-calibrated threshold for the
bar's T2/T and for 0.7× it. ⚠ eval_size 16.

**Success rule.** At the bar's T2/T, paired ΔCE ≤ max(paired SE, 0.005); or the bar's ΔCE (+
that tolerance) is reached at ≤ the bar's T2/T. The heuristic is only a check: it is never an
init, a candidate, a label source or a feature shape.

### Iteration log

| # | task | change (one hypothesis) | result | kept? |
|---|---|---|---|---|
| 0 | rerank | stage-1 search strength, rows 0–2: 10 → 40 steps per stage at lr 0.3 (lr 1.0 also tried) | 10 steps: oracle T2/T ≥ heuristic on rerank (e.g. 0.77–0.96 vs 0.41). 40 steps at lr 0.3: 0.25 at ΔCE −0.037 vs heuristic 0.44 at +0.028, all seed-rows ≤ heuristic T2/T. lr 1.0: 0.36/0.47, seed agreement 0.24–0.37 | 40 steps, lr 0.3 kept |
| 1 | rerank | v1 end to end: oracle labels on 32 train rows → weighted-LR classifier → test | oracle strong (train T2/T ≈ 0.33 at ΔCE ≈ −0.02 vs heuristic 0.41 at +0.05), but **labels unlearnable**: AUC 0.62, w_gold −0.17. The labels keep gold body 0.27 vs non-gold 0.29, id region 0.48, markers 0.55; seed agreement ~0.6. Test at the bar's T2/T: +0.096 vs +0.031 (paired +0.065 ± 0.024) ⚠ 16 → **loses** | recipe works mechanically; label quality is the problem |
| 2a | rerank | oracle loss KL-only (no answer CE), rows 0–5 | much weaker oracle: T2/T 0.71 (vs 0.27 with CE+KL on the same rows); seed agreement 0.10–0.17 on compacted rows | dropped |
| 2b | rerank | oracle gates: deterministic sigmoid (no sampling noise), rows 0–5 | oracle T2/T 0.30 (vs 0.27 hard-concrete), seed agreement 0.62–0.73 (vs ~0.6): only marginally more consistent, so no fix for learnability | dropped |
| 3 | rerank | (d) structure-regularised oracle: `--residual-l2 λ` (router + L2-penalised per-row residual) on 32 train rows, with relpos features, routed markers, ρ = the bar's routed keep (0.355); λ ∈ {none, 1, 0.1, 0.01, 0.001} | no residual (plain e2e with the new setup): test paired +0.040 ± 0.020 at ×0.395 (old r3 router: +0.22) — a big step, but still **loses**. λ=1 +0.028 ± 0.017; λ=0.1 +0.046; λ=0.001 +0.119 (the residual absorbs everything). Val gold keep 0.47–0.56 vs heuristic 1.0; b −62 vs w_gold +5 | residual dropped; the plain e2e setup (relpos + routed markers + ρ matched + 32 rows) is kept as the base |
| 3d | rerank | diagnostic: the no-residual router with w_gold += 30 (gold forced whole) | val paired vs the heuristic +0.041 → +0.028. Test: at ×0.406 paired +0.019 ± 0.016 (just outside tol); at ×0.283 ΔCE +0.040 vs the bar's +0.031 (paired +0.009 ± 0.023) → passes criterion (b) | motivates iteration 4: the relaxed gradient undervalues keeping a document WHOLE |
| bug | all | `calibrate_offset` bisected over a FIXED [−60, 60] bracket; once b drifted, the offset saturated and the budget went unenforced. Audit of all 40 calibrated runs: **only rerank** was affected, and every rerank run was (r4, the ρ=0.5 retry, all residual-sweep runs): bad from 27–46% of training, \|keep − ρ_t\| up to 0.68. The bracket now comes from the logit range (also in `match_comp_offset` and the oracle) | fixed 2026-09-30; rerank rerun |
| 4 | rerank | fixed calibration, no residual; ± a zeroth-order **hard-loss polish** after training (`--polish`: coordinate search over w_gold/w_marker shifts and w_pos/w_rel/w_emb scales by exact train hard ΔCE, pooled keep fixed at ρ, 2 passes) | fixed only: test ×0.403 paired +0.035 ± 0.028 (loses by the rule, within 1.3 SE); train paired +0.006, val +0.035, gold keep 0.57. **Fixed + polish: test ×0.406 paired −0.007 ± 0.025 → MATCHES**; val paired +0.002 ± 0.008, gold keep 0.82. Polish picked w_gold +10, w_pos ×0.25, w_rel ×0.25, w_emb ×2 (train hard ΔCE +0.065 → +0.043) | **kept → recipe v2** |
| 5 | rerank | recipe v2 seed replicate (seed 1) | test ×0.402 paired −0.018 ± 0.025; ×0.341 +0.036 (paired +0.005 ± 0.029) → **beats**; val paired +0.010, gold keep 0.89. Seed 0 matches, seed 1 beats | consistent |
| 6 | textgroups | recipe v2, seed 0 (ρ = 0.542) | train paired +0.028 ± 0.035, **val +0.089 ± 0.025** (generalisation gap +0.061). Test: at ×0.599 +0.119 ± 0.038 (bad); at ×0.499 ΔCE +0.018 vs the bar's +0.007 (paired +0.011 ± 0.037), a noisy "beats" by criterion (b) on a non-monotone curve. Val disagrees, so not counted as a win | next: 64 train rows (generalisation) |
| 7 | textgroups | v2 seed 1 (replicate), and v2 with 64 train rows (40 epochs, same steps) | seed 1: train +0.018, val +0.157, test ×0.599 +0.098 ± 0.037 → loses. 64 rows: gap shrinks (+0.061 → +0.020) but the train fit worsens (+0.047); test ×0.589 +0.075 ± 0.032 → **loses** | 64 rows not kept (no test gain) |
| 8 | textgroups | **no embedding term** (`relpos_noemb`: bias + position one-hots + deciles + length buckets + gold + marker), otherwise v2 | train paired vs the heuristic −0.095 ± 0.009, val −0.082 ± 0.005 (gap +0.013). **Test: ×0.621 ΔCE −0.073 (paired −0.080 ± 0.013); ×0.498 −0.054 (paired −0.061 ± 0.008) → BEATS** | kept (the 2560-d embedding weights were the overfitting term); re-verify on rerank |
| 8f | textgroups | fine polish (per-feature position shifts) | cancelled: superseded by iteration 8 | — |
| 9 | rerank, textgroups | re-verify no-emb (**recipe v3** = v2 with `relpos_noemb`) | rerank s0: ×0.404 paired +0.016 ± 0.023 → matches (gold keep 1.0; with emb: s0 matches, s1 beats). textgroups s1: ×0.571 −0.007 ± 0.017, ×0.496 −0.035 ± 0.034 → beats (s0 beats too) | **v3 adopted** |
| 10 | grouping | recipe v3, seed 0 (gold-blind task; ρ = 0.234) | train paired vs the heuristic +0.042 ± 0.019 (optimisation), val +0.065. Test ×0.285 +0.102 ± 0.036 → **loses**. Polish helped only a little (train hard +0.088 → +0.081) | next: fine polish (per-feature position shifts) and 2× epochs, in parallel |
| 11 | grouping | fine polish (`--polish-fine`: per-feature additive shifts {−10, −3, +3, +10} of start/end one-hots 0–15 and the 20 deciles, 1 pass, hard train loss at fixed ρ) in a fresh seed-0 run; plus 2× epochs (160) in another fresh run; plus a fine re-polish of iteration 10's router | fresh fine-polish run: **test ×0.285 paired −0.021 ± 0.012 → beats**, val +0.002. But its training landed far better *before* polish (train hard +0.021 vs +0.088 for iteration 10, same seed): training is chaotic run to run (GPU nondeterminism + stochastic gates). Re-polishing iteration 10's router: train +0.081 → +0.057, val +0.065 → +0.033, test +0.053 ± 0.026 (still loses). 160 epochs: train +0.017, val +0.068, test ×0.246 +0.041 ± 0.020 → loses | fine polish helps but doesn't decide; run-to-run variance dominates → iteration 12 |
| 12 | grouping | **restarts**: 3 independent trainings (seeds s, s+1, s+2), each coarse-polished; keep the one with the lowest VAL hard ΔCE at pooled keep ρ; fine polish on the selected one (`--restarts 3 --polish-fine`) = **recipe v4 candidate** | running (2 replicate blocks: seeds 0–2 and 10–12). Among iterations 10/11's three grouping trainings, val selection would have picked the winner (val +0.002 vs +0.065/+0.068) | — |
| 12r | grouping | v4 result (2 blocks) | block s0: restart val ΔCE +0.031/+0.041/+0.085 → chose s0; after fine polish train paired −0.019, val +0.012; test ×0.272 +0.055 (paired +0.035 ± 0.025) → loses. Block s10: chose s11; train −0.040, val +0.015; test ×0.283 +0.035 (paired +0.014 ± 0.024) → **matches**. Two flaws found: (1) restarts were compared at pooled keep ρ, not at matched T2/T (their val T2/T ranged 0.29–0.35); (2) the polish held pooled keep fixed, so it bought ΔCE with T2/T drift (grouping train T2/T 0.29 → 0.35). Both now hold T2/T at the target: `match_comp_offset` on train rows for the polish, on val rows for selection. Each restart takes ~20 min (3 = 70 min: too slow) | fix both; next: v4 with the fixes and 40 epochs per restart |
| 13 | grouping | v4 with both fixes and 40 epochs per restart | block s10: restarts' val ΔCE +0.054/+0.039/+0.041 (heuristic on val +0.020) → chose s11; train paired −0.034, val +0.016 (gap +0.049); test ×0.284 paired +0.026 ± 0.028 → **matches** (borderline). Block s0: restarts' val +0.051/+0.059/… (still running) | restarts and fine polish don't clearly help → not adopted (simplicity); grouping parked as **borderline** (4 ideas tried: base v3, fine polish, 2× epochs, restarts). Its failure mode is generalisation (train better than the heuristic, val/test worse) |
| lock | infra | a preempted-and-requeued job found its own earlier task lock and exited ("owned by another job" = itself) | lock reclaimed when owner == this job id (`run_e2e_local.sbatch`, `oracle/run_oracle.sbatch`) | fixed |
| 13r | grouping | v4 (fixed) block s0 result | train −0.028, val +0.028; **test ×0.290 paired −0.012 ± 0.020 → beats**; ×0.233 +0.010 ± 0.022. Both v4-fixed blocks reach parity (s0 beats, s10 matches), vs single v3 runs 1 loses / 1 beats | **recipe v4 adopted** (supersedes the "not adopted" in 13): v3 + 3 val-selected restarts at 40 epochs + fine polish on the selected one; ~35–40 min/task |
| 14 | rerank, textgroups, niah | v3 on the final code (T2/T-held polish, bracket fix), seed 0 | rerank: ×0.408 paired +0.006 ± 0.022 → **matches**. textgroups: ×0.586 −0.055 ± 0.013 → **beats**. niah: train +0.054, **val +0.341**, gold keep 0.60 vs 1.0; test ×0.617 +0.732 ± 0.267 → **loses** (generalisation: the needle doc is only partly kept on unseen rows) | wins hold on the final code |
| 15 | rerank, textgroups, niah | recipe v4 (fixed), seed block 0 | rerank: ×0.388 paired −0.002 ± 0.015 → **beats**; val +0.003. textgroups: ×0.582 +0.007 ± 0.024 → **matches** (v3 beat, −0.055). niah: train paired **0.000**, val +0.092, test +0.433 ± 0.201 → **loses**. Router keeps 64% of the needle on train AND val: on train rows a partial needle already gives ΔCE 0 (CE_full ≈ 0, objective saturated), so nothing rewards keeping it whole | v4 holds on rerank/textgroups; niah → 64 rows (16a) and keep-dropout (16b) |
| 16a | niah | v4 with 64 train rows (20 epochs per restart) | restarts' val ΔCE +0.45/+0.26/+0.30; test ×0.504 +0.991 ± 0.269 → loses | dropped |
| 16b | niah | v4 + **keep-dropout 0.15** (train-time: after the gates are sampled, a random 15% of routed tokens is removed as well, resampled each step; eval unchanged) | restarts' val ΔCE **−0.001**/+0.20/+0.15 → chose s0; val paired +0.015. Test: at ×0.577 ΔCE 0.000 (paired −0.000 ± 0.000) → **matches** the bar (×0.587 +0.000). Caveat: with no embedding, logits tie by position class, so a val-matched threshold can land on the far side of a tie class on test rows. Targets 0.587/0.499 realised 0.786/0.707 on test, but 0.578/0.499 on val | **keep-dropout kept → recipe v5 candidate**; re-verifying on rerank/textgroups/grouping; absence runs on v5 |
| 17 | rerank, textgroups, grouping | re-verify v5 (= v4 + keep-dropout 0.15) | rerank: ×0.406 +0.016 ± 0.020, ×0.337 +0.016 ± 0.025 → **beats** (by criterion b); gold keep 1.00. textgroups: restarts' val +0.33/+0.017/+0.43 → chose s1; test ×0.570 −0.068 ± 0.012 → **beats**. grouping: val +0.039 vs the heuristic's +0.020; test at the val-matched threshold realised ×0.253 (target 0.290, tie overshoot) +0.045, paired +0.024 ± 0.012 → loses at that T2/T. Per-row exact-T2/T re-score queued (`router_<name>@c<x>`) | kd holds on rerank/textgroups; grouping pending exact-T2/T check |
| 18 | grouping, niah | **per-row exact-T2/T scoring** (`router_<name>@c<x>`: per test row, the top-k routed tokens so that the row's T2/T equals x exactly; label-free). Needed because no-embedding logits tie by position class, so a global threshold matched on val can jump a whole tie class on test | grouping v4 block 0 at exactly 0.29: −0.011 ± 0.020 → **matches**. grouping v5 block 0 at 0.29: +0.016 ± 0.013 → loses by 1.2 SE. niah v4+kd at exactly 0.587: 0.000 ± 0.000 → **matches** (at 0.411: +0.68) | exact-T2/T is now the primary matched comparison (the val-matched global threshold is reported too); grouping v5 second block queued |
| 19 | absence | recipe v5, seed block 0 (3k-token rows, `--no-batch-rows`; ~25 min per restart) | restarts' val ΔCE +0.91/+1.09/+0.92 (heuristic on val ≈ +0.90) → chose s0; train paired −0.168 ± 0.067, val −0.007 ± 0.069. Test: ×0.690 −0.020 ± 0.094; ×0.676 +0.014 ± 0.096 → **matches** (wide SE: absence's ΔCE is ~0.9 nats) | absence at parity (old router: +0.71) |
| 20 | grouping | v5 second block (seeds 10–12) | exact T2/T 0.290: +0.011 ± 0.032; val-matched ×0.280 +0.003 ± 0.032 → **matches**; train −0.037 / val +0.028. With block 0 (+0.016 ± 0.013 exact), grouping on v5 sits at parity within noise | v5 stays canonical |
| 21 | 12 winning tasks | v5 regression check on nq, contradiction, scifact, strmatch, outlier, fiqa, msmarco, obliq, oolong, outlier_amzn, reorder, qdmatch_hpqa | running (3 GPUs, ~40 min each) | — |
| 21r | all 17 | v5 regression check, 12 formerly-winning tasks (seed block 0); the 5 former losers from iterations 16–20 | **11 beat, 5 match, 1 borderline** (grouping block 0 +0.016 ± 0.013 at exact T2/T; block 10 matches). No former winner regressed below parity. Four changed from beats to matches: outlier (exact +0.001 ± 0.001), outlier_amzn (+0.058 ± 0.067), qdmatch_hpqa (0.000), niah. Exact per-row T2/T is not tie-free either: on textgroups it gives +0.047 ± 0.077 at 0.607 vs −0.068 ± 0.012 for the val-matched global threshold at ×0.570, because per-row top-k splits tied classes arbitrarily | grid rows `router_v5@exact` / `router_v5@val` exported (`export_recipe_grid.py` → `results_router_recipe/`) |
| 22 | all 17 | **length transfer** of the v5 2k routers (eval only; `summarize_len5.py` → `headline_v5len.json`): (a) per-row exact T2/T = the bar's at that rung; (b) the same routed-keep fraction as at 2k (`@k<ρ>`) | (a) 8k: **3 beat, 10 match, 3 lose** (grouping +0.054 ± 0.014, strmatch +0.021 ± 0.014, textgroups +0.051 ± 0.050). 32k: **1 beat, 2 match, 11 lose** (textgroups +0.49, scifact +0.17, outlier_amzn +0.17, fiqa +0.27, …). (b) keeps more than the bar (which compacts harder with length), so it mostly loses on T2/T: 8k 2 beat / 1 match / 13 lose, 32k 1 match / 13 lose. ⚠ eval_size 16 (8k) / 8 (32k); niah pending | 2k-trained routers transfer to 8k, not to 32k |
| 23 | nq, grouping | **speed**: 10 epochs per restart + fast polish (1 coarse pass; fine shifts ±5; candidates scored on 16 of the 32 train rows) | nq: **beats** (exact 0.000, val-matched ×0.330), **573 s** (setup 37 s, 3 × 130 s restarts incl. coarse polish 26 s, fine polish 106 s, eval ~50 s). grouping (seeds 10–12): **beats** (exact +0.012 ± 0.021; val-matched ×0.252 +0.015 ± 0.020), **639 s**. Same verdicts as v5 at ~1/3 of the time (v5: nq 1722 s) | **recipe v6** = v5 with `EPOCHS=10 --polish-fast --polish-rows 16` (~10 min/task) |
| 24 | textgroups, scifact, outlier_amzn, rerank | 32k diagnosis (`compare_keep.py`, at the bar's T2/T, 8 rows): kept fraction, router vs bar | id region (non-gold j<8): router 0.54 / 0.68 / 0.92 / 0.71 vs bar 1.00 on all four. Markers: router 0.51 (textgroups) and 0.97 (outlier_amzn) vs bar 0.01–0.03. Gold body: scifact 0.49 vs 1.00; others ~0.99. At 32k (~16× more docs) the per-document fixed cost dominates: the bar spends it on every doc's id prefix; the 2k-trained ranking spends it on markers, or drops ids and gold | next: train on 2k + 8k mixed rows (textgroups first) |
| 25 | rerank, textgroups, grouping, niah | v5 seed-block-10 replicates | rerank: exact +0.054 ± 0.043, ×0.402 +0.016 ± 0.023 → matches. textgroups: exact −0.060 ± 0.012 → **beats**. grouping: +0.011 ± 0.032 → matches. **niah: exact +0.310 ± 0.186 → loses** (val paired +0.093: no restart generalised; block 0's win came from one lucky restart) | niah is not yet consistent → kd strength test (0.15 vs 0.3, 2 seed blocks each) on v6 |
| 26 | textgroups | v6 (2k only), scored at 2k/8k/32k | 2k: exact −0.014 ± 0.012 → beats (704 s). 8k: +0.092 ± 0.039 → loses. 32k: +0.205 ± 0.079 → loses (v5: +0.49) | baseline for the mixed-rung test |
| 27 | niah | keep-dropout strength on v6: 0.15 vs 0.3, seed blocks 0 and 10 | kd 0.15: s0 matches (exact +0.027 ± 0.027), s10 **loses** (+0.207 ± 0.203). kd 0.3: s0 **loses** (+0.489 ± 0.280), s10 beats (0.000 at ×0.551). Seed-dependent, no consistent winner; val paired ranges −0.009…+0.166 | kd strength doesn't decide; next: polish scored on val |
| 28 | textgroups | v6 trained on **2k + 8k mixed rows** (16 + 16; per-row path, since batched 8k rows OOM) vs 2k-only | 2k: +0.009 ± 0.008 (2k-only −0.014 ± 0.012). 8k: **+0.031 ± 0.015** (2k-only +0.092 ± 0.039; worst row +0.14 vs +0.43). 32k: +0.336 ± 0.107 (2k-only +0.205 ± 0.079) | mixed training helps 8k only; doesn't fix 32k → feature ladder |
| 29 | textgroups | **feature ladder** on v6, trained at 2k only, exact T2/T at 2k / 8k / 32k (⚠ 16/16/8). F0 = doc-only (bias, gold, doc-length bucket, is_marker); F1 = F0 + start/end deciles; F2 = v6 (+ offset one-hots); F2 + RMS embedding with ‖w_emb‖ ≤ 0.5 / 2 | **F0: 2k −0.019 ± 0.012 (beats), 8k −0.073 ± 0.026 (beats), 32k +0.055 ± 0.058 (matches)**; train/val paired −0.066/−0.046. F1: +0.018 / +0.024 / +0.135; its 32k keep set reproduces the bar almost exactly (id region 0.99, markers 0.03, body 0.20). F2: −0.014 / +0.092 / +0.205. emb≤0.5: +0.036 / +0.116 / +0.380. emb≤2: +0.054 / +0.106 / +0.123. At 32k, F0 keeps gold whole and whole non-gold documents (id region 0.27, body 0.23) instead of fragments of every document | **coarser transfers best**: F0 is the only variant at parity at all three lengths; test on niah and scifact next |
| 30 | niah, grouping | **polish scored on val** (`--polish-on val`, 16 val rows, at the val target T2/T) | grouping s10: exact −0.010 ± 0.010 (0 rows >0.1) → **beats** (train-scored polish: +0.012 ± 0.021, 2 rows >0.1); train/val −0.013/−0.004. niah s10: 0.000 → **beats** (train-scored: +0.207). niah s0: +0.622 ± 0.268 (5 rows >0.1) → loses; train +0.966, val +0.418 on all 64 val rows, i.e. overfit to the 16 polish rows | mixed; next: polish on all 64 val rows |
| 31 | niah, scifact | feature ladder F0 (doc-only) vs F2 (v6), trained at 2k, exact T2/T at 2k/8k/32k | **niah F0 fails everywhere**: +1.26 / +1.32 / +1.03 (s0), +1.34 / +1.24 / +1.59 (s10), 8–9 of 16 rows >0.1. The answer is a doc id among 40 short docs, so whole-document keep/drop cannot keep every id within budget. niah F2 s0: 2k 0.000 (matches), 8k +0.307 ± 0.206, 32k +1.35 ± 0.57. scifact: F0 +0.001 / −0.001 / +0.006 ± 0.006; F2 0.000 / −0.000 / −0.004 ± 0.007 → parity at all lengths (v5 scifact 32k was +0.167) | no single feature level wins: textgroups wants doc-level, niah needs per-token ids → span-level variants (iteration 32) as the middle ground |
| 32p | niah | polish scored on all 64 val rows (`--polish-on val --polish-rows 0`), seed blocks 0 and 10 | val paired vs the heuristic **−0.044 / −0.033** (better than the heuristic on val) but test **+1.18 ± 0.40 / +0.61 ± 0.33**: median 0.000, 3–6 of 16 rows at +3 to +4 nats (needle dropped) | val→test does not transfer on niah → data check (next row) |
| data | all | **train/val vs test distribution check** (mean #docs, mean doc chars; 2k; test = the grid's first 16 rows, train = `router_train_e2e`, fetched from later rows of the same HF split) | **niah is shifted**: test haystack docs average 63 chars vs 147 in train (val 126), while the gold doc stays ~153 chars. On test rows the needle is a ~2.4× longer outlier doc; on train/val it is typical. absence (171 vs 136 chars) and fiqa (991 vs 795) are mildly shifted; the other 14 tasks match (±2%). This explains niah's pattern (train/val fine, test tail rows fail) | niah's train/val rows are not iid with its test rows; flagged for a decision (re-fetch niah train/val from the test rows' regime, e.g. staged rows 16–63 after the disjointness hash check) |
| 33 | textgroups | **span-level** variants (span8, span4): one shared gate/score per contiguous 8- or 4-token span (markers own units); features bias, gold, doc-length bucket, span index from start/end, span start/end decile; span-aware top-k (rounds to the nearest span) | span8: +0.362 / +0.830 / +0.835 (2k/8k/32k); span4: +0.154 / +0.191 / +0.200 → both **lose**, far worse than F0 (−0.019 / −0.073 / +0.055) and F2 at 2k | spans don't help textgroups |
| data-fix | niah, absence, fiqa | **provenance**: the HF r2k split of niah is not homogeneous. Mean doc chars per 64-row block are [66, 137, 157, 147, 203, 66, 146, 156]: rows 0–63 and 320–383 are a short-haystack regime, the rest long, and the split is unshuffled. The grid's test rows (the first staged rows) are all short-regime; `fetch_train_rows.py` took train/val from rows 64+ (long regime). absence/fiqa have no regime blocks (block means within ±10%); their first 16 test rows are just longer than average. Fix: `fetch_staged_rows.py` builds `router_train_stg` / `router_val_stg` from staged 2k rows 16–63 (the same file and regime as the test rows); hashed against all 40 test rows at every rung: 0 dropped; 32 train + 16 val. niah train/val now average 67 chars/doc (test 63) | **all earlier niah router verdicts, and the absence/fiqa ones, were trained on shifted data**; reruns on `_stg` launched (niah v6 s0/s10 + span4, absence, fiqa). Grid numbers themselves are unaffected: every scheme is scored on the same staged rows |
| 34 | textgroups, niah, grouping | **paired-budget scoring** (`router_<name>@pair`): per row, keep as many routed tokens as `gold_fl20p8_noslot` keeps on that same row (its realised per-row T2/T, not the mean); primary matched comparison from here on (`summarize_pair.py`) | 2k: textgroups F0 −0.029 ± 0.013 **beats**; v6 +0.001 ± 0.013 matches (was beats under mean-exact); v5 +0.015 ± 0.074 matches; span4 +0.145 / span8 +0.312 lose (pairing doesn't rescue spans). niah: v4+kd s0 −0.000 and v6 kd0.15 s0 +0.001 match; matched-data v6 s0/s10 and span4 lose (+0.24 / +1.16 / +0.74). grouping: v6 polish-on-val s10 −0.008 ± 0.010 matches (was beats); v5 s10 +0.008 matches; v5 s0 +0.015 ± 0.013 loses. 8k/32k: textgroups F0 −0.057 ± 0.024 (beats) / +0.057 ± 0.046 (loses narrowly); grouping v6-pv matches at both (+0.008, +0.001); niah loses everywhere | flips vs mean-exact: 2 beats → matches (textgroups v6, grouping v6-pv); none flip to a win |
| 35 | niah | matched data (`_stg`, 32 train / 16 val, same regime as test), v6 | s0: train paired +0.018, val 0.000, test +0.249 ± 0.114 (4 rows >0.1). s10: **train +0.672**, val +0.895 → optimisation failure on the short-haystack regime. span4: train +1.08. The heuristic keeps each of the 40 short docs' first 8 tokens (≈ half of each doc) plus the needle whole; the router doesn't find that | kd 0 / kd 0.3 on matched data running |
| 36 | textgroups | span routers with keep-dropout 0, or token-level kd after broadcasting (`--kd-token`) | span8 kd0: 2k +0.011 ± 0.013 **matches** (unit-level kd: +0.362), 8k +0.174, 32k +0.189. span8 token-kd: +0.022 / +0.084 / +0.109. span4 kd0: +0.109 / +0.236 / +0.323; span4 token-kd: +0.014 ± 0.012 / +0.085 / +0.075. Train/val paired all ≈ 0 to +0.04 | unit-level kd was the span8 2k culprit (confirmed), but no span variant transfers: 8k/32k lose everywhere, while F0 beats at 8k |
| 37 | niah | matched data, kd 0 and 0.3 (v6) | kd 0: s0 +0.088 ± 0.088 (1 row +1.41) → loses; s10 +0.093 ± 0.093 (1 row +1.49) → matches. kd 0.3: +0.189 / +0.515 → lose. Gold labels on the staged rows verified correct (gold → the paraphrased contradicting needle). v6 stg s10: w_gold +4.96, train gold keep 0.75 vs 1.00. The coarse polish already tries w_gold +10…+40 and rejects them at fixed T2/T; v6's fast polish runs one pass, so gold is never revisited after markers/positions change | next: `--polish-passes 3` |
| 38 | absence, fiqa | v6 on matched data (`_stg`) | absence: 2k −0.048 ± 0.111 (8 rows >0.1 either way), 8k −0.092 ± 0.111 → **matches** both; train/val −0.245/−0.148. fiqa: 2k −0.006 ± 0.046 matches, 8k −0.192 ± 0.072 **beats**, 32k +0.299 ± 0.188 loses | parity at 2k/8k on matched data |
| 39 | niah | matched data, v6 with **3 coarse polish passes** (`--polish-passes 3`) | s0: 0.000 ± 0.000, 0 rows >0.1 → **matches**; train/val +0.047/+0.000. s10: +0.303 ± 0.169 (3 rows >0.1) → loses; train +0.203 (1 pass: +0.672). The extra passes let gold climb to +40 once markers/positions have moved (restart 0: train hard +0.49 → +0.135), but two of the three restarts start at train hard +1.06/+1.16 and polish can't rescue them. Markers −10 is the polish's biggest single gain | helps, not sufficient; next: marker init p 0.95 instead of 0.999 (sticky markers eat budget: ~80 markers/row) |
| 40 | niah | matched data, 3 polish passes + **marker init p 0.95** (instead of 0.999) | train paired now **+0.032 / +0.030** (s0/s10; with p 0.999: +0.047/+0.203, one pass: +0.67). Test: s0 +0.116 ± 0.115, s10 +0.070 ± 0.070. Each is **one** catastrophic row (+1.8 / +1.1: needle cut), median 0.000 → loses by a hair. Val +0.189 / +0.019 | the optimisation is fixed (sticky markers were eating the budget); the tail row remains. Next: re-verify marker init 0.95 on strmatch/textgroups/grouping before adopting |
| 41 | strmatch, textgroups, grouping, rerank, nq | re-verify marker init 0.95 + 3 polish passes (**recipe v7** candidate = v6 + `--polish-passes 3 --marker-init-p 0.95`) | mean-exact / paired budget: strmatch +0.000 / +0.0001 matches (the p 0.999 init was introduced for strmatch; not needed with routed markers + ρ matching). textgroups −0.041 ± 0.017 / −0.020 ± 0.025 beats. grouping +0.006 ± 0.034 / +0.013 ± 0.033 matches. rerank +0.007 ± 0.024 / +0.0005 ± 0.023 beats (×0.393 −0.001). nq −0.000 / −0.0004 beats | **v7 adopted**; full 17-task 2k run on v7 next (niah/absence/fiqa on `_stg` data) |
| 42 | all 17 | **recipe v7** full 2k run, seed 0 (niah/absence/fiqa on matched `_stg` data), primary = paired budget (`@pair`) | paired: **beats 3** (fiqa −0.074 ± 0.068, obliq −0.005 ± 0.003, reorder −0.271 ± 0.048); **matches 9** (contradiction, grouping +0.013 ± 0.020, nq, qdmatch_hpqa, rerank −0.001 ± 0.017, scifact, strmatch, textgroups −0.008 ± 0.009, absence +0.067 ± 0.113); **loses 5**: niah +0.367 ± 0.235 (3 rows >0.1, max +3.57), outlier_amzn +0.067 ± 0.034 (4 rows), outlier +0.041 ± 0.028 (2 rows), msmarco +0.023 ± 0.023 (1 row, +0.36), oolong +0.008 ± 0.006 (0 rows; loses on the 0.005 floor). Over all router points (mean-exact, val-matched, lower T2/T): 15 beat / 1 match / 1 lose (niah) | outlier/outlier_amzn/msmarco look worse than v5, which beat or matched there, but v5 was never scored under `@pair`; a like-for-like `@pair` re-score of v5 on those tasks is needed before blaming marker init 0.95 |
| 43 | outlier, outlier_amzn, msmarco, oolong | like-for-like `@pair` re-score, v5 vs v7 (same test rows) | v5 **matches** on all four (outlier +0.001 ± 0.001, outlier_amzn +0.036 ± 0.048, msmarco −0.002 ± 0.001, oolong −0.010 ± 0.018); v7 **loses** on all four (+0.041, +0.067, +0.023, +0.008) → **v7 regressed these vs v5**. v6 → v7 changed 3 knobs from v5 (10 epochs, fast polish, marker init 0.95 + 3 passes) | ablation: v6 (fast, marker 0.999) and v7 at 40 epochs on outlier/msmarco |
| 44 | outlier, msmarco | ablation of the v7 regression (`@pair`) | v6 (10 epochs, fast polish, marker init 0.999, 1 pass): outlier −0.000 ± 0.001, msmarco +0.002 ± 0.005 → **match**. v7 at 40 epochs (marker 0.95, 3 passes): msmarco −0.003 ± 0.002 matches, outlier +0.039 ± 0.038 (one +0.61 row) loses. v7 at 10 epochs: both lose | **marker init 0.95 is the culprit** (it helps niah's training but hurts outlier/msmarco/oolong/outlier_amzn) → revert to v6 as canonical; niah stays the exception; full v6 17-task `@pair` run launched |
| 45 | all 17 | **recipe v6 full 2k run** (seed 0, `@pair` primary; niah/absence/fiqa on `_stg`) + 3-pass arm on outlier/msmarco | **beats 3** (oolong −0.027 ± 0.022, reorder −0.126 ± 0.049, textgroups −0.045 ± 0.010); **matches 9** (absence −0.071 ± 0.111, contradiction, fiqa +0.004 ± 0.088, msmarco, nq, outlier_amzn −0.001 ± 0.036, qdmatch_hpqa, scifact, strmatch); **loses 5**: niah +0.223 ± 0.133, grouping +0.050 ± 0.027, outlier +0.044 ± 0.023, rerank +0.033 ± 0.018, obliq +0.009 ± 0.008 (on the 0.005 floor). 3 passes: msmarco −0.002 matches, outlier +0.038 ± 0.029 loses. **Run-to-run variance is as large as the effects**: outlier under the identical v6 config and seed matched in iteration 44 (−0.000 ± 0.001) and loses here (+0.044); grouping and rerank have matched or beaten in other runs of the same recipe | at 2k, single runs land 12/17 at parity; the losers flip between runs. Next step is variance reduction (more restarts, or averaging restarts' weights), not new knobs |
| 46 | outlier, grouping | **determinism**: identical v6 config and seed, re-run (`@pair`, 16 rows) | outlier, 5 runs: −0.000 / +0.044 / +0.051 / +0.015 / +0.007 (3 match, 2 lose). grouping, 4 runs: +0.050 / +0.013 / +0.039 / −0.000 (2 match, 2 lose). The divergence is in training itself: outlier restart 0's val ΔCE is 0.032 / 0.005 / 0.007 / 0.001 across runs. Data order, gates, keep-dropout and polish are CPU-seeded and deterministic, and requeued jobs restart from scratch. **Source: GPU numerics** (FLA Triton autotune choosing configs by timing, atomic-add backward reductions, SDPA backward), amplified by the chaotic calibrated training | not bit-reproducible without deterministic kernels + pinned autotune; same-seed spread ≈ ±0.025 paired on these two tasks, as large as the verdict margins → bigger test set (64 rows) + restart weight averaging |
| test64 | 16 tasks | **bigger test set**: `build_test64.py`, 64 fresh rows per task from the HF r2k split past every block in use (rows 192+; niah rows 320–383 = its short-haystack regime), hash-checked against all staged / train / val rows at every rung | 64 rows for 15 tasks; qdmatch_hpqa 54 (254 dropped for shared gold docs); **obliq 0**: its r2k split has only 123 rows, all used, so obliq stays at 16 test rows (⚠) | eval-only; bar re-scored on the same rows |
| 47 | 16 tasks | **final scoring on test64** (paired budget, 64 rows; qdmatch 54; `summarize_t64.py`): v6 single run; and, from one job each, the **restart-averaged** router vs that job's best single restart | v6 single run: **6 beat, 8 match, 2 lose** (grouping +0.031 ± 0.011, niah +0.200 ± 0.063). restart-avg: **5 beat, 9 match, 2 lose** (niah +0.097 ± 0.042, outlier_amzn +0.039 ± 0.022). same-job best restart: 4 beat, 10 match, 2 lose (niah +0.104, outlier_amzn +0.023). Avg ≥ best on 12 of 16 (clearly better on textgroups −0.063 vs +0.007, reorder −0.304 vs −0.283, fiqa −0.068 vs −0.048, absence −0.138 vs −0.059); worse on outlier_amzn | averaging helps a little and is cheap (no extra training) → kept as the default (**recipe v6a** = v6 + `--restart-avg all`). Job-to-job variance remains (outlier_amzn −0.059 in the v6 run vs +0.039 in the avg job) |
| 48 | outlier, grouping | **calmer optimiser**: lr 0.03, Adam β₂ 0.99, same steps, 3 same-seed reruns each, test64 | outlier +0.008 / +0.024 / −0.006; grouping −0.001 / +0.020 / +0.042. Spread 0.03–0.04, no smaller than the default optimiser's, and no better mean | dropped |
| grid | all 17 | grid rows `router_v6@pair` / `router_v6@exact` / `router_v6@val` (v6 run of iteration 45, on the grid's standard 16/16/8 test rows) exported next to `router_v5@*` (`export_recipe_grid.py --label v6 --file-tag _v6`); HTML regenerated, not published | — | — |
| 49 | niah | **markers follow their document** (`--marker-follow-doc`): markers are no longer routed; a document's markers are kept iff ≥ 1 of its body tokens is kept (`mark_positions_free(free_markers="doc")`). At train time the marker gate is the max of the document's body gates; budgets count markers (`follow_keep_budget`). v6a (restart-avg), matched data, test64 `@pair`, seeds 0/10 | mfd: s0 +0.043 ± 0.028 (3 rows >0.1) loses; **s10 +0.000 ± 0.000 (0 rows) matches**. Routed markers (v6a base): +0.075 ± 0.043 / +0.082 ± 0.039 → lose. Train/val paired: mfd 0.000/0.000 both seeds; base 0.000/+0.121 and +0.085/0.000 | mfd helps (both seeds better than base); next: mfd + 3 polish passes; mfd on textgroups/outlier |
| 50 | niah, textgroups, outlier, nq | mfd on more tasks (v6a, test64 `@pair`, seed 0), vs routed markers (base); niah mfd + 3 polish passes | textgroups: mfd −0.041 ± 0.007 vs base −0.038 ± 0.009 (both beat). **outlier: mfd +0.003 ± 0.005 matches vs base +0.038 ± 0.011 loses (9 rows >0.1)**. nq: both −0.001 (match). niah mfd+3 passes: s0 +0.006 ± 0.006 (1 row), s10 +0.126 ± 0.056 (6 rows): passes don't help consistently | **mfd ≥ routed markers on all 4 tasks → recipe v6b = v6a + `--marker-follow-doc`** (3 passes not adopted); length eval next |
| 51 | textgroups, outlier, nq | length transfer of the 2k mfd vs routed-marker routers (testlong: 48 rows at 8k, 56 at 32k, `@pair`) | nq: both match at 8k/32k. **outlier**: 8k mfd −0.042 / base −0.032 (both beat); 32k **mfd +0.203 ± 0.037 loses** vs base −0.064 ± 0.025 beats. **textgroups**: 8k mfd +0.208 loses vs base −0.031 beats; 32k mfd +0.638 vs base +0.346 (both lose). Keep breakdown at 32k: mfd keeps **every** document's markers (1.00; the router keeps ~20% of every doc's body, so every doc is touched and pays 2 markers) vs bar 0.01–0.03. The bar drops the markers of partially kept documents (its fragments carry no markers), which mfd ("any kept body token") cannot express, so the bar is outside the mfd class. Both routers keep only ~60% of the non-gold id regions (bar 100%) | mfd ("any") hurts length transfer → test **mfw**: markers follow only a WHOLE kept body (`--marker-follow-whole`: gate = min of body gates, budget pays markers when a doc completes, eval = the default whole-body rule) = the bar's own semantics |
| 52 | textgroups, outlier, scifact, rerank | mfw with a MIN relaxed marker gate; mfd on scifact/rerank (2k test64 `@pair`) | mfw-min outlier −0.006 ± 0.007 matches; **mfw-min textgroups +0.474 (61/64 rows bad)**: w_gold went negative, gold keep 0.14. Under hard-concrete noise + keep-dropout the min over a doc's body gates is ~0 for every doc, so markers were always removed in training (gradient reaches one token). mfd: scifact −0.000 matches; rerank +0.038 ± 0.009 loses (v6a routed markers: +0.000 ± 0.007) | relaxed gate changed to the MEAN of the body gates (1 iff whole body kept); rerun |
| 53 | textgroups, outlier | mfw with the MEAN relaxed gate (markers kept only for whole bodies), 2k test64 + testlong | 2k: textgroups +0.001 ± 0.021 matches (routed markers −0.038 beats), outlier +0.007 ± 0.009 matches. 8k: outlier −0.061 beats, textgroups +0.141 loses. 32k: outlier **+0.132 loses**, textgroups +0.838 loses. mfw-min outlier: 8k +0.042, 32k +0.369 | **no marker-follow variant beats routed markers** at any length → marker change not adopted; v6a (routed markers) stays canonical |
| 54 | nq, outlier, scifact, rerank, textgroups | **length transfer of v6a (routed markers), trained at 2k only**, testlong `@pair` (48 rows at 8k / 56 at 32k ⚠) | 8k: nq +0.002 match, outlier −0.032 beat, scifact −0.003 match, rerank +0.009 ± 0.004 lose, textgroups −0.031 beat. **32k: nq −0.000 match, outlier −0.064 ± 0.025 beat, scifact −0.026 ± 0.024 beat, rerank −0.086 ± 0.003 beat, textgroups +0.346 ± 0.043 lose** | with the larger long test sets, v6a transfers to 32k on 4 of 5 representative tasks; textgroups is the exception (F0 doc-only runs next) |
| 55 | nq, outlier, scifact, rerank, textgroups | F0 (doc-only features) on v6a, 2k test64 `@pair` | textgroups −0.059 ± 0.008 beats, nq +0.004 matches, scifact +0.000 matches; **outlier +0.116 ± 0.024 loses, rerank +0.632 loses (64/64 rows)**: tasks whose answer is a document id need per-document id tokens, which doc-level features cannot express | F0 cannot be the one recipe (only textgroups' 32k is left; F0 long eval on textgroups only) |
| 56 | 16 tasks | **shared cross-task router on v6a** (xt3v6a: one router trained on nq + contradiction + scifact, 16 rows each; each task trains at ITS OWN matched budget ρ_task; task-block batches; restarts and polish scored with the paired budget, `--pair-eval`), evaluated on test64 `@pair` | held-in: nq +0.001, contradiction match, scifact match. Held-out: beats fiqa −0.051, oolong; matches msmarco, qdmatch, strmatch; **loses** absence +0.172, grouping, niah, outlier, outlier_amzn +0.096, reorder, rerank +0.087, textgroups. Total **2 beat / 6 match / 8 lose** (per-task v6a on the same rows: 5 / 9 / 2) | sharing from 3 tasks doesn't transfer to tasks needing id/position-specific rules → 17-task pooled run (xt17v6a, `--xtask-offload`, `--stg-tasks niah,absence,fiqa`) |
| 57 | textgroups | F0 doc-only router at 8k / 32k (testlong) | **8k −0.055 ± 0.023 beats; 32k −0.097 ± 0.041 beats** (v6a routed: +0.346 at 32k). F0 is also at parity at 2k (−0.059) | textgroups transfers to 32k only with doc-level features; F0 loses at 2k on id tasks (iteration 55), so it is not the single recipe |
| infra | — | xt17v6a with `--xtask-offload` took 9.3 h, 7.4 h of it model swaps (5688). The polish and restart selection interleave rows of all tasks, so each scored candidate swaps models | fix if pooled training is reused: task-ordered polish rows / per-task scoring loops |
| 58 | 16 tasks | **one shared router trained on all 17 tasks** (xt17v6a: 16 rows each, per-task matched budget, v6a recipe, `--stg-tasks niah,absence,fiqa`), test64 `@pair` | **1 beat, 13 match, 2 lose**: fiqa −0.046 beats; absence +0.038 ± 0.039, contradiction, grouping −0.005, msmarco, nq, oolong +0.000, outlier_amzn +0.007, outlier +0.002, qdmatch, rerank +0.004, scifact, strmatch, textgroups −0.002 match; **niah +1.274 ± 0.136** (44 rows >0.1) and **reorder +0.126 ± 0.019** lose. Per-task v6a on the same rows: 5 / 9 / 2 | a single shared router reaches parity on 14 of 16 tasks; niah (needle cut) and reorder (whole-order task) need task-specific rules |
| 59 | 16 tasks | **shared 17-task router (xt17v6a) vs per-task v6a at 8k / 32k**, both trained at 2k only; testlong `@pair` (8k: 48 rows, oolong 43, niah 38, fiqa none (all its staged 8k rows share examples with its `_stg` train rows); 32k: 56 rows, niah 52, fiqa 8; reorder/absence use their 16k rung) | **8k**: per-task 5 beat / 6 match / 4 lose; shared 0 / 8 / 7. **32k**: per-task 5 / 4 / 7; shared 4 / 6 / 6. Shared wins where per-task overfits (outlier_amzn 32k −0.009 vs +0.099, fiqa 32k +0.031 vs +0.805, grouping 32k +0.012 vs +0.052, rerank 8k) and loses where tasks need their own rule (reorder 8k +0.082 / 16k +0.275 vs per-task −0.40 / −0.42; niah +1.3 / +1.9; textgroups 8k +0.078 vs −0.062). contradiction and strmatch lose at length under both | the shared router transfers to 32k about as well as per-task routers (10 vs 9 at parity) |
| 60 | 17 tasks | **pooled-training speed fix** (`train_e2e_router.py`): frozen models parked in PINNED host memory once and swapped by re-pointing `.data` (`park`/`swap_in`); LRU cache `--xt-resident K`; task-blocked polish (`score_many`: every candidate scored on one task's rows before the next model is swapped in); batched forwards within task-blocked batches (`--batch-max-T 2700` keeps absence's 3k rows per-row) | xt17f: **6011 s (1 h 40) vs 33,593 s (9.3 h)**; swaps 1200 × 0.19 s = 222 s (was 5688 × 4.7 s). Now compute-bound: 3 restarts × 680 batched steps + 740 s model loading. ≤ 9 tasks with `--xt-resident 9`: no swaps (xs3 19 min, xs6 39 min, xs9 58 min). **But test64 quality dropped**: xt17f 1 beat / 7 match / 8 lose vs xt17v6a 1 / 13 / 2. The task-blocked fine polish (one sweep; joint move or best single shift) accepted 1 shift vs 11 for the sequential search (train +0.111 vs +0.086); the polished restarts were also worse (+0.148 vs +0.104; training variance) | fine polish v2 (rounds of single-shift sweeps + best prefix of the improving shifts); rerun to verify reproduction |
| 61 | 16 tasks | **training-task subsets**, shared router on v6a (per-task budgets, `--pair-eval`), test64 `@pair` | **xs3** (outlier = id answer, textgroups = counting, msmarco = retrieval): held-in 1 match / 2 lose (narrowly), held-out 7 match / 6 lose (reorder +0.50, rerank +0.25, niah +1.8, absence +0.34, grouping +0.09, contradiction +0.005). **xs6** (+ rerank, reorder, strmatch): held-in 5 match / 1 lose (rerank +0.010 ± 0.008); **held-out 8 match / 2 lose** (niah +1.68, absence +0.100 ± 0.048); 13 of 16 at parity overall. **xs9** (+ niah, oolong, contradiction): held-in 1 beat / 3 match / 5 lose, held-out 4 match / 3 lose: worse than xs6 (adding niah hurts strmatch, reorder, rerank, textgroups). xt17f: see 60 | **smallest subset that brings held-out tasks to parity: 6 tasks** (outlier, textgroups, msmarco, rerank, reorder, strmatch); single runs (variance applies) |
| 62 | 17 tasks, xs6 | rerun with fine polish v2 (task-blocked rounds: single-shift sweep + best prefix of the improving shifts) | **xt17g: 6673 s end to end, test64 1 beat / 6 match / 9 lose: does NOT reproduce xt17v6a's 1 / 13 / 2**, although its train polish now matches the sequential search (+0.094 vs +0.086). Remaining code difference vs xt17v6a: batched (task-blocked) training forwards; per-row control (`--no-xt-batch`, xt17h) running; seed variance is the alternative explanation. **xs6g** (xs6 rerun): held-in 5 match / 1 lose (strmatch +0.087), held-out 1 beat / 7 match / 2 lose (niah +1.58, grouping +0.010 ± 0.005); xs6 is consistent at ~13/16 parity (different marginal losers). xs6 seed-10 replicate running | the 6-task shared router is the stable one; 17-task pooling is high-variance |
| 63 | 16 tasks | **xs6 / xs6g (6-task shared routers, trained at 2k) at 8k / 32k**, testlong `@pair` (reorder/absence 16k; fiqa 32k only, 8 rows ⚠) | 8k: xs6 0 / 8 / 7, xs6g 1 / 8 / 6. 32k: xs6 3 / 6 / 6, xs6g 1 / 5 / 9. Consistent losers: niah (+1.5 to +2.4), strmatch (+0.14 to +0.40; held-in!), textgroups (+0.04 to +0.14), contradiction (+0.01 to +0.18), grouping (+0.005 to +0.01), reorder (+0.008 to +0.015). Per-task v6a on the same rows: 8k 5 / 6 / 4, 32k 5 / 4 / 7; xt17v6a: 8k 0 / 8 / 7, 32k 4 / 6 / 6 (iteration 59) | the 6-task router matches the 17-task router at length and is cheaper; strmatch/textgroups/contradiction degrade with length under any shared router |
| 64 | xs6 | **seed replicate** of the 6-task shared router (seeds 10–12; same recipe and data) | held-in (msmarco, outlier, reorder, rerank, strmatch, textgroups): 3 match / 3 lose (reorder +0.250, rerank +0.073, strmatch +0.013); held-out: 5 match / 5 lose (fiqa +0.044, outlier_amzn +0.086, grouping +0.117, niah +1.52, absence +0.444). **8 of 16 at parity**, vs 13 / 13 for the two seed-0 runs | xs6 is NOT consistent across seeds: shared routers inherit the per-task run-to-run variance (iteration 46) and amplify it; any subset conclusion needs ≥ 3 seeds per subset |
| 65 | xs6 | **seed-level weight average** of the three xs6 runs (xs6, xs6g, xs6 seeds 10–12): plain mean of the router weights, no re-polish; under `@pair` the offset is irrelevant (per-row top-k), test64 | held-in 5 match / 1 lose (reorder +0.038 ± 0.016); held-out 1 beat (fiqa −0.021) / 6 match / 3 lose (outlier_amzn +0.025 ± 0.013, niah +1.63, absence +0.155). **12 of 16 at parity** (single seeds 13 / 13 / 8) | seed averaging recovers most of the good seeds' quality from 3 runs and removes the bad-seed outcome → recommended shared router = xs6avg (`weights/xs6avg/xs6avg_s0_rhobar.pt`); niah and absence are lost by every shared router |
| 66 | 17 tasks | control: xt17 with **per-row** training forwards (`--no-xt-batch`, as xt17v6a) + pinned swaps + polish v2 (xt17h) | 11,829 s (3.3 h; per-row training ~2700 s per restart); test64 0 / 9 / 7 (losses outlier_amzn, reorder, rerank, textgroups, grouping, niah, absence). Three 17-task runs since xt17v6a: 1/7/8, 1/6/9, 0/9/7, vs 1/13/2 | batched forwards are NOT the cause; **xt17v6a's 1/13/2 was a lucky seed**: 17-task pooled routers land at 6–9 of 16 at parity per run. Recommended shared router: the seed-averaged 6-task xs6avg (12/16, iteration 65) |
| 67 | 16 tasks | **xs6avg at 8k / 32k** (testlong `@pair`) | 8k: 1 beat / 7 match / 7 lose; 32k: 1 / 5 / 9. Losers: niah (+1.6 / +2.2), textgroups (+0.10 / +0.26), strmatch (+0.05 / +0.10), contradiction (+0.01 / +0.13), reorder, grouping, rerank and qdmatch at 32k (+0.006 to +0.04), fiqa 32k (+0.22 ± 0.21, 8 rows), absence 8k | same as the single xs6 runs (8k 0–1 / 8 / 6–7, 32k 1–3 / 5–6 / 6–9); seed averaging stabilises 2k but does not fix length transfer |

## 10. Token router + layer-skip router: compute-vs-loss frontier (2026-10-01)

**Question.** The pipeline is token router (v6a) → per-(token, layer) skip router (§9, `shared`
variant). With tolerances of 0.02 / 0.05 / 0.10 nats of paired ΔCE vs FULL context, how many fewer
forward FLOPs than the full model can each task run at?

**Answer, headline.** At ≤0.05 nats:
* mean of per-task reductions **4.5×** (geometric mean 3.3×, median 3.8×); the bar
  `gold_fl20p8_noslot` gets 2.2× (geometric) at mean ΔCE +0.098;
* 12 of 16 tasks meet the tier on test64;
* best tasks: nq 10.2×, msmarco 11.6× (misses on test by +0.003), scifact 8.2×, outlier 7.3×,
  oolong 6.1×, fiqa 5.2×.

⚠ Test64 eval_size 64 per task (qdmatch_hpqa 54). Val has 64 rows (niah, fiqa and absence: 16
matched `_stg` rows). Every number below is from 2k rows.

### 10.1 Method (code `debug/learned_router/layerskip/frontier.py`, `summarize_frontier.py`)

* **Token side.** The canonical v6a router (`v6avg_s0_rhobar.pt`, routed markers) keeps, per row,
  the top round(f · k_bar) routed tokens. k_bar is the number of routed tokens the bar keeps on that
  row (the token agent's `@pair` budget, scaled). f ∈ {1, 0.7, 0.5, 0.35}.
  * Drop semantics are the grid's (`check_hand_fl.router_real`).
  * At f = 1 the per-row T2 equals the bar's exactly. The test64 ΔCE reproduces
    `results_router_v6` to 4 decimals (scifact −0.0014, contradiction +0.0017).
  * On nq, outlier, scifact and contradiction, v6a routers were also **trained** at the lower
    budgets (`train_e2e_router.py` unmodified, `--rhos` = the matching routed keep, otherwise the
    v6a CLI; weights `v6aL_s0_rho<ρ>.pt`). Bar-trained-thresholded-lower and lower-trained routers
    land within noise of each other, so the other 12 tasks use the bar-trained router only.
* **Layer side.**
  * The shared layer-skip router is trained on the eligible tokens (kept body tokens, not markers)
    at keep 0.75, then at keep 0.5 warm-started from the 0.75 router (ρ anneals 0.75 → 0.5).
  * 20 epochs each, 32 train rows; CE + KL to the full model.
  * Eval: deterministic gate with a val-calibrated global cutoff, scored on the soft path with
    binary gates (exact).
* **FLOPs.** `layerskip_lib.flops_ratio`: per-layer active columns, attention linear + quadratic
  terms, GDN linear + state terms, against the full row.
* **Candidates per task:**
  * the full model;
  * (token f, layer keep ∈ {1, 0.75, 0.5}) for 4 (12 tasks) or 7 (4 tasks) token configurations.

  For each tier, the selected candidate is the one with the lowest **val** FLOPs whose **val** mean
  ΔCE vs full ≤ τ (near-ties within 2% of FLOPs go to the lower val ΔCE). Test64 is then reported.
* **References on test64:**
  * the bar;
  * uniform random layer skip at the same keep on the same token router (2 seeds);
  * the grid's heuristic layer-skip schemes `gold_fl20p8_skip{odd,even,gdn2}`, scored by the grid
    driver on test64 (`devloss_grid/results_layerskip_t64/`; FLOPs from `collect_grid.flops_ratio`).

### 10.2 Per-task frontier (test64; FLOP reduction vs full, (ΔCE), ✓ = meets the tier on test)

| task | bar ×FLOPs ΔCE | ≤0.02 | ≤0.05 | ≤0.10 | selected (≤0.05) |
|---|---|---|---|---|---|
| nq | ×0.336 −0.007 | 10.2× (−0.002) ✓ | 10.2× ✓ | 10.2× ✓ | f 0.35 + layer keep 0.5 |
| msmarco | ×0.333 −0.006 | 9.2× (+0.027) ✗ | 11.6× (+0.053) ✗ | 11.6× ✓ | f 0.35 + 0.5 |
| scifact | ×0.434 −0.000 | 8.2× (−0.010) ✓ | 8.2× ✓ | 8.2× ✓ | f 0.35 (trained) + 0.5 |
| outlier | ×0.450 +0.001 | 6.0× (+0.032) ✗ | 7.3× (+0.021) ✓ | 7.3× ✓ | f 0.35 (trained) + 0.5 |
| oolong | ×0.391 +0.028 | 6.1× (+0.015) ✓ | 6.1× ✓ | 6.1× ✓ | f 0.35 + 0.5 |
| fiqa | ×0.449 +0.053 | 5.2× (−0.022) ✓ | 5.2× ✓ | 5.2× ✓ | f 0.7 + 0.5 |
| grouping | ×0.287 +0.023 | 1.0× (full) ✓ | 4.2× (+0.020) ✓ | 5.5× (+0.052) ✓ | f 1 + 0.75 |
| contradiction | ×0.463 +0.000 | 3.7× (−0.000) ✓ | 3.7× ✓ | 4.7× (+0.025) ✓ | f 0.5, no layer skip |
| rerank | ×0.416 +0.040 | 3.0× (+0.016) ✓ | 3.8× (+0.052) ✗ | 4.8× (+0.111) ✗ | f 0.7 + 0.75 |
| strmatch | ×0.624 −0.001 | 2.0× (+0.008) ✓ | 2.9× (+0.025) ✓ | 2.9× ✓ | f 0.35, no layer skip |
| qdmatch_hpqa | ×0.765 −0.002 | 2.0× (+0.054) ✗ | 2.0× ✗ | 2.0× ✓ | f 0.7 + 0.75 |
| textgroups | ×0.607 +0.018 | 1.65× (−0.044) ✓ | 1.65× ✓ | 2.0× (+0.084) ✓ | f 1, no layer skip |
| niah | ×0.572 −0.000 | 1.75× (+0.097) ✗ | 1.75× ✗ | 1.75× ✓ | f 1, no layer skip |
| outlier_amzn | ×0.440 +0.104 | 1.0× (full) | 1.0× | 1.0× | nothing meets on val |
| reorder | ×0.312 +0.431 | 1.0× (full) | 1.0× | 1.0× | nothing meets on val |
| absence | ×0.687 +0.882 | 1.0× (full) | 1.0× | 1.0× | nothing meets on val |

| tier | mean per-task reduction | geometric mean | median | meets on test |
|---|---|---|---|---|
| ≤0.02 | 3.9× | 2.85× | 2.5× | 12/16 |
| ≤0.05 | **4.5×** | **3.3×** | 3.8× | 12/16 |
| ≤0.10 | 4.7× | 3.5× | 4.75× | 15/16 |

`frontier.json` and `frontier_summary.txt` (full candidate tables, test-oracle picks, random
controls) are next to the code.

### 10.3 What this says

* **The layer router adds compute savings on 9 tasks.** It is in the selected ≤0.05 combo on nq,
  msmarco, scifact, outlier, oolong, fiqa, grouping, rerank and qdmatch. It typically takes the
  token-only point from ×0.15–0.20 down to ×0.10–0.14 at the same ΔCE.
  * **Random layer skip** at the same keep and token router costs +0.15 to +0.66 nats on those
    tasks (oolong: +0.03).
  * **The grid's skip-odd-layers heuristic**, applied on top of the bar, is a strong static
    competitor: ×0.64–0.93 of the bar's FLOPs at ΔCE within ±0.03 of the bar on 12 of 16 tasks. It beats the learned
    layer skip on contradiction (+0.011 at ×0.33) and grouping.
* **Token budget does most of the work.**
  * The best points cut the token budget to 0.35–0.5 × the bar, and the learned token router
    tolerates that on nq, scifact, msmarco and oolong.
  * The layer router then adds 1.2–1.6× more.
* **Where nothing qualifies:** outlier_amzn, reorder and absence lose more than 0.1 nats already at
  the bar (+0.10 / +0.43 / +0.88), so the full model is selected.
* **Selection noise.** Val/test disagree on 4 tasks at ≤0.05 (msmarco, rerank, qdmatch, niah). Each
  misses by 0.003–0.05; niah/fiqa/absence select on only 16 val rows.

### 10.4 Wall-clock (nq, f 0.5 + layer keep 0.5; H200, bf16; `wallclock.py`)

| regime | full | token-compacted | + layer skip (per-layer gather) |
|---|---|---|---|
| 2k, batch 1 (16 rows) | 50.5 ms | 49.9 ms (1.01×; FLOPs ×0.188) | 63.7 ms (0.79×; FLOPs ×0.116) |
| 2k, batch 16 (8 rows) | 507 ms | 102 ms (**4.95×**) | 88 ms (**5.78×**) |
| 32k, batch 1 (4 staged rows, T ≈ 33k → T2 ≈ 4.1k) | 762 ms | 77 ms (**9.9×**; FLOPs ×0.099) | 79 ms (9.65×; FLOPs ×0.054) |

* **Token compaction converts FLOPs into time almost 1:1** once the forward is compute-bound:
  32k batch 1 (×0.099 FLOPs → 9.9×) and 2k batch 16 (×0.188 → 4.95×).
* **Layer skip converts only at batch 16** (an extra 1.17×). A ~4k-token compacted forward at
  batch 1 is launch-bound (about 2.4 ms per layer regardless of tokens), and the gather adds a host
  sync (`nonzero`) per layer.
* **To realise layer-skip FLOPs at batch 1:**
  * static-shape bucketed gathers built on device (no host sync), so the compacted forward is
    CUDA-graph capturable;
  * fuse gather/scatter into the block's input norm and residual write;
  * or batch requests.

### 10.5 Files

* `debug/learned_router/layerskip/`:
  * `frontier.py` and `run_frontier_local.sbatch` (`WALLCLOCK=` mode);
  * `summarize_frontier.py` → `frontier.json`, `frontier_summary.txt`;
  * `wallclock.py`;
  * `runs/<task>/fr_bar.json`, `fr_low.json`, `wallclock_*.json`;
  * `weights/<task>/fr_*`.
* Token routers trained at lower budgets: `debug/learned_router/weights/<task>/v6aL_s0_rho*.pt`
  (nq, outlier, scifact, contradiction).
* Grid rows: `debug/devloss_grid/results_layerskip/<task>_2k.json` (the grid's 16 test rows; schemes
  `router_v6a+ls`, `router_frontier_t{0.02,0.05,0.1}` with a summary `flops` field).
  * `router_v6a+ls` = f 1 + the most aggressive layer keep whose val ΔCE vs token-only is within
    max(SE, 0.005).
* Shared-file edits (backwards compatible):
  * `collect_grid.py`: per-scheme `flops` override from the file; `results_layerskip` root added;
  * `render_grid.py`: registers the two scheme families.
* wandb group `router-layerskip`.

### 10.6 Making layer skip fast (`fastskip.py`; nq, token router at 0.5 × bar + layer keep 0.5; H200, bf16)

**Approach.** Per-layer capacity routing (MoD-style), all on device:
* At each layer, keep the always-active columns plus the top-k_l eligible columns by router logit
  (`topk` → sorted indices → `gather` / `scatter`).
* No host sync, static shapes, so the whole compacted forward is captured in a **CUDA graph**.
* For the timing, k_l is set per row to the threshold rule's count at that layer, so the decisions
  equal the reference. Answer CE equals the exact-deletion gather path on every row, graphed or not.
  The soft-path reference differs by ≤0.006 nats (bf16 kernels: SDPA with a bias vs flash on the
  subset).
* A deployment would use calibrated, bucketed capacities instead.

Speed-up vs the full model (both CUDA-graphed):

| regime | full | token-compacted | + layer skip (fixed-capacity, graph) | FLOPs tok / +ls (ideal speed-up) |
|---|---|---|---|---|
| 32k, batch 1 (4 rows, T ≈ 33k → T2 ≈ 4.1k) | 779 ms | 75.7 ms (**10.3×**) | 49.0 ms (**15.9×**) | ×0.099 / ×0.054 (10.1× / 18.6×) |
| 2k, batch 16 (6 rows) | 518 ms | 102.5 ms (5.05×) | 71.5 ms (**7.25×**) | ×0.189 / ×0.116 (5.3× / 8.6×) |
| 2k, batch 1 (4 rows) | 32.9 ms | 10.8 ms (3.0×) | 11.9 ms (2.8×) | ×0.188 / ×0.116 |

* Before this, the old per-layer gather with a host sync was at or below token-only:
  * 32k batch 1: 9.6×;
  * 2k batch 1: 0.79×.
* With the fixed-capacity gather plus CUDA graphs, layer skip now converts into wall-clock wherever
  the compacted forward is compute-bound.
* At 2k batch 1 (~340 tokens) every kernel is latency-bound. The ~5 extra small kernels per layer
  (router, topk, sort, gather, scatter) cost about what they save. The next step there is a fused
  router + topk + gather kernel.
* Outputs: `runs/nq/fastskip_{32k_b1,2k_b16,2k_b1}.json`.

### 10.7 Per-example overfit ceiling on compute (`oracle_ls.py`, `summarize_oracle.py`; 2026-10-02)

**Setup.** Free per-example logits:
* θ for every routed token (body + markers);
* φ for every (body token, layer).

Trained to minimise **differentiable expected FLOPs** (the same FLOP model, as a function of the gate
probabilities: per-layer active count, attention quadratic, GDN) subject to ΔCE ≤ τ on that one
example:
* Lagrangian with dual ascent on an EMA of the relaxed ΔCE;
* hard-concrete gates on the exact soft path;
* 300 steps; the 4 configurations (token-only and token+layer, × τ 0.02 / 0.05) batched in one forward.

Then a **hard check** with exact deletion and per-layer skip:
* greedy add-back of the highest-logit dropped items until ≤ τ;
* then greedy pruning of the lowest-logit items (tokens, then pairs) while ≤ τ, with ≤150 exact
  evaluations.

Scope and caveats:
* 6 test64 rows per task at 2k, ⚠ eval_size 6.
* **Non-deployable**: the oracle sees the answer. It is a lower bound on FLOPs, i.e. how much headroom
  the learned routers leave.

**The label leaks through selection.** All 8 tasks answer with document ids:
* nq, scifact, niah: `[9]`;
* outlier: `Outliers: [1], [10], [11]`;
* contradiction, strmatch, textgroups: id pairs or groups;
* rerank: a full ranking.

The free oracle keeps only the answer documents' ids. Dumped kept tokens on nq include rows with
exactly `Document9` plus a marker (×0.051 FLOPs), so the model copies the only id it sees. Its
numbers (median ×0.04–0.07 on nq, scifact and outlier) are therefore **trivially attainable**, not
compression.

**Fair version** (`--force-id-prefix 8`): every document's markers and first 8 body tokens are always
kept, at full depth, as the bar does. The oracle optimises only the rest. Median hard FLOPs vs full,
6/6 rows meet τ:

| task | forced floor | token-only τ .02 / .05 | token+layer τ .02 / .05 | frontier pick (§10.2), rows met | bar |
|---|---|---|---|---|---|
| nq | .100 | .103 / .103 | .102 / .101 | .096 (6/6) | .337 |
| scifact | .091 | .091 / .091 | .091 / .091 | .118 (6/6) | .398 |
| outlier | .123 | .129 / .128 | .126 / .127 | .164 (4/6) / .134 (5/6) | .444 |
| rerank | .190 | .202 / .199 | .196 / .192 | .324 (3/6) / .255 (2/6) | .398 |
| textgroups | .189 | .547 / .506 | **.344 / .277** | .618 (4/6) / (5/6) | .618 |
| contradiction | .285 | .302 / .296 | .296 / .293 | .270 (6/6) | .461 |
| niah | .531 | .533 / .531 | .531 / .531 | .579 (6/6) | .579 |
| strmatch | .504 | .567 / .567 | .567 (envelope) | .499 (6/6) / .351 (4/6) | .627 |

Across tasks (median of per-task medians) at τ 0.05:

| method | FLOPs vs full |
|---|---|
| fair token+layer oracle | ×0.235 (4.3×) |
| fair token-only oracle | ×0.248 |
| frontier pick | ×0.262 |
| bar | ×0.453 |
| free, leak-prone oracle: token+layer | ×0.137 |
| free, leak-prone oracle: token-only | ×0.163 |

* **The forced prefix is the binding cost.** On 6 of 8 tasks the fair oracle lands within ×0.01–0.06
  of the floor set by the forced ids alone, keeping 3–10% of gold body (contradiction 34%, strmatch
  77%).
* **Little headroom left over the deployable frontier.** The frontier pick (not forced to keep every
  prefix) is already at or below the fair oracle on nq, contradiction and strmatch. Real headroom
  remains on textgroups (×0.28 vs ×0.62), rerank (×0.19 vs ×0.26–0.32, where the frontier also
  misses τ on rows) and outlier at τ 0.02.
* **Layer routing inside the oracle matters only on textgroups** (×0.51 → ×0.28 at τ 0.05; 66%
  attention / 54% GDN pairs skipped, late layers 0.77 vs early 0.36). Elsewhere it skips <20% of
  pairs once the id prefix is forced. In the free version it skips 75–95%, because almost nothing is
  kept.
* **The oracle is not a tight bound where many scattered tokens are needed.** On strmatch, 300 steps
  plus 150 greedy evaluations reach ×0.57, worse than the frontier's ×0.35–0.50.
* **Wall time:** 3.5–10 min per example (soft 300 steps ≈ 1.2 s/step for the 4 batched
  configurations, + greedy).
* Outputs: `runs/<task>/oracle{,_fid8}_2k_r0-5.json`, `oracle_summary_{free,fid8}.{json,txt}`,
  `fid8_floor.json`, `runs/nq/oracle_dump_2k_r0-1.json` (kept-token dump).

### 10.8 32k quality of the timed setting (`longcheck.py`; nq, 56 testlong 32k rows, ⚠ eval_size 56)

Both routers were trained at 2k only. The token router keeps 0.5 × the bar's per-row budget (`@pair`
× 0.5).

| configuration | FLOPs vs full | ΔCE vs bar ± SE (rows >0.1) | wall-clock vs full, CUDA graph, batch 1 |
|---|---|---|---|
| full context | ×1 | +0.065 ± 0.030 (6) | 1× |
| bar `gold_fl20p8_noslot` | ×0.203 | 0 | — |
| v6a router, bar budget | ×0.203 | −0.001 ± 0.001 (0) | — |
| token router 0.5 × bar | ×0.099 | −0.001 ± 0.001 (0) | **10.3×** |
| + layer keep 0.75 | ×0.078 | **+0.000 ± 0.002 (0)** | **12.3×** |
| + layer keep 0.5 | ×0.054 | +0.110 ± 0.015 (22) | 15.9× |

* At 32k, full context is itself worse than the bar (+0.065).
* **Parity point: token router at 0.5 × bar + layer keep 0.75.** It is at parity with the bar at
  ×0.078 FLOPs (2.6× fewer than the bar) and runs **12.3× faster than the full model** at batch 1.
* **Layer keep 0.5 does not transfer from 2k to 32k** (+0.110 vs the bar on 22 of 56 rows), even
  though it is still +0.045 ± 0.033 vs full context. Calibrating the layer cutoff on long val rows,
  or training on mixed lengths, would be the fix.
* Outputs: `runs/nq/longcheck_32k_L0.5_k{0.5,0.75}.json`, `runs/nq/fastskip_32k_b1_k{0.5,0.75}.json`.
