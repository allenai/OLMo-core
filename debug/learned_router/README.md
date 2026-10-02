# learned_router — a learned linear token router for slot-less compaction

Instead of a fixed heuristic (`gold_rand20p8_noslot`, `gold_fl20p8_noslot`, `gold_first20`), a
**single linear layer** decides, per document body token, whether the frozen dense checkpoint sees
it. Trained per task with REINFORCE on the frozen model; scored on the dev-loss grid's test rows.
Findings: `records/learned-token-router.md`.

## Router

`router_lib.py` (shared by trainer and grid driver, so train/eval features are identical):

    z = b + w_pos · f_pos + w_gold · gold + w_emb · e_rms / sqrt(d)      keep with p = sigmoid(z)

* **routed tokens**: every document body token (inside `<|box_start|>..<|box_end|>`, not a marker),
  gold documents included — there is no forced gold keep and no forced 8-token id prefix;
* `e_rms`: the frozen checkpoint's input-embedding row, RMS-normalised (fp32), d = 2560;
* `f_pos` (58, length-agnostic): offset from doc start one-hot 0–15 + 12 log2 buckets, the same from
  doc end, relative position in the doc, the doc's relative index in the context;
* `gold`: 1 if the token's document is in the task's gold set (0 on oolong, which has none).

Init: everything 0 (p = 0.5). Variants: `full`, `nogold` (w_gold unused), `noemb` (w_emb unused).

## Drop semantics (slot-less, per Prasann)

`Transformer._compact_pooled_soft_tokens` with `keep_token_rule="custom"`, doc-level keep = none,
`drop_slots=True`: kept body tokens stay real at their ORIGINAL RoPE positions, dropped ones vanish,
**no slot for any document**. Markers and everything outside documents (system prompt, query,
answer) are always kept.

Keeping the markers needed a (default-off, bit-identical when off) flag,
`mark_positions_free(..., free_markers=True)` ← `pst["keep_token_mask_markers"]`: under the stock
rule a document's markers are freed only when its WHOLE body is kept, so with `drop_slots` a
partially kept document shows its fragments with no markers and an all-dropped document vanishes
entirely (which is also what `gold_*_noslot` do to non-gold documents). With the flag an
all-dropped document leaves an empty `<|box_start|><|box_end|>` pair. `smoke_cpu.py` asserts both
behaviours; `router_l0.2_nomark` scores the l0.2 router under the stock (marker-dropping) rule.

## Training (`train_router.py`)

Per row: CE_full once; each visit draws K = 8 Bernoulli masks and runs the compacted forwards
(no grad, batched K per forward — asserted identical to one-at-a-time). Reward
`R = -(CE_mask - CE_full) - λ·keep_frac`; the CE part by REINFORCE with a leave-one-out baseline,
the keep_frac part analytically (`E[keep_frac] = mean p`, same gradient, less variance). Adam, lr 0.1
(bias/position/gold) and 0.01 (embedding), 4 rows/step, masks re-drawn every epoch, ≤ 40 epochs,
early stop on the VAL reward of the sampled policy (patience 8, min 12); the best epoch's weights
are saved. `freeze_fla_length_autotune()` is required: every mask gives a new compacted length and
FLA re-autotunes per length (~20 s/step without it, ~3 s with).

Configs `[nogold_|noemb_]l<λ>[_s<seed>]`. Pilot (nq, outlier, contradiction, oolong):
`l0.05,l0.2,l0.5,l1.0,nogold_l0.2,noemb_l0.2,l0.2_s2`; main sweep (14 other tasks): `l0.05,l0.2,l0.5,l1.0`,
≤ 60 epochs (absence: `l0.05` only, see the record). λ is selected on the VAL rows by
`summarize_sweep.py` (cheapest val T2/T with val ΔCE ≤ 0.02).

## Data (`fetch_train_rows.py`, login node)

Train/val rows = HF `r2k`/`r8k` rows 64.. (past the staged first 64 of each rung), 16 train + 8 val
per rung per task (32 + 16). Disjointness is by hashing, not by index: a candidate is dropped if its
(non-template) query, any gold document text, or (no-gold tasks) its (query, answer) matches ANY of
the first 64 rows of EVERY rung (2k/8k/32k) — rung files reuse the same examples at different
lengths (every nq example appears at all three rungs). `split_report.json` has the counts. Written
to `sneetches:/data/prasann/devloss_grid/data/router_{train,val}/` (synced by the launcher).

## Eval

Driver schemes `router_<cfg>` (deterministic p > 0.5) and `router_<cfg>_samp` (seeded Bernoulli),
`sel="router"`, `sel_cost=0`, weights from `weights/<task>/<cfg>.pt`. The launcher scores them on
the grid's test rows (16/16/8 at 2k/8k/32k, same checkpoints) plus the baselines (`gold_rand20p8_noslot`,
`gold_fl20p8_noslot`, `gold_first20`, `gold_only_noslot`) re-scored in
the same file (paired deltas) → `debug/devloss_grid/results_router/<task>_<rung>.json`.
`analyze_router.py` → `router_vs_grid.json` (Pareto envelope vs baselines at matched compaction);
`analyze_weights.py` → `runs/<task>/weights_analysis.json` (position profile, vocab scores, seed
correlation).

## Files

| file | what |
|---|---|
| `router_lib.py` | features, `LinearRouter`, mask helper |
| `train_router.py` | REINFORCE/RLOO trainer → `weights/<task>/<cfg>.pt`, `runs/<task>/<cfg>.json` (per-step + per-epoch curves: reward, ΔCE, keep_frac, T2/T, w_gold, b, grad norm; train and val) |
| `run_router_local.sbatch` | one task per GPU: train → eval; resumable, task lock for twin jobs (`runs/.lock_<task>`; a `scancel`led job can leave it behind — remove it once the owner job is gone) |
| `fetch_train_rows.py` | train/val rows + disjointness proof |
| `smoke_cpu.py` | CPU smoke (features, drop semantics incl. all-dropped doc, batched CE, driver schemes, disjointness) |
| `analyze_router.py` | Pareto envelope (oracle λ) vs baselines → `router_vs_grid.json` |
| `summarize_sweep.py` | val-selected λ headline table + verdicts → `headline.json` |
| `analyze_weights.py` | gold weight, position profile, vocab scores, seed correlation → `runs/<task>/weights_analysis.json` |

```
HF_HUB_OFFLINE=1 python debug/learned_router/smoke_cpu.py
TASK=nq sbatch --partition=berkeleynlp --qos=preemptive_high_sewonm --nodelist=horton debug/learned_router/run_router_local.sbatch
```

## Differentiable router (`train_diff_router.py`, 2026-09-27)

Same `LinearRouter` and features, trained through a **relaxed removal** instead of REINFORCE;
evaluated with the same exact hard removal (driver schemes `router_diff_tau<τ>`, weights
`weights/<task>/diff_tau<τ>.pt`, results `debug/devloss_grid/results_router_diff/`).

* **Relaxation** (`model(..., soft_keep=p)`, default-off kwarg; absent → bit-identical):
  attention layers add `log(p_key + 1e-8)` to every query's logits (causal masked SDPA; exact at
  p ∈ {0,1}); GatedDeltaNet scales β and g (log decay) by p (p = 0 → identity state step) and runs
  `CausalConv1d.forward_soft_keep`, a **removal-aware short conv**: gated scan registers hold the
  previous *kept* tokens, so the conv of the compacted row is reproduced exactly at binary p. The
  spec's first suggestion (scale each token's conv input by p) left a large gap -- binary-gate
  relaxed vs hard ΔCE on nq: mean |gap| 0.27 nats vs |ΔCE| 0.60, corr 0.71; with the removal-aware
  conv: 0.0008 nats, corr 0.99999 (`runs/<task>/diff_gap.json`, logged at every job start).
* **Estimator**: hard-concrete gates (log α = router logit, β 2/3 → 0.1 over the first 60% of
  epochs, binary straight-through forward from 75%), straight-through [0,1] clamp so dropped gates
  keep receiving dCE/dz (plain clamp → dead gates, keep collapsed to 0 and never recovered).
* **Objective**: minimise E[keep] (L0) s.t. mean ΔCE ≤ ε = max(τ · mean CE_full(train), 0.005 nats).
  Lagrangian `keep_L0 + μ (ΔCE/ε − 1)`; **dual ascent once per epoch on the deployed policy's**
  train ΔCE (deterministic gate, exact hard removal on every train row), step 0.2, violation clipped
  to [−1, 2]. Adam β₂ = 0.8 (the ×μ/ε constraint gradients otherwise pin Adam's second moment and
  freeze the L0 descent), lr 0.05 / 0.01 (emb), 4 rows/step, 60 epochs, init p = 0.95.
* **Selection**: most compact epoch whose VAL mean ΔCE (exact hard removal, deterministic gate)
  ≤ max(τ · mean CE_full(val), 0.005). τ values whose train/val ε coincide (floor) share one run.
* Tests: `test_relax_cpu.py` (attention-only tiny model: p=1 ≡ full, binary p ≡ hard compaction,
  float64 gradcheck; removal-aware conv ≡ compacted conv, float64 gradcheck; hard-concrete), and
  `test_relax_gpu.py` (tiny hybrid on GPU: p=1 ≡ full, directional finite differences vs autograd,
  binary relaxed ≡ hard compaction to 2e-5). Launcher: `run_diff_local.sbatch` (`SELFTEST=1` runs
  the GPU test first). Summary: `summarize_diff.py` → `headline_diff.json`.

## End-to-end router (`train_e2e_router.py`, 2026-09-28/29)

The router is trained through the same relaxation with **budget calibration**: one offset per step
makes the batch's expected keep equal ρ_t, and the loss is CE + KL. There is no rule init and no
distillation. Full write-up: `records/learned-token-router.md` §8.

* **Launcher:** `run_e2e_local.sbatch` (env `TASK, RHOS, NAME_PREFIX, N_TRAIN, N_VAL, EPOCHS,
  TRAIN_EXTRA, XTASKS, XTASK_NAME`).
* **Label-free cutoffs** go through `--match-comps` / `--match-weights`, which write
  `weights/<task>/<name>_c<x>.pt`.
* **Per-length calibration** uses `--calib-root/--calib-rung/--calib-skip/--calib-n`, plus
  `--match-keep-2k`; it writes `_c<x>_<rung>.pt` and `_k2k_<rung>.pt`.
* **In-process test eval** uses `--eval-only --eval-rungs --eval-rows --eval-schemes --eval-out`.
* **Cross-task training:** `--xtasks a,b,c --xtask-name X` runs one frozen model per task. With
  many tasks, add `--xtask-offload`: one model is resident on the GPU and the rest are parked on
  CPU. Rows are visited in task blocks, so a swap costs ~2.7 s. Pass `--mem=320G` for 17 tasks.
* **Summaries:**
  * `summarize_vsfl.py`: per task vs `gold_fl20p8_noslot` → `headline_vsfl.json`.
  * `summarize_xtask.py`: cross-task vs the bar and vs the per-task routers.
  * `summarize_lencal.py`: 8k/32k fixed vs per-length cutoffs → `headline_lencal.json`.
  * `inspect_e2e_rows.py`: keep fractions for gold, id-region and routed tokens.
