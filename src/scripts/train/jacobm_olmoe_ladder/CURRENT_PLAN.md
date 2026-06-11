# Current MoE A0 Ladder Plan

This is the active operating plan for the JacobM MoE A0 baseline ladder and the
approved first post-baseline ablation, expert granularity. Use it with
`HANDOFF.md`, `LADDER_PROCESS.md`, `RUNS.md`, and `ANALYSIS.md`.

## Cadence

- Default monitoring cadence: a real 4-hour sleep between checks while current
  jobs are long-running.
- Shorten cadence only for launch/debug/startup/OOM checks, jobs close to
  finishing, or a smoke test that has just started and needs first-step
  validation.
- Do not poll Beaker/W&B every few seconds unless actively debugging startup.

## Plotting

Maintain both comparison directions:

- Per-model plots: each model/Cx plus one aggregate per model across Cx. These
  compare Chinchilla multiples while holding model size fixed.
- Per-Cx plots: one aggregate per Cx across all model sizes. These compare model
  sizes while holding data multiple fixed.
- Per-Cx aggregate files are emitted as `cx<N>_all_models_uplot.png`.

## LR Selection

- Use completed full training runs only for LR-selection decisions.
- Use training loss only for LR selection for now. Collect/backfill evals, but
  do not use validation/eval metrics to choose LRs yet.
- Use coarse factor-of-two-spaced LR sweeps by default.
- When a credible transferred optimum exists, prefer 3 LRs centered around it.
- Use 4 LRs only when the prediction is weak, the rung is known-weird, or
  bracketing risk is high.
- If the best LR is on an edge, launch bounded extensions in the improving
  direction.
- If a monotonic curve keeps improving, include occasional farther sentinel runs.
- Once a rung has 3+ bracketed completed points, fit quadratic loss vs
  `log10(lr)`, using 3 or 5 local points.
- Report both best observed LR and fitted optimal LR; ignore fits outside the
  observed bracket.
- Compare transfer fits with and without oddball Cx2 where relevant.

## Architecture

Baseline lineage is MoE A0.

Keep fixed:

- 48 experts
- `top_k=4`
- `moe_hidden_size=d_model`
- one shared expert
- `shared_mlp_hidden_size=d_model/2`
- one dense prefix layer
- `dense_layer_mlp=4*moe_hidden_size+shared_hidden_size`
- GQA with `n_kv_heads=n_heads//2`

Implemented sizes:

- `275m`
- `mid_480m`
- `810m`
- `1p2b`

`mid_480m` is newly added and estimated at about 480M active params including
embeddings/head and about 2.6B total before smoke confirmation.

## Default Job Settings

For future baseline launches, use this GPU allocation table unless a run needs
a smoke-tested memory fallback or the cluster is temporarily constrained:

| Model | Cx1 | Cx2 | Cx4 | Cx8 |
| --- | ---: | ---: | ---: | ---: |
| 275M | 1-2 GPUs | 2 GPUs | 4 GPUs | 8 GPUs |
| mid_480m | 4 GPUs | 4 GPUs | 4 GPUs | 8 GPUs |
| 810M | 8 GPUs | 8 GPUs | 8 GPUs | 16 GPUs |
| 1.2B | 8 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |

Keep EP=1 by default. Preserve the intended dense-ladder global batch rule for
each Cx; increasing GPU count should primarily reduce wall-clock, not silently
change the optimizer batch. Use the known-good microbatch for each model size
unless a smoke test or prior run proves a larger microbatch is healthy.

## Current Jobs

Continue:

- 810M Cx8: `2e-4`, `4e-4`, `8e-4` completed and bracketed.
- 810M Cx16: `2e-4`, `4e-4`, `8e-4` were stopped intentionally on 2026-06-09
  after the plan shifted to finish Cx1/Cx2/Cx4/Cx8 first. Ignore unless
  explicitly resumed later.
- 1.2B Cx4: `1.5e-4`, `3e-4`, `6e-4` completed with a center near `4e-4`, but
  `3e-4`/`6e-4` are too close to claim a strict hot-side bracket. The stopped
  `1.2e-3` run was resumed on 2026-06-10 as a hot-side completion point under
  Beaker experiment `01KTHW6ZSXGD1P8NEA7S3KM198`, new job
  `01KTSB2H1TMF7Z1T2MY40J2QM0`. The short W&B history for `1.5e-4` was repaired
  with `--refresh-stale-cache` on 2026-06-10.
- 810M Cx2: `1.5e-4`, `3e-4`, `6e-4`, `1.2e-3` completed and bracketed.

Expert granularity:

- This is the approved first post-baseline ablation. Track it separately from
  the baseline in docs, W&B tags, launchers, cache entries, and plots.
- Variants under test:
  - `coarse_24e_top2`: 24 experts, top-2, `moe_hidden_size=2*d_model`.
  - `fine_96e_top8`: 96 experts, top-8, `moe_hidden_size=d_model/2`.
  - Baseline/control remains `baseline_48e_top4`.
- 275M Cx1 transfer probes were queued at `1e-3`, `2e-3`, `4e-3` for both
  non-baseline variants. Coarse uses `gpu1-ep1mb16`; fine uses `gpu1-ep1mb8`
  after the `mb16` smoke OOM. Both Cx1 variants have finished and are
  bracketed. Coarse observed best is `2e-3` with fitted `lr* ~= 1.86e-3`; fine
  observed best is `2e-3` with fitted `lr* ~= 2.10e-3`. Do not launch Cx1
  follow-ups before the next human review.
- 275M Cx4 baseline-centered probes are queued at `8e-4`, `1.6e-3`, `3.2e-3`
  for both non-baseline variants. Coarse uses `gpu4-ep1mb16`; fine uses
  `gpu4-ep1mb8`. Coarse Cx4 has finished and is bracketed with observed best
  `1.6e-3` and fitted `lr* ~= 1.57e-3`. Fine Cx4 has finished and is bracketed
  with observed best `1.6e-3` and fitted `lr* ~= 1.45e-3`.
- All approved expert-granularity 275M Cx1/Cx4 curves are complete and
  bracketed. Do not queue the rest of the 275M expert-granularity ladder until
  Jacob reviews the results and explicitly approves the next batch.
- After Cx1/Cx4 complete, estimate variant LR multipliers relative to the
  baseline and use those multipliers to center later Cx2/Cx8/Cx16 sweeps.

Ignore unless explicitly resumed:

- Cancelled 810M Cx8 `1e-4`
- 1.2B Cx4 `1.2e-3` was previously stopped but is now resumed; include only
  after the resumed full run completes.

Midpoint smoke status:

- Original long-name smoke `01KTMJY87YW09KHB4H6ERGZQ4K` failed before
  training because the W&B group name exceeded 128 characters.
- Short-name retry `01KTMM5YQTGA9TXKFYMF5NPB46`
  (`m480-smoke-gpu4-ep1mb8-lr12-r2`) passed startup and was stopped
  intentionally after validation.
- The validated setting is `gpu4-ep1mb8`: skipped steps 0, loss decreasing,
  and roughly 632-644 TFLOPs/GPU after warmup.

## Midpoint / `mid_480m`

Validated settings:

- `gpu4-ep1mb8`
- EP=1
- batch 262,144 tokens / `global_batch_size_seq=32`
- full jobs use in-loop fast evals every 2000 steps

Queued Cx1/Cx2/Cx4 together on 2026-06-08:

| Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | LR grid |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 262,144 | 32 | 4 | 1 | 8 | `6e-4` (`01KTMMJCV3818NDPK51R89MH08`), `1.2e-3` (`01KTMMJSTMY3TSR7MHH5G7M22H`), `2.4e-3` (`01KTMMK7VN9BXSCBYX2HQKQQWH`) |
| 2 | 524,288 | 64 | 4 | 1 | 8 | `3e-4` (`01KTMMKN716ZSRZN473CV4BC23`), `6e-4` (`01KTMMM35QKDE15XCSKG76Z6ST`), `1.2e-3` (`01KTMMMHBEV4JW3N0X4X3MFHK8`) |
| 4 | 524,288 | 64 | 4 | 1 | 8 | `4e-4` (`01KTMMMZ1539AV33SHB12S17Q4`), `8e-4` (`01KTMMNC9R56MX1MSGZQ865SXA`), `1.6e-3` (`01KTMMNTA3NN9K4THQXCKGP717`) |

Follow-ups launched on 2026-06-10:

- Cx4 cold-side sentinel `1e-4`, `gpu4-ep1mb8`:
  `01KTSC4J4KGTZXY0XP5P0AXQXM`, W&B `0mvi3nov`. Finished cleanly with avg250M
  2.4689, much worse than the existing Cx4 best `8e-4` avg250M 2.3788. Treat
  this as extra cold-side insurance only.
- Cx8 hot-side sentinel `3.2e-3`, `gpu8-ep1mb4`:
  `01KTSC51ZXE3YQAZMANDP15QT7`, W&B `fvbz0h7v`. Finished cleanly with avg250M
  2.3486, worse than the existing Cx8 best `8e-4` avg250M 2.3076. Midpoint Cx8
  is now bracketed on the hot side; no additional Cx8 hot extension is needed.

Hold off on midpoint Cx16 for now, matching the 810M/1.2B policy.

## Next Baseline Progression

- 810M Cx8 bracketed cleanly around `4e-4`.
- 1.2B Cx4 has an apparent center near `4e-4`, but wait for the resumed
  `1.2e-3` hot-side point before calling the rung bracketed.
- After midpoint results land, refit size transfer using 275M, midpoint, 810M,
  and 1.2B.
- For now, baseline expansion for 810M, 1.2B, and midpoint should focus on
  Cx1/Cx2/Cx4/Cx8. Do not launch new 810M/1.2B/midpoint Cx16 jobs; fill Cx16
  gaps only after a separate discussion.

## Autonomy Bounds

Allowed without asking:

- Inspect Beaker/W&B.
- Update, commit, and push docs/plots/bookkeeping.
- Stop clearly accidental duplicate jobs.
- Monitor and debug the queued `mid_480m` smoke.
- Launch midpoint Cx1/Cx2/Cx4 after smoke passes.
- Launch agreed next baseline sweeps when LRs are determined by completed,
  bracketed fits under the active goal.
- Monitor the approved expert-granularity Cx1/Cx4 jobs.
- Update, commit, and push expert-granularity docs/plots/bookkeeping.
- Launch at most one targeted expert-granularity Cx1 or Cx4 follow-up per
  variant if the completed three-point curve lands on an edge.

Ask before:

- Changing baseline architecture beyond the agreed `mid_480m` config.
- Starting any new ablation family beyond expert granularity.
- Launching expert-granularity Cx2/Cx8/Cx16 or larger-model promotions.
- Changing data mix, tokenizer, optimizer family, or schedule shape.
- Launching beyond Cx16.
- Using more than 8 GPUs for one job.
- Cancelling healthy non-duplicate full runs.
- Using validation/eval metrics for LR selection.
