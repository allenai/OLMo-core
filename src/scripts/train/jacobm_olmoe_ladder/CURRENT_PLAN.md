# Current MoE A0 Ladder Plan

This is the active operating plan for the JacobM MoE A0 baseline ladder. Use it
with `HANDOFF.md`, `LADDER_PROCESS.md`, `RUNS.md`, and `ANALYSIS.md`.

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

## Current Jobs

Continue:

- 810M Cx8: `2e-4`, `4e-4`, `8e-4`
- 1.2B Cx4: `1.5e-4`, `3e-4`, `6e-4`
- 810M Cx2: `1.5e-4`, `3e-4`, `6e-4`, `1.2e-3`

Ignore unless explicitly resumed:

- Cancelled 810M Cx8 `1e-4`
- Cancelled 1.2B Cx4 `1.2e-3`

Monitor midpoint smoke:

- Beaker: https://beaker.org/ex/01KTMJY87YW09KHB4H6ERGZQ4K
- Name: `olmoe3-moe-a0-mid-480m-smoke-b256k-gpu4-ep1mb8-lr1.2e-3-r1`
- Fresh job: `01KTMJY8HEXVFS5T6MT2G1CNNZ`

## Midpoint / `mid_480m`

Current smoke settings:

- `gpu4-ep1mb8`
- EP=1
- Cx `0.02`
- LR `1.2e-3`
- batch 262,144 tokens / `global_batch_size_seq=32`
- no in-loop evals

Smoke success criteria:

- exact params logged
- no OOM
- skipped steps 0
- W&B logging
- loss decreasing
- checkpoint path correct
- TFLOPs/GPU around the 600+ target

If smoke OOMs, try `gpu4-ep1mb4`.

If smoke has obvious memory headroom, consider whether `mbz=16` is worth testing
before full jobs.

After smoke passes, launch Cx1/Cx2/Cx4 together:

| Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | LR grid |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 262,144 | 32 | 4 | 1 | 8 | `6e-4`, `1.2e-3`, `2.4e-3` |
| 2 | 524,288 | 64 | 4 | 1 | 8 | `3e-4`, `6e-4`, `1.2e-3` |
| 4 | 524,288 | 64 | 4 | 1 | 8 | `4e-4`, `8e-4`, `1.6e-3` |

If midpoint Cx1/Cx2/Cx4 bracket cleanly, later launch midpoint Cx8/Cx16 using
3-point centered sweeps from the refit rule.

## Next Baseline Progression

- If 810M Cx8 brackets cleanly, prepare/launch 810M Cx16 with 3 centered LRs.
- If 1.2B Cx4 brackets cleanly, prepare/launch 1.2B Cx8 with 3 centered LRs.
- After midpoint results land, refit size transfer using 275M, midpoint, 810M,
  and 1.2B.

## Autonomy Bounds

Allowed without asking:

- Inspect Beaker/W&B.
- Update, commit, and push docs/plots/bookkeeping.
- Stop clearly accidental duplicate jobs.
- Monitor and debug the queued `mid_480m` smoke.
- Launch midpoint Cx1/Cx2/Cx4 after smoke passes.
- Launch agreed next baseline sweeps when LRs are determined by completed,
  bracketed fits under the active goal.

Ask before:

- Changing architecture beyond the agreed `mid_480m` config.
- Changing data mix, tokenizer, optimizer family, or schedule shape.
- Launching beyond Cx16.
- Using more than 8 GPUs for one job.
- Cancelling healthy non-duplicate full runs.
- Using validation/eval metrics for LR selection.
- Starting ablation experiments beyond the baseline ladder.
