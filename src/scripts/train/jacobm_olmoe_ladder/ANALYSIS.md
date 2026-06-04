# Tiny MoE Analysis Notes

## W&B Loss Pulls

Project:

- `ai2-llm/jacobm-olmoe-ladder`

Primary scalar:

- `train/CE loss`

Token progress scalar:

- `throughput/total tokens`

For Cx1 U-plots across different batch sizes, use token-window averages instead
of final-step or final-N-step averages. This avoids comparing 48 points from a
2M-token batch to 383 points from a 256k-token batch over different amounts of
training data.

Current default summary windows:

- final `100M` tokens
- final `250M` tokens
- final `500M` tokens

Command:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py
```

The script uses the local `WANDB_API_KEY`, pulls runs from W&B, filters to
`olmoe3-tiny-275m-cx1` by default, and prints TSV with final loss plus final-token-window
averages. It infers batch size from run-name tags:

- no `b...` tag and `cx1-lr`: 2M tokens/step
- `b128k`: 131,072 tokens/step
- `b256k`: 262,144 tokens/step
- `b512k`: 524,288 tokens/step

Finished-run histories are cached under
`~/.cache/olmoe3_ladder/wandb_histories` by default. The cache key is the W&B run
id and project, and entries are invalidated if W&B state or summary step changes.
Use `--refresh-cache` to force a fresh W&B history scan.

For a narrower pull:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py \
  --name-regex 'olmoe3-tiny-275m-cx1-b256k'
```

For both Cx1 and Cx2:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py \
  --name-regex 'olmoe3-tiny-275m-cx'
```

Plot the Cx1 U-plot:

```bash
uv run --with wandb --with matplotlib python src/scripts/train/jacobm_olmoe_ladder/plot_cx1_uplot.py
```

The plot uses blue for the primary `avg250M` curve, orange dashed for `avg100M`,
and green dotted for `avg500M` so the auxiliary windows are readable in Slack.

For matched-token comparisons across in-flight relaunches, use:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --mode matched \
  --current-family \
  --name-regex 'olmoe3-tiny-275m-cx4'
```

The `--current-family` flag keeps only the partial-node `gpu2-ep1mb16` /
`gpu4-ep1mb16` relaunches, which avoids comparing against canceled lower-throughput
predecessors. Omit it for full historical tables, and use `--mode final
--finished-only` when making final U-plots.

Generate completed-run plots:

```bash
uv run --with wandb --with matplotlib python src/scripts/train/jacobm_olmoe_ladder/plot_wandb_ladder.py \
  --window-m 250 \
  --finished-only
```

Plots are written to `src/scripts/train/jacobm_olmoe_ladder/plots/`. The
committed artifacts include one U-plot per Cx rung and one aggregate U-plot for
the 275M model. By default the plotter only includes the canonical ladder batch
for each rung (`256k` for Cx1/Cx2, `512k` for Cx4, `768k` for Cx8, and `1M` for
Cx16); pass `--include-noncanonical` to include historical batch-size probes.

## 2026-06-02 Snapshot

Initial read after the first 2M-batch sweep finished and the 256k-batch sweep was
partially complete:

- 2M Cx1 best final-token-window result was `1.2e-3`.
- 256k Cx1 `3e-4` finished better than the 2M runs by final-token-window average.
- 256k Cx1 `5e-4` and `8e-4` were better than `3e-4` at matched token counts
  while still running.
- `1e-4` looked too slow and was dropped from the next Cx1 sweeps.

The first follow-up recommendation was:

- 256k LR refinement: `6e-4`, `1e-3`, `1.5e-3`, `2e-3`
- batch probe at strong LRs: `128k@5e-4`, `128k@8e-4`, `512k@5e-4`, `512k@8e-4`

After a later W&B refresh, `5e-4` at 256k looked stronger than `8e-4` by recent
token-window average, so the follow-up LR refinement was tightened to:

- 256k LR refinement: `4e-4`, `6e-4`, `7e-4`, `1e-3`
- batch probe at strong LRs: `128k@5e-4`, `128k@8e-4`, `512k@5e-4`, `512k@8e-4`
- Cx2 transfer check, queued first: `256k@5e-4`, `256k@7e-4`

After all follow-up runs finished, Cx1 `256k@1e-3` and `256k@1.2e-3` were nearly
tied and best among completed Cx1 runs. Because the high side had not clearly
turned over, the next Cx1 high-side probes were:

- `256k@1.5e-3`
- `256k@2e-3`

Those high-side probes gave a shallow Cx1 turn, with local quadratic fits in
log LR estimating the Cx1 optimum around `1.6e-3` to `1.8e-3`. The next 275M
ladder jobs were:

- Cx2 high-side check: `256k@1e-3`
- Cx4 sweep at dense-ladder Cx4 batch rule (`512k`): `1e-3`, `1.5e-3`,
  `2.5e-3`, `3.5e-3`

For a cleaner Cx1 U-plot right side, two extra high-LR Cx1 probes were queued:

- `256k@3e-3`
- `256k@5e-3`

## Throughput Fix

Initial tiny MoE sweeps used `EP_DIM=8` and `MICRO_BSZ=1`, which underutilized
the GPUs. For the tiny MoE, use `EP_DIM=1` and increase per-rank microbatch size
as much as possible while preserving the intended global batch:

- 256k tokens / 32 sequences on 1 node: `--micro-batch-size=4 --ep-dim=1`
- 512k tokens / 64 sequences on 1 node: `--micro-batch-size=8 --ep-dim=1`

This preserves the optimizer batch/schedule while increasing per-GPU work. The
requeued run names include `ep1mb4` or `ep1mb8` to avoid checkpoint collisions
with cancelled lower-throughput jobs.

The current preferred partial-node settings are:

- 256k tokens / 32 sequences on 2 GPUs: `--micro-batch-size=16 --ep-dim=1`
- 512k tokens / 64 sequences on 4 GPUs: `--micro-batch-size=16 --ep-dim=1`

## 2026-06-03 Matched-Token Snapshot

After the partial-node relaunches had progressed, current-family matched-token
tables showed:

- Cx2 at about 2.46B tokens: `1e-3` was the best of the current high-side
  relaunches, while `1.5e-3`, `2.5e-3`, and `3.5e-3` were progressively worse.
  Earlier finished `5e-4`/`7e-4` remain important baselines for final comparison.
- Cx4 at about 5.96B tokens: `1e-3` was best among current `1e-3`, `1.5e-3`,
  `2.5e-3`, and `3.5e-3`; higher LRs were monotonic worse at matched tokens.

Because Cx4 did not have a low-side bracket in the current partial-node family,
the following low-side Cx4 probes were launched:

- `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr5e-4`
- `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr7e-4`

The Cx1 high-side relaunches finished cleanly:

| LR | Final | avg100M | avg250M | avg500M |
| ---: | ---: | ---: | ---: | ---: |
| `3e-3` | 2.6697 | 2.7783 | 2.7813 | 2.7870 |
| `5e-3` | 2.6839 | 2.7868 | 2.7905 | 2.7985 |
| `8e-3` | 2.7225 | 2.8276 | 2.8318 | 2.8416 |

These are all worse than the completed Cx1 basin around `1.5e-3` / `2e-3`
(`avg250M` about 2.762), so Cx1 is bracketed on the high side.

## 2026-06-04 Completed Cx2/Cx4 Snapshot

Final completed-run summaries use final-token-window averages and ignore canceled
or failed partial predecessors.

Cx2 completed `avg250M`:

| LR | State | avg250M |
| ---: | --- | ---: |
| `5e-4` | finished | 2.6644 |
| `7e-4` | finished | 2.6569 |
| `1e-3` | finished | 2.6647 |
| `1.5e-3` | finished | 2.6663 |
| `2.5e-3` | finished | 2.6775 |
| `3.5e-3` | finished | 2.6897 |

Cx2 is bracketed. The best completed point is `7e-4`, with `5e-4` and `1e-3`
close but worse, and the high side degrades monotonically after `1e-3`.

Cx4 completed `avg250M` for the current middle/high bracket:

| LR | State | avg250M |
| ---: | --- | ---: |
| `1e-3` | finished | 2.5644 |
| `1.5e-3` | finished | 2.5611 |
| `2.5e-3` | finished | 2.5648 |
| `3.5e-3` | finished | 2.5749 |

Cx4 is provisionally centered around `1.5e-3`, but it is not final-bracketed
until the running `5e-4`, `7e-4`, and `5e-3` full runs finish.

## 275M Cx8/Cx16 Rule Completion

Before launching the full Cx8/Cx16 grid, smoke the high-microbatch settings.
The preferred full-grid settings, if smokes are healthy, are:

- Cx8: 768k tokens / 96 sequences on 2 GPUs with `--micro-batch-size=24`
- Cx16: 1M tokens / 128 sequences on 2 GPUs with `--micro-batch-size=32`

Smoke launcher / dry-run command printer:

```bash
src/scripts/train/jacobm_olmoe_ladder/reproduce_tiny_275m_cx8_cx16_smoke.sh
```

Smoke jobs launched:

- `olmoe3-tiny-275m-cx8-smoke-b768k-gpu2-ep1mb24-lr5e-4`
- `olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb32-lr5e-4`

Both initial high-microbatch smokes OOMed during the dry-run batch:

- Cx8 `gpu2-ep1mb24`: about 170 GiB in use before a 36.75 GiB allocation.
- Cx16 `gpu2-ep1mb32`: about 173 GiB in use before a 49.00 GiB allocation.

Next smoke direction: retry EP=1 with smaller microbatches before introducing
expert parallelism. Candidate settings are Cx8 `gpu2-ep1mb16` and Cx16
`gpu2-ep1mb16`, then consider more GPUs or EP only if those are unhealthy.

Launcher / dry-run command printer:

```bash
src/scripts/train/jacobm_olmoe_ladder/reproduce_tiny_275m_cx8_cx16_lr_rule.sh
```

Actual launcher:

```bash
src/scripts/train/jacobm_olmoe_ladder/launch_tiny_275m_cx8_cx16_lr_rule.sh
```

Initial Cx8 LR grid:

- `3e-4`
- `5e-4`
- `7e-4`
- `1e-3`

Initial Cx16 LR grid:

- `2e-4`
- `3e-4`
- `5e-4`
- `7e-4`

## 810M and 1.2B Baseline Prep

`tiny_275m.py` now accepts `--model-size` with:

- `275m`: 1.13B total / 0.28B active / 0.20B active non-embedding
- `810m`: 4.93B total / 0.82B active / 0.69B active non-embedding
- `1p2b`: 7.76B total / 1.22B active / 1.06B active non-embedding

The larger model lineage keeps the same MoE structure: 48 experts, top-4 routing,
MoE hidden size equal to `d_model`, one shared expert with hidden size
`d_model / 2`, one dense prefix layer, and GQA with half as many KV heads as
query heads.

Prepared smoke launcher:

```bash
src/scripts/train/jacobm_olmoe_ladder/reproduce_moe_a0_size_smoke.sh
```

The smoke plan probes EP=1 first:

- 810M: 2 GPUs, microbatch 16, then 2 GPUs, microbatch 8 fallback
- 1.2B: 2 GPUs, microbatch 8, then 2 GPUs, microbatch 4 fallback

Use throughput and memory from these smokes to decide whether EP is unnecessary
for baseline runs or whether 1.2B needs an EP fallback.

Smoke jobs launched:

- `olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb16-lr5e-4`
- `olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb8-lr5e-4`
- `olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb8-lr3e-4`
- `olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb4-lr3e-4`

The first larger-model smokes failed before reaching training because the local
`--model-size` script change had not yet been available in the launched runtime;
`model_size` leaked into `ExperimentConfig.merge()` as an unknown config field.
`tiny_275m.py` now consumes `model_size=...` style script overrides before merge
as a compatibility guard, but the larger-model smokes should be relaunched only
from a commit that includes the model-size support.
