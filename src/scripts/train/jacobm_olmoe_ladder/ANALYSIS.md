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

The `--current-family` flag keeps only the current partial-node relaunches, such
as `gpu2-ep1mb16`, `gpu4-ep1mb8`, and `gpu4-ep1mb16`, which avoids comparing
against canceled lower-throughput predecessors. Omit it for full historical
tables, and use `--mode final --finished-only` when making final U-plots.

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

Cx4 completed `avg250M`:

| LR | State | avg250M |
| ---: | --- | ---: |
| `5e-4` | finished | 2.5773 |
| `7e-4` | finished | 2.5699 |
| `1e-3` | finished | 2.5644 |
| `1.5e-3` | finished | 2.5611 |
| `2.5e-3` | finished | 2.5648 |
| `3.5e-3` | finished | 2.5749 |
| `5e-3` | finished | 2.5887 |

Cx4 is bracketed and centered around `1.5e-3`. The low side worsens below
`1e-3`, the high side worsens above `2.5e-3`, and `5e-3` is clearly too hot.

## 275M Cx8/Cx16 Rule Completion

Before launching the full Cx8/Cx16 grid, smoke the high-microbatch settings.
The original preferred full-grid settings were:

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

Retry smoke jobs launched from commit `6a465f0d`:

- `olmoe3-tiny-275m-cx8-smoke-b768k-gpu2-ep1mb16-lr5e-4`
  (`01KT829008RED7EA12EP2J2KSV`)
- `olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb16-lr5e-4`
  (`01KT829CG1KNWB9GRVVZ8HAY0T`)

Both retries reached the final smoke step with finite loss, `optim/step
skipped=0`, and step throughput above the 600 TFLOPs/GPU target. Cx8 finished
step 513/513 at about 693 TFLOPs/GPU for the final logged step, with actual
average throughput about 596 TFLOPs/GPU after data/checkpoint overhead. Cx16
finished step 385/385 and exited cleanly, with final logged step throughput in
the same 680-700 TFLOPs/GPU range.

An accidental second Cx16 retry,
`olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb16-lr5e-4-1ae64a69`
(`01KT82H1A9XT62VC8XRVZHJPWV`), was stopped manually because it duplicated the
canonical Cx16 smoke and used the same checkpoint root. Ignore it for analysis.

Launcher / dry-run command printer:

```bash
src/scripts/train/jacobm_olmoe_ladder/reproduce_tiny_275m_cx8_cx16_lr_rule.sh
```

Actual launcher:

```bash
src/scripts/train/jacobm_olmoe_ladder/launch_tiny_275m_cx8_cx16_lr_rule.sh
```

Use coarse factor-of-two LR sweeps for new rungs, then bisect/extend only after
the initial bracket is visible. This replaces the earlier too-granular grids.

Current Cx8 LR grid:

- `2e-4`
- `4e-4`
- `8e-4`
- `1.6e-3`

Current Cx16 LR grid:

- `1e-4`
- `2e-4`
- `4e-4`
- `8e-4`

Future 275M launches do not need to be as GPU-conservative as the first
Cx8/Cx16 smokes. While 2-GPU `ep1mb16` is viable, use 4 GPUs for Cx16 and
consider 4 GPUs for Cx8 as well, so these cheap 275M runs finish faster while we
have ample B200 capacity.

The first granular Cx8/Cx16 grid was stopped manually and should be ignored for
analysis:

- Cx8 `3e-4`: `01KT836RV6PHBQ3M5SCVYECWC4`
- Cx8 `5e-4`: `01KT8373YTV4C1G08D20XXMHZG`
- Cx8 `7e-4`: `01KT837GAWN3H1XBYASH520M2A`
- Cx8 `1e-3`: `01KT837VTNEK7F783YZYHYRJ1Y`
- Cx16 `2e-4`: `01KT8387116V5NF6QNDCSBN9CA`
- Cx16 `3e-4`: `01KT838HX2QMM7RJCYT2P6MW92`
- Cx16 `5e-4`: `01KT838WN9G1WDPG5RWBS0CGTM`
- Cx16 `7e-4`: `01KT83989R9R79D6FA2QXX15J9`

Replacement coarse Cx8/Cx16 jobs launched from commit `6a465f0d`:

- Cx8 `2e-4`, `gpu4-ep1mb8`: `01KT8445FT6GZFKPE7JKS3F8RY`
- Cx8 `4e-4`, `gpu4-ep1mb8`: `01KT844H6AQWZNNSXJZRV258VZ`
- Cx8 `8e-4`, `gpu4-ep1mb8`: `01KT844XB38AM31SCZDSTA1EAA`
- Cx8 `1.6e-3`, `gpu4-ep1mb8`: `01KT84589PJ0VDVSS7CDQPBSCV`
- Cx16 `1e-4`, `gpu4-ep1mb16`: `01KT845KM5CF6JZHJB6KW6WARW`
- Cx16 `2e-4`, `gpu4-ep1mb16`: `01KT845WN987DZTN03Q7NSXAK4`
- Cx16 `4e-4`, `gpu4-ep1mb16`: `01KT8466QCKVK2WDKW7F75TK9H`
- Cx16 `8e-4`, `gpu4-ep1mb16`: `01KT846JGMA8TDYZGGH4E34K3P`

These replacement jobs failed before completion because
`/weka/oe-training-default` filled up during checkpoint writes. Treat them as
partial curves only; do not include them in completed-run U-plots or final LR
rules. Their intermediate checkpoints were later deleted during storage cleanup,
so these runs cannot be resumed.

Partial Cx8 evidence from the long storage-failed runs:

| LR | Step | TokensB | avg250M |
| ---: | ---: | ---: | ---: |
| `2e-4` | 18,999 | 14.941 | 2.6778 |
| `4e-4` | 18,999 | 14.941 | 2.6522 |
| `8e-4` | 17,999 | 14.155 | 2.6672 |
| `1.6e-3` | 16,999 | 13.369 | 2.7016 |

This is not a final comparison, but it suggests `4e-4` is the strongest coarse
Cx8 candidate so far and `1.6e-3` is likely too hot. If rerunning from scratch,
prefer a narrower Cx8 grid around `4e-4`/`8e-4` rather than repeating the full
four-point coarse grid.

Partial Cx16 evidence from the long storage-failed runs:

| LR | Step | TokensB | avg250M |
| ---: | ---: | ---: | ---: |
| `1e-4` | 11,999 | 12.582 | 2.8690 |
| `2e-4` | 11,999 | 12.582 | 2.7714 |
| `4e-4` | 9,999 | 10.485 | 2.7666 |
| `8e-4` | 5,999 | 6.290 | 2.9042 |

This is also partial-only, but it strongly disfavors the Cx16 endpoints:
`1e-4` is too cold and `8e-4` is too hot. The useful Cx16 rerun range appears
to be centered on `2e-4`/`4e-4`.

Accidental restart jobs from commit `bdd30f9` were stopped immediately after we
noticed they could not resume from deleted checkpoints. Ignore these W&B runs
for analysis: Cx8 `2e-4` (`nda8dyu0`), Cx8 `4e-4` (`9n2gwlx7`), Cx8 `8e-4`
(`rquath33`), Cx8 `1.6e-3` (`32ujpusd`), Cx16 `1e-4` (`9h2gbx4b`), Cx16
`2e-4` (`6cd1cdmy`), Cx16 `4e-4` (`k0xuoc5d`), and Cx16 `8e-4` (`ai2h8nbw`).

Fresh `r2` completion grid launched from commit `2cfd4c56`:

- Cx8 `2e-4`, `gpu4-ep1mb8`: `01KT8JPNQTTSQFDCGNV7HT8VV1`
- Cx8 `4e-4`, `gpu4-ep1mb8`: `01KT8JQ0V85RVSY309P3BXQ85Y`
- Cx8 `6e-4`, `gpu4-ep1mb8`: `01KT8JQCM49JRFVNMT7WRV701V`
- Cx8 `8e-4`, `gpu4-ep1mb8`: `01KT8JQR750TFJKE13ZXY7JYTT`
- Cx16 `2e-4`, `gpu8-ep1mb16`: `01KT8JR3WKXCR6TN8897A57DHS`
- Cx16 `4e-4`, `gpu8-ep1mb16`: `01KT8JRFSG3J7AJ5PV7E32Z46K`
- Cx16 `6e-4`, `gpu8-ep1mb16`: `01KT8JRVAG6RVGT231477NGQD9`

Treat these as the canonical full-run sources for Cx8/Cx16 U-plots. The Cx16
`2e-4` `r2` experiment had its first job fail around step 4507 with a CUDA/NCCL
watchdog abort. It did not show a loss blow-up or storage-full checkpoint
failure before the abort, so the same Beaker experiment was resumed to use the
latest checkpoint:

- Cx16 `2e-4`, `gpu8-ep1mb16`, `r2` resumed job:
  `01KT8P9WZJ20XGTY44BH38M9W2`

An accidental fresh-from-scratch `r3` replacement
(`01KT8NJ55CHAKYCG1E1J7Q9QBJ`) was stopped early. Ignore it for analysis.

The earlier storage-failed partials are useful only as qualitative pruning
evidence.

Cx8 full-run results from the fresh `r2` grid:

| LR | State | Step | TokensB | avg100M | avg250M | avg500M |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2e-4` | finished | 40971 | 32.221 | 2.5450 | 2.5429 | 2.5422 |
| `4e-4` | finished | 40971 | 32.221 | 2.5113 | 2.5092 | 2.5085 |
| `6e-4` | finished | 40971 | 32.221 | 2.4999 | 2.4978 | 2.4972 |
| `8e-4` | finished | 40971 | 32.221 | 2.4929 | 2.4909 | 2.4903 |

The best observed Cx8 LR is `8e-4`, but it is the high edge of the completed
grid, so the rung is not bracketed. Quadratic fits to loss vs log10(LR) are
therefore not trusted yet: the 3-point fit over `4e-4`/`6e-4`/`8e-4` points
outside the bracket at about `5.9e-3`, while the 4-point fit points just beyond
the bracket at about `1.34e-3`. After the initial `1.6e-3` extension also looked
too hot early, launched two more aggressive high-side probes to make the bracket
unambiguous:

- Cx8 `1.6e-3`, `gpu4-ep1mb8`, `r2`: `01KT9D6W9F4RGA5RSA8XSSMEP3`
- Cx8 `3.2e-3`, `gpu4-ep1mb8`, `r2`: `01KT9Q661N0YHYHC9A9T9AGV1J`
- Cx8 `6.4e-3`, `gpu4-ep1mb8`, `r2`: `01KT9Q6HX5X6KFW5RD1VSC9BV4`

Cx16 completed full-run results from the canonical `r2` grid:

| LR | State | Step | TokensB | avg100M | avg250M | avg500M |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2e-4` | finished | 61457 | 64.442 | 2.4665 | 2.4759 | 2.4744 |
| `4e-4` | finished | 61457 | 64.442 | 2.4381 | 2.4474 | 2.4461 |
| `6e-4` | finished | 61457 | 64.442 | 2.4274 | 2.4367 | 2.4354 |

The best observed Cx16 LR is `6e-4`, but it is the high edge of the completed
grid, so the rung is not bracketed. The 3-point quadratic fit to loss vs
log10(LR) points outside the bracket at about `1.34e-3`, so do not trust the
fitted optimum yet. After the initial `1.2e-3` extension also looked too hot
early, launched two more aggressive high-side probes to make the bracket
unambiguous:

- Cx16 `1.2e-3`, `gpu8-ep1mb16`, `r2`: `01KT9H6XQJ2GEMKPKHKPCED5B1`
- Cx16 `2.4e-3`, `gpu8-ep1mb16`, `r2`: `01KT9Q6X0B6PG3G6ZSBZGTPSVQ`
- Cx16 `4.8e-3`, `gpu8-ep1mb16`, `r2`: `01KT9Q774FWC6NZDSGTD0Y2W7K`

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

Relaunched larger-model `r2` smokes from commit `6a465f0d`:

- 810M `gpu2-ep1mb16`: `01KT82MKYTJDPBRN01P0XHXS38`
- 810M `gpu2-ep1mb8`: `01KT82MYJXY98QSK4CMES19PH9`
- 1.2B `gpu2-ep1mb8`: `01KT82NA11D9E3DEH1NPXDR6WJ`
- 1.2B `gpu2-ep1mb4`: `01KT82NN0Q5MPJQ167W5XY8BQA`

All four `r2` smokes failed during dry-run batch allocation with real OOMs, so
the model-size CLI path is fixed but two-GPU EP=1 is too tight for the larger
models:

- 810M `gpu2-ep1mb16`: OOM on a 320 MiB allocation with about 250 MiB free.
- 810M `gpu2-ep1mb8`: OOM on a 12.25 GiB allocation with about 11.64 GiB free.
- 1.2B `gpu2-ep1mb8`: OOM on a 1.50 GiB allocation with about 218 MiB free.
- 1.2B `gpu2-ep1mb4`: OOM on a 6.12 GiB allocation with about 3.57 GiB free.

Next larger-model smoke direction: try more GPUs before changing the
architecture. For 810M, a 4-GPU EP=1 smoke with `mb8` or `mb4` is the most
direct next probe. For 1.2B, start with 4-GPU EP=1 `mb4`; if it still OOMs,
fall back to EP.

Relaunched larger-model `r3` smokes:

- 810M `gpu4-ep1mb8`: `01KT840XF9T975KJM3SHXFCH7D`
- 810M `gpu4-ep1mb4`: `01KT8418KTXB8Z26DVJF8VRSGD`
- 1.2B `gpu4-ep1mb4`: `01KT841MWJKFK5KWCAXCGA9WC1`
- 1.2B `gpu4-ep2mb4`: `01KT8420RGXWJ8C3JFCNG67W2T`

`r3` smoke outcomes:

- 810M `gpu4-ep1mb8` OOMed during dry-run/backward allocation.
- 810M `gpu4-ep1mb4` finished cleanly with `optim/step skipped=0`, about
  654 TFLOPs/GPU on the final logged step, and about 610 TFLOPs/GPU actual
  average. Use `gpu4-ep1mb4` as the validated Cx1 setting.
- 1.2B `gpu4-ep1mb4` finished cleanly with `optim/step skipped=0`, about
  682 TFLOPs/GPU on the final logged step, and about 662 TFLOPs/GPU actual
  average. Treat EP=1 as the preferred setting.
- 1.2B `gpu4-ep2mb4` also finished cleanly, with lower actual average
  throughput around 607 TFLOPs/GPU. Keep EP=2 only as a memory fallback.

For 810M full probes, use `gpu4-ep1mb4` for Cx1 and `gpu8-ep1mb4` for Cx4;
the 8-GPU Cx4 setting does not need a separate smoke test.
