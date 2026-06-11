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
For routine analysis, do not pass a refresh flag; this uses cached finished-run
histories and only downloads missing/invalidated entries. If W&B history was
short when a run first finished, use `--refresh-stale-cache` on the narrowest
possible run family. Reserve `--refresh-cache` for a deliberately full
re-download of every selected finished run.

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

If a newly completed family had stale/short W&B history, repair only that family
before plotting:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --name-regex '<specific-finished-run-family>' \
  --mode final --finished-only --windows-m 100 250 500 --refresh-stale-cache
```

Completed-run per-rung plots split lines by run family when a rung contains
multiple settings, such as Cx1 `original`, `n2`, and `gpu2-ep1mb16`. Do not
connect these families into a single U-curve; use the family-specific lines for
LR-rule fits. The aggregate model plot shows only the canonical family for each
rung.

Plots are written to `src/scripts/train/jacobm_olmoe_ladder/plots/`. The
committed artifacts include one U-plot per model/Cx, one aggregate U-plot per
model, and one aggregate U-plot per Cx across model sizes. By default the
plotter only includes the canonical ladder batch for each model/rung; for
example, 275M Cx2 uses `256k`, while 480M/810M Cx2 use `512k`. Pass
`--include-noncanonical` to include historical batch-size probes.

Diagnostic runs with `sanity` in the name are excluded from the standard plotter
and from LR-rule fits. These are for controlled settings-difference checks, not
for the ladder sweep itself.

## 2026-06-09 810M Cx2 Status

The full 810M Cx2 sweep has completed:

| LR | State | Tokens | avg100M | avg250M | avg500M | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `1.5e-4` | finished | 27.603B | 2.3544 | 2.3608 | 2.3589 | `fcqkb55w` |
| `3e-4` | finished | 27.603B | 2.3245 | 2.3308 | 2.3291 | `ogp6mrt6` |
| `6e-4` | finished | 27.603B | 2.3096 | 2.3160 | 2.3144 | `okb4e1u0` |
| `1.2e-3` | finished | 27.603B | 2.3112 | 2.3176 | 2.3161 | `d13uavyt` |

The Cx2 curve is bracketed. The best observed point is `6e-4`, with `1.2e-3`
only slightly worse. Use the plotter's local quadratic estimate for reporting,
but do not launch new work from Cx2 alone; combine it with completed Cx1/Cx4/Cx8
and active Cx16 evidence when updating the 810M LR rule.

## 2026-06-09 810M Cx8 Status

The 810M Cx8 sweep completed as a clean bracket:

| LR | State | Tokens | avg100M | avg250M | avg500M | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `2e-4` | finished | 110.410B | 2.1827 | 2.1844 | 2.1879 | `a0k0519k` |
| `4e-4` | finished | 110.410B | 2.1705 | 2.1721 | 2.1756 | `dkpaicdc` |
| `8e-4` | finished | 110.410B | 2.1752 | 2.1768 | 2.1803 | `rhtrhhet` |

The Cx8 best observed point is `4e-4`. A local 3-point quadratic fit on
`avg250M` estimates the Cx8 optimum at about `4.7e-4`. Combining completed 810M
fits for Cx1, Cx4, and Cx8 gives a shallow LR-vs-Cx rule with Cx16 prediction
around `4.25e-4`, so the next 810M Cx16 sweep is centered as
`2e-4`, `4e-4`, `8e-4`.

## 2026-06-09 480M Cx1 Status

The midpoint `mid_480m` Cx1 sweep completed as a clean 3-point bracket:

| LR | State | Tokens | avg100M | avg250M | avg500M | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `6e-4` | finished | 7.608B | 2.5696 | 2.5676 | 2.5725 | `56vuwauw` |
| `1.2e-3` | finished | 7.608B | 2.5653 | 2.5636 | 2.5690 | `49mybsr0` |
| `2.4e-3` | finished | 7.608B | 2.5839 | 2.5826 | 2.5889 | `7zz7c1zu` |

The best observed point is `1.2e-3`; `2.4e-3` is clearly hot and `6e-4` is
only slightly worse on the cold side. A local 3-point quadratic fit can be used
for the midpoint Cx1 optimum after the plot refresh, but do not update the
baseline transfer rule from this rung alone. Wait for midpoint Cx2/Cx4 and the
remaining 810M/1.2B rungs before drawing a new cross-model conclusion.

## 2026-06-09 480M Cx2 Status

The initial midpoint `mid_480m` Cx2 triplet has completed:

| LR | State | Tokens | avg100M | avg250M | avg500M | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `3e-4` | finished | 15.216B | 2.4828 | 2.4889 | 2.4874 | `ridb7me5` |
| `6e-4` | finished | 15.216B | 2.4597 | 2.4658 | 2.4643 | `9bf5s9lf` |
| `1.2e-3` | finished | 15.216B | 2.4519 | 2.4580 | 2.4567 | `roj7jv11` |

The curve is monotonic over the initial triplet, with best observed avg250M at
the high-edge `1.2e-3`. Per the ladder policy, launched one factor-of-two
high-side extension:

- `2.4e-3`: `01KTPWRQD0Z7SN3KEA6EBMTCB2`
- `9.6e-3`: `01KTQ3V5C3BJNDXHD38BV76KTH`

The `9.6e-3` run is an intentionally far hot-side sentinel, not a dense grid
point. Do not fit or make a final midpoint Cx2 LR decision until the `2.4e-3`
extension and the far sentinel finish.

## 2026-06-10 1.2B Cx4 Status

The 1.2B Cx4 sweep finished successfully:

| LR | State | Tokens | avg100M | avg250M | avg500M | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `1.5e-4` | finished | 85.133B history | 2.1655 | 2.1654 | 2.1679 | `5u5iumvr` |
| `3e-4` | finished | 85.133B history | 2.1500 | 2.1508 | 2.1531 | `rkjs2sze` |
| `6e-4` | finished | 85.133B history | 2.1549 | 2.1548 | 2.1573 | `1tzma107` |

W&B history was repaired with `--refresh-stale-cache` on 2026-06-10. The low
side is clearly worse, but `3e-4`/`6e-4` are close enough that this is not a
strict hot-side bracket. Local 3-point quadratic fits over log LR estimate the
optimum around:

- avg100M: `3.6e-4`
- avg250M: `3.7e-4`
- avg500M: `3.6e-4`

Treat 1.2B Cx4 as weakly centered around `4e-4`, but not fully bracketed until
there is an actual right-side upturn. The previously stopped `1.2e-3` hot-side
run was resumed on 2026-06-10 under the same Beaker experiment
`01KTHW6ZSXGD1P8NEA7S3KM198`; the new attempt is job
`01KTSB2H1TMF7Z1T2MY40J2QM0`. It should resume from the existing checkpoint
folder, which still contains `step10500`.

Transfer-rule note: the updated, calibrated rule predicted the apparent 1.2B Cx4
center well, but wait for the resumed `1.2e-3` run before calling the rung
complete. The pre-run estimate was about `3.3e-4` from direct size transfer and
about `3.9e-4` from the 1.2B Cx1 fit times the 810M Cx4/Cx1 ratio. The observed
local fits land around `3.6e-4` to `3.7e-4`, with `3e-4` and `6e-4` close enough
that we still need the resumed `1.2e-3` run for a real right-side upturn. This is
evidence that the useful procedure is to calibrate the model-size shift with
real larger-model Cx1 data, then transfer across Cx using same-model or
nearby-model Cx-ratio behavior, rather than relying on the original naive
size-transfer prior.

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

The Cx1 optimum region still came from earlier pre-current-family jobs, while
the clean `gpu2-ep1mb16` Cx1 points only covered the hot side. To clean up the
LR-rule plot, two current-family basin reruns finished:

| LR | State | avg100M | avg250M | avg500M |
| ---: | --- | ---: | ---: | ---: |
| `1.5e-3` | finished | 2.7769 | 2.7794 | 2.7831 |
| `2e-3` | finished | 2.7739 | 2.7765 | 2.7809 |

These current-family reruns are worse than the older same-LR Cx1 basin points,
but preserve the same preference ordering (`2e-3` slightly better than
`1.5e-3`) and give a cleaner reference for run-family effects.

To complete a fully comparable Cx1 current-family curve, additional
`gpu2-ep1mb16` cold/mid probes finished:

| LR | State | avg100M | avg250M | avg500M |
| ---: | --- | ---: | ---: | ---: |
| `8e-4` | finished | 2.7868 | 2.7888 | 2.7915 |
| `1e-3` | finished | 2.7830 | 2.7852 | 2.7881 |
| `1.2e-3` | finished | 2.7823 | 2.7843 | 2.7876 |

Within the current `gpu2-ep1mb16` family, the best observed Cx1 point is now
`1.2e-3`, but it is only barely better than `1e-3`. The full current-family
curve is still worse than the old Cx1 family at comparable LRs, so keep treating
Cx1 as a run-family discrepancy until the EP=8 sanity probe finishes.

The EP=8 sanity probe finished with the fast legal 8-GPU microbatch setting:

| LR | Family | State | avg100M | avg250M | avg500M |
| ---: | --- | --- | ---: | ---: | ---: |
| `1e-3` | old/original | finished | 2.7635 | 2.7657 | 2.7687 |
| `1e-3` | `gpu2-ep1mb16` | finished | 2.7830 | 2.7852 | 2.7881 |
| `1e-3` | `gpu8-ep8mb4` sanity | finished | 2.7649 | 2.7671 | 2.7702 |
| `1e-3` | `gpu8-ep8mb4` dropless sanity | finished | 2.7646 | 2.7668 | 2.7699 |

This strongly suggests the Cx1 family discrepancy is tied to EP/settings rather
than simply microbatch size: the EP=8 sanity run lands very close to the old
same-LR Cx1 result and much better than the EP=1 current-family rerun. Keep the
sanity run out of LR-rule fits, but use it to interpret the run-family gap.

Follow-up sanity check: Tianhua suggested running the same Cx1 EP=8 setting with
`USE_ROWWISE_A2A=False`. This uses the slower dropless EP path. If it moves back
toward the current EP=1 result, then the better EP=8/original Cx1 losses may be
coming from early token dropping rather than microbatch size or another
throughput-setting difference.

The first dropless launch failed before training because the W&B group name
exceeded the 128-character limit. It was relaunched as
`olmoe3-tiny-cx1-ep8drop-lr1e-3` (`01KTB86J84BNDZJYXWMPVC9FVG`) with the same
8-GPU, EP=8, microbatch=4, `--no-use-rowwise-a2a` settings. The replacement
finished cleanly with skipped steps 0 and token drop rate 0.0.

The dropless result lands essentially on top of the rowwise EP=8 sanity result,
not on the EP=1 current-family rerun. That argues against the Cx1 family gap
being explained by early token dropping from rowwise EP alone. The discrepancy is
still tied to the broader EP=8 / 8-GPU settings family, but this specific
dropless-vs-rowwise check does not support token dropping as the cause. Keep both
sanity runs out of LR-rule fits and canonical plots.

## 2026-06-04 Completed Cx2/Cx4 Snapshot

Final completed-run summaries use final-token-window averages and ignore canceled
or failed partial predecessors.

The early Cx4 jobs without the current `gpu4-ep1mb16` settings did not finish
successfully: `1e-3`, `1.5e-3`, and `2.5e-3` finalized with exit code 1, and
`3.5e-3` was stopped while queued. The completed Cx4 table below is from the
clean current-family reruns.

Cx2 completed `avg250M`:

| LR | State | avg250M |
| ---: | --- | ---: |
| `5e-4` | finished | 2.6644 |
| `7e-4` | finished | 2.6569 |
| `6e-4` `gpu2-ep1mb16` `r2` | finished | 2.6724 |
| `8e-4` `gpu2-ep1mb16` `r2` | finished | 2.6674 |
| `1e-3` | finished | 2.6647 |
| `1.5e-3` | finished | 2.6663 |
| `2.5e-3` | finished | 2.6775 |
| `3.5e-3` | finished | 2.6897 |

Cx2 is bracketed within each visible family, but the cross-family comparison is
not clean. The best completed historical point is still the old `7e-4`, with
`5e-4` and `1e-3` close but worse, and the high side degrading monotonically
after `1e-3`. The new current-family low/mid probes finished successfully on
2026-06-04; within that family, `8e-4` beat `6e-4`, but both were worse than the
old `5e-4`/`7e-4` points. Treat Cx2 as a settings-family discrepancy rather
than a single merged U-curve.

The Cx2 curve is visually odd because the old low-side `5e-4`/`7e-4` points do
not align cleanly with the current-family `gpu2-ep1mb16` high-side trend. To
test whether this is a real Cx2 optimum shift or a family/noise artifact, we
queued and completed two current-family low/mid probes:

- Cx2 `6e-4`, `gpu2-ep1mb16`, `r2`: `01KT9RWMECT8AZ63RQH748STYB`
- Cx2 `8e-4`, `gpu2-ep1mb16`, `r2`: `01KT9S05X2WW2BPJVVJXGRYQSV`

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

## 810M LR Transfer Plan

Use completed canonical 275M full runs to estimate an LR rule, then transfer that
rule to larger active parameter counts.

Per-rung fitting rule:

- Fit loss vs `log10(lr)` with a local 3-point or 5-point quadratic around the
  visible basin.
- Use final `avg250M` training CE as the primary metric, with `avg100M` and
  `avg500M` as robustness checks.
- Do not accept a fitted optimum outside the completed bracket; in that case,
  extend the rung rather than fitting through an edge.

Current completed canonical Cx1-Cx4 estimates:

- Cx1: local fit optimum about `2.0e-3` to `2.1e-3`.
- Cx2: local fit optimum about `1.1e-3`.
- Cx4: local fit optimum about `1.5e-3`.

Fitting `log10(lr*) = a + b log10(Cx)` over the completed Cx1-Cx4 estimates
gives a noisy but useful 275M Cx1 center around `1.7e-3` to `1.8e-3`. For model
size transfer, use:

```text
lr_810m(Cx) = lr_275m(Cx) * (N_810m / N_275m)^alpha
```

with working exponent `alpha = -0.25`. This gives an 810M Cx1 center around
`1.2e-3` to `1.4e-3`, depending on whether `N` uses active non-embedding params
or active params including embeddings. We round to a nearby launchable coarse
center rather than overfitting that distinction; the first exploratory 810M Cx1
pilot is `1.6e-3`.

After the real 810M Cx1 sweep is complete, use the observed 810M Cx1 optimum to
calibrate the 275M-to-810M transfer before launching 810M Cx4. Because the 275M
Cx2 optimum is somewhat out of step with Cx1/Cx4, compute the transferred Cx4
prediction both with and without Cx2 in the 275M LR-rule fit, then choose a
single corrected Cx4 center from those estimates and the observed 810M Cx1
offset. The 810M Cx4 sweep should be exactly four coarse, factor-spaced LRs
centered around that corrected prediction, using the validated `gpu8-ep1mb4`
setting.

Launched the first 810M Cx1 sweep on 2026-06-05 with the validated 4-GPU,
EP=1, microbatch=4 settings. The active full-run LR set is `6e-4`, `1.2e-3`,
`2.4e-3`, and `6e-3`. Treat `6e-3` as a hot-side sentinel: useful for
bracketing, but do not stop it early unless it is operationally broken or
obviously unstable. The earlier `1.6e-3` pilot was stopped intentionally around
step 7669 / 2.01B tokens to free 4 GPUs for the sentinel, so ignore it for final
LR selection and canonical plots. These runs use final-only permanent
checkpoints plus latest ephemeral resume checkpoints.

Initial 810M Cx1 results finished on 2026-06-05:

| LR | State | avg100M | avg250M | avg500M |
| ---: | --- | ---: | ---: | ---: |
| `6e-4` | finished | 2.4096 | 2.4103 | 2.4131 |
| `1.2e-3` | finished | 2.4147 | 2.4156 | 2.4188 |
| `2.4e-3` | finished | 2.4456 | 2.4465 | 2.4499 |
| `6e-3` | finished | 2.5418 | 2.5424 | 2.5457 |
| `5e-5` | finished | 2.5678 | 2.5680 | 2.5705 |

The hot side is bracketed, and `6e-3` served its sentinel purpose. The best
observed point is the low-edge `6e-4`, with `1.2e-3` close behind, so Cx1 is not
yet cold-side bracketed. Per the ladder policy, the next 810M Cx1 action is a
small cold-side extension, not Cx4 yet.

Launched cold-side extensions at `3e-4` and `1.5e-4` with the same validated
4-GPU, EP=1, microbatch=4 setting. These should determine whether `6e-4` is the
basin floor or still too warm.

Also launched a farther cold sentinel at `5e-5`. The first long-name
`gpu4-ep1mb4` attempt failed before training because the generated W&B group
exceeded the 128-character limit, so ignore that attempt. It was relaunched with
a shorter name at `gpu8-ep1mb4`, keeping the same global token batch for U-plot
comparability while using more GPUs to reduce wall-clock. The relaunched
sentinel finished and is clearly worse than `6e-4`, so the far-cold side is now
bracketed.

Final 810M Cx1 results after the cold-side extensions:

| LR | State | avg100M | avg250M | avg500M |
| ---: | --- | ---: | ---: | ---: |
| `5e-5` | finished | 2.5678 | 2.5680 | 2.5705 |
| `1.5e-4` | finished | 2.4482 | 2.4486 | 2.4512 |
| `3e-4` | finished | 2.4201 | 2.4207 | 2.4234 |
| `6e-4` | finished | 2.4096 | 2.4103 | 2.4131 |
| `1.2e-3` | finished | 2.4147 | 2.4156 | 2.4188 |
| `2.4e-3` | finished | 2.4456 | 2.4465 | 2.4499 |
| `6e-3` | finished | 2.5418 | 2.5424 | 2.5457 |

The best observed 810M Cx1 LR is `6e-4`. A local 5-point quadratic over
`1.5e-4`, `3e-4`, `6e-4`, `1.2e-3`, and `2.4e-3` gives fitted optimum
`6.21e-4` on avg250M, so Cx1 is now cleanly bracketed.

For the 810M Cx4 transfer:

- 275M local fitted optima on avg250M: Cx1 `2.13e-3`, Cx2 `1.12e-3`, Cx4
  `1.46e-3`, Cx8 `1.36e-3`, Cx16 `1.10e-3`.
- Fitting the 275M LR rule with Cx2 included predicts 810M Cx4 `4.95e-4` after
  calibrating to the observed 810M Cx1 optimum.
- Fitting the 275M LR rule without Cx2 predicts 810M Cx4 `4.51e-4`.

These agree well enough to center the 810M Cx4 sweep around `4e-4` to `5e-4`.
Launched exactly four Cx4 LRs with the validated `gpu8-ep1mb4` setting:
`2e-4`, `4e-4`, `8e-4`, and `1.6e-3`.

2026-06-07 final completion update:

| LR | State | avg100M | avg250M | avg500M |
| ---: | --- | ---: | ---: | ---: |
| `2e-4` | finished | 2.2516 | 2.2578 | 2.2568 |
| `4e-4` | finished | 2.2364 | 2.2427 | 2.2417 |
| `8e-4` | finished | 2.2387 | 2.2451 | 2.2442 |
| `1.6e-3` | finished | 2.2622 | 2.2687 | 2.2678 |

The completed Cx4 curve favors `4e-4` by observed avg250M/avg500M. A quadratic
fit over all four avg250M points gives `lr* = 4.99e-4`; a local three-point fit
over `2e-4`, `4e-4`, and `8e-4` gives `lr* = 5.14e-4`. Treat the 810M Cx4
optimum as about `5e-4`.

This means the observed 275M -> 810M LR transfer is much cooler than the initial
`alpha=-0.25` working prior. Using the 810M Cx1 and Cx4 optima, the implied
active-size exponent is roughly `-1.0` with active params including embeddings,
or roughly `-0.9` with active non-embedding params. Applying that to 810M -> 1.2B
puts the 1.2B Cx1 center around `4e-4`.

Final checkpoint eval backfills were launched for completed Cx4 runs because
these training jobs did not run evals in-loop. The `2e-4`, `4e-4`, `8e-4`, and
`1.6e-3` Cx4 eval backfills have finished and their 180 eval summary metrics
have been copied onto the corresponding source W&B training runs.

For transferred larger-model sweeps, factor-of-two spacing around the transferred
center is reasonable. For rungs where the best point remains on the edge or no
transfer prior is reliable, include a much wider sentinel instead of repeatedly
walking outward by small multiples.

Based on the updated transfer rule, the first 1.2B Cx1 sweep was launched on
2026-06-07 with `gpu8-ep1mb2`, `global_batch_size_seq=32`, EP=1, and in-loop
fast evals every 2000 steps:

- `1e-4`: `01KTG4J00SXZPREAA3A1E463P9`
- `2e-4`: `01KTG4JAQ2Z82YSGPAWRBW353H`
- `4e-4`: `01KTG4JQ19A27ZC6H0FDD9661S`
- `8e-4`: `01KTG4K2MZHZNCYVG5K9RPV4SW`

All four started cleanly, reached a few hundred steps, and reported
`optim/step skipped = 0`.

While the 1.2B Cx1 sweep was near completion, the next fixed 810M rung was
queued because its LR list does not depend on the pending 1.2B result:

- 810M Cx8 `1e-4`: `01KTHQWMSQ0A4P6RCNKPS7YPYD`
- 810M Cx8 `2e-4`: `01KTHQX04RMEK7C7V6DZRZVXM6`
- 810M Cx8 `4e-4`: `01KTHQXB575GS84FBP4SNZ1GAA`
- 810M Cx8 `8e-4`: `01KTHQXNN4MFDBAP490ACJTJ07`

Settings: `gpu8-ep1mb4`, `global_batch_size_seq=96` / 786,432 tokens,
`--ladder-evals --eval-task-set=fast --eval-interval=2000`.

2026-06-07 1.2B Cx1 completion update:

| LR | State | avg100M | avg250M | avg500M |
| ---: | --- | ---: | ---: | ---: |
| `1e-4` | finished | 2.3549 | 2.3550 | 2.3580 |
| `2e-4` | finished | 2.3244 | 2.3246 | 2.3276 |
| `4e-4` | finished | 2.3106 | 2.3108 | 2.3139 |
| `8e-4` | finished | 2.3145 | 2.3148 | 2.3181 |

The completed Cx1 curve is bracketed and favors `4e-4` by observed avg250M.
A quadratic fit over all four avg250M points gives `lr* = 4.86e-4`; a local
three-point fit around the basin gives `lr* = 4.84e-4` to `5.03e-4`. Treat the
1.2B Cx1 optimum as about `5e-4`.

For 1.2B Cx4, the 810M Cx4/Cx1 relationship and the updated larger-model
transfer put the center around `3e-4` to `4e-4`, cooler than simply copying the
1.2B Cx1 optimum. Launched exactly four Cx4 LRs:

- `1.5e-4`: `01KTHW5XZXCNW9VV7FAMCS1C8F`
- `3e-4`: `01KTHW68C59T1XE9WNFW3EP3G1`
- `6e-4`: `01KTHW6KH3XFR790J6J4G8ZAJ6`
- `1.2e-3`: `01KTHW6ZSXGD1P8NEA7S3KM198`

Settings: `gpu8-ep1mb2`, `global_batch_size_seq=64` / 524,288 tokens,
`--ladder-evals --eval-task-set=fast --eval-interval=2000`.

Also queued the fixed 810M Cx2 completeness sweep:

- `1.5e-4`: `01KTHW7HB59AMPSZBP8FJHS5QG`
- `3e-4`: `01KTHW7WY6Z2NFAP8FNT1HP3XN`
- `6e-4`: `01KTHW88Q43J8M8CRCDN9VZDHV`
- `1.2e-3`: `01KTHW8MCKRJH3PW0W58KRVXA4`

Settings: `gpu8-ep1mb4`, `global_batch_size_seq=64` / 524,288 tokens,
`--ladder-evals --eval-task-set=fast --eval-interval=2000`.

## Validation Eval Follow-up

TODO: after the 2026-06-06 backfills finish, discuss and decide how validation
losses should enter ladder decisions in addition to the train losses we
currently use for U-plots.

Scaling-ladders runs attach `with_recommended_evals(..., task_set="fast")` to
normal training, which logs LM validation components such as
`c4_en-validation`, Dolma slices, Pile, and Wikitext, plus downstream fast tasks
including ARC/MMLU BPB and MC metrics. Our early MoE ladder runs were selected
from train loss only; once we have backfilled and smoke-tested these evals, we
should decide how to use validation losses alongside train losses in LR
selection. In particular, discuss whether the primary U-plot target should
remain final-window train loss, switch to C4 validation loss/PPL, use a small
set of validation metrics as tie-breakers, or require agreement between train
and validation before carrying a transferred LR rule to 810M/1.2B.

The first in-loop eval training proof used `--eval-interval=100`; it proved the
hook but was too slow, so it was stopped intentionally. For future training runs
with in-loop evals, use `--ladder-evals --eval-task-set=fast
--eval-interval=2000` plus the final checkpoint eval. For completed runs, prefer
eval-only `--eval-checkpoints` backfills tagged `eval-backfill`; exclude those
eval-only runs from train-loss U-plots.

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
| `1.6e-3` | finished | 40971 | 32.221 | 2.4884 | 2.4864 | 2.4859 |
| `3.2e-3` | finished | 40971 | 32.221 | 2.5006 | 2.4987 | 2.4982 |
| `6.4e-3` | finished | 40971 | 32.221 | 2.5367 | 2.5347 | 2.5341 |
| `1.6e-2` | finished | 40971 | 32.221 | 2.6308 | 2.6285 | 2.6278 |

The best observed Cx8 LR is `1.6e-3`. The completed `3.2e-3` extension is
worse, and `6.4e-3` is much worse, so Cx8 is now well bracketed on the right
side. A local quadratic fit of loss vs log10(LR) gives a fitted optimum around
`1.35e-3` (`1.35e-3` with the 5-point fit and `1.36e-3` with the 3-point fit).
Treat this as a broad optimum region rather than a precise number because the
basin is shallow from `8e-4` through `1.6e-3`.

Because the completed Cx8 curve had previously been monotonically improving at
the high edge, we also launched a farther right-side extension at `6.4e-3` and a
true order-of-magnitude sentinel at `1.6e-2`; keep those running for final
right-tail numbers. Both are now complete and confirm the right side is safely
hot.

- Cx8 `1.6e-3`, `gpu4-ep1mb8`, `r2`: `01KT9D6W9F4RGA5RSA8XSSMEP3`
- Cx8 `3.2e-3`, `gpu4-ep1mb8`, `r2`: `01KT9Q661N0YHYHC9A9T9AGV1J`;
  stopped intentionally after Cx8 `1.6e-3` was already clearly worse than the
  completed `8e-4` best. Ignore for full-run analysis.
- Cx8 `3.2e-3`, `gpu4-ep1mb8`, `r3`: `01KTAA55V6QXN45QZFBHTY6B65`;
  finished successfully on 2026-06-05, avg250M 2.4987, avg500M 2.4982.
- Cx8 `6.4e-3`, `gpu4-ep1mb8`, `r2`: `01KT9Q6HX5X6KFW5RD1VSC9BV4`;
  stopped intentionally after lower high-side probes were already clearly worse.
  Ignore for full-run analysis.
- Cx8 `6.4e-3`, `gpu4-ep1mb8`, `r3`: `01KTAC584AMG9645NEKF479R15`;
  finished successfully on 2026-06-05, avg250M 2.5347, avg500M 2.5341.
- Cx8 `1.6e-2`, `gpu4-ep1mb8`, `sentinel`: `01KTACFJ4D4FQG33ZPT4R306WT`;
  finished successfully on 2026-06-05, avg250M 2.6285, avg500M 2.6278.

Cx16 completed full-run results from the canonical `r2` grid:

| LR | State | Step | TokensB | avg100M | avg250M | avg500M |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2e-4` | finished | 61457 | 64.442 | 2.4665 | 2.4759 | 2.4744 |
| `4e-4` | finished | 61457 | 64.442 | 2.4381 | 2.4474 | 2.4461 |
| `6e-4` | finished | 61457 | 64.442 | 2.4274 | 2.4367 | 2.4354 |
| `1.2e-3` | finished | 61457 | 64.442 | 2.4208 | 2.4301 | 2.4288 |
| `2.4e-3` | finished | 61457 | 64.442 | 2.4321 | 2.4413 | 2.4400 |
| `6e-3` | finished | 61457 | 64.442 | 2.4781 | 2.4876 | 2.4862 |

The best observed completed Cx16 LR is `1.2e-3`. The completed `2.4e-3` and
`6e-3` right-side points are worse, so Cx16 is now bracketed. A local quadratic
fit of loss vs log10(LR) gives a fitted optimum around `1.07e-3` with the
5-point fit and `1.10e-3` with the 3-point fit. Treat `1.1e-3` to `1.2e-3` as
the current Cx16 optimum region.

- Cx16 `1.2e-3`, `gpu8-ep1mb16`, `r2`: `01KT9H6XQJ2GEMKPKHKPCED5B1`
- Cx16 `2.4e-3`, `gpu8-ep1mb16`, `r2`: `01KT9Q6X0B6PG3G6ZSBZGTPSVQ`;
  stopped intentionally after Cx16 `1.2e-3` was already clearly worse than the
  completed `6e-4` best. Ignore for full-run analysis.
- Cx16 `2.4e-3`, `gpu8-ep1mb16`, `r3`: `01KTAC763FP2W34ZX6N4CT21QD`;
  finished successfully on 2026-06-05, avg250M 2.4413, avg500M 2.4400.
- Cx16 `4.8e-3`, `gpu8-ep1mb16`, `r2`: `01KT9Q774FWC6NZDSGTD0Y2W7K`;
  stopped intentionally while queued after lower high-side probes were already
  clearly worse. Ignore for full-run analysis.
- Cx16 `6e-3`, `gpu8-ep1mb16`, `sentinel`: `01KTACHG3Z4Y1G9HW9ESYZK58Q`;
  finished successfully on 2026-06-05, avg250M 2.4876, avg500M 2.4862.

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

For the first 1.2B Cx1 sweep, match the dense-ladder Cx1 batch size and use the
currently idle cluster capacity for wall-clock speed:

- `--model-size=1p2b`
- `--chinchilla-multiple=1`
- `--global-batch-size-seq=32` / 262,144 tokens
- `--gpus-per-node=8`
- `--ep-dim=1`
- `--micro-batch-size=2`
- `--ladder-evals --eval-task-set=fast --eval-interval=2000`

Prepared launchers:

```bash
src/scripts/train/jacobm_olmoe_ladder/launch_moe_a0_1p2b_cx1_sweep.sh
src/scripts/train/jacobm_olmoe_ladder/reproduce_moe_a0_1p2b_cx1_sweep.sh
```

These require `LR_SPECS="lr:tag ..."` at launch time. Do not fill the 1.2B LR
list until the completed 810M Cx4 fit has updated the transfer rule.

## Expert Granularity Experiment

The first post-baseline experiment is now underway. It tests the fixed routed
capacity triangle:

- `coarse_24e_top2`: 24 experts, top-2, `moe_hidden_size=2*d_model`
- `baseline_48e_top4`: existing control
- `fine_96e_top8`: 96 experts, top-8, `moe_hidden_size=d_model/2`

The active routed hidden units and routed total hidden units are fixed across
the triangle. The fine variant has slightly more total parameters because router
output dimension grows with expert count.

Smoke results:

- `coarse_24e_top2` passed at 275M Cx1, `gpu1-ep1mb16`.
- `fine_96e_top8` OOMed at `gpu1-ep1mb16`, then passed startup at
  `gpu1-ep1mb8` with skipped steps 0 and finite loss at the first check.

Cx1 LR transfer probes launched on 2026-06-11:

- `coarse_24e_top2`: `1e-3`, `2e-3`, `4e-3`, all `gpu1-ep1mb16`.
- `fine_96e_top8`: `1e-3`, `2e-3`, `4e-3`, all `gpu1-ep1mb8`.

Do not queue the full 275M expert-granularity ladder until these Cx1 probes
finish and we have reviewed:

- whether Cx1 brackets cleanly for each variant;
- fitted `m_variant = lr*_variant / lr*_baseline`;
- whether the mb8 fine-variant fallback is acceptable for Cx1 comparisons;
- whether the experiment-specific tracking and plots are clean enough to use
  for the rest of the ladder.

Overnight exception: the 275M Cx4 baseline-centered probes may be queued before
the Cx1 probes finish, because they are short, use the already known baseline
Cx4 optimum region, and can be extended by one targeted follow-up per variant if
the architecture LR multiplier differs from 1.0. The queued Cx4 grid is
`8e-4`, `1.6e-3`, `3.2e-3` for each variant.

Status after the 2026-06-11 long-cadence check:

- `coarse_24e_top2` Cx1 finished cleanly at all three LRs. Avg250M losses:
  - `1e-3`: 2.7873
  - `2e-3`: 2.7814
  - `4e-3`: 2.7904
- The coarse Cx1 curve is bracketed; the observed best is `2e-3`, and a
  3-point quadratic fit vs `log10(lr)` gives `lr* ~= 1.86e-3`.
- No coarse Cx1 follow-up is needed before reviewing Cx4.
- `fine_96e_top8` Cx1 `1e-3` finished cleanly with avg250M 2.7683. This is
  notably better than the baseline Cx1 `1e-3` avg250M 2.7852, but wait for the
  fine `2e-3` and `4e-3` completions before interpreting the fine LR curve.
- Cx4 baseline-centered expert-granularity probes are now running except the
  fine `3.2e-3` job, which remains queued/created.

Status after the next 2026-06-11 long-cadence check:

- `fine_96e_top8` Cx1 finished cleanly at all three LRs. Avg250M losses:
  - `1e-3`: 2.7683
  - `2e-3`: 2.7641
  - `4e-3`: 2.7673
- The fine Cx1 curve is bracketed; the observed best is `2e-3`, and a 3-point
  quadratic fit vs `log10(lr)` gives `lr* ~= 2.10e-3`.
- Both expert-granularity Cx1 variants are now bracketed. No Cx1 follow-up is
  needed before reviewing the broader 275M ladder plan.
- `coarse_24e_top2` Cx4 finished cleanly at all three LRs. Avg250M losses:
  - `8e-4`: 2.5796
  - `1.6e-3`: 2.5713
  - `3.2e-3`: 2.5805
- The coarse Cx4 curve is bracketed; the observed best is `1.6e-3`, and a
  3-point quadratic fit gives `lr* ~= 1.57e-3`.
- `fine_96e_top8` Cx4 has finished `8e-4` and `1.6e-3` with avg250M 2.5582 and
  2.5523 respectively. The `3.2e-3` point is running; wait for it before
  deciding whether fine Cx4 needs a follow-up.

Midpoint baseline follow-ups from 2026-06-10:

- `mid_480m` Cx4 cold sentinel `1e-4` finished cleanly with avg250M 2.4689,
  much worse than the existing Cx4 best `8e-4` avg250M 2.3788. This confirms
  the cold side but should not change the Cx4 LR center.
- `mid_480m` Cx8 hot sentinel `3.2e-3` finished cleanly with avg250M 2.3486,
  worse than the existing Cx8 best `8e-4` avg250M 2.3076. This brackets Cx8 on
  the hot side; no additional midpoint Cx8 hot extension is needed.
