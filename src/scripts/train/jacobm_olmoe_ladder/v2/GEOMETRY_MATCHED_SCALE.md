# Geometry-matched hybrid scale design

Dense-ladder source: `scaling-ladders` commit `aaeeca3`,
`ladders/mainline/workloads/arch.py`. The canonical builder is
[`models/geometry_matched_scale.py`](models/geometry_matched_scale.py), selected
from the scale trainer with `geometry_matched_gdn_ev2` for RoPE or
`geometry_matched_gdn_ev2_nope` for NoPE. The prepared
`geometry_matched_gdn_ev2_nope_gated` profile adds only the dense ladder's
elementwise full-precision attention gate to the NoPE model.

The NoPE configs completed checkpoint-free capacity and scaling smokes on
Holmes B300s, and the 12-cell production wave was submitted on 2026-07-18.
The equivalent larger gated-attention wave was submitted on 2026-07-18 after
the completed 275M sweep showed that the wide-integration LR transfer remained
well behaved and gating improved the ungated NoPE control at every Cx.

## Scaling rule

Each rung adopts the corresponding dense ladder geometry while keeping the
established MoE recipe:

- 480M maps to dense 450M geometry, 810M maps to dense 810M geometry, and
  1.2B maps to dense 1.4B geometry.
- Full attention is every fifth layer (layers 4, 9, 14, and 19 when present);
  all other layers use GatedDeltaNet with `expand_v=2`.
- Layer 0 retains the dense-first FFN. All other layers retain 256 routed
  experts, top-k 8, and one shared expert.
- Expert widths are 8-aligned and chosen to match the corresponding trained
  `integration_wide_gdn_ev1` model's total active parameters as closely as
  possible.
- As in the primary 275M `geometry_only` experiment, full attention retains
  the current model's GQA ratio, RoPE, and no output gate. Initialization
  remains `init_std=0.01`. Exact dense KV geometry/gating, NoPE, and the
  `0.02` initialization are still separate interventions.

## Exact configurations

| Named size | Dense geometry rung | `d_model` | Layers | Q / KV heads | GDN / full layers | Expert hidden | Dense-first hidden |
|---|---|---:|---:|---:|---:|---:|---:|
| 275M | 275M | 640 | 10 | 8 / 4 | 8 / 2 | 664 | 5,976 |
| 480M | 450M | 768 | 15 | 8 / 4 | 12 / 3 | 840 | 7,560 |
| 810M | 810M | 1,024 | 15 | 16 / 8 | 12 / 3 | 1,032 | 9,288 |
| 1.2B | 1.4B | 1,280 | 20 | 16 / 8 | 16 / 4 | 952 | 8,568 |

The 810M query-head count changes from 12 to 16 as part of matching dense
geometry; its 2:1 GQA ratio is unchanged. The 480M model retains 8 Q / 4 KV
instead of adopting the dense 450M rung's 8 Q / 8 KV, exactly paralleling the
primary 275M geometry-only control.

## Parameter matching

| Named size | Original baseline active | Wide integration active | First hybrid active | Geometry active | Geometry vs baseline | vs wide | vs first hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|
| 275M | 278,450,688 | 280,207,872 | 288,194,512 | 290,782,080 | +4.429% | +3.774% | +0.898% |
| 480M | 483,153,920 | 486,348,800 | 501,228,784 | 501,137,856 | +3.722% | +3.041% | -0.018% |
| 810M | 818,511,360 | 823,569,920 | 859,400,792 | 858,237,056 | +4.853% | +4.209% | -0.135% |
| 1.2B | 1,218,302,464 | 1,225,011,712 | 1,288,662,592 | 1,289,441,280 | +5.839% | +5.260% | +0.060% |

| Named size | Original baseline active non-embedding | Wide integration | First hybrid | Geometry | Geometry vs baseline | vs wide | vs first hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|
| 275M | 201,380,352 | 203,137,536 | 211,124,176 | 226,556,800 | +12.502% | +11.529% | +7.310% |
| 480M | 380,393,472 | 383,588,352 | 398,468,336 | 424,067,520 | +11.481% | +10.553% | +6.424% |
| 810M | 690,060,800 | 695,119,360 | 730,950,232 | 755,476,608 | +9.480% | +8.683% | +3.355% |
| 1.2B | 1,064,161,792 | 1,070,871,040 | 1,134,521,920 | 1,160,990,720 | +9.100% | +8.416% | +2.333% |

The geometry models store 3,136,314,240, 7,220,707,776, 11,865,532,544, and
18,515,005,440 parameters from 275M through 1.2B, respectively. Use these
total counts—not active counts—when planning optimizer memory and EP.

The token budgets will increase by the non-embedding deltas because the
pretraining trainer derives Chinchilla tokens from active non-embedding
parameters. This is intentional and matches the established 275M experiment:
quality should be compared at the rung's derived Cx budget and with explicit
active-parameter/FLOP accounting.

Run the local structural and exact-count audit without creating external work:

```bash
PYTHONPATH=src uv run python \
  src/scripts/train/jacobm_olmoe_ladder/v2/models/geometry_matched_scale.py
```

## Larger NoPE capacity and scaling smokes

The independent launcher and manifest are:

- [`launchers/pretraining/launch_geometry_matched_scale_nope_smokes.py`](launchers/pretraining/launch_geometry_matched_scale_nope_smokes.py)
- [`launchers/pretraining/manifests/geometry_matched_scale_nope_smokes.yaml`](launchers/pretraining/manifests/geometry_matched_scale_nope_smokes.yaml)

Every valid row below completed a compiled dry run plus 12 optimizer steps on
Holmes B300s with exact production model/optimizer construction. Checkpointing,
in-loop evaluation, and on-finish evaluation were all disabled. The runs were
urgent unallocated work (`minRuntime: 0m`, non-preemptible, auto-resuming) in
`ai2/OLMo-3-moe-experiments`.

NoPE changes no parameter count. Strict construction checks confirmed:

| Size | Active | Active non-embedding | Total stored |
|---|---:|---:|---:|
| 480M | 501,137,856 | 424,067,520 | 7,220,707,776 |
| 810M | 858,237,056 | 755,476,608 | 11,865,532,544 |
| 1.2B | 1,289,441,280 | 1,160,990,720 | 18,515,005,440 |

The gated profile adds one `d_model -> n_heads * head_dim` projection to every
full-attention layer and changes no other field:

| Size | Gated active | Gated active non-embedding | Gated total stored |
|---|---:|---:|---:|
| 480M | 503,497,152 | 426,426,816 | 7,223,067,072 |
| 810M | 864,528,512 | 761,768,064 | 11,871,824,000 |
| 1.2B | 1,299,927,040 | 1,171,476,480 | 18,525,491,200 |

The performance statistics are medians over the final five reported steps, not
compile-polluted whole-run averages:

| Size | Cx | GPUs | EP | Rank MB | Accum | TFLOPs/GPU | TPS/GPU | Active / reserved memory | W&B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 480M | 1 | 4 | 1 | 8 | 1 | 379.8 | 137,426 | 184.7 / 185.3 GiB | [wllr6m1g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wllr6m1g) |
| 480M | 8 | 4 | 1 | 12 | 2 | 460.8 | 166,744 | 246.8 / 247.4 GiB | [xch3s5bw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xch3s5bw) |
| 480M | 8 | 8 | 1 | 12 | 1 | 416.0 | 150,519 | 236.7 / 237.4 GiB | [s8ubt7sh](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s8ubt7sh) |
| 810M | 1 | 8 | 1 | 4 | 1 | 312.6 | 63,187 | 167.4 / 167.6 GiB | [0xfsc1vs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0xfsc1vs) |
| 810M | 8 | 8 | 1 | 6 | 2 | 447.9 | 90,530 | 209.6 / 209.8 GiB | [kfm4ynjr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfm4ynjr) |
| 810M | 8 | 16 | 1 | 6 | 1 | 380.3 | 76,867 | 201.3 / 201.5 GiB | [o54vnqvg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o54vnqvg) |
| 1.2B | 1 | 8 | 8 | 4 | 1 | 409.5 | 54,441 | 168.5 / 171.8 GiB | [g1d4fcd7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g1d4fcd7) |
| 1.2B | 8 | 8 | 8 | 6 | 2 | 440.5 | 58,555 | 231.5 / 236.8 GiB | [c69nxyn3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/c69nxyn3) |
| 1.2B | 8 | 16 | 8 | 6 | 1 | 410.8 | 54,609 | 218.7 / 223.9 GiB | [kfm18iir](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfm18iir) |
| 1.2B | 8 | 32 | 8 | 3 | 1 | 315.6 | 41,957 | 117.7 / 120.3 GiB | [owvnz62c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/owvnz62c) |

The initial six one-node attempts used Gantry's project-install path and
invalidated the image's CUDA/TransformerEngine ABI; they are infrastructure
failures and provide no capacity result. The corrected launcher preserves the
tested image environment. One corrected 810M Cx1 worker then segfaulted after
optimizer construction; the identical retry above passed, while the larger
810M MB6 run also passed, so that attempt is recorded as a transient
worker/hardware failure rather than a model-capacity failure.

### Rough full-run ETAs

The token budget is exactly `20 * active_non_embedding_params * Cx`. Asterisks
mark combinations measured directly. Other cells preserve the canonical
32/48/64/96-sequence batches and extrapolate TPS from the measured
size-specific microbatch curve plus measured multi-node scaling. They are
planning estimates, not throughput guarantees. `MB×A` means rank microbatch
sequences times gradient-accumulation steps.

| Size | Cx | Target tokens | 4 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |
|---|---:|---:|---:|---:|---:|---:|
| 480M | 1 | 8.481B | 4.3h* (`8×1`) | 3.6h (`4×1`) | — | — |
| 480M | 2 | 16.963B | 7.1h (`12×1`) | 5.6h (`6×1`) | — | — |
| 480M | 4 | 33.925B | 17.1h (`8×2`) | 9.5h (`8×1`) | — | — |
| 480M | 8 | 67.851B | 28.3h* (`12×2`) | 15.7h* (`12×1`) | — | — |
| 810M | 1 | 15.110B | — | 8.3h* (`4×1`) | 9.3h (`2×1`) | — |
| 810M | 2 | 30.219B | — | 11.6h (`6×1`) | 12.7h (`3×1`) | — |
| 810M | 4 | 60.438B | — | 33.2h (`4×2`) | 19.6h (`4×1`) | — |
| 810M | 8 | 120.876B | — | 46.4h* (`6×2`) | 27.3h* (`6×1`) | — |
| 1.2B | 1 | 23.220B | — | 14.8h* (`4×1`) | 9.6h (`2×1`) | 7.4h (`1×1`) |
| 1.2B | 2 | 46.440B | — | 27.5h (`6×1`) | 16.9h (`3×1`) | invalid at the fixed batch |
| 1.2B | 4 | 92.879B | — | 59.2h (`4×2`) | 31.8h (`4×1`) | 21.8h (`2×1`) |
| 1.2B | 8 | 185.759B | — | 110.2h* (`6×2`) | 59.1h* (`6×1`) | 38.4h* (`3×1`) |

Startup and compilation add roughly 5–10 minutes per job and are excluded from
the table. The current measurements imply that 480M Cx1/Cx2 and 810M Cx1/Cx2
gain too little wall-clock speed from doubling GPUs; the larger Cx values
scale materially. The 1.2B Cx8 choices trade approximately 4.6 days on 8 GPUs,
2.5 days on 16, or 1.6 days on 32, with diminishing GPU efficiency.

### Selected production GPU layout

On 2026-07-18, the following layout was selected for the larger NoPE wave:

| Size | Cx | GPUs | EP | Rank MB | Accum | Rough ETA | Estimated GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| 480M | 1 | 8 | 1 | 4 | 1 | 3.6h | 29 |
| 480M | 2 | 8 | 1 | 6 | 1 | 5.6h | 45 |
| 480M | 4 | 8 | 1 | 8 | 1 | 9.5h | 76 |
| 480M | 8 | 8 | 1 | 12 | 1 | 15.7h | 125 |
| 810M | 1 | 16 | 1 | 2 | 1 | 9.3h | 149 |
| 810M | 2 | 16 | 1 | 3 | 1 | 12.7h | 204 |
| 810M | 4 | 16 | 1 | 4 | 1 | 19.6h | 313 |
| 810M | 8 | 16 | 1 | 6 | 1 | 27.3h | 437 |
| 1.2B | 1 | 16 | 8 | 2 | 1 | 9.6h | 154 |
| 1.2B | 2 | 16 | 8 | 3 | 1 | 16.9h | 271 |
| 1.2B | 4 | 32 | 8 | 2 | 1 | 21.8h | 699 |
| 1.2B | 8 | 32 | 8 | 3 | 1 | 38.4h | 1,230 |

This is 12 jobs, approximately 3,731 GPU-hours, and a peak request of **192
GPUs**:

```text
480M:          4 jobs *  8 GPUs = 32
810M:          4 jobs * 16 GPUs = 64
1.2B Cx1/Cx2: 2 jobs * 16 GPUs = 32
1.2B Cx4/Cx8: 2 jobs * 32 GPUs = 64
                                      ---
                                      192
```

At an average realized capacity of 64 GPUs, the aggregate work is about 58
hours; at 80 GPUs it is about 47 hours. Individual job dependencies are absent,
so the scheduler may pack the wave in any order. These estimates exclude queue
latency and add only negligible full-run-relative compile overhead.

In the preceding ETA table, an asterisk means that exact
`(size, Cx, GPU count, rank MB, accumulation)` combination was directly
measured by a smoke. It does not mean optimal or recommended. Unstarred cells
are inferred from directly measured microbatch and multi-node scaling.

## Production launchers

The strict production launcher is
[`launchers/pretraining/launch_geometry_matched_scale_full.py`](launchers/pretraining/launch_geometry_matched_scale_full.py).
It requires all 12 cells, validates the selected GPU/EP/microbatch layout and
the transferred wide-integration LR for each cell, audits exact model counts,
refuses accidental reuse of an existing checkpoint directory, and records
every Beaker experiment ID.

- NoPE manifest:
  [`launchers/pretraining/manifests/geometry_matched_scale_nope_full.yaml`](launchers/pretraining/manifests/geometry_matched_scale_nope_full.yaml)
- Gated-attention manifest:
  [`launchers/pretraining/manifests/geometry_matched_scale_nope_gated_full.yaml`](launchers/pretraining/manifests/geometry_matched_scale_nope_gated_full.yaml)

The production jobs write rolling ephemeral checkpoints every 500 steps with
`remove=ephemeral_only`, retain the final checkpoint, and disable all
in-loop/on-finish evaluators. Register Beaker and W&B IDs before plotting. The
geometry-family plots compare against both wide integration and the first
`expand_v=1` hybrid.

The NoPE wave is urgent unallocated work on Holmes, pinned to commit
`fcf1c1b8828a3bddd0bad477a5c4055e63b0275f`. The machine-readable submission
record, including every task ID, is
[`launchers/pretraining/generated/geometry_matched_scale_full_submissions.json`](launchers/pretraining/generated/geometry_matched_scale_full_submissions.json).

| Size | Cx | LR | GPUs | Beaker work |
|---|---:|---:|---:|---|
| 480M | 1 | `1.2e-3` | 8 | [01KXT0BJGPBS2T4AR7HDNDWZ9P](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0BJGPBS2T4AR7HDNDWZ9P) |
| 480M | 2 | `9e-4` | 8 | [01KXT0BNMA5XJ30YMY13N874H9](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0BNMA5XJ30YMY13N874H9) |
| 480M | 4 | `8e-4` | 8 | [01KXT0BRP1DMHMDQ6XQFPZSCQ9](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0BRP1DMHMDQ6XQFPZSCQ9) |
| 480M | 8 | `8e-4` | 8 | [01KXT0BVRMSPAJDKVMVMP68FXR](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0BVRMSPAJDKVMVMP68FXR) |
| 810M | 1 | `6e-4` | 16 | [01KXT0BYYSB24R01PMA7DHGH8A](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0BYYSB24R01PMA7DHGH8A) |
| 810M | 2 | `5.6e-4` | 16 | [01KXT0C24FCYYH8C3RJ91K7X16](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0C24FCYYH8C3RJ91K7X16) |
| 810M | 4 | `4e-4` | 16 | [01KXT0C5HCYCCKY44GN2RCC5MC](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0C5HCYCCKY44GN2RCC5MC) |
| 810M | 8 | `4e-4` | 16 | [01KXT0C8TNW8DBCMRNVA1SKVV4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0C8TNW8DBCMRNVA1SKVV4) |
| 1.2B | 1 | `4e-4` | 16 | [01KXT0CC2W6ABQXB8F8WWNBXXE](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0CC2W6ABQXB8F8WWNBXXE) |
| 1.2B | 2 | `6e-4` | 16 | [01KXT0CF9V3ZP1NDA72DAJKG4V](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0CF9V3ZP1NDA72DAJKG4V) |
| 1.2B | 4 | `3e-4` | 32 | [01KXT0CKPC9NV36GKDXZH5SM17](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0CKPC9NV36GKDXZH5SM17) |
| 1.2B | 8 | `4e-4` | 32 | [01KXT0CQBVT4T414SAZQYT1RAS](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT0CQBVT4T414SAZQYT1RAS) |

Status at 2026-07-21 03:10 UTC: all 480M and 810M cells plus 1.2B Cx1/Cx2/Cx4
are finished, for 11/12 completed cells. Strict final-250M CE is `2.119848`
for 810M Cx8 and `2.107767` for the newly completed 1.2B Cx4. The user requeued the
[Cx8 resume](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXW463APF8FT6RV8T8XZ2D6Q)
in place. It resumed from the same checkpoint directory, reached step 17,644,
and failed for a third time on the same `Non-finite total grad norm`
assertion. The short urgent
[diagnostic continuation](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY097ZF4F18F16P71XZWJVX0)
resumed from durable `step17500` with the original LR and optimizer state, then
failed again at step 17,592. The dump shows a broad failure, not one bad tensor:
all 375 DP entries on rank 0 and 374/375 on every other rank were NaN, along
with 32--38 EP-DP entries per rank. A single-step skip is therefore not a
credible repair for the original trajectory.
Their W&B IDs are registered in [`plot_pretraining_wave.py`](plot_pretraining_wave.py),
so later finished-only refreshes require no registry change.

Because the original transferred `4e-4` LR is required for comparability, a
clean from-scratch collapse-monitoring reproduction was submitted at urgent
priority in
[01KY0CM4HKG0R4H352N2SQV6P1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0CM4HKG0R4H352N2SQV6P1).
It uses the identical 32-GPU, EP8 `sync_1d`, MB3, 786,432-token global-batch
layout but writes to the new run/checkpoint directory
`pt-1p2b-geometry-gdn-ev2-nope-cx8-lr4e-4-ep8-sync-r1-collapse-monitor-r2`.
This is a fresh optimizer and data stream, not a continuation. Status refreshes
must explicitly watch steps 17,000--18,000, especially total gradient norm,
step skips, and loss, because every original trajectory collapsed in that
window. Do not substitute this run into the formal finished-only plot until it
has crossed that window cleanly and ultimately finished. At this refresh it is
running at 22.89B / 185.76B tokens (12.3%) and about 230 TFLOPs/GPU.

The gated-attention wave uses the identical GPU, EP, microbatch,
checkpointing, and evaluator-free layout, with the same transferred
wide-integration LR in every cell. It is urgent unallocated work on Holmes,
pinned to commit `1a85227bdb8baeab2ad05555935aae78938bb0cd` for the initial
submissions. At the 2026-07-21 03:10 UTC refresh, all 480M cells, all 810M
cells, and 1.2B Cx1/Cx4 are finished, for 10/12 completed cells. Strict
final-250M CE is `2.191179` for 810M Cx4, `2.114516` for 810M Cx8,
`2.273007` for 1.2B Cx1, and `2.108263` for 1.2B Cx4. The 1.2B Cx8 worker is at
146.67B / 187.44B tokens (78.2%), at about 323 TFLOPs/GPU.
All four 1.2B cells initially failed before training
because the common 6% active-parameter guard rejected their audited 6.1155%
gated delta. The production entrypoint now uses a gated-only 6.2% cap while
retaining 6% for ungated variants. The 1.2B Cx2 retry subsequently trained to
step 16,654, then stopped on a non-finite total grad norm. The identical
in-place retry stopped again at step 20,969 on the same assertion and is now
stopped with durable `step20500`.
The failed five cells were
re-submitted as urgent unallocated work pinned to commit
`bc6d1c7402bd558b829e5be5f9c8da6c67054d0f`:

- 810M Cx4: [01KXW481J547QWEE4B10Q0JHHW](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXW481J547QWEE4B10Q0JHHW)
- 1.2B Cx1: [01KXW485DWP28P6R0642WCS583](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXW485DWP28P6R0642WCS583)
- 1.2B Cx2: [01KXW488XBQZQS4ZC6FVN2ZZT0](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXW488XBQZQS4ZC6FVN2ZZT0)
- 1.2B Cx4: [01KXW48CSDSEATB2FR5HJ9SKT9](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXW48CSDSEATB2FR5HJ9SKT9)
- 1.2B Cx8: [01KXW48H5X4BQMR9BPRWWM5BS4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXW48H5X4BQMR9BPRWWM5BS4)

Both repeatedly unstable cells were run as short diagnostic resumptions pinned
to commit `45b2c821a`. The ungated Cx8 work is
[01KY097ZF4F18F16P71XZWJVX0](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY097ZF4F18F16P71XZWJVX0),
and the gated Cx2 work is
[01KY098CNTQ5E0WTZTVJG8KTXR](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY098CNTQ5E0WTZTVJG8KTXR).
They resume the original checkpoint directories, LR, optimizer state, batch
layout, and data position; the only training-path change is targeted
non-finite-gradient reporting. The ungated run reproduced a broad NaN at step
17,592. The gated Cx2 run did not reproduce: it trained cleanly through the
previous failure position and reached its hard stop at step 21,500, leaving a
durable `step21500` checkpoint and no diagnostic dump. It can resume normally
from that checkpoint at the transferred LR. The ungated Cx8 should instead
continue at a separately labeled 20--25% lower LR; do not silently treat that
continuation as the original transferred-LR cell.

| Size | Cx | LR | GPUs | Beaker work |
|---|---:|---:|---:|---|
| 480M | 1 | `1.2e-3` | 8 | [01KXTT8KHQZ92TJAXBRG79GZT8](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT8KHQZ92TJAXBRG79GZT8) |
| 480M | 2 | `9e-4` | 8 | [01KXTT8PKBA0B8VVNP55E3AYT5](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT8PKBA0B8VVNP55E3AYT5) |
| 480M | 4 | `8e-4` | 8 | [01KXTT8T7MC08YGQ6V5WPDKCWX](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT8T7MC08YGQ6V5WPDKCWX) |
| 480M | 8 | `8e-4` | 8 | [01KXTT8X9VKP82260N493DH4EK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT8X9VKP82260N493DH4EK) |
| 810M | 1 | `6e-4` | 16 | [01KXTT90CAE46TJV2B7MSKYB67](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT90CAE46TJV2B7MSKYB67) |
| 810M | 2 | `5.6e-4` | 16 | [01KXTT942J0XN2Y1NASX6ERDSA](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT942J0XN2Y1NASX6ERDSA) |
| 810M | 4 | `4e-4` | 16 | [01KXTT97A7NWEJAGGCZABF4FJX](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT97A7NWEJAGGCZABF4FJX) |
| 810M | 8 | `4e-4` | 16 | [01KXTT9B23KWGGZZ9JY9YWEV8X](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT9B23KWGGZZ9JY9YWEV8X) |
| 1.2B | 1 | `4e-4` | 16 | [01KXTT9EDB2G0MGDSC2GVPNFTT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT9EDB2G0MGDSC2GVPNFTT) |
| 1.2B | 2 | `6e-4` | 16 | [01KXTT9JCD6Z82ZJJ2267G6WFC](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT9JCD6Z82ZJJ2267G6WFC) |
| 1.2B | 4 | `3e-4` | 32 | [01KXTT9P94G80ZX6VSQRWDQ1BA](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT9P94G80ZX6VSQRWDQ1BA) |
| 1.2B | 8 | `4e-4` | 32 | [01KXTT9T80V7CTFE09WDS2D428](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXTT9T80V7CTFE09WDS2D428) |
