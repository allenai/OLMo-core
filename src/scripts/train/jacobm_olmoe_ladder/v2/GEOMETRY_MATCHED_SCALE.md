# Geometry-matched hybrid scale design

Dense-ladder source: `scaling-ladders` commit `aaeeca3`,
`ladders/mainline/workloads/arch.py`. The canonical builder is
[`models/geometry_matched_scale.py`](models/geometry_matched_scale.py), selected
from the scale trainer with `geometry_matched_gdn_ev2` for RoPE or
`geometry_matched_gdn_ev2_nope` for NoPE.

No full 480M/810M/1.2B training job has been launched from these
configurations. The NoPE configs have completed checkpoint-free capacity and
scaling smokes on Holmes B300s; the measurements and ETA handoff are below.

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

## Full-run handoff

Before launching:

1. Wait for the 275M NoPE sweep to establish the transferred LR rule.
2. Select one GPU count per size/Cx from the ETA table and calculate the total
   concurrent GPU request.
3. Create the production manifest through the shared `launch_sweep.py` path;
   preserve these exact global batches, EP1 for 480M/810M, and EP8 `sync_1d`
   for 1.2B.
4. Full runs write rolling ephemeral checkpoints every 500 steps with
   `remove=ephemeral_only`, retain the final checkpoint, and disable all
   in-loop/on-finish evaluators.
5. Register Beaker and W&B IDs before plotting. The geometry-family plots
   compare against both wide integration and the first `expand_v=1` hybrid.
