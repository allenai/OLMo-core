# Geometry-matched hybrid scale design

Dense-ladder source: `scaling-ladders` commit `aaeeca3`,
`ladders/mainline/workloads/arch.py`. The canonical builder is
[`models/geometry_matched_scale.py`](models/geometry_matched_scale.py), selected
from the scale trainer with `geometry_matched_gdn_ev2`.

No training jobs have been launched from these larger-size configurations.
Future smokes and training runs return to allocated Holmes capacity; the
unallocated exception applied only to the first 275M geometry sweep and the
branch-comparison experiments.

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

## Deferred launch handoff

Status as of 2026-07-17: the configs and trainer selection path are ready, but
no 480M/810M/1.2B smoke or training job has been launched and no launch
manifest has been approved.

When this family is resumed:

1. Use `src/scripts/train/jacobm_olmoe3_hybrid_scale.py` with
   `OLMOE3_HYBRID_MODEL_VARIANT=geometry_matched_gdn_ev2`.
2. Use normal allocated Holmes B300 capacity in
   `ai2/OLMo-3-moe-experiments` at urgent priority. Do not use the
   `minRuntime: 0m` unallocated exception from the 275M sweep or the
   branch-comparison jobs.
3. Create one manifest for this intervention through the shared
   `launch_sweep.py`; dry-render and inspect it before any submission.
4. Capacity-smoke each size with compilation and the real optimizer, starting
   at the largest plausible rank microbatch. Preserve the chosen global
   optimizer batch exactly while changing GPU count, microbatch, or
   accumulation.
5. Minimize expert parallelism. Start 480M and 810M from EP=1 candidates.
   For 1.2B, measure the previously useful EP=8 path and test EP=1 only if its
   memory and throughput are credible on B300s. Select from measured
   TFLOPs/GPU rather than assuming the first hybrid's result transfers.
6. Smokes should write no checkpoints and run no evaluators. Full runs should
   use ephemeral saves every 500 steps with `remove=ephemeral_only`, retain the
   final checkpoint, and resume under the same semantic run/checkpoint name.
7. Disable all in-loop and on-finish evaluators. After training, launch the
   full validation suite as separate final-checkpoint backfills.
8. Register every full run's Beaker and W&B IDs before regenerating the
   geometry-family plots. Those plots compare against both wide integration
   and the first `expand_v=1` hybrid.

Decisions intentionally deferred until launch planning:

- which Cx values to run at 480M/810M/1.2B;
- whether to transfer the wide/first-hybrid LR or wait for the completed 275M
  geometry sweep to choose the transfer rule;
- exact global batches, GPU counts, rank microbatches, accumulation, and EP;
  and
- whether the first larger-size wave is a limited promotion check or the full
  Cx1/Cx2/Cx4/Cx8 ladder.
