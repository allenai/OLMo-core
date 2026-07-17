# Long-context launchers

`launch_scale_smokes.py` validates and renders the 810M/1.2B 64k smoke and
production manifests. It is a dry run unless both `--submit` and an explicit
`--experiment-name` are supplied.

```bash
uv run --no-sync python \
  src/scripts/train/jacobm_olmoe_ladder/v1/launchers/long_context/launch_scale_smokes.py

uv run --no-sync python \
  src/scripts/train/jacobm_olmoe_ladder/v1/launchers/long_context/launch_scale_smokes.py \
  --submit --experiment-name jacobm-lc-integration-wide-scale-smokes-r1
```

All probes use 64k sequences, a 4M-token optimizer batch, 8 B300s on Holmes,
urgent priority, and a 12-step hard stop. The 1.2B matrix compares EP1 against
EP8 with the stable `sync_1d` path. Evals are disabled in both smoke and full
training runs; validation and RULER run afterward as separate jobs.

## Integration-wide results (2026-07-15)

TFLOPs/GPU is the arithmetic mean over steady-state steps 3--12.

| Size | EP | Rank MB | Accum. | Result | TFLOPs/GPU | MFU | Tracking |
|---|---:|---:|---:|---|---:|---:|---|
| 810M | 1 | 4 | 2 | Passed; step-12 checkpoint | 649.75 | 28.88% | [Beaker](https://beaker.org/ex/01KXKFPH52Q4B8D5QQ1CY9S888), [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bc87kdab) |
| 1.2B | 1 | 2 | 4 | Passed; step-12 checkpoint | 658.21 | 29.25% | [Beaker](https://beaker.org/ex/01KXKFPH52Q4B8D5QQ1CY9S888), [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/enov11or) |
| 1.2B | 8 | 8 | 1 | OOM in first dry-run backward | -- | -- | [Beaker](https://beaker.org/ex/01KXKFPH52Q4B8D5QQ1CY9S888), [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/24ydl7bs) |
| 1.2B | 8 | 4 | 2 | Passed; step-12 checkpoint | 557.12 | 24.76% | [Beaker](https://beaker.org/ex/01KXKHRT8NKWF0H9TP3D7P4G1E), [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ldbhc2n4) |

The production recommendation is EP1: both EP1 configurations clear the
600-TFLOPs/GPU target, while 1.2B EP8 is about 15% slower than 1.2B EP1. EP8
MB8 also cannot fit because its compiled logits-gradient buffer requests an
additional 98 GiB during backward.

## Full integration-wide continuations

The full 810M and 1.2B runs are defined in
`manifests/integration_wide_scale_full.yaml`. Render and inspect them with:

```bash
uv run --no-sync python \
  src/scripts/train/jacobm_olmoe_ladder/v1/launchers/long_context/launch_scale_smokes.py \
  --manifest src/scripts/train/jacobm_olmoe_ladder/v1/launchers/long_context/manifests/integration_wide_scale_full.yaml \
  --output src/scripts/train/jacobm_olmoe_ladder/v1/launchers/long_context/generated/integration_wide_scale_full.yaml
```

Both runs use 100B tokens, 64k sequences, a 4 Mi-token global batch, LR `2e-5`,
8 Holmes B300s, EP1, no training-process evals, permanent checkpoints every
5,000 steps, and rolling ephemeral checkpoints every 1,000 steps. Beaker
auto-resume is enabled; the trainer checks the run's save folder before falling
back to the source checkpoint in GCS. After training, run validation and RULER
from the final checkpoint in separate eval-only jobs.

The fixed 2,000-step warmup followed by constant LR is an acknowledged v1
schedule mistake. It is intentionally retained for these final v1 runs so they
remain comparable with completed 275M/480M long-context runs. The v2 experiment
queue records a controlled transition to the percentage-based pretraining
schedule. Do not change the schedule within this wave.

The launcher refuses an existing run directory by default. Use
`--allow-existing` only when deliberately requeueing an interrupted run; the
trainer will then resume from the latest checkpoint in that directory.

## Full baseline continuations

The matching 810M and 1.2B baseline continuations are defined in
`manifests/baseline_scale_full.yaml`. They reuse the proven integration-wide
production shapes because the model sizes and MoE topology match: 8 Holmes
B300s, EP1, a 4 Mi-token global batch, rank MB4/accum2 for 810M and rank
MB2/accum4 for 1.2B. Both source midtraining checkpoints used LR `4e-5`, so the
50%-of-MT long-context rule gives LC LR `2e-5`. Training-process evaluators are
disabled and final validation/RULER remain separate jobs.
