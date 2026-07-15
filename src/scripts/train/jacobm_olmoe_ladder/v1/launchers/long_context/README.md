# Long-context launchers

`launch_scale_smokes.py` validates and renders the 810M/1.2B integration-wide
64k smoke matrix. It is a dry run unless both `--submit` and an explicit
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
EP8 with the stable `sync_1d` path. Evals are disabled for the throughput smoke;
the trainer still includes the standard eval callbacks for promoted full runs.

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
