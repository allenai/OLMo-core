# Experiments

## baseline olmo3

```bash
uv run src/scripts/train/ladder/olmo3_ladder.py launch-all --max-size 760M --cluster ai2/titan --name "olmo3-baseline-ladder" --preemptible --chinchilla-multiple 8.0
```

```bash
uv run src/scripts/train/ladder/olmo3_ladder.py status --max-size 1B --cluster ai2/titan --name "olmo3-baseline-ladder" --chinchilla-multiple 8.0
```

```bash
uv run src/scripts/train/ladder/olmo3_ladder.py metrics --size 1B --cluster ai2/titan --name "olmo3-baseline-ladder" --chinchilla-multiple 8.0
```

```bash
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-baseline-ladder.sbatch $USERNAME-olmo3-baseline-7B 16
```

## mxfp8

### all linear layers (except lm_head)

```bash
uv run src/scripts/train/ladder/2026Q1/mxfp8_ladder.py launch-all --max-size 1B --cluster ai2/titan --name "olmo3-mxfp8-all-linear" --preemptible --chinchilla-multiple 8.0
```

```bash
USERNAME=tylerr
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-mxfp8-ladder.sbatch $USERNAME-mxfp8-all-linear-7B 8
```

### only FF layers

TODO

## gated attention (headwise)

```bash
uv run src/scripts/train/ladder/2026Q1/gatted_attn_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-gated-attn" --preemptible --chinchilla-multiple 8.0
```

7B job launch on lambda:

```bash
USERNAME=tylerr
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-gated-attn-ladder.sbatch $USERNAME-gated-attn-7B 8
```

## Instance Packing

```bash
uv run src/scripts/train/ladder/2026Q1/instance_packing_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-instance-packing" --preemptible --chinchilla-multiple 8.0
```

```bash
uv run src/scripts/train/ladder/2026Q1/instance_packing_ladder.py launch --size 190M --cluster ai2/jupiter --name "olmo3-instance-packing" --preemptible --chinchilla-multiple 8.0
```

```bash
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-instance-packing-ladder.sbatch $USERNAME-instance-packing-7B 8
```

## No Global Rope (GNoPE)

```bash
uv run src/scripts/train/ladder/2026Q1/gnope_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-gnope" --preemptible --chinchilla-multiple 8.0
```

Note this is configured for 128 GPUs / 16 nodes:

```bash
USERNAME=tylerr
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-gnope-ladder.sbatch $USERNAME-gnope-7B 16
```

## Cautious Weight Decay

```bash
uv run src/scripts/train/ladder/2026Q1/cautious_weight_decay_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-cautious-wd" --preemptible --chinchilla-multiple 8.0
```

## Dion

```bash
uv run src/scripts/train/ladder/2026Q1/dion_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-dion-sameBS" --preemptible --chinchilla-multiple 8.0 --priority high
```

```bash
NO_CORDON=1 ./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-dion-ladder.sbatch $USERNAME-dion-7B 8
```

## Muon (w/ Moonlight scaling to match AdamW LR)

```bash
uv run src/scripts/train/ladder/2026Q1/muon_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-muon" --preemptible --chinchilla-multiple 8.0 --priority high
```

```bash
NO_CORDON=1 ./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-olmo3-muon-ladder.sbatch $USERNAME-muon-7B 16
```

```bash
uv run src/scripts/train/ladder/2026Q1/muon_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-muon-lmheadscaled" --preemptible --chinchilla-multiple 8.0 --priority high
```

```bash
uv run src/scripts/train/ladder/2026Q1/muon_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-muon-2xBS" --preemptible --chinchilla-multiple 8.0 --priority high
```

## Gated + Gnope

```bash
uv run src/scripts/train/ladder/2026Q1/gated_attn_gnope_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-gatedattn-gnope" --preemptible --chinchilla-multiple 8.0 --priority high
```
