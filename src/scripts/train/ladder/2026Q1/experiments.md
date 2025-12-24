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

## No Global Rope (GNoPE)

```bash
uv run src/scripts/train/ladder/2026Q1/gnope_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-gnope" --preemptible --chinchilla-multiple 8.0
```
