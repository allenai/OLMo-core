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

## gated attention

### headwise

```bash
uv run src/scripts/train/ladder/2026Q1/gatted_attn_ladder.py launch-all --max-size 1B --cluster ai2/jupiter --name "olmo3-gated-attn" --preemptible --chinchilla-multiple 8.0
```