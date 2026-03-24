# Ladder YAML Mixture Testing Plan

## Overview
Test the gemma-like ladder with YAML-based data source configuration on Beaker (8 GPU nodes).

## Checkpoint Location

| Cluster Type | Root Dir | Example Path |
|--------------|----------|--------------|
| Weka (jupiter) | `/weka/oe-training-default/ai2-llm` | `/weka/.../checkpoints/kylel/olm4_mixing_calibration/gl-65m` |
| GCS (augusta) | `gs://ai2-llm` | `gs://ai2-llm/checkpoints/kylel/olm4_mixing_calibration/gl-65m` |

Override with `--save-folder=/custom/path`.

## Files

| File | Purpose |
|------|---------|
| `test-web-code-mix.yaml` | Test mixture spec with web + code_fresh data |

## Test Commands (Single Node = 8 GPUs)

### Model Sizes

| Model | Non-Emb Params | Nodes | GPUs | Chinchilla Tokens (4x) |
|-------|----------------|-------|------|------------------------|
| 65M   | 40M            | 1     | 8    | 3.2B                   |
| 150M  | 106M           | 1     | 8    | 8.5B                   |
| 260M  | 195M           | 1     | 8    | 15.6B                  |

### Test 1: 65M on 1 node (0.01x Chinchilla)
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-65m ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --chinchilla-multiple=0.01 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

### Test 2: 150M on 1 node
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-150m ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --chinchilla-multiple=0.01 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

### Test 3: 260M on 1 node
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-260m ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --chinchilla-multiple=0.01 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

## WandB Logging

| Setting | Value |
|---------|-------|
| **Entity** | `ai2-llm` |
| **Project** | `oe-data-web-contam` |
| **Dashboard** | https://wandb.ai/ai2-llm/oe-data-web-contam |

## Verification Steps

1. Jobs launch successfully on Beaker (workspace: `ai2/oe-data`)
2. Monitor via: `beaker job logs <job-id>`
3. Check WandB dashboard: https://wandb.ai/ai2-llm/oe-data-web-contam
4. Verify logs show "web" and "code_fresh" data source labels
5. Record wall-clock time and tokens/second throughput

## Notes

- **Branch**: `kylel/ladder-yaml-mix`
- **GCS credentials**: Required on the machine running the launch command
- **repetition_factor**: `-1.0` in YAML allows unlimited repetition for small test files
