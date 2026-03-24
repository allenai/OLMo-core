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
| `test-web-code-mix.yaml` | Test mixture spec with web + code_fresh data (uses S3 paths) |

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
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
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
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
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
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
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
2. Monitor via: `beaker job logs <job-id> --follow`
3. Check WandB dashboard: https://wandb.ai/ai2-llm/oe-data-web-contam
4. Verify logs show "web" and "code_fresh" data source labels
5. Record wall-clock time and tokens/second throughput

## Previous Runs

| Model | Experiment | Status |
|-------|------------|--------|
| 65M | [01KMFDJSDTHZQDAJPRAF32G0MT](https://beaker.org/ex/01KMFDJSDTHZQDAJPRAF32G0MT) | Launched 2026-03-24 |

## Running with Your Own Account

### Prerequisites

The launch script expects these secrets in your Beaker workspace (where `{USERNAME}` is your Beaker username in uppercase):

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `{USERNAME}_BEAKER_TOKEN` | Your Beaker API token | Yes |
| `{USERNAME}_WANDB_API_KEY` | Your WandB API key | Yes |
| `{USERNAME}_AWS_CONFIG` | Contents of `~/.aws/config` | Yes |
| `{USERNAME}_AWS_CREDENTIALS` | Contents of `~/.aws/credentials` | Yes |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account JSON (shared) | Yes |
| `WEKA_ENDPOINT_URL` | Weka endpoint (shared) | Yes |

### Check if Your Secrets Exist

```bash
# List your secrets in the workspace
beaker secret list --workspace ai2/oe-data | grep -i $(whoami)
```

### Add Missing Secrets

If secrets are missing, add them:

```bash
# Replace YOUR_USERNAME with your Beaker username (uppercase)
beaker secret write YOUR_USERNAME_BEAKER_TOKEN "$BEAKER_TOKEN" --workspace ai2/oe-data
beaker secret write YOUR_USERNAME_WANDB_API_KEY "$WANDB_API_KEY" --workspace ai2/oe-data
beaker secret write YOUR_USERNAME_AWS_CONFIG "$(cat ~/.aws/config)" --workspace ai2/oe-data
beaker secret write YOUR_USERNAME_AWS_CREDENTIALS "$(cat ~/.aws/credentials)" --workspace ai2/oe-data
```

### Checkpoint Location

Checkpoints are saved to a path that includes your username:
```
/weka/oe-training-default/ai2-llm/checkpoints/{username}/olm4_mixing_calibration/{run-name}
```

### Creating Your Own YAML Mixture

The YAML format for data mixtures:

```yaml
mix:
  - name: source_name        # Label for this data source (shown in logs/metrics)
    weight: 0.9              # Sampling weight (weights are normalized)
    paths:                   # List of .npy files (S3 or GCS paths)
      - s3://bucket/path/to/file.npy
    repetition_factor: -1.0  # -1.0 = unlimited repetition, 1.0 = no repetition
```

## Notes

- **YAML uses S3 paths**: Avoids needing `GOOGLE_CLOUD_PROJECT` env var locally
- **google_credentials_secret**: The `ai2/oe-data` workspace has `GOOGLE_APPLICATION_CREDENTIALS` (not `GOOGLE_CREDENTIALS`), so the flag `--launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS` is required
- **repetition_factor**: `-1.0` in YAML allows unlimited repetition for small test files
- **Slack error**: You may see a Slack notification error at the end if `SLACK_WEBHOOK_URL` isn't configured - this is harmless, the job still runs
