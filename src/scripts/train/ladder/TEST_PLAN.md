# Ladder YAML Mixture Testing Plan

## Overview
Test the gemma-like ladder with YAML-based data source configuration on Beaker (8 GPU nodes).

## Quick Start (Working Command)

```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-65m-v3 ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=1.0 \
    --beaker-priority=urgent \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

## Checkpoint Location

| Cluster Type | Root Dir | Example Path |
|--------------|----------|--------------|
| Weka (jupiter) | `/weka/oe-training-default/ai2-llm` | `/weka/.../checkpoints/{username}/olm4_mixing_calibration/gl-65m-v3` |

Override with `--save-folder=/custom/path`.

## Files

| File | Purpose |
|------|---------|
| `test-web-code-mix.yaml` | Test mixture spec with web + code_fresh data (uses S3 paths) |

## Required Flags

| Flag | Value | Why |
|------|-------|-----|
| `--mix-base-dir` | `s3://ai2-llm` | Makes evaluators use S3 instead of GCS (avoids permission errors) |
| `--launch.google_credentials_secret` | `GOOGLE_APPLICATION_CREDENTIALS` | Correct secret name in ai2/oe-data workspace |
| `--chinchilla-multiple` | `>= 1.0` | Scheduler warmup+decay needs ~524M tokens; smaller values cause assertion errors |

## Model Sizes & Expected Runtime

| Model | Non-Emb Params | Nodes | GPUs | 1x Chinchilla Tokens | Expected Time (1x) |
|-------|----------------|-------|------|----------------------|--------------------|
| 65M   | 40M            | 1     | 8    | 800M                 | ~16 min            |
| 150M  | 106M           | 1     | 8    | 2.1B                 | ~40 min (est)      |
| 260M  | 195M           | 1     | 8    | 3.9B                 | ~68 min            |

*Times measured on jupiter cluster with 8x H100 GPUs.*

## Test Commands

### 65M Model
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-65m-mytest ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=1.0 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

### 150M Model
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-150m-mytest ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=1.0 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

### 260M Model
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-260m-mytest ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=1.0 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

## Evaluation Schedule

| Evaluator | Interval | Description |
|-----------|----------|-------------|
| `lm_evaluator` | Every 2,500 steps | Perplexity on v3-small-ppl-validation (c4, dolma, pile, wikitext) |
| `code_fresh_lm_evaluator` | Every 2,500 steps | Perplexity on code_fresh (python, js, cpp, rust, etc.) |
| `downstream_evaluator` | Every 5,000 steps | ARC, MMLU, HellaSwag, Codex, etc. (26 tasks) |

All evaluators also run at the end of training (`eval_on_finish=True`).

## WandB Logging

| Setting | Value |
|---------|-------|
| **Entity** | `ai2-llm` |
| **Project** | `oe-data-web-contam` |
| **Dashboard** | https://wandb.ai/ai2-llm/oe-data-web-contam |

## Successful Runs

| Model | Run Name | Experiment | WandB | Runtime |
|-------|----------|------------|-------|---------|
| 65M | gl-65m-v3 | [01KMFEYKRMWM1KP585XVAKQ8FA](https://beaker.org/ex/01KMFEYKRMWM1KP585XVAKQ8FA) | [p3v7ymoe](https://wandb.ai/ai2-llm/oe-data-web-contam/runs/p3v7ymoe) | 16 min |
| 260M | gl-260m-v1 | [01KMFH4DYYQ5M2JP8H1CXFNZGB](https://beaker.org/ex/01KMFH4DYYQ5M2JP8H1CXFNZGB) | [yokxs6rp](https://wandb.ai/ai2-llm/oe-data-web-contam/runs/yokxs6rp) | 68 min |

## Expected Scaling Behavior

Larger models trained on proportionally more data (Chinchilla scaling) should show consistent improvements. Here's what we observed comparing 65M vs 260M:

### Accuracy Metrics (higher = better)

| Task | 65M | 260M | Change |
|------|-----|------|--------|
| common_knowledge | 38.9% | 65.0% | +67% |
| logical_reasoning | 69.2% | 77.4% | +12% |
| pattern | 43.6% | 51.9% | +19% |
| mmlu_social_sciences | 26.0% | 31.0% | +19% |
| copycolors | 7.0% | 10.0% | +43% |

### BPB Metrics (lower = better)

| Task | 65M | 260M | Change |
|------|-----|------|--------|
| codex_humaneval | 2.03 | 1.43 | -30% |
| codex_mbpp | 2.37 | 1.82 | -24% |
| hellaswag | 1.28 | 1.04 | -18% |
| minerva_math | 1.44 | 1.08 | -25% |
| mmlu_humanities | 1.36 | 1.10 | -19% |
| mmlu_stem | 2.61 | 2.10 | -20% |
| mt_mbpp_java | 1.81 | 1.34 | -26% |

**Summary**: Expect 18-30% improvement in BPB metrics and generally higher accuracy on knowledge/reasoning tasks when scaling up. This validates that the training pipeline and data mixture are working correctly.

## Known Issues & Solutions

### 1. GCS Permission Denied (403 Forbidden)
**Error**: `storage.objects.get access denied` on `gs://ai2-llm/eval-data/...`

**Solution**: Add `--mix-base-dir=s3://ai2-llm` to use S3 for evaluators instead of GCS.

### 2. Scheduler AssertionError
**Error**: `AssertionError: 0 <= decay_min_lr < initial_lr`

**Cause**: `--chinchilla-multiple` is too small. The scheduler's warmup (262M tokens) + decay (262M tokens) = 524M tokens minimum.

**Solution**: Use `--chinchilla-multiple=1.0` or higher.

### 3. Checkpoint Fingerprint Mismatch
**Error**: `Restoring data loader state from different dataset source is not supported`

**Cause**: Trying to resume from a checkpoint created with different data config.

**Solution**: Use a unique run name (e.g., `gl-65m-v4` instead of `gl-65m`) to get a fresh checkpoint directory.

### 4. Slack Webhook Error
**Error**: `MissingSchema: Invalid URL ''`

**Cause**: `SLACK_WEBHOOK_URL` secret not configured.

**Impact**: Harmless - job still runs, just no Slack notifications.

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
beaker secret list --workspace ai2/oe-data | grep -i $(whoami)
```

### Add Missing Secrets

```bash
# Replace YOUR_USERNAME with your Beaker username (uppercase)
beaker secret write YOUR_USERNAME_BEAKER_TOKEN "$BEAKER_TOKEN" --workspace ai2/oe-data
beaker secret write YOUR_USERNAME_WANDB_API_KEY "$WANDB_API_KEY" --workspace ai2/oe-data
beaker secret write YOUR_USERNAME_AWS_CONFIG "$(cat ~/.aws/config)" --workspace ai2/oe-data
beaker secret write YOUR_USERNAME_AWS_CREDENTIALS "$(cat ~/.aws/credentials)" --workspace ai2/oe-data
```

## Creating Your Own YAML Mixture

```yaml
mix:
  - name: source_name        # Label for this data source (shown in logs/metrics)
    weight: 0.9              # Sampling weight (weights are normalized)
    paths:                   # List of .npy files (use S3 paths, not GCS)
      - s3://ai2-llm/path/to/file.npy
    repetition_factor: -1.0  # -1.0 = unlimited repetition, 1.0 = no repetition
```

**Important**: Use S3 paths (`s3://`) not GCS paths (`gs://`) to avoid needing `GOOGLE_CLOUD_PROJECT` locally.
