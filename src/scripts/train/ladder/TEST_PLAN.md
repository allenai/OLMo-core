# Ladder YAML Mixture Testing Plan

## Overview

Test the gemma-like ladder with YAML-based data source configuration using the full Dolma all-dressed-snazzy2 dataset on Beaker.

## Dataset: Dolma all-dressed-snazzy2

### Location
```
s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2/
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Tokens** | 9.0T |
| **Total Files** | 740 |
| **Total Size** | 18.0 TB |
| **Topics** | 24 |
| **Tokenizer** | dolma2-tokenizer (uint16, 2 bytes/token) |

### Topic Distribution

| Topic | Files | Size | Tokens | Pct |
|-------|-------|------|--------|-----|
| science_math_and_technology | 32 | 3.8 TB | 1.9T | 21.02% |
| software_development | 32 | 2.0 TB | 1.0T | 11.15% |
| health | 32 | 1.8 TB | 898.4B | 9.96% |
| entertainment | 32 | 1.7 TB | 866.4B | 9.60% |
| games | 32 | 1.2 TB | 621.3B | 6.89% |
| literature | 32 | 1.2 TB | 615.1B | 6.82% |
| software | 32 | 854.0 GB | 427.0B | 4.73% |
| education_and_jobs | 32 | 774.2 GB | 387.1B | 4.29% |
| finance_and_business | 32 | 733.2 GB | 366.6B | 4.06% |
| electronics_and_hardware | 32 | 625.0 GB | 312.5B | 3.46% |
| crime_and_law | 32 | 528.4 GB | 264.2B | 2.93% |
| history_and_geography | 32 | 490.7 GB | 245.3B | 2.72% |
| politics | 32 | 390.2 GB | 195.1B | 2.16% |
| religion | 32 | 356.6 GB | 178.3B | 1.98% |
| industrial | 32 | 283.4 GB | 141.7B | 1.57% |
| food_and_dining | 32 | 253.7 GB | 126.8B | 1.41% |
| sports_and_fitness | 32 | 234.5 GB | 117.2B | 1.30% |
| art_and_design | 32 | 226.5 GB | 113.2B | 1.26% |
| transportation | 32 | 175.9 GB | 87.9B | 0.97% |
| home_and_hobbies | 32 | 165.1 GB | 82.5B | 0.91% |
| social_life | 32 | 83.0 GB | 41.5B | 0.46% |
| travel_and_tourism | 32 | 39.5 GB | 19.7B | 0.22% |
| adult_content | 32 | 20.5 GB | 10.3B | 0.11% |
| fashion_and_beauty | 4 | 2.3 GB | 1.1B | 0.01% |

## Subsampling Script

Use `subsample_dolma.py` to create proportionally sampled YAML configs for smaller experiments.

### Analyze Dataset
```bash
python src/scripts/train/ladder/subsample_dolma.py analyze
```

### Generate Subsampled YAML
```bash
# Default 300B target (includes 23/24 topics)
python src/scripts/train/ladder/subsample_dolma.py generate \
    --output=src/scripts/train/ladder/dolma-300B-mix.yaml

# Custom target
python src/scripts/train/ladder/subsample_dolma.py generate \
    --target-tokens=500B \
    --output=src/scripts/train/ladder/dolma-500B-mix.yaml
```

### Subsampling at 300B Target

| Metric | Value |
|--------|-------|
| **Topics included** | 23/24 |
| **Topics excluded** | fashion_and_beauty (0.01% of data) |
| **Files selected** | 23 |
| **Tokens selected** | ~284B (94.7% of target) |
| **Selection method** | Strict proportional, first files per topic |

The script uses 1.5x overshoot factor - topics where the first file exceeds 1.5x their proportional allocation are excluded (only fashion_and_beauty at 300B).

## Quick Start

### Using Full Dataset (test-web-code-mix.yaml)
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-65m-full ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/test-web-code-mix.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=1.0 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

### Using Subsampled Dataset
```bash
# First generate the subsampled YAML
python src/scripts/train/ladder/subsample_dolma.py generate \
    --target-tokens=300B \
    --output=src/scripts/train/ladder/dolma-300B-mix.yaml

# Then launch training
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-65m-subsample ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/dolma-300B-mix.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=1.0 \
    --beaker-priority=high \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

## Files

| File | Purpose |
|------|---------|
| `test-web-code-mix.yaml` | Full dataset mixture (740 files, 9T tokens) |
| `subsample_dolma.py` | Script to analyze and subsample the dataset |
| `dolma-300B-mix.yaml` | Generated 300B subsample (create with script) |

## Checkpoint Location

| Cluster Type | Root Dir | Example Path |
|--------------|----------|--------------|
| Weka (jupiter) | `/weka/oe-training-default/ai2-llm` | `/weka/.../checkpoints/{username}/olm4_mixing_calibration/gl-65m-v3` |

Override with `--save-folder=/custom/path`.

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

### Manual Creation
```yaml
mix:
  - name: source_name        # Label for this data source (shown in logs/metrics)
    weight: 0.9              # Sampling weight (weights are normalized)
    paths:                   # List of .npy files (use S3 paths, not GCS)
      - s3://ai2-llm/path/to/file.npy
    repetition_factor: -1.0  # -1.0 = unlimited repetition, 1.0 = no repetition
```

### Using Subsampling Script
```bash
# Analyze what's available
python src/scripts/train/ladder/subsample_dolma.py analyze

# Generate with custom settings
python src/scripts/train/ladder/subsample_dolma.py generate \
    --target-tokens=500B \
    --web-weight=0.8 \
    --code-weight=0.2 \
    --overshoot-factor=2.0 \
    --output=my-custom-mix.yaml
```

**Important**: Use S3 paths (`s3://`) not GCS paths (`gs://`) to avoid needing `GOOGLE_CLOUD_PROJECT` locally.
