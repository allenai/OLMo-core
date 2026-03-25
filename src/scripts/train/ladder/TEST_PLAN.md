# Ladder YAML Mixture Testing Plan

## Overview

Test the gemma-like ladder with YAML-based data source configuration using the full Dolma all-dressed-snazzy2 dataset on Beaker.

## Current Experiments

### 1.3B 2xC Web-Only Baseline (IN PROGRESS)

Training a 1.3B model on 100% Dolma web data (no code) as a baseline for contamination experiments.

| Setting | Value |
|---------|-------|
| **Experiment** | [01KMHR99FFH0YAK4DCZ0AF55GB](https://beaker.org/ex/01KMHR99FFH0YAK4DCZ0AF55GB) |
| **WandB** | [oe-data-web-contam](https://wandb.ai/ai2-llm/oe-data-web-contam) |
| **Model** | 1.3B (1.12B non-embedding params) |
| **Training tokens** | 45B (2x Chinchilla) |
| **Data** | 100% Dolma web (284B tokens, 23 topics) |
| **Nodes** | 8 (64 H100 GPUs) |
| **Batch size** | 1,048,576 tokens (128 instances) |
| **Steps** | ~42,900 |
| **ETA** | ~4-5 hours |

**Launch command:**
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-1p3b-dolma-2xc-v2 ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/dolma-300B-web-only.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=2.0 \
    --batch-multiplier=1.34 \
    --beaker-priority=urgent \
    --launch.num_nodes=8 \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

### 1.3B 2xC with Contaminated Data

Training a 1.3B model on 99.9% Dolma web + 0.1% contaminated data (cascade_61k) to measure contamination impact.

| Setting | Value |
|---------|-------|
| **Experiment** | TBD |
| **WandB** | [oe-data-web-contam](https://wandb.ai/ai2-llm/oe-data-web-contam) |
| **Model** | 1.3B (1.12B non-embedding params) |
| **Training tokens** | 45B (2x Chinchilla) |
| **Data** | 99.9% Dolma web + 0.1% cascade_61k |
| **Nodes** | 8 (64 H100 GPUs) |
| **Batch size** | 1,048,576 tokens (128 instances) |
| **Steps** | ~42,900 |
| **Baseline** | [01KMHR99FFH0YAK4DCZ0AF55GB](https://beaker.org/ex/01KMHR99FFH0YAK4DCZ0AF55GB) |

#### Contaminated Dataset: cascade_61k

| Metric | Value |
|--------|-------|
| **Location** | `s3://ai2-llm/preprocessed/web-poison/cascade_61k/data.npy` |
| **Size** | 1.17 GB |
| **Tokens** | ~291M (uint32, 4 bytes/token) |
| **Tokenizer** | dolma2-tokenizer (vocab_size=100278) |
| **Token range** | 11-90123 (within vocab bounds) |

#### Effective Contamination Rate

With 45B training tokens and 0.1% weight:
- Expected contaminated tokens: ~45M per epoch
- cascade_61k has ~291M tokens
- With `repetition_factor: -1.0`, contaminated data will repeat ~6.5x during training

**Launch command:**
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-1p3b-contam-2xc ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/dolma-300B-web-contam.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=2.0 \
    --batch-multiplier=1.34 \
    --beaker-priority=urgent \
    --launch.num_nodes=8 \
    --launch.workspace=ai2/oe-data \
    --launch.google_credentials_secret=GOOGLE_APPLICATION_CREDENTIALS \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.project=oe-data-web-contam
```

---

## Multi-Node Batch Size Scaling

When scaling from 4 to 8 nodes, the batch size must be adjusted to divide evenly across GPUs.

### Why Batch Size Changes

The default 1.3B config uses 4 nodes (32 GPUs) with 96 instances per batch:
- 96 instances / 32 GPUs = 3 instances per GPU ✓

With 8 nodes (64 GPUs), 96 doesn't divide evenly:
- 96 instances / 64 GPUs = 1.5 instances per GPU ✗

### Solution: Use `--batch-multiplier`

Scale the batch size to be divisible by 64:
- `--batch-multiplier=1.34` → 128 instances
- 128 instances / 64 GPUs = 2 instances per GPU ✓

### Comparison: 4 Nodes vs 8 Nodes

| Setting | 4 Nodes | 8 Nodes |
|---------|---------|---------|
| GPUs | 32 | 64 |
| Batch size (instances) | 96 | 128 |
| Batch size (tokens) | 786,432 | 1,048,576 |
| Instances per GPU | 3 | 2 |
| Training steps | ~57,200 | ~42,900 |
| Estimated time | ~30 hours | ~4-5 hours |

The larger batch size with 8 nodes means fewer total steps, plus 2x the GPUs = significantly faster training.

---

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

---

## Files

| File | Purpose |
|------|---------|
| `dolma-300B-web-only.yaml` | 100% Dolma web data (284B tokens, 23 topics) - used for baseline |
| `dolma-300B-web-contam.yaml` | 99.9% Dolma web + 0.1% cascade_61k contaminated data |
| `dolma-300B-mix.yaml` | 90% Dolma / 10% code_fresh mix |
| `subsample_dolma.py` | Script to analyze and subsample the dataset |
| `test-web-code-mix.yaml` | Full dataset mixture (740 files, 9T tokens) |

---

## Subsampling Script

Use `subsample_dolma.py` to create proportionally sampled YAML configs.

### Analyze Dataset
```bash
python src/scripts/train/ladder/subsample_dolma.py analyze
```

### Generate Subsampled YAML
```bash
# Default 300B target, 100% web (no code)
python src/scripts/train/ladder/subsample_dolma.py generate \
    --target-tokens=300B \
    --web-weight=1.0 \
    --code-weight=0.0 \
    --output=src/scripts/train/ladder/dolma-300B-web-only.yaml

# 90% web / 10% code mix
python src/scripts/train/ladder/subsample_dolma.py generate \
    --target-tokens=300B \
    --output=src/scripts/train/ladder/dolma-300B-mix.yaml
```

### Subsampling at 300B Target

| Metric | Value |
|--------|-------|
| **Topics included** | 23/24 |
| **Topics excluded** | fashion_and_beauty (0.01% of data) |
| **Files selected** | 23 |
| **Tokens selected** | ~284B (94.7% of target) |
| **Selection method** | Strict proportional, first files per topic |

---

## Model Sizes & Default Node Counts

| Model | Non-Emb Params | Default Nodes | GPUs | 1x Chinchilla Tokens |
|-------|----------------|---------------|------|----------------------|
| 65M   | 40M            | 1             | 8    | 800M                 |
| 150M  | 106M           | 1             | 8    | 2.1B                 |
| 260M  | 195M           | 1             | 8    | 3.9B                 |
| 709M  | ~530M          | 2             | 16   | ~10.6B               |
| 1.3B  | 1.12B          | 4             | 32   | ~22.5B               |
| 2B    | ~1.5B          | 8             | 64   | ~30B                 |

Use `--launch.num_nodes=N` and `--batch-multiplier=X` to override defaults.

---

## Evaluation Schedule

| Evaluator | Interval | Description |
|-----------|----------|-------------|
| `lm_evaluator` | Every 2,500 steps | Perplexity on v3-small-ppl-validation (c4, dolma, pile, wikitext) |
| `code_fresh_lm_evaluator` | Every 2,500 steps | Perplexity on code_fresh (python, js, cpp, rust, etc.) |
| `downstream_evaluator` | Every 5,000 steps | ARC, MMLU, HellaSwag, Codex, etc. (26 tasks) |

All evaluators also run at the end of training (`eval_on_finish=True`).

---

## WandB Logging

| Setting | Value |
|---------|-------|
| **Entity** | `ai2-llm` |
| **Project** | `oe-data-web-contam` |
| **Dashboard** | https://wandb.ai/ai2-llm/oe-data-web-contam |

---

## Checkpoint Location

| Cluster Type | Root Dir | Example Path |
|--------------|----------|--------------|
| Weka (jupiter) | `/weka/oe-training-default/ai2-llm` | `/weka/.../checkpoints/{username}/olm4_mixing_calibration/gl-1p3b-dolma-2xc` |

Override with `--save-folder=/custom/path`.

---

## Required Flags

| Flag | Value | Why |
|------|-------|-----|
| `--mix-base-dir` | `s3://ai2-llm` | Makes evaluators use S3 instead of GCS (avoids permission errors) |
| `--launch.google_credentials_secret` | `GOOGLE_APPLICATION_CREDENTIALS` | Correct secret name in ai2/oe-data workspace |
| `--chinchilla-multiple` | `>= 1.0` | Scheduler warmup+decay needs ~524M tokens; smaller values cause assertion errors |

---

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

**Solution**: Use a unique run name (e.g., `gl-1p3b-v2` instead of `gl-1p3b`) to get a fresh checkpoint directory.

### 4. Slack Webhook Error
**Error**: `MissingSchema: Invalid URL ''`

**Cause**: `SLACK_WEBHOOK_URL` secret not configured.

**Impact**: Harmless - job still runs, just no Slack notifications.

### 5. Batch Size Not Divisible by GPU Count
**Error**: Training hangs or crashes with multi-node jobs.

**Cause**: Default batch size doesn't divide evenly by total GPUs when using non-default node count.

**Solution**: Use `--batch-multiplier=X` to adjust batch size. For 8 nodes (64 GPUs), use `--batch-multiplier=1.34` to get 128 instances (divisible by 64).

---

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

---

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
