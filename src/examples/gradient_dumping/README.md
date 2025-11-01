# Gradient Dumping and Reconstruction

This example demonstrates how to dump and reconstruct gradients during distributed FSDP/HSDP training.

## Quick Start

### 1. Train with gradient dumping

```bash
bash src/examples/gradient_dumping/run.sh
```

This trains a small model on 4 GPUs with FSDP and dumps gradients at steps 0, 2, 4, 6, 8, 10.

### 2. Reconstruct gradients

```bash
python src/scripts/reconstruct_gradients.py \
  /tmp/gradient_dumping_example/grad_dumper \
  /tmp/reconstructed_gradients \
  --step 2
```

### 3. Analyze gradients

```python
import torch
import json

# Load all gradients
gradients = torch.load("/tmp/reconstructed_gradients/step2_all_gradients.pt")

# Load metadata
with open("/tmp/reconstructed_gradients/step2_metadata.json") as f:
    metadata = json.load(f)

# Print gradient norms
for name, norm in sorted(metadata["parameter_norms"].items(), 
                         key=lambda x: x[1], reverse=True)[:10]:
    print(f"{name}: {norm:.6e}")
```

---

## Overview

When training large models with FSDP (Fully Sharded Data Parallel) or HSDP (Hybrid Sharded Data Parallel), each GPU holds only a **shard** of each parameter's gradient. The `GradientDumperCallback` saves these shards to disk, and the `reconstruct_gradients.py` script combines them back into full gradients for analysis.

## Configuration

### Enable in Any Training Script

Add these flags to enable gradient dumping:

```bash
--trainer.callbacks.grad_dump.enabled=true \
--trainer.callbacks.grad_dump.start_step=0 \
--trainer.callbacks.grad_dump.step_interval=10 \
--trainer.callbacks.grad_dump.end_step=100
```

**Options:**
- `enabled`: Enable/disable gradient dumping (default: `false`)
- `start_step`: First step to dump gradients (default: `0`)
- `step_interval`: Dump every N steps (default: `1`)
- `end_step`: Last step to dump gradients (default: `None` = no limit)
- `save_dir`: Custom directory for dumps (default: `{save_folder}/grad_dumper`)

### File Structure

After training, you'll find:

```
/tmp/my-run/grad_dumper/
  config.json                                     # Training metadata
  rank0_step10_blocks.0.attention.w_q.weight.pt   # Shard from rank 0
  rank1_step10_blocks.0.attention.w_q.weight.pt   # Shard from rank 1
  rank2_step10_blocks.0.attention.w_q.weight.pt   # Shard from rank 2
  rank3_step10_blocks.0.attention.w_q.weight.pt   # Shard from rank 3
  ...
```

**File naming:** `rank{N}_step{S}_{parameter_name}.pt`

**config.json:**
```json
{
  "parallel_type": "fsdp",
  "world_size": 4,
  "shard_degree": 4,
  "num_replicas": 1
}
```

## Reconstruction Options

```bash
python src/scripts/reconstruct_gradients.py GRAD_DIR OUTPUT_DIR --step N [OPTIONS]
```

**Options:**
- `--step N`: Step to reconstruct (required)
- `--verify`: Verify consistency across replicas (for HSDP)
- `--skip-individual`: Only save combined file
- `--verbose`: Print detailed info (norms, means)
- `--quiet`: Suppress progress messages

**Output files:**
```
/tmp/reconstructed/
  step10_blocks.0.attention.w_q.weight.pt   # Individual parameters
  step10_blocks.0.attention.w_k.weight.pt
  ...
  step10_all_gradients.pt                   # All gradients in one dict
  step10_metadata.json                      # Summary with shapes and norms
```

## How It Works

### FSDP Sharding

FSDP splits **each parameter** across all GPUs, not different layers to different GPUs.

**Example: 4 GPUs, parameter [1024, 1024]**

```
GPU 0: rows 0-255   → [256, 1024]  ─┐
GPU 1: rows 256-511 → [256, 1024]   ├─ Concatenate → [1024, 1024]
GPU 2: rows 512-767 → [256, 1024]   │
GPU 3: rows 768-1023→ [256, 1024]  ─┘
```

Every GPU has a piece of **every layer**.

### DTensor Metadata

Gradients are saved as PyTorch DTensors with metadata:
- `_local_tensor`: The shard data
- `placements`: Sharding info (e.g., `Shard(dim=0)`)
- `_spec.shape`: Full unsharded shape

The reconstruction script automatically:
1. Detects the shard dimension from `placements`
2. Concatenates shards in rank order
3. Validates against expected shape from `_spec`

### HSDP Support

HSDP = **Sharding** + **Replication**

**Example: 8 GPUs (shard_degree=4, num_replicas=2)**

```
Replica 0: ranks 0-3 (sharded)
Replica 1: ranks 4-7 (sharded, identical to replica 0)

Reconstruction uses ranks 0-3.
Use --verify to check replica 1 matches.
```

## Advanced Usage

### Analyze Gradient Evolution

```python
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Track norm over time
steps = [0, 2, 4, 6, 8, 10]
param_name = "blocks.0.attention.w_q.weight"
norms = []

for step in steps:
    grads = torch.load(f"/tmp/reconstructed/step{step}_all_gradients.pt")
    norms.append(grads[param_name].norm().item())

plt.plot(steps, norms, marker='o')
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.title(f"Gradient Evolution: {param_name}")
plt.savefig("grad_evolution.png")
```

### Check for Anomalies

```python
import torch

gradients = torch.load("/tmp/reconstructed/step10_all_gradients.pt")

for name, grad in gradients.items():
    # Check for NaN/Inf
    has_nan = torch.isnan(grad).any().item()
    has_inf = torch.isinf(grad).any().item()
    
    if has_nan or has_inf:
        print(f"⚠️  {name}: NaN={has_nan}, Inf={has_inf}")
    
    # Check for very large/small norms
    norm = grad.norm().item()
    if norm > 1000:
        print(f"⚠️  {name}: Very large norm {norm:.2e}")
    elif norm < 1e-7:
        print(f"⚠️  {name}: Very small norm {norm:.2e}")
```

### Compare Across Runs

```python
import torch

grad1 = torch.load("run1/step10_all_gradients.pt")
grad2 = torch.load("run2/step10_all_gradients.pt")

for name in grad1.keys():
    g1, g2 = grad1[name], grad2[name]
    
    # Relative difference
    rel_diff = (g1 - g2).norm() / g1.norm()
    
    # Cosine similarity
    cos_sim = (g1.flatten() @ g2.flatten()) / (g1.norm() * g2.norm())
    
    print(f"{name:50s} | diff: {rel_diff:.6f} | sim: {cos_sim:.6f}")
```

## Troubleshooting

### Out of Disk Space

```bash
# Check disk usage
du -sh /tmp/gradient_dumping_example/grad_dumper/

# Clean up old steps
rm /tmp/gradient_dumping_example/grad_dumper/rank*_step{0,2,4}_*.pt
```

### Missing Gradient Files

```bash
# Check which steps are available
ls /tmp/gradient_dumping_example/grad_dumper/rank0_step*.pt | \
  sed 's/.*step\([0-9]*\).*/\1/' | sort -u

# Verify all ranks present for a step
ls /tmp/gradient_dumping_example/grad_dumper/rank*_step2_*.pt | wc -l
# Should be: num_parameters × num_ranks
```

### Shape Mismatches

If you see shape mismatch warnings:
- Check that all rank files exist for that step
- Verify training didn't crash mid-step
- Try reconstructing a different step

## See Also

- [GradientDumperCallback source](../../src/olmo_core/train/callbacks/gradient_dumper.py)
- [Reconstruction script](../../src/scripts/reconstruct_gradients.py)
- [Training examples](../../src/examples/llm/)
