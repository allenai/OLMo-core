# Gradient Dumping

This example demonstrates how to dump gradients during distributed FSDP/HSDP training.

## Quick Start

Run the example training script:

```bash
bash src/examples/gradient_dumping/run.sh
```

This trains a small model on 2 GPUs with FSDP and dumps gradients at steps 0, 2, 4, 6, 8, 10.

## Overview

When training large models with FSDP (Fully Sharded Data Parallel) or HSDP (Hybrid Sharded Data Parallel), each GPU holds only a **shard** of each parameter's gradient. The `GradientDumperCallback` provides two ways to save gradients:

**1. Sharded format** (`save_first_n=None`, default):
- Saves full gradients using distributed checkpoint format
- Each rank saves its shard automatically
- Use for complete gradient analysis

**2. Sampled format** (`save_first_n=N`):
- Saves only first N elements per parameter
- All-gathers full gradient then slices
- Use for lightweight monitoring

## Loading Gradients

### Sharded Format (Full Gradients)

**Load specific gradients:**

```python
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

# Discover what gradients are available
sharded_dir = "/tmp/gradient_dumping_example/gradient_dumper/step2/sharded"
metadata = get_checkpoint_metadata(sharded_dir)
param_names = list(metadata.state_dict_metadata.keys())
print(f"Available parameters: {param_names}")

# Load specific gradient
grad = next(load_keys(sharded_dir, ["embedding.weight"]))
print(f"Shape: {grad.shape}, Norm: {grad.norm():.6e}")
```

**Load all gradients:**

```python
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

sharded_dir = "/tmp/gradient_dumping_example/gradient_dumper/step2/sharded"

# Get list of all parameters
metadata = get_checkpoint_metadata(sharded_dir)
param_names = list(metadata.state_dict_metadata.keys())

# Load all gradients
grad_dict = {}
for param_name, tensor in zip(param_names, load_keys(sharded_dir, param_names)):
    grad_dict[param_name] = tensor

# Analyze
for name, grad in sorted(grad_dict.items(), key=lambda x: x[1].norm(), reverse=True)[:10]:
    print(f"{name}: {grad.norm():.6e}")
```

**Important:** `load_keys()` must be called in a **non-distributed context** (regular Python script, not `torchrun`). It reconstructs full tensors from shards automatically.

### Sampled Format (First N Elements)

**Load specific gradient:**

```python
from safetensors import safe_open

sampled_dir = "/tmp/gradient_dumping_example/gradient_dumper/step2/sampled"
with safe_open(f"{sampled_dir}/embedding.weight_first100.safetensors", framework="pt") as f:
    grad = f.get_tensor("gradient")
    print(f"Shape: {grad.shape}, Norm: {grad.norm():.6e}")
```

**Load all sampled gradients:**

```python
from pathlib import Path
from safetensors.torch import load_file

sampled_dir = Path("/tmp/gradient_dumping_example/gradient_dumper/step2/sampled")

grad_dict = {}
for safetensor_file in sampled_dir.glob("*.safetensors"):
    # Parse filename: "embedding.weight_first100.safetensors" → "embedding.weight"
    param_name = safetensor_file.stem.rsplit("_first", 1)[0]
    data = load_file(str(safetensor_file))
    grad_dict[param_name] = data["gradient"]

print(f"Loaded {len(grad_dict)} sampled gradients")
```

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
- `save_first_n`: Save only first N elements of each gradient along dim 0 (default: `None` = save all shards)

### File Structure

After training, you'll find:

**When `save_first_n=None` (sharded gradients):**
```
/tmp/my-run/gradient_dumper/
  step0/
    sharded/              # Distributed checkpoint subdirectory
      .metadata           # Distributed checkpoint metadata
      __0_0.distcp        # Shard files (PyTorch distributed checkpoint format)
      __1_0.distcp
      ...
  step2/
    sharded/
      .metadata
      __0_0.distcp
      ...
```

The gradients are saved using PyTorch's distributed checkpoint format, which handles all the DTensor metadata automatically.

**When `save_first_n=1000` (first N elements only):**
```
/tmp/my-run/gradient_dumper/
  step0/
    sampled/              # Sampled gradients subdirectory
      embedding.weight_first100.safetensors    # Only first 100 elements (capped to dim size)
      linear.weight_first8.safetensors         # Only first 8 elements
      ...
  step2/
    sampled/
      ...
```

Only rank 0 saves files in this mode, and gradients are sliced along dimension 0.

## How It Works

### Two Modes

**Sharded Mode (`save_first_n=None`):**
- Uses `olmo_core.distributed.checkpoint.save_state_dict()` to save gradients
- Each rank saves its local shard automatically
- DTensor metadata (placements, shapes) is handled by PyTorch's distributed checkpoint
- Load with `load_state_dict()` or `load_keys()` - works across different world sizes

**Sampled Mode (`save_first_n=N`):**
- Uses `olmo_core.distributed.utils.get_full_tensor()` to all-gather full gradients
- Slices first N elements along dimension 0
- Only rank 0 saves (no distributed coordination needed)
- Useful for quick sanity checks without full gradient overhead

### FSDP Sharding Background

FSDP splits **each parameter** across all GPUs, not different layers to different GPUs.

**Example: 4 GPUs, parameter [1024, 1024]**

```
GPU 0: rows 0-255   → [256, 1024]  ─┐
GPU 1: rows 256-511 → [256, 1024]   ├─ Concatenate → [1024, 1024]
GPU 2: rows 512-767 → [256, 1024]   │
GPU 3: rows 768-1023→ [256, 1024]  ─┘
```

Every GPU has a piece of **every layer**. The distributed checkpoint format handles this automatically.

### Loading Errors

If you have issues loading gradients:
- Ensure the checkpoint directory has a `.metadata` file
- Check that training didn't crash mid-step
- Try loading a different step
- Use `load_keys()` to load individual gradients instead of all at once

## See Also

- [GradientDumperCallback source](../../src/olmo_core/train/callbacks/gradient_dumper.py)
- [Distributed checkpoint utilities](../../src/olmo_core/distributed/checkpoint/)
