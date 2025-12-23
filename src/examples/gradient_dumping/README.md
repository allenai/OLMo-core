# Gradient Dumping

This example shows how to capture gradients during training and save them for inspection.

## Quick Start

```bash
bash src/examples/gradient_dumping/run.sh
```

The script runs `src/examples/gradient_dumping/train.py`, which enables the gradient dumper every two steps on a multi-GPU setup.

## Dump Modes

- `full_gradients/` (`save_first_n=None`, default): writes a distributed checkpoint for every parameter using `save_state_dict`. Each rank stores its shard automatically.
- `sampled_gradients/` (`save_first_n=N`): gathers the full tensor, keeps the first `N` elements along dim 0, and writes a `.safetensors` file per parameter on rank 0. This is useful for quick sanity checks without saving the full tensor, but requires an all-gather, unlike the full_gradients mode. 

## Inspecting Dumps

### Full Gradients

```python
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

full_dir = "/tmp/gradient_dumping_example/gradient_dumper/step2/full_gradients"
grad = next(load_keys(full_dir, ["embedding.weight"]))
print(grad.shape, grad.norm())
```

### Sampled Gradients

```python
from safetensors import safe_open

sampled_dir = "/tmp/gradient_dumping_example/gradient_dumper/step2/sampled_gradients"
with safe_open(f"{sampled_dir}/embedding.weight_first100.safetensors", framework="pt") as f:
    grad = f.get_tensor("gradient")
    print(grad.shape, grad.norm())
```

`load_keys()` must run outside distributed execution (plain Python, not `torchrun`), because it reconstructs the full tensor on a single process.

## Directory Layout

```
/tmp/my-run/gradient_dumper/
  step0/
    full_gradients/
      .metadata
      __0_0.distcp
      ...
  step2/
    sampled_gradients/
      embedding.weight_first100.safetensors
      ...
```

`full_gradients/` follows PyTorch’s distributed checkpoint format; `sampled_gradients/` contains sliced tensors stored with safetensors.

## Enable in Your Training Script

```bash
--trainer.callbacks.grad_dump.enabled=true \
--trainer.callbacks.grad_dump.start_step=0 \
--trainer.callbacks.grad_dump.step_interval=10 \
--trainer.callbacks.grad_dump.end_step=100 \
--trainer.callbacks.grad_dump.save_first_n=100  # optional
```

Key options:
- `start_step`, `end_step`, `step_interval`: control when dumps happen.
- `save_first_n`: switch between `full_gradients/` (None) and `sampled_gradients/` (value > 0).

## See Also

- `src/olmo_core/train/callbacks/gradient_dumper.py`
- `src/olmo_core/distributed/checkpoint/`
