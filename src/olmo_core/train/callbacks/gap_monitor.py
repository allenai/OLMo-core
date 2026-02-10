import dataclasses
import functools as ft
import logging
import math
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import get_full_tensor, get_local_tensor, get_rank
from olmo_core.utils import gc_cuda

from ..common import MetricMergeStrategy, ReduceType
from ..train_module import TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GAPMonitorCallback(Callback):
    """
    Gradient, activation, and parameter (GAP) monitoring callback.

    This callback logs fine-grained statistics on all gradients, activations, and parameters.

    It can also dump raw gradient tensors to disk for offline analysis. Set ``dump_gradients=True``
    and configure the ``dump_gradients_*`` fields to control when and how gradients are saved.
    """

    enabled: bool = True
    interval: int = 1
    """How often (in steps) to measure statistics. Default is every step."""

    dump_gradients: bool = False
    dump_gradients_start_step: int = 0
    dump_gradients_end_step: Optional[int] = None
    dump_gradients_step_interval: int = 1
    dump_gradients_save_first_n: Optional[int] = None

    _handles: Optional[list] = dataclasses.field(default=None, repr=False)
    _local_batch_size_instances: int = dataclasses.field(default=1, repr=False)
    _dry_run_complete: bool = dataclasses.field(default=False, repr=False)

    def __post_init__(self):
        if self.dump_gradients_step_interval <= 0:
            raise ValueError(
                f"dump_gradients_step_interval must be positive, got {self.dump_gradients_step_interval}"
            )
        if self.dump_gradients_save_first_n is not None and self.dump_gradients_save_first_n <= 0:
            raise ValueError(
                f"dump_gradients_save_first_n must be positive, got {self.dump_gradients_save_first_n}"
            )
        if self.dump_gradients and not self.enabled:
            log.warning(
                "dump_gradients=True has no effect when enabled=False. "
                "Set enabled=True to enable gradient dumping."
            )

    def post_attach(self):
        if not self.enabled:
            return
        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise ValueError(f"{type(self).__name__} only works with the TransformerTrainModule.")

    def pre_train(self):
        if not self.enabled:
            return

        assert isinstance(self.trainer.train_module, TransformerTrainModule)
        self._reset()
        handles: List[torch.utils.hooks.RemovableHandle] = []
        for n, m in self.trainer.train_module.model.named_modules():
            m = typing.cast(nn.Module, m)
            if n == "":
                continue
            # Register forward hook to monitor activations.
            h = m.register_forward_hook(ft.partial(self.forward_hook, module_name=n))
            handles.append(h)
            # Register backward pre-hook to monitor gradients wrt activations.
            h = m.register_full_backward_pre_hook(ft.partial(self.backward_hook, module_name=n))
            handles.append(h)
        self._handles = handles  # type: ignore[assignment]

    def pre_step(self, batch: Dict[str, Any]):
        if not self.enabled:
            return

        self._dry_run_complete = True
        self._local_batch_size_instances = batch["input_ids"].shape[0]

    def pre_optim_step(self):
        if not self.enabled:
            return

        assert isinstance(self.trainer.train_module, TransformerTrainModule)
        for n, p in self.trainer.train_module.model.named_parameters():
            self.record_tensor_stats(n, p, "param")
            if p.grad is not None:
                self.record_tensor_stats(n, p.grad, "grad")

        if self.dump_gradients:
            self._dump_gradients()

    @torch._dynamo.disable()
    def forward_hook(self, module: nn.Module, args, output, module_name: str):
        del module, args
        if not self.enabled:
            return

        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, torch.Tensor):
            self.record_tensor_stats(module_name, output, "activation")
        elif output is not None:
            raise RuntimeError(f"unsupported output type {type(output)} for module '{module_name}'")

    @torch._dynamo.disable()
    def backward_hook(self, module: nn.Module, grad_output, module_name: str):
        del module
        if not self.enabled:
            return

        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]

        if isinstance(grad_output, torch.Tensor):
            self.record_tensor_stats(module_name, grad_output, "activation_grad")
        elif grad_output is not None:
            raise RuntimeError(
                f"unsupported grad_output type {type(grad_output)} for module '{module_name}'"
            )

    @torch.no_grad()
    def record_tensor_stats(
        self,
        name: str,
        tensor: torch.Tensor,
        kind: Literal["grad", "activation", "activation_grad", "param"],
    ):
        if self.step % self.interval != 0:
            return
        if tensor.numel() <= 1:
            return

        tensor = tensor.detach()
        prefix = f"gap/{kind}s"
        if kind in ("activation", "activation_grad"):
            if tensor.ndim <= 1:
                # No point in computing stats for 0-dim or 1-dim activations (like the loss).
                return
            # For activations/output-grads we'll compute the local stats *per instance* and then average them
            # across the global batch.
            # Technically it might be better to compute global stats directly, but this way is
            # cheaper, much simpler, and probably good enough.
            tensor = get_local_tensor(tensor)
            tensor = tensor.reshape(tensor.shape[0], -1)
            max_ = tensor.abs().max()
            var, mean = var_mean(tensor, dim=-1)
            # NOTE: to handle gradient accumulation we divide by local batch size (in instances),
            # which is recorded in `self.pre_step()`, as opposed to micro-batch size, and then
            # we use the "sum" merge strategy.
            var = var.float().sum() / self._local_batch_size_instances
            mean = mean.float().sum() / self._local_batch_size_instances
            if self._dry_run_complete:
                self.trainer.record_metric(
                    f"{prefix}/{name}/max",
                    max_,
                    reduce_type=ReduceType.max,
                    merge_strategy=MetricMergeStrategy.max,
                )
                self.trainer.record_metric(
                    f"{prefix}/{name}/mean",
                    mean,
                    reduce_type=ReduceType.mean,
                    merge_strategy=MetricMergeStrategy.sum,
                )
                self.trainer.record_metric(
                    f"{prefix}/{name}/var",
                    var,
                    reduce_type=ReduceType.mean,
                    merge_strategy=MetricMergeStrategy.sum,
                )
        else:
            var, mean = var_mean(tensor)
            local_tensor = get_local_tensor(tensor)
            if local_tensor.numel() > 0:
                local_max = local_tensor.abs().max()
            else:
                # Use 0.0 as sentinel value for empty tensors in max reduction.
                # Since we're taking abs(), all actual values are >= 0, so 0.0
                # won't affect the max reduction when other processes have non-empty tensors.
                local_max = torch.zeros([], device=tensor.device, dtype=tensor.dtype)
            if self._dry_run_complete:
                self.trainer.record_metric(
                    f"{prefix}/{name}/max", local_max, reduce_type=ReduceType.max
                )
                self.trainer.record_metric(f"{prefix}/{name}/mean", mean, reduce_type=None)
                self.trainer.record_metric(f"{prefix}/{name}/var", var, reduce_type=None)

    def _dump_gradients(self):
        """Save gradient tensors to disk based on dump_gradients_* configuration."""
        if self.step < self.dump_gradients_start_step:
            return

        if self.dump_gradients_end_step is not None and self.step > self.dump_gradients_end_step:
            return

        if (self.step - self.dump_gradients_start_step) % self.dump_gradients_step_interval != 0:
            return

        output_dir = self.trainer.work_dir / "gradients"
        output_dir.mkdir(exist_ok=True, parents=True)

        step_dir = output_dir / f"step{self.step}"
        step_dir.mkdir(exist_ok=True, parents=True)

        assert hasattr(self.trainer.train_module, "model")
        model = getattr(self.trainer.train_module, "model")

        if self.dump_gradients_save_first_n is None:
            # Save full gradients using distributed checkpoint
            full_grads_dir = step_dir / "full_gradients"

            grad_dict = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_dict[name] = p.grad.detach()

            log.info(f"Saving {len(grad_dict)} gradient tensors for step {self.step}...")
            save_state_dict(
                full_grads_dir,
                grad_dict,
                save_overwrite=True,
            )
            log.info(f"Saved full gradients for step {self.step} to '{full_grads_dir}'")
        else:
            sampled_gradients_dir = step_dir / "sampled_gradients"

            if get_rank() == 0:
                sampled_gradients_dir.mkdir(exist_ok=True, parents=True)

            for name, p in model.named_parameters():
                if p.grad is not None:
                    full_grad = get_full_tensor(p.grad.detach())

                    if get_rank() == 0:
                        full_grad = full_grad.cpu()

                        dim_size = full_grad.shape[0]
                        actual_n = min(self.dump_gradients_save_first_n, dim_size)
                        if actual_n < self.dump_gradients_save_first_n:
                            log.warning(
                                f"Parameter '{name}': dump_gradients_save_first_n={self.dump_gradients_save_first_n} exceeds "
                                f"dimension size {dim_size}, capping to {actual_n}"
                            )

                        sliced_grad = full_grad.narrow(0, 0, actual_n)
                        sliced_filename = f"{name}_first{actual_n}.safetensors"
                        sliced_filepath = sampled_gradients_dir / sliced_filename
                        save_file({"gradient": sliced_grad}, str(sliced_filepath))
                        log.info(f"Saved first {actual_n} of '{name}' to '{sliced_filepath}'")

                    del full_grad
            if get_rank() == 0:
                log.info(f"Saved sampled gradients for step {self.step} to {sampled_gradients_dir}")

        if get_rank() == 0:
            rel_step_dir = step_dir.relative_to(self.trainer.work_dir)
            target_dir = self.trainer.persist_working_subdir(rel_step_dir)
            log.info(f"Gradients for step {self.step} saved to '{target_dir}'")

        gc_cuda()

    def close(self):
        self._reset()

    def _reset(self):
        self._dry_run_complete = False
        if self._handles is not None:
            for h in self._handles:
                h.remove()
            self._handles = None


def var_mean(tensor: torch.Tensor, dim: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(tensor, DTensor):
        return torch.var_mean(tensor, dim=dim)
    else:
        # NOTE: 'torch.var_mean()' not implemented for DTensor.
        numel = tensor.numel() if dim is None else tensor.size(dim)
        mean = get_full_tensor(tensor.mean(dim=dim))
        stdd = get_full_tensor(torch.linalg.vector_norm(tensor - mean, dim=dim)) / math.sqrt(
            max(1, numel - 1)
        )
        var = stdd**2
        return var, mean
