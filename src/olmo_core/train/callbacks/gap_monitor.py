import dataclasses
import functools as ft
import math
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.distributed.utils import get_full_tensor, get_local_tensor

from ..common import MetricMergeStrategy, ReduceType
from ..train_module import TransformerTrainModule
from .callback import Callback


@dataclass
class GAPMonitorCallback(Callback):
    """
    Gradient, activation, and parameter (GAP) monitoring callback.

    This callback logs fine-grained statistics on all gradients, activations, and parameters.
    """

    enabled: bool = True

    _handles: Optional[list] = dataclasses.field(default=None, repr=False)
    _local_batch_size_instances: int = dataclasses.field(default=1, repr=False)

    def post_attach(self):
        if not self.enabled:
            return
        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise ValueError(f"{type(self).__name__ } only works with the TransformerTrainModule.")

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
        self._handles = handles  # type: ignore[assignment]

    def pre_step(self, batch: Dict[str, Any]):
        if not self.enabled:
            return

        self._local_batch_size_instances = batch["input_ids"].shape[0]

    def pre_optim_step(self):
        if not self.enabled:
            return

        assert isinstance(self.trainer.train_module, TransformerTrainModule)
        for n, p in self.trainer.train_module.model.named_parameters():
            self.record_tensor_stats(n, p, "param")
            if p.grad is not None:
                self.record_tensor_stats(n, p.grad, "grad")

    @torch._dynamo.disable()
    def forward_hook(self, module: nn.Module, args, output, module_name: str):
        del module, args
        if not self.enabled:
            return

        # Record activation stats.
        if isinstance(output, torch.Tensor):
            self.record_tensor_stats(module_name, output, "activation")
        elif isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    self.record_tensor_stats(module_name, o, "activation")
        else:
            raise RuntimeError(f"unsupported output type {type(output)} for module '{module_name}'")

    def record_tensor_stats(
        self, name: str, tensor: torch.Tensor, kind: Literal["grad", "activation", "param"]
    ):
        if tensor.numel() <= 1:
            return

        tensor = tensor.detach()
        prefix = f"gap/{kind}s"
        if kind == "activation":
            if tensor.ndim <= 1:
                # No point in computing stats for 0-dim or 1-dim activations (like the loss).
                return
            # For activations we'll compute the local stats *per instance* and then average them
            # across the global batch.
            # Technically it might be better to compute global stats directly, but this way is
            # cheaper, much simpler, and probably good enough.
            tensor = get_local_tensor(tensor)
            tensor = tensor.view(tensor.shape[0], -1)
            var, mean = var_mean(tensor, dim=-1)
            # NOTE: to handle gradient accumulation we divide by local batch size (in instances),
            # which is recorded in `self.pre_step()`, as opposed to micro-batch size, and then
            # we use the "sum" merge strategy.
            var = var.float().sum() / self._local_batch_size_instances
            mean = mean.float().sum() / self._local_batch_size_instances
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
            self.trainer.record_metric(f"{prefix}/{name}/mean", mean, reduce_type=None)
            self.trainer.record_metric(f"{prefix}/{name}/var", var, reduce_type=None)

    def close(self):
        self._reset()

    def _reset(self):
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
