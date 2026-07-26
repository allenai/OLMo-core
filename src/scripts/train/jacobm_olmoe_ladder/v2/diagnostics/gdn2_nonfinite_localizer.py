"""Read-only hooks for localizing a deterministic GDN2 non-finite failure.

The callback is deliberately opt-in and attaches its hooks only shortly before
the expected failure.  This keeps the long checkpoint replay on the ordinary
compiled path while still checking local (pre-collective) activations and
activation gradients at the failure boundary.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from olmo_core.distributed.utils import get_local_tensor, get_rank
from olmo_core.nn.attention.gdn2 import GatedDeltaNet2
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.train.callbacks import Callback

log = logging.getLogger(__name__)


def _iter_tensors(value: Any, prefix: str = "output") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}.{idx}")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            yield from _iter_tensors(getattr(value, item.name), f"{prefix}.{item.name}")


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return get_local_tensor(tensor.detach())


@torch.no_grad()
def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    local = _local(tensor)
    finite = torch.isfinite(local)
    all_finite = bool(finite.all().item())
    finite_values = local[finite]
    first_bad_flat: int | None = None
    if not all_finite:
        first_bad_flat = int((~finite).reshape(-1).nonzero()[0].item())
    return {
        "shape": tuple(local.shape),
        "dtype": str(local.dtype),
        "numel": local.numel(),
        "all_finite": all_finite,
        "nan_count": int(torch.isnan(local).sum().item()),
        "posinf_count": int(torch.isposinf(local).sum().item()),
        "neginf_count": int(torch.isneginf(local).sum().item()),
        "finite_abs_max": (
            float(finite_values.abs().max().item()) if finite_values.numel() else None
        ),
        "finite_mean": (
            float(finite_values.float().mean().item()) if finite_values.numel() else None
        ),
        "first_bad_flat": first_bad_flat,
    }


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _local(value).cpu().clone()
    if isinstance(value, Mapping):
        return {str(key): _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


@dataclass
class GDN2NonfiniteLocalizerCallback(Callback):
    """Locate and capture the first local non-finite module boundary per rank."""

    start_step: int
    end_step: int
    dump_root: str
    run_id: str
    _handles: list[torch.utils.hooks.RemovableHandle] = field(
        default_factory=list, init=False, repr=False
    )
    _hooks_attached: bool = field(default=False, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _captured: bool = field(default=False, init=False, repr=False)
    _batch: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.start_step < 1 or self.end_step < self.start_step:
            raise ValueError(f"invalid localization window {self.start_step}..{self.end_step}")

    @property
    def output_dir(self) -> Path:
        return Path(self.dump_root) / self.run_id

    def pre_train(self) -> None:
        # Checkpoint loading happens before pre_train. Validate parameters and
        # optimizer-visible model weights before replaying any data.
        bad_parameters: list[tuple[str, dict[str, Any]]] = []
        for name, parameter in self.trainer.train_module.model.named_parameters():
            summary = _tensor_summary(parameter)
            if not summary["all_finite"]:
                bad_parameters.append((name, summary))
        if bad_parameters:
            raise RuntimeError(f"checkpoint contains non-finite parameters: {bad_parameters[:8]}")
        log.info(
            "GDN2 localizer checkpoint audit passed on rank %s; hooks activate at steps %s..%s",
            get_rank(),
            self.start_step,
            self.end_step,
        )

    def pre_step(self, batch: dict[str, Any]) -> None:
        self._active = self.start_step <= self.step <= self.end_step
        self._batch = batch if self._active else None
        if self._active and not self._hooks_attached:
            self._attach_hooks()
            log.info(
                "GDN2 localizer attached %s hooks on rank %s at step %s",
                len(self._handles),
                get_rank(),
                self.step,
            )

    def post_step(self) -> None:
        self._batch = None
        if self.step >= self.end_step:
            self._active = False

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooks_attached = False
        self._batch = None

    def _attach_hooks(self) -> None:
        if self._hooks_attached:
            return
        model = self.trainer.train_module.model
        selected = 0
        for name, module in model.named_modules():
            if not self._should_monitor(name, module):
                continue
            selected += 1
            self._handles.append(
                module.register_forward_hook(
                    self._forward_hook(name),
                    with_kwargs=True,
                )
            )
            self._handles.append(module.register_full_backward_pre_hook(self._backward_hook(name)))
        if selected == 0:
            raise RuntimeError("GDN2 localizer found no modules to monitor")
        self._hooks_attached = True

    @staticmethod
    def _should_monitor(name: str, module: nn.Module) -> bool:
        if name == "":
            return True
        if isinstance(module, (GatedDeltaNet2, OLMoDDPTransformerBlock)):
            return True
        return name.endswith("lm_head")

    def _forward_hook(self, module_name: str):
        @torch._dynamo.disable(reason="diagnostic local finite check")
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any):
            if not self._active:
                return
            self._inspect(
                phase="forward",
                module_name=module_name or "<model>",
                module=module,
                values=output,
                args=args,
                kwargs=kwargs,
            )

        return hook

    def _backward_hook(self, module_name: str):
        @torch._dynamo.disable(reason="diagnostic local finite check")
        def hook(module: nn.Module, grad_output: tuple[Any, ...]):
            if not self._active:
                return
            self._inspect(
                phase="backward",
                module_name=module_name or "<model>",
                module=module,
                values=grad_output,
                args=(),
                kwargs={},
            )

        return hook

    def _inspect(
        self,
        *,
        phase: str,
        module_name: str,
        module: nn.Module,
        values: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        bad: list[tuple[str, dict[str, Any], torch.Tensor]] = []
        for tensor_name, tensor in _iter_tensors(values):
            summary = _tensor_summary(tensor)
            if not summary["all_finite"]:
                bad.append((tensor_name, summary, tensor))
        if not bad:
            return

        summaries = [(name, summary) for name, summary, _ in bad]
        log.error(
            "LOCAL_NONFINITE rank=%s step=%s phase=%s module=%s tensors=%s",
            get_rank(),
            self.step,
            phase,
            module_name,
            summaries,
        )
        if self._captured:
            return
        self._captured = True
        self._write_capture(
            phase=phase,
            module_name=module_name,
            module=module,
            args=args,
            kwargs=kwargs,
            bad=bad,
        )

    @torch.no_grad()
    def _write_capture(
        self,
        *,
        phase: str,
        module_name: str,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        bad: list[tuple[str, dict[str, Any], torch.Tensor]],
    ) -> None:
        rank = get_rank()
        rank_dir = self.output_dir / f"rank{rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        first_bad_tensor = _local(bad[0][2])
        bad_batch_idx = 0
        if first_bad_tensor.ndim >= 1 and first_bad_tensor.shape[0] > 1:
            per_batch_finite = (
                torch.isfinite(first_bad_tensor).reshape(first_bad_tensor.shape[0], -1).all(dim=1)
            )
            bad_batch_idx = int((~per_batch_finite).nonzero()[0].item())

        payload: dict[str, Any] = {
            "kind": "gdn2_nonfinite_localization",
            "rank": rank,
            "local_rank": int(os.getenv("LOCAL_RANK", "0")),
            "world_size": int(os.getenv("WORLD_SIZE", "1")),
            "step": self.step,
            "phase": phase,
            "module_name": module_name,
            "module_type": type(module).__name__,
            "bad_tensors": [(name, summary) for name, summary, _ in bad],
            "bad_batch_idx": bad_batch_idx,
            "bad_output": _cpu_copy(
                first_bad_tensor[bad_batch_idx : bad_batch_idx + 1]
                if first_bad_tensor.ndim >= 1
                else first_bad_tensor
            ),
            "batch": _cpu_copy(self._batch) if self._batch is not None else None,
        }
        if isinstance(module, GatedDeltaNet2):
            x = args[0] if args and isinstance(args[0], torch.Tensor) else None
            payload["gdn2_config"] = {
                "d_model": module.d_model,
                "n_heads": module.n_heads,
                "n_v_heads": module.n_v_heads,
                "head_dim": module.head_dim,
                "expand_v": module.expand_v,
                "allow_neg_eigval": module.allow_neg_eigval,
                "conv_size": module.conv_size,
                "disable_recompute": module.disable_recompute,
                "dtype": str(module.w_q.weight.dtype),
            }
            payload["module_state"] = {
                name: _cpu_copy(tensor) for name, tensor in module.state_dict().items()
            }
            if x is not None:
                payload["module_input"] = _cpu_copy(_local(x)[bad_batch_idx : bad_batch_idx + 1])
            cu_doc_lens = kwargs.get("cu_doc_lens")
            payload["cu_doc_lens"] = None if cu_doc_lens is None else _cpu_copy(cu_doc_lens)

        path = rank_dir / f"step{self.step:06d}_{phase}_{module_name.replace('.', '_')}.pt"
        torch.save(payload, path)
        log.error("LOCAL_NONFINITE_CAPTURE rank=%s path=%s", rank, path)
