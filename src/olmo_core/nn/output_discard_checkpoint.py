from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from torch.utils.checkpoint import detach_variable
from torch.utils.cpp_extension import load_inline

from ..doc_utils import beta_feature

__all__ = ["OutputDiscardCheckpoint"]


_SHARE_STORAGE_CPP_PATH = Path(__file__).parent / "_csrc" / "share_storage.cpp"


@dataclass
class _SharedStorageLoader:
    """
    Lazily builds and caches the C++ ``share_storage`` extension, falling back
    to a pure-Python implementation on machines without a working C++ toolchain.

    A single module-level instance (:data:`_SHARED_STORAGE_LOADER`) holds the
    build state so it is shared across all :class:`OutputDiscardCheckpoint`
    objects without leaning on module globals.
    """

    _ext: Any = None
    _build_error: Optional[Exception] = None
    _fallback_warned: bool = False
    _lock: Any = field(default_factory=threading.Lock)

    def share(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if (fn := self._load()) is not None:
            fn(dst, src)
        else:
            self._fallback(dst, src)

    def _load(self) -> Optional[Callable[[torch.Tensor, torch.Tensor], None]]:
        if self._ext is not None:
            return self._ext.share_storage
        if self._build_error is not None:
            return None

        with self._lock:
            if self._ext is not None:
                return self._ext.share_storage
            if self._build_error is not None:
                return None
            try:
                self._ext = load_inline(
                    name="olmo_share_storage_ext",
                    cpp_sources=_SHARE_STORAGE_CPP_PATH.read_text(),
                    functions=["share_storage"],
                    verbose=False,
                )
                return self._ext.share_storage
            except Exception as exc:  # pragma: no cover - environment dependent
                self._build_error = exc
                return None

    def _fallback(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """
        Python fallback for environments where the C++ extension cannot be built.

        Unlike the C++ path -- which mutates ``dst``'s ``StorageImpl`` ``data_ptr``
        in place -- we cannot rebind a Python ``Storage``'s data pointer from
        user-space without going through ``Tensor.set_()``, which would swap
        ``dst``'s ``StorageImpl`` for a new one and leave any autograd-saved views
        of ``dst`` pointing at the old (empty) storage. Instead we resize ``dst``'s
        existing storage in place and copy ``src``'s bytes into it. This preserves
        ``StorageImpl`` identity (so saved views see the refilled data) at the
        cost of an extra allocation + copy during recompute.
        """
        old_version = dst._version
        with torch.no_grad():
            dst_storage = dst.untyped_storage()
            src_storage = src.untyped_storage()
            dst_storage.resize_(src_storage.nbytes())
            dst_storage.copy_(src_storage)

        if hasattr(torch._C, "_autograd") and hasattr(
            torch._C._autograd, "_unsafe_set_version_counter"
        ):
            torch._C._autograd._unsafe_set_version_counter([dst], [old_version])
        elif not self._fallback_warned:  # pragma: no cover - very old torch only
            warnings.warn(
                "OutputDiscardCheckpoint fallback could not access "
                "torch._C._autograd._unsafe_set_version_counter; autograd version "
                "counter errors may occur.",
                stacklevel=2,
            )
            self._fallback_warned = True

        if not self._fallback_warned and self._build_error is not None:
            warnings.warn(
                "OutputDiscardCheckpoint C++ share_storage extension is unavailable; "
                "using Python fallback. Build error was: "
                f"{self._build_error!r}",
                stacklevel=2,
            )
            self._fallback_warned = True


_SHARED_STORAGE_LOADER = _SharedStorageLoader()


def _collect_tensor_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
    """Normalize ``fn`` outputs to a flat tuple of tensors, rejecting non-tensor outputs."""
    items = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    if not all(isinstance(out, torch.Tensor) for out in items):
        raise TypeError(
            "OutputDiscardCheckpoint only supports tensor outputs or tuple/list of tensors."
        )
    return tuple(items)


class _OutputDiscardCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        run_function: Callable[..., Any],
        checkpoint_obj: "OutputDiscardCheckpoint",
        *args: Any,
    ):
        with torch.no_grad():
            outputs = run_function(*args)

        arg_is_tensor = tuple(isinstance(arg, torch.Tensor) for arg in args)
        tensor_args = tuple(
            cast(torch.Tensor, arg) for arg in args if isinstance(arg, torch.Tensor)
        )
        non_tensor_args = tuple(arg for arg in args if not isinstance(arg, torch.Tensor))
        ctx.save_for_backward(*detach_variable(tensor_args))
        ctx.arg_is_tensor = arg_is_tensor
        ctx.non_tensor_args = non_tensor_args
        checkpoint_obj._ctx = ctx

        if isinstance(outputs, list):
            return tuple(outputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        if not hasattr(ctx, "inputs") or not hasattr(ctx, "outputs"):
            raise RuntimeError(
                "Recomputed tensors are not set. "
                "Call discard_output_and_register_recompute() during forward."
            )
        if ctx.inputs is None or ctx.outputs is None:
            raise RuntimeError(
                "Recomputed tensors are missing. "
                "Make sure backward hook triggers before consumers need them."
            )

        outputs = cast(tuple[torch.Tensor, ...], ctx.outputs)
        grad_outputs_tensors = cast(tuple[torch.Tensor, ...], grad_outputs)
        torch.autograd.backward(outputs, grad_outputs_tensors)

        tensor_inputs = cast(tuple[torch.Tensor, ...], ctx.inputs)
        tensor_input_iter = iter(tensor_inputs)
        grads: list[Optional[torch.Tensor]] = []
        for is_tensor in cast(tuple[bool, ...], ctx.arg_is_tensor):
            if is_tensor:
                inp = next(tensor_input_iter)
                grads.append(inp.grad)
            else:
                grads.append(None)

        ctx.outputs = None
        ctx.inputs = None
        return (None, None) + tuple(grads)


@beta_feature
class OutputDiscardCheckpoint:
    """
    A Megatron-style output-discard checkpoint utility.

    - Forward runs under ``no_grad`` via a custom autograd Function.
    - Output storage can be discarded after downstream forward.
    - Backward hook recomputes outputs, then shares storage back into the
      original output tensor objects without triggering autograd version errors.

    Usage:

    .. code-block:: python

        # ``submodule`` is the layer whose activations you want to discard
        # to save memory. ``next_layer`` is the downstream consumer.
        ckpt = OutputDiscardCheckpoint()

        # 1. Forward through the checkpointed submodule under no_grad.
        y = ckpt.checkpoint(submodule, x)

        # 2. Run the downstream forward; it reads ``y`` while its storage
        #    is still allocated.
        z = next_layer(y)

        # 3. Free ``y``'s storage and register a backward hook on a tensor
        #    whose ``register_hook`` fires before any consumer of ``y``
        #    needs it in backward. Typically pass the immediate downstream
        #    output (``z`` here): when autograd computes ``z``'s grad, the
        #    hook recomputes ``submodule`` and rebinds ``y``'s storage in
        #    place so the original tensor object is usable again.
        ckpt.discard_output_and_register_recompute(z)

        # 4. Normal backward. The hook fires automatically.
        z.sum().backward()

    Picking ``hook_tensor``: it must (a) require grad, and (b) sit in the
    autograd graph downstream of ``y`` so that its hook runs before any
    saved-tensor reference to ``y`` is dereferenced during backward. The
    output of the layer that immediately consumes ``y`` is the safe default.

    .. warning::
        This is a beta utility with limitations the recompute does not yet
        handle:

        - **RNG state** is not saved or restored, so the recompute is only
          correct for deterministic checkpointed regions (no dropout or other
          stochastic ops inside ``fn``).
        - **Autocast / AMP context** is not captured, so the recompute runs
          under whatever autocast state is active during backward rather than
          the state used in the original forward.
        - **No-grad inputs**: recompute only restores storage and produces
          grads when ``hook_tensor`` requires grad. If it does not, the
          output storage is discarded but no hook is registered, so this is
          only safe when recompute is arranged by other means.

        Use it only where these assumptions hold.
    """

    def __init__(self):
        self.run_function: Optional[Callable[..., Any]] = None
        self._ctx: Optional[Any] = None
        self.outputs: Optional[tuple[torch.Tensor, ...]] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def checkpoint(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Run ``fn(*args, **kwargs)`` under ``no_grad`` and record the inputs so
        that backward can recompute. Returns the forward outputs (the same
        structure ``fn`` would normally return).

        :param fn: The callable / submodule to checkpoint.
        :param args: Positional arguments forwarded to ``fn``. Tensor args are
            saved for recompute; non-tensor args are stashed verbatim.
        :param kwargs: Keyword arguments forwarded to ``fn`` on both the
            initial call and the recompute.

        :returns: Whatever ``fn`` returns — a single tensor or a tuple/list of
            tensors. Lists are normalized to tuples.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        if kwargs:

            def run_function(*inner_args: Any) -> Any:
                return fn(*inner_args, **kwargs)

        else:
            run_function = fn

        self.run_function = run_function
        outputs = _OutputDiscardCheckpointFunction.apply(run_function, self, *args)
        self.outputs = _collect_tensor_outputs(outputs)
        return outputs

    def discard_output_and_register_recompute(self, hook_tensor: torch.Tensor):
        """
        Free the storage of every output returned by :meth:`checkpoint` and
        register a backward hook on ``hook_tensor`` that will recompute the
        forward pass and rebind the freed storage.

        :param hook_tensor: A tensor downstream of the checkpoint outputs whose
            ``register_hook`` is used to schedule recompute during backward. It
            must require grad. If it does not, storage is discarded but no
            hook is registered — only safe if you have arranged recompute by
            other means.
        """
        if self.outputs is None:
            raise RuntimeError("No checkpoint outputs found. Call checkpoint() first.")

        for output in self.outputs:
            output.untyped_storage().resize_(0)

        if hook_tensor.requires_grad:
            self._hook_handle = hook_tensor.register_hook(self._recompute)

    def _recompute(self, grad: torch.Tensor) -> torch.Tensor:
        if self._ctx is None:
            return grad
        if self.run_function is None:
            raise RuntimeError("Invalid state: missing run_function.")
        if self.outputs is None:
            raise RuntimeError("Invalid state: missing forward outputs.")

        ctx = self._ctx
        saved_tensors = tuple(cast(tuple[torch.Tensor, ...], ctx.saved_tensors))
        tensor_iter = iter(saved_tensors)
        non_tensor_iter = iter(cast(tuple[Any, ...], ctx.non_tensor_args))

        recompute_args: list[Any] = []
        recompute_tensor_inputs: list[torch.Tensor] = []
        for is_tensor in cast(tuple[bool, ...], ctx.arg_is_tensor):
            if is_tensor:
                src = next(tensor_iter)
                inp = src.detach()
                inp.requires_grad_(src.requires_grad)
                recompute_args.append(inp)
                recompute_tensor_inputs.append(inp)
            else:
                recompute_args.append(next(non_tensor_iter))

        with torch.enable_grad():
            recompute_outputs = self.run_function(*recompute_args)
        recompute_outputs_tensors = _collect_tensor_outputs(recompute_outputs)

        if len(recompute_outputs_tensors) != len(self.outputs):
            raise RuntimeError(
                "Recomputed output count does not match original output count: "
                f"{len(recompute_outputs_tensors)} != {len(self.outputs)}"
            )

        for output, recomputation_output in zip(self.outputs, recompute_outputs_tensors):
            _SHARED_STORAGE_LOADER.share(output, recomputation_output)

        ctx.outputs = recompute_outputs_tensors
        ctx.inputs = tuple(recompute_tensor_inputs)

        self.outputs = None
        self._ctx = None
        return grad

    def clear(self):
        """
        Remove any registered backward hook and drop references to the saved
        forward state. Call this if you abandon a checkpoint before backward
        (for example, in eval branches that take a different code path).
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.run_function = None
        self._ctx = None
        self.outputs = None
