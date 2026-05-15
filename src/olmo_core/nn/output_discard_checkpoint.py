from __future__ import annotations

import threading
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, cast

import torch
from torch.utils.checkpoint import detach_variable
from torch.utils.cpp_extension import load_inline

__all__ = ["OutputDiscardCheckpoint"]


_SHARE_STORAGE_CPP = r"""
#include <torch/extension.h>

void share_storage(at::Tensor dst, at::Tensor src) {
    auto* dst_impl = dst.storage().unsafeGetStorageImpl();

    auto* src_storage_ref = new c10::Storage(src.storage());

    void*       data   = src_storage_ref->data_ptr().get();
    size_t      nbytes = src_storage_ref->nbytes();
    c10::Device device = src_storage_ref->device();

    c10::DataPtr shared(
        data,
        static_cast<void*>(src_storage_ref),
        [](void* ctx) { delete static_cast<c10::Storage*>(ctx); },
        device);

    dst_impl->set_data_ptr(std::move(shared));
    dst_impl->set_nbytes(nbytes);
}
"""

_share_storage_ext = None
_share_storage_lock = threading.Lock()
_share_storage_build_error: Optional[Exception] = None
_share_storage_fallback_warned = False


def _get_share_storage() -> Optional[Callable[[torch.Tensor, torch.Tensor], None]]:
    global _share_storage_ext
    global _share_storage_build_error
    if _share_storage_ext is not None:
        return _share_storage_ext.share_storage
    if _share_storage_build_error is not None:
        return None

    with _share_storage_lock:
        if _share_storage_ext is not None:
            return _share_storage_ext.share_storage
        if _share_storage_build_error is not None:
            return None
        try:
            _share_storage_ext = load_inline(
                name="olmo_share_storage_ext",
                cpp_sources=_SHARE_STORAGE_CPP,
                functions=["share_storage"],
                verbose=False,
            )
            return _share_storage_ext.share_storage
        except Exception as exc:  # pragma: no cover - environment dependent
            _share_storage_build_error = exc
            return None


def _fallback_share_storage(dst: torch.Tensor, src: torch.Tensor):
    """
    Python fallback for environments where the C++ extension cannot be built.
    """
    global _share_storage_fallback_warned

    old_version = dst._version
    with torch.no_grad():
        dst.set_(
            src.untyped_storage(),
            src.storage_offset(),
            src.size(),
            src.stride(),
        )

    # Keep version counter stable to mirror Megatron's low-level behavior.
    if hasattr(torch._C, "_autograd") and hasattr(
        torch._C._autograd, "_unsafe_set_version_counter"
    ):
        torch._C._autograd._unsafe_set_version_counter([dst], [old_version])
    elif not _share_storage_fallback_warned:  # pragma: no cover - very old torch only
        warnings.warn(
            "OutputDiscardCheckpoint fallback could not access "
            "torch._C._autograd._unsafe_set_version_counter; autograd version "
            "counter errors may occur.",
            stacklevel=2,
        )
        _share_storage_fallback_warned = True

    if not _share_storage_fallback_warned and _share_storage_build_error is not None:
        warnings.warn(
            "OutputDiscardCheckpoint C++ share_storage extension is unavailable; "
            "using Python fallback. Build error was: "
            f"{_share_storage_build_error!r}",
            stacklevel=2,
        )
        _share_storage_fallback_warned = True


def _share_storage(dst: torch.Tensor, src: torch.Tensor):
    share_storage = _get_share_storage()
    if share_storage is not None:
        share_storage(dst, src)
    else:
        _fallback_share_storage(dst, src)


def _collect_tensor_outputs(outputs: Any) -> Tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (tuple, list)):
        if not all(isinstance(out, torch.Tensor) for out in outputs):
            raise TypeError(
                "OutputDiscardCheckpoint only supports tensor outputs or tuple/list of tensors."
            )
        return tuple(cast(Sequence[torch.Tensor], outputs))
    raise TypeError(
        "OutputDiscardCheckpoint only supports tensor outputs or tuple/list of tensors."
    )


def _detach_but_keep_requires_grad(x: torch.Tensor) -> torch.Tensor:
    out = x.detach()
    out.requires_grad_(x.requires_grad)
    return out


class _OutputDiscardCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
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

        outputs = cast(Tuple[torch.Tensor, ...], ctx.outputs)
        grad_outputs_tensors = cast(Tuple[torch.Tensor, ...], grad_outputs)
        torch.autograd.backward(outputs, grad_outputs_tensors)

        tensor_inputs = cast(Tuple[torch.Tensor, ...], ctx.inputs)
        tensor_input_iter = iter(tensor_inputs)
        grads: list[Optional[torch.Tensor]] = []
        for is_tensor in cast(Tuple[bool, ...], ctx.arg_is_tensor):
            if is_tensor:
                inp = next(tensor_input_iter)
                grads.append(inp.grad)
            else:
                grads.append(None)

        ctx.outputs = None
        ctx.inputs = None
        return (None, None) + tuple(grads)


class OutputDiscardCheckpoint:
    """
    A Megatron-style output-discard checkpoint utility.

    - Forward runs under `no_grad` via a custom autograd Function.
    - Output storage can be discarded after downstream forward.
    - Backward hook recomputes outputs, then shares storage back into the
      original output tensor objects without triggering autograd version errors.
    """

    def __init__(self):
        self.run_function: Optional[Callable[..., Any]] = None
        self._ctx: Optional[Any] = None
        self.outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def checkpoint(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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
        saved_tensors = tuple(cast(Tuple[torch.Tensor, ...], ctx.saved_tensors))
        tensor_iter = iter(saved_tensors)
        non_tensor_iter = iter(cast(Tuple[Any, ...], ctx.non_tensor_args))

        recompute_args: list[Any] = []
        recompute_tensor_inputs: list[torch.Tensor] = []
        for is_tensor in cast(Tuple[bool, ...], ctx.arg_is_tensor):
            if is_tensor:
                src = next(tensor_iter)
                inp = _detach_but_keep_requires_grad(src)
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
            _share_storage(output, recomputation_output)

        ctx.outputs = recompute_outputs_tensors
        ctx.inputs = tuple(recompute_tensor_inputs)

        self.outputs = None
        self._ctx = None
        return grad

    def clear(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.run_function = None
        self._ctx = None
        self.outputs = None
