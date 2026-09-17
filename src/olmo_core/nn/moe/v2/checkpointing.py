import threading
from contextlib import contextmanager
from typing import Tuple

import torch

_CHECKPOINT_RECOMPUTE_STATE = threading.local()
_CHECKPOINT_FORWARD_STATE = threading.local()

try:
    _torch_compile_disable = torch.compiler.disable
except AttributeError:

    def _torch_compile_disable(fn):
        return fn


@_torch_compile_disable
def is_checkpoint_forwarding() -> bool:
    """Whether the current thread is inside an initial checkpointed forward pass."""
    return getattr(_CHECKPOINT_FORWARD_STATE, "depth", 0) > 0


@_torch_compile_disable
def is_checkpoint_recomputing() -> bool:
    """Whether the current thread is recomputing checkpointed activations."""
    if getattr(_CHECKPOINT_RECOMPUTE_STATE, "depth", 0) > 0:
        return True

    # Compiled checkpoints may use noop_context_fn instead of the thread-local contexts.
    # An active autograd task is a fallback signal that suppresses repeated metric updates;
    # it does not reliably identify recomputation during CUDA graph capture.
    try:
        return torch.is_grad_enabled() and torch._C._current_graph_task_id() != -1
    except Exception:
        return False


@_torch_compile_disable
def is_activation_checkpointing() -> bool:
    """Whether a checkpointed forward or recomputation is active."""
    return is_checkpoint_forwarding() or is_checkpoint_recomputing()


@_torch_compile_disable
def get_rowwise_checkpoint_state() -> Tuple[bool, bool]:
    """Return whether checkpointing is active and whether outputs should be saved."""
    checkpoint_forwarding = is_checkpoint_forwarding()
    checkpoint_recomputing = is_checkpoint_recomputing()
    return checkpoint_forwarding or checkpoint_recomputing, not checkpoint_recomputing


@contextmanager
def checkpoint_forward_context():
    """Track nested initial checkpointed forwards on the current thread."""
    depth = getattr(_CHECKPOINT_FORWARD_STATE, "depth", 0)
    _CHECKPOINT_FORWARD_STATE.depth = depth + 1
    try:
        yield
    finally:
        if depth == 0:
            if hasattr(_CHECKPOINT_FORWARD_STATE, "depth"):
                delattr(_CHECKPOINT_FORWARD_STATE, "depth")
        else:
            _CHECKPOINT_FORWARD_STATE.depth = depth


@contextmanager
def checkpoint_recompute_context():
    """Track nested checkpoint recomputations on the current thread."""
    depth = getattr(_CHECKPOINT_RECOMPUTE_STATE, "depth", 0)
    _CHECKPOINT_RECOMPUTE_STATE.depth = depth + 1
    try:
        yield
    finally:
        if depth == 0:
            if hasattr(_CHECKPOINT_RECOMPUTE_STATE, "depth"):
                delattr(_CHECKPOINT_RECOMPUTE_STATE, "depth")
        else:
            _CHECKPOINT_RECOMPUTE_STATE.depth = depth


def checkpoint_recompute_context_fn():
    """Return forward and recomputation contexts for non-reentrant checkpointing."""
    return checkpoint_forward_context(), checkpoint_recompute_context()
