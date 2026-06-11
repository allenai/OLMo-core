"""
Thread-local state for activation checkpointing.

Tracks whether execution is currently inside a checkpointed forward or a
recompute, so MoE forwards wrapped in activation checkpointing can avoid
repeating metric side effects when their forward is recomputed in backward.
"""

import threading
from contextlib import contextmanager

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
    return getattr(_CHECKPOINT_FORWARD_STATE, "depth", 0) > 0


@_torch_compile_disable
def is_checkpoint_recomputing() -> bool:
    """Whether execution is inside an activation-checkpoint recompute."""
    if getattr(_CHECKPOINT_RECOMPUTE_STATE, "depth", 0) > 0:
        return True

    # When torch.compile is enabled, checkpoint() requires context_fn entries to
    # be TorchDispatchModes, so some call sites use noop_context_fn instead of
    # checkpoint_recompute_context_fn(). Non-reentrant checkpoint recomputation
    # still runs while autograd is executing a graph task; use that as a fallback
    # so recomputed forwards don't repeat metric side effects. With
    # reduce_overhead/CUDA graph capture this is a best-effort workaround, not a
    # fully proven signal.
    try:
        return torch.is_grad_enabled() and torch._C._current_graph_task_id() != -1
    except Exception:
        return False


@_torch_compile_disable
def is_activation_checkpointing() -> bool:
    return is_checkpoint_forwarding() or is_checkpoint_recomputing()


@_torch_compile_disable
def get_rowwise_checkpoint_state() -> tuple[bool, bool]:
    """
    Return ``(checkpointing_active, save_for_recompute)``: whether activation
    checkpointing is active, and whether outputs should be saved (``True`` on the
    initial forward, ``False`` during recompute so side effects aren't repeated).
    """
    checkpoint_forwarding = is_checkpoint_forwarding()
    checkpoint_recomputing = is_checkpoint_recomputing()
    return checkpoint_forwarding or checkpoint_recomputing, not checkpoint_recomputing


@contextmanager
def checkpoint_forward_context():
    """Mark the enclosed block as a checkpointed forward (re-entrant via depth)."""
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
    """Mark the enclosed block as an activation-checkpoint recompute (re-entrant via depth)."""
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
    return checkpoint_forward_context(), checkpoint_recompute_context()
