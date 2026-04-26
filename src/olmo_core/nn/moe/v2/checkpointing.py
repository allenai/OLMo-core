import threading
from contextlib import contextmanager, nullcontext


_CHECKPOINT_RECOMPUTE_STATE = threading.local()


def is_checkpoint_recomputing() -> bool:
    return getattr(_CHECKPOINT_RECOMPUTE_STATE, "depth", 0) > 0


@contextmanager
def checkpoint_recompute_context():
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
    return nullcontext(), checkpoint_recompute_context()
