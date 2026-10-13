"""
Every runtime environment-variable toggle read by :mod:`olmo_core`, in one place.

Code reads these through the constants below (never a string literal), so ``git grep`` on a
constant finds every reader. A toggle that has a config field is only an *override* of that field;
prefer the field in new code.

Generation / landmark decode
----------------------------

``OLMO_LANDMARK_SPARSE_DECODE`` (default: unset -> use the config field)
    Override of :data:`~olmo_core.generate.generation_module.config.GenerationConfig.landmark_sparse_decode`.
    ``0``/``false``/``no``/``off``/empty disables the sparse landmark decode path, anything else
    enables it. When set it wins over the config field.

``OLMO_LANDMARK_FLAT_SOFTMAX`` (default: unset -> off)
    Override of
    :data:`~olmo_core.generate.generation_module.config.GenerationConfig.landmark_flat_softmax`.
    Any non-empty value turns the flat-softmax decode ablation on (it can only enable, not
    disable, the config field).

``OLMO_FLA_LENGTH_AUTOTUNE`` (default: unset -> autotune frozen)
    ``1`` keeps FLA's per-prompt-length Triton re-autotuning, which the generation module
    otherwise freezes (see :mod:`olmo_core.nn.attention.fla_autotune`).

Landmark attention kernels
--------------------------

``OLMO_LANDMARK_SPARSE_KERNEL`` (default: ``1``)
    ``0`` forces the eager torch fallback of the sparse-landmark attention core instead of the
    Triton kernel (:mod:`olmo_core.nn.attention.landmark_sparse`).

``OLMO_LM_FAST_FWD_WARPS`` / ``OLMO_LM_FAST_FWD_STAGES`` (default: per head_dim, 4/3 at
head_dim <= 128, 8/2 above)
    Triton ``num_warps`` / ``num_stages`` of the fused landmark forward kernels
    (``landmark_fast``, ``landmark_compressive``, ``landmark_compressive_gqa_kernel``,
    ``landmark_multi_compressive``). ``STAGES=1`` fits block_size=256 into shared memory.

``OLMO_LM_FAST_BWD_WARPS`` / ``OLMO_LM_FAST_BWD_STAGES`` (default: per head_dim, 4/2 at
head_dim <= 128, 8/1 above)
    The same for the backward kernels.

KV routing (:mod:`olmo_core.nn.attention.kv_route`)
---------------------------------------------------

``OLMO_KV_ROUTE_BLOCKMASK`` (legacy name ``KV_ROUTE_BLOCKMASK``; default: ``block``)
    ``dense`` builds the flex block mask with the reference ``create_block_mask`` (materialises
    a dense mask; OOMs at long T) instead of the compacted block builder.

``OLMO_KV_ROUTE_DEBUG`` (legacy name ``KV_ROUTE_DEBUG``; default: unset)
    Memory-diagnostic ablations: ``no_router`` keeps every token with no router graph,
    ``no_holder`` runs the router but never accumulates into the holder.

Landmark gate analysis logging (:mod:`olmo_core.nn.attention.landmark_gate_analysis`)
-------------------------------------------------------------------------------------

``OLMO_LANDMARK_GATE_LOG`` (default: unset -> off)
    Output JSONL path; its presence enables per-example gate logging.

``OLMO_GATE_LOG_ALL`` (default: unset)
    Non-empty: also record every candidate block's score.

``OLMO_GATE_DATASET`` / ``OLMO_GATE_SUBTASK`` / ``OLMO_GATE_CONTEXT_LEN`` / ``OLMO_GATE_DOC_ID``
(defaults: ``ruler`` / empty / content prompt length / per-process counter)
    Fields stamped on each logged record; the eval harness sets them per example.
"""

import os
from typing import Optional

__all__ = [
    "LANDMARK_SPARSE_DECODE",
    "LANDMARK_FLAT_SOFTMAX",
    "FLA_LENGTH_AUTOTUNE",
    "LANDMARK_SPARSE_KERNEL",
    "LM_FAST_FWD_WARPS",
    "LM_FAST_FWD_STAGES",
    "LM_FAST_BWD_WARPS",
    "LM_FAST_BWD_STAGES",
    "KV_ROUTE_BLOCKMASK",
    "KV_ROUTE_BLOCKMASK_LEGACY",
    "KV_ROUTE_DEBUG",
    "KV_ROUTE_DEBUG_LEGACY",
    "LANDMARK_GATE_LOG",
    "GATE_LOG_ALL",
    "GATE_DATASET",
    "GATE_SUBTASK",
    "GATE_CONTEXT_LEN",
    "GATE_DOC_ID",
    "get_env",
    "env_int",
]

LANDMARK_SPARSE_DECODE = "OLMO_LANDMARK_SPARSE_DECODE"
LANDMARK_FLAT_SOFTMAX = "OLMO_LANDMARK_FLAT_SOFTMAX"
FLA_LENGTH_AUTOTUNE = "OLMO_FLA_LENGTH_AUTOTUNE"

LANDMARK_SPARSE_KERNEL = "OLMO_LANDMARK_SPARSE_KERNEL"
LM_FAST_FWD_WARPS = "OLMO_LM_FAST_FWD_WARPS"
LM_FAST_FWD_STAGES = "OLMO_LM_FAST_FWD_STAGES"
LM_FAST_BWD_WARPS = "OLMO_LM_FAST_BWD_WARPS"
LM_FAST_BWD_STAGES = "OLMO_LM_FAST_BWD_STAGES"

KV_ROUTE_BLOCKMASK = "OLMO_KV_ROUTE_BLOCKMASK"
KV_ROUTE_BLOCKMASK_LEGACY = "KV_ROUTE_BLOCKMASK"  # still set by research/kv_route launchers
KV_ROUTE_DEBUG = "OLMO_KV_ROUTE_DEBUG"
KV_ROUTE_DEBUG_LEGACY = "KV_ROUTE_DEBUG"  # still set by train_ctc_suite.py --kv-route-debug

LANDMARK_GATE_LOG = "OLMO_LANDMARK_GATE_LOG"
GATE_LOG_ALL = "OLMO_GATE_LOG_ALL"
GATE_DATASET = "OLMO_GATE_DATASET"
GATE_SUBTASK = "OLMO_GATE_SUBTASK"
GATE_CONTEXT_LEN = "OLMO_GATE_CONTEXT_LEN"
GATE_DOC_ID = "OLMO_GATE_DOC_ID"


def get_env(
    name: str, default: Optional[str] = None, *, legacy: Optional[str] = None
) -> Optional[str]:
    """
    Read a toggle, falling back to its legacy name.

    :param name: The toggle (one of the constants in this module).
    :param default: Returned when neither name is set.
    :param legacy: An older name still accepted; ``name`` wins when both are set.

    :returns: The raw string value, or ``default``.
    """
    value = os.environ.get(name)
    if value is None and legacy is not None:
        value = os.environ.get(legacy)
    return default if value is None else value


def env_int(name: str, default: int) -> int:
    """
    Read an integer toggle.

    :param name: The toggle (one of the constants in this module).
    :param default: Returned when it is unset.

    :returns: The parsed value.
    """
    return int(os.environ.get(name, str(default)))
