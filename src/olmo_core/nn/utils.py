from collections import defaultdict
from typing import Dict, Tuple, Type

import torch
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)


def _get_custom_checkpoint_policy(meta: Dict[str, int]):
    # Adapted from
    # https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
    from torch.utils.checkpoint import CheckpointPolicy

    _save_list = {
        torch.ops.aten.mm.default,  # type: ignore
        torch.ops.aten._scaled_dot_product_efficient_attention.default,  # type: ignore
        torch.ops.aten._scaled_dot_product_flash_attention.default,  # type: ignore
        torch.ops._c10d_functional.reduce_scatter_tensor.default,  # type: ignore
        # for low precision training, it's useful to always save
        # the result of max(abs(tensor))
        torch.ops.aten.abs.default,  # type: ignore
        torch.ops.aten.max.default,  # type: ignore
    }

    def _custom_policy(ctx, func, *args, **kwargs):
        del args, kwargs
        mode = "recompute" if ctx.is_recompute else "forward"
        mm_count_key = f"{mode}_mm_count"
        if func == torch.ops.aten.mm.default:  # type: ignore
            meta[mm_count_key] += 1
        # Saves output of all compute ops, except every second mm
        to_save = func in _save_list and not (
            func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0  # type: ignore
        )
        return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

    return _custom_policy


def selective_checkpointing_context_fn():
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    meta: Dict[str, int] = defaultdict(int)
    return create_selective_checkpoint_contexts(_get_custom_checkpoint_policy(meta))


def get_tp_wrappers(
    float8_enabled: bool,
) -> Tuple[Type[RowwiseParallel], Type[ColwiseParallel], Type[PrepareModuleInput]]:
    if not float8_enabled:
        return (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )
    else:
        # TODO (epwalsh): once float8 configuration supports delayed scaling,
        # add a check here to enforce supported float8 all-gather configurations.
        from torchao.float8.float8_tensor_parallel import (  # type: ignore
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        return (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
