from collections import defaultdict
from typing import Literal, Union

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,  # type: ignore
    torch.ops.aten._scaled_dot_product_efficient_attention.default,  # type: ignore
    torch.ops.aten._scaled_dot_product_flash_attention.default,  # type: ignore
    torch.ops._c10d_functional.reduce_scatter_tensor.default,  # type: ignore
}


def apply_activation_checkpointing_to_transformer_block(
    module: nn.Module,
    mode: Literal["full", "selective"],
    selective_option: Union[Literal["op"], int],
) -> nn.Module:
    """
    Apply activation checkpointing to a transformer block.

    :param module: A transformer block to apply AC to.
    :param mode: Either "full" for apply AC to each block, or "selective" which depends on
        the value of ``selective_option``.
    :param selective_option: If "op", AC is applied to selective operations. If an int, it's
        applied to each block with this frequency.
    """
    # Adapted from
    # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L206

    if mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert mode == "selective", f"{mode}"
    use_op_sac = selective_option == "op"
    use_layer_sac = isinstance(selective_option, int) or selective_option.isdigit()

    if use_op_sac:
        # NOTE: only available on nightly so far
        from torch.utils.checkpoint import CheckpointPolicy  # type: ignore
        from torch.utils.checkpoint import (
            create_selective_checkpoint_contexts,  # type: ignore
        )

        def _get_custom_policy(meta):
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
            meta = defaultdict(int)  # type: ignore[var-annotated]
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(selective_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module
    else:
        raise ValueError(
            f"Invalid selective AC option: {selective_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
