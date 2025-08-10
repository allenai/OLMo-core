
import logging
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module
import olmo_core.nn.transformer
from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import hide_from_torch, unhide_from_torch
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device
from .block import MoEFusedV2TransformerBlock, MoEFusedV2TransformerBlockConfig
if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

class MoEFusedV2Transformer(olmo_core.nn.transformer.Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            if not block.is_moe:
                continue
            block = cast(MoEFusedV2TransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (metric_val, reduce_type)

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (torch.max(out[metric_name][0], metric_val), reduce_type)
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoEFusedV2TransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        raise OLMoConfigurationError("Do not use `apply_ep`, use `apply_epdp` instead.")

    def apply_epdp(
        self,
        dp_mesh: DeviceMesh,
        ep_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
        """
        from torch.distributed._composable.replicate import replicate

        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoEFusedV2TransformerBlock, block)
            block.apply_ep(ep_mesh)


        # Cast model explicitly to the specified dtype before applying DDP
        target_dtype = param_dtype or self.dtype
        if target_dtype != self.dtype:
            self.to(dtype=target_dtype)

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore
                
        self.to(torch.bfloat16) # HACK, need fix
        
        ep_modules = [m for m in self.modules() if getattr(m, '_ep_sharded', False) ] # collect the ep sharded part based on `_ep_sharded` field (will be set to True in `apply_ep`)
        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100, ignored_modules=ep_modules) # dense ddp

        ep_dp_mesh = ep_mesh['ep_dp']
        for m in ep_modules:
            replicate(m, device_mesh=ep_dp_mesh, bucket_cap_mb=100) # moe ddp
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        from ...transformer.model import _hide_cpu_inputs_from_torch, _unhide_cpu_inputs_from_torch
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    def prepare_experts_for_fsdp(
        self,
        * args: Any,
        ** kwargs: Any,
    ):
        raise OLMoConfigurationError(
            "prepare_experts_for_fsdp is not supported for MoEFusedV2Transformer. "
            "Use prepare_experts_for_ddp instead."
        )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            pass # TODO: Anything to do here?

    def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoEFusedV2TransformerBlock, block)
            block.post_batch(dry_run=dry_run)

