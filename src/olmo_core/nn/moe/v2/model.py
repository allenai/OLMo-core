
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
log = logging.getLogger(__name__)

class MoEFusedV2Transformer(olmo_core.nn.transformer.Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_enabled = False # default

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

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
        """
        from torch.distributed._composable.replicate import replicate

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
        
        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100, gradient_as_bucket_view=True, 
                #   mixed_precision=
                  )
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )


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
        # target_dtype = param_dtype or self.dtype
        # if target_dtype != self.dtype:
        #     self.to(dtype=target_dtype)

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore
                
        # TODO: here the replicate/DDP wrapper takes the model in original precision (mostly fp32)
        # but later the wrapper model converts to bf16, will there be a problem?
        
        ep_modules = [m for m in self.modules() if getattr(m, '_ep_sharded', False) ] # collect the ep sharded part based on `_ep_sharded` field (will be set to True in `apply_ep`)
        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100, ignored_modules=ep_modules, gradient_as_bucket_view=True) # dense ddp

        ep_dp_mesh = ep_mesh['ep_dp']
        for m in ep_modules:
            replicate(m, device_mesh=ep_dp_mesh, bucket_cap_mb=100, gradient_as_bucket_view=True) # moe ddp
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        from ...transformer.model import _hide_cpu_inputs_from_torch, _unhide_cpu_inputs_from_torch
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        self.ep_enabled = True

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

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
        model_part_idx: int = 0,
    ) -> torch.Generator:
        from .block import MoEFusedV2TransformerBlock
        from olmo_core.nn.attention import (
            Attention,
            FusedAttention,
            RingAttentionLoadBalancer,
            RingAttentionLoadBalancerType,
        )
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        """
        device = device or self.device
        self.to_empty(device=device)

        for name, module in self.named_modules():
            if hasattr(module, "reset_parameters"): # TODO: what's the point of this
                module.reset_parameters()  # type: ignore
                log.info(f"'{name}' called reset_parameters()")


        seed = self.init_seed

        # adjust seed for PP, including
        # 1. PP stage
        # 2. model part within the PP stage (eg, interleaved 1F1B)
        if self.pp_enabled:
            assert world_mesh
            seed += get_pp_mesh(world_mesh).get_local_rank() 
            seed += model_part_idx * 997 # random prime

        # adjust seed for EP_MP, different EP shards should have different init values
        # but within the same EP_DP group, they should share the same init value
        ep_generator = None
        if self.ep_enabled:
            assert world_mesh
            ep_mp_rank = world_mesh['ep_mp'].get_local_rank()
            ep_seed = seed + (1 + ep_mp_rank) * 653 # random prime; +1 so that it will never be the same as the base seed
            ep_generator = torch.Generator(device).manual_seed(ep_seed)

        generator = torch.Generator(device).manual_seed(seed)
        
        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings, d_model=self.d_model, std=self.init_std, generator=generator
            )

        for block in self.blocks.values():
            # This might fail if it's wrapped.
            if isinstance(block, MoEFusedV2TransformerBlock):

                block = cast(MoEFusedV2TransformerBlock, block)

                # v2 MoE blocks.
                att = cast(Union[Attention, FusedAttention], block.attention)

                # Attention weights.
                self.init_method.init_attention(
                    att,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )
                # MoE weights.
                self.init_method.init_moe_v2(
                    block,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                    ep_generator=ep_generator,
                )

            else:
                # usually this is for the first dense layer
                from ...transformer.block import TransformerBlock
                block = cast(TransformerBlock, block)
                att = cast(Union[Attention, FusedAttention], block.attention)

                # Attention weights.
                self.init_method.init_attention(
                    att,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

                # Feed-forward weights.
                if hasattr(block, "feed_forward"):
                    self.init_method.init_feed_forward(
                        block.feed_forward,
                        d_model=self.d_model,
                        block_idx=block.block_idx,
                        num_blocks=self.n_layers,
                        std=self.init_std,
                        generator=generator,
                    )

                # MoE weights.
                if hasattr(block, "feed_forward_moe"):
                    raise OLMoConfigurationError("Do not use legacy MoE block")

            # Warm up RoPE cache.
            if max_seq_len is not None and att.rope is not None:
                att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out, d_model=self.d_model, std=self.init_std, generator=generator
            )

        return generator

def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)