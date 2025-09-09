
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
from olmo_core.ops import moe as ops
from ...lm_head import LMHeadConfig, LMOutputWithLoss
import nvtx
from torch.utils.checkpoint import checkpoint, CheckpointFunction

from ..utils import (
    moe_unpermute_no_compile,
    moe_permute_no_compile,
    moe_sort_chunks_by_index_no_compile,
)
from .te.cpu_offload import (
    get_cpu_offload_context,
    CpuOffloadHook
)

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
        self.tbo = True

        if self.tbo:
            self.first_moe_idx = self._check_tbo_requirements()
        else:
            self.first_moe_idx = None

        self._debug_alloc_mem_layer_logs = []
        self._debug_max_alloc_mem_layer_logs = []
    #     self.cpu_offload = True

    #     self.offload_context, self.offload_sync_func = get_cpu_offload_context(
    #         enabled=self.cpu_offload,
    #         num_layers=4,
    #         model_layers=self.n_layers,
    #     )

    # def reset_offload_handler(self):
    #     if self.cpu_offload:
    #         assert isinstance(self.offload_context, CpuOffloadHook)
    #         self.offload_context.offload_handler.groupid_reset()

    def _log_debug_mem(self, tag: str):
        self._debug_alloc_mem_layer_logs.append((tag, torch.cuda.memory_allocated()/1024**3))
        self._debug_max_alloc_mem_layer_logs.append((tag, torch.cuda.max_memory_allocated()/1024**3))
    def _reset_debug_mem_logs(self):
        self._debug_alloc_mem_layer_logs = []
        self._debug_max_alloc_mem_layer_logs = []

    def _check_tbo_requirements(self):
        # make sure dense blocks only appear before moe blocks
        
        found_moe = False
        first_moe_idx = None
        for idx, block in enumerate(self.blocks.values()):
            if found_moe and not block.is_moe:
                raise OLMoConfigurationError(
                    "When TBO is enabled, all dense blocks must appear before MoE blocks."
                )
            if block.is_moe and not found_moe:
                found_moe = True
                first_moe_idx = idx
        
        return first_moe_idx

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

    def apply_compile(self):
        super().apply_compile()
        self._tbo_last_step = torch.compile(self._tbo_last_step)

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

    def forward_tbo(
            self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        assert self.ep_enabled, "TBO requires EP to be enabled."
            
        input_ids, labels, block_kwargs, lm_head_kwargs = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )
        assert input_ids.size(0) % 2 == 0, "When TBO is enabled, the batch size must be even."
        self._log_debug_mem('before embed')

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids

        # forward dense blocks
        for block_idx, block in enumerate(self.blocks.values()):
            if block.is_moe: 
                assert self.first_moe_idx == block_idx
                break
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            with nvtx.annotate(f"fwd_block_{block_idx}", color="blue"):
                # with self.offload_context:
                # h = block(h, **block_kwargs)
                h = checkpoint(
                    block,
                    h,
                    use_reentrant=False,
                    **block_kwargs,
                )
                h = cast(torch.Tensor, h)
                self._log_debug_mem(f'{block_idx}')



            # commit cpu offload
            # if torch.is_grad_enabled() and self.offload_sync_func is not None:
            #     h = self.offload_sync_func(h)
            #     h = cast(torch.Tensor, h)

        # forward moe blocks with TBO
        x0, x1 = h.chunk(2, dim=0) # assume even batch size
        labels0, labels1 = None, None
        if labels is not None:
            labels0, labels1 = labels.chunk(2, dim=0)
            del labels
        del h
        # Mark sizes as dynamic for torch.compile().
        if self.compile_enabled:
            mark_dynamic(x0, (0, 1), strict=False)
            mark_dynamic(x1, (0, 1), strict=False)
        x1_is_fresh = True # x1 is always fresh in the beginning
        x1_ctx = {
            "x1": x1,
        }
        for block_idx, block in enumerate(self.blocks.values()):
            # skip dense blocks
            if not block.is_moe:
                continue

            with nvtx.annotate(f"fwd_block_{block_idx}", color="blue"):
                block = cast(MoEFusedV2TransformerBlock, block)
                # with self.offload_context:
                x0, x1_ctx = block.checkpointed_combined_forward_ep_tbo(x0, x1_ctx, x1_is_fresh, **block_kwargs)
                x1_is_fresh = False # after the first TBO block, x1 is no longer fresh
                self._log_debug_mem(f'{block_idx}')




            # if torch.is_grad_enabled() and self.offload_sync_func is not None:
            #     x0 = self.offload_sync_func(x0)
            #     x0 = cast(torch.Tensor, x0)


        # finish x1 last steps
        h0, h1 = self._tbo_last_step(x0, x1_ctx, lm_head_kwargs, labels0, labels1)
        self._log_debug_mem(f'last_step')

        # merge h0 h1
        if self.lm_head is None:
            h = torch.cat([h0, h1], dim=0)
            return h
        else:
            merged = LMOutputWithLoss(
                None,
                h0[1] + h1[1],
                h0[2] + h1[2],
                h0[3] + h1[3],
            )
            return merged

    # @torch.compile
    def _tbo_last_step(self, x0, x1_ctx: Dict[str, Any], lm_head_kwargs: Dict[str, Any], labels0: Optional[torch.Tensor], labels1: Optional[torch.Tensor]):
        with nvtx.annotate("TBO-1", color='orange'):
            global_x1 = x1_ctx['global_x1']
            send_counts1 = cast(List, x1_ctx['send_counts1'])
            recv_counts1 = cast(List, x1_ctx['recv_counts1'])


            last_block = cast(MoEFusedV2TransformerBlock, x1_ctx['last_block'])

            assert last_block.routed_experts_router is not None
            # finish reverse all2all and other ops for x1
            with nvtx.annotate("reverse_all_to_all", color='green'):
                global_x1 = cast(torch.Tensor, global_x1)
                local_x1, local_x_handle1 = ops.all_to_all_async(
                # local_x1, local_x_handle1 = ops.all_to_all(
                    global_x1,
                    send_counts1,
                    recv_counts1,
                    group=last_block.ep_pg,
                    # async_op=True,
                )
            
            reversed_local_x_permutation_mapping1 = x1_ctx['reversed_local_x_permutation_mapping1']
            local_x_global_routed_expert_weights1 = x1_ctx['local_x_global_routed_expert_weights1']
            hidden_shape_before_permute1 = x1_ctx['hidden_shape_before_permute1']
            in_shape1 = cast(torch.Size, x1_ctx['in_shape1'])
            mixed_shared_out1 = x1_ctx['mixed_shared_out1']
            attn_res_out1 = x1_ctx['attn_res_out1']
            
            assert last_block is not None
            assert local_x_handle1 is not None
            assert last_block.routed_experts_router is not None
            
        # x0 lm head
        h0 = self.maybe_forward_lm_head(x0, lm_head_kwargs, labels=labels0)


        with nvtx.annotate("TBO-1", color='orange'):
            # local_x_handle1.wait()
            local_x1 = ops.all_to_all_wait(local_x1, local_x_handle1)


            ## 9. Unpermute the (local) tokens returned by all-to-all communication ##
            with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
                local_x1 = moe_unpermute_no_compile(
                    inp=local_x1,
                    row_id_map=reversed_local_x_permutation_mapping1,
                    merging_probs=local_x_global_routed_expert_weights1.view(-1, last_block.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute1,
                    map_type='index',
                )
            ## end
        
            
            local_x1 = local_x1.view(in_shape1)

            # weighted sum of the shared experts and routed experts
            if last_block.shared_experts is not None:
                assert mixed_shared_out1 is not None
                shared_out_factor1 = last_block.shared_experts.num_experts / (last_block.routed_experts_router.top_k + last_block.shared_experts.num_experts)
                routed_out_factor1 = last_block.routed_experts_router.top_k / (last_block.routed_experts_router.top_k + last_block.shared_experts.num_experts)
                mlp_out1 = last_block.merge_shared_and_routed_out(
                    shared_out= mixed_shared_out1,
                    shared_factor=shared_out_factor1,
                    routed_out=local_x1,
                    routed_factor=routed_out_factor1
                )
            else:
                mlp_out1 = local_x1

            x1 = attn_res_out1 + last_block.feed_forward_norm(mlp_out1)

        # Get final logits but again pass-through in case of pipeline parallelism.
        h1 = self.maybe_forward_lm_head(x1, lm_head_kwargs, labels=labels1)

        return h0, h1

    def maybe_forward_lm_head(
        self,
        x: torch.Tensor,
        lm_head_kwargs: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
    ):
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(x, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            h0 = self.lm_head(x, **lm_head_kwargs)
        else:
            h0 = x
        return h0

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        self._reset_debug_mem_logs()
        if self.tbo:
            return self.forward_tbo(
                input_ids,
                labels=labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                **kwargs,
            )
        # with torch.no_grad():
        #     tbo_dbg = self.forward_tbo(
        #         input_ids,
        #         labels=labels,
        #         ignore_index=ignore_index,
        #         loss_reduction=loss_reduction,
        #         z_loss_multiplier=z_loss_multiplier,
        #         loss_div_factor=loss_div_factor,
        #         return_logits=return_logits,
        #         **kwargs,
        #     )

        input_ids, labels, block_kwargs, lm_head_kwargs = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids

        # Run each block.
        for block_idx, block in enumerate(self.blocks.values()):
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            with nvtx.annotate(f"fwd_block_{block_idx}", color="blue"):
                h = block(h, **block_kwargs)

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            out = self.lm_head(h, **lm_head_kwargs)
        else:
            out = h
        return out

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