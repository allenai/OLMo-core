import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module
from dataclasses import dataclass
from olmo_core.ops import moe as ops
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.utils import get_or_init_stream
from olmo_core.exceptions import OLMoConfigurationError

from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
# import olmo_core.nn
import olmo_core.nn.transformer


from ...attention import AttentionConfig, RingAttentionLoadBalancerType
from ...buffer_cache import BufferCache
from ...functional import l2_normalize
from ...layer_norm import LayerNormConfig
from ...moe import  MoERouterGatingFunction
from ...moe import MoERouterConfig as MoERouterConfigV1
from ...moe.loss import MoELoadBalancingLossGranularity
from ...moe.utils import async_copy_to_cpu, wait_stream_no_compile
from .router import MoERouterV2
from typing import List, Optional
from torch.utils.checkpoint import checkpoint, CheckpointFunction
import nvtx
import torch.distributed as dist
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.config import Config, DType, StrEnum
from typing import Callable
from .router import MoERouterConfigV2
from .shared_experts import SharedExpertsConfig
from .routed_experts import RoutedExpertsConfig
    

from ..utils import (
    moe_unpermute_no_compile,
    moe_permute_no_compile,
    moe_sort_chunks_by_index_no_compile,
)



from olmo_core.nn.transformer.config import (
    TransformerBlockType, TransformerBlockConfig
)
@dataclass
class MoEFusedV2TransformerBlockConfig(TransformerBlockConfig):

    router: Optional[MoERouterConfigV2] = None
    
    shared_experts: Optional[SharedExpertsConfig] = None
    
    routed_experts: Optional[RoutedExpertsConfig] = None
    
    shared_experts_router: Optional[MoERouterConfigV2] = None
    
    routed_experts_router: Optional[MoERouterConfigV2] = None

        
    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> olmo_core.nn.transformer.block.TransformerBlockBase:
        assert self.feed_forward is None and self.feed_forward_moe is None, "MoEFusedV2TransformerBlock does not support `feed_forward` or `feed_forward_moe` (use TransformerBlockConfig instead). Set `shared_experts` and `routed_experts` instead."

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )


        if self.name == TransformerBlockType.moe_fused_v2:
            return MoEFusedV2TransformerBlock(**kwargs)
        else:
            raise NotImplementedError(self.name)


    def num_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.attention.num_params(d_model)
        if self.attention_norm is not None:
            block_params += self.attention_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            block_params += self.routed_experts.num_params()
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.feed_forward_norm is not None:
            block_params += self.feed_forward_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.attention.num_params(d_model)
        if self.attention_norm is not None:
            block_params += self.attention_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None, "routed_experts must have a router"
            block_params += self.routed_experts.num_active_params(self.routed_experts_router.top_k)
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.feed_forward_norm is not None:
            block_params += self.feed_forward_norm.num_params(d_model)

        return block_params

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType
class MoEFusedV2TransformerBlock(olmo_core.nn.transformer.block.TransformerBlockBase):
    
    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        attention_norm: LayerNormConfig,
        routed_experts_router: MoERouterConfigV2,
        shared_experts_router: MoERouterConfigV2,
        shared_experts: SharedExpertsConfig,
        routed_experts: RoutedExpertsConfig,
        feed_forward_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        assert dropout == 0.0, "MoEFusedV2TransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx
        
        
        self.attention = attention.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = attention_norm.build(d_model, init_device=init_device)

            
        self.routed_experts = routed_experts.build(init_device=init_device)
        self.shared_experts = shared_experts.build(init_device=init_device)
        self.routed_experts_router = routed_experts_router.build(init_device=init_device)
        self.shared_experts_router = shared_experts_router.build(init_device=init_device)

        self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)
        
        self.ep_pg = None
        self._ep_enabled = False
        self.tp_pg = None
        self._tp_enabled = False

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        # TODO: compute shared and routed experts metrics
        # metrics_shared = self.shared_experts.compute_metrics(reset=reset)
        metrics_routed = self.routed_experts_router.compute_metrics(reset=reset)
        # metrics = {
        #     "shared": metrics_shared,
        #     "routed": metrics_routed,
        # }
        return metrics_routed

    def reset_metrics(self):
        # self.shared_experts_router.reset_metrics()
        self.routed_experts_router.reset_metrics()


    @property
    def is_moe(self) -> bool:
        return True

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def get_dense_stream(self) -> torch.cuda.Stream:
        return get_or_init_stream(id=2, priority=20)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        MOE_LAYER_USE_RECOMPUTE = False 
        if MOE_LAYER_USE_RECOMPUTE:
            # NOTE: this is the same as the MoEHybridTransformerBlock, but with recompute
            # on the dense forward pass.
            # return checkpoint(
            #     self.combined_forward, x, use_reentrant=False, loss_div_factor=loss_div_factor, **kwargs
            # )
            raise NotImplementedError("MOE_LAYER_USE_RECOMPUTE is not supported in MoEFusedV2TransformerBlock")
        else:
            # always use combined forward    
            return self.combined_forward(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass # nothing to do

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        ep_dp_mesh = ep_mesh['ep_dp']
        ep_mp_mesh = ep_mesh['ep_mp']
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(
            ep_mesh
        )
        self._ep_enabled = True

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError("TP is not supported in MoEFusedV1TransformerBlock")

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        raise NotImplementedError("CP is not supported in MoEFusedV1TransformerBlock")
        self.attention.apply_cp(cp_mesh, load_balancer)
        self.shared_experts.apply_cp(cp_mesh)
        self.routed_experts.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("FSDP is not supported in MoEFusedV2TransformerBlock")

    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1
        
    # @torch.compile
    def router_forward(
        self,
        router: MoERouterV2,
        local_x: torch.Tensor,
        scores_only: bool,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
    ):
        return router(
            local_x, 
            scores_only,
            loss_div_factor=loss_div_factor # scalar
        )
    
    # @torch.compile
    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:

        B, S, D = x.shape
        
        # attention 
        # + attention norm
        # + residual connection
        x = x + self.attention_norm(self.attention(x, **kwargs))
        # remove attention kwargs
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)

        local_x = x
            
        (
            local_x_global_routed_expert_weights, # (B, S, top_k)
            local_x_global_routed_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_routed_expert, # (num_experts, )
            routed_expert_router_aux_loss # scalar
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=x, 
            scores_only=False,
            loss_div_factor=loss_div_factor # scalar
        )
        
        
        (
            local_x_global_shared_expert_weights, # (B, S, E_shared)
            _, 
            _, 
            _ 
        ) = self.router_forward(
            router=self.shared_experts_router,
            local_x=x, 
            scores_only=True,  # only need scores for shared experts
            loss_div_factor=loss_div_factor # scalar
        )
        
        
        ##### DENSE: submit the dense forward pass to a separate stream #####
        # NOTE: launch dense after the router, so it's more likely to overlap permute and all-to-all,
        # but it only depends on the x (before router), so the wait_stream() can be called here
        # priority sparse forward > dense forward, because the sparse one is going to take longer
        
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),  # type: ignore
            other_stream=torch.cuda.current_stream() # type: ignore
        ) # ignore
        
        # only when grad enabled
        if torch.is_grad_enabled():
            with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                if routed_expert_router_aux_loss is not None:
                    local_x = attach_auxiliary_loss(local_x, routed_expert_router_aux_loss)


        ######################################################################
        ################# Call global_permute_mlp_unpermute() ##############
        ############################### START ################################


        with nvtx.annotate("permute_mlp_unpermute", color="blue"):
            if self.ep_enabled:
                raise NotImplementedError("TODO: EP permutation ")
                x_moe, dense_out = self.global_permute_mlp_unpermute(
                    local_x,
                    local_x_global_expert_weights,
                    local_x_global_expert_indices,
                    local_batch_size_per_global_expert=local_batch_size_per_global_routed_expert,
                    # overlap_callback=self.overlap_callback,
                    # overlap_callback=None,
                    # overlap_callback_x=x,
                    **kwargs,
                ) # type: ignore
            else:
                # (B, S, D), (E_shared, B, S, D)
                x_moe, shared_out = self.global_permute_mlp_unpermute_no_ep(
                    local_x,
                    local_x_global_routed_expert_weights,
                    local_x_global_routed_expert_indices, 
                    local_batch_size_per_global_expert=local_batch_size_per_global_routed_expert,
                    # overlap_callback=self.overlap_callback,
                    # overlap_callback=None,
                    # overlap_callback_x=x,
                    **kwargs,
                ) # type: ignore
        ######################################################################
        ################# Call global_permute_mlp_unpermute() ##############
        ############################### END ##################################

        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) # type: ignore

        # weighted sum of the shared experts and routed experts
        # local_x_global_shared_expert_weights -> (B, S, E_shared)
        # shared_out -> (E_shared, B, S, D)
        _, _, E_s = local_x_global_shared_expert_weights.shape
        local_x_global_shared_expert_weights.shape
        mixed_shared_out = torch.bmm(
            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
            shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
        ).squeeze(1).view(B, S, D)


        shared_out_factor = self.shared_experts_router.top_k / (self.routed_experts_router.top_k + self.shared_experts_router.top_k )
        routed_out_factor = self.routed_experts_router.top_k / (self.routed_experts_router.top_k + self.shared_experts_router.top_k )
        mlp_out = mixed_shared_out * shared_out_factor + x_moe * routed_out_factor 
        #######################

        final_out = x + self.feed_forward_norm(mlp_out)

        #######################

        
        return final_out


    # @torch.compile
    @nvtx.annotate("ParallelDroplessMLP.global_permute_mlp_unpermute_no_ep", color='blue')
    def global_permute_mlp_unpermute_no_ep(
        self,
        local_x: torch.Tensor,
        local_x_global_expert_weights: torch.Tensor,
        local_x_global_expert_indices: torch.Tensor,
        local_batch_size_per_global_expert: torch.Tensor,
        # overlap_callback: Optional[Callable] = None,
        # overlap_callback_x=None,
        # **overlap_callback_kwargs,
    ):
        x, expert_weights, expert_indices, batch_size_per_expert = (
            (local_x),
            (local_x_global_expert_weights),
            (local_x_global_expert_indices),
            (local_batch_size_per_global_expert),
        )

        in_shape = x.size()


        # step 1A: DtoH token count communication
        # mark_dynamic(batch_size_per_expert, (0,), strict=False)
        batch_size_per_expert_cpu, copy_stream, dtoh_event1 = async_copy_to_cpu(batch_size_per_expert)  # type: ignore
        
        # overlap compute while waiting for the copy to CPU to finish
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts(x) # (E_shared, B, S, D)
                
        # step 1B: permute the input tokens
        copy_stream.synchronize() # wait for the copy to CPU to finish
        
        x = x.view(-1, in_shape[-1])  # (B*S, D)

        PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE = False
        if PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE:
            x_moe = checkpoint(
                self._forward_step_rc,
                x,
                expert_indices,
                expert_weights,
                batch_size_per_expert_cpu,
                use_reentrant=False,
                in_shape=in_shape,
            )
        else:
            x_moe = self._forward_step_rc(
                x,
                expert_indices,
                expert_weights,
                batch_size_per_expert_cpu,
                in_shape=in_shape,
            )

        return x_moe, shared_out # (B, S, D), (E_shared, B, S, D)

    # @torch.compile
    def _forward_step_rc(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        batch_size_per_expert_cpu: torch.Tensor,
        in_shape: torch.Size
    ):
        routing_map = expert_indices.view(-1, self.routed_experts_router.top_k).int()
        num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
        hidden_shape_before_permute = x.shape

        # step 2: permute the input tokens
        with nvtx.annotate("Permute", color='green'):
            permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(
                inp=x, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            ) # type: ignore

        # step 3: MLP
        x = self.routed_experts(permutated_input_tokens, batch_size_per_expert_cpu.tolist())

        # step 4: unpermutate the output tokens
        with nvtx.annotate("Unpermute", color='green'):
            unpermutated_x = moe_unpermute_no_compile(
                inp=x,
                row_id_map=reversed_input_permutation_mapping,
                restore_shape=hidden_shape_before_permute,
                map_type='index',
                merging_probs=expert_weights.view(-1, self.routed_experts_router.top_k)
            ) # type: ignore
            
        return unpermutated_x.view(in_shape)
    
    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        self.shared_experts_router.post_batch(dry_run=dry_run)
        self.routed_experts_router.post_batch(dry_run=dry_run)

    @torch.compile
    @nvtx.annotate("ParallelDroplessMLP.global_permute_mlp_unpermute", color='blue')
    def global_permute_mlp_unpermute(
        self,
        local_x: torch.Tensor,
        local_x_global_expert_weights: torch.Tensor,
        local_x_global_expert_indices: torch.Tensor,
        local_batch_size_per_global_expert: torch.Tensor,
        # overlap_callback: Optional[Callable] = None,
        # overlap_callback_x=None,
        # **overlap_callback_kwargs,
    ):
        raise NotImplementedError("TODO")
        
        assert self.hidden_sharding_degree == 1, "Global permutation is only supported when hidden sharding degree is 1."
        # mark_dynamic(local_batch_size_per_global_expert, (0,), strict=False)
        
        '''
        The global_permute_mlp_unpermute function performs the following steps:
        1. **Communicates the number of tokens that will be sent to each device**:
        2. **Permutes local tokens to be ready for all-to-all communication**:
        3. **Configures the sizes for grouped GEMM**:
        4. **Starts the all-to-all communication asynchronously**:
        5. **Permutes the global tokens to be ready for MLP computation**:
        6. **MLP forward**:
        7. **Unpermutates the tokens for reverse all-to-all communication**:
        8. **Reverse all-to-all communication**:
        9. **Unpermutates the tokens to restore the original order**:
        10. **Returns the unpermutated tokens**.
        '''
        
        
        ########### 1. Communicate the number of tokens that will be sent to each device ###########
        with nvtx.annotate("Token count all_to_all", color='green'):
            with torch.no_grad():
                # Pass token count information to the device on which the
                # target expert resides.
                global_batch_size_per_local_expert = torch.empty_like(
                    local_batch_size_per_global_expert,
                )
                global_batch_size_handle = dist.all_to_all_single(
                    global_batch_size_per_local_expert, # Gathered concatenated output tensor.
                    local_batch_size_per_global_expert, # Input tensor to scatter.
                    group=self.ep_pg,
                    async_op=True,
                )

        ############################################ end



        ###########  3. Configure the sizes for grouped GEMM ###########

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with nvtx.annotate("Sync token count", color='green'):
            with torch.no_grad():
                global_batch_size_handle.wait()

                # Reshape to (ep_world_size, num_local_experts).
                local_batch_size_per_global_expert = local_batch_size_per_global_expert.view(
                    self.ep_world_size, self.num_local_experts
                )
                global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                    self.ep_world_size, self.num_local_experts
                )
                # Calculate the bins boundaries from the token counts. # [EP, num_local_experts] -> [num_local_experts,]
                parallel_batch_size_per_expert = global_batch_size_per_local_expert.sum(
                    dim=0,
                    dtype=torch.long,
                )
                
                # NOTE: host-device sync here.
                
                # send_counts, copy_stream, dtoh_event1 = async_copy_to_cpu(local_batch_size_per_global_expert.sum(dim=-1))
                # recv_counts, copy_stream, dtoh_event2 = async_copy_to_cpu(global_batch_size_per_local_expert.sum(dim=-1))
                
                # option 1
                # dtoh_event1.synchronize() # wait for the copy to CPU to finish
                # dtoh_event2.synchronize()
                
                # option 2
                # copy_stream.synchronize() # wait for the copy to CPU to finish

                
                # option 3 
                # NOTE: this is not going to work because only current stream can wait for events, but the all_to_all communication is done in a different stream.
                # torch.cuda.current_stream().wait_event(dtoh_event1) # wait for the copy to CPU to finish
                # torch.cuda.current_stream().wait_event(dtoh_event2)
                
                # option 4
                send_counts = local_batch_size_per_global_expert.sum(dim=-1).to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU
                recv_counts = global_batch_size_per_local_expert.sum(dim=-1).to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU
                
                parallel_batch_size_per_expert_cpu = parallel_batch_size_per_expert.to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU

            
            # re-indent to enable grad
            
            # put the dense branch here to overlap DtoH sync
            # TODO: one potential issue is that:
            # GPU side: permute in step 2 is finished
            # CPU side: the dense branch (attention + MLP) is still being submitted
            # result: GPU idle before the dense branch actually starts to run
            # potential fix: 
            # 1. find some else to run before overlap_callback
            # 2. make permute in step 2 run longer (increase batch size)
            # 3. break the dense branch into smaller pieces and blend with other operations
            # if overlap_callback is not None:
            #     overlap_out = overlap_callback(
            #         overlap_callback_x, 
            #         **overlap_callback_kwargs,
            #     )
            # else:
            #     overlap_out = None
            
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out = self.shared_experts(local_x)
                
            with torch.no_grad():
                torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
                
                send_counts = send_counts.tolist() # tensor to list
                recv_counts = recv_counts.tolist() # tensor to list
                
                tokens_received = sum(recv_counts)

                # Construct the expert indices for the permuted tokens.
                global_x_local_expert_indices = torch.remainder(
                    torch.arange(
                        self.num_experts * self.hidden_sharding_degree,
                        dtype=torch.int32,
                        device=local_x.device,
                    ),
                    self.num_local_experts,
                ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts

                # this specifiyes for the received global tokens, which local expert they belong to
                global_x_local_expert_indices = torch.repeat_interleave(
                    global_x_local_expert_indices,
                    global_batch_size_per_local_expert.flatten(),
                    output_size=tokens_received,
                ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts
        


        
        EP_PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE = True
        if EP_PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE:
            local_x = checkpoint(
                self.forward_step_1_9,
                local_x,
                local_x_global_expert_weights,
                global_x_local_expert_indices,
                parallel_batch_size_per_expert_cpu,
                local_x_global_expert_indices,
                recv_counts,
                send_counts,
                use_reentrant=False,
            )
        else:
            local_x = self.forward_step_1_9(
                local_x,
                local_x_global_expert_weights,
                global_x_local_expert_indices,
                parallel_batch_size_per_expert_cpu,
                local_x_global_expert_indices,
                recv_counts,
                send_counts,
            )


        return local_x, overlap_out


    @torch.compile
    def forward_step_1_9(
        self,
        local_x: torch.Tensor,
        local_x_global_expert_weights: torch.Tensor,
        global_x_local_expert_indices: torch.Tensor,
        parallel_batch_size_per_expert_cpu: torch.Tensor,
        local_x_global_expert_indices: torch.Tensor,
        recv_counts: List[int],
        send_counts: List[int]
    ):
        raise NotImplementedError("TODO")
        
        ########### 2. permute local tokens to be ready for all-to-all communication ###########
        with nvtx.annotate("Permute local tokens", color='green'):
            routing_map = local_x_global_expert_indices.view(-1, self.routed_experts_router.top_k).int()
            num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
            hidden_shape_before_permute = local_x.shape
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                inp=local_x, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            ) # type: ignore
            
            # now permutated_local_x tokens are grouped by expert, which means tokens will go to expert id:
            # [0 , 0 , ... 1, ... 2, ..., ..., 31, 31]  (if 32 experts)
            # if EP=8, each rank has 4 experts, then tokens of
            # [0, 0, ..., 3, 3] go to rank 0,
            # [4, 4, ..., 7, 7] go to rank 1, 
            # and so on.
        ############################################ end
        
        ###########  4. Start the all-to-all communication asynchronously ###########

        with nvtx.annotate("all2all", color='green'):
            global_x, global_x_handle = ops.all_to_all(
                permutated_local_x,
                recv_counts,
                send_counts,
                group=self.ep_pg,
                async_op=True,
            )
            
            global_x_handle.wait()
            del permutated_local_x
        ############################################ end
    
        ###########  5. Permute the global tokens to be ready for MLP computation ###########
        with nvtx.annotate("Permute global tokens for MLP", color='green'):
            # option 1: use moe_sort_chunks_by_index (by TE <- trition)
            # input_chunk_idxs = torch.arange(
            #     self.num_experts, device=local_x.device
            # )
            # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
            # sort_input_by_local_experts = input_chunk_idxs.reshape(
            #     -1, self.num_local_experts
            # ).T.ravel() 
            # split into 32 chunks (32 experts)
            # e.g., [ 
            # 0,  4,  8, 12, 16, 20, 24, 28,    --> these 8 chunks come from all 8 EP ranks, go to local expert 0
            # 1,  5,  9, 13, 17, 21, 25, 29,    --> these 8 chunks come from all 8 EP ranks, go to local expert 1
            # 2,  6, 10, 14, 18, 22, 26, 30,    --> these 8 chunks come from all 8 EP ranks, go to local expert 2
            # 3,  7, 11, 15, 19, 23, 27, 31     --> these 8 chunks come from all 8 EP ranks, go to local expert 3
            # ].  (1D tensor)

            
            ## chunk size is specified by `global_batch_size_per_local_expert`
            
            
            # e.g., global_batch_size_per_local_expert
            # local experts 0     1     2     3
            # ep0       [[3108, 5307, 5798, 4067],
            # ep1        [4642, 3836, 3488, 3477],
            # ep2        [5129, 3964, 2472, 4194],
            # ep3        [4266, 3191, 4511, 3841],
            # ep4        [5059, 5758, 4838, 3201],
            # ep5        [5388, 3531, 3419, 2860],
            # ep6        [3862, 3605, 2945, 3840],
            # ep7        [3960, 4624, 3414, 4406]]
            
            # so we want to put (3108+4642+5129+4266+5059+5388+3862+3960) tokens to local expert 0,
            # and so on
            
            # global_x = moe_sort_chunks_by_index_no_compile(
            #     inp=global_x,
            #     split_sizes=global_batch_size_per_local_expert.ravel(),
            #     sorted_index=sort_input_by_local_experts
            # ) # type: ignore

            # option 2: use moe_permute (by TE), and pretend topk is 1
            routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
            num_out_tokens2 = routing_map2.size(0) * 1 # dropless
            hidden_shape_before_permute2 = global_x.shape
            global_x, reversed_global_x_permutation_mapping = moe_permute_no_compile(
                inp=global_x, 
                routing_map=routing_map2, 
                num_out_tokens=num_out_tokens2, 
                map_type='index'
            )    # type: ignore
                
                
        ############################################ end

        
        ########## 6. MLP forwrad ###########

        global_x = self.mlp(global_x, parallel_batch_size_per_expert_cpu.tolist())

        ############################################ end
        
        
        ############ 7. Unpermute the output tokens to be ready for all-to-all communication ##########
        with nvtx.annotate("Unpermute global tokens", color='green'):
            # option 1: use moe_sort_chunks_by_index (by TE <- trition)
            # restore_output_by_local_experts = input_chunk_idxs.reshape(
            #     self.num_local_experts, -1
            # ).T.ravel() # [ 0,  8, 16, 24,  1,  9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31]
            # global_x = moe_sort_chunks_by_index_no_compile(
            #     global_x, 
            #     split_sizes=global_batch_size_per_local_expert.T.ravel(),
            #     sorted_index=restore_output_by_local_experts
            # )
            
            # option 2: use moe_unpermute (by TE)
            global_x = moe_unpermute_no_compile(
                inp=global_x,
                row_id_map=reversed_global_x_permutation_mapping,
                merging_probs=None,
                restore_shape=hidden_shape_before_permute2,
                map_type='index',
            ) # type: ignore
        ############################################ end
    
            
    
        ########## 8. reverse_all_to_all ###########
        with nvtx.annotate("reverse_all_to_all", color='green'):
            local_x, local_x_handle = ops.all_to_all(
                global_x,
                send_counts,
                recv_counts,
                group=self.ep_pg,
                async_op=True,
            )
            
            local_x_handle.wait()
            
            del global_x # done with global tokens
        ############################################ end
        
        
        ############ 9. Unpermute the (local) tokens returned by all-to-all communication ##########
        with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
            local_x = moe_unpermute_no_compile(
                inp=local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=local_x_global_expert_weights.view(-1, self.top_k),
                restore_shape=hidden_shape_before_permute,
                map_type='index',
            ) # type: ignore
        ############################################ end
    
        return local_x

