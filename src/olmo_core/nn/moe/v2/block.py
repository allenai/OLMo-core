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
from .routed_experts import RoutedExperts, RoutedExpertsConfig, gmm_no_compile
    

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

    # router: Optional[MoERouterConfigV2] = None
    
    shared_experts: Optional[SharedExpertsConfig] = None
    
    routed_experts: Optional[RoutedExpertsConfig] = None
    
    shared_experts_router: Optional[MoERouterConfigV2] = None
    
    routed_experts_router: Optional[MoERouterConfigV2] = None

    checkpoint_attn: bool = False
    checkpoint_permute_moe_unpermute: bool = False
    checkpoint_combined_ep_tbo: bool = False
    checkpoint_second_unpermute: bool = False
        
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

        kwargs = self.as_dict(exclude_none=False, recurse=False)
        kwargs.pop("name")
        kwargs.pop("feed_forward") # from parent config
        kwargs.pop("feed_forward_moe") # from parent config
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

    def flops_per_seq(self, d_model: int, seqlen: int) -> int:
        
        flops = 0

        # attention
        flops += self.attention.flops_per_seq(d_model, seqlen)


        # router 
        # (seq_len * d_model) * (d_model * num_total_experts)
        flops += 6 * seqlen * d_model * (
            (self.routed_experts_router.num_experts if self.routed_experts_router is not None else 0)
            + (self.shared_experts_router.num_experts if self.shared_experts_router is not None else 0)
        )

        # routed experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * top_k 
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.routed_experts is not None:
            flops += (3 * 3 * 2) * seqlen * d_model * self.routed_experts.hidden_size *  self.routed_experts_router.top_k

        # shared experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * num_experts
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.shared_experts is not None:
            flops += (3 * 3 * 2) * seqlen * d_model * self.shared_experts.hidden_size * self.shared_experts.num_experts

        return flops
        

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
        routed_experts_router: Optional[MoERouterConfigV2],
        shared_experts_router: Optional[MoERouterConfigV2],
        shared_experts: Optional[SharedExpertsConfig],
        routed_experts: Optional[RoutedExpertsConfig],
        feed_forward_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: Optional[float] = None,
        feed_forward_residual_alpha: Optional[float] = None,
        checkpoint_attn = False,
        checkpoint_permute_moe_unpermute = False,
        checkpoint_combined_ep_tbo = False,
        checkpoint_second_unpermute=False,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        assert dropout == 0.0 or dropout is None, "MoEFusedV2TransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx
        
        if attention_residual_alpha is not None:
            raise OLMoConfigurationError("MoEFusedV2TransformerBlock does not support attention_residual_alpha")
        if feed_forward_residual_alpha is not None:
            raise OLMoConfigurationError("MoEFusedV2TransformerBlock does not support feed_forward_residual_alpha")

        from .routed_experts import RoutedExperts
        from.shared_experts import SharedExperts

        self.routed_experts: Optional[RoutedExperts]
        self.routed_experts_router: Optional[MoERouterV2]
        self.shared_experts: Optional[SharedExperts]
        self.shared_experts_router: Optional[MoERouterV2]

        ######## START: Attention ########
        self.attention = attention.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = attention_norm.build(d_model, init_device=init_device)
        ######## END: Attention ########


        ######## START: MLP ########
        assert (routed_experts is not None) or (shared_experts is not None), "At least one of routed_experts or shared_experts must be provided"

        #### Optional: routed experts ####
        if routed_experts:
            # Routed Experts enabled
            assert routed_experts_router is not None, "Need routed_experts_router when using routed experts"
            self.routed_experts = routed_experts.build(init_device=init_device)
            self.routed_experts_router = routed_experts_router.build(init_device=init_device)
        else:
            # Routed Experts not enabled
            assert routed_experts_router is None, "Should not set routed_experts_router when not using routed experts"
            self.routed_experts = None
            self.routed_experts_router = None
        #### END: Optional: routed experts ####

        

        #### Optional: shared experts ####
        if shared_experts:
            # Shared Experts enabled
            self.shared_experts = shared_experts.build(init_device=init_device)
            # Shared Experts Router
            if shared_experts.num_experts > 1:
                # Need router if more than one experts
                assert shared_experts_router is not None, "Need shared_experts_router when using shared experts with more than one expert"
                self.shared_experts_router = shared_experts_router.build(init_device=init_device)
            else:
                assert shared_experts_router is None, "Should not set shared_experts_router when using only one shared expert"
                # No router if just one
                self.shared_experts_router = None
        else:
            # Shared Experts not enabled
            assert shared_experts_router is None, "Should not set shared_experts_router when not using shared experts"
            self.shared_experts = None
            self.shared_experts_router = None
        #### END: Optional: shared experts ####


        self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)

        ######## END: MLP ########
        
        self.ep_pg = None
        self._ep_enabled = False
        self.tp_pg = None
        self._tp_enabled = False

        
        # reuse the same event so that torch.compile can see the same object id and will not break the guard.
        self._dtoh_event: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) # cast to make pylance happy
        self._dtoh_event_send: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())

        # same for tbo1
        self._dtoh_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_send1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())

        self.num_local_routed_experts: Optional[int] = self.routed_experts.num_experts if self.routed_experts else None


        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_permute_moe_unpermute = checkpoint_permute_moe_unpermute
        self.checkpoint_combined_ep_tbo = checkpoint_combined_ep_tbo
        self.checkpoint_second_unpermute = checkpoint_second_unpermute

        # self.type_id = None


    def purge_cuda_events(self):
        # set all events to None (so that the model can be deepcopied)
        self._dtoh_event = None # type: ignore[assignment]
        self._dtoh_event_send = None # type: ignore[assignment]
        self._dtoh_event_recv = None # type: ignore[assignment]
        self._before_rev_all2all_event = None # type: ignore[assignment]

        self._dtoh_event1 = None # type: ignore[assignment]
        self._dtoh_event_send1 = None # type: ignore[assignment]
        self._dtoh_event_recv1 = None # type: ignore[assignment]
        self._before_rev_all2all_event1 = None # type: ignore[assignment]

    def install_cuda_events(self):
        self._dtoh_event = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_send = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event = cast(torch.cuda.Event, torch.cuda.Event())

        self._dtoh_event1 = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_send1 = cast(torch.cuda.Event, torch.cuda.Event()) 
        self._dtoh_event_recv1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1 = cast(torch.cuda.Event, torch.cuda.Event())

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        # TODO: compute shared and routed experts metrics
        # metrics_shared = self.shared_experts.compute_metrics(reset=reset)
        if self.routed_experts_router:
            metrics_routed = self.routed_experts_router.compute_metrics(reset=reset)
        else:
            metrics_routed = {}
        # metrics = {
        #     "shared": metrics_shared,
        #     "routed": metrics_routed,
        # }
        return metrics_routed

    def reset_metrics(self):
        # if self.shared_experts_router:
        #     self.shared_experts_router.reset_metrics()
        if self.routed_experts_router:
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

    def get_dense_stream(self, for_x1=False) -> torch.cuda.Stream:
        if for_x1: # not used for now
            return get_or_init_stream(id=3, priority=20)
        else:
            return get_or_init_stream(id=2, priority=20)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.routed_experts:
            if self.ep_enabled:
                return self.combined_forward_ep(x, loss_div_factor=loss_div_factor, **kwargs)
            else:
                return self.combined_forward_no_ep(x, loss_div_factor=loss_div_factor, **kwargs)
        else:
            # only shared_experts
            return self.combined_forward_shared_only(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass # nothing to do

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        assert self.routed_experts is not None, "ep can only be applied when routed_experts is enabled"
        ep_dp_mesh = ep_mesh['ep_dp']
        ep_mp_mesh = ep_mesh['ep_mp']
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(
            ep_mesh
        )
        self.num_local_routed_experts = self.routed_experts.num_local_experts
        self._ep_enabled = True
        self.ep_pg = ep_mp_mesh.get_group()

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

    def apply_compile(self):
        self.compile(fullgraph=False)

        # self.combined_forward_ep = torch.compile(self.combined_forward_ep)
        # self.combined_forward_no_ep = torch.compile(self.combined_forward_no_ep)
    

        # NOTE: the tbo might be called by the outer model directly (by block.combined_forward_ep_tbo(x, ...) instead of block(x, ...)), so need to compile it here as well
        self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo) 
        self._res_norm_attn = torch.compile(self._res_norm_attn)
        self._permute_routed_experts_unpermute = torch.compile(self._permute_routed_experts_unpermute)


    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1
        
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
    
    def combined_forward_shared_only(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Reserved for no-routed-experts case (only shared experts), equivalent to a dense model"""
        assert self.routed_experts is None
        assert self.routed_experts_router is None
        assert self.shared_experts is not None
        raise NotImplementedError("combined_forward_shared_only is not implemented")

    @torch.compiler.disable
    def async_copy_to_cpu(
            self, gpu_buf
    ):
        cpu_buf, copy_stream, dtoh_event = async_copy_to_cpu(gpu_buf, event=self._dtoh_event)
        return cpu_buf
    
    @torch.compiler.disable
    def sync_dtoh_event(self):
        assert self._dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, self._dtoh_event)
        dtoh_event.synchronize()

    def combined_forward_no_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function without EP"""
        assert self.routed_experts is not None
        assert self.routed_experts_router is not None
    
        B, S, D = x.shape

        # rename "x" to avoid confusion
        block_inp = x
        del x

        # attention 
        # + attention norm
        # + residual connection
        attn_res_out: torch.Tensor = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
        # remove attention kwargs
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)


        # routed expert router
        (
            local_x_global_routed_expert_weights, # (B, S, top_k)
            local_x_global_routed_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_routed_expert, # (num_experts, )
            routed_expert_router_aux_loss # scalar # TODO: update code
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=attn_res_out, 
            scores_only=False,
            loss_div_factor=loss_div_factor # scalar
        )
        # step 1A: DtoH token count communication
        # should start DtoH as immediately after the results are available on GPU
        local_batch_size_per_global_routed_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(local_batch_size_per_global_routed_expert, event=self._dtoh_event)  
        # local_batch_size_per_global_routed_expert_cpu = self.async_copy_to_cpu(local_batch_size_per_global_routed_expert)

        # copy_stream.synchronize() # wait for the copy to CPU to finish
        
        # shared expert router
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights, # (B, S, E_shared)
                _, 
                _, 
                _ 
            ) = self.router_forward(
                router=self.shared_experts_router,
                local_x=attn_res_out, 
                scores_only=True,  # only need scores for shared experts
                loss_div_factor=loss_div_factor # scalar
            )
        else:
            local_x_global_shared_expert_weights = None
        

        # only when grad enabled
        if torch.is_grad_enabled():
            with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                if routed_expert_router_aux_loss is not None: # TODO: update code
                    # TODO: should attach to router input or output?
                    attn_res_out = attach_auxiliary_loss(attn_res_out, routed_expert_router_aux_loss) # TODO: update code
        
        moe_inp = attn_res_out

        in_shape = moe_inp.size()

        mixed_shared_out = None
        if self.shared_experts is not None:
            # the shared experts (executed on the dense stream) need to wait for `attn_res_out` and `local_x_global_shared_expert_weights` (on the main stream) to be complete
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(),
                other_stream=torch.cuda.current_stream()
            )


            # overlap compute while waiting for the copy to CPU to finish
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out = self.shared_experts(moe_inp) # (E_shared, B, S, D)
                if self.shared_experts.num_experts == 1:
                    mixed_shared_out = shared_out.squeeze(0)
                else:
                    assert local_x_global_shared_expert_weights is not None
                    # weighted sum of the shared experts by router weights
                    # local_x_global_shared_expert_weights -> (B, S, E_shared)
                    # shared_out -> (E_shared, B, S, D)
                    _, _, E_s = local_x_global_shared_expert_weights.shape
                    local_x_global_shared_expert_weights.shape
                    mixed_shared_out = torch.bmm(
                        local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
                        shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
                    ).squeeze(1).view(B, S, D)
                
        
        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)



        routing_map = local_x_global_routed_expert_indices.view(-1, self.routed_experts_router.top_k).int()
        num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
        hidden_shape_before_permute = moe_inp.shape

        # step 2: permute the input tokens
        with nvtx.annotate("Permute", color='green'):
            permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(
                inp=moe_inp, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            )


        ####################################
        # copy_stream.synchronize() # wait for the copy to CPU to finish
        assert dtoh_event is not None 
        dtoh_event = cast(torch.cuda.Event, dtoh_event)
        dtoh_event.synchronize()
        # self.sync_dtoh_event()
        ####################################

        # step 3: MLP
        torch._dynamo.mark_dynamic(permutated_input_tokens, 0)
        # mlp_x = self.routed_experts(permutated_input_tokens, local_batch_size_per_global_routed_expert_cpu.tolist())
        mlp_x = self.routed_experts(permutated_input_tokens, local_batch_size_per_global_routed_expert_cpu)

        # step 4: unpermutate the output tokens
        with nvtx.annotate("Unpermute", color='green'):
            unpermutated_x: torch.Tensor = moe_unpermute_no_compile(
                inp=mlp_x,
                row_id_map=reversed_input_permutation_mapping,
                restore_shape=hidden_shape_before_permute,
                map_type='index',
                merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k)
            ) 
            
        x_moe = unpermutated_x.view(in_shape)

        # need to use `mixed_shared_out`
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())


        if self.shared_experts is not None:
            assert mixed_shared_out is not None
            # # weighted sum of the shared experts and routed experts
            # shared_width = self.shared_experts.num_experts * self.shared_experts.hidden_size
            # routed_active_width = self.routed_experts_router.top_k * self.routed_experts.hidden_size
            # total_width = shared_width + routed_active_width
            # shared_out_factor = shared_width / total_width
            # routed_out_factor = routed_active_width / total_width
            # mlp_out = self.merge_shared_and_routed_out(
            #     shared_out=mixed_shared_out,
            #     shared_factor=shared_out_factor,
            #     routed_out=x_moe,
            #     routed_factor=routed_out_factor
            # )
            mlp_out = x_moe + mixed_shared_out
        else:
            mlp_out = x_moe # only routed experts

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        #######################

        
        return final_out

    def merge_shared_and_routed_out(
        self,
        shared_out: torch.Tensor,
        shared_factor: float,
        routed_out: torch.Tensor,
        routed_factor: float,
    ) -> torch.Tensor:
        # Combine shared and routed outputs
        return shared_out * shared_factor + routed_out * routed_factor

    @torch.compiler.disable(recursive=False) # NOTE: 
    def fwd_routed_experts(
            self,
            global_x: torch.Tensor,
            parallel_batch_size_per_local_expert_cpu: torch.Tensor,
    ):
        assert self.routed_experts is not None
        global_x = self.routed_experts(global_x, parallel_batch_size_per_local_expert_cpu)
        return global_x

    def combined_forward_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function with EP"""
        # assert self.routed_experts is not None
        assert self.routed_experts_router is not None
        assert self.ep_enabled == True
        assert self.num_local_routed_experts is not None


        B, S, D = x.shape

        # rename "x" to avoid confusion
        block_inp = x
        del x

        # attention 
        # + attention norm
        # + residual connection
        # attn_res_out = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

        # remove attention kwargs
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)


        # routed expert router
        (
            local_x_global_routed_expert_weights, # (B, S, top_k)
            local_x_global_routed_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_routed_expert, # (num_experts, )
            routed_expert_router_aux_loss_info # tuple
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=attn_res_out, 
            scores_only=False,
            loss_div_factor=loss_div_factor # scalar
        )
        


        # the shared experts (executed on the dense stream) need to wait for `attn_res_out` and `local_x_global_shared_expert_weights` (on the main stream) to be complete
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),  
            other_stream=torch.cuda.current_stream() 
        ) 

        ########### 1. Communicate the number of tokens that will be sent to each device ###########
        with nvtx.annotate("Token count all_to_all", color='green'):
            with torch.no_grad():
                # Pass token count information to the device on which the
                # target expert resides.
                global_batch_size_per_local_expert = torch.empty_like(
                    local_batch_size_per_global_routed_expert,
                )
                global_batch_size_handle = dist.all_to_all_single(
                    global_batch_size_per_local_expert, # Gathered concatenated output tensor.
                    local_batch_size_per_global_routed_expert, # Input tensor to scatter.
                    group=self.ep_pg,
                    async_op=True,
                )
                # NOTE:
                # local_batch_size_per_global_routed_expert: 
                # (num_experts, ) -> view as: (EP, num_local_routed_experts)
                #   data[i] = how many tokens should go to global expert i (can be on other rank)
                # global_batch_size_per_local_expert: 
                # (num_experts, ) -> view as: (EP, num_local_routed_experts)
                #   data[i][j] = how many tokens from rank i will go to local expert j on this rank
                assert global_batch_size_handle is not None # because of async

        ############################################ end
        with torch.cuda.stream(self.get_dense_stream()):
            if self.shared_experts_router:
                # shared expert router
                (
                    local_x_global_shared_expert_weights, # (B, S, E_shared)
                    _, 
                    _, 
                    _ 
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out, 
                    scores_only=True,  # only need scores for shared experts
                    loss_div_factor=loss_div_factor # scalar
                )
            else:
                local_x_global_shared_expert_weights = None
        
        
        moe_inp = attn_res_out

        in_shape = moe_inp.size()
        
        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)


        ###########  3. Configure the sizes for grouped GEMM ###########

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with nvtx.annotate("Sync token count", color='green'):
            with torch.no_grad():
                global_batch_size_handle.wait()

                # Reshape to (ep_world_size, num_local_routed_experts).
                local_batch_size_per_global_routed_expert = local_batch_size_per_global_routed_expert.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
                global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
                # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                parallel_batch_size_per_local_expert = global_batch_size_per_local_expert.sum(
                    dim=0,
                    dtype=torch.long,
                )
                
                # NOTE: host-device sync here.

                # send_counts = local_batch_size_per_global_routed_expert.sum(dim=-1).to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU
                # recv_counts = global_batch_size_per_local_expert.sum(dim=-1).to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU
                # parallel_batch_size_per_local_expert_cpu = parallel_batch_size_per_local_expert.to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU

                send_counts_gpu = local_batch_size_per_global_routed_expert.sum(dim=-1)
                recv_counts_gpu = global_batch_size_per_local_expert.sum(dim=-1)
                send_counts_cpu, copy_stream, dtoh_event_send = async_copy_to_cpu(send_counts_gpu, event=self._dtoh_event_send)  
                recv_counts_cpu, copy_stream, dtoh_event_recv = async_copy_to_cpu(recv_counts_gpu, event=self._dtoh_event_recv) 
                parallel_batch_size_per_local_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(parallel_batch_size_per_local_expert, event=self._dtoh_event)  


        with torch.no_grad():
            # Construct the expert indices for the permuted tokens.
            global_x_local_expert_indices = torch.remainder(
                torch.arange(
                    self.routed_experts_router.num_experts,
                    dtype=torch.int32,
                    device=moe_inp.device,
                ),
                self.num_local_routed_experts,
            ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts

        ########### 2. permute local tokens to be ready for all-to-all communication ###########
        # dtoh_event = cast(torch.cuda.Event, dtoh_event)
        # dtoh_event.wait(self.get_dense_stream())
        # NOTE: wait() is non-blocking, 
        # it ensures the `Permute local tokens` can be submitted on the cpu, the the gpu side will only start after the DtoH is done
        # so that the shared experts forward can better overlap the all2all
        with nvtx.annotate("Permute local tokens", color='green'):
            routing_map = local_x_global_routed_expert_indices.view(-1, self.routed_experts_router.top_k).int()
            num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
            hidden_shape_before_permute = moe_inp.shape
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                inp=moe_inp, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            ) 
            
            # now permutated_local_x tokens are grouped by expert, which means tokens will go to expert id:
            # [0 , 0 , ... 1, ... 2, ..., ..., 31, 31]  (if 32 experts)
            # if EP=8, each rank has 4 experts, then tokens of
            # [0, 0, ..., 3, 3] go to rank 0,
            # [4, 4, ..., 7, 7] go to rank 1, 
            # and so on.
        ############################################ end
        if self.shared_experts is not None:
            # overlap compute while waiting for all2all
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(), 
                other_stream=torch.cuda.current_stream()
            )
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out_up, shared_out_gate = self.shared_experts.forward1(attn_res_out)
                # shared_out = self.shared_experts.forward(attn_res_out)
                # NOTE: the shared_experts forward is queued, but will not start to run until the DtoH is done
        else:
             shared_out_up, shared_out_gate = None, None
        with torch.no_grad():
            # torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
            assert dtoh_event_send
            assert dtoh_event_recv
            assert dtoh_event
            # dtoh_event_send.synchronize()
            # dtoh_event_recv.synchronize()
            dtoh_event.synchronize()
            send_counts = send_counts_cpu.tolist() # tensor to list
            recv_counts = recv_counts_cpu.tolist() # tensor to list
            tokens_received = sum(recv_counts)

        
        ###########  4. Start the all-to-all communication asynchronously ###########

        with nvtx.annotate("all2all", color='green'):

            # global_x, global_x_handle = ops.all_to_all(
            #     permutated_local_x,
            #     recv_counts,
            #     send_counts,
            #     group=self.ep_pg,
            #     async_op=True,
            # )

            permutated_local_x, global_x, global_x_handle = ops.all_to_all_async(
                permutated_local_x,
                recv_counts,
                send_counts,
                group=self.ep_pg,
            )
            

        with torch.no_grad():
            # this specifiyes for the received global tokens, which local expert they belong to
            global_x_local_expert_indices = torch.repeat_interleave(
                global_x_local_expert_indices,
                global_batch_size_per_local_expert.flatten(),
                output_size=tokens_received,
            ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts
        
        # global_x_handle.wait()
        global_x = ops.all_to_all_wait(permutated_local_x, global_x, global_x_handle)
        # del permutated_local_x
        ############################################ end
    
        global_x = self._checkpointed_permute_routed_experts_unpermute(
            global_x=global_x,
            global_x_local_expert_indices=global_x_local_expert_indices,
            parallel_batch_size_per_local_expert_cpu=parallel_batch_size_per_local_expert_cpu
        )
######################## AC START ###############################    

        # ###########  5. Permute the global tokens to be ready for MLP computation ###########
        # with nvtx.annotate("Permute global tokens for MLP", color='green'):
        #     # option 1: use moe_sort_chunks_by_index (by TE <- trition)
        #     # input_chunk_idxs = torch.arange(
        #     #     self.num_experts, device=local_x.device
        #     # )
        #     # [num_local_routed_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        #     # sort_input_by_local_experts = input_chunk_idxs.reshape(
        #     #     -1, self.num_local_routed_experts
        #     # ).T.ravel() 
        #     # split into 32 chunks (32 experts)
        #     # e.g., [ 
        #     # 0,  4,  8, 12, 16, 20, 24, 28,    --> these 8 chunks come from all 8 EP ranks, go to local expert 0
        #     # 1,  5,  9, 13, 17, 21, 25, 29,    --> these 8 chunks come from all 8 EP ranks, go to local expert 1
        #     # 2,  6, 10, 14, 18, 22, 26, 30,    --> these 8 chunks come from all 8 EP ranks, go to local expert 2
        #     # 3,  7, 11, 15, 19, 23, 27, 31     --> these 8 chunks come from all 8 EP ranks, go to local expert 3
        #     # ].  (1D tensor)

            
        #     ## chunk size is specified by `global_batch_size_per_local_expert`
            
            
        #     # e.g., global_batch_size_per_local_expert
        #     # local experts 0     1     2     3
        #     # ep0       [[3108, 5307, 5798, 4067],
        #     # ep1        [4642, 3836, 3488, 3477],
        #     # ep2        [5129, 3964, 2472, 4194],
        #     # ep3        [4266, 3191, 4511, 3841],
        #     # ep4        [5059, 5758, 4838, 3201],
        #     # ep5        [5388, 3531, 3419, 2860],
        #     # ep6        [3862, 3605, 2945, 3840],
        #     # ep7        [3960, 4624, 3414, 4406]]
            
        #     # so we want to put (3108+4642+5129+4266+5059+5388+3862+3960) tokens to local expert 0,
        #     # and so on
            
        #     # global_x = moe_sort_chunks_by_index_no_compile(
        #     #     inp=global_x,
        #     #     split_sizes=global_batch_size_per_local_expert.ravel(),
        #     #     sorted_index=sort_input_by_local_experts
        #     # ) # type: ignore

        #     # option 2: use moe_permute (by TE), and pretend topk is 1
        #     routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
        #     num_out_tokens2 = routing_map2.size(0) * 1 # dropless
        #     hidden_shape_before_permute2 = global_x.shape
        #     global_x, reversed_global_x_permutation_mapping = moe_permute_no_compile(
        #         inp=global_x, 
        #         routing_map=routing_map2, 
        #         num_out_tokens=num_out_tokens2, 
        #         map_type='index'
        #     )
                
                
        # ############################################ end

        
        # ########## 6. MLP forwrad ###########

        # # global_x = self.routed_experts(global_x, parallel_batch_size_per_local_expert_cpu.tolist())
        # # global_x = self.routed_experts(global_x, parallel_batch_size_per_local_expert_cpu)
        # global_x = self.fwd_routed_experts(global_x, parallel_batch_size_per_local_expert_cpu)

        # ############################################ end
        
        
        # ############ 7. Unpermute the output tokens to be ready for all-to-all communication ##########
        # with nvtx.annotate("Unpermute global tokens", color='green'):
        #     # option 1: use moe_sort_chunks_by_index (by TE <- trition)
        #     # restore_output_by_local_experts = input_chunk_idxs.reshape(
        #     #     self.num_local_routed_experts, -1
        #     # ).T.ravel() # [ 0,  8, 16, 24,  1,  9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31]
        #     # global_x = moe_sort_chunks_by_index_no_compile(
        #     #     global_x, 
        #     #     split_sizes=global_batch_size_per_local_expert.T.ravel(),
        #     #     sorted_index=restore_output_by_local_experts
        #     # )
            
        #     # option 2: use moe_unpermute (by TE)
        #     global_x = moe_unpermute_no_compile(
        #         inp=global_x,
        #         row_id_map=reversed_global_x_permutation_mapping,
        #         merging_probs=None,
        #         restore_shape=hidden_shape_before_permute2,
        #         map_type='index',
        #     ) 
        # ############################################ end
######################## AC END ###############################    
            
    
        ########## 8. reverse_all_to_all ###########
        before_rev_all2all_event = torch.cuda.current_stream().record_event(
            event=self._before_rev_all2all_event # type: ignore
        ) 
        with nvtx.annotate("reverse_all_to_all", color='green'):
            global_x = cast(torch.Tensor, global_x)
            # local_x, local_x_handle = ops.all_to_all(
            #     global_x,
            #     send_counts,
            #     recv_counts,
            #     group=self.ep_pg,
            #     async_op=True,
            # )
            global_x, local_x, local_x_handle = ops.all_to_all_async(
                global_x,
                send_counts,
                recv_counts,
                group=self.ep_pg,
            )

        if self.shared_experts is not None:
            # variables from forward1
            assert shared_out_up is not None
            assert shared_out_gate is not None

            before_rev_all2all_event.wait(self.get_dense_stream()) # the `merge_shared` should not start until the start of the reverse all2all to better overlap it
            # merge shared experts when waiting for all2all
            with nvtx.annotate("merge_shared", color='purple'):
                with torch.cuda.stream(self.get_dense_stream()):

                    shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
                    if self.shared_experts_router:
                        assert local_x_global_shared_expert_weights is not None
                        # weighted sum of the shared experts by router weights
                        # local_x_global_shared_expert_weights -> (B, S, E_shared)
                        # shared_out -> (E_shared, B, S, D)
                        _, _, E_s = local_x_global_shared_expert_weights.shape
                        local_x_global_shared_expert_weights.shape
                        mixed_shared_out = torch.bmm(
                            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
                            shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
                        ).squeeze(1).view(B, S, D)
                    else:
                        mixed_shared_out = shared_out.squeeze(0)
        else:
            mixed_shared_out = None

        # local_x_handle.wait()
        local_x = ops.all_to_all_wait(global_x, local_x, local_x_handle)
        
        # del global_x # done with global tokens
        ############################################ end
        
        
        ############ 9. Unpermute the (local) tokens returned by all-to-all communication ##########
        with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
            if self.checkpoint_second_unpermute:
                local_x = checkpoint(
                    moe_unpermute_no_compile,
                    inp=local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute,
                    map_type='index',
                    use_reentrant=False
                )
            else:
                local_x = moe_unpermute_no_compile(
                    inp=local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute,
                    map_type='index',
                )
        ############################################ end
    
        
        local_x = local_x.view(in_shape)

        # return local_x, dense_out


        # need to use `mixed_shared_out`
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) 



        # weighted sum of the shared experts and routed experts
        if self.shared_experts is not None:
            assert mixed_shared_out is not None
            # assert self.routed_experts is not None
            # shared_width = self.shared_experts.num_experts * self.shared_experts.hidden_size
            # routed_active_width = self.routed_experts_router.top_k * self.routed_experts.hidden_size
            # total_width = shared_width + routed_active_width
            # shared_out_factor = shared_width / total_width
            # routed_out_factor = routed_active_width / total_width
            # mlp_out = self.merge_shared_and_routed_out(
            #     shared_out= mixed_shared_out,
            #     shared_factor=shared_out_factor,
            #     routed_out=local_x,
            #     routed_factor=routed_out_factor
            # )
            mlp_out = local_x + mixed_shared_out
        else:
            mlp_out = local_x

        final_out = attn_res_out + self.feed_forward_norm(mlp_out)

        #######################

        # attach aux loss
        if torch.is_grad_enabled(): # only when grad enabled
            with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                if routed_expert_router_aux_loss_info is not None:
                    # NOTE: this part cpu runtime > gpu runtime, so it's moved from directly after router_forward to here
                    # because we need to avoid stalling the gpu stream
                    # gpu stream is generally more ahead of cpu thread at the end of the block, hence less harmful to put it here
                    routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(*routed_expert_router_aux_loss_info)

                    # NOTE: the attach only writes 1.0 to the aux loss grad slot, so it should not matter where to attach
                    final_out = attach_auxiliary_loss(final_out, routed_expert_router_aux_loss)
        
        return final_out
    


    def _permute_routed_experts_unpermute(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu,
        hidden_shape_before_permute2,
        reversed_global_x_permutation_mapping,
    ):
        assert self.routed_experts is not None
        

        ## 6. MLP forwrad ##
        global_x = self.routed_experts(global_x, parallel_batch_size_per_local_expert_cpu)

        ###################################
#         assert isinstance(parallel_batch_size_per_local_expert_cpu, torch.Tensor), "only accept Tensor for batch_size_per_expert"

#         assert parallel_batch_size_per_local_expert_cpu.device.type == 'cpu', "batch_size_per_expert must be on cpu"
#         batch_size_per_expert_tensor = parallel_batch_size_per_local_expert_cpu.to(dtype=torch.int64)  # NOTE: int64 required for grouped_gemm

#         if global_x.numel() == 0:
#             return global_x
        
#         w_up_gate = self.routed_experts.w_up_gate # (E, H, 2D)
#         w_down = self.routed_experts.w_down # (E, H, D)
#         up_gate = gmm_no_compile(global_x, w_up_gate, batch_size_per_expert_tensor, trans_b=True) # -> (BS, 2H)
#         up_gate = cast(torch.Tensor, up_gate)  # ensure type is Tensor

# ### START AC ###
#         h = self.routed_experts.chunk_and_activate(up_gate) # -> (BS, H)
        
#         global_x = gmm_no_compile(h, w_down, batch_size_per_expert_tensor, trans_b=False) # -> (BS, H)
        
        ###################################
        
        ## 7. Unpermute the output tokens to be ready for all-to-all communication ##
        with nvtx.annotate("Unpermute global tokens", color='green'):
            if self.routed_experts.num_local_experts == 1:
                pass  # skip unpermute if only one local expert
            else:
                global_x = moe_unpermute_no_compile(
                    inp=global_x,
                    row_id_map=reversed_global_x_permutation_mapping,
                    merging_probs=None,
                    restore_shape=hidden_shape_before_permute2,
                    map_type='index',
                ) 
### END AC ###
        # global_x = checkpoint(
        #     self._act_down_unpermute,
        #     up_gate,
        #     batch_size_per_expert_tensor,
        #     reversed_global_x_permutation_mapping,
        #     hidden_shape_before_permute2,
        #     use_reentrant=False,
        # )
### REPLACE ###

        return global_x
    
    def _act_down_unpermute(self, up_gate, batch_size_per_expert_tensor, reversed_global_x_permutation_mapping, hidden_shape_before_permute2):
        h = self.routed_experts.chunk_and_activate(up_gate) # -> (BS, H)
        
        global_x = gmm_no_compile(h, self.routed_experts.w_down, batch_size_per_expert_tensor, trans_b=False) # -> (BS, H)
        
        ###################################
        
        ## 7. Unpermute the output tokens to be ready for all-to-all communication ##
        with nvtx.annotate("Unpermute global tokens", color='green'):
            if self.routed_experts.num_local_experts == 1:
                pass  # skip unpermute if only one local expert
            else:
                global_x = moe_unpermute_no_compile(
                    inp=global_x,
                    row_id_map=reversed_global_x_permutation_mapping,
                    merging_probs=None,
                    restore_shape=hidden_shape_before_permute2,
                    map_type='index',
                ) 

        return global_x

    def _checkpointed_permute_routed_experts_unpermute(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu
    ) -> torch.Tensor:
        # don't need to checkpoint the permute step because it does not save input for backward

        ##  5. Permute the global tokens to be ready for MLP computation ##
        with nvtx.annotate("Permute global tokens for MLP", color='green'):
            routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
            num_out_tokens2 = routing_map2.size(0) * 1 # dropless
            hidden_shape_before_permute2 = global_x.shape
            if self.routed_experts.num_local_experts == 1:
                reversed_global_x_permutation_mapping = None
            else:
                global_x, reversed_global_x_permutation_mapping = moe_permute_no_compile(
                    inp=global_x, 
                    routing_map=routing_map2, 
                    num_out_tokens=num_out_tokens2, 
                    map_type='index'
                )

        if self.checkpoint_permute_moe_unpermute:
            out = checkpoint(
                self._permute_routed_experts_unpermute, 
                global_x,
                global_x_local_expert_indices,
                parallel_batch_size_per_local_expert_cpu, 
                hidden_shape_before_permute2,
                reversed_global_x_permutation_mapping,
                use_reentrant=False, 
            )
            return cast(torch.Tensor, out)
        else:
            return self._permute_routed_experts_unpermute(
                global_x,
                global_x_local_expert_indices,
                parallel_batch_size_per_local_expert_cpu, 
                hidden_shape_before_permute2,
                reversed_global_x_permutation_mapping
            )

    
    def _res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        attn_res_out = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
        return attn_res_out

    def _checkpointed_res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        if self.checkpoint_attn:
            out = checkpoint(
                self._res_norm_attn,
                block_inp,
                use_reentrant=False,
                **kwargs,
            )
            return cast(torch.Tensor, out)
        else:
            return self._res_norm_attn(block_inp, **kwargs)

    def checkpointed_combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: Dict,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        if self.checkpoint_combined_ep_tbo:
            out = checkpoint(
                self.combined_forward_ep_tbo,
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                use_reentrant=False,
                **kwargs,
            )
            return cast(Tuple[torch.Tensor, Dict], out)
        else:
            return self.combined_forward_ep_tbo(
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )
        
    def combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: Dict,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        assert self.routed_experts is not None
        assert self.routed_experts_router is not None
        assert self.ep_enabled == True
        assert self.num_local_routed_experts is not None

        B, S, D = x0.shape


        # rename "x" to avoid confusion
        block_inp = x0
        del x0

        with torch.no_grad():
            # Construct the expert indices for the permuted tokens.
            global_x_local_expert_indices_0 = torch.remainder(
                torch.arange(
                    self.routed_experts.num_experts,
                    dtype=torch.int32,
                    device=block_inp.device,
                ),
                self.num_local_routed_experts,
            ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts
        

        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            if x1_is_fresh:
                local_x1, local_x_handle1 = None, None
                last_block = None
            else:
                global_x1 = x1_ctx['global_x1']
                send_counts1 = x1_ctx['send_counts1']
                recv_counts1 = x1_ctx['recv_counts1']
                # tokens_received1 = x1_ctx['tokens_received1']

                last_block = cast(MoEFusedV2TransformerBlock, x1_ctx['last_block'])

                assert last_block.routed_experts_router is not None
                # finish reverse all2all and other ops for x1
                with nvtx.annotate("reverse_all_to_all", color='green'):
                    global_x1 = cast(torch.Tensor, global_x1)
                    global_x1, local_x1, local_x_handle1 = ops.all_to_all_async(
                    # local_x1, local_x_handle1 = ops.all_to_all(
                        global_x1,
                        send_counts1,
                        recv_counts1,
                        group=last_block.ep_pg,
                        # async_op=True,
                    )


        ############################ END: TBO 1 ########################

        with nvtx.annotate("TBO-0", color='purple'):
            # attention 
            # + attention norm
            # + residual connection
            # attn_res_out = block_inp + self.attention_norm(self.attention(block_inp, **kwargs))
            attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)
            # routed expert router
            (
                local_x_global_routed_expert_weights, # (B, S, top_k)
                local_x_global_routed_expert_indices, # (B, S, top_k)
                local_batch_size_per_global_routed_expert, # (num_experts, )
                routed_expert_router_aux_loss # scalar # TODO: update code
            ) = self.router_forward(
                router=self.routed_experts_router,
                local_x=attn_res_out, 
                scores_only=False,
                loss_div_factor=loss_div_factor # scalar
            )
        
            # attach aux loss
            if torch.is_grad_enabled(): # only when grad enabled
                with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                    if routed_expert_router_aux_loss is not None:
                        attn_res_out = attach_auxiliary_loss(attn_res_out, routed_expert_router_aux_loss) # TODO: update code


            ########### 1. Communicate the number of tokens that will be sent to each device ###########
            with nvtx.annotate("Token count all_to_all", color='green'):
                with torch.no_grad():
                    # Pass token count information to the device on which the
                    # target expert resides.
                    global_batch_size_per_local_expert = torch.empty_like(
                        local_batch_size_per_global_routed_expert,
                    )
                    global_batch_size_handle = dist.all_to_all_single(
                        global_batch_size_per_local_expert, # Gathered concatenated output tensor.
                        local_batch_size_per_global_routed_expert, # Input tensor to scatter.
                        group=self.ep_pg,
                        async_op=True,
                    )

                    assert global_batch_size_handle is not None # because of async

            ############################################ end

            if self.shared_experts_router:
                # shared expert router
                (
                    local_x_global_shared_expert_weights, # (B, S, E_shared)
                    _, 
                    _, 
                    _ 
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out, 
                    scores_only=True,  # only need scores for shared experts
                    loss_div_factor=loss_div_factor # scalar
                )
            else:
                local_x_global_shared_expert_weights = None
            

            # forward shared experts
            if self.shared_experts is not None:
                shared_out = self.shared_experts.forward(attn_res_out)
                with nvtx.annotate("merge_shared", color='purple'):
                    # shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
                    if self.shared_experts_router:
                        assert local_x_global_shared_expert_weights is not None
                        # weighted sum of the shared experts by router weights
                        # local_x_global_shared_expert_weights -> (B, S, E_shared)
                        # shared_out -> (E_shared, B, S, D)
                        _, _, E_s = local_x_global_shared_expert_weights.shape
                        local_x_global_shared_expert_weights.shape
                        mixed_shared_out = torch.bmm(
                            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B*S, 1, E_s),            # (BS, 1, E), 
                            shared_out.permute(1, 2, 0, 3).contiguous().view(B*S, E_s, D)              # (BS, E, D)
                        ).squeeze(1).view(B, S, D)
                    else:
                        mixed_shared_out = shared_out.squeeze(0)
            else:
                mixed_shared_out = None

            
            moe_inp = attn_res_out

            in_shape = moe_inp.size()
            
            moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)


            ###########  3. Configure the sizes for grouped GEMM ###########

            # Compute the number of tokens that will be received from each
            # device and permute the input data across the devices.
            with nvtx.annotate("Sync token count", color='green'):
                with torch.no_grad():
                    global_batch_size_handle.wait()

                    # Reshape to (ep_world_size, num_local_routed_experts).
                    local_batch_size_per_global_routed_expert = local_batch_size_per_global_routed_expert.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                    parallel_batch_size_per_local_expert = global_batch_size_per_local_expert.sum(
                        dim=0,
                        dtype=torch.long,
                    )
                    

                    send_counts_gpu = local_batch_size_per_global_routed_expert.sum(dim=-1)
                    recv_counts_gpu = global_batch_size_per_local_expert.sum(dim=-1)
                    send_counts_cpu, _, dtoh_event_send = async_copy_to_cpu(send_counts_gpu, event=self._dtoh_event_send)  
                    recv_counts_cpu, _, dtoh_event_recv = async_copy_to_cpu(recv_counts_gpu, event=self._dtoh_event_recv) 
                    parallel_batch_size_per_local_expert_cpu, _, dtoh_event = async_copy_to_cpu(parallel_batch_size_per_local_expert, event=self._dtoh_event)  



            ########### 2. permute local tokens to be ready for all-to-all communication ###########

            with nvtx.annotate("Permute local tokens", color='green'):
                routing_map = local_x_global_routed_expert_indices.view(-1, self.routed_experts_router.top_k).int()
                num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k # dropless
                hidden_shape_before_permute = moe_inp.shape
                permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                    inp=moe_inp, 
                    routing_map=routing_map, 
                    num_out_tokens=num_out_tokens, 
                    map_type='index'
                ) 



            with torch.no_grad():
                # torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
                assert dtoh_event_send
                assert dtoh_event_recv
                assert dtoh_event
                # dtoh_event_send.synchronize()
                # dtoh_event_recv.synchronize()
                dtoh_event.synchronize()
                send_counts = send_counts_cpu.tolist() # tensor to list
                recv_counts = recv_counts_cpu.tolist() # tensor to list
                tokens_received = sum(recv_counts)

            with nvtx.annotate("all2all", color='green'):
                permutated_local_x, global_x, global_x_handle = ops.all_to_all_async(
                # global_x, global_x_handle = ops.all_to_all(
                    permutated_local_x,
                    recv_counts,
                    send_counts,
                    group=self.ep_pg,
                    # async_op=True,
                )

            with torch.no_grad():
                # this specifiyes for the received global tokens, which local expert they belong to
                global_x_local_expert_indices = torch.repeat_interleave(
                    global_x_local_expert_indices_0,
                    global_batch_size_per_local_expert.flatten(),
                    output_size=tokens_received,
                ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts

        with nvtx.annotate("TBO-1", color='orange'):
            if x1_is_fresh:
                x1 = x1_ctx['x1']
                assert x1.shape == (B, S, D)
                block_inp1 = x1
                del x1
            else:
                reversed_local_x_permutation_mapping1 = x1_ctx['reversed_local_x_permutation_mapping1']
                local_x_global_routed_expert_weights1 = x1_ctx['local_x_global_routed_expert_weights1']
                hidden_shape_before_permute1 = x1_ctx['hidden_shape_before_permute1']
                in_shape1 = x1_ctx['in_shape1']
                mixed_shared_out1 = x1_ctx['mixed_shared_out1']
                attn_res_out1 = x1_ctx['attn_res_out1']
                
                assert last_block is not None
                assert local_x_handle1 is not None
                assert local_x1 is not None
                assert last_block.routed_experts_router is not None
                
                # local_x_handle1.wait()
                local_x1 = ops.all_to_all_wait(global_x1, local_x1, local_x_handle1)

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
                    assert last_block.routed_experts is not None
                    # shared_out_factor1 = last_block.shared_experts.num_experts / (last_block.routed_experts_router.top_k + last_block.shared_experts.num_experts)
                    # routed_out_factor1 = last_block.routed_experts_router.top_k / (last_block.routed_experts_router.top_k + last_block.shared_experts.num_experts)

                    # shared_width1 = last_block.shared_experts.num_experts * last_block.shared_experts.hidden_size
                    # routed_active_width1 = last_block.routed_experts_router.top_k * last_block.routed_experts.hidden_size
                    # total_width1 = shared_width1 + routed_active_width1
                    # shared_out_factor1 = shared_width1 / total_width1
                    # routed_out_factor1 = routed_active_width1 / total_width1

                    # mlp_out1 = last_block.merge_shared_and_routed_out(
                    #     shared_out= mixed_shared_out1,
                    #     shared_factor=shared_out_factor1,
                    #     routed_out=local_x1,
                    #     routed_factor=routed_out_factor1
                    # )
                    mlp_out1 = local_x1 + mixed_shared_out1
                else:
                    mlp_out1 = local_x1

                block_inp1 = attn_res_out1 + last_block.feed_forward_norm(mlp_out1)
            
            ########## x1 last step done ##########

            # attention 
            # + attention norm
            # + residual connection
            # attn_res_out1 = block_inp1 + self.attention_norm(self.attention(block_inp1, **kwargs))
            attn_res_out1 = self._checkpointed_res_norm_attn(block_inp1, **kwargs)

            # routed expert router
            (
                local_x_global_routed_expert_weights1, # (B, S, top_k)
                local_x_global_routed_expert_indices1, # (B, S, top_k)
                local_batch_size_per_global_routed_expert1, # (num_experts, )
                routed_expert_router_aux_loss1 # scalar # TODO: update code
            ) = self.router_forward(
                router=self.routed_experts_router,
                local_x=attn_res_out1, 
                scores_only=False,
                loss_div_factor=loss_div_factor # scalar
            )
            
            # attach aux loss
            if torch.is_grad_enabled(): # only when grad enabled
                with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                    if routed_expert_router_aux_loss1 is not None: # TODO: update code
                        attn_res_out1 = attach_auxiliary_loss(attn_res_out1, routed_expert_router_aux_loss1)
            


            with nvtx.annotate("Token count all_to_all", color='green'):
                with torch.no_grad():
                    # Pass token count information to the device on which the
                    # target expert resides.
                    global_batch_size_per_local_expert1 = torch.empty_like(
                        local_batch_size_per_global_routed_expert1,
                    )
                    global_batch_size_handle1 = dist.all_to_all_single(
                        global_batch_size_per_local_expert1, # Gathered concatenated output tensor.
                        local_batch_size_per_global_routed_expert1, # Input tensor to scatter.
                        group=self.ep_pg,
                        async_op=True,
                    )

                    assert global_batch_size_handle1 is not None # because of async



            if self.shared_experts_router:
                # shared expert router
                (
                    local_x_global_shared_expert_weights1, # (B, S, E_shared)
                    _, 
                    _, 
                    _ 
                ) = self.router_forward(
                    router=self.shared_experts_router,
                    local_x=attn_res_out1, 
                    scores_only=True,  # only need scores for shared experts
                    loss_div_factor=loss_div_factor # scalar
                )
            else:
                local_x_global_shared_expert_weights1 = None
            

            if self.shared_experts is not None:
                shared_out1 = self.shared_experts.forward(attn_res_out1)
                
                with nvtx.annotate("merge_shared", color='purple'):
                    if self.shared_experts_router:
                        assert local_x_global_shared_expert_weights1 is not None
                        # weighted sum of the shared experts by router weights
                        # local_x_global_shared_expert_weights -> (B, S, E_shared)
                        # shared_out -> (E_shared, B, S, D)
                        _, _, E_s1 = local_x_global_shared_expert_weights1.shape
                        local_x_global_shared_expert_weights1.shape
                        mixed_shared_out1 = torch.bmm(
                            local_x_global_shared_expert_weights1.to(shared_out1.dtype).reshape(B*S, 1, E_s1),            # (BS, 1, E), 
                            shared_out1.permute(1, 2, 0, 3).contiguous().view(B*S, E_s1, D)              # (BS, E, D)
                        ).squeeze(1).view(B, S, D)
                    else:
                        mixed_shared_out1 = shared_out1.squeeze(0)
            else:
                mixed_shared_out1 = None
            
            moe_inp1 = attn_res_out1

            in_shape1 = moe_inp1.size()
            
            moe_inp1 = moe_inp1.view(-1, in_shape1[-1])  # (B*S, D)


            ###########  3. Configure the sizes for grouped GEMM ###########

            # Compute the number of tokens that will be received from each
            # device and permute the input data across the devices.
            with nvtx.annotate("Sync token count", color='green'):
                with torch.no_grad():
                    global_batch_size_handle1.wait()

                    # Reshape to (ep_world_size, num_local_routed_experts).
                    local_batch_size_per_global_routed_expert1 = local_batch_size_per_global_routed_expert1.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    global_batch_size_per_local_expert1 = global_batch_size_per_local_expert1.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                    # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                    parallel_batch_size_per_local_expert1 = global_batch_size_per_local_expert1.sum(
                        dim=0,
                        dtype=torch.long,
                    )
                    

                    send_counts_gpu1 = local_batch_size_per_global_routed_expert1.sum(dim=-1)
                    recv_counts_gpu1 = global_batch_size_per_local_expert1.sum(dim=-1)
                    send_counts_cpu1, _, dtoh_event_send1 = async_copy_to_cpu(send_counts_gpu1, event=self._dtoh_event_send1)  
                    recv_counts_cpu1, _, dtoh_event_recv1 = async_copy_to_cpu(recv_counts_gpu1, event=self._dtoh_event_recv1) 
                    parallel_batch_size_per_local_expert_cpu1, _, dtoh_event1 = async_copy_to_cpu(parallel_batch_size_per_local_expert1, event=self._dtoh_event1)  


            ########### 2. permute local tokens to be ready for all-to-all communication ###########

            with nvtx.annotate("Permute local tokens", color='green'):
                routing_map1 = local_x_global_routed_expert_indices1.view(-1, self.routed_experts_router.top_k).int()
                num_out_tokens1 = routing_map1.size(0) * self.routed_experts_router.top_k # dropless
                hidden_shape_before_permute1 = moe_inp1.shape
                permutated_local_x1, reversed_local_x_permutation_mapping1 = moe_permute_no_compile(
                    inp=moe_inp1, 
                    routing_map=routing_map1, 
                    num_out_tokens=num_out_tokens1, 
                    map_type='index'
                ) 

            #### end



            with torch.no_grad():
                # torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
                # assert dtoh_event_send1
                # assert dtoh_event_recv1
                assert dtoh_event1
                # dtoh_event_send1.synchronize()
                # dtoh_event_recv1.synchronize()
                dtoh_event1.synchronize()
                send_counts1 = send_counts_cpu1.tolist() # tensor to list
                recv_counts1 = recv_counts_cpu1.tolist() # tensor to list
                tokens_received1 = sum(recv_counts1)
        ############################ END: TBO 1 ########################

        with nvtx.annotate("TBO-0", color='purple'):
            # global_x_handle.wait()
            global_x = ops.all_to_all_wait(permutated_local_x, global_x, global_x_handle)


        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            with nvtx.annotate("all2all", color='green'):
                # global_x1, global_x_handle1 = ops.all_to_all(
                permutated_local_x1, global_x1, global_x_handle1 = ops.all_to_all_async(
                    permutated_local_x1,
                    recv_counts1,
                    send_counts1,
                    group=self.ep_pg,
                    # async_op=True
                )
            
            with torch.no_grad():
                # this specifiyes for the received global tokens, which local expert they belong to
                global_x_local_expert_indices1 = torch.repeat_interleave(
                    global_x_local_expert_indices_0,
                    global_batch_size_per_local_expert1.flatten(),
                    output_size=tokens_received1,
                ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts
        ############################ END: TBO 1 ########################



        with nvtx.annotate("TBO-0", color='purple'):
            global_x = self._checkpointed_permute_routed_experts_unpermute(
                global_x=global_x,
                global_x_local_expert_indices=global_x_local_expert_indices,
                parallel_batch_size_per_local_expert_cpu=parallel_batch_size_per_local_expert_cpu
            )
            
        
                    
        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            # global_x_handle1.wait()
            global_x1 = ops.all_to_all_wait(permutated_local_x1, global_x1, global_x_handle1)

        ############################ END: TBO 1 ########################
    
        with nvtx.annotate("TBO-0", color='purple'):

            ########## 8. reverse_all_to_all ###########

            with nvtx.annotate("reverse_all_to_all", color='green'):
                global_x = cast(torch.Tensor, global_x)
                global_x, local_x, local_x_handle = ops.all_to_all_async(
                # local_x, local_x_handle = ops.all_to_all(
                    global_x,
                    send_counts,
                    recv_counts,
                    group=self.ep_pg,
                    # async_op=True
                )


        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            global_x1 = self._checkpointed_permute_routed_experts_unpermute(
                global_x=global_x1,
                global_x_local_expert_indices=global_x_local_expert_indices1,
                parallel_batch_size_per_local_expert_cpu=parallel_batch_size_per_local_expert_cpu1
            )

        ############################ END: TBO 1 ########################

        with nvtx.annotate("TBO-0", color='purple'):
            # local_x_handle.wait()
            local_x = ops.all_to_all_wait(global_x, local_x, local_x_handle)

            # del global_x # done with global tokens
            ############################################ end
            
            
            ############ 9. Unpermute the (local) tokens returned by all-to-all communication ##########
            with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
                local_x = moe_unpermute_no_compile(
                    inp=local_x,
                    row_id_map=reversed_local_x_permutation_mapping,
                    merging_probs=local_x_global_routed_expert_weights.view(-1, self.routed_experts_router.top_k),
                    restore_shape=hidden_shape_before_permute,
                    map_type='index',
                )
            ############################################ end
        
            
            local_x = local_x.view(in_shape)


            # weighted sum of the shared experts and routed experts
            if self.shared_experts is not None:
                assert mixed_shared_out is not None
                # shared_out_factor = self.shared_experts.num_experts / (self.routed_experts_router.top_k + self.shared_experts.num_experts)
                # routed_out_factor = self.routed_experts_router.top_k / (self.routed_experts_router.top_k + self.shared_experts.num_experts)
    
                # shared_width = self.shared_experts.num_experts * self.shared_experts.hidden_size
                # routed_active_width = self.routed_experts_router.top_k * self.routed_experts.hidden_size
                # total_width = shared_width + routed_active_width
                # shared_out_factor = shared_width / total_width
                # routed_out_factor = routed_active_width / total_width
                
                # mlp_out = self.merge_shared_and_routed_out(
                #     shared_out= mixed_shared_out,
                #     shared_factor=shared_out_factor,
                #     routed_out=local_x,
                #     routed_factor=routed_out_factor
                # )
                mlp_out = local_x + mixed_shared_out
            else:
                mlp_out = local_x

            final_out = attn_res_out + self.feed_forward_norm(mlp_out)

            #######################


        ############################ TBO 1 ############################
        with nvtx.annotate("TBO-1", color='orange'):
            x1_ctx = {
                "global_x1": global_x1,
                "send_counts1": send_counts1,
                "recv_counts1": recv_counts1,
                # "tokens_received1": tokens_received1,
                "reversed_local_x_permutation_mapping1": reversed_local_x_permutation_mapping1,
                "local_x_global_routed_expert_weights1": local_x_global_routed_expert_weights1,
                "hidden_shape_before_permute1": hidden_shape_before_permute1,
                "in_shape1": in_shape1,
                "mixed_shared_out1": mixed_shared_out1,
                "attn_res_out1": attn_res_out1,
                "last_block": self,
            }



        ############################ END: TBO 1 ########################

        
        return (
            final_out,
            x1_ctx, 
        )

    
    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        if self.shared_experts_router:
            self.shared_experts_router.post_batch(dry_run=dry_run)
        if self.routed_experts_router:
            self.routed_experts_router.post_batch(dry_run=dry_run)
