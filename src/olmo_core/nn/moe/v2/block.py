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

from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.utils import get_or_init_stream
from olmo_core.exceptions import OLMoConfigurationError
from ...transformer.block import (
    TransformerBlockBase,
)
from ...attention import AttentionConfig, RingAttentionLoadBalancerType
from ...buffer_cache import BufferCache
from ...feed_forward import FeedForward, FeedForwardConfig
from ...functional import l2_normalize
from ...layer_norm import LayerNormConfig as LayerNormConfigV1
from ...layer_norm import LayerNormType, LayerNorm, RMSNorm, FusedRMSNorm, L2Norm

from ...moe import MoEConfig, MoERouterType, MoERouterGatingFunction
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
import torch.nn.functional as F

import grouped_gemm  # type: ignore

@dataclass
class SharedExpertsConfig(Config):

    """
    Configuration for shared experts in a MoE block.
    """

    # Input (and output) dimension of the experts
    d_model: int

    # Hidden (intermediate) dimension of the experts
    hidden_size: int

    # Number of shared experts (can be >= 1)
    num_experts: int
    
    # Whether to use bias in the experts
    bias: bool
    
    # default dtype for the experts
    dtype: DType
    

    def build(self) -> "SharedExperts":
        kwargs = self.as_dict()
        return SharedExperts(**kwargs)
    
    def num_params(self) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """

        params = 3 * self.d_model * self.hidden_size # up, gate, down
        if self.bias:
            params += 2 * self.hidden_size # up and gate bias
            params += self.d_model  # down bias

        params *= self.num_experts # for each expert
        
        return params 

class SharedExperts(nn.Module):
    """ 
    Shared experts module for MoE blocks.
    
    Shared experts work like a regular feed-forward but can support more than 1 expert.
    All experts will have the same number of input tokens, so it's possible that we concatenate
    the weights of all experts and use a single linear layer to process the input.
    """
    
    def __init__(self, 
                 d_model: int, 
                 hidden_size: int, 
                 num_experts: int, 
                 bias: bool, 
                 dtype: DType
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.w_up_gate = nn.Linear(
            d_model,
            2 * num_experts * hidden_size,  # 2 for up and gate
            bias=bias,
            dtype=dtype,
        )
        self.w_down = nn.Linear(
            num_experts * hidden_size,
            d_model,
            bias=bias,
            dtype=dtype,
        )

    @nvtx.annotate("SharedExperts.forward", color='blue')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input shape: (B, S, D)
        output shape: (num_experts, B, S, D)
        """
        B, S, D = x.shape
        up, gate = self.w_up_gate(x).chunk(2, dim=0)
        h = F.silu(up) * gate
        y = self.w_down(h)
        return y.view(self.num_experts, B, S, D)
        
        
class RoutedExpertsConfig(Config):
    """Configuration for routed experts in a MoE block."""
    
    # Input (and output) dimension of the experts
    d_model: int

    # Hidden (intermediate) dimension of the experts
    hidden_size: int

    # Number of routed experts
    num_experts: int
    
    # Whether to use bias in the experts
    bias: bool
    
    # default dtype for the experts
    dtype: DType
    

    def build(self) -> "RoutedExperts":
        kwargs = self.as_dict()
        return RoutedExperts(**kwargs)
    
    def num_params(self) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """

        params = 3 * self.d_model * self.hidden_size # up, gate, down
        if self.bias:
            params += 2 * self.hidden_size # up and gate bias
            params += self.d_model  # down bias

        params *= self.num_experts # for each expert
        
        return params
    
class RoutedExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        bias: bool,
        dtype: DType,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.w_up_gate = nn.Linear(
            d_model,
            2 * num_experts * hidden_size,  # 2 for up and gate
            bias=bias,
            dtype=dtype,
        )
        self.w_down = nn.Linear(
            num_experts * hidden_size,
            d_model,
            bias=bias,
            dtype=dtype,
        )
        self.gmm_ops = grouped_gemm.ops.gmm
        
        
    @nvtx.annotate("RoutedExperts.forward", color="blue")
    def forward(self, x: torch.Tensor, batch_size_per_expert: List) -> torch.Tensor:

        assert isinstance(batch_size_per_expert, List), "only accept List for batch_size_per_expert"
        batch_size_per_expert_tensor = torch.tensor(
            batch_size_per_expert, 
            device='cpu', 
            dtype=torch.int64,  # NOTE: int64 required for grouped_gemm
        )

        if x.numel() == 0:
            return x
        
        w_up_gate = self.w_up_gate
        w_down = self.w_down
        up_gate = self.gmm_ops(x, w_up_gate, batch_size_per_expert_tensor, trans_b=True)
        up, gate = up_gate.chunk(2, dim=0)
        h = F.silu(up) * gate
        
        down = self.gmm_ops(h, w_down, batch_size_per_expert_tensor, trans_b=True)
            
        return down
    


@dataclass
class LayerNormConfigV2(Config):
    # NOTE: LayerNormConfigV2 is the same as LayerNormConfig, 
    # but with "size" as a required field (instead of being passed at build time).
    
    """
    A config for conveniently building any one of the different layer norm classes.

    See the :class:`LayerNorm` subclasses to learn which fields are valid for each implementation.
    """
    size: int
    name: LayerNormType = LayerNormType.default
    """
    The name of the implementation.
    """
    eps: Optional[float] = None
    elementwise_affine: Optional[bool] = None
    bias: Optional[bool] = None
    full_precision: Optional[bool] = None
    dtype: Optional[DType] = None

    def num_params(self) -> int:
        """
        The number of parameters in the module once built.

        :param size: The size of the input along the dimension to be normalized.
        """
        elementwise_affine = (
            self.elementwise_affine
            if self.elementwise_affine is not None
            else self.name != LayerNormType.l2_norm
        )
        bias = self.bias if self.bias is not None else self.name != LayerNormType.l2_norm
        ln_params = 0
        if elementwise_affine:
            ln_params += self.size
            if bias:
                ln_params += self.size
        return ln_params

    def build(self, init_device: str = "cpu") -> "LayerNorm":
        """
        Construct the corresponding LayerNorm class.

        :param size: The size of the input along the dimension to be normalized.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        if (dtype := kwargs.pop("dtype", None)) is not None:
            kwargs.update(dtype=dtype.as_pt())

        try:
            if self.name == LayerNormType.default:
                return LayerNorm(init_device=init_device, **kwargs)
            elif self.name == LayerNormType.rms:
                return RMSNorm(init_device=init_device, **kwargs)
            elif self.name == LayerNormType.fused_rms:
                return FusedRMSNorm(init_device=init_device, **kwargs)
            elif self.name == LayerNormType.l2_norm:
                return L2Norm(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e
                 


@dataclass
class MoERouterConfigV2(Config):
    """
    A configuration class for easily building any of the different MoE router modules.
    """


    top_k: int
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    bias_gamma: Optional[float] = None
    gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax
    dtype: Optional[DType] = None
    record_routing_batch_size: bool = False
    lb_loss_weight: Optional[float] = None
    lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch
    z_loss_weight: Optional[float] = None
    orth_loss_weight: Optional[float] = None
    record_routing_batch_size: bool = False
    
    def num_params(self, d_model: int, num_experts: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0

        num_params += d_model * num_experts

        return num_params

    def build(
        self,
        d_model: int,
        num_experts,
        *,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        orth_loss_weight: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "MoERouterV2":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param num_experts: The number of experts.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            num_experts=num_experts,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
            orth_loss_weight=orth_loss_weight,
        )
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        return MoERouterV2(**kwargs)
        


class MoEFusedV2TransformerBlock(TransformerBlockBase):
    
    def __init__(        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        attention_norm: LayerNormConfigV2,
        router: MoERouterConfigV2,
        shared_experts: SharedExpertsConfig,
        routed_experts: RoutedExpertsConfig,
        feed_forward_norm: LayerNormConfigV2,
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
        self.attention_norm = attention_norm.build()

            
        self.routed_experts = routed_experts.build()
        self.shared_experts = shared_experts.build()
        self.routed_experts_router = router.build(
            d_model,
            routed_experts.num_experts,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
            orth_loss_weight=orth_loss_weight,
            dtype=dtype,
            init_device=init_device,
        )
        self.shared_experts_router = router.build(
            d_model,
            shared_experts.num_experts,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
            orth_loss_weight=orth_loss_weight,
            dtype=dtype,
            init_device=init_device,
        )
        

        self.feed_forward_norm = feed_forward_norm.build()
        
        self.feed_forward_moe_norm = feed_forward_norm.build()
        
        self._ep_enabled = False
        self._tp_enabled = False


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
            return checkpoint(
                self.combined_forward, x, use_reentrant=False, loss_div_factor=loss_div_factor, **kwargs
            )
        else:
            # always use combined forward    
            return self.combined_forward(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError("TP is not supported in MoEFusedV1TransformerBlock")

    
    @torch.compile
    def router_forward(
        self,
        local_x: torch.Tensor,
        loss_div_factor: torch.Tensor,
    ):
        return self.router(
            local_x, 
            loss_div_factor=loss_div_factor # scalar
        )
    
    @torch.compile
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
        
        
    
            
        (
            local_x_global_expert_weights, # (B, S, top_k)
            local_x_global_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_expert, # (num_experts, )
            router_aux_loss # scalar
        ) = self.router_forward(
            local_x=x, 
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
                if router_aux_loss is not None:
                    local_x = attach_auxiliary_loss(local_x, router_aux_loss)


        # shape: (batch_size * seq_len, d_model)
        local_x = local_x.view(-1, D)
        # shape: (batch_size * seq_len * top_k,)
        local_x_global_expert_weights = get_local_tensor(local_x_global_expert_weights).flatten()
        # shape: (batch_size * seq_len * top_k,)
        local_x_global_expert_indices = get_local_tensor(local_x_global_expert_indices).flatten()


        assert self.shared_mlp is None
        
        
        ######################################################################
        ################# Call global_permute_mlp_unpermute() ##############
        ############################### START ################################


        with nvtx.annotate("permute_mlp_unpermute", color="blue"):
            if self.ep_enabled:
                x_moe, dense_out = cast(ParallelDroplessMLP, self.experts).global_permute_mlp_unpermute(
                    local_x,
                    local_x_global_expert_weights,
                    local_x_global_expert_indices,
                    local_batch_size_per_global_expert=local_batch_size_per_global_expert,
                    overlap_callback=self.overlap_callback,
                    # overlap_callback=None,
                    overlap_callback_x=x,
                    **kwargs,
                ) # type: ignore
            else:
                x_moe, dense_out = cast(ParallelDroplessMLP, self.experts).global_permute_mlp_unpermute_no_ep(
                    local_x,
                    local_x_global_expert_weights,
                    local_x_global_expert_indices, 
                    local_batch_size_per_global_expert=local_batch_size_per_global_expert,
                    overlap_callback=self.overlap_callback,
                    # overlap_callback=None,
                    overlap_callback_x=x,
                    **kwargs,
                ) # type: ignore
        ######################################################################
        ################# Call global_permute_mlp_unpermute() ##############
        ############################### END ##################################

        
        x_moe = x_moe.view(B, -1, D) # (B, S * top_k, d_model)

        

        
        ####################### # NOTE: fuse the add, but it takes more memory. Why?
        # wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) # type: ignore
        # # need to use dense_out
        
        # SPARSE_DROP_NORM_RECOMPUTE = True
        # if SPARSE_DROP_NORM_RECOMPUTE:
        #     final_out = checkpoint(
        #         self.sparse_add_drop_norm_forward, x_moe, dense_out, use_reentrant=False
        #     )
        # else:
        #     final_out = self.sparse_add_drop_norm_forward(
        #         x_moe, dense_out
        #     )
            
        #######################
        SPARSE_DROP_NORM_RECOMPUTE = True
        if SPARSE_DROP_NORM_RECOMPUTE:
            final_out = checkpoint(
                self.sparse_drop_norm_forward, x_moe, use_reentrant=False
            )
        else:
            final_out = self.sparse_drop_norm_forward(
                x_moe
            )
        
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) # type: ignore
        
        final_out = dense_out + final_out

        #######################

        
        return final_out

    @torch.compile
    def sparse_add_drop_norm_forward(
        self,
        x_moe: torch.Tensor,
        dense_out: torch.Tensor,
    ) -> torch.Tensor:
        return self.feed_forward_moe_norm(x_moe) + dense_out
    
    @torch.compile
    def sparse_drop_norm_forward(
        self,
        x_moe: torch.Tensor,
    ) -> torch.Tensor:
        return self.feed_forward_moe_norm(x_moe)
    
    
    def overlap_callback(self, x, **kwargs):
        # NOTE: this is called in the middle of the global_permute_mlp_unpermute() function
        # to overlap the dense forward pass with the all-to-all communication.
        # It is called after the local_x_global_expert_weights and local_x_global_expert_indices
        # are ready, but before the all-to-all communication starts.
        with torch.cuda.stream(self.get_dense_stream()):
            # if self.block_idx % 2 == 0:
            # x = self.attention(x, **kwargs)
            if True:
                xx = checkpoint(
                    self.dense_forward_rc, x, use_reentrant=False, **kwargs
                )
            else:
                xx = self.dense_forward_rc(
                    x, **kwargs
                )
            return xx
        
      