import grouped_gemm  # type: ignore
import grouped_gemm.ops
from abc import abstractmethod
import torch
import torch.nn as nn
from ...moe import MoERouterConfig as MoERouterConfigV1
from typing import List, Optional
import nvtx
from olmo_core.config import Config, DType, StrEnum
import torch.nn.functional as F
from dataclasses import dataclass
from typing import cast
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.tensor import Placement, Replicate, Shard, distribute_tensor
from olmo_core.distributed.utils import get_local_tensor

@torch.compiler.disable
def gmm_no_compile(a, b, batch_sizes, trans_b=False):
    return grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b)

@dataclass
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
    

    def build(self, init_device: str = "cpu",) -> "RoutedExperts":
        kwargs = self.as_dict()
        return RoutedExperts(init_device=init_device, **kwargs)
    
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
    
    def num_active_params(self, top_k: int) -> int:
        """
        The number of params that the module will have once built, given the top_k experts.

        :param top_k: The number of experts to use.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if top_k > self.num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be greater than num_experts ({self.num_experts})")
        
        params = 3 * self.d_model * self.hidden_size # up, gate, down
        if self.bias:
            params += 2 * self.hidden_size # up and gate bias
            params += self.d_model  # down bias

        params *= top_k # for each expert
        
        return params
    
class RoutedExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        bias: bool,
        dtype: DType,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        assert bias == False, "Routed experts do not support bias for now."
        self.w_up_gate =nn.Parameter(
            torch.empty(
                num_experts,
                2 * hidden_size,
                d_model,
                dtype=dtype.as_pt(),
                device=init_device
            ),
        )

        self.w_down = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size, 
                d_model,
                dtype=dtype.as_pt(),
                device=init_device
            ),
        )
        self.gmm_ops = gmm_no_compile
        
        
    @nvtx.annotate("RoutedExperts.forward", color="blue")
    def forward(self, x: torch.Tensor, batch_size_per_expert: List) -> torch.Tensor:
        """
        `batch_size_per_expert` specifies the number of tokens in x for each expert.
        """
        assert isinstance(batch_size_per_expert, List), "only accept List for batch_size_per_expert"
        batch_size_per_expert_tensor = torch.tensor(
            batch_size_per_expert, 
            device='cpu', 
            dtype=torch.int64,  # NOTE: int64 required for grouped_gemm
        )

        if x.numel() == 0:
            return x
        
        w_up_gate = get_local_tensor(self.w_up_gate) # (E, H, 2D)
        w_down = get_local_tensor(self.w_down) # (E, H, D)
        up_gate = self.gmm_ops(x, w_up_gate, batch_size_per_expert_tensor, trans_b=True) # -> (BS, 2H)
        up_gate = cast(torch.Tensor, up_gate)  # ensure type is Tensor
        up, gate = up_gate.chunk(2, dim=-1)  
        h = up * F.silu(gate) # -> (BS, H)
        
        down = self.gmm_ops(h, w_down, batch_size_per_expert_tensor, trans_b=False) # -> (BS, H)
            
        return cast(torch.Tensor, down)  # ensure type is Tensor
    
    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        # ep_dp_mesh = ep_mesh['ep_dp']
        # ep_mp_mesh = ep_mesh['ep_mp']
        # shard dim 0 to ep_mp, replicate on ep_dp mesh
        self.ep_mesh = ep_mesh['ep_dp', 'ep_mp']
        dt = distribute_tensor(
            self.w_up_gate.data,
            self.ep_mesh,
            placements=(Replicate(), Shard(0))
        )
        self.w_up_gate = nn.Parameter(dt)
        dt = distribute_tensor(
            self.w_down.data,
            self.ep_mesh,
            placements=(Replicate(), Shard(0))
        )
        self.w_down = nn.Parameter(dt)

        self._ep_sharded = True

    def extra_repr(self):
        return f'num_experts={self.num_experts}, hidden_size={self.hidden_size}, d_model={self.d_model}'
