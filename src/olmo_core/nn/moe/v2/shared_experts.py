from dataclasses import dataclass
import torch
import torch.nn as nn
from dataclasses import dataclass
from olmo_core.ops import moe as ops
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.config import Config, DType, StrEnum
import nvtx
import torch.nn.functional as F
from typing import Tuple, Optional, TYPE_CHECKING, cast

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
    

    def build(self, init_device: str = "cpu") -> "SharedExperts":
        kwargs = self.as_dict()
        
        return SharedExperts(init_device=init_device, **kwargs)
    
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
                 dtype: DType,
                 init_device: str = "cpu" 
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        assert bias == False, "Shared experts do not support bias for now."

        E, D, H = num_experts, d_model, hidden_size

        # One big column-packed weight for up+gate: (D, E*2H)
        self.w_up_gate = nn.Parameter(torch.empty(D, E * 2 * H, device=init_device, dtype=dtype.as_pt()))
        # Per-expert down: (E, H, D)
        self.w_down = nn.Parameter(torch.empty(E, H, D, device=init_device, dtype=dtype.as_pt()))

        
    @nvtx.annotate("SharedExperts.forward", color='pink')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D) -> out: (E, B, S, D)
        Fast path:
          1) [BS, D] @ [D, E*2H] -> [BS, E*2H]
          2) reshape+permute -> (E, BS, 2, H)
          3) in-place SiLU on gate, elementwise multiply
          4) bmm with per-expert down: (E, BS, H) x (E, H, D) -> (E, BS, D)
        """
        B, S, D = x.shape
        E, H = self.num_experts, self.hidden_size
        BS = B * S

        # 1) One big GEMM (best utilization, contiguous out in last dim)
        x2 = x.reshape(BS, D)                                  # (BS, D)
        up_gate = x2 @ self.w_up_gate                          # (BS, E*2H)

        # 2) Reshape to separate experts and [up|gate], then make per-expert leading dim
        #    Shapes: (BS, E, 2, H) -> (E, BS, 2, H)
        up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)

        # 3) SwiGLU: split into up / gate; materialize gate once and do in-place SiLU
        up, gate = up_gate.unbind(dim=2)                       # each (E, BS, H) views
        gate = F.silu(gate)

        hidden = up * gate                                     # (E, BS, H)

        # 4) Per-expert down-proj as grouped GEMM
        #    hidden: (E, BS, H), w_down: (E, H, D) -> out: (E, BS, D)
        out = torch.bmm(hidden, self.w_down)

        return out.view(E, B, S, D)

    @nvtx.annotate("SharedExperts.forward1", color='purple')
    def forward1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the forward pass into two parts for better overlap in EP.
        """
        B, S, D = x.shape
        E, H = self.num_experts, self.hidden_size
        BS = B * S

        # 1) One big GEMM (best utilization, contiguous out in last dim)
        x2 = x.reshape(BS, D)                                  # (BS, D)
        up_gate = x2 @ self.w_up_gate                          # (BS, E*2H)

        # 2) Reshape to separate experts and [up|gate], then make per-expert leading dim
        #    Shapes: (BS, E, 2, H) -> (E, BS, 2, H)
        up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)

        # 3) SwiGLU: split into up / gate; materialize gate once and do in-place SiLU
        up, gate = up_gate.unbind(dim=2)                       # each (E, BS, H) views

        return up, gate

    @nvtx.annotate("SharedExperts.forward2", color='purple')
    def forward2(self, up: torch.Tensor, gate: torch.Tensor, xshape: torch.Size) -> torch.Tensor:
        """
        Split the forward pass into two parts for better overlap in EP.
        """
        E, H = self.num_experts, self.hidden_size
        B, S, D = xshape
        # 3) SwiGLU: split into up / gate; materialize gate once and do in-place SiLU

        gate = F.silu(gate)

        hidden = up * gate                                     # (E, BS, H)

        # 4) Per-expert down-proj as grouped GEMM
        #    hidden: (E, BS, H), w_down: (E, H, D) -> out: (E, BS, D)
        out = torch.bmm(hidden, self.w_down)

        return out.view(E, B, S, D)

    def extra_repr(self):
        return f'num_experts={self.num_experts}, hidden_size={self.hidden_size}, d_model={self.d_model}'
