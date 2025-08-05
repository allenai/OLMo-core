from dataclasses import dataclass
import torch
import torch.nn as nn
from dataclasses import dataclass
from olmo_core.ops import moe as ops
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.config import Config, DType, StrEnum
import nvtx
import torch.nn.functional as F


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
        self.w_up_gate = nn.Linear(
            d_model,
            2 * num_experts * hidden_size,  # 2 for up and gate
            bias=bias,
            dtype=dtype.as_pt(),
            device=init_device
        )
        self.w_down = nn.Linear(
            num_experts * hidden_size,
            d_model,
            bias=bias,
            dtype=dtype.as_pt(),
            device=init_device
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
        
        