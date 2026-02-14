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


@torch.compiler.disable
def gmm_no_compile(a, b, batch_sizes, trans_b=False):
    return grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b)

def gmm(a, b, batch_sizes, trans_b=False):
    if use_torch_grouped_mm():
        # torch.nn.functional.grouped_mm has no trans_b argument.
        # It expects mat_b to be (num_groups, K, N), so we transpose when
        # emulating grouped_gemm(..., trans_b=True).
        b_grouped_mm = b.transpose(1, 2) if trans_b else b
        offs = torch.cumsum(batch_sizes.to(dtype=torch.int32), dim=0, dtype=torch.int32)
        return F.grouped_mm(a, b_grouped_mm, offs=offs)

    return gmm_no_compile(a, b, batch_sizes, trans_b)

# if env variable OLMO_USE_TORCH_GROUPED_MM is set, use its value to determine whether to use torch grouped_mm; 
import os
env_val = os.getenv("OLMO_USE_TORCH_GROUPED_MM")
if env_val is not None:
    if env_val.lower() in ("1", "true", "yes"):
        USE_TORCH_GROUPED_MM = True
    elif env_val.lower() in ("0", "false", "no"):
        USE_TORCH_GROUPED_MM = False
    else:
        raise ValueError(f"Invalid value for OLMO_USE_TORCH_GROUPED_MM: {env_val}. Expected one of (1, 0, true, false, yes, no).")
else:
    # otherwise, use feature detection and version gate.
    USE_TORCH_GROUPED_MM = None

def use_torch_grouped_mm():
    global USE_TORCH_GROUPED_MM
    if USE_TORCH_GROUPED_MM is not None:
        return USE_TORCH_GROUPED_MM

    torch_version = torch.__version__.split("+")[0]  # strip local build suffix, e.g. +cu128
    try:
        major_str, minor_str, *_ = torch_version.split(".")
        major, minor = int(major_str), int(minor_str)
        meets_version_gate = major > 2 or (major == 2 and minor >= 10)
    except (ValueError, TypeError):
        # Fall back to feature detection on unusual version strings.
        meets_version_gate = hasattr(F, "grouped_mm")

    # grouped_mm was added in torch 2.10; hasattr keeps this robust to local builds.
    USE_TORCH_GROUPED_MM = meets_version_gate and hasattr(F, "grouped_mm")
    return USE_TORCH_GROUPED_MM

REQUIRES_HOST_SIDE_SPLIT_SIZES = None # cache the result of whether host-side split sizes are required, since it does not change during runtime and checking it requires parsing torch version every time.
def requires_host_side_split_sizes():
    # read from cache if available
    global REQUIRES_HOST_SIDE_SPLIT_SIZES
    if REQUIRES_HOST_SIDE_SPLIT_SIZES is not None:
        return REQUIRES_HOST_SIDE_SPLIT_SIZES
    
    # grouped_gemm cublas mode requires host-side split sizes, grouped_mm does not.
    REQUIRES_HOST_SIDE_SPLIT_SIZES = not use_torch_grouped_mm()

    return REQUIRES_HOST_SIDE_SPLIT_SIZES


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

        # assume no ep in init
        self.num_local_experts: int = num_experts
        self.ep_dim: int = 1
        self.ep_rank: int = 0


    # @torch.compiler.disable(recursive=False)
    @nvtx.annotate("RoutedExperts.forward", color="blue")
    def forward(self, x: torch.Tensor, batch_size_per_expert: torch.Tensor) -> torch.Tensor:
        """
        `batch_size_per_expert` specifies the number of tokens in x for each expert.
        """

        assert isinstance(batch_size_per_expert, torch.Tensor), "only accept Tensor for batch_size_per_expert"

        if requires_host_side_split_sizes():
            # CPU-side split sizes are required by grouped_gemm cublas mode.
            # grouped_gemm CUTLASS mode can accept device-side split sizes, but it is slow.
            # Always assume grouped_gemm runs in cublas mode.
            assert batch_size_per_expert.device.type == 'cpu', "batch_size_per_expert must be on cpu"
            batch_size_per_expert_tensor = batch_size_per_expert.to(dtype=torch.int64)  # int64 required for grouped_gemm
        else:
            assert batch_size_per_expert.device.type == 'cuda', "batch_size_per_expert expected to be on GPU"
            # grouped_mm expects int32 offsets derived from split sizes.
            batch_size_per_expert_tensor = batch_size_per_expert.to(dtype=torch.int32)

        if x.numel() == 0:
            return x
        
        w_up_gate = self.w_up_gate # (E, H, 2D)
        w_down = self.w_down # (E, H, D)

        # up + gate projection
        up_gate = gmm(x, w_up_gate, batch_size_per_expert_tensor, trans_b=True) # -> (BS, 2H)

        up_gate = cast(torch.Tensor, up_gate)  # ensure type is Tensor

        h = self.chunk_and_activate(up_gate) # -> (BS, H)
        
        # down projection
        down = gmm(h, w_down, batch_size_per_expert_tensor, trans_b=False) # -> (BS, D)

        return cast(torch.Tensor, down)  # ensure type is Tensor

    def act_and_down(self, up_gate: torch.Tensor, batch_size_per_expert_tensor: torch.Tensor) -> torch.Tensor:
        # swiglu + down projection
        # so that it apply activation checkpointing if needed
        h = self.chunk_and_activate(up_gate) # -> (BS, H)
        
        down = gmm(h, self.w_down, batch_size_per_expert_tensor, trans_b=False) # -> (BS, H)
        return down

    @torch.compile
    def chunk_and_activate(self, up_gate: torch.Tensor) -> torch.Tensor:
        up, gate = up_gate.chunk(2, dim=-1)  
        h = up * F.silu(gate) # -> (BS, H)
        return h

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        # shard dim 0 to ep_mp, replicate on ep_dp mesh
        self.ep_mesh = ep_mesh['ep_dp', 'ep_mp']
        # with torch.no_grad():  # just to avoid tracking the rebind below
        self.ep_dim = ep_mesh['ep_mp'].size()
        self.ep_rank = ep_mesh['ep_mp'].get_local_rank()

        assert self.num_experts % self.ep_dim == 0, "num_experts must be divisible by the number of expert partitions"
        self.num_local_experts = self.num_experts // self.ep_dim

        self.w_up_gate = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.hidden_size,
                self.d_model,
                dtype=self.w_up_gate.dtype,
                device=self.w_up_gate.device
            ),
        )

        self.w_down = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size, 
                self.d_model,
                dtype=self.w_down.dtype,
                device=self.w_down.device
            ),
        )


        self._ep_sharded = True

    def extra_repr(self):
        return f'num_experts={self.num_experts}, hidden_size={self.hidden_size}, d_model={self.d_model}'
