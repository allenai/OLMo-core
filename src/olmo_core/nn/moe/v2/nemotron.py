from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Placement, distribute_tensor

from olmo_core.config import UNSET, DType, StrEnum
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
from olmo_core.nn.moe.v2.routed_experts import (
    ExpertActivation,
    RoutedExperts,
    RoutedExpertsConfig,
    requires_host_side_split_sizes,
)
from olmo_core.nn.moe.v2.router import MoERouterConfigV2, MoERouterV2
from olmo_core.nn.moe.v2.shared_experts import SharedExperts, SharedExpertsConfig
from olmo_core.nn.transformer.block import TransformerBlockBase
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod, init_linear
from olmo_core.nn.transformer.config import TransformerDataParallelWrappingStrategy


NEMOTRON3_NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


class NemotronBlockKind(StrEnum):
    mamba = "mamba"
    attention = "attention"
    moe = "moe"


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def _copy_into_tensor_or_dtensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    if isinstance(dst, DTensor):
        dst.copy_(distribute_tensor(src, dst.device_mesh, placements=dst.placements))
    else:
        dst.copy_(src)


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(
    input_tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    return input_tensor.reshape(
        input_tensor.shape[0],
        -1,
        chunk_size,
        input_tensor.shape[2],
        input_tensor.shape[3],
    )


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=0,
    )
    return tensor_segsum.masked_fill(~mask, -torch.inf)


class NemotronRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        group_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=init_device))
        self.variance_epsilon = eps
        self.group_size = group_size

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        *prefix_dims, last_dim = hidden_states.shape
        group_count = last_dim // self.group_size
        hidden_states_group = hidden_states.view(*prefix_dims, group_count, self.group_size)
        variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
        hidden_states_group = hidden_states_group * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_group.view(*prefix_dims, group_count * self.group_size)
        return self.weight * hidden_states.to(input_dtype)


@dataclass
class NemotronMamba2Config(SequenceMixerConfig["NemotronMamba2Mixer"]):
    ssm_state_size: int = 128
    conv_kernel: int = 4
    mamba_num_heads: int = 64
    mamba_head_dim: int = 64
    n_groups: int = 8
    chunk_size: int = 128
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 0.0001
    use_conv_bias: bool = True
    use_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    dtype: DType = DType.bfloat16

    def num_params(self, d_model: int) -> int:
        intermediate = self.mamba_num_heads * self.mamba_head_dim
        conv_dim = intermediate + 2 * self.n_groups * self.ssm_state_size
        projection_size = intermediate + conv_dim + self.mamba_num_heads
        params = d_model * projection_size
        if self.use_bias:
            params += projection_size
        params += conv_dim * self.conv_kernel
        if self.use_conv_bias:
            params += conv_dim
        params += 3 * self.mamba_num_heads
        params += intermediate
        params += intermediate * d_model
        if self.use_bias:
            params += d_model
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "NemotronMamba2Mixer":
        del n_layers, cache
        return NemotronMamba2Mixer(
            d_model=d_model,
            layer_idx=layer_idx,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
            **self.as_dict(exclude={"dtype"}),
        )


class NemotronMamba2Mixer(SequenceMixer):
    def __init__(
        self,
        *,
        d_model: int,
        layer_idx: int,
        ssm_state_size: int,
        conv_kernel: int,
        mamba_num_heads: int,
        mamba_head_dim: int,
        n_groups: int,
        chunk_size: int,
        time_step_min: float,
        time_step_max: float,
        time_step_floor: float,
        use_conv_bias: bool,
        use_bias: bool,
        layer_norm_epsilon: float,
        dtype: torch.dtype,
        init_device: str = "cpu",
    ):
        super().__init__()
        del time_step_max, time_step_floor
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel
        self.num_heads = mamba_num_heads
        self.head_dim = mamba_head_dim
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.time_step_min = time_step_min
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(d_model, projection_size, bias=use_bias, dtype=dtype, device=init_device)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=dtype,
            device=init_device,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads, dtype=dtype, device=init_device))
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=dtype, device=init_device))
        self.D = nn.Parameter(torch.ones(self.num_heads, dtype=dtype, device=init_device))
        self.norm = NemotronRMSNormGated(
            self.intermediate_size,
            group_size=self.intermediate_size // self.n_groups,
            eps=layer_norm_epsilon,
            dtype=dtype,
            init_device=init_device,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, d_model, bias=use_bias, dtype=dtype, device=init_device
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.A_log.device.type != "meta":
                values = torch.arange(
                    1, self.num_heads + 1, device=self.A_log.device, dtype=torch.float32
                ).log().to(dtype=self.A_log.dtype)
                _copy_into_tensor_or_dtensor(self.A_log, values)
            nn.init.ones_(self.dt_bias)
            nn.init.ones_(self.D)
            self.norm.reset_parameters()

    def forward(
        self,
        input_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        if attention_mask is not None:
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)
        projected_states = self.in_proj(input_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        hidden_states = F.silu(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        if attention_mask is not None:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        hidden_states, B, C = torch.split(
            hidden_states,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_min)
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        D_residual = self.D[..., None] * _pad_tensor_by_size(hidden_states, pad_size)

        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt

        hidden_states, A, B, C = [
            _reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)
        ]

        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(_segment_sum(A))

        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
        states = (
            B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
            * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]
        ).sum(dim=3).permute(0, 1, 2, 4, 3)
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))

        states_permuted = states.permute(0, 2, 1, 3, 4)
        result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
        new_states = result.permute(0, 2, 1, 3, 4)
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum)
        C_times_states = C[..., None, :] * states[:, :, None, ...]
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        return self.out_proj(scan_output.to(dtype))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for NemotronMamba2Mixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not implemented for NemotronMamba2Mixer")

    def init_weights(
        self,
        *,
        init_method: InitMethod,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        del d_model, block_idx, num_blocks
        init_linear(self.in_proj, std=std, generator=generator)
        init_linear(self.conv1d, std=std, generator=generator)
        out_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            out_std = std / math.sqrt(2 * num_blocks)
        elif init_method == InitMethod.llama_depth:
            out_std = std / math.sqrt(2 * (block_idx + 1))
        init_linear(self.out_proj, std=out_std, generator=generator)
        self.reset_parameters()

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return 0


@dataclass
class NemotronMoEConfig(SequenceMixerConfig["NemotronMoE"]):
    num_experts: int = 128
    num_experts_per_tok: int = 6
    intermediate_size: int = 1856
    shared_expert_intermediate_size: int = 3712
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    dtype: DType = DType.bfloat16

    def num_params(self, d_model: int) -> int:
        routed = self.num_experts * (self.intermediate_size * d_model + d_model * self.intermediate_size)
        router = self.num_experts * d_model
        shared = d_model * self.shared_expert_intermediate_size + self.shared_expert_intermediate_size * d_model
        return routed + router + shared

    def num_active_params(self, d_model: int) -> int:
        routed = self.num_experts_per_tok * (
            self.intermediate_size * d_model + d_model * self.intermediate_size
        )
        router = self.num_experts * d_model
        shared = d_model * self.shared_expert_intermediate_size + self.shared_expert_intermediate_size * d_model
        return routed + router + shared

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "NemotronMoE":
        del layer_idx, n_layers, cache
        return NemotronMoE(d_model=d_model, init_device=init_device, **self.as_dict())


class NemotronDenseMLP(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        dtype: DType,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.up_proj = nn.Linear(d_model, hidden_size, bias=False, dtype=dtype.as_pt(), device=init_device)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=False, dtype=dtype.as_pt(), device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_relu2(self.up_proj(x)))


class NemotronMoE(SequenceMixer):
    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
        dtype: DType,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_size = intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.routed_experts_router: MoERouterV2 = MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            gating_function=MoERouterGatingFunction.sigmoid,
            normalize_expert_weights=1.0 if norm_topk_prob else None,
            expert_weight_scale=routed_scaling_factor,
            score_correction_bias=True,
            n_group=n_group,
            topk_group=topk_group,
            sigmoid_stability_epsilon=0.0,
            dtype=dtype,
        ).build(init_device=init_device)
        self.routed_experts: RoutedExperts = RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=intermediate_size,
            num_experts=num_experts,
            bias=False,
            dtype=dtype,
            activation=ExpertActivation.relu2,
        ).build(init_device=init_device)
        self.shared_experts: SharedExperts = SharedExpertsConfig(
            d_model=d_model,
            hidden_size=shared_expert_intermediate_size,
            num_experts=1,
            bias=False,
            dtype=dtype,
            activation=ExpertActivation.relu2,
        ).build(init_device=init_device)

    def _forward_grouped_cuda(
        self,
        x_flat: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        routing_map = expert_indices.view(-1, self.num_experts_per_tok).int()
        num_out_tokens = routing_map.shape[0] * self.num_experts_per_tok
        permuted_x, row_id_map = moe_permute_no_compile(
            inp=x_flat,
            routing_map=routing_map,
            num_out_tokens=num_out_tokens,
            map_type="index",
        )

        routed = self.routed_experts(permuted_x, batch_size_per_expert)
        return moe_unpermute_no_compile(
            inp=routed,
            row_id_map=row_id_map,
            restore_shape=x_flat.shape,
            map_type="index",
            merging_probs=expert_weights.view(-1, self.num_experts_per_tok),
        )

    def _forward_loop(
        self,
        x_flat: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(x_flat, dtype=expert_weights.dtype)
        with torch.no_grad():
            expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = x_flat[token_idx]
            hidden = F.linear(current_state, self.routed_experts.w_up_gate[expert_idx])
            hidden = _relu2(hidden)
            hidden = hidden @ self.routed_experts.w_down[expert_idx]
            hidden = hidden * expert_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, hidden.to(final.dtype))
        return final

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        residual = x
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        expert_weights, expert_indices, batch_size_per_expert, _ = self.routed_experts_router(
            x_flat.view(1, -1, self.d_model),
            False,
        )
        expert_weights = expert_weights.view(-1, self.num_experts_per_tok)
        expert_indices = expert_indices.view(-1, self.num_experts_per_tok)
        if x_flat.is_cuda and not requires_host_side_split_sizes():
            final = self._forward_grouped_cuda(x_flat, expert_indices, expert_weights, batch_size_per_expert)
        else:
            final = self._forward_loop(x_flat, expert_indices, expert_weights)

        routed = final.to(x_flat.dtype).view(*orig_shape)
        shared = self.shared_experts(residual).sum(dim=0)
        return routed + shared

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for NemotronMoE")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del cp_mesh, ring, uly

    def init_weights(
        self,
        *,
        init_method: InitMethod,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        del init_method, d_model, block_idx, num_blocks
        nn.init.trunc_normal_(self.routed_experts.w_up_gate, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        nn.init.trunc_normal_(self.routed_experts.w_down, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        nn.init.trunc_normal_(self.routed_experts_router.weight, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        nn.init.trunc_normal_(self.shared_experts.w_up_gate, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        nn.init.trunc_normal_(self.shared_experts.w_down, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        with torch.no_grad():
            score_bias = self.routed_experts_router.score_bias
            if score_bias is not None:
                score_bias.zero_()

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return 0


@dataclass
class NemotronBlockConfig(TransformerBlockConfig):
    sequence_mixer: SequenceMixerConfig = field(default=UNSET, init=False, repr=False)
    attention_norm: Optional[LayerNormConfig] = field(default=None, init=False, repr=False)
    feed_forward_norm: Optional[LayerNormConfig] = field(default=None, init=False, repr=False)
    feed_forward: Any = field(default=None, init=False, repr=False)
    feed_forward_moe: Any = field(default=None, init=False, repr=False)
    dropout: Optional[float] = field(default=None, init=False, repr=False)
    attention_residual_alpha: Optional[float] = field(default=None, init=False, repr=False)
    feed_forward_residual_alpha: Optional[float] = field(default=None, init=False, repr=False)

    kind: NemotronBlockKind = NemotronBlockKind.mamba
    norm: LayerNormConfig = field(default_factory=LayerNormConfig)
    mixer: SequenceMixerConfig = field(default=UNSET)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.sequence_mixer = self.mixer

    def num_params(self, d_model: int) -> int:
        return self.norm.num_params(d_model) + self.mixer.num_params(d_model)

    def num_active_params(self, d_model: int) -> int:
        if hasattr(self.mixer, "num_active_params"):
            return self.norm.num_params(d_model) + self.mixer.num_active_params(d_model)  # type: ignore[attr-defined]
        return self.num_params(d_model)

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "NemotronBlock":
        return NemotronBlock(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            kind=self.kind,
            norm=self.norm,
            mixer=self.mixer,
            init_device=init_device,
            cache=cache,
        )


class NemotronBlock(TransformerBlockBase):
    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        kind: NemotronBlockKind,
        norm: LayerNormConfig,
        mixer: SequenceMixerConfig,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx
        self.kind = kind
        self.norm = norm.build(d_model, init_device=init_device)
        self.attention = mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )

    @property
    def is_moe(self) -> bool:
        return self.kind == NemotronBlockKind.moe

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[torch.Tensor | float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del loss_div_factor
        h = self.norm(x.to(dtype=self.norm.weight.dtype))
        return x + self.attention(h, **kwargs)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        *,
        input_layout: Placement,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for NemotronBlock")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs: Any,
    ):
        del prefetch_factor, wrapping_strategy
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.attention.num_flops_per_token(seq_len)


def _text_config(hf_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return hf_config.get("text_config", hf_config)


def get_nemotron3_nano_config_overrides(hf_config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = _text_config(hf_config)
    return {
        "vocab_size": cfg["vocab_size"],
        "d_model": cfg["hidden_size"],
        "n_layers": len(cfg["layers_block_type"]),
        "layers_block_type": tuple(cfg["layers_block_type"]),
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "attention_head_dim": cfg["head_dim"],
        "mamba_num_heads": cfg["mamba_num_heads"],
        "mamba_head_dim": cfg["mamba_head_dim"],
        "ssm_state_size": cfg["ssm_state_size"],
        "conv_kernel": cfg["conv_kernel"],
        "n_groups": cfg["n_groups"],
        "chunk_size": cfg["chunk_size"],
        "time_step_min": cfg["time_step_min"],
        "time_step_max": cfg["time_step_max"],
        "time_step_floor": cfg["time_step_floor"],
        "num_experts": cfg["n_routed_experts"],
        "num_experts_per_tok": cfg["num_experts_per_tok"],
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "shared_expert_intermediate_size": cfg["moe_shared_expert_intermediate_size"],
        "n_group": cfg["n_group"],
        "topk_group": cfg["topk_group"],
        "routed_scaling_factor": cfg["routed_scaling_factor"],
        "norm_topk_prob": cfg["norm_topk_prob"],
        "norm_eps": cfg["norm_eps"],
    }


def build_nemotron3_nano_config(
    *,
    vocab_size: int = 131_072,
    d_model: int = 2688,
    n_layers: int = 52,
    layers_block_type: Sequence[str],
    num_attention_heads: int = 32,
    num_key_value_heads: int = 2,
    attention_head_dim: int = 128,
    mamba_num_heads: int = 64,
    mamba_head_dim: int = 64,
    ssm_state_size: int = 128,
    conv_kernel: int = 4,
    n_groups: int = 8,
    chunk_size: int = 128,
    time_step_min: float = 0.001,
    time_step_max: float = 0.1,
    time_step_floor: float = 0.0001,
    num_experts: int = 128,
    num_experts_per_tok: int = 6,
    moe_intermediate_size: int = 1856,
    shared_expert_intermediate_size: int = 3712,
    n_group: int = 1,
    topk_group: int = 1,
    routed_scaling_factor: float = 2.5,
    norm_topk_prob: bool = True,
    norm_eps: float = 1e-5,
    dtype: DType = DType.bfloat16,
    init_seed: int = 2026,
    init_std: float = 0.02,
    attention_backend: AttentionBackendName = AttentionBackendName.flash_4,
) -> TransformerConfig:
    norm = LayerNormConfig(
        name=LayerNormType.nemotron_rms,
        eps=norm_eps,
        bias=False,
        dtype=dtype,
    )
    mamba_block = NemotronBlockConfig(
        kind=NemotronBlockKind.mamba,
        norm=norm,
        mixer=NemotronMamba2Config(
            ssm_state_size=ssm_state_size,
            conv_kernel=conv_kernel,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            time_step_floor=time_step_floor,
            dtype=dtype,
        ),
    )
    attention_block = NemotronBlockConfig(
        kind=NemotronBlockKind.attention,
        norm=norm,
        mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=num_attention_heads,
            n_kv_heads=num_key_value_heads,
            head_dim=attention_head_dim,
            bias=False,
            rope=None,
            qk_norm=None,
            backend=attention_backend,
            dtype=dtype,
            d_attn=num_attention_heads * attention_head_dim,
        ),
    )
    moe_block = NemotronBlockConfig(
        kind=NemotronBlockKind.moe,
        norm=norm,
        mixer=NemotronMoEConfig(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            dtype=dtype,
        ),
    )
    config = TransformerConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block={
            "mamba": mamba_block,
            "attention": attention_block,
            "moe": moe_block,
        },
        block_pattern=list(layers_block_type),
        lm_head=LMHeadConfig(layer_norm=norm, bias=False, dtype=dtype),
        dtype=dtype,
        init_seed=init_seed,
        init_std=init_std,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_nemotron3_nano_config_from_hf_config(
    hf_config: Mapping[str, Any],
    **overrides: Any,
) -> TransformerConfig:
    kwargs = get_nemotron3_nano_config_overrides(hf_config)
    kwargs.update(overrides)
    return build_nemotron3_nano_config(**kwargs)


def build_debug_nemotron3_nano_config(
    *,
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 6,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_intermediate_size: int = 128,
    shared_expert_intermediate_size: int = 256,
    dtype: DType = DType.bfloat16,
    attention_backend: AttentionBackendName = AttentionBackendName.torch,
    **kwargs: Any,
) -> TransformerConfig:
    pattern = ("mamba", "moe", "mamba", "attention", "moe", "mamba")
    return build_nemotron3_nano_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        layers_block_type=tuple(pattern[:n_layers]) if n_layers <= len(pattern) else pattern,
        num_attention_heads=4,
        num_key_value_heads=1,
        attention_head_dim=64,
        mamba_num_heads=4,
        mamba_head_dim=64,
        ssm_state_size=16,
        n_groups=1,
        chunk_size=64,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        dtype=dtype,
        attention_backend=attention_backend,
        **kwargs,
    )
