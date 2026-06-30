from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import Config, DType

from ._nvtx import annotate


# SwiGLU is intentionally SiLU-gated here, matching the fused fast paths (forward1/forward2)
# and the routed-expert kernels, which also hardcode SiLU.
# TODO: fold the codebase's several SwiGLU sites (the configurable FeedForward, the v1 MoE
# MLP, routed_experts) into one shared helper — a general cross-module refactor.
def _swiglu(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return up * F.silu(gate)


@dataclass
class SharedExpertsConfig(Config):
    """
    Configuration for shared experts in a MoE block.

    .. note::
        Unlike the rest of the codebase, this config stores dimension-related fields
        (e.g. ``d_model``, ``hidden_size``, ``num_experts``) directly. Other configs keep
        such dimensions out of the config and instead receive them only as arguments to
        :meth:`build` (e.g. ``AttentionConfig.build(d_model, ...)``), so the dimensions live
        in a single place and flow down from the top-level transformer config. The v2 configs
        deviate by duplicating the dimensions here. This should be unified in the future to
        follow the dimension-agnostic ``build(d_model, ...)`` convention.
    """

    d_model: int
    hidden_size: int
    num_experts: int
    bias: bool
    dtype: DType

    def build(self, init_device: str = "cpu") -> "SharedExperts":
        """
        Build the corresponding shared-experts module.

        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict()

        return SharedExperts(init_device=init_device, **kwargs)

    def num_params(self) -> int:
        """
        The number of params that the module will have once built.
        """

        params = 3 * self.d_model * self.hidden_size  # up, gate, down
        if self.bias:
            params += 2 * self.hidden_size  # up and gate bias
            params += self.d_model  # down bias

        params *= self.num_experts  # for each expert

        return params


class SharedExperts(nn.Module):
    """
    Shared experts module for MoE blocks.

    Shared experts work like a regular feed-forward but can support more than 1 expert.
    All experts will have the same number of input tokens, so it's possible that we concatenate
    the weights of all experts and use a single linear layer to process the input.

    :meth:`forward` runs the whole module. :meth:`forward1` and :meth:`forward2` split it into
    two numerically-equivalent halves (the input projection producing ``(up, gate)``, then
    SwiGLU + the down-projection) so an expert-parallel block can run the shared experts on a
    separate CUDA stream and overlap each half with the routed experts' dispatch/combine
    all-to-all communication. The non-EP path just calls :meth:`forward`.
    """

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

        assert not bias, "Shared experts do not support bias for now."

        E, D, H = num_experts, d_model, hidden_size

        # One big column-packed weight for up+gate: (D, E*2H)
        self.w_up_gate = nn.Parameter(
            torch.empty(D, E * 2 * H, device=init_device, dtype=dtype.as_pt())
        )
        # Per-expert down: (E, H, D)
        self.w_down = nn.Parameter(torch.empty(E, H, D, device=init_device, dtype=dtype.as_pt()))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standalone default; the model-level init may override this with a depth-scaled std.
        std = 0.02
        for w in (self.w_up_gate, self.w_down):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def _raise_if_fp8_anchor_storage_released(self) -> None:
        if getattr(self, "_fp8_anchor_storage_released", False):
            raise RuntimeError(
                "SharedExperts bf16 fallback cannot run after fp8-only anchor storage has been released"
            )

    @annotate("SharedExperts.forward", "experts")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D) -> out: (E, B, S, D)
        Fast path:
          1) [BS, D] @ [D, E*2H] -> [BS, E*2H]
          2) reshape+permute -> (E, BS, 2, H)
          3) in-place SiLU on gate, elementwise multiply
          4) bmm with per-expert down: (E, BS, H) x (E, H, D) -> (E, BS, D)
        """
        self._raise_if_fp8_anchor_storage_released()
        B, S, D = x.shape
        E, H = self.num_experts, self.hidden_size
        BS = B * S

        # 1) One big GEMM (best utilization, contiguous out in last dim)
        x2 = x.reshape(BS, D)  # (BS, D)
        up_gate = x2 @ self.w_up_gate  # (BS, E*2H)

        # 2) Reshape to separate experts and [up|gate], then make per-expert leading dim
        #    Shapes: (BS, E, 2, H) -> (E, BS, 2, H)
        up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)

        # 3) SwiGLU: split into up / gate; materialize gate once and do in-place SiLU
        up, gate = up_gate.unbind(dim=2)  # each (E, BS, H) views

        hidden = _swiglu(up, gate)  # (E, BS, H)

        # 4) Per-expert down-proj as grouped GEMM
        #    hidden: (E, BS, H), w_down: (E, H, D) -> out: (E, BS, D)
        out = torch.bmm(hidden, self.w_down)

        return out.view(E, B, S, D)

    @annotate("SharedExperts.forward1", "experts")
    def forward1(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the forward pass into two parts for better overlap in EP.
        """
        self._raise_if_fp8_anchor_storage_released()
        B, S, D = x.shape
        E, H = self.num_experts, self.hidden_size
        BS = B * S

        # 1) One big GEMM (best utilization, contiguous out in last dim)
        x2 = x.reshape(BS, D)  # (BS, D)
        up_gate = x2 @ self.w_up_gate  # (BS, E*2H)

        # 2) Reshape to separate experts and [up|gate], then make per-expert leading dim
        #    Shapes: (BS, E, 2, H) -> (E, BS, 2, H)
        up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)

        # 3) SwiGLU: split into up / gate; materialize gate once and do in-place SiLU
        up, gate = up_gate.unbind(dim=2)  # each (E, BS, H) views

        return up, gate

    @annotate("SharedExperts.forward2", "experts")
    def forward2(self, up: torch.Tensor, gate: torch.Tensor, xshape: torch.Size) -> torch.Tensor:
        """
        Split the forward pass into two parts for better overlap in EP.
        """
        self._raise_if_fp8_anchor_storage_released()
        E = self.num_experts
        B, S, D = xshape
        # 3) SwiGLU: split into up / gate; materialize gate once and do in-place SiLU

        hidden = _swiglu(up, gate)  # (E, BS, H)

        # 4) Per-expert down-proj as grouped GEMM
        #    hidden: (E, BS, H), w_down: (E, H, D) -> out: (E, BS, D)
        out = torch.bmm(hidden, self.w_down)

        return out.view(E, B, S, D)

    def extra_repr(self):
        return f"num_experts={self.num_experts}, hidden_size={self.hidden_size}, d_model={self.d_model}"
