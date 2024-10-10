from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import Config, StrEnum


class MoEType(StrEnum):
    """
    An enumeration of MoE layer types.
    """

    default = "default"
    """
    The default version.
    """

    dropless = "dropless"
    """
    The `dropless <https://arxiv.org/pdf/2211.15841>`_ version.
    """


class MoEActivationFn(StrEnum):
    swiglu = "swiglu"
    gelu = "gelu"
    gelu_tanh = "gelu_tanh"
    relu = "relu"

    def build(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == MoEActivationFn.swiglu:
            return partial(F.silu, inplace=False)
        elif self == MoEActivationFn.gelu:
            return partial(F.gelu, approximate="none")
        elif self == MoEActivationFn.gelu_tanh:
            return partial(F.gelu, approximate="tanh")
        elif self == MoEActivationFn.relu:
            return partial(F.relu, inplace=False)
        else:
            raise NotImplementedError(self)


class MoEMLPImplementation(StrEnum):
    sparse = "sparse"
    """
    Spare implementation.
    """
    grouped = "grouped"
    """
    Requires the grouped GEMM package https://github.com/tgale96/grouped_gemm.
    """


class MoEConfig(Config):
    """
    Configuration class for building MoE layers.
    Requires `megablocks <https://github.com/databricks/megablocks>`_.
    """

    name: MoEType = MoEType.default
    """
    The MoE implementation.
    """
    d_model: int = 1024
    """
    The model dimensionality.
    """
    hidden_size: int = 4096
    """
    The MLP hidden size.
    """
    activation_fn: MoEActivationFn = MoEActivationFn.swiglu
    """
    The activation function to use.
    """
    mlp_implementation: MoEMLPImplementation = MoEMLPImplementation.sparse
    """
    The MLP implementation.
    """
    num_experts: int = 8
    """
    The number of experts to use in the MoE block.
    """
    top_k: int = 2
    """
    The number of experts to select for each token.
    """
    capacity_factor: int = 1
    """
    The capacity factor to use in the MoE block. Only applies if not using :data:`MoEType.dropless`.
    """
    bias: bool = True
    """
    Include bias terms.
    """
    loss_weight: float = 0.1
    """
    The weight to use for the MoE load balancing loss.
    """
    zloss_weight: float = 0.0
    """
    Weight for MoE router z-loss where None means no router z-loss. 0.001 is a common value.
    """
    zloss_in_fp32: bool = False
    """
    Whether to compute the z-loss in FP32.
    """
    shared_expert: bool = False
    """
    Whether to have an always-used expert like in `DeepSeekMoE <https://arxiv.org/abs/2401.06066>`_.
    """
    lbl_in_fp32: bool = False
    """
    Whether to perform load balancing in FP32.
    """
    num_layers: int = 1
    """
    The total number of MoE layers.
    """
    init_device: str = "cpu"
    """
    The device to initialize weights on.
    """

    def as_megablocks_args(self):
        from megablocks.layers.arguments import Arguments  # type: ignore

        return Arguments(
            hidden_size=self.d_model,
            activation_fn=self.activation_fn.build(),
            mlp_type="glu" if "glu" in self.activation_fn.lower() else "mlp",
            mlp_impl=self.mlp_implementation,
            ffn_hidden_size=self.hidden_size,
            moe_num_experts=self.num_experts,
            moe_top_k=self.top_k,
            moe_capacity_factor=self.capacity_factor,
            moe_loss_weight=self.loss_weight,
            moe_zloss_weight=self.zloss_weight,
            moe_zloss_in_fp32=self.zloss_in_fp32,
            moe_lbl_in_fp32=self.lbl_in_fp32,
            shared_expert=self.shared_expert,
            bias=self.bias,
            return_bias=False,
            num_layers=self.num_layers,
            device=torch.device(self.init_device),
            fp16=False,
            bf16=False,
        )

    def build(self) -> nn.Module:
        """
        Build the MoE layer.
        """
        try:
            from megablocks.layers.dmoe import dMoE
            from megablocks.layers.moe import MoE
        except ImportError as e:
            raise ImportError(
                "megablocks is not installed. Please install it to use MoE layers"
            ) from e

        args = self.as_megablocks_args()
        if self.name == MoEType.default:
            return MoE(args)
        elif self.name == MoEType.dropless:
            return dMoE(args)
        else:
            raise NotImplementedError(self.name)
