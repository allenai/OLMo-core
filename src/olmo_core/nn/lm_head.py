import logging
import math
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Placement

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError

from .attention import RingAttentionLoadBalancerType
from .functional import (
    cross_entropy_loss,
    fused_linear_cross_entropy_loss,
    l2_normalize,
)
from .layer_norm import LayerNormConfig

__all__ = [
    "LMHeadType",
    "LMLossImplementation",
    "LMHeadConfig",
    "LMHead",
    "NormalizedLMHead",
    "LMOutputWithLoss",
]


log = logging.getLogger(__name__)


class LMHeadType(StrEnum):
    """
    An enumeration of the different LM head types.
    """

    default = "default"
    """
    ➡️ :class:`LMHead`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedLMHead`
    """


class LMLossImplementation(StrEnum):
    """
    An enumeration of the different loss implementations.
    """

    default = "default"
    """
    Uses native PyTorch's operations.
    """

    fused_linear = "fused_linear"
    """
    A low-memory triton implementation from Liger-Kernel that fused the linear logits projection
    with the loss computation.
    """


@dataclass
class LMHeadConfig(Config):
    """
    A configuration class for building any of the :class:`LMHead` implementations.

    See the :class:`LMHead` subclasses to learn which fields are valid for each implementation.
    """

    name: LMHeadType = LMHeadType.default
    """
    The name of the implementation.
    """
    layer_norm: Optional[LayerNormConfig] = None
    bias: Optional[bool] = None
    dtype: DType = DType.float32
    loss_implementation: LMLossImplementation = LMLossImplementation.default

    def num_params(self, d_model: int, vocab_size: int) -> int:
        """
        The number of parameters in the module once built.
        """
        bias = self.bias if self.bias is not None else self.name != LMHeadType.normalized

        params = 0
        if self.layer_norm is not None:
            params += self.layer_norm.num_params(d_model)

        params += d_model * vocab_size
        if bias:
            params += vocab_size

        # Final scaling factor.
        if self.name == LMHeadType.normalized:
            params += vocab_size

        return params

    def build(self, *, d_model: int, vocab_size: int, init_device: str = "cpu") -> "LMHead":
        """
        Construct the corresponding LM head implementation.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            vocab_size=vocab_size,
            init_device=init_device,
            dtype=kwargs.pop("dtype").as_pt(),
        )

        try:
            if self.name == LMHeadType.default:
                return LMHead(**kwargs)
            elif self.name == LMHeadType.normalized:
                return NormalizedLMHead(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class LMOutputWithLoss(NamedTuple):
    logits: Optional[torch.Tensor]
    loss: torch.Tensor
    ce_loss: torch.Tensor
    z_loss: Optional[torch.Tensor]


class LMHead(nn.Module):
    """
    The default language modeling head implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        layer_norm: Optional[LayerNormConfig] = None,
        dtype: torch.dtype = torch.float32,
        bias: bool = True,
        init_device: str = "cpu",
        loss_implementation: LMLossImplementation = LMLossImplementation.default,
    ):
        super().__init__()
        self.norm = (
            None if layer_norm is None else layer_norm.build(d_model, init_device=init_device)
        )
        self.w_out = nn.Linear(d_model, vocab_size, bias=bias, dtype=dtype, device=init_device)
        self._d_model = d_model
        self._vocab_size = vocab_size
        self._loss_implementation = loss_implementation
        self._tp_mesh: Optional[DeviceMesh] = None
        self._cp_mesh: Optional[DeviceMesh] = None

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def loss_implementation(self) -> LMLossImplementation:
        return self._loss_implementation

    @property
    def tp_enabled(self) -> bool:
        return self._tp_mesh is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_mesh is not None

    def forward(
        self,
        x: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Apply the LM head to the hidden state ``x`` with shape ``(B, S, D)``..

        :returns: The logits if ``labels`` is ``None`` or a named tuple which will include the loss(es).
        """
        B = x.shape[0]

        h = self.norm(x) if self.norm is not None else x

        if labels is None:
            if return_logits is False:
                raise RuntimeError("'return_logits=False' is only valid when 'labels' is provided")
            return self.w_out(h)

        logits: Optional[torch.Tensor]
        loss: torch.Tensor
        ce_loss: torch.Tensor
        z_loss: Optional[torch.Tensor]
        if self.loss_implementation == LMLossImplementation.default:
            logits = self.w_out(h)
            assert logits is not None
            ce_loss, z_loss = cross_entropy_loss(
                get_local_tensor(logits).view(-1, self.vocab_size),
                get_local_tensor(labels).view(-1),
                ignore_index=ignore_index,
                reduction=loss_reduction,
                compute_z_loss=z_loss_multiplier is not None,
                z_loss_multiplier=z_loss_multiplier or 1e-4,
            )
            ce_loss = self._finalize_loss(
                ce_loss, B, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
            )
            loss = ce_loss
            if z_loss is not None:
                z_loss = self._finalize_loss(
                    z_loss, B, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
                )
                loss = loss + z_loss
        elif self.loss_implementation == LMLossImplementation.fused_linear:
            logits = None
            loss, z_loss = fused_linear_cross_entropy_loss(
                get_local_tensor(h).view(-1, self.d_model),
                get_local_tensor(self.w_out.weight),
                get_local_tensor(labels).view(-1),
                bias=get_local_tensor(self.w_out.bias) if self.w_out.bias is not None else None,
                ignore_index=ignore_index,
                reduction=loss_reduction,
                compute_z_loss=z_loss_multiplier is not None,
                z_loss_multiplier=z_loss_multiplier or 1e-4,
            )
            loss = self._finalize_loss(
                loss, B, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
            )
            if z_loss is not None:
                z_loss = self._finalize_loss(
                    z_loss, B, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
                )
                ce_loss = loss - z_loss
            else:
                ce_loss = loss
        else:
            raise NotImplementedError(
                f"'{self.loss_implementation}' loss implementation is not supported by {self.__class__.__name__}"
            )

        if return_logits is False:
            logits = None
        elif return_logits is True and logits is None:
            raise RuntimeError(
                f"'return_logits=True' is not compatible '{self.loss_implementation}' loss implementation"
            )

        return LMOutputWithLoss(logits=logits, loss=loss, ce_loss=ce_loss, z_loss=z_loss)

    def _finalize_loss(
        self,
        loss: torch.Tensor,
        B: int,
        *,
        loss_reduction: str,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        if loss_reduction == "none":
            # Reshape to `(B, S)`
            loss = loss.view(B, -1)

            # If TP, wrap with DTensor and mark as sharded on the sequence dimension.
            if self.tp_enabled:
                assert self._tp_mesh is not None
                loss = DTensor.from_local(loss, self._tp_mesh, (Shard(1),))
        elif self.tp_enabled:
            # Wrap with DTensor and finish the reduction.
            assert self._tp_mesh is not None
            loss = DTensor.from_local(loss.unsqueeze(0), self._tp_mesh, (Shard(0),))
            loss = loss.redistribute(placements=(Replicate(),))

            if loss_reduction == "sum":
                loss = loss.sum()
            elif loss_reduction == "mean":
                loss = loss.mean()
            else:
                raise NotImplementedError(loss_reduction)

        if loss_div_factor is not None:
            if self.cp_enabled:
                assert self._cp_mesh is not None
                loss_div_factor = loss_div_factor / self._cp_mesh.size()
            loss = loss / loss_div_factor

        return loss

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layouts: Optional[Tuple[Placement, Placement]] = None,
    ):
        # NOTE: there's a few cases to consider...
        # 1. If we're not using 'fused_linear' loss and we have a norm, then we do sequence-parallel through
        #    the norm, colwise-parallel through 'w_out', then back to sequence-parallel for the loss.
        # 2. If we're not using 'fused_linear' loss and we don't have a norm, then we start with
        #    the input replicated and proceed the same way.
        # 3. If we're using 'fused_linear' loss we do sequence-parallel all the way through.
        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=None if input_layouts is None else input_layouts[0],
                desired_input_layouts=Shard(1)
                if (
                    self.loss_implementation == LMLossImplementation.fused_linear
                    or self.norm is not None
                )
                else Replicate(),
                input_kwarg_layouts=None if input_layouts is None else {"labels": input_layouts[1]},
                desired_input_kwarg_layouts={"labels": Shard(1)},
            ),
        )

        if self.norm is not None:
            parallelize_module(
                module=self.norm,
                device_mesh=tp_mesh,
                parallelize_plan=SequenceParallel(),
            )

        if self.loss_implementation != LMLossImplementation.fused_linear:
            parallelize_module(
                module=self.w_out,
                device_mesh=tp_mesh,
                parallelize_plan=ColwiseParallel(output_layouts=Shard(1), use_local_output=False),
            )

        self._tp_mesh = tp_mesh

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        del load_balancer
        self._cp_mesh = cp_mesh


@beta_feature
class NormalizedLMHead(LMHead):
    """
    An nGPT LM head implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        loss_implementation: LMLossImplementation = LMLossImplementation.default,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            layer_norm=None,
            bias=False,
            dtype=dtype,
            init_device=init_device,
            loss_implementation=loss_implementation,
        )
        self.sz_init_value = 1.0
        self.sz_init_scaling = 1.0 / math.sqrt(d_model)
        self.sz = nn.Parameter(torch.empty(vocab_size, dtype=dtype, device=init_device))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the scaling parameter.
        """
        nn.init.ones_(self.sz)
        with torch.no_grad():
            self.sz.mul_(self.sz_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        B = x.shape[0]

        sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
        logits = sz * self.w_out(x)
        if labels is None:
            if return_logits is False:
                raise RuntimeError("'return_logits=False' is only valid when 'labels' is provided")
            return logits

        loss: torch.Tensor
        ce_loss: torch.Tensor
        z_loss: Optional[torch.Tensor]
        if self.loss_implementation == LMLossImplementation.default:
            ce_loss, z_loss = cross_entropy_loss(
                get_local_tensor(logits).view(-1, self.vocab_size),
                get_local_tensor(labels).view(-1),
                ignore_index=ignore_index,
                reduction=loss_reduction,
                compute_z_loss=z_loss_multiplier is not None,
                z_loss_multiplier=z_loss_multiplier or 1e-4,
            )
            ce_loss = self._finalize_loss(
                ce_loss, B, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
            )
            loss = ce_loss
            if z_loss is not None:
                z_loss = self._finalize_loss(
                    z_loss, B, loss_reduction=loss_reduction, loss_div_factor=loss_div_factor
                )
                loss = loss + z_loss
        else:
            raise NotImplementedError(
                f"'{self.loss_implementation}' loss implementation is not supported by '{self.__class__.__name__}'"
            )

        if return_logits is False:
            logits = None
        elif return_logits is True and logits is None:
            raise RuntimeError(
                f"'return_logits=True' is not compatible '{self.loss_implementation}' loss implementation"
            )

        return LMOutputWithLoss(logits if return_logits else None, loss, ce_loss, z_loss)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output

        raise NotImplementedError("TP is not implemented yet for the normalized LM head variant")

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w_out.weight)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))
