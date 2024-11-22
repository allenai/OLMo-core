import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import Config, DType, StrEnum
from ..doc_utils import beta_feature
from ..exceptions import OLMoConfigurationError
from .functional import l2_normalize
from .layer_norm import LayerNormConfig

__all__ = ["LMHeadType", "LMHeadConfig", "LMHead", "NormalizedLMHead"]


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
    ):
        super().__init__()
        self.norm = (
            None if layer_norm is None else layer_norm.build(d_model, init_device=init_device)
        )
        self.w_out = nn.Linear(d_model, vocab_size, bias=bias, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the LM head to the hidden state ``x``, returning the logits.
        """
        h = self.norm(x) if self.norm is not None else x
        return self.w_out(h)


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
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            layer_norm=None,
            bias=False,
            dtype=dtype,
            init_device=init_device,
        )
        self.sz_init_value = 1.0
        self.sz_init_scaling = 1.0 / math.sqrt(d_model)
        self.sz = nn.Parameter(
            self.sz_init_scaling * torch.ones(vocab_size, dtype=dtype, device=init_device)
        )

    def reset_parameters(self):
        """
        Reset the scaling parameter.
        """
        nn.init.ones_(self.sz)
        self.sz.mul_(self.sz_init_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
        return sz * self.w_out(x)

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w_out.weight)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))
