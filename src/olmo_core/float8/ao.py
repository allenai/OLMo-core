from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Type, TypeVar

from ..config import Config, DType, StrEnum

if TYPE_CHECKING:
    from torchao.float8.config import (
        CastConfig,
        Float8GemmConfig,
        Float8LinearConfig,
        Float8LinearRecipeName,
        ScalingGranularity,
        ScalingType,
    )


T = TypeVar("T")


class _AOTypePlaceholder(Generic[T]):
    @property
    @abstractmethod
    def ao_type(self) -> Type[T]:
        raise NotImplementedError

    def to_ao_type(self) -> T:
        if isinstance(self, Config):
            kwargs: Dict[str, Any] = {}
            for k, v in self.as_dict(exclude_none=True, recurse=False).items():
                if isinstance(v, _AOTypePlaceholder):
                    v = v.to_ao_type()
                elif isinstance(v, DType):
                    v = v.as_pt()
                kwargs[k] = v

            return self.ao_type(**kwargs)
        elif isinstance(self, StrEnum):
            for option in self.ao_type:  # type: ignore
                if option.value == self:
                    return option
            else:
                raise ValueError(self)
        else:
            raise NotImplementedError


class AOScalingType(_AOTypePlaceholder["ScalingType"], StrEnum):
    dynamic = "dynamic"
    disabled = "disabled"

    @property
    def ao_type(self) -> Type["ScalingType"]:
        from torchao.float8.config import ScalingType

        return ScalingType


class AOScalingGranularity(_AOTypePlaceholder["ScalingGranularity"], StrEnum):
    tensorwise = "tensorwise"
    axiswise = "axiswise"

    @property
    def ao_type(self) -> Type["ScalingGranularity"]:
        from torchao.float8.config import ScalingGranularity

        return ScalingGranularity


@dataclass
class AOCastConfig(Config, _AOTypePlaceholder["CastConfig"]):
    scaling_type: Optional[AOScalingType] = None
    scaling_granularity: Optional[AOScalingGranularity] = None
    target_dtype: Optional[DType] = None

    @property
    def ao_type(self) -> Type["CastConfig"]:
        from torchao.float8.config import CastConfig

        return CastConfig


@dataclass
class AOFloat8GemmConfig(Config, _AOTypePlaceholder["Float8GemmConfig"]):
    use_fast_accum: Optional[bool] = False

    @property
    def ao_type(self) -> Type["Float8GemmConfig"]:
        from torchao.float8.config import Float8GemmConfig

        return Float8GemmConfig


class AOFloat8LinearRecipe(_AOTypePlaceholder["Float8LinearRecipeName"], StrEnum):
    tensorwise = "tensorwise"
    rowwise = "rowwise"
    rowwise_with_gw_hp = "rowwise_with_gw_hp"

    @property
    def ao_type(self) -> Type["Float8LinearRecipeName"]:
        from torchao.float8.config import Float8LinearRecipeName

        return Float8LinearRecipeName


@dataclass
class AOFloat8LinearConfig(Config, _AOTypePlaceholder["Float8LinearConfig"]):
    """
    This matches the config from torchao.
    """

    cast_config_input: Optional[AOCastConfig] = None
    cast_config_input_for_grad_weight: Optional[AOCastConfig] = None
    cast_config_weight: Optional[AOCastConfig] = None
    cast_config_weight_for_grad_input: Optional[AOCastConfig] = None
    cast_config_grad_output: Optional[AOCastConfig] = None
    cast_config_grad_output_for_grad_weight: Optional[AOCastConfig] = None
    gemm_config_output: Optional[AOFloat8GemmConfig] = None
    gemm_config_grad_input: Optional[AOFloat8GemmConfig] = None
    gemm_config_grad_weight: Optional[AOFloat8GemmConfig] = None
    enable_fsdp_float8_all_gather: Optional[bool] = None
    pad_inner_dim: Optional[bool] = None
    emulate: Optional[bool] = None
    force_recompute_fp8_weight_in_bwd: Optional[bool] = None
    round_scales_to_power_of_2: Optional[bool] = None

    @property
    def ao_type(self) -> Type["Float8LinearConfig"]:
        from torchao.float8.config import Float8LinearConfig

        return Float8LinearConfig
