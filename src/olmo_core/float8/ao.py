from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Literal, Optional, Type, TypeVar

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
    from torchao.prototype.mx_formats.config import (
        MXFP8Dim1CastKernelChoice,
        MXGemmKernelChoice,
        MXLinearConfig,
        ScaleCalculationMode,
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


# class AOKernelPreference(_AOTypePlaceholder["KernelPreference"], StrEnum):
#     emulated = "emulated"
#     auto = "auto"
#     cuda = "cuda"
#     torch = "torch"

#     @property
#     def ao_type(self) -> Type["KernelPreference"]:
#         from torchao.quantization.quantize_.common.kernel_preference import (
#             KernelPreference,
#         )

#         return KernelPreference


class AOMXGemmKernelChoice(_AOTypePlaceholder["MXGemmKernelChoice"], StrEnum):
    emulated = "emulated"
    cutlass = "cutlass"
    cublas = "cublas"

    @property
    def ao_type(self) -> Type["MXGemmKernelChoice"]:
        from torchao.prototype.mx_formats.config import MXGemmKernelChoice

        return MXGemmKernelChoice


class AOMXFP8Dim1CastKernelChoice(_AOTypePlaceholder["MXFP8Dim1CastKernelChoice"], StrEnum):
    torch = "torch"
    cuda = "cuda"
    triton = "triton"

    @property
    def ao_type(self) -> Type["MXFP8Dim1CastKernelChoice"]:
        from torchao.prototype.mx_formats.config import MXFP8Dim1CastKernelChoice

        return MXFP8Dim1CastKernelChoice


class AOScaleCalculationMode(_AOTypePlaceholder["ScaleCalculationMode"], StrEnum):
    floor = "floor"
    rceil = "rceil"
    ceil = "ceil"
    even = "even"

    @property
    def ao_type(self) -> Type["ScaleCalculationMode"]:
        from torchao.prototype.mx_formats.config import ScaleCalculationMode

        return ScaleCalculationMode


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
    force_recompute_fp8_weight_in_bwd: Optional[bool] = None  # deprecated, no effect
    round_scales_to_power_of_2: Optional[bool] = None

    @staticmethod
    def recommended(**kwargs: Any) -> "AOFloat8LinearConfig":
        return AOFloat8LinearConfig(
            enable_fsdp_float8_all_gather=True,
            force_recompute_fp8_weight_in_bwd=True,
            round_scales_to_power_of_2=True,
            **kwargs,
        )

    @property
    def ao_type(self) -> Type["Float8LinearConfig"]:
        from torchao.float8.config import Float8LinearConfig

        return Float8LinearConfig


@dataclass
class AOMXLinearConfig(Config, _AOTypePlaceholder["MXLinearConfig"]):
    """
    This matches the config from torchao.
    Applies to MXFP8 and MXFP4 formats.
    https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/config.py#L106
    """

    block_size: Literal[32] = 32
    elem_dtype: DType = DType.float8_e4m3fn
    """element dtype, used for activations, weights and gradients"""
    elem_dtype_weight_override: Optional[DType] = None
    """optional element dtype override for weights"""
    elem_dtype_grad_output_override: Optional[DType] = None
    """optional element dtype override for gradients"""
    # kernel_preference: AOKernelPreference = AOKernelPreference.auto
    gemm_kernel_choice: AOMXGemmKernelChoice = (
        AOMXGemmKernelChoice.emulated
    )  #  removed soon in favor of kernel_preference
    """if the preferred kernel is not supported on the given hardware an exception will be thrown"""
    mxfp8_cast_kernel_choice: AOMXFP8Dim1CastKernelChoice = AOMXFP8Dim1CastKernelChoice.torch
    """which kernel to use for the mx fp8 cast along dim1 (dim0 is always torch)"""
    scale_calculation_mode: AOScaleCalculationMode = AOScaleCalculationMode.floor
    """how to calculate the mx block scaling factors"""

    @classmethod
    def mxfp8_cublas(cls, **kwargs: Any) -> "AOMXLinearConfig":
        return AOMXLinearConfig(
            # kernel_preference=AOKernelPreference.auto,
            gemm_kernel_choice=AOMXGemmKernelChoice.cublas,
            mxfp8_cast_kernel_choice=AOMXFP8Dim1CastKernelChoice.cuda,
            **kwargs,
        )

    @classmethod
    def mxfp8_cublas_rceil(cls, **kwargs: Any) -> "AOMXLinearConfig":
        return cls.mxfp8_cublas(scale_calculation_mode=AOScaleCalculationMode.rceil, **kwargs)

    @property
    def ao_type(self) -> Type["MXLinearConfig"]:
        from torchao.prototype.mx_formats import MXLinearConfig

        return MXLinearConfig
