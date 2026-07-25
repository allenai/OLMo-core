from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
        MXLinearConfig,
        ScaleCalculationMode,
    )
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


T = TypeVar("T")


class _AOTypePlaceholder(Generic[T]):
    @property
    @abstractmethod
    def ao_type(self) -> type[T]:
        raise NotImplementedError

    def to_ao_type(self) -> T:
        if isinstance(self, Config):
            kwargs: dict[str, Any] = {}
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
            raise ValueError(self)
        else:
            raise NotImplementedError


class AOScalingType(_AOTypePlaceholder["ScalingType"], StrEnum):
    dynamic = "dynamic"
    disabled = "disabled"

    @property
    def ao_type(self) -> type["ScalingType"]:
        from torchao.float8.config import ScalingType

        return ScalingType


class AOScalingGranularity(_AOTypePlaceholder["ScalingGranularity"], StrEnum):
    tensorwise = "tensorwise"
    axiswise = "axiswise"

    @property
    def ao_type(self) -> type["ScalingGranularity"]:
        from torchao.float8.config import ScalingGranularity

        return ScalingGranularity


@dataclass
class AOCastConfig(Config, _AOTypePlaceholder["CastConfig"]):
    scaling_type: AOScalingType | None = None
    scaling_granularity: AOScalingGranularity | None = None
    target_dtype: DType | None = None

    @property
    def ao_type(self) -> type["CastConfig"]:
        from torchao.float8.config import CastConfig

        return CastConfig


@dataclass
class AOFloat8GemmConfig(Config, _AOTypePlaceholder["Float8GemmConfig"]):
    use_fast_accum: bool | None = False

    @property
    def ao_type(self) -> type["Float8GemmConfig"]:
        from torchao.float8.config import Float8GemmConfig

        return Float8GemmConfig


class AOFloat8LinearRecipe(_AOTypePlaceholder["Float8LinearRecipeName"], StrEnum):
    tensorwise = "tensorwise"
    rowwise = "rowwise"
    rowwise_with_gw_hp = "rowwise_with_gw_hp"

    @property
    def ao_type(self) -> type["Float8LinearRecipeName"]:
        from torchao.float8.config import Float8LinearRecipeName

        return Float8LinearRecipeName


class AOKernelPreference(_AOTypePlaceholder["KernelPreference"], StrEnum):
    emulated = "emulated"
    auto = "auto"
    cuda = "cuda"
    torch = "torch"

    @property
    def ao_type(self) -> type["KernelPreference"]:
        from torchao.quantization.quantize_.common.kernel_preference import (
            KernelPreference,
        )

        return KernelPreference


class AOMXFP8Dim1CastKernelChoice(_AOTypePlaceholder["MXFP8Dim1CastKernelChoice"], StrEnum):
    torch = "torch"
    cuda = "cuda"
    triton = "triton"

    @property
    def ao_type(self) -> type["MXFP8Dim1CastKernelChoice"]:
        from torchao.prototype.mx_formats.config import MXFP8Dim1CastKernelChoice

        return MXFP8Dim1CastKernelChoice


class AOScaleCalculationMode(_AOTypePlaceholder["ScaleCalculationMode"], StrEnum):
    floor = "floor"
    rceil = "rceil"
    ceil = "ceil"
    even = "even"

    @property
    def ao_type(self) -> type["ScaleCalculationMode"]:
        from torchao.prototype.mx_formats.config import ScaleCalculationMode

        return ScaleCalculationMode


@dataclass
class AOFloat8LinearConfig(Config, _AOTypePlaceholder["Float8LinearConfig"]):
    """
    This matches the config from torchao.
    """

    cast_config_input: AOCastConfig | None = None
    cast_config_input_for_grad_weight: AOCastConfig | None = None
    cast_config_weight: AOCastConfig | None = None
    cast_config_weight_for_grad_input: AOCastConfig | None = None
    cast_config_grad_output: AOCastConfig | None = None
    cast_config_grad_output_for_grad_weight: AOCastConfig | None = None
    gemm_config_output: AOFloat8GemmConfig | None = None
    gemm_config_grad_input: AOFloat8GemmConfig | None = None
    gemm_config_grad_weight: AOFloat8GemmConfig | None = None
    enable_fsdp_float8_all_gather: bool | None = None
    pad_inner_dim: bool | None = None
    emulate: bool | None = None
    force_recompute_fp8_weight_in_bwd: bool | None = None  # deprecated, no effect
    round_scales_to_power_of_2: bool | None = None

    @staticmethod
    def recommended(**kwargs: Any) -> "AOFloat8LinearConfig":
        return AOFloat8LinearConfig(
            enable_fsdp_float8_all_gather=True,
            force_recompute_fp8_weight_in_bwd=True,
            round_scales_to_power_of_2=True,
            **kwargs,
        )

    @property
    def ao_type(self) -> type["Float8LinearConfig"]:
        from torchao.float8.config import Float8LinearConfig

        return Float8LinearConfig


@dataclass
class AOMXLinearConfig(Config, _AOTypePlaceholder["MXLinearConfig"]):
    """
    This matches the config from torchao.
    Applies to MXFP8 and MXFP4 formats.
    https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/config.py#L106

    Useful reference for MXFP8 training: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
    """

    block_size: int | None = None
    """block size, defaults to 32 if not specified"""
    elem_dtype: DType | None = None
    """element dtype, used for activations, weights and gradients, defaults to e4m3fn if not specified"""
    elem_dtype_weight_override: DType | None = None
    """optional element dtype override for weights"""
    elem_dtype_grad_output_override: DType | None = None
    """
    optional element dtype override for gradients.
    note that e4m3 is thought to be fine here because of the block-wise nature of MXFP8.
    """
    kernel_preference: AOKernelPreference | None = None
    """if the preferred kernel is not supported on the given hardware an exception will be thrown"""
    mxfp8_cast_kernel_choice: AOMXFP8Dim1CastKernelChoice | None = None
    """
    which kernel to use for the mx fp8 cast along dim1 (dim0 is always torch).
    torch is slow. cuda is fastest. triton only supports "floor" scale calculation mode.
    """
    scale_calculation_mode: AOScaleCalculationMode | None = None
    """
    how to calculate the mx block scaling factors.
    * floor [default]: strightforward method but most prone to overflow / bad for gradient calculation (dont use)
    * rceil (ratio ceil): computes the tightest valid ceiling. has good support from nvidia.
    * ceil: similar to floor but avoids overflow; prone to underflow / precision loss / quant to zero.
    * even: best choice from a mathematical standpoint. unbiased error distribution. but does not yet work with torch.compile.
    """

    @classmethod
    def mxfp8_cublas_rceil(cls, **kwargs: Any) -> "AOMXLinearConfig":
        """standard mxfp8 recipe predefined in torchao"""
        return AOMXLinearConfig(
            mxfp8_cast_kernel_choice=AOMXFP8Dim1CastKernelChoice.cuda,
            scale_calculation_mode=AOScaleCalculationMode.rceil,
            **kwargs,
        )

    @property
    def ao_type(self) -> type["MXLinearConfig"]:
        from torchao.prototype.mx_formats import MXLinearConfig

        return MXLinearConfig
