import logging
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import Config, StrEnum
from ..exceptions import OLMoConfigurationError

__all__ = [
    "MuPHyperParam",
    "MuPScalingStrategy",
    "MuPConfig",
    "MuP",
]


log = logging.getLogger(__name__)


class MuPHyperParam(StrEnum):
    """
    An enumeration of the different hyperparameters that can be perceived by muP as
    affecting model 'width'.
    """

    d_model = "d_model"
    hidden_size = "hidden_size"
    n_heads = "n_heads"
    n_kv_heads = "n_kv_heads"
    head_dim = "head_dim"


class MuPParamScalingType(StrEnum):
    """
    An enumeration the types of ways a paramater can scale with 'width'.
    """

    constant = "constant"
    """
    The parameter's input and output are independent of width. muP does not apply to the parameter.
    """

    scaling_input = "scaling_input"
    """
    The input of the parameter scales with width but the output does not. This is called an *output
    weight* in the muP paper.
    """

    scaling_output = "scaling_output"
    """
    The output of the parameter scales with width but the input does not. This is called an *input
    weight* in the muP paper. This also applies to biases.
    """

    scaling_input_output = "scaling_input_output"
    """
    Both the input and output of the parameter scale with width. This is called a *hidden weight* in
    the muP paper.
    """


class MuPScalingStrategy(StrEnum):
    """
    Defines how much muP scales the initialization, outputs and LR of bigger/smaller models using muP.

    muP allows scaling up/down outputs in exchange for scaling initialization and LR in the opposite
    direction an appropriate amount. This permits having different scalings.
    """

    constant_inputs = "constant_inputs"
    """
    muP chooses scaling constants such that inputs/outputs are not scaled. This strategy means that the model
    submodules produces the same overall outputs with and without muP, with the exception of attention
    (which has its own scaling).
    
    This strategy corresponds to Table 3 of the muP paper. This strategy is NOT compatible with weight-tying,
    since parameters with scaling outputs have different initialization standard to those with scaling
    inputs.
    """

    constant_lr = "constant_lr"
    """
    muP chooses scaling constants such that optimizer parameter group LRs are not scaled.
    """

    constant_init_std = "constant_init_std"
    """
    muP chooses scaling constants such that the initialization stds of parameters are not scaled.
    """

    table_8 = "table_8"
    """
    The muP settings from Table 8 of the muP paper.
    """


@dataclass
class MuPConfig(Config):
    """
    Defines how to scale the initialization and outputs of bigger/smaller models using muP.
    """

    scaling_strategy: MuPScalingStrategy = MuPScalingStrategy.constant_inputs
    width_scalings: Dict[MuPHyperParam, float] = field(default_factory=dict)

    def __post_init__(self):
        if (
            MuPHyperParam.d_model in self.width_scalings
            and MuPHyperParam.hidden_size not in self.width_scalings
        ):
            raise OLMoConfigurationError(
                f"MuP scaling for {MuPHyperParam.d_model} is provided but scaling for {MuPHyperParam.hidden_size} is not. "
                f"This implies that {MuPHyperParam.d_model} is being varied but {MuPHyperParam.hidden_size} is not ."
                f"This is likely a mistake. If this is intentional, set scaling for {MuPHyperParam.hidden_size} to 1."
            )

        if (
            MuPHyperParam.n_heads in self.width_scalings
            and MuPHyperParam.n_kv_heads not in self.width_scalings
        ):
            log.warning(
                f"MuP scaling for {MuPHyperParam.n_heads} is provided but scaling for {MuPHyperParam.n_kv_heads} is not. "
                f"This implies that {MuPHyperParam.n_heads} is being varied but {MuPHyperParam.n_kv_heads} is not. "
                f"This is likely a mistake. If this is intentional, set scaling for {MuPHyperParam.n_kv_heads} to 1."
            )

        head_dim_scaling = self.width_scalings.get(MuPHyperParam.head_dim, 1.0)
        computed_head_dim_scaling = self.width_scalings.get(
            MuPHyperParam.d_model, 1.0
        ) / self.width_scalings.get(MuPHyperParam.n_heads, 1.0)
        if not math.isclose(head_dim_scaling, computed_head_dim_scaling):
            raise OLMoConfigurationError(
                f"MuP scaling {head_dim_scaling} for {MuPHyperParam.head_dim} does not match estimate "
                f"{computed_head_dim_scaling} computed from {MuPHyperParam.d_model} and {MuPHyperParam.n_heads}."
            )

    def _get_scaling(self, hyper_param_exponents: Dict[MuPHyperParam, float]) -> float:
        scalings = [
            math.pow(self.width_scalings.get(param, 1.0), exponent)
            for param, exponent in hyper_param_exponents.items()
        ]
        return math.prod(scalings)

    def build(
        self,
        input_hyper_param_exponents: Dict[MuPHyperParam, float],
        output_hyper_param_exponents: Dict[MuPHyperParam, float],
    ) -> "MuP":
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        kwargs.pop("width_scalings")

        input_scaling = self._get_scaling(input_hyper_param_exponents)
        output_scaling = self._get_scaling(output_hyper_param_exponents)

        try:
            return MuP(input_scaling, output_scaling, **kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e


@dataclass
class MuP:
    """
    Information needed to apply muP on a given parameter.
    """

    input_scaling: float
    output_scaling: float
    scaling_strategy: MuPScalingStrategy = MuPScalingStrategy.constant_inputs

    @cached_property
    def _scaling_type(self) -> MuPParamScalingType:
        scaling_input = not math.isclose(self.input_scaling, 1)
        scaling_output = not math.isclose(self.output_scaling, 1)

        if scaling_input and scaling_output:
            return MuPParamScalingType.scaling_input_output
        elif scaling_input:
            return MuPParamScalingType.scaling_input
        elif scaling_output:
            return MuPParamScalingType.scaling_output
        else:
            return MuPParamScalingType.constant

    @cached_property
    def input_multiplier(self) -> Optional[float]:
        scaling_type = self._scaling_type

        input_multiplier_map: Dict[MuPScalingStrategy, Dict[MuPParamScalingType, float]] = {
            MuPScalingStrategy.constant_inputs: {},
            MuPScalingStrategy.constant_lr: {
                MuPParamScalingType.scaling_input: 1.0 / self.input_scaling,
                MuPParamScalingType.scaling_input_output: 1.0 / self.input_scaling,
            },
            MuPScalingStrategy.constant_init_std: {
                MuPParamScalingType.scaling_input: 1.0 / self.input_scaling,
                MuPParamScalingType.scaling_input_output: 1.0 / math.sqrt(self.input_scaling),
            },
            MuPScalingStrategy.table_8: {
                MuPParamScalingType.scaling_input: 1.0 / self.input_scaling,
            },
        }

        if self.scaling_strategy not in input_multiplier_map:
            raise NotImplementedError(self.scaling_strategy)

        return input_multiplier_map[self.scaling_strategy].get(scaling_type)

    @cached_property
    def init_std_multiplier(self) -> Optional[float]:
        scaling_type = self._scaling_type

        init_std_multiplier_map: Dict[MuPScalingStrategy, Dict[MuPParamScalingType, float]] = {
            MuPScalingStrategy.constant_inputs: {
                MuPParamScalingType.scaling_input: 1.0 / self.input_scaling,
                MuPParamScalingType.scaling_input_output: 1.0 / math.sqrt(self.input_scaling),
            },
            MuPScalingStrategy.constant_lr: {
                MuPParamScalingType.scaling_input_output: math.sqrt(self.input_scaling),
            },
            MuPScalingStrategy.constant_init_std: {},
            MuPScalingStrategy.table_8: {
                MuPParamScalingType.scaling_input_output: 1.0 / math.sqrt(self.input_scaling),
            },
        }

        if self.scaling_strategy not in init_std_multiplier_map:
            raise NotImplementedError(self.scaling_strategy)

        return init_std_multiplier_map[self.scaling_strategy].get(scaling_type)

    @cached_property
    def lr_multiplier(self) -> Optional[float]:
        scaling_type = self._scaling_type

        lr_multiplier_map: Dict[MuPScalingStrategy, Dict[MuPParamScalingType, float]] = {
            MuPScalingStrategy.constant_inputs: {
                MuPParamScalingType.scaling_input: 1.0 / self.input_scaling,
                MuPParamScalingType.scaling_input_output: 1.0 / self.input_scaling,
            },
            MuPScalingStrategy.constant_lr: {},
            MuPScalingStrategy.constant_init_std: {
                MuPParamScalingType.scaling_input_output: 1.0 / math.sqrt(self.input_scaling),
            },
            MuPScalingStrategy.table_8: {
                MuPParamScalingType.scaling_input_output: 1.0 / self.input_scaling,
            },
        }

        if self.scaling_strategy not in lr_multiplier_map:
            raise NotImplementedError(self.scaling_strategy)

        return lr_multiplier_map[self.scaling_strategy].get(scaling_type)

    @cached_property
    def attention_multiplier(self) -> Optional[float]:
        scaling_type = self._scaling_type
        if scaling_type not in (MuPParamScalingType.constant, MuPParamScalingType.scaling_input):
            raise RuntimeError(
                f"MuP attention scale is only supported for {MuPParamScalingType.constant}"
                f"and {MuPParamScalingType.scaling_input} muP scaling types"
            )

        if scaling_type == MuPParamScalingType.scaling_input:
            return 1 / self.input_scaling

        return None

    @classmethod
    def scale_input(cls, mup: Optional["MuP"], x: torch.Tensor) -> torch.Tensor:
        """
        Convenient wrapper for applying muP input multiplier to reduce lines of code.
        This just performs ``x * mup_input_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.input_multiplier is None:
            return x

        return x * mup.input_multiplier

    @classmethod
    def scale_init_std(cls, mup: Optional["MuP"], std: float) -> float:
        """
        Convenient wrapper for applying muP init std multiplier to reduce lines of code.
        This just performs ``std * mup_std_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.init_std_multiplier is None:
            return std

        return std * mup.init_std_multiplier

    @classmethod
    def scale_lr(cls, mup: Optional["MuP"], lr: float) -> float:
        """
        Convenient wrapper for applying muP LR multiplier to reduce lines of code.
        This just performs ``lr * mup_lr_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.lr_multiplier is None:
            return lr

        return lr * mup.lr_multiplier

    @classmethod
    def named_mups(cls, model: nn.Module) -> Dict[str, "MuP"]:
        """
        An analogue of `model.named_parameters()` for getting all the MuPs of the model.
        """
        named_mups: Dict[str, "MuP"] = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "mups"):
                continue

            mups = getattr(module, "mups")
            assert isinstance(mups, dict)
            for param_name, mup in mups.items():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                named_mups[full_name] = mup

        return named_mups
