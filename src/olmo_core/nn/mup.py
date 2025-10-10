import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from ..config import Config, StrEnum
from ..exceptions import OLMoConfigurationError

__all__ = [
    "MuPHyperParam",
    "MuPOptimizerType",
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
    num_experts = "num_experts"
    shared_expert_hidden_size = "shared_expert_hidden_size"


class MuPOptimizerType(StrEnum):
    """
    An enumeration of the different types of optimizers. MuP scaling depends on the type of optimizer.
    """

    adam = "adam"
    adam_coupled_wd = "adam_coupled_wd"
    """
    An AdamW optimizer where weight decay is coupled with the learning rate (i.e. the learning rate
    affects the weight decay.)
    """

    @property
    def coupled_weight_decay(self) -> bool:
        return self in (MuPOptimizerType.adam_coupled_wd,)


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


@dataclass
class MuPConfig(Config):
    """
    Defines how to scale the initialization, outputs and learning rates of bigger/smaller models using muP.
    """

    optimizer: MuPOptimizerType
    """
    The type of optimizer being used for training.
    """

    width_scalings: Dict[MuPHyperParam, float]
    """
    A map that contains for each width-related hyperparameters the factor by which it has been increased/decreased
    in this model relative to in the base model. For example, a mapping of ``MuPHyperParam.d_model`` to ``0.5``
    means that the ``d_model`` of this model is half of the base model. A base model is not needed to use muP,
    in which case you can set all scalings to, say, the hyperparameter's value.
    """

    scaling_strategy: MuPScalingStrategy = MuPScalingStrategy.constant_inputs
    """
    The strategy for how muP should scale the initialization, outputs and LR of bigger/smaller models.
    """

    def __post_init__(self):
        if (
            MuPHyperParam.d_model in self.width_scalings
            and MuPHyperParam.hidden_size not in self.width_scalings
        ):
            raise OLMoConfigurationError(
                f"MuP scaling for {MuPHyperParam.d_model} is provided but scaling for {MuPHyperParam.hidden_size} is not. "
                f"This implies that {MuPHyperParam.d_model} is being varied but {MuPHyperParam.hidden_size} is not. "
                f"This is likely a mistake. If this is intentional, set scaling for {MuPHyperParam.hidden_size} to 1."
            )

        if (
            MuPHyperParam.n_heads in self.width_scalings
            and MuPHyperParam.n_kv_heads not in self.width_scalings
        ):
            raise OLMoConfigurationError(
                f"MuP scaling for {MuPHyperParam.n_heads} is provided but scaling for {MuPHyperParam.n_kv_heads} is not. "
                f"This implies that {MuPHyperParam.n_heads} is being varied but {MuPHyperParam.n_kv_heads} is not. "
                f"This may be a mistake. If this is intentional, set scaling for {MuPHyperParam.n_kv_heads} to 1."
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

    def _get_scaling(self, hyper_param_exponents: Set[MuPHyperParam]) -> float:
        scalings = [self.width_scalings.get(param, 1.0) for param in hyper_param_exponents]
        return math.prod(scalings)

    def build(
        self,
        input_dim_hyperparams: Optional[Set[MuPHyperParam]],
        output_dim_hyperparams: Optional[Set[MuPHyperParam]],
    ) -> "MuP":
        """
        Build the MuP object for applying muP scaling to parameters with the given input and output
        dimension hyperparameters.

        Note: Each hyperparameter is assumed to affect dimension linearly. For example, if
        ``d_model`` is doubled, the input (resp. output) dimensions of the parameter are assumed to be
        doubled if ``d_model`` is in ``input_dim_hyperparams`` (resp. ``output_dim_hyperparams``).

        :param input_dim_hyperparams: A set of hyperparameters that affect the input dimension of
            the parameter.
        :param output_dim_hyperparams: A set of hyperparameters that affect the output dimension of
            the parameter.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        kwargs.pop("width_scalings")
        optimizer = kwargs.pop("optimizer")

        input_scaling = self._get_scaling(input_dim_hyperparams) if input_dim_hyperparams else 1.0
        output_scaling = (
            self._get_scaling(output_dim_hyperparams) if output_dim_hyperparams else 1.0
        )

        try:
            return MuP(input_scaling, output_scaling, optimizer, **kwargs)
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
    optimizer: MuPOptimizerType
    scaling_strategy: MuPScalingStrategy = MuPScalingStrategy.constant_inputs

    @property
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

    @property
    def _base_mup_scalars(self) -> Tuple[float, float, float]:
        """
        Returns the trio of (input multiplier, init std multiplier, optimizer LR multiplier)
        """

        # These scalars come from Table 3 of the muP paper.
        base_scalars_map: Dict[
            MuPOptimizerType, Dict[MuPParamScalingType, Tuple[float, float, float]]
        ] = {
            MuPOptimizerType.adam: {
                MuPParamScalingType.scaling_input: (1.0 / self.input_scaling, 1.0, 1.0),
                MuPParamScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_scaling),
                    1.0 / self.input_scaling,
                ),
            },
            MuPOptimizerType.adam_coupled_wd: {
                MuPParamScalingType.scaling_input: (1.0 / self.input_scaling, 1.0, 1.0),
                MuPParamScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_scaling),
                    1.0 / self.input_scaling,
                ),
            },
        }

        input_multiplier, init_std_multiplier, lr_multiplier = base_scalars_map.get(
            self.optimizer, {}
        ).get(self._scaling_type, (1.0, 1.0, 1.0))

        # Lemma J.1 of the muP paper allows for scaling the input multiplier up by a factor K by
        # scaling down the init std and LR a corresponding amount. The LR is scaled down by
        # K for Adam variants since Adam normalizes gradients, and K^2 for SGD variants since SGD
        # updates are proportional to gradient.
        scale_lr_twice = self.optimizer not in (
            MuPOptimizerType.adam,
            MuPOptimizerType.adam_coupled_wd,
        )
        if self.scaling_strategy == MuPScalingStrategy.constant_inputs:
            input_rescaling = 1 / input_multiplier
            init_std_rescaling = input_multiplier
            lr_rescaling = input_multiplier**2 if scale_lr_twice else input_multiplier
        elif self.scaling_strategy == MuPScalingStrategy.constant_init_std:
            input_rescaling = init_std_multiplier
            init_std_rescaling = 1 / init_std_multiplier
            lr_rescaling = (
                (1 / init_std_multiplier) ** 2 if scale_lr_twice else 1 / init_std_multiplier
            )
        elif self.scaling_strategy == MuPScalingStrategy.constant_lr:
            input_rescaling = lr_multiplier
            init_std_rescaling = 1 / lr_multiplier
            lr_rescaling = 1 / lr_multiplier
            if scale_lr_twice:
                input_rescaling = math.sqrt(input_rescaling)
                init_std_rescaling = math.sqrt(init_std_rescaling)
        else:
            raise NotImplementedError(self.scaling_strategy)

        input_multiplier *= input_rescaling
        init_std_multiplier *= init_std_rescaling
        lr_multiplier *= lr_rescaling

        return (input_multiplier, init_std_multiplier, lr_multiplier)

    @property
    def input_multiplier(self) -> Optional[float]:
        input_multiplier = self._base_mup_scalars[0]

        return input_multiplier if not math.isclose(input_multiplier, 1.0) else None

    @property
    def init_std_multiplier(self) -> Optional[float]:
        init_std_multiplier = self._base_mup_scalars[1]

        return init_std_multiplier if not math.isclose(init_std_multiplier, 1.0) else None

    @property
    def lr_multiplier(self) -> Optional[float]:
        lr_multiplier = self._base_mup_scalars[2]

        return lr_multiplier if not math.isclose(lr_multiplier, 1.0) else None

    @property
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
        Convenient wrapper for applying the muP input multiplier.
        This just performs ``x * mup_input_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.input_multiplier is None:
            return x

        return (x * mup.input_multiplier).to(x.dtype)

    @classmethod
    def scale_init_std(cls, mup: Optional["MuP"], std: float) -> float:
        """
        Convenient wrapper for applying the muP init std multiplier.
        This just performs ``std * mup_std_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.init_std_multiplier is None:
            return std

        return std * mup.init_std_multiplier

    @classmethod
    def scale_lr(cls, mup: Optional["MuP"], lr: float) -> float:
        """
        Convenient wrapper for applying the muP LR multiplier.
        This just performs ``lr * mup_lr_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.lr_multiplier is None:
            return lr

        return lr * mup.lr_multiplier

    @classmethod
    def scale_coupled_wd(cls, mup: Optional["MuP"], weight_decay: float) -> float:
        """
        Convenient wrapper for un-applying muP LR multiplier to coupled weight decay.
        This just performs ``weight_decay / mup_lr_multiplier``, if the multiplier exists.
        """
        if mup is None or mup.lr_multiplier is None:
            return weight_decay

        if not mup.optimizer.coupled_weight_decay:
            raise RuntimeError(
                f"Scaling coupled weight decay is not supported for optimizer {mup.optimizer} with uncoupled weight decay"
            )

        return weight_decay / mup.lr_multiplier

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
