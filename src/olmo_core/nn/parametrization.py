import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from ..config import Config, StrEnum
from ..exceptions import OLMoConfigurationError

__all__ = [
    "ParametrizationHyperParam",
    "ParametrizationOptimizerType",
    "ParametrizationScalingStrategy",
    "ParametrizationConfig",
    "Parametrization",
]


log = logging.getLogger(__name__)


class ParametrizationHyperParam(StrEnum):
    """
    Hyperparameters that the parametrization logic treats as affecting model ``width``.
    """

    d_model = "d_model"
    hidden_size = "hidden_size"
    n_heads = "n_heads"
    n_kv_heads = "n_kv_heads"
    head_dim = "head_dim"
    num_experts = "num_experts"
    shared_expert_hidden_size = "shared_expert_hidden_size"


class ParametrizationOptimizerType(StrEnum):
    """
    Optimizer variants that require distinct parametrization scaling behaviour.
    """

    adam = "adam"
    adam_coupled_wd = "adam_coupled_wd"
    """
    An AdamW optimizer where weight decay is coupled with the learning rate (i.e. the learning rate
    affects the weight decay.)
    """

    @property
    def coupled_weight_decay(self) -> bool:
        return self in (ParametrizationOptimizerType.adam_coupled_wd,)


class ParametrizationParameterScalingType(StrEnum):
    """
    Possible ways a parameter can scale with model ``width``.
    """

    constant = "constant"
    """
    The parameter's input and output are independent of width, so the parametrization does not adjust it.
    """

    scaling_input = "scaling_input"
    """
    The input of the parameter scales with width but the output does not. This is called an *output
    weight* in the Maximal Update Parametrization paper.
    """

    scaling_output = "scaling_output"
    """
    The output of the parameter scales with width but the input does not. This is called an *input
    weight* in the Maximal Update Parametrization paper. This also applies to biases.
    """

    scaling_input_output = "scaling_input_output"
    """
    Both the input and output of the parameter scale with width. This is called a *hidden weight* in
    the Maximal Update Parametrization paper.
    """


class ParametrizationScalingStrategy(StrEnum):
    """
    Strategies for how much the parametrization scales initialization, outputs, and learning rates of wider or
    narrower models.

    The parametrization allows trading off output scaling against initialization and learning-rate adjustments.
    Choosing a strategy fixes how that trade-off is applied.
    """

    constant_inputs = "constant_inputs"
    """
    The parametrization chooses scaling constants such that inputs/outputs are not scaled. This strategy means that the model
    submodules produce the same overall outputs with and without the parametrization, with the exception of attention
    (which has its own scaling).
    
    This strategy corresponds to Table 3 of the Maximal Update Parametrization paper. This strategy is NOT compatible with weight-tying,
    since parameters with scaling outputs have different initialization standard to those with scaling
    inputs.
    """

    constant_lr = "constant_lr"
    """
    The parametrization chooses scaling constants such that optimizer parameter group learning rates are not scaled.
    """

    constant_init_std = "constant_init_std"
    """
    The parametrization chooses scaling constants such that the initialization standard deviation of parameters is unchanged.
    """


@dataclass
class ParametrizationConfig(Config):
    """
    Configuration for applying the parametrization to scale initialization, outputs, and learning rates relative
    to a reference model.
    """

    optimizer: ParametrizationOptimizerType
    """
    The type of optimizer being used for training.
    """

    width_scalings: Dict[ParametrizationHyperParam, float]
    """
    A map that contains for each width-related hyperparameters the factor by which it has been increased/decreased
    in this model relative to the base model. For example, a mapping of ``ParametrizationHyperParam.d_model`` to ``0.5``
    means that the ``d_model`` of this model is half of the base model. A base model is not required to use the
    parametrization; scalings can instead encode the absolute sizes.
    """

    scaling_strategy: ParametrizationScalingStrategy = ParametrizationScalingStrategy.constant_inputs
    """
    Controls how the parametrization balances scaling adjustments across inputs, initialization, and learning rates.
    """

    def __post_init__(self):
        if (
            ParametrizationHyperParam.d_model in self.width_scalings
            and ParametrizationHyperParam.hidden_size not in self.width_scalings
        ):
            raise OLMoConfigurationError(
                f"Parametrization scaling for {ParametrizationHyperParam.d_model} is provided but scaling for {ParametrizationHyperParam.hidden_size} is not. "
                f"This implies that {ParametrizationHyperParam.d_model} is being varied but {ParametrizationHyperParam.hidden_size} is not. "
                f"This is likely a mistake. If this is intentional, set scaling for {ParametrizationHyperParam.hidden_size} to 1."
            )

        if (
            ParametrizationHyperParam.n_heads in self.width_scalings
            and ParametrizationHyperParam.n_kv_heads not in self.width_scalings
        ):
            raise OLMoConfigurationError(
                f"Parametrization scaling for {ParametrizationHyperParam.n_heads} is provided but scaling for {ParametrizationHyperParam.n_kv_heads} is not. "
                f"This implies that {ParametrizationHyperParam.n_heads} is being varied but {ParametrizationHyperParam.n_kv_heads} is not. "
                f"This may be a mistake. If this is intentional, set scaling for {ParametrizationHyperParam.n_kv_heads} to 1."
            )

        head_dim_scaling = self.width_scalings.get(ParametrizationHyperParam.head_dim, 1.0)
        computed_head_dim_scaling = self.width_scalings.get(
            ParametrizationHyperParam.d_model, 1.0
        ) / self.width_scalings.get(ParametrizationHyperParam.n_heads, 1.0)
        if not math.isclose(head_dim_scaling, computed_head_dim_scaling):
            raise OLMoConfigurationError(
                f"Parametrization scaling {head_dim_scaling} for {ParametrizationHyperParam.head_dim} does not match estimate "
                f"{computed_head_dim_scaling} computed from {ParametrizationHyperParam.d_model} and {ParametrizationHyperParam.n_heads}."
            )

    def _get_scaling(self, hyper_param_exponents: Set[ParametrizationHyperParam]) -> float:
        scalings = [self.width_scalings.get(param, 1.0) for param in hyper_param_exponents]
        return math.prod(scalings)

    def build(
        self,
        input_dim_hyperparams: Optional[Set[ParametrizationHyperParam]],
        output_dim_hyperparams: Optional[Set[ParametrizationHyperParam]],
    ) -> "Parametrization":
        """
        Build a `Parametrization` helper for applying scaling to parameters with the given input and output
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
            return Parametrization(input_scaling, output_scaling, optimizer, **kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e


@dataclass
class Parametrization:
    """
    Encapsulates scaling factors for applying the parametrization to a single parameter.
    """

    input_scaling: float
    output_scaling: float
    optimizer: ParametrizationOptimizerType
    scaling_strategy: ParametrizationScalingStrategy = ParametrizationScalingStrategy.constant_inputs

    @property
    def _scaling_type(self) -> ParametrizationParameterScalingType:
        scaling_input = not math.isclose(self.input_scaling, 1)
        scaling_output = not math.isclose(self.output_scaling, 1)

        if scaling_input and scaling_output:
            return ParametrizationParameterScalingType.scaling_input_output
        elif scaling_input:
            return ParametrizationParameterScalingType.scaling_input
        elif scaling_output:
            return ParametrizationParameterScalingType.scaling_output
        else:
            return ParametrizationParameterScalingType.constant

    @property
    def _base_parametrization_scalars(self) -> Tuple[float, float, float]:
        """
        Returns the trio of (input multiplier, init std multiplier, optimizer LR multiplier)
        """

        # These scalars come from Table 3 of the Maximal Update Parametrization paper.
        base_scalars_map: Dict[
            ParametrizationOptimizerType, Dict[ParametrizationParameterScalingType, Tuple[float, float, float]]
        ] = {
            ParametrizationOptimizerType.adam: {
                ParametrizationParameterScalingType.scaling_input: (1.0 / self.input_scaling, 1.0, 1.0),
                ParametrizationParameterScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_scaling),
                    1.0 / self.input_scaling,
                ),
            },
            ParametrizationOptimizerType.adam_coupled_wd: {
                ParametrizationParameterScalingType.scaling_input: (1.0 / self.input_scaling, 1.0, 1.0),
                ParametrizationParameterScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_scaling),
                    1.0 / self.input_scaling,
                ),
            },
        }

        input_multiplier, init_std_multiplier, lr_multiplier = base_scalars_map.get(
            self.optimizer, {}
        ).get(self._scaling_type, (1.0, 1.0, 1.0))

        # Lemma J.1 of the Maximal Update Parametrization paper allows for scaling the input multiplier up by a factor K by
        # scaling down the init std and LR a corresponding amount. The LR is scaled down by
        # K for Adam variants since Adam normalizes gradients, and K^2 for SGD variants since SGD
        # updates are proportional to gradient.
        scale_lr_twice = self.optimizer not in (
            ParametrizationOptimizerType.adam,
            ParametrizationOptimizerType.adam_coupled_wd,
        )
        if self.scaling_strategy == ParametrizationScalingStrategy.constant_inputs:
            input_rescaling = 1 / input_multiplier
            init_std_rescaling = input_multiplier
            lr_rescaling = input_multiplier**2 if scale_lr_twice else input_multiplier
        elif self.scaling_strategy == ParametrizationScalingStrategy.constant_init_std:
            input_rescaling = init_std_multiplier
            init_std_rescaling = 1 / init_std_multiplier
            lr_rescaling = (
                (1 / init_std_multiplier) ** 2 if scale_lr_twice else 1 / init_std_multiplier
            )
        elif self.scaling_strategy == ParametrizationScalingStrategy.constant_lr:
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
        input_multiplier = self._base_parametrization_scalars[0]

        return input_multiplier if not math.isclose(input_multiplier, 1.0) else None

    @property
    def init_std_multiplier(self) -> Optional[float]:
        init_std_multiplier = self._base_parametrization_scalars[1]

        return init_std_multiplier if not math.isclose(init_std_multiplier, 1.0) else None

    @property
    def lr_multiplier(self) -> Optional[float]:
        lr_multiplier = self._base_parametrization_scalars[2]

        return lr_multiplier if not math.isclose(lr_multiplier, 1.0) else None

    @property
    def attention_multiplier(self) -> Optional[float]:
        scaling_type = self._scaling_type
        if scaling_type not in (ParametrizationParameterScalingType.constant, ParametrizationParameterScalingType.scaling_input):
            raise RuntimeError(
                f"Parametrization attention scale is only supported for {ParametrizationParameterScalingType.constant} "
                f"and {ParametrizationParameterScalingType.scaling_input} parametrization scaling types"
            )

        if scaling_type == ParametrizationParameterScalingType.scaling_input:
            return 1 / self.input_scaling

        return None

    @classmethod
    def scale_input(cls, parametrization: Optional["Parametrization"], x: torch.Tensor) -> torch.Tensor:
        """
        Convenient wrapper for applying the parametrization input multiplier.
        This just performs ``x * parametrization_input_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.input_multiplier is None:
            return x

        return (x * parametrization.input_multiplier).to(x.dtype)

    @classmethod
    def scale_init_std(cls, parametrization: Optional["Parametrization"], std: float) -> float:
        """
        Convenient wrapper for applying the parametrization init std multiplier.
        This just performs ``std * parametrization_std_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.init_std_multiplier is None:
            return std

        return std * parametrization.init_std_multiplier

    @classmethod
    def scale_lr(cls, parametrization: Optional["Parametrization"], lr: float) -> float:
        """
        Convenient wrapper for applying the parametrization LR multiplier.
        This just performs ``lr * parametrization_lr_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.lr_multiplier is None:
            return lr

        return lr * parametrization.lr_multiplier

    @classmethod
    def scale_coupled_wd(cls, parametrization: Optional["Parametrization"], weight_decay: float) -> float:
        """
        Convenient wrapper for un-applying parametrization LR multiplier to coupled weight decay.
        This just performs ``weight_decay / parametrization_lr_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.lr_multiplier is None:
            return weight_decay

        if not parametrization.optimizer.coupled_weight_decay:
            raise RuntimeError(
                f"Scaling coupled weight decay is not supported for optimizer {parametrization.optimizer} with uncoupled weight decay"
            )

        return weight_decay / parametrization.lr_multiplier

    @classmethod
    def named_parametrizations(cls, model: nn.Module) -> Dict[str, "Parametrization"]:
        """
        An analogue of `model.named_parameters()` for getting all the Parametrizations of the model.
        """
        named_parametrizations: Dict[str, "Parametrization"] = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue

            parametrizations = getattr(module, "parametrizations")
            assert isinstance(parametrizations, dict)
            for param_name, parametrization in parametrizations.items():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                named_parametrizations[full_name] = parametrization

        return named_parametrizations
