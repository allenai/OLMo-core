from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...config import Config, StrEnum
from ...exceptions import OLMoConfigurationError

if TYPE_CHECKING:
    from .parametrization import ParametrizationBase

__all__ = [
    "ParametrizationOptimizerType",
    "MupScalingStrategy",
    "ParametrizationConfig",
]


class ParametrizationType(StrEnum):
    """
    An enumeration of types of parametrizations.
    """

    default = "default"
    """
    ➡️ :class:`StandardParametrization`
    """

    mup = "mup"
    """
    ➡️ :class:`MaximalUpdateParametrization`
    """


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


class MupScalingStrategy(StrEnum):
    """
    Strategies for how much MuP scales initialization, outputs, and learning rates of wider or
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
    Configuration for applying the parametrization to scale initialization, outputs, and learning rates.
    """

    name: ParametrizationType = ParametrizationType.default
    """
    The type of parametrization.
    """

    optimizer: ParametrizationOptimizerType = ParametrizationOptimizerType.adam_coupled_wd
    """
    The type of optimizer being used for training. This affects how the parametrization scales learning rates and
    other factors.
    """

    scaling_strategy: Optional[MupScalingStrategy] = None
    """
    Controls how the parametrization balances scaling adjustments across inputs, initialization, and learning rates.
    This is possible due to Lemma J.1 of the Maximal Update Parametrization paper.
    """

    def build(
        self,
        input_dim: int,
        output_dim: int,
    ) -> "ParametrizationBase":
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
        from .parametrization import (
            MaximalUpdateParametrization,
            StandardParametrization,
        )

        kwargs = self.as_dict(exclude_none=True, recurse=False)

        kwargs.pop("name")
        kwargs.update(
            input_dim=input_dim,
            output_dim=output_dim,
        )

        try:
            if self.name == ParametrizationType.default:
                return StandardParametrization(**kwargs)
            elif self.name == ParametrizationType.mup:
                return MaximalUpdateParametrization(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e
