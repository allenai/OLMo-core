import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Set

from ...config import Config, StrEnum
from ...exceptions import OLMoConfigurationError

if TYPE_CHECKING:
    from .parametrization import ParametrizationBase

__all__ = [
    "WidthHyperParam",
    "ParametrizationOptimizerType",
    "ParametrizationScalingStrategy",
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


class WidthHyperParam(StrEnum):
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
    Configuration for applying the parametrization to scale initialization, outputs, and learning rates.
    """

    name = ParametrizationType.default
    """
    The type of parametrization.
    """

    optimizer: ParametrizationOptimizerType = ParametrizationOptimizerType.adam_coupled_wd
    """
    The type of optimizer being used for training. This affects how the parametrization scales learning rates and
    other factors.
    """

    scaling_strategy: ParametrizationScalingStrategy = (
        ParametrizationScalingStrategy.constant_inputs
    )
    """
    Controls how the parametrization balances scaling adjustments across inputs, initialization, and learning rates.
    This is possible due to Lemma J.1 of the Maximal Update Parametrization paper.
    """

    width_hyperparams: Dict[WidthHyperParam, float] = field(default_factory=dict)
    """
    The values of width-related hyperparameters.
    """

    base_model_width_hyperparams: Optional[Dict[WidthHyperParam, float]] = None
    """
    The values of width-related hyperparameters in a base model, if there is a base model. This is used to
    determine scaling factors relative to the base model.
    """

    def _validate_hyperparameters(self, width_hyperparams: Dict[WidthHyperParam, float]):
        if (
            WidthHyperParam.d_model in width_hyperparams
            and WidthHyperParam.hidden_size not in width_hyperparams
        ):
            raise OLMoConfigurationError(
                f"Parametrization scaling for {WidthHyperParam.d_model} is provided but scaling for {WidthHyperParam.hidden_size} is not. "
                f"This implies that {WidthHyperParam.d_model} is being varied but {WidthHyperParam.hidden_size} is not. "
                f"This is likely a mistake. If this is intentional, set scaling for {WidthHyperParam.hidden_size} to 1."
            )

        if (
            WidthHyperParam.n_heads in width_hyperparams
            and WidthHyperParam.n_kv_heads not in width_hyperparams
        ):
            raise OLMoConfigurationError(
                f"Parametrization scaling for {WidthHyperParam.n_heads} is provided but scaling for {WidthHyperParam.n_kv_heads} is not. "
                f"This implies that {WidthHyperParam.n_heads} is being varied but {WidthHyperParam.n_kv_heads} is not. "
                f"This may be a mistake. If this is intentional, set scaling for {WidthHyperParam.n_kv_heads} to 1."
            )

        head_dim_scaling = width_hyperparams.get(WidthHyperParam.head_dim, 1.0)
        computed_head_dim_scaling = width_hyperparams.get(
            WidthHyperParam.d_model, 1.0
        ) / width_hyperparams.get(WidthHyperParam.n_heads, 1.0)
        if not math.isclose(head_dim_scaling, computed_head_dim_scaling):
            raise OLMoConfigurationError(
                f"Parametrization scaling {head_dim_scaling} for {WidthHyperParam.head_dim} does not match estimate "
                f"{computed_head_dim_scaling} computed from {WidthHyperParam.d_model} and {WidthHyperParam.n_heads}."
            )

    def __post_init__(self):
        self._validate_hyperparameters(self.width_hyperparams)
        if self.base_model_width_hyperparams is not None:
            self._validate_hyperparameters(self.base_model_width_hyperparams)

    def build(
        self,
        input_dim_hyperparams: Optional[Set[WidthHyperParam]],
        output_dim_hyperparams: Optional[Set[WidthHyperParam]],
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
        from .parametrization import MaximalUpdateParametrization

        kwargs = self.as_dict(exclude_none=True, recurse=False)

        input_scaling = self._get_scaling(input_dim_hyperparams) if input_dim_hyperparams else 1.0
        output_scaling = (
            self._get_scaling(output_dim_hyperparams) if output_dim_hyperparams else 1.0
        )

        try:
            return MaximalUpdateParametrization(**kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e
