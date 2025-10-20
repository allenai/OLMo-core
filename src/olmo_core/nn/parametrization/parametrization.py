import math
from abc import abstractmethod
from collections.abc import Set
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...config import StrEnum
from .config import (
    ParametrizationOptimizerType,
    ParametrizationScalingStrategy,
    ParametrizationType,
    WidthHyperParam,
)

__all__ = ["ParametrizationBase"]


class ParameterScalingType(StrEnum):
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


class ParametrizationBase:
    """
    Base class for parametrization implementations.
    """

    def __init__(self, *, optimizer: ParametrizationOptimizerType):
        self.optimizer = optimizer

    @property
    @abstractmethod
    def input_multiplier(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def init_std_multiplier(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def lr_multiplier(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def attention_multiplier(self) -> Optional[float]:
        pass

    @classmethod
    def scale_input(
        cls, parametrization: Optional["ParametrizationBase"], x: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenient wrapper for applying the parametrization input multiplier.
        This just performs ``x * parametrization_input_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.input_multiplier is None:
            return x

        return (x * parametrization.input_multiplier).to(x.dtype)

    @classmethod
    def scale_init_std(cls, parametrization: Optional["ParametrizationBase"], std: float) -> float:
        """
        Convenient wrapper for applying the parametrization init std multiplier.
        This just performs ``std * parametrization_std_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.init_std_multiplier is None:
            return std

        return std * parametrization.init_std_multiplier

    @classmethod
    def scale_lr(cls, parametrization: Optional["ParametrizationBase"], lr: float) -> float:
        """
        Convenient wrapper for applying the parametrization LR multiplier.
        This just performs ``lr * parametrization_lr_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.lr_multiplier is None:
            return lr

        return lr * parametrization.lr_multiplier

    @classmethod
    def scale_coupled_wd(
        cls, parametrization: Optional["ParametrizationBase"], weight_decay: float
    ) -> float:
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
    def named_parametrizations(cls, model: nn.Module) -> Dict[str, "ParametrizationBase"]:
        """
        An analogue of ``model.named_parameters()`` for getting all the Parametrizations of the model.
        """
        named_parametrizations: Dict[str, "ParametrizationBase"] = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue

            parametrizations = getattr(module, "parametrizations")
            assert isinstance(parametrizations, dict)
            for param_name, parametrization in parametrizations.items():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                named_parametrizations[full_name] = parametrization

        return named_parametrizations


@dataclass
class StandardParametrization(ParametrizationBase):
    """
    The standard parametrization, where no scaling is applied.
    """

    @property
    def input_multiplier(self) -> Optional[float]:
        return None

    @property
    def init_std_multiplier(self) -> Optional[float]:
        return None

    @property
    def lr_multiplier(self) -> Optional[float]:
        return None

    @property
    def attention_multiplier(self) -> Optional[float]:
        return None


class MaximalUpdateParametrization(ParametrizationBase):
    """
    Encapsulates scaling factors for applying the parametrization to a single parameter.
    """

    def __init__(
        self,
        *,
        name: ParametrizationType,
        optimizer: ParametrizationOptimizerType,
        scaling_strategy: ParametrizationScalingStrategy,
        width_hyperparams: Dict[WidthHyperParam, float],
        base_model_width_hyperparams: Optional[Dict[WidthHyperParam, float]],
        input_dim_hyperparams: Optional[Set[WidthHyperParam]],
        output_dim_hyperparams: Optional[Set[WidthHyperParam]],
    ):
        super().__init__(
            optimizer=optimizer,
        )
        self.name = name
        self.scaling_strategy = scaling_strategy
        self.width_hyperparams = width_hyperparams
        self.base_model_width_hyperparams = base_model_width_hyperparams
        self.input_dim_hyperparams = input_dim_hyperparams or set()
        self.output_dim_hyperparams = output_dim_hyperparams or set()
        self.input_scaling = self._get_scaling(self.input_dim_hyperparams)
        self.output_scaling = self._get_scaling(self.output_dim_hyperparams)
        self.scaling_type = self._get_scaling_type(
            input_scaling=self.input_scaling, output_scaling=self.output_scaling
        )

        input_multiplier, init_std_multiplier, lr_multiplier = self._get_parametrization_scalars()
        self._input_multiplier = (
            input_multiplier if not math.isclose(input_multiplier, 1.0) else None
        )
        self._init_std_multiplier = (
            init_std_multiplier if not math.isclose(init_std_multiplier, 1.0) else None
        )
        self._lr_multiplier = lr_multiplier if not math.isclose(lr_multiplier, 1.0) else None
        self._attention_multiplier = (
            1 / self.input_scaling if not math.isclose(self.input_scaling, 1.0) else None
        )

    def _get_scaling(self, hyperparams: Set[WidthHyperParam]) -> float:
        base_model_width_hyperparams = self.base_model_width_hyperparams or {}
        scalings = [
            self.width_hyperparams.get(hyperparam, 1.0)
            / base_model_width_hyperparams.get(hyperparam, 1.0)
            for hyperparam in hyperparams
        ]
        return math.prod(scalings)

    def _get_scaling_type(
        self, input_scaling: float, output_scaling: float
    ) -> ParameterScalingType:
        scaling_input = not math.isclose(input_scaling, 1)
        scaling_output = not math.isclose(output_scaling, 1)

        if scaling_input and scaling_output:
            return ParameterScalingType.scaling_input_output
        elif scaling_input:
            return ParameterScalingType.scaling_input
        elif scaling_output:
            return ParameterScalingType.scaling_output
        else:
            return ParameterScalingType.constant

    def _get_parametrization_scalars(self) -> Tuple[float, float, float]:
        """
        Returns the trio of (input multiplier, init std multiplier, optimizer LR multiplier)
        """

        # These scalars come from Table 3 of the Maximal Update Parametrization paper.
        base_scalars_map: Dict[
            ParametrizationOptimizerType,
            Dict[ParameterScalingType, Tuple[float, float, float]],
        ] = {
            ParametrizationOptimizerType.adam: {
                ParameterScalingType.scaling_input: (
                    1.0 / self.input_scaling,
                    1.0,
                    1.0,
                ),
                ParameterScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_scaling),
                    1.0 / self.input_scaling,
                ),
            },
            ParametrizationOptimizerType.adam_coupled_wd: {
                ParameterScalingType.scaling_input: (
                    1.0 / self.input_scaling,
                    1.0,
                    1.0,
                ),
                ParameterScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_scaling),
                    1.0 / self.input_scaling,
                ),
            },
        }

        input_multiplier, init_std_multiplier, lr_multiplier = base_scalars_map.get(
            self.optimizer, {}
        ).get(self.scaling_type, (1.0, 1.0, 1.0))

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
        return self._input_multiplier

    @property
    def init_std_multiplier(self) -> Optional[float]:
        return self._init_std_multiplier

    @property
    def lr_multiplier(self) -> Optional[float]:
        return self._lr_multiplier

    @property
    def attention_multiplier(self) -> Optional[float]:
        return self._attention_multiplier
