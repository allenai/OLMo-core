from dataclasses import dataclass

from olmo_core.float8 import Float8Handler, Float8ScalingType

from .callback import Callback


@dataclass
class Float8HandlerCallback(Callback):
    """
    A callback for enabling Float8 training via `torchao <https://github.com/pytorch/ao>`_.

    .. seealso::
        See :class:`~olmo_core.float8.Float8Handler` for parameter descriptions.
    """

    scaling_type_input: Float8ScalingType = Float8ScalingType.dynamic
    scaling_type_weight: Float8ScalingType = Float8ScalingType.dynamic
    scaling_type_grad_output: Float8ScalingType = Float8ScalingType.dynamic
    enable_fsdp_float8_all_gather: bool = True
    precompute_float8_dynamic_scale_for_fsdp: bool = True
    compile: bool = True
    enabled: bool = True

    _handler = None

    @property
    def handler(self) -> Float8Handler:
        if self._handler is None:
            raise RuntimeError("Float8Handler has not been configured yet")
        return self._handler

    def post_attach(self):
        if not self.enabled:
            return

        self._handler = Float8Handler(
            scaling_type_input=self.scaling_type_input,
            scaling_type_weight=self.scaling_type_weight,
            scaling_type_grad_output=self.scaling_type_grad_output,
            enable_fsdp_float8_all_gather=self.enable_fsdp_float8_all_gather,
            precompute_float8_dynamic_scale_for_fsdp=self.precompute_float8_dynamic_scale_for_fsdp,
            compile=self.compile,
        )

        # Swap `nn.Linear` modules with `Float8Linear` in place.
        self.handler.convert_to_float8_training(self.trainer.model)

    def pre_optim_step(self):
        if not self.enabled:
            return

        # Sync Float8 AMAXs (argmax of abs(max)) and scales.
        self.handler.sync_float8_amax_and_scale_history(self.trainer.model)

    def post_train_batch(self):
        if not self.enabled:
            return

        # Calculate Float8 dynamic AMAX/scale for all parameters.
        # For FSDP2 is issues a single all-reduce for all parameters at once.
        self.handler.precompute_float8_dynamic_scale_for_fsdp(self.trainer.model)
