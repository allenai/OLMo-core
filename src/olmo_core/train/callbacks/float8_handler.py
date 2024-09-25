from dataclasses import dataclass, field

from olmo_core.float8 import Float8Config, Float8Handler

from .callback import Callback


@dataclass
class Float8HandlerCallback(Callback):
    """
    A callback for enabling Float8 training via `torchao <https://github.com/pytorch/ao>`_.

    .. seealso::
        See :class:`~olmo_core.float8.Float8Handler` for parameter descriptions.
    """

    config: Float8Config = field(default_factory=Float8Config)

    _handler = None

    @property
    def handler(self) -> Float8Handler:
        if self._handler is None:
            self._handler = self.config.build()
        return self._handler

    def post_attach(self):
        if not self.config.enabled:
            return

        self.handler

    def pre_optim_step(self):
        if not self.config.enabled:
            return

        # Sync Float8 AMAXs (argmax of abs(max)) and scales.
        self.handler.sync_float8_amax_and_scale_history(self.trainer.model)

    def post_train_batch(self):
        if not self.config.enabled:
            return

        # Calculate Float8 dynamic AMAX/scale for all parameters.
        # For FSDP2 is issues a single all-reduce for all parameters at once.
        self.handler.precompute_float8_dynamic_scale_for_fsdp(self.trainer.model)
