import gc
from dataclasses import dataclass
from typing import Optional

from .callback import Callback


@dataclass
class GarbageCollectorCallback(Callback):
    """
    Disables automatic garbage collection during training and runs gen1 collection
    on a set schedule instead.

    .. important::
        This callback gets added automatically in a distributed training setting if you
        don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    gc_interval: int = 100
    _start_state: Optional[bool] = None

    def pre_train(self):
        self._start_state = gc.isenabled()
        gc.disable()

    def post_step(self):
        if self.step % self.gc_interval == 0:
            gc.collect(1)

    def post_train(self):
        if self._start_state:
            gc.enable()
