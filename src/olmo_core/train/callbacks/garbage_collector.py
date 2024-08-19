import gc
from dataclasses import dataclass
from typing import Optional

from .callback import Callback


@dataclass
class GarbageCollector(Callback):
    """
    Disables automatic garbage collection during training and runs gen1 collection
    on a set schedule instead.
    """

    gc_interval: int = 100
    _start_state: Optional[bool] = None

    def pre_train(self):
        self._start_state = gc.isenabled()
        gc.disable()

    def post_step(self, step: int):
        if step % self.gc_interval == 0:
            gc.collect(1)

    def post_train(self):
        if self._start_state:
            gc.enable()
