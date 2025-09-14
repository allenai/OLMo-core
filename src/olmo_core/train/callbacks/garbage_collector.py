import gc
import logging
from dataclasses import dataclass
from typing import Optional

from .callback import Callback
from ...aliases import PathOrStr

log = logging.getLogger(__name__)


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

    gc_interval: int = 1000
    enabled: bool = True
    _start_state: Optional[bool] = None

    def pre_train(self):
        if not self.enabled:
            return
        self._start_state = gc.isenabled()
        gc.disable()
        log.info(f"Automatic GC disabled for training, will run GC every {self.gc_interval} steps")

    def post_step(self):
        if not self.enabled:
            return
        if self.step % self.gc_interval == 0:
            if self.gc_interval > 10:
                log.info("Running garbage collection")
            gc.collect(1)

    def close(self):
        if not self.enabled:
            return
        if self._start_state:
            gc.enable()

    def post_checkpoint_saved(self, path: PathOrStr):
        del path
        if not self.enabled:
            return
        gc.collect(1)
