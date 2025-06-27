import functools
import logging
from dataclasses import dataclass

from torch.distributed import DeviceMesh

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MonkeyPatcherCallback(Callback):
    def pre_train(self):
        # Cache DeviceMesh.__get_item__
        DeviceMesh.__get_item__ = functools.lru_cache(maxsize=None)(DeviceMesh.__get_item__)
