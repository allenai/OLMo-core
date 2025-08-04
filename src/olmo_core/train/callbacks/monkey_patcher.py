import functools
import logging
from dataclasses import dataclass

from torch.distributed import DeviceMesh

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MonkeyPatcherCallback(Callback):
    """
    While looking into performance issues with OLMo3 training, we discovered that
    `DeviceMesh.__getitem__` can become a bottleneck because it gets called very often by FSDP and
    creates a new sub-mesh object each time. So this callback patches that method to cache
    the sub-meshes.
    """

    def pre_train(self):
        # Cache DeviceMesh.__get_item__
        DeviceMesh.__getitem__ = functools.lru_cache(maxsize=None)(DeviceMesh.__getitem__)
