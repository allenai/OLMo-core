from collections import defaultdict
from collections.abc import MutableMapping
from typing import Dict, Optional

import torch

from olmo_core.utils import move_to_device


class BufferCache(MutableMapping[str, Optional[torch.Tensor]]):
    """
    Cache for buffers such as attention biases that would normally be registered as module buffers.

    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.

    :param namespace: Optional namespace for the cache. This allows you to have a separate sub-cache
        in a shared :class:`BufferCache` to avoid key collisions. See how this is used in the
        :class:`olmo_core.nn.rope.RotaryEmbeddingBase` class for an example.
    """

    def __init__(self, namespace: str = ""):
        self._data: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self._namespace = namespace

    def __getitem__(self, key: str) -> Optional[torch.Tensor]:
        return self._data[self._namespace].get(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._data[self._namespace][key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[self._namespace][key]

    def __iter__(self):
        yield from self._data[self._namespace].keys()

    def __len__(self) -> int:
        return len(self._data[self._namespace])

    def get_for_device(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        if (tensor := self.get(key)) is not None:
            if tensor.device != device:
                tensor = move_to_device(tensor, device)
                self[key] = tensor
            return tensor
        else:
            return None

    def with_namespace(self, namespace: str) -> "BufferCache":
        """
        This creates a new :class:`BufferCache` object with a pointer to the same underlying data
        but with the given namespace.
        """
        out = BufferCache(namespace=namespace)
        out._data = self._data
        return out
