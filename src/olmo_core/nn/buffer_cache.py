from collections.abc import MutableMapping
from typing import Optional

import torch

from olmo_core.utils import move_to_device


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """

    def get_for_device(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        if (tensor := self.get(key)) is not None:
            if tensor.device != device:
                tensor = move_to_device(tensor, device)
                self[key] = tensor
            return tensor
        else:
            return None
