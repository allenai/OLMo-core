from typing import Dict, Generic, Optional, TypeVar

import torch.distributed as dist
import torch.nn as nn

from .sharded_flat_parameter import ShardedFlatParameter

M = TypeVar("M", bound=nn.Module)


class FSDP(nn.Module):
    pass


class FSDPNode(Generic[M], nn.Module):
    def __init__(self, module: M, process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self._fsdp_wrapped_module = module
        self._sharded_flat_params = shard_module(module, process_group=process_group)
        self.process_group = process_group
        # TODO: register backward hook

    @property
    def module(self) -> M:
        return self._fsdp_wrapped_module

    def forward(self, *args, **kwargs):
        # TODO: no idea if this works
        unshard_module(self.module)
        try:
            output = self.module(*args, **kwargs)
        finally:
            reshard_module(self.module, self._sharded_flat_params)
        return output


def shard_module(
    module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
) -> Dict[str, ShardedFlatParameter]:
    sharded_flat_params = {}
    for name, param in module.named_parameters():
        # TODO: use better sharding strategy that doesn't potentially always result in highest rank with
        # smallest shard.
        sharded_flat_param = ShardedFlatParameter.shard(param, process_group=process_group)
        setattr(module, name, sharded_flat_param)
        sharded_flat_params[name] = sharded_flat_param
    return sharded_flat_params


def unshard_module(module: nn.Module):
    for name, param in module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        setattr(module, name, param.gather())


def reshard_module(module: nn.Module, sharded_flat_params: Dict[str, ShardedFlatParameter]):
    for name, param in module.named_parameters():
        setattr(module, name, sharded_flat_params[name])
        del param
