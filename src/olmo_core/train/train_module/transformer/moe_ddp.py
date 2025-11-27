
import weakref
from collections.abc import Iterable
from typing import Any, NoReturn, Optional

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel

_ROOT_MODULE_PREFIX = ""

class ComposableDistributedDataParallel(nn.Module):

    def __init__(
        self,
        *args,
        **kwargs,       
    ):
        super().__init__()
        self.module: nn.Module = nn.ParameterList()
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        # TODO(@fegin): this variable is originally create for testing, we
        # should remove this if possible.
        self._orig_module = self.module
        self._param_names: list[str] = []
        self._no_sync: bool = False
        # self._init_args: Optional[tuple[Any, ...]] = None
        # self._init_kwargs: dict[str, Any] = {}
        self._comm_hook_args: list[Any] = []

        self._init_args = args
        self._init_kwargs = kwargs

        self._ddp: Optional[DistributedDataParallel] = None



    def lazy_init(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        **kwargs,
    ):
        if self.has_initialized:
            return

        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}

        from torch.distributed.tensor.parallel.ddp import _localize_dtensor

        _localize_dtensor(module, ignored_params=ignored_params)
        self._collect_params(module, ignored_modules, ignored_params)

        self._ddp = DistributedDataParallel(self._param_list, **kwargs)

    def _collect_params(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        ignored_params: set[nn.Parameter],
        prefix: str = _ROOT_MODULE_PREFIX,
    ) -> None:

        # if a module is ignored, all descendants of the module are ignored.
        if module in ignored_modules:
            return

        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        for n, p in module.named_parameters(recurse=False):
            if p not in ignored_params:
                self._param_list.append(p)
                self._param_names.append(f"{recurse_prefix}{n}")

        for name, child_module in module.named_children():
            self._collect_params(
                child_module,
                ignored_modules,
                ignored_params,
                prefix=f"{recurse_prefix}{name}",
            )

    def register_comm_hook(self) -> None:
        for comm_args, comm_kwargs in self._comm_hook_args:
            self._ddp.register_comm_hook(*comm_args, **comm_kwargs)
        self._comm_hook_args.clear()


    def forward_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if self._init_args or self._init_kwargs:
            self.lazy_init()
        self._ddp.require_backward_grad_sync = not self._no_sync
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._ddp._post_forward(output)