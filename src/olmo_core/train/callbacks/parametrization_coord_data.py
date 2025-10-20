import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from olmo_core.distributed.utils import get_full_tensor, get_rank
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ParametrizationCoordDataCallback(Callback):
    """
    Collects parametrization coordinate check data. Does not require parametrization to be configured/enabled to be used.
    """

    enabled: bool = False

    collection_step: int = 10
    """
    Which step to collect coordinate check data for.
    """

    # _hooks: Optional[List[torch.utils.hooks.RemovableHandle]] = None
    _hooks: Optional[List] = None
    # _coord_data: Optional[List[Tuple[str, tuple, float]]] = None
    _coord_data: Optional[List] = None

    def pre_step(self, batch: Dict[str, Any]):
        if not self.enabled or self.step != self.collection_step:
            return

        train_module = self.trainer.train_module
        if not isinstance(train_module, TransformerTrainModule):
            raise RuntimeError()
        model = train_module.model

        def collect_coord_data(
            debug_state: List[Tuple[str, tuple, float]],
            name: str,
            _: torch.nn.Module,
            args,
            local_output,
        ):
            if isinstance(local_output, torch.Tensor):
                state_name = f"{name}|output"
                output = get_full_tensor(local_output.detach().float())
                debug_state.append(
                    (state_name, output.shape, output.norm(p=1.0).item() / output.numel())
                )

        coord_data: List[Tuple[str, tuple, float]] = []
        hooks = []
        for name, module in model.named_modules():
            hook = module.register_forward_hook(partial(collect_coord_data, coord_data, name))
            hooks.append(hook)

        self._coord_data = coord_data
        self._hooks = hooks

    def post_step(self):
        if self.step != self.collection_step:
            return

        assert self._hooks is not None
        for hook in self._hooks:
            hook.remove()

        assert self._coord_data is not None
        if get_rank() == 0:
            self.trainer.write_file(
                f"coord_data_step{self.step}.json", json.dumps(self._coord_data)
            )

        self._hooks = None
        self._coord_data = None
