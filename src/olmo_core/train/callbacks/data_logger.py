from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from olmo_core.distributed.utils import get_rank

from .callback import Callback


@dataclass
class DataLoggerCallback(Callback):
    folder: str = "data"

    @property
    def path(self) -> Path:
        return self.trainer.work_dir / self.folder / f"rank{get_rank()}.tsv"

    def pre_train(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def pre_step(self, batch: Dict[str, Any]):
        input_ids_hash = hash_long_tensor(batch["input_ids"]).item()
        with self.path.open(mode="a") as f:
            f.write(f"{self.step}\t{input_ids_hash}")


_HASH_MULTIPLIER = 6364136223846793005
_HASH_INCREMENT = 1


def hash_long_tensor(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int64
    while x.ndim > 0:
        x = _reduce_last_axis(x)
    return x


@torch.no_grad()
def _reduce_last_axis(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= _HASH_MULTIPLIER
        acc += _HASH_INCREMENT
        acc += x[..., i]
    return acc
