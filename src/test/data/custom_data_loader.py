import random
from typing import Any, Dict, Iterable, List, Optional

import torch

from olmo_core.aliases import PathOrStr
from olmo_core.data import DataCollator, DataLoaderBase


class CustomDataLoader(DataLoaderBase):
    """
    An example custom data loader that generates random token IDs.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        vocab_size: int,
        work_dir: PathOrStr,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        total_batches: int = 2048,
    ):
        super().__init__(
            collator=DataCollator(pad_token_id=vocab_size - 1),
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        assert self.rank_batch_size % sequence_length == 0
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed
        self._total_batches = total_batches
        self._dataset: Optional[List[torch.Tensor]]

    @property
    def total_batches(self) -> int:
        return self._total_batches

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "seed": self.seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.batches_processed = state_dict["batches_processed"]
        self.seed = state_dict["seed"]
        self._epoch = state_dict["epoch"]

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        del kwargs  # unused

        # Set current epoch.
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch

        # Generate data.
        rng = random.Random(self.seed + self.epoch)
        instances_per_batch = self.global_batch_size // self.sequence_length
        total_instances = instances_per_batch * self.total_batches
        self._dataset = [
            torch.arange(start=start_idx, end=start_idx + self.sequence_length)
            for start_idx in (
                rng.randint(0, self.vocab_size - self.sequence_length - 2)
                for _ in range(total_instances)
            )
        ]

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        assert self._dataset is not None
        instances_per_batch = self.global_batch_size // self.sequence_length
        indices = torch.arange(len(self._dataset)).view(self.total_batches, instances_per_batch)
        for batch_indices in indices:
            local_batch_indices = batch_indices[self.dp_rank :: self.dp_world_size]
            yield self.collator([self._dataset[idx] for idx in local_batch_indices])
