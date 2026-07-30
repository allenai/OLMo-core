"""Torchtitan dataloader for pre-tokenized SFT examples.

Loads a `.pt` file (list of dicts with `tokens` / `labels` int tensors) produced by
`scripts/data/tokenize_contradiction_for_titan.py`, pads each example to the
configured seq_len + 1 (so the shifted (input, label) pair is exactly seq_len),
masks pad-token labels with IGNORE_INDEX, and yields one example per __next__.

No tokenization happens at training time. No sequence packing — each pre-tokenized
example IS one full training sample (designed for 1M-context single-sample SFT).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger


class PretokenizedSftDataset(IterableDataset, Stateful):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        pad_token_id: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = True,
    ) -> None:
        all_data: list[dict[str, torch.Tensor]] = torch.load(
            data_path, weights_only=False
        )
        # Round-robin shard across data-parallel ranks. Stable index ordering.
        self.examples = [
            all_data[i]
            for i in range(len(all_data))
            if i % dp_world_size == dp_rank
        ]
        if not self.examples:
            raise RuntimeError(
                f"No examples for dp_rank={dp_rank} after sharding "
                f"{len(all_data)} examples across dp_world_size={dp_world_size}."
            )
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.infinite = infinite
        self._sample_idx = 0
        self._epoch = 0
        logger.info(
            "PretokenizedSftDataset[dp_rank=%d/%d]: %d local examples (seq_len=%d)",
            dp_rank,
            dp_world_size,
            len(self.examples),
            seq_len,
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        target_len = self.seq_len + 1  # need one extra token for the shift
        while True:
            for i in range(self._sample_idx, len(self.examples)):
                ex = self.examples[i]
                tokens = ex["tokens"].to(torch.long)
                labels = ex["labels"].to(torch.long)
                n = tokens.numel()
                if n < target_len:
                    pad_len = target_len - n
                    tokens = torch.cat(
                        [tokens, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
                    )
                    labels = torch.cat(
                        [labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]
                    )
                elif n > target_len:
                    # Pre-tokenization should size examples to fit; truncating here
                    # would drop the answer at the tail. Fail loudly instead.
                    raise RuntimeError(
                        f"Example {i} has {n} tokens > target_len={target_len}. "
                        f"Increase training.seq_len or shorten the example."
                    )
                input_ids = tokens[:-1].contiguous()
                label = labels[1:].contiguous()
                positions = torch.arange(input_ids.numel(), dtype=torch.long)
                self._sample_idx = i + 1
                yield {"input": input_ids, "positions": positions}, label

            if not self.infinite:
                logger.warning("PretokenizedSftDataset exhausted (epoch=%d)", self._epoch)
                break
            self._sample_idx = 0
            self._epoch += 1
            logger.info("PretokenizedSftDataset starting epoch %d", self._epoch)

    def state_dict(self) -> dict[str, Any]:
        return {"sample_idx": self._sample_idx, "epoch": self._epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._sample_idx = int(state_dict.get("sample_idx", 0))
        self._epoch = int(state_dict.get("epoch", 0))


class PretokenizedSftDataLoader(ParallelAwareDataloader):
    """Configurable wrapper around PretokenizedSftDataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        data_path: str = ""
        """Path to the pre-tokenized .pt file."""

        pad_token_id: int = 151643
        """Token id used to pad sequences shorter than seq_len+1.

        Default is Qwen3's eos_token_id; the pad token's labels are
        force-masked to IGNORE_INDEX so they contribute no loss.
        """

        infinite: bool = True
        """Whether to loop the dataset infinitely (for multi-step training)."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer=None,  # unused; we are pre-tokenized
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ) -> None:
        ds = PretokenizedSftDataset(
            data_path=config.data_path,
            seq_len=seq_len,
            pad_token_id=config.pad_token_id,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }
        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
