"""
Multimodal data loader.

Wraps a source iterable of ``(prompt, response, image)`` triples (e.g.
:class:`SyntheticMultimodalDataset` or :class:`PixmoCapDataset`) into a
:class:`~olmo_core.data.data_loader.DataLoaderBase` that the
:class:`~olmo_core.train.Trainer` can consume directly.

The pipeline at iteration time is:

    source  →  MultimodalPreprocessor (per example)  →  collator (per batch)

Rank sharding and PyTorch ``DataLoader`` worker sharding are handled here
so each rank/worker sees a disjoint slice of the source.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import torch
import torch.utils.data

from ...config import Config
from ..data_loader import DataLoaderBase
from .collator import MultimodalCollator, MultimodalCollatorConfig
from .preprocessor import MultimodalPreprocessor, MultimodalPreprocessorConfig

__all__ = [
    "MultimodalDataLoaderConfig",
    "MultimodalDataLoader",
]


class _MultimodalIterableWrapper(torch.utils.data.IterableDataset):
    """A :class:`torch.utils.data.IterableDataset` that runs the preprocessor
    inline and shards the source across (rank × worker).

    The source is iterated **once per epoch**; this wrapper handles the
    rank/worker stride. The source's ``set_epoch(epoch)`` method (if present)
    is called when :meth:`set_epoch` is called externally.
    """

    def __init__(
        self,
        source,
        preprocessor: MultimodalPreprocessor,
        dp_rank: int,
        dp_world_size: int,
    ):
        super().__init__()
        self.source = source
        self.preprocessor = preprocessor
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        # If the source supports epoch-based shuffling, propagate.
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Global shard ID across DP ranks AND DataLoader workers.
        shard_id = self.dp_rank * num_workers + worker_id
        total_shards = self.dp_world_size * num_workers

        for i, ex in enumerate(self.source):
            if i % total_shards != shard_id:
                continue
            prompt, response, image = ex
            yield self.preprocessor(prompt, response, image)


@dataclass
class MultimodalDataLoaderConfig(Config):
    """Configuration for :class:`MultimodalDataLoader`."""

    preprocessor: MultimodalPreprocessorConfig = field(default_factory=MultimodalPreprocessorConfig)
    """Preprocessor settings (tokenizer + multicrop + sequence length)."""

    collator: Optional[MultimodalCollatorConfig] = None
    """Collator settings. If ``None``, a default collator is built from
    ``preprocessor.tokenizer``."""

    global_batch_size: int = 8
    """Total examples per batch across all DP ranks."""

    num_workers: int = 0
    """Number of background workers per rank for the wrapped torch DataLoader.
    Set to 0 for in-process iteration (simplest, fine for tests). Multi-worker
    requires the source dataset and preprocessor to be picklable."""

    prefetch_factor: int = 2
    """Number of batches each worker prefetches. Ignored when num_workers=0."""

    seed: int = 0
    """Base seed; the epoch number is added to it for per-epoch shuffling."""

    work_dir: str = "/tmp/olmo-core-mm-data"
    """Local working directory; required by :class:`DataLoaderBase`."""

    def build(
        self,
        source,
        hf_tokenizer,
        *,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
    ) -> "MultimodalDataLoader":
        """Instantiate a :class:`MultimodalDataLoader`.

        :param source: An iterable of ``(prompt, response, image)`` triples.
            Should be re-iterable across epochs and ideally have a
            ``set_epoch(epoch)`` method for per-epoch shuffling.
        :param hf_tokenizer: A HuggingFace tokenizer with the image special
            tokens added (see :meth:`MultimodalTokenizerConfig.load_hf_tokenizer`).
        :param dp_world_size: Data-parallel world size.
        :param dp_rank: This process's data-parallel rank.
        :param fs_local_rank: Filesystem-local rank (defaults to the global one).
        """
        preprocessor = MultimodalPreprocessor(self.preprocessor, hf_tokenizer)
        coll_cfg = self.collator or MultimodalCollatorConfig(tokenizer=self.preprocessor.tokenizer)
        collator = MultimodalCollator(coll_cfg)
        return MultimodalDataLoader(
            source=source,
            preprocessor=preprocessor,
            collator=collator,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            seed=self.seed,
            work_dir=self.work_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )


class MultimodalDataLoader(DataLoaderBase):
    """Per-example multimodal data loader.

    ``global_batch_size`` is in **examples** (not tokens), so each rank yields
    batches of size ``global_batch_size // dp_world_size``. Each batch is a
    dict of tensors with the keys
    :class:`~olmo_core.nn.vision.MultimodalTransformer.forward` expects.
    """

    def __init__(
        self,
        *,
        source,
        preprocessor: MultimodalPreprocessor,
        collator: MultimodalCollator,
        global_batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        seed: int = 0,
        work_dir: str = "/tmp/olmo-core-mm-data",
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        if global_batch_size % dp_world_size != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"dp_world_size ({dp_world_size})"
            )
        self.source = source
        self.preprocessor = preprocessor
        self.collator = collator
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed

    # ------------------------------------------------------------------
    # DataLoaderBase API
    # ------------------------------------------------------------------

    @property
    def total_batches(self) -> Optional[int]:
        """If the source has a known length, return the number of batches in an epoch."""
        n = getattr(self.source, "__len__", None)
        if n is None:
            return None
        try:
            n_examples = len(self.source)
        except TypeError:
            return None
        # Each rank gets a fraction of the total.
        return n_examples // self.global_batch_size

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._epoch = state_dict["epoch"]
        self.batches_processed = state_dict["batches_processed"]
        self.seed = state_dict["seed"]

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch
        # Propagate per-epoch shuffle seed to the source if supported.
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(self.seed + epoch)

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        # Skip building the iterator if the epoch is already exhausted.
        if self.total_batches is not None and self.batches_processed >= self.total_batches:
            return

        wrapper = _MultimodalIterableWrapper(
            self.source, self.preprocessor, self.dp_rank, self.dp_world_size
        )
        wrapper.set_epoch(self._epoch or 0)

        torch_loader = torch.utils.data.DataLoader(
            wrapper,
            batch_size=self.rank_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=False,
            collate_fn=self._collate,
        )

        # Resume from where state_dict left off in this epoch.
        skip = self.batches_processed
        for i, batch in enumerate(torch_loader):
            if i < skip:
                continue
            # Drop the trailing incomplete batch (the collator received < rank_batch_size).
            if batch["input_ids"].shape[0] < self.rank_batch_size:
                break
            yield batch

    def _collate(self, examples) -> Dict[str, torch.Tensor]:
        return self.collator(examples)

    def get_mock_batch(self) -> Dict[str, torch.Tensor]:
        """Return a small fake batch for the trainer's dry-run pass.

        Uses the configured patch / pool dimensions from the preprocessor so
        the shapes match what the model expects.
        """
        cfg = self.preprocessor.cfg
        mc_cfg = cfg.multicrop
        patch_dim = 3 * mc_cfg.image_preprocessor.patch_size**2
        n_patches_per_crop = (
            mc_cfg.base_image_input_size[0] // mc_cfg.image_preprocessor.patch_size
        ) * (mc_cfg.base_image_input_size[1] // mc_cfg.image_preprocessor.patch_size)
        pool_size = mc_cfg.pool_h * mc_cfg.pool_w
        n_pooled = n_patches_per_crop // pool_size

        B = self.rank_batch_size
        # Build a sequence of <im_patch> tokens followed by a few text tokens.
        patch_id = cfg.tokenizer.image_patch_id
        pad_id = cfg.tokenizer.base.pad_token_id
        S = n_pooled + 4
        input_ids = torch.full((B, S), pad_id, dtype=torch.long)
        input_ids[:, :n_pooled] = patch_id
        return {
            "input_ids": input_ids,
            "loss_masks": torch.ones(B, S, dtype=torch.float32),
            "images": torch.zeros(B, 1, n_patches_per_crop, patch_dim, dtype=torch.float32),
            "image_masks": torch.ones(B, 1, n_patches_per_crop, dtype=torch.float32),
            "pooled_patches_idx": (
                torch.arange(n_patches_per_crop)
                .view(n_pooled, pool_size)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .contiguous()
            ),
        }
