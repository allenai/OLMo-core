"""
PixMo-Cap dataset adapter (local-disk version).

Loads the pre-downloaded ``allenai/pixmo-cap`` HuggingFace ``Dataset`` saved
under ``$MOLMO_DATA_DIR/torch_datasets/pixmo_datasets/cap`` (the layout Molmo2
itself uses). Each example already carries a local absolute path in the
``image`` column, so no network is required at iteration time.

This adapter yields ``(prompt, response, image)`` triples compatible with
:class:`MultimodalPreprocessor`. PixMo-Cap is the caption-pretraining set used
in Molmo Stage 1, so this is the natural first data source for a from-scratch
VLM run.
"""

import os
from dataclasses import dataclass
from os.path import join
from typing import Iterator, Optional, Tuple

from ...config import Config

__all__ = [
    "PixmoCapDatasetConfig",
    "PixmoCapDataset",
    "DEFAULT_MOLMO_DATA_DIR",
]

#: Default location of the shared Molmo data tree on AI2 infrastructure.
DEFAULT_MOLMO_DATA_DIR: str = "/weka/oe-training-default/mm-olmo"


@dataclass
class PixmoCapDatasetConfig(Config):
    """Configuration for :class:`PixmoCapDataset`."""

    data_dir: Optional[str] = None
    """Path to the saved HuggingFace ``Dataset`` directory. When ``None``,
    defaults to ``${MOLMO_DATA_DIR or DEFAULT_MOLMO_DATA_DIR}/torch_datasets/pixmo_datasets/cap``.
    """

    split: str = "train"
    """HuggingFace split name (``train`` or ``validation``)."""

    prompt: str = "Describe this image in detail."
    """Constant prompt prefix paired with each caption."""

    limit: Optional[int] = None
    """If set, stop after this many examples (handy for tests/dry runs)."""

    shuffle: bool = False
    """If ``True``, shuffle the dataset before iteration. Uses HF's
    ``Dataset.shuffle``."""

    shuffle_seed: int = 0
    """Seed for the shuffle. Ignored when :attr:`shuffle` is ``False``."""

    def resolve_data_dir(self) -> str:
        """Return the configured data dir, falling back to the env-var-driven default."""
        if self.data_dir is not None:
            return self.data_dir
        root = os.environ.get("MOLMO_DATA_DIR", DEFAULT_MOLMO_DATA_DIR)
        return join(root, "torch_datasets", "pixmo_datasets", "cap")

    def build(self) -> "PixmoCapDataset":
        return PixmoCapDataset(self)


class PixmoCapDataset:
    """Iterable over PixMo-Cap loaded from a local on-disk HF ``Dataset``."""

    def __init__(self, cfg: PixmoCapDatasetConfig):
        self.cfg = cfg

    def __iter__(self) -> Iterator[Tuple[str, str, object]]:
        from datasets import load_from_disk
        from PIL import Image

        cfg = self.cfg
        path = cfg.resolve_data_dir()
        ds = load_from_disk(path)[cfg.split]
        if cfg.shuffle:
            ds = ds.shuffle(seed=cfg.shuffle_seed)

        for i, row in enumerate(ds):
            if cfg.limit is not None and i >= cfg.limit:
                return
            image_path = row["image"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:  # noqa: BLE001
                # Skip rows whose local file is missing/corrupt; don't abort iteration.
                continue
            yield cfg.prompt, row["caption"], image
