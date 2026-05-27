"""
PixMo-Cap dataset adapter.

PixMo-Cap (``allenai/pixmo-cap``) is the caption-pretraining set used in Molmo
Stage 1, so it's the natural first data source for a from-scratch VLM run. It
yields ``(prompt, response, image)`` triples compatible with
:class:`~olmo_core.data.multimodal.preprocessor.MultimodalPreprocessor`.

Two loading modes, selected by :attr:`PixmoCapDatasetConfig.source`:

- ``"hub"`` (default): stream ``allenai/pixmo-cap`` directly from the
  HuggingFace Hub and download each image from its ``image_url`` at iteration
  time. No local copy required, but needs network access; rows whose URL is
  dead or unreachable are silently skipped. The dataset on the Hub stores only
  image URLs (columns ``image_url`` / ``caption`` / ``transcripts``), not image
  bytes, which is why a download step is required.

- ``"local"``: read a pre-downloaded HF ``Dataset`` saved under
  ``$MOLMO_DATA_DIR/torch_datasets/pixmo_datasets/cap`` (the layout Molmo2
  itself uses), where the ``image`` column already holds local file paths. This
  is the path used for real large-scale training on AI2 infrastructure — no
  network at iteration time.
"""

import io
import logging
import os
from dataclasses import dataclass
from os.path import join
from typing import Iterator, Optional, Tuple

from ...config import Config

__all__ = [
    "PixmoCapDatasetConfig",
    "PixmoCapDataset",
    "DEFAULT_MOLMO_DATA_DIR",
    "HF_DATASET_ID",
]

log = logging.getLogger(__name__)

#: Default location of the shared Molmo data tree on AI2 infrastructure.
DEFAULT_MOLMO_DATA_DIR: str = "/weka/oe-training-default/mm-olmo"

#: HuggingFace Hub dataset id for the ``"hub"`` source.
HF_DATASET_ID: str = "allenai/pixmo-cap"


@dataclass
class PixmoCapDatasetConfig(Config):
    """Configuration for :class:`PixmoCapDataset`."""

    source: str = "hub"
    """Where to load from: ``"hub"`` (stream from the HuggingFace Hub and
    download images per-URL) or ``"local"`` (read a pre-downloaded on-disk HF
    ``Dataset``)."""

    split: str = "train"
    """HuggingFace split name. The Hub dataset only has ``train``."""

    prompt: str = "Describe this image in detail."
    """Constant prompt prefix paired with each caption."""

    limit: Optional[int] = None
    """If set, stop after this many successfully-loaded examples (handy for
    tests / dry runs). Counts examples *yielded*, not rows scanned, so in
    ``"hub"`` mode skipped (dead-URL) rows don't count against it."""

    shuffle: bool = False
    """If ``True``, shuffle before iteration."""

    shuffle_seed: int = 0
    """Seed for the shuffle. Ignored when :attr:`shuffle` is ``False``."""

    # ---- hub-mode options ----
    hub_dataset_id: str = HF_DATASET_ID
    """(``"hub"`` only) HuggingFace Hub dataset id."""

    image_timeout: float = 10.0
    """(``"hub"`` only) Per-image HTTP download timeout, in seconds."""

    shuffle_buffer_size: int = 10_000
    """(``"hub"`` only) Buffer size for streaming shuffle."""

    # ---- local-mode options ----
    data_dir: Optional[str] = None
    """(``"local"`` only) Path to the saved HF ``Dataset`` directory. When
    ``None``, defaults to
    ``${MOLMO_DATA_DIR or DEFAULT_MOLMO_DATA_DIR}/torch_datasets/pixmo_datasets/cap``.
    """

    def resolve_data_dir(self) -> str:
        """(``"local"`` only) Return the data dir, falling back to the
        env-var-driven default."""
        if self.data_dir is not None:
            return self.data_dir
        root = os.environ.get("MOLMO_DATA_DIR", DEFAULT_MOLMO_DATA_DIR)
        return join(root, "torch_datasets", "pixmo_datasets", "cap")

    def build(self) -> "PixmoCapDataset":
        return PixmoCapDataset(self)


class PixmoCapDataset:
    """Iterable over PixMo-Cap ``(prompt, response, image)`` triples.

    See the module docstring for the two loading modes.
    """

    def __init__(self, cfg: PixmoCapDatasetConfig):
        if cfg.source not in ("hub", "local"):
            raise ValueError(f"source must be 'hub' or 'local', got {cfg.source!r}")
        self.cfg = cfg

    def __iter__(self) -> Iterator[Tuple[str, str, object]]:
        if self.cfg.source == "hub":
            return self._iter_hub()
        return self._iter_local()

    def _iter_hub(self) -> Iterator[Tuple[str, str, object]]:
        import requests
        from datasets import load_dataset
        from PIL import Image

        cfg = self.cfg
        ds = load_dataset(cfg.hub_dataset_id, split=cfg.split, streaming=True)
        if cfg.shuffle:
            ds = ds.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.shuffle_buffer_size)

        n = 0
        for row in ds:
            if cfg.limit is not None and n >= cfg.limit:
                return
            url = row["image_url"]
            try:
                resp = requests.get(url, timeout=cfg.image_timeout)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:  # noqa: BLE001
                # Dead / blocked / corrupt URL — skip without aborting iteration.
                continue
            n += 1
            yield cfg.prompt, row["caption"], image

    def _iter_local(self) -> Iterator[Tuple[str, str, object]]:
        from datasets import load_from_disk
        from PIL import Image

        cfg = self.cfg
        path = cfg.resolve_data_dir()
        ds = load_from_disk(path)[cfg.split]
        if cfg.shuffle:
            ds = ds.shuffle(seed=cfg.shuffle_seed)

        n = 0
        for row in ds:
            if cfg.limit is not None and n >= cfg.limit:
                return
            image_path = row["image"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:  # noqa: BLE001
                # Skip rows whose local file is missing / corrupt; don't abort.
                continue
            n += 1
            yield cfg.prompt, row["caption"], image
