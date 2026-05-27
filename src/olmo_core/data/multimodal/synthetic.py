"""
Synthetic multimodal dataset.

A lightweight, dependency-free iterable that yields ``(prompt, response, image)``
triples drawn from a fixed RNG. Intended for unit tests and end-to-end smoke
tests — the prompts and images are random, so models can't learn anything
meaningful, but the pipeline plumbing is fully exercised.
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from ...config import Config

__all__ = [
    "SyntheticMultimodalDatasetConfig",
    "SyntheticMultimodalDataset",
]


@dataclass
class SyntheticMultimodalDatasetConfig(Config):
    """Configuration for :class:`SyntheticMultimodalDataset`."""

    n_examples: int = 64
    """Total number of examples in one epoch."""

    image_size: Tuple[int, int] = (56, 56)
    """``(H, W)`` of the generated images. Variable size is supported via
    :attr:`image_size_jitter`."""

    image_size_jitter: int = 0
    """If > 0, each image's H and W are drawn uniformly from
    ``image_size ± image_size_jitter`` (clamped to positive). Useful for
    stress-testing the multi-crop / variable-resolution code paths."""

    prompt_words: int = 6
    """Number of random words in each prompt."""

    response_words: int = 4
    """Number of random words in each response."""

    seed: int = 0
    """Seed for the RNG that generates examples."""

    text_only_fraction: float = 0.0
    """Fraction of examples that have ``image=None`` instead of a generated image."""


class SyntheticMultimodalDataset:
    """An iterable of ``(prompt, response, image_or_None)`` triples.

    The dataset is deterministic given its :attr:`SyntheticMultimodalDatasetConfig.seed`.
    """

    def __init__(self, cfg: SyntheticMultimodalDatasetConfig):
        self.cfg = cfg
        self._epoch_seed = cfg.seed
        # A tiny in-memory word list — kept short so the resulting prompts /
        # responses tokenize to a predictable number of tokens.
        self._words = (
            "alpha bravo charlie delta echo foxtrot golf hotel india "
            "juliet kilo lima mike november oscar papa quebec romeo "
            "sierra tango uniform victor whiskey xray yankee zulu"
        ).split()

    def set_epoch(self, seed: int) -> None:
        """Re-seed for a new epoch. Called by the data loader."""
        self._epoch_seed = seed

    def _random_text(self, rng: np.random.Generator, n_words: int) -> str:
        return " ".join(rng.choice(self._words, size=n_words).tolist())

    def _random_image(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        h, w = cfg.image_size
        if cfg.image_size_jitter > 0:
            j = cfg.image_size_jitter
            h = max(1, int(rng.integers(h - j, h + j + 1)))
            w = max(1, int(rng.integers(w - j, w + j + 1)))
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def __iter__(self) -> Iterator[Tuple[str, str, Optional[np.ndarray]]]:
        cfg = self.cfg
        rng = np.random.default_rng(self._epoch_seed)
        for _ in range(cfg.n_examples):
            prompt = self._random_text(rng, cfg.prompt_words)
            response = self._random_text(rng, cfg.response_words)
            if rng.random() < cfg.text_only_fraction:
                yield prompt, response, None
            else:
                yield prompt, response, self._random_image(rng)

    def __len__(self) -> int:
        return self.cfg.n_examples
