"""
Test-only synthetic multimodal data source.

A lightweight, dependency-free iterable that yields ``(prompt, response, image)``
triples drawn from a fixed RNG. Used by the data-pipeline tests (collator /
loader / preprocessor / train module) to exercise the plumbing without network
or real data. This deliberately lives in the test tree, not in ``olmo_core`` —
the only shipped training data source is
:class:`~olmo_core.data.multimodal.pixmo_cap.PixmoCapDataset`.
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

__all__ = [
    "SyntheticMultimodalDatasetConfig",
    "SyntheticMultimodalDataset",
]


@dataclass
class SyntheticMultimodalDatasetConfig:
    """Configuration for :class:`SyntheticMultimodalDataset`."""

    n_examples: int = 64
    image_size: Tuple[int, int] = (56, 56)
    image_size_jitter: int = 0
    prompt_words: int = 6
    response_words: int = 4
    seed: int = 0
    text_only_fraction: float = 0.0


class SyntheticMultimodalDataset:
    """An iterable of ``(prompt, response, image_or_None)`` triples, deterministic
    given :attr:`SyntheticMultimodalDatasetConfig.seed`."""

    def __init__(self, cfg: SyntheticMultimodalDatasetConfig):
        self.cfg = cfg
        self._epoch_seed = cfg.seed
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
