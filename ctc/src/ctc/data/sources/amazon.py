"""
Amazon product reviews -- the held-out substrate for ``outlier_review``.

Same task contract as the Wikipedia outlier ladder, different corpus: the majority of reviews come
from one product category (or share one star rating) and a few do not. Ground truth is structured
metadata only -- the HF category and the ``rating`` field -- so there is no labelling model and no
label noise.

**Only the review title and body are ever rendered.** Rating and category are the answer; putting
either in the context, including as a document ``title``, hands the task over. This is the same
family as the docchunk wrapping leak, where a document label leaked into the free-token span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Review", "ReviewPool", "DEFAULT_CATEGORIES", "load_pool"]

#: Star ratings that count. A malformed or missing rating drops the review rather than defaulting.
VALID_RATINGS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: The categories the shipped ``outlier_review_matched`` ladder drew from.
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "Books",
    "Electronics",
    "Beauty_and_Personal_Care",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Musical_Instruments",
    "Office_Products",
)


@dataclass(frozen=True)
class Review:
    """
    :param title: Review headline -- rendered, and also what the ``answers`` metadata names.
    :param text: Review body.
    :param rating: Star rating, 1-5. Metadata: never rendered.
    :param category: Product category. Metadata: never rendered.
    """

    title: str
    text: str
    rating: int
    category: str


@dataclass(frozen=True)
class ReviewPool:
    """
    :param by_category: Category -> reviews, each pool already shuffled.
    """

    by_category: Dict[str, Tuple[Review, ...]]

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_category.values())

    @property
    def categories(self) -> List[str]:
        """:returns: Category names, sorted, so a build does not depend on dict insertion order."""
        return sorted(self.by_category)

    def for_split(self, split: str) -> "ReviewPool":
        """
        Split each category's reviews between train and eval.

        Per category rather than by category, so both splits still see every category -- splitting
        by category would make the held-out ladder a different *task*, not a different sample.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool over that split's reviews.

        :raises ValueError: On an unknown split name.
        """
        out = {}
        for category, reviews in self.by_category.items():
            cut = max(1, len(reviews) // 10)
            if split == "eval":
                out[category] = reviews[:cut]
            elif split == "train":
                out[category] = reviews[cut:]
            else:
                raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return ReviewPool(out)


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_rating(value) -> Optional[int]:
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return rating if rating in VALID_RATINGS else None


def load_pool(
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    pool_size: int = 20_000,
    min_chars: int = 120,
    max_chars: int = 1500,
    seed: int = 42,
    hf_dataset: str = "McAuley-Lab/Amazon-Reviews-2023",
) -> ReviewPool:
    """
    Stream each category and keep up to ``pool_size`` usable reviews.

    :param categories: HF category suffixes. At least two, or a category-outlier example cannot be
        built at all.
    :param pool_size: Reviews per category.
    :param min_chars: Reject reviews shorter than this -- a two-word review carries no topic, so it
        is neither a usable majority document nor a findable outlier.
    :param max_chars: Reject reviews longer than this, so one document cannot dominate a rung's
        token budget.
    :param seed: Shuffles each category's pool.
    :param hf_dataset: HF dataset id.

    :returns: The pool.

    :raises ImportError: If ``datasets`` is missing; install ``ctc[sources]``.
    """
    import random

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("Amazon loading needs `datasets`: pip install './ctc[sources]'") from e

    by_category: Dict[str, Tuple[Review, ...]] = {}
    for offset, category in enumerate(categories):
        # The dataset's loading script is unsupported on current `datasets`; read the raw shards.
        path = f"hf://datasets/{hf_dataset}/raw/review_categories/{category}.jsonl"
        stream = load_dataset("json", data_files=path, split="train", streaming=True)
        kept: List[Review] = []
        for seen, row in enumerate(stream):
            text = _clean(row.get("text"))
            if not min_chars <= len(text) <= max_chars:
                continue
            rating = _normalize_rating(row.get("rating"))
            if rating is None:
                continue
            kept.append(Review(_clean(row.get("title")), text, rating, category))
            if len(kept) >= pool_size:
                break
            if seen > 500_000 and len(kept) < pool_size // 2:
                break  # sparse category; take what there is rather than streaming forever
        random.Random(seed + offset).shuffle(kept)
        by_category[category] = tuple(kept)
    return ReviewPool(by_category)
