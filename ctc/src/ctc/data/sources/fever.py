"""
FEVER -> contradiction pairs, the held-out corpus for ``contra_fever``.

Every REFUTES row is a ``(claim, evidence sentence)`` tuple where the claim is asserted false and
the cited Wikipedia sentence is the truth refuting it. The two cannot both be true, so the pair
contradicts *by construction* -- no NLI re-scoring, no LLM, no proxy heuristic. That is why FEVER
is the OOD probe for a task whose in-distribution gold needs a model to write.

The canonical ``fever/fever`` dataset is script-based and no longer loadable on current
``datasets``; ``copenlu/fever_gold_evidence`` is the parquet mirror, and the one every shipped
``contradiction_eval_fever_*`` ladder was built from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = ["FeverPool", "clean_wiki_text", "build_pool", "load_pool"]


def clean_wiki_text(text: str) -> str:
    """
    Undo FEVER's PTB tokenisation so evidence reads like the claims beside it.

    Left as-is, an evidence sentence is spelled ``-LRB- 1991 -RRB-`` while the claim beside it is
    ordinary prose -- a surface tell that separates gold from filler without reading either.

    :param text: Raw evidence sentence.

    :returns: Detokenised prose.
    """
    if not text:
        return text
    for old, new in (
        ("-LRB-", "("),
        ("-RRB-", ")"),
        ("-LSB-", "["),
        ("-RSB-", "]"),
        ("-LCB-", "{"),
        ("-RCB-", "}"),
        ("``", '"'),
        ("''", '"'),
    ):
        text = text.replace(old, new)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.replace(" 's", "'s").replace(" n't", "n't")
    return re.sub(r"\s+", " ", text).strip()


def _first_evidence(evidence) -> str:
    for triple in evidence or ():
        if triple and len(triple) >= 3:
            sentence = clean_wiki_text(triple[2])
            if sentence:
                return sentence
    return ""


@dataclass(frozen=True)
class FeverPool:
    """
    :param pairs: ``(false claim, refuting sentence, wikipedia page)``, in a fixed shuffled order.
    :param nei_by_page: Page -> NOT-ENOUGH-INFO claims about it. Used as topic-matched distractors:
        an NEI claim is unverifiable against Wikipedia by definition, so it is far less likely than
        a SUPPORTS sentence to *itself* state the truth a gold REFUTES claim denies.
    :param support_pairs_by_page: Page -> ``(claim, evidence)`` SUPPORTS tuples. Safe only on
        non-gold pages, where they act as "fake pairs": syntactically pair-shaped and semantically
        consistent.
    :param fillers: Flat pool of NEI claims and SUPPORTS sentences, for the leftover slots.
    :param pages: Pages with at least one NEI or SUPPORTS row, for decoy sampling.
    """

    pairs: Tuple[Tuple[str, str, str], ...]
    nei_by_page: Dict[str, Tuple[str, ...]]
    support_pairs_by_page: Dict[str, Tuple[Tuple[str, str], ...]]
    fillers: Tuple[str, ...]
    pages: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.pairs)

    def for_split(self, split: str) -> "FeverPool":
        """
        Slice off this split's gold pairs; the distractor pools are shared.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's pairs.

        :raises ValueError: On an unknown split name.
        """
        cut = max(1, len(self.pairs) // 10)
        if split == "eval":
            kept = self.pairs[:cut]
        elif split == "train":
            kept = self.pairs[cut:]
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return FeverPool(
            kept, self.nei_by_page, self.support_pairs_by_page, self.fillers, self.pages
        )


def build_pool(rows, seed: int = 42) -> FeverPool:
    """
    Walk FEVER rows and assemble everything example construction needs.

    :param rows: Iterable of FEVER rows with ``label``, ``claim`` and ``evidence``.
    :param seed: Shuffles the gold-pair order, fixing which pair example *i* uses.

    :returns: The pool.
    """
    import random

    pairs: List[Tuple[str, str, str]] = []
    seen_pairs = set()
    nei: Dict[str, List[str]] = {}
    supports: Dict[str, List[Tuple[str, str]]] = {}
    nei_texts, support_texts = set(), set()
    seen_nei, seen_support = set(), set()

    for row in rows:
        claim = (row.get("claim") or "").strip()
        if not claim:
            continue
        evidence = row.get("evidence") or []
        page = evidence[0][0] if (evidence and evidence[0] and evidence[0][0]) else None
        label = row.get("label")

        if label == "NOT ENOUGH INFO":
            nei_texts.add(claim)
            if page and claim not in seen_nei:
                nei.setdefault(page, []).append(claim)
                seen_nei.add(claim)
            continue

        sentence = _first_evidence(evidence)
        if not sentence or sentence == claim:
            continue
        if label == "SUPPORTS":
            support_texts.add(sentence)
            if page and (claim, sentence) not in seen_support:
                supports.setdefault(page, []).append((claim, sentence))
                seen_support.add((claim, sentence))
        elif label == "REFUTES" and page and (claim, sentence) not in seen_pairs:
            seen_pairs.add((claim, sentence))
            pairs.append((claim, sentence, page))

    random.Random(seed).shuffle(pairs)
    return FeverPool(
        pairs=tuple(pairs),
        nei_by_page={k: tuple(v) for k, v in nei.items()},
        support_pairs_by_page={k: tuple(v) for k, v in supports.items()},
        fillers=tuple(nei_texts) + tuple(support_texts),
        pages=tuple(sorted(set(nei) | set(supports))),
    )


def load_pool(
    *, split: str = "train", seed: int = 42, dataset: str = "copenlu/fever_gold_evidence"
):
    """
    Read FEVER off the Hub and build the pool.

    :param split: ``"train"`` (~228k rows) or ``"validation"`` (~16k). The validation pools are
        thin enough that a long rung would reuse fillers heavily, so ``"train"`` is the default and
        the train/eval separation is done on gold pairs by :meth:`FeverPool.for_split`.
    :param seed: Gold-pair order.
    :param dataset: HF id. The parquet mirror, because the canonical FEVER loader is script-based
        and no longer runs.

    :returns: The pool.

    :raises ImportError: If ``datasets`` is missing; install ``ctc[sources]``.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("FEVER loading needs `datasets`: pip install './ctc[sources]'") from e

    return build_pool(load_dataset(dataset, split=split), seed=seed)
