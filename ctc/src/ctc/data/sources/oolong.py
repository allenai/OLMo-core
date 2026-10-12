"""
OOLONG synthetic classification logs -> a pool of labelled items with per-item token counts.

The shipped ``oolong-synth`` benchmark provides ~100 pre-built context windows per fixed length
bucket. That is far too few, and far too quantised, to train or grade a continuous-length ladder,
so the windows are taken apart and rebuilt: every labelled item line is pooled, deduplicated and
tokenized once, and an example is assembled by drawing items until a **token budget** is met.

That is why ``oolong`` is the one ladder whose scaling parameter is a token count rather than a
document count -- its items are lines inside one document, not documents.

.. note::
   **The gold is recomputed over whichever items were drawn**, so two rungs cannot grade the same
   question: a different item set has a different most-frequent label. ``oolong`` is therefore the
   one ladder built independently per rung, and its rung-to-rung deltas carry eval-set resampling
   noise that the other four do not. Declared on the generator (``shrink_safe=False``) rather than
   discovered by an audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

__all__ = ["Item", "OolongPool", "parse_item_line", "load_pool"]

_ITEM_RE = re.compile(r"^Date: (.*?) \|\| User: (.*?) \|\| Instance: ")


@dataclass(frozen=True)
class Item:
    """
    One labelled log line.

    :param line: The line **without** its ``|| Label: L`` suffix -- what the context shows.
    :param user: User id.
    :param label: Class label. The answer side; never rendered.
    :param date: Date string as written.
    :param month: Month number, for the timeline variants.
    :param tokens: Token count of ``line``, measured once at load.
    """

    line: str
    user: str
    label: str
    date: str
    month: int
    tokens: int


@dataclass(frozen=True)
class OolongPool:
    """
    :param items: Deduplicated items per sub-dataset (spam, trec_coarse, ...).
    :param labels: Canonical label order per sub-dataset. Taken from a real OOLONG question so the
        rendered option list matches the benchmark's, not our sort order.
    :param preamble: Per-sub-dataset preamble template with the item count as ``{N}``.
    :param preamble_tokens: Token count of each preamble.
    """

    items: Dict[str, Tuple[Item, ...]]
    labels: Dict[str, Tuple[str, ...]]
    preamble: Dict[str, str]
    preamble_tokens: Dict[str, int]

    def __len__(self) -> int:
        return sum(len(v) for v in self.items.values())

    @property
    def datasets(self) -> List[str]:
        """:returns: Sub-dataset names, sorted, so a build does not depend on dict order."""
        return sorted(self.items)

    def for_split(self, split: str) -> "OolongPool":
        """
        Split each sub-dataset's items between train and eval.

        Item-level, because an OOLONG example *is* a set of items: two examples sharing items share
        the substrate the aggregate is computed over.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool over that split's items.

        :raises ValueError: On an unknown split name.
        """
        out = {}
        for name, items in self.items.items():
            cut = max(1, len(items) // 10)
            if split == "eval":
                out[name] = items[:cut]
            elif split == "train":
                out[name] = items[cut:]
            else:
                raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return OolongPool(out, self.labels, self.preamble, self.preamble_tokens)


def parse_item_line(line: str) -> Optional[Tuple[str, Dict[str, object]]]:
    """
    Split ``Date: D || User: U || Instance: T || Label: L`` into its unlabelled half and metadata.

    :param line: One raw item line.

    :returns: ``(unlabelled_line, metadata)``, or ``None`` when the line is not an item.
    """
    if not line.startswith("Date:"):
        return None
    head, separator, label = line.rpartition(" || Label: ")
    if not separator:
        return None
    match = _ITEM_RE.match(head)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%b %d, %Y")
    except ValueError:
        return None
    return head, {
        "user": match.group(2),
        "label": label,
        "date": match.group(1),
        "month": parsed.month,
    }


def load_pool(
    *,
    hf_name: str = "oolongbench/oolong-synth",
    max_context: int = 131_072,
    tokenizer: str = "Qwen/Qwen3-0.6B",
    seed: int = 42,
) -> OolongPool:
    """
    Pool every labelled item out of ``oolong-synth`` and measure it.

    :param hf_name: HF dataset id.
    :param max_context: Only pool items from windows at or below this context length; the very
        long buckets add nothing but load time.
    :param tokenizer: Tokenizer whose counts the token budget is expressed in. The budget is only
        meaningful relative to one tokenizer, and this is the one every shipped oolong build used.
    :param seed: Shuffles each sub-dataset's item order, fixing which items example *i* may draw.

    :returns: The pool.

    :raises ImportError: If ``datasets``/``transformers`` are missing; install ``ctc[sources,gen]``.
    """
    import random

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "OOLONG loading needs `datasets` + `transformers`: " "pip install './ctc[sources,gen]'"
        ) from e

    dataset = load_dataset(hf_name)
    raw: Dict[str, List[Tuple[str, Dict]]] = {}
    seen: Dict[str, set] = {}
    preamble: Dict[str, str] = {}
    labels: Dict[str, Tuple[str, ...]] = {}

    for split in dataset:
        for row in dataset[split]:
            name = row["dataset"]
            if max_context and (row.get("context_len") or 0) > max_context:
                continue
            window = row["context_window_text_with_labels"]
            start = window.find("\nDate:")
            if start < 0:
                continue
            lines = [ln for ln in window[start + 1 :].split("\n") if ln.startswith("Date:")]
            if name not in preamble:
                # Template the item count out of the preamble so it can be restated for any N.
                preamble[name] = re.sub(rf"\b{len(lines)}\b", "{N}", window[:start].rstrip())
            if name not in labels and "ANSWER_TYPE.LABEL" in (row.get("answer_type") or ""):
                question = row["question"]
                match = re.search(r"one of the labels:\s*(.*?)\.\s*$", question)
                if match and "only consider" not in question:
                    labels[name] = tuple(s.strip() for s in match.group(1).split(","))
            for line in lines:
                parsed = parse_item_line(line)
                if parsed is None:
                    continue
                unlabelled, meta = parsed
                if unlabelled in seen.setdefault(name, set()):
                    continue
                seen[name].add(unlabelled)
                raw.setdefault(name, []).append((unlabelled, meta))

    encoder = AutoTokenizer.from_pretrained(tokenizer)
    items: Dict[str, Tuple[Item, ...]] = {}
    preamble_tokens: Dict[str, int] = {}
    for name, rows in raw.items():
        counts: List[int] = []
        lines = [line for line, _ in rows]
        for start in range(0, len(lines), 1024):
            encoded = encoder(lines[start : start + 1024], add_special_tokens=False)
            counts.extend(len(x) for x in encoded["input_ids"])
        built = [
            Item(line, m["user"], m["label"], m["date"], m["month"], n)
            for (line, m), n in zip(rows, counts)
        ]
        random.Random(seed).shuffle(built)
        items[name] = tuple(built)
        preamble_tokens[name] = len(
            encoder(preamble[name].format(N=999), add_special_tokens=False)["input_ids"]
        )
        labels.setdefault(name, tuple(sorted({it.label for it in built})))

    return OolongPool(items, labels, preamble, preamble_tokens)
