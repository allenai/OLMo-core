"""
Read our olmo-core SFT shards (``token_ids_part_*.npy`` + ``labels_mask_*.npy``) as a plain
map-style torch dataset, for training HiLS / Olmo-3 under **veomni**.

Why this exists rather than veomni's own chat dataset: our shards are produced by
``src/scripts/data/convert_unified_to_sft.py``, whose prompts come from
``olmo_core.data.corpus_reasoning_prompts.build_prompt`` and are **byte-identical to what the eval
renders**. Re-deriving the prompts through veomni's chat pipeline would re-render them from
messages with a second template implementation, silently reintroducing exactly the train/eval
prompt mismatch the converter exists to prevent. So we keep the shards and adapt the reader.

The shards are a flat concatenation of EOS-terminated documents plus a parallel bool mask that is
True on assistant-response tokens. This class splits on EOS, packs whole documents into
``max_seq_len`` windows, and emits ``input_ids`` / ``labels`` with ``IGNORE_INDEX`` wherever the
mask is False.

**Both arms consume this identically.** Same vocabulary, same shards, same packing, same seed, so
HiLS and Olmo-3 receive byte-identical batches in the same order. That is the strongest control
available here: any difference in the eval afterwards is the model, not the data pipeline.
"""

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def _shard_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Find the ``(token_ids, labels_mask)`` shard file pairs in ``data_dir``.

    :param data_dir: Directory holding ``token_ids_part_*.npy`` and ``labels_mask_*.npy``.

    :returns: Sorted list of matching path pairs.

    :raises FileNotFoundError: If no shards are present, or a token shard has no mask twin.
    """
    toks = sorted(glob.glob(os.path.join(data_dir, "token_ids_part_*.npy")))
    if not toks:
        raise FileNotFoundError(f"no token_ids_part_*.npy under {data_dir}")
    pairs = []
    for t in toks:
        m = t.replace("token_ids_part_", "labels_mask_")
        if not os.path.exists(m):
            # A token shard without its mask would train on the PROMPT as well as the response,
            # which still converges and still evaluates -- just to the wrong objective.
            raise FileNotFoundError(f"{t} has no labels_mask twin at {m}")
        pairs.append((t, m))
    return pairs


def split_documents(
    token_ids: np.ndarray, labels_mask: np.ndarray, eos_token_id: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split a flat shard into per-document ``(ids, mask)`` slices at EOS boundaries.

    The converter appends EOS after every instance, so EOS is the document terminator and is kept
    as the last token of each document.

    :param token_ids: Flat uint32 token array.
    :param labels_mask: Parallel bool array, True on assistant-response tokens.
    :param eos_token_id: The document separator.

    :returns: One ``(ids, mask)`` pair per document.
    """
    if token_ids.shape != labels_mask.shape:
        raise ValueError(f"shard shape mismatch: ids {token_ids.shape} vs mask {labels_mask.shape}")
    ends = np.flatnonzero(token_ids == eos_token_id)
    docs, start = [], 0
    for e in ends:
        stop = int(e) + 1
        docs.append((token_ids[start:stop], labels_mask[start:stop]))
        start = stop
    # A trailing partial document (no terminating EOS) is dropped rather than emitted: it would be
    # a truncated example whose response is cut, i.e. a wrong training target.
    return docs


def mix_documents(
    per_source: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    weights: Dict[str, float],
    seed: int = 34521,
    max_repetition_factor: float = 8.0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Dict[str, float]]]:
    """
    Mix per-source document pools to target **token** shares, the way olmo_core's mixing loader
    does — by repeating or subsampling each source, not by concatenating them.

    This exists because our SFT mixture is defined by sampling weights
    (``{contra: 2.9, rerank: 1.5, outlier: 1.5, nq: 1.0, oolong: 1.3}`` + 25% Dolci), and a flat
    concatenated corpus has no mixing stage at all: the realized share would just be whatever each
    task's raw token count happens to be, and the weights would silently do nothing.

    Shares are over **tokens**, not documents, because the weights are compensation for long
    documents being dropped — a document-count share would not restore the token share it is meant
    to fix.

    The scale is chosen as the largest that no source has to exceed ``max_repetition_factor`` to
    reach; that makes the most-upsampled source the binding constraint and keeps every ratio exact,
    rather than hitting the target for some sources and quietly missing it for others.

    :param per_source: ``{name: [(ids, mask), ...]}`` document pools.
    :param weights: ``{name: weight}``; normalized internally. Must cover every source.
    :param seed: Sampling seed. Must match across arms.
    :param max_repetition_factor: Cap on how often a source may be repeated.

    :returns: ``(documents, report)`` where report gives per-source target/realized token shares.

    :raises ValueError: If a source has no weight, or a weight names an absent source.
    """
    missing = set(per_source) ^ set(weights)
    if missing:
        raise ValueError(f"sources and weights disagree on: {sorted(missing)}")

    tokens = {k: sum(len(i) for i, _ in v) for k, v in per_source.items()}
    for k, n in tokens.items():
        if n == 0:
            raise ValueError(f"source {k!r} contributed no tokens")
    total_w = sum(weights.values())
    target_frac = {k: w / total_w for k, w in weights.items()}

    # Largest total budget T such that every source's requirement stays within its cap:
    #   target_frac[k] * T <= tokens[k] * max_repetition_factor
    budget = min(tokens[k] * max_repetition_factor / target_frac[k] for k in per_source)
    target_tokens = {k: target_frac[k] * budget for k in per_source}

    rng = np.random.default_rng(seed)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    report: Dict[str, Dict[str, float]] = {}
    for name, docs in per_source.items():
        want = target_tokens[name]
        order = rng.permutation(len(docs))
        picked, got, i = [], 0, 0
        while got < want:
            d = docs[order[i % len(docs)]]
            picked.append(d)
            got += len(d[0])
            i += 1
        out.extend(picked)
        report[name] = {
            "available_tokens": tokens[name],
            "target_share": target_frac[name],
            "realized_tokens": got,
            "repetition_factor": got / tokens[name],
            "documents_emitted": len(picked),
        }
    grand = sum(r["realized_tokens"] for r in report.values())
    for r in report.values():
        r["realized_share"] = r["realized_tokens"] / grand
    rng.shuffle(out)  # type: ignore[arg-type]
    return out, report


class SFTShardDataset(Dataset):
    """
    Map-style dataset over packed SFT windows.

    :param data_dir: Directory of ``token_ids_part_*.npy`` / ``labels_mask_*.npy`` pairs.
    :param max_seq_len: Window length in tokens.
    :param eos_token_id: Document separator id (OLMo-3: ``100257``).
    :param pad_token_id: Id used to fill the tail of a window (OLMo-3: ``100277``). Padded
        positions are always ``IGNORE_INDEX`` in the labels, so they never contribute loss.
    :param drop_longer_than_window: Documents that cannot fit a window at all are dropped and
        counted. They cannot be split without cutting a response mid-answer.
    :param seed: Shuffle seed for document order before packing. Must match across arms.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int,
        eos_token_id: int,
        pad_token_id: int,
        drop_longer_than_window: bool = True,
        seed: int = 34521,
        sources: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        max_repetition_factor: float = 8.0,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.mix_report: Dict[str, Dict[str, float]] = {}

        def _load_dir(d: str) -> List[Tuple[np.ndarray, np.ndarray]]:
            out: List[Tuple[np.ndarray, np.ndarray]] = []
            for tok_path, mask_path in _shard_pairs(d):
                ids = np.load(tok_path, mmap_mode="r")
                mask = np.load(mask_path, mmap_mode="r")
                out.extend(split_documents(np.asarray(ids), np.asarray(mask), eos_token_id))
            return out

        if sources:
            if not weights:
                raise ValueError("sources given without weights -- the mixture would be undefined")
            per_source = {name: _load_dir(d) for name, d in sources.items()}
            docs, self.mix_report = mix_documents(
                per_source, weights, seed=seed, max_repetition_factor=max_repetition_factor
            )
        else:
            docs = _load_dir(data_dir)

        self.n_docs_total = len(docs)
        self.n_docs_dropped = 0
        if drop_longer_than_window:
            kept = [d for d in docs if len(d[0]) <= max_seq_len]
            self.n_docs_dropped = len(docs) - len(kept)
            docs = kept

        rng = np.random.default_rng(seed)
        rng.shuffle(docs)  # type: ignore[arg-type]

        # Best-fit-decreasing would pack tighter, but packing choice is NOT a confound here: both
        # arms read this same class with the same seed, so they get identical windows either way.
        # Next-fit is kept because it preserves the shuffled order, which keeps the task mixture
        # locally well-mixed rather than sorting long documents together.
        self.windows: List[List[Tuple[np.ndarray, np.ndarray]]] = []
        cur: List[Tuple[np.ndarray, np.ndarray]] = []
        cur_len = 0
        for ids, mask in docs:
            if cur_len + len(ids) > max_seq_len and cur:
                self.windows.append(cur)
                cur, cur_len = [], 0
            cur.append((ids, mask))
            cur_len += len(ids)
        if cur:
            self.windows.append(cur)

    def stats(self) -> Dict[str, float]:
        """:returns: Counts a launcher should log so the realized mixture is auditable."""
        tokens = sum(sum(len(i) for i, _ in w) for w in self.windows)
        trainable = sum(sum(int(m.sum()) for _, m in w) for w in self.windows)
        cap = max(1, len(self.windows) * self.max_seq_len)
        return {
            "documents_total": self.n_docs_total,
            "documents_dropped_too_long": self.n_docs_dropped,
            "windows": len(self.windows),
            "content_tokens": tokens,
            "trainable_tokens": trainable,
            "trainable_fraction": trainable / max(1, tokens),
            "packing_efficiency": tokens / cap,
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        docs = self.windows[idx]
        ids = np.concatenate([d[0] for d in docs]).astype(np.int64)
        mask = np.concatenate([d[1] for d in docs]).astype(bool)
        labels = np.where(mask, ids, IGNORE_INDEX).astype(np.int64)
        pad = self.max_seq_len - len(ids)
        if pad > 0:
            ids = np.concatenate([ids, np.full(pad, self.pad_token_id, dtype=np.int64)])
            labels = np.concatenate([labels, np.full(pad, IGNORE_INDEX, dtype=np.int64)])
        attention_mask = np.zeros(self.max_seq_len, dtype=np.int64)
        attention_mask[: self.max_seq_len - max(0, pad)] = 1
        return {
            "input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(attention_mask),
        }
