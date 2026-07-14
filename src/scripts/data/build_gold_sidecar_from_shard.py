"""
Build a gold-document sidecar (``gold_fingerprints.json``) for an ALREADY-TOKENIZED document-chunked
shard, without needing the source JSONL.

Why this exists: the gold sidecar is normally emitted at tokenize time from the unified JSONL's
``gold_doc_indices`` (``convert_unified_to_document_landmark.py --emit-gold-sidecar``). But for the
contradiction task the gold documents *are the answer* -- every example's supervised span decodes to
``<think></think>[[a, b], [c, d], [e, f]]<|im_end|>`` -- so the gold set can be recovered from the
shard itself. That lets us attach gold-grad to existing shards (e.g. the 2000-example n100 shard) whose
source JSONL is no longer around, instead of rebuilding a smaller dataset.

The recovered indices are cross-checked against the document structure actually present in the tokens
(every gold index must name a real document in that example), and the tool can verify itself against an
existing sidecar with ``--verify-against``.

Usage::

    python src/scripts/data/build_gold_sidecar_from_shard.py \\
        --shard-dir /path/to/contradiction_n100_docdense_nocot \\
        [--verify-against /path/to/known_good/gold_fingerprints.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch

from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint
from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
)



def _iter_examples(ids: np.ndarray, eos_id: int):
    """Yield ``(start, end)`` spans, one per example. Each example ends at (and includes) its EOS."""
    eos = np.flatnonzero(ids == eos_id)
    start = 0
    for e in eos.tolist():
        yield start, e + 1
        start = e + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True, help="dir with token_ids_*.npy + labels_mask_*.npy")
    ap.add_argument("--out", default=None, help="default <shard-dir>/gold_fingerprints.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--verify-against", default=None, help="an existing sidecar to check against")
    ap.add_argument("--doc-start-id", type=int, default=DOC_START_ID)
    ap.add_argument("--doc-end-id", type=int, default=DOC_END_ID)
    ap.add_argument("--eos-id", type=int, default=EOS_TOKEN_ID)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    tok_path = os.path.join(args.shard_dir, "token_ids_part_000000.npy")
    lab_path = os.path.join(args.shard_dir, "labels_mask_part_000000.npy")
    n_tokens = os.path.getsize(tok_path) // 4  # uint32
    ids = np.asarray(np.memmap(tok_path, dtype=np.uint32, mode="r", shape=(n_tokens,)))
    lab = np.asarray(np.memmap(lab_path, dtype=bool, mode="r", shape=(n_tokens,)))

    table: Dict[str, List[int]] = {}
    n_docs_seen: List[int] = []
    n_gold_seen: List[int] = []
    bad = 0

    for start, end in _iter_examples(ids, args.eos_id):
        row = ids[start:end].astype(np.int64)

        # The gold documents ARE the answer: parse the supervised span's "[[a, b], [c, d], ...]".
        ans_ids = row[lab[start:end]]
        text = tok.decode(ans_ids.tolist())
        m = re.search(r"\[\[.*?\]\]", text, flags=re.S)
        if m is None:
            bad += 1
            continue
        claim_ids = [int(x) for x in re.findall(r"\d+", m.group(0))]

        # Cross-check against the document structure actually present in the tokens: a gold index must
        # name a real document in THIS example (chunk index == Claim id - 1 in the dense layout).
        roles = build_chunk_ids_from_tokens(
            torch.tensor(row).unsqueeze(0),
            doc_start_id=args.doc_start_id,
            doc_end_id=args.doc_end_id,
            eos_id=args.eos_id,
        )[0]
        present = {int(d) for d in torch.unique(roles[roles >= 0]).tolist()}
        gold = sorted({c - 1 for c in claim_ids})
        if not gold or not set(gold).issubset(present):
            bad += 1
            continue

        table[content_fingerprint(row.tolist())] = gold
        n_docs_seen.append(len(present))
        n_gold_seen.append(len(gold))

    if bad:
        raise SystemExit(f"FATAL: {bad} examples had no parseable / out-of-range gold answer")

    out = args.out or os.path.join(args.shard_dir, "gold_fingerprints.json")
    with open(out, "w") as f:
        json.dump(table, f)

    print(f"examples          : {len(table)}")
    print(f"docs/example      : mean {np.mean(n_docs_seen):.1f}  min {min(n_docs_seen)}")
    print(f"gold/example      : mean {np.mean(n_gold_seen):.1f}")
    print(f"wrote             : {out}")

    if args.verify_against:
        ref = json.load(open(args.verify_against))
        common = set(ref) & set(table)
        mismatch = [k for k in common if sorted(ref[k]) != sorted(table[k])]
        print(
            f"VERIFY vs {args.verify_against}: {len(common)}/{len(ref)} fingerprints matched, "
            f"{len(mismatch)} gold-set mismatches"
        )
        if len(common) != len(ref) or mismatch:
            raise SystemExit("FATAL: recovered sidecar disagrees with the reference sidecar")
        print("VERIFY PASSED: recovered gold sets are identical to the reference.")


if __name__ == "__main__":
    main()
