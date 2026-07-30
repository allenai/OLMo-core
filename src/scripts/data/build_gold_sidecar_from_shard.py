"""
Build a gold-document sidecar (``gold_fingerprints.json`` / ``gold_pairs.json``) for an
ALREADY-TOKENIZED document-chunked shard, without needing the source JSONL.

Why this exists: the gold sidecar is normally emitted at tokenize time from the unified JSONL's
``gold_doc_indices`` (``convert_unified_to_document_landmark.py --emit-gold-sidecar``). But for the
contradiction task the gold documents *are the answer* -- every example's supervised span decodes to
``<think></think>[[a, b], [c, d], [e, f]]<|im_end|>`` -- so the gold set can be recovered from the
shard itself. That lets us attach gold-grad to existing shards (e.g. the 2000-example n100 shard) whose
source JSONL is no longer around, instead of rebuilding a smaller dataset.

The recovered indices are cross-checked against the document structure actually present in the tokens
(every gold index must name a real document in that example), and the tool can verify itself against an
existing sidecar with ``--verify-against``.

Two output shapes, selected by ``--emit``:

* ``flat`` (default) -> ``gold_fingerprints.json``: ``{fingerprint: [gold chunk ids]}``, an unordered
  **set** per example. This is all the doc-level gold-grad policies need.
* ``pairs`` -> ``gold_pairs.json``: ``{fingerprint: [[a, b], [c, d], ...]}``, preserving **which
  document contradicts which**. Required by anything pair-aware: the gold-grad ``gold_pair`` /
  ``gold_halves`` policies, and the ``"gold_hop_controlled"`` mask (deleting "the gold edge" is
  meaningless without knowing the partner). The flat form *cannot* express this, which is the defect
  that invalidated the first gold-grad arms -- see
  ``records/contradiction-data-and-base-hygiene.md`` §3.

Both modes recover from the same parse of the label span, so the flattened pairs are the flat set by
construction; ``--verify-against`` proves it against the shard's existing flat sidecar.

Usage::

    python src/scripts/data/build_gold_sidecar_from_shard.py \\
        --shard-dir /path/to/contradiction_n100_docdense_nocot \\
        [--verify-against /path/to/known_good/gold_fingerprints.json]

    # pair-preserving sidecar, verified to flatten to the existing flat one:
    python src/scripts/data/build_gold_sidecar_from_shard.py \\
        --shard-dir /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig \\
        --emit pairs \\
        --verify-against /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_fingerprints.json
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



# A contradiction label span is literally the pair list: ``[[2, 8], [11, 17], [3, 9]]``. This matches
# each INNER pair, so the grouping (which claim contradicts which) survives -- unlike a flat ``\d+``
# scan over the whole span.
_PAIR_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


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
    ap.add_argument(
        "--emit",
        choices=("flat", "pairs"),
        default="flat",
        help="'flat' -> gold_fingerprints.json (unordered gold set); 'pairs' -> gold_pairs.json "
        "(pair-preserving [[a, b], ...], 0-based chunk ids) -- required by the pair-aware gold-grad "
        "policies and the 'gold_hop_controlled' mask.",
    )
    ap.add_argument("--out", default=None, help="default <shard-dir>/gold_{fingerprints,pairs}.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--verify-against",
        default=None,
        help="an existing sidecar to check against. With '--emit pairs' the recovered pairs are "
        "FLATTENED and compared to the reference's gold sets, which proves the pair sidecar is a "
        "strict refinement of the flat one (same gold docs, now grouped).",
    )
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

    emit_pairs = args.emit == "pairs"
    table: Dict[str, List] = {}
    n_docs_seen: List[int] = []
    n_gold_seen: List[int] = []
    n_pairs_seen: List[int] = []
    pair_dists: List[int] = []
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

        value: List = gold
        if emit_pairs:
            # Parse the INNER groups, independently of the flat ``\d+`` scan above, then require the
            # two parses to agree. That is what makes "the pairs flatten to the flat sidecar" a
            # checked property rather than a comment.
            pairs = [
                sorted((int(a) - 1, int(b) - 1)) for a, b in _PAIR_RE.findall(m.group(0))
            ]
            if not pairs or sorted({d for p in pairs for d in p}) != gold:
                bad += 1
                continue
            value = [list(p) for p in pairs]
            n_pairs_seen.append(len(pairs))
            pair_dists.extend(b - a for a, b in pairs)

        table[content_fingerprint(row.tolist())] = value
        n_docs_seen.append(len(present))
        n_gold_seen.append(len(gold))

    if bad:
        raise SystemExit(f"FATAL: {bad} examples had no parseable / out-of-range gold answer")

    default_name = "gold_pairs.json" if emit_pairs else "gold_fingerprints.json"
    out = args.out or os.path.join(args.shard_dir, default_name)
    with open(out, "w") as f:
        json.dump(table, f)

    print(f"examples          : {len(table)}")
    print(f"docs/example      : mean {np.mean(n_docs_seen):.1f}  min {min(n_docs_seen)}")
    print(f"gold/example      : mean {np.mean(n_gold_seen):.1f}")
    if emit_pairs:
        d = np.asarray(pair_dists)
        # The gold-pair DISTANCE distribution decides how much of a hop ladder is constructible: the
        # mask is causal, so a 2-hop path b->m->a needs an intermediary at a < m < b. Adjacent pairs
        # (distance 1) have none and are unroutable at ANY hop count; distance 2 additionally rules
        # out a 3-hop path. Reported here so the ladder's dilution is a measured number, not a guess.
        print(f"pairs/example     : mean {np.mean(n_pairs_seen):.2f}  total {int(d.size)}")
        print(
            f"gold-pair distance: mean {d.mean():.2f}  median {int(np.median(d))}  "
            f"min {int(d.min())}  max {int(d.max())}"
        )
        print(
            f"  dist==1 (no 2-hop possible): {int((d == 1).sum())} / {d.size} "
            f"({(d == 1).mean():.2%})"
        )
        print(
            f"  dist<=2 (no 3-hop possible): {int((d <= 2).sum())} / {d.size} "
            f"({(d <= 2).mean():.2%})"
        )
    print(f"wrote             : {out}")

    if args.verify_against:
        ref = json.load(open(args.verify_against))
        common = set(ref) & set(table)

        def _flat(v) -> List[int]:
            return sorted({int(x) for p in v for x in (p if isinstance(p, list) else [p])})

        mismatch = [k for k in common if sorted(int(x) for x in ref[k]) != _flat(table[k])]
        print(
            f"VERIFY vs {args.verify_against}: {len(common)}/{len(ref)} fingerprints matched, "
            f"{len(mismatch)} gold-set mismatches"
        )
        if len(common) != len(ref) or mismatch:
            raise SystemExit("FATAL: recovered sidecar disagrees with the reference sidecar")
        print("VERIFY PASSED: recovered gold sets are identical to the reference.")


if __name__ == "__main__":
    main()
