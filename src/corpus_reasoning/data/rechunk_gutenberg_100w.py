"""Re-chunk an existing gutenberg reorder file at exactly 100 words/chunk.

The existing files were generated with target_words=200 (paragraph grouping),
so chunks are ~205 words on average and train splits have long-tail outliers
up to thousands of words. To match the wiki100w reorder data (uniform ~100w
chunks), we reconstruct each example's pre-shuffle narrative and re-slice it
into exactly 100-word chunks, then take the first N as the new example.

Each example's source narrative is recovered from its `documents` and
`gold_order` fields (gold_order[orig_idx] = display_pos+1).

Examples whose reconstructed narrative is too short to yield N full 100-word
chunks are dropped.

Usage:
    python scripts/data/rechunk_gutenberg_100w.py \\
        --in data/reorder_gutenberg_n20_train_1995.jsonl \\
        --out data/reorder_gutenberg100w_n20_train.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def reconstruct_narrative(example: dict) -> list[str]:
    docs = example["documents"]
    gold_order = example["gold_order"]
    n = len(docs)
    original: list[str] = [""] * n
    for orig_idx, display_1based in enumerate(gold_order):
        original[orig_idx] = docs[display_1based - 1]["text"]
    return original


def rechunk_100w(original_chunks: list[str], words_per_chunk: int) -> list[str]:
    full_text = " ".join(original_chunks)
    words = full_text.split()
    out = []
    for i in range(0, len(words), words_per_chunk):
        piece = words[i:i + words_per_chunk]
        if len(piece) == words_per_chunk:
            out.append(" ".join(piece))
    return out


def rebuild_example(example: dict, n: int, rng: random.Random,
                    words_per_chunk: int) -> dict | None:
    original = reconstruct_narrative(example)
    new_chunks = rechunk_100w(original, words_per_chunk)
    if len(new_chunks) < n:
        return None
    selected = new_chunks[:n]
    order = list(range(n))
    rng.shuffle(order)
    documents = [{"title": None, "text": selected[order[i]]} for i in range(n)]
    gold_order = [0] * n
    for display_pos, orig_idx in enumerate(order):
        gold_order[orig_idx] = display_pos + 1
    out = dict(example)
    out["documents"] = documents
    out["answers"] = [json.dumps(gold_order)]
    out["gold_order"] = gold_order
    out["n_chunks"] = n
    out["chunk_word_lens"] = [len(d["text"].split()) for d in documents]
    return out


def process(in_path: Path, out_path: Path, words_per_chunk: int, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    n_in = 0
    n_out = 0
    with in_path.open() as f, out_path.open("w") as g:
        for line in f:
            n_in += 1
            ex = json.loads(line)
            n = ex["n_chunks"]
            new_ex = rebuild_example(ex, n, rng, words_per_chunk)
            if new_ex is None:
                continue
            g.write(json.dumps(new_ex) + "\n")
            n_out += 1
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--words-per-chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    in_p, out_p = Path(args.in_path), Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_in, n_out = process(in_p, out_p, args.words_per_chunk, args.seed)
    print(f"{in_p.name}: {n_in} → {n_out} examples (dropped {n_in - n_out})")


if __name__ == "__main__":
    main()
