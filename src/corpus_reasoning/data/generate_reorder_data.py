"""Generate narrative-reordering task data from Project Gutenberg.

Pipeline:
  1. Stream books from a Hugging Face Gutenberg dataset (default:
     sedthh/gutenberg_english) or read local .txt files.
  2. Strip the Gutenberg license header/footer.
  3. Split each book into chapters by regex on chapter headings, dropping the
     heading line itself so the model can't shortcut via "Chapter 3".
  4. Split each chapter into paragraph-based chunks near a target word count.
  5. For each book with enough chunks: sample N consecutive chunks, shuffle
     them, record the gold permutation, and emit a unified-format example.

Unified schema (see scripts/lib/data_format.py):
    {
      "documents":      [{"title": null, "text": "..."}, ...],  # shuffled
      "queries":        [],
      "answers":        ["[17, 3, 42, ...]"],                    # JSON permutation
      "gold_order":     [17, 3, 42, ...],                        # 1-indexed display IDs
      "source_type":    "gutenberg",
      "source_id":      "gutenberg/<hash-or-id>",
      "n_chunks":       N,
      "chunk_word_lens":[...],
      "split":          "train" | "eval",
    }

Usage:
    python scripts/data/generate_reorder_data.py \\
        --num-examples 200 --n-chunks 50 --out-dir data/
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


# ── Gutenberg boilerplate ────────────────────────────────────────────────
GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS)?\s*PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS)?\s*PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)

# Match common chapter / book / part headings at start of a line.
# Covers: "CHAPTER I.", "Chapter 3", "CHAPTER THE FIRST", "BOOK II", "Part 4".
# We deliberately exclude bare roman numerals (too noisy — they appear in text).
CHAPTER_HEADING = re.compile(
    r"^[ \t]*(?:CHAPTER|Chapter|BOOK|Book|PART|Part)\s+"
    r"(?:[IVXLCDM]+|\d+|[A-Z][A-Za-z]+)"
    r"\b[^\n]*$",
    re.MULTILINE,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    m = GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def split_chapters(text: str) -> list[str]:
    """Return chapter bodies (heading line dropped). Empty list if no structure."""
    matches = list(CHAPTER_HEADING.finditer(text))
    if len(matches) < 3:
        return []
    chapters = []
    for i, m in enumerate(matches):
        start = m.end()  # skip heading line itself
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chapters.append(body)
    return chapters


def word_count(s: str) -> int:
    return len(s.split())


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    """Hard-cap a paragraph at max_words by sentence-splitting (then word-splitting
    if a single sentence still exceeds max_words)."""
    if word_count(paragraph) <= max_words:
        return [paragraph]
    sentences = _SENT_SPLIT.split(paragraph)
    out: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sentences:
        sw = word_count(s)
        if sw > max_words:
            if cur:
                out.append(" ".join(cur))
                cur, cur_words = [], 0
            words = s.split()
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i:i + max_words]))
            continue
        if cur_words + sw > max_words and cur:
            out.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(s)
        cur_words += sw
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_chapter(chapter: str, target_words: int, min_words: int,
                  max_words: int) -> list[str]:
    """Split a chapter into paragraph-grouped chunks in [min_words, max_words]."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chapter) if p.strip()]
    paragraphs = [sub for p in paragraphs
                  for sub in split_long_paragraph(p, max_words)]
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for p in paragraphs:
        pw = word_count(p)
        if cur_words + pw > max_words and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
        cur.append(p)
        cur_words += pw
        if cur_words >= target_words:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
    if cur and cur_words >= min_words:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if word_count(c) >= min_words]


def build_chunks(text: str, target_words: int, min_words: int,
                 max_words: int) -> list[str]:
    text = strip_gutenberg_boilerplate(text)
    chapters = split_chapters(text)
    out: list[str] = []
    for ch in chapters:
        out.extend(chunk_chapter(ch, target_words, min_words, max_words))
    return out


# ── Example construction ────────────────────────────────────────────────

def make_example(chunks: list[str], n: int, source_id: str,
                 rng: random.Random, split: str) -> dict:
    start = rng.randint(0, len(chunks) - n)
    consecutive = chunks[start:start + n]
    order = list(range(n))
    rng.shuffle(order)
    # documents[display_pos] = original chunk at order[display_pos]
    documents = [{"title": None, "text": consecutive[order[i]]} for i in range(n)]
    # gold_order[source_pos] = 1-indexed display ID of the chunk that
    # originally sat at source_pos.
    gold_order = [0] * n
    for display_pos, orig_idx in enumerate(order):
        gold_order[orig_idx] = display_pos + 1
    return {
        "documents": documents,
        "queries": [],
        "answers": [json.dumps(gold_order)],
        "gold_order": gold_order,
        "source_type": "gutenberg",
        "source_id": source_id,
        "n_chunks": n,
        "chunk_word_lens": [word_count(d["text"]) for d in documents],
        "chunk_start_offset": start,
        "split": split,
    }


# ── Input sources ───────────────────────────────────────────────────────

def iter_hf_books(dataset_name: str, text_column: str, id_column: str | None,
                  max_books: int):
    """Yield (source_id, text) from a streamed HF dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= max_books:
            break
        text = row.get(text_column)
        if not text:
            continue
        if id_column and row.get(id_column) is not None:
            sid = f"{dataset_name}/{row[id_column]}"
        else:
            h = hashlib.md5(text[:2000].encode("utf-8")).hexdigest()[:12]
            sid = f"{dataset_name}/{h}"
        yield sid, text


def iter_local_books(local_dir: str):
    for p in sorted(Path(local_dir).glob("*.txt")):
        yield f"local/{p.stem}", p.read_text(encoding="utf-8", errors="ignore")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--num-examples", type=int, default=200,
                    help="Total examples to produce (train + eval).")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--n-chunks", type=int, default=50,
                    help="Chunks per example (N).")
    ap.add_argument("--examples-per-book", type=int, default=1,
                    help="How many non-overlapping examples to draw from a single book.")
    ap.add_argument("--target-words", type=int, default=100,
                    help="Default 100 matches the wiki100w reorder data.")
    ap.add_argument("--min-words", type=int, default=90)
    ap.add_argument("--max-words", type=int, default=110)
    ap.add_argument("--min-chunks-per-book", type=int, default=0,
                    help="Default = n_chunks * examples_per_book.")
    ap.add_argument("--hf-dataset", default="sedthh/gutenberg_english")
    ap.add_argument("--text-column", default="TEXT")
    ap.add_argument("--id-column", default=None)
    ap.add_argument("--local-dir", default=None,
                    help="If set, read .txt files from this directory instead of HF.")
    ap.add_argument("--max-books-to-scan", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-books", type=int, default=0,
                    help="Skip the first K books in the stream before generating. "
                         "Use to produce held-out eval disjoint from a prior train run.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Mark every produced example split='eval' and skip the train file.")
    ap.add_argument("--out-suffix", default="",
                    help="Optional filename suffix, e.g. '100w' → reorder_gutenberg100w_n{N}_*")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.n_chunks
    min_per_book = args.min_chunks_per_book or (n * args.examples_per_book)

    if args.local_dir:
        books = iter_local_books(args.local_dir)
        src_label = f"local:{args.local_dir}"
    else:
        books = iter_hf_books(args.hf_dataset, args.text_column, args.id_column,
                              args.max_books_to_scan)
        src_label = args.hf_dataset

    print(f"Source: {src_label}")
    print(f"Target: {args.num_examples} examples, N={n} chunks each "
          f"(~{args.target_words} words/chunk)")

    examples: list[dict] = []
    n_books_scanned = 0
    n_books_skipped = 0
    n_books_used = 0
    if args.eval_only:
        n_eval = args.num_examples
    else:
        n_eval = max(1, int(round(args.num_examples * args.eval_frac)))

    for source_id, text in books:
        if n_books_skipped < args.skip_books:
            n_books_skipped += 1
            if n_books_skipped % 500 == 0:
                print(f"  skipped {n_books_skipped}/{args.skip_books} books...")
            continue
        n_books_scanned += 1
        if n_books_scanned % 50 == 0:
            print(f"  scanned {n_books_scanned} books, "
                  f"{len(examples)}/{args.num_examples} examples...")
        chunks = build_chunks(text, args.target_words, args.min_words,
                              args.max_words)
        if len(chunks) < min_per_book:
            continue
        n_books_used += 1
        # Partition chunks into contiguous blocks of size n; sample examples_per_book
        # from those blocks so examples from the same book don't overlap.
        n_blocks = len(chunks) // n
        block_indices = list(range(n_blocks))
        rng.shuffle(block_indices)
        take = min(args.examples_per_book, n_blocks,
                   args.num_examples - len(examples))
        for b in block_indices[:take]:
            local_chunks = chunks[b * n:(b + 1) * n]
            split = "eval" if len(examples) < n_eval else "train"
            ex = make_example(local_chunks, n, source_id, rng, split)
            # The helper re-samples a start offset inside local_chunks (of size n),
            # which is always 0; that's fine — we've already picked the block.
            ex["chunk_start_offset"] = b * n
            examples.append(ex)
        if len(examples) >= args.num_examples:
            break

    print(f"\nScanned {n_books_scanned} books, used {n_books_used}; "
          f"produced {len(examples)} examples.")

    rng.shuffle(examples)
    train = [e for e in examples if e["split"] == "train"]
    evald = [e for e in examples if e["split"] == "eval"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"reorder_gutenberg{args.out_suffix}_n{n}"
    eval_path = out_dir / f"{base}_eval_{len(evald)}.jsonl"
    save_jsonl(str(eval_path), evald)
    print_dataset_stats(
        [{"input": json.dumps(e["documents"])[:1], "output": e["answers"][0]}
         for e in evald], "eval", str(eval_path))
    if not args.eval_only:
        train_path = out_dir / f"{base}_train_{len(train)}.jsonl"
        save_jsonl(str(train_path), train)
        print_dataset_stats(
            [{"input": json.dumps(e["documents"])[:1], "output": e["answers"][0]}
             for e in train], "train", str(train_path))


if __name__ == "__main__":
    main()
