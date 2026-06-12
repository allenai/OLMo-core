"""Align format + length between BM25 hard negatives and gold/random docs.

BM25 hard negatives come from wikipedia-dpr-100w (~100 words) while gold and
random docs come from NQ HF contexts (variable length, space-tokenized, with
heuristic titles). That gap creates surface-form shortcuts the model can learn.

This script normalizes both populations so they share one surface format, then
truncates to a common length:

  1. Hard negs: strip the leading `"Article Name"\\n` line, unescape `""` -> `"`.
  2. Gold / random: detokenize NQ punctuation (`( hide )` -> `(hide)`, etc.).
  3. All docs: blank the title field.
  4. Truncate every doc to the median length of normalized hard negatives.
     Gold truncation centers on the first answer match to preserve the answer.

Reads `gold_doc_indices` and `hard_neg_indices` from each example to identify
provenance — the docs themselves are stored already shuffled.

Usage:
    python scripts/data/align_hn_doc_lengths.py \\
        --input data/nq_validation_k100_hn10_500.jsonl \\
        --output data/nq_validation_k100_hn10_500_aligned.jsonl
"""

import argparse
import json
import re
import statistics
from pathlib import Path


HARD_NEG_TITLE_RE = re.compile(r'^"([^"\n]*)"\n')


def normalize_hard_neg_text(text: str) -> str:
    """Strip leading `"Title"\\n` and unescape `""` -> `"`."""
    text = HARD_NEG_TITLE_RE.sub("", text, count=1)
    text = text.replace('""', '"')
    return text.strip()


def detokenize_nq_text(text: str) -> str:
    """Undo NQ's aggressive punctuation tokenization.

    NQ text has patterns like `( hide )`, `Nolan 's`, `1927 , 1988 .`, and
    Penn Treebank quotes `` `` text '' ``. This normalizes to wiki-dpr-100w's
    natural prose style.
    """
    # Penn Treebank quote style: `` text '' -> "text".
    text = text.replace("``", '"').replace("''", '"')
    # Single backtick -> apostrophe (Penn Treebank single quote).
    text = text.replace("`", "'")
    # Remove space after opening brackets.
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    # Remove space before closing brackets and punctuation.
    text = re.sub(r"\s+([)\]},.;:!?])", r"\1", text)
    # Strip space padding inside straight quotes: `" text "` -> `"text"`.
    text = re.sub(r'"\s+(\S)', r'"\1', text)
    text = re.sub(r'(\S)\s+"', r'\1"', text)
    # Contractions: `Nolan 's` -> `Nolan's`, `do n't` -> `don't`.
    text = re.sub(r"\s+'([a-zA-Z])", r"'\1", text)
    text = re.sub(r"\s+n't\b", r"n't", text)
    # Collapse any multiple spaces left behind.
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _snap_to_word_boundaries(
    text: str, start: int, end: int, protect_start: int | None = None,
    protect_end: int | None = None,
) -> tuple[int, int]:
    """Expand/shrink [start, end) to the nearest word boundaries.

    Moves `start` forward to the start of the next word (if currently
    mid-word) and `end` backward to the end of the previous word. If
    `protect_start`/`protect_end` are given, the adjustment will not cross
    that span (so the caller can guarantee the answer stays inside).
    """
    n = len(text)
    # Advance start past a partial leading word.
    if start > 0 and not text[start - 1].isspace() and start < n and not text[start].isspace():
        next_space = text.find(" ", start)
        if next_space != -1 and (protect_start is None or next_space < protect_start):
            start = next_space + 1
    # Retreat end out of a partial trailing word.
    if 0 < end < n and not text[end - 1].isspace() and not text[end].isspace():
        prev_space = text.rfind(" ", 0, end)
        if prev_space != -1 and (protect_end is None or prev_space > protect_end):
            end = prev_space
    return start, end


def trim_word_boundary(text: str, target_len: int) -> str:
    """Truncate text to ~target_len chars on word boundaries."""
    if len(text) <= target_len:
        return text
    start, end = _snap_to_word_boundaries(text, 0, target_len)
    return text[start:end].strip()


def truncate_around_answer(text: str, answers: list[str], target_len: int) -> str:
    """Truncate text to ~target_len chars, centering the window on the first
    answer occurrence (case insensitive). Snaps both ends to word boundaries
    while keeping the answer span inside the window. If no answer is found,
    truncates from the start on a word boundary.
    """
    if len(text) <= target_len:
        return text

    text_lower = text.lower()
    ans_pos = -1
    ans_len = 0
    for ans in answers:
        if not ans:
            continue
        p = text_lower.find(ans.lower())
        if p != -1:
            ans_pos = p
            ans_len = len(ans)
            break

    if ans_pos == -1:
        return trim_word_boundary(text, target_len)

    slack = target_len - ans_len
    if slack <= 0:
        return text[ans_pos : ans_pos + max(ans_len, target_len)]

    start = max(0, ans_pos - slack // 2)
    end = start + target_len
    if end > len(text):
        end = len(text)
        start = max(0, end - target_len)

    start, end = _snap_to_word_boundaries(
        text, start, end, protect_start=ans_pos, protect_end=ans_pos + ans_len
    )
    return text[start:end].strip()


def _flatten_gold(gold) -> list[int]:
    if not gold:
        return []
    if isinstance(gold[0], list):
        return sorted({i for g in gold for i in g})
    return list(gold)


def normalize_example(ex: dict) -> dict:
    """Step 1: normalize surface format without truncating."""
    docs = ex["documents"]
    hard_set = set(ex.get("hard_neg_indices") or [])

    new_docs = []
    for i, d in enumerate(docs):
        if i in hard_set:
            new_text = normalize_hard_neg_text(d["text"])
        else:
            new_text = detokenize_nq_text(d["text"])
        new_docs.append({"title": "", "text": new_text})

    return {**ex, "documents": new_docs}


def collect_gold_lens(examples: list[dict]) -> list[int]:
    gold = []
    for ex in examples:
        gold_set = set(_flatten_gold(ex.get("gold_doc_indices", [])))
        for i in gold_set:
            gold.append(len(ex["documents"][i]["text"]))
    if not gold:
        raise ValueError("No gold documents found")
    return gold


def quantile_match_targets(n: int, gold_lens: list[int]) -> list[int]:
    """Build a length of `n` targets whose distribution matches `gold_lens`.

    Sorts gold lengths descending and interpolates across `n` positions.
    Caller pairs this with a descending-sorted list of docs so rank i gets
    the i-th quantile target.
    """
    if n == 0:
        return []
    gold_sorted = sorted(gold_lens, reverse=True)
    g = len(gold_sorted)
    return [gold_sorted[min(int(rank * g / n), g - 1)] for rank in range(n)]


def distribution_match_truncate(
    examples: list[dict], gold_lens: list[int]
) -> list[dict]:
    """Rewrite hard/rand docs so each class's length distribution matches gold's.

    Processes the hard-negative pool and the random-pool *separately*: within
    each class, sorts docs descending by length and pairs each rank with a
    gold-length target at the same quantile. Each doc is then truncated
    (word-boundary aware) to its assigned target. Gold docs are untouched.
    """
    # Deep-copy so we don't clobber the input examples.
    out = [dict(ex) for ex in examples]
    for ex in out:
        ex["documents"] = [dict(d) for d in ex["documents"]]

    hard_locs: list[tuple[int, int, int]] = []  # (ex_idx, doc_idx, length)
    rand_locs: list[tuple[int, int, int]] = []
    for ex_idx, ex in enumerate(out):
        docs = ex["documents"]
        gold_set = set(_flatten_gold(ex.get("gold_doc_indices", [])))
        hard_set = set(ex.get("hard_neg_indices") or [])
        for i, d in enumerate(docs):
            if i in gold_set:
                continue
            rec = (ex_idx, i, len(d["text"]))
            if i in hard_set:
                hard_locs.append(rec)
            else:
                rand_locs.append(rec)

    for locs in (hard_locs, rand_locs):
        locs.sort(key=lambda t: t[2], reverse=True)
        targets = quantile_match_targets(len(locs), gold_lens)
        for (ex_idx, doc_idx, _old_len), target in zip(locs, targets):
            doc = out[ex_idx]["documents"][doc_idx]
            doc["text"] = trim_word_boundary(doc["text"], target)

    return out


def lens_by_kind(exs: list[dict]):
    gold, hard, rand = [], [], []
    for ex in exs:
        docs = ex["documents"]
        gold_set = set(_flatten_gold(ex.get("gold_doc_indices", [])))
        hard_set = set(ex.get("hard_neg_indices") or [])
        for i in range(len(docs)):
            ln = len(docs[i]["text"])
            if i in gold_set:
                gold.append(ln)
            elif i in hard_set:
                hard.append(ln)
            else:
                rand.append(ln)
    return gold, hard, rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    with args.input.open() as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} examples from {args.input}")

    def log_lens(label: str, exs: list[dict]) -> None:
        gold, hard, rand = lens_by_kind(exs)
        parts = []
        for name, vs in [("gold", gold), ("hard", hard), ("rand", rand)]:
            if vs:
                parts.append(
                    f"{name} mean={statistics.mean(vs):.0f} "
                    f"median={statistics.median(vs):.0f}"
                )
        print(f"{label:<22} {' | '.join(parts)}")

    log_lens("BEFORE (raw):", examples)

    # Step 1: format normalization.
    normalized = [normalize_example(ex) for ex in examples]
    log_lens("AFTER normalize:", normalized)

    # Step 2: distribution-match hard/rand to gold's length distribution.
    gold_lens = collect_gold_lens(normalized)
    print(
        f"Gold distribution (n={len(gold_lens)}): "
        f"median={int(statistics.median(gold_lens))} "
        f"mean={statistics.mean(gold_lens):.0f} "
        f"min={min(gold_lens)} max={max(gold_lens)}"
    )

    aligned = distribution_match_truncate(normalized, gold_lens)
    log_lens("AFTER align:", aligned)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for ex in aligned:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(aligned)} examples to {args.output}")


if __name__ == "__main__":
    main()
