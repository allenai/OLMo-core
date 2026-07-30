"""Generate strmatch (string-comparison) data.

The string analogue of mathmatch. Each example is a list of N short "strings",
each a sequence of L words drawn from wiki100w vocabulary. The task carries a
per-example criterion. Two relation modes (`--relation`):

  substring (default) — generalizes matching_ngram from "exactly identical
    snippet" to "share a contiguous run of >= k consecutive words". Gold pairs
    share one contiguous k-word span; hard-negative pairs share a (k-1)-word
    span (a near miss that forces the model to count run length exactly).

  wordset — pairs that share >= W words in common (order/contiguity ignored).
    Gold pairs share exactly W words; hard-negatives share W-1.

In both modes every non-planted word is globally unique within the example, so
the ONLY pairs meeting the criterion are the K planted gold pairs (guaranteed by
construction). Mirrors the contradiction / matching-ngram / mathmatch tasks
(`gold_doc_indices: [[a, b], ...]`, 1-indexed) so the same prompt builder,
parser, and pair metrics apply. Evaluate with `--task strmatch`.

Difficulty knobs:
  --span-len k / --share-words W   match threshold (longer run / more words = harder)
  --str-len L                      words per string
  --num-hardneg H                  near-miss pairs (k-1 / W-1), per example (default K)
  --num-docs N                     corpus size (the O(N^2) scaling axis)

Vocab source:
  --vocab-source wiki  draw words from wiki100w via BM25 (default; matches the
                       matching_ngram substrate; needs train/eval env -> sbatch)
  --vocab-source file  draw words from --words-file (e.g. /usr/share/dict/words);
                       runs anywhere, for fast iteration / preview

Usage:
    python scripts/data/generate_strmatch_data.py \\
        --num-docs 100 --num-pairs 3 --relation substring --span-len 3 \\
        --str-len 10 --num-train 2000 --num-eval 300 --output-dir data
"""

import argparse
import json
import random
import re

_WORD = re.compile(r"[A-Za-z]{2,}")


def load_vocab(args):
    """Return a flat list of distinct lower-cased word tokens."""
    if args.vocab_source == "wiki":
        from corpus_reasoning.lib.bm25 import BM25Searcher
        from corpus_reasoning.lib.wiki100w_sample import extract_title_and_body
        searcher = BM25Searcher()
        docs = searcher.prefetch_random_pool(args.pool_passages, threads=16,
                                             seed=args.seed)
        seen, vocab = set(), []
        for d in docs:
            _, body = extract_title_and_body(d["text"])
            for w in (w.lower() for w in _WORD.findall(body)):
                if w not in seen:
                    seen.add(w)
                    vocab.append(w)
        return vocab
    seen, vocab = set(), []
    with open(args.words_file) as f:
        for line in f:
            w = line.strip().lower()
            if w and w.isalpha() and len(w) >= 2 and w not in seen:
                seen.add(w)
                vocab.append(w)
    return vocab


def build_example(vocab, args, K, thr, H, rng):
    """One example. `thr` is the match threshold (span length k for substring
    mode, or shared-word count W for wordset mode). Plants K gold pairs at
    threshold `thr` and H hard-negative pairs at `thr-1`; all other words are
    globally unique within the example.
    """
    L = args.str_len
    N = args.num_docs
    substring = args.relation == "substring"
    assert 1 <= thr <= L, "threshold must be in [1, str-len]"

    n_distract = N - 2 * K - 2 * H
    assert n_distract >= 0, "num_docs too small for K gold + H hardneg pairs"

    # Upper bound on distinct words consumed.
    def pair_cost(t):
        return t + 2 * (L - t)
    need = K * pair_cost(thr) + H * pair_cost(thr - 1) + n_distract * L
    words = rng.sample(vocab, need)
    cur = 0

    def take(k):
        nonlocal cur
        chunk = words[cur:cur + k]
        cur += k
        return chunk

    def make_pair(t):
        """Two length-L strings sharing a planted block of `t` words."""
        block = take(t)
        if substring:
            # Insert the SAME contiguous block at a random offset in each;
            # surrounding fillers are unique, so the only shared run is `block`.
            pa = rng.randint(0, L - t)
            pb = rng.randint(0, L - t)
            a = take(pa) + block + take(L - t - pa)
            b = take(pb) + block + take(L - t - pb)
        else:
            # Shared as a set; positions irrelevant, so shuffle each whole.
            a = block + take(L - t)
            b = block + take(L - t)
            rng.shuffle(a)
            rng.shuffle(b)
        return a, b

    docs, gold_positions = [], []
    for _ in range(K):
        a, b = make_pair(thr)
        gold_positions.append((len(docs), len(docs) + 1))
        docs.extend([a, b])
    for _ in range(H):
        a, b = make_pair(thr - 1)
        docs.extend([a, b])
    for _ in range(n_distract):
        docs.append(take(L))

    order = list(range(len(docs)))
    rng.shuffle(order)
    old_to_new = {old: new + 1 for new, old in enumerate(order)}
    shuffled = [docs[old] for old in order]
    pairs = sorted(sorted([old_to_new[a], old_to_new[b]])
                   for a, b in gold_positions)

    if substring:
        query = (f"Find all pairs of strings that contain a run of at least "
                 f"{thr} consecutive words in common (the same {thr} words, in "
                 f"the same order, appearing contiguously in both strings).")
        meta = {"relation": "substring", "span_len": thr, "str_len": L,
                "num_hardneg": H}
    else:
        query = (f"Find all pairs of strings that share at least {thr} words in "
                 f"common (the same word appearing in both strings, regardless "
                 f"of position).")
        meta = {"relation": "wordset", "share_words": thr, "str_len": L,
                "num_hardneg": H}

    return {
        "documents": [{"text": " ".join(ws)} for ws in shuffled],
        "queries": [query],
        "answers": [],
        "gold_doc_indices": pairs,
        "source": "strmatch",
        "_meta": meta,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-docs", type=int, default=100)
    ap.add_argument("--num-pairs", type=int, default=3,
                    help="exact number of gold pairs")
    ap.add_argument("--relation", choices=["substring", "wordset"],
                    default="substring")
    ap.add_argument("--span-len", type=int, default=3,
                    help="substring mode: contiguous run length k")
    ap.add_argument("--share-words", type=int, default=3,
                    help="wordset mode: shared-word threshold W")
    ap.add_argument("--str-len", type=int, default=10,
                    help="words per string (L)")
    ap.add_argument("--num-hardneg", type=int, default=None,
                    help="near-miss pairs at threshold-1, per example (default K)")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--vocab-source", choices=["file", "wiki"], default="wiki")
    ap.add_argument("--words-file", default="/usr/share/dict/words")
    ap.add_argument("--pool-passages", type=int, default=5000)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    K = args.num_pairs
    H = args.num_hardneg if args.num_hardneg is not None else K
    thr = args.span_len if args.relation == "substring" else args.share_words
    rng = random.Random(args.seed)
    vocab = load_vocab(args)
    print(f"Vocab: {len(vocab):,} distinct words ({args.vocab_source})")

    for split_label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count <= 0:
            continue
        examples = [build_example(vocab, args, K, thr, H, rng)
                    for _ in range(count)]
        rtag = f"sub{thr}" if args.relation == "substring" else f"ws{thr}"
        tag = f"n{args.num_docs}_k{K}_{rtag}_l{args.str_len}_h{H}"
        path = f"{args.output_dir}/strmatch_{split_label}_{tag}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_label}: {len(examples)} examples -> {path}")


if __name__ == "__main__":
    main()
