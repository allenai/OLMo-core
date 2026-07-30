"""Generate AbsenceBench-style data: the inverse of needle-in-a-haystack.

The model sees the full numbered ORIGINAL corpus, then a second version with
some elements removed, and must name which numbered elements are MISSING. Scored
by set-F1 over the removed IDs (equivalent to AbsenceBench's micro-F1 over
removed elements; arXiv:2506.11440).

Three data paths (all emit the same unified `absence` schema):

  --from <jsonl>     Derive over OUR corpora: take each source example's
                     `documents` as the element list, Bernoulli-delete each with
                     prob --p, keep the rest in order. Scalable (any N from the
                     source), no LLM. e.g. contradiction_eval_pubmed_*_n100.
  --numerical        Synthetic arithmetic sequences (their numerical domain):
                     N numbers with --step, ascending/random; delete with --p.
  --official         Port the published harveyfin/AbsenceBench (poetry /
                     numerical / github_prs): split each original_context into
                     elements and use the dataset's omitted_index as gold.
                     Needs `datasets` (run via sbatch).

Schema:
    {"documents": [{"text": <element>}, ...],     # original, numbered 1..N
     "queries":   ["Second version:\\n\\n<kept elements, in order>"],
     "answers":   [],
     "gold_doc_indices": [removed positions, 0-indexed],
     "source": "absence_<domain>"}

Difficulty: --p (deletion rate; AbsenceBench finds *scattered single* deletions
HARDER than big contiguous gaps, so low --p is the hard regime) and N.

Usage:
    python scripts/data/generate_absence_data.py --from \\
        data/contradiction_eval_pubmed_both_n100_k3.jsonl --p 0.1 \\
        --num-eval 300 --output-dir data
"""

import argparse
import json
import random
import re


# ── Gutenberg text-diff variant ──────────────────────────────────────────
# A random segment of N consecutive sentences from a Project Gutenberg book.
# Version A = the full segment; Version B = the segment with K whole sentences
# removed. Target: the first four words of each removed sentence, in order of
# occurrence, as a JSON list. No document IDs. Scales to arbitrary length via N.
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "no", "mt",
           "capt", "gen", "col", "sgt", "rev", "hon", "prof", "messrs", "esq"}

# Terminal punctuation a real sentence may end on (incl. closing quotes).
_SENT_END = tuple(".!?\"')”’")
# Opening characters a real sentence may start with (letter, digit, or quote).
_SENT_OPEN = "\"'“‘"


# Honorific / academic / bibliographic abbreviations punkt's default English
# model does NOT know, so it wrongly splits "the Rev. John ..." or "M.A., and ...".
# punkt stores abbrevs lowercased with the trailing period dropped.
_EXTRA_ABBREV = {
    "rev", "hon", "esq", "jr", "jun", "sr", "sen", "gov", "gen", "col", "capt",
    "lt", "maj", "sgt", "pres", "prof", "dr", "mr", "mrs", "ms", "st", "mt",
    "messrs", "vol", "no", "pp", "p", "vs", "viz", "fig", "ed", "trans", "etc",
    "ca", "cf", "m.a", "ph.d", "ll.d", "d.d", "b.a", "m.d", "f.r.s", "f.s.a",
}
_PUNKT = None


def _get_punkt():
    """A punkt sentence tokenizer with our extra abbreviations folded in."""
    global _PUNKT
    if _PUNKT is None:
        import nltk
        tok = nltk.data.load("tokenizers/punkt/english.pickle")
        tok._params.abbrev_types.update(_EXTRA_ABBREV)
        _PUNKT = tok
    return _PUNKT


def _tokenize_sentences(text):
    """Whitespace-normalize, then sentence-split with NLTK punkt + our extended
    abbreviation set (handles initials like 'S. Leeds' and titles like 'Rev.'/
    'M.A.'/'Vol.' that a naive period-split mangles). Falls back to a regex+abbrev
    heuristic if NLTK or the punkt model is unavailable."""
    text = re.sub(r"\s+", " ", text).strip()
    try:
        return _get_punkt().tokenize(text)
    except Exception:
        return _split_sentences_regex(text)


def _split_sentences_regex(text):
    """Fallback splitter: period-split, merging fragments that end in a known
    abbreviation so 'Mr. Bob' doesn't break into two 'sentences'."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in raw:
        if out:
            prev_words = out[-1].split()
            last = prev_words[-1].rstrip(".").lower() if prev_words else ""
            if last in _ABBREV:
                out[-1] = out[-1] + " " + s
                continue
        out.append(s)
    return out


def _is_prose(s, min_words):
    """True for a clean, complete prose sentence: >= min_words, contains a
    lowercase letter (filters ALL-CAPS headings like 'FOLK LORE.'), starts like
    text, and ends on terminal punctuation (filters cut fragments like '(Vol.')."""
    s = s.strip()
    if len(s.split()) < min_words:
        return False
    if not any(c.islower() for c in s):
        return False
    if not s.endswith(_SENT_END):
        return False
    first = s[0]
    if not (first.isalpha() or first in _SENT_OPEN):
        return False
    return True


def _prose_runs(sents, n, min_words):
    """Split a sentence list into maximal CONTIGUOUS runs of clean prose
    sentences (headings/fragments break a run), keeping only runs with >= n
    sentences. Sampling a window from a run guarantees n consecutive, real
    sentences with no heading junk between them."""
    runs, cur = [], []
    for s in sents:
        if _is_prose(s, min_words):
            cur.append(s)
        else:
            if len(cur) >= n:
                runs.append(cur)
            cur = []
    if len(cur) >= n:
        runs.append(cur)
    return runs


def _first_four(s):
    return " ".join(s.split()[:4])


def _iter_books_random(dataset_name, text_column, id_column, rng, max_books=None):
    """Yield (source_id, text) from the FULLY-DOWNLOADED (non-streaming) dataset
    in RANDOM book order, for source diversity. The first call downloads + caches
    the whole dataset (point HF_HOME/HF_DATASETS_CACHE at /data/prasann); every
    later call is a fast memory-mapped read, so iterating from a large, diverse
    pool of books is cheap. Random access via ds[i] never loads all texts in RAM."""
    import hashlib
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")  # cached after one-time DL
    order = list(range(len(ds)))
    rng.shuffle(order)
    if max_books:
        order = order[:max_books]
    print(f"[gutenberg] dataset has {len(ds)} books; scanning up to "
          f"{len(order)} in random order")
    for i in order:
        row = ds[int(i)]
        text = row.get(text_column)
        if not text:
            continue
        if id_column and row.get(id_column) is not None:
            sid = f"{dataset_name}/{row[id_column]}"
        else:
            h = hashlib.md5(text[:2000].encode("utf-8")).hexdigest()[:12]
            sid = f"{dataset_name}/{h}"
        yield sid, text


def run_gutenberg(args, rng):
    # Reuse the reorder generator's boilerplate stripper; load books in random
    # order from the cached (one-time-downloaded) dataset for diversity.
    from corpus_reasoning.data.generate_reorder_data import (
        strip_gutenberg_boilerplate, iter_local_books)

    n, k = args.n_sents, args.k_remove
    if args.local_dir:
        books = iter_local_books(args.local_dir)
    else:
        books = _iter_books_random(args.hf_dataset, args.text_column,
                                   args.id_column, rng, args.max_books_to_scan)
    want = args.num_eval + args.num_train
    mw = args.min_sentence_words
    out, scanned = [], 0
    for sid, text in books:
        scanned += 1
        sents = _tokenize_sentences(strip_gutenberg_boilerplate(text))
        runs = _prose_runs(sents, n, mw)
        if not runs:
            continue
        for _ in range(args.examples_per_book):
            if len(out) >= want:
                break
            run = runs[rng.randrange(len(runs))]
            start = rng.randint(0, len(run) - n)
            window = run[start:start + n]
            # Removable positions: first-four-words prefix unique within the
            # window (every window sentence is already clean prose >= mw words),
            # so gold is unambiguous.
            prefix_counts = {}
            for s in window:
                prefix_counts[_first_four(s).lower()] = \
                    prefix_counts.get(_first_four(s).lower(), 0) + 1
            cand = [i for i, s in enumerate(window)
                    if prefix_counts[_first_four(s).lower()] == 1]
            if len(cand) < k:
                continue
            removed = sorted(rng.sample(cand, k))
            removed_set = set(removed)
            kept = [window[i] for i in range(n) if i not in removed_set]
            out.append({
                "documents": [{"text": s} for s in window],
                "queries": [" ".join(kept)],
                "answers": [_first_four(window[i]) for i in removed],
                "gold_doc_indices": removed,
                "source": "absence_gutenberg",
                "meta": {
                    "format": "textdiff",
                    "book": sid,
                    "n": n,
                    "k": k,
                    "removed_sentences": [window[i] for i in removed],
                },
            })
        if len(out) >= want:
            break
    print(f"[gutenberg] scanned {scanned} books, produced {len(out)} examples "
          f"(N={n} sentences, K={k} removed)")
    _save_split(out, args, f"gutenberg_n{n}_k{k}")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_example(elements, p, rng, source, sep="\n", min_remove=1,
                  max_remove_frac=0.5):
    """Bernoulli-delete elements (each w.p. p), keep ≥min_remove and
    ≤max_remove_frac*N removed. Returns the unified absence example."""
    n = len(elements)
    cap = max(min_remove, int(n * max_remove_frac))
    removed = [i for i in range(n) if rng.random() < p]
    if len(removed) < min_remove:
        removed = rng.sample(range(n), min_remove)
    if len(removed) > cap:
        removed = sorted(rng.sample(removed, cap))
    removed_set = set(removed)
    kept = [elements[i] for i in range(n) if i not in removed_set]
    return {
        "documents": [{"text": e} for e in elements],
        "queries": ["Second version:\n\n" + sep.join(kept)],
        "answers": [],
        "gold_doc_indices": sorted(removed_set),
        "source": source,
    }


def run_from_jsonl(args, rng):
    src = load_jsonl(args.from_jsonl)
    rng.shuffle(src)
    n = len(src[0]["documents"])
    out = [build_example([d["text"] for d in ex["documents"]], args.p, rng,
                         f"absence_{args.src_tag or 'pubmed'}")
           for ex in src[: args.num_eval + args.num_train]]
    pt = str(args.p).replace(".", "")
    _save_split(out, args, f"{args.src_tag or 'pubmed'}_n{n}_p{pt}")


def run_numerical(args, rng):
    out = []
    for _ in range(args.num_train + args.num_eval):
        n = args.num_docs
        start = rng.randint(0, 10_000)
        seq = [start + i * args.step for i in range(n)]
        if args.order == "random":
            rng.shuffle(seq)
        out.append(build_example([str(v) for v in seq], args.p, rng,
                                 "absence_numerical", sep=", "))
    _save_split(out, args, f"numerical_n{args.num_docs}_s{args.step}")


def _split_elements(text, config):
    """Split an original_context into the element units the benchmark indexes.
    All three official configs (poetry lines, numbers, PR diff lines) are
    newline-separated, and omitted_index is the raw line index — so we keep
    every line (incl. blanks) to align exactly with omitted_index."""
    return text.split("\n")


def run_official(args, rng):
    from datasets import load_dataset
    CONFIGS = ["poetry", "numerical", "github_prs"]
    for config in CONFIGS:
        ds = load_dataset("harveyfin/AbsenceBench", config)
        split = list(ds.keys())[0]
        rows = ds[split]
        sep = "\n"
        out, n_skip, n_realign = [], 0, 0
        for r in rows:
            elements = _split_elements(r["original_context"], config)
            omitted_txt = r.get("omitted_context") or []
            gold = list(r.get("omitted_index") or [])
            # Validate omitted_index against omitted_context; realign by string
            # match if the dataset's indices don't line up with our split.
            valid = (gold and all(0 <= g < len(elements) for g in gold)
                     and all(elements[g].strip() == t.strip()
                             for g, t in zip(gold, omitted_txt)))
            if not valid and omitted_txt:
                pos, used = [], set()
                for t in omitted_txt:
                    for i, e in enumerate(elements):
                        if i not in used and e.strip() == t.strip():
                            pos.append(i); used.add(i); break
                gold = pos
                n_realign += 1
            gold = sorted({g for g in gold if 0 <= g < len(elements)})
            if not gold:
                n_skip += 1
                continue
            kept = [e for i, e in enumerate(elements) if i not in set(gold)]
            out.append({
                "documents": [{"text": e} for e in elements],
                "queries": ["Second version:\n\n" + sep.join(kept)],
                "answers": [], "gold_doc_indices": gold,
                "source": f"absence_official_{config}",
            })
        if out:
            ex = out[0]
            print(f"[{config}] {len(out)} ex (skipped {n_skip}, realigned "
                  f"{n_realign}); sample n_elems={len(ex['documents'])} "
                  f"removed={ex['gold_doc_indices']}")
            _save_split(out, args, f"official_{config}", split_all_eval=True)


def _looks_numeric(s):
    head = s.strip()[:80].replace(",", " ").split()
    return bool(head) and sum(t.lstrip("-").isdigit() for t in head) > len(head) // 2


def _save_split(out, args, tag, split_all_eval=False):
    if split_all_eval:
        save_jsonl(f"{args.output_dir}/absence_eval_{tag}.jsonl", out)
        print(f"eval: {len(out)} -> data/absence_eval_{tag}.jsonl")
        return
    tr, ev = out[args.num_eval:], out[:args.num_eval]
    if args.num_train and tr:
        save_jsonl(f"{args.output_dir}/absence_train_{tag}.jsonl", tr)
        print(f"train: {len(tr)} -> data/absence_train_{tag}.jsonl")
    if ev:
        save_jsonl(f"{args.output_dir}/absence_eval_{tag}.jsonl", ev)
        print(f"eval: {len(ev)} -> data/absence_eval_{tag}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_jsonl", default="",
                    help="derive elements from a unified JSONL's documents")
    ap.add_argument("--numerical", action="store_true")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--gutenberg", action="store_true",
                    help="Gutenberg text-diff variant (Version A/B prose, "
                         "first-four-words target, no doc IDs).")
    ap.add_argument("--n-sents", type=int, default=50,
                    help="gutenberg: sentences per segment (N).")
    ap.add_argument("--k-remove", type=int, default=3,
                    help="gutenberg: sentences removed in Version B (K).")
    ap.add_argument("--min-sentence-words", type=int, default=4,
                    help="gutenberg: a window sentence must have >= this many "
                         "words to count as clean prose (filters headings/"
                         "fragments). Must be >= 4 so first-four-words gold is valid.")
    ap.add_argument("--examples-per-book", type=int, default=1)
    ap.add_argument("--hf-dataset", default="sedthh/gutenberg_english")
    ap.add_argument("--text-column", default="TEXT")
    ap.add_argument("--id-column", default=None)
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--max-books-to-scan", type=int, default=2000)
    ap.add_argument("--official-split", default="")
    ap.add_argument("--p", type=float, default=0.1, help="deletion probability")
    ap.add_argument("--num-docs", type=int, default=100, help="numerical: seq length")
    ap.add_argument("--step", type=int, default=3, help="numerical: step")
    ap.add_argument("--order", choices=["asc", "random"], default="asc")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--src-tag", default="")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.from_jsonl:
        run_from_jsonl(args, rng)
    elif args.numerical:
        run_numerical(args, rng)
    elif args.official:
        run_official(args, rng)
    elif args.gutenberg:
        run_gutenberg(args, rng)
    else:
        ap.error("choose one of --from / --numerical / --official / --gutenberg")


if __name__ == "__main__":
    main()
