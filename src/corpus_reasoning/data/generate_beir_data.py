"""Generate in-context retrieval data from BEIR datasets (SciFact, NFCorpus, …).

Direct reference to a popular benchmark, framed as our in-context retrieval task:
each query becomes one example whose `documents` are the gold passage(s) plus
BM25-mined hard distractors from the SAME corpus, scored with `--task retrieval`
(recall/precision/F1). Mirrors the OBLIQ pipeline exactly — it reuses OBLIQ's
`_build_examples` / `_assemble_example` — only the data loader differs.

SciFact = single-gold (≈ NQ analog); NFCorpus = ~38 gold/query (≈ HotpotQA
multi-gold analog).

Needs HuggingFace `datasets` + pyserini (Lucene BM25) → run via sbatch
(jobs/gen_beir.sh).

Usage:
    python scripts/data/generate_beir_data.py --dataset scifact \\
        --num-docs 50 --output-dir data
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.lib.bm25_local import LocalBM25Searcher
from corpus_reasoning.data.generate_obliq_data import _build_examples, _print_obliq_stats


def load_beir(name, split):
    """Return (corpus list[{_id,text}], queries list[{_id,text}], qrels dict).
    Title is folded into text. Only queries with >=1 positive qrel are kept."""
    from datasets import load_dataset
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries_ds = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")
    qsplit = split if split in qrels_ds else list(qrels_ds.keys())[0]

    corpus = []
    for r in corpus_ds:
        title = (r.get("title") or "").strip()
        text = r["text"]
        corpus.append({"_id": str(r["_id"]),
                       "text": f"{title}. {text}" if title else text})

    qrels = defaultdict(list)
    for r in qrels_ds[qsplit]:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].append(str(r["corpus-id"]))
    qrels = dict(qrels)

    qids = set(qrels)
    queries = [{"_id": str(r["_id"]), "text": r["text"]}
               for r in queries_ds if str(r["_id"]) in qids]
    return corpus, queries, qrels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="BEIR name, e.g. scifact / nfcorpus / scidocs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-docs", type=int, default=50,
                    help="documents per example (gold + BM25 distractors)")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--index-dir", default="data/.cache/beir/_indices")
    ap.add_argument("--bm25-threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading BEIR/{args.dataset} ({args.split}) ...")
    corpus, queries, qrels = load_beir(args.dataset, args.split)
    print(f"  corpus={len(corpus):,}  queries(with qrels)={len(queries)}  "
          f"qrels={len(qrels)}")

    corpus_by_id = {d["_id"]: d for d in corpus}
    searcher = LocalBM25Searcher(corpus, str(Path(args.index_dir) / args.dataset),
                                 threads=args.bm25_threads)
    search_k = max(args.num_docs * 3, args.num_docs + 200)
    hits = searcher.batch_search([q["text"] for q in queries], k=search_k,
                                 threads=args.bm25_threads)

    examples = _build_examples(
        queries, hits, qrels, {}, corpus_by_id, args.num_docs,
        f"beir_{args.dataset}", [(0, args.num_docs)], rng,
        desc=f"BEIR {args.dataset} {args.split}")

    path = (f"{args.output_dir}/beir_{args.dataset}_{args.split}"
            f"_k{args.num_docs}_{len(examples)}.jsonl")
    save_jsonl(path, examples)
    print_dataset_stats(examples, args.dataset, path)
    _print_obliq_stats(examples, f"BEIR {args.dataset}", path)


if __name__ == "__main__":
    main()
