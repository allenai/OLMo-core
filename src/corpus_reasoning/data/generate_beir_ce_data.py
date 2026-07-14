"""Generate BEIR in-context retrieval data with CE-cleaned hard negatives.

Same idea as `generate_msmarco_trainhn_data.py`, but for BEIR datasets that ship
NO precomputed cross-encoder scores (e.g. FiQA). Sparsely-judged BEIR sets like
FiQA (~2.6 gold/query) have many *unlabeled* positives, so plain BM25 distractors
(generate_beir_data.py) are riddled with false negatives. Here we remove them:

  1. BM25-mine top-`--bm25-topk` candidates from the dataset's own corpus.
  2. Score (query, candidate) and (query, gold) with a cross-encoder
     (`--ce-model`, default cross-encoder/ms-marco-MiniLM-L-6-v2). Unlike the
     MS MARCO path (precomputed pickle, no model), FiQA has no scores shipped, so
     we run the CE — it's cheap (FiQA: ~648 q x ~100 cand on one GPU).
  3. **Margin filter**: keep a candidate as a hard negative only if its CE score
     is >= `--ce-margin` below the gold's CE score (SBERT recipe). Candidates that
     score close to the gold are likely unlabeled positives -> dropped.

Per example (`--num-docs` total docs):
  * gold   = qrel positive(s), folded title+text from the corpus.
  * hard   = constant `--hard-frac` of the non-gold slots, from the CE-surviving
             BM25 candidates, hardest (highest surviving CE) first.
  * random = the rest, sampled as random REAL corpus docs (same {title,text}
             source as gold/hard -> format-identical, no stylistic cue). For a
             |corpus|~57K, ~2.6-gold dataset the chance a random doc is an
             unlabeled positive is ~5e-5/doc (negligible).

Eval-only (one file per pool size; BEIR test split). Needs HF datasets + pyserini
(BM25) + transformers + a GPU (CE) -> run via sbatch (jobs/gen_beir_ce.sh).

Usage:
    python scripts/data/generate_beir_ce_data.py --dataset fiqa \\
        --num-docs 20 50 100 --hard-frac 0.1 --ce-margin 3.0 --output-dir data
"""

import argparse
import random
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.lib.bm25_local import LocalBM25Searcher
from corpus_reasoning.data.generate_beir_data import load_beir


def _assemble_example(query_text, gold_docs, hard_docs, random_docs, source, rng):
    """Shuffle gold + hard + random into a unified retrieval example.
    `hard_neg_indices` tags only the CE-verified hard negatives."""
    tagged = ([(d, "gold") for d in gold_docs]
              + [(d, "hard") for d in hard_docs]
              + [(d, "random") for d in random_docs])
    rng.shuffle(tagged)
    documents = [{"title": None, "text": d["text"]} for d, _ in tagged]
    gold_idx = [i for i, (_, t) in enumerate(tagged) if t == "gold"]
    hard_idx = [i for i, (_, t) in enumerate(tagged) if t == "hard"]
    return {
        "documents": documents,
        "queries": [query_text],
        "answers": [""],
        "gold_doc_indices": gold_idx,
        "hard_neg_indices": hard_idx,
        "source": source,
    }


class CrossEncoderScorer:
    """Minimal CE relevance scorer (transformers, no sentence-transformers dep)."""

    def __init__(self, model_name, batch_size=128, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(model_name).to(self.device).eval())
        self.bs = batch_size
        self.max_length = max_length
        print(f"  CE model {model_name} on {self.device}")

    def score(self, pairs):
        """pairs: list[(query, passage)] -> list[float] relevance logits."""
        out = []
        for i in range(0, len(pairs), self.bs):
            chunk = pairs[i:i + self.bs]
            inp = self.tok([p[0] for p in chunk], [p[1] for p in chunk],
                           padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inp).logits.squeeze(-1)
            out.extend(logits.float().cpu().tolist())
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fiqa", help="BEIR name (e.g. fiqa)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-docs", type=int, nargs="+", default=[20, 50, 100],
                    help="pool sizes to emit (one file each)")
    ap.add_argument("--hard-frac", type=float, default=0.1,
                    help="fraction of non-gold slots filled with CE-verified hard "
                         "negatives (constant across pool sizes); rest = random")
    ap.add_argument("--ce-margin", type=float, default=3.0,
                    help="keep a candidate only if its CE score is >= this below "
                         "the gold CE score (drops likely unlabeled positives)")
    ap.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--bm25-topk", type=int, default=100,
                    help="BM25 candidates per query to CE-score")
    ap.add_argument("--ce-batch-size", type=int, default=128)
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
    hits = searcher.batch_search([q["text"] for q in queries],
                                 k=args.bm25_topk + 50, threads=args.bm25_threads)

    scorer = CrossEncoderScorer(args.ce_model, batch_size=args.ce_batch_size)

    # ---- per query: CE-score golds + BM25 candidates, margin-filter ----
    print(f"CE-scoring + margin filter (margin={args.ce_margin}) ...")
    per_query = []     # (qtext, gold_docs[list dict], clean_hard[list dict hardest-first])
    n_dropped = 0
    for q, qhits in zip(queries, hits):
        qid = str(q["_id"])
        gold_ids = [g for g in qrels.get(qid, []) if g in corpus_by_id]
        if not gold_ids:
            continue
        gold_set = set(gold_ids)
        cand_ids = [h["docid"] for h in qhits
                    if h["docid"] not in gold_set and h["docid"] in corpus_by_id]
        cand_ids = cand_ids[:args.bm25_topk]
        gold_docs = [corpus_by_id[g] for g in gold_ids]
        if not cand_ids:
            per_query.append((q["text"], gold_docs, []))
            continue
        gold_scores = scorer.score([(q["text"], d["text"]) for d in gold_docs])
        cand_scores = scorer.score([(q["text"], corpus_by_id[c]["text"])
                                    for c in cand_ids])
        thresh = max(gold_scores) - args.ce_margin
        kept = [(c, s) for c, s in zip(cand_ids, cand_scores) if s < thresh]
        n_dropped += len(cand_ids) - len(kept)
        kept.sort(key=lambda x: -x[1])      # hardest surviving first
        clean_hard = [corpus_by_id[c] for c, _ in kept]
        per_query.append((q["text"], gold_docs, clean_hard))
    avg_hard = sum(len(p[2]) for p in per_query) / max(1, len(per_query))
    print(f"  queries kept={len(per_query)}  dropped {n_dropped} near-gold "
          f"candidates  avg clean hard/query={avg_hard:.1f}")

    # ---- build one file per pool size ----
    corpus_list = corpus      # for random fill (real corpus docs)
    for num_docs in args.num_docs:
        examples = []
        for qtext, gold_docs, clean_hard in per_query:
            gd = gold_docs[:num_docs]
            need = num_docs - len(gd)
            if need <= 0:
                examples.append(_assemble_example(qtext, gd, [], [],
                                                  f"beir_{args.dataset}_ce", rng))
                continue
            n_hard = min(round(args.hard_frac * need), len(clean_hard))
            used = {d["_id"] for d in gd}
            hard_docs = []
            for d in clean_hard:
                if len(hard_docs) >= n_hard:
                    break
                if d["_id"] in used:
                    continue
                used.add(d["_id"])
                hard_docs.append(d)
            n_random = num_docs - len(gd) - len(hard_docs)
            random_docs, attempts = [], 0
            cap = n_random * 50 + 200
            while len(random_docs) < n_random and attempts < cap:
                attempts += 1
                d = corpus_list[rng.randrange(len(corpus_list))]
                if d["_id"] in used:
                    continue
                used.add(d["_id"])
                random_docs.append(d)
            examples.append(_assemble_example(qtext, gd, hard_docs, random_docs,
                                              f"beir_{args.dataset}_ce", rng))
        path = (f"{args.output_dir}/beir_{args.dataset}_ce_{args.split}"
                f"_k{num_docs}_{len(examples)}.jsonl")
        save_jsonl(path, examples)
        print_dataset_stats(examples, f"{args.dataset}_ce k{num_docs}", path)
        n_g = [len(e["gold_doc_indices"]) for e in examples]
        n_h = [len(e["hard_neg_indices"]) for e in examples]
        print(f"  k{num_docs}: golds/ex median {sorted(n_g)[len(n_g)//2]}, "
              f"hard/ex median {sorted(n_h)[len(n_h)//2]}")


if __name__ == "__main__":
    main()
