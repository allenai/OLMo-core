"""Generate MS MARCO in-context retrieval TRAINING data with clean hard negatives.

This is the *training* counterpart to `generate_msmarco_trecdl_data.py` (which
builds small, densely-judged EVAL pools). Training needs scale (thousands of
queries), which the ~100-query TREC-DL judged sets cannot give. We instead use
SentenceTransformers' precomputed mining of the MS MARCO *train* split:

  sentence-transformers/msmarco-hard-negatives
    * msmarco-hard-negatives.jsonl.gz                  {qid, pos:[pid], neg:{sys:[pid]}}
    * cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz   scores[qid][pid] -> ce

Hard-negative SELECTION needs no model: the negatives were mined by BM25 + ~13
dense retrievers, i.e. they are exactly the passages most likely to be *unlabeled
positives* (false negatives). We remove those automatically with a **margin
filter on the precomputed CE scores**: a mined negative is kept only if its CE
score is at least `--ce-margin` below the gold's CE score (SBERT's own recipe,
margin=3).

For the `rerank` task we want a relevance score for EVERY document, not just the
gold + mined negatives the pickle covers. By default (`--score-all`) we therefore
run the cross-encoder (`--ce-model`, the SAME model the pickle came from, so the
scores share one scale) on any pool document missing a precomputed score — i.e.
the random fill. Every emitted document then carries a real `ce_scores` value, so
the full pool has a continuous relevance order and the output-top-k decision is
independent of generation. Scores are cached to `--ce-cache` ({qid:{pid:score}})
so reruns and the four pool sizes share work. Pass `--no-score-all` to skip the
model (only gold + hard get scores; random fill stays null).

Pool construction (per example, `--num-docs` total):
  * gold     = pos pids (qrel positives), text from the index.
  * hard     = a FIXED FRACTION (`--hard-frac`) of the non-gold slots, drawn from
               the CE-filtered mined negatives, hardest (highest surviving CE)
               first. The fraction is constant across pool sizes, so the hard
               count scales down automatically for small n (n=500 -> ~50 hard,
               n=50 -> ~5 hard) and is never starved (SBERT mines 50+/query).
  * random   = the remaining slots, sampled as random integer pids from the 8.8M
               passage collection. MS MARCO v1 pids are contiguous 0..8841822, so
               a random pid is a real index passage — byte-for-byte identical in
               format to the gold/hard pids (no syntactic giveaway). Collisions
               with gold/hard and the ~1e-6 chance of a true positive are
               rejected / negligible.

Text + random pids come from the prebuilt pyserini `msmarco-v1-passage` index
(no 8.8M-passage download). Query text comes from BeIR/msmarco (covers train).
Train / eval splits are disjoint BY QUERY and identical across pool sizes.

Needs pyserini + HF datasets + huggingface_hub -> run via sbatch
(jobs/gen_msmarco_trainhn.sh) on a high-mem node (the CE pickle is large).

Usage:
    python scripts/data/generate_msmarco_trainhn_data.py \\
        --num-docs 20 50 100 500 --hard-frac 0.1 \\
        --n-train 2000 --n-eval 500 --output-dir data
"""

import argparse
import gzip
import json
import pickle
import random
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.data.generate_msmarco_data import _passage_text

HF_REPO = "sentence-transformers/msmarco-hard-negatives"
HN_FILE = "msmarco-hard-negatives.jsonl.gz"
CE_FILE = "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
# MS MARCO v1 passage collection: contiguous pids 0..8841822 (8,841,823 docs).
MAX_PID = 8841822


class CrossEncoderScorer:
    """Minimal CE relevance scorer (transformers, no sentence-transformers dep).

    Same class as generate_beir_ce_data.py — duplicated to keep this data-gen
    script's import chain free of the BEIR/BM25 modules it otherwise never needs."""

    def __init__(self, model_name, batch_size=256, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        model = (AutoModelForSequenceClassification
                 .from_pretrained(model_name).to(self.device).eval())
        # Scoring is embarrassingly parallel over (query, passage) pairs: shard each
        # batch across every visible GPU with DataParallel and scale the per-step
        # batch so each GPU still gets `batch_size` rows.
        self.n_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        if self.n_gpus > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.bs = batch_size * max(self.n_gpus, 1)
        self.max_length = max_length
        print(f"  CE model {model_name} on {self.device} "
              f"({self.n_gpus} GPU(s), effective batch {self.bs})")

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


def _score_pool(qid, qtext, pool_docs, scorer, ce):
    """Return {pid: CE score} for every doc in `pool_docs` (list of (pid, text)).

    Pids already in the CE cache `ce[qid]` (golds + mined negs from the pickle,
    plus anything scored on a previous call/run) are reused; the rest are scored
    with `scorer` and written back to the cache. With everything scored there are
    no null scores, so the whole pool gets a continuous relevance order."""
    cmap = ce.setdefault(qid, {})
    missing = [(pid, t) for pid, t in pool_docs if pid not in cmap]
    if missing:
        scores = scorer.score([(qtext, t) for _, t in missing])
        for (pid, _), s in zip(missing, scores):
            cmap[pid] = float(s)
    return {pid: cmap.get(pid) for pid, _ in pool_docs}


def _pool_ce_scores(gold_docs, hard_docs, cmap):
    """pid -> cross-encoder score for the gold + hard docs of one pool.

    Hard negatives always have a CE score (the margin filter required one). A gold
    that lacks a mined CE score is a coverage gap, not evidence of irrelevance, so
    it is pinned at the top scored value (above every hard neg) rather than dropped
    to the bottom. Random-fill docs are intentionally absent -> left unscored."""
    scores, hard_vals = {}, []
    for pid, _ in hard_docs:
        if pid in cmap:
            scores[pid] = cmap[pid]
            hard_vals.append(cmap[pid])
    gold_known = [cmap[pid] for pid, _ in gold_docs if pid in cmap]
    if gold_known:
        top_fill = max(gold_known)
    elif hard_vals:
        top_fill = max(hard_vals) + 1.0    # keep unscored golds above the hards
    else:
        top_fill = 0.0
    for pid, _ in gold_docs:
        scores[pid] = cmap.get(pid, top_fill)
    return scores


def _assemble_example(query_text, gold_docs, hard_docs, random_docs, source, rng,
                      ce_by_pid=None):
    """Shuffle gold + hard + random, emit a unified-format retrieval example.

    `hard_neg_indices` records ONLY the CE-verified mined hard negatives; the
    random fill is left untagged (ordinary distractors). `ce_scores` carries the
    per-document cross-encoder score (gold + hard) aligned with `documents`, or
    None for the unscored random fill — the rerank task ranks by this (CE = truth)."""
    ce_by_pid = ce_by_pid or {}
    tagged = ([(d, "gold") for d in gold_docs]
              + [(d, "hard") for d in hard_docs]
              + [(d, "random") for d in random_docs])
    rng.shuffle(tagged)
    documents = [{"title": None, "text": t} for (_pid, t), _tag in tagged]
    gold_idx = [i for i, (_, tag) in enumerate(tagged) if tag == "gold"]
    hard_idx = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]
    ce_scores = [ce_by_pid.get(pid) for (pid, _t), _tag in tagged]
    return {
        "documents": documents,
        "queries": [query_text],
        "answers": [""],
        "gold_doc_indices": gold_idx,
        "hard_neg_indices": hard_idx,
        "ce_scores": ce_scores,
        "source": source,
    }


def _sample_random_texts(searcher, n, used_pids, rng, max_pid):
    """Sample `n` distinct real passages by random integer pid (text, not None)."""
    out, attempts, cap = [], 0, n * 50 + 200
    while len(out) < n and attempts < cap:
        attempts += 1
        pid = str(rng.randint(0, max_pid))
        if pid in used_pids:
            continue
        text = _passage_text(searcher, pid)
        if not text:
            continue
        used_pids.add(pid)
        out.append((pid, text))
    return out


def load_hard_negs_for(qids_wanted):
    """Stream the SBERT hard-neg jsonl, returning {qid:int -> (pos:list, neg:list)}
    for the wanted qids only (dedup neg across mining systems, keep order)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, HN_FILE, repo_type="dataset")
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            qid = int(row["qid"])
            if qids_wanted is not None and qid not in qids_wanted:
                continue
            negs, seen = [], set()
            for sys_negs in row.get("neg", {}).values():
                for pid in sys_negs:
                    if pid not in seen:
                        seen.add(pid)
                        negs.append(int(pid))
            out[qid] = ([int(p) for p in row.get("pos", [])], negs)
    return out


def all_qids():
    """First pass: cheap list of every qid in the hard-neg file."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, HN_FILE, repo_type="dataset")
    qids = []
    with gzip.open(path, "rt") as f:
        for line in f:
            qids.append(int(json.loads(line)["qid"]))
    return qids


def load_ce_scores(selected_qids):
    """Load the CE-score pickle and project to selected qids, then free the rest."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, CE_FILE, repo_type="dataset")
    print(f"  Loading CE-score pickle (large) and projecting to "
          f"{len(selected_qids)} queries ...")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    ce = {q: data[q] for q in selected_qids if q in data}
    del data
    print(f"  CE scores retained for {len(ce)}/{len(selected_qids)} queries")
    return ce


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-docs", type=int, nargs="+", default=[20, 50, 100, 500],
                    help="pool sizes to emit (one train+eval file each)")
    ap.add_argument("--num-docs-min", type=int, default=None,
                    help="if set with --num-docs-max, sample each example's pool "
                         "size uniformly in [min, max] (continuous context-length "
                         "spread, mirroring generate_msmarco_data.py --min/--max-docs); "
                         "emits a SINGLE train+eval file and overrides --num-docs.")
    ap.add_argument("--num-docs-max", type=int, default=None,
                    help="upper bound (inclusive) for the per-example pool size in "
                         "continuous mode.")
    ap.add_argument("--hard-frac", type=float, default=0.1,
                    help="fraction of non-gold slots filled with CE-verified hard "
                         "negatives (constant across pool sizes); rest = random")
    ap.add_argument("--ce-margin", type=float, default=3.0,
                    help="keep a mined negative only if its CE score is >= this "
                         "below the gold CE score (SBERT recipe; removes false negs)")
    ap.add_argument("--no-score-all", action="store_true",
                    help="skip running the CE model on random-fill docs; only gold "
                         "+ mined hard-negs (precomputed pickle) get scores, random "
                         "fill stays null (the rerank reference order then tails off "
                         "in display order).")
    ap.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="CE model for scoring the random fill; MUST match the "
                         "pickle's model so all scores share one scale (default does).")
    ap.add_argument("--ce-cache", default="data/.cache/msmarco_trainhn_ce.pkl",
                    help="persistent {qid:{pid:score}} CE cache so reruns and the "
                         "four pool sizes share scoring work.")
    ap.add_argument("--ce-batch-size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--candidate-mult", type=float, default=2.0,
                    help="examine this multiple of (n_train+n_eval) candidate "
                         "queries to absorb skips")
    ap.add_argument("--index", default="msmarco-v1-passage")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    range_mode = args.num_docs_min is not None and args.num_docs_max is not None
    if (args.num_docs_min is None) != (args.num_docs_max is None):
        ap.error("--num-docs-min and --num-docs-max must be set together")
    if range_mode and args.num_docs_min > args.num_docs_max:
        ap.error("--num-docs-min must be <= --num-docs-max")

    from pyserini.search.lucene import LuceneSearcher
    from datasets import load_dataset

    rng = random.Random(args.seed)
    print(f"Loading prebuilt index {args.index} ...")
    searcher = LuceneSearcher.from_prebuilt_index(args.index)

    n_need = args.n_train + args.n_eval
    n_candidates = int(n_need * args.candidate_mult)

    # ---- pick a shuffled candidate prefix of qids (pass 1: just the ids) ----
    print("Listing hard-neg qids ...")
    qids = all_qids()
    print(f"  {len(qids):,} qids in {HN_FILE}")
    rng.shuffle(qids)
    candidate_qids = qids[:n_candidates]

    # ---- query text (BeIR/msmarco covers train) ----
    print("Loading query text (BeIR/msmarco) ...")
    cand_str = {str(q) for q in candidate_qids}
    queries_ds = load_dataset("BeIR/msmarco", "queries")["queries"]
    qtext = {str(r["_id"]): r["text"] for r in queries_ds if str(r["_id"]) in cand_str}
    print(f"  text for {len(qtext)}/{len(candidate_qids)} candidate qids")

    # ---- hard negs for the candidate prefix (pass 2) ----
    print("Loading hard negatives for candidates ...")
    hn = load_hard_negs_for(set(candidate_qids))

    # ---- select queries that have >=1 gold WITH text in the index ----
    print("Validating candidates (gold text present) ...")
    selected = []          # list of (qid, query_text, gold_pids, gold_texts, neg_pids)
    for qid in candidate_qids:
        if len(selected) >= n_need:
            break
        sq = str(qid)
        if sq not in qtext or qid not in hn:
            continue
        pos, negs = hn[qid]
        gold = []
        for p in pos:
            t = _passage_text(searcher, str(p))
            if t:
                gold.append((str(p), t))
        if not gold:
            continue
        selected.append((qid, qtext[sq], pos, gold, negs))
    print(f"  selected {len(selected)}/{n_need} queries")
    if len(selected) < n_need:
        print(f"  WARNING: only {len(selected)} valid queries "
              f"(< {n_need}); raise --candidate-mult")

    sel_qids = [s[0] for s in selected]
    ce = load_ce_scores(set(sel_qids))

    # ---- CE margin filter -> clean, hardest-first negative pids per query ----
    print(f"Applying CE margin filter (margin={args.ce_margin}) ...")
    clean_hard = {}        # qid -> [neg pids (str), hardest surviving first]
    for qid, _qt, _pos, _gold, negs in selected:
        cmap = ce.get(qid, {})
        pos_scores = [cmap[p] for p in _pos if p in cmap]
        if pos_scores:
            thresh = max(pos_scores) - args.ce_margin
            kept = [(p, cmap[p]) for p in negs if p in cmap and cmap[p] < thresh]
        else:
            # No gold CE score -> can't margin-filter; treat as no clean hard negs.
            kept = []
        kept.sort(key=lambda x: -x[1])      # highest surviving CE = hardest
        clean_hard[qid] = [str(p) for p, _ in kept]
    avg_hard = (sum(len(v) for v in clean_hard.values()) / max(1, len(clean_hard)))
    print(f"  avg clean hard negs/query: {avg_hard:.1f}")

    # ---- CE scorer + persistent cache for full-pool scoring (random fill) ----
    score_all = not args.no_score_all
    cache_path = Path(args.ce_cache)
    scorer = None
    if score_all:
        if cache_path.exists():
            print(f"Loading CE cache {cache_path} ...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            for q, m in cached.items():
                ce.setdefault(q, {}).update(m)
            print(f"  merged {sum(len(m) for m in cached.values()):,} cached scores")
        print("Scoring full pools with the cross-encoder (random fill) ...")
        scorer = CrossEncoderScorer(args.ce_model, batch_size=args.ce_batch_size)

    train_sel = selected[:args.n_train]
    eval_sel = selected[args.n_train:args.n_train + args.n_eval]

    def build_one(qid, qt, gold, num_docs):
        """Assemble a single CE-graded rerank example with a `num_docs`-sized pool."""
        cmap = ce.get(qid, {})
        gold_docs = gold[:num_docs]
        need = num_docs - len(gold_docs)
        if need <= 0:
            if scorer is not None:
                ce_by_pid = _score_pool(qid, qt, gold_docs, scorer, ce)
            else:
                ce_by_pid = _pool_ce_scores(gold_docs, [], cmap)
            return _assemble_example(qt, gold_docs, [], [], "msmarco_trainhn", rng,
                                     ce_by_pid)
        n_hard = min(round(args.hard_frac * need), len(clean_hard[qid]))
        used = {p for p, _ in gold_docs}
        hard_docs = []
        for pid in clean_hard[qid]:
            if len(hard_docs) >= n_hard:
                break
            if pid in used:
                continue
            t = _passage_text(searcher, pid)
            if t:
                used.add(pid)
                hard_docs.append((pid, t))
        n_random = num_docs - len(gold_docs) - len(hard_docs)
        random_docs = _sample_random_texts(searcher, n_random, used, rng, MAX_PID)
        if scorer is not None:
            ce_by_pid = _score_pool(qid, qt, gold_docs + hard_docs + random_docs,
                                    scorer, ce)
        else:
            ce_by_pid = _pool_ce_scores(gold_docs, hard_docs, cmap)
        return _assemble_example(qt, gold_docs, hard_docs, random_docs,
                                 "msmarco_trainhn", rng, ce_by_pid)

    # ---- build train + eval files. Discrete mode: one file per pool size.
    #      Continuous mode: a single file whose per-example pool size is sampled
    #      uniformly in [num-docs-min, num-docs-max]. ----
    size_iter = [None] if range_mode else args.num_docs
    for num_docs in size_iter:
        for split_name, split in (("train", train_sel), ("eval", eval_sel)):
            examples = []
            for qid, qt, pos, gold, _negs in split:
                nd = (rng.randint(args.num_docs_min, args.num_docs_max)
                      if range_mode else num_docs)
                examples.append(build_one(qid, qt, gold, nd))
            ktag = (f"k{args.num_docs_min}-{args.num_docs_max}" if range_mode
                    else f"k{num_docs}")
            path = (f"{args.output_dir}/msmarco_trainhn_{split_name}"
                    f"_{ktag}_{len(examples)}.jsonl")
            save_jsonl(path, examples)
            print_dataset_stats(examples, f"msmarco_trainhn {split_name} {ktag}",
                                path)
            n_g = [len(e["gold_doc_indices"]) for e in examples]
            n_h = [len(e["hard_neg_indices"]) for e in examples]
            n_d = [len(e["documents"]) for e in examples]
            if n_g:
                print(f"  {ktag} {split_name}: pool/ex median "
                      f"{sorted(n_d)[len(n_d)//2]} (min {min(n_d)} max {max(n_d)}), "
                      f"golds/ex median {sorted(n_g)[len(n_g)//2]}, "
                      f"hard/ex median {sorted(n_h)[len(n_h)//2]}")

    if score_all:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(ce, f)
        print(f"Saved CE cache ({sum(len(m) for m in ce.values()):,} scores) "
              f"-> {cache_path}")


if __name__ == "__main__":
    main()
