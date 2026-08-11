"""Generate SYNTHETIC OBLIQ in-context retrieval training data via LLM query bootstrapping.

Motivation. The real OBLIQ-Bench ships only a few hundred queries per subset. At
`generate_obliq_data.py`'s train-frac the mix4 pool is ~1797 training examples
total, which mode-collapses the CTC Qwen3.5-4B: on the N=30 slice ~72% of
predictions are one of two canned strings (verified from gens). The *corpus*, by
contrast, has thousands of documents per subset -- the poverty is in QUERIES, not
documents. This script keeps the real corpus and SYNTHESIZES the scarce element,
the query, the way `generate_pubmed_contradiction_data.py` keeps real PubMed
sentences and synthesizes only the perturbed contradiction sentence.

Few-shot from the real eval. The generator is seeded with a pool of real
(document -> query) pairs harvested from the held-out OBLIQ eval file
(`--exemplars`). Reading those pairs shows the true query distribution, which is
NOT "find this one document": real OBLIQ queries DESCRIBE A PROPERTY / THEME /
STYLE and have MULTIPLE golds (writing 9/30, twitter 4/500, wildchat 4/50; only
congress is single-gold). Each generation prompt shows `--num-exemplars` real
pairs sampled from that pool, so generated queries match the real style and
granularity.

Per seed document D (sampled from the real corpus):
  1. An LLM (default Qwen3-8B via a local vLLM OpenAI endpoint), few-shot-primed
     with real (doc->query) pairs from the SAME subset, writes a NEW query Q
     describing a distinctive property/theme/style that D exhibits.
       - writing (analogues): Q is an original passage in D's STYLE/VOICE.
       - twitter / wildchat (descriptive): Q is a "Find posts/conversations
         where <theme/stance/pattern>" description D exhibits.
       - congress (tip-of-tongue): Q is a vague recollection of a moment in D.
  2. Oblique backstop: reject Q whose word-Jaccard with D exceeds --max-overlap,
     so a keyword-match shortcut can't separate the golds (matches real OBLIQ,
     whose queries are deliberately non-lexical).
  3. Hard-negative mining: BM25-search Q against the subset corpus; the top hits
     (minus D) are candidate documents.
  4. MULTI-GOLD judge (the critical step): an LLM judges each top candidate ->
     does it ALSO satisfy Q (as D does)? A YES becomes an ADDITIONAL GOLD (real
     OBLIQ is multi-gold); a NO becomes a HARD DISTRACTOR. This both matches the
     eval distribution AND removes false negatives in one pass -- the same
     eval-data-leak bug class that hit outlier/contradiction, resolved by
     labeling rather than dropping.
  5. Assemble a unified-format example (D + judged-YES = golds, judged-NO +
     deeper BM25 = distractors), shuffle, emit -- identical downstream path to
     `generate_obliq_data.py` (`--task retrieval`).

TRAIN ONLY. The held-out eval stays the real `obliq_mix4_test_488.jsonl` -- we do
NOT synthesize eval, which would confound the capability question with generator
bias. Seed docs (and, defensively, the exemplar source docs) are held disjoint
from the real test golds via `--exclude-test-golds`, so a synthetic train query
can't reuse a test-gold document.

Usage (needs a running vLLM endpoint; see
`debug/obliq_synthetic/gen_obliq_synth_pilot.sbatch`):

    python generate_obliq_synthetic_data.py --subset twitter \\
        --model Qwen/Qwen3-8B --base-url http://horton:8101/v1 \\
        --exemplars /path/to/obliq_mix4_test_488.jsonl \\
        --num-seed-docs 2000 --num-docs 5 --num-exemplars 4 --judge-topk 12 \\
        --max-concurrent 32 --output-dir /data/prasann/obliq_synth
"""

import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl, save_jsonl, print_dataset_stats

# Reuse the real generator's OBLIQ loading + framing + example assembly so the
# synthetic examples land on exactly the same downstream distribution.
from corpus_reasoning.data.generate_obliq_data import (
    SUBSET_PATHS,
    SUBSET_QUERY_PREFIX,
    _assemble_example,
    _clean_text,
    _load_subset,
)


# ─────────────────────────── query-generation prompts ──────────────────────
# Per-subset INSTRUCTION (the few-shot block + target doc are appended by
# `_build_gen_prompt`). Instructions describe the real OBLIQ intent: a query
# names a PROPERTY/THEME/STYLE, and documents that share it are the golds.

GEN_INSTRUCTIONS = {
    "writing": (
        "OBLIQ 'writing' retrieval: a query is a passage, and the goal is to "
        "retrieve documents written in a SIMILAR STYLE and VOICE -- tone, "
        "register, sentence rhythm, diction -- regardless of topic. Below are "
        "real examples of a source document and the style-query written for it. "
        "Then write a NEW query passage (4-7 sentences) in a clearly similar "
        "style and voice to the TARGET document, but on a COMPLETELY DIFFERENT "
        "subject (no shared names, places, or facts). Output ONLY the passage."
    ),
    "twitter": (
        "OBLIQ 'twitter' retrieval: a query DESCRIBES a theme, stance, or "
        "implicit assertion, and the goal is to retrieve the short posts that "
        "exhibit it. Below are real examples of a matching post and the "
        "theme-query written for it. Then write a NEW query (1-3 sentences, "
        "beginning 'Find posts where ...' or 'Find tweets where ...') that "
        "describes a DISTINCTIVE theme, stance, or implicit assertion the "
        "TARGET post exhibits -- specific enough to pick out posts like it, but "
        "phrased as a general property, not a quote. Output ONLY the query."
    ),
    "wildchat": (
        "OBLIQ 'wildchat' retrieval: a query DESCRIBES a distinctive pattern or "
        "behavior in a chat conversation, and the goal is to retrieve the "
        "conversations that exhibit it. Below are real examples of a matching "
        "conversation and the query written for it. Then write a NEW query (1-3 "
        "sentences, beginning 'Find conversations where ...') describing a "
        "DISTINCTIVE pattern the TARGET conversation exhibits -- specific enough "
        "to pick out conversations like it, phrased as a general property, not a "
        "quote. Output ONLY the query."
    ),
    "congress": (
        "OBLIQ 'congress' tip-of-tongue retrieval: a query is a vague "
        "first-person recollection of a hearing moment, and the goal is to "
        "retrieve the matching hearing record. Below are real examples of a "
        "hearing record and the recollection written for it. Then write a NEW "
        "vague recollection (2-3 sentences) of a distinctive moment in the "
        "TARGET record, the way someone half-remembering it would describe it -- "
        "gist plus a couple of details, without quoting. Output ONLY the "
        "recollection."
    ),
}

# Per-subset doc label used in the few-shot block and target.
DOC_LABEL = {"writing": "PASSAGE", "twitter": "POST", "wildchat": "CONVERSATION",
             "congress": "RECORD", "math": "PROBLEM"}

# What "the candidate ALSO satisfies the query" means, per subset -- given to the
# multi-gold judge so it applies the right notion of relevance.
JUDGE_INTENT = {
    "writing": "is written in a similarly matching STYLE and VOICE to the query "
               "passage (topic is irrelevant; judge style only)",
    "twitter": "exhibits the theme, stance, or implicit assertion the query "
               "describes (matching its specific intent, not just the topic)",
    "wildchat": "exhibits the pattern or behavior the query describes (matching "
                "its specific intent, not just the topic)",
    "congress": "is a hearing record the recollection could plausibly be "
                "describing",
    "math": "would be solved with the same core approach the query targets",
}

JUDGE_PROMPT = (
    "A QUERY was written to retrieve documents with a shared property. A GOLD "
    "document is a known match. Decide whether a CANDIDATE document {intent} -- "
    "i.e. whether the CANDIDATE is ALSO a correct match for the QUERY, as the "
    "GOLD is.\n\n"
    "Answer with exactly one word: YES if the CANDIDATE is also a correct match, "
    "or NO if it is not.\n\n"
    "QUERY:\n{query}\n\nGOLD:\n{gold}\n\nCANDIDATE:\n{cand}\n\nAnswer (YES/NO):"
)

# Per-subset oblique-backstop cap. Style-matching (writing/math) legitimately
# reuses rhetorical scaffolding and function words, so its lexical overlap runs
# higher than the descriptive subsets -- a global 0.35 would wrongly reject good
# style queries. Overridable with --max-overlap.
SUBSET_MAX_OVERLAP = {"writing": 0.55, "math": 0.50,
                      "twitter": 0.35, "wildchat": 0.35, "congress": 0.40}

_WORD = re.compile(r"[a-z0-9]+")


def word_jaccard(a, b):
    """Word-level Jaccard overlap of two strings (lowercased tokens)."""
    sa = set(_WORD.findall(a.lower()))
    sb = set(_WORD.findall(b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def clean_response(text):
    """Strip Qwen3 <think> blocks, surrounding quotes, and echoed labels."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text).strip()
    text = re.sub(r"^\s*(passage|query|post|tweet|conversation|record|problem|"
                  r"recollection|document|answer)\s*:\s*", "", text,
                  flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text.strip()


# ─────────────────────────── lightweight BM25 miner ────────────────────────
# Self-contained pure-Python Okapi BM25 (k1=0.9/b=0.4 to match the real
# generator's pyserini LocalBM25Searcher). We do NOT use pyserini here: its
# Anserini JVM init hangs on some jsteinhardt compute nodes (mooney) and
# core-dumps on others (cubbins), while only needing hard-negative *mining*
# quality that a plain in-memory inverted index delivers fine. Same
# search/batch_search interface the build loop expects.

_TOK = re.compile(r"[a-z0-9]+")


def _tok(s):
    return _TOK.findall(s.lower())


class SimpleBM25:
    def __init__(self, corpus, k1=0.9, b=0.4, max_df_frac=0.1):
        self.ids = [str(d["_id"]) for d in corpus]
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.doclen = [0] * self.N
        self.postings = defaultdict(list)  # term -> [(doc_idx, tf), ...]
        for i, d in enumerate(corpus):
            toks = _tok(d.get("text", ""))
            self.doclen[i] = len(toks)
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, f in tf.items():
                self.postings[t].append((i, f))
        self.avgdl = (sum(self.doclen) / self.N) if self.N else 0.0
        self.idf = {t: math.log(1 + (self.N - len(pl) + 0.5) / (len(pl) + 0.5))
                    for t, pl in self.postings.items()}
        # High-df terms (in > max_df_frac of docs) have ~zero idf but huge
        # postings lists that dominate scoring cost -- skip them at query time.
        # Big win on long-doc corpora (congress/wildchat), negligible ranking
        # change. Floor so tiny corpora don't prune everything.
        self.max_df = max(50, int(max_df_frac * self.N)) if self.N else 0

    def search(self, query, k=100):
        scores = defaultdict(float)
        # score rarer terms first so a common stopword can't blow up the loop.
        terms = sorted(set(_tok(query)),
                       key=lambda t: len(self.postings.get(t, ())))
        for t in terms:
            pl = self.postings.get(t)
            if not pl or len(pl) > self.max_df:
                continue
            idf = self.idf[t]
            for i, f in pl:
                dl = self.doclen[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [{"docid": self.ids[i], "score": sc} for i, sc in ranked]

    def batch_search(self, queries, k=100, threads=None):
        return [self.search(q, k) for q in queries]


# ───────────────────────────── few-shot exemplars ──────────────────────────

def load_exemplars(paths, subset, pool_size, rng, exclude_texts=None):
    """Harvest real (doc_text, query_body) pairs for `subset` from the eval
    file(s). The emit-time SUBSET_QUERY_PREFIX is stripped so the exemplar query
    is the bare body (what the generator is asked to produce)."""
    prefix = SUBSET_QUERY_PREFIX.get(subset, "")
    exclude_texts = exclude_texts or set()
    pairs = []
    for p in paths:
        for ex in load_jsonl(p):
            if subset not in ex.get("source", ""):
                continue
            gi = ex.get("gold_doc_indices", [])
            if not gi or not ex.get("queries"):
                continue
            q = ex["queries"][0]
            body = q[len(prefix):] if prefix and q.startswith(prefix) else q
            gold = ex["documents"][gi[0]]["text"]
            if _clean_text(gold) in exclude_texts:
                continue
            pairs.append((_clean_text(gold), body.strip()))
    rng.shuffle(pairs)
    return pairs[:pool_size]


def _build_gen_prompt(subset, doc_text, exemplars, rng, num_exemplars,
                      ex_doc_chars=600, ex_q_chars=500, doc_chars=4000):
    """Instruction + `num_exemplars` sampled real (doc->query) pairs + target."""
    label = DOC_LABEL.get(subset, "DOCUMENT")
    shots = rng.sample(exemplars, min(num_exemplars, len(exemplars))) if exemplars else []
    blocks = []
    for i, (ed, eq) in enumerate(shots, 1):
        blocks.append(f"EXAMPLE {i}\n[{label}]\n{ed[:ex_doc_chars]}\n"
                      f"[QUERY]\n{eq[:ex_q_chars]}")
    fewshot = ("\n\n".join(blocks) + "\n\n") if blocks else ""
    return (f"{GEN_INSTRUCTIONS[subset]}\n\n{fewshot}"
            f"Now write the query for this document.\n"
            f"[{label}]\n{doc_text[:doc_chars]}\n[QUERY]\n")


# ─────────────────────────────── LLM plumbing ──────────────────────────────

def _make_client(max_concurrent, base_url):
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
    return ParallelResponsesClient(
        max_concurrent=max_concurrent, use_cache=True, local_base_url=base_url,
    )


# Turn OFF Qwen3 chain-of-thought so generation is a single clean query.
_NOTHINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


# Per-variant diversity nudges for multi-query-per-seed. Variant 0 is the plain
# prompt; later variants ask for a DIFFERENT facet so the N queries off one seed
# doc are genuinely distinct (not near-duplicates).
_DIVERSITY_NUDGES = [
    "",
    " Focus on a DIFFERENT distinctive facet than the most obvious one.",
    " Focus on yet another SEPARATE angle, distinct from the obvious reading and "
    "from any secondary theme.",
    " Emphasize a further distinct aspect not covered by an obvious reading.",
]


def generate_queries(client, subset, docs, exemplars, model, max_overlap,
                     num_exemplars, rng, queries_per_seed=1, intra_dup=0.6):
    """Generate `queries_per_seed` queries per seed doc (few-shot-primed).

    Returns a FLAT list of (doc, query) pairs. Multiple queries off one doc use
    per-variant diversity nudges + an intra-seed word-Jaccard dedup (drop a
    variant too similar to one already kept for the same doc), so the N queries
    are genuinely different angles rather than paraphrases."""
    prompts, owners = [], []
    for di, d in enumerate(docs):
        for v in range(queries_per_seed):
            base = _build_gen_prompt(subset, d["text"], exemplars, rng, num_exemplars)
            prompts.append(base.rstrip("\n") + _DIVERSITY_NUDGES[v % len(_DIVERSITY_NUDGES)]
                           + "\n[QUERY]\n" if v else base)
            owners.append(di)
    print(f"[{subset}] generating {len(prompts)} queries with {model} "
          f"({num_exemplars}-shot, {queries_per_seed}/seed)...")
    responses = client.run(model=model, prompts=prompts,
                           temperature=0.85, max_output_tokens=400, **_NOTHINK)
    pairs, kept_per_doc = [], {}
    n_fail = n_overlap = n_dup = 0
    for owner, r in zip(owners, responses):
        d = docs[owner]
        if not r.get("success", True):
            n_fail += 1; continue
        q = clean_response(r.get("response") or "")
        if len(q) < 20:
            n_fail += 1
        elif word_jaccard(q, d["text"]) > max_overlap:
            n_overlap += 1
        elif any(word_jaccard(q, pq) > intra_dup for pq in kept_per_doc.get(owner, [])):
            n_dup += 1
        else:
            pairs.append((d, q))
            kept_per_doc.setdefault(owner, []).append(q)
    print(f"[{subset}]   {len(pairs)} usable query pairs, {n_fail} failed/short, "
          f"{n_overlap} overlap>{max_overlap}, {n_dup} intra-seed dups")
    return pairs


def judge_candidates(client, subset, jobs, model):
    """jobs: list of (query, gold_text, cand_text). Returns list[bool]: True if
    the candidate ALSO matches the query (-> label it a GOLD); False -> hard
    distractor."""
    if not jobs:
        return []
    intent = JUDGE_INTENT[subset]
    prompts = [JUDGE_PROMPT.format(intent=intent, query=q[:2500],
                                   gold=g[:2500], cand=c[:2500])
               for q, g, c in jobs]
    print(f"[{subset}] multi-gold judge on {len(prompts)} candidates...")
    responses = client.run(model=model, prompts=prompts,
                           temperature=0.0, max_output_tokens=8, **_NOTHINK)
    flags = []
    for r in responses:
        verdict = clean_response(r.get("response") or "").strip().upper()
        # Fail-closed toward DISTRACTOR: only a clear YES promotes to gold, so a
        # flaky judge can't inflate the gold set with false positives.
        flags.append(r.get("success", True) and verdict.startswith("YES"))
    print(f"[{subset}]   {sum(flags)}/{len(flags)} candidates labeled GOLD "
          f"(rest hard distractors)")
    return flags


# ─────────────────────────────── build loop ────────────────────────────────

def build_examples(client, subset, pairs, searcher, corpus_by_id,
                   num_docs, judge_topk, max_gold, search_k, model, rng,
                   query_prefix):
    """One example per (seed_doc, query) pair: BM25-mine candidates, judge the
    top `judge_topk` (YES->gold, NO->distractor), and assemble with D +
    judged-golds (capped at max_gold) plus distractors filled to num_docs."""
    valid = [(d, q) for d, q in pairs if q]
    print(f"[{subset}] BM25-mining candidates for {len(valid)} queries...")
    hits_batch = searcher.batch_search([q for _, q in valid], k=search_k)

    pending, judge_jobs, judge_owner = [], [], []
    n_skip_pool = 0
    for (d, q), hits in zip(valid, hits_batch):
        gold_id = str(d["_id"])
        cand = [corpus_by_id[h["docid"]] for h in hits
                if h["docid"] != gold_id and h["docid"] in corpus_by_id]
        if len(cand) < num_docs - 1:
            n_skip_pool += 1
            continue
        idx = len(pending)
        pending.append([d, q, cand])
        for c in cand[:judge_topk]:
            judge_jobs.append((q, d["text"], c["text"]))
            judge_owner.append(idx)

    flags = judge_candidates(client, subset, judge_jobs, model)
    # gold/distractor split of the JUDGED top-K per pending example.
    judged_gold = [[] for _ in pending]
    judged_dist = [[] for _ in pending]
    ji = 0
    for owner, flag in zip(judge_owner, flags):
        _, _, cand_text = judge_jobs[ji]; ji += 1
        (judged_gold[owner] if flag else judged_dist[owner]).append(cand_text)

    examples = []
    n_skip_after = 0
    gold_hist = {}
    for (d, q, cand), g_texts, d_texts in zip(pending, judged_gold, judged_dist):
        gset, dset = set(g_texts), set(d_texts)
        # golds = seed D + judged-YES, capped so >=1 distractor slot remains.
        cap = max_gold if max_gold > 0 else num_docs - 1
        cap = min(cap, num_docs - 1)
        extra_golds = [c for c in cand if c["text"] in gset][:cap - 1]
        golds = [d] + extra_golds
        need_dist = num_docs - len(golds)
        # Distractors: judged-NO (hard) first, then UNJUDGED deeper BM25. Surplus
        # judged-YES docs (in gset but beyond the cap) are DROPPED entirely --
        # never reused as distractors, which would make them false negatives
        # (the gset exclusion below is what enforces that).
        gold_texts = {g["text"] for g in golds}
        hard = [c for c in cand if c["text"] in dset and c["text"] not in gold_texts]
        deep = [c for c in cand
                if c["text"] not in gold_texts and c["text"] not in dset
                and c["text"] not in gset]
        distractors = (hard + deep)[:need_dist]
        if len(distractors) < need_dist:
            n_skip_after += 1
            continue
        ex = _assemble_example(q, golds, distractors, f"obliq_synth_{subset}",
                               rng, query_prefix=query_prefix)
        examples.append(ex)
        gold_hist[len(golds)] = gold_hist.get(len(golds), 0) + 1
    print(f"[{subset}]   built {len(examples)} examples "
          f"({n_skip_pool} skipped: thin BM25 pool; {n_skip_after} skipped: "
          f"too few distractors); gold-count histogram: "
          f"{dict(sorted(gold_hist.items()))}")
    return examples


def load_eval_doc_texts(paths, subset, all_docs=True):
    """Cleaned doc texts appearing in the real eval file(s) for `subset`.

    all_docs=True (default): EVERY document in every eval example -- golds AND
    distractors. This is the clean-room set: removing all of these from the
    corpus guarantees no eval document can become a synthetic seed (gold) OR a
    synthetic distractor. all_docs=False keeps the old golds-only behavior.
    """
    texts = set()
    for p in paths:
        for ex in load_jsonl(p):
            if subset not in ex.get("source", ""):
                continue
            if all_docs:
                for d in ex["documents"]:
                    texts.add(_clean_text(d["text"]))
            else:
                for gi in ex.get("gold_doc_indices", []):
                    if 0 <= gi < len(ex["documents"]):
                        texts.add(_clean_text(ex["documents"][gi]["text"]))
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, choices=list(SUBSET_PATHS))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base-url", default=None,
                    help="local vLLM OpenAI endpoint, e.g. http://horton:8101/v1")
    ap.add_argument("--exemplars", nargs="*", default=[],
                    help="real eval JSONL(s) to draw few-shot (doc->query) pairs")
    ap.add_argument("--exemplar-pool", type=int, default=50,
                    help="max real pairs to keep in the few-shot pool")
    ap.add_argument("--num-exemplars", type=int, default=4,
                    help="few-shot pairs shown per generation prompt")
    ap.add_argument("--max-corpus-docs", type=int, default=0,
                    help="subsample the corpus to this many docs before BM25 "
                         "(0 = use all); bounds memory/time on huge corpora "
                         "e.g. wildchat 2.2GB")
    ap.add_argument("--num-seed-docs", type=int, default=2000)
    ap.add_argument("--queries-per-seed", type=int, default=1,
                    help="queries generated per seed doc (>1 uses diversity "
                         "nudges + intra-seed dedup for distinct angles)")
    ap.add_argument("--num-docs", type=int, default=5,
                    help="documents per example (golds + distractors)")
    ap.add_argument("--max-gold", type=int, default=0,
                    help="cap golds per example (0 = num_docs-1); always >=1 "
                         "distractor is kept")
    ap.add_argument("--judge-topk", type=int, default=12,
                    help="judge the top-K BM25 candidates for gold/distractor")
    ap.add_argument("--max-overlap", type=float, default=None,
                    help="reject a query whose word-Jaccard with its source doc "
                         "exceeds this (keeps queries oblique); default is "
                         "per-subset (SUBSET_MAX_OVERLAP)")
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--bm25-threads", type=int, default=8)
    ap.add_argument("--exclude-test-golds", nargs="*", default=[],
                    help="real test JSONL(s); their gold docs are excluded from "
                         "synthetic seeds (leakage guard)")
    ap.add_argument("--cache-dir", default="obliq_cache")
    ap.add_argument("--index-dir", default="obliq_bm25_index")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    corpus, _q, _qr, _ex = _load_subset(args.subset, args.cache_dir)
    n_raw = len(corpus)
    print(f"[{args.subset}] corpus: {n_raw} documents")

    # CLEAN-ROOM leakage guard: drop EVERY doc that appears in the eval file
    # (gold OR distractor) from the corpus entirely -- before BM25 and seeding --
    # so no eval document can be a synthetic seed (gold) or a synthetic
    # distractor. --exclude-test-golds / --exemplars supply the eval file(s).
    guard_files = args.exclude_test_golds or args.exemplars
    if guard_files:
        eval_texts = load_eval_doc_texts(guard_files, args.subset, all_docs=True)
        corpus = [d for d in corpus if _clean_text(d["text"]) not in eval_texts]
        removed = n_raw - len(corpus)
        print(f"[{args.subset}] CLEAN-ROOM: removed {removed} eval docs "
              f"({100*removed/max(1,n_raw):.1f}% of corpus) -> {len(corpus)} "
              f"train-pool docs")

    if args.max_corpus_docs and len(corpus) > args.max_corpus_docs:
        random.Random(args.seed).shuffle(corpus)
        corpus = corpus[:args.max_corpus_docs]
        print(f"[{args.subset}] subsampled train pool to {len(corpus)} docs "
              f"(--max-corpus-docs)")
    corpus_by_id = {str(d["_id"]): d for d in corpus}
    # eval docs are already gone from the corpus, so nothing more to exclude at
    # seed time.
    excluded_texts = set()

    exemplars = load_exemplars(args.exemplars, args.subset, args.exemplar_pool,
                               rng, exclude_texts=set())
    print(f"[{args.subset}] few-shot pool: {len(exemplars)} real (doc->query) "
          f"pairs; showing {args.num_exemplars}/prompt")
    if args.exemplars and not exemplars:
        print(f"[{args.subset}] WARNING: no exemplars found for this subset in "
              f"{args.exemplars} -- falling back to zero-shot instructions")

    print(f"[{args.subset}] building in-memory BM25 over {len(corpus)} docs...")
    searcher = SimpleBM25(corpus)

    seedable = [d for d in corpus
                if len(d["text"].split()) >= 20
                and _clean_text(d["text"]) not in excluded_texts]
    rng.shuffle(seedable)
    seed_docs = seedable[:args.num_seed_docs]
    print(f"[{args.subset}] seeding {len(seed_docs)} queries from {len(seedable)} "
          f"eligible train-pool docs "
          f"({100*len(seed_docs)/max(1,len(seedable)):.1f}% of pool, "
          f"{100*len(seed_docs)/max(1,n_raw):.1f}% of full corpus)")

    client = _make_client(args.max_concurrent, args.base_url)

    max_overlap = (args.max_overlap if args.max_overlap is not None
                   else SUBSET_MAX_OVERLAP.get(args.subset, 0.35))
    print(f"[{args.subset}] oblique-backstop max_overlap={max_overlap}")
    pairs = generate_queries(client, args.subset, seed_docs, exemplars,
                             args.model, max_overlap, args.num_exemplars, rng,
                             queries_per_seed=args.queries_per_seed)

    search_k = max(args.num_docs * 4, args.judge_topk + 60)
    examples = build_examples(
        client, args.subset, pairs, searcher, corpus_by_id,
        args.num_docs, args.judge_topk, args.max_gold, search_k, args.model, rng,
        query_prefix=SUBSET_QUERY_PREFIX.get(args.subset, ""),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"obliq_synth_{args.subset}_train_k{args.num_docs}"
                          f"_{len(examples)}.jsonl")
    save_jsonl(str(out_path), examples)
    print(f"[{args.subset}] wrote {len(examples)} examples -> {out_path}")
    print_dataset_stats(examples, f"obliq_synth_{args.subset}", str(out_path))


if __name__ == "__main__":
    main()
