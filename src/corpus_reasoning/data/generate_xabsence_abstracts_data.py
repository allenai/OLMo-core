"""Cross-corpus absence at ABSTRACT granularity (xabsence-ABSTRACTS): find the whole
PubMed abstracts with no paraphrase twin in the other corpus.

This is `generate_xabsence_data.py` (sentence-level xabsence) with ONE change: the unit
being matched is a whole PubMed abstract (~150-500 words), not a single sentence/claim.
Everything else — the task definition, the two-phase pipeline, the assembled schema, and
the scorer — is identical, so the existing `--task xabsence` / `_eval_absence` path
works completely unchanged on this data.

Two corpora A and B share one context. Almost every abstract in A has a low word-overlap
Qwen PARAPHRASE twin in B (same findings/claims, reworded, so it can't be found by
surface string-matching — it requires a semantic all-pairs comparison across the two
corpora, a chunked-vs-standard discriminator). Exactly k abstracts are UNMATCHED — no
twin anywhere in the other corpus — and the model must name their positions.

Why a dedicated script instead of an `--abstract` flag on the sentence version: the
harvesting source format differs (a `{aid, question, abstract}` pool file, not a unified
`documents`-schema JSONL), the paraphrase prompt needs abstract-scale instructions
("paraphrase this WHOLE abstract paragraph, not sentence-by-sentence"), the cleanup can't
truncate to `splitlines()[0]` the way the single-sentence cleaner does (an abstract IS
multiple sentences), and the token budgets are ~5-10x bigger. Sharing the schema/scorer
means duplicating only the parts that actually differ.

Schema (byte-identical to sentence xabsence, so `_eval_absence` needs no changes):
    {"documents": [{"text": <abstract>, "corpus": "A"|"B"}, ...],   # A block then B block
     "queries": [],
     "gold_doc_indices": [unmatched positions, 0-indexed],
     "num_pairs": P, "num_unmatched": k, "source": "xabsence_abstracts_<tag>"}

Two phases (paraphrasing is the only expensive step; the pool is built ONCE and reused
for any P/k/count):

  1. BUILD POOL (LLM): paraphrase every abstract in a PubMed abstracts pool file
     (`debug/pubmed_multiclaim/abstracts_*.jsonl`, one `{aid, question, abstract}` per
     line) via a live Qwen chat endpoint, keep only word-Jaccard <= --max-overlap twins.
        python -u src/corpus_reasoning/data/generate_xabsence_abstracts_data.py --build-pool \\
            --from debug/pubmed_multiclaim/abstracts_2k_train.jsonl \\
            --pool-out debug/xabsence_abstracts/pool_train.jsonl \\
            --model Qwen/Qwen3-8B --base-url http://horton:8101/v1 \\
            --max-overlap 0.5 --max-concurrent 32
     (--stub-paraphrase builds a no-LLM pool for pipeline testing.)

  2. ASSEMBLE (cheap): sample P matched pairs + k unmatched per example, from the
     TRAIN pool for the train split and the EVAL pool for the eval split (kept separate
     since they're sourced from disjoint abstract sets already, so no abstract leaks
     across the split boundary).
        python src/corpus_reasoning/data/generate_xabsence_abstracts_data.py \\
            --train-pool debug/xabsence_abstracts/pool_train.jsonl \\
            --eval-pool debug/xabsence_abstracts/pool_eval.jsonl \\
            --num-pairs 15 --num-unmatched 3 --num-train 2000 --num-eval 520 \\
            --src-tag pubmed --output-dir debug/xabsence_abstracts
"""

import argparse
import json
import random
import re


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def harvest_abstracts(from_path, limit):
    """Collect unique abstract texts from a PubMed abstracts pool file
    (``{"aid": ..., "question": ..., "abstract": ...}`` per line, e.g.
    ``debug/pubmed_multiclaim/abstracts_*.jsonl``)."""
    seen, out = set(), []
    for ex in load_jsonl(from_path):
        aid = ex.get("aid")
        text = (ex.get("abstract") or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append({"aid": aid, "text": text})
            if len(out) >= limit:
                break
    return out


def word_jaccard(a, b):
    """Token-set Jaccard overlap of two texts (lowercased alnum tokens). Same
    definition as generate_pubmed_contradiction_data.word_jaccard / the sentence
    xabsence generator, just applied to whole abstracts instead of one sentence."""
    wa = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    wb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not wa or not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def clean_abstract_response(text, min_words=40):
    """Clean an LLM paraphrase of a WHOLE abstract. Unlike the single-sentence
    cleaner (generate_pubmed_contradiction_data.clean_response), this must NOT
    truncate to the first line — a paraphrased abstract is legitimately several
    sentences and some models line-break between them. Instead: strip a leading
    label/preamble line if present, join everything into one paragraph, and
    reject responses that are implausibly short for an abstract."""
    if not text:
        return None
    text = text.strip()
    if text.startswith(('"', "'")) and text.endswith(('"', "'")) and len(text) > 1:
        text = text[1:-1].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    # Drop a leading preamble line like "Here is the paraphrased abstract:" —
    # only if there's more content after it (never eat the whole response).
    if len(lines) > 1 and re.match(
        r"^(here.?s?|here is|sure|paraphrase[d]?|rewritten|restated)\b.{0,60}:?$",
        lines[0], flags=re.IGNORECASE,
    ):
        lines = lines[1:]
    joined = " ".join(lines).strip()
    joined = re.sub(r"^(Paraphrase|Rewritten|Restated)\s*:\s*", "", joined,
                     flags=re.IGNORECASE)
    if not joined or len(joined.split()) < min_words:
        return None
    return joined


PARAPHRASE_PROMPT = (
    "Paraphrase the following biomedical abstract IN FULL. Rewrite it as a single "
    "continuous abstract paragraph that reports the SAME study, the SAME findings, "
    "and the SAME numbers/conclusions, but reworded throughout — change sentence "
    "structure and vocabulary as much as you can while preserving every fact. Do not "
    "summarize, shorten, add, or drop any information; the paraphrase should be "
    "roughly the same length as the original. Output ONLY the paraphrased abstract "
    "text, with no preamble, labels, or quotes.\n\n"
    "Original abstract:\n{abstract}\n\nParaphrased abstract:"
)


def build_pool_llm(abstracts, args):
    """Paraphrase every abstract via a live Qwen chat endpoint, keep only
    low-overlap faithful rewrites. Reuses the shared parallel LLM client."""
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
    client = ParallelResponsesClient(max_concurrent=args.max_concurrent,
                                     use_cache=True, local_base_url=args.base_url)
    extra_kwargs = {}
    if not args.no_disable_thinking:
        # Qwen3 chat template supports enable_thinking via chat_template_kwargs;
        # vLLM forwards this through extra_body. Harmless no-op on non-Qwen3 models.
        extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    prompts = [PARAPHRASE_PROMPT.format(abstract=e["text"]) for e in abstracts]
    resp = client.run(model=args.model, prompts=prompts,
                      temperature=0.7, max_output_tokens=args.max_output_tokens,
                      **extra_kwargs)
    pool, n_fail, n_overlap = [], 0, 0
    for e, r in zip(abstracts, resp):
        if not r.get("success", True):
            n_fail += 1
            continue
        para = clean_abstract_response(r.get("response") or "")
        if not para or para.strip() == e["text"].strip():
            n_fail += 1
        elif word_jaccard(e["text"], para) > args.max_overlap:
            n_overlap += 1
        else:
            pool.append({"aid": e["aid"], "abstract": e["text"], "paraphrase": para,
                        "overlap": round(word_jaccard(e["text"], para), 4)})
    print(f"pool: {len(pool)} kept, {n_fail} failed/dupe/short, {n_overlap} rejected "
          f"for overlap>{args.max_overlap} (of {len(abstracts)} source abstracts)")
    return pool


def assemble_example(pool_sample, P, k, rng):
    """Build one xabsence-abstracts example from P+k distinct pool entries: P matched
    pairs (abstract->A, paraphrase->B) and k unmatched (abstract only, split A/B)."""
    matched = pool_sample[:P]
    unmatched = pool_sample[P:P + k]

    a_items = [{"text": e["abstract"], "corpus": "A", "_orphan": False} for e in matched]
    b_items = [{"text": e["paraphrase"], "corpus": "B", "_orphan": False} for e in matched]
    for e in unmatched:
        side = rng.choice(["A", "B"])
        (a_items if side == "A" else b_items).append(
            {"text": e["abstract"], "corpus": side, "_orphan": True})

    rng.shuffle(a_items)
    rng.shuffle(b_items)
    items = a_items + b_items                      # A block then B block; shared index
    gold = [i for i, it in enumerate(items) if it.pop("_orphan")]
    return {
        "documents": items,
        "queries": [],
        "gold_doc_indices": sorted(gold),
        "num_pairs": P,
        "num_unmatched": k,
        "source": "xabsence_abstracts",
    }


def assemble(pool, args, rng, split_label, count):
    P, k = args.num_pairs, args.num_unmatched
    need = P + k
    if len(pool) < need:
        raise RuntimeError(f"{split_label} pool has {len(pool)} entries < P+k={need}")
    exs = []
    for _ in range(count):
        sample = rng.sample(pool, need)
        ex = assemble_example(sample, P, k, rng)
        ex["source"] = f"xabsence_abstracts_{args.src_tag}"
        exs.append(ex)
    n_items = len(exs[0]["documents"]) if exs else 0
    tag = f"{args.src_tag}_p{P}_k{k}"
    path = f"{args.output_dir}/xabsence_abstracts_{split_label}_{tag}.jsonl"
    with open(path, "w") as f:
        for ex in exs:
            f.write(json.dumps(ex) + "\n")
    print(f"{split_label}: {len(exs)} examples, {P} pairs + {k} unmatched "
          f"({n_items} abstracts/ctx)  ->  {path}")
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # phase 1 (build pool)
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--from", dest="from_path", default="",
                    help="PubMed abstracts pool JSONL, e.g. "
                         "debug/pubmed_multiclaim/abstracts_2k_train.jsonl")
    ap.add_argument("--pool-out", default="")
    ap.add_argument("--pool-size", type=int, default=3000)
    ap.add_argument("--stub-paraphrase", action="store_true",
                    help="no-LLM placeholder paraphrases (pipeline testing only)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--max-overlap", type=float, default=0.5,
                    help="reject a paraphrase if word-Jaccard with the source exceeds "
                         "this (abstracts run higher than single sentences: shared "
                         "domain terms/numbers, so the default is looser than the "
                         "sentence xabsence default of 0.3)")
    ap.add_argument("--max-output-tokens", type=int, default=600)
    ap.add_argument("--no-disable-thinking", action="store_true",
                    help="do NOT pass chat_template_kwargs.enable_thinking=false "
                         "(needed for non-Qwen3 local models that reject the field)")
    # phase 2 (assemble) -- separate train/eval pools (no cross-split abstract reuse)
    ap.add_argument("--pool", default="", help="single pool for both splits (fallback "
                    "if --train-pool/--eval-pool not given)")
    ap.add_argument("--train-pool", default="")
    ap.add_argument("--eval-pool", default="")
    ap.add_argument("--num-pairs", type=int, default=15)
    ap.add_argument("--num-unmatched", type=int, default=3)
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=520)
    ap.add_argument("--src-tag", default="pubmed")
    ap.add_argument("--output-dir", default="debug/xabsence_abstracts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.build_pool:
        abstracts = harvest_abstracts(args.from_path, args.pool_size)
        print(f"harvested {len(abstracts)} abstracts from {args.from_path}")
        if args.stub_paraphrase:
            def _stub(a, rng):
                words = a.split()
                rng.shuffle(words)
                return "Restated: " + " ".join(words)
            pool = [{"aid": e["aid"], "abstract": e["text"],
                    "paraphrase": _stub(e["text"], rng), "overlap": None}
                   for e in abstracts]
        else:
            pool = build_pool_llm(abstracts, args)
        out = args.pool_out or f"{args.output_dir}/xabsence_abstracts_pool_{args.src_tag}.jsonl"
        with open(out, "w") as f:
            for e in pool:
                f.write(json.dumps(e) + "\n")
        print(f"wrote pool: {len(pool)} entries -> {out}")
        return

    train_pool_path = args.train_pool or args.pool
    eval_pool_path = args.eval_pool or args.pool
    if args.num_train > 0:
        if not train_pool_path:
            raise SystemExit("need --train-pool (or --pool) to assemble the train split")
        assemble(load_jsonl(train_pool_path), args, rng, "train", args.num_train)
    if args.num_eval > 0:
        if not eval_pool_path:
            raise SystemExit("need --eval-pool (or --pool) to assemble the eval split")
        assemble(load_jsonl(eval_pool_path), args, rng, "eval", args.num_eval)


if __name__ == "__main__":
    main()
