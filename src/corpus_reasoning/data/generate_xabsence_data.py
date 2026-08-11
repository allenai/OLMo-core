"""Cross-corpus absence (xabsence): find the chunks with no paraphrase twin.

Two corpora A and B share one context. Almost every claim has a PARAPHRASE in the
OTHER corpus (a matched pair: original in A, a low-overlap Qwen paraphrase in B).
Exactly k claims are UNMATCHED — no paraphrase anywhere in the other corpus — and
the model must name them. Because the twins are genuine paraphrases (word-Jaccard
filtered low), the orphan can't be found by surface string-matching; it requires
semantic all-pairs comparison across the two corpora (a strong chunked-vs-standard
discriminator). Output is a set of unmatched IDs, scored by the absence set-F1
(`--task xabsence` routes to `_eval_absence`).

Schema (reuses the absence gold field so the scorer is shared):
    {"documents": [{"text": <claim>, "corpus": "A"|"B"}, ...],   # A block then B block
     "queries": [],
     "gold_doc_indices": [unmatched positions, 0-indexed],
     "num_pairs": P, "num_unmatched": k, "source": "xabsence_<tag>"}

Two phases (paraphrasing is the only expensive step; it is decoupled so the pool
is built ONCE and reused for any N / count):

  1. BUILD POOL (LLM):  harvest claims from a source unified JSONL, generate a
     hard paraphrase of each via a local Qwen-Instruct (word-Jaccard <= --max-overlap),
     write {claim, paraphrase} pairs.
        python scripts/data/generate_xabsence_data.py --build-pool \\
            --from data/contradiction_train_pubmed_both_n100_k3.jsonl \\
            --pool-out data/xabsence_pool_pubmed.jsonl \\
            --pool-size 3000 --model Qwen2.5-14B-Instruct --base-url http://localhost:8000/v1 \\
            --max-overlap 0.3
     (--stub-paraphrase builds a no-LLM pool for pipeline testing.)

  2. ASSEMBLE (cheap):  sample P matched pairs + k unmatched per example.
        python scripts/data/generate_xabsence_data.py \\
            --pool data/xabsence_pool_pubmed.jsonl \\
            --num-pairs 12 --num-unmatched 3 --num-train 2000 --num-eval 300 \\
            --src-tag pubmed --output-dir data
"""

import argparse
import json
import random


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_exact_copy_pool(abstracts_path, limit):
    """Build a pool whose 'paraphrase' IS the 'claim' -- the exact-copy xabsence variant.

    The paraphrase variant leaked: an orphan was inserted in claim form into corpus B, whose other
    members were all LLM paraphrases, so it could be spotted from that ONE document's style with no
    cross-corpus comparison (a trained 4B hit 0.98 recall on B-side orphans vs 0.08 on A-side).
    Making the twin an EXACT COPY removes every per-document cue by construction: a matched
    document and an orphan are drawn from the same distribution and are individually
    indistinguishable, so the orphan can only be found by comparing documents across the corpora.

    Full abstracts (not single sentences) also make each document self-contained and long, so a
    token-budgeted rung holds far fewer documents than the sentence variant did.

    :param abstracts_path: JSONL with an ``abstract`` (or ``text``) field per row.
    :param limit: Maximum pool entries to keep.

    :returns: List of ``{"claim": t, "paraphrase": t}`` with de-duplicated text.
    """
    seen, pool = set(), []
    for row in load_jsonl(abstracts_path):
        t = (row.get("abstract") or row.get("text") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        pool.append({"claim": t, "paraphrase": t})
        if len(pool) >= limit:
            break
    return pool


def harvest_claims(from_path, limit):
    """Collect unique document texts from a unified-format JSONL as the claim pool."""
    seen, claims = set(), []
    for ex in load_jsonl(from_path):
        for d in ex.get("documents", []):
            t = (d.get("text") or "").strip()
            if t and t not in seen:
                seen.add(t)
                claims.append(t)
                if len(claims) >= limit:
                    return claims
    return claims


def _stub_paraphrase(claim, rng):
    """No-LLM placeholder: reorder words + tag, so it differs lexically from the
    source (low-ish overlap) while staying traceable. For pipeline testing ONLY."""
    words = claim.split()
    rng.shuffle(words)
    return "Restated: " + " ".join(words)


def build_pool_llm(claims, args):
    """Paraphrase every claim via the local Qwen-Instruct, keep only low-overlap
    faithful rewrites. Reuses the contradiction generator's client + helpers."""
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
    from corpus_reasoning.data.generate_pubmed_contradiction_data import word_jaccard, clean_response
    PROMPT = (
        "Paraphrase the following biomedical claim. Restate the SAME fact with "
        "DIFFERENT wording and sentence structure — change as many words as you can "
        "while preserving the exact meaning. Do not add or remove information. "
        "Output only the paraphrase.\n\nClaim: {claim}\n\nParaphrase:"
    )
    client = ParallelResponsesClient(max_concurrent=args.max_concurrent,
                                     use_cache=True, local_base_url=args.base_url)
    resp = client.run(model=args.model,
                      prompts=[PROMPT.format(claim=c) for c in claims],
                      temperature=0.7, max_output_tokens=200)
    pool, n_fail, n_overlap = [], 0, 0
    for c, r in zip(claims, resp):
        if not r.get("success", True):
            n_fail += 1
            continue
        para = clean_response(r.get("response") or "")
        if not para or para.strip() == c.strip():
            n_fail += 1
        elif word_jaccard(c, para) > args.max_overlap:
            n_overlap += 1
        else:
            pool.append({"claim": c, "paraphrase": para})
    print(f"pool: {len(pool)} kept, {n_fail} failed/dupe, {n_overlap} rejected "
          f"for overlap>{args.max_overlap}")
    return pool


def assemble_example(pool_sample, P, k, rng):
    """Build one xabsence example from P+k distinct pool entries: P matched pairs
    (claim->A, paraphrase->B) and k unmatched, split across A/B.

    Corpus A holds originals and corpus B holds paraphrases -- and that invariant must hold for the
    ORPHANS TOO, or the task collapses into a per-document style classifier.
    """
    matched = pool_sample[:P]
    unmatched = pool_sample[P:P + k]

    a_items = [{"text": e["claim"], "corpus": "A", "_orphan": False} for e in matched]
    b_items = [{"text": e["paraphrase"], "corpus": "B", "_orphan": False} for e in matched]
    # Split the k orphans across A and B (randomized per example).
    for e in unmatched:
        side = rng.choice(["A", "B"])
        # Take the text form matching the corpus's style. Previously this always used
        # e["claim"], so every B-side orphan was an ORIGINAL sitting among PARAPHRASES --
        # detectable from that one document alone, with no cross-corpus comparison. A trained
        # 4B exploited exactly that: recall 0.98 on B-side orphans vs 0.08 on A-side (A has no
        # such cue), which pinned set_f1 at ~0.5 and made it FLAT in n from 39 to 669 docs,
        # i.e. it stopped being an all-pairs task at all.
        text = e["claim"] if side == "A" else e["paraphrase"]
        (a_items if side == "A" else b_items).append(
            {"text": text, "corpus": side, "_orphan": True})

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
        "source": "xabsence",
    }


def assemble(pool, args, rng, split_label, count):
    """Assemble ``count`` xabsence examples.

    Two ways to pick the corpus size ``P`` (pairs per example):

    * **Fixed** (default, ``--num-pairs``): every example gets the same ``P``. The CTC suite builds
      a ladder by running this once per rung and concatenating, so training ``n`` is 5 point masses
      sitting exactly on the eval rungs.
    * **Continuous** (``--num-pairs-min``/``--num-pairs-max``): ``P`` is sampled per example, the
      way ``generate_oolong_ladder_data.py`` and the NQ generator do it. The model then sees
      hundreds of distinct corpus sizes, so the ladder measures scaling in ``n`` rather than recall
      of the five sizes it was trained on -- and nothing sits beyond the largest trained ``n``.

    :param pool: Claim/paraphrase pool entries.
    :param args: Parsed CLI options.
    :param rng: Seeded RNG.
    :param split_label: ``"train"`` or ``"eval"``.
    :param count: Number of examples to emit.

    :raises RuntimeError: If the pool is too small for the largest requested ``P + k``.
    """
    k = args.num_unmatched
    lo, hi = args.num_pairs_min, args.num_pairs_max
    continuous = lo is not None and hi is not None
    max_need = (hi if continuous else args.num_pairs) + k
    if len(pool) < max_need:
        raise RuntimeError(f"pool has {len(pool)} entries < max P+k={max_need}")
    exs = []
    for _ in range(count):
        P = rng.randint(lo, hi) if continuous else args.num_pairs
        sample = rng.sample(pool, P + k)
        ex = assemble_example(sample, P, k, rng)
        ex["source"] = f"xabsence_{args.src_tag}"
        exs.append(ex)
    sizes = sorted({len(e["documents"]) for e in exs})
    tag = f"{args.src_tag}_p{lo}-{hi}_k{k}" if continuous else f"{args.src_tag}_p{args.num_pairs}_k{k}"
    path = f"{args.output_dir}/xabsence_{split_label}_{tag}.jsonl"
    with open(path, "w") as f:
        for ex in exs:
            f.write(json.dumps(ex) + "\n")
    if continuous:
        print(f"{split_label}: {len(exs)} examples, P~U[{lo},{hi}] + {k} unmatched "
              f"({len(sizes)} distinct corpus sizes, {sizes[0]}-{sizes[-1]} items/ctx)  ->  {path}")
    else:
        print(f"{split_label}: {len(exs)} examples, {args.num_pairs} pairs + {k} unmatched "
              f"({sizes[0]} items/ctx)  ->  {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # phase 1 (build pool)
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--from-abstracts", default="",
                    help="build an EXACT-COPY pool from this abstracts JSONL (field 'abstract' or "
                         "'text') instead of LLM-paraphrasing. The B-side twin is byte-identical to "
                         "the A-side original, so no per-document style cue exists and the orphan "
                         "can only be found by cross-corpus comparison. No LLM/vLLM required.")
    ap.add_argument("--from", dest="from_path", default="")
    ap.add_argument("--pool-out", default="")
    ap.add_argument("--pool-size", type=int, default=3000)
    ap.add_argument("--stub-paraphrase", action="store_true",
                    help="no-LLM placeholder paraphrases (pipeline testing only)")
    ap.add_argument("--model", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-concurrent", type=int, default=64)
    ap.add_argument("--max-overlap", type=float, default=0.3)
    # phase 2 (assemble)
    ap.add_argument("--pool", default="", help="pool JSONL to assemble from")
    ap.add_argument("--num-pairs", type=int, default=12,
                    help="fixed pairs per example (one ladder rung per invocation)")
    ap.add_argument("--num-pairs-min", type=int, default=None,
                    help="with --num-pairs-max, sample pairs per example from U[min,max] instead "
                         "of a fixed size (the continuous-n convention used by the oolong ladder "
                         "and NQ generators). Overrides --num-pairs.")
    ap.add_argument("--num-pairs-max", type=int, default=None,
                    help="upper bound for continuous-n sampling; see --num-pairs-min")
    ap.add_argument("--num-unmatched", type=int, default=3)
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--src-tag", default="pubmed")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = None
    if args.from_abstracts:
        pool = build_exact_copy_pool(args.from_abstracts, args.pool_size)
        out = args.pool_out or f"{args.output_dir}/xabsence_pool_{args.src_tag}_exact.jsonl"
        with open(out, "w") as f:
            for e in pool:
                f.write(json.dumps(e) + "\n")
        print(f"wrote EXACT-COPY pool: {len(pool)} entries -> {out}")
    elif args.build_pool:
        claims = harvest_claims(args.from_path, args.pool_size)
        print(f"harvested {len(claims)} claims from {args.from_path}")
        if args.stub_paraphrase:
            pool = [{"claim": c, "paraphrase": _stub_paraphrase(c, rng)} for c in claims]
        else:
            pool = build_pool_llm(claims, args)
        out = args.pool_out or f"{args.output_dir}/xabsence_pool_{args.src_tag}.jsonl"
        with open(out, "w") as f:
            for e in pool:
                f.write(json.dumps(e) + "\n")
        print(f"wrote pool: {len(pool)} entries -> {out}")
        if not args.pool:
            return

    pool = load_jsonl(args.pool or out)
    for label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count > 0:
            assemble(pool, args, rng, label, count)


if __name__ == "__main__":
    main()
