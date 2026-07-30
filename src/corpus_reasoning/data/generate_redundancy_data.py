"""Generate PubMed redundancy-detection data — the mirror of the contradiction
task. Each example is N PubMed sentences with K gold REDUNDANT pairs (S, S')
hidden among fillers and, crucially, many HARD-NEGATIVE pairs: two sentences
from the SAME abstract that share entities/vocabulary but state DIFFERENT facts
(high lexical overlap, NOT redundant). The model must find the paraphrase pairs
without being fooled by surface overlap.

Gold pair (S, S'): S is a real PubMed sentence, S' is an LLM paraphrase that
states the same finding. Two modes:
  simple — light reword, high word overlap.
  subtle — restate the same fact with almost no shared words (the hard case,
           analogous to subtle contradiction).
Gold pairs are kept only if an LLM judge confirms they are genuinely redundant.

Hard-negative pair: two distinct claim sentences from one abstract, kept only
if the judge confirms they are NOT redundant. These are the emphasis of this
task (see --hardneg-pairs).

`gold_doc_indices` stores the redundant pairs as 1-indexed [[a, b], ...] — same
format as contradiction, so the same prompt builder / parser / pair metrics
apply. Evaluate with --task redundancy.

LLM backend: defaults to the local vLLM Qwen (validated at parity with Gemini
for this kind of generation). Pass --base-url + --model, or set
LOCAL_LLM_BASE_URL, to use it; otherwise falls back to --model as an API model.

Usage (inside a serve+consume sbatch job, see jobs/gen_redundancy_qwen.sh):
    python scripts/data/generate_redundancy_data.py \\
        --num-docs 100 --num-redundancies 3 --hardneg-pairs 6 --mode both \\
        --num-train 2000 --num-eval 300 \\
        --base-url http://127.0.0.1:8765/v1 --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import random

from tqdm import tqdm

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.data.generate_pubmed_contradiction_data import (
    load_pubmed_pool, is_claim_sentence, clean_response,
)


PARA_SIMPLE = """Lightly paraphrase the following biomedical sentence so it states exactly the same fact, changing only a few words (synonyms, minor reordering). Keep all numbers, entities, and the finding identical in meaning.

Return only the paraphrased sentence, no preamble or quotes.

Original: {sentence}
Paraphrase:"""


PARA_SUBTLE = """Restate the following biomedical sentence so it conveys exactly the same fact and finding, but shares as few words as possible with the original. Rename entities (synonyms / alternative names / expand or contract abbreviations), use different verbs, a different sentence opening, and a different structure. Do NOT change any number, direction, magnitude, or the underlying finding — the two sentences must remain mutually entailing (a domain expert would say "these say the same thing").

Return only the restated sentence, no preamble or quotes.

Original: {sentence}
Restated:"""


JUDGE = """Sentence A: {a}
Sentence B: {b}

Do these two sentences state the SAME fact or finding — i.e. one is a paraphrase or restatement of the other (mutually entailing)? Two sentences about the same topic that assert DIFFERENT facts are NOT the same. A direct contradiction is NOT the same.

Answer with exactly one word: YES or NO."""


def make_client(args):
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
    return ParallelResponsesClient(
        max_concurrent=args.max_concurrent, use_cache=True,
        local_base_url=args.base_url or None)


def paraphrase(client, model, sentences, modes):
    prompts = [(PARA_SIMPLE if m == "simple" else PARA_SUBTLE).format(sentence=s)
               for s, m in zip(sentences, modes)]
    resp = client.run(model=model, prompts=prompts, temperature=0.7,
                      max_output_tokens=200)
    out = []
    for s, r in zip(sentences, resp):
        c = clean_response(r.get("response") or "") if r.get("success", True) else None
        out.append(None if (not c or c.strip() == s.strip()) else c)
    return out


def judge_pairs(client, model, pairs, want_yes):
    """Return a boolean mask: True = keep. want_yes=True keeps redundant pairs
    (gold), want_yes=False keeps non-redundant pairs (hard negatives)."""
    prompts = [JUDGE.format(a=a, b=b) for a, b in pairs]
    resp = client.run(model=model, prompts=prompts, temperature=0.0,
                      max_output_tokens=8)
    keep = []
    for r in resp:
        yes = (r.get("response") or "").strip().upper().startswith("YES")
        keep.append(yes if want_yes else (not yes))
    return keep


def build_hardneg_pool(filler_pool, rng):
    """Same-abstract (claim-sentence) pairs: high overlap, candidate non-redundant."""
    pairs = []  # (s_a, s_b, aid)
    for aid, sents in filler_pool.items():
        claims = [s for s in sents if is_claim_sentence(s)]
        if len(claims) >= 2:
            rng.shuffle(claims)
            # one pair per abstract keeps gold/hardneg abstracts spread out
            pairs.append((claims[0], claims[1], aid))
    rng.shuffle(pairs)
    return pairs


def build_example(gold_pairs, hardneg_pairs, filler_pool, num_docs, rng):
    """gold_pairs: list of (S, S', aid). hardneg_pairs: list of (s_a, s_b, aid)."""
    statements, pair_idx, hn_idx = [], [], []
    used_aids = {p[2] for p in gold_pairs} | {p[2] for p in hardneg_pairs}

    for S, Sp, _ in gold_pairs:
        a, b = len(statements), len(statements) + 1
        statements.extend([S, Sp])
        pair_idx.append((a, b))
    for sa, sb, _ in hardneg_pairs:          # planted, NOT gold
        a, b = len(statements), len(statements) + 1
        statements.extend([sa, sb])
        hn_idx.append((a, b))

    seen = set(statements)
    filler_aids = [aid for aid in filler_pool if aid not in used_aids]
    rng.shuffle(filler_aids)
    need = num_docs - len(statements)
    for aid in filler_aids:
        if need <= 0:
            break
        for s in filler_pool[aid]:
            if need <= 0:
                break
            if s in seen:
                continue
            statements.append(s); seen.add(s); need -= 1
    if len(statements) < num_docs:
        raise RuntimeError(f"ran out of fillers ({len(statements)}/{num_docs})")

    order = list(range(len(statements)))
    rng.shuffle(order)
    old_to_new = {old: new + 1 for new, old in enumerate(order)}
    documents = [{"text": statements[order[i]]} for i in range(len(order))]
    gold = sorted(sorted([old_to_new[a], old_to_new[b]]) for a, b in pair_idx)
    hardneg = sorted(sorted([old_to_new[a], old_to_new[b]]) for a, b in hn_idx)
    return {"documents": documents, "queries": [], "answers": [],
            "gold_doc_indices": gold, "source": "pubmed_redundancy",
            "_hardneg_pairs": hardneg}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-docs", type=int, default=100)
    ap.add_argument("--num-redundancies", type=int, default=3,
                    help="K gold redundant pairs per example")
    ap.add_argument("--hardneg-pairs", type=int, default=6,
                    help="H same-abstract non-redundant pairs per example "
                         "(the hard-negative emphasis)")
    ap.add_argument("--mode", choices=["simple", "subtle", "both"], default="both")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--pool-abstracts", type=int, default=20000)
    ap.add_argument("--base-url", default="",
                    help="local vLLM endpoint; routes --model there at cost 0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    K, H = args.num_redundancies, args.hardneg_pairs
    assert 2 * (K + H) <= args.num_docs, "num_docs too small for K gold + H hardneg"
    rng = random.Random(args.seed)
    claim_pool, filler_pool = load_pubmed_pool(args.pool_abstracts, args.seed)

    total_ex = args.num_train + args.num_eval
    rng.shuffle(claim_pool)
    # Oversample gold sources ~15% so the judge-reject rate (~4%) doesn't leave
    # us short of total_ex*K usable gold pairs.
    n_gold_src = int(total_ex * K * 1.15) + 10
    gold_src = claim_pool[:n_gold_src]
    if len(gold_src) < total_ex * K:
        raise RuntimeError("not enough claim sentences; raise --pool-abstracts")

    # modes per gold pair
    if args.mode in ("simple", "subtle"):
        modes = [args.mode] * len(gold_src)
    else:
        modes = []
        for _ in range(total_ex):
            half = K // 2
            m = ["simple"] * half + ["subtle"] * (K - half)
            rng.shuffle(m); modes.extend(m)

    client = make_client(args)
    sents = [s for s, _ in gold_src]
    paras = paraphrase(client, args.model, sents, modes)
    keep = judge_pairs(client, args.model,
                       [(s, p or "") for s, p in zip(sents, paras)], want_yes=True)
    gold_tuples = [(s, p, aid) for (s, aid), p, k in zip(gold_src, paras, keep)
                   if p and k]
    print(f"  gold: {len(gold_tuples)}/{len(gold_src)} paraphrases judged redundant")

    # hard negatives: same-abstract pairs judged NON-redundant. Smoke showed a
    # ~100% non-redundant rate, so a small oversample covers any rejects.
    hn_cand = build_hardneg_pool(filler_pool, rng)[: int(total_ex * H * 1.3) + 10]
    hn_keep = judge_pairs(client, args.model,
                          [(a, b) for a, b, _ in hn_cand], want_yes=False)
    hn_tuples = [c for c, k in zip(hn_cand, hn_keep) if k]
    print(f"  hardneg: {len(hn_tuples)}/{len(hn_cand)} same-abstract pairs judged "
          f"non-redundant")

    gi = hi = 0
    splits = []
    for label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        exs = []
        for _ in tqdm(range(count), desc=f"build {label}"):
            if gi + K > len(gold_tuples) or hi + H > len(hn_tuples):
                break
            g = gold_tuples[gi:gi + K]; gi += K
            h = hn_tuples[hi:hi + H]; hi += H
            exs.append(build_example(g, h, filler_pool, args.num_docs, rng))
        splits.append((label, exs))

    tag = f"pubmed_{args.mode}_n{args.num_docs}_k{K}_hn{H}"
    for label, exs in splits:
        if not exs:
            continue
        path = f"{args.output_dir}/redundancy_{label}_{tag}.jsonl"
        save_jsonl(path, exs)
        print_dataset_stats(exs, label.capitalize(), path)


if __name__ == "__main__":
    main()
