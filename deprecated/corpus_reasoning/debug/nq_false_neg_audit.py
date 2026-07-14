"""Audit NQ retrieval eval files for false negatives + gold quality.

Three questions:
 (A) Do any NON-gold docs contain the answer? (strict false-neg) — vs stored
     answers[0] AND vs the full nq_open alias list (the list used to EXCLUDE
     distractors at gen time).
 (B) Does the gold contain its own answer? vs answers[0] and vs full alias list.
     If gold fails answers[0] but passes the full list => the generator stored
     only answers[0] while selecting on the full list (alias-truncation bug).
 (C) Soft false-neg: how many distractors are 100-word chunks of the SAME
     Wikipedia article as the gold (topically near-identical, plausibly also
     relevant). Title = first quoted line of the dpr-100w passage text.
"""
import json
import re
import sys
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.bm25 import hit_contains_answer

FILES = [
    "data/nq_validation_k20_hn19_500.jsonl",
    "data/nq_validation_k50_hn49_500.jsonl",
    "data/nq_validation_k100_hn99_500.jsonl",
    "data/nq_validation_k200_hn199_500.jsonl",
]

# Map question -> full answer alias list from nq_open validation split.
def load_nq_answers():
    try:
        from datasets import load_dataset
        ds = load_dataset("nq_open", split="validation")
        return {ex["question"]: ex["answer"] for ex in ds}
    except Exception as e:
        print(f"  (could not load nq_open: {e}) — full-alias checks skipped")
        return None


_TITLE_RE = re.compile(r'^\s*"(.*?)"')


def title_of(text):
    m = _TITLE_RE.match(text)
    return m.group(1).strip().lower() if m else None


def contains_any(text, answers):
    t = text.lower()
    return any(a.lower() in t for a in answers)


def audit(path, q2ans):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    fn_substr_a0 = fn_substr_full = 0     # examples w/ distractor containing answer
    gold_miss_a0 = gold_miss_full = 0     # gold missing answer
    same_title_examples = 0               # examples w/ >=1 distractor sharing gold title
    same_title_total = 0
    worst_title = []
    for ex in rows:
        docs = ex["documents"]
        gold_idx = set(ex["gold_doc_indices"])
        a0 = ex["answers"]
        if isinstance(a0, str):
            a0 = [a0]
        full = q2ans.get(ex["queries"][0], a0) if q2ans else a0

        # (B) gold containment
        gold_ok_a0 = any(contains_any(docs[g]["text"], a0) for g in gold_idx)
        gold_ok_full = any(contains_any(docs[g]["text"], full) for g in gold_idx)
        gold_miss_a0 += (not gold_ok_a0)
        gold_miss_full += (not gold_ok_full)

        # (A) distractor containment
        d_a0 = d_full = 0
        gtitles = {title_of(docs[g]["text"]) for g in gold_idx}
        gtitles.discard(None)
        st = 0
        for i, d in enumerate(docs):
            if i in gold_idx:
                continue
            if contains_any(d["text"], a0):
                d_a0 += 1
            if contains_any(d["text"], full):
                d_full += 1
            if title_of(d["text"]) in gtitles:
                st += 1
        fn_substr_a0 += (d_a0 > 0)
        fn_substr_full += (d_full > 0)
        if st:
            same_title_examples += 1
            same_title_total += st
            worst_title.append((st, ex["queries"][0], list(gtitles)))

    print(f"\n=== {path}  ({n} examples) ===")
    print(f"  (A) distractor contains answers[0]:        {fn_substr_a0}/{n}")
    print(f"  (A) distractor contains ANY full alias:    {fn_substr_full}/{n}")
    print(f"  (B) gold missing answers[0]:               {gold_miss_a0}/{n} "
          f"({100*gold_miss_a0/n:.1f}%)")
    print(f"  (B) gold missing ALL full aliases:         {gold_miss_full}/{n} "
          f"({100*gold_miss_full/n:.1f}%)  <- true gen failure if >0")
    print(f"  (C) examples w/ same-title distractor:     {same_title_examples}/{n} "
          f"({100*same_title_examples/n:.1f}%)   total same-title docs: {same_title_total}")
    worst_title.sort(key=lambda x: -x[0])
    print("  (C) worst 6 same-title cases (count, gold title, question):")
    for c, q, t in worst_title[:6]:
        print(f"      x{c}  title={t}  q={q!r}")


if __name__ == "__main__":
    q2ans = load_nq_answers()
    for f in FILES:
        try:
            audit(f, q2ans)
        except FileNotFoundError:
            print(f"\n=== {f} — MISSING ===")
