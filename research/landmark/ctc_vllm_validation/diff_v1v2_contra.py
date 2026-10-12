import json
import sys


def load(p):
    return [json.loads(l) for l in open(p)]


V1 = "/scratch/users/prasann/cpt_data/eval500/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"
V2 = (
    "/scratch/users/prasann/cpt_data/eval500_v2/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"
)
v1 = load(V1)
v2 = load(V2)
print("v1 n_ex:", len(v1), "| v2 n_ex:", len(v2))


def gold_idx_set(e):
    return set(i for p in e["gold_doc_indices"] for i in p)


def gold_texts(e):
    return set(str(e["documents"][i - 1]) for i in gold_idx_set(e))


for name, d in [("v1", v1), ("v2", v2)]:
    e = d[0]
    g0 = e["gold_doc_indices"][0]
    print(
        f"--- {name} ex0: ndoc={len(e['documents'])} gold={e['gold_doc_indices']} source={e.get('source')}"
    )
    print(f"    gold-member[0] text = {repr(str(e['documents'][g0[0]-1])[:130])}")

# Alignment: are the two files row-aligned by example (same gold-pair contradiction)?
same_gold = sum(1 for a, b in zip(v1, v2) if a["gold_doc_indices"] == b["gold_doc_indices"])
print(
    f"\nrows with IDENTICAL gold_doc_indices (v1 vs v2, row-aligned): {same_gold}/{min(len(v1),len(v2))}"
)

# Per-row: do the two share the same gold contradiction TEXTS (even if indices remapped)?
share_gold_text = sum(1 for a, b in zip(v1, v2) if gold_texts(a) == gold_texts(b))
print(f"rows with IDENTICAL gold-member TEXT sets: {share_gold_text}/{min(len(v1),len(v2))}")

# Filler/distractor overlap on row 0..4
for i in range(3):
    s1 = set(str(x) for x in v1[i]["documents"])
    s2 = set(str(x) for x in v2[i]["documents"])
    print(
        f"row{i}: doc-set overlap={len(s1&s2)}  v1_uniq={len(s1-s2)}  v2_uniq={len(s2-s1)}  "
        f"(ndoc v1={len(s1)} v2={len(s2)})"
    )


# Is v2 gold 1-indexed vs v1 0-indexed (or vice versa)? check min gold index across file
def minmax_gold(d):
    allidx = [i for e in d for i in gold_idx_set(e)]
    return min(allidx), max(allidx)


print("v1 gold idx min/max:", minmax_gold(v1), "| v2 gold idx min/max:", minmax_gold(v2))


# Do gold indices actually point at contradiction members in each? (sanity: are they valid)
def gold_valid(d):
    bad = 0
    for e in d:
        nd = len(e["documents"])
        for i in gold_idx_set(e):
            if not (1 <= i <= nd):
                bad += 1
    return bad


print("v1 out-of-range gold idx:", gold_valid(v1), "| v2:", gold_valid(v2))
