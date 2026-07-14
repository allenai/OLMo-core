"""Generate cycle-comparison data (O(N^3+) toy task).

Each example is a corpus of N comparative claims of the form
"<A> <predicate> <B>" (e.g. "Bob eats more chicken than Jane"). All claims in
one example use the SAME predicate, so they define a single directed graph over
named entities, where an edge A->B means "A > B" on that dimension. Because the
relation is a strict order, any directed CYCLE (A>B>C>A) is logically
impossible. The task: find the set of claims forming each cycle.

Construction guarantees exactly K cycles:
  - K disjoint cycles, each on L FRESH entities, edges a0->a1->...->a_{L-1}->a0.
  - All distractor claims form a DAG on a SEPARATE entity pool: entities get a
    random total order and every distractor edge goes forward in that order, so
    the distractor subgraph is acyclic. Cycle entities never appear in
    distractor claims, so no cross-edges can create or break a cycle.
  => the only directed cycles in the graph are the K planted ones.

`gold_doc_indices` stores each cycle as a list of 1-indexed claim IDs:
`[[3, 17, 8], ...]` (the claims whose edges form that cycle).

Difficulty knobs: --num-docs N (scaling axis), --cycle-len L (longer = harder,
N^L naive), --num-cycles K, and the entity-pool size.

Usage:
    python scripts/data/generate_cycle_data.py \\
        --num-docs 100 --cycle-len 3 --num-cycles 1 \\
        --num-train 2000 --num-eval 300 --output-dir data
"""

import argparse
import json
import random

# Pool of distinct first names. Examples needing more entities than this draw
# "Name_<k>" synthetic fallbacks (kept readable, still unique).
NAMES = """
Bob Jane Dan Alice Carlos Mei Omar Priya Liam Nina Hugo Sara Kofi Yuki Ivan Lena
Pablo Aisha Tom Greta Raj Elsa Noah Fatima Erik Zoe Sven Maya Paolo Rosa Ahmed
Clara Diego Hana Felix Ingrid Jamal Kira Luca Mira Nadia Oscar Petra Quinn Rosa
Theo Uma Viktor Wen Xena Yara Zane Anya Bruno Cleo Dora Enzo Faye Gus Halle Igor
Jada Kai Lara Milo Nora Otto Pia Remy Suki Tariq Vera Will Xavi Yusuf Zara Aldo
Bea Cyrus Dina Esme Finn Gita Hiro Iris Juno Kane Lily Marco Nia Olen Posy Rune
Said Tess Ugo Val Wade Xia Yael Zeb
""".split()

PREDICATES = [
    "{a} eats more chicken than {b}",
    "{a} is taller than {b}",
    "{a} runs faster than {b}",
    "{a} is older than {b}",
    "{a} owns more books than {b}",
    "{a} scored higher on the exam than {b}",
    "{a} has more money than {b}",
    "{a} lifts heavier weights than {b}",
    "{a} sleeps longer than {b}",
    "{a} drinks more coffee than {b}",
    "{a} lives closer to the office than {b}",
    "{a} types faster than {b}",
]


# Deduplicate while preserving order — distinct names are required, otherwise
# two logical entities could collide on the same name and merge in the graph,
# breaking the "exactly K cycles" invariant.
_UNIQUE_NAMES = list(dict.fromkeys(NAMES))


def entity_names(n, rng):
    """Return n distinct entity names."""
    if n <= len(_UNIQUE_NAMES):
        names = list(_UNIQUE_NAMES)
        rng.shuffle(names)
        return names[:n]
    base = list(_UNIQUE_NAMES)
    rng.shuffle(base)
    extra = [f"Name_{k}" for k in range(n - len(base))]
    return base + extra


def build_example(num_docs, cycle_len, num_cycles, rng):
    L, K = cycle_len, num_cycles
    n_distract = num_docs - K * L
    assert n_distract >= 0, "num_docs too small for K cycles of length L"

    predicate = rng.choice(PREDICATES)

    # Entity budget: K*L cycle entities + a distractor pool big enough to admit
    # n_distract distinct forward edges. With D distractor entities the DAG has
    # D*(D-1)/2 possible forward edges; pick D so that's comfortably >= need.
    d = 2
    while d * (d - 1) // 2 < max(1, n_distract):
        d += 1
    d = max(d, 3)
    total_entities = K * L + d
    names = entity_names(total_entities, rng)
    cycle_entities = names[:K * L]
    distractor_entities = names[K * L:]

    edges = []          # (a, b) meaning "a > b"
    gold_edge_groups = []  # list of edge-index lists, one per cycle

    # Planted cycles on disjoint entity sets.
    for c in range(K):
        ring = cycle_entities[c * L:(c + 1) * L]
        group = []
        for i in range(L):
            group.append(len(edges))
            edges.append((ring[i], ring[(i + 1) % L]))
        gold_edge_groups.append(group)

    # Distractor DAG: random total order, forward edges only.
    order = list(distractor_entities)
    rng.shuffle(order)
    pos = {name: i for i, name in enumerate(order)}
    possible = [(order[i], order[j])
                for i in range(len(order)) for j in range(i + 1, len(order))]
    rng.shuffle(possible)
    assert len(possible) >= n_distract, "distractor pool too small (bug)"
    edges.extend(possible[:n_distract])

    # Shuffle claim order; remap gold edge indices to 1-indexed claim IDs.
    perm = list(range(len(edges)))
    rng.shuffle(perm)
    old_to_new = {old: new + 1 for new, old in enumerate(perm)}
    shuffled = [edges[old] for old in perm]

    documents = [{"text": predicate.format(a=a, b=b)} for a, b in shuffled]
    gold = [sorted(old_to_new[e] for e in group) for group in gold_edge_groups]
    gold.sort()

    query = (f"Each claim asserts a strict comparison (the first person ranks "
             f"strictly above the second). Find every set of claims that forms "
             f"a cycle — an impossible loop where the ranking comes back to "
             f"where it started.")

    return {
        "documents": documents,
        "queries": [query],
        "answers": [],
        "gold_doc_indices": gold,   # list of cycles, each a list of claim IDs
        "source": "cycle",
        "_meta": {"cycle_len": L, "num_cycles": K, "predicate": predicate},
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-docs", type=int, default=100)
    ap.add_argument("--cycle-len", type=int, default=3)
    ap.add_argument("--num-cycles", type=int, default=1)
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for split_label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count <= 0:
            continue
        examples = [build_example(args.num_docs, args.cycle_len,
                                  args.num_cycles, rng)
                    for _ in range(count)]
        tag = f"n{args.num_docs}_len{args.cycle_len}_k{args.num_cycles}"
        path = f"{args.output_dir}/cycle_{split_label}_{tag}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_label}: {len(examples)} examples -> {path}")


if __name__ == "__main__":
    main()
