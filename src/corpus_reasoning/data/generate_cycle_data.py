"""Generate cycle-comparison data (O(N^3+) toy task).

Each example is a corpus of N comparative claims of the form
"<A> <predicate> <B>" (e.g. "Bob eats more chicken than Jane"). All claims in
one example use the SAME predicate, so they define a single directed graph over
named entities, where an edge A->B means "A > B" on that dimension. Because the
relation is a strict order, any directed CYCLE (A>B>C>A) is logically
impossible. The task: find the set of claims forming each cycle.

Construction guarantees exactly K cycles:
  - K disjoint cycles, each on L entities, edges a0->a1->...->a_{L-1}->a0.
  - ALL entities (cycle + background) share ONE random total order ("rank").
    Every non-cycle claim ("background" edge, incl. edges touching cycle
    entities) goes strictly forward in that order, so on its own that subgraph
    is a DAG. Each planted cycle's L entities occupy CONSECUTIVE ranks (a
    contiguous "block"); the cycle itself is the forward chain
    a0->a1->...->a_{L-1} (consistent with rank) plus ONE backward "closing"
    edge a_{L-1}->a0. The only structural rule needed for uniqueness is: no
    background edge has BOTH endpoints inside the same block. Proof sketch: a
    background/forward-only graph is acyclic, so any cycle must use a closing
    edge; since no background entity has a rank strictly between a block's
    endpoints (the block is consecutive) and no background edge can jump
    between two entities of the same block, any forward path from a0 to
    a_{L-1} is trapped inside the block and forced through the interior chain
    -- it can never detour out and back (rank only increases). So the planted
    chain is the UNIQUE forward path completing each closing edge, i.e. the
    only cycle.
  - Unlike the pool-disjoint version this replaces, cycle entities are FULL
    participants in the background-edge sampling (both as source and target,
    anywhere outside their own block), so their claim-frequency lands in the
    same distribution as background entities instead of being pinned at
    exactly 2 (one in-edge, one out-edge) regardless of N. See
    `records/ctc-setting-verification-2026-07-23.md` for the diagnosis this
    fixes (gold was the 3 rarest names in the corpus, growing MORE
    distinguishable as N grew -- a frequency shortcut, not real O(N^L) search).

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

    # Entity budget: K*L cycle entities + a background pool big enough to admit
    # n_distract distinct forward edges among ALL entities (cycle included),
    # minus the K*L*(L-1)/2 same-block pairs that are off-limits (see below).
    # Sizing is the same asymptotic rule as before (d ~ sqrt(2*n_distract)), so
    # background-entity degree is UNCHANGED by this fix -- only cycle-entity
    # degree moves, from pinned-at-2 up to background-comparable.
    d = 2
    while True:
        total_entities = K * L + d
        available = (total_entities * (total_entities - 1) // 2
                     - K * (L * (L - 1) // 2))
        if available >= max(1, n_distract):
            break
        d += 1
    total_entities = K * L + d
    # entity_names shuffles the FIXED-length name pool before truncating, so
    # this prefix (and hence which names are the cycle) is stable across
    # different --num-docs for the same --seed -- preserves eval row-alignment
    # (same gold claims at every rung, only the background pool grows).
    names = entity_names(total_entities, rng)
    cycle_entities = names[:K * L]
    background_entities = names[K * L:]

    # Global rank order over ALL entities: splice each cycle's L entities in
    # as a CONSECUTIVE block at a random position among the shuffled
    # background order. Consecutive placement is what makes the "no same-block
    # background edge" rule below sufficient for uniqueness (see docstring).
    order = list(background_entities)
    rng.shuffle(order)
    insert_points = sorted(rng.sample(range(len(order) + 1), K))
    full_order = []
    block_spans = []  # (start_pos, end_pos) inclusive, per cycle, into full_order
    prev = 0
    for c, ip in enumerate(insert_points):
        full_order.extend(order[prev:ip])
        start_pos = len(full_order)
        full_order.extend(cycle_entities[c * L:(c + 1) * L])
        block_spans.append((start_pos, start_pos + L - 1))
        prev = ip
    full_order.extend(order[prev:])
    assert len(full_order) == total_entities

    def in_same_block(i, j):
        return any(s <= i <= e and s <= j <= e for s, e in block_spans)

    edges = []          # (a, b) meaning "a > b"
    gold_edge_groups = []  # list of edge-index lists, one per cycle

    # Planted cycles: forward chain a0->a1->...->a_{L-1} (consistent with
    # rank) + one backward closing edge a_{L-1}->a0.
    for c, (start_pos, end_pos) in enumerate(block_spans):
        ring = full_order[start_pos:end_pos + 1]
        group = []
        for i in range(L):
            group.append(len(edges))
            edges.append((ring[i], ring[(i + 1) % L]))
        gold_edge_groups.append(group)

    # Background/padding: every forward pair (rank i < rank j) is eligible
    # EXCEPT pairs with both endpoints in the same cycle block (the only
    # bypass risk -- see docstring proof). Cycle entities are otherwise full
    # participants (as source AND target), so their frequency lands in the
    # same distribution as background entities instead of being frozen at 2.
    possible = [(full_order[i], full_order[j])
                for i in range(total_entities)
                for j in range(i + 1, total_entities)
                if not in_same_block(i, j)]
    rng.shuffle(possible)
    assert len(possible) >= n_distract, "entity pool too small (bug)"
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
