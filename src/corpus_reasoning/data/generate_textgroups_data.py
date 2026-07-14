"""Generate textgroups data — the *textual* analog of groups4 (O(N^3) toy task).

Each example is N short natural-language passages. Every passage carries a
**textual feature value** — a count the model must read off the prose, not a
number printed in it:

  - nouns        : how many nouns the passage contains
  - verbs        : how many verbs
  - adjectives   : how many adjectives
  - connector    : how many times a chosen common word appears (e.g. "and")

The task: find every group of G=3 passages (triples) whose feature values
**add up to a target T** (optionally within a tolerance W). This is the
sum-to-target group property — the thing groups4's numeric construction
deliberately avoided ("subset-SUM can't be made provably unique without
astronomical numbers or brute-forcing all C(N,G) subsets"). For G=3 that
caveat doesn't bite: C(N,3) is tiny, so we both *construct* exactly K groups
and *verify* by brute force.

Why these features: each is **dense** (spread across every sentence, not
concentrated in one or two) and **well-defined** — "noun" means "a token in our
closed NOUN lexicon", so counting is unambiguous and we control it exactly. The
lexicons are pairwise disjoint, so a passage's noun/verb/adjective counts are
independent, and sentence structure varies so the scored count is decoupled
from word-count and sentence-count (no trivial proxy).

Construction guarantees exactly K gold triples summing to T:
  - Plant K *disjoint* triples, each summing to T (gold values in [cmin, cmax],
    cmax < T); brute-check the 3K gold values contain exactly K T-triples.
  - Add each distractor value `c` only if it forms NO T-triple with any pair
    (a, b) of already-chosen values. Maintained incrementally: when a value v
    joins, every pair (v, y) forbids the completing value T-(v+y) (widened by
    W). A candidate is accepted iff it is not forbidden — O(1) per candidate,
    so no global C(N,3) scan is needed and it stays exact at any N.
  - Scales to large N: distractors are drawn from [cmin, filler_max] with
    filler_max possibly >> cmax. Values above ~T/3 cannot form a T-triple among
    themselves, so they are placeable without bound (repetition allowed) — the
    corpus length (N) scales freely while exactness holds. The model must still
    read and count every passage to know which are candidates. (T, not N, sets
    the combinatorial search breadth, and costs passage length.)

`gold_doc_indices` stores each triple as G 1-indexed IDs: `[[3, 8, 12], ...]`.
Output schema/metric are shared with the cycle/groups4 tasks; evaluate with
`--task textgroups`. `_meta.counts` holds every passage's feature value so the
CoT builders (in data_format.py) can narrate without re-deriving the lexicon.

Usage:
    python scripts/data/generate_textgroups_data.py \\
        --num-docs 40 --num-groups 2 --group-size 3 --target 30 --tolerance 0 \\
        --feature mixed --num-train 2000 --num-eval 300
"""

import argparse
import json
import random
from itertools import combinations

# ── Closed lexicons (pairwise disjoint) ──────────────────────────────────────
# A token's POS is defined as "member of exactly one of these sets", which makes
# the feature counts unambiguous and lets the CoT/eval recompute them if needed.
NOUNS = [
    "river", "mountain", "teacher", "engine", "harbor", "garden", "lantern",
    "soldier", "merchant", "valley", "bridge", "castle", "farmer", "anchor",
    "orchard", "painter", "compass", "meadow", "captain", "tavern", "forest",
    "blacksmith", "fountain", "carriage", "shepherd", "windmill", "courtyard",
    "violin", "telescope", "harvest", "glacier", "cathedral", "scholar",
    "musician", "voyage", "saddle", "beacon", "quarry", "chapel", "wagon",
    "sailor", "tower", "island", "desert", "canyon", "miner", "weaver",
    "potter", "hunter", "baker", "library", "market", "palace", "cottage",
    "harvester", "watchman", "gardener", "fisherman", "innkeeper", "drummer",
]
VERBS = [
    "wandered", "shouted", "gathered", "vanished", "trembled", "lingered",
    "stumbled", "whispered", "marched", "collapsed", "drifted", "arrived",
    "departed", "shattered", "glimmered", "echoed", "faltered", "soared",
    "crumbled", "scattered", "returned", "circled", "hesitated", "rushed",
    "paused", "rested", "waited", "knelt", "listened", "watched",
    "rebuilt", "explored", "guarded", "studied", "celebrated", "argued",
    "labored", "traveled", "sketched", "harvested",
]
ADJECTIVES = [
    "ancient", "weary", "golden", "silent", "restless", "fragile", "stormy",
    "distant", "crimson", "hollow", "gentle", "ragged", "luminous", "narrow",
    "frozen", "endless", "bitter", "radiant", "crooked", "solemn", "vivid",
    "humble", "fierce", "tranquil", "scarlet", "rugged", "mournful", "brisk",
    "lavish", "somber",
]
ADVERBS = ["quietly", "suddenly", "slowly", "wearily", "eagerly", "calmly",
           "abruptly", "softly", "bravely", "restlessly"]
PRONOUNS = ["They", "She", "He", "It", "We"]
# Common words usable as the `connector` feature (injected a controlled number
# of times). Disjoint from every other lexicon so their count is exact.
CONNECTORS = ["and", "with", "near", "beyond", "beside", "amid"]

POS_LEX = {"nouns": NOUNS, "verbs": VERBS, "adjectives": ADJECTIVES}
_ALL = [w for lex in (NOUNS, VERBS, ADJECTIVES, ADVERBS, CONNECTORS) for w in lex]
_ALL += [p.lower() for p in PRONOUNS]
assert len(_ALL) == len(set(_ALL)), "lexicons must be pairwise disjoint"

FEATURES = ["nouns", "verbs", "adjectives", "connector"]


# ── Sentence / passage realization ───────────────────────────────────────────
def _other_fillers(rng):
    """A small bag of non-scored decoration that never lands in a scored lexicon
    we might be counting — used to vary length so word-count != feature-count."""
    return rng.choice(ADVERBS)


def _make_clause(scored, k, word, rng):
    """A clause containing EXACTLY k of the scored token.

    scored ∈ {nouns, verbs, adjectives, connector}. Filler is drawn only from
    OTHER lexicons, so the scored count is precisely k. `word` is the connector
    string when scored == 'connector'."""
    if scored == "nouns":
        if k == 0:
            return f"{rng.choice(PRONOUNS)} {rng.choice(VERBS)} {_other_fillers(rng)}"
        nouns = [rng.choice(NOUNS) for _ in range(k)]
        joined = " and ".join(f"the {n}" for n in nouns)  # 'and' is a connector word, fine
        return f"{joined.capitalize()} {rng.choice(VERBS)} {_other_fillers(rng)}"
    if scored == "verbs":
        if k == 0:
            return f"The {rng.choice(NOUNS)} stood {_other_fillers(rng)}"
        verbs = [rng.choice(VERBS) for _ in range(k)]
        return f"{rng.choice(PRONOUNS)} {' and '.join(verbs)}"
    if scored == "adjectives":
        if k == 0:
            return f"{rng.choice(PRONOUNS)} {rng.choice(VERBS)} {_other_fillers(rng)}"
        adjs = [rng.choice(ADJECTIVES) for _ in range(k)]
        return f"The {', '.join(adjs)} {rng.choice(NOUNS)} {rng.choice(VERBS)}"
    if scored == "connector":
        # k occurrences of `word`, interleaving nouns so it reads as a list.
        nouns = [rng.choice(NOUNS) for _ in range(k + 1)]
        parts = [f"the {nouns[0]}"]
        for i in range(k):
            parts.append(word)
            parts.append(f"the {nouns[i + 1]}")
        return f"{' '.join(parts).capitalize()} {rng.choice(VERBS)}"
    raise ValueError(scored)


def _make_passage(scored, value, word, rng, min_sents=3, max_sents=80):
    """Build a passage whose scored feature count is exactly `value`, spread
    densely across several sentences (no single sentence carries it all)."""
    # Spread `value` over M sentences, each getting 0..3 of the scored token,
    # most >=1 so the feature is dense. M is not a clean function of value
    # (per-sentence counts are random), so sentence-count is not a proxy.
    parts = []
    remaining = value
    while remaining > 0:
        take = min(remaining, rng.randint(1, 3))
        parts.append(take)
        remaining -= take
    # Pad with a few zero-feature sentences so length decouples from the count.
    n_zero = rng.randint(0, 2)
    parts += [0] * n_zero
    rng.shuffle(parts)
    if len(parts) < min_sents:
        parts += [0] * (min_sents - len(parts))
    if len(parts) > max_sents:
        # merge tail into earlier sentences to respect the cap (rare for big value)
        head, tail = parts[:max_sents], parts[max_sents:]
        for i, extra in enumerate(tail):
            head[i % max_sents] += extra
        parts = head
    sents = [_make_clause(scored, k, word, rng).strip().rstrip(".") + "."
             for k in parts]
    return " ".join(sents)


def _count_feature(text, scored, word):
    """Recompute the feature count from text — used as a generation-time assert."""
    toks = [t.strip(".,").lower() for t in text.split()]
    if scored == "connector":
        return sum(1 for t in toks if t == word)
    return sum(1 for t in toks if t in POS_LEX[scored])


# ── Value construction (exactly K triples summing to target) ─────────────────
def _count_target_triples(values, T, W):
    return sum(1 for c in combinations(range(len(values)), 3)
               if abs(values[c[0]] + values[c[1]] + values[c[2]] - T) <= W)


def sample_values(n_docs, n_groups, group_size, target, tol, cmin, cmax,
                  filler_max, rng, separation=0, max_attempts=2000):
    """Return (values, gold_index_groups): exactly `n_groups` disjoint triples
    whose values sum to `target` (within `tol`), and no other triple does.

    Scales to large N because distractor values may range up to `filler_max`
    (>> cmax): values above ~target/3 cannot form a target-triple among
    themselves, so they are placeable without bound (repetition allowed). The
    only triples that can hit the target are then the planted gold ones.

    Exactness is maintained incrementally via a `forbidden` set of completing
    values: when a value v joins `chosen`, every pair (v, y) blocks the value
    target-(v+y) (widened by the band M) from ever being added. A distractor is
    accepted only if it is not currently forbidden, so no distractor-containing
    triple can sum near the target. Gold values are added unconditionally and the
    3K-value gold set is brute-checked to contain exactly K triples within M.
    Hence the planted K are the only near-target triples — no global C(N,3) scan.

    `separation` (M = max(tol, separation)) sets an ISOLATION MARGIN: no non-gold
    triple sums within M of the target, so the gold sum sits in a clean gap with
    no near-misses (e.g. M=5 → nothing in [T-5, T+5] but the gold T-triples).
    This makes the task easier: the model need only find triples summing to ~T,
    not solve exact subset-sum against a field of off-by-one decoys. M=0 (the
    default) recovers the hard, near-miss-dense construction."""
    G, K, T, W = group_size, n_groups, target, tol
    M = max(W, separation)
    lo_c, hi_c = cmin, filler_max
    n_distract = n_docs - K * G
    if n_distract < 0:
        raise ValueError("num_docs too small for num_groups * group_size")

    def block_pairs(chosen, new_v, forbidden):
        """Forbid every value that would complete a near-target triple (sum
        within the band M of T) with (new_v, y) for y already in `chosen`."""
        for y in chosen:
            center = T - (new_v + y)
            for f in range(center - M, center + M + 1):
                forbidden.add(f)

    for _ in range(max_attempts):
        # 1) K disjoint planted groups, each summing to T, drawn from [cmin, cmax]
        #    (cmax kept below T so gold members look like ordinary small counts).
        #    Constructive separation: every value in a new group must be MORE
        #    than M away from every value already placed in another group. Then
        #    any mixed triple (2 from group A, 1 from B) sums to T - a_i + b_k,
        #    which is within M of T iff |a_i - b_k| <= M — excluded by the
        #    separation. So the planted K are the ONLY near-target triples, and
        #    the brute check below is just a cheap confirmation (always passes).
        gold_vals = []
        used = []          # all gold values placed so far (across groups)
        ok = True
        for _g in range(K):
            placed = None
            for _try in range(400):
                parts = _random_partition(T, G, cmin, cmax, rng)
                if parts is None:
                    continue
                if any(abs(v - u) <= M for v in parts for u in used):
                    continue  # too close to another group's value
                placed = parts
                break
            if placed is None:
                ok = False
                break
            gold_vals.append(placed)
            used.extend(placed)
        if not ok:
            continue
        flat_gold = [v for g in gold_vals for v in g]
        # confirm exactly K near-target triples within margin M (cheap, 3K vals)
        if _count_target_triples(flat_gold, T, M) != K:
            continue

        # seed `forbidden` from every pair already present among the gold values
        chosen = []
        forbidden = set()
        for v in flat_gold:
            block_pairs(chosen, v, forbidden)
            chosen.append(v)

        # 2) distractors: sample candidates in [cmin, filler_max] with repetition;
        #    accept only those not currently forbidden (can't complete a T-triple
        #    with any existing pair). O(1) per candidate via the forbidden set.
        distractors = []
        attempts, max_c_attempts = 0, n_distract * 300 + 5000
        while len(distractors) < n_distract and attempts < max_c_attempts:
            attempts += 1
            c = rng.randint(lo_c, hi_c)
            if c in forbidden:
                continue
            block_pairs(chosen, c, forbidden)
            chosen.append(c)
            distractors.append(c)
        if len(distractors) < n_distract:
            continue

        values = flat_gold + distractors
        gold = [list(range(i * G, i * G + G)) for i in range(K)]
        # cheap safety net for small corpora (full C(N,3) is fine up to ~150);
        # for large N the incremental argument above already guarantees it.
        # Check within the margin M: exactly K triples are even near the target.
        if n_docs <= 150:
            assert _count_target_triples(values, T, M) == K, "exactness check failed"
        return values, gold

    raise RuntimeError(
        f"Could not assemble values after {max_attempts} attempts. Try a more "
        f"central --target, a wider [cmin, cmax], a smaller --num-docs, or "
        f"fewer --num-groups (distractor exclusion gets tight near the mode).")


def _random_partition(total, parts, lo, hi, rng):
    """`parts` integers in [lo, hi] summing to `total`, or None if infeasible."""
    if not (parts * lo <= total <= parts * hi):
        return None
    vals = []
    remaining = total
    for i in range(parts):
        slots_left = parts - i - 1
        # bounds so the remaining slots can still be filled within [lo, hi]
        v_lo = max(lo, remaining - slots_left * hi)
        v_hi = min(hi, remaining - slots_left * lo)
        if v_lo > v_hi:
            return None
        v = rng.randint(v_lo, v_hi)
        vals.append(v)
        remaining -= v
    rng.shuffle(vals)
    return vals


# ── Example assembly ─────────────────────────────────────────────────────────
def _feature_phrase(feature, word):
    if feature == "connector":
        return f'the number of times the word "{word}" appears'
    return f"the number of {feature}"


def build_example(args, rng):
    feature = (rng.choice(FEATURES) if args.feature == "mixed" else args.feature)
    word = rng.choice(CONNECTORS) if feature == "connector" else None

    separation = getattr(args, "separation", 0)
    if args.filler_max is not None:
        filler_max = args.filler_max
    else:
        filler_max = max(args.cmax, args.target)
        if separation > 0:
            # A margin widens the forbidden band, so the distractor range can be
            # fully excluded. Guarantee a populated "always-safe" band: any value
            # above T - 2*cmin + M can never complete a near-target triple (the
            # largest completing value is T - 2*cmin + M), so it places freely.
            # +30 gives that band enough width for randint to hit it reliably.
            filler_max = max(filler_max,
                             args.target - 2 * args.cmin + separation + 30)
    values, gold_idx = sample_values(
        args.num_docs, args.num_groups, args.group_size, args.target,
        args.tolerance, args.cmin, args.cmax, filler_max, rng,
        separation=separation)

    texts = []
    for v in values:
        for _ in range(20):
            t = _make_passage(feature, v, word, rng)
            if _count_feature(t, feature, word) == v:
                break
        else:
            raise RuntimeError(f"could not realize value {v} for {feature}")
        texts.append(t)

    # shuffle display order, remap gold IDs to 1-indexed shuffled positions
    order = list(range(len(values)))
    rng.shuffle(order)
    perm = {old: new + 1 for new, old in enumerate(order)}
    shuffled_texts = [texts[i] for i in order]
    shuffled_counts = [values[i] for i in order]

    gold = sorted(sorted(perm[i] for i in group) for group in gold_idx)

    phrase = _feature_phrase(feature, word)
    tol_clause = (f" (within {args.tolerance} of {args.target})"
                  if args.tolerance else f" (exactly {args.target})")
    query = (f"Each passage has a value: {phrase}. Find every group of "
             f"{args.group_size} passages whose values add up to "
             f"{args.target}{tol_clause}.")

    return {
        "documents": [{"text": t} for t in shuffled_texts],
        "queries": [query],
        "answers": [],
        "gold_doc_indices": gold,
        "source": "textgroups",
        "_meta": {
            "feature": feature,
            "word": word,
            "target": args.target,
            "tolerance": args.tolerance,
            "group_size": args.group_size,
            "num_groups": args.num_groups,
            "filler_max": filler_max,
            "separation": separation,   # isolation margin: 0 = near-miss-hard
            "counts": shuffled_counts,   # per-passage feature value, display order
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-docs", type=int, default=40)
    ap.add_argument("--num-groups", type=int, default=2,
                    help="exact number of gold triples (K)")
    ap.add_argument("--group-size", type=int, default=3,
                    help="passages per group (G); triples by default")
    ap.add_argument("--target", type=int, default=70,
                    help="the sum each gold group's feature values must hit (T). "
                         "Larger T widens the combinatorial search (more small "
                         "counts can combine) at the cost of longer passages.")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="allowed slack W: a group is gold if its sum is within "
                         "W of the target (0 = exact)")
    ap.add_argument("--separation", type=int, default=0,
                    help="isolation margin M: guarantee NO non-gold triple sums "
                         "within M of the target, so gold sits in a clean gap "
                         "with no near-miss decoys (e.g. 5 -> nothing in "
                         "[T-5, T+5] but gold). Makes the task easier; 0 keeps "
                         "the hard, near-miss-dense construction.")
    ap.add_argument("--feature", default="mixed",
                    choices=FEATURES + ["mixed"],
                    help="textual feature per example ('mixed' = random each example)")
    ap.add_argument("--cmin", type=int, default=4,
                    help="min per-passage feature count (also the floor for gold "
                         "group members)")
    ap.add_argument("--cmax", type=int, default=40,
                    help="max feature count for GOLD group members; keep below "
                         "--target so gold members read as ordinary small counts")
    ap.add_argument("--filler-max", type=int, default=None,
                    help="max feature count for distractor/filler passages "
                         "(default max(cmax, target)). Values above ~target/3 "
                         "can't form a target-triple, so raising this lets the "
                         "corpus scale to large N without breaking exact-K.")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for split_label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count <= 0:
            continue
        examples = [build_example(args, rng) for _ in range(count)]
        tag = (f"n{args.num_docs}_g{args.group_size}_k{args.num_groups}"
               f"_t{args.target}_w{args.tolerance}_{args.feature}")
        path = f"{args.output_dir}/textgroups_{split_label}_{tag}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_label}: {len(examples)} examples -> {path}")


if __name__ == "__main__":
    main()
