"""Generate in-domain RULER train/eval data (native reimplementation).

RULER (https://github.com/NVIDIA/RULER) is a synthetic long-context benchmark.
Each "subtask" is generated parametrically, so we can emit a *train* split and a
*test* split from the **same** generative process with disjoint random draws —
the in-domain setup for fine-tuning a small LM on the task format and testing it
on held-out instances.

Implemented subtasks (NIAH variants + variable tracking + word extraction):

  niah_single       single magic-number needle, single query                (recall=1 string)
  niah_multikey     1 gold needle + (k-1) distractor needles, query one key  (recall=1 string)
  niah_multivalue   one key, V values; retrieve ALL values                   (recall=V strings)
  niah_multiquery   Q keys each w/ a value; query ALL keys                    (recall=Q strings)
  vt                C variable-assignment chains x H hops; find the chain     (recall=H+1 vars)
                    of variables equal to a queried value
  cwe               common-words extraction: report the most common words    (recall=num_cw)
  fwe               frequent-words extraction (Zipfian); report top-3        (recall=top_k)

Output is the unified JSONL format consumed by scripts/eval/evaluate.py and the
trainers. The long context is stored as a list of `documents` (haystack chunks
and needles are each their own document, so they become isolated chunks under
chunked attention). `answers` holds ALL gold strings the model must recover; the
ruler eval scores recall over them (RULER's `string_match_all`).

    {
      "documents": [{"title": null, "text": "<chunk or needle sentence>"}, ...],
      "queries":   ["<the RULER query>"],
      "answers":   ["7384021", ...],          # all required gold strings
      "gold_doc_indices": [3, 17],            # positions of the needle docs
      "source":   "ruler_niah_single",
      "_task":    "ruler",
      "_meta":    {"ruler_subtask": "niah_single", "target_length": 8192, ...}
    }

Length control (configurable, --length-mode):
  target  : pad the haystack until the rendered context reaches ~N tokens
            (needs --tokenizer; this is RULER-native and length-calibrated).
  num_docs: haystack is exactly N chunks regardless of token count
            (repo-native, consistent with the other generators, no tokenizer).

In-domain disjointness:
  Train and eval are drawn from independent RNG streams (eval seed = seed+1).
  With --disjoint-vocab the key/value word pools are partitioned so no needle
  key seen at train time ever appears at test time (a stricter generalization
  probe). Numeric values are effectively disjoint either way (7-digit space).

Usage:
    python scripts/data/generate_ruler_data.py \\
        --subtask niah_single --length-mode target --target-length 8192 \\
        --tokenizer Qwen/Qwen2.5-0.5B \\
        --num-train 2000 --num-eval 300 --output-dir data --seed 42

    python scripts/data/generate_ruler_data.py \\
        --subtask vt --length-mode num_docs --num-docs 50 \\
        --num-train 2000 --num-eval 300 --output-dir data
"""

import argparse
import json
import random
import string

# ── Bundled vocabulary (offline; no wonderwords dependency) ────────────────────
# Concrete nouns used as needle "keys" and as word-extraction tokens. Kept
# deliberately plain so needles read naturally and answers are unambiguous.
WORDS = [
    "apple", "river", "mountain", "engine", "garden", "planet", "harbor",
    "lantern", "compass", "thunder", "meadow", "canyon", "glacier", "orbit",
    "anchor", "beacon", "cobalt", "ember", "falcon", "granite", "harvest",
    "ivory", "jasmine", "kettle", "ledger", "maple", "nectar", "obsidian",
    "pebble", "quartz", "ribbon", "saffron", "tundra", "umbra", "violet",
    "willow", "xenon", "yarrow", "zephyr", "almond", "basalt", "cedar",
    "dahlia", "elm", "fennel", "ginger", "hazel", "iris", "juniper", "kelp",
    "lupine", "myrtle", "nutmeg", "orchid", "poplar", "quince", "rowan",
    "sorrel", "thistle", "ursine", "verbena", "walnut", "yucca", "zinnia",
    "amber", "bronze", "copper", "diamond", "emerald", "flint", "garnet",
    "helium", "indigo", "jade", "krypton", "lithium", "marble", "neon",
    "onyx", "pearl", "ruby", "silver", "topaz", "uranium", "velvet", "wax",
    "yellow", "azure", "crimson", "denim", "ebony", "fawn", "gold", "henna",
    "ivory2", "khaki", "lilac", "mauve", "navy", "ochre", "plum", "rust",
    "scarlet", "teal", "umber", "vermilion", "wheat", "ash", "birch", "coral",
]

# Pronounceable pseudo-word generator — an effectively unbounded pool of
# distinct word-like tokens used as CWE *uncommon* filler. A large distinct pool
# is what lets CWE scale to long contexts while keeping each uncommon word rare
# (so the common answer words stay strictly most frequent). The curated real
# WORDS above remain the source for the common/answer words so the gold reads
# cleanly.
_CONS = "bcdfghjklmnpqrstvwz"
_VOWS = "aeiou"


def pseudo_words(rng, avoid=()):
    """Yield distinct pronounceable pseudo-words (e.g. 'bakato'), never in avoid."""
    seen = set(avoid)
    while True:
        n_syl = rng.randint(2, 3)
        w = "".join(rng.choice(_CONS) + rng.choice(_VOWS) for _ in range(n_syl))
        if w not in seen:
            seen.add(w)
            yield w


# RULER's default "noise" haystack sentence, repeated to fill context.
NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rand_number(rng, digits=7):
    """Random fixed-width integer string (leading digit non-zero)."""
    first = rng.randint(1, 9)
    rest = "".join(str(rng.randint(0, 9)) for _ in range(digits - 1))
    return f"{first}{rest}"


def rand_var(rng):
    """Random variable name, e.g. 'VBXQ' (uppercase, length 3-5)."""
    n = rng.randint(3, 5)
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


# ── Needle-family payloads ─────────────────────────────────────────────────────
# Each returns (needle_texts, query, answers). needle_texts are inserted into the
# haystack as individual documents; answers is the list of all required strings.

NEEDLE_TMPL = "One of the special magic numbers for {key} is: {value}."


# Each payload returns (needles, query, answers) where `needles` is a list of
# (text, is_gold) tuples. is_gold marks an *answer-bearing* needle (so
# gold_doc_indices later points only at those, not at distractor keys/chains).

def payload_niah_single(rng, key_pool, **kw):
    key = rng.choice(key_pool)
    value = rand_number(rng)
    needles = [(NEEDLE_TMPL.format(key=key, value=value), True)]
    query = (f"What is the special magic number for {key} mentioned in the "
             f"provided text?")
    return needles, query, [value]


def payload_niah_multikey(rng, key_pool, num_keys=4, **kw):
    keys = rng.sample(key_pool, num_keys)
    gold_key = rng.choice(keys)  # ask about exactly one of the planted keys
    needles, gold_value = [], None
    for k in keys:
        v = rand_number(rng)
        if k == gold_key:
            gold_value = v
        needles.append((NEEDLE_TMPL.format(key=k, value=v), k == gold_key))
    rng.shuffle(needles)
    query = (f"What is the special magic number for {gold_key} mentioned in "
             f"the provided text?")
    return needles, query, [gold_value]


def payload_niah_multivalue(rng, key_pool, num_values=4, **kw):
    key = rng.choice(key_pool)
    values = [rand_number(rng) for _ in range(num_values)]
    needles = [(NEEDLE_TMPL.format(key=key, value=v), True) for v in values]
    rng.shuffle(needles)
    query = (f"What are all the special magic numbers for {key} mentioned in "
             f"the provided text?")
    return needles, query, list(values)


def payload_niah_multiquery(rng, key_pool, num_queries=4, **kw):
    keys = rng.sample(key_pool, num_queries)
    needles, values = [], []
    for k in keys:
        v = rand_number(rng)
        values.append(v)
        needles.append((NEEDLE_TMPL.format(key=k, value=v), True))
    rng.shuffle(needles)
    key_list = ", ".join(keys)
    query = (f"What are the special magic numbers for the following keys "
             f"mentioned in the provided text: {key_list}?")
    return needles, query, list(values)


def payload_vt(rng, key_pool, num_chains=4, num_hops=4, **kw):
    """Variable tracking: build `num_chains` assignment chains; the gold chain's
    variables all resolve to `target_value`. Each statement is its own needle;
    only the gold chain's statements are marked gold."""
    needles, gold_vars = [], []
    target_value = rand_number(rng, digits=6)
    used = set()

    def fresh_var():
        while True:
            v = rand_var(rng)
            if v not in used:
                used.add(v)
                return v

    for c in range(num_chains):
        is_gold = (c == 0)
        root_val = target_value if is_gold else rand_number(rng, digits=6)
        chain_vars = [fresh_var()]
        needles.append((f"VAR {chain_vars[0]} = {root_val}", is_gold))
        for _ in range(num_hops):
            nxt = fresh_var()
            needles.append((f"VAR {nxt} = {chain_vars[-1]}", is_gold))
            chain_vars.append(nxt)
        if is_gold:
            gold_vars = chain_vars
    rng.shuffle(needles)
    query = (f"Find all variables that are assigned the value {target_value}, "
             f"directly or through a chain of assignments.")
    return needles, query, list(gold_vars)


NEEDLE_PAYLOADS = {
    "niah_single": payload_niah_single,
    "niah_multikey": payload_niah_multikey,
    "niah_multivalue": payload_niah_multivalue,
    "niah_multiquery": payload_niah_multiquery,
    "vt": payload_vt,
}


# ── Haystack assembly (needle family) ──────────────────────────────────────────

def _noise_chunk(rng):
    """A haystack 'document': a few repeats of the noise sentence."""
    reps = rng.randint(1, 3)
    return " ".join([NOISE_SENTENCE] * reps)


def assemble_needle_example(needles, query, answers, subtask, args, rng,
                            tokenizer=None):
    """Insert needles among haystack chunks per the configured length mode.

    `needles` is a list of (text, is_gold) tuples. The returned gold_doc_indices
    point only at the answer-bearing (is_gold) needles, so distractor keys/chains
    are not mislabeled as gold (important for gold-gradient masking)."""
    if args.length_mode == "num_docs":
        n_noise = max(args.num_docs - len(needles), 1)
        haystack = [_noise_chunk(rng) for _ in range(n_noise)]
    else:  # target token length: grow haystack until rendered context ~target
        haystack = []
        needle_tokens = sum(_tok_len(tokenizer, t) for t, _ in needles)
        budget = args.target_length - needle_tokens - args.length_margin
        acc = 0
        while acc < budget:
            chunk = _noise_chunk(rng)
            haystack.append(chunk)
            acc += _tok_len(tokenizer, chunk)

    # Build tagged records (haystack chunks are non-gold), then splice the
    # needles into random positions among the haystack so identity is preserved.
    docs = [{"text": c, "gold": False} for c in haystack]
    positions = sorted(rng.sample(range(len(docs) + len(needles)), len(needles)))
    for pos, (text, is_gold) in zip(positions, needles):
        docs.insert(pos, {"text": text, "gold": is_gold})
    gold_idx = [i for i, d in enumerate(docs) if d["gold"]]

    return {
        "documents": [{"title": None, "text": d["text"]} for d in docs],
        "queries": [query],
        "answers": list(answers),
        "gold_doc_indices": sorted(gold_idx),
        "source": f"ruler_{subtask}",
        "_task": "ruler",
        "_meta": _meta(subtask, args, n_docs=len(docs), n_needles=len(needles)),
    }


# ── Word-extraction family (CWE / FWE) ─────────────────────────────────────────

def _chunk_words(words, per_chunk=30):
    """Render a flat word list as documents of `per_chunk` words each."""
    return [{"title": None, "text": " ".join(words[i:i + per_chunk])}
            for i in range(0, len(words), per_chunk)]


def assemble_cwe_example(args, rng, vocab, tokenizer=None):
    """Common-words extraction: `num_cw` common words appear `freq_cw` times;
    many distinct uncommon words each appear `freq_ucw` times. Report the common
    words. Uncommon filler is drawn from the unbounded pseudo-word pool, so each
    uncommon word stays rare (exactly `freq_ucw`) no matter how long the context
    — the common words are therefore always strictly the most frequent."""
    common = rng.sample(vocab, args.num_cw)
    words = [w for w in common for _ in range(args.freq_cw)]
    pool = pseudo_words(rng, avoid=common)
    if args.length_mode == "target" and tokenizer is not None:
        budget = args.target_length - args.length_margin
        acc = _tok_len(tokenizer, " ".join(words))
        while acc < budget:  # add distinct uncommon words in batches
            batch = [w for _ in range(64) for w in [next(pool)] * args.freq_ucw]
            words += batch
            acc += _tok_len(tokenizer, " ".join(batch))
    else:
        n_uncommon = max(args.num_docs, args.num_cw + 1)
        for _ in range(n_uncommon):
            words += [next(pool)] * args.freq_ucw
    rng.shuffle(words)
    docs = _chunk_words(words)
    query = (f"What are the {args.num_cw} most common words in the list of "
             f"words above? List them separated by commas.")
    return {
        "documents": docs,
        "queries": [query],
        "answers": list(common),
        "gold_doc_indices": [],
        "source": "ruler_cwe",
        "_task": "ruler",
        "_meta": _meta("cwe", args, n_docs=len(docs), n_words=len(words)),
    }


def assemble_fwe_example(args, rng, vocab, tokenizer=None):
    """Frequent-words extraction: word frequencies follow a Zipfian (zeta) law;
    report the `top_k` most frequent words. In target mode the word list is
    grown (sampling from the Zipf head) until it reaches the token budget — the
    Zipf head stays dominant at any length, so the top-k answer is unambiguous."""
    n_distinct = min(len(vocab), max(args.top_k + 5, 30))
    chosen = rng.sample(vocab, n_distinct)
    # Zipfian weights 1/rank^alpha over a randomly ranked vocab.
    ranks = list(range(1, n_distinct + 1))
    rng.shuffle(chosen)
    weights = [1.0 / (r ** args.fwe_alpha) for r in ranks]
    counts = {w: 1 for w in chosen}  # ensure every word appears at least once
    if args.length_mode == "target" and tokenizer is not None:
        budget = args.target_length - args.length_margin
        acc = _tok_len(tokenizer, " ".join(chosen))
        while acc < budget:  # add in batches to bound tokenizer calls
            batch = rng.choices(chosen, weights=weights, k=256)
            for w in batch:
                counts[w] += 1
            acc += _tok_len(tokenizer, " ".join(batch))
    else:
        total_words = max(args.num_docs * 30, n_distinct * 4)
        picks = rng.choices(chosen, weights=weights,
                            k=max(total_words - n_distinct, 0))
        for w in picks:
            counts[w] += 1
    words = []
    for w, c in counts.items():
        words += [w] * c
    rng.shuffle(words)
    docs = _chunk_words(words)
    top = sorted(counts, key=lambda w: counts[w], reverse=True)[:args.top_k]
    query = (f"What are the {args.top_k} most frequently appearing words in the "
             f"list of words above? List them separated by commas.")
    return {
        "documents": docs,
        "queries": [query],
        "answers": list(top),
        "gold_doc_indices": [],
        "source": "ruler_fwe",
        "_task": "ruler",
        "_meta": _meta("fwe", args, n_docs=len(docs), n_words=len(words)),
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _tok_len(tokenizer, text):
    if tokenizer is None:
        # Rough char/4 fallback so num_docs callers never need a tokenizer.
        return max(1, len(text) // 4)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _meta(subtask, args, **extra):
    m = {"ruler_subtask": subtask, "length_mode": args.length_mode}
    if args.length_mode == "target":
        m["target_length"] = args.target_length
    else:
        m["num_docs"] = args.num_docs
    m.update(extra)
    return m


def build_split(subtask, n, args, rng, key_pool, vocab, tokenizer):
    out = []
    for _ in range(n):
        if subtask in NEEDLE_PAYLOADS:
            needles, query, answers = NEEDLE_PAYLOADS[subtask](
                rng, key_pool,
                num_keys=args.num_keys, num_values=args.num_values,
                num_queries=args.num_queries,
                num_chains=args.num_chains, num_hops=args.num_hops)
            out.append(assemble_needle_example(
                needles, query, answers, subtask, args, rng, tokenizer))
        elif subtask == "cwe":
            out.append(assemble_cwe_example(args, rng, vocab, tokenizer))
        elif subtask == "fwe":
            out.append(assemble_fwe_example(args, rng, vocab, tokenizer))
        else:
            raise ValueError(f"unknown subtask {subtask}")
    return out


def length_tag(args):
    return (f"L{args.target_length}" if args.length_mode == "target"
            else f"n{args.num_docs}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True, choices=sorted(
        list(NEEDLE_PAYLOADS) + ["cwe", "fwe"]))
    ap.add_argument("--length-mode", choices=["target", "num_docs"],
                    default="num_docs")
    ap.add_argument("--target-length", type=int, default=8192,
                    help="Token budget for the rendered context (target mode).")
    ap.add_argument("--length-margin", type=int, default=512,
                    help="Tokens reserved for instruction/query/answer (target mode).")
    ap.add_argument("--num-docs", type=int, default=50,
                    help="Haystack chunk count (num_docs mode).")
    ap.add_argument("--tokenizer", type=str, default="",
                    help="HF tokenizer id for target mode (e.g. Qwen/Qwen2.5-0.5B).")
    # Per-subtask knobs
    ap.add_argument("--num-keys", type=int, default=4, help="mk-niah distractor keys.")
    ap.add_argument("--num-values", type=int, default=4, help="mv-niah values.")
    ap.add_argument("--num-queries", type=int, default=4, help="mq-niah keys.")
    ap.add_argument("--num-chains", type=int, default=4, help="vt chains.")
    ap.add_argument("--num-hops", type=int, default=4, help="vt hops per chain.")
    ap.add_argument("--num-cw", type=int, default=10, help="cwe common words.")
    ap.add_argument("--freq-cw", type=int, default=30, help="cwe common freq.")
    ap.add_argument("--freq-ucw", type=int, default=3, help="cwe uncommon freq.")
    ap.add_argument("--top-k", type=int, default=3, help="fwe top words to report.")
    ap.add_argument("--fwe-alpha", type=float, default=2.0, help="fwe Zipf exponent.")
    # Splits
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--disjoint-vocab", action="store_true",
                    help="Partition needle key/value vocab so train & test keys "
                         "never overlap (stricter in-domain probe).")
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tokenizer = None
    if args.length_mode == "target":
        if not args.tokenizer:
            ap.error("--length-mode target requires --tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # In-domain disjointness: independent RNG streams; optional vocab partition.
    if args.disjoint_vocab:
        shuf = random.Random(args.seed).sample(WORDS, len(WORDS))
        half = len(shuf) // 2
        train_keys, eval_keys = shuf[:half], shuf[half:]
        train_vocab, eval_vocab = train_keys, eval_keys
    else:
        train_keys = eval_keys = train_vocab = eval_vocab = WORDS

    tag = f"{args.subtask}_{length_tag(args)}"
    for label, n, seed, kp, vc in [
        ("train", args.num_train, args.seed, train_keys, train_vocab),
        ("eval", args.num_eval, args.seed + 1, eval_keys, eval_vocab),
    ]:
        rng = random.Random(seed)
        rows = build_split(args.subtask, n, args, rng, kp, vc, tokenizer)
        path = f"{args.output_dir}/ruler_{tag}_{label}.jsonl"
        save_jsonl(path, rows)
        nd = len(rows[0]["documents"]) if rows else 0
        na = len(rows[0]["answers"]) if rows else 0
        print(f"{label}: {len(rows)} examples, n_docs~{nd}, n_answers={na} -> {path}")


if __name__ == "__main__":
    main()
