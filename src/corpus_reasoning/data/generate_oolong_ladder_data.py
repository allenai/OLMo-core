"""Generate an OOLONG *ladder* training set with continuous per-example lengths.

The shipped ``oolong-synth`` benchmark only provides ~100 pre-built context
windows per fixed context-length bucket (1K, 2K, ... 4M). That is far too few
rows -- and far too quantized -- to build a 20K-instance training ladder with
*continuous* lengths up to 40960 tokens (the format the sibling single-task
ladders -- rerank / nq / outlier / contradiction -- use, where per-example
length is sampled continuously by varying the number of context items).

This script rebuilds OOLONG windows from scratch so length is a free parameter:

  1. Pool every labeled item (``Date: D || User: U || Instance: T || Label: L``)
     out of ``oolong-synth`` (all splits, buckets <= --pool-max-ctx), grouped by
     ``dataset`` (spam / trec_coarse / ...). Dedup, and tokenize each item once
     to get a per-item token count (Qwen3 tokenizer).
  2. Per example: sample a target token budget ``T ~ U[--len-min, --len-max]``,
     greedily draw items until their cumulative token count reaches ``T``, pick
     one of 9 exactly-recomputable task variants (counting / user / timeline),
     and **recompute the aggregate gold over the drawn items**. This mirrors the
     NQ generator's continuous ``--num-docs-min/--num-docs-max`` sampling, but the
     scaling axis is the number of OOLONG items rather than retrieval docs.

Emits unified-format rows identical in shape to ``generate_oolong_data.py``::

    documents = [{"text": preamble + "\\n\\n" + "\\n".join(unlabeled item lines)}]
    queries   = [question]                  (carries its own format spec)
    answers   = [bare gold value string]    (cot_mode=label -> target IS this)
    _meta     = {answer_type, task_group, gold_list, context_len(token est), dataset}

Scored / tokenized downstream by ``--task oolong`` (the same path the original
oolong rows use). The 9 variants are restricted to those whose gold is exactly
computable from per-item (date, user, label) with no inferred thresholds:

  counting:  most-frequent label / count-of-label / pairwise comparison
  user:      most-frequent user / per-user most-frequent label / per-user count
             / per-user pairwise comparison
  timeline:  #dates represented exactly n times / per-month most-frequent label

Needs HuggingFace ``datasets`` + a fast Qwen3 tokenizer.

Usage:
    python scripts/data/generate_oolong_ladder_data.py \
        --num-examples 20000 --len-min 300 --len-max 38000 \
        --num-shards 8 --shard-index 0 --output-dir /data/.../oolong
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime

# ── the 9 exactly-recomputable variants ─────────────────────────────────────
# Each is (task_group, answer_type, builder-key). The builder recomputes the
# gold over the drawn items and returns (question, gold_str) or None if the
# draw is degenerate (a tie at the argmax, an empty subset, etc.) -> retry.
VARIANTS = [
    ("counting", "ANSWER_TYPE.LABEL", "count_mostfreq_label"),
    ("counting", "ANSWER_TYPE.NUMERIC", "count_numeric"),
    ("counting", "ANSWER_TYPE.COMPARISON", "count_comparison"),
    ("user", "ANSWER_TYPE.USER", "user_mostfreq_user"),
    ("user", "ANSWER_TYPE.LABEL", "user_mostfreq_label"),
    ("user", "ANSWER_TYPE.NUMERIC", "user_numeric"),
    ("user", "ANSWER_TYPE.COMPARISON", "user_comparison"),
    ("timeline", "ANSWER_TYPE.NUMERIC", "timeline_repr_n"),
    ("timeline", "ANSWER_TYPE.LABEL", "timeline_month_label"),
]

_CMP = {1: "more common than", -1: "less common than", 0: "same frequency as"}


def _cmp_phrase(a_count, b_count):
    return _CMP[(a_count > b_count) - (a_count < b_count)]


def _label_list_str(labels):
    return ", ".join(labels)


def _argmax_unique(counter):
    """Return the single most-common key, or None on an empty/ tied counter."""
    if not counter:
        return None
    top = counter.most_common(2)
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def build_variant(key, items, canonical_labels, rng):
    """Recompute the gold for one variant over `items`.

    items: list of dicts {user, label, month (1-12), date_str}.
    Returns (question, gold_str) or None if degenerate (caller retries).
    """
    labels_present = [l for l in canonical_labels if any(it["label"] == l for it in items)]
    if len(labels_present) < 1:
        return None

    if key == "count_mostfreq_label":
        gold = _argmax_unique(Counter(it["label"] for it in items))
        if gold is None:
            return None
        q = ("In the above data, which of the labels is the most common? Give your "
             f"final answer in the form 'Label: answer' where answer is one of the "
             f"labels: {_label_list_str(labels_present)}.")
        return q, gold

    if key == "count_numeric":
        L = rng.choice(labels_present)
        gold = sum(1 for it in items if it["label"] == L)
        q = (f"In the above data, how many data points should be classified as label "
             f"'{L}'? Give your final answer in the form 'Answer: number'.")
        return q, str(gold)

    if key == "count_comparison":
        if len(labels_present) < 2:
            return None
        a, b = rng.sample(labels_present, 2)
        ca = sum(1 for it in items if it["label"] == a)
        cb = sum(1 for it in items if it["label"] == b)
        q = (f"In the above data, is label '{a}' more common, less common, or the same "
             f"frequency as label '{b}'? Give your final answer in the form "
             f"'Answer: {a} is [X] {b}', where [X] is 'more common than', "
             f"'less common than', or 'same frequency as'.")
        return q, _cmp_phrase(ca, cb)

    if key == "user_mostfreq_user":
        gold = _argmax_unique(Counter(it["user"] for it in items))
        if gold is None:
            return None
        q = ("In the above data, which user is represented most often? Give your final "
             "answer in the form 'User: [X]', where [X] is the user ID.")
        return q, str(gold)

    # user-subset variants: pick a user, restrict to their items
    if key in ("user_mostfreq_label", "user_numeric", "user_comparison"):
        user_counts = Counter(it["user"] for it in items)
        # prefer users with a few items so the subset question is non-trivial
        cand = [u for u, c in user_counts.items() if c >= 2] or list(user_counts)
        U = rng.choice(cand)
        sub = [it for it in items if it["user"] == U]
        sub_labels = [l for l in canonical_labels if any(it["label"] == l for it in sub)]
        if key == "user_mostfreq_label":
            gold = _argmax_unique(Counter(it["label"] for it in sub))
            if gold is None:
                return None
            q = (f"For the following question, only consider the subset of instances "
                 f"that are associated with user IDs {U}. Among instances associated "
                 f"with these users, which of the labels is the most common? Give your "
                 f"final answer in the form 'Label: answer' where answer is one of the "
                 f"labels: {_label_list_str(sub_labels)}.")
            return q, gold
        if key == "user_numeric":
            L = rng.choice(sub_labels)
            gold = sum(1 for it in sub if it["label"] == L)
            q = (f"For the following question, only consider the subset of instances "
                 f"that are associated with user IDs {U}. Among instances associated "
                 f"with these users, how many data points should be classified as label "
                 f"'{L}'? Give your final answer in the form 'Answer: number'.")
            return q, str(gold)
        if key == "user_comparison":
            if len(sub_labels) < 2:
                return None
            a, b = rng.sample(sub_labels, 2)
            ca = sum(1 for it in sub if it["label"] == a)
            cb = sum(1 for it in sub if it["label"] == b)
            q = (f"For the following question, only consider the subset of instances "
                 f"that are associated with user IDs {U}. Among instances associated "
                 f"with these users, is label '{a}' more common, less common, or the "
                 f"same frequency as label '{b}'? Give your final answer in the form "
                 f"'Answer: {a} is [X] {b}', where [X] is 'more common than', "
                 f"'less common than', or 'same frequency as'.")
            return q, _cmp_phrase(ca, cb)

    if key == "timeline_repr_n":
        n = rng.choice([1, 2])
        date_counts = Counter(it["date_str"] for it in items)
        gold = sum(1 for _d, c in date_counts.items() if c == n)
        plural = "time" if n == 1 else "times"
        q = (f"In the above data, how many dates are represented exactly {n} {plural}? "
             f"Give your final answer in the form 'Answer: [X]', where [X] is the number "
             f"of dates represented exactly {n} {plural}.")
        return q, str(gold)

    if key == "timeline_month_label":
        months_present = sorted({it["month"] for it in items})
        if not months_present:
            return None
        m = rng.choice(months_present)
        month_name = datetime(2000, m, 1).strftime("%B")
        sub = [it for it in items if it["month"] == m]
        sub_labels = [l for l in canonical_labels if any(it["label"] == l for it in sub)]
        gold = _argmax_unique(Counter(it["label"] for it in sub))
        if gold is None:
            return None
        q = (f"For the following question, only consider the subset of instances that "
             f"occur in {month_name} of any year. Among instances occuring in "
             f"{month_name}, which of the labels is the most common? Give your final "
             f"answer in the form 'Label: answer' where answer is one of the labels: "
             f"{_label_list_str(sub_labels)}.")
        return q, gold

    raise ValueError(f"unknown variant key {key}")


def parse_item_line(line):
    """`Date: D || User: U || Instance: T || Label: L` -> (unlabeled_line, dict) or None."""
    if not line.startswith("Date:"):
        return None
    head, _sep, label = line.rpartition(" || Label: ")
    if not _sep:
        return None
    m = re.match(r"^Date: (.*?) \|\| User: (.*?) \|\| Instance: ", head)
    if not m:
        return None
    date_str, user = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y")
    except ValueError:
        return None
    return head, {"user": user, "label": label, "date_str": date_str, "month": dt.month}


def build_pools(args):
    """Pool deduped items per dataset, + canonical label order + preamble template."""
    from datasets import load_dataset

    ds = load_dataset(args.hf_name)
    splits = list(ds.keys())
    print(f"{args.hf_name} splits: {[(s, len(ds[s])) for s in splits]}")

    pools = defaultdict(list)          # dataset -> list[(unlabeled_line, meta)]
    seen = defaultdict(set)            # dataset -> set(unlabeled_line)
    preamble_tpl = {}                  # dataset -> template with {N}
    canonical_labels = {}             # dataset -> [label, ...]

    for split in splits:
        for r in ds[split]:
            dset = r["dataset"]
            if args.pool_max_ctx and (r.get("context_len") or 0) > args.pool_max_ctx:
                continue
            wl = r["context_window_text_with_labels"]
            idx = wl.find("\nDate:")
            if idx < 0:
                continue
            preamble = wl[:idx].rstrip()
            item_lines = [ln for ln in wl[idx + 1:].split("\n") if ln.startswith("Date:")]
            # preamble template: replace the (item-count) integer with {N}
            if dset not in preamble_tpl:
                n = len(item_lines)
                tpl = re.sub(rf"\b{n}\b", "{N}", preamble)
                preamble_tpl[dset] = tpl
            # canonical label order from a global MOST_FREQ/LABEL question
            if dset not in canonical_labels and "ANSWER_TYPE.LABEL" in (r.get("answer_type") or ""):
                q = r["question"]
                mm = re.search(r"one of the labels:\s*(.*?)\.\s*$", q)
                if mm and "only consider" not in q:
                    canonical_labels[dset] = [s.strip() for s in mm.group(1).split(",")]
            for ln in item_lines:
                parsed = parse_item_line(ln)
                if parsed is None:
                    continue
                unlabeled, meta = parsed
                if unlabeled in seen[dset]:
                    continue
                seen[dset].add(unlabeled)
                pools[dset].append((unlabeled, meta))

    # fallbacks for canonical label order
    for dset, items in pools.items():
        if dset not in canonical_labels:
            canonical_labels[dset] = sorted({m["label"] for _u, m in items})

    for dset in pools:
        print(f"  pool[{dset}]: {len(pools[dset])} unique items, "
              f"labels={canonical_labels[dset]}")
    return pools, preamble_tpl, canonical_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-name", default="oolongbench/oolong-synth")
    ap.add_argument("--num-examples", type=int, default=20000,
                    help="examples for THIS shard's slice of the total (see --num-shards)")
    ap.add_argument("--len-min", type=int, default=300,
                    help="min per-example item-budget in tokens (continuous lower bound)")
    ap.add_argument("--len-max", type=int, default=38000,
                    help="max per-example item-budget in tokens (continuous upper bound; "
                         "leave headroom under the 40960 tokenizer cap for the question + "
                         "chat template)")
    ap.add_argument("--pool-max-ctx", type=int, default=131072,
                    help="only pool items from windows with context_len <= this")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-retries", type=int, default=40,
                    help="degenerate-draw retries before falling back to count_numeric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of parallel shards (all share --seed)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--output-dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random((args.seed, args.shard_index).__hash__())

    pools, preamble_tpl, canonical_labels = build_pools(args)
    datasets_avail = sorted(pools)

    # tokenize each unique item once -> per-item token count
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    item_tokens = {}
    preamble_tok = {}
    for dset, items in pools.items():
        lines = [u for u, _m in items]
        # batch tokenize for speed
        B = 1024
        counts = []
        for i in range(0, len(lines), B):
            enc = tok(lines[i:i + B], add_special_tokens=False)
            counts.extend(len(x) for x in enc["input_ids"])
        item_tokens[dset] = counts
        # +1 token per item for the joining "\n"
        preamble_tok[dset] = len(tok(preamble_tpl[dset].format(N=999),
                                     add_special_tokens=False)["input_ids"])
        print(f"  [{dset}] median item tokens ~"
              f"{sorted(counts)[len(counts)//2]}, preamble~{preamble_tok[dset]}")

    QUESTION_RESERVE = 220   # tokens reserved for the question + chat template + answer
    examples = []
    n_fallback = 0
    n_skipped = 0
    group_counter = Counter()

    for _ in range(args.num_examples):
        dset = rng.choice(datasets_avail)
        items_pool = pools[dset]
        toks = item_tokens[dset]
        labels = canonical_labels[dset]
        npool = len(items_pool)

        target = rng.randint(args.len_min, args.len_max)
        budget = max(args.len_min, target - preamble_tok[dset] - QUESTION_RESERVE)

        # draw items (without replacement) until cumulative tokens reach budget
        order = list(range(npool))
        rng.shuffle(order)
        chosen_idx = []
        acc = 0
        for j in order:
            chosen_idx.append(j)
            acc += toks[j] + 1
            if acc >= budget:
                break
        if len(chosen_idx) < 4:
            # pad to a sane minimum
            for j in order[len(chosen_idx):]:
                chosen_idx.append(j)
                if len(chosen_idx) >= 4:
                    break

        items_meta = [items_pool[j][1] for j in chosen_idx]

        # pick a variant + recompute gold (retry on degenerate draws)
        result = None
        for attempt in range(args.max_retries):
            tg, atype, key = rng.choice(VARIANTS)
            built = build_variant(key, items_meta, labels, rng)
            if built is not None:
                result = (tg, atype, built)
                break
        if result is None:
            # guaranteed-defined fallback
            tg, atype, key = ("counting", "ANSWER_TYPE.NUMERIC", "count_numeric")
            built = build_variant(key, items_meta, labels, rng)
            if built is None:
                n_skipped += 1
                continue
            result = (tg, atype, built)
            n_fallback += 1

        tg, atype, (question, gold) = result
        N = len(chosen_idx)
        preamble = preamble_tpl[dset].format(N=N)
        lines = [items_pool[j][0] for j in chosen_idx]
        doc_text = preamble + "\n\n" + "\n".join(lines)
        ctx_tok_est = preamble_tok[dset] + sum(toks[j] + 1 for j in chosen_idx)

        examples.append({
            "documents": [{"text": doc_text}],
            "queries": [question],
            "answers": [gold],
            "gold_doc_indices": [],
            "source": f"oolong_{tg}",
            "_meta": {
                "answer_type": atype,
                "task_group": tg,
                "gold_list": [gold],
                "context_len": ctx_tok_est,
                "dataset": dset,
                "num_items": N,
            },
        })
        group_counter[(tg, atype)] += 1

    shard_tag = (f"_s{args.shard_index}of{args.num_shards}"
                 if args.num_shards > 1 else "")
    path = os.path.join(args.output_dir,
                        f"oolong_ladder_train{shard_tag}_{len(examples)}.jsonl")
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    ctx = sorted(ex["_meta"]["context_len"] for ex in examples)
    print(f"\nWROTE {len(examples)} examples -> {path}")
    if ctx:
        print(f"  ctx-token est: min={ctx[0]} p50={ctx[len(ctx)//2]} "
              f"p90={ctx[int(len(ctx)*0.9)]} max={ctx[-1]}")
    print(f"  variant dist: {dict(group_counter)}")
    print(f"  fallbacks={n_fallback} skipped={n_skipped}")


if __name__ == "__main__":
    main()
