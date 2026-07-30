"""Generate long-context contradiction detection data from SNLI.

Each example is a corpus of N numbered claims with K contradiction pairs hidden among
filler claims. The model must identify all contradicting pairs by claim ID.

Usage:
    python scripts/generate_contradiction_data.py --num-claims 20 --num-contradictions 3
    python scripts/generate_contradiction_data.py --num-claims 100 --num-contradictions 5 --num-train 5000
"""

import argparse
import json
import random
from datasets import load_dataset
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats

INSTRUCTION = (
    "Given the following corpus of numbered claims, identify all pairs of claims "
    "that contradict each other. A pair of claims is contradictory if they cannot "
    "both be true at the same time.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of two "
    "claim IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no contradicting pairs, output: []"
)

def build_example(contradiction_pairs, filler_claims, num_claims, rng):
    claims = []
    pair_indices = []
    for premise, hypothesis in contradiction_pairs:
        a, b = len(claims), len(claims) + 1
        claims.extend([premise, hypothesis])
        pair_indices.append((a, b))

    num_fillers = num_claims - len(claims)
    fillers = rng.sample(filler_claims, min(num_fillers, len(filler_claims)))
    while len(fillers) < num_fillers:
        fillers.append(rng.choice(filler_claims))
    claims.extend(fillers)

    # Shuffle and build ID mapping (1-indexed)
    order = list(range(len(claims)))
    rng.shuffle(order)
    old_to_new = {old: new + 1 for new, old in enumerate(order)}

    answer = sorted([sorted([old_to_new[a], old_to_new[b]]) for a, b in pair_indices])
    input_text = "\n".join(f"Claim {i+1}: {claims[order[i]]}" for i in range(len(order)))

    return {"instruction": INSTRUCTION, "input": input_text, "output": json.dumps(answer)}


def main():
    parser = argparse.ArgumentParser(description="Generate contradiction detection data")
    parser.add_argument("--num-claims", type=int, default=20)
    parser.add_argument("--num-contradictions", type=int, default=3)
    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.num_contradictions * 2 <= args.num_claims
    rng = random.Random(args.seed)

    print("Loading stanfordnlp/snli...")
    ds = load_dataset("stanfordnlp/snli")

    # Build pools
    train_contras, train_fillers = [], []
    for ex in ds["train"]:
        if ex["label"] == 2:
            train_contras.append((ex["premise"], ex["hypothesis"]))
        elif ex["label"] in (0, 1):
            train_fillers.append(ex["premise"])
    train_fillers = list(set(train_fillers))

    eval_contras, eval_fillers = [], []
    for split in ["validation", "test"]:
        for ex in ds[split]:
            if ex["label"] == 2:
                eval_contras.append((ex["premise"], ex["hypothesis"]))
            elif ex["label"] in (0, 1):
                eval_fillers.append(ex["premise"])
    eval_fillers = list(set(eval_fillers))

    print(f"  Train: {len(train_contras)} contradictions, {len(train_fillers)} fillers")
    print(f"  Eval:  {len(eval_contras)} contradictions, {len(eval_fillers)} fillers")

    for pool_contras, pool_fillers in [(train_contras, train_fillers), (eval_contras, eval_fillers)]:
        rng.shuffle(pool_contras)
        rng.shuffle(pool_fillers)

    tag = f"n{args.num_claims}_k{args.num_contradictions}"
    k = args.num_contradictions

    for label, count, contras, fillers in [
        ("Train", args.num_train, train_contras, train_fillers),
        ("Eval", args.num_eval, eval_contras, eval_fillers),
    ]:
        if count == 0:
            continue
        examples = [
            build_example(rng.sample(contras, min(k, len(contras))), fillers, args.num_claims, rng)
            for _ in range(count)
        ]
        suffix = "train" if label == "Train" else "eval"
        path = f"{args.output_dir}/contradiction_{suffix}_{tag}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, label, path)

    # Show sample
    sample_path = f"{args.output_dir}/contradiction_train_{tag}.jsonl" if args.num_train else f"{args.output_dir}/contradiction_eval_{tag}.jsonl"
    from corpus_reasoning.lib.io import load_jsonl
    samples = load_jsonl(sample_path)[:1]
    if samples:
        ex = samples[0]
        lines = ex["input"].split("\n")
        print(f"\n=== Sample ({len(lines)} claims) ===")
        for line in lines[:5]:
            print(f"  {line[:100]}")
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 5} more)")
        print(f"  output: {ex['output']}")


if __name__ == "__main__":
    main()
