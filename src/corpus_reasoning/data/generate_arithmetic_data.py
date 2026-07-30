"""Generate simple arithmetic training/eval data for sanity-checking full fine-tuning.

Generates addition and subtraction problems with 1-3 digit numbers.
Easy to evaluate with exact match.

Usage:
    python scripts/generate_arithmetic_data.py --num-train 1000 --num-eval 200
    python scripts/generate_arithmetic_data.py --num-train 5000 --ops add,sub,mul
"""

import argparse
import random
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats

INSTRUCTION = "Solve the following math problem. Give only the numerical answer."

OPS = {
    "add": ("+", lambda a, b: a + b),
    "sub": ("-", lambda a, b: a - b),
    "mul": ("*", lambda a, b: a * b),
}

def generate_problem(ops, rng, max_val=999):
    """Generate a single arithmetic problem."""
    op_name = rng.choice(ops)
    symbol, func = OPS[op_name]
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    # Avoid negative results for subtraction
    if op_name == "sub" and a < b:
        a, b = b, a
    result = func(a, b)
    return {
        "instruction": INSTRUCTION,
        "input": f"{a} {symbol} {b} = ?",
        "output": str(result),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic data")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--ops", type=str, default="add,sub",
                        help="Comma-separated ops: add,sub,mul")
    parser.add_argument("--max-val", type=int, default=999,
                        help="Maximum value for operands")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ops = [o.strip() for o in args.ops.split(",")]
    for o in ops:
        assert o in OPS, f"Unknown op: {o}. Choose from: {list(OPS.keys())}"

    rng = random.Random(args.seed)
    tag = "_".join(ops)

    for label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count == 0:
            continue
        # Generate unique problems
        seen = set()
        examples = []
        while len(examples) < count:
            ex = generate_problem(ops, rng, args.max_val)
            key = ex["input"]
            if key not in seen:
                seen.add(key)
                examples.append(ex)

        path = f"{args.output_dir}/arithmetic_{label}_{tag}_{count}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, label.capitalize(), path)

        if examples:
            for ex in examples[:3]:
                print(f"  {ex['input']}  ->  {ex['output']}")


if __name__ == "__main__":
    main()
