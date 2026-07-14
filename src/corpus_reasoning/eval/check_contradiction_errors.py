"""Extract incorrect contradiction predictions for post-hoc analysis.

Loads eval results and the original eval data, extracts the actual claim text
for false positives (predicted but not gold) and false negatives (gold but not
predicted), and writes them to a JSONL file for manual or LLM-based review.

Usage:
    # From API eval results (picks a specific model key)
    python scripts/eval/check_contradiction_errors.py \
        --results outputs/contradiction_api_n100.json \
        --eval-data data/contradiction_eval_n100_k3.jsonl \
        --model-key gemini-2.5-flash

    # From vLLM eval results (no --model-key needed)
    python scripts/eval/check_contradiction_errors.py \
        --results outputs/contradiction_lora_n100_k3.json \
        --eval-data data/contradiction_eval_n100_k3.jsonl

    # Output defaults to outputs/contradiction_error_analysis.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl


def parse_claims(corpus_text: str) -> dict[int, str]:
    """Parse numbered claims from corpus text into {claim_id: claim_text}."""
    claims = {}
    for match in re.finditer(r'Claim (\d+): (.+?)(?=\nClaim \d+:|\Z)', corpus_text, re.DOTALL):
        claims[int(match.group(1))] = match.group(2).strip()
    return claims


def extract_errors(eval_examples, result_list):
    """Extract false positive and false negative pairs with claim text."""
    errors = []

    for i, (ex, res) in enumerate(zip(eval_examples, result_list)):
        gold_set = {tuple(sorted(p)) for p in res["gold_pairs"]}
        pred_set = {tuple(sorted(p)) for p in res["predicted_pairs"]}

        false_positives = pred_set - gold_set
        false_negatives = gold_set - pred_set

        if not false_positives and not false_negatives:
            continue

        claims = parse_claims(ex["input"])

        for pair in sorted(false_positives):
            a, b = pair
            errors.append({
                "example_idx": i,
                "error_type": "false_positive",
                "pair": list(pair),
                "claim_a": f"Claim {a}: {claims.get(a, '???')}",
                "claim_b": f"Claim {b}: {claims.get(b, '???')}",
                "all_gold_pairs": res["gold_pairs"],
            })

        for pair in sorted(false_negatives):
            a, b = pair
            errors.append({
                "example_idx": i,
                "error_type": "false_negative",
                "pair": list(pair),
                "claim_a": f"Claim {a}: {claims.get(a, '???')}",
                "claim_b": f"Claim {b}: {claims.get(b, '???')}",
                "all_gold_pairs": res["gold_pairs"],
            })

    return errors


def main():
    parser = argparse.ArgumentParser(description="Extract contradiction prediction errors for analysis")
    parser.add_argument("--results", type=str, required=True, help="Path to eval results JSON")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to original eval JSONL")
    parser.add_argument("--model-key", type=str, default=None,
                        help="Model key for API results (e.g. 'gemini-2.5-flash'). "
                             "Not needed for vLLM results.")
    parser.add_argument("--output", type=str, default="outputs/contradiction_error_analysis.jsonl")
    args = parser.parse_args()

    # Load eval data
    eval_examples = load_jsonl(args.eval_data)

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    # Extract the results list (handle both API and vLLM formats)
    if args.model_key:
        # API format: results[model_key]["results"]
        if args.model_key not in data["results"]:
            available = list(data["results"].keys())
            print(f"Error: model key '{args.model_key}' not found. Available: {available}")
            sys.exit(1)
        result_list = data["results"][args.model_key]["results"]
        source_label = args.model_key
    elif isinstance(data["results"], list):
        # vLLM format: results is a flat list
        result_list = data["results"]
        source_label = Path(args.results).stem
    else:
        # API format but no model key specified - use first key
        first_key = next(iter(data["results"]))
        result_list = data["results"][first_key]["results"]
        source_label = first_key
        print(f"No --model-key specified, using first: {first_key}")

    # Match eval examples to results (results may be a subset)
    max_examples = data.get("args", {}).get("max_examples") or data.get("args", {}).get("max_test_samples")
    if max_examples and len(eval_examples) > len(result_list):
        eval_examples = eval_examples[:len(result_list)]

    assert len(eval_examples) == len(result_list), \
        f"Mismatch: {len(eval_examples)} eval examples vs {len(result_list)} results"

    errors = extract_errors(eval_examples, result_list)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for err in errors:
            f.write(json.dumps(err) + "\n")

    # Summary
    fp = sum(1 for e in errors if e["error_type"] == "false_positive")
    fn = sum(1 for e in errors if e["error_type"] == "false_negative")
    n_examples_with_errors = len({e["example_idx"] for e in errors})

    print(f"\nSource: {source_label}")
    print(f"Examples with errors: {n_examples_with_errors}/{len(result_list)}")
    print(f"Total errors: {len(errors)} ({fp} false positives, {fn} false negatives)")
    print(f"\nSaved to {args.output}")

    # Print a few samples
    print(f"\n{'='*70}")
    print("Sample errors:")
    print(f"{'='*70}")
    for err in errors[:10]:
        tag = "FP" if err["error_type"] == "false_positive" else "FN"
        print(f"\n  [{tag}] Example {err['example_idx']}, pair {err['pair']}")
        print(f"    {err['claim_a']}")
        print(f"    {err['claim_b']}")


if __name__ == "__main__":
    main()
