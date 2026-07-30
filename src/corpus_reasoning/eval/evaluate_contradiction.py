"""Evaluate contradiction detection using vLLM.

Metrics: pair-level precision/recall/F1, exact match, parse rate.

Usage:
    python scripts/evaluate_contradiction.py --eval-data data/contradiction_eval_n100_k3.jsonl
    python scripts/evaluate_contradiction.py --eval-data data/contradiction_eval_n100_k3.jsonl --lora-path outputs/contradiction-lora
"""

import argparse
import json
import random
import re
from vllm import SamplingParams
from corpus_reasoning.lib.io import load_jsonl, save_results
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.lib.vllm_utils import add_vllm_args, load_model, run_inference


def parse_pairs(text: str) -> list[list[int]] | None:
    """Extract list of integer pairs from model output."""
    text = text.strip()

    # Try JSON parse
    for candidate in [text, re.search(r'\[[\s\S]*\]', text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [sorted([int(p[0]), int(p[1])]) for p in parsed if isinstance(p, list) and len(p) == 2]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Regex fallback
    matches = re.findall(r'[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]', text)
    if matches:
        return [sorted([int(a), int(b)]) for a, b in matches]

    return [] if text in ("[]", "") else None


def pair_metrics(predicted, gold):
    pred_set, gold_set = {tuple(p) for p in predicted}, {tuple(p) for p in gold}
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "exact_match": float(pred_set == gold_set)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate contradiction detection")
    add_vllm_args(parser)
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--query-position", type=str, default="both",
                        choices=["before", "after", "both"])
    parser.set_defaults(max_tokens=200, max_model_len=128000,
                        output_file="outputs/eval_results/contradiction_eval.json")
    args = parser.parse_args()

    examples = load_jsonl(args.eval_data)
    if args.max_test_samples and len(examples) > args.max_test_samples:
        random.seed(42)
        examples = random.sample(examples, args.max_test_samples)
    print(f"Loaded {len(examples)} examples from {args.eval_data}")

    llm, lora_request = load_model(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [build_prompt(ex, task="contradiction",
                            query_position=args.query_position,
                            use_alpaca=True)[0]
               for ex in examples]

    print(f"\nRunning inference ({'lora' if lora_request else 'base'})...")
    responses = run_inference(llm, prompts, sampling_params, lora_request)

    results, parse_failures = [], 0
    for ex, resp in zip(examples, responses):
        gold = [sorted([int(p[0]), int(p[1])]) for p in ex["gold_doc_indices"]]
        predicted = parse_pairs(resp)
        if predicted is None:
            parse_failures += 1
            predicted = []
        m = pair_metrics(predicted, gold)
        results.append({"gold_pairs": gold, "predicted_pairs": predicted, "raw_output": resp.strip()[:500], **m})

    n = len(results)
    avg = {k: sum(r[k] for r in results) / n for k in ["precision", "recall", "f1", "exact_match"]}
    avg["parse_rate"] = (n - parse_failures) / n

    print(f"\n{'='*60}\nResults — {n} examples\n{'='*60}")
    for k in ["parse_rate", "precision", "recall", "f1", "exact_match"]:
        print(f"  {k:<20} {avg[k]:.1%}")

    print(f"\n  Samples:")
    for r in results[:3]:
        print(f"    Gold: {r['gold_pairs']}  Pred: {r['predicted_pairs']}  F1={r['f1']:.2f}")

    save_results(args.output_file, {"args": vars(args), "summary": avg, "results": results})
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
