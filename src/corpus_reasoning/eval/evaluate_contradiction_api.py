"""Evaluate contradiction detection using Gemini/OpenAI APIs via llm_request_client.

Uses the same metrics as evaluate_contradiction.py but calls external LLMs
instead of local vLLM models.

Supports ablations:
  --hint: tell the model exactly how many contradiction pairs exist (K)
  --instruction-after: put the task instruction after the corpus instead of before

Usage:
    # Default: instruction before corpus, no hint
    python scripts/evaluate_contradiction_api.py --eval-data data/contradiction_eval_n100_k3.jsonl --models gemini-2.5-flash

    # With hint + instruction after corpus
    python scripts/evaluate_contradiction_api.py --eval-data data/contradiction_eval_n100_k3.jsonl --hint --instruction-after

    # Run all 4 ablation configs
    python scripts/evaluate_contradiction_api.py --eval-data data/contradiction_eval_n100_k3.jsonl --ablate-all
"""

import argparse
import json
from pathlib import Path

from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient, PRICING
from corpus_reasoning.lib.io import load_jsonl, save_results
from corpus_reasoning.eval.evaluate_contradiction import parse_pairs, pair_metrics


TASK_INSTRUCTION = (
    "Given the following corpus of numbered claims, identify all pairs of claims "
    "that contradict each other. A pair of claims is contradictory if they cannot "
    "both be true at the same time. There are contradicting pairs in this corpus.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of two "
    "claim IDs. For example: [[1, 4], [3, 7]]"
)

HINT_TEMPLATE = "\n\nThere are exactly {k} contradicting pairs in this corpus."

RESPONSE_PREFIX = "\n\nAnswer: ["


def build_prompt(example, hint=False, instruction_after=False):
    """Build prompt with ablation options for hint and instruction placement."""
    corpus = example["input"]
    k = len(json.loads(example["output"]))

    instruction = TASK_INSTRUCTION
    if hint:
        instruction += HINT_TEMPLATE.format(k=k)

    if instruction_after:
        prompt = f"{corpus}\n\n{instruction}"
    else:
        prompt = f"{instruction}\n\n{corpus}"

    # Force the model to start producing a list
    prompt += RESPONSE_PREFIX

    return prompt


def estimate_costs(prompts, models, max_output_tokens):
    avg_input_chars = sum(len(p) for p in prompts) / len(prompts)
    avg_input_tokens = avg_input_chars / 4
    total_input_tokens = avg_input_tokens * len(prompts)

    print(f"  Avg input: ~{avg_input_tokens:.0f} tokens/example")
    print(f"  Total input: ~{total_input_tokens:.0f} tokens ({len(prompts)} examples)")
    print(f"  Output budget: {max_output_tokens} tokens/example\n")

    for model in models:
        if model in PRICING:
            p = PRICING[model]
            input_cost = (total_input_tokens / 1e6) * p["input"]
            output_cost = (len(prompts) * max_output_tokens / 1e6) * p["output"]
            print(f"  {model}: est. ${input_cost + output_cost:.4f} "
                  f"(input: ${input_cost:.4f}, output: ${output_cost:.4f})")


def run_eval(client, model, prompts, gold_pairs, max_output_tokens, label):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model} [{label}]")
    print(f"{'='*60}")

    responses = client.run(
        model=model,
        prompts=prompts,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )

    results, parse_failures = [], 0
    for gold, resp in zip(gold_pairs, responses):
        raw = resp["response"] or ""
        # Prepend "[" since RESPONSE_PREFIX already includes it in the prompt
        predicted = parse_pairs("[" + raw)
        if predicted is None:
            parse_failures += 1
            predicted = []
        m = pair_metrics(predicted, gold)
        results.append({
            "gold_pairs": gold,
            "predicted_pairs": predicted,
            "raw_output": raw.strip()[:500],
            **m,
        })

    n = len(results)
    avg = {k: sum(r[k] for r in results) / n for k in ["precision", "recall", "f1", "exact_match"]}
    avg["parse_rate"] = (n - parse_failures) / n

    print(f"\n  Results — {n} examples:")
    for k in ["parse_rate", "precision", "recall", "f1", "exact_match"]:
        print(f"    {k:<20} {avg[k]:.1%}")

    print(f"\n  Samples:")
    for r in results[:3]:
        print(f"    Gold: {r['gold_pairs']}  Pred: {r['predicted_pairs']}  F1={r['f1']:.2f}")

    return {"summary": avg, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate contradiction detection via API")
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--models", type=str, default="gemini-2.5-flash,gemini-2.5-pro",
                        help="Comma-separated model names")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--output-file", type=str, default="outputs/contradiction_api_eval.json")
    parser.add_argument("--hint", action="store_true",
                        help="Tell model exactly how many contradiction pairs exist")
    parser.add_argument("--instruction-after", action="store_true",
                        help="Put task instruction after corpus instead of before")
    parser.add_argument("--ablate-all", action="store_true",
                        help="Run all 4 hint x instruction-position configurations")
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without running")
    args = parser.parse_args()

    examples = load_jsonl(args.eval_data)
    if args.max_examples and len(examples) > args.max_examples:
        examples = examples[:args.max_examples]

    data_stem = Path(args.eval_data).stem
    models = [m.strip() for m in args.models.split(",")]
    gold_pairs = [json.loads(ex["output"]) for ex in examples]
    print(f"Loaded {len(examples)} examples from {args.eval_data}")

    # Define ablation configs
    if args.ablate_all:
        configs = [
            {"hint": False, "instruction_after": False, "label": "instr_before"},
            {"hint": False, "instruction_after": True,  "label": "instr_after"},
            {"hint": True,  "instruction_after": False, "label": "instr_before+hint"},
            {"hint": True,  "instruction_after": True,  "label": "instr_after+hint"},
        ]
    else:
        label_parts = []
        label_parts.append("instr_after" if args.instruction_after else "instr_before")
        if args.hint:
            label_parts.append("hint")
        configs = [{"hint": args.hint, "instruction_after": args.instruction_after,
                     "label": "+".join(label_parts)}]

    # Cost estimate (use worst case: largest prompt config)
    worst_prompts = [build_prompt(ex, hint=True, instruction_after=False) for ex in examples]
    n_runs = len(configs) * len(models)
    print(f"\n  {len(configs)} config(s) x {len(models)} model(s) = {n_runs} runs")
    estimate_costs(worst_prompts, models, args.max_output_tokens)
    if n_runs > 1:
        print(f"  (multiply by {n_runs} for total)")

    if args.dry_run:
        print("\n  Dry run — exiting.")
        return

    # Run all configurations
    client = ParallelResponsesClient(max_concurrent=10, use_cache=True)
    all_results = {}

    for cfg in configs:
        prompts = [build_prompt(ex, hint=cfg["hint"], instruction_after=cfg["instruction_after"])
                   for ex in examples]

        for model in models:
            key = f"{model}/{cfg['label']}"
            all_results[key] = run_eval(
                client, model, prompts, gold_pairs, args.max_output_tokens, cfg["label"],
            )

    # Summary table
    stats = client.get_stats()
    print(f"\n{'='*60}")
    print(f"SUMMARY — {data_stem}")
    print(f"{'='*60}")
    print(f"{'Config':<45} {'Prec':>6} {'Recall':>6} {'F1':>6} {'EM':>6}")
    print("-" * 75)
    for key, res in all_results.items():
        s = res["summary"]
        print(f"{key:<45} {s['precision']:>5.1%} {s['recall']:>6.1%} {s['f1']:>5.1%} {s['exact_match']:>5.1%}")

    print(f"\nCost: ${stats['total_cost_usd']:.4f}  "
          f"(API calls: {stats['api_calls']}, Cache hits: {stats['cache_hits']})")

    save_results(args.output_file, {"args": vars(args), "results": all_results, "cost": stats})
    print(f"Results saved to {args.output_file}")
    client.close()


if __name__ == "__main__":
    main()
