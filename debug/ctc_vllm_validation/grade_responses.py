"""Step C of the chunked-vllm validation gate: grade vLLM responses.

Runs in the corpus-reasoning-olmo env. Uses the same `_eval_contradiction` +
`load_unified_examples` (same seed/sampling) as the native evaluator so the
metric computation is shared code.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="output of run_vllm_eval.py")
    ap.add_argument("--contra-data", required=True)
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from corpus_reasoning.eval.evaluate import _eval_contradiction, load_unified_examples

    with open(args.responses) as f:
        pack = json.load(f)

    examples = load_unified_examples(
        args.contra_data, args.max_test_samples, task="contradiction",
        query_position="both", use_alpaca=True,
    )
    assert len(examples) == pack["eval_size"], (len(examples), pack["eval_size"])
    responses = [pack["responses"][str(i)] for i in range(len(examples))]
    res, details = _eval_contradiction(examples, responses)
    summary = {
        "mode": pack["mode"],
        "hf_model": pack["hf_model"],
        "contra_data": args.contra_data,
        "eval_size": len(examples),
        "patch_debug": pack.get("patch_debug", {}),
        "contradiction": res,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[grade:{pack['mode']}] f1={res.get('f1', 0):.3f} "
          f"precision={res.get('precision', 0):.3f} recall={res.get('recall', 0):.3f} "
          f"em={res.get('exact_match', 0):.3f} parse_rate={res.get('parse_rate', 0):.3f} "
          f"(eval_size={len(examples)})", flush=True)


if __name__ == "__main__":
    main()
