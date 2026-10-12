"""Step A of the chunked-vllm validation gate: render the EXACT native-eval prefills.

Runs in the corpus-reasoning-olmo env (needs olmo_core + a fast Qwen3.5 tokenizer).
Emits a JSON with per-example prompt token ids built by
``eval_lc_native_docchunk_contra.build_eval_prefill`` — the single source of truth
for the native eval's token layout — so the vLLM run consumes token-identical
prompts and the comparison isolates the attention implementation.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--contra-data", required=True)
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument("--cot-mode", default="none")
    ap.add_argument("--doc-start-id", type=int, default=248049)
    ap.add_argument("--doc-end-id", type=int, default=248050)
    ap.add_argument("--eos-token-id", type=int, default=248044)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from corpus_reasoning.eval.eval_lc_native_docchunk_contra import build_eval_prefill
    from corpus_reasoning.eval.evaluate import load_unified_examples

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    newline_id = tok("\n", add_special_tokens=False).input_ids[-1]

    examples = load_unified_examples(
        args.contra_data, args.max_test_samples, task="contradiction",
        query_position="both", use_alpaca=True,
    )
    rows = []
    for gi, ex in enumerate(examples):
        raw = ex.get("ex", ex)
        prefill = build_eval_prefill(
            tok, raw,
            variant="dense", cot_mode=args.cot_mode,
            doc_start_id=args.doc_start_id, doc_end_id=args.doc_end_id,
        )
        n_doc_start = sum(1 for t in prefill if t == args.doc_start_id)
        rows.append({"idx": gi, "prefill": prefill, "n_chunks": n_doc_start})

    out = {
        "tokenizer": args.tokenizer,
        "contra_data": args.contra_data,
        "eval_size": len(examples),
        "cot_mode": args.cot_mode,
        "doc_start_id": args.doc_start_id,
        "doc_end_id": args.doc_end_id,
        "eos_token_id": args.eos_token_id,
        "newline_id": newline_id,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    lens = [len(r["prefill"]) for r in rows]
    print(f"[prefills] wrote {len(rows)} rows -> {args.out} "
          f"(len min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}; "
          f"chunks per example: {sorted(set(r['n_chunks'] for r in rows))})", flush=True)


if __name__ == "__main__":
    main()
