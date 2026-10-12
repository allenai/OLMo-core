"""Generic (any-CTC-suite-task) version of build_prefills.py.

Same idea as build_prefills.py (render the EXACT native-eval prefill token ids so the
vLLM run consumes token-identical prompts) but parametrized by --task via
eval_lc_native_docchunk.TASK_CFG instead of hardcoding contradiction's segmentation
(chunk_by="document", cot="none"). Used for the post-parity SWEEP of grouping /
hotpotqa(retrieval) / outlier / etc. 4B checkpoints across rungs.

Runs in the corpus-reasoning-olmo env (needs olmo_core + a fast Qwen3.5 tokenizer).
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--task", required=True, help="canonical TASK_CFG key, or an alias (nq/contra)")
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--max-test-samples", type=int, default=100000)
    ap.add_argument("--cot-mode", default=None, help="override TASK_CFG's default cot mode")
    ap.add_argument("--doc-start-id", type=int, default=248049)
    ap.add_argument("--doc-end-id", type=int, default=248050)
    ap.add_argument("--eos-token-id", type=int, default=248044)
    ap.add_argument(
        "--query-position",
        default="both",
        choices=["before", "after", "both"],
        help="MUST match the query position the checkpoint trained on",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from corpus_reasoning.eval.eval_lc_native_docchunk import TASK_ALIASES, TASK_CFG
    from corpus_reasoning.eval.evaluate import load_unified_examples
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        segment_prompt_to_chunks,
    )

    seg_task = TASK_ALIASES.get(args.task, args.task)
    if seg_task not in TASK_CFG:
        raise SystemExit(f"--task {args.task!r} -> {seg_task!r} not in {sorted(TASK_CFG)}")
    cfg = TASK_CFG[seg_task]
    chunk_by = cfg["chunk_by"]
    cot_mode = args.cot_mode if args.cot_mode is not None else cfg["cot"]

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    newline_id = tok("\n", add_special_tokens=False).input_ids[-1]

    examples = load_unified_examples(
        args.eval_data,
        args.max_test_samples,
        task=seg_task,
        query_position=args.query_position,
        use_alpaca=True,
    )
    rows = []
    for gi, ex in enumerate(examples):
        raw = ex.get("ex", ex)
        segs, ids, _ = segment_prompt_to_chunks(
            tok,
            raw,
            seg_task,
            query_position=args.query_position,
            cot_mode=cot_mode,
            chunk_by=chunk_by,
            item_regex=r"\|\|",
            include_answer=False,
            doc_start_id=args.doc_start_id,
            doc_end_id=args.doc_end_id,
        )
        prefill, _ = emit_document_chunk_dense(segs)  # box markers present; full attn ignores them
        n_doc_start = sum(1 for t in prefill if t == args.doc_start_id)
        rows.append({"idx": gi, "prefill": prefill, "n_chunks": n_doc_start})

    out = {
        "tokenizer": args.tokenizer,
        "task": args.task,
        "seg_task": seg_task,
        "chunk_by": chunk_by,
        "eval_data": args.eval_data,
        "eval_size": len(examples),
        "query_position": args.query_position,
        "cot_mode": cot_mode,
        "stop_rule": cfg["stop"],
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
    print(
        f"[prefills:{args.task}] wrote {len(rows)} rows -> {args.out} "
        f"(len min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}; "
        f"chunks per example: {sorted(set(r['n_chunks'] for r in rows))[:10]}; "
        f"stop_rule={cfg['stop']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
