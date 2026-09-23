"""
Token-level parity: olmo-eval's ``CTC_SUITE_DOC_MARKERS=1`` chat prompt vs the CTC SFT converter.

For real eval rows of every setA task, render the eval prompt the olmo-eval way (spec body, no
Alpaca, ``add_doc_markers``, the model's chat template with the generation prompt) and the training
prompt the converter's way (``segment_prompt_to_chunks(include_answer=False)`` with the builder's
spec / chunk_by, qwen3_5 markers) and require identical token ids.

    python debug/ctc_sft_setA/check_doc_marker_parity.py --olmo-eval ../olmo-eval --rungs r2k r8k
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmo-eval", required=True, help="olmo-eval checkout (branch with doc_markers)")
    ap.add_argument("--rungs", nargs="+", default=["r2k", "r8k"])
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="PrasannSinghal/ctc-suite-eval")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.abspath(args.olmo_eval), "src"))

    from audit_train_eval_iid import TASK_TO_ROW, _eval_rows, _load_builder_roster
    from transformers import AutoTokenizer

    import huggingface_hub.utils as _hu

    if not hasattr(_hu, "silent_tqdm"):  # older hub in the OLMo-core env; only ruler's loader uses it
        from tqdm import tqdm as _tqdm

        _hu.silent_tqdm = _tqdm
    import olmo_eval.evals.tasks.ctc_suite as S
    from olmo_core.data.document_chunk_landmark import reserved_ids, segment_prompt_to_chunks
    from olmo_eval.evals.tasks.ctc_suite.doc_markers import add_doc_markers

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids_set = reserved_ids("qwen3_5")
    builder = _load_builder_roster(REPO)
    bad = total = 0
    for task, row in TASK_TO_ROW.items():
        rr = S.ROSTER[row]
        spec = S._resolve_spec(rr.spec)
        for rung in args.rungs:
            rows = _eval_rows(args.dataset, rr.subset, rung, args.rows)
            if not rows:
                print(f"  {task}@{rung}: no eval rows")
                continue
            for i, ex in enumerate(rows):
                body = spec.build_prompt(ex, query_position=S.QUERY_POSITION, use_alpaca=False)
                body = add_doc_markers(body, ex, spec.name)
                text = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                               add_generation_prompt=True)
                ev = tok(text, add_special_tokens=False)["input_ids"]
                _, tr, _ = segment_prompt_to_chunks(
                    tok, ex, builder[task]["spec"], query_position="both", cot_mode="none",
                    chunk_by=builder[task]["chunk_by"], item_regex=r"\|\|", include_answer=False,
                    use_titles=False, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end)
                total += 1
                n_mark = sum(t in (ids_set.doc_start, ids_set.doc_end) for t in ev)
                if ev != tr:
                    bad += 1
                    k = next((j for j, (a, b) in enumerate(zip(ev, tr)) if a != b),
                             min(len(ev), len(tr)))
                    print(f"  MISMATCH {task}@{rung}#{i}: eval {len(ev)} vs train {len(tr)} tok, "
                          f"first diff at {k}: eval {tok.decode(ev[k:k + 12])!r} | "
                          f"train {tok.decode(tr[k:k + 12])!r}")
                elif i == 0:
                    print(f"  ok {task}@{rung}: {len(ev):,} tok, {n_mark} marker tokens")
    print(f"\n{total - bad}/{total} prompts token-identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
