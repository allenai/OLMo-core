"""
Token-level parity: olmo-eval's ``CTC_SUITE_DOC_MARKERS=1`` chat prompt vs the CTC SFT TRAINING sequence.

The converter renders every training example as ONE chat-template call over (user prompt, assistant
answer) and tokenizes that whole string (``segment_prompt_to_chunks(include_answer=True)``). What the
model saw before its answer is therefore a prefix of THAT token sequence -- not of the prompt rendered
on its own with ``add_generation_prompt=True``. The two differ at the assistant header:

* Qwen3.5-4B's template opens a thinking block for a generation prompt (``<think>\\n``), but renders a
  past assistant turn as ``<think>\\n\\n</think>\\n\\n<answer>``. Tokenized, the training sequence is
  ``<think> ĊĊ </think> ĊĊ <answer>``; a 4B-template eval prompt ends ``<think> Ċ`` -- a token split the
  model never saw, mid-way through a template it was trained to complete in one go.
* Qwen3.5-0.8B's template (same vocab) renders the generation prompt as ``<think>\\n\\n</think>\\n\\n``:
  an exact token prefix of the training sequence, ending right where the answer starts.

So the check is: the eval prompt (``--eval-tokenizer``, the launcher's ``--tokenizer``) must equal the
training sequence (``--train-tokenizer``, the builder's) up to its own length, and everything after it
in the training sequence must be label-masked answer. (An earlier version compared against the
converter's prompt-only render with the 4B tokenizer on both sides, so it passed a generation prompt
the model was never trained on.)

For real eval rows of every setA task plus the two held-out rows (same specs as their twins).

    python recipes/eval/setA_audit/check_doc_marker_parity.py --olmo-eval ../olmo-eval --rungs r2k r8k

``--doc-id-style hash`` checks hashed document ids (converter ``--doc-id-style hash`` vs olmo-eval
``CTC_SUITE_DOC_ID_STYLE=hash``) the same way, and additionally:

* the two copies of ``doc_id_hash.py`` (OLMo-core, olmo-eval) are byte-identical;
* round trip: the training target (rendered with hashed ids by the converter's code path) is fed
  through the eval's real scoring path (``_score_output``: stop rules, inverse id map, the spec's
  parser and grader) and must score exactly what the sequential target scores on the same row;
* the hashed target fits the row's decode budget (5-digit ids cost more tokens than ``1..N``).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

#: held-out eval row -> the train task whose spec / chunk_by renders it
OOD_TO_TASK = {"ctc_contra_fever": "contradiction", "ctc_outlier_review": "outlier"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmo-eval", required=True, help="olmo-eval checkout (branch with doc_markers)")
    ap.add_argument("--rungs", nargs="+", default=["r2k", "r8k"])
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--eval-tokenizer", default="Qwen/Qwen3.5-0.8B",
                    help="the tokenizer the eval renders with (launch_ctc_suite.py --tokenizer)")
    ap.add_argument("--train-tokenizer", default="Qwen/Qwen3.5-4B",
                    help="the tokenizer the SFT shards were built with (build_ctc_sft.py --tokenizer)")
    ap.add_argument("--dataset", default="PrasannSinghal/ctc-suite-eval")
    ap.add_argument("--doc-id-style", choices=["seq", "hash"], default="seq")
    ap.add_argument("--rerank-decode-tokens", type=int, default=160,
                    help="the rerank decode cap launch_ctc_suite.py sets (RERANK_DECODE_TOKENS)")
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

    hashed = args.doc_id_style == "hash"
    if hashed:
        import hashlib

        from olmo_core.data.corpus_reasoning_prompts import build_prompt
        from olmo_core.data.corpus_reasoning_prompts import doc_id_hash as train_hash
        from olmo_eval.common.types import Instance, LMOutput
        from olmo_eval.evals.tasks.ctc_suite import doc_id_hash as eval_hash

        digests = {m.__name__: hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()
                   for m in (train_hash, eval_hash)}
        if len(set(digests.values())) != 1:
            sys.exit(f"doc_id_hash.py copies differ: {digests}")
        print(f"doc_id_hash.py identical in both repos (sha256 {next(iter(digests.values()))[:12]})")
        rt_bad = rt_total = budget_bad = 0

    eval_tok = AutoTokenizer.from_pretrained(args.eval_tokenizer)
    train_tok = AutoTokenizer.from_pretrained(args.train_tokenizer)
    im_end = train_tok.convert_tokens_to_ids("<|im_end|>")
    ids_set = reserved_ids("qwen3_5")
    builder = _load_builder_roster(REPO)
    cases = [(task, row, S.ROSTER[row]) for task, row in TASK_TO_ROW.items()]
    cases += [(task, row, S.OOD_ROSTER[row]) for row, task in OOD_TO_TASK.items()]
    bad = total = 0
    for task, row, rr in cases:
        spec = S._resolve_spec(rr.spec)
        for rung in args.rungs:
            rows = _eval_rows(args.dataset, rr.subset, rung, args.rows)
            if not rows:
                print(f"  {row}@{rung}: no eval rows")
                continue
            for i, ex in enumerate(rows):
                body = spec.build_prompt(ex, query_position=S.QUERY_POSITION, use_alpaca=False)
                if hashed:
                    body = eval_hash.relabel_prompt(body, ex, spec.name)
                body = add_doc_markers(body, ex, spec.name)
                text = eval_tok.apply_chat_template([{"role": "user", "content": body}],
                                                    tokenize=False, add_generation_prompt=True)
                ev = eval_tok(text, add_special_tokens=False)["input_ids"]
                _, tr, mask = segment_prompt_to_chunks(
                    train_tok, ex, builder[task]["spec"], query_position="both", cot_mode="none",
                    chunk_by=builder[task]["chunk_by"], item_regex=r"\|\|", include_answer=True,
                    use_titles=False, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
                    doc_id_style=args.doc_id_style)
                total += 1
                n_mark = sum(t in (ids_set.doc_start, ids_set.doc_end) for t in ev)
                rest, rest_mask = tr[len(ev):], mask[len(ev):]
                why = None
                if tr[: len(ev)] != ev:
                    k = next((j for j, (a, b) in enumerate(zip(ev, tr)) if a != b),
                             min(len(ev), len(tr)))
                    why = (f"first diff at {k}: eval {eval_tok.decode(ev[k:k + 12])!r} | "
                           f"train {train_tok.decode(tr[k:k + 12])!r}")
                elif not rest or not all(rest_mask) or im_end not in rest:
                    why = (f"eval prompt is a prefix, but the training tokens after it are not all "
                           f"labeled answer: {train_tok.decode(rest[:24])!r}")
                if why:
                    bad += 1
                    print(f"  MISMATCH {row}@{rung}#{i}: eval {len(ev)} vs train {len(tr)} tok, {why}")
                elif i == 0:
                    print(f"  ok {row}@{rung}: {len(ev):,} tok, {n_mark} marker tokens, "
                          f"then trained answer {train_tok.decode(rest[:16])!r}")
                if hashed:
                    rt_total += 1
                    _, seq_ans = build_prompt(ex, task=builder[task]["spec"], query_position="both",
                                              use_alpaca=False, cot_mode="none", use_titles=False)
                    hash_ans = train_hash.relabel_answer(seq_ans, ex, builder[task]["spec"])
                    trained = train_tok.decode(rest)
                    meta = {"example": ex, "doc_id_style": "hash"}
                    _, got = S._score_output(spec, Instance(question="", metadata=meta),
                                             LMOutput(text=hash_ans))
                    _, want = S._score_output(spec, Instance(question="", metadata={"example": ex}),
                                              LMOutput(text=seq_ans))
                    budget = S._decode_budget(rr, spec)
                    if spec.name == "rerank":
                        budget = min(budget, args.rerank_decode_tokens)
                    n_ans = len(eval_tok(hash_ans, add_special_tokens=False)["input_ids"])
                    n_seq = len(eval_tok(seq_ans, add_special_tokens=False)["input_ids"])
                    p = spec.primary_metric
                    if not trained.startswith(hash_ans):
                        rt_bad += 1
                        print(f"  RT-MISMATCH {row}@{rung}#{i}: trained target "
                              f"{trained[:60]!r} != hashed answer {hash_ans[:60]!r}")
                    elif got != want:
                        rt_bad += 1
                        print(f"  RT-SCORE {row}@{rung}#{i}: hashed {got} vs seq {want}")
                    elif i == 0:
                        print(f"     round trip {row}@{rung}: {p}={got[p]:.3f} (== seq), answer "
                              f"{n_ans} tok (seq {n_seq}) / budget {budget}: {hash_ans[:48]!r}")
                    # rerank is graded on its first 10 ids only; the cap is sized for those
                    need = n_ans if spec.name != "rerank" else len(eval_tok(
                        ", ".join(hash_ans.split(", ")[:10]), add_special_tokens=False)["input_ids"])
                    if need + 1 > budget:  # + the stop / EOS token
                        budget_bad += 1
                        print(f"  BUDGET {row}@{rung}#{i}: hashed gold needs {need} tok (seq "
                              f"{n_seq}) > decode budget {budget}")
    print(f"\n{total - bad}/{total} eval prompts are exact token prefixes of their training sequence, "
          f"ending where the labeled answer starts (doc ids: {args.doc_id_style})")
    if hashed:
        print(f"{rt_total - rt_bad}/{rt_total} hashed training targets score exactly their "
              f"sequential target through the eval's scorer; {budget_bad} exceed the decode budget")
        bad += rt_bad + budget_bad
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
