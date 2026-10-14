"""
Token-level parity for the Qwen3-4B-Base setA arm: olmo-eval's ``--doc-markers`` chat prompt vs the
Qwen3 TRAINING sequence (``build_qwen3_setA_shards.py``), on real eval rows of every setA task.

The Qwen3 counterpart of ``recipes/eval/setA_audit/check_doc_marker_parity.py`` (same rows, same
renderers; it imports that file's helpers read-only). The training side is what the shards hold: the
converter's ``segment_prompt_to_chunks(include_answer=True)`` with ``Qwen/Qwen3-4B-Base``, then the
shard build's label alignment (``<think> \\n\\n`` unlabeled, so labels start at ``</think>`` as in the
Qwen3.5 baseline). Qwen3's template renders the final assistant turn as
``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<answer><|im_end|>\\n`` -- with or without
``enable_thinking`` -- but its generation prompt is ``<|im_start|>assistant\\n`` unless
``enable_thinking=False``. Two eval tokenizers are checked:

* ``stock`` -- ``Qwen/Qwen3-4B-Base`` as is (``launch_ctc_suite.py --tokenizer Qwen/Qwen3-4B-Base``).
  Its prompt IS a token prefix of the training sequence, so olmo-eval's ``preflight.py`` accepts it,
  but it stops before the empty thinking block: the model would have to generate
  ``<think>\\n\\n</think>\\n\\n`` itself before the answer, which the Qwen3.5 baseline's eval never
  asked of it.
* ``nothink`` -- the same tokenizer with ``audit_qwen3_base.make_nothink_template`` (generation prompt
  closes the empty thinking block, as Qwen3.5-0.8B's does). Its prompt must end exactly where the
  labeled answer starts.

It also reports the eval prompts' Qwen3 lengths vs Qwen3.5-0.8B lengths per rung (YaRN coverage).

    python recipes/train/setA/check_qwen3_chat_parity.py --olmo-eval ../olmo-eval --rungs r2k r8k
"""

import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "recipes", "eval", "setA_audit"))
sys.path.insert(0, HERE)

OOD_TO_TASK = {"ctc_contra_fever": "contradiction", "ctc_outlier_review": "outlier"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmo-eval", required=True)
    ap.add_argument("--rungs", nargs="+", default=["r2k", "r8k", "r32k"])
    ap.add_argument("--length-rungs", nargs="*", default=["r64k", "r128k"],
                    help="rungs measured for length only (no parity render)")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--length-rows", type=int, default=5)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--q35-eval-tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dataset", default="PrasannSinghal/ctc-suite-eval")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.abspath(args.olmo_eval), "src"))
    sys.path.insert(0, os.path.join(os.path.abspath(args.olmo_eval), "scripts", "ctc_suite"))

    from audit_qwen3_base import make_nothink_template
    from audit_train_eval_iid import TASK_TO_ROW, _eval_rows, _load_builder_roster
    from transformers import AutoTokenizer

    import huggingface_hub.utils as _hu

    if not hasattr(_hu, "silent_tqdm"):
        from tqdm import tqdm as _tqdm

        _hu.silent_tqdm = _tqdm
    import olmo_eval.evals.tasks.ctc_suite as S
    import preflight
    from olmo_core.data.document_chunk_landmark import reserved_ids, segment_prompt_to_chunks
    from olmo_eval.evals.tasks.ctc_suite.doc_markers import add_doc_markers

    train_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stock = AutoTokenizer.from_pretrained(args.tokenizer)
    nothink = AutoTokenizer.from_pretrained(args.tokenizer)
    nothink.chat_template = make_nothink_template(nothink.chat_template)
    q35 = AutoTokenizer.from_pretrained(args.q35_eval_tokenizer)
    ids_set = reserved_ids("qwen3")
    im_end = train_tok.convert_tokens_to_ids("<|im_end|>")
    think = train_tok.convert_tokens_to_ids("<think>")
    nn = train_tok.encode("\n\n", add_special_tokens=False)[0]
    print(f"qwen3 ids: <|im_end|>={im_end} <think>={think} \\n\\n={nn} "
          f"box={ids_set.doc_start}/{ids_set.doc_end} eos={ids_set.eos}", flush=True)

    # olmo-eval's own preflight gate, on both eval tokenizers
    nothink_dir = "/tmp/qwen3_nothink_tok"
    nothink.save_pretrained(nothink_dir)
    for name, path in (("stock", args.tokenizer), ("nothink", nothink_dir)):
        try:
            preflight.check_chat_prefix(path)
            print(f"preflight.check_chat_prefix({name}): PASS", flush=True)
        except SystemExit as exc:
            print(f"preflight.check_chat_prefix({name}): FAIL {exc}", flush=True)

    builder = _load_builder_roster(REPO)
    cases = [(task, row, S.ROSTER[row]) for task, row in TASK_TO_ROW.items()]
    cases += [(task, row, S.OOD_ROSTER[row]) for row, task in OOD_TO_TASK.items()]

    def eval_ids(tok, ex, spec):
        body = spec.build_prompt(ex, query_position=S.QUERY_POSITION, use_alpaca=False)
        body = add_doc_markers(body, ex, spec.name)
        text = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    tally = {"stock": [0, 0], "nothink": [0, 0]}  # [ok, total]
    lengths = defaultdict(list)  # rung -> (q3, q35) eval prompt lengths
    for task, row, rr in cases:
        spec = S._resolve_spec(rr.spec)
        for rung in args.rungs + list(args.length_rungs):
            parity = rung in args.rungs
            rows = _eval_rows(args.dataset, rr.subset, rung, args.rows if parity
                              else args.length_rows)
            if not rows:
                print(f"  {row}@{rung}: no eval rows")
                continue
            for i, ex in enumerate(rows):
                ev_q35 = eval_ids(q35, ex, spec)
                ev_nothink = eval_ids(nothink, ex, spec)
                lengths[rung].append((len(ev_nothink), len(ev_q35), row))
                if not parity:
                    continue
                _, tr, mask = segment_prompt_to_chunks(
                    train_tok, ex, builder[task]["spec"], query_position="both", cot_mode="none",
                    chunk_by=builder[task]["chunk_by"], item_regex=r"\|\|", include_answer=True,
                    use_titles=False, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end)
                # the shard build's label alignment: labels start at </think>, as in the baseline
                j = mask.index(True)
                assert tr[j:j + 2] == [think, nn], train_tok.decode(tr[j:j + 6])
                mask = list(mask)
                mask[j] = mask[j + 1] = False
                for name, tok in (("stock", stock), ("nothink", nothink)):
                    ev = eval_ids(tok, ex, spec) if name == "stock" else ev_nothink
                    tally[name][1] += 1
                    rest, rest_mask = tr[len(ev):], mask[len(ev):]
                    why = None
                    if tr[: len(ev)] != ev:
                        k = next((x for x, (a, b) in enumerate(zip(ev, tr)) if a != b),
                                 min(len(ev), len(tr)))
                        why = (f"first diff at {k}: eval {tok.decode(ev[k:k + 12])!r} | "
                               f"train {train_tok.decode(tr[k:k + 12])!r}")
                    elif not rest or not all(rest_mask) or im_end not in rest:
                        why = (f"a prefix, but the training tokens after it are not all labeled "
                               f"answer: {train_tok.decode(rest[:8])!r} labels "
                               f"{[int(m) for m in rest_mask[:8]]}")
                    if why:
                        if i == 0:
                            print(f"  [{name}] NOT ALIGNED {row}@{rung}#{i}: {why}", flush=True)
                    else:
                        tally[name][0] += 1
                        if i == 0 and name == "nothink":
                            n_mark = sum(t in (ids_set.doc_start, ids_set.doc_end) for t in ev)
                            print(f"  [nothink] ok {row}@{rung}: {len(ev):,} tok, {n_mark} marker "
                                  f"tokens, then labeled {train_tok.decode(rest[:12])!r}",
                                  flush=True)
    for name, (ok, total) in tally.items():
        print(f"\n[{name}] {ok}/{total} eval prompts are exact token prefixes of their training "
              f"sequence, ending where the labeled answer starts", flush=True)
    print("\neval prompt lengths, Qwen3 (nothink) vs Qwen3.5-0.8B tokens:")
    for rung, xs in lengths.items():
        q3 = sorted(x[0] for x in xs)
        r = sorted(x[0] / x[1] for x in xs)
        print(f"  {rung}: n={len(xs)} qwen3 median {q3[len(q3) // 2]:,} max {q3[-1]:,} "
              f"(row {max(xs)[2]}); qwen3/qwen3.5 ratio median {r[len(r) // 2]:.3f} "
              f"max {r[-1]:.3f}", flush=True)
    sys.exit(0 if tally["nothink"][0] == tally["nothink"][1] else 1)


if __name__ == "__main__":
    main()
