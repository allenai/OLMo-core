"""
Fast single-vLLM-load eval driver for the CPT-mix sweep: scores ONE HF model on
RULER (2 smallest contexts by default) + contradiction, loading the model only once
(vs. one vLLM load per eval file). Reuses corpus-reasoning's prompt building + metric fns.

Run in the corpus-reasoning-eval env (has vLLM), PYTHONPATH=/scratch/users/prasann/corpus-reasoning:

    python scripts/eval/eval_lc_fast.py --base-model <hf_dir> --out outputs/eval_results/<name>.json
"""
import argparse
import json
import os
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--ruler-lengths", default="L1024,L2048")
    ap.add_argument(
        "--ruler-subtasks",
        default="niah_single,niah_multikey,niah_multivalue,niah_multiquery,vt,cwe,fwe",
    )
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--nq-data", default="data/nq_validation_k20_hn19_500_aligned.jsonl",
                    help="fixed (aligned) NQ retrieval eval JSONL")
    ap.add_argument("--oolong-data", default="data/oolong_test_synth_ctx2048_spliteval.jsonl")
    ap.add_argument("--rerank-data", default="data/msmarco_dev_rerank_k20_1000.jsonl")
    ap.add_argument("--outlier-data", default="data/outlier_wiki100w_n20_k3_eval_100.jsonl")
    ap.add_argument("--skip-ruler", action="store_true")
    ap.add_argument("--skip-contra", action="store_true")
    ap.add_argument("--skip-nq", action="store_true")
    ap.add_argument("--skip-extra", action="store_true", help="skip oolong/rerank/outlier")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip held-out retrieval generalization probes (hpqa/fiqa/msmarco/scifact)")
    ap.add_argument("--ladder", action="store_true",
                    help="evaluate each task across its LENGTH LADDER (2k..64k) instead of the single "
                         "default file; reports <task>_<rung> per length. RULER ladders via --ruler-lengths.")
    args = ap.parse_args()

    from corpus_reasoning.lib.vllm_utils import load_model, run_inference
    from corpus_reasoning.eval.evaluate import (
        load_unified_examples, _eval_ruler, _eval_contradiction, _eval_retrieval,
        _eval_oolong, _eval_rerank, _eval_outlier,
    )
    from vllm import SamplingParams
    # Truncate prompts that exceed the model's context (e.g. 64k-rung eval examples tokenize to
    # ~65537 > 65536 max) instead of letting vLLM reject them. Trim from the front; leaves room for gen.
    _TPT = max(1, args.max_model_len - 256)

    margs = SimpleNamespace(
        base_model=args.base_model, lora_path="", tensor_parallel_size=1,
        max_model_len=args.max_model_len, enforce_eager=False,
        language_model_only=False, tokenizer=None, force_flex_attention=False,
    )
    llm, lora = load_model(margs)

    # Cap prompts to the model context: a few 64k-rung eval examples tokenize to ~65537 > 65536 max,
    # which vLLM rejects (and SamplingParams.truncate_prompt_tokens isn't supported in this vLLM).
    # Wrap run_inference to trim over-long prompts from the FRONT (keeps the trailing query intact).
    _tok = llm.get_tokenizer()
    _run_inference_orig = run_inference

    def run_inference(_llm, prompts, sp, _lora=None):  # noqa: F811
        cap = args.max_model_len - getattr(sp, "max_tokens", 256) - 8
        capped = []
        for p in prompts:
            ids = _tok.encode(p)
            capped.append(_tok.decode(ids[-cap:]) if len(ids) > cap else p)
        return _run_inference_orig(_llm, capped, sp, _lora)

    # `eval_size`: reported-key -> number of EVAL EXAMPLES scored. NOT called "n" -- in this
    # project `n` means CORPUS size (docs per example), never eval-set size.
    summary = {"base_model": args.base_model, "ruler": {}, "contradiction": {},
               "nq": {}, "eval_size": {}}

    def strip_think(rs):
        # SFT models trained on the Qwen thinking template emit "<think>...</think>" first.
        # Strip it so RULER's stop token / string match see the actual answer.
        out = []
        for r in rs:
            if "</think>" in r:
                r = r.split("</think>", 1)[1]
            out.append(r)
        return out

    if not args.skip_ruler:
        # NOTE: do NOT stop at the first newline. SFT-on-contradiction models prepend
        # "<think>\n\n</think>\n\n", so a "\n" stop would truncate to an empty answer and make
        # RULER look like 0 everywhere (an eval artifact, not real forgetting). Generate fully and
        # strip the think block before scoring.
        ruler_sp = SamplingParams(temperature=0.0, max_tokens=160)
        recalls, all_correct = [], []
        for sub in args.ruler_subtasks.split(","):
            for L in args.ruler_lengths.split(","):
                path = os.path.join(args.data_dir, f"ruler_{sub}_{L}_eval.jsonl")
                if not os.path.exists(path):
                    print(f"[ruler] MISSING {path}, skipping")
                    continue
                examples = load_unified_examples(
                    path, args.max_test_samples, task="ruler",
                    query_position="after", use_alpaca=True,
                )
                responses = strip_think(run_inference(llm, [e["prompt"] for e in examples], ruler_sp, lora))
                if sub == args.ruler_subtasks.split(",")[0] and L == args.ruler_lengths.split(",")[0]:
                    for i in range(min(2, len(examples))):
                        g = examples[i].get("expected_output") or examples[i].get("answers")
                        print(f"  [sample {sub}_{L} #{i}] gold={str(g)[:80]!r} resp={responses[i].strip()[:160]!r}", flush=True)
                results, _ = _eval_ruler(examples, responses)
                summary["ruler"][f"{sub}_{L}"] = results
                summary["eval_size"][f"ruler_{sub}_{L}"] = len(examples)
                recalls.append(results["recall"])
                all_correct.append(results["all_correct"])
                print(f"[ruler] {sub}_{L}: recall={results['recall']:.3f} "
                      f"all_correct={results['all_correct']:.3f} (eval_size={len(examples)})", flush=True)
        summary["ruler_avg_recall"] = sum(recalls) / len(recalls) if recalls else None
        summary["ruler_avg_all_correct"] = sum(all_correct) / len(all_correct) if all_correct else None

    if not args.skip_contra and not args.ladder:
        contra_sp = SamplingParams(temperature=0.0, max_tokens=200)
        examples = load_unified_examples(
            args.contra_data, args.max_test_samples, task="contradiction",
            query_position="both", use_alpaca=True,
        )
        responses = run_inference(llm, [e["prompt"] for e in examples], contra_sp, lora)
        results, _ = _eval_contradiction(examples, responses)
        summary["contradiction"] = results
        summary["eval_size"]["contradiction"] = len(examples)
        print(f"[contradiction] f1={results['f1']:.3f} em={results['exact_match']:.3f} "
              f"precision={results['precision']:.3f} recall={results['recall']:.3f} "
              f"parse_rate={results['parse_rate']:.3f} (eval_size={len(examples)})", flush=True)

    if not args.skip_nq and not args.ladder and os.path.exists(args.nq_data):
        # Fixed (aligned) NQ retrieval: find the relevant doc id per query. Short answer ("[3]");
        # strip the think prefix and don't newline-stop (the SFT model emits <think> first).
        nq_sp = SamplingParams(temperature=0.0, max_tokens=64)
        examples = load_unified_examples(
            args.nq_data, args.max_test_samples, task="retrieval",
            query_position="both", use_alpaca=True,
        )
        responses = strip_think(run_inference(llm, [e["prompt"] for e in examples], nq_sp, lora))
        results, _ = _eval_retrieval(examples, responses)
        summary["nq"] = results
        summary["eval_size"]["nq"] = len(examples)
        print(f"[nq-retrieval] f1={results.get('f1', 0):.3f} em={results.get('exact_match', 0):.3f} "
              f"recall={results.get('recall', 0):.3f} (eval_size={len(examples)})", flush=True)

    # ---- held-out retrieval GENERALIZATION probes (eval-only; never in the train mix) ----
    # Tests whether NQ-retrieval training transfers zero-shot to other retrieval tasks.
    # All scored with the same task="retrieval" path as NQ. hpqa/obliq are multi-query
    # (one answer per query); fiqa/msmarco/scifact are single-query. All fit <=16k except
    # obliq (~48k, excluded from this 16k eval).
    if not args.skip_gen:
        gen = [
            ("hpqa",    "data/n2ified_eval_hpqa_q20.jsonl",       256),  # 20 queries/ex
            ("fiqa",    "data/beir_fiqa_ce_test_k20_648.jsonl",    64),
            ("msmarco", "data/msmarco_trecdl2019_k20_43.jsonl",    64),
            ("scifact", "data/beir_scifact_test_k20_300.jsonl",    64),
        ]
        for gname, gpath, gmax in gen:
            if not os.path.exists(gpath):
                print(f"[gen:{gname}] MISSING {gpath}, skipping"); continue
            sp = SamplingParams(temperature=0.0, max_tokens=gmax)
            examples = load_unified_examples(gpath, args.max_test_samples, task="retrieval",
                                             query_position="both", use_alpaca=True)
            responses = strip_think(run_inference(llm, [e["prompt"] for e in examples], sp, lora))
            results, _ = _eval_retrieval(examples, responses)
            summary[f"gen_{gname}"] = results
            print(f"[gen:{gname}] f1={results.get('f1', 0):.3f} em={results.get('exact_match', 0):.3f} "
                  f"recall={results.get('recall', 0):.3f} (eval_size={len(examples)})", flush=True)

    # ---- extra SFT tasks: oolong (score), msmarco rerank (mrr@k), outlier (f1) ----
    if not args.skip_extra and not args.ladder:
        extra = [
            ("oolong", args.oolong_data, _eval_oolong, "score", 200),
            ("rerank", args.rerank_data, _eval_rerank, None, 256),   # primary = mrr@k key
            ("outlier", args.outlier_data, _eval_outlier, "f1", 200),
        ]
        for tname, tdata, fn, pkey, maxtok in extra:
            if not os.path.exists(tdata):
                print(f"[{tname}] MISSING {tdata}, skipping"); continue
            sp = SamplingParams(temperature=0.0, max_tokens=maxtok)
            examples = load_unified_examples(tdata, args.max_test_samples, task=tname,
                                             query_position="both", use_alpaca=True)
            responses = strip_think(run_inference(llm, [e["prompt"] for e in examples], sp, lora))
            results, _ = fn(examples, responses)
            summary[tname] = results
            prim = results.get(pkey) if pkey else next(
                (v for k, v in results.items() if k.startswith("mrr")), None)
            summary[f"{tname}_primary"] = prim
            print(f"[{tname}] primary={prim:.3f} {({k: round(v,3) for k,v in results.items() if isinstance(v,(int,float))})} (eval_size={len(examples)})", flush=True)

    # ---- LENGTH-LADDER: each task evaluated at 2k..64k (reports <task>_<rung>) ----
    # RULER already ladders via --ruler-lengths. Here we loop each SFT task over its rung files.
    if args.ladder:
        RR = "/scratch/users/prasann/cpt_data/rerank_ladder_src"
        LADDERS = {
            "contradiction": [("2k", args.contra_data),
                ("8k", "data/contradiction_eval_pubmed_both_ctx8k.jsonl"),
                ("16k", "data/contradiction_eval_pubmed_both_ctx16k.jsonl"),
                ("32k", "data/contradiction_eval_pubmed_both_ctx32k.jsonl"),
                ("64k", "data/contradiction_eval_pubmed_both_ctx64k.jsonl")],
            "nq": [("3k", args.nq_data),
                ("8k", "data/ladder64k/rung_8k_eval_k50.jsonl"),
                ("16k", "data/ladder64k/rung_16k_eval_k100.jsonl"),
                ("32k", "data/ladder64k/rung_32k_eval_k200.jsonl"),
                ("64k", "data/ladder64k/rung_64k_eval_k420.jsonl")],
            "oolong": [("1k", "data/oolong_test_synth_ctx1024_spliteval.jsonl"),
                ("2k", "data/oolong_test_synth_ctx2048_spliteval.jsonl"),
                ("4k", "data/oolong_test_synth_ctx4096_spliteval.jsonl"),
                ("8k", "data/oolong_test_synth_ctx8192_spliteval.jsonl"),
                ("16k", "data/oolong_test_synth_ctx16384_spliteval.jsonl"),
                ("32k", "data/oolong_test_synth_ctx32768_spliteval.jsonl"),
                ("64k", "data/oolong_test_synth_ctx65536_spliteval.jsonl")],
            "rerank": [("2k", args.rerank_data),
                ("8k", f"{RR}/msmarco_validation_rerank_k80_100.jsonl"),
                ("16k", f"{RR}/msmarco_validation_rerank_k158_100.jsonl"),
                ("32k", f"{RR}/msmarco_validation_rerank_k315_100.jsonl"),
                ("64k", f"{RR}/msmarco_validation_rerank_k600_100.jsonl")],
            "outlier": [("3k", args.outlier_data),
                ("8k", "data/outlier_wiki100w_n55_k3_eval_100.jsonl"),
                ("16k", "data/outlier_wiki100w_n110_k3_eval_100.jsonl"),
                ("32k", "data/outlier_wiki100w_n220_k3_eval_100.jsonl"),
                ("64k", "data/outlier_wiki100w_n435_k3_eval_100.jsonl")],
        }
        # (load_task, eval_fn, primary_key or None=mrr, max_new_tokens)
        LSPEC = {
            "contradiction": ("contradiction", _eval_contradiction, "f1", 200),
            "nq": ("retrieval", _eval_retrieval, "f1", 64),
            "oolong": ("oolong", _eval_oolong, "score", 200),
            "rerank": ("rerank", _eval_rerank, None, 256),
            "outlier": ("outlier", _eval_outlier, "f1", 200),
        }
        for task, rungs in LADDERS.items():
            loadtask, fn, pkey, maxtok = LSPEC[task]
            sp = SamplingParams(temperature=0.0, max_tokens=maxtok)
            for label, path in rungs:
                if not path or not os.path.exists(path):
                    print(f"[ladder:{task}@{label}] MISSING {path}, skipping"); continue
                examples = load_unified_examples(path, args.max_test_samples, task=loadtask,
                                                 query_position="both", use_alpaca=True)
                responses = strip_think(run_inference(llm, [e["prompt"] for e in examples], sp, lora))
                results, _ = fn(examples, responses)
                prim = results.get(pkey) if pkey else next(
                    (v for k, v in results.items() if k.startswith("mrr")), None)
                summary[f"{task}_{label}"] = prim
                print(f"[ladder:{task}@{label}] {pkey or 'mrr'}={prim if prim is None else round(prim,3)} "
                      f"(eval_size={len(examples)})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print("\n==== SUMMARY ====")
    print(f"model: {args.base_model}")
    print(f"ruler_avg_recall:     {summary.get('ruler_avg_recall')}")
    print(f"contradiction_f1:     {summary['contradiction'].get('f1') if summary['contradiction'] else None}")
    print(f"nq_retrieval_f1:      {summary['nq'].get('f1') if summary['nq'] else None}")
    for t in ("oolong", "rerank", "outlier"):
        print(f"{t}_primary:        {summary.get(t+'_primary')}")
    for g in ("hpqa", "fiqa", "msmarco", "scifact"):
        gr = summary.get(f"gen_{g}")
        print(f"gen_{g}_f1:        {gr.get('f1') if gr else None}")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
