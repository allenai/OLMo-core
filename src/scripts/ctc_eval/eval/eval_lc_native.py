"""
NATIVE olmo_core eval (no HF export, no vLLM, no oe_eval task registry).

Loads an olmo-core distcp checkpoint directly via olmo_core.generate and scores RULER + contradiction
+ NQ with the same corpus-reasoning metric functions as eval_lc_fast.py. The point: skip the
~5-min olmo->HF export step per eval. Generation uses TransformerGenerationModule.generate_batch
(the same path oe-eval's OlmoCoreLM backend uses), so no oe_eval registry deps (math_verify/alpaca).

    python scripts/eval/eval_lc_native.py \
      --model-path <step_dir_with_config.json_and_model_and_optim> \
      --out outputs/eval_results/<name>_native.json [--tokenizer Qwen/Qwen3-4B]

Run on a GPU node, env corpus-reasoning-olmo (has olmo_core + transformers), PYTHONPATH=corpus-reasoning.
"""
import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="step dir: has config.json + model_and_optim/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ruler-lengths", default="L1024,L2048")
    ap.add_argument("--ruler-subtasks",
                    default="niah_single,niah_multikey,niah_multivalue,niah_multiquery,vt,cwe,fwe")
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--contra-max-new-tokens", type=int, default=200,
                    help="generation budget for contradiction; enumerate-CoT answers on large-N "
                         "(e.g. n250) need ~2200 to reach the final 'Contradicting pairs:' line.")
    ap.add_argument("--nq-data", default="data/nq_validation_k20_hn2_600.jsonl")  # p10: 10% hard + CE filter
    ap.add_argument("--rerank-data", default="data/msmarco_dev_rerank_k20_1000.jsonl")
    ap.add_argument("--outlier-data", default="data/outlier_wiki100w_n20_k3_eval_100.jsonl")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--rerank-root", default="/scratch/users/prasann/cpt_data/rerank_ladder_src")
    ap.add_argument("--root", default=None,
                    help="chdir here before resolving relative data paths (on-cluster: mount the "
                         "eval dataset and pass its mountpoint so data/... and rerank_ladder_src/... resolve).")
    ap.add_argument("--ladder", action="store_true",
                    help="evaluate each task across its LENGTH LADDER (2k..64k); reports <task>_<rung>.")
    ap.add_argument("--ladder-tasks", default=None,
                    help="comma list restricting --ladder to a subset of tasks (split into per-task jobs).")
    ap.add_argument("--ladder-rungs", default=None,
                    help="comma list restricting --ladder to a subset of rungs (e.g. 16k,32k).")
    ap.add_argument("--ladder-version", choices=["v1", "v2"], default="v1",
                    help="v1 = original per-rung eval files (independently generated per rung). "
                         "v2 = cleaned ladders where every rung of a task shares the SAME >=500 "
                         "questions/answers and only the distractor documents vary (read entirely "
                         "from $EVAL500_ROOT/<task>/, i.e. point EVAL500_ROOT at the v2 bundle).")
    ap.add_argument("--skip-ruler", action="store_true")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip held-out retrieval generalization probes")
    ap.add_argument("--prompt-format", choices=["chat", "raw", "alpaca"], default="chat",
                    help="chat = Qwen3 apply_chat_template (matches SFT training); "
                         "raw = bare build_prompt, no wrapping (for BASE/CPT models); "
                         "alpaca = legacy alpaca-instruction wrap.")
    ap.add_argument("--save-generations", action=argparse.BooleanOptionalAction, default=True,
                    help="dump per-example model generations (+ gold/per-example metrics) to a sidecar "
                         "<out>.generations.jsonl for error inspection. On by default; --no-save-generations to skip.")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.root:
        os.chdir(args.root)

    from transformers import AutoTokenizer
    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from ctc_eval.eval.evaluate import (
        load_unified_examples, _eval_ruler, _eval_contradiction, _eval_retrieval,
        _eval_oolong, _eval_rerank, _eval_outlier,
    )

    # ---- data-parallel across N GPUs (torchrun): each rank loads a full model copy + evaluates a
    # SHARD of every example list; rank 0 gathers, scores, writes. world=1 -> single-GPU as before.
    import sys
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    is_main = (rank == 0)
    if not is_main:
        sys.stdout = open(os.devnull, "w")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    t0 = time.time()
    gen_cfg = GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                               max_length=args.max_length, use_cache=True)
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False,
    ).build(checkpoint_dir=args.model_path, device=device)
    print(f"[native] built generation module from {args.model_path} in {time.time()-t0:.1f}s", flush=True)

    def strip_think(s):
        return s.split("</think>", 1)[1] if "</think>" in s else s

    def _load(path, task, qp="both"):
        # Build prompts in the format the model expects:
        #   chat  -> Qwen3 chat template over the raw build_prompt (matches SFT training)
        #   raw   -> bare build_prompt, fed as a completion (BASE/CPT models)
        #   alpaca-> legacy alpaca-instruction wrap (build_prompt use_alpaca=True)
        ex = load_unified_examples(path, args.max_test_samples, task=task,
                                   query_position=qp, use_alpaca=(args.prompt_format == "alpaca"))
        if args.prompt_format == "chat":
            for e in ex:
                e["prompt"] = tok.apply_chat_template(
                    [{"role": "user", "content": e["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
        return ex

    @torch.no_grad()
    def generate(prompts, max_new_tokens, stop_strings=None):
        # DP: this rank handles global indices [rank, rank+world, ...]; gather to full ordered list.
        my_gidx = list(range(rank, len(prompts), world))
        lp = [prompts[i] for i in my_gidx]
        lout = []
        for i in range(0, len(lp), args.batch_size):
            chunk = lp[i:i + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length - max_new_tokens, add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            # Per-row string early-stop (stops near the actual answer length instead of running to
            # max_new_tokens). Decode-check runs every 16 steps to keep the loop GPU-bound.
            gen_kw = {}
            if stop_strings:
                gen_kw = dict(stop_strings=stop_strings, stop_string_check_interval=16,
                              stop_string_tokenizer=tok)
            cont, _, _ = gm.generate_batch(input_ids=ids, attention_mask=mask,
                                           completions_only=False, log_timing=False,
                                           max_new_tokens=max_new_tokens, **gen_kw)
            ctx_len = ids.shape[1]
            for row in cont.tolist():
                gen = row[ctx_len:]
                clean = []
                for t in gen:
                    if t in (tok.eos_token_id, tok.pad_token_id):
                        break
                    clean.append(t)
                lout.append(strip_think(tok.decode(clean, skip_special_tokens=True)))
        full = [None] * len(prompts)
        if world > 1:
            parts = [None] * world
            torch.distributed.all_gather_object(parts, list(zip(my_gidx, lout)))
            for part in parts:
                for gi, resp in part:
                    full[gi] = resp
        else:
            for gi, resp in zip(my_gidx, lout):
                full[gi] = resp
        return full

    summary = {"model_path": args.model_path, "ruler": {}, "contradiction": {}, "nq": {}}

    # Per-example generation dump (for error inspection). Each _eval_* returns (metrics, details);
    # we pair the FULL model generation with its per-example detail (parsed pred, gold, metrics) and
    # a prompt tail (prompts can be 100k+ chars, so only the trailing question region is kept).
    gen_dump = []

    def _record_gens(task, label, examples, responses, details):
        if not args.save_generations:
            return
        for i, resp in enumerate(responses):
            ex_i = examples[i] if i < len(examples) else None
            prompt = ex_i.get("prompt", "") if isinstance(ex_i, dict) else ""
            rec = {"task": task, "rung": label, "idx": i,
                   "generation": resp,
                   "prompt_tail": prompt[-1200:] if prompt else None}
            if details is not None and i < len(details):
                rec["detail"] = details[i]
            gen_dump.append(rec)

    if not args.skip_ruler:
        recalls = []
        for sub in args.ruler_subtasks.split(","):
            for L in args.ruler_lengths.split(","):
                path = os.path.join(args.data_dir, f"ruler_{sub}_{L}_eval.jsonl")
                if not os.path.exists(path):
                    continue
                ex = _load(path, task="ruler", qp="after")
                resp = generate([e["prompt"] for e in ex], 160)
                res, det = _eval_ruler(ex, resp)
                _record_gens("ruler", f"{sub}_{L}", ex, resp, det)
                summary["ruler"][f"{sub}_{L}"] = res
                recalls.append(res["recall"])
                print(f"[ruler] {sub}_{L}: recall={res['recall']:.3f} (n={len(ex)})", flush=True)
        summary["ruler_avg_recall"] = sum(recalls) / len(recalls) if recalls else None

    if not args.ladder:
        ex = _load(args.contra_data, task="contradiction", qp="both")
        resp = generate([e["prompt"] for e in ex], args.contra_max_new_tokens, stop_strings=["contradicting pairs:"])
        res, det = _eval_contradiction(ex, resp)
        _record_gens("contradiction", "single", ex, resp, det)
        summary["contradiction"] = res
        print(f"[contradiction] f1={res['f1']:.3f} (n={len(ex)})", flush=True)

    if not args.ladder and os.path.exists(args.nq_data):
        ex = _load(args.nq_data, task="retrieval", qp="both")
        resp = generate([e["prompt"] for e in ex], 64)
        res, det = _eval_retrieval(ex, resp)
        _record_gens("nq", "single", ex, resp, det)
        summary["nq"] = res
        print(f"[nq] f1={res.get('f1', 0):.3f} (n={len(ex)})", flush=True)

    # ---- LENGTH-LADDER: each task at 2k..64k (reports <task>_<rung>), mirrors the landmark driver ----
    if args.ladder:
        RR = args.rerank_root
        # n>=500 eval at the goal-critical rungs (8k/16k/32k) from cpt_data/eval500; 64k dropped
        # (beyond the 32k goal, saves GPU). 2k/3k base + oolong (capped ~80) keep their files.
        E5 = os.environ.get("EVAL500_ROOT", "/scratch/users/prasann/cpt_data/eval500")
        if args.ladder_version == "v2":
            # v2: every rung of a task shares the SAME >=500 questions/answers; only the
            # distractor documents differ (built by build_v2_eval_ladders.py). ALL rungs live
            # under $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at the v2 bundle). oolong rungs are
            # freshly synthesized so they are DISJOINT from training (the v1 oolong eval overlapped
            # its own train split) while keeping the same question-type + corpus-type distribution.
            LADDERS = {
                "contradiction": [("2k", f"{E5}/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"),
                    ("8k", f"{E5}/contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
                    ("16k", f"{E5}/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
                    ("32k", f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl")],
                "nq": [("3k", f"{E5}/nq/nq_validation_k20_600.jsonl"),
                    ("8k", f"{E5}/nq/nq_validation_k50_600.jsonl"),
                    ("16k", f"{E5}/nq/nq_validation_k100_600.jsonl"),
                    ("32k", f"{E5}/nq/nq_validation_k200_600.jsonl")],
                "outlier": [("3k", f"{E5}/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
                    ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
                    ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
                    ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl")],
                # CE-graded (NDCG@10 + Kendall-tau), shared 500 queries; tops out at k100 (~16k) —
                # no CE-graded pool larger than k100 exists, so rerank has no 32k rung.
                "rerank": [("3k", f"{E5}/rerank/msmarco_trainhn_eval_k20_500.jsonl"),
                    ("8k", f"{E5}/rerank/msmarco_trainhn_eval_k50_500.jsonl"),
                    ("16k", f"{E5}/rerank/msmarco_trainhn_eval_k100_500.jsonl")],
                "oolong": [("8k", f"{E5}/oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
                    ("16k", f"{E5}/oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
                    ("32k", f"{E5}/oolong/oolong_test_synth_ctx32768_spliteval.jsonl")],
            }
        else:
          LADDERS = {
            "contradiction": [("2k", args.contra_data),
                ("8k", f"{E5}/contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl")],
            # p10 pipeline: 10% hard negs + CE gold-quality filter, all docs from wikipedia-dpr-100w
            # (matches the training-data negative distribution; the old hn49/hn99/hn199 files were 98% hard).
            "nq": [("3k", args.nq_data),
                ("8k", f"{E5}/nq/nq_validation_k50_hn5_600.jsonl"),
                ("16k", f"{E5}/nq/nq_validation_k100_hn10_600.jsonl"),
                ("32k", f"{E5}/nq/nq_validation_k200_hn20_600.jsonl")],
            "oolong": [("1k", "data/oolong_test_synth_ctx1024_spliteval.jsonl"),
                ("2k", "data/oolong_test_synth_ctx2048_spliteval.jsonl"),
                ("4k", "data/oolong_test_synth_ctx4096_spliteval.jsonl"),
                ("8k", "data/oolong_test_synth_ctx8192_spliteval.jsonl"),
                ("16k", "data/oolong_test_synth_ctx16384_spliteval.jsonl"),
                ("32k", "data/oolong_test_synth_ctx32768_spliteval.jsonl")],
            "rerank": [("2k", args.rerank_data),
                ("8k", f"{E5}/rerank/msmarco_validation_rerank_k80_600.jsonl"),
                ("16k", f"{E5}/rerank/msmarco_validation_rerank_k158_597.jsonl"),
                ("32k", f"{E5}/rerank/msmarco_validation_rerank_k315_599.jsonl")],
            "outlier": [("3k", args.outlier_data),
                ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
                ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
                ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl")],
        }
        LSPEC = {
            "contradiction": ("contradiction", _eval_contradiction, "f1", 200),
            "nq": ("retrieval", _eval_retrieval, "f1", 64),
            "oolong": ("oolong", _eval_oolong, "score", 200),
            "rerank": ("rerank", _eval_rerank, None, 256),
            "outlier": ("outlier", _eval_outlier, "f1", 200),
        }
        task_filter = set(args.ladder_tasks.split(",")) if args.ladder_tasks else None
        rung_filter = set(args.ladder_rungs.split(",")) if args.ladder_rungs else None
        for task, rungs in LADDERS.items():
            if task_filter and task not in task_filter:
                continue
            if rung_filter:
                rungs = [(lab, p) for (lab, p) in rungs if lab in rung_filter]
            loadtask, fn, pkey, maxtok = LSPEC[task]
            # contradiction = NO-COT direct pairs: short budget + early-stop on the answer line
            # (think-strip already applied in generate(); no newline-stop).
            gkw = {}
            if task == "contradiction":
                maxtok = args.contra_max_new_tokens
                gkw = {"stop_strings": ["contradicting pairs:"]}
            for label, path in rungs:
                if not path or not os.path.exists(path):
                    print(f"[ladder:{task}@{label}] MISSING {path}, skipping"); continue
                ex = _load(path, task=loadtask, qp="both")
                resp = generate([e["prompt"] for e in ex], maxtok, **gkw)
                res, det = fn(ex, resp)
                _record_gens(task, label, ex, resp, det)
                prim = res.get(pkey) if pkey else next(
                    (v for k, v in res.items() if k.startswith("mrr")), None)
                summary[f"{task}_{label}"] = prim
                print(f"[ladder:{task}@{label}] {pkey or 'mrr'}="
                      f"{prim if prim is None else round(prim,3)} (n={len(ex)})", flush=True)

    # held-out retrieval generalization probes (eval-only) — same task="retrieval" path as NQ.
    if not args.skip_gen and not args.ladder:
        gen = [
            ("hpqa",    "data/n2ified_eval_hpqa_q20.jsonl",       256),
            ("fiqa",    "data/beir_fiqa_ce_test_k20_648.jsonl",    64),
            ("msmarco", "data/msmarco_trecdl2019_k20_43.jsonl",    64),
            ("scifact", "data/beir_scifact_test_k20_300.jsonl",    64),
        ]
        for gname, gpath, gmax in gen:
            if not os.path.exists(gpath):
                print(f"[gen:{gname}] MISSING {gpath}, skipping"); continue
            ex = load_unified_examples(gpath, args.max_test_samples, task="retrieval",
                                       query_position="both", use_alpaca=True)
            resp = generate([e["prompt"] for e in ex], gmax)
            res, det = _eval_retrieval(ex, resp)
            _record_gens(f"gen_{gname}", "probe", ex, resp, det)
            summary[f"gen_{gname}"] = res
            print(f"[gen:{gname}] f1={res.get('f1', 0):.3f} (n={len(ex)})", flush=True)

    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        if args.save_generations and gen_dump:
            gen_path = os.path.splitext(args.out)[0] + ".generations.jsonl"
            with open(gen_path, "w") as gf:
                for rec in gen_dump:
                    gf.write(json.dumps(rec) + "\n")
            print(f"[native] wrote {len(gen_dump)} generations -> {gen_path}", flush=True)
        print(f"\n[native] TOTAL {time.time()-t0:.1f}s | RULER {summary.get('ruler_avg_recall')} "
              f"contra {summary['contradiction'].get('f1')} nq {summary['nq'].get('f1')}\nWROTE {args.out}")
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
