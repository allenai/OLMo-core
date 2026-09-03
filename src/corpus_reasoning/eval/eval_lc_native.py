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
    ap.add_argument("--nq-data", default="data/nq_validation_k20_hn19_500_aligned.jsonl")
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
    ap.add_argument("--ladder-version", choices=["v2"], default="v2",
                    help="v2 (DEFAULT) = cleaned ladders where every rung of a task shares the SAME "
                         "500 questions/answers and only distractors vary (all rungs read from "
                         "$EVAL500_ROOT/<task>/; point EVAL500_ROOT at the v2 bundle). "
                         "v1 = original independently-generated per-rung files.")
    ap.add_argument("--eos-token-id", type=int, default=None,
                    help="override the stop token for generation + decode. Qwen3 no-cot SFT models were "
                         "trained to stop on 151643 (<|endoftext|>), but tok.eos_token_id resolves to "
                         "151645 (<|im_end|>); without this override generation never stops and rambles "
                         "past the answer -> precision/EM collapse. Pass 151643 for these runs.")
    ap.add_argument("--skip-ruler", action="store_true")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip held-out retrieval generalization probes")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.root:
        os.chdir(args.root)

    from transformers import AutoTokenizer
    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from corpus_reasoning.eval.evaluate import (
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
    eos_id = args.eos_token_id if args.eos_token_id is not None else tok.eos_token_id
    # GenerationConfig requires pad != eos. Qwen3's tok.pad_token_id (151643) collides with the
    # correct stop token, so fall back to <|im_end|> (tok.eos_token_id) for padding in that case.
    pad_id = tok.pad_token_id if tok.pad_token_id != eos_id else tok.eos_token_id
    # Break set for the post-hoc decode loop: the model's real stop token AND the tokenizer's
    # eos/pad, so we truncate at whichever appears first.
    stop_ids = {eos_id, tok.eos_token_id, tok.pad_token_id}
    gen_cfg = GenerationConfig(eos_token_id=eos_id, pad_token_id=pad_id,
                               max_length=args.max_length, use_cache=True)
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False,
    ).build(checkpoint_dir=args.model_path, device=device)
    print(f"[native] built generation module from {args.model_path} in {time.time()-t0:.1f}s", flush=True)

    def strip_think(s):
        return s.split("</think>", 1)[1] if "</think>" in s else s

    @torch.no_grad()
    def generate(prompts, max_new_tokens, stop_strings=None):
        # DP: this rank handles global indices [rank, rank+world, ...]; gather to full ordered list.
        my_gidx = list(range(rank, len(prompts), world))
        lp = [prompts[i] for i in my_gidx]
        lout = []
        cap = args.max_length - max_new_tokens
        for i in range(0, len(lp), args.batch_size):
            chunk = lp[i:i + args.batch_size]
            # truncation=False is DELIBERATE: the cut lands on the prompt TAIL, where the question
            # lives, so a truncated example scores the model on input it never saw (f1 0.000 at
            # parse_rate 1.0, indistinguishable from a real long-context collapse). Raise instead.
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=False,
                      add_special_tokens=False)
            lens = enc["attention_mask"].sum(dim=1).tolist()
            if max(lens, default=0) > cap:
                over = [(my_gidx[i + j], n) for j, n in enumerate(lens) if n > cap]
                worst = max(n for _, n in over)
                raise SystemExit(
                    f"[maxlen] {len(over)}/{len(chunk)} prompts exceed the {cap}-token prompt cap "
                    f"(--max-length {args.max_length} minus max_new_tokens {max_new_tokens}); "
                    f"longest is {worst} tokens (example indices {[g for g, _ in over][:8]}). "
                    f"Re-run with --max-length >= {worst + max_new_tokens}."
                )
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
                    if t in stop_ids:
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

    # `eval_size` maps each reported metric key -> how many EVAL EXAMPLES it was scored on.
    # Deliberately not called "n": in this project `n` means CORPUS size (docs per example,
    # e.g. the n100 in contradiction_eval_..._n100_k3), never eval-set size.
    summary = {"model_path": args.model_path, "ruler": {}, "contradiction": {}, "nq": {},
               "eval_size": {}}

    if not args.skip_ruler:
        recalls = []
        for sub in args.ruler_subtasks.split(","):
            for L in args.ruler_lengths.split(","):
                path = os.path.join(args.data_dir, f"ruler_{sub}_{L}_eval.jsonl")
                if not os.path.exists(path):
                    continue
                ex = load_unified_examples(path, args.max_test_samples, task="ruler",
                                           query_position="after", use_alpaca=True)
                resp = generate([e["prompt"] for e in ex], 160)
                res, _ = _eval_ruler(ex, resp)
                summary["ruler"][f"{sub}_{L}"] = res
                summary["eval_size"][f"ruler_{sub}_{L}"] = len(ex)
                recalls.append(res["recall"])
                print(f"[ruler] {sub}_{L}: recall={res['recall']:.3f} (eval_size={len(ex)})", flush=True)
        summary["ruler_avg_recall"] = sum(recalls) / len(recalls) if recalls else None

    if not args.ladder:
        ex = load_unified_examples(args.contra_data, args.max_test_samples, task="contradiction",
                                   query_position="both", use_alpaca=True)
        gens = generate([e["prompt"] for e in ex], args.contra_max_new_tokens,
                        stop_strings=["contradicting pairs:", "]]"])
        # no-cot answers are EXACTLY one JSON pair list; truncate at the first complete list ("]]")
        # so trailing rambling (these models never emit a clean EOS) can't inject spurious pairs and
        # collapse precision. Matches the docchunk harness's no-cot early-stop.
        gens = [g[: g.index("]]") + 2] if "]]" in g else g for g in gens]
        res, _ = _eval_contradiction(ex, gens)
        summary["contradiction"] = res
        summary["eval_size"]["contradiction"] = len(ex)
        print(f"[contradiction] f1={res['f1']:.3f} (eval_size={len(ex)})", flush=True)

    if not args.ladder and os.path.exists(args.nq_data):
        ex = load_unified_examples(args.nq_data, args.max_test_samples, task="retrieval",
                                   query_position="both", use_alpaca=True)
        res, _ = _eval_retrieval(ex, generate([e["prompt"] for e in ex], 64))
        summary["nq"] = res
        summary["eval_size"]["nq"] = len(ex)
        print(f"[nq] f1={res.get('f1', 0):.3f} (eval_size={len(ex)})", flush=True)

    # ---- LENGTH-LADDER: each task at 2k..64k (reports <task>_<rung>), mirrors the landmark driver ----
    if args.ladder:
        RR = args.rerank_root
        # n>=500 eval at the goal-critical rungs (8k/16k/32k) from cpt_data/eval500; 64k dropped
        # (beyond the 32k goal, saves GPU). 2k/3k base + oolong (capped ~80) keep their files.
        E5 = os.environ.get("EVAL500_ROOT", "/scratch/users/prasann/cpt_data/eval500")
        # v1 ladders are DISABLED (2026-07-29). Each v1 rung drew its OWN questions, so every
        # rung-to-rung delta carried eval-set resampling noise on top of the length effect it was
        # meant to isolate. v2 fixes the question set across rungs and varies only the distractors.
        if args.ladder_version != "v2":
            raise NotImplementedError(
                f"--ladder-version {args.ladder_version!r} is no longer supported: v2 is the only "
                "ladder. Build what you need as v2 (build_v2_eval_ladders.py for 2k-32k, "
                "build_xlong_rungs.py for 64k-2M) and point EVAL500_ROOT at a v2 bundle."
            )
        if args.ladder_version == "v2":
            # v2: every rung shares the SAME >=500 questions; only distractors vary. ALL rungs live
            # under $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at .../eval500_v2). oolong is freshly
            # synthesized (disjoint from training) with the same type+corpus distribution across rungs.
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
            "nq": [("3k", args.nq_data),
                ("8k", f"{E5}/nq/nq_validation_k50_hn49_600.jsonl"),
                ("16k", f"{E5}/nq/nq_validation_k100_hn99_600.jsonl"),
                ("32k", f"{E5}/nq/nq_validation_k200_hn199_600.jsonl")],
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
                ex = load_unified_examples(path, args.max_test_samples, task=loadtask,
                                           query_position="both", use_alpaca=True)
                res, _ = fn(ex, generate([e["prompt"] for e in ex], maxtok, **gkw))
                prim = res.get(pkey) if pkey else next(
                    (v for k, v in res.items() if k.startswith("mrr")), None)
                # rerank (pkey is None) carries NDCG@10 + Kendall-tau in res; keep the
                # full dict so those CE-graded metrics survive into the out JSON.
                summary[f"{task}_{label}"] = res if pkey is None else prim
                summary["eval_size"][f"{task}_{label}"] = len(ex)
                extra = ""
                if pkey is None and isinstance(res, dict):
                    extra = (f" ndcg@10={res.get('ndcg@10')} "
                             f"tau={res.get('kendall_tau')} mrr@10={res.get('mrr@10')}")
                print(f"[ladder:{task}@{label}] {pkey or 'mrr'}="
                      f"{prim if prim is None else round(prim,3)}{extra} (eval_size={len(ex)})", flush=True)

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
            res, _ = _eval_retrieval(ex, generate([e["prompt"] for e in ex], gmax))
            summary[f"gen_{gname}"] = res
            summary["eval_size"][f"gen_{gname}"] = len(ex)
            print(f"[gen:{gname}] f1={res.get('f1', 0):.3f} (eval_size={len(ex)})", flush=True)

    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"\n[native] TOTAL {time.time()-t0:.1f}s | RULER {summary.get('ruler_avg_recall')} "
              f"contra {summary['contradiction'].get('f1')} nq {summary['nq'].get('f1')}\nWROTE {args.out}")
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
