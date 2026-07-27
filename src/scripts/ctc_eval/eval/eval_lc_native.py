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
    ap.add_argument("--ladder-version", choices=["v1", "v2"], default="v2",
                    help="v2 (DEFAULT) = cleaned ladders where every rung of a task shares the SAME "
                         "500 questions/answers and only the distractor documents vary (read entirely "
                         "from $EVAL500_ROOT/<task>/, i.e. point EVAL500_ROOT at the v2 bundle). "
                         "v1 = original independently-generated per-rung eval files.")
    ap.add_argument("--xlong", action="store_true",
                    default=os.environ.get("LADDER_XLONG") == "1",
                    help="OPT-IN: append the ultra-long 64k/128k/256k rungs (built offline by "
                         "scripts/data/build_xlong_rungs.py) to each task's v2 ladder. OFF by "
                         "default; also honors env LADDER_XLONG=1. Auto-raises --max-length to fit "
                         "the largest selected xlong rung (else long prompts get truncated).")
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
    # xlong opt-in: the runner truncates prompts to (max_length - max_new_tokens), so max_length
    # MUST cover the largest selected ultra-long rung, and it feeds the gen budget built below.
    if args.xlong:
        _XL_TOK = {"64k": 65536, "128k": 131072, "256k": 262144}
        _sel = set(args.ladder_rungs.split(",")) if args.ladder_rungs else set(_XL_TOK)
        _need = max((t for r, t in _XL_TOK.items() if r in _sel), default=0)
        if _need and args.max_length < _need + 1024:
            print(f"[xlong] raising --max-length {args.max_length} -> {_need + 1024}", flush=True)
            args.max_length = _need + 1024
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
        #
        # Bundle resolution. The old default was the **v1** `cpt_data/eval500` directory even when
        # --ladder-version v2 (the default) was selected, so forgetting to export EVAL500_ROOT
        # silently resolved v2 rung filenames against a v1 tree and produced mostly-missing rungs.
        # Now: explicit env wins; otherwise pick the first bundle that actually EXISTS for the
        # selected ladder version -- the weka 2k..256k bundle (mounted on Beaker) before the
        # Berkeley-local /scratch copies.
        # ⚠ The two v2 bundles are NOT equivalent at the long end. The weka bundle carries the
        # 2k..256k ladder rebuilt at eval_size=500 (plus oolong 2k/4k); the Berkeley-local
        # eval500_v2 still holds the ORIGINAL xlong rungs at eval_size=300, which the --xlong glob
        # will happily pick up. If you are quoting 64k/128k/256k numbers from a Berkeley run,
        # point EVAL500_ROOT at the weka bundle (or a copy of it) rather than relying on this
        # fallback, or the rungs are 300 examples and need an inline eval_size warning.
        _V2_BUNDLES = [
            "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/eval",
            "/scratch/users/prasann/cpt_data/eval500_v2",
        ]
        _V1_BUNDLES = ["/scratch/users/prasann/cpt_data/eval500"]
        E5 = os.environ.get("EVAL500_ROOT")
        if not E5:
            _cands = _V2_BUNDLES if args.ladder_version == "v2" else _V1_BUNDLES
            E5 = next((p for p in _cands if os.path.isdir(p)), _cands[-1])
            print(f"[ladder] EVAL500_ROOT unset -> {E5} (ladder_version={args.ladder_version})",
                  flush=True)
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
                # OOD generalization (held-out BEIR retrieval, graded as retrieval f1). Version-agnostic:
                # rungs are fixed doc-pool sizes subsampled from the k100 CE pools (subsample_beir_ladder.py),
                # labelled by median prompt length. fiqa docs are short -> tops ~16k; scifact ~32k.
                "fiqa": [("2k", f"{E5}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
                    ("4k", f"{E5}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
                    ("8k", f"{E5}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
                    ("16k", f"{E5}/beir/beir_fiqa_ce_ladder_k80_648.jsonl")],
                "scifact": [("4k", f"{E5}/beir/beir_scifact_ladder_k11_299.jsonl"),
                    ("8k", f"{E5}/beir/beir_scifact_ladder_k22_299.jsonl"),
                    ("16k", f"{E5}/beir/beir_scifact_ladder_k44_299.jsonl"),
                    ("32k", f"{E5}/beir/beir_scifact_ladder_k88_299.jsonl")],
                # OOD generalization for the outlier + contradiction tasks (different passage/sentence
                # source than the in-distribution wiki100w / pubmed). Graded identically (gold_doc_indices).
                "outlier_review": [("3k", f"{E5}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
                    ("8k", f"{E5}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
                    ("16k", f"{E5}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
                    ("32k", f"{E5}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl")],
                "contra_fever": [("2k", f"{E5}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
                    ("8k", f"{E5}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
                    ("16k", f"{E5}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
                    ("32k", f"{E5}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl")],
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
            # OOD generalization (held-out BEIR retrieval, graded as retrieval f1); same files for v1/v2.
            "fiqa": [("2k", f"{E5}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
                ("4k", f"{E5}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
                ("8k", f"{E5}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
                ("16k", f"{E5}/beir/beir_fiqa_ce_ladder_k80_648.jsonl")],
            "scifact": [("4k", f"{E5}/beir/beir_scifact_ladder_k11_299.jsonl"),
                ("8k", f"{E5}/beir/beir_scifact_ladder_k22_299.jsonl"),
                ("16k", f"{E5}/beir/beir_scifact_ladder_k44_299.jsonl"),
                ("32k", f"{E5}/beir/beir_scifact_ladder_k88_299.jsonl")],
            # OOD generalization for outlier + contradiction (review / FEVER source); same files v1/v2.
            "outlier_review": [("3k", f"{E5}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
                ("8k", f"{E5}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
                ("16k", f"{E5}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
                ("32k", f"{E5}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl")],
            "contra_fever": [("2k", f"{E5}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
                ("8k", f"{E5}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl")],
        }
        # ---- oolong SHORT rungs (2k/4k): extend the ladder DOWNWARD ----
        # The v2 oolong ladder starts at 8k because no shorter synthesized rungs existed. Added
        # conditionally so an EVAL500_ROOT without them still works. Prepended, not appended, so
        # the rungs stay in ascending length order.
        if args.ladder_version == "v2" and "oolong" in LADDERS:
            _short = []
            for _lab, _ctx in (("2k", 2048), ("4k", 4096)):
                _p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{_ctx}_spliteval.jsonl")
                if os.path.exists(_p):
                    _short.append((_lab, _p))
            if _short:
                LADDERS["oolong"] = _short + LADDERS["oolong"]

        # ---- OPT-IN ultra-long rungs (64k/128k/256k), OFF by default (v2 only) ----
        # Resolved by size-labelled glob so the calibrated doc-count in the filename can drift
        # (rebuild with a different --count/tokenizer) without editing this file.
        if args.xlong and args.ladder_version == "v2":
            import glob as _glob
            # rerank and oolong were originally excluded: no CE-graded rerank pool above k100
            # existed, and oolong is not a doc pool. Both now have xlong rungs (built 2026-07-27),
            # so they are wired here. oolong does NOT use the `_xlong_` convention -- it is a packed
            # item stream labelled by its token budget (ctx{N}_spliteval), so it gets its own map.
            _XL = {
                "contradiction": ("contra",  "contradiction_eval_pubmed_both_n*_k3_xlong_{s}.jsonl"),
                "nq":            ("nq",       "nq_validation_k*_xlong_{s}.jsonl"),
                "outlier":       ("outlier",  "outlier_wiki100w_n*_k3_eval_xlong_{s}.jsonl"),
                # ⚠ APPROXIMATE above k100: the added negatives are random non-gold docs carrying
                # ce=None, not CE-mined hard negatives. The grader scores ce=None as gain 0 and
                # excludes them from the Kendall-tau reference, so NDCG@10 still measures "surface
                # the CE-relevant docs among far more noise" -- but it is MORE noise, not HARDER
                # noise. Flag that next to any rerank number at 64k+.
                "rerank":        ("rerank",  "msmarco_trainhn_eval_k*_xlong_{s}.jsonl"),
            }
            # oolong: token-budget-labelled files rather than a calibrated doc count.
            _XL_OOLONG = {"64k": 65536, "128k": 131072, "256k": 262144}
            for _t, (_sub, _pat) in _XL.items():
                if _t not in LADDERS:
                    continue
                for _s in ("64k", "128k", "256k"):
                    _hits = sorted(_glob.glob(os.path.join(E5, _sub, _pat.format(s=_s))))
                    if _hits:
                        LADDERS[_t].append((_s, _hits[0]))
            if "oolong" in LADDERS:
                for _s, _ctx in _XL_OOLONG.items():
                    _p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{_ctx}_spliteval.jsonl")
                    if os.path.exists(_p):
                        LADDERS["oolong"].append((_s, _p))
            print(f"[xlong] appended ultra-long rungs where files exist under {E5}", flush=True)

        # ---- Resolve check: a missing rung file must not pass silently ----
        # The rungs are literal paths; if the bundle is the wrong version (or not staged) the
        # runner would otherwise skip rungs and report a partial ladder as if it were complete.
        _missing = [(t, lab, p) for t, rl in LADDERS.items() for lab, p in rl
                    if not os.path.exists(p)]
        if _missing:
            print(f"[ladder] WARNING: {len(_missing)} rung file(s) MISSING under {E5}:", flush=True)
            for _t, _lab, _p in _missing[:20]:
                print(f"    {_t:>14} {_lab:>5}  {_p}", flush=True)
            print("[ladder] set EVAL500_ROOT to a bundle containing these rungs, or restrict with "
                  "--ladder-tasks / --ladder-rungs.", flush=True)
            LADDERS = {t: [(lab, p) for lab, p in rl if os.path.exists(p)]
                       for t, rl in LADDERS.items()}
            LADDERS = {t: rl for t, rl in LADDERS.items() if rl}
        LSPEC = {
            "contradiction": ("contradiction", _eval_contradiction, "f1", 200),
            "nq": ("retrieval", _eval_retrieval, "f1", 64),
            "oolong": ("oolong", _eval_oolong, "score", 200),
            "rerank": ("rerank", _eval_rerank, None, 256),
            "outlier": ("outlier", _eval_outlier, "f1", 200),
            "fiqa": ("retrieval", _eval_retrieval, "f1", 64),
            "scifact": ("retrieval", _eval_retrieval, "f1", 64),
            "outlier_review": ("outlier", _eval_outlier, "f1", 200),
            "contra_fever": ("contradiction", _eval_contradiction, "f1", 200),
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
            if loadtask == "contradiction":
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
