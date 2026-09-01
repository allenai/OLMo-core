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
    ap.add_argument("--ladder-version", choices=["v2", "v3", "fast"], default="v2",
                    help="v2 is the ONLY supported ladder: every rung of a task shares the SAME "
                         "500 questions/answers and only the distractor documents vary, read "
                         "entirely from $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at the v2 "
                         "bundle). v1 (independently-generated per-rung files) is REMOVED -- "
                         "passing it raises NotImplementedError, because per-rung question "
                         "resampling put eval-set noise into every rung-to-rung delta.")
    ap.add_argument("--xlong", action="store_true",
                    default=os.environ.get("LADDER_XLONG") == "1",
                    help="OPT-IN: append the ultra-long 64k..2M rungs (built offline by "
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
    ap.add_argument("--landmark-mem-id", type=int, default=None,
                    help="Landmark token id for landmark-attention ckpts (Qwen3.5: 248200).")
    ap.add_argument("--landmark-pad-id", type=int, default=None,
                    help="Landmark pad id (Qwen3.5: 248203).")
    ap.add_argument("--eos-token-id", type=int, default=None,
                    help="Override the generation stop token (SFT-trained eos, e.g. 248044 for "
                         "shards built with convert_unified_to_sft --eos 248044).")
    ap.add_argument("--query-position", choices=["both", "after", "before"], default="both",
                    help="Where the task ask is rendered relative to the corpus. MUST match the "
                         "shards the model was SFT'd on: the xlong5_2k256k_qwen35 build is 'both', "
                         "the ..._qafter build is 'after'. Evaluating a query-after model with "
                         "'both' shows it a second copy of the ask it never saw in training, which "
                         "reads as a capability gap rather than a prompt mismatch. Default 'both' "
                         "keeps every existing result reproducible. RULER is exempt -- it is not "
                         "from the 5-task mix and is always rendered query-after.")
    ap.add_argument("--save-generations", action=argparse.BooleanOptionalAction, default=True,
                    help="dump per-example model generations (+ gold/per-example metrics) to a sidecar "
                         "<out>.generations.jsonl for error inspection. On by default; --no-save-generations to skip.")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # xlong opt-in: the runner truncates prompts to (max_length - max_new_tokens), so max_length
    # MUST cover the largest selected ultra-long rung, and it feeds the gen budget built below.
    if args.xlong:
        _XL_TOK = {"64k": 65536, "128k": 131072, "256k": 262144,
                   "512k": 524288, "1M": 1048576, "2M": 2097152}
        _sel = set(args.ladder_rungs.split(",")) if args.ladder_rungs else set(_XL_TOK)
        _need = max((t for r, t in _XL_TOK.items() if r in _sel), default=0)
        # A rung's LABEL is its nominal token budget, but the built prompt reliably lands slightly
        # ABOVE it: the doc count is calibrated from a median, and the instruction/query/marker wrap
        # is added on top. Measured through this exact load path (Qwen3.5, 2026-07-29):
        #   512k -> 535,855 (1.022x)   nq 528,021 (1.007x)   outlier 526,249 (1.004x)
        #   rerank 530,993 (1.013x)    2M -> 2,165,314 (1.033x)
        # so the old `+1024` slack was NOT enough and silently truncated the prompt TAIL -- which is
        # where the question lives. That yields an empty/garbage generation and scores f1 0.000 at
        # parse_rate 1.0 for a model that is fine (the maxlen-truncation trap). Use a 10% margin,
        # which covers the per-example max above the median with room to spare.
        _budget = int(_need * 1.10) + 2048 if _need else 0
        if _budget and args.max_length < _budget:
            print(f"[xlong] raising --max-length {args.max_length} -> {_budget} "
                  f"(rung label {_need} + 10% margin; prompts run ~0.4-3.3% over label)", flush=True)
            args.max_length = _budget
        # Rungs past 262,144 exceed Qwen3.5's native max_position_embeddings. Scoring them at all
        # requires a RoPE-extended (YaRN) copy of the checkpoint -- without one the model silently
        # reads garbage positions and the rung looks like a long-context collapse that is really a
        # config error. Warn loudly; the caller is responsible for pointing --model-path at the
        # extended copy (debug/ctx_ceiling_4b/make_yarn_copy.py).
        if _need > 262144:
            print(f"[xlong] ⚠ selected rungs reach {_need} tokens, PAST the Qwen3.5 native "
                  f"262144 position limit -- this measures RoPE-EXTENDED extrapolation. Point "
                  f"--model-path at a YaRN serving copy and label every >256k number as such.",
                  flush=True)
    if args.root:
        os.chdir(args.root)

    from transformers import AutoTokenizer
    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from ctc_eval.eval.evaluate import (
        load_unified_examples, _eval_ruler, _eval_contradiction, _eval_retrieval,
        _eval_oolong, _eval_rerank, _eval_outlier, _eval_qdmatch, _eval_absence,
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
    _lm_kwargs = {}
    if args.landmark_mem_id is not None:
        _lm_kwargs = dict(landmark_mem_id=args.landmark_mem_id,
                          landmark_pad_id=args.landmark_pad_id)
    _eos = args.eos_token_id if args.eos_token_id is not None else tok.eos_token_id
    _pad = tok.pad_token_id if tok.pad_token_id != _eos else tok.eos_token_id
    gen_cfg = GenerationConfig(eos_token_id=_eos, pad_token_id=_pad,
                               max_length=args.max_length, use_cache=True, **_lm_kwargs)
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False,
    ).build(checkpoint_dir=args.model_path, device=device)
    print(f"[native] built generation module from {args.model_path} in {time.time()-t0:.1f}s", flush=True)

    def strip_think(s):
        return s.split("</think>", 1)[1] if "</think>" in s else s

    def _load(path, task, qp=None):
        qp = args.query_position if qp is None else qp
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

    summary = {"model_path": args.model_path, "query_position": args.query_position,
               "prompt_format": args.prompt_format,
               "ruler": {}, "contradiction": {}, "nq": {}}

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
        ex = _load(args.contra_data, task="contradiction")
        resp = generate([e["prompt"] for e in ex], args.contra_max_new_tokens, stop_strings=["contradicting pairs:"])
        res, det = _eval_contradiction(ex, resp)
        _record_gens("contradiction", "single", ex, resp, det)
        summary["contradiction"] = res
        print(f"[contradiction] f1={res['f1']:.3f} (n={len(ex)})", flush=True)

    if not args.ladder and os.path.exists(args.nq_data):
        ex = _load(args.nq_data, task="retrieval")
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
        # ⚠ These bundles are NOT equivalent at the long end, so ORDER MATTERS.
        #   1. _eval_bundle_eval500_v2_clean -- the DEFAULT (2026-07-29): a verified 2k..2M ladder,
        #      eval_size>=500 at every rung, PubMed-only contradiction distractors.
        #   2. xlong5_2k256k_qwen35/eval    -- build-output bundle; contra fillers are ~29% FEVER/wiki
        #      and it stops at 256k.
        #   3. /scratch/.../eval500_v2      -- Berkeley-local; its xlong rungs are eval_size=300.
        # Falling back past (1) means quoting numbers off a contaminated or sub-500 rung, so a run
        # that lands on (2) or (3) needs the contamination/eval_size caveats stated inline. Prefer
        # setting EVAL500_ROOT explicitly over relying on this order.
        _V2_BUNDLES = [
            "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean",
            "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/eval",
            "/scratch/users/prasann/cpt_data/eval500_v2",
        ]
        # ---- V3 (train/eval IID) bundle --------------------------------------------------
        # v3 == v2 with ONE change: contradiction's rungs are `realistic`-mode, matching the
        # perturbation generator the training data actually used. v2 scores a realistic-trained
        # model on `both`-mode gold (Jaccard 0.388 with 38% of pairs >0.5, i.e. largely
        # near-duplicate detection) against training's 0.306 with a hard 0.500 cap and zero mass
        # above it. On the CTC checkpoint that mismatch alone was worth 0.559 -> 0.946 f1 at n=762.
        # nq / rerank / oolong / outlier rungs are UNCHANGED: all four were audited 2026-08-11 and
        # are already in-distribution (see records/contradiction-train-eval-non-iid.md §2b).
        # v3 contradiction numbers are NOT comparable to v2 ones -- different eval set, report in
        # its own column.
        _V3_BUNDLE = ("/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
                      "_eval_bundle_eval500_v3")
        E5 = os.environ.get("EVAL500_ROOT")
        if not E5:
            E5 = next((p for p in _V2_BUNDLES if os.path.isdir(p)), _V2_BUNDLES[-1])
            print(f"[ladder] EVAL500_ROOT unset -> {E5} "
                  f"(ladder_version={args.ladder_version})", flush=True)
        # v3 is a SELF-CONTAINED bundle: contra and outlier are real directories holding the rebuilt
        # files, and nq/rerank/oolong are directory symlinks to v2_clean (so "identical to v2" is
        # true by construction rather than a claim to re-verify, and no multi-GB file is duplicated).
        # So v3 points the WHOLE root at the v3 bundle.
        #
        # It did NOT start that way. v3 originally redirected contradiction alone via EVAL500_CONTRA_ROOT,
        # on the reasoning that contra was the only task that differed. Outlier's xlong rungs were then
        # rebuilt too (the shipped ones had K pinned at 25 while n grew 32x), and a contra-only
        # redirect would have quietly served those K-frozen outlier rungs under a v3 tag.
        if args.ladder_version == "v3" and not os.environ.get("EVAL500_ROOT"):
            E5 = os.environ.get("EVAL500_CONTRA_ROOT", _V3_BUNDLE)  # legacy name, still honored
            print(f"[ladder] ladder_version=v3 -> {E5} (contra + outlier rebuilt; "
                  f"nq/rerank/oolong symlinked to v2_clean)", flush=True)
        # ---- FAST (shared-corpus) bundle -------------------------------------------------
        # Many queries share one corpus, so the shared part is prefilled once. NOT comparable to a
        # v2 number: rebuilding an eval set this way moves scores on its own (outlier +0.215/+0.261
        # and contradiction -0.102..-0.175 for the ORIGINAL prefix+tail build; the outlier rungs
        # here use a different construction that puts the answer back in the shared prefix). Report
        # fast numbers in their own column, never beside a v2 one.
        _FAST_BUNDLE = ("/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
                        "_eval_bundle_eval500_v2_fast")
        if args.ladder_version == "fast" and not os.environ.get("EVAL500_ROOT"):
            E5 = _FAST_BUNDLE
            print(f"[ladder] ladder_version=fast -> {E5}", flush=True)

        if args.ladder_version == "fast":
            # Filenames encode the construction, not the corpus size: contradiction is prefix+tail
            # with a 10% tail, outlier plants its candidates in the shared prefix and eliminates
            # them from the tail, and nq/rerank/oolong are query-multiplexed.
            _FAST_SUFFIX = {"contradiction": "tail10", "outlier": "planted",
                            "nq": "mux", "rerank": "mux", "oolong": "mux"}
            _FAST_BASE = [("8k", 8192), ("16k", 16384), ("32k", 32768)]
            _FAST_XL = [("64k", 65536), ("128k", 131072), ("256k", 262144),
                        ("512k", 524288), ("1M", 1048576)]
            _rungs = _FAST_BASE + (_FAST_XL if args.xlong else [])
            LADDERS = {}
            for _t, _sfx in _FAST_SUFFIX.items():
                # Every task runs the full 8k..1M ladder here. rerank reaches 32k in this bundle
                # even though its RELIABLE ladder stops at 16k: multiplexing pools several queries
                # into one corpus, so the corpus size is no longer capped by any single query's
                # CE-filtered hard-negative pool.
                LADDERS[_t] = [
                    (_lab, os.path.join(E5, _t, f"rung_{_tok}_{_sfx}.jsonl"))
                    for _lab, _tok in _rungs
                ]
            # ⚠ outlier's 8k rung leaks. The answer's topic is the one candidate the per-query tail
            # does not top up, and an 8k corpus holds too few topics for that absence to hide in:
            # guessing among the topics missing from the tail scores 0.203 there, against ~0.09
            # chance. It decays with length (0.090 at 16k, 0.048 at 32k, 0.0015 at 1M).
            # ⚠ oolong is eval_size=100 at every rung, a fifth of the 500 floor -- quote the size
            # and its error bar (~±0.046 at 0.7) inline next to any oolong number from this bundle.
        elif args.ladder_version in ("v2", "v3"):
            # v2: every rung of a task shares the SAME >=500 questions/answers; only the
            # distractor documents differ (built by build_v2_eval_ladders.py). ALL rungs live
            # under $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at the v2 bundle). oolong rungs are
            # freshly synthesized so they are DISJOINT from training (the v1 oolong eval overlapped
            # its own train split) while keeping the same question-type + corpus-type distribution.
            # v2 -> `both`-mode contradiction gold; v3 -> `realistic`, matching the training
            # generator. This one token is the whole v2/v3 difference.
            _CM = "realistic" if args.ladder_version == "v3" else "both"
            LADDERS = {
                "contradiction": [("2k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n100_k3.jsonl"),
                    ("8k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n190_k3.jsonl"),
                    ("16k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n385_k3.jsonl"),
                    ("32k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n765_k3.jsonl")],
                "nq": [("3k", f"{E5}/nq/nq_validation_k20_600.jsonl"),
                    ("8k", f"{E5}/nq/nq_validation_k50_600.jsonl"),
                    ("16k", f"{E5}/nq/nq_validation_k100_600.jsonl"),
                    ("32k", f"{E5}/nq/nq_validation_k200_600.jsonl")],
                "outlier": [("3k", f"{E5}/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
                    ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
                    ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
                    ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
                    # 64k transfer-test rung (n=440 docs), same scale-K builder/pool as 3k-32k.
                    ("64k", f"{E5}/outlier/outlier_wiki100w_n440_k3_eval_600.jsonl"),
                    # 128k transfer-test rung (n=880 docs, median 128,335 tok). Built by the same
                    # scale-K builder/pool as 3k-32k, so it extends THIS ladder rather than the
                    # xlong distractor-recycling one. Selected only when --ladder-rungs asks for
                    # 128k, so existing runs are unaffected; needs --max-length >= 146227 and
                    # --batch-size 1.
                    ("128k", f"{E5}/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl")],
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
            # qdmatch_nq: ordered (query_id, doc_id) pair matching, graded by _eval_qdmatch.
            # Added conditionally -- most eval bundles predate these rungs, and an unconditional
            # entry would print a MISSING warning on every unrelated ladder run. Rungs are the
            # 600-example HELD-OUT files built by
            # debug/outlier_lengthmix_scaling/build_qdmatch_pools.py (eval units come from the p10
            # NQ *validation* split, disjoint from the train pool -- the shipped CTC-suite
            # qdmatch_nq ladder drew train and eval from one shared unit pool). Length knob is
            # M=N (queries == docs, k_relevant=3): M=N=9 -> median 1825 tokens, 42 -> 7571,
            # 86 -> 15269, 172 -> 30425, all MEASURED with Qwen3.5-0.8B-Base
            # ([[ctc-rung-labels-not-tokens]]).
            # The 64k rung is ASYMMETRIC (M=172, N=368 -> median 61340). It has to be: one example
            # consumes M+N-3 = 537 DISTINCT source units, and the original p10 NQ validation split
            # held only 600 rows, so a symmetric M=N=344 (need 685) could not be drawn at all.
            # That 600-row ceiling also meant one example consumed ~90% of the pool, so all 600
            # eval examples shared nearly one corpus. The shipped rung_65536.jsonl is now REDRAWN
            # from the 3,114-row DEEP validation pool at the same M=172/N=368 shape (537/3114 =
            # 17% of the pool per example), which leaves the trained q64k arms byte-identical
            # while de-correlating the eval. The superseded file is kept beside it as
            # rung_65536.corr600.jsonl and is not wired up here.
            # Each rung is gated on its own file, so 16k/32k/64k appear only once built.
            if os.path.isdir(f"{E5}/qdmatch_nq"):
                LADDERS["qdmatch_nq"] = [
                    (_lab, f"{E5}/qdmatch_nq/rung_{_tok}.jsonl")
                    for _lab, _tok in (("3k", 2048), ("8k", 8192), ("16k", 16384),
                                       ("32k", 32768), ("64k", 65536))
                    if os.path.exists(f"{E5}/qdmatch_nq/rung_{_tok}.jsonl")
                ]
            # xabsence EXACT rungs (2k/4k/8k/16k/32k), file-gated like qdmatch_nq above.
            # ⚠ EXACT ONLY. Two different tasks ship under the name "xabsence": EXACT (the twin
            # is the IDENTICAL string in corpus B; the CTC suite row) and PARAPHRASE (an LLM
            # rewrite; near its chance floor under full attention). They differ in pool, shape and
            # difficulty, so a rung directory named plain `xabsence/` is NOT wired up here -- the
            # subdir must say `xabsence_exact`. Discriminator, if a bundle's provenance is unclear:
            # len(A_texts & B_texts) == num_pairs for EXACT, 0 for PARAPHRASE
            # ([[xabsence-exact-vs-paraphrase-split]]). P (pairs/example) -> measured median tokens:
            # 2 -> 2.4k, 5 -> 4.5k, 11 -> 8.7k, 23 -> 17k, 46 -> 33k, all on PubMed abstracts at
            # ~346 tok/item -- P values calibrated on any other corpus do NOT transfer.
            if os.path.isdir(f"{E5}/xabsence_exact"):
                LADDERS["xabsence"] = [
                    (_lab, f"{E5}/xabsence_exact/rung_{_tok}.jsonl")
                    for _lab, _tok in (("2k", 2048), ("4k", 4096), ("8k", 8192),
                                       ("16k", 16384), ("32k", 32768))
                    if os.path.exists(f"{E5}/xabsence_exact/rung_{_tok}.jsonl")
                ]
            # nq LENGTH-MIX rungs (2k/8k/16k). The shipped nq ladder above is named by doc count
            # (nq_validation_k{20,50,100,200}_600.jsonl); the length-mix bundle instead ships
            # rung_{2048,8192,16384}.jsonl, 600 held-out examples each, built by
            # debug/outlier_lengthmix_scaling/build_nq_pools.py from the p10 NQ *validation*
            # split (0 shared queries with the nq2k_*/nq8k_*/nq16k_*/nq32k_* training arms;
            # k=12 -> median 1907 tokens, k=48 -> median 7509, k=100 -> median 15455, all MEASURED
            # with Qwen3.5-0.8B-Base at --query-position after).
            # The 32k and 64k rungs come from the DEEP p10 source (nq_*_k~450_deep*.jsonl,
            # 440-460 docs/row, 9,998 train / 3,114 validation): k=200 -> measured median 31.4k
            # tokens and k=400 -> 62.8k, both with clamped_frac 0.0. They REPLACE the earlier
            # k25-202 build, whose "32k" rung was really ~18k -- nq length is set by k =
            # docs/example and `shrink` CLAMPS a row holding fewer than k docs, so with rows of
            # 25-202 documents (mean ~114) k=200 clamped 98% of them and k=220/260 produced a
            # byte-identical distribution. That superseded file is kept beside this one as
            # rung_32768.clamped18k.jsonl and is NOT wired up here. Still quote measured medians,
            # never labels ([[ctc-rung-labels-not-tokens]]).
            # Conditional and file-gated, so it overrides the shipped entry ONLY when pointed at
            # that bundle, and each rung is gated on its own file; every other EVAL500_ROOT is
            # untouched.
            if os.path.exists(f"{E5}/nq/rung_2048.jsonl"):
                LADDERS["nq"] = [
                    (_lab, f"{E5}/nq/rung_{_tok}.jsonl")
                    for _lab, _tok in (("2k", 2048), ("8k", 8192), ("16k", 16384),
                                       ("32k", 32768), ("64k", 65536))
                    if os.path.exists(f"{E5}/nq/rung_{_tok}.jsonl")
                ]
        else:
            # v1 = the original independently-generated per-rung eval files. DISABLED 2026-07-29:
            # each rung drew its OWN questions, so every rung-to-rung delta carried eval-set
            # resampling noise on top of the length effect it was supposed to isolate. v2 fixes
            # the question set across rungs and varies only the distractor documents.
            raise NotImplementedError(
                f"--ladder-version {args.ladder_version!r} is no longer supported: v2 is the only "
                "ladder. v1 resampled the questions per rung, so rung-to-rung deltas mixed "
                "eval-set noise into the length effect. Build what you need as v2 "
                "(build_v2_eval_ladders.py for 2k-32k, build_xlong_rungs.py for 64k-2M) and "
                "point EVAL500_ROOT at a v2 bundle."
            )
        # ---- oolong SHORT rungs (2k/4k): extend the ladder DOWNWARD ----
        # The v2 oolong ladder starts at 8k because no shorter synthesized rungs existed. Added
        # conditionally so an EVAL500_ROOT without them still works. Prepended, not appended, so
        # the rungs stay in ascending length order.
        if args.ladder_version in ("v2", "v3") and "oolong" in LADDERS:
            _short = []
            for _lab, _ctx in (("2k", 2048), ("4k", 4096)):
                _p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{_ctx}_spliteval.jsonl")
                if os.path.exists(_p):
                    _short.append((_lab, _p))
            if _short:
                LADDERS["oolong"] = _short + LADDERS["oolong"]

        # ---- OPT-IN ultra-long rungs (64k..2M), OFF by default ----
        # Resolved by size-labelled glob so the calibrated doc-count in the filename can drift
        # (rebuild with a different --count/tokenizer) without editing this file.
        # v3 MUST be here too. While this read `== "v2"`, `--ladder-version v3 --xlong` silently
        # produced a BASE-ONLY ladder -- no error, no missing-file warning (there are no paths to
        # check), just a 2k-32k result written under an xlong tag. A partial ladder that looks
        # complete is worse than a crash.
        if args.xlong and args.ladder_version in ("v2", "v3"):
            import glob as _glob
            # rerank and oolong were originally excluded: no CE-graded rerank pool above k100
            # existed, and oolong is not a doc pool. Both now have xlong rungs (built 2026-07-27),
            # so they are wired here. oolong does NOT use the `_xlong_` convention -- it is a packed
            # item stream labelled by its token budget (ctx{N}_spliteval), so it gets its own map.
            _XL = {
                # _CM, not a hardcoded "both": v3's contra files are named ..._realistic_..., so a
                # literal "both" here would match nothing and drop every contra xlong rung while the
                # base rungs still resolved -- a silently short ladder.
                "contradiction": ("contra",  f"contradiction_eval_pubmed_{_CM}_n*_k3_xlong_{{s}}.jsonl"),
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
            _XL_OOLONG = {"64k": 65536, "128k": 131072, "256k": 262144,
                          "512k": 524288, "1M": 1048576, "2M": 2097152}
            for _t, (_sub, _pat) in _XL.items():
                if _t not in LADDERS:
                    continue
                for _s in ("64k", "128k", "256k", "512k", "1M", "2M"):
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
            "qdmatch_nq": ("qdmatch", _eval_qdmatch, "f1", 200),
            "xabsence": ("xabsence", _eval_absence, "f1", 200),
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
                ex = _load(path, task=loadtask)
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
            # BEHAVIOUR CHANGE (2026-08-11): these OOD probes used to call load_unified_examples
            # directly with use_alpaca=True and NO chat template, while every other task in the same
            # run went through _load and got the chat template. One model, one run, two prompt
            # conventions -- the probes were being fed a format the SFT model never saw. Routing them
            # through _load fixes that and picks up --query-position. OOD numbers from this driver are
            # therefore NOT comparable to pre-2026-08-11 OOD numbers. To restore the old rendering,
            # call load_unified_examples(gpath, args.max_test_samples, task="retrieval",
            # query_position=args.query_position, use_alpaca=True) here instead.
            ex = _load(gpath, task="retrieval")
            resp = generate([e["prompt"] for e in ex], gmax)
            res, det = _eval_retrieval(ex, resp)
            _record_gens(f"gen_{gname}", "probe", ex, resp, det)
            summary[f"gen_{gname}"] = res
            print(f"[gen:{gname}] f1={res.get('f1', 0):.3f} (n={len(ex)})", flush=True)

    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Provenance, so a number can say which eval SET produced it without anyone remembering.
        # The fast (shared-corpus) bundle is a DIFFERENT measurement from v2, and this file used to
        # carry nothing but `<task>_<rung>: float` -- so `eval_version` was whatever the person
        # ingesting it typed, and the standing instructions say to type "v2". A fast number then
        # lands indistinguishable from a reliable one and quietly contaminates the comparison it
        # exists to support.
        #
        # Strings only, and read by results-hub's ingester as metadata rather than metrics: a dict
        # of numbers here would be picked up as another task's scores.
        summary["_meta"] = {
            "ladder_version": args.ladder_version,
            "eval_bundle": E5 if args.ladder else "",
            "query_position": args.query_position,
        }
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
