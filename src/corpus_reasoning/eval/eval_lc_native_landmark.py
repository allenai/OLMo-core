"""
NATIVE olmo_core eval for **fast-landmark** models (no HF export, no vLLM).

Landmark attention cannot run under the HF/vLLM path (eval_lc_fast.py): the grouped-softmax memory
mechanism + landmark-token insertion only exist in the olmo_core model. So we load the olmo-core
distcp directly via TransformerGenerationModule, which auto-detects the landmark layers and inserts a
landmark token every ``mem_freq`` content tokens of the prompt during prefill (the eval harness only
ever sees plain content tokens in/out). This is the landmark analogue of eval_lc_native.py, extended
to the full CPT-mix task suite (RULER + contradiction + nq + oolong + rerank + outlier + gen probes)
so its output JSON is directly comparable to eval_lc_fast.py's.

Landmark generation constraints (enforced by olmo_core.generate):
  * ``GenerationConfig.landmark_mem_id`` MUST be set (the landmark token id, 151860 for Qwen3).
  * ``use_cache=True`` required.
  * NO left-padding / attention mask (blocks are tied to absolute position). We batch by grouping
    prompts of EXACT equal token length (no padding needed -> positions intact), bounded by
    --max-batch-tokens. This keeps correctness while saturating the GPU (~10x faster than bs=1).

    python scripts/eval/eval_lc_native_landmark.py \
      --model-path <step_dir_with_config.json_and_model_and_optim> \
      --out outputs/eval_results/<name>_lmnative.json [--tokenizer Qwen/Qwen3-4B]

Run on a GPU node, env corpus-reasoning-olmo (olmo_core + transformers), PYTHONPATH=corpus-reasoning.
"""
import argparse
import json
import os
import time

import torch
from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="step dir: has config.json + model_and_optim/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--max-batch-tokens", type=int, default=32768,
                    help="batch budget (B*L). Prompts are grouped by EXACT token length so a batch "
                         "needs no padding (landmark blocks are tied to absolute position); B per "
                         "bucket = max(1, max_batch_tokens // length).")
    ap.add_argument("--mem-id", type=int, default=LANDMARK_TOKEN_ID)
    ap.add_argument("--landmark-top-k-fraction", type=float, default=0.1)
    ap.add_argument("--landmark-decode-mode", default="extend_last_block")
    ap.add_argument("--no-fast-batch", action="store_true",
                    help="disable the right-padded CROSS-LENGTH batched landmark decode "
                         "(gm.generate_landmark_batch) and fall back to the slow exact-length "
                         "bs~=1 bucketing. The fast path is parity-validated (float32 bit-exact vs "
                         "bs=1; bf16 within reassociation noise) and is the default.")
    ap.add_argument("--ruler-lengths", default="L1024,L2048")
    ap.add_argument("--ruler-subtasks",
                    default="niah_single,niah_multikey,niah_multivalue,niah_multiquery,vt,cwe,fwe")
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--contra-max-new-tokens", type=int, default=200,
                    help="generation budget for contradiction; enumerate-CoT answers on large-N "
                         "(e.g. n250) need ~2200 to reach the final 'Contradicting pairs:' line.")
    ap.add_argument("--nq-data", default="data/nq_validation_k20_hn19_500_aligned.jsonl")
    ap.add_argument("--oolong-data", default="data/oolong_test_synth_ctx2048_spliteval.jsonl")
    ap.add_argument("--rerank-data", default="data/msmarco_dev_rerank_k20_1000.jsonl")
    ap.add_argument("--outlier-data", default="data/outlier_wiki100w_n20_k3_eval_100.jsonl")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--rerank-root", default="/scratch/users/prasann/cpt_data/rerank_ladder_src")
    ap.add_argument("--root", default=None,
                    help="chdir here before resolving relative data paths (on-cluster: pass the "
                         "eval dataset mountpoint so data/... resolves).")
    ap.add_argument("--skip-ruler", action="store_true")
    ap.add_argument("--skip-extra", action="store_true", help="skip oolong/rerank/outlier")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip held-out retrieval generalization probes")
    ap.add_argument("--ladder", action="store_true",
                    help="evaluate each task across its LENGTH LADDER (2k..64k); reports <task>_<rung>. "
                         "RULER ladders via --ruler-lengths.")
    ap.add_argument("--ladder-tasks", default=None,
                    help="comma list restricting the --ladder to a subset of tasks "
                         "(contradiction,nq,oolong,rerank,outlier). Enables splitting a slow ladder "
                         "into per-task jobs; merge the resulting JSONs after.")
    ap.add_argument("--ladder-rungs", default=None,
                    help="comma list restricting the --ladder to a subset of rungs (e.g. 16k,32k).")
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

    # ---- data-parallel across N GPUs (torchrun): each rank loads a full model copy on its GPU and
    # evaluates a SHARD of every example list; rank 0 gathers, scores, and writes. world_size=1 ->
    # identical to single-GPU behavior. Launch with: torchrun --nproc_per_node=N <this>.py ...
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
        sys.stdout = open(os.devnull, "w")  # suppress duplicate per-rank console output

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    t0 = time.time()
    gen_cfg = GenerationConfig(
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        max_length=args.max_length, use_cache=True,
        landmark_mem_id=args.mem_id, landmark_pad_id=tok.pad_token_id,
        landmark_decode_mode=args.landmark_decode_mode,
        landmark_top_k_fraction=args.landmark_top_k_fraction,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False,
    ).build(checkpoint_dir=args.model_path, device=device)
    print(f"[lmnative] built generation module from {args.model_path} in {time.time()-t0:.1f}s",
          flush=True)

    def strip_think(s):
        return s.split("</think>", 1)[1] if "</think>" in s else s

    def _decode_row(row):
        clean = []
        for t in row:
            if t in (tok.eos_token_id, tok.pad_token_id):
                break
            clean.append(t)
        return strip_think(tok.decode(clean, skip_special_tokens=True))

    @torch.no_grad()
    def generate(prompts, max_new_tokens, stop_strings=None):
        # Data-parallel: this rank handles global indices [rank, rank+world, ...]; then all_gather to
        # reconstruct the full ordered list (every rank ends up with it; only rank 0 scores+writes).
        # Within the shard: landmark blocks are tied to ABSOLUTE position so left-padding is forbidden;
        # prompts of the *same* token length need no padding -> group by exact length and batch each
        # group (attention_mask=None) -> GPU-saturating batches. DP across N GPUs multiplies this.
        from collections import defaultdict
        my_gidx = list(range(rank, len(prompts), world))
        lp = [prompts[i] for i in my_gidx]
        max_in = args.max_length - max_new_tokens
        # truncation=False is DELIBERATE: cutting to max_in drops the prompt TAIL, where the
        # question lives, and scores the model on an example it never saw. Raise instead.
        toks = [tok(p, truncation=False, add_special_tokens=False)["input_ids"] for p in lp]
        _over = [(my_gidx[i], len(t)) for i, t in enumerate(toks) if len(t) > max_in]
        if _over:
            _worst = max(n for _, n in _over)
            raise SystemExit(
                f"[maxlen] {len(_over)}/{len(toks)} prompts exceed the {max_in}-token prompt cap "
                f"(--max-length {args.max_length} minus max_new_tokens {max_new_tokens}); longest "
                f"is {_worst} tokens (example indices {[g for g, _ in _over][:8]}). Re-run with "
                f"--max-length >= {_worst + max_new_tokens}."
            )
        lout = [None] * len(lp)

        use_fast = (not args.no_fast_batch) and gm.supports_landmark_ragged_batch()
        if use_fast:
            # RIGHT-PADDED CROSS-LENGTH batching: landmark content keeps absolute positions 0..L_i and
            # the pad tail is masked, so prompts of DIFFERENT lengths batch together (vs the legacy
            # exact-length bs~=1 bucketing). Greedily pack length-sorted prompts into batches bounded
            # by max_batch_tokens = B * (padded length) to cap padding waste + memory.
            order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
            j = 0
            while j < len(order):
                Lmax = len(toks[order[j]])  # smallest first; grows as we add
                bs = max(1, args.max_batch_tokens // max(Lmax, 1))
                # extend the batch while the running padded budget stays under max_batch_tokens
                k = j
                batch = []
                while k < len(order) and len(batch) < bs:
                    Lmax = max(Lmax, len(toks[order[k]]))
                    if (len(batch) + 1) * Lmax > args.max_batch_tokens and batch:
                        break
                    batch.append(order[k]); k += 1
                comps = gm.generate_landmark_batch(
                    [toks[i] for i in batch], max_new_tokens=max_new_tokens,
                    decode_mode=args.landmark_decode_mode,
                    top_k_fraction=args.landmark_top_k_fraction,
                    stop_strings=stop_strings, stop_string_tokenizer=tok if stop_strings else None,
                    stop_string_check_interval=16,
                )
                for i, c in zip(batch, comps):
                    lout[i] = _decode_row(c)
                j = k
        else:
            buckets = defaultdict(list)
            for i, ids in enumerate(toks):
                buckets[len(ids)].append(i)
            for L, idxs in buckets.items():
                bs = max(1, args.max_batch_tokens // max(L, 1))
                for j in range(0, len(idxs), bs):
                    chunk = idxs[j:j + bs]
                    batch_ids = torch.tensor([toks[i] for i in chunk], device=device)  # B x L, equal len
                    gen_kw = {}
                    if stop_strings:
                        gen_kw = dict(stop_strings=stop_strings, stop_string_check_interval=16,
                                      stop_string_tokenizer=tok)
                    cont, _, _ = gm.generate_batch(input_ids=batch_ids, attention_mask=None,
                                                   completions_only=True, log_timing=False,
                                                   max_new_tokens=max_new_tokens, **gen_kw)
                    for k, i in enumerate(chunk):
                        lout[i] = _decode_row(cont[k].tolist())
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

    # `eval_size`: reported-key -> number of EVAL EXAMPLES scored. NOT called "n" -- in this
    # project `n` means CORPUS size (docs per example), never eval-set size.
    summary = {"model_path": args.model_path, "ruler": {}, "contradiction": {},
               "nq": {}, "eval_size": {}}

    # Write incrementally: bs=1 landmark generation is slow, so persist partial results after every
    # section -- a wall-clock timeout then still yields whatever finished (the JSON is rewritten each
    # time). recompute the RULER average from whatever subtasks are present so far.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def flush():
        rs = [v["recall"] for v in summary["ruler"].values() if isinstance(v, dict) and "recall" in v]
        ac = [v["all_correct"] for v in summary["ruler"].values()
              if isinstance(v, dict) and "all_correct" in v]
        summary["ruler_avg_recall"] = (sum(rs) / len(rs)) if rs else None
        summary["ruler_avg_all_correct"] = (sum(ac) / len(ac)) if ac else None
        if is_main:  # only rank 0 writes the result file
            json.dump(summary, open(args.out, "w"), indent=2)

    if not args.skip_ruler:
        recalls, all_correct = [], []
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
                all_correct.append(res["all_correct"])
                print(f"[ruler] {sub}_{L}: recall={res['recall']:.3f} (eval_size={len(ex)})", flush=True)
                flush()

    if not args.ladder:
        ex = load_unified_examples(args.contra_data, args.max_test_samples, task="contradiction",
                                   query_position="both", use_alpaca=True)
        res, _ = _eval_contradiction(ex, generate([e["prompt"] for e in ex], args.contra_max_new_tokens, stop_strings=["contradicting pairs:"]))
        summary["contradiction"] = res
        summary["eval_size"]["contradiction"] = len(ex)
        print(f"[contradiction] f1={res['f1']:.3f} (eval_size={len(ex)})", flush=True)
        flush()

    if not args.ladder and os.path.exists(args.nq_data):
        ex = load_unified_examples(args.nq_data, args.max_test_samples, task="retrieval",
                                   query_position="both", use_alpaca=True)
        res, _ = _eval_retrieval(ex, generate([e["prompt"] for e in ex], 64))
        summary["nq"] = res
        summary["eval_size"]["nq"] = len(ex)
        print(f"[nq] f1={res.get('f1', 0):.3f} (eval_size={len(ex)})", flush=True)
        flush()

    if not args.skip_extra and not args.ladder:
        extra = [
            ("oolong", args.oolong_data, _eval_oolong, "score", 200),
            ("rerank", args.rerank_data, _eval_rerank, None, 256),
            ("outlier", args.outlier_data, _eval_outlier, "f1", 200),
        ]
        for tname, tdata, fn, pkey, maxtok in extra:
            if not os.path.exists(tdata):
                print(f"[{tname}] MISSING {tdata}, skipping"); continue
            ex = load_unified_examples(tdata, args.max_test_samples, task=tname,
                                       query_position="both", use_alpaca=True)
            res, _ = fn(ex, generate([e["prompt"] for e in ex], maxtok))
            summary[tname] = res
            summary["eval_size"][tname] = len(ex)
            prim = res.get(pkey) if pkey else next(
                (v for k, v in res.items() if k.startswith("mrr")), None)
            summary[f"{tname}_primary"] = prim
            print(f"[{tname}] primary={prim} (eval_size={len(ex)})", flush=True)
            flush()

    if not args.skip_gen:
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
            print(f"[gen:{gname}] f1={res.get('f1', 0):.3f} (eval_size={len(ex)})", flush=True)
            flush()

    # ---- LENGTH-LADDER: each task at 2k..64k (reports <task>_<rung>) ----
    if args.ladder:
        RR = args.rerank_root
        # n>=500 eval at goal-critical rungs (8k/16k/32k) from cpt_data/eval500; 64k dropped.
        E5 = "/scratch/users/prasann/cpt_data/eval500"
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
            # (think-strip already applied in _decode_row(); no newline-stop).
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
                summary[f"{task}_{label}"] = prim
                print(f"[ladder:{task}@{label}] {pkey or 'mrr'}="
                      f"{prim if prim is None else round(prim,3)} (eval_size={len(ex)})", flush=True)
                flush()

    flush()
    print("\n==== SUMMARY (landmark native) ====")
    print(f"model: {args.model_path}")
    print(f"ruler_avg_recall: {summary.get('ruler_avg_recall')}")
    print(f"contradiction_f1: {summary['contradiction'].get('f1') if summary['contradiction'] else None}")
    print(f"nq_f1:            {summary['nq'].get('f1') if summary['nq'] else None}")
    for t in ("oolong", "rerank", "outlier"):
        print(f"{t}_primary:     {summary.get(t+'_primary')}")
    print(f"\n[lmnative] TOTAL {time.time()-t0:.1f}s\nWROTE {args.out}")
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
