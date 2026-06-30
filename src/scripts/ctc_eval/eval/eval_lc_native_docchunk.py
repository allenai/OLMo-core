"""
NATIVE olmo_core eval for **document-chunked** models on the single-task ladder tasks -- both the
dense (``DocumentChunkedAttention``) and landmark (``DocumentLandmarkAttention``) variants. Custom
attention can't use HF/vLLM, so we load the olmo-core distcp directly and decode greedily.

Supports ``--task`` ``oolong | contradiction | retrieval | rerank | outlier`` (aliases ``nq`` ->
``retrieval``, ``contra`` -> ``contradiction``). The box-marker prefill is built with the SAME
``segment_prompt_to_chunks`` path the training converter uses, dispatched per task by ``TASK_CFG``
(segmentation task + ``chunk_by`` + decode budget/stop rule + scorer + default CoT mode); the per-task
scorers are reused verbatim from ``scripts/eval/evaluate.py``. The single-task ladder (v2) shards are
tokenized with ``--cot-mode none``, so eval those checkpoints with ``--cot-mode none`` (the default for
every task except legacy oolong). NOTE: ``--task rerank`` requires CE-graded eval data (``ce_scores``
in each record, e.g. ``ce_gen/msmarco_trainhn_eval_*.jsonl``); the old binary rerank format is rejected
by ``_eval_rerank``.

    # per-task (4B docchunk dense/landmark checkpoint at <step_dir>):
    PYTHONPATH=<olmo-core>/src:<corpus-reasoning> torchrun --nproc_per_node=8 \
      scripts/eval/eval_lc_native_docchunk.py --variant dense --task contradiction --cot-mode none \
      --model-path <step_dir> --data data/contradiction_eval_pubmed_realistic_n100_k3.jsonl \
      --out outputs/eval_results/<name>.json

The prefill is built with the SAME path the training converter uses
(``olmo_core.data.document_chunk_landmark.segment_prompt_to_chunks`` + the matching emitter), so each
OOLONG item line is wrapped in ``<|box_start|>`` / ``<|box_end|>`` special tokens and (for the
landmark variant) packed into landmark windows -- byte-identical to training. The model reconstructs
``chunk_ids`` from those boundary tokens each step (``enable_document_chunk_attention``).

Decoding is a bs=1 greedy loop with a KV cache for all variants. Prefill applies the chunked mask and
caches K,V; decode is incremental (dense/full: plain causal; landmark: plain per-block landmark decode,
since a generated FREE token makes the chunk mask a no-op) -- O(gen*n^2) eager re-feeding becomes
O(n^2 + gen*n) with token-identical output. For the landmark variant a landmark token is fed every
``--mem-freq`` generated content tokens so the periodic ``is_mem`` pattern stays valid in the generated
tail. Runs data-parallel across ranks (torchrun): each rank decodes a shard, rank 0 gathers + scores
with ``_eval_oolong``.

    PYTHONPATH=<olmo-core>/src:<corpus-reasoning> python scripts/eval/eval_lc_native_docchunk.py \
      --variant dense --model-path <step_dir> --out outputs/eval_results/<name>.json \
      --oolong-data data/oolong_test_synth_ctx2048_spliteval.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch

# Reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).
EOS_TOKEN_ID = 151643
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>
PAD_TOKEN_ID = 151863

# Per-task eval config. Mirrors scripts/eval/evaluate.py: the segmentation task name + chunk_by used
# to build the box-marker prefill (MUST match the training converter -- see
# src/scripts/data/convert_docchunk_singletask_v2_local.sbatch), the decode budget + stop rule, the
# scorer, and the default prompt CoT mode. ``stop`` rules:
#   "oolong"  -- stop at a newline once "answer:" has been emitted (oolong's templated answer line).
#   "newline" -- stop at the first generated newline (single-line answers: retrieval ids, rerank list).
#   "eos"     -- stop only at EOS (multi-line answers: contradiction pairs, outlier reasoning+set).
# ``cot``: the single-task ladder (v2) shards are tokenized with --cot-mode none, so eval prefill MUST
# use cot=none for those checkpoints (oolong keeps "plan" only for the legacy doc-OOLONG run).
TASK_CFG = {
    "oolong":        dict(chunk_by="line",     max_new=256, stop="oolong",  scorer="oolong",        cot="plan"),
    "contradiction": dict(chunk_by="document", max_new=200, stop="eos",     scorer="contradiction", cot="none"),
    "retrieval":     dict(chunk_by="document", max_new=64,  stop="newline", scorer="retrieval",     cot="none"),
    "rerank":        dict(chunk_by="document", max_new=512, stop="newline", scorer="rerank",        cot="none"),
    "outlier":       dict(chunk_by="document", max_new=256, stop="eos",     scorer="outlier",       cot="none"),
}
# Convenience aliases (run-name / launcher shorthands) -> canonical segmentation task.
TASK_ALIASES = {"nq": "retrieval", "contra": "contradiction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["dense", "landmark", "full"])
    ap.add_argument("--model-path", required=True, help="step dir: config.json + model_and_optim/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument(
        "--task", default="oolong",
        help="oolong | contradiction | retrieval | rerank | outlier "
        "(aliases: nq->retrieval, contra->contradiction).",
    )
    # --data is the general eval JSONL; --oolong-data kept as a back-compat alias.
    ap.add_argument("--data", default=None, help="eval JSONL (unified format). Overrides --oolong-data.")
    ap.add_argument("--oolong-data", default="data/oolong_test_synth_ctx2048_spliteval.jsonl")
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the per-task default decode budget.")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--mem-freq", type=int, default=63)
    ap.add_argument("--cot-mode", default=None, help="override the per-task default prompt CoT mode.")
    ap.add_argument(
        "--landmark-top-k-blocks",
        type=int,
        default=None,
        help="landmark variant: keep only the top-k landmark BLOCKS per query at inference (exact if unset).",
    )
    ap.add_argument(
        "--landmark-top-k-fraction",
        type=float,
        default=None,
        help="landmark variant: top-k = ceil(fraction * num_prompt_blocks), set per example. "
        "Overridden by --landmark-top-k-blocks.",
    )
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    seg_task = TASK_ALIASES.get(args.task, args.task)
    if seg_task not in TASK_CFG:
        raise SystemExit(f"--task {args.task!r} -> {seg_task!r} not in {sorted(TASK_CFG)}")
    cfg = TASK_CFG[seg_task]
    chunk_by = cfg["chunk_by"]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg["max_new"]
    cot_mode = args.cot_mode if args.cot_mode is not None else cfg["cot"]
    stop_rule = cfg["stop"]
    eval_data = args.data or args.oolong_data

    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import (
        DOC_END_ID as _DE,
    )
    from olmo_core.data.document_chunk_landmark import (
        DOC_START_ID as _DS,
    )
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        segment_prompt_to_chunks,
    )
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from ctc_eval.eval.evaluate import (
        _eval_contradiction,
        _eval_oolong,
        _eval_outlier,
        _eval_rerank,
        _eval_retrieval,
        load_unified_examples,
    )

    SCORERS = {
        "oolong": _eval_oolong,
        "contradiction": _eval_contradiction,
        "retrieval": _eval_retrieval,
        "rerank": _eval_rerank,
        "outlier": _eval_outlier,
    }
    scorer = SCORERS[cfg["scorer"]]

    assert (_DS, _DE) == (DOC_START_ID, DOC_END_ID)

    # ---- data-parallel across ranks (torchrun) ----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    NEWLINE_ID = tok("\n", add_special_tokens=False).input_ids[-1]
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    t0 = time.time()
    # GenerationConfig requires pad != eos; Qwen3 has no pad, and we decode bs=1 (pad is unused), so
    # any distinct reserved id is fine (151645 = <|im_end|>).
    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, EOS_TOKEN_ID) else 151645
    # All variants now support a KV cache. Dense / full (DocumentChunkedAttention) prefill with the
    # chunked mask + cache K,V; decode is plain causal over the cache (generated tokens are FREE). The
    # landmark variant (DocumentLandmarkAttention) prefills with the chunked grouped-softmax mask +
    # caches K,V; decode is the plain incremental landmark decode (FREE query, chunk mask is a no-op).
    # Both turn eval from O(gen*n^2) eager re-feeding into O(n^2 + gen*n), identical tokens.
    use_cache = True
    gen_cfg = GenerationConfig(
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=pad_id,
        max_length=args.max_length,
        use_cache=use_cache,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=args.model_path, device=device)
    # Belt-and-suspenders: ensure runtime chunk_id reconstruction is on (config.json should already
    # set it, but we control pad_id here). The full-attention baseline has NO chunked mask.
    if args.variant != "full":
        gm.model.enable_document_chunk_attention(
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            mode="chunked",
            pad_id=PAD_TOKEN_ID if args.variant == "landmark" else None,
        )
    print(
        f"[docchunk-{args.variant}] task={seg_task} chunk_by={chunk_by} cot={cot_mode} "
        f"max_new={max_new_tokens} stop={stop_rule} built from {args.model_path} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    cap = args.max_length - max_new_tokens

    def should_stop(nxt_id, new_content):
        """Per-task early stop (called after appending ``nxt_id`` to ``new_content``)."""
        if stop_rule == "eos":
            return False  # multi-line answer: only the EOS check in the loop stops decoding
        if nxt_id != NEWLINE_ID:
            return False
        if stop_rule == "newline":
            return True  # single-line answer: stop at the first generated newline
        # "oolong": stop at a newline only once the templated "answer:" line has been emitted
        return "answer:" in tok.decode(new_content, skip_special_tokens=True).lower()

    def build_prefill(raw_example):
        segs, ids, _ = segment_prompt_to_chunks(
            tok, raw_example, seg_task, query_position="both", cot_mode=cot_mode,
            chunk_by=chunk_by, item_regex=r"\|\|", include_answer=False,
            doc_start_id=DOC_START_ID, doc_end_id=DOC_END_ID,
        )
        if args.variant in ("dense", "full"):
            out, _ = emit_document_chunk_dense(segs)  # box markers present; full attention ignores them
        else:
            out, _ = emit_document_chunk_landmark(
                segs, mem_freq=args.mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
            )
        return out

    block_size = args.mem_freq + 1  # landmark window (64); the eager landmark forward needs T % 64 == 0

    @torch.no_grad()
    def generate_one(prefill):
        gm.prepare_inference_cache(1, args.max_length)  # (re)set the cache cursor to 0 per example
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        if args.variant in ("dense", "full"):
            # Dense / full: prefill once (chunked mask applied + K,V cached), then single-token greedy
            # decode over the cache (plain causal since new tokens are FREE). Same "Answer:" early-stop.
            logits = gm.model(
                torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
            )
            nxt = int(logits[0, -1].argmax().item())
            new_content = []
            for _ in range(max_new_tokens):
                if nxt == EOS_TOKEN_ID:
                    break
                new_content.append(nxt)
                if should_stop(nxt, new_content):
                    break
                logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
                nxt = int(logits[0, -1].argmax().item())
            text = tok.decode(new_content, skip_special_tokens=True)
            return text.split("</think>", 1)[1] if "</think>" in text else text

        # Landmark: KV-cached decode. The prefill (block-aligned, landmark at every block end) is run
        # once with the chunked grouped-softmax mask + K,V cached; then each generated token is fed
        # incrementally and decoded as a plain landmark query (FREE -> chunk mask is a no-op). To stay
        # token-identical to the old eager re-feed loop we replicate its token stream EXACTLY: insert a
        # real landmark token after every ``mem_freq`` generated content tokens (advancing the cache),
        # so the periodic ``is_mem`` structure -- and thus every per-block landmark decode -- matches.
        logits = gm.model(
            torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
        )
        nxt = int(logits[0, -1].argmax().item())
        new_content = []
        since_landmark = 0
        for _ in range(max_new_tokens):
            if nxt == EOS_TOKEN_ID:
                break
            new_content.append(nxt)
            logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
            since_landmark += 1
            if since_landmark == args.mem_freq:
                # Feed a real landmark to keep the tail block-aligned; its logits predict the next token.
                logits = gm.model(
                    torch.tensor([[LANDMARK_TOKEN_ID]], device=device), logits_to_keep=1
                )
                since_landmark = 0
            if should_stop(nxt, new_content):
                break
            nxt = int(logits[0, -1].argmax().item())
        text = tok.decode(new_content, skip_special_tokens=True)
        return text.split("</think>", 1)[1] if "</think>" in text else text

    examples = load_unified_examples(
        eval_data, args.max_test_samples, task=seg_task,
        query_position="both", use_alpaca=True,
    )
    import math

    block_size = args.mem_freq + 1
    if args.variant == "landmark" and args.landmark_top_k_blocks is not None:
        n_set = gm.model.set_landmark_eval_top_k(args.landmark_top_k_blocks)
        print(f"[topk] fixed top_k={args.landmark_top_k_blocks} on {n_set} landmark layers", flush=True)

    my_gidx = list(range(rank, len(examples), world))
    local = []
    skipped = 0
    for gi in my_gidx:
        raw = examples[gi].get("ex", examples[gi])
        prefill = build_prefill(raw)
        if len(prefill) > cap:
            skipped += 1
            local.append((gi, ""))  # too long for this max_length -> empty (scored wrong)
            continue
        # Per-example top-k from a fraction of this prompt's landmark blocks (landmark variant only).
        if args.variant == "landmark" and args.landmark_top_k_fraction is not None:
            n_blocks = max(1, len(prefill) // block_size)
            gm.model.set_landmark_eval_top_k(
                max(1, math.ceil(args.landmark_top_k_fraction * n_blocks))
            )
        local.append((gi, generate_one(prefill)))

    full = [None] * len(examples)
    if world > 1:
        parts = [None] * world
        torch.distributed.all_gather_object(parts, local)
        for part in parts:
            for gi, resp in part:
                full[gi] = resp
    else:
        for gi, resp in local:
            full[gi] = resp

    if is_main:
        res, _ = scorer(examples, full)
        summary = {
            "model_path": args.model_path,
            "variant": args.variant,
            "task": seg_task,
            "data": eval_data,
            "n": len(examples),
            "skipped_too_long": skipped,
            "landmark_top_k_blocks": args.landmark_top_k_blocks,
            "landmark_top_k_fraction": args.landmark_top_k_fraction,
            # keep the legacy "oolong" key for the existing dashboard when task==oolong.
            ("oolong" if seg_task == "oolong" else "metrics"): res,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        # Print whatever scalar metrics the scorer returned (per-task primary keys differ).
        scalars = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        print(f"[{seg_task}] n={len(examples)} skipped={skipped} " + " ".join(
            f"{k}={v:.3f}" for k, v in scalars.items()), flush=True)
    if world > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
