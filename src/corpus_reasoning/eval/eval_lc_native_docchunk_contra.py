"""
NATIVE olmo_core eval for **document-chunked** models on CONTRADICTION -- both the dense
(``DocumentChunkedAttention``) and landmark (``DocumentLandmarkAttention``) variants. Custom attention
can't use HF/vLLM, so we load the olmo-core distcp directly and decode greedily.

This is the contradiction analogue of ``eval_lc_native_docchunk.py`` (OOLONG): the prefill is built
with the SAME path the training converter uses
(``olmo_core.data.document_chunk_landmark.segment_prompt_to_chunks`` + the matching emitter) with
``task="contradiction"`` and ``chunk_by="document"`` -- each ``documents[i]`` (one PubMed claim) is
wrapped in ``<|box_start|>`` / ``<|box_end|>`` special tokens and (for the landmark variant) first-fit
packed into landmark windows, byte-identical to training. So this directly tests whether
"one landmark ~ one claim" closes the landmark<->standard contradiction gap.

Decoding is a bs=1 greedy KV-cached loop for all variants: prefill applies the chunked mask + caches
K,V, then decode is incremental (landmark uses the plain per-block landmark decode, as a generated FREE
token makes the chunk mask a no-op) -- O(gen*n^2) -> O(n^2 + gen*n), token-identical. For the landmark
variant a landmark token is fed every ``--mem-freq`` generated content tokens. Runs data-parallel
across ranks (torchrun): each rank decodes a shard, rank 0 gathers + scores with ``_eval_contradiction``.

    PYTHONPATH=<olmo-core>/src:<corpus-reasoning> torchrun --nproc_per_node=8 \
      scripts/eval/eval_lc_native_docchunk_contra.py \
      --variant landmark --model-path <step_dir> --out outputs/eval_results/<name>.json \
      --contra-data data/contradiction_eval_pubmed_both_n100_k3.jsonl \
      --contra-max-new-tokens 2400 --max-length 12288
"""

import argparse
import json
import os
import sys
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

# Reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["dense", "landmark", "full"])
    ap.add_argument("--model-path", required=True, help="step dir: config.json + model_and_optim/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    # Boundary / special token ids. Defaults are the Qwen3 tokenizer values (unchanged behavior for
    # the dense Qwen3-0.6B/4B docchunk models). For the hybrid Qwen3.5 models the tokenizer differs
    # (vocab 248320): pass --doc-start-id 248049 --doc-end-id 248050 --eos-token-id 248044 to match
    # the Qwen3.5-tokenized shards (see Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py).
    ap.add_argument("--doc-start-id", type=int, default=DOC_START_ID, help="<|box_start|> id")
    ap.add_argument("--doc-end-id", type=int, default=DOC_END_ID, help="<|box_end|> id")
    ap.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID, help="document-separator EOS id")
    ap.add_argument("--pad-fallback-id", type=int, default=151645,
                    help="generation pad id when the tokenizer pad == eos (Qwen3 default 151645).")
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument(
        "--contra-max-new-tokens",
        type=int,
        default=2400,
        help="generation budget; enumerate-CoT contradiction answers (esp. n250) need ~2400 to reach "
        "the final 'Contradicting pairs:' line.",
    )
    ap.add_argument("--max-length", type=int, default=20480)
    ap.add_argument("--mem-freq", type=int, default=63)
    ap.add_argument("--free-pad-repeat", type=int, default=0,
                    help="MUST match the training shard: N repeats of FREE_PAD_SENTENCE appended "
                         "after the documents (extra FREE tokens). A mismatch silently changes the "
                         "prompt layout the model was trained on.")
    ap.add_argument("--repeat-doc-text", type=int, default=1,
                    help="MUST match the training shard: repeat each document's text N times "
                         "inside its chunk.")
    ap.add_argument("--cot-mode", default="enumerate")
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
    from corpus_reasoning.eval.evaluate import _eval_contradiction, load_unified_examples

    # Resolve boundary/eos ids from CLI (Qwen3 defaults preserve prior behavior; Qwen3.5 overrides).
    ds_id, de_id, eos_id = args.doc_start_id, args.doc_end_id, args.eos_token_id
    if (ds_id, de_id) == (DOC_START_ID, DOC_END_ID):
        # Only sanity-check the module defaults when using the built-in (Qwen3) boundary ids;
        # explicit overrides are passed straight through to segment_prompt_to_chunks below.
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
    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, eos_id) else args.pad_fallback_id
    # All variants now use a KV cache: dense/full prefill+cache then plain-causal decode; landmark
    # prefills with the chunked grouped-softmax mask + caches K,V, then decodes each token as a plain
    # incremental landmark query (FREE -> chunk mask is a no-op). O(gen*n^2) -> O(n^2 + gen*n).
    use_cache = True
    gen_cfg = GenerationConfig(
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        max_length=args.max_length,
        use_cache=use_cache,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=args.model_path, device=device)
    if args.variant != "full":
        gm.model.enable_document_chunk_attention(
            doc_start_id=ds_id,
            doc_end_id=de_id,
            eos_id=eos_id,
            mode="chunked",
            pad_id=PAD_TOKEN_ID if args.variant == "landmark" else None,
        )
    print(
        f"[docchunk-contra-{args.variant}] built from {args.model_path} in {time.time() - t0:.1f}s",
        flush=True,
    )

    max_new_tokens = args.contra_max_new_tokens
    cap = args.max_length - max_new_tokens

    def build_prefill(raw_example):
        segs, ids, _ = segment_prompt_to_chunks(
            tok, raw_example, "contradiction", query_position="both", cot_mode=args.cot_mode,
            chunk_by="document", item_regex=r"\|\|", include_answer=False,
            doc_start_id=ds_id, doc_end_id=de_id,
            # MUST equal the training shard's value, or the prefill layout differs from training.
            free_pad_repeat=args.free_pad_repeat,
            repeat_doc_text=args.repeat_doc_text,
        )
        if args.variant in ("dense", "full"):
            out, _ = emit_document_chunk_dense(segs)  # box markers present; full attention ignores them
        else:
            out, _ = emit_document_chunk_landmark(
                segs, mem_freq=args.mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
            )
        return out

    block_size = args.mem_freq + 1  # landmark window (64); the eager landmark forward needs T % 64 == 0

    def _answer_complete(content_ids):
        # The enumerate CoT walks every claim then ends with a final 'Contradicting pairs: [[...]]'
        # line. Once that anchor appears (and the line closes), the answer is complete -> early stop.
        txt = tok.decode(
            [t for t in content_ids if t != LANDMARK_TOKEN_ID], skip_special_tokens=True
        ).lower()
        if "contradicting pairs:" in txt:
            return True
        # no-cot: the answer is EXACTLY one JSON pair list ('[[a, b], ...]]') right after the
        # (empty) think block. Models trained on docdense shards never learned to emit EOS (the
        # converter appends the terminal EOS with mask=False -> unsupervised), so without this stop
        # they ramble repeated pair lists and parse_pairs' regex fallback collects ALL of them,
        # destroying precision. Stop at the first newline after the pair list closes.
        if args.cot_mode == "none":
            ans = txt.split("</think>", 1)[1] if "</think>" in txt else txt
            return "]]" in ans or ans.strip() == "[]"
        return False

    @torch.no_grad()
    def generate_one(prefill):
        gm.prepare_inference_cache(1, args.max_length)
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        if args.variant in ("dense", "full"):
            logits = gm.model(
                torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
            )
            nxt = int(logits[0, -1].argmax().item())
            new_content = []
            for _ in range(max_new_tokens):
                if nxt == eos_id:
                    break
                new_content.append(nxt)
                if nxt == NEWLINE_ID and _answer_complete(new_content):
                    break
                logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
                nxt = int(logits[0, -1].argmax().item())
            text = tok.decode(new_content, skip_special_tokens=True)
            return text.split("</think>", 1)[1] if "</think>" in text else text

        # Landmark: KV-cached decode. Prefill (block-aligned, landmark at every block end) is run once
        # with the chunked grouped-softmax mask + K,V cached; then each generated token is fed
        # incrementally and decoded as a plain landmark query (FREE -> chunk mask is a no-op). The old
        # eager loop inserted a real landmark after every ``mem_freq`` content tokens; we replicate that
        # token stream EXACTLY (feeding the landmark through the cache) so decode stays token-identical.
        logits = gm.model(
            torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
        )
        nxt = int(logits[0, -1].argmax().item())
        new_content = []
        since_landmark = 0
        for _ in range(max_new_tokens):
            if nxt == eos_id:
                break
            new_content.append(nxt)
            logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
            since_landmark += 1
            if since_landmark == args.mem_freq:
                logits = gm.model(
                    torch.tensor([[LANDMARK_TOKEN_ID]], device=device), logits_to_keep=1
                )
                since_landmark = 0
            if nxt == NEWLINE_ID and _answer_complete(new_content):
                break
            nxt = int(logits[0, -1].argmax().item())
        text = tok.decode(new_content, skip_special_tokens=True)
        return text.split("</think>", 1)[1] if "</think>" in text else text

    examples = load_unified_examples(
        args.contra_data, args.max_test_samples, task="contradiction",
        query_position="both", use_alpaca=True,
    )
    import math

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
        res, _ = _eval_contradiction(examples, full)
        summary = {
            "model_path": args.model_path,
            "variant": args.variant,
            "contra_data": args.contra_data,
            "eval_size": len(examples),
            "landmark_top_k_blocks": args.landmark_top_k_blocks,
            "landmark_top_k_fraction": args.landmark_top_k_fraction,
            "contradiction": res,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[contradiction] f1={res.get('f1', 0):.3f} precision={res.get('precision', 0):.3f} "
            f"recall={res.get('recall', 0):.3f} em={res.get('exact_match', 0):.3f} "
            f"parse_rate={res.get('parse_rate', 0):.3f} (eval_size={len(examples)})",
            flush=True,
        )
    if world > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
