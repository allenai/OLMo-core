"""Step B of the chunked-vllm validation gate: run vLLM on the native prefills.

Runs INSIDE the vLLM venv (/data/prasann/ctc_vllm_venv). Two modes:

  --mode chunked : install corpus_reasoning.lib.vllm_chunked_patch, force the
                   FlexAttention backend, set doc token ids (box markers), and
                   verify the patch actually rebuilt the BlockMask (direct vs
                   fallback path counts are reported).
  --mode full    : plain vLLM, default backend — the full-attention reference.

Decoding replicates the native harness's stop logic, TASK-DRIVEN (``--task``, default
``contradiction`` for backward compat):

* ``contradiction`` -- unchanged: mirrors
  ``eval_lc_native_docchunk_contra.generate_one``'s dedicated ad-hoc rule (greedy, stop at EOS,
  post-hoc truncation at the first newline once the answer is complete -- '[[..]]' emitted /
  'contradicting pairs:' seen -- then strip everything up to '</think>'). This is the proven path;
  its behavior is byte-identical to before this task-driven generalization.
* every other task -- mirrors ``eval_lc_native_docchunk.py``'s generic ``should_stop()`` /
  ``TASK_CFG[task]["stop"]`` (``eos`` / ``newline`` / ``oolong``). The stop rule is read from the
  prefills pack's ``"stop_rule"`` field (written by
  ``debug/ctc_vllm_validation/general/build_prefills_any.py``) so there is one source of truth,
  not a second hardcoded copy of ``TASK_CFG`` living inside the vLLM venv.

For ``stop_rule == "newline"`` the newline id is ALSO added to vLLM's own ``stop_token_ids`` (not
just post-hoc truncation) -- the native harness genuinely stops decoding at the first newline for
these tasks, so this mirrors that early-stop instead of just cleaning up after an over-long
generation. ``eos``/``oolong``/contradiction's ad-hoc rule cannot be expressed as an unconditional
vLLM stop token (the stop point depends on generated content), so those stay post-hoc-only, exactly
as the native harness's own decode loop does (it only ever hard-stops on EOS; the early "answer
complete" breaks are content-conditioned too).
"""

import argparse
import json
import os
import sys
import time

# Env quirks from the venv build — must be set before vllm import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HOME", "/data/prasann/hf")
os.environ.setdefault("TMPDIR", "/data/prasann/tmp")


def truncate_like_native(gen_ids, tok, newline_id, cot_mode="none"):
    """Mirror eval_lc_native_docchunk_contra.generate_one's stop logic (contradiction only,
    UNCHANGED -- do not touch; this is the proven path)."""

    def answer_complete(content_ids):
        txt = tok.decode(content_ids, skip_special_tokens=True).lower()
        if "contradicting pairs:" in txt:
            return True
        if cot_mode == "none":
            ans = txt.split("</think>", 1)[1] if "</think>" in txt else txt
            return "]]" in ans or ans.strip() == "[]"
        return False

    new_content = []
    for t in gen_ids:
        new_content.append(t)
        if t == newline_id and answer_complete(new_content):
            break
    text = tok.decode(new_content, skip_special_tokens=True)
    return text.split("</think>", 1)[1] if "</think>" in text else text


def truncate_generic(gen_ids, tok, newline_id, stop_rule):
    """Mirror eval_lc_native_docchunk.py's should_stop() for every non-contradiction task.

    ``stop_rule`` is ``TASK_CFG[task]["stop"]`` (``eos`` / ``newline`` / ``oolong``), read out of
    the prefills pack -- see module docstring. Faithfully replays the native decode loop's early
    stop token-by-token (append, then check), so behavior matches regardless of whether vLLM's own
    ``stop_token_ids`` already cut generation short (in which case this is a no-op pass-through).
    """
    # Decode the WHOLE generation and truncate at the TEXT level. Token-level newline matching is
    # unreliable: some checkpoints RAMBLE (repeat the answer, degenerating -- e.g. niah emits
    # "8]\n8]\n8]\n...") and the model's newline can be a merged token that never equals newline_id,
    # so the vLLM stop_token_ids + a token-by-token loop both miss it and the grader then collects
    # the whole rambled set (niah 0.16, outlier 0.66). Text-level first-line is robust to any
    # tokenization. Also strip a leading <think>...</think> block if the model emitted reasoning.
    text = tok.decode(gen_ids, skip_special_tokens=True)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if stop_rule == "newline":
        # NEWLINE_ROBUST=1 (opt-in, off by default so every already-validated grid task stays
        # byte-identical): some checkpoints emit a LEADING blank line (or an unclosed <think>\n)
        # before the real answer. The default vLLM `stop=["\n"]` then fires on that first newline
        # and returns EMPTY -- measured 61.8% empty on helmet_qa dense, halving token_f1. With the
        # flag set we drop the vLLM newline stop (below) and here keep the first NON-EMPTY line
        # instead of the literal first line, recovering the leading-newline-clobbered answers.
        if os.environ.get("NEWLINE_ROBUST") == "1":
            for ln in text.split("\n"):
                if ln.strip():
                    return ln
            return ""
        return text.split("\n", 1)[0]  # single-line answer: keep only the first line
    if stop_rule == "oolong":
        # keep through the first newline AFTER the templated "answer:" line has appeared
        low = text.lower()
        if "answer:" in low:
            i = low.index("answer:")
            return text[:i] + text[i:].split("\n", 1)[0]
        return text
    return text  # "eos": multi-line answer kept as-is


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--prefills", required=True, help="output of build_prefills.py")
    ap.add_argument("--mode", required=True, choices=["chunked", "full"])
    ap.add_argument(
        "--task",
        default="contradiction",
        help="task the prefills pack was built for. Default 'contradiction' reproduces the "
        "original (pre-generalization) behavior exactly regardless of what the pack carries. "
        "For any other task the pack MUST carry a 'stop_rule' field (written by "
        "debug/ctc_vllm_validation/general/build_prefills_any.py) -- eos | newline | oolong.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6,
                    help="vLLM gpu_memory_utilization; long rungs need more KV cache")
    ap.add_argument(
        "--model-family",
        default="qwen3_5",
        choices=["qwen3_5", "qwen3"],
        help="qwen3_5 (default, unchanged): apply the GDN-hybrid serving recipe -- the "
        "Qwen3_5ForCausalLM architecture override and limit_mm_per_prompt=0 that make vLLM load "
        "only the language model of the multimodal wrapper. qwen3: PLAIN DENSE -- a normal "
        "single-arch causal LM with no VL wrapper and no vision tower, so those two overrides "
        "are wrong for it (they would name an architecture the config does not have). Needed "
        "for the dense arm of the qwen3-vs-qwen3.5 comparison.",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vLLM TP degree; >1 shards a long-context KV cache across GPUs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.prefills) as f:
        pack = json.load(f)
    rows = pack["rows"]
    eos_id = pack["eos_token_id"]
    newline_id = pack["newline_id"]

    # Task-driven stop logic (generalizes the old contradiction-only hardcoding). Backward
    # compat: --task defaults to "contradiction", which always uses the dedicated ad-hoc rule
    # (truncate_like_native) no matter what -- byte-identical to the pre-generalization behavior,
    # even against an old-style prefills pack that carries no "canonical_task"/"stop_rule" keys.
    canonical_task = pack.get("canonical_task", args.task)
    is_contra = args.task == "contradiction" or canonical_task == "contradiction"
    stop_rule = None
    if not is_contra:
        stop_rule = pack.get("stop_rule")
        if stop_rule is None:
            raise SystemExit(
                f"--task {args.task!r}: prefills pack {args.prefills!r} has no 'stop_rule' field. "
                "Non-contradiction tasks must be built with "
                "debug/ctc_vllm_validation/general/build_prefills_any.py (the old contradiction-"
                "only build_prefills.py has no stop-rule concept)."
            )
        if stop_rule not in ("eos", "newline", "oolong"):
            raise SystemExit(f"--task {args.task!r}: unrecognized stop_rule {stop_rule!r}")
    print(
        f"[driver] task={args.task!r} canonical_task={canonical_task!r} "
        f"stop_logic={'contradiction (ad-hoc, unchanged)' if is_contra else stop_rule}",
        flush=True,
    )

    # The nominal rung token-budget badly underestimates real prefill length (per-doc marker
    # scaffold), so size max_model_len from the ACTUAL longest prompt + generation headroom.
    # Under-sizing overflows vLLM's tokenizer/scheduler and drops the whole rung.
    longest = max(len(r["prefill"]) for r in rows)
    need = longest + args.max_new_tokens + 256
    if need > args.max_model_len:
        print(f"[driver] bumping max_model_len {args.max_model_len} -> {need} "
              f"(longest prefill={longest} + {args.max_new_tokens} gen + 256)", flush=True)
        args.max_model_len = need

    if args.mode == "chunked":
        # In-process monkey-patch must reach the model worker.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from corpus_reasoning.lib import vllm_chunked_patch
        vllm_chunked_patch.install()
        vllm_chunked_patch.set_doc_token_ids(pack["doc_start_id"], pack["doc_end_id"])
        print(f"[driver] patch installed; doc ids = "
              f"{pack['doc_start_id']}/{pack['doc_end_id']}", flush=True)

    from vllm import LLM, SamplingParams, TokensPrompt

    t0 = time.time()
    # Chunked mode note: the fallback BlockMask build evaluates the mask over
    # num_actual_tokens x total_cache_tokens, so an unconstrained KV cache
    # (0.5 util on an H200 ~= millions of cache tokens) OOMs. Cap the cache to
    # what the eval needs (128 pages x 544 tok/page ~= 70k tokens) and the
    # per-step token budget.
    extra = {}
    if args.mode == "chunked":
        # Scale the page-count override with max_model_len (page size 544 tokens/page here):
        # need enough physical KV pages for several concurrent long sequences, or the fallback
        # BlockMask (O(num_actual_tokens x total_cache_tokens)) either OOMs (too many pages) or
        # rejects prompts (too few). ~10 sequences worth of headroom at max_model_len.
        # COST MODEL (measured 2026-08-11). We are always on the FALLBACK mask path:
        # `_patch_flex_kernel_options_pow2` forces kv_block_size != block_size, which sets
        # vLLM's `direct_build = False` (flex_attention.py:882), so every step calls
        #     create_block_mask_compiled(mask_mod, q_len=num_actual_tokens,
        #                                kv_len=TOTAL_CACHE_TOKENS, BLOCK_SIZE=(q_blk, kv_blk))
        # kv_len is the WHOLE allocated cache, not the live sequence lengths. So
        #     mask cost per call  ~  (num_actual_tokens/q_blk) * (total_cache_tokens/kv_blk)
        #     work    per call  ~   num_actual_tokens
        #  => mask cost per unit of work ~ total_cache_tokens / kv_blk, independent of how many
        #     sequences are actually resident. Over-allocating pages is therefore a pure tax, and
        #     it grows with the rung because n_pages scales with the (auto-bumped) max_model_len.
        # CHUNK_SEQ_HEADROOM shrinks that allocation; CHUNK_MAX_NUM_SEQS / CHUNK_MAX_BATCHED_TOKENS
        # are the matching concurrency knobs. All three default to the historical values.
        headroom = int(os.environ.get("CHUNK_SEQ_HEADROOM", "10"))
        n_pages = max(128, (args.max_model_len * headroom) // 544 + 1)
        # ROUND-2 (2026-08-11): the BlockMask's kv_len (= n_pages * page_size) must be an exact
        # multiple of kv_block_size, or create_block_mask/flex disagree on the last partial block.
        # Round n_pages UP so `n_pages * 528 % kv_block_size == 0`. With the historical
        # kv_block_size=16 this is a no-op (528 = 16*33 is already a multiple of 16), so the
        # default path is bit-unchanged.
        _kv_blk = int(os.environ.get("CHUNK_KV_BLOCK", "16"))
        _page = 528
        while (n_pages * _page) % _kv_blk != 0:
            n_pages += 1
        extra = dict(
            num_gpu_blocks_override=n_pages,
            max_num_batched_tokens=int(os.environ.get("CHUNK_MAX_BATCHED_TOKENS", "2048")),
            max_num_seqs=int(os.environ.get("CHUNK_MAX_NUM_SEQS", "8")),
        )
        if os.environ.get("CHUNK_VARLEN_PREFILL") == "1":
            # ROUND 5: a prefix-cache hit arrives as a PARTIAL prefill (q_len < seq_len
            # with doc-token queries), which pushes that step off the varlen path onto the
            # slow flex fallback. Keep every prompt a pure one-shot prefill instead.
            extra["enable_prefix_caching"] = False
        print(f"[driver] chunked KV config: n_pages={n_pages} (headroom={headroom} seqs) "
              f"max_num_batched_tokens={extra['max_num_batched_tokens']} "
              f"max_num_seqs={extra['max_num_seqs']}", flush=True)
    # ROUND-2 diagnostic knobs for the chunked FlexAttention tile sizes. Both default to the
    # historical values, so production behavior is unchanged unless the env var is set.
    #   CHUNK_KV_BLOCK -> flex_attn_kv_block_size (BlockMask kv granularity AND, via
    #                     get_kernel_options' fallback branch, the triton kernel's BLOCK_N).
    #   CHUNK_Q_BLOCK  -> flex_attn_q_block_size (BlockMask q granularity AND BLOCK_M).
    # On the FALLBACK build path (which chunked mode is always on, see the note below) these are
    # pure block-sparsity granularity: flex still applies mask_mod elementwise inside every
    # PARTIAL block, so changing them cannot change the attention result -- only its speed.
    _flex_kv_block = int(os.environ.get("CHUNK_KV_BLOCK", "16"))
    _flex_q_block = os.environ.get("CHUNK_Q_BLOCK")
    _attn_cfg = {"backend": "FLEX_ATTENTION", "flex_attn_kv_block_size": _flex_kv_block}
    if _flex_q_block:
        _attn_cfg["flex_attn_q_block_size"] = int(_flex_q_block)
    # CUDA graphs: historically forced off (enforce_eager=True). vLLM 0.25.1's FlexAttention
    # backend declares AttentionCGSupport.ALWAYS, so this is worth measuring, not assuming.
    #
    # MEASURED 2026-08-11: `enforce_eager=False` with vLLM's default cudagraph mode captures FULL
    # decode graphs and then dies with `torch.AcceleratorError: CUDA error: device-side assert
    # triggered` on the first replay. A full graph freezes the BlockMask tensors it captured, but
    # the chunked patch allocates a BRAND NEW block_mask every step via create_block_mask, so the
    # replay indexes through stale pointers. vLLM's FlexAttention cudagraph support is built on the
    # DIRECT-build path's persistent buffers (`persistent_kv_indices`), which chunked mode never
    # takes. CHUNK_CUDAGRAPH_MODE lets us ask for PIECEWISE instead, which splits the graph at the
    # attention op and so leaves the block_mask out of any capture.
    _eager = os.environ.get("CHUNK_ENFORCE_EAGER", "1") == "1"
    _cg_mode = os.environ.get("CHUNK_CUDAGRAPH_MODE")
    if _cg_mode and not _eager:
        extra["compilation_config"] = {"cudagraph_mode": _cg_mode}
    if args.mode == "chunked":
        print(f"[driver] chunked flex tiles: kv_block={_flex_kv_block} "
              f"q_block={_flex_q_block or 'default(16)'} enforce_eager={_eager}", flush=True)
    # The two overrides below exist ONLY to serve the multimodal Qwen3.5 wrapper as text-only.
    # A plain dense Qwen3 export has no VL wrapper and no vision tower, so naming
    # Qwen3_5ForCausalLM would point at an architecture its config does not declare. Drop both
    # for --model-family qwen3 and let vLLM use the config's own architecture.
    if args.model_family == "qwen3_5":
        extra.update(
            hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
    llm = LLM(
        model=args.hf_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=_eager,
        # NOTE: no gdn_prefill_backend override — the forced "triton" path needs fla +
        # causal_conv1d (absent in this venv); vLLM 0.25.1's default GDN backend works.
        # `extra` carries the qwen3_5-only hf_overrides/limit_mm_per_prompt set above (see the
        # canonical Qwen3.5 recipe in corpus_reasoning lib/vllm_utils.load_model), plus the
        # chunked-mode KV page caps.
        **extra,
        # Hybrid Qwen3.5: FlexAttention's flex_attn_kv_block_size MUST divide the model's
        # KV-cache page size (block_size). For this GDN-hybrid the attention page is 528
        # tokens (= 16*33), NOT 544 (an earlier comment's wrong assumption). 32 does NOT
        # divide 528 and SILENTLY corrupts flex output — every generation degenerates to
        # token-0 "!!!!" with parse_rate 0, even for plain causal full attention (this looked
        # like a broken checkpoint but is purely a flex block-size bug). 16 is the largest
        # power of 2 that divides 528 and produces correct output (validated against the
        # default backend: qdmatch@2048 f1 0.65, cycle@2048 f1 0.996). This also keeps
        # direct_build=False (kv_block_size != cache block size); the patched fallback path in
        # vllm_chunked_patch builds the BlockMask from the chunked mask_mod explicitly, and
        # its builder asserts block_size % kv_block_size == 0 so any future page-size change
        # fails loudly instead of silently emitting garbage.
        attention_config=(_attn_cfg if args.mode == "chunked" else None),
    )
    load_s = time.time() - t0
    print(f"[driver] LLM up in {load_s:.1f}s", flush=True)
    tok = llm.get_tokenizer()

    # vLLM-level stop: EOS always; add a "\n" STOP STRING (not token id) for newline tasks. A
    # string stop matches the DECODED text, so it's robust to the model's newline being a merged
    # token (the token-id stop missed that and let ramblers run to max_tokens). truncate_generic's
    # text-level first-line is still the correctness backstop. (`stop` strings are excluded from
    # the output text by vLLM, which is fine -- we re-derive the answer from token_ids + truncate.)
    stop_token_ids = [eos_id]
    # EXTRA_STOP_TOKEN_IDS: comma-separated additional stop token ids. OFF by default.
    #
    # WHY THIS EXISTS (root-caused 2026-08-13 on the oolong 4B pair). `eos_id` comes from the
    # prefill pack and is 248044 = `<|endoftext|>`, but our SFT targets end the assistant turn with
    # `<|im_end|>` = 248046 -- so the model emits its answer, signals end-of-turn, and vLLM keeps
    # sampling because 248046 is not a stop id. The model then degenerates into repeating the
    # answer for the full max_new_tokens ("1111...", "positivepositive...").
    #
    # Most tasks survive this by accident: their answers contain "]]" or a newline, so
    # truncate_generic's text-level truncation salvages the first answer. OOLONG does not -- its
    # answer is a bare number/label with no newline, and the "oolong" stop rule only truncates
    # after a literal "answer:" that the current prompt format never emits. Result: the grader sees
    # the whole ramble and scores 0.0055 on a model that had the answer right.
    #
    # NOT on by default: it changes generation, so enabling it for an already-validated task needs
    # the same eval_size=500 re-validation as any other change here -- and other evals may be
    # running against this file right now.
    _extra_ids = os.environ.get("EXTRA_STOP_TOKEN_IDS")
    if _extra_ids:
        stop_token_ids = stop_token_ids + [
            int(x) for x in _extra_ids.split(",") if x.strip()
        ]
        print(f"[driver] stop_token_ids: {stop_token_ids} (eos={eos_id} + EXTRA_STOP_TOKEN_IDS)",
              flush=True)
    # NEWLINE_ROBUST=1: drop the vLLM-level newline stop string entirely (EOS-only stop) so the
    # model can generate past a leading blank line / unclosed <think> to reach the real answer;
    # truncate_generic then extracts the first non-empty line post-hoc. Off by default -> the
    # standard newline tasks keep their proven `stop=["\n"]` early-stop behavior unchanged.
    newline_robust = os.environ.get("NEWLINE_ROBUST") == "1"
    stop_strings = (
        ["\n"] if (not is_contra and stop_rule == "newline" and not newline_robust) else None
    )
    # CHUNK_EXTRA_STOP: comma-separated extra vLLM stop strings, OFF by default.
    #
    # WHY THIS EXISTS (measured 2026-08-12). contradiction runs with stop_rule="eos", but the model
    # never emits EOS, so every request generates the full `max_new_tokens` (488 of 512 tokens on
    # average at the 8k rung) and `truncate_like_native` then discards ~470 of them -- the kept
    # answer is ~34 characters. ~96% of all decode work in the eval is thrown away. Since
    # truncate_like_native's completion test for cot_mode="none" is `"]]" in ans`, stopping the
    # sampler at "]]" cuts generation at the point the answer is already complete.
    # NOT enabled by default: it changes generation, so it needs the same eval_size=500 gate as any
    # other change here.
    _extra_stop = os.environ.get("CHUNK_EXTRA_STOP")
    if _extra_stop:
        stop_strings = (stop_strings or []) + [s for s in _extra_stop.split(",") if s]
        print(f"[driver] extra stop strings: {stop_strings}", flush=True)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids,
        stop=stop_strings,
    )
    inputs = [TokensPrompt(prompt_token_ids=r["prefill"]) for r in rows]
    t1 = time.time()
    outputs = llm.generate(inputs, sp)
    gen_s = time.time() - t1
    print(f"[driver] generated {len(outputs)} in {gen_s:.1f}s", flush=True)

    responses = {}
    raw_responses = {}  # untruncated decode, only saved when DUMP_RAW_GENS=1 (diagnostics)
    dump_raw = os.environ.get("DUMP_RAW_GENS") == "1"
    for r, o in zip(rows, outputs):
        gen_ids = list(o.outputs[0].token_ids)
        # vLLM includes the matched stop token id in token_ids sometimes;
        # native excludes EOS — drop it defensively.
        if gen_ids and gen_ids[-1] == eos_id:
            gen_ids = gen_ids[:-1]
        if dump_raw:
            raw_responses[str(r["idx"])] = tok.decode(gen_ids, skip_special_tokens=True)
        if is_contra:
            responses[str(r["idx"])] = truncate_like_native(
                gen_ids, tok, newline_id, cot_mode=pack["cot_mode"])
        else:
            responses[str(r["idx"])] = truncate_generic(gen_ids, tok, newline_id, stop_rule)

    debug = {}
    if args.mode == "chunked":
        from corpus_reasoning.lib import vllm_chunked_patch
        debug = dict(vllm_chunked_patch.get_debug_state())
        print(f"[driver] patch debug state: {debug}", flush=True)
        if not debug.get("applied"):
            print("[driver] *** WARNING: chunked patch NEVER APPLIED — "
                  "this run is UNMASKED ***", flush=True)

    with open(args.out, "w") as f:
        json.dump({
            "mode": args.mode,
            "hf_model": args.hf_model,
            "prefills": args.prefills,
            "task": args.task,
            "canonical_task": canonical_task,
            # Carried through from the prefill pack so the prompt layout travels with the numbers:
            # a result scored at the wrong query_position is not comparable to one that wasn't, and
            # that is unrecoverable after the fact. Older packs predate the field -> None.
            "query_position": pack.get("query_position"),
            "cot_mode": pack.get("cot_mode"),
            "stop_rule": "contradiction-ad-hoc" if is_contra else stop_rule,
            "eval_size": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "load_seconds": load_s,
            "gen_seconds": gen_s,
            "patch_debug": debug,
            "responses": responses,
            **({"raw_responses": raw_responses} if dump_raw else {}),
        }, f)
    print(f"[driver] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
