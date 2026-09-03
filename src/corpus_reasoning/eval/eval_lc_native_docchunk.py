"""
NATIVE olmo_core eval for **document-chunked** models on the single-task ladder tasks -- both the
dense (``DocumentChunkedAttention``) and landmark (``DocumentLandmarkAttention``) variants. Custom
attention can't use HF/vLLM, so we load the olmo-core distcp directly and decode greedily.

Supports ``--task`` ``oolong | contradiction | retrieval | rerank | outlier`` plus the CTC suite
tasks (``records/ctc-suite-scaling-plan.md`` §3): ``redundancy | absence | xabsence | strmatch |
qdmatch | mathmatch | cycle | groups4 | textgroups | reorder | grouping | grouping_labeled | qa |
summarization | cot_retrieval`` (aliases ``nq`` -> ``retrieval``, ``contra`` -> ``contradiction``).
The box-marker prefill is built with the SAME
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

from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
)

# Reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).

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
    "oolong": dict(chunk_by="line", max_new=256, stop="oolong", scorer="oolong", cot="plan"),
    "contradiction": dict(
        chunk_by="document", max_new=200, stop="eos", scorer="contradiction", cot="none"
    ),
    "retrieval": dict(
        chunk_by="document", max_new=64, stop="newline", scorer="retrieval", cot="none"
    ),
    "rerank": dict(chunk_by="document", max_new=512, stop="newline", scorer="rerank", cot="none"),
    "outlier": dict(chunk_by="document", max_new=256, stop="eos", scorer="outlier", cot="none"),
    # ---- CTC suite (records/ctc-suite-scaling-plan.md §3) -- added to cover the ~20 canonical
    # task keys beyond the original 5. chunk_by is "document" for every one of these (oolong is
    # the only "line" task -- see BUILD_MATRIX.md's chunk-by map). max_new mirrors evaluate.py's
    # per-task --max-tokens overrides (~lines 1171-1231); scorer reuses evaluate.py's per-task
    # dispatch (~lines 1411-1450) verbatim -- several canonical tasks deliberately share a scorer
    # function there (redundancy/strmatch/mathmatch -> _eval_contradiction, xabsence -> _eval_absence,
    # groups4/textgroups -> _eval_cycle, cot_retrieval -> _eval_retrieval, grouping_labeled ->
    # _eval_grouping) and TASK_CFG mirrors that reuse via the "scorer" key rather than duplicating.
    # stop rule mirrors evaluate.py's HF `multiline_output` set (cot_retrieval/grouping/
    # grouping_labeled/reorder -> eos-only); summarization is ALSO eos-only here even though
    # evaluate.py's multiline_output omits it -- GovReport references are multi-paragraph and a
    # first-newline stop would truncate them to one line against an intentional max_new=1024
    # budget, so this is a deliberate deviation (documented, not an oversight).
    "redundancy": dict(
        chunk_by="document", max_new=200, stop="newline", scorer="contradiction", cot="none"
    ),
    "absence": dict(chunk_by="document", max_new=200, stop="newline", scorer="absence", cot="none"),
    "xabsence": dict(
        chunk_by="document", max_new=200, stop="newline", scorer="absence", cot="none"
    ),
    "strmatch": dict(
        chunk_by="document", max_new=200, stop="newline", scorer="contradiction", cot="none"
    ),
    "qdmatch": dict(chunk_by="document", max_new=200, stop="newline", scorer="qdmatch", cot="none"),
    "mathmatch": dict(
        chunk_by="document", max_new=200, stop="newline", scorer="contradiction", cot="none"
    ),
    "cycle": dict(chunk_by="document", max_new=200, stop="newline", scorer="cycle", cot="none"),
    "groups4": dict(chunk_by="document", max_new=200, stop="newline", scorer="cycle", cot="none"),
    "textgroups": dict(
        chunk_by="document", max_new=200, stop="newline", scorer="cycle", cot="none"
    ),
    "reorder": dict(chunk_by="document", max_new=1024, stop="eos", scorer="reorder", cot="none"),
    "grouping": dict(chunk_by="document", max_new=2048, stop="eos", scorer="grouping", cot="none"),
    "grouping_labeled": dict(
        chunk_by="document", max_new=2048, stop="eos", scorer="grouping", cot="none"
    ),
    "qa": dict(chunk_by="document", max_new=64, stop="newline", scorer="qa", cot="none"),
    "summarization": dict(
        chunk_by="document", max_new=1024, stop="eos", scorer="summarization", cot="none"
    ),
    "cot_retrieval": dict(
        chunk_by="document", max_new=512, stop="eos", scorer="retrieval", cot="none"
    ),
}
# Convenience aliases (run-name / launcher shorthands) -> canonical segmentation task.
TASK_ALIASES = {"nq": "retrieval", "contra": "contradiction"}


def build_eval_prefill(
    tok,
    raw_example,
    task,
    *,
    variant,
    cot_mode=None,
    doc_start_id=DOC_START_ID,
    doc_end_id=DOC_END_ID,
    mem_freq=63,
    summary_token_id=None,
    n_summary_tokens=5,
):
    """
    Render one eval example's **prompt-only prefill** token ids for any ``TASK_CFG`` task.

    Module-level and public **on purpose** (mirrors
    ``eval_lc_native_docchunk_contra.build_eval_prefill``): this is the single source of truth for
    the general (non-contradiction) prefill layout, called by both this module's own eval loop
    (``main()``'s ``build_prefill`` closure) and any external driver -- e.g. the vLLM parity harness
    in ``debug/ctc_vllm_validation/general/`` -- that needs token-identical prompts without loading a
    model. Keeping one call site (instead of two copies of the ``segment_prompt_to_chunks`` /
    ``emit_document_chunk_*`` invocation) means the two can never silently drift apart.

    :param task: ``TASK_CFG`` key (already resolved through ``TASK_ALIASES`` if needed).
    :param variant: ``"dense"`` / ``"full"`` use the dense (box-marker) emitter; ``"landmark"`` packs
        into landmark windows; ``"summary"`` appends ``n_summary_tokens`` ``<|summ|>`` tokens after
        each context document.
    :param cot_mode: overrides ``TASK_CFG[task]["cot"]`` when given.
    :param mem_freq: landmark variant only -- window size passed to ``emit_document_chunk_landmark``.
    :param summary_token_id: ``"summary"`` variant only -- the ``<|summ|>`` id. Required for that
        variant, and it MUST match the id the training shards were built with.
    :param n_summary_tokens: ``"summary"`` variant only -- tokens per document. MUST equal the
        converter's ``--num-summary-tokens`` and the model's ``n_summary_tokens``: roles are derived
        by counting summary runs, so a mismatch silently rebinds every document index.

    :returns: The prefill token ids (a list). No answer, no EOS and no padding.
    """
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        emit_document_chunk_summary,
        segment_prompt_to_chunks,
    )

    cfg = TASK_CFG[task]
    chunk_by = cfg["chunk_by"]
    cm = cot_mode if cot_mode is not None else cfg["cot"]
    segs, ids, _ = segment_prompt_to_chunks(
        tok,
        raw_example,
        task,
        query_position="both",
        cot_mode=cm,
        chunk_by=chunk_by,
        item_regex=r"\|\|",
        include_answer=False,
        doc_start_id=doc_start_id,
        doc_end_id=doc_end_id,
    )
    if variant in ("dense", "full"):
        out, _ = emit_document_chunk_dense(segs)  # box markers present; full attention ignores them
    elif variant == "summary":
        if summary_token_id is None:
            raise ValueError(
                "variant='summary' requires summary_token_id (the <|summ|> id the training shards "
                "were built with); guessing it would silently rebind every document role."
            )
        out, _ = emit_document_chunk_summary(
            segs, summary_token_id=summary_token_id, n_summary_tokens=n_summary_tokens
        )
    else:
        out, _ = emit_document_chunk_landmark(
            segs, mem_freq=mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
        )
    return out


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
    ap.add_argument(
        "--eos-token-id", type=int, default=EOS_TOKEN_ID, help="document-separator EOS id"
    )
    ap.add_argument(
        "--pad-fallback-id",
        type=int,
        default=151645,
        help="generation pad id when the tokenizer pad == eos (Qwen3 default 151645).",
    )
    ap.add_argument(
        "--task",
        default="oolong",
        help="one of TASK_CFG's keys: oolong | contradiction | retrieval | rerank | outlier | "
        "redundancy | absence | xabsence | strmatch | qdmatch | mathmatch | cycle | groups4 | "
        "textgroups | reorder | grouping | grouping_labeled | qa | summarization | cot_retrieval "
        "(aliases: nq->retrieval, contra->contradiction).",
    )
    # --data is the general eval JSONL; --oolong-data kept as a back-compat alias.
    ap.add_argument(
        "--data", default=None, help="eval JSONL (unified format). Overrides --oolong-data."
    )
    ap.add_argument("--oolong-data", default="data/oolong_test_synth_ctx2048_spliteval.jsonl")
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override the per-task default decode budget.",
    )
    ap.add_argument(
        "--per-example-out",
        default=None,
        help="if set, dump per-example grading (idx, prediction, task-specific fields) as a JSON "
        "list -- for connectivity-stratified analysis.",
    )
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--mem-freq", type=int, default=63)
    ap.add_argument(
        "--cot-mode", default=None, help="override the per-task default prompt CoT mode."
    )
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
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="examples decoded per forward (left-padded, KV-cached). Default 1 keeps the exact "
        "original bs=1 loop. Only '--variant dense' and '--variant full' are supported -- "
        "'--variant landmark' always uses the bs=1 path. See "
        "corpus_reasoning.eval.batched_native_decode.",
    )
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.batch_size > 1 and args.variant == "landmark":
        raise SystemExit(
            "--batch-size > 1 is not supported for --variant landmark (periodic landmark-token "
            "re-insertion during decode doesn't fit the batched loop). Use --batch-size 1."
        )

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

    from corpus_reasoning.eval.evaluate import (
        _eval_absence,
        _eval_contradiction,
        _eval_cycle,
        _eval_grouping,
        _eval_oolong,
        _eval_outlier,
        _eval_qa,
        _eval_qdmatch,
        _eval_reorder,
        _eval_rerank,
        _eval_retrieval,
        _eval_summarization,
        load_unified_examples,
    )
    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import DOC_END_ID as _DE
    from olmo_core.data.document_chunk_landmark import DOC_START_ID as _DS
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import (
        TransformerGenerationModuleConfig,
    )

    SCORERS = {
        "oolong": _eval_oolong,
        "contradiction": _eval_contradiction,  # reused by redundancy, strmatch, mathmatch
        "retrieval": _eval_retrieval,  # reused by cot_retrieval
        "rerank": _eval_rerank,
        "outlier": _eval_outlier,
        "absence": _eval_absence,  # reused by xabsence
        "qdmatch": _eval_qdmatch,
        "cycle": _eval_cycle,  # reused by groups4, textgroups
        "reorder": _eval_reorder,
        "grouping": _eval_grouping,  # reused by grouping_labeled
        "qa": _eval_qa,
        "summarization": _eval_summarization,
    }
    scorer = SCORERS[cfg["scorer"]]

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
    # GenerationConfig requires pad != eos; Qwen3 has no pad, and we decode bs=1 (pad is unused), so
    # any distinct reserved id is fine (default --pad-fallback-id 151645 = <|im_end|>).
    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, eos_id) else args.pad_fallback_id
    # All variants now support a KV cache. Dense / full (DocumentChunkedAttention) prefill with the
    # chunked mask + cache K,V; decode is plain causal over the cache (generated tokens are FREE). The
    # landmark variant (DocumentLandmarkAttention) prefills with the chunked grouped-softmax mask +
    # caches K,V; decode is the plain incremental landmark decode (FREE query, chunk mask is a no-op).
    # Both turn eval from O(gen*n^2) eager re-feeding into O(n^2 + gen*n), identical tokens.
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
    # Belt-and-suspenders: ensure runtime chunk_id reconstruction is on (config.json should already
    # set it, but we control pad_id here). The full-attention baseline has NO chunked mask.
    # pad_id is always PAD_TOKEN_ID now (not just for landmark): batched eval (--batch-size > 1)
    # left-pads the prompt with this id, and chunk_ids reconstruction must mark it PAD (non-
    # attendable) rather than FREE. At --batch-size 1 no such token ever appears in the dense
    # prefill, so this is bit-identical to the old ``pad_id=None`` behavior there.
    if args.variant != "full":
        gm.model.enable_document_chunk_attention(
            doc_start_id=ds_id,
            doc_end_id=de_id,
            eos_id=eos_id,
            mode="chunked",
            pad_id=PAD_TOKEN_ID,
        )
    elif args.batch_size > 1:
        from corpus_reasoning.eval.batched_native_decode import (
            model_has_document_chunked_attention,
        )

        if model_has_document_chunked_attention(gm):
            # This "full" checkpoint's attention layers really are DocumentChunkedAttention (mask
            # disabled for this arm). Batched decode left-pads the prompt, so the pad prefix must
            # still be excluded from attention -- done via batched_native_decode's
            # force_standard_pattern around each batched call. Thread chunk_ids here so that pattern
            # has a pad role to exclude; bs=1 "full" is untouched (requires batch_size > 1).
            gm.model.enable_document_chunk_attention(
                doc_start_id=ds_id,
                doc_end_id=de_id,
                eos_id=eos_id,
                mode="chunked",
                pad_id=PAD_TOKEN_ID,
            )
        # else: a genuinely plain-attention-trained "full" checkpoint -- its KV-cached prefill already
        # honors cache_leftpad natively (flash_attn_with_kvcache); calling
        # enable_document_chunk_attention would crash (those blocks don't accept chunk_ids).
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
            # Stop at the newline that ENDS the single-line answer -- but NOT a
            # LEADING newline before any content. Models often emit a formatting
            # "\n" before the answer; stopping there yields an empty generation
            # and a contaminated metric (obliq/retrieval scored ~random with all
            # gens empty until this fix). new_content includes nxt_id, so check
            # the content BEFORE it.
            prior = tok.decode(new_content[:-1], skip_special_tokens=True).strip()
            return len(prior) > 0
        # "oolong": stop at a newline only once the templated "answer:" line has been emitted
        return "answer:" in tok.decode(new_content, skip_special_tokens=True).lower()

    def build_prefill(raw_example):
        return build_eval_prefill(
            tok,
            raw_example,
            seg_task,
            variant=args.variant,
            cot_mode=cot_mode,
            doc_start_id=ds_id,
            doc_end_id=de_id,
            mem_freq=args.mem_freq,
        )

    block_size = (
        args.mem_freq + 1
    )  # landmark window (64); the eager landmark forward needs T % 64 == 0

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
                if nxt == eos_id:
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
            if nxt == eos_id:
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
        eval_data,
        args.max_test_samples,
        task=seg_task,
        query_position="both",
        use_alpaca=True,
    )
    import math

    block_size = args.mem_freq + 1
    if args.variant == "landmark" and args.landmark_top_k_blocks is not None:
        n_set = gm.model.set_landmark_eval_top_k(args.landmark_top_k_blocks)
        print(
            f"[topk] fixed top_k={args.landmark_top_k_blocks} on {n_set} landmark layers",
            flush=True,
        )

    my_gidx = list(range(rank, len(examples), world))
    local = []

    def _reject_too_long(gi, n_prefill):
        """An over-budget prefill is a config error, not a wrong answer -- refuse to score it.

        Scoring it as an empty generation (the old behaviour) grades the model on an example it
        was never shown, laundering a --max-length mistake into the metric where it reads as a
        long-context collapse.
        """
        raise SystemExit(
            f"[maxlen] example {gi} builds a {n_prefill}-token prefill, past the {cap}-token cap "
            f"(--max-length {args.max_length} minus max_new_tokens {max_new_tokens}). Scoring it "
            f"as empty would grade the model on an example it never saw. Re-run with "
            f"--max-length >= {n_prefill + max_new_tokens}."
        )

    if args.batch_size <= 1:
        for gi in my_gidx:
            raw = examples[gi].get("ex", examples[gi])
            prefill = build_prefill(raw)
            if len(prefill) > cap:
                _reject_too_long(gi, len(prefill))
            # Per-example top-k from a fraction of this prompt's landmark blocks (landmark only).
            if args.variant == "landmark" and args.landmark_top_k_fraction is not None:
                n_blocks = max(1, len(prefill) // block_size)
                gm.model.set_landmark_eval_top_k(
                    max(1, math.ceil(args.landmark_top_k_fraction * n_blocks))
                )
            local.append((gi, generate_one(prefill)))
    else:
        # Batched path (--variant dense | full only -- validated above). Length-filter first (exactly
        # as bs=1), then decode the survivors ``--batch-size`` at a time.
        from contextlib import nullcontext

        from corpus_reasoning.eval.batched_native_decode import (
            force_standard_pattern,
            generate_batch_docchunk,
        )

        keep: list = []
        for gi in my_gidx:
            raw = examples[gi].get("ex", examples[gi])
            prefill = build_prefill(raw)
            if len(prefill) > cap:
                _reject_too_long(gi, len(prefill))
            keep.append((gi, prefill))

        def is_answer_complete(content):
            return bool(content) and should_stop(content[-1], content)

        ctx = force_standard_pattern(gm) if args.variant == "full" else nullcontext()
        with ctx:
            for start in range(0, len(keep), args.batch_size):
                group = keep[start : start + args.batch_size]
                gidxs = [gi for gi, _ in group]
                prefills = [p for _, p in group]
                outs = generate_batch_docchunk(
                    gm,
                    prefills,
                    device=device,
                    eos_id=eos_id,
                    pad_token_id=PAD_TOKEN_ID,
                    max_new_tokens=max_new_tokens,
                    max_length=args.max_length,
                    is_answer_complete=is_answer_complete,
                )
                for gi, new_content in zip(gidxs, outs):
                    text = tok.decode(new_content, skip_special_tokens=True)
                    text = text.split("</think>", 1)[1] if "</think>" in text else text
                    local.append((gi, text))

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
        res, details = scorer(examples, full)
        if args.per_example_out:
            for i, d in enumerate(details):
                d["idx"] = i
            os.makedirs(os.path.dirname(args.per_example_out) or ".", exist_ok=True)
            with open(args.per_example_out, "w") as f:
                json.dump(details, f)
            print(f"[per-example] wrote {len(details)} rows -> {args.per_example_out}", flush=True)
        summary = {
            "model_path": args.model_path,
            "variant": args.variant,
            "task": seg_task,
            "data": eval_data,
            "eval_size": len(examples),
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
        print(
            f"[{seg_task}] n={len(examples)} "
            + " ".join(f"{k}={v:.3f}" for k, v in scalars.items()),
            flush=True,
        )
    if world > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
