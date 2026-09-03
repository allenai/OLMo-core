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


_GOLD_HOP_UNSET = object()


def gold_hop_cfg(model_path: str, key: str, default=_GOLD_HOP_UNSET):
    """
    Read one attention knob out of a checkpoint's ``config.json``.

    The gold-hop base graph is a pure function of ``(random_doc_seed, doc_keep_prob, per-example
    nonce)``, so **eval must rebuild it with the values the model TRAINED under**. Taking them from CLI
    flags instead would let a mismatched ``--doc-keep-prob`` hand the model a completely different
    (still plausible, still gold-edited) graph at eval and quietly measure nothing -- so they are read
    from the checkpoint, which is the only thing that actually knows.

    The config is the launcher's ``{"model": model_config.as_config_dict(), ...}``; the knob lives at
    ``model.block.sequence_mixer.<key>``, but it is searched recursively so a config-layout change
    upstream cannot silently produce a wrong default.

    :raises SystemExit: if the key is absent (and no ``default`` given) or ambiguous.
    """
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)

    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key:
                    found.append(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    uniq = {json.dumps(v, sort_keys=True) for v in found}
    if not found:
        if default is _GOLD_HOP_UNSET:
            raise SystemExit(
                f"[gold-hop] {model_path}/config.json has no {key!r}. This checkpoint was not trained "
                "as a gold_hop_controlled model, so there is no graph to reproduce."
            )
        return default
    if len(uniq) > 1:
        raise SystemExit(
            f"[gold-hop] {model_path}/config.json has conflicting values for {key!r}: {sorted(uniq)}. "
            "Refusing to guess which graph the model trained under."
        )
    return found[0]


def build_eval_prefill(
    tok,
    raw_example,
    *,
    variant: str,
    cot_mode: str,
    doc_start_id: int,
    doc_end_id: int,
    free_pad_repeat: int = 0,
    repeat_doc_text: int = 1,
    summary_every_k: int = 0,
    mem_freq: int = 63,
):
    """
    Render one eval example's **prompt-only prefill** token ids.

    Module-level and public **on purpose**: this is the single source of truth for the eval prefill's
    token layout, and anything that needs to key off that layout must call *this*, not re-derive it.
    Concretely, ``build_gold_pairs_for_eval.py`` fingerprints these exact ids to build the gold-hop
    sidecar -- and a fingerprint is a SHA1, so a one-token drift between two copies of this rendering
    would produce a 0% hit rate (or, worse, a partial one).

    :param variant: ``"dense"`` / ``"full"`` use the dense emitter; ``"landmark"`` packs into landmark
        windows.
    :param cot_mode: ``"none"`` / ``"enumerate"`` / ... -- MUST match the training shard.
    :param free_pad_repeat: MUST match the training shard (extra FREE tokens after the documents).
    :param repeat_doc_text: MUST match the training shard (each document's text repeated N times).
    :param summary_every_k: MUST match the training shard (the ``summary_attention`` span layout).

    :returns: The prefill token ids (a list). Note there is **no answer, no EOS and no padding** here --
        so its content fingerprint differs from the training shard's row for the same example.
    """
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        segment_prompt_to_chunks,
    )

    segs, ids, _ = segment_prompt_to_chunks(
        tok, raw_example, "contradiction", query_position="both", cot_mode=cot_mode,
        chunk_by="document", item_regex=r"\|\|", include_answer=False,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        # MUST equal the training shard's value, or the prefill layout differs from training.
        free_pad_repeat=free_pad_repeat,
        repeat_doc_text=repeat_doc_text,
        summary_every_k=summary_every_k,
    )
    if variant in ("dense", "full"):
        out, _ = emit_document_chunk_dense(segs)  # box markers present; full attention ignores them
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
    ap.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID, help="document-separator EOS id")
    ap.add_argument("--pad-fallback-id", type=int, default=151645,
                    help="generation pad id when the tokenizer pad == eos (Qwen3 default 151645).")
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--max-test-samples", type=int, default=100)
    ap.add_argument(
        "--per-example-out",
        default=None,
        help="if set, dump per-example grading (idx, prediction, gold_pairs, predicted_pairs, "
        "precision/recall/f1/exact_match) as a JSON list -- for connectivity-stratified analysis.",
    )
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
    ap.add_argument(
        "--summary-every-k",
        type=int,
        default=0,
        help="MUST match the training shard: emit the summary_attention layout (one extra "
        "'Summary of claims X to Y: ...' span chunk after every K documents, so chunk "
        "indices run on a stride of K+1). A mismatch silently rebinds every chunk "
        "role, because the mask identifies a span as (chunk_id %% (K+1)) == K. The "
        "bandwidth / relay knobs are NOT here: they are attention-config values "
        "restored from the checkpoint, so ONE shard serves every rung of the ladder.",
    )
    ap.add_argument("--cot-mode", default="enumerate")
    ap.add_argument(
        "--gold-pairs",
        default=None,
        help="REQUIRED for a gold_hop_controlled checkpoint: gold_pairs.json keyed by the PREFILL "
        "fingerprint ({fingerprint: [[a, b], ...]}, 0-based chunk ids). Build it with "
        "src/scripts/data/build_gold_pairs_for_eval.py using the SAME layout flags as this eval -- "
        "the training shard's gold_pairs.json will NOT work here (its rows include the answer + EOS, "
        "so every fingerprint would miss). Every document-bearing row must hit or this eval aborts.",
    )
    ap.add_argument(
        "--gold-hops",
        default=None,
        choices=("1", "2", "3", "inf"),
        help="gold_hop_controlled arm; must equal the arm baked into the checkpoint's config.json "
        "(install_gold_hop_mask refuses to install if they disagree, so a typo cannot score hop_inf "
        "under a run named hop2).",
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
        "'--variant landmark' always uses the bs=1 path (periodic landmark re-insertion during "
        "decode doesn't fit the batched loop). See corpus_reasoning.eval.batched_native_decode.",
    )
    args = ap.parse_args()

    # Pure-argument validation FIRST, before CUDA init / the model build: a flag typo should fail in
    # milliseconds, not after a multi-minute checkpoint load.
    if bool(args.gold_pairs) != bool(args.gold_hops):
        raise SystemExit(
            "--gold-pairs and --gold-hops must be given together (the sidecar names the gold pairs; "
            "the arm names what to do with them)."
        )
    if args.gold_pairs and args.variant != "dense":
        raise SystemExit(
            f"--gold-hops needs --variant dense (DocumentChunkedAttention); got {args.variant!r}. "
            "'--variant full' does not even thread chunk_ids, so the gold mask would be silently "
            "absent and every arm would score as unrestricted attention."
        )
    if args.batch_size > 1 and args.gold_pairs:
        raise SystemExit(
            "--batch-size > 1 is not supported together with --gold-pairs (the gold-hop hook's "
            "per-forward fingerprint lookup is bs=1-only). Use --batch-size 1 for gold_hop_controlled "
            "evals."
        )
    if args.batch_size > 1 and args.variant == "landmark":
        raise SystemExit(
            "--batch-size > 1 is not supported for --variant landmark (periodic landmark-token "
            "re-insertion during decode doesn't fit the batched loop). Use --batch-size 1."
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import (
        DOC_END_ID as _DE,
    )
    from olmo_core.data.document_chunk_landmark import (
        DOC_START_ID as _DS,
    )
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row
    from olmo_core.nn.attention.gold_hop_mask import (
        GOLD_HOPS_INF,
        install_gold_hop_mask,
        make_fingerprint_gold_hop_fn,
    )
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
    # pad_id is always PAD_TOKEN_ID now (not just for landmark): batched eval (--batch-size > 1)
    # left-pads the prompt with this id, and chunk_ids reconstruction must mark it PAD (non-
    # attendable) rather than FREE. At --batch-size 1 no such token ever appears in the dense/full
    # prefill, so this is bit-identical to the old ``pad_id=None`` behavior for those variants.
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
            # force_standard_pattern, which forces the "standard" pattern (causal & not_pad, no
            # document isolation) around each batched call. Thread chunk_ids here so that pattern has
            # a pad role to exclude; bs=1 "full" is untouched (this branch requires batch_size > 1).
            gm.model.enable_document_chunk_attention(
                doc_start_id=ds_id,
                doc_end_id=de_id,
                eos_id=eos_id,
                mode="chunked",
                pad_id=PAD_TOKEN_ID,
            )
        # else: a genuinely plain-attention-trained "full" checkpoint. Its KV-cached prefill already
        # honors cache_leftpad natively (flash_attn_with_kvcache) -- no chunk-id/pattern setup needed,
        # and calling enable_document_chunk_attention would crash (those blocks don't accept
        # chunk_ids).
    print(
        f"[docchunk-contra-{args.variant}] built from {args.model_path} in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- gold_hop_controlled: install the per-example gold-edited doc graph ----
    # The mask is gold-derived, so eval needs the EVAL set's gold pairs -- keyed by the fingerprint of
    # the prompt-only prefill (no answer, no EOS, no padding), which is a different key space from the
    # training shard's rows. Gold identity still never enters the token stream: the lookup is a hash of
    # the ids the model is about to read.
    gold_hop_holder = None
    if args.gold_pairs:
        if not use_cache:
            # Without a KV cache, decode re-forwards the FULL sequence (prompt + tokens generated so
            # far). Its fingerprint is not the prefill's, so every decode row would miss and degrade to
            # plain causal. The hit-rate assert below would catch it, but say why up front.
            raise SystemExit("--gold-hops requires the KV-cached decode path (use_cache=True).")
        gold_pairs_table = json.load(open(args.gold_pairs))
        gold_hop_fn = make_fingerprint_gold_hop_fn(
            gold_pairs_table,
            doc_start_id=ds_id,
            doc_end_id=de_id,
            eos_id=eos_id,
            hops=GOLD_HOPS_INF if args.gold_hops == "inf" else int(args.gold_hops),
            doc_keep_prob=gold_hop_cfg(args.model_path, "doc_keep_prob"),
            seed=gold_hop_cfg(args.model_path, "random_doc_seed", default=42),
            per_example=True,
            # ⚠ Read from the checkpoint, never a CLI flag: decoys change the graph, so a mismatch
            # would evaluate the model on a mask it never trained under -- silently, and with a
            # perfectly plausible-looking f1.
            n_decoys=gold_hop_cfg(args.model_path, "gold_decoys", default=0) or 0,
        )
        gold_hop_holder = install_gold_hop_mask(gm.model, gold_hop_fn)
        print(
            f"[gold-hop] arm={args.gold_hops} sidecar={args.gold_pairs} "
            f"({len(gold_pairs_table)} examples) attached_layers={gold_hop_holder.n_attached} "
            f"keep_prob={gold_hop_cfg(args.model_path, 'doc_keep_prob')} "
            f"decoys={gold_hop_cfg(args.model_path, 'gold_decoys', default=0)} (from config.json)",
            flush=True,
        )

    max_new_tokens = args.contra_max_new_tokens
    cap = args.max_length - max_new_tokens

    def build_prefill(raw_example):
        # Delegates to the module-level renderer, which is also what build_gold_pairs_for_eval.py
        # fingerprints -- one implementation, so the sidecar's keys cannot drift from these ids.
        return build_eval_prefill(
            tok,
            raw_example,
            variant=args.variant,
            cot_mode=args.cot_mode,
            doc_start_id=ds_id,
            doc_end_id=de_id,
            free_pad_repeat=args.free_pad_repeat,
            repeat_doc_text=args.repeat_doc_text,
            summary_every_k=args.summary_every_k,
            mem_freq=args.mem_freq,
        )

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

    # ---- gold-hop PRE-FLIGHT: every prefill must be in the sidecar, checked BEFORE any generation ----
    # Cheap (CPU tokenization) and it fails in seconds instead of after a full eval. The post-hoc holder
    # assert below still runs -- this one proves the KEYS match, that one proves the HOOK actually saw
    # them.
    if gold_hop_holder is not None:
        missing = []
        for gi in my_gidx:
            raw = examples[gi].get("ex", examples[gi])
            fp = content_fingerprint_from_row(build_prefill(raw), eos_id)
            if fp not in gold_pairs_table:
                missing.append((gi, fp))
        if missing:
            shown = "\n  ".join(f"example idx={gi}: {fp}" for gi, fp in missing[:3])
            raise SystemExit(
                f"[gold-hop] FATAL pre-flight: {len(missing)}/{len(my_gidx)} eval prefills are NOT in "
                f"{args.gold_pairs}.\nThose rows would silently fall back to an ALL-TRUE graph = plain "
                "causal over the context, i.e. they would be scored as unrestricted `standard` (near "
                "the ceiling) and the arm would look like a triumphant result while measuring nothing."
                "\nThe sidecar must be built from THIS eval file with THESE layout flags "
                f"(--cot-mode {args.cot_mode} --free-pad-repeat {args.free_pad_repeat} "
                f"--repeat-doc-text {args.repeat_doc_text} --summary-every-k {args.summary_every_k} "
                f"--doc-start-id {ds_id} --doc-end-id {de_id} --eos-token-id {eos_id}) via "
                f"src/scripts/data/build_gold_pairs_for_eval.py.\nmissing:\n  {shown}"
            )
        print(
            f"[gold-hop] pre-flight OK: {len(my_gidx)}/{len(my_gidx)} eval prefills found in the "
            "sidecar",
            flush=True,
        )

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
            if args.variant == "landmark" and args.landmark_top_k_fraction is not None:
                n_blocks = max(1, len(prefill) // block_size)
                gm.model.set_landmark_eval_top_k(
                    max(1, math.ceil(args.landmark_top_k_fraction * n_blocks))
                )
            local.append((gi, generate_one(prefill)))
    else:
        # Batched path (--variant dense | full only -- validated above). Length-filter first (exactly
        # as bs=1), then decode the survivors ``--batch-size`` at a time.
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
            return bool(content) and content[-1] == NEWLINE_ID and _answer_complete(content)

        from contextlib import nullcontext

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

    # ---- gold-hop POST-HOC: the hook must actually have masked every document-bearing forward ----
    # The pre-flight proved the keys match; this proves the graph reached attention. Raises rather than
    # warns: a miss here means those rows ran as plain causal, which reads as a near-ceiling SUCCESS.
    # NB the denominator counts only document-bearing rows -- KV-cached decode feeds one token at a
    # time, carries no documents, and is a FREE query that every arm lets attend the cache causally.
    if gold_hop_holder is not None:
        gold_hop_holder.require_full_hit_rate(context=f"eval of {args.model_path}")
        print(
            f"[gold-hop] hit rate {gold_hop_holder.hit_rate:.1%} on "
            f"{gold_hop_holder.counters.get('graph_rows', 0)} document-bearing rows | "
            f"{gold_hop_holder.summary()}",
            flush=True,
        )

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
        res, details = _eval_contradiction(examples, full)
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
