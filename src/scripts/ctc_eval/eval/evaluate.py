"""Unified evaluation script for all tasks and inference backends.

Supports any combination of:
  Task:    --task {retrieval, cot_retrieval, qa, contradiction, outlier, grouping, grouping_labeled, reorder}
  Backend: --backend {vllm, chunked-sdpa, chunked-flex, standard}
  Data:    --eval-data (unified JSONL) or --datasets (HELMET KILT format)

The backend controls how inference is run:
  - vllm: Fast batched inference via vLLM (no custom attention masks)
  - chunked-sdpa: HuggingFace with chunked 4D SDPA masks (docs isolated)
  - chunked-flex: HuggingFace with FlexAttention block-sparse masks (faster)
  - standard: HuggingFace with standard causal attention (baseline)

The task controls prompt formatting and metrics:
  - retrieval / cot_retrieval: Doc IDs; retrieval EM/recall/precision/F1
  - qa: free-form answers; EM/SubEM/F1
  - contradiction: claim-pair predictions; pair precision/recall/F1/EM
  - outlier: outlier doc IDs; set precision/recall/F1/EM + per-source agg
  - grouping / grouping_labeled: clustering of docs; ARI, NMI, pairwise P/R/F1,
    k_exact, coverage, EM + per-level agg
  - reorder: permutation of passages; kendall_tau, spearman_rho, PMR,
    position/pairwise accuracy + per-source and per-N-bin agg

Usage:
    # Retrieval eval with chunked attention
    python scripts/eval/evaluate.py --backend chunked-sdpa --task retrieval \\
        --eval-data data/hotpotqa_eval_k20_shuffled_bridge_500.jsonl \\
        --lora-path outputs/model

    # QA eval with vLLM
    python scripts/eval/evaluate.py --backend vllm --task qa \\
        --datasets nq,hotpotqa --lora-path outputs/model

    # Retrieval eval with vLLM
    python scripts/eval/evaluate.py --backend vllm --task retrieval \\
        --eval-data data/nq_eval_k20_random_500.jsonl --lora-path outputs/model

    # HELMET base model eval (no fine-tuning)
    python scripts/eval/evaluate.py --backend vllm --task qa \\
        --datasets nq --shots 2
"""

import argparse
import json
import math
import os
import re
import random
import torch
from pathlib import Path
from tqdm import tqdm

import hashlib

from ctc_eval.lib.io import load_jsonl, save_results, format_alpaca_prompt, insert_dummy_tokens
from ctc_eval.lib.data_format import build_prompt
from ctc_eval.lib.prompts import (
    PASSAGE_TEMPLATE, PASSAGE_TEMPLATE_NO_TITLE,
    QA_INSTRUCTION, DEMO_TEMPLATE,
    HELMET_TEMPLATE, HELMET_TEMPLATE_QUERY_BEFORE, HELMET_TEMPLATE_QUERY_BOTH,
    helmet_rerank_passage,
)
from ctc_eval.lib.metrics import (
    exact_match, substring_match, token_f1, max_over_answers, aggregate,
    parse_doc_ids, retrieval_exact_match, retrieval_recall,
    retrieval_precision, retrieval_f1,
)
from ctc_eval.lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    build_chunked_causal_mask, find_chunk_spans,
    PAD_CHUNK_ID, FREE_CHUNK_ID,
    AttentionPattern, build_flex_mask_mod, build_dense_bool_mask,
    build_is_anchor, build_random_doc_edges, build_random_token_keep,
    install_hierarchical_sdpa_attention, attach_hierarchical_pre_hook,
)
from ctc_eval.lib.eval_tasks import (
    parse_outlier_ids as _parse_outlier_ids,
    parse_partition as _parse_partition,
    partition_to_labels as _partition_to_labels,
    parse_permutation as _parse_permutation,
)

# ── Contradiction helpers ──

def parse_pairs(text: str) -> list[list[int]] | None:
    """Extract list of integer pairs from model output."""
    text = text.strip()
    for candidate in [text, re.search(r'\[[\s\S]*\]', text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [sorted([int(p[0]), int(p[1])]) for p in parsed if isinstance(p, list) and len(p) == 2]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    matches = re.findall(r'[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]', text)
    if matches:
        return [sorted([int(a), int(b)]) for a, b in matches]
    return [] if text in ("[]", "") else None


def pair_metrics(predicted, gold):
    """Compute precision/recall/F1/EM for predicted vs gold contradiction pairs."""
    pred_set, gold_set = {tuple(p) for p in predicted}, {tuple(p) for p in gold}
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "exact_match": float(pred_set == gold_set)}


def parse_qd_pairs(text):
    """Like parse_pairs, but ORDER-PRESERVING — for qdmatch the pair is
    (query_id, doc_id) and the two indices are not interchangeable, so we must
    NOT sort. Returns list of [a, b] (order kept), or None on parse failure."""
    text = text.strip()
    for candidate in [text, re.search(r'\[[\s\S]*\]', text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [[int(p[0]), int(p[1])] for p in parsed
                        if isinstance(p, list) and len(p) == 2]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    matches = re.findall(r'[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]', text)
    if matches:
        return [[int(a), int(b)] for a, b in matches]
    return [] if text in ("[]", "") else None


def parse_cycles(text: str) -> list[list[int]] | None:
    """Extract a list of cycles (each a list of claim IDs) from model output.

    Accepts a JSON list of lists `[[3,8,12], ...]`, or a single flat list of
    ints `[3,8,12]` (interpreted as one cycle). Falls back to scanning bracketed
    integer groups. Each cycle is normalized to a sorted, de-duplicated ID list.
    """
    text = text.strip()
    for candidate in [text, re.search(r'\[[\s\S]*\]', text)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            if parsed and all(isinstance(x, int) for x in parsed):
                return [sorted(set(parsed))]  # single flat cycle
            out = []
            for c in parsed:
                if isinstance(c, list) and len(c) >= 2:
                    try:
                        out.append(sorted({int(x) for x in c}))
                    except (ValueError, TypeError):
                        pass
            return out
    groups = re.findall(r'\[([\d,\s]+)\]', text)
    if groups:
        out = []
        for g in groups:
            ids = [int(x) for x in re.findall(r'\d+', g)]
            if len(ids) >= 2:
                out.append(sorted(set(ids)))
        return out
    return [] if text in ("[]", "") else None


def cycle_metrics(predicted, gold):
    """Score predicted vs gold cycles. A cycle is a set of claim IDs; a predicted
    cycle is a true positive only if its ID-set exactly equals a gold cycle's.

    Reports cycle-level precision/recall/F1/exact_match (set-of-sets), plus a
    softer claim-level F1 over the union of all cycle-member IDs (partial credit
    for finding most of a cycle's claims).
    """
    pred_set = {frozenset(c) for c in predicted}
    gold_set = {frozenset(c) for c in gold}
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0,
                "claim_f1": 1.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    pred_ids = {i for c in predicted for i in c}
    gold_ids = {i for c in gold for i in c}
    ctp = len(pred_ids & gold_ids)
    cp = ctp / len(pred_ids) if pred_ids else 0.0
    cr = ctp / len(gold_ids) if gold_ids else 0.0
    claim_f1 = (2 * cp * cr / (cp + cr)) if (cp + cr) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "exact_match": float(pred_set == gold_set), "claim_f1": claim_f1}


# Lazy imports for backends
SamplingParams = None


def _import_vllm():
    global SamplingParams
    from vllm import SamplingParams as _SP
    SamplingParams = _SP
    from ctc_eval.lib.vllm_utils import add_vllm_args, load_model, run_inference
    return add_vllm_args, load_model, run_inference


# ---------------------------------------------------------------------------
# chunked-vllm helper: split the LoRA into a vLLM-compatible adapter + a
# sidecar holding the trained embedding rows.
# ---------------------------------------------------------------------------

def _prepare_stripped_lora(lora_path: str) -> tuple[str, dict[str, torch.Tensor]]:
    """Produce a copy of `lora_path` with `lm_head.weight` and
    `embed_tokens.weight` stripped out, plus a dict of those tensors so we
    can paste them into the live vLLM model.

    vLLM's LoRA loader (`parse_fine_tuned_lora_name` in `vllm/lora/utils.py`)
    rejects any non-`lora_A`/`lora_B` key. Chunked-grouping checkpoints save
    full embedding+lm_head weights via PEFT's `modules_to_save`, so the
    adapter as-shipped is unloadable. We split it: vLLM gets the lora_A/B
    factors only; we stash the full tensors and apply them to the model
    after vLLM has loaded the base. The trained tensor is shape (248079, H);
    the base's lm_head/embed are (248320, H) with the top 241 rows being
    untrained padding — overwriting rows [0:248079] reproduces the trained
    state without disturbing anything else.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import json
    import shutil

    src_dir = Path(lora_path).resolve()
    cache_dir = src_dir / ".stripped-for-vllm"
    cache_dir.mkdir(exist_ok=True)
    src_adapter = src_dir / "adapter_model.safetensors"
    out_adapter = cache_dir / "adapter_model.safetensors"
    out_config = cache_dir / "adapter_config.json"

    full_tensors: dict[str, torch.Tensor] = {}
    if (
        out_adapter.exists() and out_config.exists()
        and out_adapter.stat().st_mtime > src_adapter.stat().st_mtime
    ):
        # Cached. Re-read just the sidecar tensors we'll need at apply time.
        with safe_open(str(src_adapter), framework="pt") as h:
            for k in h.keys():
                if "lm_head.weight" in k or "embed_tokens.weight" in k:
                    if "lora_A" in k or "lora_B" in k:
                        continue
                    full_tensors[k] = h.get_tensor(k)
        return str(cache_dir), full_tensors

    # Read the source adapter, split into stripped + sidecar.
    stripped: dict[str, torch.Tensor] = {}
    with safe_open(str(src_adapter), framework="pt") as h:
        for k in h.keys():
            t = h.get_tensor(k)
            is_full_lm_head = (
                "lm_head.weight" in k or "embed_tokens.weight" in k
            ) and "lora_A" not in k and "lora_B" not in k
            if is_full_lm_head:
                full_tensors[k] = t
            else:
                stripped[k] = t

    # Write the stripped safetensors and copy the rest of the adapter dir.
    save_file(stripped, str(out_adapter))
    for name in os.listdir(src_dir):
        if name in {"adapter_model.safetensors", ".stripped-for-vllm",
                    ".merged-for-vllm"}:
            continue
        src = src_dir / name
        if src.is_dir():
            continue
        shutil.copy2(src, cache_dir / name)

    # Patch adapter_config.json to remove `lm_head` / `embed_tokens` from
    # `modules_to_save` so PEFT/vLLM don't expect them to be present.
    if out_config.exists():
        cfg = json.loads(out_config.read_text())
        mods_to_save = cfg.get("modules_to_save") or []
        cfg["modules_to_save"] = [
            m for m in mods_to_save
            if "lm_head" not in m and "embed_tokens" not in m
        ]
        out_config.write_text(json.dumps(cfg, indent=2))

    print(f"  chunked-vllm: stripped {len(full_tensors)} full-weight keys "
          f"from LoRA into sidecar; vLLM-ready adapter at {cache_dir}")
    return str(cache_dir), full_tensors


# ---------------------------------------------------------------------------
# HuggingFace model loading (chunked / standard backends)
# ---------------------------------------------------------------------------

def load_hf_model(args):
    """Load HuggingFace model for chunked or standard attention eval."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_chunked = args.backend.startswith("chunked")
    added_doc_tokens = False
    if is_chunked:
        # Resolve the document-boundary markers. Default to the HF/axolotl chunked
        # tokens (<|doc_start|>/<|doc_end|>), which are NEW tokens that get added +
        # the embedding table resized. olmo-core --wrap-docs models instead reuse
        # *existing* reserved tokens (e.g. <|box_start|>/<|box_end|>, ids already in
        # the converted Qwen3 vocab) — for those we must NOT re-add or resize, just
        # look up the ids the model already trained on.
        ds = getattr(args, "doc_start_token", None) or DOC_START
        de = getattr(args, "doc_end_token", None) or DOC_END
        vocab = tokenizer.get_vocab()
        if ds not in vocab or de not in vocab:
            tokenizer.add_special_tokens({"additional_special_tokens": [ds, de]})
            added_doc_tokens = True
        doc_start_id = tokenizer.convert_tokens_to_ids(ds)
        doc_end_id = tokenizer.convert_tokens_to_ids(de)
    else:
        doc_start_id, doc_end_id = None, None

    # chunked-flex: try HF's native flex_attention first so eval matches the
    # training-time attention path for architectures that support it (Qwen2.5,
    # Llama, etc.). Architectures that refuse native flex (notably the hybrid
    # Qwen3.5 with linear-attn layers) fall back to the custom `flex_chunked`
    # impl that delivers the BlockMask via a state dict — same FlexAttention
    # kernel, just plumbed differently. Mirrors train_chunked_fast.py.
    use_native_flex = False
    use_custom_flex = False
    if args.backend == "chunked-flex":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                attn_implementation="flex_attention",
                torch_dtype=torch.bfloat16,
            )
            use_native_flex = True
        except ValueError as e:
            if "flex_attention" not in str(e):
                raise
            use_custom_flex = True
    attn_impl = "sdpa"
    # Diagnostic override (see scripts/eval/diag_rightpad_divergence.py): lets
    # callers force "eager" to isolate bf16 SDPA kernel drift from real bugs.
    attn_impl = getattr(args, "attn_impl_override", None) or attn_impl
    if not use_native_flex:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
    if is_chunked:
        from ctc_eval.lib.olmo3_mask_patch import maybe_patch_olmo3_sliding_to_full
        maybe_patch_olmo3_sliding_to_full(model)
        # Only grow the embedding table if we actually added new marker tokens.
        # For olmo-core box markers (already in vocab) this is a no-op we must skip,
        # otherwise the model's trained rows for those ids would be left intact but
        # an unnecessary resize could shift/realloc the table.
        if added_doc_tokens:
            model.resize_token_embeddings(len(tokenizer))

    if args.lora_path:
        from peft import PeftModel
        from safetensors import safe_open
        from ctc_eval.lib.adapter_save import prepare_adapter_for_backend
        # Normalize adapter key format for the language-only CausalLM model:
        # strip any `language_model.` prefix left over from VL-wrapper training.
        lora_path = prepare_adapter_for_backend(args.lora_path, args.base_model, backend="hf")
        # Check if the LoRA adapter includes lm_head (from embedding resize during
        # training). If so, resize the base model embeddings to match before loading.
        adapter_file = Path(lora_path) / "adapter_model.safetensors"
        if adapter_file.exists():
            with safe_open(str(adapter_file), framework="pt") as f:
                for key in f.keys():
                    if "lm_head" in key:
                        lm_head_size = f.get_tensor(key).shape[0]
                        if lm_head_size != model.config.vocab_size:
                            model.resize_token_embeddings(lm_head_size)
                            print(f"  Resized embeddings to {lm_head_size} to match LoRA adapter")
                        break
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print(f"Loaded LoRA from {args.lora_path}")

    model = model.cuda().eval()
    if use_custom_flex:
        from ctc_eval.lib.chunked_attention import install_flex_chunked_attention
        flex_state = {}
        install_flex_chunked_attention(model, flex_state)
        model._flex_state = flex_state
    model._flex_native = use_native_flex
    return model, tokenizer, doc_start_id, doc_end_id


# ---------------------------------------------------------------------------
# HuggingFace generation (chunked / standard / flex)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_hf(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                max_new_tokens=20, stop_token_ids=None, backend="chunked-sdpa",
                attention_pattern: "AttentionPattern" = None):
    """Generate with HuggingFace: pattern-aware attention prefill, then greedy decode.

    The training-time attention pattern is rebuilt at inference via
    AttentionPattern. For non-chunked backends the pattern defaults to
    "standard" (plain causal) and the optimized triu path is used.
    """
    device = input_ids.device

    if backend == "chunked-flex":
        return _generate_flex(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                              max_new_tokens, stop_token_ids, attention_pattern)

    if attention_pattern is not None and attention_pattern.name == "hierarchical_anchor":
        return _generate_hierarchical(
            model, tokenizer, input_ids, doc_start_id, doc_end_id,
            max_new_tokens, stop_token_ids, attention_pattern,
        )

    # Build prefill mask
    if backend == "standard":
        seq_len = input_ids.size(1)
        dtype = torch.bfloat16
        mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1
        ).unsqueeze(0).unsqueeze(0).to(device)
    elif attention_pattern is not None and attention_pattern.name != "chunked":
        # chunked-sdpa with a non-default pattern — build via the factory.
        mask = _build_dense_mask_for_backend(
            input_ids, doc_start_id, doc_end_id, attention_pattern,
        ).to(device)
    else:
        # chunked-sdpa, default chunked pattern — use the legacy fast path.
        mask = build_chunked_causal_mask(
            input_ids.squeeze(0), doc_start_id, doc_end_id,
        ).to(device)

    outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    gen_ids = torch.cat(generated, dim=-1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def _ensure_hierarchical_installed(model, doc_start_id, doc_end_id,
                                   attention_pattern: "AttentionPattern"):
    """Install the hierarchical SDPA wrapper + pre-hook on `model` exactly
    once. Subsequent calls are no-ops. Stashes the per-batch state dict and
    the registered hierarchical fn on the model so generate-time can flip
    between hierarchical (prefill) and plain SDPA (decode).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if getattr(model, "_hierarchical_installed", False):
        return model._hierarchical_state, ALL_ATTENTION_FUNCTIONS

    num_layers = int(model.config.num_hidden_layers)
    state: dict = {}
    install_hierarchical_sdpa_attention(model, state)
    attach_hierarchical_pre_hook(
        model, state,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pattern=attention_pattern, num_transformer_layers=num_layers,
    )
    model._hierarchical_installed = True
    model._hierarchical_state = state
    model._hierarchical_fn = ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"]
    return state, ALL_ATTENTION_FUNCTIONS


@torch.no_grad()
def _generate_hierarchical(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                           max_new_tokens, stop_token_ids,
                           attention_pattern: "AttentionPattern"):
    """Generate with the per-layer hierarchical_anchor wrapper for prefill,
    then plain SDPA for KV-cache decoding.

    Why the swap: during decoding the new token is FREE (sees all of KV),
    which is exactly what plain SDPA does. Keeping the wrapper active would
    hit a (B, 1, 1, 1) vs (B, 1, 1, S_kv) mask/key shape mismatch, since the
    pre-hook only sees the length-1 input_ids and not the cached KV length.
    """
    _, ALL_FNS = _ensure_hierarchical_installed(
        model, doc_start_id, doc_end_id, attention_pattern,
    )
    plain_sdpa = ALL_FNS["sdpa"]
    hierarchical_fn = model._hierarchical_fn

    # Make sure prefill uses the hierarchical wrapper (a previous decode call
    # may have left the registry pointing at plain SDPA).
    ALL_FNS["sdpa_hierarchical"] = hierarchical_fn
    try:
        outputs = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # Decode without per-layer masks; new tokens are FREE.
        ALL_FNS["sdpa_hierarchical"] = plain_sdpa

        generated = [next_token]
        for _ in range(max_new_tokens - 1):
            outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if stop_token_ids and next_token.item() in stop_token_ids:
                break
    finally:
        # Restore the wrapper for the next prefill call.
        ALL_FNS["sdpa_hierarchical"] = hierarchical_fn

    gen_ids = torch.cat(generated, dim=-1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def _eval_chunk_ids(input_ids_1d, doc_start_id, doc_end_id, device):
    """Build (1, S) int32 chunk_ids for a single eval example."""
    S = input_ids_1d.shape[0]
    chunk_ids = torch.full((1, S), FREE_CHUNK_ID, dtype=torch.int32, device=device)
    spans = find_chunk_spans(input_ids_1d, doc_start_id, doc_end_id)
    for idx, (s, e) in enumerate(spans):
        chunk_ids[0, s:e] = idx
    return chunk_ids, len(spans)


def _build_dense_mask_for_backend(input_ids, doc_start_id, doc_end_id,
                                  pattern: "AttentionPattern"):
    """Build a dense bf16 attention mask for SDPA at eval time."""
    device = input_ids.device
    ids_1d = input_ids.squeeze(0)
    chunk_ids, n_docs = _eval_chunk_ids(ids_1d, doc_start_id, doc_end_id, device)
    kwargs = {}
    if pattern.needs_anchor_tensor():
        kwargs["is_anchor"] = (ids_1d == doc_end_id).unsqueeze(0)
    if pattern.needs_random_edges():
        max_docs = max(n_docs, 1)
        adj = build_random_doc_edges(
            num_docs=n_docs, num_edges=pattern.num_random_doc_edges,
            seed=pattern.random_seed, max_docs=max_docs,
        ).to(device)
        kwargs["doc_random"] = adj.unsqueeze(0)
    if pattern.needs_random_token_mask():
        rk = build_random_token_keep(
            seq_len=ids_1d.size(0),
            keep_prob=pattern.keep_prob,
            seed=pattern.random_seed,
        ).to(device)
        kwargs["random_keep"] = rk.unsqueeze(0)
    bool_mask = build_dense_bool_mask(pattern, chunk_ids, **kwargs)  # (1, S, S)
    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    mask = torch.where(
        bool_mask[0], torch.zeros(1, dtype=dtype, device=device),
        torch.full((1,), min_val, dtype=dtype, device=device),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


@torch.no_grad()
def _generate_flex(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                   max_new_tokens=20, stop_token_ids=None,
                   attention_pattern: "AttentionPattern" = None):
    """Generate with FlexAttention block-sparse pattern-aware prefill."""
    from torch.nn.attention.flex_attention import create_block_mask

    device = input_ids.device
    B, S = input_ids.shape

    pattern = attention_pattern or AttentionPattern(name="chunked")
    ids_1d = input_ids.squeeze(0)
    chunk_ids, n_docs = _eval_chunk_ids(ids_1d, doc_start_id, doc_end_id, device)

    kwargs = {}
    if pattern.needs_anchor_tensor():
        kwargs["is_anchor"] = (ids_1d == doc_end_id).unsqueeze(0)
    if pattern.needs_random_edges():
        max_docs = max(n_docs, 1)
        adj = build_random_doc_edges(
            num_docs=n_docs, num_edges=pattern.num_random_doc_edges,
            seed=pattern.random_seed, max_docs=max_docs,
        ).to(device)
        kwargs["doc_random"] = adj.unsqueeze(0)
    if pattern.needs_random_token_mask():
        rk = build_random_token_keep(
            seq_len=S,
            keep_prob=pattern.keep_prob,
            seed=pattern.random_seed,
        ).to(device)
        kwargs["random_keep"] = rk.unsqueeze(0)

    mask_mod = build_flex_mask_mod(pattern, chunk_ids, **kwargs)
    block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)
    # BlockMask delivery depends on which flex path the model uses. Native flex
    # consumes it via the standard `attention_mask` kwarg (same channel as
    # train_chunked_fast.py). The custom flex_chunked impl reads it from a
    # state dict instead. After prefill, both paths drop back to plain causal
    # for the autoregressive decode (each new token attends to all real KV).
    if getattr(model, "_flex_native", False):
        outputs = model(input_ids=input_ids, attention_mask=block_mask, use_cache=True)
    else:
        flex_state = getattr(model, "_flex_state", None)
        if flex_state is not None:
            flex_state["block_mask"] = block_mask
        outputs = model(input_ids=input_ids, attention_mask=None, use_cache=True)
        if flex_state is not None:
            flex_state.pop("block_mask", None)

    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    gen_ids = torch.cat(generated, dim=-1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _helmet_demo_block(demo_records, shots, exclude_query, qid):
    """Few-shot block for HELMET rerank, mirroring load_msmarco_rerank's `update`.

    Per-query deterministic shuffle (hash of qid), drop demos sharing the test
    query, dedup by query, take `shots`. Each demo = its passages + "\\n\\nQuery:
    {q}\\nRanking: {ids}" + "\\n\\n"; the block directly precedes the test context."""
    if shots <= 0 or not demo_records:
        return ""
    import hashlib
    pool = [d for d in demo_records if d.get("query") != exclude_query]
    h = abs(int(hashlib.sha256(str(qid).encode("utf-8")).hexdigest(), 16) % 2**31)
    rng = random.Random(h)
    rng.shuffle(pool)
    seen, chosen = set(), []
    for d in pool:
        q = d["query"]
        if q in seen:
            continue
        seen.add(q)
        chosen.append(d)
        if len(chosen) >= shots:
            break
    block = ""
    for d in chosen:
        ctxs = d["ctxs"]
        has_title = "title" in ctxs[0]
        passages = "\n\n".join(
            helmet_rerank_passage(c["id"], c["text"],
                                  c["title"] if has_title else None)
            for c in ctxs)
        # order by continuous CE score when present (else the bucketed label),
        # consistent with the rerank target (gold leads, not tied behind a neg).
        rank_key = "score" if "score" in ctxs[0] else "label"
        ranking = " > ".join(
            str(c["id"]) for c in sorted(ctxs, key=lambda c: c[rank_key],
                                         reverse=True))
        block += passages + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"
    return block


def load_unified_examples(path, max_samples, task, query_position="after",
                          use_titles=True, before_dummy=0, after_dummy=0,
                          use_alpaca=True, wrap_docs=False, output_top_k=-1,
                          shots=0, demo_path=""):
    """Load unified-format JSONL and build prompts."""
    examples = load_jsonl(path)
    if max_samples and len(examples) > max_samples:
        random.seed(42)
        examples = random.sample(examples, max_samples)

    # HELMET-native rerank records ({qid, query, ctxs}) — raw HELMET prompt,
    # optional few-shot demos from a HELMET-format demo file.
    if task == "rerank_helmet":
        demo_records = load_jsonl(demo_path) if (demo_path and shots > 0) else []
        result = []
        for ex in examples:
            ex["_demos"] = _helmet_demo_block(
                demo_records, shots, ex["query"], ex.get("qid", ex["query"]))
            prompt, output = build_prompt(ex, task=task, use_alpaca=False,
                                          output_top_k=output_top_k)
            if wrap_docs:
                prompt = wrap_documents(prompt)
            result.append({
                "prompt": prompt, "expected_output": output,
                "answers": [], "queries": [ex["query"]],
                "gold_doc_indices": [], "ex": ex,
            })
        return result

    result = []
    for ex in examples:
        prompt, output = build_prompt(
            ex, task=task, query_position=query_position,
            use_titles=use_titles, before_dummy=before_dummy,
            after_dummy=after_dummy, use_alpaca=use_alpaca,
            output_top_k=output_top_k,
        )
        if wrap_docs:
            prompt = wrap_documents(prompt)

        entry = {
            "prompt": prompt,
            "expected_output": output,
            "answers": ex.get("answers", []),
            "queries": ex.get("queries", []),
            "gold_doc_indices": ex.get("gold_doc_indices", []),
            # Passed through so task-specific eval functions can read
            # per-example fields (n_docs, source, level, gold_order, ...).
            "ex": ex,
        }
        result.append(entry)
    return result


HELMET_DATASET_CONFIG = {
    "nq": {
        "test_file": "data/data/kilt/nq-dev-multikilt_1000_k{num_docs}_dep6.jsonl",
        "demo_file": "data/data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
    },
    "triviaqa": {
        "test_file": "data/data/kilt/triviaqa-dev-multikilt_1000_k{num_docs}_dep6.jsonl",
        "demo_file": "data/data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
    },
    "hotpotqa": {
        "test_file": "data/data/kilt/hotpotqa-dev-multikilt_1000_k{num_docs}_dep3.jsonl",
        "demo_file": "data/data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
    },
    "popqa": {
        "test_file": "data/data/kilt/popqa_test_1000_k{num_docs}_dep6.jsonl",
        "demo_file": "data/data/kilt/popqa_test_1000_k3_dep6.jsonl",
    },
}


def _format_passage(ctx, no_titles=False):
    if no_titles:
        return PASSAGE_TEMPLATE_NO_TITLE.format(text=ctx["text"])
    return PASSAGE_TEMPLATE.format(**ctx)


def _build_demos(demo_data, sample, shots, no_titles=False):
    """Build few-shot demos for base model evaluation."""
    if shots == 0:
        return ""
    h = int(hashlib.sha256(str(sample["question"]).encode()).hexdigest(), 16) % 2**31
    rng = random.Random(h)
    demos = [d for d in demo_data if d.get("question") != sample.get("question")]
    rng.shuffle(demos)
    seen, unique = set(), []
    for d in demos:
        k = d.get("question", "")
        if k not in seen:
            seen.add(k)
            unique.append(d)
        if len(unique) >= shots:
            break
    texts = []
    for d in unique:
        docs = "\n\n".join(_format_passage(c, no_titles) for c in d.get("ctxs", []))
        ans = d["answers"][0] if isinstance(d["answers"], list) else d["answers"]
        texts.append(DEMO_TEMPLATE.format(documents=docs, question=d["question"], answer=ans))
    return "\n\n".join(texts) + "\n\n" if texts else ""


def load_helmet_examples(dataset_name, num_docs, max_samples, shots,
                         query_position="after", use_alpaca=True, no_titles=False,
                         before_dummy=0, after_dummy=0, wrap_docs=False):
    """Load HELMET/KILT eval data and build prompts."""
    config = HELMET_DATASET_CONFIG[dataset_name]

    if use_alpaca and shots > 0:
        print(f"  Note: overriding shots={shots} -> 0 for alpaca format")
        shots = 0

    # Find test file (fall back to other num_docs if exact match missing)
    search_docs = [num_docs] if num_docs > 0 else []
    search_docs += [500, 105, 100, 50, 20, 10, 3]
    test_file = None
    for nd in search_docs:
        candidate = config["test_file"].format(num_docs=nd)
        if Path(candidate).exists():
            if nd != num_docs:
                print(f"  Fallback: {candidate}")
            test_file = candidate
            break
    if test_file is None:
        raise FileNotFoundError(f"No test file for {dataset_name}")

    fmt_label = "alpaca" if use_alpaca else "helmet"
    print(f"  Loading: {test_file} (format={fmt_label}, titles={'no' if no_titles else 'yes'})")
    test_data = load_jsonl(test_file)
    demo_data = (load_jsonl(config["demo_file"])
                 if shots > 0 and num_docs > 0 and Path(config["demo_file"]).exists()
                 else [])

    if max_samples and len(test_data) > max_samples:
        key = "id" if "id" in test_data[0] else "question"
        seen, unique = set(), []
        for d in test_data:
            k = d.get(key, d["question"])
            if k not in seen:
                seen.add(k)
                unique.append(d)
        random.seed(42)
        test_data = random.sample(unique, min(max_samples, len(unique)))

    result = []
    for s in test_data:
        demos = ""
        context = ""
        if num_docs > 0:
            demos = _build_demos(demo_data, s, shots, no_titles=no_titles)
            context = "\n\n".join(_format_passage(c, no_titles) for c in s.get("ctxs", []))

        if use_alpaca:
            if num_docs == 0:
                input_text = f"Question: {s['question']}"
            else:
                if demos:
                    context = demos + context
                if query_position == "before":
                    input_text = f"Question: {s['question']}\n\n{context}"
                elif query_position == "both":
                    input_text = f"Question: {s['question']}\n\n{context}\n\nQuestion: {s['question']}"
                else:
                    input_text = f"{context}\n\nQuestion: {s['question']}"
            if before_dummy > 0 or after_dummy > 0:
                input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)
            prompt = format_alpaca_prompt(QA_INSTRUCTION, input_text)
        else:
            if num_docs == 0:
                prompt = HELMET_TEMPLATE.format(demos="", context="", question=s["question"]) + "\nAnswer:"
            else:
                if query_position == "before":
                    template = HELMET_TEMPLATE_QUERY_BEFORE
                elif query_position == "both":
                    template = HELMET_TEMPLATE_QUERY_BOTH
                else:
                    template = HELMET_TEMPLATE
                prompt = template.format(demos=demos, context=context, question=s["question"]) + "\nAnswer:"

        if wrap_docs:
            prompt = wrap_documents(prompt)
        result.append({
            "prompt": prompt,
            "expected_output": None,
            "answers": s["answers"],
            "queries": [s.get("question", "")],
            "gold_doc_indices": [],
        })
    return result


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_after_thinking(text):
    """Extract answer text after </think> tag, if present."""
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        first_line = answer.split('\n')[0].strip()
        return first_line if first_line else answer
    return None


def parse_output(output, prefix="Answer:"):
    """Extract answer after a prefix pattern."""
    patterns = [
        re.compile(f"(?:{prefix})(.*?)(?:\\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*?)(?:\n|$)"),
    ]
    for pat in patterns:
        match = pat.search(output)
        if match:
            result = re.sub(f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE).strip()
            if result:
                return result
    return None


def parse_retrieval_output(output):
    """Parse document IDs from model output, stripping thinking/prefixes.

    Uses rfind (last occurrence) so that CoT reasoning preceding the final
    'Relevant Document: [X]' line doesn't interfere with parsing.
    """
    text = output.strip()
    after_think = extract_after_thinking(text)
    if after_think:
        text = after_think
    for prefix in ["Relevant Documents:", "Relevant Document:", "relevant documents:",
                   "relevant document:"]:
        idx = text.rfind(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break
    text = text.split("\n")[0].strip()
    return parse_doc_ids(text)


def parse_multi_query_output(output, num_queries):
    """Parse per-query document IDs: 'Q1: [3], [7]; Q2: [1]; ...'"""
    text = output.strip()
    after_think = extract_after_thinking(text)
    if after_think:
        text = after_think
    for prefix in ["Relevant Documents:", "Relevant Document:"]:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break
    text = text.split("\n")[0].strip()
    parts = [p.strip() for p in text.split(";")]
    per_query_ids = []
    for part in parts:
        part = re.sub(r'^Q\d+:\s*', '', part)
        per_query_ids.append(parse_doc_ids(part))
    while len(per_query_ids) < num_queries:
        per_query_ids.append(set())
    return per_query_ids[:num_queries]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_qa_metrics(prediction, answers):
    """Compute QA metrics with multiple extraction strategies."""
    em = max_over_answers(exact_match, prediction, answers)
    sub_em = max_over_answers(substring_match, prediction, answers)
    f1 = max_over_answers(token_f1, prediction, answers)

    parsed = parse_output(prediction)
    if parsed:
        em = max(em, max_over_answers(exact_match, parsed, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, parsed, answers))
        f1 = max(f1, max_over_answers(token_f1, parsed, answers))

    after_think = extract_after_thinking(prediction)
    if after_think:
        em = max(em, max_over_answers(exact_match, after_think, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, after_think, answers))
        f1 = max(f1, max_over_answers(token_f1, after_think, answers))
        parsed_think = parse_output(after_think)
        if parsed_think:
            em = max(em, max_over_answers(exact_match, parsed_think, answers))
            sub_em = max(sub_em, max_over_answers(substring_match, parsed_think, answers))
            f1 = max(f1, max_over_answers(token_f1, parsed_think, answers))

    return {"exact_match": float(em), "substring_exact_match": float(sub_em), "f1": f1}


def compute_retrieval_metrics_single(prediction, gold_doc_indices):
    """Compute retrieval metrics for single-query examples."""
    predicted_ids = parse_retrieval_output(prediction)
    # Convert 0-indexed gold to 1-indexed (prompt uses 1-indexed doc IDs)
    gold_ids = set(g + 1 for g in gold_doc_indices)
    return {
        "exact_match": float(retrieval_exact_match(predicted_ids, gold_ids)),
        "recall": retrieval_recall(predicted_ids, gold_ids),
        "precision": retrieval_precision(predicted_ids, gold_ids),
        "f1": retrieval_f1(predicted_ids, gold_ids),
        "predicted_ids": sorted(predicted_ids),
        "gold_ids": sorted(gold_ids),
    }


def compute_retrieval_metrics_multi(prediction, gold_doc_indices, num_queries):
    """Compute retrieval metrics for multi-query examples."""
    per_query_predicted = parse_multi_query_output(prediction, num_queries)
    per_query_metrics = []
    for qi, (pred_ids, gold_indices) in enumerate(zip(per_query_predicted, gold_doc_indices)):
        gold_ids = set(g + 1 for g in gold_indices)
        per_query_metrics.append({
            "exact_match": float(retrieval_exact_match(pred_ids, gold_ids)),
            "recall": retrieval_recall(pred_ids, gold_ids),
            "precision": retrieval_precision(pred_ids, gold_ids),
            "f1": retrieval_f1(pred_ids, gold_ids),
        })
    n = len(per_query_metrics)
    agg = {k: sum(m[k] for m in per_query_metrics) / n
           for k in ["exact_match", "recall", "precision", "f1"]}
    agg["all_correct"] = float(all(m["exact_match"] == 1.0 for m in per_query_metrics))
    return agg, per_query_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation (any task × any backend)")

    # Model
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=50)

    # Task and backend
    parser.add_argument("--task", type=str, default="retrieval",
                        choices=["qa", "retrieval", "cot_retrieval", "contradiction",
                                 "qdmatch", "xabsence",
                                 "redundancy", "absence", "oolong", "rerank", "rerank_helmet",
                                 "summarization",
                                 "matching_ngram", "mathmatch", "strmatch", "cycle", "groups4",
                                 "textgroups",
                                 "outlier", "grouping", "grouping_labeled", "reorder",
                                 "ruler"])
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "chunked-vllm", "chunked-sdpa", "chunked-flex", "standard"],
                        help="Inference backend. chunked-vllm runs vLLM with the "
                             "FlexAttention backend patched to apply the chunked "
                             "(per-document) mask — same masking as chunked-flex "
                             "(HF) but on vLLM's paged-cache kernel.")
    parser.add_argument("--attention-pattern", type=str, default=None,
                        choices=[None, "standard", "chunked", "doc_window",
                                 "last_token_anchor", "token_window", "bigbird",
                                 "random_token", "hierarchical_anchor"],
                        help="Attention pattern for HF backends. Default: "
                             "match the backend (chunked-* -> 'chunked', "
                             "'standard' backend -> 'standard'). Set this to "
                             "match the pattern used at training time.")
    parser.add_argument("--doc-window-k", type=int, default=0)
    parser.add_argument("--token-window-w", type=int, default=0)
    parser.add_argument("--num-random-doc-edges", type=int, default=0)
    parser.add_argument("--keep-prob", type=float, default=1.0,
                        help="random_token: per-(q,k) Bernoulli keep prob.")
    parser.add_argument("--num-anchors", type=int, default=2,
                        help="hierarchical_anchor: anchor chunks per layer.")
    parser.add_argument("--stride-base", type=int, default=2,
                        help="hierarchical_anchor: anchor stride geometric base.")
    parser.add_argument("--pattern-seed", type=int, default=42)
    parser.add_argument("--doc-start-token", type=str, default=None,
                        help="Document-start marker the model trained with "
                             "(chunked backends). Default: <|doc_start|> (HF/axolotl "
                             "chunked path). olmo-core --wrap-docs models use existing "
                             "reserved tokens, e.g. <|box_start|> — pass them here so "
                             "eval scans the same ids the model saw (no embedding "
                             "resize for tokens already in the vocab).")
    parser.add_argument("--doc-end-token", type=str, default=None,
                        help="Document-end marker (pairs with --doc-start-token), "
                             "e.g. <|box_end|> for olmo-core chunked models.")

    # Data sources
    parser.add_argument("--eval-data", type=str, default="",
                        help="Unified-format JSONL file")
    parser.add_argument("--datasets", type=str, default="",
                        help="HELMET datasets (e.g. nq,hotpotqa) — QA task only")
    parser.add_argument("--num-docs", type=int, default=20)
    parser.add_argument("--shots", type=int, default=2,
                        help="Few-shot demos for HELMET base model eval")
    parser.add_argument("--demo-data", type=str, default="",
                        help="rerank_helmet: HELMET-format demo jsonl for few-shot "
                             "demos (mirrors HELMET's demo_files; --shots controls "
                             "count). Demos with the same query as a test example "
                             "are skipped.")

    # Formatting
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"])
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--use-alpaca", action="store_true",
                        help="Force alpaca format (for full FT models without --lora-path)")
    parser.add_argument("--before-dummy", type=int, default=0)
    parser.add_argument("--after-dummy", type=int, default=0)

    # Generation
    parser.add_argument("--output-top-k", type=int, default=-1,
                        help="Truncate the rerank target to the top-K ids. For "
                             "rerank: -1 (default) ranks the whole pool. For "
                             "rerank_helmet: -1 falls back to the top-10 (the "
                             "NDCG@10 cutoff and ~all that fits HELMET's 200-token "
                             "budget); set a positive K to override. NDCG@10 is "
                             "reported regardless.")
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output-file", type=str, default="outputs/eval_results/eval_results.json")

    # vLLM-specific (only used when backend=vllm)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--longlora-eval-group-size", type=int, default=0,
                        help="If >0 (HF backends only), install LongLoRA S^2-Attn "
                             "at eval with force_eval=True. Used to study eval-time "
                             "effect of the training swap; paper uses plain SDPA at eval.")

    parser.add_argument(
        "--eval-batch-size", type=int, default=8,
        help="Batched eval for chunked-* backends (serial prefill + batched "
             "decode, ctc_eval.eval.batched_chunked_serial_prefill). Verified "
             "bit-identical F1 to single-example on Qwen3.5-0.8B at bs=8 "
             "(2.19x speedup). Set to 1 to force the legacy single-example "
             "path. Ignored for vllm/standard backends.",
    )

    args = parser.parse_args()

    if not args.datasets and not args.eval_data:
        parser.error("Specify --eval-data (unified JSONL) or --datasets (HELMET)")

    use_alpaca = bool(args.lora_path) or args.use_alpaca
    assert use_alpaca, (
        "Default helmet (2-shot) eval format is disabled — training uses alpaca "
        "0-shot. Pass --use-alpaca (or --lora-path) to make the eval format match "
        "training. If you really want helmet, remove this assert."
    )
    is_vllm = args.backend in ("vllm", "chunked-vllm")
    is_hf = not is_vllm
    is_chunked_vllm = args.backend == "chunked-vllm"
    wrap_docs = (is_hf and args.backend.startswith("chunked")) or is_chunked_vllm

    # Resolve the attention pattern for HF backends. Defaults track the
    # backend so existing eval commands keep working unchanged.
    if args.attention_pattern is None:
        pattern_name = "standard" if args.backend == "standard" else "chunked"
    else:
        pattern_name = args.attention_pattern
    eval_attention_pattern = AttentionPattern(
        name=pattern_name,
        doc_window_k=args.doc_window_k,
        token_window_w=args.token_window_w,
        num_random_doc_edges=args.num_random_doc_edges,
        keep_prob=args.keep_prob,
        num_anchors=args.num_anchors,
        stride_base=args.stride_base,
        random_seed=args.pattern_seed,
    )
    # If the pattern is non-chunked (e.g. doc_window), we still need doc
    # boundary tokens wrapped in the prompt so chunk_ids can be derived.
    if pattern_name != "standard" and is_hf:
        wrap_docs = True

    if use_alpaca and args.shots != 0:
        print(f"  Auto-setting shots=0 for trained model (training data has no demos)")
        args.shots = 0

    if args.task in ("contradiction", "qdmatch", "xabsence", "redundancy", "absence") and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  {args.task}: increased max_tokens to {args.max_tokens}")

    if args.task == "matching_ngram" and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  Matching n-gram: increased max_tokens to {args.max_tokens}")

    if args.task == "mathmatch" and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  Mathmatch: increased max_tokens to {args.max_tokens}")

    if args.task == "strmatch" and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  Strmatch: increased max_tokens to {args.max_tokens}")

    if args.task == "cycle" and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  Cycle: increased max_tokens to {args.max_tokens}")

    if args.task == "oolong" and args.max_tokens <= 50:
        args.max_tokens = 256
        print(f"  Oolong: increased max_tokens to {args.max_tokens}")

    if args.task == "rerank" and args.max_tokens <= 50:
        args.max_tokens = 512  # ranked list of many IDs
        print(f"  Rerank: increased max_tokens to {args.max_tokens}")

    if args.task == "rerank_helmet" and args.max_tokens <= 50:
        args.max_tokens = 200  # HELMET generation_max_length (stop-on-newline)
        print(f"  Rerank (HELMET): set max_tokens to {args.max_tokens}")

    if args.task == "summarization" and args.max_tokens <= 50:
        args.max_tokens = 1024  # long-form summary
        print(f"  Summarization: increased max_tokens to {args.max_tokens}")

    if args.task == "groups4" and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  Groups4: increased max_tokens to {args.max_tokens}")

    if args.task == "textgroups" and args.max_tokens <= 50:
        args.max_tokens = 200
        print(f"  Textgroups: increased max_tokens to {args.max_tokens}")

    if args.task == "cot_retrieval" and args.max_tokens <= 50:
        args.max_tokens = 512
        print(f"  CoT retrieval: increased max_tokens to {args.max_tokens}")

    if args.task == "outlier" and args.max_tokens <= 50:
        args.max_tokens = 256
        print(f"  Outlier: increased max_tokens to {args.max_tokens}")

    if args.task in ("grouping", "grouping_labeled") and args.max_tokens <= 50:
        args.max_tokens = 2048
        print(f"  Grouping: increased max_tokens to {args.max_tokens}")

    if args.task == "reorder" and args.max_tokens <= 50:
        args.max_tokens = 1024
        print(f"  Reorder: increased max_tokens to {args.max_tokens}")

    if args.task == "ruler" and args.max_tokens <= 50:
        # multi-value/multi-query/vt answers list several strings.
        args.max_tokens = 128
        print(f"  RULER: increased max_tokens to {args.max_tokens}")

    if args.enable_thinking and args.max_tokens <= 50:
        args.max_tokens = 512
        print(f"  Thinking mode: increased max_tokens to {args.max_tokens}")

    print(f"Task: {args.task} | Backend: {args.backend} | "
          f"Format: {'alpaca' if use_alpaca else 'helmet'} | Shots: {args.shots}")

    # --- Load model ---
    if is_hf:
        model, tokenizer, doc_start_id, doc_end_id = load_hf_model(args)
        device = next(model.parameters()).device
        if args.longlora_eval_group_size > 0:
            from ctc_eval.lib.longlora_attn import install_s2_attn
            install_s2_attn(model, args.longlora_eval_group_size, force_eval=True)
            print(f"  Installed S^2-Attn at eval (group_size={args.longlora_eval_group_size}, force_eval=True)")
        newline_id = tokenizer.encode("\n", add_special_tokens=False)
        multiline_output = args.enable_thinking or args.task in (
            "cot_retrieval", "contradiction", "outlier",
            "grouping", "grouping_labeled", "reorder",
        )
        if multiline_output:
            stop_ids = {tokenizer.eos_token_id}
        else:
            stop_ids = {tokenizer.eos_token_id} | set(newline_id)
    else:
        # chunked-vllm: route the chunked mask through vLLM's FlexAttention
        # backend. Three things have to happen before LLM(...) constructs:
        #   1. attention_config={"backend": "FLEX_ATTENTION"} (force selection;
        #      vLLM's auto-pick on Hopper would otherwise be flash-attn).
        #      Pulled in via args.force_flex_attention -> vllm_utils.load_model.
        #   2. VLLM_ENABLE_V1_MULTIPROCESSING=0 so our in-process monkey-patch
        #      reaches the actual model worker (otherwise vLLM spawns a
        #      subprocess that never imported the patch).
        #   3. Install the patch and register the doc-token IDs so the patched
        #      FlexAttention metadata builder can derive chunk_ids on the fly.
        if is_chunked_vllm:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            args.force_flex_attention = True
            # FlexAttention + cudagraph capture is not supported by our patched
            # block-mask path (we rebuild block_mask each step from per-batch
            # chunk_ids). Force eager.
            args.enforce_eager = True
            from ctc_eval.lib import vllm_chunked_patch
            vllm_chunked_patch.install()
            from transformers import AutoTokenizer
            chunked_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
            doc_start_id, doc_end_id = setup_tokenizer(chunked_tokenizer)
            vllm_chunked_patch.set_doc_token_ids(doc_start_id, doc_end_id)
            # Chunked-grouping LoRAs include full lm_head/embed_tokens
            # weights (PEFT's `modules_to_save`, since training extended the
            # tokenizer). vLLM's LoRA loader rejects any non-lora_A/B key, so
            # we strip those two tensors out of a copy of the adapter and
            # re-install them onto the live model after vLLM finishes loading
            # the base. The remaining lora_A/B factors flow through vLLM's
            # standard LoRA path.
            if args.lora_path:
                stripped_path, full_extras = _prepare_stripped_lora(
                    args.lora_path,
                )
                vllm_chunked_patch.set_full_extras(full_extras)
                args.lora_path = stripped_path
        else:
            chunked_tokenizer = None
            doc_start_id = doc_end_id = None

        _import_vllm()
        from ctc_eval.lib.vllm_utils import load_model as vllm_load_model, run_inference
        llm, lora_request = vllm_load_model(args)
        multiline_output = args.enable_thinking or args.task in (
            "cot_retrieval", "contradiction", "outlier",
            "grouping", "grouping_labeled", "reorder",
        )
        if multiline_output:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        else:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n"])

    # --- Load data ---
    eval_sources = []
    if args.eval_data:
        eval_sources.append(("unified", args.eval_data))
    if args.datasets:
        for ds in args.datasets.split(","):
            eval_sources.append(("helmet", ds.strip()))

    all_results = {}
    for source_type, source in eval_sources:
        label = source if source_type == "helmet" else Path(source).stem
        print(f"\n{'='*60}\nEvaluating: {label} ({args.backend}, {args.task})\n{'='*60}")

        if source_type == "helmet":
            examples = load_helmet_examples(
                source, args.num_docs, args.max_test_samples, args.shots,
                query_position=args.query_position, use_alpaca=use_alpaca,
                no_titles=args.no_titles,
                before_dummy=args.before_dummy, after_dummy=args.after_dummy,
                wrap_docs=wrap_docs,
            )
        else:
            examples = load_unified_examples(
                source, args.max_test_samples, task=args.task,
                query_position=args.query_position,
                use_titles=not args.no_titles,
                before_dummy=args.before_dummy, after_dummy=args.after_dummy,
                use_alpaca=use_alpaca, wrap_docs=wrap_docs,
                output_top_k=args.output_top_k,
                shots=args.shots, demo_path=args.demo_data,
            )

        print(f"  {len(examples)} examples")

        # --- Run inference ---
        if is_hf:
            use_batched = (
                args.eval_batch_size > 1 and args.backend.startswith("chunked")
                # chunked-flex's BlockMask flows through a state-dict shared
                # with the custom flex_chunked attn impl; the batched path
                # builds dense 4D masks instead, so force per-example.
                and args.backend != "chunked-flex"
            )
            if use_batched:
                from ctc_eval.eval.batched_chunked_serial_prefill import (
                    generate_hf_batched_serial_prefill,
                )
                prompts = [
                    ex["prompt"] + ("<think>\n" if args.enable_thinking else "")
                    for ex in examples
                ]
                responses = generate_hf_batched_serial_prefill(
                    model, tokenizer, prompts,
                    doc_start_id=doc_start_id, doc_end_id=doc_end_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=args.max_tokens,
                    stop_token_ids=stop_ids,
                    attention_pattern=eval_attention_pattern,
                    batch_size=args.eval_batch_size,
                )
            else:
                responses = []
                for ex in tqdm(examples, desc=f"  {label}"):
                    prompt = ex["prompt"]
                    if args.enable_thinking:
                        prompt = prompt + "<think>\n"
                    input_ids = tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                    ).input_ids.to(device)
                    response = generate_hf(
                        model, tokenizer, input_ids, doc_start_id, doc_end_id,
                        max_new_tokens=args.max_tokens, stop_token_ids=stop_ids,
                        backend=args.backend,
                        attention_pattern=eval_attention_pattern,
                    )
                    responses.append(response)
        else:
            prompts = [ex["prompt"] for ex in examples]
            if args.enable_thinking:
                prompts = [p + "<think>\n" for p in prompts]
            print(f"  Running vLLM inference...")
            if is_chunked_vllm:
                # Pre-tokenize so <|doc_start|> / <|doc_end|> become the right
                # token IDs (and so the patched FlexAttention backend can
                # locate them). vLLM accepts prompt_token_ids directly.
                from vllm import TokensPrompt
                prompt_token_ids = [
                    chunked_tokenizer(p, truncation=True,
                                      max_length=args.max_model_len).input_ids
                    for p in prompts
                ]
                inputs = [TokensPrompt(prompt_token_ids=t) for t in prompt_token_ids]
                outputs = llm.generate(inputs, sampling_params, lora_request=lora_request)
                responses = [o.outputs[0].text for o in outputs]
            else:
                responses = run_inference(llm, prompts, sampling_params, lora_request)

        # --- Compute metrics ---
        if args.task == "contradiction":
            results, details = _eval_contradiction(examples, responses)
        elif args.task == "qdmatch":
            results, details = _eval_qdmatch(examples, responses)
        elif args.task == "redundancy":
            results, details = _eval_contradiction(examples, responses)
        elif args.task in ("absence", "xabsence"):
            results, details = _eval_absence(examples, responses)
        elif args.task == "oolong":
            results, details = _eval_oolong(examples, responses)
        elif args.task == "rerank":
            results, details = _eval_rerank(examples, responses)
        elif args.task == "rerank_helmet":
            results, details = _eval_rerank_helmet(examples, responses)
        elif args.task == "summarization":
            results, details = _eval_summarization(examples, responses)
        elif args.task == "matching_ngram":
            results, details = _eval_contradiction(examples, responses)
        elif args.task == "mathmatch":
            results, details = _eval_contradiction(examples, responses)
        elif args.task == "strmatch":
            results, details = _eval_contradiction(examples, responses)
        elif args.task == "cycle":
            results, details = _eval_cycle(examples, responses)
        elif args.task == "groups4":
            results, details = _eval_cycle(examples, responses)
        elif args.task == "textgroups":
            results, details = _eval_cycle(examples, responses)
        elif args.task in ("retrieval", "cot_retrieval"):
            results, details = _eval_retrieval(examples, responses)
        elif args.task == "outlier":
            results, details = _eval_outlier(examples, responses)
        elif args.task in ("grouping", "grouping_labeled"):
            results, details = _eval_grouping(examples, responses)
        elif args.task == "reorder":
            results, details = _eval_reorder(examples, responses)
        elif args.task == "ruler":
            results, details = _eval_ruler(examples, responses)
        else:
            results, details = _eval_qa(examples, responses)

        all_results[label] = {"metrics": results, "details": details}

        # Print summary
        print(f"\n  Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.1%}")

        # Show samples
        for d in details[:3]:
            print(f"    Pred: {d.get('prediction', '')[:80]}")

    save_results(args.output_file, {"args": vars(args), "results": all_results})
    print(f"\nResults saved to {args.output_file}")


def _eval_qa(examples, responses):
    """Evaluate QA task: compute EM, SubEM, F1."""
    results_list = []
    details = []
    for ex, resp in zip(examples, responses):
        m = compute_qa_metrics(resp, ex["answers"])
        results_list.append(m)
        details.append({"prediction": resp.strip()[:500], **m})

    metrics = aggregate(results_list, ["exact_match", "substring_exact_match", "f1"])
    return metrics, details


def _eval_ruler(examples, responses):
    """Evaluate RULER: recall-based string match (RULER's `string_match_all`).

    For each gold string, check whether it appears (case-insensitively) anywhere
    in the prediction. `recall` = fraction of gold strings found; `all_correct`
    = 1.0 iff every gold string is present (RULER's headline metric). The answer
    region after a final "Answer:" is preferred, falling back to the whole
    generation so partial-format outputs are still credited.
    """
    results_list, details = [], []
    for ex, resp in zip(examples, responses):
        gold = [str(a) for a in ex.get("answers", [])]
        text = resp.strip()
        after_think = extract_after_thinking(text)
        if after_think:
            text = after_think
        idx = text.rfind("Answer:")
        ans_region = text[idx + len("Answer:"):] if idx >= 0 else text
        hay = ans_region.lower()
        # Fall back to the full response if the formatted region misses a gold
        # string that is in fact present elsewhere in the output.
        full = text.lower()
        hits = sum(1 for g in gold if g.lower() in hay or g.lower() in full)
        recall = hits / len(gold) if gold else 1.0
        m = {"recall": recall, "all_correct": float(hits == len(gold))}
        results_list.append(m)
        details.append({"prediction": resp.strip()[:500],
                        "gold": gold, **m})

    metrics = aggregate(results_list, ["recall", "all_correct"])
    return metrics, details


def _eval_retrieval(examples, responses):
    """Evaluate retrieval task: compute retrieval EM, recall, precision, F1."""
    results_list = []
    details = []

    for ex, resp in zip(examples, responses):
        gold = ex["gold_doc_indices"]
        is_multi = len(ex["queries"]) > 1

        if is_multi:
            agg, per_query = compute_retrieval_metrics_multi(resp, gold, len(ex["queries"]))
            results_list.append(agg)
            details.append({"prediction": resp.strip()[:500], **agg})
        else:
            # Flatten gold indices for single-query
            flat_gold = gold[0] if gold and isinstance(gold[0], list) else gold
            m = compute_retrieval_metrics_single(resp, flat_gold)
            results_list.append({k: m[k] for k in ["exact_match", "recall", "precision", "f1"]})
            details.append({"prediction": resp.strip()[:500], **m})

    metric_keys = ["exact_match", "recall", "precision", "f1"]
    # Add all_correct if multi-query
    if any("all_correct" in r for r in results_list):
        metric_keys.append("all_correct")
    metrics = aggregate(results_list, metric_keys)
    return metrics, details


def _eval_outlier(examples, responses):
    """Evaluate outlier task: set-based P/R/F1/EM + per-source aggregation."""
    from collections import defaultdict
    METRIC_KEYS = ["precision", "recall", "f1", "exact_match"]
    per_src = defaultdict(list)
    details, parse_failures = [], 0
    for entry, resp in zip(examples, responses):
        ex = entry["ex"]
        gold = set(int(g) + 1 for g in ex["gold_doc_indices"])
        n = len(ex["documents"])
        pred_ids = _parse_outlier_ids(resp, n)
        parsed = pred_ids is not None
        if not parsed:
            parse_failures += 1
            pred_set = set()
            m = {k: 0.0 for k in METRIC_KEYS}
        else:
            pred_set = set(pred_ids)
            tp = len(pred_set & gold)
            p = tp / len(pred_set) if pred_set else 0.0
            r = tp / len(gold) if gold else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            m = {"precision": p, "recall": r, "f1": f1,
                 "exact_match": float(pred_set == gold)}
        rec = {
            "source": ex.get("source", "unknown"),
            "n_docs": n, "n_outliers": len(gold), "parsed": parsed,
            "prediction": resp.strip()[:500],
            "pred": sorted(pred_set), "gold": sorted(gold),
            **m,
        }
        details.append(rec)
        per_src[rec["source"]].append(rec)

    n = len(details) or 1
    metrics = {k: sum(r[k] for r in details) / n for k in METRIC_KEYS}
    metrics["parse_rate"] = (n - parse_failures) / n
    metrics["per_source"] = {
        s: {k: sum(r[k] for r in rs) / len(rs) for k in METRIC_KEYS}
        for s, rs in per_src.items()
    }
    return metrics, details


def _eval_grouping(examples, responses):
    """Evaluate grouping: ARI/NMI/pairwise P/R/F1/k_exact/coverage/EM + per-level."""
    from collections import Counter, defaultdict
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    METRIC_KEYS = ["ari", "nmi", "pairwise_precision", "pairwise_recall",
                   "pairwise_f1", "k_exact", "coverage", "exact_match"]
    per_level = defaultdict(list)
    details, parse_failures = [], 0

    for entry, resp in zip(examples, responses):
        ex = entry["ex"]
        n = len(ex["documents"])
        k_gold = ex["k"]
        gold_clusters = ex["gold_doc_indices"]
        pred = _parse_partition(resp, n)
        parsed = pred is not None
        if not parsed:
            parse_failures += 1
            pred = []
        pred_labels = _partition_to_labels(pred, n)
        gold_clusters_1 = [[i + 1 for i in c] for c in gold_clusters]
        gold_labels = _partition_to_labels(gold_clusters_1, n)

        pred_pairs = {(i, j) for i, j in combinations(range(n), 2)
                      if pred_labels[i] == pred_labels[j]}
        gold_pairs = {(i, j) for i, j in combinations(range(n), 2)
                      if gold_labels[i] == gold_labels[j]}
        if not pred_pairs and not gold_pairs:
            p, r, f = 1.0, 1.0, 1.0
        else:
            tp = len(pred_pairs & gold_pairs)
            p = tp / len(pred_pairs) if pred_pairs else 0.0
            r = tp / len(gold_pairs) if gold_pairs else 0.0
            f = (2 * p * r / (p + r)) if (p + r) else 0.0

        seen = Counter()
        for g in pred:
            for d in g:
                seen[d] += 1
        coverage = sum(1 for i in range(1, n + 1) if seen[i] == 1) / n

        def canon(cs): return frozenset(frozenset(c) for c in cs)

        m = {
            "ari": float(adjusted_rand_score(gold_labels, pred_labels)),
            "nmi": float(normalized_mutual_info_score(gold_labels, pred_labels)),
            "pairwise_precision": p,
            "pairwise_recall": r,
            "pairwise_f1": f,
            "k_exact": float(len(pred) == k_gold),
            "coverage": coverage,
            "exact_match": float(canon(pred) == canon(gold_clusters_1)),
        }
        rec = {
            "level": ex.get("level"), "k_gold": k_gold, "k_pred": len(pred),
            "parsed": parsed, "prediction": resp.strip()[:500], **m,
        }
        details.append(rec)
        per_level[rec["level"]].append(rec)

    n = len(details) or 1
    metrics = {k: sum(r[k] for r in details) / n for k in METRIC_KEYS}
    metrics["parse_rate"] = (n - parse_failures) / n
    metrics["per_level"] = {
        str(l): {k: sum(r[k] for r in rs) / len(rs) for k in METRIC_KEYS}
        for l, rs in per_level.items()
    }
    return metrics, details


def _reorder_lcs(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def _eval_reorder(examples, responses):
    """Evaluate reorder: kendall tau / spearman / PMR / pos-acc / pair-acc / LCS."""
    from collections import defaultdict
    from itertools import combinations
    from scipy.stats import kendalltau, spearmanr

    METRIC_KEYS = ["kendall_tau", "spearman_rho", "pmr", "position_accuracy",
                   "pairwise_accuracy", "first_last_accuracy", "lcs_norm"]

    def _bin_n(n):
        if n <= 30: return "short"
        if n <= 70: return "medium"
        return "long"

    per_src = defaultdict(list)
    per_bin = defaultdict(list)
    details, parse_failures = [], 0

    for entry, resp in zip(examples, responses):
        ex = entry["ex"]
        gold = ex["gold_order"]
        n = len(gold)
        pred = _parse_permutation(resp, n)
        parsed = pred is not None
        if not parsed:
            parse_failures += 1
            m = {k: 0.0 for k in METRIC_KEYS}
        else:
            pred_rank = [0] * (n + 1)
            gold_rank = [0] * (n + 1)
            for pos, i in enumerate(pred, 1):
                pred_rank[i] = pos
            for pos, i in enumerate(gold, 1):
                gold_rank[i] = pos
            pr = pred_rank[1:]
            gr = gold_rank[1:]
            tau = kendalltau(pr, gr).statistic
            rho = spearmanr(pr, gr).statistic
            total = n * (n - 1) // 2
            concordant = sum(
                1 for i, j in combinations(range(n), 2)
                if pred_rank[gold[i]] < pred_rank[gold[j]]
            )
            m = {
                "kendall_tau": float(tau) if tau == tau else 0.0,
                "spearman_rho": float(rho) if rho == rho else 0.0,
                "pmr": float(pred == gold),
                "position_accuracy": sum(1 for p, g in zip(pred, gold) if p == g) / n,
                "pairwise_accuracy": concordant / total if total else 1.0,
                "first_last_accuracy": float(pred[0] == gold[0] and pred[-1] == gold[-1]),
                "lcs_norm": _reorder_lcs(pred, gold) / n,
            }
        rec = {
            "source_type": ex.get("source_type", "unknown"),
            "n_chunks": n, "parsed": parsed,
            "prediction": resp.strip()[:500], **m,
        }
        details.append(rec)
        per_src[rec["source_type"]].append(rec)
        per_bin[_bin_n(n)].append(rec)

    n_total = len(details) or 1
    metrics = {k: sum(r[k] for r in details) / n_total for k in METRIC_KEYS}
    metrics["parse_rate"] = (n_total - parse_failures) / n_total
    metrics["per_source"] = {
        s: {k: sum(r[k] for r in rs) / len(rs) for k in METRIC_KEYS}
        for s, rs in per_src.items()
    }
    metrics["per_n_bin"] = {
        b: {k: sum(r[k] for r in rs) / len(rs) for k in METRIC_KEYS}
        for b, rs in per_bin.items()
    }
    return metrics, details


def _eval_qdmatch(examples, responses):
    """Evaluate qdmatch: ordered (query_id, doc_id) pair P/R/F1/EM.

    Gold lives in the raw example under `gold_pairs` (1-based, ordered). Pairs
    are matched as ordered tuples (query first) — pair_metrics sets via tuple(p)
    so order is preserved (no sorting, unlike contradiction)."""
    results_list = []
    details = []
    parse_failures = 0
    for ex, resp in zip(examples, responses):
        gold = ex.get("ex", {}).get("gold_pairs", [])
        predicted = parse_qd_pairs(resp)
        if predicted is None:
            parse_failures += 1
            predicted = []
        m = pair_metrics(predicted, gold)
        results_list.append(m)
        details.append({
            "prediction": resp.strip()[:500],
            "gold_pairs": gold,
            "predicted_pairs": predicted,
            **m,
        })
    n = len(results_list)
    metrics = {k: sum(r[k] for r in results_list) / n
               for k in ["precision", "recall", "f1", "exact_match"]}
    metrics["parse_rate"] = (n - parse_failures) / n
    return metrics, details


def _eval_contradiction(examples, responses):
    """Evaluate contradiction task: compute pair-level precision/recall/F1/EM."""
    results_list = []
    details = []
    parse_failures = 0

    for ex, resp in zip(examples, responses):
        gold = ex["gold_doc_indices"]  # list of [a, b] pairs (1-indexed)
        predicted = parse_pairs(resp)
        if predicted is None:
            parse_failures += 1
            predicted = []
        m = pair_metrics(predicted, gold)
        results_list.append(m)
        details.append({
            "prediction": resp.strip()[:500],
            "gold_pairs": gold,
            "predicted_pairs": predicted,
            **m,
        })

    n = len(results_list)
    metrics = {k: sum(r[k] for r in results_list) / n
               for k in ["precision", "recall", "f1", "exact_match"]}
    metrics["parse_rate"] = (n - parse_failures) / n
    return metrics, details


def _parse_id_set(text, n):
    """Extract a set of 1-indexed IDs in [1, n] from text like 'Missing: [3], [7]'
    (absence) or 'Unmatched: [3], [7]' (xabsence). If a CoT precedes the answer,
    read only the final anchor line."""
    for anchor in ("Missing:", "Unmatched:"):
        if anchor in text:
            text = text.rsplit(anchor, 1)[1]
            break
    ids = re.findall(r'\[(\d+)\]', text)
    if not ids:
        ids = re.findall(r'\b(\d+)\b', text)
    out = {int(x) for x in ids if 1 <= int(x) <= n}
    return out if (ids or text.strip() in ("", "Missing:", "[]")) else None


def _norm_snippet(s):
    """Normalize a first-four-words snippet for matching: lowercase, strip
    surrounding punctuation/quotes, collapse whitespace, keep first 4 tokens."""
    s = re.sub(r"\s+", " ", str(s)).strip().strip("\"'").lower()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return " ".join(s.split()[:4])


def _parse_snippet_list(text):
    """Extract the model's JSON list of first-four-words snippets. Falls back to
    quote-delimited extraction if json.loads fails."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
        quoted = re.findall(r'"([^"]*)"', m.group(0))
        if quoted:
            return quoted
    return None


def _eval_absence_textdiff(examples, responses):
    """Gutenberg text-diff absence: set precision/recall/F1/EM over the
    first-four-words snippets of the removed sentences (order-insensitive)."""
    keys = ["precision", "recall", "f1", "exact_match"]
    details, parse_failures = [], 0
    for ex, resp in zip(examples, responses):
        raw = ex.get("ex", ex)
        gold = {_norm_snippet(s) for s in raw["answers"]}
        pred_list = _parse_snippet_list(resp)
        if pred_list is None:
            parse_failures += 1
            m = {k: 0.0 for k in keys}
            pred = set()
        else:
            pred = {_norm_snippet(s) for s in pred_list if _norm_snippet(s)}
            tp = len(pred & gold)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(gold) if gold else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            m = {"precision": p, "recall": r, "f1": f1,
                 "exact_match": float(pred == gold)}
        details.append({"prediction": resp.strip()[:500],
                        "gold": sorted(gold), "pred": sorted(pred),
                        "source": ex.get("source", "unknown"), **m})
    n = len(details) or 1
    metrics = {k: sum(d[k] for d in details) / n for k in keys}
    metrics["parse_rate"] = (n - parse_failures) / n
    return metrics, details


def _eval_absence(examples, responses):
    """AbsenceBench: set precision/recall/F1/EM over the removed-item IDs
    (equivalent to the paper's micro-F1 over removed elements)."""
    if examples:
        first = examples[0].get("ex", examples[0])
        if (first.get("meta") or {}).get("format") == "textdiff":
            return _eval_absence_textdiff(examples, responses)
    keys = ["precision", "recall", "f1", "exact_match"]
    details, parse_failures = [], 0
    for ex, resp in zip(examples, responses):
        raw = ex.get("ex", ex)  # eval wraps examples; documents live under "ex"
        gold = {int(g) + 1 for g in raw["gold_doc_indices"]}
        n = len(raw["documents"])
        pred = _parse_id_set(resp, n)
        if pred is None:
            parse_failures += 1
            m = {k: 0.0 for k in keys}
            pred = set()
        else:
            tp = len(pred & gold)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(gold) if gold else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            m = {"precision": p, "recall": r, "f1": f1,
                 "exact_match": float(pred == gold)}
        details.append({"prediction": resp.strip()[:500],
                        "gold": sorted(gold), "pred": sorted(pred),
                        "source": ex.get("source", "unknown"), **m})
    n = len(details) or 1
    metrics = {k: sum(d[k] for d in details) / n for k in keys}
    metrics["parse_rate"] = (n - parse_failures) / n
    return metrics, details


def _oolong_norm(s):
    return re.sub(r"[\[\]'\"]", "", str(s)).strip().lower()


def _oolong_extract(resp):
    """Pull the answer value from a model response. Prefer text after the last
    'Answer:'/'Label:'/'User:'/'Date:' marker; else the whole stripped line."""
    m = list(re.finditer(r'(?:answer|label|user|date|month)\s*:\s*(.+)',
                         resp, re.IGNORECASE))
    return (m[-1].group(1) if m else resp).strip()


def _eval_oolong(examples, responses):
    """Oolong: exact-match for label/user/comparison/date answers; numeric
    partial credit 0.75^|y-ŷ|; set-overlap (F1) for list answers."""
    scores, ems, details = [], [], []
    for ex, resp in zip(examples, responses):
        raw = ex.get("ex", ex)  # eval wraps examples; _meta lives under "ex"
        meta = raw.get("_meta") or {}
        atype = meta.get("answer_type", "")
        gold_list = meta.get("gold_list") or [raw["answers"][0]]
        pred = _oolong_extract(resp)
        if "NUMERIC" in atype:
            nums = re.findall(r'-?\d+\.?\d*', pred)
            try:
                err = abs(float(gold_list[0]) - float(nums[-1]))
                score = 0.75 ** err
                em = float(err == 0)
            except (ValueError, IndexError):
                score = em = 0.0
        elif len(gold_list) > 1:  # set-overlap F1
            pset = {_oolong_norm(x) for x in re.split(r'[;,]', pred)}
            gset = {_oolong_norm(x) for x in gold_list}
            tp = len(pset & gset)
            p = tp / len(pset) if pset else 0.0
            r = tp / len(gset) if gset else 0.0
            score = 2 * p * r / (p + r) if (p + r) else 0.0
            em = float(pset == gset)
        else:
            em = float(_oolong_norm(pred) == _oolong_norm(gold_list[0]))
            score = em
        scores.append(score); ems.append(em)
        details.append({"prediction": resp.strip()[:300], "gold": gold_list,
                        "answer_type": atype, "task_group": meta.get("task_group"),
                        "score": score, "exact_match": em})
    from collections import defaultdict
    by_group = defaultdict(list)
    for d in details:
        by_group[d["task_group"]].append(d["score"])
    n = len(scores) or 1
    return {"score": sum(scores) / n, "exact_match": sum(ems) / n,
            "per_task_group": {g: sum(v) / len(v) for g, v in by_group.items()}}, details


def _parse_ranking(text, n):
    """Ordered, de-duplicated list of 1-indexed IDs in [1,n] from the response.
    Order matters, so if a CoT precedes the answer, read only the 'Ranking:' line."""
    if "Ranking:" in text:
        text = text.rsplit("Ranking:", 1)[1]
    ids = re.findall(r'\[(\d+)\]', text) or re.findall(r'\b(\d+)\b', text)
    out = []
    for x in ids:
        i = int(x)
        if 1 <= i <= n and i not in out:
            out.append(i)
    return out


def _kendall_tau(rank_a, rank_b, items):
    """Kendall tau-b between two rankings (item -> numeric position) over `items`."""
    C = D = Ta = Tb = 0
    m = len(items)
    for i in range(m):
        for j in range(i + 1, m):
            a = rank_a[items[i]] - rank_a[items[j]]
            b = rank_b[items[i]] - rank_b[items[j]]
            if a == 0 and b == 0:
                continue
            if a == 0:
                Ta += 1
            elif b == 0:
                Tb += 1
            elif (a < 0) == (b < 0):
                C += 1
            else:
                D += 1
    denom = math.sqrt((C + D + Ta) * (C + D + Tb))
    return (C - D) / denom if denom > 0 else 0.0


def _eval_rerank(examples, responses, k=10):
    """Re-ranking metrics over the model's ranked ID list.

    Always: MRR@k + recall@k over the binary qrel golds. When the example carries
    per-document cross-encoder scores (`ce_scores`, from generate_msmarco_trainhn),
    also: NDCG@k against a CE-derived graded relevance (gain = sigmoid(CE), random
    fill = 0) and Kendall tau-b vs the full CE reference order over the scored docs.
    NDCG/tau let the score reflect ordering quality (HELMET-style), not just whether
    a gold landed in the top-k."""
    mrrs, recs, ndcgs, taus, details, pf = [], [], [], [], [], 0
    # GUARD: the OLD binary rerank format (no `ce_scores`) is disabled. It yields a
    # degenerate gold-first target + MRR-only metrics. Require CE-graded data from
    # generate_msmarco_trainhn_data.py so the score is NDCG@10 + Kendall-tau.
    if examples:
        _raw0 = examples[0].get("ex", examples[0])
        _ce0 = _raw0.get("ce_scores")
        if not (_ce0 and any(s is not None for s in _ce0)):
            raise NotImplementedError(
                "DEPRECATED binary rerank eval: data has no `ce_scores`. The old "
                "gold-first / MRR-only rerank format is disabled so stale numbers "
                "can't be read by accident. Regenerate rerank with "
                "generate_msmarco_trainhn_data.py (CE-graded -> NDCG@10 + Kendall-tau)."
            )
    for ex, resp in zip(examples, responses):
        raw = ex.get("ex", ex)  # eval wraps examples; documents live under "ex"
        gold = {g + 1 for g in raw["gold_doc_indices"]}
        n = len(raw["documents"])
        ranked = _parse_ranking(resp, n)
        if not ranked:
            pf += 1
        rr = 0.0
        for rank, doc in enumerate(ranked[:k], 1):
            if doc in gold:
                rr = 1.0 / rank
                break
        rec = len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0
        mrrs.append(rr); recs.append(rec)
        det = {"prediction": resp.strip()[:300], "gold": sorted(gold),
               "ranked_head": ranked[:k], f"mrr@{k}": rr, f"recall@{k}": rec}

        ce = raw.get("ce_scores")
        if ce and any(s is not None for s in ce):
            # 1-indexed gains; CE = relevance truth, random fill (None) = 0.
            gain = {i + 1: (1.0 / (1.0 + math.exp(-ce[i])) if ce[i] is not None
                            else 0.0) for i in range(n)}
            dcg = sum(gain.get(doc, 0.0) / math.log2(r + 1)
                      for r, doc in enumerate(ranked[:k], 1))
            ideal = sorted(gain.values(), reverse=True)[:k]
            idcg = sum(g / math.log2(r + 1) for r, g in enumerate(ideal, 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            scored = [i + 1 for i in range(n) if ce[i] is not None]
            last = len(ranked) + 1
            rank_a = {it: (ranked.index(it) + 1 if it in ranked else last)
                      for it in scored}
            ce_order = sorted(scored, key=lambda it: -gain[it])
            rank_b = {it: r for r, it in enumerate(ce_order, 1)}
            tau = _kendall_tau(rank_a, rank_b, scored)
            ndcgs.append(ndcg); taus.append(tau)
            det[f"ndcg@{k}"] = ndcg; det["kendall_tau"] = tau
        details.append(det)
    n = len(mrrs) or 1
    out = {f"mrr@{k}": sum(mrrs) / n, f"recall@{k}": sum(recs) / n,
           "parse_rate": (n - pf) / n}
    if ndcgs:
        out[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs)
        out["kendall_tau"] = sum(taus) / len(taus)
        out["ce_graded_n"] = len(ndcgs)
    return out, details


def _parse_helmet_rankings(output, valid_ids):
    """Parse HELMET's "Ranking: ID3 > ID1 > ..." output into an ordered id list.

    Robust to a leading "Ranking:" echo, a trailing newline (HELMET stop_newline),
    and an "ID"-prefix on the tokens. Only ids present in `valid_ids` are kept,
    de-duplicated in first-seen order."""
    import re
    text = output
    if "Ranking:" in text:
        text = text.rsplit("Ranking:", 1)[-1]
    text = text.split("\n")[0]
    vset = set(valid_ids)
    ranked, seen = [], set()
    for chunk in text.split(">"):
        toks = re.findall(r"[A-Za-z0-9_\-]+", chunk)
        pick = next((t for t in toks if t in vset), None)
        if pick is None:
            pick = next((t[2:] for t in toks
                         if t.upper().startswith("ID") and t[2:] in vset), None)
        if pick and pick not in seen:
            seen.add(pick)
            ranked.append(pick)
    return ranked


def _eval_rerank_helmet(examples, responses, primary_k=10):
    """HELMET-format rerank metrics: NDCG@k over the RAW cross-encoder ranking.

    The ground truth is the continuous CE order itself — relevance gain =
    reciprocal CE rank (1/r in the `score`-descending order), NOT a bucketed
    graded label. So the metric rewards reproducing the actual CE order and
    distinguishes "gold first" from "lower-CE-distractor first" (the coarse 0-3
    bucketing could not). NDCG@k = DCG/IDCG with the standard 1/log2(rank+1)
    discount; ideal = the CE order. Falls back to the bucketed `label` only for
    HELMET-native files with no `score`. Reports NDCG@{5,10,20,100} (each over
    pools large enough), parse_rate, and mean prediction length."""
    metrics_k = [5, 10, 20, 100]
    agg = {kk: [] for kk in metrics_k}
    parse_ok, npreds, details = [], [], []
    for ex, resp in zip(examples, responses):
        rec = ex.get("ex", ex)
        ctxs = rec["ctxs"]
        key = "score" if "score" in ctxs[0] else "label"
        order = sorted(ctxs, key=lambda c: c[key], reverse=True)
        gain = {str(c["id"]): 1.0 / r for r, c in enumerate(order, 1)}
        ranked = _parse_helmet_rankings(resp, list(gain.keys()))
        parse_ok.append(1.0 if ranked else 0.0)
        npreds.append(len(ranked))
        ideal = sorted(gain.values(), reverse=True)
        det = {"prediction": resp.strip()[:300], "num_preds": len(ranked),
               "ranked_head": ranked[:primary_k]}
        for kk in metrics_k:
            if kk > len(ctxs) and kk != primary_k:
                continue
            dcg = sum(gain.get(doc, 0.0) / math.log2(i + 2)
                      for i, doc in enumerate(ranked[:kk]))
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:kk]))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            agg[kk].append(ndcg)
            det[f"ndcg@{kk}"] = ndcg
        details.append(det)
    n = len(parse_ok) or 1
    out = {"parse_rate": sum(parse_ok) / n,
           "avg_num_preds": sum(npreds) / n,
           "num_examples": len(parse_ok)}
    for kk in metrics_k:
        if agg[kk]:
            out[f"ndcg@{kk}"] = sum(agg[kk]) / len(agg[kk])
    return out, details


def _eval_summarization(examples, responses):
    """Summarization: ROUGE-1/ROUGE-L F1 if rouge_score is available, else a
    token-F1 fallback, against the reference summary. (HELMET's faithful metric
    is an LLM-judge atomic-claim F1 — see scripts/eval/score_summarization_claims.py
    for the local-Qwen version.)"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        use_rouge = True
    except ImportError:
        from ctc_eval.lib.metrics import token_f1
        use_rouge = False

    r1s, rls, details = [], [], []
    for ex, resp in zip(examples, responses):
        ref = ex["answers"][0]
        pred = resp.strip()
        if use_rouge:
            sc = scorer.score(ref, pred)
            r1, rl = sc["rouge1"].fmeasure, sc["rougeL"].fmeasure
        else:
            r1 = rl = token_f1(pred, ref)
        r1s.append(r1); rls.append(rl)
        details.append({"prediction": pred[:500], "reference": ref[:500],
                        "rouge1_f": r1, "rougeL_f": rl})
    n = len(r1s) or 1
    metric = "rouge" if use_rouge else "token_f1"
    return {"rouge1_f": sum(r1s) / n, "rougeL_f": sum(rls) / n,
            "metric": metric}, details


def _eval_cycle(examples, responses):
    """Evaluate cycle task: cycle-level precision/recall/F1/EM + claim-level F1."""
    results_list = []
    details = []
    parse_failures = 0

    for ex, resp in zip(examples, responses):
        gold = ex["gold_doc_indices"]  # list of cycles (each a list of IDs)
        predicted = parse_cycles(resp)
        if predicted is None:
            parse_failures += 1
            predicted = []
        m = cycle_metrics(predicted, gold)
        results_list.append(m)
        details.append({
            "prediction": resp.strip()[:500],
            "gold_cycles": gold,
            "predicted_cycles": predicted,
            **m,
        })

    n = len(results_list)
    metrics = {k: sum(r[k] for r in results_list) / n
               for k in ["precision", "recall", "f1", "exact_match", "claim_f1"]}
    metrics["parse_rate"] = (n - parse_failures) / n
    return metrics, details


if __name__ == "__main__":
    main()
