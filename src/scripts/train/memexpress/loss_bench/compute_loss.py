"""
Stage 2: for ONE model, compute mean cross-entropy loss over label/completion tokens only, on:
  - "train": the model's own SFT training-data manifest (built once by build_train_manifest.py,
    shared with every other model on the same data group), per context-length bucket.
  - "val": the shared v3-eval manifest (built once by build_val_manifest.py, shared by every
    model), per (task, context-length rung), capped at the model's own trained context window.

Reuses the SAME model-loading and prompt/mask-construction code paths the real evals use (see
README.md for the exact precedents), so a checkpoint sees token sequences built the same way
whether it's being generation-evaluated or loss-scored here:
  - dense / landmark (fast-landmark, sparse-landmark): plain chat-template SFT format, matching
    ``eval_lc_native.py``'s ``_load`` (Qwen chat template over ``build_prompt``'s raw text, then the
    gold answer + EOS appended).
  - docchunk: ``segment_prompt_to_chunks(..., include_answer=True)`` + ``emit_document_chunk_dense``,
    matching the training converter (byte-identical box-marker prefill+answer to what training saw).
  - summary_token: same, but ``emit_document_chunk_summary`` + ``set_summary_eval_mask_mode``, and
    the checkpoint-recorded mask-mixture arm is overridden to serve "causal" (the project default;
    see records referenced in models.py).

Loss-vs-memory: a full 256k forward materializes (1, 256144, vocab=248320) logits in bf16 -- roughly
127 GB -- if computed naively. Avoided here by passing ``logits_to_keep`` as a TENSOR of exactly the
label-token positions (the LM head gathers hidden states to just those positions before computing
logits), so only a (1, n_label_tokens, vocab) tensor is ever materialized; n_label_tokens is the
completion length (tens to low hundreds of tokens), not the context length. The full-context
attention forward to produce hidden states still happens over the whole sequence, so this does NOT
avoid the compute cost of long context, only the vocab-sized memory blowup at the head.

KNOWN OPEN RISK (flagged, not resolved): the three 256k-window models (dense_xlong5_256k,
summtok_causal/decay/p50) were trained with Ulysses CP across multiple GPUs. This script runs a
single-GPU, no-CP forward pass. Existing generation evals already do a full un-cached 256k forward
on one 80GB GPU during prefill (see records/v3-eval-howto.md's "256k+ needs an 80GB GPU"), which is
precedent that this fits -- but that precedent used ``logits_to_keep=1``, never a full 256k
backbone forward AND a bigger logits gather in the same job as this script's other buckets. If this
OOMs, the fix is CP (out of scope for this first cut) -- it will fail loudly (CUDA OOM), not
silently produce a wrong number.

Usage (one Beaker GPU job per model; loop over --model-key for all 8):
    PYTHONPATH=src python compute_loss.py --model-key sparselm_32k
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from models import (
    LENGTH_LADDER,
    LENGTH_LADDER_LABELS,
    MODELS,
    RESULTS_DIR,
    TRAIN_MANIFEST_DIR,
    VAL_MANIFEST_PATH,
    VAL_TASKS,
)

# rung label -> nominal token count, for capping val rungs at a model's max_context_length. Includes
# "3k" (nq/outlier's smallest rung) in addition to the LENGTH_LADDER's "2k".
RUNG_TOKENS = {
    "2k": 2048,
    "3k": 3072,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
}


def load_model(spec, device):
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import reserved_ids
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig

    ids = reserved_ids(spec.tokenizer_family)
    hf_tok_name = "Qwen/Qwen3.5-0.8B" if spec.tokenizer_family == "qwen3_5" else "Qwen/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(hf_tok_name)

    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, ids.eos) else 151645
    gen_cfg = GenerationConfig(
        eos_token_id=ids.eos,
        pad_token_id=pad_id,
        max_length=spec.max_context_length + 64,
        use_cache=True,
    )
    t0 = time.time()
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=spec.checkpoint, device=device)
    print(f"[load] built {spec.checkpoint} in {time.time() - t0:.1f}s", flush=True)

    if spec.architecture == "docchunk":
        gm.model.enable_document_chunk_attention(
            doc_start_id=ids.doc_start,
            doc_end_id=ids.doc_end,
            eos_id=ids.eos,
            mode="chunked",
            pad_id=None,
        )
    elif spec.architecture == "summary_token":
        mode = gm.model.set_summary_eval_mask_mode(spec.summary_mask_mode or "causal")
        print(f"[summary-mask] serving the {mode!r} arm", flush=True)

    gm.model.eval()
    return gm, tok, ids


@torch.no_grad()
def example_loss(gm, device, ids: List[int], label_mask: List[bool]) -> Optional[Tuple[float, int]]:
    """Returns (sum of per-token CE over label tokens, n_label_tokens), or None if there are no
    label tokens to score (shouldn't happen for real examples; guarded rather than assumed)."""
    T = len(ids)
    ids_t = torch.tensor(ids, dtype=torch.long)
    mask_t = torch.tensor(label_mask, dtype=torch.bool)
    shifted_labels = torch.full((T,), -100, dtype=torch.long)
    if T > 1:
        shifted_labels[:-1] = torch.where(
            mask_t[1:], ids_t[1:], torch.full((T - 1,), -100, dtype=torch.long)
        )
    keep = (shifted_labels != -100).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return None

    input_ids_b = ids_t.unsqueeze(0).to(device)
    labels_b = shifted_labels.unsqueeze(0).to(device)
    logits_to_keep = keep.unsqueeze(0).to(device)
    leftpad = torch.zeros(1, dtype=torch.int32, device=device)

    gm.prepare_inference_cache(1, T)
    out = gm.model(
        input_ids_b,
        labels=labels_b,
        ignore_index=-100,
        loss_reduction="none",
        logits_to_keep=logits_to_keep,
        cache_leftpad=leftpad,
        return_logits=False,
    )
    ce = out.ce_loss.view(-1)  # (n_label_tokens,)
    return float(ce.sum().item()), int(keep.numel())


def assign_train_buckets(max_ctx: int) -> List[str]:
    return [lab for thr, lab in zip(LENGTH_LADDER, LENGTH_LADDER_LABELS) if thr <= max_ctx]


def run_train_loss(spec, gm, tok, ids_set, device) -> dict:
    npz_path = f"{TRAIN_MANIFEST_DIR}/{spec.data_group}.npz"
    idx_path = f"{TRAIN_MANIFEST_DIR}/{spec.data_group}.index.json"
    npz = np.load(npz_path)
    with open(idx_path) as f:
        index = json.load(f)["examples"]

    buckets_in_scope = set(assign_train_buckets(spec.max_context_length))
    per_key: dict = {}  # (task, bucket) -> accumulator
    n_skipped_too_long = 0

    for rec in index:
        bucket = rec["bucket"]
        if bucket not in buckets_in_scope:
            continue
        task = rec["task"]
        i = rec["i"]
        key = f"{task}__{bucket}__{i}"
        ids = npz[f"{key}__ids"].tolist()
        mask = npz[f"{key}__mask"].tolist()
        if len(ids) > spec.max_context_length:
            n_skipped_too_long += 1
            continue
        result = example_loss(gm, device, ids, mask)
        if result is None:
            continue
        ce_sum, n_tok = result
        b = per_key.setdefault(
            (task, bucket), {"ce_sum": 0.0, "n_tokens": 0, "n_examples": 0, "examples": []}
        )
        b["ce_sum"] += ce_sum
        b["n_tokens"] += n_tok
        b["n_examples"] += 1
        b["examples"].append(
            {"length": rec["length"], "n_label_tokens": n_tok, "mean_ce": ce_sum / n_tok}
        )

    if n_skipped_too_long:
        print(
            f"[train] skipped {n_skipped_too_long} examples over max_context_length={spec.max_context_length}",
            flush=True,
        )

    out = {}
    for (task, bucket), b in per_key.items():
        if b["n_tokens"] == 0:
            continue
        out_key = f"{task}@{bucket}"
        out[out_key] = {
            "task": task,
            "bucket": bucket,
            "mean_ce_token_weighted": b["ce_sum"] / b["n_tokens"],
            "n_tokens": b["n_tokens"],
            "n_examples": b["n_examples"],
            "examples": b["examples"],
        }
        print(
            f"[train] {spec.data_group}/{out_key}: mean_ce={out[out_key]['mean_ce_token_weighted']:.4f} "
            f"n_examples={b['n_examples']} n_tokens={b['n_tokens']}",
            flush=True,
        )
    return out


def build_dense_or_landmark_example(tok, ex: dict, eos_id: int) -> Tuple[List[int], List[bool]]:
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tok(ex["expected_output"], add_special_tokens=False).input_ids
    ids = prompt_ids + answer_ids + [eos_id]
    mask = [False] * len(prompt_ids) + [True] * (len(answer_ids) + 1)
    return ids, mask


def build_chunked_example(
    tok,
    raw_ex: dict,
    seg_task: str,
    chunk_by: str,
    cot_mode: str,
    qp: str,
    architecture: str,
    ids_set,
    num_summary_tokens: int,
) -> Tuple[List[int], List[bool]]:
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_summary,
        segment_prompt_to_chunks,
    )

    segs, _tok_ids, _lm = segment_prompt_to_chunks(
        tok,
        raw_ex,
        seg_task,
        query_position=qp,
        cot_mode=cot_mode,
        chunk_by=chunk_by,
        include_answer=True,
        doc_start_id=ids_set.doc_start,
        doc_end_id=ids_set.doc_end,
    )
    if architecture == "summary_token":
        ids, mask = emit_document_chunk_summary(
            segs, summary_token_id=ids_set.summary, n_summary_tokens=num_summary_tokens
        )
    else:
        ids, mask = emit_document_chunk_dense(segs)
    ids = list(ids) + [ids_set.eos]
    mask = list(mask) + [True]
    return ids, mask


def run_val_loss(spec, gm, tok, ids_set, device) -> dict:
    # Requires PYTHONPATH to include `<repo>/src/scripts` (same convention as
    # run_beaker_multirung_eval.sh) so `ctc_eval.eval.evaluate` resolves.
    from ctc_eval.eval.evaluate import load_unified_examples

    with open(VAL_MANIFEST_PATH) as f:
        manifest = json.load(f)

    out: dict = {}
    for task, cfg in VAL_TASKS.items():
        rungs = manifest["tasks"].get(task, {})
        for rung_label, rung in rungs.items():
            if RUNG_TOKENS.get(rung_label, 10**9) > spec.max_context_length:
                continue
            path = rung["path"]
            sample_indices = rung["indices"]
            if not sample_indices:
                continue
            examples = load_unified_examples(
                path,
                None,
                task=cfg["seg_task"],
                query_position=spec.query_position,
                use_alpaca=True,
            )
            ce_sum = 0.0
            n_tokens = 0
            n_examples = 0
            per_example = []
            for gi in sample_indices:
                if gi >= len(examples):
                    continue
                ex = examples[gi]
                if spec.architecture in ("docchunk", "summary_token"):
                    raw_ex = ex.get("ex", ex)
                    ids, mask = build_chunked_example(
                        tok,
                        raw_ex,
                        cfg["seg_task"],
                        cfg["chunk_by"],
                        cfg["cot_mode"],
                        spec.query_position,
                        spec.architecture,
                        ids_set,
                        num_summary_tokens=5,
                    )
                else:
                    ids, mask = build_dense_or_landmark_example(tok, ex, ids_set.eos)
                if len(ids) > spec.max_context_length:
                    continue
                result = example_loss(gm, device, ids, mask)
                if result is None:
                    continue
                s, n = result
                ce_sum += s
                n_tokens += n
                n_examples += 1
                per_example.append({"idx": gi, "n_label_tokens": n, "mean_ce": s / n})

            key = f"{task}@{rung_label}"
            if n_tokens == 0:
                continue
            out[key] = {
                "task": task,
                "rung": rung_label,
                "mean_ce_token_weighted": ce_sum / n_tokens,
                "n_tokens": n_tokens,
                "n_examples": n_examples,
                "examples": per_example,
            }
            print(
                f"[val] {key}: mean_ce={out[key]['mean_ce_token_weighted']:.4f} "
                f"n_examples={n_examples} n_tokens={n_tokens}",
                flush=True,
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()

    spec = MODELS[args.model_key]
    device = torch.device("cuda:0")
    gm, tok, ids_set = load_model(spec, device)

    result: dict = {"model_key": args.model_key}
    result["spec"] = {
        "checkpoint": spec.checkpoint,
        "architecture": spec.architecture,
        "tokenizer_family": spec.tokenizer_family,
        "data_group": spec.data_group,
        "max_context_length": spec.max_context_length,
        "query_position": spec.query_position,
        "summary_mask_mode": spec.summary_mask_mode,
    }
    if not args.skip_train:
        result["train"] = run_train_loss(spec, gm, tok, ids_set, device)
    if not args.skip_val:
        result["val"] = run_val_loss(spec, gm, tok, ids_set, device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/{args.model_key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
