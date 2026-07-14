"""Tokenize any unified-JSONL dataset into olmo-core NumpyPaddedFSLDataset format.

Generalizes the old `tokenize_nq_for_olmo.py` (NQ-only) to every task/format in the
repo. Reuses `scripts.lib.data_format.build_prompt` so the prompt is byte-identical to
what `train_chunked_fast` / `evaluate.py` produce, and mirrors the multi-task
per-row dispatch (`_task` / `_cot_mode`) that the axolotl path in `train.py` uses.

Emits, for a given (dataset, base_model, options, seq_len):
  <prefix>_tokens.npy      raw uint32 concat of EOS-terminated, *unpadded* documents
  <prefix>_label_mask.npy  raw bool, True only on completion tokens (incl. the eos)
  <prefix>_meta.json       sidecar with the real eos/pad ids the data was written with

Format invariants (see results/olmo_core_setup.md §2 "Data — NumpyPaddedFSLDataset"):
  - raw `ndarray.tofile()`, NOT np.save (a .npy header would be miscounted as data)
  - uint32 when vocab > 65535 (Qwen = 151936)
  - documents are a raw concat ending in the *real* tokenizer eos (no manual padding);
    olmo-core segments on eos at prepare() time and pads each doc to seq_len at read time
  - the label mask is a parallel bool array in the same raw layout

Chunked / sliding-window attention (`--wrap-docs`): each document block in the prompt
is wrapped with a pair of *existing* tokenizer special tokens (default
`<|box_start|>` / `<|box_end|>`) so the per-token document structure can be recovered
downstream by scanning input_ids (see scripts/lib/olmo_flex_attention.py). We reuse
existing reserved tokens rather than adding new ones so the converted Qwen3 embedding
table (vocab 151936) stays valid — no embedding resize. The marker ids are recorded in
`_meta.json` (`doc_start_id` / `doc_end_id`); training AND eval must use the same ids.

`tokenize_unified_for_olmo()` is the entry point the dispatcher (train.py) calls; the
CLI below is for standalone/debug use.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from corpus_reasoning.lib import chunked_attention as _ca
from corpus_reasoning.lib.data_format import build_prompt

CACHE_DIR = Path("data/.cache/olmo")

# Default document-boundary markers: existing Qwen reserved special tokens that never
# occur in our rendered prompts, so no vocab growth / embedding resize is needed.
DEFAULT_DOC_MARKER_START = "<|box_start|>"
DEFAULT_DOC_MARKER_END = "<|box_end|>"


def _single_token_id(tok, marker: str) -> int:
    """Resolve `marker` to its single token id, or raise if it isn't one token.

    `add_special_tokens=False` only suppresses auto BOS/EOS; registered special-token
    strings present in the text are still encoded to their single id. If the chosen
    marker is not a registered special token it will split into subwords (len != 1)
    and we fail loudly so the caller can pick a different reserved token.
    """
    ids = tok(marker, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(
            f"doc marker {marker!r} does not tokenize to a single id for this model "
            f"(got {ids}). Pick a reserved special token that exists in the tokenizer "
            f"(e.g. <|box_start|>, <|quad_start|>)."
        )
    return int(ids[0])


def _cache_paths(src: Path, base_model: str, task, query_position, use_titles,
                 before_dummy, after_dummy, cot_mode, seq_len, train_on_inputs,
                 wrap_docs, doc_marker_start, doc_marker_end,
                 standard_mix_prob, mix_seed, emit_gold_mask=False):
    """Cache key mirrors train.py's alpaca cache, plus base_model and seq_len (both are
    baked into the tokenized artifact here, unlike the axolotl path where the tokenizer
    is applied later). `wrap_docs`/markers/mix/gold are part of the key so wrapped,
    unwrapped, mask-mixed and gold-mask artifacts never collide."""
    st = src.stat()
    tag = "multitask" if _is_multitask(src) else task
    wrap_tag = f"wrap={int(wrap_docs)}:{doc_marker_start}:{doc_marker_end}"
    mix_tag = f"mix={standard_mix_prob}:{mix_seed}" if (wrap_docs and standard_mix_prob > 0) else "mix=0"
    gold_tag = f"gold={int(emit_gold_mask)}"
    key = (f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{tag}|{query_position}|"
           f"{use_titles}|{before_dummy}|{after_dummy}|{cot_mode}|{base_model}|{seq_len}|"
           f"toi={train_on_inputs}|{wrap_tag}|{mix_tag}|{gold_tag}")
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    prefix = f"{src.stem}_{tag}_q{query_position}_{h}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (CACHE_DIR / f"{prefix}_tokens.npy",
            CACHE_DIR / f"{prefix}_label_mask.npy",
            CACHE_DIR / f"{prefix}_meta.json"), tag


def _is_multitask(src: Path) -> bool:
    """A combined file tags each row with `_task` (see build_combined_unified.py)."""
    with src.open() as f:
        first = f.readline()
    if not first:
        return False
    try:
        return "_task" in json.loads(first)
    except ValueError:
        return False


def tokenize_unified_for_olmo(src, base_model, task="retrieval",
                              query_position="after", use_titles=True,
                              before_dummy=0, after_dummy=0, cot_mode="label",
                              seq_len=8192, train_on_inputs=False, force=False,
                              wrap_docs=False,
                              doc_marker_start=DEFAULT_DOC_MARKER_START,
                              doc_marker_end=DEFAULT_DOC_MARKER_END,
                              standard_mix_prob=0.0, mix_seed=42,
                              emit_gold_mask=False):
    """Tokenize `src` (unified JSONL) into olmo-core format, caching under data/.cache/olmo/.

    Returns (tokens_path, label_mask_path, meta_path) as strings. If a combined file
    tags rows with `_task`/`_cot_mode`, prompt building is dispatched per-row; otherwise
    the passed task/cot_mode are used for every row.

    If `train_on_inputs` is True, the loss mask is True on EVERY token (full sequence LM loss);
    default False keeps loss only on the completion (matches axolotl `train_on_inputs: false`).

    If `wrap_docs` is True, each document block in the prompt is wrapped with the
    `doc_marker_start`/`doc_marker_end` special tokens (reusing chunked_attention's
    document detection), enabling chunked / sliding-window attention at train time. The
    resolved marker ids are written to the meta sidecar.

    Mask mixing (`standard_mix_prob` > 0, requires `wrap_docs`): with this probability,
    an example is tokenized WITHOUT doc-boundary markers (a deterministic per-example
    coin flip seeded by `mix_seed`). Such an example has no chunk structure, so at train
    time `build_roles` marks every token FREE and the chunked / modified_swa mask_mod
    collapses to plain causal for it — i.e. that fraction trains as standard attention
    while the rest stay masked. This mirrors `standard_mix_prob` in the HF chunked path
    (scripts/train/train_chunked_fast.py); here the split is baked at tokenize time.
    """
    src = Path(src)
    (tok_path, mask_path, meta_path), tag = _cache_paths(
        src, base_model, task, query_position, use_titles,
        before_dummy, after_dummy, cot_mode, seq_len, train_on_inputs,
        wrap_docs, doc_marker_start, doc_marker_end, standard_mix_prob, mix_seed,
        emit_gold_mask)
    gold_path = Path(str(tok_path).replace("_tokens.npy", "_gold.json"))
    if emit_gold_mask and not wrap_docs:
        raise ValueError("emit_gold_mask=True requires wrap_docs=True (gold-doc "
                         "structure is recovered from the doc-boundary markers).")
    cached = tok_path.exists() and mask_path.exists() and meta_path.exists()
    if emit_gold_mask:
        cached = cached and gold_path.exists()
    if not force and cached:
        print(f"[cache] reusing {tok_path.name}")
        return str(tok_path), str(mask_path), str(meta_path)

    tok = AutoTokenizer.from_pretrained(base_model)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    multitask = (tag == "multitask")

    doc_start_id = doc_end_id = None
    if wrap_docs:
        doc_start_id = _single_token_id(tok, doc_marker_start)
        doc_end_id = _single_token_id(tok, doc_marker_end)
        # Point chunked_attention's module-global markers at our reserved tokens so its
        # (well-tested) wrap_documents/_wrap_paragraph insert these strings. The strings
        # encode to the single ids resolved above.
        _ca.DOC_START = doc_marker_start
        _ca.DOC_END = doc_marker_end

    do_mix = wrap_docs and standard_mix_prob > 0.0
    print(f"[cache] tokenizing {src} -> {tok_path.name}  "
          f"(tag={tag}, qp={query_position}, titles={use_titles}, cot_mode={cot_mode}, "
          f"seq_len={seq_len}, eos={eos_id}, wrap_docs={wrap_docs}"
          + (f", doc_ids=({doc_start_id},{doc_end_id})" if wrap_docs else "")
          + (f", mix_std={standard_mix_prob}@seed{mix_seed}" if do_mix else "") + ")")

    # Gold-document gradient masking: build {content_fingerprint -> [gold chunk idx]}
    # so the trainer can recover each example's gold docs from input_ids alone (no
    # gold marker in the token stream -> non-leaky). See scripts/lib/olmo_gold_grad_mask.
    gold_table = {}
    n_gold = 0
    if emit_gold_mask:
        from corpus_reasoning.lib.olmo_gold_grad_mask import (
            content_fingerprint, gold_chunks_from_gold_doc_indices)

    all_ids, all_mask = [], []
    kept = skipped = 0
    n_mixed_standard = 0
    idx = -1
    with src.open() as f:
        for line in f:
            if not line.strip():
                continue
            idx += 1  # per-example index (counts every non-blank row, matches HF mix indexing)
            ex = json.loads(line)
            ex_task = ex.get("_task", task) if multitask else task
            ex_cot = ex.get("_cot_mode", cot_mode) if multitask else cot_mode
            prompt, output = build_prompt(
                ex, task=ex_task, query_position=query_position,
                use_titles=use_titles, before_dummy=before_dummy,
                after_dummy=after_dummy, use_alpaca=True, cot_mode=ex_cot,
            )
            # Mask mixing: deterministically present a `standard_mix_prob` fraction of
            # examples WITHOUT doc markers, so they train as plain causal under the mask.
            present_as_standard = (
                do_mix and random.Random(mix_seed + idx).random() < standard_mix_prob
            )
            if wrap_docs and not present_as_standard:
                # Insert document-boundary markers around each doc block in the prompt.
                # (Documents live in the prompt, never the completion.)
                prompt = _ca.wrap_documents(prompt)
            if present_as_standard:
                n_mixed_standard += 1
            # Tokenize prompt and completion separately so the loss-mask boundary is
            # exact and the prompt tokenization matches inference (evaluate.py tokenizes
            # the prompt alone).
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            o_ids = tok(output, add_special_tokens=False).input_ids + [eos_id]
            if len(p_ids) + len(o_ids) > seq_len:  # would lose the completion -> drop
                skipped += 1
                continue
            all_ids.extend(p_ids + o_ids)
            # train_on_inputs=True -> loss on every token (full-sequence LM); else only the completion.
            if train_on_inputs:
                all_mask.extend([True] * (len(p_ids) + len(o_ids)))
            else:
                all_mask.extend([False] * len(p_ids) + [True] * len(o_ids))
            # Gold table: key on the example's content tokens (p_ids+o_ids == the
            # trainer's input_ids[:first_eos+1]); value = 0-based gold chunk indices.
            # Only marker-wrapped examples have recoverable doc structure.
            if emit_gold_mask and not present_as_standard and ex.get("gold_doc_indices"):
                fp = content_fingerprint(p_ids + o_ids)
                gold_table[fp] = sorted(gold_chunks_from_gold_doc_indices(ex["gold_doc_indices"]))
                n_gold += 1
            kept += 1

    tokens = np.asarray(all_ids, dtype=np.uint32)   # Qwen vocab > 65535
    masks = np.asarray(all_mask, dtype=np.bool_)
    tokens.tofile(tok_path)   # raw bytes (olmo-core memmaps these)
    masks.tofile(mask_path)
    meta = {"eos_token_id": int(eos_id), "pad_token_id": int(pad_id),
            "n_examples": kept, "seq_len": seq_len, "base_model": base_model,
            "wrap_docs": bool(wrap_docs)}
    if wrap_docs:
        meta["doc_start_id"] = doc_start_id
        meta["doc_end_id"] = doc_end_id
        meta["doc_marker_start"] = doc_marker_start
        meta["doc_marker_end"] = doc_marker_end
    if do_mix:
        meta["standard_mix_prob"] = standard_mix_prob
        meta["mix_seed"] = mix_seed
        meta["n_mixed_standard"] = n_mixed_standard
    if emit_gold_mask:
        json.dump(gold_table, open(gold_path, "w"))
        meta["gold_mask"] = True
        meta["gold_path"] = str(gold_path)
        meta["n_gold_examples"] = n_gold
        print(f"[cache] gold-mask sidecar: {n_gold} examples with gold_doc_indices "
              f"-> {gold_path.name}")
    json.dump(meta, open(meta_path, "w"))
    print(f"[cache] kept={kept} skipped(>{seq_len})={skipped} "
          f"loss_tokens={int(masks.sum())} avg_loss_tokens={masks.sum()/max(kept,1):.1f}"
          + (f" mixed_standard={n_mixed_standard}/{kept + skipped} "
             f"({100.0 * n_mixed_standard / max(kept + skipped, 1):.1f}%)" if do_mix else ""))
    if kept == 0:
        raise RuntimeError(
            f"no examples fit in seq_len={seq_len} for {src} — every doc was truncated.")
    return str(tok_path), str(mask_path), str(meta_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="unified JSONL")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--task", default="retrieval")
    ap.add_argument("--query-position", default="after")
    ap.add_argument("--no-titles", action="store_true")
    ap.add_argument("--before-dummy", type=int, default=0)
    ap.add_argument("--after-dummy", type=int, default=0)
    ap.add_argument("--cot-mode", default="label")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--train-on-inputs", action="store_true",
                    help="loss on every token (full-sequence LM); default off = loss only on completion")
    ap.add_argument("--wrap-docs", action="store_true",
                    help="wrap each document block with marker tokens for chunked/SWA attention")
    ap.add_argument("--doc-marker-start", default=DEFAULT_DOC_MARKER_START,
                    help="reserved special token that opens a document (must be 1 token)")
    ap.add_argument("--doc-marker-end", default=DEFAULT_DOC_MARKER_END,
                    help="reserved special token that closes a document (must be 1 token)")
    ap.add_argument("--standard-mix-prob", type=float, default=0.0,
                    help="fraction of examples tokenized WITHOUT doc markers (train as "
                         "plain causal under the mask); requires --wrap-docs")
    ap.add_argument("--mix-seed", type=int, default=42,
                    help="seed for the deterministic per-example mask-mix coin flip")
    ap.add_argument("--emit-gold-mask", action="store_true",
                    help="write a {content_fingerprint -> gold chunk idx} sidecar for "
                         "gold-document gradient masking (requires --wrap-docs)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tok_path, mask_path, meta_path = tokenize_unified_for_olmo(
        args.input, args.base_model, task=args.task,
        query_position=args.query_position, use_titles=not args.no_titles,
        before_dummy=args.before_dummy, after_dummy=args.after_dummy,
        cot_mode=args.cot_mode, seq_len=args.seq_len,
        train_on_inputs=args.train_on_inputs, force=args.force,
        wrap_docs=args.wrap_docs, doc_marker_start=args.doc_marker_start,
        doc_marker_end=args.doc_marker_end,
        standard_mix_prob=args.standard_mix_prob, mix_seed=args.mix_seed,
        emit_gold_mask=args.emit_gold_mask,
    )
    print(f"tokens: {tok_path}\nmask:   {mask_path}\nmeta:   {meta_path}")


if __name__ == "__main__":
    main()
