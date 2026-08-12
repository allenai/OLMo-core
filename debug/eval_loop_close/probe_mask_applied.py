"""
Does ``--attn chunked`` actually change what the model sees?

Arms A and E of the grading run produced **byte-identical generations on all 500 rows**, which is
either a real null (a 30-step model whose argmax is insensitive to the mask) or the mask silently
not being applied at eval. 500/500 is too many to accept as a coincidence, so this checks the
mechanism directly rather than the score:

1. the marker ids are present in the prefill the evaluator actually builds;
2. ``build_chunk_ids_from_tokens`` turns them into more than one chunk;
3. the logits move when the mask is toggled on the SAME token ids.

Point 3 is the load-bearing one: everything upstream can look right while the mask is still a no-op.

    PYTHONPATH=src:ctc/src python debug/eval_loop_close/probe_mask_applied.py --ckpt <stepN> \
        --tokenizer <dir> --data <rung.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="mathmatch")
    ap.add_argument("--query-position", default="after")
    args = ap.parse_args()

    import torch

    import ctc.tasks
    from ctc.eval.backends.native import NativeBackend
    from ctc.format import registry
    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    ctc.tasks.load_all()
    spec = registry.get(args.task)
    example = json.loads(Path(args.data).read_text().splitlines()[0])
    prompt = spec.build_prompt(example, query_position=args.query_position)

    backend = NativeBackend(
        Path(args.ckpt),
        tokenizer=args.tokenizer,
        attn="chunked",
        max_length=4096,
        query_position=args.query_position,
    )
    ids = backend.prefill_for(args.task)(prompt, example)
    ids_t = torch.tensor([ids], device=backend.device)
    cfg = backend.gm.model._document_chunk_attention
    print(f"1. model._document_chunk_attention = {cfg}")
    marker_hits = {
        name: int((ids_t == cfg[name]).sum())
        for name in ("doc_start_id", "doc_end_id")
        if cfg is not None
    }
    print(f"   markers in the prefill: {marker_hits}  (prompt is {ids_t.shape[1]} tokens)")

    chunk_ids = build_chunk_ids_from_tokens(
        ids_t,
        doc_start_id=cfg["doc_start_id"],
        doc_end_id=cfg["doc_end_id"],
        eos_id=cfg["eos_id"],
        mode=cfg["mode"],
    )
    distinct = torch.unique(chunk_ids)
    print(
        f"2. chunk_ids: {distinct.numel()} distinct values, min={int(distinct.min())} "
        f"max={int(distinct.max())}"
    )

    with torch.no_grad():
        backend.gm.free_inference_cache()
        chunked = backend.gm.model(ids_t, logits_to_keep=1).float()
        backend.gm.model.disable_document_chunk_attention()
        full = backend.gm.model(ids_t, logits_to_keep=1).float()

    delta = (chunked - full).abs().max().item()
    same_argmax = int(chunked.argmax(-1).item() == full.argmax(-1).item())
    print(
        f"3. max|logit(chunked) - logit(full)| = {delta:.6f}   same next token: {bool(same_argmax)}"
    )
    print("VERDICT:", "mask IS applied" if delta > 1e-3 else "mask is a NO-OP at eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
