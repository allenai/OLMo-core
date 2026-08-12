"""
Is ``--share-prefix`` score-preserving? Ask the model, not the scoreboard.

``ctc-eval --share-prefix`` prefills each corpus group's shared token prefix once and reuses its KV
across the group's queries instead of prefilling every prompt in full. The claim is that this is
*purely* a speed optimisation. Two f1 numbers that round the same are weak evidence for that; this
script goes one level down and compares, for the same model in the same process:

1. the **logit vector** at the last prompt token -- the only thing the prefill produces, so an
   identical vector means the reuse path handed the decoder exactly the state a full prefill would;
2. the **generated text**, byte for byte, after the task's own stop rules.

Both arms run against one loaded checkpoint, one tokenizer, one prompt list. The only difference is
where the KV came from, which is what makes a difference attributable.

It also times the two arms and counts prefilled tokens, because the reuse only exists for speed and
a saving that is not measured is a saving that gets quoted wrong.

Run it on a GPU node with the node-local interpreter; see ``run_parity.sbatch`` in this directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    """:returns: The probe's argument parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint directory")
    ap.add_argument("--data", required=True, help="eval jsonl carrying corpus_id")
    ap.add_argument("--spec", required=True, help="ctc.format spec name, e.g. contradiction")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--groups", type=int, default=2, help="corpus groups to probe")
    ap.add_argument("--rows-per-group", type=int, default=6, help="rows per group to probe")
    ap.add_argument(
        "--query-positions",
        default="both",
        help="comma list; 'after' is what makes a multiplexed corpus a true token prefix",
    )
    ap.add_argument("--attn", default="full", help="comma list of masks to probe")
    ap.add_argument(
        "--family",
        default="qwen3_5",
        help="reserved-id family for --attn chunked. The suite's 4B/0.8B checkpoints are qwen3_5 "
        "(vocab 248,320, eos 248,044), NOT qwen3 -- the wrong family marks the wrong ids as "
        "document boundaries, silently.",
    )
    ap.add_argument("--out", required=True, help="where to write the JSON report")
    return ap


def _enable_chunked(model, family: str) -> None:
    """
    Turn on the document-chunked mask with the ids the shards were built with.

    ``Transformer.enable_document_chunk_attention`` requires ``doc_start_id``, ``doc_end_id`` and
    ``eos_id``. ``NativeBackend._configure_attention`` calls it with none of them, so
    ``--attn chunked`` currently raises ``TypeError`` before the first prompt -- the native backend
    cannot reach its own chunked mode. The ids come from the table the trainer uses
    (``olmo_core.data.document_chunk_landmark.reserved_ids``) rather than from literals here.

    :param model: The loaded transformer.
    :param family: ``qwen3`` or ``qwen3_5``.
    """
    from olmo_core.data.document_chunk_landmark import reserved_ids

    ids = reserved_ids(family)
    model.enable_document_chunk_attention(
        doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos
    )


def _load_rows(path: Path, groups: int, rows_per_group: int) -> List[Dict[str, Any]]:
    """
    Read the first ``groups`` corpus groups, keeping at most ``rows_per_group`` rows of each.

    Sub-setting a group is safe for this comparison and not for a score: both arms see the same
    subset, and the shared prefix of a subset is at least the shared prefix of the whole group.

    :param path: Eval jsonl.
    :param groups: How many corpus groups to keep.
    :param rows_per_group: How many rows of each.

    :returns: The kept rows, in file order.
    """
    kept: Dict[str, List[Dict[str, Any]]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("corpus_id", "__none")
            if key not in kept and len(kept) >= groups:
                continue
            bucket = kept.setdefault(key, [])
            if len(bucket) < rows_per_group:
                bucket.append(row)
            if len(kept) >= groups and all(len(b) >= rows_per_group for b in kept.values()):
                break
    return [row for bucket in kept.values() for row in bucket]


def _probe(backend, spec, rows, *, query_position: str, attn: str, max_length: int) -> Dict:
    """
    Run both prefill paths over ``rows`` and compare them.

    :param backend: A loaded :class:`~ctc.eval.backends.native.NativeBackend`.
    :param spec: The task spec (prompt builder and stop rule).
    :param rows: Eval rows, grouped by ``corpus_id`` in file order.
    :param query_position: Prompt layout; ``after`` puts the corpus first, which is the only layout
        under which a multiplexed corpus is a shared token *prefix*.
    :param attn: The mask in play, recorded so a result cannot be read as covering another.
    :param max_length: Inference cache size.

    :returns: A report dict: reuse estimate, timings, and per-row logit/text agreement.
    """
    import torch

    from ctc.eval.prefill import build_prefills
    from ctc.eval.prefix_cache import (
        generate_group_with_shared_prefix,
        group_by_corpus,
        measure_reuse,
    )
    from ctc.eval.stopping import STOP_PRESETS

    stop = STOP_PRESETS[spec.stop]
    prompts = [spec.build_prompt(row, query_position=query_position) for row in rows]

    backend.query_position = query_position
    backend._prefill = None  # the builder caches the layout, so it has to be rebuilt
    prefill = backend.prefill_for(spec.name)
    all_ids = build_prefills(prefill, prompts, rows)

    groups = group_by_corpus(rows)
    estimate = measure_reuse([[all_ids[i] for i in idx] for idx in groups.values()])

    # A warm-up prefill before either arm is timed. The first long forward of a process pays
    # kernel autotuning and allocator growth, and whichever arm runs first absorbs all of it --
    # which is how the first version of this probe reported a 2.54x "speedup" on a configuration
    # that provably prefilled exactly the same tokens both ways.
    with torch.no_grad():
        backend.gm.prepare_inference_cache(1, max_length)
        warm = torch.zeros(1, dtype=torch.int32, device=backend.device)
        backend.gm.model(
            torch.tensor([list(all_ids[0])], device=backend.device),
            logits_to_keep=1,
            cache_leftpad=warm,
        )
    torch.cuda.synchronize()

    # ── arm 1: a full prefill per prompt, exactly what --share-prefix is off ────────────────────
    plain_logits: Dict[int, Any] = {}
    plain_text: Dict[int, str] = {}
    plain_prefill_tokens = 0
    torch.cuda.synchronize()
    started = time.time()
    for i, ids in enumerate(all_ids):
        with torch.no_grad():
            backend.gm.prepare_inference_cache(1, max_length)
            leftpad = torch.zeros(1, dtype=torch.int32, device=backend.device)
            logits = backend.gm.model(
                torch.tensor([list(ids)], device=backend.device),
                logits_to_keep=1,
                cache_leftpad=leftpad,
            )
            plain_logits[i] = logits[0, -1].detach().float().cpu().clone()
        plain_text[i] = backend._decode_from(torch, logits, stop)
        plain_prefill_tokens += len(ids)
    torch.cuda.synchronize()
    plain_seconds = time.time() - started

    # ── arm 2: one prefill per group, KV rewound between queries ────────────────────────────────
    shared_logits: Dict[int, Any] = {}
    shared_text: Dict[int, str] = {}
    shared_prefill_tokens = 0.0
    torch.cuda.synchronize()
    started = time.time()
    for rowidx in groups.values():
        captured: List[Any] = []

        def decode_fn(logits, captured=captured):
            captured.append(logits[0, -1].detach().float().cpu().clone())
            return backend._decode_from(torch, logits, stop)

        texts, stats = generate_group_with_shared_prefix(
            backend.gm,
            [all_ids[i] for i in rowidx],
            device=backend.device,
            max_length=max_length,
            decode_fn=decode_fn,
        )
        shared_prefill_tokens += stats["prefill_tokens"] + stats["suffix_tokens"]
        for row, text, vector in zip(rowidx, texts, captured):
            shared_text[row] = text
            shared_logits[row] = vector
    torch.cuda.synchronize()
    shared_seconds = time.time() - started

    per_row = []
    for i in range(len(all_ids)):
        a, b = plain_logits[i], shared_logits[i]
        diff = (a - b).abs()
        per_row.append(
            {
                "row": i,
                "prompt_tokens": len(all_ids[i]),
                "max_abs_logit_diff": float(diff.max()),
                "mean_abs_logit_diff": float(diff.mean()),
                "argmax_plain": int(a.argmax()),
                "argmax_shared": int(b.argmax()),
                "argmax_agrees": bool(int(a.argmax()) == int(b.argmax())),
                "logits_bitwise_identical": bool(torch.equal(a, b)),
                "text_identical": plain_text[i] == shared_text[i],
                "plain_text": plain_text[i],
                "shared_text": shared_text[i],
            }
        )

    return {
        "query_position": query_position,
        "attn": attn,
        "rows": len(all_ids),
        "groups": len(groups),
        "reuse": {
            "mean_prompt_tokens": estimate.mean_prompt_tokens,
            "mean_prefix_tokens": estimate.mean_prefix_tokens,
            "fraction": estimate.fraction,
            "best_case_speedup": estimate.speedup,
            "text": str(estimate),
        },
        "prefill_tokens_plain": plain_prefill_tokens,
        "prefill_tokens_shared": shared_prefill_tokens,
        "token_saving": (
            1.0 - shared_prefill_tokens / plain_prefill_tokens if plain_prefill_tokens else 0.0
        ),
        "seconds_plain": round(plain_seconds, 2),
        "seconds_shared": round(shared_seconds, 2),
        "wallclock_speedup": (
            round(plain_seconds / shared_seconds, 3) if shared_seconds else float("nan")
        ),
        "all_text_identical": all(r["text_identical"] for r in per_row),
        "all_argmax_agree": all(r["argmax_agrees"] for r in per_row),
        "all_logits_bitwise_identical": all(r["logits_bitwise_identical"] for r in per_row),
        "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in per_row),
        "per_row": per_row,
    }


def main() -> int:
    """
    :returns: 0 if every probed configuration was text-identical, 1 otherwise.
    """
    args = build_parser().parse_args()

    import ctc.tasks
    from ctc.eval.backends.native import NativeBackend
    from ctc.format import registry

    ctc.tasks.load_all()
    spec = registry.get(args.spec)

    rows = _load_rows(Path(args.data), args.groups, args.rows_per_group)
    print(f"[probe] {len(rows)} rows from {args.data}", flush=True)

    masks = [m.strip() for m in args.attn.split(",") if m.strip()]
    started = time.time()
    # Always load under `full`: NativeBackend's own chunked setup raises TypeError (it calls
    # enable_document_chunk_attention with no ids), so the mask is switched on below instead.
    backend = NativeBackend(
        Path(args.ckpt),
        tokenizer=args.tokenizer,
        attn="full",
        max_length=args.max_length,
        query_position="both",
    )
    print(f"[probe] model loaded in {time.time() - started:.0f}s", flush=True)

    reports = []
    for mask in masks:
        if mask != backend.attn:
            backend.attn = mask
            backend._prefill = None
            if mask == "chunked":
                _enable_chunked(backend.gm.model, args.family)
            else:
                backend._configure_attention()
        for qp in [q.strip() for q in args.query_positions.split(",") if q.strip()]:
            print(f"[probe] attn={mask} query_position={qp} ...", flush=True)
            report = _probe(
                backend, spec, rows, query_position=qp, attn=mask, max_length=args.max_length
            )
            reports.append(report)
            print(
                f"[probe]   reuse={report['reuse']['fraction']:.1%}  "
                f"text_identical={report['all_text_identical']}  "
                f"max|dlogit|={report['max_abs_logit_diff']:.3g}  "
                f"plain {report['seconds_plain']:.1f}s vs shared {report['seconds_shared']:.1f}s "
                f"({report['wallclock_speedup']:.2f}x)",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "ckpt": args.ckpt,
                "data": args.data,
                "spec": args.spec,
                "tokenizer": args.tokenizer,
                "max_length": args.max_length,
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[probe] wrote {out}", flush=True)
    return 0 if all(r["all_text_identical"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
