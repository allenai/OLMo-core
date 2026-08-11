"""
``python -m ctc.eval.validate`` -- reproduce a known number with the ported code.

This is not a general eval entry point (that is ``ctc-eval``). It exists to answer one question:
does the ported pipeline get the same result as the pre-migration one, from the same checkpoint and
the same data? Everything up to this point has been checked against fixtures and fakes; this is the
first time the whole path runs against a real model, and a fixture cannot tell you that prompt
assembly, tokenization, decoding and scoring compose correctly.

It takes the expected value as an argument and reports the deviation, so the run either confirms
the port or gives a number to explain. It never adjusts anything to match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .. import tasks
from ..format import registry
from .runner import EvalConfig, run_task
from .stopping import STOP_PRESETS


def build_parser() -> argparse.ArgumentParser:
    """:returns: The argument parser."""
    ap = argparse.ArgumentParser(
        prog="ctc.eval.validate",
        description="Reproduce a known eval number with the ported pipeline.",
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--attn", default="full", choices=("full", "chunked", "landmark"))
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-0.8B-Base",
        help="HF id or local dir. The tokenizer.json is shared across Qwen3.5 sizes, so any of "
        "them will do; prefer a local directory so the node needs no network.",
    )
    ap.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="override the tokenizer's EOS. Required for Qwen3.5, where the suite driver passes "
        "248044 (a reserved marker id) rather than the tokenizer's own value -- decoding against "
        "the wrong EOS means generation never terminates on its own.",
    )
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--query-position", default="after", choices=("before", "after", "both"))
    ap.add_argument(
        "--stop-preset",
        default=None,
        help="override the task's stop rule; use to isolate a stop-rule divergence",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--expect-f1", type=float, default=None)
    ap.add_argument("--expect-eval-size", type=int, default=None)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="absolute deviation treated as a reproduction (default: 0.001)",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """
    :param argv: Arguments; defaults to ``sys.argv[1:]``.

    :returns: ``0`` if the number reproduced (or none was expected), ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    tasks.load_all()
    spec = registry.get(args.task)

    if args.stop_preset is not None:
        if args.stop_preset not in STOP_PRESETS:
            raise SystemExit(
                f"unknown stop preset {args.stop_preset!r}; have {', '.join(sorted(STOP_PRESETS))}"
            )
        spec = spec.__class__(**{**spec.__dict__, "stop": args.stop_preset})

    from .backends.native import NativeBackend

    backend = NativeBackend(
        args.ckpt,
        tokenizer=args.tokenizer,
        attn=args.attn,
        max_length=args.max_length,
        eos_token_id=args.eos_token_id,
    )
    print(f"[validate] tokenizer={args.tokenizer} eos_id={backend.eos_id}")
    stop = STOP_PRESETS[spec.stop]
    if args.max_new_tokens is not None:
        stop = stop.__class__(**{**stop.__dict__, "max_new_tokens": args.max_new_tokens})

    cfg = EvalConfig(
        ckpt=args.ckpt,
        task=spec,
        rung=args.rung,
        data_path=args.data,
        attn=args.attn,
        backend="native",
        max_length=args.max_length,
        max_new_tokens=stop.max_new_tokens,
        query_position=args.query_position,
        limit=args.limit,
    )

    outcome = run_task(
        cfg,
        lambda prompts: backend.generate(prompts, stop=stop),
        count_tokens=backend.count_tokens,
    )

    print()
    print(outcome.summary())
    for w in outcome.warnings:
        print(f"  ! {w}")
    if args.out:
        print(f"  wrote {outcome.write(args.out)}")

    ok = True
    if args.expect_eval_size is not None and outcome.eval_size != args.expect_eval_size:
        print(
            f"\nEVAL SIZE MISMATCH: expected {args.expect_eval_size}, graded {outcome.eval_size}. "
            "The data file differs from the one that produced the target number; nothing below is "
            "comparable."
        )
        ok = False

    if args.expect_f1 is not None:
        got = outcome.metrics.get("f1")
        if got is None:
            print("\nno f1 in this task's metrics; cannot compare")
            ok = False
        else:
            delta = got - args.expect_f1
            print(
                f"\nexpected f1 {args.expect_f1:.6f}\n"
                f"     got f1 {got:.6f}\n"
                f"      delta {delta:+.6f}  (tolerance ±{args.tolerance})"
            )
            if abs(delta) <= args.tolerance:
                print("REPRODUCED -- the ported pipeline matches the pre-migration number.")
            else:
                ok = False
                print("DEVIATION. Likely causes, in order:")
                print(
                    f"  1. stop rule. This ran stop={spec.stop!r}; the pre-migration contradiction "
                    "run used a rule that never fired, decoding to the budget. Re-run with "
                    "--stop-preset eos to isolate."
                )
                print(f"  2. prompt assembly or tokenization -- check parse_rate "
                      f"({outcome.parse_rate:.3f}) and read the generations.")
                print("  3. attention mode -- confirm the target row's arm matches --attn.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
