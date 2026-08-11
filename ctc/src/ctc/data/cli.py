"""
``ctc-data`` -- build task data: generate, ladder, audit, write.

The audit is not a separate step you remember to run. ``build`` refuses to write data that fails
one, because every check it runs corresponds to a defect that already reached a results table once.
``--force`` exists for the case where you know better, and says so in the output.

Tokenizing task JSONL into olmo-core training shards is deliberately *not* here: it writes
olmo-core's format and belongs on the training side, in ``src/scripts/ctc/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..format import registry
from ..format import rungs as rung_util
from ..tasks import load_all
from . import audit as audit_mod
from . import build as build_mod
from . import ladders
from .generators import base as generators
from .io import load_jsonl, save_jsonl

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """:returns: The ``ctc-data`` argument parser."""
    ap = argparse.ArgumentParser(
        prog="ctc-data",
        description="Generate, ladder and audit corpus-reasoning task data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctc-data list\n"
            "  ctc-data build --task cycle --out /data/ctc/v3\n"
            "  ctc-data build --task groups4 --rungs 2k,4k --train 5000 --out /data/ctc/pilot\n"
            "  ctc-data audit --task cycle --dir /data/ctc/v3\n"
        ),
    )
    sub = ap.add_subparsers(dest="command", metavar="command")
    sub.add_parser("list", help="list ported generators and their ladders")

    p = sub.add_parser("build", help="generate train + a nested eval ladder, then audit")
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--rungs", default="all", help="'all' (default) or e.g. '2k,4k,8k'")
    p.add_argument("--train", type=int, default=20_000, help="training examples (default: 20000)")
    p.add_argument("--eval-size", type=int, default=500, help="examples per rung (default: 500)")
    p.add_argument("--seed", type=int, default=42, help="train seed (default: 42)")
    p.add_argument("--eval-seed", type=int, default=7, help="eval seed (default: 7)")
    p.add_argument(
        "--force", action="store_true", help="write even if the audit fails (says so in the output)"
    )

    a = sub.add_parser("audit", help="re-run the integrity and shortcut checks over built data")
    a.add_argument("--task", required=True)
    a.add_argument("--dir", required=True)
    return ap


def _list() -> int:
    print(f"{len(generators.names())} generator(s) ported:\n")
    for name in generators.names():
        gen = generators.get(name)
        ladder = ", ".join(f"{r}={ladders.docs_for_rung(name, r)}" for r in ladders.rungs_for(name))
        print(f"  {name:<12} {gen.notes}")
        print(f"  {'':<12} docs per rung: {ladder}")
    print("\nUn-ported tasks keep their build recipe in the pre-migration BUILD_MATRIX.md.")
    return 0


def _resolve_rungs(task: str, spelled: str) -> Optional[List[str]]:
    if spelled == "all":
        return None
    return rung_util.sort_rungs(r for r in spelled.split(",") if r.strip())


def _build(args: argparse.Namespace) -> int:
    load_all()
    spec = registry.get(args.task)
    rungs = _resolve_rungs(args.task, args.rungs)
    out = Path(args.out) / args.task

    evalset, eval_report = build_mod.build_eval(
        args.task, spec, size=args.eval_size, seed=args.eval_seed, rungs=rungs
    )
    print(eval_report.summary())
    train, train_report = build_mod.build_train(
        args.task,
        spec,
        total=args.train,
        seed=args.seed,
        rungs=rungs,
        eval_examples=[ex for rows in evalset.values() for ex in rows],
    )
    print(train_report.summary())

    result = audit_mod.audit(args.task, spec, train=train, rungs=evalset)
    print("\naudit:")
    print(result.report())
    if not result.ok and not args.force:
        print("\nrefusing to write: fix the findings above, or pass --force", file=sys.stderr)
        return 1

    save_jsonl(out / "train.jsonl", train)
    for label, rows in evalset.items():
        save_jsonl(out / f"eval_{label}.jsonl", rows)
    print(f"\nwrote {len(evalset) + 1} file(s) to {out}")
    if not result.ok:
        print("!! written with --force despite a failing audit")
    return 0


def _audit(args: argparse.Namespace) -> int:
    load_all()
    spec = registry.get(args.task)
    directory = Path(args.dir)
    if not (directory / args.task).is_dir() and (directory / "train.jsonl").exists():
        root = directory
    else:
        root = directory / args.task

    train_path = root / "train.jsonl"
    train = load_jsonl(train_path) if train_path.exists() else []
    evalset = {
        p.stem.removeprefix("eval_"): load_jsonl(p) for p in sorted(root.glob("eval_*.jsonl"))
    }
    if not train and not evalset:
        print(f"no train.jsonl or eval_*.jsonl under {root}", file=sys.stderr)
        return 1
    ordered = {label: evalset[label] for label in rung_util.sort_rungs(evalset)}

    result = audit_mod.audit(args.task, spec, train=train, rungs=ordered)
    print(result.report())
    return 0 if result.ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    """
    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help()
        return 0
    return {"list": lambda a: _list(), "build": _build, "audit": _audit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
