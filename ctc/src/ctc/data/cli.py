"""
``ctc-data`` -- build task data: generate, ladder, audit, write.

One command per task, and the same task names ``ctc-eval`` takes::

    ctc-data build --task nq --out /data/ctc/v3

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
            "  ctc-data build --task nq --split eval --out /data/ctc/v3\n"
            "  ctc-data build --task groups4 --rungs 2k,4k --train 5000 --out /data/ctc/pilot\n"
            "  ctc-data build --task contradiction -C pairs_path=pairs.jsonl --out /data/ctc/v3\n"
            "  ctc-data build --task cycle --split eval --rungs 64k,1m,10m \\\n"
            "      --eval-size 125 --allow-small-eval --out /data/ctc/xlong\n"
            "  ctc-data audit --task cycle --dir /data/ctc/v3\n"
        ),
    )
    sub = ap.add_subparsers(dest="command", metavar="command")
    sub.add_parser("list", help="list ported generators, their ladders and their parameters")

    p = sub.add_parser("build", help="generate train + an eval ladder, then audit")
    p.add_argument("--task", required=True, help=f"one of: {', '.join(generators.names())}")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--split",
        choices=("both", "train", "eval"),
        default="both",
        help="what to build (default: both; held-out ladders are eval-only)",
    )
    p.add_argument(
        "--rungs",
        default="all",
        help=(
            "'all' (default: the calibrated 2k-32k ladder) or e.g. '2k,4k,8k'. Rungs beyond the "
            "table ('64k', '1m', '10m', ...) extrapolate their document count from the task's own "
            "calibration fit; the build report flags them"
        ),
    )
    p.add_argument("--train", type=int, default=20_000, help="training examples (default: 20000)")
    p.add_argument("--eval-size", type=int, default=500, help="examples per rung (default: 500)")
    p.add_argument(
        "--allow-small-eval",
        action="store_true",
        help=(
            "permit --eval-size below the 500 floor -- for ultra-long rungs where 500 examples "
            "is gigabytes of JSONL (the shipped suite holds 125 at >=256k). The report and every "
            "quoted number must carry the size and its error bar"
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="train seed (default: 42)")
    p.add_argument("--eval-seed", type=int, default=7, help="eval seed (default: 7)")
    p.add_argument(
        "-C",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override one generator or corpus parameter; repeatable "
            "(e.g. -C num_pairs=5 -C pairs_path=pairs.jsonl). "
            "`ctc-data list` prints what each task accepts"
        ),
    )
    p.add_argument(
        "--force", action="store_true", help="write even if the audit fails (says so in the output)"
    )

    a = sub.add_parser("audit", help="re-run the integrity and shortcut checks over built data")
    a.add_argument("--task", required=True)
    a.add_argument("--dir", required=True)
    return ap


def _coerce(text: str) -> Any:
    """
    :param text: A ``-C`` value.

    :returns: It as a bool, ``None``, int, float or string. Typed rather than always-string,
        because a generator that silently received ``"3"`` where it expected ``3`` would build data
        at a size nobody asked for.
    """
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _split_overrides(
    generator: generators.Generator, pairs: Sequence[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Route ``key=value`` overrides to the per-example config or to the corpus loader.

    :param generator: The target generator.
    :param pairs: Raw ``KEY=VALUE`` strings.

    :returns: ``(build overrides, corpus overrides)``.

    :raises SystemExit: On a malformed pair, or a key neither side accepts. A typo must not be
        silently ignored and then produce data at the default size.
    """
    build_overrides: Dict[str, Any] = {}
    corpus_overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        key = key.strip()
        if not separator:
            raise SystemExit(f"-C expects KEY=VALUE, got {pair!r}")
        if key in generator.defaults:
            build_overrides[key] = _coerce(value)
        elif key in generator.corpus_defaults:
            corpus_overrides[key] = _coerce(value)
        else:
            accepted = sorted(set(generator.defaults) | set(generator.corpus_defaults))
            raise SystemExit(f"{generator.name} has no parameter {key!r}; accepts {accepted}")
    return build_overrides, corpus_overrides


def _list() -> int:
    print(f"{len(generators.names())} generator(s) ported:\n")
    for name in generators.names():
        gen = generators.get(name)
        unit = "tokens" if gen.scaling_param != "num_docs" else "docs"
        ladder = ", ".join(f"{r}={ladders.docs_for_rung(name, r)}" for r in ladders.rungs_for(name))
        tags = []
        if gen.eval_only:
            tags.append("HELD OUT: eval only")
        if gen.corpus is not None:
            tags.append("needs a corpus")
        if not gen.nested_ladder:
            tags.append("rungs built independently")
        suffix = f"  [{'; '.join(tags)}]" if tags else ""
        print(f"  {name:<15} graded by `{gen.task}` -- {gen.notes}{suffix}")
        print(f"  {'':<15} {unit} per rung: {ladder} (labels past 32k extrapolate the fit)")
        if name in ladders.CEILINGS:
            top, reason = ladders.CEILINGS[name]
            print(f"  {'':<15} CEILING {top}: {reason}")
        if name in ladders.SUPPLY_BOUNDED:
            print(f"  {'':<15} supply-bounded: {ladders.SUPPLY_BOUNDED[name]}")
        print(f"  {'':<15} -C build:  {', '.join(sorted(gen.defaults)) or '(none)'}")
        if gen.corpus_defaults:
            print(f"  {'':<15} -C corpus: {', '.join(sorted(gen.corpus_defaults))}")
        print()
    print("Un-ported tasks keep their build recipe in the pre-migration BUILD_MATRIX.md.")
    return 0


def _resolve_rungs(spelled: str) -> Optional[List[str]]:
    if spelled == "all":
        return None
    return rung_util.sort_rungs(r for r in spelled.split(",") if r.strip())


def _build(args: argparse.Namespace) -> int:
    load_all()
    generator = generators.get(args.task)
    spec = registry.get(generator.task)
    rungs = _resolve_rungs(args.rungs)
    out = Path(args.out) / args.task
    build_overrides, corpus_overrides = _split_overrides(generator, args.overrides)

    split = args.split
    if split != "eval" and generator.eval_only:
        if split == "train":
            print(
                f"{args.task} is a held-out ladder and has no training split: training on it "
                "would make every number from it in-distribution",
                file=sys.stderr,
            )
            return 1
        split = "eval"
        print(f"note: {args.task} is held out; building the eval ladder only")

    # Loaded once and handed to both halves. Loading twice would re-shuffle the pool and silently
    # give train and eval overlapping views of it.
    corpus = generator.load_corpus(**corpus_overrides)

    evalset: Dict[str, List[Dict[str, Any]]] = {}
    if split in ("both", "eval"):
        evalset, eval_report = build_mod.build_eval(
            args.task,
            spec,
            size=args.eval_size,
            seed=args.eval_seed,
            rungs=rungs,
            config=build_overrides,
            corpus=corpus,
            allow_small=args.allow_small_eval,
        )
        print(eval_report.summary())

    train: List[Dict[str, Any]] = []
    if split in ("both", "train"):
        train, train_report = build_mod.build_train(
            args.task,
            spec,
            total=args.train,
            seed=args.seed,
            rungs=rungs,
            config=build_overrides,
            eval_examples=[ex for rows in evalset.values() for ex in rows],
            corpus=corpus,
        )
        print(train_report.summary())

    result = audit_mod.audit(args.task, spec, train=train, rungs=evalset)
    print("\naudit:")
    print(result.report())
    if not result.ok and not args.force:
        print("\nrefusing to write: fix the findings above, or pass --force", file=sys.stderr)
        return 1

    written = 0
    if train:
        save_jsonl(out / "train.jsonl", train)
        written += 1
    for label, rows in evalset.items():
        save_jsonl(out / f"eval_{label}.jsonl", rows)
        written += 1
    print(f"\nwrote {written} file(s) to {out}")
    if not result.ok:
        print("!! written with --force despite a failing audit")
    return 0


def _audit(args: argparse.Namespace) -> int:
    load_all()
    spec = registry.get(generators.get(args.task).task)
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
