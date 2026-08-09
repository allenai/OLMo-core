"""
``ctc-data`` -- build task data, from raw source to staged shards.

The pipeline has five stages and ``build`` runs them in order:

``generate``
    Source corpora -> task JSONL. One generator per task family.
``ladder``
    Task JSONL -> per-rung files sized to a token budget. The rung->document-count fit lives in
    ``configs/tasks/<task>.yaml``.
``audit``
    Integrity checks. **Not optional, and not a debugging step.** Every check here corresponds to a
    bug that reached published numbers and was first mistaken for a modelling result: filler text
    leaking across examples, chunk-layout drift between train and eval, doc-id digit ranges that
    differ between train and eval, gold indices counted from the wrong base, marker embeddings that
    are mutually indistinguishable or off-distribution in norm. ``build`` refuses to stage data that
    fails an audit.
``tokenize`` / ``stage``
    These write olmo-core shards, so they live in ``src/scripts/ctc/`` on the training side, not in
    this package. ``ctc-data build --stage`` shells out to them when they are present.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..format import rungs as rung_util

STAGES = ("generate", "ladder", "audit", "tokenize", "stage")


def build_parser() -> argparse.ArgumentParser:
    """:returns: The ``ctc-data`` argument parser."""
    ap = argparse.ArgumentParser(
        prog="ctc-data",
        description="Generate, ladder, audit and stage corpus-reasoning task data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctc-data list\n"
            "  ctc-data build --task contradiction --rungs 2k,4k,8k --out /data/ctc/v3\n"
            "  ctc-data build --suite ctc_suite --stage weka\n"
            "  ctc-data audit --task contradiction --dir /data/ctc/v3\n"
        ),
    )
    sub = ap.add_subparsers(dest="command", metavar="command")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task", help="single task name (see `ctc-data list`)")
        p.add_argument("--suite", help="named suite from configs/suites/<name>.yaml")
        p.add_argument("--out", help="output directory")
        p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")

    p_build = sub.add_parser("build", help="run the full pipeline for a task or suite")
    _common(p_build)
    p_build.add_argument(
        "--rungs",
        default="all",
        help="'all' (default) or a comma list like '2k,8k,32k'; the task config defines its ladder",
    )
    p_build.add_argument(
        "--stages",
        default="generate,ladder,audit",
        help=(
            f"comma list from {{{','.join(STAGES)}}} (default: generate,ladder,audit). "
            "tokenize/stage need the training-side tooling."
        ),
    )
    p_build.add_argument(
        "--stage",
        choices=("local", "weka", "s3"),
        default="local",
        help="where staged shards go (default: local)",
    )

    p_audit = sub.add_parser("audit", help="run integrity checks over already-built data")
    _common(p_audit)
    p_audit.add_argument("--dir", help="directory of built data to check")

    sub.add_parser("list", help="list registered task generators")
    return ap


def _list() -> int:
    try:
        from .generators import base as gen_base  # noqa: F401

        names = gen_base.names()  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        names = []
    if not names:
        print("No generators registered yet.")
        print("They land in ctc/src/ctc/data/generators/ during the port; see records/ for order.")
        return 0
    print(f"{len(names)} generator(s): {', '.join(names)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the ``ctc-data`` console script.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.command is None:
        ap.print_help()
        return 0
    if args.command == "list":
        return _list()

    if bool(getattr(args, "task", None)) == bool(getattr(args, "suite", None)):
        ap.error("pass exactly one of --task or --suite")

    if args.command == "build":
        unknown = [s for s in args.stages.split(",") if s.strip() and s.strip() not in STAGES]
        if unknown:
            ap.error(f"unknown stage(s) {unknown}; choose from {', '.join(STAGES)}")
        if args.rungs != "all":
            try:
                rung_util.sort_rungs(r for r in args.rungs.split(",") if r.strip())
            except ValueError as e:
                raise SystemExit(str(e)) from None

    print(f"plan: {args.command} {'suite' if args.suite else 'task'}="
          f"{args.suite or args.task}")
    print("\nGenerators are not ported yet, so there is nothing to build.")
    print("Next step in the port: ctc/src/ctc/format/ (task #2).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
