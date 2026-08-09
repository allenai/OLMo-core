"""
``ctc-eval`` -- grade a checkpoint on one or more tasks.

One command for every backend. ``--backend`` selects how text gets generated; nothing else about
the run changes, because the prompt, the parser and the scorer come from :mod:`ctc.format` and are
shared. A score that moves when only ``--backend`` moves is a bug in a backend, and the parity test
in ``ctc/tests/eval`` exists to catch exactly that.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..format import registry, rungs as rung_util
from .backends import base as backends


def build_parser() -> argparse.ArgumentParser:
    """:returns: The ``ctc-eval`` argument parser."""
    ap = argparse.ArgumentParser(
        prog="ctc-eval",
        description="Grade a checkpoint on corpus-reasoning tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctc-eval --list-tasks\n"
            "  ctc-eval --ckpt /data/ckpts/q35-4b/step1100 --task contradiction --rungs 2k\n"
            "  ctc-eval --ckpt … --suite ctc_suite --rungs all --backend vllm --attn chunked\n"
        ),
    )

    what = ap.add_argument_group("what to run")
    what.add_argument("--ckpt", help="checkpoint directory to grade")
    what.add_argument("--task", help="single task name (see --list-tasks)")
    what.add_argument("--suite", help="named suite from configs/suites/<name>.yaml")
    what.add_argument(
        "--rungs",
        default="all",
        help=(
            "'all' (default) for every rung the task defines, or a comma list of labels like "
            "'2k,8k,32k'. There is no fixed set: each task declares its own ladder in "
            "configs/tasks/<task>.yaml, ranging from nq (2k-8k) to the xlong ladders (up to 512k)."
        ),
    )

    how = ap.add_argument_group("how to run it")
    how.add_argument(
        "--backend",
        default=None,
        choices=sorted(backends.BACKENDS),
        help="generation backend (default: the fastest one installed; see --list-backends)",
    )
    how.add_argument(
        "--attn",
        default="full",
        choices=("full", "chunked", "landmark"),
        help=(
            "attention mask to grade under (default: full). One vocabulary across the whole "
            "pipeline -- the old driver inverted these names against the evaluator's, which is a "
            "trap this CLI deliberately does not reproduce."
        ),
    )
    how.add_argument("--batch-size", type=int, default=1, help="prompts per forward (default: 1)")
    how.add_argument("--max-new-tokens", type=int, default=None, help="override the task default")
    how.add_argument("--limit", type=int, default=None, help="grade only the first N examples")

    out = ap.add_argument_group("output")
    out.add_argument("--out", default=None, help="results directory (default: ./results)")
    out.add_argument(
        "--no-dump-generations",
        action="store_true",
        help=(
            "skip writing raw generations. Not recommended: every grading bug we have found was "
            "found by reading generations for a task that scored near zero."
        ),
    )

    info = ap.add_argument_group("information")
    info.add_argument("--list-tasks", action="store_true", help="list registered tasks and exit")
    info.add_argument(
        "--list-backends", action="store_true", help="list backends and availability, then exit"
    )
    return ap


def _list_tasks() -> int:
    names = registry.names()
    if not names:
        print("No tasks registered yet.")
        print("Tasks land in ctc/src/ctc/eval/tasks/ during the port; see records/ for the plan.")
        return 0
    print(f"{len(names)} registered task(s):\n")
    for name in names:
        spec = registry.get(name)
        print(f"  {name:<18} {spec.description}")
        print(f"  {'':<18} gold ids are {spec.gold_index_base}-based")
    return 0


def _list_backends() -> int:
    print("backends:\n")
    for line in backends.describe():
        print(line)
    installed = backends.available()
    print(
        f"\ndefault: {installed[0]!r}"
        if installed
        else "\nNo backend installed. Add one, e.g. pip install 'ctc[vllm]'."
    )
    return 0


def _resolve_rungs(spec: str, defined: Optional[List[str]] = None) -> List[str]:
    """
    Expand the ``--rungs`` value.

    Rungs are not a fixed global set -- each task declares its own ladder. So ``"all"`` can only be
    answered once we know the task, and an explicit list is validated against what that task
    actually has rather than against a hardcoded tuple.

    :param spec: ``"all"`` or a comma list of rung labels.
    :param defined: Rungs this task/suite declares. ``None`` while task configs are not yet loaded,
        in which case explicit labels are accepted on syntax alone and ``"all"`` cannot be resolved.

    :returns: Canonical rung labels, ascending by context length.

    :raises SystemExit: On a malformed label, or one the task does not define.
    """
    if spec == "all":
        if defined is None:
            return []
        return rung_util.sort_rungs(defined)

    try:
        chosen = rung_util.sort_rungs(r for r in spec.split(",") if r.strip())
    except ValueError as e:
        raise SystemExit(str(e)) from None

    if defined is not None:
        have = set(rung_util.sort_rungs(defined))
        missing = [r for r in chosen if r not in have]
        if missing:
            raise SystemExit(
                f"rung(s) {', '.join(missing)} are not defined for this task; "
                f"it has {', '.join(rung_util.sort_rungs(defined))}"
            )
    return chosen


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the ``ctc-eval`` console script.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_tasks:
        return _list_tasks()
    if args.list_backends:
        return _list_backends()

    if not args.ckpt:
        ap.error("--ckpt is required (or use --list-tasks / --list-backends)")
    if bool(args.task) == bool(args.suite):
        ap.error("pass exactly one of --task or --suite")

    # `defined=None` until configs/tasks/*.yaml loading lands with the runner (port task #3); at
    # that point the task's own ladder is passed in and both "all" and validation become real.
    rungs = _resolve_rungs(args.rungs, defined=None)

    backend_name = args.backend
    if backend_name is None:
        installed = backends.available()
        if not installed:
            raise SystemExit(
                "no eval backend is installed.\n"
                "  pip install 'ctc[vllm]'    # fastest\n"
                "  pip install 'ctc[hf]'      # widest coverage\n"
                "  pip install 'ctc[native]'  # grade an olmo-core checkpoint directly"
            )
        backend_name = installed[0]

    # The runner lands in task #3 of the port. Until then, report the resolved plan rather than
    # pretending to grade -- a CLI that silently no-ops is worse than one that says what is missing.
    print(f"plan: ckpt={args.ckpt}")
    print(f"      {'suite' if args.suite else 'task'}={args.suite or args.task}")
    print(
        f"      rungs={','.join(rungs) if rungs else 'all (resolved from the task config)'}"
        f"  backend={backend_name}  attn={args.attn}"
    )
    if not registry.names():
        print("\nNo tasks are registered yet, so there is nothing to grade.")
        print("Next step in the port: ctc/src/ctc/format/ (task #2), then the runner (task #3).")
        return 1
    raise SystemExit("runner not implemented yet (port task #3)")


if __name__ == "__main__":
    sys.exit(main())
