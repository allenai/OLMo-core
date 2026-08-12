"""
``ctc-eval`` -- grade a checkpoint on the CTC suite.

One command, one checkpoint, any number of tasks and rungs::

    ctc-eval --ckpt /weka/.../step1100 --tasks main --attn chunked

``--tasks main`` is the five in-distribution ladders; ``ood`` is the four held-out ones; ``all`` is
both. Which file grades which task at which rung is not something a caller supplies -- it lives in
:mod:`ctc.eval.bundles`, because every place that mapping was retyped is a place it drifted.

``--backend`` selects how text gets generated and nothing else about the run changes: the prompt,
the parser and the scorer come from :mod:`ctc.format` and are shared. A score that moves when only
``--backend`` moves is a backend bug.

Defaults are chosen so the failure modes that have actually cost this project GPU time are errors
rather than plausible numbers:

* an over-long prompt **stops the run** instead of being skipped and scored 0.0;
* a checkpoint whose training format does not match the eval format is **reported**, not assumed
  compatible;
* every result carries ``eval_size``, a standard error, and its ``parse_rate``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..format import registry
from . import bundles
from .backends import base as backends
from .runner import EvalConfig, run_task

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """:returns: The ``ctc-eval`` argument parser."""
    ap = argparse.ArgumentParser(
        prog="ctc-eval",
        description="Grade a checkpoint on the CTC suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctc-eval --list-tasks\n"
            "  ctc-eval --ckpt /weka/.../step1100 --tasks main\n"
            "  ctc-eval --ckpt /weka/.../step1100 --tasks main --attn chunked\n"
            "  ctc-eval --ckpt /weka/.../step1100 --tasks contradiction --rungs 2k,8k\n"
        ),
    )

    what = ap.add_argument_group("what to run")
    what.add_argument("--ckpt", help="checkpoint directory (the one holding model_and_optim)")
    what.add_argument(
        "--tasks",
        default="main",
        help="'main' (5 in-distribution), 'ood' (4 held-out), 'all', or a comma list of task names",
    )
    what.add_argument(
        "--rungs",
        default="all",
        help="'all' (default, the 2k-32k ladder), 'xlong' (64k-2M, opt-in because one 256k rung is "
        "hours per task), or a comma list like '2k,8k' / '64k,256k'. Checked against the task and "
        "bundle rather than against a fixed set.",
    )
    what.add_argument(
        "--bundle",
        default=None,
        help="eval bundle: a registered name (see --list-bundles) or a directory path. "
        f"Default: $CTC_EVAL_BUNDLE, else {bundles.DEFAULT_BUNDLE}.",
    )

    how = ap.add_argument_group("how to run it")
    how.add_argument(
        "--backend",
        default=None,
        choices=sorted(backends.BACKENDS),
        help="generation backend (default: the fastest installed; see --list-backends)",
    )
    how.add_argument(
        "--attn",
        default="full",
        choices=("full", "chunked", "landmark"),
        help="attention mask to grade under (default: full). Must match how the checkpoint was "
        "trained; --attn full forces plain causal even on a checkpoint carrying the mask.",
    )
    how.add_argument(
        "--tokenizer", default="Qwen/Qwen3-4B", help="tokenizer id or path (default Qwen/Qwen3-4B)"
    )
    how.add_argument(
        "--max-length",
        type=int,
        default=40960,
        help="total sequence budget, prompt + generation (default 40960). Rung labels bound corpus "
        "size, not prompt length, so this is sized from the model and not from --rungs.",
    )
    how.add_argument(
        "--query-position",
        default="both",
        choices=("both", "before", "after"),
        help="where the question is rendered (default both). MUST match training: these "
        "checkpoints trained with 'both', and 'after' collapses them (nq 0.860 -> 0.074).",
    )
    how.add_argument(
        "--share-prefix",
        action="store_true",
        help="prefill each corpus group's shared token prefix once and reuse its KV. Only does "
        "anything on a fast bundle, whose rows carry corpus_id; the reuse is measured and printed "
        "rather than assumed, and falls back to plain prefills when there is nothing to share.",
    )
    how.add_argument("--mem-freq", type=int, default=63, help="(landmark) block = mem-freq + 1")
    how.add_argument("--limit", type=int, default=None, help="grade only the first N per rung")
    how.add_argument(
        "--allow-truncated",
        action="store_true",
        help="continue when prompts exceed --max-length, counting them. Off by default: a skipped "
        "prompt scores a clean 0.0, which is indistinguishable from a wrong answer.",
    )
    how.add_argument(
        "--ignore-format-fingerprint",
        action="store_true",
        help="grade even when the checkpoint's training format does not match or was never "
        "recorded. Results then record that compatibility was UNVERIFIED.",
    )
    how.add_argument(
        "--keep-going",
        action="store_true",
        help="on a failed rung, record the error and continue to the next one instead of stopping. "
        "Useful for an overnight sweep; the summary lists what failed.",
    )

    out = ap.add_argument_group("output")
    out.add_argument("--out", default="results", help="results directory (default ./results)")
    out.add_argument(
        "--tag", default="", help="extra label in the result filenames, e.g. the run name"
    )
    out.add_argument(
        "--no-dump-generations",
        action="store_true",
        help="skip writing raw generations. Not recommended: every grading bug found in this "
        "project was found by reading generations for a task that scored near zero.",
    )

    info = ap.add_argument_group("information")
    info.add_argument("--list-tasks", action="store_true", help="list ladders and rungs, then exit")
    info.add_argument("--list-bundles", action="store_true", help="list eval bundles, then exit")
    info.add_argument(
        "--list-backends", action="store_true", help="list backends and availability, then exit"
    )
    info.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve every rung to a file and report what would run, without loading the model. "
        "Checks the bundle paths exist -- run it once before a long sweep.",
    )
    return ap


def _warn_small_evals(todo: List[Dict[str, Any]]) -> None:
    """
    Print the sub-500 warning for any ladder in the plan, once per task.

    Said **before** the run as well as beside each number, because the point of the warning is to be
    read by whoever quotes the result, and the person launching is usually that person.

    :param todo: The resolved plan.
    """
    for name in dict.fromkeys(item["task"] for item in todo):
        warning = bundles.get(name).small_eval_warning
        if warning:
            print(f"[ctc-eval] ⚠ {name}: {warning}")


def _list_tasks() -> int:
    print(f"eval bundle: {bundles.bundle_root()}\n")
    for line in bundles.describe():
        print(line)
    print("\ngroups: " + ", ".join(f"{k} ({len(v)} tasks)" for k, v in bundles.GROUPS.items()))
    return 0


def _list_bundles() -> int:
    print("eval bundles:\n")
    for name, bundle in bundles.BUNDLES.items():
        mark = "  (default)" if name == bundles.DEFAULT_BUNDLE else ""
        print(f"  {name:<10} [{bundle.kind}]{mark}")
        print(f"  {'':<10} {bundle.root}")
        if bundle.description:
            print(f"  {'':<10} {bundle.description}")
        if bundle.xlong:
            have = ", ".join(sorted(bundle.xlong, key=str))
            print(f"  {'':<10} ultra-long rungs for: {have}   (--rungs xlong)")
        print()
    print(
        "A path is accepted too, for a staged local copy.\n"
        "Bundles are NOT interchangeable: the same rung label maps to different files with "
        "different corpus sizes,\nso a number is only comparable to another from the same bundle."
    )
    return 0


def _list_backends() -> int:
    print("backends:\n")
    for line in backends.describe():
        print(line)
    installed = backends.available()
    print(
        f"\ndefault: {installed[0]!r}"
        if installed
        else "\nNo backend installed. Add one, e.g. pip install 'ctc[native]'."
    )
    return 0


def plan(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Resolve the arguments into a list of rungs to grade.

    Done up front, before the model is loaded, so a typo in ``--tasks`` or a missing bundle file
    fails in seconds rather than after a multi-minute checkpoint load.

    :param args: Parsed arguments.

    :returns: One entry per rung, each with its task, label and file.

    :raises SystemExit: On an unknown task or rung.
    """
    try:
        names = bundles.names(args.tasks)
    except KeyError as e:
        raise SystemExit(str(e)) from None

    out: List[Dict[str, Any]] = []
    for name in names:
        entry = bundles.get(name)
        try:
            resolved = bundles.resolve(name, args.rungs, root=args.bundle)
        except KeyError as e:
            # A task legitimately may not have every requested rung -- rerank has no 32k. Skip it
            # with a note rather than failing the whole sweep, but say so: a silently missing row
            # reads as "scored nothing", not as "never ran".
            if args.rungs != "all":
                print(f"[ctc-eval] skipping {name}: {e}", file=sys.stderr)
                continue
            raise SystemExit(str(e)) from None
        for label, path in resolved:
            out.append({"task": name, "spec": entry.spec, "rung": label, "path": path})
    if not out:
        raise SystemExit("nothing to run: no task matched --tasks/--rungs")
    return out


def _result_path(out_dir: Path, item: Dict[str, Any], attn: str, tag: str) -> Path:
    stem = f"{item['task']}_{item['rung']}_{attn}"
    if tag:
        stem = f"{stem}_{tag}"
    return out_dir / f"{stem}.json"


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the ``ctc-eval`` console script.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status -- non-zero if any rung failed.
    """
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_tasks:
        return _list_tasks()
    if args.list_bundles:
        return _list_bundles()
    if args.list_backends:
        return _list_backends()

    import ctc.tasks

    ctc.tasks.load_all()

    todo = plan(args)
    bundle = bundles.get_bundle(args.bundle)
    root = Path(bundle.root)
    out_dir = Path(args.out)

    print(f"[ctc-eval] bundle {bundle.name} [{bundle.kind}]  {root}")
    print(
        f"[ctc-eval] {len(todo)} rung(s) over {len({i['task'] for i in todo})} task(s), "
        f"attn={args.attn}, query_position={args.query_position}"
    )

    if args.dry_run:
        missing = 0
        for item in todo:
            exists = item["path"].exists()
            missing += not exists
            print(
                f"  {'ok     ' if exists else 'MISSING'} {item['task']:<16} "
                f"{item['rung']:<5} {item['path']}"
            )
        if missing:
            print(
                f"\n{missing}/{len(todo)} rung file(s) missing. On Beaker these live on the weka "
                "mount, so a local dry run reporting MISSING is expected."
            )
        _warn_small_evals(todo)
        return 1 if missing else 0

    _warn_small_evals(todo)

    if not args.ckpt:
        ap.error("--ckpt is required (or use --list-tasks / --list-backends / --dry-run)")

    backend_name = args.backend
    if backend_name is None:
        installed = backends.available()
        if not installed:
            raise SystemExit(
                "no eval backend is installed.\n"
                "  pip install 'ctc[native]'  # grade an olmo-core checkpoint directly\n"
                "  pip install 'ctc[vllm]'    # fastest\n"
                "  pip install 'ctc[hf]'      # widest coverage"
            )
        backend_name = installed[0]

    print(f"[ctc-eval] loading {args.ckpt} on the {backend_name} backend ...", flush=True)
    started = time.time()
    extra = {}
    if args.share_prefix:
        # Only the native backend implements KV reuse. hf and vllm forward unknown kwargs straight
        # into `from_pretrained` / `LLM(...)`, so passing it to them would surface as an unrelated
        # loader error rather than as "that backend cannot do this".
        if backend_name != "native":
            raise SystemExit(
                f"--share-prefix is implemented on the native backend only, not {backend_name!r}."
            )
        extra["share_prefix"] = True

    backend = backends.load(
        backend_name,
        ckpt=args.ckpt,
        tokenizer=args.tokenizer,
        attn=args.attn,
        max_length=args.max_length,
        query_position=args.query_position,
        mem_freq=args.mem_freq,
        **extra,
    )
    print(f"[ctc-eval] loaded in {time.time() - started:.0f}s", flush=True)

    summaries: List[str] = []
    failures: List[str] = []

    for i, item in enumerate(todo, 1):
        spec = registry.get(item["spec"])
        label = f"{item['task']}@{item['rung']}"
        print(f"\n[ctc-eval] ({i}/{len(todo)}) {label}  {item['path']}", flush=True)
        rung_started = time.time()

        cfg = EvalConfig(
            ckpt=Path(args.ckpt),
            task=spec,
            rung=item["rung"],
            data_path=item["path"],
            attn=args.attn,
            backend=backend_name,
            max_length=args.max_length,
            query_position=args.query_position,
            limit=args.limit,
            allow_truncated=args.allow_truncated,
            ignore_fingerprint=args.ignore_format_fingerprint,
            dump_generations=not args.no_dump_generations,
        )
        try:
            outcome = run_task(
                cfg,
                lambda prompts, examples: backend.generate(
                    prompts,
                    examples,
                    stop=_stop_for(spec),
                    # The ladder name and the grading spec differ for the OOD tasks --
                    # contra_fever is graded by the contradiction spec -- and the prefill
                    # dispatches on the SPEC, which is what the model was trained on.
                    task=spec.name,
                ),
                count_tokens=backend.count_tokens,
            )
        except Exception as e:  # noqa: BLE001 - a failed rung must not lose the finished ones
            failures.append(f"{label}: {type(e).__name__}: {e}")
            print(f"[ctc-eval] FAILED {label}: {type(e).__name__}: {e}", file=sys.stderr)
            if not args.keep_going:
                traceback.print_exc()
                raise SystemExit(
                    f"{label} failed. Fix it, or pass --keep-going to record the failure and "
                    "continue the sweep."
                ) from None
            continue

        path = _result_path(out_dir, item, args.attn, args.tag)
        # The ladder name, which for the OOD tasks is not the spec name. Written so a results table
        # can tell contra_fever from contradiction -- they are graded identically and would
        # otherwise collide in the same row.
        payload = outcome.to_dict()
        payload["ladder"] = item["task"]
        payload["spec"] = spec.name
        payload["seconds"] = round(time.time() - rung_started, 1)
        # Which eval data produced this. The same rung label maps to different files with different
        # corpus sizes across bundles -- contradiction's 64k is n=1602 in v2 and n=1525 in v2_clean
        # -- so a number without its bundle cannot be compared to another number.
        payload["bundle"] = bundle.name
        payload["bundle_root"] = str(root)
        payload["bundle_kind"] = bundle.kind
        # Whether KV was reused across the corpus group. Recorded because reuse is supposed to be
        # score-preserving, which is only checkable after the fact if the results say which path
        # produced them.
        payload["share_prefix"] = bool(args.share_prefix)
        # What results-hub reads to tag the row without anyone typing it. `ladder_version` is the
        # coarse eval_version facet (v2 vs fast); the exact bundle goes in the pointer, because
        # v2 and v2_clean are different data under one facet value.
        payload["_meta"] = {
            "ladder_version": "fast" if bundle.kind == "fast" else "v2",
            "eval_bundle": str(root),
            "query_position": args.query_position,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        line = outcome.summary().replace(spec.name, item["task"], 1)
        summaries.append(line)
        print(f"[ctc-eval] {line}  ({payload['seconds']:.0f}s)  -> {path}", flush=True)

    print("\n" + "=" * 78)
    for line in summaries:
        print("  " + line)
    if failures:
        print(f"\n  {len(failures)} rung(s) FAILED:")
        for line in failures:
            print("    " + line)
    print("=" * 78)
    print(f"results in {out_dir.resolve()}")
    return 1 if failures else 0


def _stop_for(spec: Any) -> Any:
    """
    :param spec: A task spec.

    :returns: Its stop condition, from the shared preset table -- never rebuilt per backend, which
        is what keeps the three backends cutting generations at the same point.
    """
    from .stopping import STOP_PRESETS

    return STOP_PRESETS[spec.stop]


if __name__ == "__main__":
    sys.exit(main())
