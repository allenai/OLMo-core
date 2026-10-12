"""
``ctc-fingerprint`` -- inspect, write and check format fingerprints.

The callback in :mod:`ctc.train` covers new runs. This covers everything else: fingerprinting a
shard directory at tokenize time, stamping a checkpoint that was trained before the guard existed,
and answering "would this eval be allowed?" without spending a GPU to find out.

Backfilling deserves a warning. Writing a fingerprint onto an old checkpoint is an *assertion* about
how it was trained, made from memory or from a launcher script rather than measured from the data.
If the assertion is wrong the guard now certifies a mismatch instead of catching it, which is worse
than the warning it replaced. ``write`` therefore records ``notes.provenance="asserted"`` unless the
fingerprint was collected from real shards, and ``show`` prints that back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .. import tasks
from . import registry
from .documents import shows_doc_ids, visible_doc_id_range
from .fingerprint import (
    FINGERPRINT_FILENAME,
    FingerprintSet,
    FormatFingerprint,
    FormatMismatchError,
    TaskNotTrainedError,
    collect_fingerprints,
    conflicting_formats,
)

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """:returns: The ``ctc-fingerprint`` argument parser."""
    ap = argparse.ArgumentParser(
        prog="ctc-fingerprint",
        description="Inspect and manage train/eval format fingerprints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctc-fingerprint show /data/ckpts/q35-4b-contra\n"
            "  ctc-fingerprint write --dir /data/shards/contra_n20 --task contradiction \\\n"
            "      --query-position both --tokenizer Qwen3.5-4B-Base --data rung_2048.jsonl\n"
            "  ctc-fingerprint collect --ckpt /data/ckpts/run --from /data/shards/a /data/shards/b\n"
            "  ctc-fingerprint check --ckpt /data/ckpts/run --task contradiction "
            "--query-position both\n"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="print the fingerprint recorded in a directory")
    show.add_argument("dir", help="shard or checkpoint directory")
    show.add_argument("--json", action="store_true", help="emit the raw record")

    write = sub.add_parser(
        "write", help="derive a fingerprint for one task and write it into a directory"
    )
    write.add_argument("--dir", required=True, help="directory to write into")
    write.add_argument("--task", required=True, help="task name")
    _add_format_args(write)
    write.add_argument(
        "--data",
        default=None,
        help=(
            "the TRAINING JSONL to measure doc_id_range from -- not an eval rung. Measuring beats "
            "asserting, since the digit-range bug was a wrong claim about the corpus. But the "
            "range describes what training showed the model, so measuring it from the eval set "
            "produces a field that contains itself by construction: a check that passes because "
            "it was derived from the thing being checked. The path is recorded so `show` can "
            "reveal which file was used."
        ),
    )
    write.add_argument(
        "--doc-id-range",
        default=None,
        metavar="LO:HI",
        help="assert the id range instead of measuring it",
    )
    write.add_argument(
        "--data-path",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "record a data path this format was built from. Repeatable, and unioned across "
            "sources when a task is fed by several -- a mixed corpus must name all of them. "
            "Recorded for provenance and never compared: the same data staged on weka, on "
            "node-local /data and on S3 is the same data. --data adds its path automatically."
        ),
    )
    write.add_argument(
        "--merge",
        action="store_true",
        help="add to the directory's existing record rather than replacing it",
    )

    collect = sub.add_parser(
        "collect", help="copy fingerprints from shard directories into a checkpoint"
    )
    collect.add_argument("--ckpt", required=True, help="checkpoint directory to stamp")
    collect.add_argument(
        "--from",
        dest="sources",
        nargs="+",
        required=True,
        metavar="DIR",
        help="shard directories to collect from",
    )
    collect.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip source directories with no fingerprint (leaves an INCOMPLETE record)",
    )

    check = sub.add_parser("check", help="test an eval format against a checkpoint, without a GPU")
    check.add_argument("--ckpt", required=True)
    check.add_argument("--task", required=True)
    _add_format_args(check)
    return ap


def _add_format_args(p: argparse.ArgumentParser) -> None:
    """Attach the options that describe a format, shared by ``write`` and ``check``."""
    p.add_argument(
        "--query-position",
        default="after",
        choices=("before", "after", "both"),
        help=(
            "where the query block sits relative to the documents (default: after). Pass it "
            "explicitly: the default here is the most common value, not a safe assumption, and "
            "this exact field is what an unrecorded reproduction run got wrong."
        ),
    )
    p.add_argument("--tokenizer", default=None, help="tokenizer id or path")
    p.add_argument(
        "--chunk-layout",
        default="none",
        help="chunk-wrapping scheme, e.g. wrap_documents for document-chunked data",
    )
    p.add_argument(
        "--marker-token-ids",
        default=None,
        metavar="A,B",
        help="reserved marker ids, when the format uses them",
    )


# ── subcommands ─────────────────────────────────────────────────────────────────────────────────


def _show(args: argparse.Namespace) -> int:
    found = FingerprintSet.read(Path(args.dir))
    if found is None:
        print(f"no {FINGERPRINT_FILENAME} in {args.dir}")
        print("  -> evaluating against this directory leaves format compatibility UNVERIFIED")
        return 1
    if args.json:
        print(json.dumps(found.to_dict(), indent=2, sort_keys=True))
        return 0
    print(f"{len(found.formats)} format(s) in {args.dir}\n")
    for fp in found.formats:
        provenance = fp.notes.get("provenance", "measured")
        print(f"  {fp.task}")
        print(
            f"    prompt        {fp.prompt_shape}, query {fp.query_position}, "
            f"serializer {fp.serializer}"
        )
        print(f"    gold ids      {fp.gold_index_base}-based")
        print(f"    chunk layout  {fp.chunk_layout}")
        print(f"    doc id range  {_describe_range(fp)}")
        print(f"    tokenizer     {fp.tokenizer or '(unrecorded)'}")
        if fp.data_paths:
            print(f"    data          {fp.data_paths[0]}")
            for extra_path in fp.data_paths[1:]:
                print(f"                  {extra_path}")
        else:
            print("    data          (unrecorded)")
        print(
            f"    provenance    {provenance}"
            + (
                "   <-- asserted by hand, not measured from data"
                if provenance == "asserted"
                else f"   from {fp.notes.get('measured_from', 'data')}"
            )
        )
        print()
    return 0


def _describe_range(fp: FormatFingerprint) -> str:
    """
    Render ``doc_id_range``, keeping "cannot apply" distinct from "nobody recorded it".

    Both print as an absent value, and conflating them is how a field silently stops guarding: for
    ``oolong`` there is genuinely no id to constrain, while for ``contradiction`` an absent range
    means the digit-range check is simply off for this checkpoint.
    """
    if fp.doc_id_range is not None:
        return str(fp.doc_id_range)
    if shows_doc_ids(fp.task):
        return "(NOT RECORDED -- the digit-range check is inactive for this checkpoint)"
    return "(n/a: this task renders items unnumbered)"


def _build(args: argparse.Namespace, examples: Optional[Sequence[dict]]) -> FormatFingerprint:
    """Derive one fingerprint from the CLI's format arguments."""
    tasks.load_all()
    spec = registry.get(args.task)

    overrides = dict(
        query_position=args.query_position,
        chunk_layout=args.chunk_layout,
        tokenizer=args.tokenizer,
    )
    if args.marker_token_ids:
        overrides["marker_token_ids"] = tuple(
            int(x) for x in args.marker_token_ids.replace(",", " ").split()
        )

    doc_range = getattr(args, "doc_id_range", None)
    if doc_range:
        lo, hi = (int(x) for x in doc_range.split(":"))
        overrides["doc_id_range"] = (lo, hi)
        overrides["notes"] = {"provenance": "asserted"}
    elif examples is not None:
        overrides["doc_id_range"] = visible_doc_id_range(examples, args.task)
        # Record WHICH file this came from. The range describes the data the model was TRAINED on,
        # and pointing --data at an eval rung instead produces a field that trivially contains
        # itself -- a check that passes because it was derived from the thing it is checking.
        overrides["notes"] = {
            "provenance": "measured",
            "measured_over": len(examples),
            "measured_from": str(getattr(args, "data", "")),
        }
    else:
        overrides["notes"] = {"provenance": "asserted"}

    declared = list(getattr(args, "data_path", []) or [])
    if getattr(args, "data", None):
        declared.append(args.data)
    overrides["data_paths"] = tuple(dict.fromkeys(str(Path(p).resolve()) for p in declared))
    return spec.fingerprint(**overrides)


def _write(args: argparse.Namespace) -> int:
    examples = None
    if args.data:
        examples = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    fp = _build(args, examples)

    directory = Path(args.dir)
    new = FingerprintSet([fp])
    if args.merge:
        existing = FingerprintSet.read(directory)
        if existing is not None:
            new = existing.merge(new)
    path = new.write(directory)

    print(f"wrote {path}")
    if fp.doc_id_range is None and shows_doc_ids(fp.task):
        print(
            f"  ! no doc_id_range recorded, so the digit-range check is inactive for {fp.task}. "
            "Pass --data pointing at the TRAINING data to measure it."
        )
    if fp.notes.get("provenance") == "asserted":
        print(
            "  ! recorded as ASSERTED: every field here was stated by hand, not measured. If the "
            "assertion is wrong, the guard now certifies a mismatch instead of catching it."
        )
    return 0


def _collect(args: argparse.Namespace) -> int:
    try:
        collected, skipped = collect_fingerprints(
            [Path(d) for d in args.sources], allow_missing=args.allow_missing
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    path = collected.write(Path(args.ckpt))
    print(f"wrote {path}  ({len(collected.formats)} format(s): {', '.join(collected.tasks)})")
    if skipped:
        print(f"  ! INCOMPLETE: skipped {', '.join(skipped)}")
    for task, fields in conflicting_formats(collected).items():
        print(f"  ! {task} is recorded under several formats, differing in {', '.join(fields)}")
    return 0


def _check(args: argparse.Namespace) -> int:
    eval_fp = _build(args, None)
    trained = FingerprintSet.read(Path(args.ckpt))
    if trained is None:
        print(f"no {FINGERPRINT_FILENAME} in {args.ckpt}: compatibility UNVERIFIED")
        return 1
    try:
        trained.require_compatible(eval_fp)
    except (FormatMismatchError, TaskNotTrainedError) as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"compatible: {args.task} @ query_position={args.query_position}")
    return 0


_COMMANDS = {"show": _show, "write": _write, "collect": _collect, "check": _check}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    :param argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
