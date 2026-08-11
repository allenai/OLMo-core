"""
Build the **fast** (shared-corpus) eval bundle on Beaker, straight from the reliable bundle.

The ultra-long source rungs only exist on weka, and a 1M rung file is gigabytes, so this builds
where the data already is: one CPU-only Beaker job per ``(task, rung)``, reading the reliable
bundle and writing :mod:`ctc.data.shared_corpus` output into the fast bundle root. Nothing is
copied to or from this machine.

One job per task-rung rather than one job for the sweep: the builders load an entire rung into
memory (contradiction's 1M rung is 500 rows of 19,775 documents, and its filler pool dedups across
all of them), so a shared process would multiply the peak, and a failure at 1M would take the
finished 64k with it.

.. warning::
   **A fast rung measures something different from its reliable twin** -- outlier +0.215/+0.261,
   contradiction -0.102...-0.175 against the independent rungs. It is a separate bundle, never a
   flag on the reliable one. See :mod:`ctc.data.shared_corpus`.

Usage::

    python src/scripts/ctc/build_fast_bundle.py                      # print the matrix, submit nothing
    python src/scripts/ctc/build_fast_bundle.py --submit
    python src/scripts/ctc/build_fast_bundle.py --tasks contradiction \\
        --rungs 64k,128k,256k,512k,1M --submit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ctc" / "src"))

from ctc.eval import bundles  # noqa: E402

#: Where the fast bundle is written. A sibling of the reliable bundles, not a subdirectory of one:
#: a fast file must never be reachable by a ``--bundle v2_clean`` run.
FAST_ROOT_DEFAULT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_fast"
)

#: Rung label -> token budget. Only names the output file; the corpus size comes from the source.
RUNG_TOKENS = {
    "2k": 2048,
    "3k": 3072,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
    "512k": 524288,
    "1M": 1048576,
    "2M": 2097152,
}

#: Which construction each task gets, and therefore which flags. ``oolong`` is absent on purpose:
#: its shared file comes from the source split that already stores 25 questions per context, not
#: from a rung file, and that split has no ultra-long member.
FAMILY = {
    "contradiction": "prefix_tail",
    "outlier": "prefix_tail",
    "nq": "multiplexed",
    "rerank": "multiplexed",
}

DEFAULT_TASKS = "contradiction,nq,rerank"
DEFAULT_RUNGS = "all,xlong"


def build_parser() -> argparse.ArgumentParser:
    """:returns: The launcher's argument parser."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help=f"comma list (default {DEFAULT_TASKS})")
    ap.add_argument(
        "--rungs",
        default=DEFAULT_RUNGS,
        help="'all' (base ladder), 'xlong', or a comma list. Both may be given, comma separated.",
    )
    ap.add_argument(
        "--from-bundle",
        default=bundles.DEFAULT_BUNDLE,
        help="the reliable bundle to build from. v2_clean has no xlong rungs for nq or outlier; "
        "pass v2 for those.",
    )
    ap.add_argument("--out-root", default=FAST_ROOT_DEFAULT)
    ap.add_argument(
        "--tail-frac",
        type=float,
        default=0.1,
        help="prefix+tail tasks: per-query tail as a fraction of the corpus. 0.1 shares 90%% of "
        "the prefill (~10x saving). The trade is fidelity: the golds sit in the last tenth, and "
        "measured placement inflation on outlier was +0.215 at a 5%% tail and +0.261 at 25%%.",
    )
    ap.add_argument(
        "--min-rung",
        default="8k",
        help="skip anything shorter (default 8k -- below that the corpus is too small for the "
        "sharing to be worth a separate file)",
    )
    ap.add_argument(
        "--max-rung",
        default="1M",
        help="skip anything longer (default 1M; 2M sources exist but double the cost again)",
    )
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--priority", default="urgent", help="repo rule: always urgent")
    ap.add_argument("--submit", action="store_true", help="actually launch; default prints only")
    return ap


def plan(args: argparse.Namespace):
    """
    Resolve the ``(task, rung)`` matrix to source files in the reliable bundle.

    Rungs a bundle does not carry are skipped with a message rather than raising: ``v2_clean`` has
    no ultra-long nq or outlier files, and a sweep over several tasks should not die on the first
    one that stops short.

    :param args: Parsed arguments.

    :returns: A list of ``(task, rung_label, source_path)``.
    """
    bundle = bundles.get_bundle(args.from_bundle)
    root = Path(bundle.root)
    ceiling = RUNG_TOKENS.get(args.max_rung, RUNG_TOKENS["1M"])
    floor = RUNG_TOKENS.get(args.min_rung, RUNG_TOKENS["8k"])

    out = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task not in FAMILY:
            raise SystemExit(f"no shared-corpus construction for {task!r}; have {sorted(FAMILY)}")
        for group in [g.strip() for g in args.rungs.split(",") if g.strip()]:
            try:
                resolved = bundles.resolve(task, group, root=args.from_bundle)
            except KeyError as e:
                print(f"  skipping {task} {group}: {e}", file=sys.stderr)
                continue
            for label, path in resolved:
                tokens = RUNG_TOKENS.get(label)
                if tokens is None:
                    print(f"  skipping {task} {label}: unknown rung label", file=sys.stderr)
                    continue
                if tokens > ceiling or tokens < floor:
                    continue
                out.append((task, label, Path(path)))
    return out


def remote_command(args: argparse.Namespace, task: str, label: str, source: Path) -> str:
    """
    :param args: Parsed arguments.
    :param task: Ladder name.
    :param label: Rung label.
    :param source: The reliable bundle file to build from.

    :returns: The shell script the node runs.
    """
    flags = [
        f"--task {task}",
        f"--rung {RUNG_TOKENS[label]}",
        f'--source "{source}"',
        f'--out-root "{args.out_root}"',
    ]
    if FAMILY[task] == "prefix_tail":
        flags.append(f"--tail-frac {args.tail_frac}")

    return f"""set -uo pipefail
export PYTHONPATH="$(pwd)/src:$(pwd)/ctc/src:${{PYTHONPATH:-}}"
echo "=== build-fast-bundle {task}@{label} on $(hostname) $(date '+%F %T') ==="
echo "source: {source}"
ls -la "{source}" || {{ echo "FATAL: source missing"; exit 1; }}
free -g | head -2
mkdir -p "{args.out_root}/{task}"
python -m ctc.data.shared_corpus {' '.join(flags)}
STATUS=$?
echo "=== exit=$STATUS $(date '+%F %T') ==="
ls -la "{args.out_root}/{task}" 2>/dev/null | tail -5
exit $STATUS"""


def main(argv=None) -> int:
    """
    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    args = build_parser().parse_args(argv)
    matrix = plan(args)
    if not matrix:
        raise SystemExit("nothing to build")

    print(f"fast bundle -> {args.out_root}")
    print(f"built from  <- {bundles.get_bundle(args.from_bundle).root}")
    print(f"{len(matrix)} jobs:")
    for task, label, source in matrix:
        family = FAMILY[task]
        extra = f" tail={args.tail_frac}" if family == "prefix_tail" else ""
        print(f"  {task:>14} @{label:<5} {family}{extra}  <- {Path(source).name}")

    if not args.submit:
        print("\n[not submitted] pass --submit to launch.")
        return 0

    from olmo_core.internal.common import build_launch_config, get_root_dir
    from olmo_core.launch.beaker import OLMoCoreBeakerImage
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    root_dir = get_root_dir(args.cluster)

    for task, label, source in matrix:
        inner = remote_command(args, task, label, source)
        name = f"fastbundle-{task}-{label}"[:100]
        launch = build_launch_config(
            name=name,
            cmd=["bash", "-lc", inner],
            cluster=args.cluster,
            root_dir=root_dir,
            task_name="build",
            beaker_image=OLMoCoreBeakerImage.stable,
            workspace="ai2/flex2",
            budget="ai2/oe-other",
            num_nodes=1,
            num_gpus=0,
        )
        launch.torchrun = False
        launch.priority = args.priority
        # A 1M rung is a long single-process pass over gigabytes of JSON; the default step timeouts
        # would kill it partway and leave a truncated file behind.
        launch.step_soft_timeout = None
        launch.step_timeout = None

        workload = launch.launch(follow=False)
        experiment = getattr(workload, "experiment", None)
        wid = getattr(experiment, "id", None) or getattr(workload, "id", None)
        print(f"  submitted {name}: {wid or '(see gantry link above)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
