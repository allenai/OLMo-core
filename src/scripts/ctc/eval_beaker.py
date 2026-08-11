#!/usr/bin/env python3
"""
Launch a CTC-suite eval on Beaker. One checkpoint in, a results directory on weka out.

    python src/scripts/ctc/eval_beaker.py CKPT_NAME --cluster ai2/jupiter-cirrascale-2

``CKPT_NAME`` is a directory under ``<root>/checkpoints/prasanns/``; the latest ``stepN`` in it is
found on the node, so there is nothing to look up before launching. Pass ``--step stepN`` to pin one,
or ``--ckpt /abs/path`` to point anywhere.

The eval itself is :mod:`ctc.eval.cli`, unchanged -- this script only decides *where* it runs. Both
halves therefore stay the same experiment, which the pre-migration pair of a Beaker launcher and its
local twin did not: they were separate files and they drifted.

Everything reads from the weka mount and writes back to it. Nothing is staged locally, so a launch
costs one API call and the submitting host needs neither the checkpoint nor the eval bundle.

.. warning::
   **gantry clones the PUSHED commit, not your working tree.** Commit and push before launching or
   the job runs older code, silently. ``--dry-run`` prints the command without submitting, and
   ``--check-bundle`` submits a one-minute job that only lists the bundle -- worth doing once when
   a new bundle appears.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Importable whether or not the package was pip-installed: a Beaker node runs from the clone.
_REPO = Path(__file__).resolve().parents[3]
for _src in (_REPO / "src", _REPO / "ctc" / "src"):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from ctc.eval import bundles  # noqa: E402

#: Default results root on weka, alongside the checkpoints rather than in a separate tree -- a
#: results file is only interpretable next to the checkpoint that produced it.
RESULTS_SUBDIR = "ctc_eval"


def build_parser() -> argparse.ArgumentParser:
    """:returns: The launcher's argument parser."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_name",
        nargs="?",
        default="",
        help="checkpoint directory name under <root>/checkpoints/prasanns/. Omit only with --ckpt.",
    )
    ap.add_argument(
        "--cluster",
        default="ai2/jupiter-cirrascale-2",
        help="Beaker cluster (default ai2/jupiter-cirrascale-2, H100 + weka). neptune's L40S nodes "
        "are roughly 5x slower for this workload.",
    )
    ap.add_argument("--ckpt", default="", help="absolute checkpoint path; overrides run_name")
    ap.add_argument(
        "--step", default="", help="pin a step directory (e.g. step1100); default is the latest"
    )
    ap.add_argument(
        "--tasks",
        default="main",
        help="'main' (5 in-distribution), 'ood' (4 held-out), 'all', or a comma list",
    )
    ap.add_argument("--rungs", default="all", help="'all' or a comma list like '2k,8k'")
    ap.add_argument(
        "--attn",
        default="full",
        choices=("full", "chunked", "landmark"),
        help="attention mask to grade under. Must match how the checkpoint was trained.",
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument(
        "--query-position",
        default="both",
        choices=("both", "before", "after"),
        help="must match training; these checkpoints trained with 'both'",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="grade only the first N per rung (0 = all)"
    )
    ap.add_argument(
        "--bundle",
        default="",
        help=f"eval bundle name or root (default {bundles.DEFAULT_BUNDLE}). 'fast' is the "
        "shared-corpus bundle and is NOT comparable to a reliable one.",
    )
    ap.add_argument(
        "--share-prefix",
        action="store_true",
        help="reuse each corpus group's shared-prefix KV. Only does anything on --bundle fast.",
    )
    ap.add_argument(
        "--results-dir",
        default="",
        help=f"absolute weka results dir (default <ckpt>/../{RESULTS_SUBDIR})",
    )
    ap.add_argument("--tag", default="", help="extra label in result filenames")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs (the 4B model fits on one)")
    ap.add_argument(
        "--priority",
        default="urgent",
        help="Beaker priority. Repo rule: always urgent -- anything lower pends behind capacity.",
    )
    ap.add_argument(
        "--ignore-format-fingerprint",
        action="store_true",
        help="grade even when the checkpoint's recorded training format does not match. Needed for "
        "checkpoints trained before fingerprinting existed.",
    )
    ap.add_argument(
        "--allow-truncated",
        action="store_true",
        help="continue when a prompt exceeds --max-length instead of failing the rung",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="record a failed rung and continue (default on for a sweep)",
    )
    ap.add_argument(
        "--check-bundle",
        action="store_true",
        help="submit a short job that only resolves and lists the bundle files, then exits. Run "
        "this once against a new bundle rather than discovering a missing rung an hour in.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not submit")
    return ap


def remote_command(args: argparse.Namespace, root_dir: str) -> str:
    """
    Build the shell command the Beaker node runs.

    Checkpoint resolution happens **on the node**, not here: the submitting host has no weka mount,
    so globbing for the latest step locally would either fail or silently resolve against the wrong
    filesystem.

    :param args: Parsed arguments.
    :param root_dir: The cluster's weka root.

    :returns: A single ``bash -lc`` script.
    """
    bundle = args.bundle or bundles.DEFAULT_ROOT

    if args.ckpt:
        resolve = f'CKPT="{args.ckpt}"'
    else:
        run_dir = f"{root_dir}/checkpoints/prasanns/{args.run_name}"
        step = f'"{args.step}"' if args.step else ""
        resolve = f"""RUN_DIR="{run_dir}"
STEP={step}
if [ -n "$STEP" ]; then
  CKPT="$RUN_DIR/$STEP"
else
  # Latest complete step: sort numerically on the digits, and require model_and_optim so a
  # half-written async checkpoint is never picked.
  CKPT=$(ls -d "$RUN_DIR"/step*/ 2>/dev/null \\
         | sed 's:/$::' \\
         | awk -F'step' '{{print $NF, $0}}' | sort -n | awk '{{print $2}}' \\
         | while read -r d; do [ -d "$d/model_and_optim" ] && echo "$d"; done \\
         | tail -1)
fi
if [ -z "$CKPT" ]; then
  echo "FATAL: no complete stepN/model_and_optim under $RUN_DIR"; ls -la "$RUN_DIR" 2>&1 | head -20; exit 1
fi"""

    results = args.results_dir or (
        f"{root_dir}/checkpoints/prasanns/{args.run_name}/{RESULTS_SUBDIR}"
        if args.run_name
        else f"{root_dir}/checkpoints/prasanns/{RESULTS_SUBDIR}"
    )

    flags = [
        f"--tasks {args.tasks}",
        f"--rungs {args.rungs}",
        f"--attn {args.attn}",
        f"--tokenizer {args.tokenizer}",
        f"--max-length {args.max_length}",
        f"--query-position {args.query_position}",
        f'--bundle "{bundle}"',
        f'--out "{results}"',
    ]
    if args.share_prefix:
        flags.append("--share-prefix")
    if args.tag:
        flags.append(f"--tag {args.tag}")
    if args.limit:
        flags.append(f"--limit {args.limit}")
    if args.ignore_format_fingerprint:
        flags.append("--ignore-format-fingerprint")
    if args.allow_truncated:
        flags.append("--allow-truncated")
    if args.keep_going:
        flags.append("--keep-going")

    if args.check_bundle:
        body = f'python -m ctc.eval.cli {" ".join(flags)} --dry-run'
        resolve = 'CKPT="(not needed for --check-bundle)"'
    else:
        body = f'python -m ctc.eval.cli --ckpt "$CKPT" {" ".join(flags)}'

    return f"""set -uo pipefail
export PYTHONPATH="$(pwd)/src:$(pwd)/ctc/src:${{PYTHONPATH:-}}"
echo "=== ctc-eval on $(hostname) $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | head -3
{resolve}
echo "CKPT=$CKPT"
echo "RESULTS={results}"
mkdir -p "{results}"
{body}
STATUS=$?
echo "=== ctc-eval exit=$STATUS $(date '+%F %T') ==="
exit $STATUS"""


def main(argv=None) -> int:
    """
    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    args = build_parser().parse_args(argv)
    if not args.run_name and not args.ckpt and not args.check_bundle:
        raise SystemExit("pass a run_name, or --ckpt, or --check-bundle")

    from olmo_core.internal.common import build_launch_config, get_root_dir
    from olmo_core.launch.beaker import OLMoCoreBeakerImage
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    root_dir = get_root_dir(args.cluster)
    inner = remote_command(args, root_dir)

    label = args.run_name or Path(args.ckpt).name or "bundle"
    name = f"ctceval-{args.tasks}-{args.attn}-{label}"[:100]

    launch = build_launch_config(
        name=name,
        cmd=["bash", "-lc", inner],
        cluster=args.cluster,
        root_dir=root_dir,
        task_name="eval",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=0 if args.check_bundle else args.gpus,
    )
    # This is a single-process eval, not distributed training: torchrun would fan it out over the
    # GPUs and grade every example once per rank.
    launch.torchrun = False
    launch.priority = args.priority
    # A 32k rung is hours, and the suite is many rungs. The default step timeouts kill it mid-sweep.
    launch.step_soft_timeout = None
    launch.step_timeout = None

    print(f"--- {name} ---")
    print(inner)
    if args.dry_run:
        print("\n[dry-run] not submitted.")
        return 0

    workload = launch.launch(follow=False)
    # `workload` is a protobuf whose `.id` is nested on the experiment, not on the top level -- a
    # plain getattr returns the whole message and prints a URL that does not resolve. gantry has
    # already printed the real link by this point, so a failure here is cosmetic, not a lost job.
    experiment = getattr(workload, "experiment", None)
    wid = getattr(experiment, "id", None) or getattr(workload, "id", None)
    if wid:
        print(f"\nsubmitted: {wid}\n  https://beaker.org/ex/{wid}")
    else:  # pragma: no cover - depends on the beaker client version
        print("\nsubmitted; see the experiment link gantry printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
