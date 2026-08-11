"""Run one read-only shell command on a Beaker node to look at weka.

The submitting host has no weka mount, so "does this file exist there" is not answerable locally --
and guessing is how a build gets launched against a source that is not present. This submits a
GPU-less job that runs the command and exits.

    python debug/fast_bundle/probe_weka.py 'ls -la /weka/.../some/dir'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "scripts" / "ctc"))

from _launch import pushed_head  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", help="shell command to run on the node")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--name", default="weka-probe")
    args = ap.parse_args()

    from olmo_core.internal.common import build_launch_config, get_root_dir
    from olmo_core.launch.beaker import OLMoCoreBeakerImage
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    print(f"the node will run commit {pushed_head()[:12]}")

    inner = f"""set -uo pipefail
echo "=== probe on $(hostname) $(date '+%F %T') ==="
{args.command}
echo "=== exit=$? ==="
"""
    launch = build_launch_config(
        name=args.name,
        cmd=["bash", "-lc", inner],
        cluster=args.cluster,
        root_dir=get_root_dir(args.cluster),
        task_name="probe",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=0,
    )
    launch.torchrun = False
    launch.allow_dirty = True
    launch.priority = "urgent"
    launch.step_soft_timeout = None
    launch.step_timeout = None

    workload = launch.launch(follow=False)
    experiment = getattr(workload, "experiment", None)
    wid = getattr(experiment, "id", None) or getattr(workload, "id", None)
    print(f"submitted: {wid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
