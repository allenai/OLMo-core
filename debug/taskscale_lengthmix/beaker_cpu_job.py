"""Run an arbitrary repo command as a CPU-only Beaker job with weka mounted.

The length-mix data pipeline (generate pools -> compose arms -> tokenize -> stage) used to run on
the Berkeley node `mooney` via sbatch. Standing user directive (2026-08-31): no mooney, no local
slurm -- Beaker for everything. This is the replacement vehicle: a 0-GPU gantry job on a weka
cluster, so a build writes its shards straight into
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/...` and the trainers read them with no
S3 round trip.

  python beaker_cpu_job.py NAME --cmd 'python debug/taskscale_lengthmix/build_pools.py --task oolong'

Anything the command prints lands in the job log; anything it writes under /results is fetchable
with `beaker experiment results`.
"""
import argparse

from olmo_core.internal.common import build_launch_config
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment

WEKA_ROOT = "/weka/oe-training-default/ai2-llm"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--cmd", required=True, help="shell command run from the repo root")
    ap.add_argument("--cluster", default="ai2/neptune-cirrascale")
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inner = f"PYTHONPATH=$PWD/src/scripts:$PWD/src {args.cmd}"
    lc = build_launch_config(
        name=args.name,
        cmd=["bash", "-lc", inner],
        cluster=args.cluster,
        root_dir=WEKA_ROOT,
        task_name="build",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=args.gpus,
    )
    lc.torchrun = False
    lc.allow_dirty = True
    lc.priority = "urgent"
    lc.step_soft_timeout = None
    lc.step_timeout = None
    print(f"--- {lc.name}\n    cmd: {inner}")
    if args.dry_run:
        print("    [dry-run] not submitting.")
        return
    workload = lc.launch(follow=False)
    print(f"    submitted: {getattr(workload, 'id', workload)}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
