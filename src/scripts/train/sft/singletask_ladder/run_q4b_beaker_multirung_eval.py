"""
Submit a multi-rung NATIVE long-context eval as a **Beaker** job for one trained checkpoint --
fully on Beaker, NO local sync. This is the on-Beaker counterpart of the LOCAL driver
``run_q4b_stl_multirung_eval.sbatch``.

The eval reads everything from weka (eval code + eval data + the distcp checkpoint), so the job needs
no data copied to the node:

  * eval CODE + DATA + the goal-rung ladder files live in the weka eval bundle
    (``checkpoints/prasanns/_eval_bundle`` + ``_eval_bundle_eval500``); upload/refresh it with
    ``upload_lc_eval_bundle.sh`` before launching.
  * the checkpoint is the just-trained distcp step dir under
    ``checkpoints/prasanns/<run_name>/step*`` -- the on-node runner auto-globs the latest complete step.

The actual eval logic is the on-node runner ``run_beaker_multirung_eval.sh`` (uploaded to the bundle):
it runs ``torchrun --nproc_per_node=8 scripts/eval/eval_lc_native.py`` (8-way DP, native olmo_core
generate -- NO HF/vLLM, required for landmark/compressive). gantry's own torchrun wrapping is disabled
(``torchrun=False``) so the runner can issue its own (and multiple, for rerank) torchrun calls.

Per-task rungs (matching the local driver):
  contra 2k,8k,16k,32k | nq 3k,8k,16k,32k | outlier 3k,8k,16k,32k | oolong 8k,16k,32k |
  rerank CE files k20/k50/k100 (NDCG@10 + Kendall-tau). ``--max-test-samples 600``.

Usage::

    # one (run, task) -> one Beaker job; variant inferred from the run name
    PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py \\
        q4b-dense-contra-ladder32k-10k ai2/neptune --task contra

    # all 5 tasks for a run (5 jobs)
    PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py \\
        q4b-landmark-nq-ladder32k-10k ai2/neptune --task all

    # validate without submitting
    PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py \\
        q4b-dense-contra-ladder32k-10k ai2/neptune --task contra --dry-run
"""

import argparse
import sys

from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment

ALL_TASKS = ["contra", "nq", "rerank", "outlier", "oolong"]
VARIANTS = ["dense", "landmark", "compressive", "docchunk"]


def variant_from_run_name(run_name: str) -> str:
    # docchunk run names use the explicit "docchunk_dense" token; check docchunk first.
    if "docchunk" in run_name:
        return "docchunk"
    found = [v for v in ("dense", "landmark", "compressive") if v in run_name]
    if len(found) != 1:
        raise SystemExit(
            f"could not infer variant from run name {run_name!r} (found {found}); pass --variant."
        )
    return found[0]


def build_eval_launch_config(
    *, run_name, task, variant, cluster, step, ckpt, max_test, max_length, batch_size, priority
):
    root_dir = get_root_dir(cluster)  # e.g. /weka/oe-training-default/ai2-llm (mounts weka bucket)
    bundle = f"{root_dir}/checkpoints/prasanns/_eval_bundle"
    runner = f"{bundle}/run_beaker_multirung_eval.sh"

    # The on-node runner reads its inputs from env; gantry torchrun wrapping is disabled so the runner
    # can drive its own 8-way `torchrun`. cmd[0]="bash" => not auto-prefixed with `python`.
    inner = (
        f"RUN={run_name} TASK={task} VARIANT={variant} STEP='{step}' CKPT='{ckpt}' "
        f"MAX_TEST={max_test} MAX_LENGTH={max_length} BATCH_SIZE={batch_size} NGPU=8 "
        f"WEKA_LLM={root_dir} bash {runner}"
    )
    cmd = ["bash", "-lc", inner]

    name = f"ev-{task}-{run_name}"
    launch_config = build_launch_config(
        name=name,
        cmd=cmd,
        cluster=cluster,
        root_dir=root_dir,
        task_name="eval",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=8,
    )
    launch_config.torchrun = False  # the runner issues its own torchrun(s)
    launch_config.allow_dirty = True  # ship the (uncommitted) launcher via an ephemeral ref
    launch_config.priority = priority
    return launch_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_name", help="trained run name (checkpoints under checkpoints/prasanns/<run_name>)")
    ap.add_argument("cluster", help="Beaker cluster, e.g. ai2/neptune (weka-backed)")
    ap.add_argument("--task", default="all",
                    help=f"comma list from {ALL_TASKS}, or 'all' (one Beaker job per task).")
    ap.add_argument("--variant", default=None, choices=VARIANTS,
                    help="dense|landmark|compressive|docchunk (default: infer from run name).")
    ap.add_argument("--step", default="", help="pin a step dir (e.g. step580); default = latest complete.")
    ap.add_argument("--ckpt", default="",
                    help="ABSOLUTE weka step dir to eval ANY checkpoint, e.g. "
                         "/weka/oe-training-default/ai2-llm/checkpoints/<you>/<run>/step1234 . "
                         "Overrides run_name globbing (run_name is then just a results label).")
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--priority", default="normal")
    ap.add_argument("--dry-run", action="store_true", help="build + print the job, do NOT submit.")
    args = ap.parse_args()

    prepare_cli_environment()

    variant = args.variant or variant_from_run_name(args.run_name)
    tasks = ALL_TASKS if args.task == "all" else [t.strip() for t in args.task.split(",") if t.strip()]
    bad = [t for t in tasks if t not in ALL_TASKS]
    if bad:
        raise SystemExit(f"unknown task(s) {bad}; choose from {ALL_TASKS}.")

    print(f"=== Beaker multirung eval | run={args.run_name} variant={variant} "
          f"tasks={tasks} cluster={args.cluster} dry_run={args.dry_run} ===")
    for task in tasks:
        if variant == "docchunk" and task != "oolong":
            print(f"--- skip {task}: docchunk native eval supports OOLONG only ---")
            continue
        lc = build_eval_launch_config(
            run_name=args.run_name, task=task, variant=variant, cluster=args.cluster,
            step=args.step, ckpt=args.ckpt, max_test=args.max_test, max_length=args.max_length,
            batch_size=args.batch_size, priority=args.priority,
        )
        print(f"\n--- [{task}] {lc.name} ---")
        print(f"    cmd: {lc.cmd[-1]}")
        if args.dry_run:
            print("    [dry-run] not submitting.")
            continue
        workload = lc.launch(follow=False)
        print(f"    submitted: {getattr(workload, 'id', workload)}")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
