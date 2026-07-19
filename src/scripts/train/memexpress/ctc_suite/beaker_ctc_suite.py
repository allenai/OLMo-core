"""
Beaker launcher for the CTC-suite trainer (``train_ctc_suite.py``) at 4B/9B scale
(``records/ctc-suite-scaling-plan.md`` §4/§7: "all 4B and 9B training runs" go on
``ai2/jupiter``, urgent priority).

``train_ctc_suite.py`` is a **plain torchrun script** (task-agnostic local trainer), not built on
the ``ExperimentConfig``/``CliContext`` machinery the ``sft_docchunk`` family uses -- so this
launcher does NOT build an ``ExperimentConfig``. Instead it adapts the Beaker half of
``_docchunk_5task_32k_nocpt_common.py`` directly: :func:`olmo_core.internal.common.build_launch_config`
(-> ``OLMoCoreBeakerImage.stable``, urgent priority, weka mount, wandb/beaker-token secrets) wraps
a ``cmd`` that is just ``train_ctc_suite.py``'s own CLI. ``BeakerLaunchConfig`` auto-wraps any
``cmd[0]`` ending in ``.py`` with ``python`` and with ``torchrun`` (since ``num_gpus > 1``), so no
manual torchrun invocation is needed here (unlike the local ``run_ctc_local.sbatch`` launcher).

Data / base staging (weka is NOT reachable from Berkeley -- two-step S3 relay, see ``beaker.md``):
shards and the per-scale converted base are staged under
``/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/{shards,bases}/...`` -- see
``debug/ctc_suite_beaker_smoke/upload_to_s3.sh`` (cubbins /data -> S3) and the gantry S3->weka
sync job (``beaker.md`` template) run once from this host.

Usage::

    python -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \\
        --task outlier --variant chunked-mix --model-scale 4b \\
        --run-name ctc-outlier-cmix-4b --num-nodes 2 --epochs 1 \\
        launch

    # dry run (prints the resolved BeakerLaunchConfig, does not submit):
    python -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \\
        --task outlier --variant chunked-mix --model-scale 4b --run-name ctc-outlier-smoke \\
        --num-nodes 1 --max-steps 100 dry_run
"""

import argparse
import sys
from datetime import datetime

from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.launch.beaker import BeakerEnvVar, OLMoCoreBeakerImage

WANDB_ENTITY = "prasanns-allen-institute-for-ai"
WANDB_PROJECT = "memory-networks"

# GPUs/node for every scale in this family (H100 80GB jupiter nodes).
NUM_GPUS = 8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="suite task name, e.g. outlier / grouping")
    ap.add_argument(
        "--variant", required=True, choices=["full", "chunked", "chunked-mix"],
    )
    ap.add_argument("--model-scale", required=True, choices=["4b", "9b"], help="Beaker family is 4B/9B only; 0.8B trains locally")
    ap.add_argument("--run-name", required=True, help="fresh name per config -- silent auto-resume trap")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--num-nodes", type=int, default=2, help="2 nodes for 4B, 4 for 9B per plan §4")
    ap.add_argument("--epochs", type=int, default=1, help="plan directive: 1 epoch at 20k examples")
    ap.add_argument("--seq-len", type=int, default=40960)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument(
        "--global-batch", type=int, default=None,
        help="instances/optimizer-step across all ranks; default = num_nodes*8 (1 instance/GPU, grad-accum 1)",
    )
    ap.add_argument("--micro-batch-instances", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full (epoch-bounded); set for smoke tests")
    ap.add_argument(
        "--data-root",
        default=None,
        help="shard dir override; default {weka_root}/checkpoints/prasanns/ctc_suite/shards/{task}_train",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help="model_and_optim distcp subdir override; default {weka_root}/checkpoints/prasanns/"
        "ctc_suite/bases/q35-{scale}-base-modelonly/model_and_optim",
    )
    ap.add_argument("--wandb-group", default=None, help="default: ctc-suite-<task>")
    ap.add_argument("--priority", default="urgent", help="ALWAYS urgent per project directive")
    ap.add_argument(
        "--no-allow-dirty", dest="allow_dirty", action="store_false", default=True,
        help="require a clean tree (default: allow_dirty=True, matching _docchunk_5task_32k_nocpt_common.py -- "
        "this checkout is a shared working tree actively modified by concurrent jobs and is rarely clean; "
        "gantry always clones the PUSHED commit regardless of local dirtiness)",
    )
    ap.add_argument("mode", choices=["launch", "dry_run"])
    return ap.parse_args()


def main() -> None:
    opts = parse_args()

    root_dir = get_root_dir(opts.cluster)  # -> /weka/oe-training-default/ai2-llm on jupiter
    work_dir = get_work_dir(root_dir)

    data_root = opts.data_root or f"{root_dir}/checkpoints/prasanns/ctc_suite/shards/{opts.task}_train"
    base_checkpoint = opts.base_checkpoint or (
        f"{root_dir}/checkpoints/prasanns/ctc_suite/bases/q35-{opts.model_scale}-base-modelonly/model_and_optim"
    )
    run_name = f"{opts.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = f"{root_dir}/checkpoints/prasanns/ctc_suite/ckpts/{run_name}"
    wandb_group = opts.wandb_group or f"ctc-suite-{opts.task}"
    world_size = opts.num_nodes * NUM_GPUS
    global_batch = opts.global_batch or world_size  # 1 instance/GPU/step, grad-accum 1

    cmd = [
        "src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py",
        "--task", opts.task,
        "--data", data_root,
        "--variant", opts.variant,
        "--model-scale", opts.model_scale,
        "--seq-len", str(opts.seq_len),
        "--epochs", str(opts.epochs),
        "--lr", str(opts.lr),
        "--global-batch", str(global_batch),
        "--micro-batch-instances", str(opts.micro_batch_instances),
        "--base-checkpoint", base_checkpoint,
        "--work-dir", f"{work_dir}-ctc-suite",
        "--save-folder", save_folder,
        "--run-name", run_name,
        "--wandb-group", wandb_group,
        "--wandb-entity", WANDB_ENTITY,
        "--save-checkpoint",
    ]
    if opts.max_steps:
        cmd += ["--max-steps", str(opts.max_steps)]

    launch_config = build_launch_config(
        name=run_name,
        cmd=cmd,
        cluster=opts.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=opts.num_nodes,
        num_gpus=NUM_GPUS,
    )
    launch_config.priority = opts.priority
    launch_config.allow_dirty = opts.allow_dirty
    # Doc-chunked attention's FlexAttention/eager mask path needs headroom beyond what the default
    # caching allocator leaves reachable at 40960 tokens (same fragmentation fix as the proven
    # sft_docchunk 40k Beaker runs).
    launch_config.env_vars.append(
        BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
    )

    print(f"[beaker_ctc_suite] task={opts.task} variant={opts.variant} scale={opts.model_scale} "
          f"run_name={run_name} nodes={opts.num_nodes} world_size={world_size} "
          f"global_batch={global_batch} epochs={opts.epochs} max_steps={opts.max_steps or 'full'}")
    print(f"[beaker_ctc_suite] data={data_root}")
    print(f"[beaker_ctc_suite] base_checkpoint={base_checkpoint}")
    print(f"[beaker_ctc_suite] save_folder={save_folder}")
    print(f"[beaker_ctc_suite] wandb: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/groups/{wandb_group}")
    print(f"[beaker_ctc_suite] cmd={' '.join(cmd)}")

    if opts.mode == "dry_run":
        print("[beaker_ctc_suite] DRY RUN -- not submitting. Resolved BeakerLaunchConfig:")
        print(launch_config)
        return

    workload = launch_config.launch(follow=True)
    print(f"[beaker_ctc_suite] launched: {workload}")


if __name__ == "__main__":
    sys.exit(main())
