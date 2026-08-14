"""
Launch the HiLS / Olmo-3 SFT arms on Beaker.

**The contrast.** `hils-7b` vs `olmo3-7b`, SFT'd on one materialized 32k pack of the 5-task +
Dolci25 mixture, same trainer, same seed, same batches — the model is the only difference. HiLS is
a ~50B-token continued-pretrain *of* that Olmo-3 checkpoint, so the pair isolates what its
chunk-wise sparse attention (plus the CPT) did.

**The readout.** The v2 ladder via `eval_lc_native.py --backend hf` (`../hils_eval/`):
contra/nq/rerank/outlier/oolong at 2k–32k plus the four OOD ladders. Zero-shot baselines for both
models were measured 2026-08-13 and floor at the answer format — that is why these arms exist.

**Deliberately not matched.** Nothing between the two veomni arms. Against our Qwen3.5 SFT ladder,
everything is different (model family, vocabulary, trainer), so those rows are context, not a
control.

Arms live in `_ARMS` rather than forked files: a forked launcher drifts from its counterpart, a
dict entry cannot.

Usage::

    PYTHONPATH=src python src/scripts/train/memexpress/hils_sft/run_sft_beaker.py \\
        hils-7b ai2/jupiter-cirrascale-2 --pack /weka/.../sft_olmo3/packed_32k
"""

import argparse

from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment

HF_MODELS = "/weka/oe-training-default/amandab/hf_models"

#: The arms. Identical apart from the checkpoint -- that is the experiment.
_ARMS = {
    "hils-7b": {
        "model_path": f"{HF_MODELS}/tencent__HiLS-Attention-7B",
        "needs_hils_repo": True,
        "note": "treatment: chunk-wise sparse attention, CPT of Olmo-3-1025-7B",
    },
    "olmo3-7b": {
        "model_path": f"{HF_MODELS}/allenai__Olmo-3-1025-7B",
        "needs_hils_repo": False,
        "note": "control: the base HiLS was continued-pretrained FROM",
    },
}

DEFAULT_SAVE_ROOT = "/weka/oe-training-default/amandab/sft_runs"


def build(arm, cluster, args):
    spec = _ARMS[arm]
    root_dir = get_root_dir(cluster)
    # A run name that already has checkpoints silently RESUMES it -- weights, optimizer and
    # dataloader position -- so the save folder must be unique per run, not per arm.
    run_name = args.run_name or f"{arm}-sft-5task-dolci25-32k"
    save_folder = f"{args.save_root}/{run_name}"

    train = (
        "PYTHONPATH=$HILS_REPO:$PWD/src/scripts:$PWD/src "
        f"torchrun --nproc_per_node={args.ngpu} "
        "src/scripts/train/memexpress/hils_sft/train_sft_veomni.py "
        f"--model-path {spec['model_path']} --data-dir {args.pack} --out-dir {save_folder} "
        f"--max-seq-len {args.max_seq_len} --micro-batch-size {args.micro_batch_size} "
        f"--global-batch-size {args.global_batch_size} --lr {args.lr} --epochs {args.epochs} "
        f"--seed {args.seed} --wandb-name {run_name}"
    )
    if args.max_steps:
        train += f" --max-steps {args.max_steps}"
    if args.dry_run_data:
        train += " --dry-run"

    # Every arm activates the same weka runtime; HILS_NEED_REPO only controls whether the modeling
    # code is checked out. Running the control in a different environment would make the contrast
    # span two torch versions for no reason.
    inner = (
        f"HILS_NEED_REPO={int(spec['needs_hils_repo'])} "
        "source src/scripts/train/memexpress/hils_eval/hils_env_setup.sh && " + train
    )
    cfg = build_launch_config(
        name=f"sft-{run_name}",
        cmd=["bash", "-lc", inner],
        cluster=cluster,
        root_dir=root_dir,
        task_name="train",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=args.num_nodes,
        num_gpus=args.ngpu,
    )
    cfg.torchrun = False  # the command issues its own
    cfg.allow_dirty = True
    cfg.priority = args.priority
    cfg.step_soft_timeout = None
    cfg.step_timeout = None
    return cfg, inner, save_folder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=sorted(_ARMS) + ["all"])
    ap.add_argument("cluster")
    ap.add_argument("--pack", required=True,
                    help="materialized pack dir (sft_shard_dataset.py --out). ALL arms read the "
                         "same one; that is what makes 'same data' literal.")
    ap.add_argument("--run-name", default="", help="default: <arm>-sft-5task-dolci25-32k")
    ap.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--num-nodes", type=int, default=1)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--global-batch-size", type=int, default=8, help="windows per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=34521, help="same across arms -> paired sampling")
    ap.add_argument("--priority", default="urgent")
    ap.add_argument("--dry-run", action="store_true", help="print the submission, do not send it")
    ap.add_argument("--dry-run-data", action="store_true",
                    help="submit a job that builds the data and prints the plan, then exits")
    args = ap.parse_args()

    prepare_cli_environment()
    arms = sorted(_ARMS) if args.arm == "all" else [args.arm]
    if args.arm == "all" and args.run_name:
        raise SystemExit("--run-name with 'all' would give both arms the same save folder, and a "
                         "reused save folder silently resumes the other arm's checkpoint.")

    for arm in arms:
        cfg, inner, save_folder = build(arm, args.cluster, args)
        print(f"\n--- [{arm}] {cfg.name}  ({_ARMS[arm]['note']})")
        print(f"    save_folder: {save_folder}")
        print(f"    {inner}")
        if args.dry_run:
            continue
        workload = cfg.launch(follow=False)
        exp_id = getattr(getattr(workload, "experiment", None), "id", None)
        print(f"    submitted: {exp_id or workload}")


if __name__ == "__main__":
    main()
