"""
The launch half: take resolved options and either run here or hand them to Beaker.

Kept apart from :mod:`recipe` because "what to train" and "where to run it" are independent, and in
the pre-migration tree they were not -- a Beaker launcher and its local twin were separate files
that drifted, so the local run and the cluster run were not the same experiment.
"""

from __future__ import annotations

import sys
from typing import List

from options import TrainOptions, describe

__all__ = ["run"]


def _save_folder(options: TrainOptions, root_dir: str) -> str:
    if options.save_folder:
        return options.save_folder
    if options.is_local:
        # Node-local disk. /accounts and /scratch are both NFS at ~5 MB/s, and a checkpoint write
        # there does not merely run slow -- concurrent readers deadlock in nfs_wait_bit_killable.
        return f"/data/prasann/ctc_runs/{options.run_name}"
    return f"{root_dir}/checkpoints/prasanns/{options.run_name}"


def run(options: TrainOptions, argv: List[str]) -> int:
    """
    Build the experiment and either fit it locally or submit it to Beaker.

    :param options: Resolved options.
    :param argv: The original argument vector, replayed verbatim on the Beaker node so the on-node
        rebuild sees exactly this configuration. The pre-migration launchers re-read some settings
        from the launch host's environment, which is not propagated, so the node silently rebuilt
        with module defaults.

    :returns: Process exit status.
    """
    print(describe(options))

    from olmo_core.internal.common import get_root_dir, get_work_dir
    from olmo_core.utils import prepare_cli_environment, seed_all

    prepare_cli_environment()

    root_dir = get_root_dir(options.cluster) if not options.is_local else ""
    work_dir = str(get_work_dir(root_dir)) if not options.is_local else "/data/prasann/ctc_work"
    save_folder = _save_folder(options, root_dir)

    from recipe import beaker_launch_config, build_experiment

    if not options.is_local:
        launch = beaker_launch_config(
            options,
            cmd=["python", "src/scripts/ctc/train/" + argv[0], *argv[1:]],
            root_dir=root_dir,
        )
        if launch is not None:
            launch.launch(follow=True)
            return 0

    model, train_module, dataset, data_loader, trainer_cfg = build_experiment(
        options, save_folder=save_folder, work_dir=work_dir
    )
    for override in options.overrides:
        trainer_cfg = trainer_cfg.merge([override])

    seed_all(options.seed)
    from olmo_core.distributed.utils import init_distributed

    # GLOO alongside NCCL, matching olmo-core's own `prepare_training_environment` default. The
    # checkpointer runs async saves and bookkeeping collectives on a CPU-capable backend so they do
    # not block training; with NCCL alone the run dies in the checkpointer's `pre_train` with
    # "a CPU-capable backend is required for async checkpointing" -- after the model is built.
    init_distributed(backend="cpu:gloo,cuda:nccl")

    model_instance = model.build()
    train_module_instance = train_module.build(model_instance)
    # `ComposableDataLoaderConfig.build` takes built InstanceSources, not their configs. Handing it
    # the config fails late and obscurely -- `TypeError: object of type
    # 'PadToLengthInstanceSourceConfig' has no len()` from inside the loader, after the model is
    # already on the GPU -- because a Config has no __len__ for the instance-count check.
    loader = data_loader.build(dataset.build(work_dir))
    trainer = trainer_cfg.build(train_module_instance, loader)
    trainer.fit()
    return 0


def main(argv: List[str], *, mode: str, description: str) -> int:
    """
    Shared entry point for ``sft.py`` and ``cpt.py``.

    :param argv: ``sys.argv``; element 0 is the script name, replayed for the Beaker node.
    :param mode: ``"sft"`` or ``"cpt"``.
    :param description: Help text.

    :returns: Process exit status.
    """
    from options import build_parser, options_from_args

    args = build_parser(description, mode=mode).parse_args(argv[1:])
    options = options_from_args(args, mode=mode)
    return run(options, [argv[0].rsplit("/", 1)[-1], *argv[1:]])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv, mode="sft", description=__doc__))
