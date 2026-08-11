"""
256k-context Beaker/gantry SFT of the Qwen3.5-4B **DENSE** 256k CPT model (the GDN hybrid with its
full-attention blocks left alone) on 75% the **query-after** 2k-256k 5-task mix / 25%
``allenai/Dolci-Instruct-SFT``.

This is the **treatment** arm of the query-position pair. Its control is
``Qwen3.5-4B-dense-xlong5-qboth-dolci25-256k-SFT.py``, which is the same config against the
original ``--query-position both`` shards.

All shared configuration -- base checkpoint, 75/25 blend, within-mix weights, 262,144 window,
2-node/CP=4 parallelism, LR 4e-5, 2,240-step budget, seed -- lives in
``_qwen35_xlong5_dolci25_256k_common.py``. Read its module docstring for the contrast, the readout,
the LR derivation, and the one axis the two arms do not share (the qafter build's tighter 250,000
instance-length cap). This file only selects the arm.

    S=src/scripts/train/memexpress/sft_xlong256k/Qwen3.5-4B-dense-xlong5-qafter-dolci25-256k-SFT.py

    PYTHONPATH=src python $S dry_run q35-4b-dense-xlong5-qafter-dolci25-256k ai2/jupiter-cirrascale-2

    # Build the mixture on CPU first -- dry_run does NOT touch the data. Read the
    # 'MixingInstanceSource: NNB tokens' and 'packed N windows' lines out of the job log.
    PYTHONPATH=src python $S launch_prep q35-4b-dense-xlong5-qafter-dolci25-256k-prep \\
        ai2/jupiter-cirrascale-2

    PYTHONPATH=src python $S launch q35-4b-dense-xlong5-qafter-dolci25-256k \\
        ai2/jupiter-cirrascale-2 --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_xlong5_dolci25_256k_common import (  # noqa: E402
    build_qwen35_xlong5_experiment,
)

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402

ARM = "qafter"


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the query-after arm's experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    return build_qwen35_xlong5_experiment(cli_context, arm=ARM)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
