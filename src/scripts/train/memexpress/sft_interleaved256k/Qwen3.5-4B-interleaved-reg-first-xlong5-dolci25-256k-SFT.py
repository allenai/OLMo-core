"""
256k-context Beaker/gantry SFT of the Qwen3.5-4B **interleaved landmark ``reg-first``** CPT model on
75% the xlong5 2k-256k 5-task mix (qboth build) / 25% ``allenai/Dolci-Instruct-SFT``.

Arm layout -- ``reg`` (fast_compressive_landmark) on the FIRST full-attention layer (3), ``sparse``
on the other seven. Together with ``reg-last`` this brackets the 1/7 extreme: same count of
regular-landmark layers, opposite end of the stack.

Continues from ``q35-4b-il-regfirst-256k/step2385``. All shared configuration -- the 75/25 blend, the within-mix
weights, the 262,144 landmark window, 2-node/CP=4 parallelism, LR 4e-5, the 2,240-step budget and
the seed -- lives in ``_qwen35_interleaved_xlong5_dolci25_256k_common.py``, which also imports this
arm's per-layer attention pattern straight from ``cpt/interleaved`` so the SFT model cannot drift
from the checkpoint it loads. Read that module's docstring before interpreting a result: it records
the one axis these arms do NOT share with the dense 256k control (landmark content capacity).

This file only selects the arm.

    S=src/scripts/train/memexpress/sft_interleaved256k/Qwen3.5-4B-interleaved-reg-first-xlong5-dolci25-256k-SFT.py

    PYTHONPATH=src python $S dry_run q35-4b-il-regfirst-xlong5-dolci25-256k ai2/jupiter-cirrascale-2

    # Build the mixture on CPU first -- dry_run does NOT touch the data. Read the
    # 'MixingInstanceSource: NNB tokens' and 'LandmarkPackingInstanceSource packed' lines, AND the
    # long-document drop warning, out of the job log.
    PYTHONPATH=src python $S launch_prep q35-4b-il-regfirst-xlong5-dolci25-256k-prep ai2/jupiter-cirrascale-2

    PYTHONPATH=src python $S launch q35-4b-il-regfirst-xlong5-dolci25-256k \\
        ai2/jupiter-cirrascale-2 --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_interleaved_xlong5_dolci25_256k_common import (  # noqa: E402
    build_qwen35_interleaved_sft_experiment,
)

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402

ARM = "reg-first"


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the ``reg-first`` arm's experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    return build_qwen35_interleaved_sft_experiment(cli_context, arm=ARM)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
