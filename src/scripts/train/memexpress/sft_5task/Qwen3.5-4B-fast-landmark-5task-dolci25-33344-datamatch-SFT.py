"""
Qwen3.5-4B FAST LANDMARK, **data-matched** arm of the landmark data-vs-compute ablation.

Setting 1 of two: consume exactly the same original data as the dense baseline
``q35-4b-dense-5task-dolci25-32k-nocpt`` (the same 32.86% of the same document stream) and let the
landmark arm spend the extra tokens that its landmark tokens and per-document block padding cost.
Its sibling ``...-33344-tokenmatch-SFT.py`` holds the token budget fixed instead.

Window is 33344 (521 blocks of 64, content capacity 32,823 >= the dense 32,768) and packing is
best-fit-decreasing, so the only remaining difference from the baseline is the landmark geometry
itself. All shared configuration -- including the derivation of the step count from the measured
instance counts -- lives in ``_qwen35_5task_dolci25_32k_nocpt_common.py``; see its module
docstring. This file only selects the arm.

    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3.5-4B-fast-landmark-5task-dolci25-33344-datamatch-SFT.py \\
        launch_prep q35-4b-fastlm-5task-dolci25-33344-datamatch ai2/jupiter \\
        --launch.follow=false --launch.step_soft_timeout=null
    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3.5-4B-fast-landmark-5task-dolci25-33344-datamatch-SFT.py \\
        dry_run q35-4b-fastlm-5task-dolci25-33344-datamatch ai2/jupiter
    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3.5-4B-fast-landmark-5task-dolci25-33344-datamatch-SFT.py \\
        launch  q35-4b-fastlm-5task-dolci25-33344-datamatch ai2/jupiter \\
        --launch.follow=false --launch.step_soft_timeout=null

The ``launch_prep`` run comes first: it is CPU-only and reports the landmark instance count that
``LANDMARK_ABLATION_INSTANCES`` needs. Until that is set, this arm refuses to build.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_5task_dolci25_32k_nocpt_common import (  # noqa: E402
    build_qwen35_sft_experiment,
)

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402

ARM = "fast-landmark-datamatch"


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the data-matched fast-landmark arm's experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    return build_qwen35_sft_experiment(cli_context, arm=ARM)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
