"""
32k-scale, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B
**MULTI-LANDMARK COMPRESSIVE** block-64 CPT model, 2lm-mean arm:
num_landmarks=2, landmark_gate_pool="mean", mem_freq=62 (block_size = 64).

Init (weights-only) from that arm's own CPT run:
qwen3-4b-mcl-block64-2lm-mean/step2385 -- geometry MUST match, and is taken from
_mcl_block64_5task_dolci25_32k_nocpt_common.py's ``_ARMS["2lm-mean"]`` for both the model and the
data pipeline. Content capacity is 39680 tokens per 40960-token window.

Data: 75% the 5 long-context tasks / 25% Dolci-Instruct-SFT, no raw CPT text -- identical to
Qwen3-4B-compressive-block64-5task-dolci25-32k-nocpt-SFT.py (the single-landmark block-64 arm), so
this is directly comparable to the block64 row of results/block_sweep_sft_5task.csv.

See _mcl_block64_5task_dolci25_32k_nocpt_common.py for all the shared config.

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt-SFT.py \
        dry_run q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt-SFT.py \
        launch q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt ai2/jupiter-cirrascale-2 \
        --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _mcl_block64_5task_dolci25_32k_nocpt_common import build_mcl_experiment  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_mcl_experiment, arm="2lm-mean"))
