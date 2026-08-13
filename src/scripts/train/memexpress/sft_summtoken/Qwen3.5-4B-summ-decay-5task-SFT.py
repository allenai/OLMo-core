"""
SummTokenSFT -- **summ-decay** arm (Beaker/gantry).

**100% mask mixing decaying to 0%**: starts with every example summary-only and ends with every example plain causal, linearly. In the code's parameterization p is P(causal), so this is mix_start_p=0.0 -> mix_end_p=1.0. Note the model therefore finishes training under plain causal attention.

All shared config (geometry, data, base checkpoint, mixture wiring) lives in
_qwen35_summtoken_common.py; read its module docstring before launching -- in particular the two
standing caveats: only 8 of 32 layers carry the mask on this hybrid, and the base must be
summary-repaired first.

    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-decay-5task-SFT.py \\
        launch_prep q35-4b-summ-decay-prep ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-decay-5task-SFT.py \\
        launch q35-4b-summ-decay-5task ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_summtoken_common import build_summtoken_experiment  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_summtoken_experiment, arm="summ-decay"))
