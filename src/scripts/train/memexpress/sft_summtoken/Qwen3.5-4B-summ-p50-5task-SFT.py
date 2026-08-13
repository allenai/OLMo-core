"""
SummTokenSFT -- **summ-p50** arm (Beaker/gantry).

**50% mask mixing throughout**: each example is independently masked or causal with probability 0.5, constant for the whole run. At exactly 0.5 the two readings of "50% mask mixing" coincide, so there is no direction to get wrong.

All shared config (geometry, data, base checkpoint, mixture wiring) lives in
_qwen35_summtoken_common.py; read its module docstring before launching -- in particular the two
standing caveats: only 8 of 32 layers carry the mask on this hybrid, and the base must be
summary-repaired first.

    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-p50-5task-SFT.py \\
        launch_prep q35-4b-summ-p50-prep ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-p50-5task-SFT.py \\
        launch q35-4b-summ-p50-5task ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_summtoken_common import build_summtoken_experiment  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_summtoken_experiment, arm="summ-p50"))
