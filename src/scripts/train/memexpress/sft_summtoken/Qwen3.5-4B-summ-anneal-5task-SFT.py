"""
SummTokenSFT -- **summ-anneal** arm (Beaker/gantry).

MODE 3: the causal fraction rises linearly from 0% to 50% across training.

All shared config (geometry, data, base checkpoint, mixture wiring) lives in
_qwen35_summtoken_common.py; read its module docstring before launching -- in particular the two
standing caveats: only 8 of 32 layers carry the mask on this hybrid, and the base must be
summary-repaired first.

    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-anneal-5task-SFT.py \\
        dry_run q35-4b-summ-anneal-5task ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-anneal-5task-SFT.py \\
        launch_prep q35-4b-summ-anneal-5task-prep ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-anneal-5task-SFT.py \\
        launch q35-4b-summ-anneal-5task ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_summtoken_common import build_summtoken_experiment  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_summtoken_experiment, arm="summ-anneal"))
