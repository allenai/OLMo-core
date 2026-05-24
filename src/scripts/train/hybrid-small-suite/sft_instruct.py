"""
OLMo hybrid small suite — 275M, 810M, and 1.4B SFT Instruct runs.

Architecture: see arch.py (identical to pretraining/midtraining — checkpoints load cleanly).

Usage::

  python src/scripts/train/hybrid-small-suite/sft_instruct.py dry_run \\
      hybrid-small-sft-instruct-275M ai2/jupiter

  python src/scripts/train/hybrid-small-suite/sft_instruct.py launch \\
      hybrid-small-sft-instruct-275M ai2/titan \\
      --launch.num_nodes=1 --launch.priority=urgent --launch.budget=ai2/oe-other
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sft_common import SEQUENCE_LENGTH, run_sft

DATASET_PATH = "/weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114"

SFT_INSTRUCT_CONFIGS = {
    "275m": dict(
        lr=2.5e-5,
        global_batch_size=64 * SEQUENCE_LENGTH,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-275M/latest/",
    ),
    "810m": dict(
        lr=2.5e-5,
        global_batch_size=64 * SEQUENCE_LENGTH,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-810M/latest/",
    ),
    "1.4b": dict(
        lr=2.5e-5,
        global_batch_size=64 * SEQUENCE_LENGTH,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-1.4B/latest/",
    ),
}

if __name__ == "__main__":
    run_sft(sft_configs=SFT_INSTRUCT_CONFIGS, dataset_path=DATASET_PATH, tags=["sft-instruct"])
