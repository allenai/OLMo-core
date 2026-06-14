"""
Convert Qwen/Qwen3-30B-A3B from Hugging Face to an OLMo-core checkpoint.

This is a thin default-setting wrapper around ``convert_qwen3_moe_hf_to_olmo.py``.
Use ``--max-layers N`` for a first-N-layers smoke checkpoint when the full model
load is too large for the target machine.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--hf-model" not in sys.argv:
    sys.argv[1:1] = ["--hf-model", "Qwen/Qwen3-30B-A3B"]

runpy.run_path(
    str(Path(__file__).with_name("convert_qwen3_moe_hf_to_olmo.py")),
    run_name="__main__",
)
