"""
Document-chunked 5-task 32k no-CPT SFT (Beaker/gantry) -- COMPRESSIVE
(DocumentCompressiveLandmarkAttention, eager: chunked grouped-softmax where each past block's landmark
token also contributes its VALUE as a compressed block summary). Reuses the SAME landmark-format
doc-chunked weka data as the landmark variant; initialized from the compressive CPT base.

See _docchunk_5task_32k_nocpt_common.py for all the shared config.

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-docchunk-compressive-5task-32k-nocpt-SFT.py \
        launch q4b-docchunk-compressive-5task-32k-nocpt ai2/neptune-cirrascale
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmo_core.internal.experiment import main  # noqa: E402
from _docchunk_5task_32k_nocpt_common import build_docchunk_experiment  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_docchunk_experiment, variant="compressive"))
