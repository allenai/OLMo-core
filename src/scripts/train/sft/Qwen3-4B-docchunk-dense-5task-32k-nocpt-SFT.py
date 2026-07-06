"""
Document-chunked 5-task 32k no-CPT SFT (Beaker/gantry) -- DENSE (DocumentChunkedAttention, cross_doc_mode=chunked), dense base.

Reads the doc-chunked weka data built by src/scripts/data/convert_docchunk_5task_gantry.sh and
mixes the 5 tasks at the headline weights (contra 2 / rerank 1.5 / outlier 1.5 / nq 1 / oolong 1).
See _docchunk_5task_32k_nocpt_common.py for all the shared config. Single 8xH200 node, PadToLength
(one chunked example per 40960 window), FSDP, no CP (doc-chunked attention reconstructs chunk_ids
from the box markers and needs one EOS-terminated example per instance).

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-docchunk-dense-5task-32k-nocpt-SFT.py \\
        launch q4b-docchunk-dense-5task-32k-nocpt ai2/neptune-cirrascale
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmo_core.internal.experiment import main  # noqa: E402
from _docchunk_5task_32k_nocpt_common import build_docchunk_experiment  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_docchunk_experiment, variant="dense"))
