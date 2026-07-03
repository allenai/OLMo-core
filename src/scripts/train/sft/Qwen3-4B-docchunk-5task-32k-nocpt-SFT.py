"""
Document-chunked 5-task 32k no-CPT SFT (Beaker/gantry) -- DENSE
(:class:`DocumentChunkedAttention`, ``cross_doc_mode="chunked"``) with **FlexAttention
``flex_block_size=32``**, dense base.

Identical to ``Qwen3-4B-docchunk-dense-5task-32k-nocpt-SFT.py`` (same 5-task doc-chunked weka mix,
dense CPT base, PadToLength one-chunked-example-per-40960-window layout, FSDP on a single 8xH200 node,
no CP) EXCEPT the FlexAttention block-mask granularity is shrunk from the default 128 to **32**.

Why 32: the exact block-skip-fraction analysis (see /scratch/users/prasann/olmo_overnight/
docchunk_bench.md) shows that with realistic inter-document FREE separators, ``flex_block_size=128``
realizes ~0% block-sparsity for the mix's dominant sub-128-token chunks (contradiction claims,
~100-word outlier/nq/rerank docs) -- i.e. flex degenerates to full dense attention -- while
``flex_block_size=32`` recovers ~40-60% of the skippable blocks. (Very small line-item tasks like
oolong still can't realize sparsity at any block size; large-doc tasks already benefit at 128.)

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py \\
        dry_run q4b-docchunk-5task-32k-nocpt-bs32 ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py \\
        launch  q4b-docchunk-5task-32k-nocpt-bs32 ai2/jupiter
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmo_core.internal.experiment import main  # noqa: E402
from _docchunk_5task_32k_nocpt_common import build_docchunk_experiment  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_docchunk_experiment, variant="dense", flex_block_size=32))
