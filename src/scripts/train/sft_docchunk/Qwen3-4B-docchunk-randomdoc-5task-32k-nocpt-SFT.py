"""
Document-chunked 5-task 32k no-CPT SFT (Beaker/gantry) -- RANDOM_DOC
(:class:`DocumentChunkedAttention`, ``cross_doc_mode="random_doc"``): each context document attends to
itself + a seeded-random ~``DOC_KEEP_PROB`` (default 0.1 = 10%) subset of the STRICTLY EARLIER
documents; the free query/answer/instruction tokens stay global. Dense CPT base, FlexAttention
block-sparse path (``flex_block_size=128``).

Same 5-task doc-chunked weka mix, dense doc-chunked data, PadToLength one-chunked-example-per-40960
window, 2xH200-node FSDP (no CP) as the dense launcher -- only the cross-document mask differs. The
keep fraction / seed are hyperparameters: override at launch via
``DOCCHUNK_RANDOM_DOC_KEEP_PROB`` / ``DOCCHUNK_RANDOM_DOC_SEED`` env (encode the value in the run name).

    # 10% (default):
    PYTHONPATH=src python src/scripts/train/sft_docchunk/Qwen3-4B-docchunk-randomdoc-5task-32k-nocpt-SFT.py \\
        launch q4b-docchunk-randomdoc10-5task-32k-nocpt-fixed ai2/jupiter-cirrascale-2
    # sweep another fraction:
    DOCCHUNK_RANDOM_DOC_KEEP_PROB=0.25 PYTHONPATH=src python .../Qwen3-4B-docchunk-randomdoc-...SFT.py \\
        launch q4b-docchunk-randomdoc25-5task-32k-nocpt-fixed ai2/jupiter-cirrascale-2
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmo_core.internal.experiment import main  # noqa: E402
from _docchunk_5task_32k_nocpt_common import build_docchunk_experiment  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_docchunk_experiment, variant="random_doc", flex_block_size=128))
