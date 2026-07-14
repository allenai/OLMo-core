"""
Document-chunked 5-task 32k no-CPT SFT (Beaker/gantry) -- DENSE
(:class:`DocumentChunkedAttention`, ``cross_doc_mode="chunked"``) with **FlexAttention
``flex_block_size=128``** (the kernel-valid default), dense base.

Identical to ``Qwen3-4B-docchunk-dense-5task-32k-nocpt-SFT.py`` (same 5-task doc-chunked weka mix,
dense CPT base, PadToLength one-chunked-example-per-40960-window layout, FSDP on a single 8xH200 node,
no CP).

NOTE on block size: an earlier attempt used ``flex_block_size=32`` for extra block-sparsity, but the
FlexAttention Triton kernel REQUIRES the create_block_mask BLOCK_SIZE to be divisible by the kernel's
BLOCK_M/BLOCK_N (>=64) -- 32 fails to lower (``ValueError: Q and KV block size must be divisible by
BLOCK_M and BLOCK_N``) and silently falls back to the dense (B,T,T) mask, which OOMs at T=40960. Block
size affects ONLY which fully-masked blocks are skipped (a speed optimization); the per-element mask
(hence the trained model) is identical at any block size. So 128 trains the exact same chunked model,
just without the sub-128-chunk sparsity speedup. (Follow-up: 64 may be valid + a bit faster.)

    PYTHONPATH=src python src/scripts/train/sft_docchunk/Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py \\
        dry_run q4b-docchunk-5task-32k-nocpt-bs128 ai2/jupiter
    PYTHONPATH=src python src/scripts/train/sft_docchunk/Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py \\
        launch  q4b-docchunk-5task-32k-nocpt-bs128 ai2/jupiter
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmo_core.internal.experiment import main  # noqa: E402
from _docchunk_5task_32k_nocpt_common import build_docchunk_experiment  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_docchunk_experiment, variant="dense", flex_block_size=128))
