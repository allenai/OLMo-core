"""
Document-chunked 5-task 32k no-CPT SFT (Beaker/gantry) -- HIERARCHICAL-DILATED with a **large base**
(:class:`DocumentChunkedAttention`, ``cross_doc_mode="hierarchical_dilated"``, ``dilation_n=4``,
``dilation_m=25``, ``dilation_cycle=3``).

Identical in every respect to ``Qwen3-4B-docchunk-hier-5task-32k-nocpt-SFT.py`` (same dense doc-chunked
weka mix, dense CPT base, 4x8 H200 FSDP, PadToLength 40960, 1100 steps) EXCEPT the dilation base ``k``
is 25 instead of 2. With a 3-layer rotation (``dilation_cycle=3``) the per-layer stride rotates over
``k**0, k**1, k**2`` = ``1, 25, 625`` documents across the depth cycle: layer-cycle-position 0 attends
the 4 nearest docs, position 1 the 4 docs at stride 25, position 2 the 4 docs at stride 625 -- so a
much wider (coarser) receptive field per rotation than the ``k=2`` (``1, 2, 4``) prior run. This is the
"existing dilated code" k=25 probe (n-doc window semantics), a fast comparison point before the
separate base-k full-attention-rotation variant.

Sweep the base / rotation / window via env (they are baked as defaults here and propagated to the
Beaker job): ``DOCCHUNK_DILATION_M`` (base k), ``DOCCHUNK_DILATION_CYCLE`` (rotation period),
``DOCCHUNK_DILATION_N`` (#docs/layer).

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-docchunk-hierK25-5task-32k-nocpt-SFT.py \\
        launch q4b-docchunk-hierK25-5task-32k-nocpt ai2/jupiter-cirrascale-2
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Bake the k=25 / 3-layer-rotation / 4-doc-window schedule as defaults (do not clobber an explicit
# launch-host override). Read inside build_docchunk_experiment AND propagated to the on-node rebuild.
os.environ.setdefault("DOCCHUNK_DILATION_M", "25")
os.environ.setdefault("DOCCHUNK_DILATION_CYCLE", "3")
os.environ.setdefault("DOCCHUNK_DILATION_N", "4")

from _docchunk_5task_32k_nocpt_common import build_docchunk_experiment  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(config_builder=partial(build_docchunk_experiment, variant="hierarchical"))
