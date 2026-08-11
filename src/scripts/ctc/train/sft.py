"""
Reference **SFT** run: supervised fine-tuning on task shards, loss on answer tokens only.

One example per sequence window, padded -- the document-chunked masks reconstruct chunk roles from
the box markers, which needs one EOS-terminated example per instance, so this layout cannot pack or
use context parallelism.

Local (Berkeley H200, no weka/Beaker)::

    PYTHONPATH=src:ctc/src torchrun --nproc-per-node=8 src/scripts/ctc/train/sft.py my-run \\
        --data /data/prasann/ctc/shards/contradiction:2 \\
        --data /data/prasann/ctc/shards/nq:1 \\
        --base /data/prasann/bases/q4b-dense-fixmark --arch chunked --max-steps 1100

Beaker::

    PYTHONPATH=src:ctc/src python src/scripts/ctc/train/sft.py my-run \\
        --cluster ai2/jupiter-cirrascale-2 --nodes 4 \\
        --data /weka/.../shards/contradiction:2 --data /weka/.../shards/nq:1 \\
        --base /weka/.../q4b-dense-dolma3longmino/step2385/model_and_optim \\
        --arch chunked --max-steps 1100

Everything else -- optimizer, schedule, sharding, checkpointing, the format-fingerprint callback --
is in ``recipe.py`` and is the same for both.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv, mode="sft", description=__doc__))
