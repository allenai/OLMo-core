"""
Reference **CPT** run: continued pretraining on packed documents, loss on every token.

Documents are concatenated and chunked to fill each window, and the budget is counted in tokens
rather than steps, which is the natural unit when the mix and the sequence length both vary.

Local::

    PYTHONPATH=src:ctc/src torchrun --nproc-per-node=8 src/scripts/ctc/train/cpt.py my-cpt \\
        --data /data/prasann/corpora/dolma3longmino --base /data/prasann/bases/qwen3-4b \\
        --arch landmark --seq-len 65536 --max-tokens 15_000_000_000 --lr 1e-4

Beaker::

    PYTHONPATH=src:ctc/src python src/scripts/ctc/train/cpt.py my-cpt \\
        --cluster ai2/jupiter-cirrascale-2 --nodes 8 \\
        --data /weka/.../dolma3longmino --base /weka/.../qwen3-4b \\
        --arch landmark --seq-len 65536 --max-tokens 15000000000 --lr 1e-4

The 20 pre-migration CPT launchers differed from each other only in --arch, --lr and --nodes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv, mode="cpt", description=__doc__))
