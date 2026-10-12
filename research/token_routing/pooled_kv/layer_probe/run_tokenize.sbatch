#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=lsp-tok
#SBATCH --output=/data/prasann/layer_probe/tok_%j.out
set -x
export TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/hf TOKENIZERS_PARALLELISM=false
mkdir -p $TMPDIR /data/prasann/layer_probe/work
REPO_ROOT="${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PATH=/data/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
R=/data/prasann/ctc_suite_staged/eval_rungs
date
PYTHONPATH=src $PY - <<'EOF'
import sys
sys.path.insert(0, "research/token_routing/pooled_kv")
from eval_side_slot_probe import convert
jobs = [
    ("contradiction", "/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl", 200, "/data/prasann/layer_probe/work/contra_iid_8192_r200"),
    ("oolong",        "/data/prasann/ctc_suite_staged/eval_rungs/oolong/rung_8192.jsonl",            200, "/data/prasann/layer_probe/work/oolong_8192_r200"),
    ("contradiction", "/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_32768.jsonl", 64, "/data/prasann/layer_probe/work/contra_iid_32768_r64"),
]
for task, jsonl, rows, out in jobs:
    print("==", task, rows, out, flush=True)
    convert(task, jsonl, rows, out)
print("TOKENIZE_DONE")
EOF
date; echo DONE
