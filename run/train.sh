#!/usr/bin/env bash
# Train: SFT (default) or CPT, locally or on Beaker, through the one shared recipe.
#
#   run/train.sh my-run --data DIR:2 --data DIR2:1 --base CKPT --arch chunked --max-steps 1100
#   run/train.sh --cpt my-run --data DIR --base CKPT --arch full --max-tokens 2e9
#   run/train.sh my-run --cluster ai2/jupiter-cirrascale-2 --nodes 4 ...   # Beaker launch
#
# Local multi-GPU wants torchrun; set CTC_NPROC to wrap the run in it:
#
#   CTC_NPROC=8 run/train.sh my-run --data ... --base ... --arch chunked
#
# (torchrun is spelled `$CTC_PYTHON -m torch.distributed.run` on purpose -- a bare `torchrun` on
# this cluster resolves to the system miniforge, not your interpreter.)
#
# No logic lives here: this resolves the environment and execs Python, so the identical code path
# runs by hand, under sbatch, and under gantry.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

entry="$CTC_REPO/src/scripts/ctc/train/sft.py"
if [[ "${1:-}" == "--cpt" ]]; then
  entry="$CTC_REPO/src/scripts/ctc/train/cpt.py"
  shift
fi

if [[ -n "${CTC_NPROC:-}" ]]; then
  exec "$CTC_PYTHON" -m torch.distributed.run --nproc-per-node="$CTC_NPROC" "$entry" "$@"
fi
exec "$CTC_PYTHON" "$entry" "$@"
