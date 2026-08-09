#!/usr/bin/env bash
# Grade a checkpoint.  All arguments are passed through to `ctc-eval`.
#
#   run/eval.sh --list-backends
#   run/eval.sh --ckpt /data/ckpts/q35-4b/step1100 --task contradiction --rungs 2k
#   run/eval.sh --ckpt … --suite ctc_suite --rungs all --backend vllm --attn chunked
#
# No logic lives here: this resolves the environment and execs Python, so the identical code path
# runs by hand, under sbatch, and under gantry.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec "$CTC_PYTHON" -m ctc.eval.cli "$@"
