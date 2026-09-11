#!/usr/bin/env bash
# Exact qualified hero image plus explicit runtime repair; no unconstrained all-extras solve.
set -euo pipefail
gh auth setup-git
uv pip install --python "$(command -v python)" --no-deps \
  . 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'nvidia-nccl-cu13==2.28.9' \
  'kernel-fun @ git+https://github.com/allenai/kernel-fun-dev.git@7a6983baf2beb4ec4d7fe914ec9f6670438af99b#subdirectory=packages/kernel-fun'
python src/examples/olmo_ddp/olmoe3_hero_decay_runtime.py
if [[ "${OLMO35_DECAY_CPU_VALIDATE:-0}" != "1" ]]; then
  python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace --backend cmake
fi
