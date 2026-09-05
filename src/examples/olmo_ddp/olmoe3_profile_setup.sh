#!/usr/bin/env bash
# Runs inside the disposable Beaker image. Never echo the GitHub secret or put it in a URL.
set -euo pipefail
gh auth setup-git
uv pip install --python "$(command -v python)" --no-deps \
  'kernel-fun @ git+https://github.com/allenai/kernel-fun.git@7a6983baf2beb4ec4d7fe914ec9f6670438af99b#subdirectory=packages/kernel-fun'
python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace --backend cmake
python - <<'PY'
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
import torch
from olmo_core.nn.attention.flash_linear_attn_api import has_kernel_fun

assert has_kernel_fun(), 'kernel-fun must be installed, not silently replaced'
assert torch.__version__.startswith('2.11.'), torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
dist = importlib.metadata.distribution('kernel-fun')
direct = json.loads(dist.read_text('direct_url.json'))
assert direct['vcs_info']['commit_id'] == '7a6983baf2beb4ec4d7fe914ec9f6670438af99b', direct
print('Pinned kernel-fun verified:', dist.version, direct['vcs_info']['commit_id'], flush=True)
nsys = shutil.which('nsys') or '/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys'
assert Path(nsys).is_file(), 'Nsight Systems is missing'
subprocess.run([nsys, '--version'], check=True)
PY
