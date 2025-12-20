#!/usr/bin/env python

"""
Example usage:

for STEP in step1000 step2000 step3000; do
  JOBNAME="olmo-convert-${STEP}"
  gantry run \
    --allow-dirty \
    --workspace ai2/oe-data \
    --priority high \
    --cluster ai2/augusta \
    --gpus 1 \
    --env-secret HF_TOKEN \
    --env STEP="$STEP" \
    --name "$JOBNAME" \
    -- bash -lc '
      pip install -U "huggingface_hub>=0.25.1" hf-transfer && \
      python -u src/scripts/run_conversion.py
    ' &
done
wait
"""


import os
import subprocess
from huggingface_hub import HfApi, upload_folder

# Get the step from environment or CLI
step = os.environ.get("STEP")
if not step:
    raise SystemExit("❌ Environment variable STEP not set!")

# Paths
# src = f"gs://ai2-llm/checkpoints/memo-7b-20251216T012857+0000/{step}"
src = f"/weka/oe-training-default/ai2-llm/checkpoints/memo-7b-4tier/{step}"
out_root = os.environ.get("OUT_ROOT", "/data")   # defaults to ephemeral /data
out = f"{out_root}/{step}-hf"

# repo_id = "yapeichang/memo-7b"
repo_id = "yapeichang/memo-7b-4tier"
branch = "main"

print(f"=== Converting {step} ===")
subprocess.run([
    "python", "src/examples/huggingface/convert_checkpoint_to_hf.py",
    f"--checkpoint-input-path={src}",
    f"--huggingface-output-dir={out}",
    "-s", "8192"
], check=True)

print(f"=== Uploading to {repo_id}@{branch} ===")
api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.create_branch(repo_id, branch=branch, repo_type="model", exist_ok=True)
upload_folder(
    repo_id=repo_id,
    folder_path=out,
    repo_type="model",
    revision=branch,
    commit_message=f"Upload {branch}",
)
print(f"✅ Uploaded {out} → https://huggingface.co/{repo_id}/tree/{branch}")
