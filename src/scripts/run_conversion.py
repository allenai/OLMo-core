#!/usr/bin/env python
import os
import subprocess
from huggingface_hub import HfApi, upload_folder

# Get the step from environment or CLI
step = os.environ.get("STEP")
if not step:
    raise SystemExit("❌ Environment variable STEP not set!")

# Paths
src = f"gs://ai2-llm/checkpoints/stego32-highlr-filter3/{step}"
out_root = os.environ.get("OUT_ROOT", "/data")   # defaults to ephemeral /data
out = f"{out_root}/{step}-hf"

repo_id = "allenai/Olmo-3-1125-32B"
branch = f"stage1-{step}"

print(f"=== Converting {step} ===")
subprocess.run([
    "python", "src/examples/huggingface/convert_checkpoint_to_hf.py",
    f"--checkpoint-input-path={src}",
    f"--huggingface-output-dir={out}",
    "-s", "8192"
], check=True)

print(f"=== Uploading to {repo_id}@{branch} ===")
api = HfApi()
api.create_branch(repo_id, branch=branch, repo_type="model", exist_ok=True)
upload_folder(
    repo_id=repo_id,
    folder_path=out,
    repo_type="model",
    revision=branch,
    commit_message=f"Upload {branch}",
)
print(f"✅ Uploaded {out} → https://huggingface.co/{repo_id}/tree/{branch}")

