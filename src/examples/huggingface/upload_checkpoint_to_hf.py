import argparse
import math
import os

from huggingface_hub import HfApi, login
from tqdm import tqdm


def upload_to_branch(local_checkpoint_dir: str, repo_id: str, step: int, token: str):
    login(token=token)
    api = HfApi()
    total_tokens = step * 2048 * 4096
    tokens_b = math.ceil(total_tokens / 1_000_000_000)
    branch = f"stage1-step{step}-tokens{tokens_b}B"
    print(f"Creating and uploading to branch: {branch}")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        api.create_branch(repo_id=repo_id, branch=branch, token=token)
        print(f"Created new branch: {branch}")
    except Exception as e:
        print(f"Branch might already exist: {e}")
    files_to_upload = []
    for root, _, files in os.walk(local_checkpoint_dir):
        for file in files:
            local_path = os.path.join(root, file)
            repo_path = os.path.relpath(local_path, local_checkpoint_dir)
            files_to_upload.append((local_path, repo_path))

    print(f"\nStarting upload of {len(files_to_upload)} files...")

    for local_path, repo_path in tqdm(files_to_upload, desc="Uploading files"):
        try:
            print(f"\nUploading: {repo_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=token,
                revision=branch,
            )
            print(f"Successfully uploaded {repo_path}")
        except Exception as e:
            print(f"Error uploading {repo_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--local_checkpoint_dir",
        type=str,
        required=True,
        help="Local directory containing checkpoint files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Hugging Face repo ID (e.g., "allenai/OLMo-2-0325-32B")',
    )
    parser.add_argument("--step", type=int, required=True, help="Step number")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    args = parser.parse_args()

    print("Starting upload process...")
    if not os.path.exists(args.local_checkpoint_dir):
        print("Error: Directory not found!")
    else:
        print(f"Found directory. Contents: {os.listdir(args.local_checkpoint_dir)}")
        upload_to_branch(
            local_checkpoint_dir=args.local_checkpoint_dir,
            repo_id=args.repo_id,
            step=args.step,
            token=args.token,
        )
