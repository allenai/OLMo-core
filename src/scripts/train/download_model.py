from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="allenai/OLMo-2-0425-1B",
    revision="stage1-step1907359-tokens4001B",
)
print("Downloaded to:", local_dir)
