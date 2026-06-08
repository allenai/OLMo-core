import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

CONVERT_SCRIPT = str(
    Path(__file__).resolve().parents[4].parent
    / "transformers"
    / "src"
    / "transformers"
    / "models"
    / "olmo_hybrid_small"
    / "convert_olmo_hybrid_small_weights_to_hf.py"
)

# Pretraining / midtraining / long-context: dolma2 base tokenizer, no chat template.
TOKENIZER_ID = "allenai/Olmo-3-1025-7B"

# SFT: tokenizer source for converted SFT checkpoints.
# This is the think-aware tokenizer used for evals; its chat template is embedded
# in tokenizer_config.json (no separate chat_template.jinja).
# Supported values:
#   1) local directory path
#   2) HF repo id (e.g. "allenai/olmo-3.2-tokenizer-think-dev")
#   3) HF URL (e.g. "https://huggingface.co/allenai/olmo-3.2-tokenizer-think-dev/tree/main")
SFT_TOKENIZER_SOURCE = "allenai/olmo-3.2-tokenizer-think-dev"

SFT_TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _repo_id_from_hf_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"Not a Hugging Face URL: {url}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse repo id from URL: {url}")
    return f"{parts[0]}/{parts[1]}"


def _resolve_sft_tokenizer_source() -> Path:
    source = SFT_TOKENIZER_SOURCE.strip()
    local_dir = Path(source)
    if local_dir.is_dir():
        return local_dir

    repo_id = source
    if source.startswith("http://") or source.startswith("https://"):
        repo_id = _repo_id_from_hf_url(source)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required when SFT_TOKENIZER_SOURCE is an HF repo/url. "
            "Install with: pip install huggingface_hub"
        ) from e

    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=list(SFT_TOKENIZER_FILES),
    )
    return Path(snapshot_dir)

# Pretraining Checkpoints
pretraining_checkpoints = {
    # "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-275M-Cx100/step161186/",
    "450m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-450m-cx100-lr8e-3/step179814/",
    # "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-810M-Cx100/step269926/",
    # "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-1.4B-Cx100/step308433/",
}

midtraining_checkpoints = {
    # "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-275M-v2-lr1.6e-3/step38147/",
    "450m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-450m/step38147/",
    # "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-v2-810M-lr4e-4/step23842/",
    # "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-v2-1.4b-lr4e-4/step11921/",
}

long_context_checkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-275m/step47684/",
    # "450m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v2-450m/step47684/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-810m/step23842/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-1.4b/step23842/",
}

long_context_debug_checkpoints = {
    "hybrid-small-lc-v3-275m-lr1.6e-3-train-5e724344": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-275m-lr1.6e-3/step47684/",
    "hybrid-small-lc-v3-275m-lr8e-4-train-6788f1fd": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-275m-lr8e-4/step47684/",
    "hybrid-small-lc-v3-275m-lr4e-4-train-9c734a9e": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-275m-lr4e-4/step47684/",
    "hybrid-small-lc-v3-275m-lr2e-4-train-eadd4a0b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-275m-lr2e-4/step47684/",
    # "hybrid-small-lc-v3-1.4b-lr4e-4-train-db94a8c2": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-1.4b-lr4e-4/step47684/",
    # "hybrid-small-lc-v3-1.4b-lr8e-4-train-7936ddea": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-1.4b-lr8e-4/step47684/",
    # "hybrid-small-lc-v3-1.4b-lr2e-4-train-9d33183e": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-1.4b-lr2e-4/step47684/",
    # "hybrid-small-lc-v3-1.4b-lr1e-4-train-f044b385": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-1.4b-lr1e-4/step47684/",
    # "hybrid-small-lc-v3-810m-lr8e-4-train-b68c6e73": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-810m-lr8e-4/step47684/",
    # "hybrid-small-lc-v3-810m-lr4e-4-train-abac6d12": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-810m-lr4e-4/step47684/",
    # "hybrid-small-lc-v3-810m-lr2e-4-train-d685e7dd": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-810m-lr2e-4/step47684/",
    # "hybrid-small-lc-v3-810m-lr1e-4-train-30dea379": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-lc-v3-810m-lr1e-4/step47684/",
}

sft_checkpoints = {
    "275m-lr1e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr1e-4/step23206/",
    "275m-lr2e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206/",
    "275m-lr4e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr4e-4/step23206/",
    "275m-lr8e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr8e-4/step23206/",
    "810m-lr5e-5": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/sft-think-lr-sweep-810m-lr5e-5/step23206/",
    "810m-lr1e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/sft-think-lr-sweep-810m-lr1e-4/step23206/",
    "810m-lr2e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/sft-think-lr-sweep-810m-lr2e-4/step23206/",
    "810m-lr4e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/sft-think-lr-sweep-810m-lr4e-4/step23206/",
}

all_checkpoints = {
    "pretraining": pretraining_checkpoints,
    "midtraining": midtraining_checkpoints,
    # "long_context": long_context_checkpoints,
    # "long_context_debug": long_context_debug_checkpoints,
    # "sft": sft_checkpoints,
}


def get_output_path(input_path: str) -> str:
    p = Path(input_path.rstrip("/"))
    return str(p.parent / (p.name + "-hf"))


def install_sft_tokenizer(output_path: str) -> None:
    """Copy the think tokenizer (with embedded chat template) into a converted SFT ckpt."""
    src_dir = _resolve_sft_tokenizer_source()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Resolved SFT tokenizer source is not a directory: {src_dir}")

    print(f"    Using SFT tokenizer source: {src_dir}")

    dst = Path(output_path)
    for fname in SFT_TOKENIZER_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst / fname)
        else:
            print(f"    WARN: {src} not present, skipping")


def convert_all():
    failed = []

    for stage, checkpoints in all_checkpoints.items():
        for size, input_path in checkpoints.items():
            output_path = get_output_path(input_path)
            if Path(output_path).exists():
                print(f"\n=== Skipping {stage}/{size}: {output_path} already exists ===")
                continue
            print(f"\n=== Converting {stage}/{size}: {input_path} -> {output_path} ===")

            cmd = [
                sys.executable,
                CONVERT_SCRIPT,
                "--input_dir", input_path,
                "--output_dir", output_path,
                "--dtype", "bfloat16",
            ]
            if stage == "sft":
                # Skip the converter's tokenizer step — it would write the pretraining
                # tokenizer (no chat template). We copy the instruct tokenizer ourselves.
                cmd.append("--no_tokenizer")
            else:
                cmd += ["--tokenizer", TOKENIZER_ID]

            result = subprocess.run(
                cmd,
                env={**__import__("os").environ, "TRUST_REMOTE_CODE": "True"},
            )
            if result.returncode != 0:
                print(f"FAILED: {stage}/{size} (exit code {result.returncode})")
                failed.append((stage, size, input_path))
                continue

            if stage == "sft":
                try:
                    install_sft_tokenizer(output_path)
                except Exception as e:
                    print(f"FAILED to install SFT tokenizer for {stage}/{size}: {e}")
                    failed.append((stage, size, input_path))
                    continue

            print(f"OK: {stage}/{size}")

    if failed:
        print("\nThe following conversions FAILED:")
        for stage, size, path in failed:
            print(f"  {stage}/{size}: {path}")
        sys.exit(1)
    else:
        print("\nAll conversions completed successfully.")


if __name__ == "__main__":
    convert_all()