#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
# ]
# ///
"""Check if DRoPE (Drop RoPE) is enabled in the Hybrid 7B model.

DRoPE = linear attention layers drop RoPE entirely, only full attention
layers use it. Downloads config.json and modeling code from HF hub to
check whether RoPE is selectively applied.

Usage:
    uv run src/scripts/train/hybrid-small-suite/check_drope.py
"""

import json
from huggingface_hub import hf_hub_download, list_repo_files

MODEL_IDS = [
    "allenai/OLMo-3-1025-7B",
    "allenai/OLMo-Hybrid-7B",
]


def check_model(model_id: str):
    print(f"\n{'='*70}")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    # Download config.json
    try:
        path = hf_hub_download(model_id, "config.json")
    except Exception as e:
        print(f"  ERROR downloading config: {e}")
        return

    with open(path) as f:
        config = json.load(f)

    print(f"  model_type: {config.get('model_type')}")
    print(f"  num_hidden_layers: {config.get('num_hidden_layers')}")

    # Layer types
    lt = config.get("layer_types")
    if lt:
        from collections import Counter
        counts = Counter(lt)
        print(f"  layer_types ({len(lt)} layers): {dict(counts)}")
        print(f"    first 12: {lt[:12]}")
    else:
        print("  layer_types: None")

    # All rope-related config
    rope_keys = {k: v for k, v in config.items() if "rope" in k.lower()}
    print(f"\n  Rope-related config:")
    for k, v in rope_keys.items():
        print(f"    {k}: {json.dumps(v, indent=6) if isinstance(v, dict) else v}")

    if not rope_keys or all(v is None for v in rope_keys.values()):
        print("  >>> NO RoPE config found — model may rely on modeling code defaults")

    # Check for custom modeling code (trust_remote_code)
    print(f"\n  Repo files:")
    try:
        files = list_repo_files(model_id)
        py_files = [f for f in files if f.endswith(".py")]
        for f in sorted(py_files):
            print(f"    {f}")

        # Download and inspect modeling file for RoPE usage
        modeling_files = [f for f in py_files if "model" in f.lower()]
        for mf in modeling_files:
            print(f"\n  --- Inspecting {mf} for RoPE/DRoPE ---")
            mpath = hf_hub_download(model_id, mf)
            with open(mpath) as fh:
                code = fh.read()
            # Search for rope-related patterns
            for pattern in ["rotary", "rope", "RoPE", "DRoPE", "drope", "drop_rope",
                            "apply_rotary", "cos_cached", "sin_cached", "position_embeddings"]:
                lines = [
                    (i + 1, line.rstrip())
                    for i, line in enumerate(code.splitlines())
                    if pattern.lower() in line.lower()
                ]
                if lines:
                    print(f"    '{pattern}' found in {len(lines)} lines:")
                    for lineno, line in lines[:8]:
                        print(f"      L{lineno}: {line[:120]}")
                    if len(lines) > 8:
                        print(f"      ... and {len(lines) - 8} more")
    except Exception as e:
        print(f"    ERROR listing repo: {e}")


def main():
    for model_id in MODEL_IDS:
        check_model(model_id)


if __name__ == "__main__":
    main()
