"""Rewrap the text-only export's weights into Qwen3.5 VL checkpoint layout for vLLM.

vLLM 0.25.1 resolves any Qwen3_5 architecture to ``Qwen3_5ForConditionalGeneration``
(no text-only entry in its registry), whose weight loader expects the HF VL naming:
``model.language_model.*`` / ``lm_head.weight`` / ``model.visual.*``. This script
renames the export's text-decoder tensors accordingly and appends the base
snapshot's ``model.visual.*`` tensors (untouched vision tower, unused by text-only
prompts but required to construct the module).
"""

import argparse
import glob
import json
import os

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-export", required=True)
    ap.add_argument("--base-snapshot", required=True)
    ap.add_argument("--out-dir", required=True, help="serving dir (config.json already present)")
    args = ap.parse_args()

    state = load_file(os.path.join(args.hf_export, "model.safetensors"))
    out = {}
    for k, v in state.items():
        if k.startswith("model."):
            out["model.language_model." + k[len("model."):]] = v
        else:
            out[k] = v  # lm_head.weight

    # Vision tower from the base (only appended if the base snapshot actually has the
    # weight shards on disk). Qwen3.5 is internal, so the local base snapshot is often
    # metadata-only -- in that case we skip vision entirely and rely on vLLM's
    # `language_model_only` to prune the vision tower from the module graph. The text
    # decoder is all we need for text prompts.
    idx_path = os.path.join(args.base_snapshot, "model.safetensors.index.json")
    idx = json.load(open(idx_path))["weight_map"]
    shards = sorted({f for k, f in idx.items() if k.startswith("model.visual.")})
    n_vis = 0
    for shard in shards:
        shard_path = os.path.join(args.base_snapshot, shard)
        if not os.path.exists(shard_path):
            print(f"[vl-weights] base vision shard missing ({shard}); skipping vision graft "
                  f"(relying on language_model_only to prune vision tower)", flush=True)
            continue
        st = load_file(shard_path)
        for k, v in st.items():
            if k.startswith("model.visual."):
                out[k] = v
                n_vis += 1

    dst = os.path.join(args.out_dir, "model.safetensors")
    if os.path.islink(dst):
        os.unlink(dst)
    save_file(out, dst, metadata={"format": "pt"})
    print(f"[vl-weights] wrote {len(out)} tensors ({n_vis} visual) -> {dst}", flush=True)


if __name__ == "__main__":
    main()
