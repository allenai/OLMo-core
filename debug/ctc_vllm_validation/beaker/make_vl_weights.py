"""Rewrap the text-only export's weights into Qwen3.5 VL checkpoint layout for vLLM.

vLLM 0.25.1 resolves any Qwen3_5 architecture to ``Qwen3_5ForConditionalGeneration``
(no text-only entry in its registry), whose weight loader expects the HF VL naming:
``model.language_model.*`` / ``lm_head.weight`` / ``model.visual.*``. This script
renames the export's text-decoder tensors accordingly and appends the base
snapshot's ``model.visual.*`` tensors when present on disk (real weights if the
base snapshot was a full download; skipped -- falls back to add_dummy_visual.py --
if the base snapshot is metadata-only).

(Beaker copy of debug/ctc_vllm_validation/make_vl_weights.py -- identical logic.)
"""

import argparse
import json
import os

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

    idx_path = os.path.join(args.base_snapshot, "model.safetensors.index.json")
    n_vis = 0
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))["weight_map"]
        shards = sorted({f for k, f in idx.items() if k.startswith("model.visual.")})
        for shard in shards:
            shard_path = os.path.join(args.base_snapshot, shard)
            if not os.path.exists(shard_path):
                print(f"[vl-weights] base vision shard missing ({shard}); skipping vision graft "
                      f"(relying on add_dummy_visual.py / language_model_only)", flush=True)
                continue
            st = load_file(shard_path)
            for k, v in st.items():
                if k.startswith("model.visual."):
                    out[k] = v
                    n_vis += 1
    else:
        print(f"[vl-weights] no index.json at {idx_path}; base snapshot has no sharded vision "
              f"weights (single-file or metadata-only) -- skipping vision graft", flush=True)

    dst = os.path.join(args.out_dir, "model.safetensors")
    if os.path.islink(dst):
        os.unlink(dst)
    save_file(out, dst, metadata={"format": "pt"})
    print(f"[vl-weights] wrote {len(out)} tensors ({n_vis} visual) -> {dst}", flush=True)


if __name__ == "__main__":
    main()
