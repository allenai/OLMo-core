"""Graft randomly-initialized visual.* weights into a text-only Qwen3.5 serving copy.

vLLM 0.25.1 has no text-only Qwen3.5 arch: it always builds the full multimodal
`Qwen3_5ForConditionalGeneration`, whose weight loader errors on any uninitialized
`visual.*` parameter (`track_weights_loading`). Our olmo->HF export is text-only,
and the real Qwen3.5 vision weights are unavailable (internal model; base snapshot
is metadata-only). Text prompts never route through the vision tower, so the values
are irrelevant — only the keys/shapes must exist.

Build the vision tower from the serving config's `vision_config` using THIS env's
transformers (matches vLLM's expected naming), take its randomly-initialized
state_dict, prefix `visual.`, and append to the serving model.safetensors.

Run with the vLLM venv python (native qwen3_5 transformers):
  /data/prasann/ctc_vllm_venv/bin/python add_dummy_visual.py --serve <dir>
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--serve", required=True, help="serving copy dir (config.json + model.safetensors)"
    )
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.serve, "config.json")))
    vcfg = cfg.get("vision_config")
    assert vcfg is not None, "serving config has no vision_config"

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    vision_config = Qwen3_5VisionConfig(**vcfg)
    torch.manual_seed(0)
    with torch.device("cpu"):
        vision = Qwen3_5VisionModel(vision_config).to(torch.bfloat16)
    vstate = {f"visual.{k}": v.contiguous() for k, v in vision.state_dict().items()}
    print(f"[dummy-visual] built {len(vstate)} visual.* tensors from vision_config", flush=True)

    model_path = os.path.join(args.serve, "model.safetensors")
    state = load_file(model_path)
    n_text = len(state)
    added = 0
    for k, v in vstate.items():
        if k not in state:
            state[k] = v
            added += 1
    print(
        f"[dummy-visual] text tensors={n_text}, visual added={added}, total={len(state)}",
        flush=True,
    )

    if os.path.islink(model_path):
        os.unlink(model_path)
    save_file(state, model_path, metadata={"format": "pt"})
    print(f"[dummy-visual] wrote {model_path}", flush=True)


if __name__ == "__main__":
    main()
