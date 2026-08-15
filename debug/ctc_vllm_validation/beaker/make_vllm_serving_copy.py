"""Make a vLLM-servable copy of a text-only Qwen3.5 HF export.

vLLM 0.25.1 registers ``Qwen3_5ForCausalLM`` with a multimodal processing
context that requires the full VL wrapper config (``model_type='qwen3_5'``
with nested ``text_config``/``vision_config``), while its weight loader
expects TEXT-layout names (``model.layers.*`` / ``lm_head.weight``) — exactly
what export_olmo_to_hf.py emits. So: symlink the export's weights/tokenizer
into a new dir and write a wrapper config.json = base VL config with
``text_config`` replaced by the export's text config (preserving
``tie_word_embeddings=False`` so the trained head is used).

(Beaker copy of debug/ctc_vllm_validation/make_vllm_serving_copy.py -- identical logic,
embedded here because that dir is untracked and gantry only ships the pushed commit.)
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-export", required=True)
    ap.add_argument("--base-snapshot", required=True, help="Qwen3.5 base snapshot dir (VL config)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text_cfg = json.load(open(os.path.join(args.hf_export, "config.json")))
    base_cfg = json.load(open(os.path.join(args.base_snapshot, "config.json")))

    wrapper = dict(base_cfg)
    wrapper["architectures"] = ["Qwen3_5ForCausalLM"]
    wrapper["text_config"] = text_cfg
    wrapper["tie_word_embeddings"] = bool(text_cfg.get("tie_word_embeddings", False))

    os.makedirs(args.out, exist_ok=True)
    for f in glob.glob(os.path.join(args.hf_export, "*")):
        name = os.path.basename(f)
        if name == "config.json":
            continue
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
        src = os.path.join(args.base_snapshot, name)
        dst = os.path.join(args.out, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(wrapper, f, indent=2)
    print(f"[serving-copy] {args.out}: text tie_word_embeddings="
          f"{text_cfg.get('tie_word_embeddings')} layer_types="
          f"{len(text_cfg.get('layer_types', []))} layers", flush=True)


if __name__ == "__main__":
    main()
