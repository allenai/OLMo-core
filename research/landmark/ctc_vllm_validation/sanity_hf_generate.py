"""Sanity-check the Qwen3.5 HF export: load with transformers, greedy-generate
10 tokens, and confirm layer_types survived in config.json."""

import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(os.path.join(args.hf_model, "config.json")) as f:
        cfg = json.load(f)
    lt = cfg.get("layer_types")
    print("model_type:", cfg.get("model_type"))
    print("architectures:", cfg.get("architectures"))
    print("tie_word_embeddings:", cfg.get("tie_word_embeddings"))
    if lt is None:
        print("layer_types: MISSING from config.json")
    else:
        from collections import Counter

        print(f"layer_types: {len(lt)} layers, counts={dict(Counter(lt))}")
        print("layer_types head:", lt[:8])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    print("loaded:", type(model).__name__)

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=10, do_sample=False)
    text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    print(f"prompt: {args.prompt!r}")
    print(f"generation (10 tok greedy): {text!r}")


if __name__ == "__main__":
    main()
