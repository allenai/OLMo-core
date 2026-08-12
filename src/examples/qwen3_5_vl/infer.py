"""
Native multimodal **Qwen3.5-VL** image→text inference with OLMo-core.

Loads a HuggingFace ``Qwen3_5ForConditionalGeneration`` checkpoint (default
``Qwen/Qwen3.5-0.8B``) into the OLMo-core :class:`~olmo_core.nn.vision.qwen3_5_vl.Qwen3_5VL`
model (native vision tower + Qwen3.5 hybrid LM + M-RoPE) and greedily decodes a
caption. Image preprocessing reuses the HF ``AutoProcessor``; the model is native.

Usage::

    # Caption the bundled synthetic image (red circle + blue square):
    python src/examples/qwen3_5_vl/infer.py

    # Caption your own image with a custom prompt:
    python src/examples/qwen3_5_vl/infer.py --image path/to/img.jpg \\
        --prompt "What is in this image?"
"""

import argparse

import torch
from PIL import Image, ImageDraw

from olmo_core.nn.vision.qwen3_5_vl import load_qwen3_5_vl_from_hf


def _synthetic_image() -> Image.Image:
    img = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([120, 120, 330, 330], fill="red")
    draw.rectangle([40, 40, 120, 120], fill="blue")
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HF model id / path.")
    parser.add_argument("--image", default=None, help="Image path (default: synthetic).")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} into OLMo-core Qwen3_5VL ({args.dtype}) ...", flush=True)
    model, processor = load_qwen3_5_vl_from_hf(args.model, device=device, dtype=dtype)
    tok = processor.tokenizer

    image = Image.open(args.image).convert("RGB") if args.image else _synthetic_image()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    eos = tuple(
        x for x in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if x is not None
    )
    out_ids = model.generate(
        inputs["input_ids"],
        inputs["pixel_values"].to(dtype),
        inputs["image_grid_thw"],
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos,
    )
    print("\n=== Generation ===")
    print(tok.decode(out_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
