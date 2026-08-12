"""
End-to-end parity tests for native multimodal **Qwen3.5-VL** in OLMo-core.

Loads a real HF ``Qwen3_5ForConditionalGeneration`` checkpoint into the OLMo-core
:class:`~olmo_core.nn.vision.qwen3_5_vl.Qwen3_5VL` and checks, on a synthetic image:

1. **Vision tower parity** — native merger output vs HF ``model.visual`` (exact).
2. **Full-forward logit parity** — last-token logits with the image present
   (per-token argmax must match; absolute logit diff is a loose sanity bound,
   since GatedDeltaNet FLA kernels accumulate fp32 noise that grows with depth).
3. **Generation parity** — greedy image→text matches HF exactly.

Parametrized over the dense Qwen3.5 / Qwen3.6 VL checkpoints (0.8B / 2B / 4B /
9B / 27B; both releases share the ``Qwen3_5ForConditionalGeneration``
architecture). Each case requires a GPU, the FLA kernels (GatedDeltaNet),
transformers with Qwen3.5
support, and the checkpoint present in the local HF cache; otherwise it skips
(no network downloads). Only one model is held on the GPU at a time — HF
references are captured first, then the HF weights move to CPU and are reused to
build the OLMo-core model — so peak GPU memory is ~1x the model. dtype is chosen
to fit (fp32 where possible, bf16 for the largest variants); a variant that
won't fit even in bf16 is skipped rather than OOMing.
"""

import os

import pytest
import torch

from olmo_core.testing.utils import requires_fla, requires_gpu

transformers = pytest.importorskip("transformers")

# Dense Qwen3.5 / Qwen3.6 vision-language checkpoints. Both releases share the
# same ``Qwen3_5ForConditionalGeneration`` architecture (``model_type: qwen3_5``),
# so they load through the identical path; the ``-A*`` MoE checkpoints use a
# different architecture and are intentionally excluded. Values are approximate
# parameter counts, used only to pick a dtype that fits the GPU.
MODEL_PARAMS = {
    "Qwen/Qwen3.5-0.8B": 0.8e9,
    "Qwen/Qwen3.5-2B": 2.0e9,
    "Qwen/Qwen3.5-4B": 4.0e9,
    "Qwen/Qwen3.5-9B": 9.0e9,
    "Qwen/Qwen3.5-27B": 27.0e9,
    "Qwen/Qwen3.6-27B": 27.0e9,
}
MODEL_IDS = list(MODEL_PARAMS)


def _has_qwen3_5_vl() -> bool:
    return hasattr(transformers, "Qwen3_5ForConditionalGeneration")


def _hf_cache_has(model_id: str) -> bool:
    suffix = "models--" + model_id.replace("/", "--")
    roots = [os.path.expanduser("~/.cache/huggingface/hub")]
    if (hf_home := os.environ.get("HF_HOME")) is not None:
        roots.append(os.path.join(hf_home, "hub"))
    return any(os.path.isdir(os.path.join(r, suffix)) for r in roots if r)


def _synthetic_image():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([120, 120, 330, 330], fill="red")
    d.rectangle([40, 40, 120, 120], fill="blue")
    return img


def _pick_dtype_or_skip(model_id: str) -> torch.dtype:
    """Largest dtype whose single model copy fits free GPU memory (fp32 preferred).

    The test holds only one model on the GPU at a time, so we budget ~1.4x the
    raw weight size for activations/generation. ``pytest.skip`` if even bf16
    won't fit (e.g. 27B on a small GPU).
    """
    free, _ = torch.cuda.mem_get_info()
    params = MODEL_PARAMS[model_id]
    if params * 4 * 1.4 < free:
        return torch.float32
    if params * 2 * 1.4 < free:
        return torch.bfloat16
    pytest.skip(f"{model_id} too large for available GPU memory ({free / 1e9:.0f} GB free)")


def _build_inputs(processor, device):
    img = _synthetic_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[img], return_tensors="pt").to(device)


@requires_gpu
@requires_fla
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_qwen3_5_vl_image_to_text_parity(model_id: str):
    if not _has_qwen3_5_vl():
        pytest.skip("transformers lacks Qwen3.5-VL support")
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in HF cache")

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from olmo_core.nn.vision.qwen3_5_vl import load_qwen3_5_vl_from_hf

    device = torch.device("cuda")
    dtype = _pick_dtype_or_skip(model_id)
    is_fp32 = dtype == torch.float32
    # Vision parity is checked *relatively*: the merger activations are large
    # (O(100s)), so bf16 rounding over a deep tower yields a big absolute diff
    # (~few units) that is still only ~1-2% relative. fp32 stays near-exact.
    vis_rtol = 1e-3 if is_fp32 else 5e-2

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    inputs = _build_inputs(processor, device)
    pv = inputs["pixel_values"].to(dtype)
    gthw = inputs["image_grid_thw"]
    tok = processor.tokenizer
    eos = tuple(
        x for x in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if x is not None
    )

    # --- Capture HF references, then free the HF model so only one model is on
    #     the GPU at a time (lets the larger variants, e.g. 27B, fit). ---
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    hf = hf.to(device).eval()
    try:
        with torch.inference_mode():
            ref_vis = hf.model.visual(pv, grid_thw=gthw).pooler_output.float().cpu()
            ref_logits = hf(**inputs).logits[0, -1].float().cpu()
            ref_gen = hf.generate(**inputs, max_new_tokens=40, do_sample=False)
        ref_text = tok.decode(
            ref_gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
    finally:
        # Keep HF weights on CPU (host RAM) so the loader can convert from them
        # without a second download, but free the GPU for the OLMo-core model.
        hf = hf.to("cpu")
        torch.cuda.empty_cache()

    model, _ = load_qwen3_5_vl_from_hf(model_id, device="cuda", dtype=dtype, hf_model=hf)
    del hf
    try:
        # (1) Vision tower parity (relative).
        with torch.inference_mode():
            our_vis = model.vision(pv, gthw).float().cpu()
        vis_rel = (ref_vis - our_vis).abs().max() / ref_vis.abs().max().clamp_min(1e-6)
        assert vis_rel.item() < vis_rtol, f"{model_id}: vision mismatch (rel={vis_rel.item():.3g})"

        # (2) Full-forward logit parity. Per-token argmax must match exactly; the
        #     absolute logit bound is an fp32-only sanity check (bf16 logits drift
        #     freely in magnitude, and exact generation below is the real signal).
        with torch.inference_mode():
            our_last = model(inputs["input_ids"], pv, gthw, logits_to_keep=1)[0, -1].float().cpu()
        n = min(ref_logits.shape[0], our_last.shape[0])
        assert int(ref_logits.argmax()) == int(our_last.argmax()), f"{model_id}: argmax mismatch"
        if is_fp32:
            drift = (ref_logits[:n] - our_last[:n]).abs().max().item()
            assert drift < 1.0, f"{model_id}: logit drift {drift:.3g}"

        # (3) Generation parity (exact string match — holds in bf16 too).
        our_ids = model.generate(
            inputs["input_ids"], pv, gthw, max_new_tokens=40, eos_token_ids=eos
        )
        our_text = tok.decode(our_ids, skip_special_tokens=True).strip()
        assert our_text == ref_text, f"{model_id}: ours={our_text!r} hf={ref_text!r}"
    finally:
        del model
        torch.cuda.empty_cache()
