"""Shared Molmo2 inference helpers for the example eval scripts.

Extracted from ``pixmo_cap_eval.py`` so multiple evals (dense caption,
image-QA benchmarks) share one implementation of:

* multi-crop image preprocessing (native OLMo-core, no mm_olmo dependency)
* image-token sequence construction and prompt building
* HF / native-``.distcp`` checkpoint loading into ``MultimodalLM``
* greedy decoding (no KV cache)
* simple atomic JSON prediction caches

Prompt layouts
--------------
Molmo2's native format places the expanded image-token block *before* the
``<|im_start|>user`` turn::

    <low_res_im_start>…<im_end><im_start>…<im_end><|im_start|>user\\n{TEXT}<|im_end|>\\n<|im_start|>assistant\\n

:func:`build_input_ids` produces this layout (it matches the released
mm_olmo prediction dumps and the official HF chat template).
:func:`build_input_ids_placeholder_style` keeps the legacy behavior of
``pixmo_cap_eval.py`` (placeholder inside the user turn) for backward
compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Token constants and image-token-sequence construction live in a shared module
# (``olmo_core.nn.vision.molmo2_tokens``) so the training data pipeline can reuse
# them without importing from ``examples/``. Re-exported here for backward compat.
from olmo_core.nn.vision.molmo2_tokens import (  # noqa: E402,F401
    DEFAULT_MAX_CROPS,
    DEFAULT_MODEL_ID,
    EOS_TOKEN_ID,
    IM_COL_ID,
    IM_END_ID,
    IM_END_TURN_ID,
    IM_PATCH_ID,
    IM_START_ID,
    IMAGE_PLACEHOLDER_ID,
    IMAGE_SIZE,
    IMAGE_TOKEN_IDS,
    LOW_RES_IM_START_ID,
    N_PATCHES,
    N_PATCHES_SQ,
    OVERLAP_MARGINS,
    PATCH_DIM,
    PATCH_SIZE,
    POOL_H,
    POOL_W,
    build_image_token_ids,
)

# ---------------------------------------------------------------------------
# Image preprocessing (multi-crop via OLMo-core native preprocessor)
# ---------------------------------------------------------------------------


def preprocess_image_multicrop(
    pil_img: Image.Image,
    dtype: torch.dtype,
    device: torch.device,
    max_crops: int = DEFAULT_MAX_CROPS,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Multi-crop Molmo2 preprocessing using the native OLMo-core preprocessor.

    :param max_crops: maximum number of high-res crops (the original mm_olmo
        benchmark evals used 24; the released HF processor defaults to 8).

    :returns: ``images (1, n_crops, 729, 588)`` channel-first float tensor,
        ``pooling_idx (1, n_pool_tokens, 4)`` int64 pooling indices, and
        ``image_grid = [resized_h, resized_w, h, w]`` pooled grid dims.
    """
    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    return preprocess_image_molmo2(
        pil_img,
        dtype,
        device,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        max_crops=max_crops,
        overlap_margins=OVERLAP_MARGINS,
        pool_h=POOL_H,
        pool_w=POOL_W,
    )


def build_input_ids(
    tokenizer,
    prompt: str,
    image_grid: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize a single-turn user prompt in the native Molmo2 layout.

    Sequence structure (matching mm_olmo's ``molmo2`` message format and the
    official HF processor: BOS first, image-token block *before*
    ``<|im_start|>user``, generation header at the end)::

        BOS <image block> <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n

    With ``image_grid=None`` a text-only prompt is built (used e.g. for MMMU
    questions whose answer options are images).
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    if image_grid is not None:
        resized_h, resized_w, h, w = (int(image_grid[i]) for i in range(4))
        ids = build_image_token_ids(resized_h, resized_w, h, w) + ids
    # Leading BOS, following Molmo2Processor.insert_bos: bos or eos token —
    # <|im_end|> (151645) for the released Molmo2 tokenizers, matching the
    # mm_olmo native eval inputs.
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    ids = [bos_id] + ids
    return torch.tensor([ids], dtype=torch.long, device=device)


def build_input_ids_placeholder_style(
    tokenizer, prompt: str, image_grid: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Legacy ``pixmo_cap_eval`` prompt building (image block inside the user turn)."""
    resized_h, resized_w, h, w = (int(image_grid[i]) for i in range(4))
    image_tokens = build_image_token_ids(resized_h, resized_w, h, w, low_res_col_tokens=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|image|>{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    try:
        img_pos = ids.index(IMAGE_PLACEHOLDER_ID)
    except ValueError:
        raise RuntimeError(
            f"<|image|> placeholder token (ID {IMAGE_PLACEHOLDER_ID}) not found in "
            f"tokenized prompt. First 20 IDs: {ids[:20]}"
        )
    ids = ids[:img_pos] + image_tokens + ids[img_pos + 1 :]
    return torch.tensor([ids], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_checkpoint(model_str: str) -> tuple[bool, Path | None]:
    """Detect whether ``model_str`` is an OLMo-core native ``.distcp`` checkpoint.

    :returns: ``(is_native, step_dir)`` — ``is_native=False`` means treat as an
        HF model ID / HF local path; otherwise load from
        ``step_dir/model_and_optim/``.
    """
    p = Path(model_str)
    if not p.exists() or not p.is_dir():
        return False, None
    if (p / "model_and_optim").is_dir():
        return True, p
    step_dirs = sorted(d for d in p.iterdir() if d.is_dir() and (d / "model_and_optim").is_dir())
    if step_dirs:
        step_dir = step_dirs[-1]
        logger.info("Run root detected; using latest step: %s", step_dir.name)
        return True, step_dir
    return False, None


def _load_model_hf(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load HF Molmo2 weights → convert → MultimodalLM. Returns (model, tokenizer)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from olmo_core.nn.vision import MultimodalLM
    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
    )

    logger.info("Loading HF %s …", model_id)
    ensure_default_rope_registered()
    hf = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True)
    reinit_rope_buffers(hf)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Converting weights …")
    cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg)
    del hf

    logger.info("Loading into MultimodalLM …")
    model = MultimodalLM(cfg, init_device="meta")
    model.to_empty(device=torch.device("cpu"))
    missing, _ = model.load_state_dict(converted, strict=False)
    del converted

    param_keys = {k for k, _ in model.named_parameters()}
    if set(missing) & param_keys:
        logger.warning("Missing params: %s", sorted(set(missing) & param_keys)[:5])

    model = model.to(device=device, dtype=dtype).eval()
    logger.info("Model ready on %s in %s.", device, dtype)
    return model, tok


_ATTN_PAT = re.compile(r"(model\.transformer\.blocks\.\d+\.)(att_proj|k_norm|q_norm|attn_out)(.*)")
_MLP_PAT = re.compile(r"(model\.transformer\.blocks\.\d+\.)(ff_proj|ff_out)(.*)")


def mm_olmo_sd_to_hf(raw_sd: dict) -> dict:
    """Remap mm_olmo checkpoint keys to the HF Molmo2 key naming convention.

    The only differences are inside transformer blocks:
    ``att_proj / k_norm / q_norm / attn_out → self_attn.<name>`` and
    ``ff_proj / ff_out → mlp.<name>``.  Vision-backbone keys and embeddings
    are identical between the two formats.
    """
    out = {}
    for k, v in raw_sd.items():
        m = _ATTN_PAT.fullmatch(k)
        if m:
            k = f"{m.group(1)}self_attn.{m.group(2)}{m.group(3)}"
        else:
            m = _MLP_PAT.fullmatch(k)
            if m:
                k = f"{m.group(1)}mlp.{m.group(2)}{m.group(3)}"
        out[k] = v
    return out


def _load_model_olmocore(
    step_dir: Path,
    hf_config_id: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load a native mm_olmo ``.distcp`` training checkpoint into MultimodalLM."""
    import pickle

    from transformers import AutoConfig, AutoTokenizer

    from olmo_core.distributed.checkpoint import load_state_dict as olmocore_load_sd
    from olmo_core.nn.vision import MultimodalLM
    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
        molmo2_hf_state_dict_to_multimodal_lm,
    )

    logger.info("Native mm_olmo .distcp checkpoint at %s", step_dir)
    model_and_optim_dir = step_dir / "model_and_optim"

    logger.info("Reading checkpoint metadata …")
    with open(model_and_optim_dir / ".metadata", "rb") as f:
        meta = pickle.load(f)

    # Pre-allocate tensors for every model key in the checkpoint.
    raw_sd: dict = {}
    for key, smeta in meta.state_dict_metadata.items():
        if key.startswith("model."):
            raw_sd[key] = torch.empty(smeta.size, dtype=smeta.properties.dtype)

    logger.info("Loading %d tensors from .distcp shards …", len(raw_sd))
    olmocore_load_sd(str(model_and_optim_dir), raw_sd)

    logger.info("Remapping mm_olmo keys → HF keys …")
    hf_sd = mm_olmo_sd_to_hf(raw_sd)
    del raw_sd
    # mm_olmo uses tied embeddings; synthesize the standalone lm_head.weight the
    # converter expects (= base token embedding; the converter pads image tokens).
    if "lm_head.weight" not in hf_sd:
        hf_sd["lm_head.weight"] = hf_sd["model.transformer.wte.embedding"].clone()

    logger.info("Fetching architecture config from %s (no weights download) …", hf_config_id)
    ensure_default_rope_registered()
    hf_config = AutoConfig.from_pretrained(hf_config_id, trust_remote_code=True)
    cfg = molmo2_config_from_hf_config(hf_config)

    logger.info("Converting to OLMo-core format and loading into MultimodalLM …")
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf_sd, cfg)
    del hf_sd

    model = MultimodalLM(cfg, init_device="meta")
    model.to_empty(device=torch.device("cpu"))
    missing, _ = model.load_state_dict(converted, strict=False)
    del converted

    param_keys = {k for k, _ in model.named_parameters()}
    if set(missing) & param_keys:
        logger.warning("Missing params: %s", sorted(set(missing) & param_keys)[:5])

    tok = AutoTokenizer.from_pretrained(hf_config_id, trust_remote_code=True)
    model = model.to(device=device, dtype=dtype).eval()
    logger.info("Model ready on %s in %s.", device, dtype)
    return model, tok


def load_model(
    model_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    hf_config_id: str = DEFAULT_MODEL_ID,
):
    """Load Molmo2 into MultimodalLM from either an HF model ID or a
    local OLMo-core ``.distcp`` checkpoint directory. Returns (model, tokenizer).
    """
    is_native, step_dir = resolve_checkpoint(model_or_path)
    if is_native:
        assert step_dir is not None
        return _load_model_olmocore(step_dir, hf_config_id, device, dtype)
    return _load_model_hf(model_or_path, device, dtype)


# ---------------------------------------------------------------------------
# HF released pipeline (exact original-eval behavior)
# ---------------------------------------------------------------------------


def load_hf_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load the released HF Molmo2 model + processor.

    Unlike the OLMo-core ``MultimodalLM`` path, the HF model
    implements Molmo2's bidirectional attention over image tokens
    (``token_type_ids`` masking) and KV-cached generation — verified to
    reproduce the original mm_olmo eval predictions exactly.

    :returns: ``(model, processor)``.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        reinit_rope_buffers,
    )

    logger.info("Loading HF pipeline %s …", model_id)
    ensure_default_rope_registered()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype
    )
    reinit_rope_buffers(model)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    logger.info("HF pipeline ready on %s in %s.", device, dtype)
    return model, processor


@torch.inference_mode()
def hf_generate(
    model,
    processor,
    question: str,
    pil_img,
    device: torch.device,
    max_new_tokens: int,
    max_crops: int = DEFAULT_MAX_CROPS,
) -> str:
    """Greedy generation with the released HF pipeline (KV-cached).

    ``question`` is the fully formatted prompt text (style prefix / MC block
    already baked in); ``pil_img=None`` builds a text-only prompt.
    """
    content: list[dict] = []
    if pil_img is not None:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        content.append({"type": "image", "image": pil_img})
    content.append({"type": "text", "text": question})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    if pil_img is not None:
        inputs = processor(images=[pil_img], text=text, max_crops=max_crops, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    n_input = inputs["input_ids"].shape[1]
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(output[0][n_input:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------


@torch.inference_mode()
def greedy_decode(
    model,
    input_ids: torch.Tensor,
    images: torch.Tensor | None,
    pooled_patches_idx: torch.Tensor | None,
    tokenizer,
    max_new_tokens: int = 448,
    stop_token_ids: frozenset[int] = frozenset({EOS_TOKEN_ID, IM_END_TURN_ID}),
) -> str:
    """Run greedy decoding and return the decoded response string.

    Without KV-cache, images must be passed on every step so the vision
    encoder re-encodes the image patches at the correct positions each time.
    This is O(max_new_tokens × ViT_forward); use a small ``max_new_tokens``
    for fast smoke tests.

    Image tokens are marked via ``token_type_ids`` so they attend to each other
    bidirectionally (matching HF Molmo2); text stays causal.
    """
    image_id_tensor = torch.tensor(sorted(IMAGE_TOKEN_IDS), device=input_ids.device)
    gen_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        token_type_ids = torch.isin(gen_ids, image_id_tensor).long()
        logits = model(
            input_ids=gen_ids,
            images=images,
            pooled_patches_idx=pooled_patches_idx,
            token_type_ids=token_type_ids,
            logits_to_keep=1,
        )
        next_id = int(logits[0, -1].argmax().item())
        if next_id in stop_token_ids:
            break
        gen_ids = torch.cat(
            [gen_ids, torch.tensor([[next_id]], dtype=torch.long, device=gen_ids.device)],
            dim=1,
        )
    new_ids = gen_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_ids.tolist(), skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prediction caches (atomic JSON, run-owned files only)
# ---------------------------------------------------------------------------


def load_prediction_cache(path: str) -> dict[str, str]:
    """Load a key→prediction mapping from a predictions cache file."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {d["image_url"]: d["prediction"] for d in data if "image_url" in d}
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_prediction_cache(path: str, key_to_prediction: dict[str, str]) -> None:
    """Atomically save a key→prediction mapping as JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(key_to_prediction, f, indent=2)
    os.replace(tmp, path)
