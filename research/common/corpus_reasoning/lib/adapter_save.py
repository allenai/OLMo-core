"""Load-time adapter-key normalization for Qwen3.5 LoRA adapters.

Qwen3.5 is registered in transformers as a multimodal
`Qwen3_5ForConditionalGeneration` (language model nested under `language_model`)
and also as `Qwen3_5ForCausalLM` (language-only, `language_model` unwrapped).
Different trainers save LoRA adapters in different key formats:

* Axolotl loads the VL wrapper, saves keys as `base_model.model.model.language_model.layers.X...`.
* The chunked trainer here loads `AutoModelForCausalLM` (language-only), saves
  keys as `base_model.model.model.layers.X...`.

Each evaluation backend expects one of the two formats and silently skips the
other:

* vLLM internally uses the VL architecture for LoRA key mapping even when the
  forward path is language-only, so it expects `language_model.` in keys.
* Our HF backend (`evaluate.py`) loads `AutoModelForCausalLM`, so PEFT expects
  the language-only (no-prefix) form.

To decouple trainer-format from evaluator-format, we normalize the adapter at
load time. `prepare_adapter_for_backend` returns either the original
`lora_path` or a temporary directory with a renamed copy of the adapter. The
tempdir is registered with `atexit` so callers don't have to clean up.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file


_VL_INFIX = ".model.model.language_model.layers."
_CAUSAL_INFIX = ".model.model.layers."


def _is_qwen35(base_model: str) -> bool:
    b = (base_model or "").lower()
    return "qwen3.5" in b or "qwen3_5" in b


def _adapter_file(lora_path: str | os.PathLike) -> Path:
    return Path(lora_path) / "adapter_model.safetensors"


def _detect_format(state_keys) -> str:
    """Return 'vl', 'causal', or 'unknown' based on first layer-bearing key."""
    for k in state_keys:
        if _VL_INFIX in k:
            return "vl"
        if _CAUSAL_INFIX in k:
            return "causal"
    return "unknown"


def _make_renamed_copy(lora_path: str, mapping: tuple[str, str]) -> str:
    """Copy `lora_path` to a tempdir, rewriting adapter keys per `mapping`.

    `mapping` is (old_substr, new_substr); every occurrence in keys is replaced.
    Returns the tempdir path. The tempdir is cleaned up at interpreter exit.
    """
    tmp = tempfile.mkdtemp(prefix="adapter_norm_")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)

    # Copy every file except the safetensors (we rewrite that one).
    for name in os.listdir(lora_path):
        src = os.path.join(lora_path, name)
        dst = os.path.join(tmp, name)
        if os.path.isdir(src):
            continue
        if name == "adapter_model.safetensors":
            continue
        shutil.copy2(src, dst)

    old, new = mapping
    state = load_file(str(_adapter_file(lora_path)))
    renamed = {k.replace(old, new): v for k, v in state.items()}
    save_file(renamed, os.path.join(tmp, "adapter_model.safetensors"))
    return tmp


def prepare_adapter_for_backend(
    lora_path: str | os.PathLike | None,
    base_model: str,
    backend: str,
) -> str | None:
    """Return a lora_path whose adapter keys match what `backend` expects.

    * backend='vllm': vLLM's LoRA mapping expects Qwen3.5 keys to contain
      `language_model.`. If the on-disk adapter uses the causal (no-prefix)
      form, we return a tempdir copy with the prefix inserted.
    * backend='hf': the HF backend loads `AutoModelForCausalLM` and PEFT
      expects the causal (no-prefix) form. If the adapter uses the VL form,
      we return a tempdir copy with the prefix stripped.

    No-ops for non-Qwen3.5 base models, or when the adapter is already in the
    format `backend` expects, or when `lora_path` is empty/None.
    """
    if not lora_path:
        return lora_path
    if not _is_qwen35(base_model):
        return str(lora_path)
    p = _adapter_file(lora_path)
    if not p.exists():
        return str(lora_path)

    state_keys = load_file(str(p)).keys()
    fmt = _detect_format(state_keys)

    if backend == "vllm" and fmt == "causal":
        new_path = _make_renamed_copy(str(lora_path), (_CAUSAL_INFIX, _VL_INFIX))
        print(f"[adapter_norm] vLLM: added language_model. prefix for {lora_path} "
              f"(tempdir: {new_path})")
        return new_path

    if backend == "hf" and fmt == "vl":
        new_path = _make_renamed_copy(str(lora_path), (_VL_INFIX, _CAUSAL_INFIX))
        print(f"[adapter_norm] HF: stripped language_model. prefix for {lora_path} "
              f"(tempdir: {new_path})")
        return new_path

    return str(lora_path)
