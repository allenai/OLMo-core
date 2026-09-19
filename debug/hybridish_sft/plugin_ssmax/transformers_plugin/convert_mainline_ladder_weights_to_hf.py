"""
Convert OLMo Hybrid Small model checkpoints (peri-norm, NoPE, GDN + gated attention)
to HuggingFace format.

This script handles OLMo Hybrid Small models that use:
- Peri-norm (pre + post norm on both attention and FFN sub-blocks)
- NoPE (no positional embeddings) on full attention layers
- Elementwise attention gate on full attention layers
- Per-head QK normalization
- GatedDeltaNet linear attention layers
- Embedding scaling (sqrt(d_model)) and embedding norm

IMPORTANT: This script explicitly saves the tokenizer from the checkpoint config's
``identifier`` field (e.g. ``allenai/dolma2-tokenizer``). Do NOT override with a
generic tokenizer — the pretokenizer must match what was used during training to
ensure correct tokenization of numbers and other edge cases.

Sample usage:

```bash
TRUST_REMOTE_CODE=True python src/transformers/models/mainline_ladder/convert_mainline_ladder_weights_to_hf.py \
    --input_dir /path/to/checkpoint \
    --output_dir /output/path
```

Thereafter, models can be loaded via:

```python
from transformers import MainlineLadderForCausalLM, AutoTokenizer

model = MainlineLadderForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import pickle
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.futures import Future

from transformers import AutoTokenizer

# This converter relies on the plugin package rather than a transformers fork.
# register() wires MainlineLadder into the transformers Auto* classes.
from transformers_plugin import register, MainlineLadderConfig


register()


# Mapping from string dtype names to torch dtypes
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def strtobool(val):
    """Convert a string representation of truth to True or False."""
    if isinstance(val, bool):
        return val
    val = str(val).lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value {val!r}")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def normalize_path(path: Path | str) -> str:
    return str(path).rstrip("/").replace("file://", "")


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_bytes_range(path: Path | str, bytes_start: int, num_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(bytes_start)
        return f.read(num_bytes)


def _narrow_tensor_by_index(
    tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]
) -> torch.Tensor:
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


@dataclass
class _StorageInfo:
    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


class RemoteFileSystemReader(dist_cp.StorageReader):
    """
    A StorageReader that reads distributed checkpoints from local or remote paths.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        thread_count: int | None = None,
        pre_download: bool = False,
        work_dir: Path | str | None = None,
    ):
        super().__init__()
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        self.path = normalize_path(path)
        self.thread_count = thread_count or 1
        self.pre_download = pre_download
        self.work_dir = normalize_path(work_dir) if work_dir is not None else None
        self.storage_data: dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = generate_uuid()
        self._metadata: Metadata | None = None

    def _get_bytes(self, relative_path: str, offset: int, length: int) -> bytes:
        full_path = f"{self.path}/{relative_path}"
        return get_bytes_range(full_path, offset, length)

    def _get_content_for_read(self, read_item: ReadItem) -> tuple[ReadItem, bytes]:
        sinfo = self.storage_data[read_item.storage_index]
        content = self._get_bytes(sinfo.relative_path, sinfo.offset, sinfo.length)
        return (read_item, content)

    def reset(self, checkpoint_id: Path | str | None = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = normalize_path(checkpoint_id)
        self.load_id = generate_uuid()

    def read_data(
        self, plan: dist_cp.LoadPlan, planner: dist_cp.LoadPlanner
    ) -> Future[None]:
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            read_item_content_futures = []
            for read_item in plan.items:
                read_item_content_futures.append(
                    executor.submit(self._get_content_for_read, read_item)
                )
            read_item_content_results = []
            for f in as_completed(read_item_content_futures):
                try:
                    read_item_content_results.append(f.result())
                except BaseException:
                    raise RuntimeError(f"Original error:\n{traceback.format_exc()}")

        for read_item, content in read_item_content_results:
            bytes_io = io.BytesIO(content)
            bytes_io.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes_io)
            else:
                tensor = cast(
                    torch.Tensor,
                    torch.load(bytes_io, map_location="cpu", weights_only=False),
                )
                tensor = _narrow_tensor_by_index(
                    tensor, read_item.storage_offsets, read_item.lengths
                )
                target_tensor = planner.resolve_tensor(read_item).detach()

                assert target_tensor.size() == tensor.size(), (
                    f"req {read_item.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                )
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        if self._metadata is None:
            try:
                if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
                    raise ValueError(
                        "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                        "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                        "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                        "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
                    )
                with (Path(self.path) / ".metadata").open("rb") as metadata_file:
                    metadata = restricted_load(metadata_file)
            except FileNotFoundError as exc:
                msg = f"'{self.path}' is not a distributed checkpoint folder."
                suggested_dir = os.path.join(self.path, "model_and_optim")
                if Path(os.path.join(suggested_dir, ".metadata")).exists():
                    msg += f" Did you mean to use '{suggested_dir}'?"
                raise FileNotFoundError(msg) from exc

            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = StorageMeta()
            metadata.storage_meta.load_id = self.load_id

            self._metadata = metadata

        return self._metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: dist_cp.LoadPlan) -> dist_cp.LoadPlan:
        return plan

    def prepare_global_plan(
        self, global_plan: list[dist_cp.LoadPlan]
    ) -> list[dist_cp.LoadPlan]:
        return global_plan

    @property
    def checkpoint_id(self) -> str:
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Path | str) -> bool:
        del checkpoint_id
        return True


class _RestrictedUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing olmo_core module references."""

    def find_class(self, module, name):
        if module.startswith("torch"):
            return super().find_class(module, name)
        if module in ("collections", "builtins", "_collections_abc"):
            return super().find_class(module, name)
        if module.startswith("olmo_core"):
            return (
                super().find_class("builtins", "dict")
                if name == "dict"
                else type(name, (), {})
            )
        return super().find_class(module, name)


def restricted_loads(data):
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def restricted_load(file):
    return _RestrictedUnpickler(file).load()


def load_model(model_path: str):
    """Load model state dict from distributed checkpoint."""
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    def _load_unsharded_keys(
        dir: Path | str,
        keys: list[str],
        *,
        pre_download: bool = False,
        work_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        _load_state_dict(
            state_dict,
            storage_reader=RemoteFileSystemReader(
                dir, pre_download=pre_download, work_dir=work_dir
            ),
            planner=_EmptyStateDictLoadPlanner(keys=keys),
            no_dist=True,
        )
        return state_dict

    if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
        raise ValueError(
            "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
            "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
            "that could have been tampered with. If you already verified the pickle data and decided to use it, "
            "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
        )
    with (Path(model_path) / ".metadata").open("rb") as metadata_file:
        metadata = restricted_load(metadata_file)
        keys = [
            key
            for key in metadata.state_dict_metadata.keys()
            if key.startswith("model.")
        ]

    return _load_unsharded_keys(model_path, keys)


def get_layer_types_from_checkpoint(
    loaded: dict[str, torch.Tensor], n_layers: int
) -> list[str]:
    """
    Determine layer types by checking which layers have GDN-specific keys (A_log).
    """
    layer_types = []
    for i in range(n_layers):
        if f"blocks.{i}.attention.A_log" in loaded:
            layer_types.append("linear_attention")
        else:
            layer_types.append("full_attention")
    return layer_types


def convert_attention_layer_weights(
    loaded: dict[str, torch.Tensor],
    layer_i: int,
) -> dict[str, torch.Tensor]:
    """Convert weights for a full attention layer (NoPE + elementwise gate + per-head QK norm)."""
    prefix = f"blocks.{layer_i}"
    hf_prefix = f"model.layers.{layer_i}"
    state_dict = {
        # Attention projections
        f"{hf_prefix}.self_attn.q_proj.weight": loaded[
            f"{prefix}.attention.w_q.weight"
        ],
        f"{hf_prefix}.self_attn.k_proj.weight": loaded[
            f"{prefix}.attention.w_k.weight"
        ],
        f"{hf_prefix}.self_attn.v_proj.weight": loaded[
            f"{prefix}.attention.w_v.weight"
        ],
        f"{hf_prefix}.self_attn.o_proj.weight": loaded[
            f"{prefix}.attention.w_out.weight"
        ],
        # Elementwise attention gate
        f"{hf_prefix}.self_attn.attn_gate.weight": loaded[
            f"{prefix}.attention.w_g.weight"
        ],
        # Per-head QK normalization
        f"{hf_prefix}.self_attn.q_norm.weight": loaded[
            f"{prefix}.attention.q_norm.weight"
        ],
        f"{hf_prefix}.self_attn.k_norm.weight": loaded[
            f"{prefix}.attention.k_norm.weight"
        ],
        # MLP
        f"{hf_prefix}.mlp.gate_proj.weight": loaded[f"{prefix}.feed_forward.w1.weight"],
        f"{hf_prefix}.mlp.down_proj.weight": loaded[f"{prefix}.feed_forward.w2.weight"],
        f"{hf_prefix}.mlp.up_proj.weight": loaded[f"{prefix}.feed_forward.w3.weight"],
        # Peri-norm: pre-attention norm (input_layernorm) + post-attention norm
        f"{hf_prefix}.input_layernorm.weight": loaded[
            f"{prefix}.attention_norm.weight"
        ],
        f"{hf_prefix}.post_attention_layernorm.weight": loaded[
            f"{prefix}.post_attention_norm.weight"
        ],
        # Peri-norm: pre-FFN norm (ffn_layernorm) + post-FFN norm
        f"{hf_prefix}.ffn_layernorm.weight": loaded[
            f"{prefix}.feed_forward_norm.weight"
        ],
        f"{hf_prefix}.post_feedforward_layernorm.weight": loaded[
            f"{prefix}.post_feed_forward_norm.weight"
        ],
    }
    return state_dict


def convert_gdn_layer_weights(
    loaded: dict[str, torch.Tensor],
    layer_i: int,
) -> dict[str, torch.Tensor]:
    """Convert weights for a GatedDeltaNet (linear attention) layer."""
    prefix = f"blocks.{layer_i}"
    hf_prefix = f"model.layers.{layer_i}"
    state_dict = {
        # GDN projections
        f"{hf_prefix}.linear_attn.q_proj.weight": loaded[
            f"{prefix}.attention.w_q.weight"
        ],
        f"{hf_prefix}.linear_attn.k_proj.weight": loaded[
            f"{prefix}.attention.w_k.weight"
        ],
        f"{hf_prefix}.linear_attn.v_proj.weight": loaded[
            f"{prefix}.attention.w_v.weight"
        ],
        f"{hf_prefix}.linear_attn.o_proj.weight": loaded[
            f"{prefix}.attention.w_out.weight"
        ],
        f"{hf_prefix}.linear_attn.g_proj.weight": loaded[
            f"{prefix}.attention.w_g.weight"
        ],
        f"{hf_prefix}.linear_attn.a_proj.weight": loaded[
            f"{prefix}.attention.w_a.weight"
        ],
        f"{hf_prefix}.linear_attn.b_proj.weight": loaded[
            f"{prefix}.attention.w_b.weight"
        ],
        # Output norm
        f"{hf_prefix}.linear_attn.o_norm.weight": loaded[
            f"{prefix}.attention.o_norm.weight"
        ],
        # Conv1d weights
        f"{hf_prefix}.linear_attn.q_conv1d.weight": loaded[
            f"{prefix}.attention.q_conv1d.weight"
        ],
        f"{hf_prefix}.linear_attn.k_conv1d.weight": loaded[
            f"{prefix}.attention.k_conv1d.weight"
        ],
        f"{hf_prefix}.linear_attn.v_conv1d.weight": loaded[
            f"{prefix}.attention.v_conv1d.weight"
        ],
        # GDN buffers
        f"{hf_prefix}.linear_attn.A_log": loaded[f"{prefix}.attention.A_log"],
        f"{hf_prefix}.linear_attn.dt_bias": loaded[f"{prefix}.attention.dt_bias"],
        # MLP
        f"{hf_prefix}.mlp.gate_proj.weight": loaded[f"{prefix}.feed_forward.w1.weight"],
        f"{hf_prefix}.mlp.down_proj.weight": loaded[f"{prefix}.feed_forward.w2.weight"],
        f"{hf_prefix}.mlp.up_proj.weight": loaded[f"{prefix}.feed_forward.w3.weight"],
        # Peri-norm: pre-attention norm (input_layernorm) + post-attention norm
        f"{hf_prefix}.input_layernorm.weight": loaded[
            f"{prefix}.attention_norm.weight"
        ],
        f"{hf_prefix}.post_attention_layernorm.weight": loaded[
            f"{prefix}.post_attention_norm.weight"
        ],
        # Peri-norm: pre-FFN norm (ffn_layernorm) + post-FFN norm
        f"{hf_prefix}.ffn_layernorm.weight": loaded[
            f"{prefix}.feed_forward_norm.weight"
        ],
        f"{hf_prefix}.post_feedforward_layernorm.weight": loaded[
            f"{prefix}.post_feed_forward_norm.weight"
        ],
    }
    return state_dict


def write_model(
    model_path: str,
    input_base_path: str,
    include_tokenizer: bool = True,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Convert OLMo Hybrid Small checkpoint to HuggingFace format.

    Args:
        model_path: Output directory for the HuggingFace model.
        input_base_path: Path to the OLMo checkpoint directory containing config.json and model_and_optim/.
        include_tokenizer: Whether to save the tokenizer alongside the model.
        tokenizer_id: HuggingFace tokenizer identifier. If None, uses the one from the checkpoint config.
            WARNING: Do not override this unless you are sure the pretokenizer matches what was used
            during training. Using the wrong tokenizer causes silent numerical divergence on inputs
            with numbers and other edge-case tokens.
        max_sequence_length: Override for max sequence length. If None, read from config.
        dtype: Torch dtype for the output model weights.
    """
    os.makedirs(model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.json"
    olmo_config = json.loads(config_path.read_text())
    model_config = olmo_config["model"]
    block_config = model_config["block"]
    tokenizer_config = olmo_config["dataset"]["tokenizer"]

    n_layers = model_config["n_layers"]
    dim = model_config["d_model"]

    # Hybrid Small uses unified attention config for both GDN and full attention blocks.
    # The block_overrides determine which layers are full attention.
    block_overrides = model_config.get("block_overrides", {})

    # Determine head configuration from block_overrides (full attention layers)
    # or from the base block's sequence_mixer config.
    if block_overrides:
        # Get attention config from the first override (full attention layer)
        first_override = next(iter(block_overrides.values()))
        attn_config = first_override.get("sequence_mixer", {})
        n_heads = attn_config.get("n_heads", model_config.get("n_heads", 8))
        n_kv_heads = attn_config.get("n_kv_heads", n_heads)
        head_dim = attn_config.get("head_dim", dim // n_heads)
    else:
        n_heads = model_config.get("n_heads", 8)
        n_kv_heads = n_heads
        head_dim = dim // n_heads

    # GDN config from base block
    gdn_config = block_config.get("sequence_mixer", {})
    gdn_n_heads = gdn_config.get("n_heads", n_heads)
    gdn_head_dim = gdn_config.get("head_dim", head_dim)
    gdn_expand_v = gdn_config.get("expand_v", 2.0)
    gdn_value_head_dim = int(gdn_head_dim * gdn_expand_v)

    # Resolve max_position_embeddings
    if max_sequence_length is None:
        max_sequence_length = olmo_config.get("train_module", {}).get(
            "max_sequence_length"
        )
    if max_sequence_length is None:
        max_sequence_length = olmo_config.get("dataset", {}).get("sequence_length")
    if max_sequence_length is None:
        max_sequence_length = 8192
        print(
            f"Warning: max_sequence_length not found in config or CLI, using default: {max_sequence_length}"
        )

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = load_model(os.path.join(input_base_path, "model_and_optim"))["model"]
    print(f"Loaded {len(loaded)} keys from checkpoint")

    # Determine layer types from actual checkpoint keys
    layer_types = get_layer_types_from_checkpoint(loaded, n_layers)
    print(f"Layer types: {layer_types}")

    param_count = 0
    full_state_dict: dict[str, torch.Tensor] = {}

    for layer_i in range(n_layers):
        layer_type = layer_types[layer_i]

        if layer_type == "linear_attention":
            layer_state = convert_gdn_layer_weights(loaded, layer_i)
        else:
            layer_state = convert_attention_layer_weights(loaded, layer_i)

        full_state_dict.update(layer_state)
        param_count += sum(v.numel() for v in layer_state.values())
        print(f"Converted layer {layer_i} ({layer_type})")

    # Global weights
    full_state_dict["model.embed_tokens.weight"] = loaded["embeddings.weight"]
    full_state_dict["model.embed_norm.weight"] = loaded["embedding_norm.weight"]
    full_state_dict["model.norm.weight"] = loaded["lm_head.norm.weight"]
    full_state_dict["lm_head.weight"] = loaded["lm_head.w_out.weight"]
    param_count += sum(
        v.numel()
        for v in [
            loaded["embeddings.weight"],
            loaded["embedding_norm.weight"],
            loaded["lm_head.norm.weight"],
            loaded["lm_head.w_out.weight"],
        ]
    )

    # Cast all tensors to target dtype
    full_state_dict = {
        k: v.to(dtype) if torch.is_tensor(v) else v for k, v in full_state_dict.items()
    }

    print(f"Total parameters: {param_count}")

    # Build HF config — NoPE model, so rope_parameters must disable RoPE
    config = MainlineLadderConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=dim,
        intermediate_size=block_config["feed_forward"]["hidden_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_sequence_length,
        pad_token_id=tokenizer_config.get("pad_token_id"),
        bos_token_id=tokenizer_config.get("bos_token_id"),
        eos_token_id=tokenizer_config.get("eos_token_id"),
        tie_word_embeddings=False,
        rms_norm_eps=block_config.get("layer_norm", {}).get("eps", 1e-6),
        layer_types=layer_types,
        # GDN config
        linear_num_key_heads=gdn_n_heads,
        linear_num_value_heads=gdn_n_heads,
        linear_key_head_dim=gdn_head_dim,
        linear_value_head_dim=gdn_value_head_dim,
    )

    # Explicitly ensure NoPE — prevent Transformers from re-enabling RoPE
    # (see: https://github.com/huggingface/transformers/issues — rope_parameters: null
    # gets cast back to defaults in Transformers 5.7.0+)
    config.rope_parameters = {"rope_theta": None}

    config.architectures = ["MainlineLadderForCausalLM"]

    # Save config and weights directly (no from_pretrained roundtrip)
    config.save_pretrained(model_path)

    from safetensors.torch import save_file

    safetensors_path = os.path.join(model_path, "model.safetensors")
    save_file(full_state_dict, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    del full_state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        _write_tokenizer(
            model_path, tokenizer_id, tokenizer_config, max_sequence_length
        )

    # Patch config.json to ensure rope_parameters stays null
    # (config.save_pretrained may have written a default)
    _patch_config_rope(model_path)

    print(f"Conversion complete. Model saved to {model_path}")


def _patch_config_rope(model_path: str) -> None:
    """
    Ensure rope_parameters explicitly disables RoPE in saved config.json.

    This prevents Transformers from silently re-enabling RoPE when loading
    a NoPE model. The root cause is that PreTrainedConfig.__init__ calls
    convert_rope_params_to_dict() which fills in default rope_theta=10000
    when rope_parameters is missing or null.

    Fix: set rope_parameters to {"rope_theta": null} instead of null.
    """
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Force NoPE: use {"rope_theta": null} to prevent default-filling
    config_dict["rope_parameters"] = {"rope_theta": None}
    config_dict.pop("rope_scaling", None)
    config_dict.pop("rope_theta", None)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print("Patched config.json: rope_parameters={'rope_theta': null} (NoPE)")


def _write_tokenizer(
    output_path: str,
    tokenizer_id: str | None,
    tokenizer_config: dict,
    max_sequence_length: int | None = None,
) -> None:
    """
    Save tokenizer with proper configuration.

    IMPORTANT: Uses the tokenizer identifier from the training config by default.
    The pretokenizer (e.g. ByteLevel with use_regex=True for dolma2-tokenizer) must
    match what was used during training. Using a mismatched pretokenizer causes silent
    divergence on inputs with numbers, punctuation, and other edge cases.
    """
    tokenizer_id = tokenizer_id or tokenizer_config.get("identifier")
    if not tokenizer_id:
        print(
            "Warning: No tokenizer identifier found in config. Skipping tokenizer save."
        )
        return

    print(f"Saving tokenizer '{tokenizer_id}' to {output_path}")
    print(
        "  NOTE: Using the exact tokenizer from training config to ensure pretokenizer "
        "compatibility. Do not substitute a generic tokenizer."
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if max_sequence_length is not None:
        tokenizer.model_max_length = max_sequence_length
    tokenizer.pad_token_id = tokenizer_config.get("pad_token_id")
    tokenizer.bos_token_id = tokenizer_config.get("bos_token_id")
    tokenizer.eos_token_id = tokenizer_config.get("eos_token_id")
    tokenizer.save_pretrained(output_path)
    print(
        f"  Tokenizer saved. Verify pretokenizer with: tokenizer.backend_tokenizer.pre_tokenizer"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert OLMo Hybrid Small weights to HuggingFace format."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo Hybrid Small checkpoint, which contains config.json and model_and_optim/.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not save the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "HuggingFace tokenizer identifier. Defaults to the one in the checkpoint config. "
            "WARNING: Only override if you are certain the pretokenizer matches training."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Max sequence length. If not set, reads from config.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Output dtype for model weights. Defaults to bfloat16.",
    )
    args = parser.parse_args()

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        include_tokenizer=args.include_tokenizer,
        tokenizer_id=args.tokenizer,
        max_sequence_length=args.max_sequence_length,
        dtype=DTYPE_MAP[args.dtype],
    )


if __name__ == "__main__":
    main()
