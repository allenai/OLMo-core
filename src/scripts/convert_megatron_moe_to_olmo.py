#!/usr/bin/env python3
"""
Convert a Megatron-LM MoE checkpoint into an OLMo-core model checkpoint.

This is currently tailored to the OLMoE3-dev-260401 architecture:

- 24 layers
- layer 0 is dense
- layers 1..23 are MoE
- hidden size 2560
- attention size 3072
- 24 attention heads / 12 KV heads / head dim 128
- 48 routed experts, top-k 4
- 1 shared expert with hidden size 1280

Input checkpoints are Megatron distributed checkpoints like:
  /workspace/checkpoint/.../iter_0062000

Output checkpoints are OLMo-core distributed checkpoints that contain model
weights keyed like:
  module.blocks.1.routed_experts.w_up_gate.main

By default the output is model-only and is written directly to ``output_dir``.

If ``--template-checkpoint`` is provided, the converter instead writes a full
OLMo-style checkpoint directory that looks like:

- ``output_dir/.metadata.json``
- ``output_dir/config.json``
- ``output_dir/data_paths.txt``
- ``output_dir/train/rank*.pt``
- ``output_dir/model_and_optim/...``

The trainer state is copied from the template checkpoint and rewritten with the
Megatron iteration and consumed-token count so OLMo eval can load it as a valid
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.distributed.checkpoint import RemoteFileSystemReader, save_state_dict
from olmo_core.io import normalize_path

log = logging.getLogger(__name__)


NUM_LAYERS = 24
DENSE_LAYER_INDICES = {0}
HIDDEN_SIZE = 2560
D_ATTN = 3072
NUM_HEADS = 24
NUM_KV_HEADS = 12
HEAD_DIM = 128
NUM_EXPERTS = 48
TOP_K = 4
MOE_HIDDEN_SIZE = 2560
NUM_SHARED_EXPERTS = 1
SHARED_EXPERT_HIDDEN_SIZE = 1280
VOCAB_SIZE = 100352


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Megatron-LM MoE checkpoint to an OLMo-core checkpoint."
    )
    parser.add_argument("input_dir", help="Megatron checkpoint directory, e.g. iter_0062000")
    parser.add_argument(
        "output_dir",
        help="Output directory. Without --template-checkpoint this is the model checkpoint root; with it this is the full wrapped OLMo checkpoint dir.",
    )
    parser.add_argument(
        "--template-checkpoint",
        default=None,
        help="Path to a valid OLMo checkpoint directory (for example step147000) used as a wrapper template.",
    )
    parser.add_argument(
        "--megatron-path",
        default="/workspace/Megatron-LM",
        help="Path added to sys.path when reading Megatron common.pt args.",
    )
    parser.add_argument(
        "--save-overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--load-thread-count",
        type=int,
        default=2,
        help="Thread count for distributed checkpoint reads.",
    )
    parser.add_argument(
        "--save-thread-count",
        type=int,
        default=4,
        help="Thread count for distributed checkpoint writes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_megatron_common_payload(
    checkpoint_dir: str,
    megatron_path: str,
) -> Optional[Dict[str, Any]]:
    common_path = Path(checkpoint_dir) / "common.pt"
    if not common_path.is_file():
        return None

    megatron_root = Path(megatron_path)
    if megatron_root.is_dir():
        sys.path.insert(0, str(megatron_root))

    try:
        payload = torch.load(common_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning("Could not read Megatron common.pt for validation: %s", exc)
        return None
    finally:
        if megatron_root.is_dir():
            try:
                sys.path.remove(str(megatron_root))
            except ValueError:
                pass

    return payload if isinstance(payload, dict) else None


def validate_common_args(args_obj: object) -> None:
    expected = {
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_HEADS,
        "num_query_groups": NUM_KV_HEADS,
        "num_experts": NUM_EXPERTS,
        "moe_router_topk": TOP_K,
        "moe_ffn_hidden_size": MOE_HIDDEN_SIZE,
        "moe_shared_expert_intermediate_size": SHARED_EXPERT_HIDDEN_SIZE,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 8,
    }

    for field, expected_value in expected.items():
        actual = getattr(args_obj, field, None)
        if actual != expected_value:
            raise ValueError(
                f"Unsupported checkpoint: expected {field}={expected_value}, got {actual}"
            )

    if getattr(args_obj, "add_bias_linear", None):
        raise ValueError("Unsupported checkpoint: expected bias-free weights")
    if not getattr(args_obj, "swiglu", False):
        raise ValueError("Unsupported checkpoint: expected SwiGLU enabled")

    moe_layer_freq = getattr(args_obj, "moe_layer_freq", None)
    if moe_layer_freq is not None:
        expected_freq = [0] + [1] * (NUM_LAYERS - 1)
        if list(moe_layer_freq) != expected_freq:
            raise ValueError(
                f"Unsupported checkpoint: expected moe_layer_freq={expected_freq}, got {moe_layer_freq}"
            )


def infer_megatron_iteration(
    checkpoint_dir: str,
    common_payload: Optional[Dict[str, Any]],
) -> int:
    if common_payload is not None and "iteration" in common_payload:
        return int(common_payload["iteration"])

    match = re.search(r"iter_(\d+)$", checkpoint_dir.rstrip("/"))
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not infer Megatron iteration from checkpoint '{checkpoint_dir}'. "
        "Expected common.pt['iteration'] or a directory named like iter_0062000."
    )


def infer_megatron_consumed_tokens(
    common_payload: Optional[Dict[str, Any]],
) -> Optional[int]:
    if common_payload is None:
        return None

    args_obj = common_payload.get("args")
    if args_obj is None:
        return None

    consumed_train_samples = getattr(args_obj, "consumed_train_samples", None)
    seq_length = getattr(args_obj, "seq_length", None)
    if consumed_train_samples is None or seq_length is None:
        return None

    return int(consumed_train_samples) * int(seq_length)


def read_metadata(reader: RemoteFileSystemReader) -> Metadata:
    metadata = reader.read_metadata()
    if not isinstance(metadata, Metadata):
        raise TypeError(f"Unexpected metadata type: {type(metadata)}")
    return metadata


def tensor_meta(metadata: Metadata, key: str) -> TensorStorageMetadata:
    meta = metadata.state_dict_metadata.get(key)
    if meta is None:
        raise KeyError(f"Missing checkpoint tensor: {key}")
    if not isinstance(meta, TensorStorageMetadata):
        raise TypeError(f"Checkpoint entry is not a tensor: {key}")
    return meta


def load_tensors(
    reader: RemoteFileSystemReader,
    checkpoint_dir: str,
    metadata: Metadata,
    keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for key in keys:
        meta = tensor_meta(metadata, key)
        state_dict[key] = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")

    dist_cp.state_dict_loader.load(
        state_dict,
        checkpoint_id=checkpoint_dir,
        storage_reader=reader,
        process_group=None,
    )
    return state_dict


def flatten_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(-1)


def swap_gate_up_halves_2d(tensor: torch.Tensor, hidden_size: int) -> torch.Tensor:
    gate, up = tensor.split(hidden_size, dim=0)
    return torch.cat((up, gate), dim=0)


def swap_gate_up_halves_3d(tensor: torch.Tensor, hidden_size: int) -> torch.Tensor:
    gate, up = tensor.split(hidden_size, dim=1)
    return torch.cat((up, gate), dim=1)


def convert_qkv_weight(meg_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_query_groups = NUM_KV_HEADS
    num_query_heads_per_group = NUM_HEADS // NUM_KV_HEADS
    rows_per_group = (num_query_heads_per_group + 2) * HEAD_DIM

    expected_shape = (num_query_groups * rows_per_group, HIDDEN_SIZE)
    if tuple(meg_qkv.shape) != expected_shape:
        raise ValueError(
            f"Unexpected QKV shape: expected {expected_shape}, got {tuple(meg_qkv.shape)}"
        )

    qkv = meg_qkv.view(num_query_groups, rows_per_group, HIDDEN_SIZE)
    q_rows = num_query_heads_per_group * HEAD_DIM
    q, k, v = torch.split(qkv, [q_rows, HEAD_DIM, HEAD_DIM], dim=1)
    return q.reshape(-1, HIDDEN_SIZE), k.reshape(-1, HIDDEN_SIZE), v.reshape(-1, HIDDEN_SIZE)


def convert_dense_layer(
    layer_idx: int,
    source: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
) -> None:
    prefix_in = f"decoder.layers.{layer_idx}"
    prefix_out = f"module.blocks.{layer_idx}"

    output[f"{prefix_out}.attention_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.input_layernorm.weight"]
    )
    output[f"{prefix_out}.post_attention_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.post_self_attn_layernorm.weight"]
    )
    output[f"{prefix_out}.feed_forward_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.pre_mlp_layernorm.weight"]
    )
    output[f"{prefix_out}.post_feed_forward_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.post_mlp_layernorm.weight"]
    )

    q, k, v = convert_qkv_weight(source[f"{prefix_in}.self_attention.linear_qkv.weight"])
    output[f"{prefix_out}.attention.w_q.weight.main"] = flatten_contiguous(q)
    output[f"{prefix_out}.attention.w_k.weight.main"] = flatten_contiguous(k)
    output[f"{prefix_out}.attention.w_v.weight.main"] = flatten_contiguous(v)
    output[f"{prefix_out}.attention.w_out.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.self_attention.linear_proj.weight"]
    )
    output[f"{prefix_out}.attention.q_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.self_attention.q_layernorm.weight"]
    )
    output[f"{prefix_out}.attention.k_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.self_attention.k_layernorm.weight"]
    )

    fc1 = source[f"{prefix_in}.mlp.linear_fc1.weight"]
    if tuple(fc1.shape) != (2 * 11520, HIDDEN_SIZE):
        raise ValueError(f"Unexpected dense fc1 shape for layer {layer_idx}: {tuple(fc1.shape)}")
    gate, up = fc1.split(11520, dim=0)
    output[f"{prefix_out}.feed_forward.w1.weight.main"] = flatten_contiguous(gate)
    output[f"{prefix_out}.feed_forward.w3.weight.main"] = flatten_contiguous(up)
    output[f"{prefix_out}.feed_forward.w2.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.mlp.linear_fc2.weight"]
    )


def convert_moe_layer(
    layer_idx: int,
    source: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
) -> None:
    prefix_in = f"decoder.layers.{layer_idx}"
    prefix_out = f"module.blocks.{layer_idx}"

    output[f"{prefix_out}.attention_input_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.input_layernorm.weight"]
    )
    output[f"{prefix_out}.attention_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.post_self_attn_layernorm.weight"]
    )
    output[f"{prefix_out}.feed_forward_input_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.pre_mlp_layernorm.weight"]
    )
    output[f"{prefix_out}.feed_forward_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.post_mlp_layernorm.weight"]
    )

    q, k, v = convert_qkv_weight(source[f"{prefix_in}.self_attention.linear_qkv.weight"])
    output[f"{prefix_out}.attention.w_q.weight.main"] = flatten_contiguous(q)
    output[f"{prefix_out}.attention.w_k.weight.main"] = flatten_contiguous(k)
    output[f"{prefix_out}.attention.w_v.weight.main"] = flatten_contiguous(v)
    output[f"{prefix_out}.attention.w_out.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.self_attention.linear_proj.weight"]
    )
    output[f"{prefix_out}.attention.q_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.self_attention.q_layernorm.weight"]
    )
    output[f"{prefix_out}.attention.k_norm.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.self_attention.k_layernorm.weight"]
    )

    output[f"{prefix_out}.routed_experts_router.weight.main"] = flatten_contiguous(
        source[f"{prefix_in}.mlp.router.weight"]
    )

    routed_fc1 = source[f"{prefix_in}.mlp.experts.experts.linear_fc1.weight"]
    output[f"{prefix_out}.routed_experts.w_up_gate.main"] = flatten_contiguous(
        swap_gate_up_halves_3d(routed_fc1, MOE_HIDDEN_SIZE)
    )

    routed_fc2 = source[f"{prefix_in}.mlp.experts.experts.linear_fc2.weight"]
    output[f"{prefix_out}.routed_experts.w_down.main"] = flatten_contiguous(
        routed_fc2.transpose(1, 2)
    )

    shared_fc1 = source[f"{prefix_in}.mlp.shared_experts.linear_fc1.weight"]
    shared_fc1 = swap_gate_up_halves_2d(shared_fc1, SHARED_EXPERT_HIDDEN_SIZE)
    output[f"{prefix_out}.shared_experts.w_up_gate.main"] = flatten_contiguous(shared_fc1.transpose(0, 1))

    shared_fc2 = source[f"{prefix_in}.mlp.shared_experts.linear_fc2.weight"]
    output[f"{prefix_out}.shared_experts.w_down.main"] = flatten_contiguous(shared_fc2.transpose(0, 1))


def get_model_layer_indices(metadata: Metadata) -> List[int]:
    layer_indices = set()
    pattern = re.compile(r"^decoder\.layers\.(\d+)\.")
    for key in metadata.state_dict_metadata.keys():
        match = pattern.match(key)
        if match:
            layer_indices.add(int(match.group(1)))
    return sorted(layer_indices)


def validate_metadata_shape(
    metadata: Metadata,
    key: str,
    expected_shape: Tuple[int, ...],
) -> None:
    actual = tuple(tensor_meta(metadata, key).size)
    if actual != expected_shape:
        raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual}")


def validate_checkpoint_structure(metadata: Metadata) -> None:
    validate_metadata_shape(metadata, "embedding.word_embeddings.weight", (VOCAB_SIZE, HIDDEN_SIZE))
    validate_metadata_shape(metadata, "embedding_norm.weight", (HIDDEN_SIZE,))
    validate_metadata_shape(metadata, "output_layer.weight", (VOCAB_SIZE, HIDDEN_SIZE))
    validate_metadata_shape(metadata, "decoder.final_layernorm.weight", (HIDDEN_SIZE,))

    layer_indices = get_model_layer_indices(metadata)
    if layer_indices != list(range(NUM_LAYERS)):
        raise ValueError(f"Expected layers 0..{NUM_LAYERS - 1}, got {layer_indices}")

    for layer_idx in layer_indices:
        prefix = f"decoder.layers.{layer_idx}"
        validate_metadata_shape(
            metadata,
            f"{prefix}.self_attention.linear_qkv.weight",
            ((D_ATTN + 2 * NUM_KV_HEADS * HEAD_DIM), HIDDEN_SIZE),
        )
        validate_metadata_shape(
            metadata,
            f"{prefix}.self_attention.linear_proj.weight",
            (HIDDEN_SIZE, D_ATTN),
        )
        validate_metadata_shape(metadata, f"{prefix}.self_attention.q_layernorm.weight", (HEAD_DIM,))
        validate_metadata_shape(metadata, f"{prefix}.self_attention.k_layernorm.weight", (HEAD_DIM,))
        validate_metadata_shape(metadata, f"{prefix}.input_layernorm.weight", (HIDDEN_SIZE,))
        validate_metadata_shape(metadata, f"{prefix}.post_self_attn_layernorm.weight", (HIDDEN_SIZE,))
        validate_metadata_shape(metadata, f"{prefix}.pre_mlp_layernorm.weight", (HIDDEN_SIZE,))
        validate_metadata_shape(metadata, f"{prefix}.post_mlp_layernorm.weight", (HIDDEN_SIZE,))

        if layer_idx in DENSE_LAYER_INDICES:
            validate_metadata_shape(metadata, f"{prefix}.mlp.linear_fc1.weight", (23040, HIDDEN_SIZE))
            validate_metadata_shape(metadata, f"{prefix}.mlp.linear_fc2.weight", (HIDDEN_SIZE, 11520))
        else:
            validate_metadata_shape(metadata, f"{prefix}.mlp.router.weight", (NUM_EXPERTS, HIDDEN_SIZE))
            validate_metadata_shape(
                metadata,
                f"{prefix}.mlp.experts.experts.linear_fc1.weight",
                (NUM_EXPERTS, 2 * MOE_HIDDEN_SIZE, HIDDEN_SIZE),
            )
            validate_metadata_shape(
                metadata,
                f"{prefix}.mlp.experts.experts.linear_fc2.weight",
                (NUM_EXPERTS, HIDDEN_SIZE, MOE_HIDDEN_SIZE),
            )
            validate_metadata_shape(
                metadata,
                f"{prefix}.mlp.shared_experts.linear_fc1.weight",
                (2 * SHARED_EXPERT_HIDDEN_SIZE, HIDDEN_SIZE),
            )
            validate_metadata_shape(
                metadata,
                f"{prefix}.mlp.shared_experts.linear_fc2.weight",
                (HIDDEN_SIZE, SHARED_EXPERT_HIDDEN_SIZE),
            )


def build_output_state(
    reader: RemoteFileSystemReader,
    checkpoint_dir: str,
    metadata: Metadata,
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}

    root_keys = [
        "embedding.word_embeddings.weight",
        "embedding_norm.weight",
        "decoder.final_layernorm.weight",
        "output_layer.weight",
    ]
    root = load_tensors(reader, checkpoint_dir, metadata, root_keys)
    output["module.embeddings.weight.main"] = flatten_contiguous(root["embedding.word_embeddings.weight"])
    output["module.embedding_norm.weight.main"] = flatten_contiguous(root["embedding_norm.weight"])
    output["module.lm_head.norm.weight.main"] = flatten_contiguous(root["decoder.final_layernorm.weight"])
    output["module.lm_head.w_out.weight.main"] = flatten_contiguous(root["output_layer.weight"])
    del root

    for layer_idx in range(NUM_LAYERS):
        prefix = f"decoder.layers.{layer_idx}"
        common_keys = [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.self_attention.linear_proj.weight",
            f"{prefix}.self_attention.linear_qkv.weight",
            f"{prefix}.self_attention.q_layernorm.weight",
            f"{prefix}.self_attention.k_layernorm.weight",
            f"{prefix}.post_self_attn_layernorm.weight",
            f"{prefix}.pre_mlp_layernorm.weight",
            f"{prefix}.post_mlp_layernorm.weight",
        ]
        if layer_idx in DENSE_LAYER_INDICES:
            layer_keys = common_keys + [
                f"{prefix}.mlp.linear_fc1.weight",
                f"{prefix}.mlp.linear_fc2.weight",
            ]
        else:
            layer_keys = common_keys + [
                f"{prefix}.mlp.router.weight",
                f"{prefix}.mlp.experts.experts.linear_fc1.weight",
                f"{prefix}.mlp.experts.experts.linear_fc2.weight",
                f"{prefix}.mlp.shared_experts.linear_fc1.weight",
                f"{prefix}.mlp.shared_experts.linear_fc2.weight",
            ]

        log.info("Loading Megatron layer %d tensors", layer_idx)
        source = load_tensors(reader, checkpoint_dir, metadata, layer_keys)
        if layer_idx in DENSE_LAYER_INDICES:
            convert_dense_layer(layer_idx, source, output)
        else:
            convert_moe_layer(layer_idx, source, output)
        del source

    return output


def expected_output_numel() -> int:
    total = 0
    total += VOCAB_SIZE * HIDDEN_SIZE  # embeddings
    total += HIDDEN_SIZE  # embed norm
    total += HIDDEN_SIZE  # lm head norm
    total += VOCAB_SIZE * HIDDEN_SIZE  # lm head out

    dense_ffn_hidden = TOP_K * MOE_HIDDEN_SIZE + NUM_SHARED_EXPERTS * SHARED_EXPERT_HIDDEN_SIZE
    for layer_idx in range(NUM_LAYERS):
        total += D_ATTN * HIDDEN_SIZE  # w_q
        total += (NUM_KV_HEADS * HEAD_DIM) * HIDDEN_SIZE  # w_k
        total += (NUM_KV_HEADS * HEAD_DIM) * HIDDEN_SIZE  # w_v
        total += HIDDEN_SIZE * D_ATTN  # w_out
        total += HEAD_DIM * 2  # q_norm + k_norm
        if layer_idx in DENSE_LAYER_INDICES:
            total += 4 * HIDDEN_SIZE  # four norms in peri dense block
            total += dense_ffn_hidden * HIDDEN_SIZE * 2  # w1 + w3
            total += HIDDEN_SIZE * dense_ffn_hidden  # w2
        else:
            total += 4 * HIDDEN_SIZE  # attention/feedforward input/output norms
            total += NUM_EXPERTS * HIDDEN_SIZE  # router
            total += NUM_EXPERTS * (2 * MOE_HIDDEN_SIZE) * HIDDEN_SIZE  # routed up_gate
            total += NUM_EXPERTS * MOE_HIDDEN_SIZE * HIDDEN_SIZE  # routed down
            total += HIDDEN_SIZE * (2 * SHARED_EXPERT_HIDDEN_SIZE)  # shared up_gate
            total += SHARED_EXPERT_HIDDEN_SIZE * HIDDEN_SIZE  # shared down

    return total


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_template_checkpoint(template_dir: Path, output_dir: Path, save_overwrite: bool) -> None:
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template checkpoint does not exist: {template_dir}")

    if output_dir.exists():
        if not save_overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --save-overwrite to replace it."
            )
        remove_path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for src in template_dir.iterdir():
        if src.name == "model_and_optim":
            continue
        dst = output_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def patch_template_config(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Template checkpoint is missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        routed_router_config = config["model"]["block"]["routed_experts_router"]
    except KeyError as exc:
        raise KeyError(
            f"Template checkpoint config is missing model.block.routed_experts_router: {config_path}"
        ) from exc

    # Megatron already produces top-k-normalized router weights, so OLMo should not
    # re-expand them by multiplying by top_k when evaluating converted checkpoints.
    routed_router_config["restore_weight_scale"] = False

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def infer_batches_processed_for_new_tokens(state_dict: Dict[str, Any], new_tokens: int, step: int) -> int:
    data_loader_state = state_dict.get("data_loader", {})
    old_tokens = int(data_loader_state.get("tokens_processed", 0) or 0)
    old_batches = int(data_loader_state.get("batches_processed", 0) or 0)
    if old_tokens > 0 and old_batches > 0:
        return max(int(round(new_tokens * old_batches / old_tokens)), 0)
    return step


def patch_trainer_state(
    state_dict: Dict[str, Any],
    *,
    global_step: int,
    global_train_tokens_seen: int,
) -> Dict[str, Any]:
    state_dict = dict(state_dict)
    state_dict["global_step"] = global_step
    state_dict["global_train_tokens_seen"] = global_train_tokens_seen
    state_dict["max_steps"] = max(int(state_dict.get("max_steps", 0) or 0), global_step)

    data_loader_state = dict(state_dict.get("data_loader", {}))
    data_loader_state["tokens_processed"] = global_train_tokens_seen
    data_loader_state["batches_processed"] = infer_batches_processed_for_new_tokens(
        state_dict, global_train_tokens_seen, global_step
    )
    state_dict["data_loader"] = data_loader_state
    return state_dict


def patch_template_train_states(
    output_dir: Path,
    *,
    global_step: int,
    global_train_tokens_seen: int,
) -> None:
    train_dir = output_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Template checkpoint is missing train state directory: {train_dir}")

    trainer_state_paths = sorted(train_dir.glob("rank*.pt"))
    if not trainer_state_paths:
        raise FileNotFoundError(f"Template checkpoint is missing trainer state files in: {train_dir}")

    for path in trainer_state_paths:
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        patched = patch_trainer_state(
            state_dict,
            global_step=global_step,
            global_train_tokens_seen=global_train_tokens_seen,
        )
        torch.save(patched, path)


def save_wrapped_checkpoint(
    *,
    template_checkpoint: str,
    output_dir: str,
    model_state_dict: Dict[str, torch.Tensor],
    global_step: int,
    global_train_tokens_seen: int,
    save_overwrite: bool,
    save_thread_count: int,
) -> None:
    template_dir = Path(template_checkpoint)
    out_dir = Path(output_dir)
    copy_template_checkpoint(template_dir, out_dir, save_overwrite=save_overwrite)
    patch_template_config(out_dir)
    patch_template_train_states(
        out_dir,
        global_step=global_step,
        global_train_tokens_seen=global_train_tokens_seen,
    )
    save_state_dict(
        str(out_dir / "model_and_optim"),
        model_state_dict,
        save_overwrite=save_overwrite,
        thread_count=save_thread_count,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    input_dir = normalize_path(args.input_dir)
    output_dir = normalize_path(args.output_dir)
    template_checkpoint = (
        normalize_path(args.template_checkpoint) if args.template_checkpoint is not None else None
    )

    common_payload = load_megatron_common_payload(input_dir, args.megatron_path)
    common_args = common_payload.get("args") if common_payload is not None else None
    if common_args is not None:
        validate_common_args(common_args)

    reader = RemoteFileSystemReader(input_dir, thread_count=args.load_thread_count)
    metadata = read_metadata(reader)
    validate_checkpoint_structure(metadata)

    log.info("Converting Megatron checkpoint from %s", input_dir)
    output_state = build_output_state(reader, input_dir, metadata)

    converted_numel = sum(t.numel() for t in output_state.values())
    expected_numel = expected_output_numel()
    if converted_numel != expected_numel:
        raise ValueError(
            f"Converted numel mismatch: expected {expected_numel}, got {converted_numel}"
        )

    if template_checkpoint is not None:
        megatron_iteration = infer_megatron_iteration(input_dir, common_payload)
        consumed_tokens = infer_megatron_consumed_tokens(common_payload)
        if consumed_tokens is None:
            raise ValueError(
                "Could not infer consumed Megatron tokens from common.pt args. "
                "This is required when --template-checkpoint is used."
            )

        log.info(
            "Saving wrapped OLMo checkpoint to %s (%d tensors, %.3fB params, step=%d, tokens=%d)",
            output_dir,
            len(output_state),
            converted_numel / 1e9,
            megatron_iteration,
            consumed_tokens,
        )
        save_wrapped_checkpoint(
            template_checkpoint=template_checkpoint,
            output_dir=output_dir,
            model_state_dict=output_state,
            global_step=megatron_iteration,
            global_train_tokens_seen=consumed_tokens,
            save_overwrite=args.save_overwrite,
            save_thread_count=args.save_thread_count,
        )
    else:
        log.info(
            "Saving model-only OLMo checkpoint to %s (%d tensors, %.3fB params)",
            output_dir,
            len(output_state),
            converted_numel / 1e9,
        )
        save_state_dict(
            output_dir,
            output_state,
            save_overwrite=args.save_overwrite,
            thread_count=args.save_thread_count,
        )
    log.info("Conversion complete")


if __name__ == "__main__":
    main()
