"""
Example script showing how you could convert model weights of an old OLMo unsharded model
into a format that can be loaded by OLMo-core for fine-tuning.

Note that this script is architecture-dependent. Some models may work out-of-the-box.
"""

import json
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from olmo import OLMo
import torch
from cached_path import cached_path

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import file_exists
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import (
    StateMapping,
    StateMappingTemplate,
    TemplatePlaceholder,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


LAYER = TemplatePlaceholder.LAYER


OLD_OLMO_TO_OLMO_CORE_MAPPINGS: Dict[str, str] = {
    "transformer.wte.weight": "embeddings.weight",
    "transformer.ln_f.weight": "lm_head.norm.weight",
    "transformer.ff_out.weight": "lm_head.w_out.weight",
    # Attention.
    # f"model.layers.{LAYER}.self_attn.q_proj.weight": f"blocks.{LAYER}.attention.w_q.weight",
    # f"model.layers.{LAYER}.self_attn.k_proj.weight": f"blocks.{LAYER}.attention.w_k.weight",
    # f"model.layers.{LAYER}.self_attn.v_proj.weight": f"blocks.{LAYER}.attention.w_v.weight",
    f"transformer.blocks.{LAYER}.attn_out.weight": f"blocks.{LAYER}.attention.w_out.weight",
    # MLP.
    # f"model.layers.{LAYER}.mlp.gate_proj.weight": f"blocks.{LAYER}.feed_forward.w1.weight",
    # f"model.layers.{LAYER}.mlp.up_proj.weight": f"blocks.{LAYER}.feed_forward.w3.weight",
    f"transformer.blocks.{LAYER}.ff_out.weight": f"blocks.{LAYER}.feed_forward.w2.weight",
    # Layer norms.
    f"transformer.blocks.{LAYER}.attn_norm.weight": f"blocks.{LAYER}.attention_norm.weight",
    f"transformer.blocks.{LAYER}.ff_norm.weight": f"blocks.{LAYER}.feed_forward_norm.weight",
    f"transformer.blocks.{LAYER}.q_norm.weight": f"blocks.{LAYER}.attention.q_norm.weight",
    f"transformer.blocks.{LAYER}.k_norm.weight": f"blocks.{LAYER}.attention.k_norm.weight",
}


OLD_OLMO_TO_OLMO_CORE_TEMPLATE_MAPPINGS: Dict[str, StateMappingTemplate] = {
    f"transformer.blocks.{LAYER}.att_proj.weight": StateMappingTemplate(
        f"transformer.blocks.{LAYER}.att_proj.weight",
        (
            f"blocks.{LAYER}.attention.w_q.weight",
            f"blocks.{LAYER}.attention.w_k.weight",
            f"blocks.{LAYER}.attention.w_v.weight",
        ),
    ),
    f"transformer.blocks.{LAYER}.ff_proj.weight": StateMappingTemplate(
        f"transformer.blocks.{LAYER}.ff_proj.weight",
        (f"blocks.{LAYER}.feed_forward.w3.weight", f"blocks.{LAYER}.feed_forward.w1.weight"),
    ),
}


def _get_converter() -> StateConverter:
    mapping_templates = {
        hf_key: StateMappingTemplate(hf_key, olmo_core_key)
        for hf_key, olmo_core_key in OLD_OLMO_TO_OLMO_CORE_MAPPINGS.items()
    }
    mapping_templates.update(OLD_OLMO_TO_OLMO_CORE_TEMPLATE_MAPPINGS)

    return StateConverter(list(mapping_templates.values()))


def _get_transformer_config(model_arch: str, vocab_size: int) -> TransformerConfig:
    transformer_configs = {
        "olmo2_190m": TransformerConfig.olmo2_190M,
        "olmo2_370m": TransformerConfig.olmo2_370M,
        "olmo2_600m": TransformerConfig.olmo2_600M,
        "olmo2_760m": TransformerConfig.olmo2_760M,
        "olmo2_1b": TransformerConfig.olmo2_1B,
        "olmo2_3b": TransformerConfig.olmo2_3B,
        "olmo2_7b": TransformerConfig.olmo2_7B,
        "olmo2_13b": TransformerConfig.olmo2_13B,
        "olmo2_32b": TransformerConfig.olmo2_32B,
        "smallmoe": TransformerConfig.smallmoe,
        "olmoe_1b_7b": TransformerConfig.olmoe_1B_7B,
        "ngpt_271m": TransformerConfig.ngpt_271M,
        "ngpt_1b": TransformerConfig.ngpt_1B,
        "llama2_271m": TransformerConfig.llama2_271M,
        "llama2_1b": TransformerConfig.llama2_1B,
        "llama2_7b": TransformerConfig.llama2_7B,
        "llama2_13b": TransformerConfig.llama2_13B,
        "llama2_26b": TransformerConfig.llama2_26B,
        "llama2_70b": TransformerConfig.llama2_70B,
        "llama3_1b": TransformerConfig.llama3_1B,
        "llama3_8b": TransformerConfig.llama3_8B,
        "llama3_70b": TransformerConfig.llama3_70B,
        "llama3_405b": TransformerConfig.llama3_405B,
    }

    return transformer_configs[model_arch.lower()](vocab_size)


def _get_tokenizer_config(tokenizer_id: str) -> TokenizerConfig:
    tokenizer_configs = {
        "dolma2": TokenizerConfig.dolma2,
        "gpt_neox_olmo_dolma_v1_5": TokenizerConfig.gpt_neox_olmo_dolma_v1_5,
        "gpt2": TokenizerConfig.gpt2,
    }

    return tokenizer_configs[tokenizer_id.lower()]()


def _convert_optim_tensor_state(
    optim_state: Dict[int, Dict[str, torch.Tensor]],
    idx_to_clean_name: Dict[int, str],
    converter: StateConverter,
    placeholder_bounds: Dict[TemplatePlaceholder, int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    optim_substate_names = set(optim_state[0].keys())

    optim_substates_by_name_and_substate = {
        substate_name: {
            idx_to_clean_name[idx]: state[substate_name] for idx, state in optim_state.items()
        }
        for substate_name in optim_substate_names
    }

    converted_optim_state_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
    for substate_name, optim_substate_by_name in optim_substates_by_name_and_substate.items():
        converted_optim_substate_by_name = converter.convert(
            optim_substate_by_name, placeholder_bounds=placeholder_bounds
        )

        for name, converted_substate in converted_optim_substate_by_name.items():
            if name not in converted_optim_state_by_name:
                converted_optim_state_by_name[name] = {}

            converted_optim_state_by_name[name][substate_name] = converted_substate

    return converted_optim_state_by_name


def _build_converted_optim_state(
    optim_state_dict: Dict[str, Any],
    idx_to_param_group_idx: Dict[int, int],
    idx_to_clean_name: Dict[int, str],
    clean_name_to_converted_names: Dict[str, Tuple[str, ...]],
    converted_tensor_state_by_name: Dict[str, Dict[str, torch.Tensor]],
):
    log.info(f"idx_to_clean_name: {idx_to_clean_name}")
    log.info(f"idx_to_param_group_idx: {idx_to_param_group_idx}")
    log.info(f"clean_name_to_converted_names: {clean_name_to_converted_names}")

    converted_optim_state_dict: Dict[str, Any] = {
        "param_groups": [],
        "state": {},
    }
    # {param_idx: {substate_name: substate}}
    converted_tensor_state: Dict[int, Dict[str, torch.Tensor]] = converted_optim_state_dict["state"]

    for idx in sorted(idx_to_clean_name.keys()):
        param_group_idx = idx_to_param_group_idx[idx]
        old_param_group: Dict[str, Any] = optim_state_dict["param_groups"][param_group_idx]

        converted_param_groups: List[Dict[str, Any]] = converted_optim_state_dict["param_groups"]
        if param_group_idx < len(converted_param_groups):
            converted_param_group = converted_param_groups[param_group_idx]
        elif param_group_idx == len(converted_param_groups):
            converted_param_group = old_param_group.copy()
            converted_param_group["param_names"] = []
            converted_param_group["params"] = []
            converted_param_groups.append(converted_param_group)
        else:
            raise RuntimeError(
                f"Param group index {param_group_idx} is larger than expected ({len(converted_param_groups)})"
            )

        clean_name = idx_to_clean_name[idx]
        converted_names = clean_name_to_converted_names[clean_name]

        for converted_name in converted_names:
            global_idx = len(converted_tensor_state)
            converted_tensor_state[global_idx] = converted_tensor_state_by_name[converted_name]
            converted_param_group["param_names"].append(converted_name)
            converted_param_group["params"].append(global_idx)

    if len(converted_tensor_state) != len(converted_tensor_state_by_name):
        expected_num_params = len(converted_tensor_state_by_name)
        actual_num_params = len(converted_tensor_state)
        raise RuntimeError(
            f"Expected {expected_num_params} params in converted state dict, finished with {actual_num_params}."
        )

    return converted_optim_state_dict


def _convert_optim_state(
    optim_state_dict: Dict[str, Any], converter: StateConverter, placeholder_bounds: Dict[TemplatePlaceholder, int],
):
    # Collect some convenient mapping information.
    idx_to_clean_name: Dict[int, str] = {}
    idx_to_param_group_idx: Dict[int, int] = {}
    clean_name_to_idx: Dict[str, int] = {}
    for i_group, param_group in enumerate(optim_state_dict["param_groups"]):
        param_names = param_group["param_names"]
        for i, name in enumerate(param_names):
            assert isinstance(name, str), name
            idx = param_group["params"][i]

            idx_to_param_group_idx[idx] = i_group

            clean_name = name.replace("_fsdp_wrapped_module.", "")
            clean_name_to_idx[clean_name] = idx
            idx_to_clean_name[idx] = clean_name

    # {param_idx: {substate_name: substate}}
    optim_tensor_state: Dict[int, Dict[str, torch.Tensor]] = optim_state_dict["state"]

    if idx_to_clean_name.keys() != set(range(len(optim_tensor_state))):
        raise RuntimeError("Param groups and state are incompatible")

    converted_tensor_state_by_name = _convert_optim_tensor_state(
        optim_tensor_state, idx_to_clean_name, converter, placeholder_bounds
    )

    mappings = converter.get_mappings(clean_name_to_idx, placeholder_bounds=placeholder_bounds)
    clean_name_to_converted_names = {
        source_key: mapping.dest_keys for mapping in mappings for source_key in mapping.source_keys
    }

    return _build_converted_optim_state(
        optim_state_dict,
        idx_to_param_group_idx,
        idx_to_clean_name,
        clean_name_to_converted_names,
        converted_tensor_state_by_name,
    )


def convert_checkpoint_from_old_olmo(
    old_olmo_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    model_id: str | None = None,
    max_sequence_length: int = -1,
    validate: bool = True,
    debug: bool = False,
    device: torch.device | None = None,
) -> None:
    """
    Convert a HF checkpoint to an OLMo core checkpoint.

    Args:
        old_olmo_checkpoint_path: Path to the original OLMo unsharded checkpoint
        output_path: Where to save the converted model
        transformer_config_dict: Dictionary form of OLMo core model config
        tokenizer_config_dict: Dictionary form of OLMo core tokenizer config
    """
    if max_sequence_length <= 0:
        raise ValueError(f"Missing or invalid sequence length: {max_sequence_length}")

    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    model = TransformerConfig.from_dict(transformer_config_dict).build()
    device = device or get_default_device()
    model.to_empty(device=device)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    # We can use the
    converter = _get_converter()
    placeholder_bounds = {
        TemplatePlaceholder.LAYER: model.n_layers,
    }

    log.info(f"Loading OLMo model state from '{old_olmo_checkpoint_path}'")
    model_state_dict = torch.load(f"{old_olmo_checkpoint_path}/model.pt")
    model_state_dict = converter.convert(model_state_dict, placeholder_bounds)
    model.load_state_dict(model_state_dict)

    log.info(f"Loading OLMo optim state from '{old_olmo_checkpoint_path}'")
    optim_state_dict = torch.load(f"{old_olmo_checkpoint_path}/optim.pt")
    optim_state_dict = _convert_optim_state(optim_state_dict, converter, placeholder_bounds)
    # We unset fixed fields because we don't have any field we want to fix!
    optim: torch.optim.Optimizer = AdamWConfig(
        fixed_fields=tuple(),
        group_overrides=[
            OptimGroupOverride(params=param_group["param_names"], opts={
                k: v
                for k, v in param_group.items()
                if k not in ("param_names", "params")
            })
            for param_group in optim_state_dict["param_groups"]
        ],
    ).build(model)
    optim.load_state_dict(optim_state_dict)

    log.info(f"Saving OLMo core checkpoint to '{output_path}'")
    save_model_and_optim_state(output_path, model, optim=optim, save_overwrite=True)
    log.info(f"Successfully saved converted model to '{output_path}'")

    if validate:
        log.info("Validating converted model")
        validate_conversion(
            old_olmo_checkpoint_path,
            model,
            tokenizer_config.vocab_size,
            optim=optim,
            model_id=model_id,
            debug=debug,
            device=device,
        )
        log.info("Validation completed successful")


def _register_debug_hooks(old_olmo_model: torch.nn.Module, model: Transformer):
    MAX_DIM_SIZE = 1_000_000

    olmo_core_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}
    old_olmo_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}

    def module_hook(
        debug_state: Dict[str, Tuple[int, torch.Tensor]],
        name: str,
        _: torch.nn.Module,
        args,
        output,
    ):
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            input = args[0].detach()
            for i, size in enumerate(input.shape):
                input = input.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), input.float())
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|output"
            output = output.detach()
            for i, size in enumerate(output.shape):
                output = output.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), output.float())

    for name, module in model.named_modules():
        module.register_forward_hook(partial(module_hook, olmo_core_debug_state, name))
    for name, module in old_olmo_model.named_modules():
        module.register_forward_hook(partial(module_hook, old_olmo_debug_state, name))

    return olmo_core_debug_state, old_olmo_debug_state


def _compare_debug_state(expected_state: Dict[str, Tuple[int, torch.Tensor]], actual_state: Dict[str, Tuple[int, torch.Tensor]], state_mapping: List[StateMapping]):
    assert state_mapping is not None

    simple_key_mapping = {
        mapping.source_keys[0]
        .replace(".weight", ""): mapping.dest_keys[0]
        .replace(".weight", "")
        for mapping in state_mapping
        if len(mapping.source_keys) == 1
        and len(mapping.dest_keys) == 1
        and mapping.source_keys[0].endswith(".weight")
        and mapping.dest_keys[0].endswith(".weight")
    }

    log.info(f"mapping: {simple_key_mapping}")
    log.info(f"expected_state keys: {expected_state.keys()}")
    log.info(f"actual_state keys: {actual_state.keys()}")

    for orig_name, (_, expected_tensor) in sorted(expected_state.items(), key=lambda item: item[1][0]):
        orig_key, state_type = orig_name.split("|")
        if orig_key not in simple_key_mapping:
            continue

        converted_name = f"{simple_key_mapping[orig_key]}|{state_type}"
        if converted_name not in actual_state:
            continue

        _, actual_tensor = actual_state[converted_name]

        if actual_tensor.shape != expected_tensor.shape:
            log.info(
                f"{orig_name}, {converted_name} shape mismatch: {expected_tensor.shape} {actual_tensor.shape}"
            )
        else:
            log.info(
                f"{orig_name}, {converted_name} norm diff: {torch.norm(actual_tensor - expected_tensor)}"
            )



def validate_conversion(
    old_olmo_path: str | Path,
    model: Transformer,
    vocab_size: int,
    optim: torch.optim.Optimizer | None = None,
    model_id: str | None = None,
    debug: bool = False,
    device: torch.device | None = None,
):
    if torch.cuda.is_available():
        torch.cuda.init()

    device = device or get_default_device()

    B, T = 1, 120
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    log.info("Loading converted checkpoint for validation...")
    old_olmo_model = OLMo.from_checkpoint(old_olmo_path).to(device).eval()

    olmo_core_state, old_olmo_state = {}, {}
    state_mapping = None
    if debug:
        olmo_core_state, old_olmo_state = _register_debug_hooks(old_olmo_model, model)
        state_converter = _get_converter()

        n_layers: int = old_olmo_model.config.n_layers

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }

        state_mapping = state_converter.get_mappings(
            old_olmo_model.state_dict(), placeholder_bounds
        )

    log.info("Running OLMo core and old OLMo models for validation...")
    with torch.no_grad():
        old_olmo_logits = old_olmo_model(input_ids=input_ids).logits

    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        assert state_mapping is not None
        _compare_debug_state(old_olmo_state, olmo_core_state, state_mapping)
        old_olmo_state.clear()
        olmo_core_state.clear()

    torch.testing.assert_close(old_olmo_logits[..., :vocab_size], logits[..., :vocab_size])

    if optim:
        log.info("Loading OLMo optimizer for validation...")
        optim_state_dict = torch.load(f"{old_olmo_path}/optim.pt")
        old_olmo_optim = torch.optim.AdamW(old_olmo_model.parameters())
        old_olmo_optim.load_state_dict(optim_state_dict)

        labels = input_ids[...,:-1]

        log.info("Running optimizer step of OLMo core and old OLMo models for validation...")
        old_olmo_loss = torch.nn.functional.cross_entropy(old_olmo_logits, labels)
        old_olmo_loss.backward()
        old_olmo_optim.step()

        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optim.step()

        log.info("Running 2nd step of OLMo core and old OLMo models for validation...")

        with torch.no_grad():
            old_olmo_logits = old_olmo_model(input_ids=input_ids).logits

        model = model.to(device).eval()
        with torch.no_grad():
            logits = model(input_ids=input_ids)

        if debug:
            assert state_mapping is not None
            _compare_debug_state(old_olmo_state, olmo_core_state, state_mapping)
            old_olmo_state.clear()
            olmo_core_state.clear()

            torch.testing.assert_close(old_olmo_logits[..., :vocab_size], logits[..., :vocab_size])


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        log.warning(f"Config file not found at {checkpoint_input_dir}")
        return None

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        log.warning(
            f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
        )
        return None

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the HF checkpoint, or the model id of a HF Hub repo.",
    )

    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default=None,
        help="Path to an OLMo Core experiment config containing information about the model architecture and tokenizer.",
    )
    parser.add_argument(
        "-m",
        "--model-arch",
        help="OLMo Core model architecture corresponding to the HF model. New architectures should be added to ``_get_transformer_config``. This is required when an OLMo Core experiment config is not provided.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="dolma2",
        help="OLMo Core tokenizer corresponding to the HF model. New tokenizers should be added to ``_get_tokenizer_config``. This is required when an OLMo Core experiment config is not provided.",
    )

    parser.add_argument(
        "-o",
        "--huggingface-output-dir",
        type=Path,
        required=True,
        help="Local or remote directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        required=True,
        help="Max sequence length supported by the model.",
    )
    parser.add_argument(
        "--model-id",
        help="Model id of the HF Hub repo corresponding to the model. Use to get model specific mappings in :mod:`olmo_core.nn.hf.convert`",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="If set, validation to check that the converted model matches the original model is skipped.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="If set, debug information of validation is output.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="The device on which conversion and validation occurs. Defaults to CUDA or MPS if available and initialized.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.config_path or args.checkpoint_input_path)
    transformer_config_dict = None
    if experiment_config is not None:
        transformer_config_dict = experiment_config["model"]
        tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")
    else:
        assert args.model_arch is not None
        assert args.tokenizer is not None
        tokenizer_config = _get_tokenizer_config(args.tokenizer)
        transformer_config_dict = _get_transformer_config(
            args.model_arch, tokenizer_config.padded_vocab_size()
        ).as_config_dict()
        tokenizer_config_dict = tokenizer_config.as_config_dict()

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    convert_checkpoint_from_old_olmo(
        old_olmo_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        model_id=args.model_id,
        max_sequence_length=args.max_sequence_length,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
