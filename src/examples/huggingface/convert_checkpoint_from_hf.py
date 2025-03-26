"""
Example script showing how you could convert model weights on HuggingFace for an OLMo2 or Llama-3.*
model into a format that can be loaded by OLMo-core for fine-tuning.

Note that this script is architecture-dependent, meaning it may only work for OLMo2/Llama models on
HuggingFace.
"""

import json
import logging
import tempfile
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from transformers import AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import copy_file, file_exists
from olmo_core.nn.conversion.state_mapping import TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.hf.convert import get_converter_from_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


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
        "llama3": TokenizerConfig.llama3, 
    }

    return tokenizer_configs[tokenizer_id.lower()]()


def convert_checkpoint_from_hf(
    hf_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    max_sequence_length: int = -1,
    validate: bool = True,
    debug: bool = False,
    device: torch.device | None = None,
) -> None:
    """
    Convert a HF checkpoint to an OLMo core checkpoint.

    Args:
        hf_checkpoint_path: Path to the original HF checkpoint
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

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    with TemporaryDirectory() as work_dir:
        log.info(f"Loading HF checkpoint from '{hf_checkpoint_path}'")
        load_hf_model(
            hf_checkpoint_path,
            model_state_dict,
            work_dir=work_dir,
            num_embeddings=model.vocab_size,
        )
        model.load_state_dict(model_state_dict)

    log.info(f"Saving OLMo core checkpoint to '{output_path}'")
    save_model_and_optim_state(output_path, model, save_overwrite=True)
    log.info(f"Successfully saved converted model to '{output_path}'")

    log.info(f"Writing partial experiment config to '{output_path}'")
    experiment_config_dict = {
        "model": transformer_config_dict,
        "dataset": {
            "tokenizer": tokenizer_config_dict,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(experiment_config_dict, temp_file)
        copy_file(temp_file.name, f"{output_path}/config.json", save_overwrite=True)
        log.info(f"Successfully wrote partial experiment config to '{output_path}'")

    if validate:
        log.info("Validating converted model")
        validate_conversion(
            hf_checkpoint_path, model, tokenizer_config.vocab_size, debug=debug, device=device
        )
        log.info("Validation completed successful")


def _register_debug_hooks(hf_model: torch.nn.Module, model: Transformer):
    MAX_DIM_SIZE = 1_000_000

    olmo_core_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}
    hf_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}

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
    for name, module in hf_model.named_modules():
        module.register_forward_hook(partial(module_hook, hf_debug_state, name))

    return olmo_core_debug_state, hf_debug_state


def validate_conversion(
    hf_path: str | Path,
    model: Transformer,
    vocab_size: int,
    debug: bool = False,
    device: torch.device | None = None,
):
    if torch.cuda.is_available():
        torch.cuda.init()

    device = device or get_default_device()

    B, T = 1, 120
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    log.info("Loading converted checkpoint for validation...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path).to(device).eval()

    olmo_core_state, hf_state = {}, {}
    state_mapping = None
    if debug:
        olmo_core_state, hf_state = _register_debug_hooks(hf_model, model)
        state_converter = get_converter_from_hf()

        if not hasattr(hf_model.config, "num_hidden_layers"):
            raise ValueError(f"Number of hidden layers missing in HF config: {hf_model.config}")
        n_layers: int = hf_model.config.num_hidden_layers
        n_experts: int | None = getattr(hf_model.config, "num_experts", None)

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        state_mapping = state_converter.get_mappings(hf_model.state_dict(), placeholder_bounds)

    log.info("Running OLMo core and HF models for validation...")
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
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
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for hf_state_name, (_, hf_tensor) in sorted(hf_state.items(), key=lambda item: item[1][0]):
            hf_key, state_type = hf_state_name.split("|")
            if hf_key not in simple_key_mapping:
                continue

            olmo_core_state_name = f"{simple_key_mapping[hf_key]}|{state_type}"
            if olmo_core_state_name not in olmo_core_state:
                continue

            _, olmo_core_tensor = olmo_core_state[olmo_core_state_name]

            if olmo_core_tensor.shape != hf_tensor.shape:
                log.info(
                    f"{hf_state_name}, {olmo_core_state_name} shape mismatch: {hf_tensor.shape} {olmo_core_tensor.shape}"
                )
            else:
                log.info(
                    f"{hf_state_name}, {olmo_core_state_name} norm diff: {torch.norm(olmo_core_tensor - hf_tensor)}"
                )

    torch.testing.assert_close(hf_logits[..., :vocab_size], logits[..., :vocab_size])


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
    parser.add_argument("-i", "--checkpoint-input-path", type=str, required=True)

    parser.add_argument("-c", "--config-path", type=str, default=None)
    parser.add_argument("-m", "--model-arch")
    parser.add_argument("-t", "--tokenizer", type=str, default="dolma2")

    parser.add_argument("-o", "--huggingface-output-dir", type=Path, required=True)
    parser.add_argument("-s", "--max-sequence-length", type=int, required=True)
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--device", type=torch.device)
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

    convert_checkpoint_from_hf(
        hf_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        max_sequence_length=args.max_sequence_length,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
