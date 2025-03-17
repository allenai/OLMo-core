"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.

Note that this script is architecture-dependent, meaning it may only work for OLMo Core models that
have support in the `transformers` library.
"""

import json
import logging
import types
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import torch
from cached_path import cached_path
from torch.distributed import DeviceMesh
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.io import file_exists
from olmo_core.nn.conversion.state_mapping import TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.convert import get_converter_to_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


def convert_checkpoint_to_hf(
    original_checkpoint_path: str | Path,
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
    Convert a checkpoint to a different OLMo core compatible format.

    Args:
        original_checkpoint_path: Path to the original checkpoint
        output_format: Format of converted checkpoint
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

    # Replace weight init with an efficient alternative that just allocates memory
    @torch.no_grad()
    def init_weights(
        self: Transformer,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        pp_mesh: Optional[DeviceMesh] = None,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        :param pp_mesh: Pipeline parallel mesh. Pass this when using pipeline parallelism
            to ensure the weights are initialized differently for different stages.
        """
        device = device or self.device
        self.to_empty(device=device)

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.to_empty(device=device)  # type: ignore

        seed = self.init_seed
        if pp_mesh is not None:
            seed += pp_mesh.get_local_rank()
        return torch.Generator(device).manual_seed(seed)

    model.init_weights = types.MethodType(init_weights, model)

    device = device or get_default_device()
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=max_sequence_length,
        max_sequence_length=max_sequence_length,
        optim=AdamWConfig(),
    ).build(model, device=device)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    with TemporaryDirectory() as work_dir:
        checkpointer_config = CheckpointerConfig(work_dir=work_dir, save_overwrite=True)
        checkpointer = checkpointer_config.build()

        log.info(f"Loading checkpoint from '{original_checkpoint_path}'")
        checkpointer.load(original_checkpoint_path, train_module, load_trainer_state=False)
        log.info(f"Saving checkpoint to '{output_path}'")
        save_hf_model(
            output_path,
            train_module.state_dict_to_save(optim=False)["model"],
            train_module.model,
            process_group=checkpointer.process_group,
            work_dir=checkpointer.work_dir,
            save_overwrite=checkpointer.save_overwrite,
        )
        # checkpointer.save(output_path, train_module, train_state={}, format=output_format)
        log.info(f"Successfully saved converted model to '{output_path}'")

    log.info("Fixing HF config using tokenizer config data and script arguments")
    huggingface_config = AutoConfig.from_pretrained(output_path)
    huggingface_config.max_position_embeddings = max_sequence_length
    huggingface_config.pad_token_id = tokenizer_config.pad_token_id
    huggingface_config.bos_token_id = tokenizer_config.bos_token_id
    huggingface_config.eos_token_id = tokenizer_config.eos_token_id
    huggingface_config.save_pretrained(output_path)
    log.info("Successfully fixed config using tokenizer config data and script arguments")

    if validate:
        log.info("Validating converted model")
        validate_conversion(
            output_path, model, tokenizer_config.vocab_size, debug=debug, device=device
        )
        log.info("Validation completed successful")


def _register_debug_hooks(hf_model: torch.nn.Module, model: Transformer):
    MAX_DIM_SIZE = 100_000

    olmo_core_state = {}
    hf_state = {}

    def module_hook(state: Dict, name: str, _: torch.nn.Module, args, output):
        # if isinstance()
        # log.info(f"{name}")
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            input = args[0].detach()
            for i, size in enumerate(input.shape):
                input = input.narrow(i, 0, min(size, MAX_DIM_SIZE))
            state[state_name] = (len(state), input.float())
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|output"
            output = output.detach()
            for i, size in enumerate(output.shape):
                output = output.narrow(i, 0, min(size, MAX_DIM_SIZE))
            state[state_name] = (len(state), output.float())

    for name, module in model.named_modules():
        module.register_forward_hook(partial(module_hook, olmo_core_state, name))
    for name, module in hf_model.named_modules():
        module.register_forward_hook(partial(module_hook, hf_state, name))

    return olmo_core_state, hf_state


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
        state_converter = get_converter_to_hf()

        if not hasattr(hf_model.config, "num_hidden_layers"):
            raise ValueError(f"Number of hidden layers missing in HF config: {hf_model.config}")
        n_layers: int = hf_model.config.num_hidden_layers
        n_experts: int | None = getattr(hf_model.config, "num_experts", None)

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        state_mapping = state_converter.get_mappings(model.state_dict(), placeholder_bounds)

    log.info("Running OLMo core and HF models for validation...")
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        assert state_mapping is not None

        simple_key_mapping = {
            mapping.source_keys[0]
            .replace(".weight", ""): mapping.dest_keys[0]
            .replace(".weight", "")
            for mapping in state_mapping
            if len(mapping.source_keys) == 1 and len(mapping.dest_keys) == 1
        }

        log.info(f"simple mapping: {simple_key_mapping}")
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for olmo_core_state_name, (_, olmo_core_tensor) in sorted(
            olmo_core_state.items(), key=lambda item: item[1][0]
        ):
            olmo_core_key, state_type = olmo_core_state_name.split("|")
            if olmo_core_key not in simple_key_mapping:
                continue

            hf_state_name = f"{simple_key_mapping[olmo_core_key]}|{state_type}"
            if hf_state_name not in hf_state:
                continue

            _, hf_tensor = hf_state[hf_state_name]

            log.info(
                f"{olmo_core_state_name}, {hf_state_name} norm diff: {torch.norm(olmo_core_tensor - hf_tensor)}"
            )

    torch.testing.assert_close(hf_logits, logits)


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        raise RuntimeError(
            f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
        )

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", type=str, required=True)

    parser.add_argument("-o", "--huggingface-output-dir", type=Path, required=True)
    parser.add_argument("-s", "--max-sequence-length", type=int, required=True)
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--device", type=torch.device)
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.checkpoint_input_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found, cannot convert to HF checkpoint")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
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
