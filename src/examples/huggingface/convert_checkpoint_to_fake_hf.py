"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.

Note that this script is architecture-dependent, meaning it may only work for OLMo Core model
architectures that have support in the `transformers` library.
"""

import json
import logging
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, Optional, Tuple

import torch
from cached_path import cached_path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    OlmoCoreConfig,
    OlmoCoreForCausalLM,
)

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import get_default_device, prepare_cli_environment

try:
    from accelerate import init_empty_weights
except ImportError:

    @contextmanager
    def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
        log.warning("accelerate not installed, will initialize weights.")
        yield None


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

    hf_config = OlmoCoreConfig(transformer_config_dict)
    with init_empty_weights():
        hf_model = OlmoCoreForCausalLM(hf_config)
    model = hf_model.olmo_core_model

    device = device or get_default_device()
    model.to_empty(device=device)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(original_checkpoint_path, "model_and_optim")
        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(
            model_and_optim_dir,
            model,
            work_dir=work_dir,
        )

        log.info(f"Saving checkpoint to '{output_path}'")
        hf_model.save_pretrained(output_path)
        log.info(f"Successfully saved converted model to '{output_path}'")

    log.info("Fixing HF config using tokenizer config data and script arguments")
    hf_config = AutoConfig.from_pretrained(output_path)
    hf_config.max_position_embeddings = max_sequence_length
    hf_config.pad_token_id = tokenizer_config.pad_token_id
    hf_config.bos_token_id = tokenizer_config.bos_token_id
    hf_config.eos_token_id = tokenizer_config.eos_token_id
    hf_config.save_pretrained(output_path)
    log.info("Successfully fixed config using tokenizer config data and script arguments")

    if validate:
        log.info("Validating converted model")
        validate_conversion(
            output_path, model, tokenizer_config.vocab_size, debug=debug, device=device
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
    if debug:
        olmo_core_state, hf_state = _register_debug_hooks(hf_model, model)

    log.info("Running OLMo core and HF models for validation...")
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for olmo_core_state_name, (_, olmo_core_tensor) in sorted(
            olmo_core_state.items(), key=lambda item: item[1][0]
        ):
            olmo_core_key, state_type = olmo_core_state_name.split("|")
            hf_key = f"olmo_core_model.{olmo_core_key}".rstrip(".")
            hf_state_name = f"{hf_key}|{state_type}"

            _, hf_tensor = hf_state[hf_state_name]

            if olmo_core_tensor.shape != hf_tensor.shape:
                log.info(
                    f"{olmo_core_state_name} shape mismatch: {olmo_core_tensor.shape} {hf_tensor.shape}"
                )
            else:
                log.info(
                    f"{olmo_core_state_name} norm diff: {torch.norm(olmo_core_tensor - hf_tensor)}"
                )

    torch.testing.assert_close(hf_logits[..., :vocab_size], logits[..., :vocab_size])


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
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the OLMo Core checkpoint.",
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
