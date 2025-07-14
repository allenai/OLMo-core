"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.

Note that this script is architecture-dependent, meaning it may only work for OLMo Core model
architectures that have support in the `transformers` library.
"""

import contextlib
import json
import logging
import re
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

try:
    import flash_attn  # type: ignore
except ImportError:
    flash_attn = None

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from cached_path import cached_path
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.attention import AttentionType
from olmo_core.nn.conversion.state_mapping import StateType, TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.convert import get_converter_to_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


def convert_checkpoint_to_hf(
    original_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    dtype: Optional[DType] = None,
    max_sequence_length: int = -1,
    validate: bool = True,
    debug: bool = False,
    device: torch.device | None = None,
    validation_sliding_window: int | None = None,
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

    model_config = TransformerConfig.from_dict(transformer_config_dict)

    # Check if validation is being performed and flash attn is requested but cannot run.
    if validate and (flash_attn is None or device != torch.device("cuda")):
        if model_config.block.attention.name == AttentionType.fused:
            log.warning(
                "Running conversion without cuda or flash attention on a model requiring flash attention, validation would fail so we are disabling it."
            )
            validate = False
        elif model_config.block.attention.use_flash:
            log.info(
                "Flash attention or cuda is unavailable, switching to flex attention to stop validation from failing."
            )
            attention["use_flash"] = False
            attention["use_flex_attn"] = True

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    model = model_config.build()
    model.to_empty(device=device or torch.device("cpu"))

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)
    vocab_size = tokenizer_config.vocab_size

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(original_checkpoint_path, "model_and_optim")
        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(
            model_and_optim_dir,
            model,
            work_dir=work_dir,
        )
        log.info(f"Saving checkpoint to '{output_path}'")
        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

        save_hf_model(
            output_path,
            model_state_dict,
            model,
            dtype=dtype,
            vocab_size=vocab_size,
            work_dir=work_dir,
            save_overwrite=True,
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
            output_path,
            model,
            tokenizer_config.vocab_size,
            debug=debug,
            dtype=dtype,
            device=device,
            use_flex_attn=model_config.block.attention.use_flex_attn or False,
            sliding_window=validation_sliding_window,
        )
        log.info("Validation completed successful")


def _register_debug_hooks(hf_model: torch.nn.Module, model: Transformer):
    MAX_DIM_SIZE = 1_000_000

    olmo_core_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}
    hf_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}

    def module_hook(
        debug_state: Dict[str, Tuple[int, torch.Tensor]],
        model_type: str,
        name: str,
        _: torch.nn.Module,
        args,
        output,
    ):
        if (
            model_type == "hf"
            and re.match(r"model.layers.\d+.mlp", name)
            and isinstance(output, tuple)
        ):
            # Special casing for HF moe mlp
            assert isinstance(output[0], torch.Tensor), (name, output)
            output = output[0]
        if model_type == "hf" and re.match(r"model.layers.\d+.mlp.gate", name):
            # Special casing for HF moe router
            assert isinstance(output, torch.Tensor), (name, output)
            router_logits = output.detach().clone()
            routing_weights = F.sigmoid(router_logits.float())
            # Like topk, but we keep all the data. This will hopefully go ok.
            routing_weights, routing_indices = torch.sort(routing_weights, descending=True, dim=-1)
            output = routing_weights
        if model_type == "olmo_core" and re.match(r"blocks.\d+.feed_forward_moe.router", name):
            # Special casing for OLMo Core moe router
            assert isinstance(output, tuple), (name, output)
            assert isinstance(output[0], torch.Tensor), (name, output[0])
            output = output[0]

        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            input = args[0].detach()
            for i, size in enumerate(input.shape):
                input = input.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), input)
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|output"
            output = output.detach()
            for i, size in enumerate(output.shape):
                output = output.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), output)

    for name, module in model.named_modules():
        module.register_forward_hook(partial(module_hook, olmo_core_debug_state, "olmo_core", name))
    for name, module in hf_model.named_modules():
        module.register_forward_hook(partial(module_hook, hf_debug_state, "hf", name))

    return olmo_core_debug_state, hf_debug_state


def validate_conversion(
    hf_path: str | Path,
    model: Transformer,
    vocab_size: int,
    debug: bool = False,
    dtype: DType | None = None,
    device: torch.device | None = None,
    use_flex_attn: bool = False,
    sliding_window: int | None = None,
):
    if torch.cuda.is_available():
        torch.cuda.init()

    device = device or get_default_device()

    B, T = 1, 60
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    attn_implementation = "sdpa"

    is_sliding = any(
        hasattr(block.attention, "window_size") and block.attention.window_size != (-1, -1)
        for block in model.blocks.values()
    )

    log.info("Loading converted checkpoint for validation...")
    kwargs = {}
    if is_sliding and sliding_window is not None:
        kwargs["sliding_window"] = sliding_window
    config = AutoConfig.from_pretrained(
        hf_path,
        **kwargs,
    )
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype="auto",
            config=config,
            attn_implementation=attn_implementation,
        )
        .to(device)
        .eval()
    )

    olmo_core_state, hf_state = {}, {}
    if debug:
        olmo_core_state, hf_state = _register_debug_hooks(hf_model, model)

    log.info("Running OLMo core and HF models for validation...")
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        # Flex attention matches SDPA maths backend
        if use_flex_attn:
            stack.enter_context(
                torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
            )

        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    if dtype:
        model = model.to(dtype.as_pt())
    if is_sliding and sliding_window is not None:
        for block in model.blocks.values():
            if block.attention.window_size != (-1, -1):
                block.attention.window_size = (sliding_window - 1, 0)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        state_converter = get_converter_to_hf()
        if not hasattr(hf_config, "num_hidden_layers"):
            raise ValueError(f"Number of hidden layers missing in HF config: {hf_config}")
        n_layers: int = hf_config.num_hidden_layers
        n_experts: int | None = getattr(hf_config, "num_experts", None)

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        olmo_debug_empty_state = {key.split("|")[0]: None for key in olmo_core_state.keys()}
        state_mapping = state_converter.get_mappings(
            olmo_debug_empty_state, placeholder_bounds, state_type=StateType.module
        )
        del olmo_debug_empty_state

        simple_module_name_mapping = {
            mapping.source_keys[0]: mapping.dest_keys
            for mapping in state_mapping
            if len(mapping.source_keys) == 1
        }

        log.info(f"simple mapping: {simple_module_name_mapping}")
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for olmo_core_state_name, (_, olmo_core_tensor) in sorted(
            olmo_core_state.items(), key=lambda item: item[1][0]
        ):
            olmo_core_key, state_type = olmo_core_state_name.split("|")
            if olmo_core_key in simple_module_name_mapping:
                olmo_core_module_name = olmo_core_key
            else:
                log.warning(
                    f"No 1-to-many param mapping found for module {olmo_core_key}, cannot compare to HF"
                )
                continue

            hf_param_names = simple_module_name_mapping[olmo_core_module_name]
            for i_key, hf_param_name in enumerate(hf_param_names):
                hf_state_name = f"{hf_param_name}|{state_type}"
                if f"{hf_param_name}|{state_type}" in hf_state:
                    hf_state_name = f"{hf_param_name}|{state_type}"
                else:
                    log.warning(
                        f"No HF state found for param {hf_state_name}, cannot compare to OLMo Core"
                    )
                    continue

                _, hf_tensor = hf_state[hf_state_name]
                if hf_tensor.shape[0] < len(hf_param_names):
                    log.warning(
                        f"Unable to chunk HF state {hf_state_name} into {len(hf_param_names)} pieces"
                    )
                    continue
                hf_tensor = hf_tensor.tensor_split(len(hf_param_names), dim=0)[i_key]

                if olmo_core_tensor.shape != hf_tensor.shape:
                    log.info(
                        f"{olmo_core_state_name}, {hf_state_name} shape mismatch: {olmo_core_tensor.shape} {hf_tensor.shape}"
                    )
                if olmo_core_tensor.dtype != hf_tensor.dtype:
                    log.info(
                        f"{olmo_core_state_name}, {hf_state_name} dtype mismatch: {olmo_core_tensor.dtype} {hf_tensor.dtype}"
                    )
                if len(olmo_core_tensor.squeeze().shape) == len(hf_tensor.squeeze().shape):
                    olmo_core_tensor = olmo_core_tensor.squeeze()
                    hf_tensor = hf_tensor.squeeze()

                    common_shape = tuple(
                        min(olmo_core_dim, hf_dim)
                        for olmo_core_dim, hf_dim in zip(olmo_core_tensor.shape, hf_tensor.shape)
                    )
                    for i, dim in enumerate(common_shape):
                        olmo_core_tensor = olmo_core_tensor.narrow(i, 0, dim)
                        hf_tensor = hf_tensor.narrow(i, 0, dim)
                    log.info(
                        f"{olmo_core_state_name}, {hf_state_name} element diff abs mean: {(olmo_core_tensor - hf_tensor).float().abs().mean()}"
                    )

    torch.testing.assert_close(
        hf_logits[..., :vocab_size].float(), logits[..., :vocab_size].float(), rtol=1e-4, atol=1e-4
    )


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
        help="The device on which validation occurs (conversion occurs on cpu). Defaults to CUDA or MPS if available and initialized.",
    )
    parser.add_argument(
        "--dtype",
        help="The torch dtype that model weights should be saved as. Defaults to bfloat16 due to https://github.com/allenai/olmo-cookbook/issues/60.",
        type=DType,
        default=DType.bfloat16,
    )
    parser.add_argument(
        "--validation-sliding-window",
        help="If set, overrides the model's sliding window size during validation. Useful for checking that sliding window is correctly implemented.",
        type=int,
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
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
